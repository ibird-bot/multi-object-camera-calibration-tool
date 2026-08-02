"""
Bootstrap: turn raw observations into a starting point the solver can refine.

Order matters, because each step depends on the previous one:

  1. Per-camera intrinsics, via OpenCV's own planar calibration on that
     camera's views. Cheap, and far better than "focal = image width".
  2. Per (camera, frame, board) pose by PnP  ->  T_cam_board.
  3. Camera extrinsics relative to the reference, from frames where two
     cameras see the SAME board:  T_c_ref = T_c_board * inv(T_ref_board).
     Averaged over all such frames.
  4. Board poses in the rig frame:  T_rig_board = inv(T_cam_rig) * T_cam_board.

Step 3 is the one that fails silently on real data: if two cameras never share
a board in any frame, their relative pose is simply not observable. That is
detected and raised rather than filled with a plausible-looking identity.
"""

from __future__ import annotations

import cv2
import numpy as np

from mlti_cal.models.manifolds import (
    IDENTITY_POSE,
    matrix_to_quat,
    pose_compose,
    pose_from_rt,
    pose_inverse,
    quat_to_matrix,
)
from mlti_cal.problem.types import CalibrationSystem


class InitializationError(RuntimeError):
    """Raised when a starting point cannot be established honestly."""


def _quaternion_average(quats: np.ndarray) -> np.ndarray:
    """
    Chordal L2 mean of unit quaternions: principal eigenvector of sum(q q^T).

    Sign-insensitive by construction, which matters because q and -q are the
    same rotation and a naive componentwise mean of mixed signs collapses
    toward zero.
    """
    A = np.zeros((4, 4))
    for q in quats:
        q = q / np.linalg.norm(q)
        A += np.outer(q, q)
    w, V = np.linalg.eigh(A)
    q = V[:, int(np.argmax(w))]
    return q / np.linalg.norm(q)


def _average_poses(poses: list[np.ndarray]) -> np.ndarray:
    P = np.asarray(poses, dtype=float)
    t = np.median(P[:, 0:3], axis=0)  # median: robust to a bad PnP
    q = _quaternion_average(P[:, 3:7])
    return np.concatenate([t, q])


def initialize_intrinsics(
    system: CalibrationSystem, min_views: int = 4, verbose: bool = False
) -> dict[str, float]:
    """
    Per-camera intrinsics via `cv2.calibrateCamera`, treating every
    (frame, board) as an independent planar view.

    Returns the OpenCV RMS per camera. Cameras with too few views keep their
    existing (default) parameters and are reported with RMS = nan rather than
    being silently left looking calibrated.
    """
    rms: dict[str, float] = {}
    for cid, cam in system.cameras.items():
        obj_pts, img_pts = [], []
        for obs in system.observations:
            if obs.camera != cid or obs.num_points < 6:
                continue
            board = system.boards[obs.board]
            obj_pts.append(board.object_points[obs.point_ids].astype(np.float32))
            img_pts.append(obs.image_points.astype(np.float32))
        if len(obj_pts) < min_views:
            rms[cid] = float("nan")
            continue

        flags = 0
        if cam.model_name == "fisheye_kb":
            # cv2.fisheye wants (N,1,3)/(N,1,2) and its own flag set.
            obj_f = [p.reshape(-1, 1, 3) for p in obj_pts]
            img_f = [p.reshape(-1, 1, 2) for p in img_pts]
            K = np.eye(3)
            D = np.zeros((4, 1))
            try:
                err, K, D, _, _ = cv2.fisheye.calibrate(
                    obj_f,
                    img_f,
                    cam.image_size,
                    K,
                    D,
                    flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6),
                )
            except cv2.error as exc:  # pragma: no cover - depends on data
                rms[cid] = float("nan")
                if verbose:
                    print(f"  {cid}: fisheye calibrate failed ({exc})")
                continue
            cam.params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *np.asarray(D).ravel()[:4]])
            rms[cid] = float(err)
        else:
            err, K, D, _, _ = cv2.calibrateCamera(
                obj_pts, img_pts, cam.image_size, None, None, flags=flags
            )
            d = np.asarray(D).ravel()
            d = np.pad(d, (0, max(0, 5 - d.size)))[:5]
            # OpenCV distCoeffs order is (k1,k2,p1,p2,k3) -- same as ours.
            cam.params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *d])
            rms[cid] = float(err)
        if verbose:
            print(f"  {cid}: OpenCV init RMS = {rms[cid]:.4f} px over {len(obj_pts)} views")
    return rms


def solve_pnp_poses(system: CalibrationSystem) -> dict[tuple[str, str, str], np.ndarray]:
    """(frame, camera, board) -> T_cam_board, by PnP with current intrinsics."""
    out: dict[tuple[str, str, str], np.ndarray] = {}
    for obs in system.observations:
        if obs.num_points < 4:
            continue
        cam = system.cameras[obs.camera]
        board = system.boards[obs.board]
        obj = board.object_points[obs.point_ids].astype(np.float64)
        img = obs.image_points.astype(np.float64)
        K = cam.model.matrix(cam.params)
        dist = cam.model.distortion(cam.params)
        if cam.model_name == "fisheye_kb":
            # Undistort to normalised rays, then PnP with an identity camera.
            und = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), K, dist.reshape(4, 1)).reshape(
                -1, 2
            )
            ok, rvec, tvec = cv2.solvePnP(
                obj, und, np.eye(3), np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        out[(obs.frame, obs.camera, obs.board)] = pose_from_rt(rvec, tvec)
    return out


def initialize_extrinsics(
    system: CalibrationSystem, pnp: dict[tuple[str, str, str], np.ndarray]
) -> dict[str, int]:
    """
    Relative camera poses from co-observed boards. Returns support counts.

    Raises InitializationError for any camera with zero shared observations --
    an unobservable extrinsic must not be papered over with identity.
    """
    ref = system.reference_camera
    support: dict[str, int] = {}
    for cid, cam in system.cameras.items():
        if cid == ref:
            cam.extrinsic = IDENTITY_POSE.copy()
            support[cid] = -1
            continue
        candidates = []
        for (frame, camera, board), T_cam_board in pnp.items():
            if camera != cid:
                continue
            T_ref_board = pnp.get((frame, ref, board))
            if T_ref_board is None:
                continue
            # T_cam_ref = T_cam_board * inv(T_ref_board)
            candidates.append(pose_compose(T_cam_board, pose_inverse(T_ref_board)))
        if not candidates:
            raise InitializationError(
                f"camera {cid!r} never observes a board at the same time as the "
                f"reference camera {ref!r}; its extrinsic is not observable. "
                f"Add frames where both cameras see one board, or make {cid!r} "
                f"the reference of a separate system."
            )
        cam.extrinsic = _average_poses(candidates)
        support[cid] = len(candidates)
    return support


def initialize_board_poses(
    system: CalibrationSystem, pnp: dict[tuple[str, str, str], np.ndarray]
) -> None:
    """T_rig_board per (frame, board), preferring the camera that saw most points."""
    best: dict[tuple[str, str], tuple[int, str]] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for obs in system.observations:
        counts[(obs.frame, obs.camera, obs.board)] = obs.num_points
    for (frame, camera, board), n in counts.items():
        key = (frame, board)
        if (frame, camera, board) not in pnp:
            continue
        if key not in best or n > best[key][0]:
            best[key] = (n, camera)

    for (frame, board), (_, camera) in best.items():
        T_cam_board = pnp[(frame, camera, board)]
        T_cam_rig = system.cameras[camera].extrinsic
        # T_rig_board = inv(T_cam_rig) * T_cam_board
        system.board_poses[(frame, board)] = pose_compose(pose_inverse(T_cam_rig), T_cam_board)

    missing = [k for k in system.frame_board_pairs if k not in system.board_poses]
    if missing:
        raise InitializationError(
            f"could not initialise poses for {len(missing)} (frame, board) pairs, "
            f"first few: {missing[:5]}"
        )


def initialize_system(
    system: CalibrationSystem, do_intrinsics: bool = True, verbose: bool = False
) -> dict:
    """Run the full bootstrap. Returns a small report dict."""
    report: dict = {}
    if do_intrinsics:
        if verbose:
            print("Initialising intrinsics with OpenCV...")
        report["intrinsics_rms"] = initialize_intrinsics(system, verbose=verbose)
    pnp = solve_pnp_poses(system)
    report["pnp_solved"] = len(pnp)
    report["extrinsic_support"] = initialize_extrinsics(system, pnp)
    initialize_board_poses(system, pnp)
    report["board_poses"] = len(system.board_poses)
    if verbose:
        print(
            f"  PnP poses: {report['pnp_solved']}, "
            f"board poses: {report['board_poses']}, "
            f"extrinsic support: {report['extrinsic_support']}"
        )
    return report
