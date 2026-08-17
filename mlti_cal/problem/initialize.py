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

import dataclasses
from collections.abc import Callable

import cv2
import numpy as np

from mlti_cal.models.manifolds import (
    IDENTITY_POSE,
    pose_compose,
    pose_from_rt,
    pose_inverse,
)
from mlti_cal.problem.settings import InitSettings
from mlti_cal.problem.types import CalibrationSystem


class InitializationError(RuntimeError):
    """Raised when a starting point cannot be established honestly."""


def _cv_calib_flag(name: str) -> int:
    """
    Look a CALIB_* flag up across both namespaces OpenCV has put it in.

    OpenCV 4 exposes the fisheye flags only as `cv2.fisheye.CALIB_*`; OpenCV 5
    -- the version this project pins -- exposes them only at the top level, so
    the 4.x spelling raises AttributeError the moment a fisheye camera is
    bootstrapped. Resolving by name keeps both working instead of pinning the
    code to one OpenCV generation.
    """
    for namespace in (cv2.fisheye, cv2):
        flag = getattr(namespace, name, None)
        if flag is not None:
            return int(flag)
    raise AttributeError(f"OpenCV {cv2.__version__} exposes no {name} in cv2 or cv2.fisheye")


def _fisheye_flags(settings: InitSettings) -> int:
    """The two fisheye fit flags the user can turn off, as one bitmask."""
    flags = 0
    if settings.fisheye_recompute_extrinsic:
        flags |= _cv_calib_flag("CALIB_RECOMPUTE_EXTRINSIC")
    if settings.fisheye_fix_skew:
        flags |= _cv_calib_flag("CALIB_FIX_SKEW")
    return flags


#: Parameter indices each CALIB_FIX_* flag covers, per model. OpenCV can only
#: hold these GROUPS: there is no flag for "fx alone", so pinning one half of a
#: pair buys no flag at all. Pins that no flag covers are still honoured -- they
#: are written back after the fit -- they simply do not constrain the fit
#: itself, which is what `unsupported_pins` exists to warn about.
_PINHOLE_FIX_FLAGS: tuple[tuple[tuple[int, ...], str], ...] = (
    ((0, 1), "CALIB_FIX_FOCAL_LENGTH"),
    ((2, 3), "CALIB_FIX_PRINCIPAL_POINT"),
    ((4,), "CALIB_FIX_K1"),
    ((5,), "CALIB_FIX_K2"),
    ((6, 7), "CALIB_FIX_TANGENT_DIST"),
    ((8,), "CALIB_FIX_K3"),
)
_FISHEYE_FIX_FLAGS: tuple[tuple[tuple[int, ...], str], ...] = (
    ((0, 1), "CALIB_FIX_FOCAL_LENGTH"),
    ((2, 3), "CALIB_FIX_PRINCIPAL_POINT"),
    ((4,), "CALIB_FIX_K1"),
    ((5,), "CALIB_FIX_K2"),
    ((6,), "CALIB_FIX_K3"),
    ((7,), "CALIB_FIX_K4"),
)


def _fix_flag_table(model_name: str) -> tuple[tuple[tuple[int, ...], str], ...]:
    """
    Which CALIB_FIX_* groups apply, by model.

    Empty for every model OpenCV cannot fit: those are bootstrapped from a
    pinhole fit for K alone, so no OpenCV flag holds any of THEIR parameters
    and claiming otherwise would report pins as supported when they are only
    written back afterwards.
    """
    if model_name == "fisheye_kb":
        return _FISHEYE_FIX_FLAGS
    if model_name == "pinhole_radtan":
        return _PINHOLE_FIX_FLAGS
    return ()


def _resolve_flag(model_name: str, flag_name: str) -> int | None:
    """
    The flag's value for this model's OpenCV entry point, or None if absent.

    Namespace matters and is easy to get wrong: `cv2.fisheye.CALIB_*` and
    `cv2.CALIB_*` are DIFFERENT bit values for the same name, so a fisheye flag
    passed to `cv2.calibrateCamera` silently means something else. Pinhole
    therefore resolves against `cv2` only.
    """
    if model_name == "fisheye_kb":
        for namespace in (cv2.fisheye, cv2):
            flag = getattr(namespace, flag_name, None)
            if flag is not None:
                return int(flag)
        return None
    flag = getattr(cv2, flag_name, None)
    return None if flag is None else int(flag)


def _pin_flags(model_name: str, pins) -> int:
    """
    CALIB_* flags holding `pins` during the fit, or 0 when nothing is pinned.

    CALIB_USE_INTRINSIC_GUESS comes along with any pin and is not optional:
    without it OpenCV sets a FIXED distortion coefficient to zero rather than to
    the supplied one, which would turn "hold k1 at 0.12" into "hold k1 at 0".
    """
    if not pins:
        return 0
    want = {int(i) for i in pins}
    flags = _resolve_flag(model_name, "CALIB_USE_INTRINSIC_GUESS") or 0
    for group, flag_name in _fix_flag_table(model_name):
        if want.issuperset(group):
            value = _resolve_flag(model_name, flag_name)
            if value is not None:
                flags |= value
    return flags


def unsupported_pins(model_name: str, indices) -> tuple[int, ...]:
    """
    Which pinned indices OpenCV cannot hold during the bootstrap fit.

    They are still forced to the requested value afterwards; they just do not
    constrain the fit, so the other parameters are fitted against a value that
    is then overwritten. Worth saying out loud, not worth refusing.
    """
    want = {int(i) for i in indices}
    covered: set[int] = set()
    for group, flag_name in _fix_flag_table(model_name):
        if want.issuperset(group) and _resolve_flag(model_name, flag_name) is not None:
            covered.update(group)
    return tuple(sorted(want - covered))


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


def _average_poses(poses: list[np.ndarray], translation: str = "median") -> np.ndarray:
    """
    Combine repeated estimates of ONE pose.

    Translation is median by default -- robust to a single bad PnP -- or mean
    when the caller has established there are no outliers. Rotation is always
    the quaternion eigenvector mean: there is no meaningful componentwise
    median of a rotation, so offering one would be a lie.
    """
    P = np.asarray(poses, dtype=float)
    t = np.mean(P[:, 0:3], axis=0) if translation == "mean" else np.median(P[:, 0:3], axis=0)
    q = _quaternion_average(P[:, 3:7])
    return np.concatenate([t, q])


def initialize_intrinsics(
    system: CalibrationSystem,
    min_views: int | None = None,
    verbose: bool = False,
    fixed: dict[str, dict[int, float]] | None = None,
    on_progress: Callable[[str], None] | None = None,
    settings: InitSettings | None = None,
) -> dict[str, float]:
    """
    Per-camera intrinsics via `cv2.calibrateCamera`, treating every
    (frame, board) as an independent planar view.

    Returns the OpenCV RMS per camera. Cameras with too few views keep their
    existing (default) parameters and are reported with RMS = nan rather than
    being silently left looking calibrated.

    Args:
        fixed: camera id -> {parameter index: value the user already knows}.
            Each value is written in before the fit, held there by the matching
            CALIB_FIX_* flag where OpenCV has one, and written back afterwards
            in every case. The flags are the optimisation; the write-back is the
            contract, so "I typed 960 for cx" survives even where no flag exists
            (see `unsupported_pins`). Note that a pin forces
            CALIB_USE_INTRINSIC_GUESS: without it OpenCV resets any FIXED
            distortion coefficient to zero instead of to the supplied value.
    """
    cfg = settings or InitSettings()
    if min_views is None:
        min_views = cfg.min_views_per_camera
    rms: dict[str, float] = {}
    for n, (cid, cam) in enumerate(system.cameras.items(), start=1):
        if on_progress is not None:
            on_progress(f"calibrating {cid} ({n}/{len(system.cameras)})...")
        pins = {int(i): float(v) for i, v in (fixed or {}).get(cid, {}).items()}
        for idx, value in pins.items():
            cam.params[idx] = value
        obj_pts, img_pts = [], []
        for obs in system.observations:
            if obs.camera != cid or obs.num_points < cfg.min_points_for_calibration:
                continue
            board = system.boards[obs.board]
            obj_pts.append(board.object_points[obs.point_ids].astype(np.float32))
            img_pts.append(obs.image_points.astype(np.float32))
        if len(obj_pts) < min_views:
            rms[cid] = float("nan")
            continue

        flags = _pin_flags(cam.model_name, pins)
        if cam.model_name == "fisheye_kb":
            # cv2.fisheye wants (N,1,3)/(N,1,2) and its own flag set.
            obj_f = [p.reshape(-1, 1, 3) for p in obj_pts]
            img_f = [p.reshape(-1, 1, 2) for p in img_pts]
            K = cam.model.matrix(cam.params) if pins else np.eye(3)
            D = (cam.params[4:8] if pins else np.zeros(4)).reshape(4, 1).astype(float)
            try:
                err, K, D, _, _ = cv2.fisheye.calibrate(
                    obj_f,
                    img_f,
                    cam.image_size,
                    K,
                    D,
                    flags=_fisheye_flags(cfg) | flags,
                    criteria=cfg.fisheye_criteria,
                )
            except cv2.error as exc:  # pragma: no cover - depends on data
                rms[cid] = float("nan")
                if verbose:
                    print(f"  {cid}: fisheye calibrate failed ({exc})")
                continue
            cam.params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *np.asarray(D).ravel()[:4]])
            rms[cid] = float(err)
        elif cam.model_name == "pinhole_radtan":
            K0 = cam.model.matrix(cam.params) if pins else None
            D0 = np.asarray(cam.params[4:9], dtype=float).reshape(1, 5) if pins else None
            err, K, D, _, _ = cv2.calibrateCamera(
                obj_pts, img_pts, cam.image_size, K0, D0, flags=flags
            )
            d = np.asarray(D).ravel()
            d = np.pad(d, (0, max(0, 5 - d.size)))[:5]
            # OpenCV distCoeffs order is (k1,k2,p1,p2,k3) -- same as ours.
            cam.params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *d])
            rms[cid] = float(err)
        else:
            # OpenCV cannot fit this model, so it supplies K only.
            #
            # This is honest rather than clever: every model here reduces to the
            # pinhole at zero distortion (Double Sphere at xi=alpha=0, EUCM at
            # alpha=0, division at kappa=0, thin prism at all-zero), so a pinhole
            # fit IS a point on that model's own manifold, and the bundle
            # adjustment moves off it. FOV is the exception -- its parameter is
            # unidentifiable at w=0 -- and `default_params` starts it at 0.5 for
            # exactly that reason, so the value already in `cam.params` is kept.
            #
            # The distortion OpenCV fitted is DISCARDED, not translated: its k1
            # is not this model's first coefficient, and reusing the number
            # because it sits in the same slot is how a bootstrap silently
            # starts somewhere meaningless.
            err, K, _, _, _ = cv2.calibrateCamera(obj_pts, img_pts, cam.image_size, None, None)
            cam.params[0] = K[0, 0]
            cam.params[1] = K[1, 1]
            cam.params[2] = K[0, 2]
            cam.params[3] = K[1, 2]
            # OpenCV's RMS describes ITS model, not this one, so it is not
            # reported as if it were this camera's initialisation error.
            rms[cid] = float("nan")
            if verbose:
                print(
                    f"  {cid}: {cam.model_name} has no OpenCV fit; took K from a "
                    f"pinhole fit (RMS {err:.4f} px) and left distortion at its default"
                )
        # The contract, independent of what OpenCV did with the flags: a pinned
        # component holds the value that was asked for, exactly.
        for idx, value in pins.items():
            cam.params[idx] = value
        if on_progress is not None:
            got = rms[cid]
            on_progress(
                f"{cid}: OpenCV RMS {got:.4f} px over {len(obj_pts)} views"
                if got == got
                else f"{cid}: too few usable views, keeping default parameters"
            )
        if verbose:
            print(f"  {cid}: OpenCV init RMS = {rms[cid]:.4f} px over {len(obj_pts)} views")
    return rms


def replace_threshold(cfg: InitSettings, focal: float) -> InitSettings:
    """`cfg` with its RANSAC threshold converted from pixels to normalised units."""
    if not cfg.pnp_ransac or focal <= 0:
        return cfg
    return dataclasses.replace(
        cfg, ransac_reproj_threshold_px=cfg.ransac_reproj_threshold_px / focal
    )


def _run_pnp(obj: np.ndarray, img: np.ndarray, K: np.ndarray, dist: np.ndarray, cfg: InitSettings):
    """
    One PnP call, RANSAC or not. Returns (ok, rvec, tvec, inliers or None).

    RANSAC is a different OpenCV entry point rather than a flag, and its
    threshold is in PIXELS of reprojection error -- which for the fisheye path
    means normalised units, since the points have already been undistorted to
    rays. That is why the caller scales it by the focal length there.
    """
    if not cfg.pnp_ransac:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cfg.pnp_flag)
        return ok, rvec, tvec, None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj,
        img,
        K,
        dist,
        flags=cfg.pnp_flag,
        reprojectionError=float(cfg.ransac_reproj_threshold_px),
        confidence=float(cfg.ransac_confidence),
        iterationsCount=int(cfg.ransac_max_iterations),
    )
    return ok, rvec, tvec, inliers


def solve_pnp_poses(
    system: CalibrationSystem, settings: InitSettings | None = None
) -> dict[tuple[str, str, str], np.ndarray]:
    """(frame, camera, board) -> T_cam_board, by PnP with current intrinsics."""
    cfg = settings or InitSettings()
    out: dict[tuple[str, str, str], np.ndarray] = {}
    for obs in system.observations:
        if obs.num_points < cfg.min_points_for_pnp:
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
            # The threshold the user typed is in pixels, but `und` is in
            # normalised units, so it has to be divided by the focal length --
            # otherwise a 3 px threshold becomes 3 RADIANS and RANSAC accepts
            # everything, silently doing nothing at all.
            scaled = replace_threshold(cfg, float(K[0, 0]))
            ok, rvec, tvec, _ = _run_pnp(obj, und, np.eye(3), np.zeros(5), scaled)
        elif cam.model.opencv_compatible:
            ok, rvec, tvec, _ = _run_pnp(obj, img, K, dist, cfg)
        else:
            # OpenCV would read this model's coefficients as its own -- Double
            # Sphere's (xi, alpha) as (k1, k2) -- and undistort by a law the
            # camera does not obey. So the model inverts its OWN projection
            # (Newton on the analytic Jacobian it already provides) and PnP runs
            # on the resulting rays with an identity camera.
            und = cam.model.undistort_to_normalised(cam.params, img)
            scaled = replace_threshold(cfg, float(K[0, 0]))
            ok, rvec, tvec, _ = _run_pnp(obj, und, np.eye(3), np.zeros(5), scaled)
        if not ok:
            continue
        out[(obs.frame, obs.camera, obs.board)] = pose_from_rt(rvec, tvec)
    return out


def initialize_extrinsics(
    system: CalibrationSystem,
    pnp: dict[tuple[str, str, str], np.ndarray],
    settings: InitSettings | None = None,
) -> dict[str, int]:
    """
    Relative camera poses from co-observed boards. Returns support counts.

    Raises InitializationError for any camera with zero shared observations --
    an unobservable extrinsic must not be papered over with identity.
    """
    cfg = settings or InitSettings()
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
        cam.extrinsic = _average_poses(candidates, cfg.translation_average)
        support[cid] = len(candidates)
    return support


def initialize_board_poses(
    system: CalibrationSystem,
    pnp: dict[tuple[str, str, str], np.ndarray],
    settings: InitSettings | None = None,
) -> None:
    """
    T_rig_board per (frame, board).

    Two ways to pick it when several cameras saw the board in one frame, both
    exposed because they fail differently. `max_corners` trusts the single
    best-conditioned PnP -- one bad view is either used or not. `average`
    combines every camera's estimate, which is steadier but folds extrinsic
    error into the board pose, making a wrong extrinsic harder to see in the
    residuals.
    """
    cfg = settings or InitSettings()
    best: dict[tuple[str, str], tuple[int, str]] = {}
    per_pair: dict[tuple[str, str], list[np.ndarray]] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for obs in system.observations:
        counts[(obs.frame, obs.camera, obs.board)] = obs.num_points
    for (frame, camera, board), n in counts.items():
        key = (frame, board)
        if (frame, camera, board) not in pnp:
            continue
        if key not in best or n > best[key][0]:
            best[key] = (n, camera)
        # T_rig_board = inv(T_cam_rig) * T_cam_board
        T_cam_rig = system.cameras[camera].extrinsic
        per_pair.setdefault(key, []).append(
            pose_compose(pose_inverse(T_cam_rig), pnp[(frame, camera, board)])
        )

    for key, (_, camera) in best.items():
        frame, board = key
        if cfg.board_pose_source == "average":
            system.board_poses[key] = _average_poses(per_pair[key], cfg.translation_average)
        else:
            T_cam_rig = system.cameras[camera].extrinsic
            system.board_poses[key] = pose_compose(
                pose_inverse(T_cam_rig), pnp[(frame, camera, board)]
            )

    missing = [k for k in system.frame_board_pairs if k not in system.board_poses]
    if missing:
        raise InitializationError(
            f"could not initialise poses for {len(missing)} (frame, board) pairs, "
            f"first few: {missing[:5]}"
        )


def initialize_system(
    system: CalibrationSystem,
    do_intrinsics: bool = True,
    verbose: bool = False,
    fixed_intrinsics: dict[str, dict[int, float]] | None = None,
    on_progress: Callable[[str], None] | None = None,
    settings: InitSettings | None = None,
) -> dict:
    """
    Run the full bootstrap. Returns a small report dict.

    `fixed_intrinsics` (camera id -> {index: value}) is passed straight to
    `initialize_intrinsics`; the values it names are known, not estimated. The
    later steps see them as ordinary intrinsics, so PnP and the extrinsics are
    computed against the values the user supplied.
    """
    cfg = settings or InitSettings()
    report: dict = {"settings": cfg.to_dict()}
    if do_intrinsics:
        if verbose:
            print("Initialising intrinsics with OpenCV...")
        report["intrinsics_rms"] = initialize_intrinsics(
            system,
            verbose=verbose,
            fixed=fixed_intrinsics,
            on_progress=on_progress,
            settings=cfg,
        )
    elif fixed_intrinsics:
        # Skipping the OpenCV fit does not skip what the user typed: PnP below
        # reads cam.params, so the pins must be in place either way.
        for cid, pins in fixed_intrinsics.items():
            for idx, value in pins.items():
                system.cameras[cid].params[int(idx)] = float(value)
    if on_progress is not None:
        on_progress(f"solving PnP over {len(system.observations)} observations...")
    pnp = solve_pnp_poses(system, settings=cfg)
    report["pnp_solved"] = len(pnp)
    if on_progress is not None:
        on_progress(f"{len(pnp)} PnP pose(s); placing cameras in the rig frame...")
    report["extrinsic_support"] = initialize_extrinsics(system, pnp, settings=cfg)
    if on_progress is not None:
        on_progress("placing the boards...")
    initialize_board_poses(system, pnp, settings=cfg)
    report["board_poses"] = len(system.board_poses)
    if verbose:
        print(
            f"  PnP poses: {report['pnp_solved']}, "
            f"board poses: {report['board_poses']}, "
            f"extrinsic support: {report['extrinsic_support']}"
        )
    return report
