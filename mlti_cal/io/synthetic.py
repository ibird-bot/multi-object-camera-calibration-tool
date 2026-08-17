"""
Synthetic multi-camera / multi-board dataset generation with ground truth.

This exists so the honesty claims can actually be tested. Without a dataset
whose true intrinsics, true extrinsics and true noise level are known, a
statement like "the reported uncertainty brackets the real error" is
unfalsifiable -- and an unfalsifiable uncertainty claim is precisely the
failure mode this project is built to expose.

Everything is generated in metres and pixels; board geometry is metric, so the
recovered translations are metric too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlti_cal.models.camera import get_model
from mlti_cal.models.manifolds import (
    IDENTITY_POSE,
    matrix_to_quat,
    pose_inverse,
    quat_to_matrix,
    so3_exp,
)
from mlti_cal.problem.types import Board, CalibrationSystem, Camera, Observation

#: Ground-truth distortion per model, for the models OpenCV cannot fit.
#:
#: Values sit in each model's own well-behaved range rather than at a token
#: small number: a synthetic camera with near-zero distortion cannot show
#: whether the model recovers distortion at all, which is the one thing an
#: end-to-end test of a new model needs to prove.
_TRUTH_DISTORTION: dict[str, tuple[float, ...]] = {
    "double_sphere": (-0.17, 0.56),  # xi, alpha
    "eucm": (0.60, 1.10),  # alpha, beta
    "fov": (0.85,),  # w, radians
    "halcon_division": (-0.10,),  # kappa
    "matlab": (0.0, -0.26, 0.09, -0.015, 8e-4, -6e-4),  # skew, k1..k3, p1, p2
    "thin_prism": (
        -0.26,
        0.09,
        8e-4,
        -6e-4,
        -0.015,
        1e-3,
        -4e-4,
        1e-4,
        2e-4,
        -8e-5,
        1.5e-4,
        -1e-4,
    ),  # fmt: skip
}

#: Focal length appropriate to each model's geometry. The projective models
#: (double sphere, EUCM, FOV) compress a wide field into the same sensor, so a
#: pinhole-sized focal length would put every board corner off the image.
_TRUTH_FOCAL: dict[str, float] = {
    "double_sphere": 360.0,
    "eucm": 370.0,
    "fov": 520.0,
    "halcon_division": 820.0,
    "matlab": 900.0,
    "thin_prism": 900.0,
}


def _truth_for(model_name: str, w: int, h: int, rng) -> np.ndarray:
    """Ground-truth intrinsics for a model with no OpenCV reference fit."""
    if model_name not in _TRUTH_DISTORTION:
        raise KeyError(
            f"no synthetic ground truth for camera model {model_name!r}; add one to "
            f"_TRUTH_DISTORTION so the model can be exercised end to end"
        )
    f = _TRUTH_FOCAL[model_name]
    return np.array(
        [
            f + 0.02 * f * rng.standard_normal(),
            f + 0.02 * f * rng.standard_normal(),
            w / 2 + 8.0 * rng.standard_normal(),
            h / 2 + 8.0 * rng.standard_normal(),
            *_TRUTH_DISTORTION[model_name],
        ]
    )


def charuco_object_points(squares_x: int, squares_y: int, square_length: float) -> np.ndarray:
    """
    Interior chessboard corners of a Charuco board, in the board frame.

    Matches OpenCV's CharucoBoard.getChessboardCorners() ordering: row-major
    from the top-left interior corner, z = 0, x right, y down.
    """
    xs = np.arange(1, squares_x) * square_length
    ys = np.arange(1, squares_y) * square_length
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])


@dataclass
class GroundTruth:
    """Everything the solver is not allowed to see."""

    camera_params: dict[str, np.ndarray] = field(default_factory=dict)
    camera_extrinsics: dict[str, np.ndarray] = field(default_factory=dict)
    board_poses: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    pixel_noise_std: float = 0.0

    def to_dict(self) -> dict:
        return {
            "camera_params": {k: v.tolist() for k, v in self.camera_params.items()},
            "camera_extrinsics": {k: v.tolist() for k, v in self.camera_extrinsics.items()},
            "board_poses": {f"{f}|{b}": p.tolist() for (f, b), p in self.board_poses.items()},
            "pixel_noise_std": self.pixel_noise_std,
        }


def _look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, -1.0, 0.0)) -> np.ndarray:
    """T_cam_world for a camera at `eye` looking at `target` (OpenCV axes)."""
    f = target - eye
    f = f / np.linalg.norm(f)
    up = np.asarray(up, dtype=float)
    if abs(f @ (up / np.linalg.norm(up))) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R_wc = np.column_stack([r, d, f])  # camera axes in world
    R_cw = R_wc.T
    return np.concatenate([-R_cw @ eye, matrix_to_quat(R_cw)])


def generate_dataset(
    num_cameras: int = 2,
    num_frames: int = 14,
    boards_spec: tuple[tuple[str, int, int, float], ...] = (("board_A", 9, 7, 0.03),),
    model_name: str = "pinhole_radtan",
    image_size: tuple[int, int] = (1280, 720),
    pixel_noise_std: float = 0.3,
    outlier_fraction: float = 0.0,
    outlier_magnitude: float = 12.0,
    seed: int = 7,
    init_perturbation: bool = True,
) -> tuple[CalibrationSystem, GroundTruth]:
    """
    Build a system whose observations come from known parameters.

    The returned system carries a deliberately WRONG starting guess (crude
    intrinsics, identity-ish extrinsics, PnP-free board poses set to None) so
    that a solve has real work to do. `init_perturbation=False` starts at
    ground truth, which is only useful for isolating solver behaviour.
    """
    rng = np.random.default_rng(seed)
    model = get_model(model_name)
    w, h = image_size

    # ---- ground-truth cameras -------------------------------------------
    gt = GroundTruth(pixel_noise_std=pixel_noise_std)
    system = CalibrationSystem()

    for i in range(num_cameras):
        cid = f"cam{i}"
        if model_name == "pinhole_radtan":
            true_params = np.array(
                [
                    900.0 + 40.0 * rng.standard_normal(),
                    900.0 + 40.0 * rng.standard_normal(),
                    w / 2 + 12.0 * rng.standard_normal(),
                    h / 2 + 12.0 * rng.standard_normal(),
                    -0.26 + 0.03 * rng.standard_normal(),
                    0.09 + 0.02 * rng.standard_normal(),
                    8e-4 * rng.standard_normal(),
                    8e-4 * rng.standard_normal(),
                    -0.015 + 5e-3 * rng.standard_normal(),
                ]
            )
        elif model_name == "fisheye_kb":
            true_params = np.array(
                [
                    420.0 + 15.0 * rng.standard_normal(),
                    420.0 + 15.0 * rng.standard_normal(),
                    w / 2 + 8.0 * rng.standard_normal(),
                    h / 2 + 8.0 * rng.standard_normal(),
                    -0.02,
                    3e-3,
                    -1e-3,
                    2e-4,
                ]
            )
        else:
            true_params = _truth_for(model_name, w, h, rng)
        gt.camera_params[cid] = true_params

        # cam0 defines the rig; the others sit on a baseline looking inward.
        if i == 0:
            extr = IDENTITY_POSE.copy()
        else:
            baseline = 0.18 * i
            eye = np.array([baseline, 0.02 * rng.standard_normal(), 0.0])
            extr = _look_at(eye, np.array([0.0, 0.0, 1.4]))
        gt.camera_extrinsics[cid] = extr

        start_params = true_params.copy()
        if init_perturbation:
            start_params = model.default_params(image_size)
        system.add_camera(
            Camera(
                id=cid,
                model_name=model_name,
                image_size=image_size,
                params=start_params,
                extrinsic=IDENTITY_POSE.copy() if init_perturbation else extr.copy(),
                is_reference=(i == 0),
            )
        )

    for bid, sx, sy, sq in boards_spec:
        system.add_board(
            Board(
                id=bid,
                object_points=charuco_object_points(sx, sy, sq),
                kind="charuco",
                meta={"squares_x": sx, "squares_y": sy, "square_length": sq},
            )
        )

    # ---- frames ----------------------------------------------------------
    n_obs_total = 0
    for f in range(num_frames):
        frame = f"frame{f:03d}"
        for bi, (bid, sx, sy, sq) in enumerate(boards_spec):
            # Spread boards over depth, lateral offset and tilt so the problem
            # is well conditioned; a board that never tilts leaves focal length
            # and distance almost unidentifiable.
            cx_off = 0.10 * rng.standard_normal() + 0.12 * bi
            cy_off = 0.08 * rng.standard_normal()
            depth = rng.uniform(0.85, 1.9)
            tilt = np.array(
                [
                    rng.uniform(-0.55, 0.55),
                    rng.uniform(-0.55, 0.55),
                    rng.uniform(-0.35, 0.35),
                ]
            )
            R = so3_exp(tilt)
            board_extent = np.array([(sx - 1) * sq, (sy - 1) * sq, 0.0]) * 0.5
            t = np.array([cx_off, cy_off, depth]) - R @ board_extent
            pose = np.concatenate([t, matrix_to_quat(R)])
            gt.board_poses[(frame, bid)] = pose

            board_pts = system.boards[bid].object_points
            X_rig = board_pts @ R.T + t

            for cid in system.cameras:
                E = gt.camera_extrinsics[cid]
                R_E, t_E = quat_to_matrix(E[3:7]), E[0:3]
                X_cam = X_rig @ R_E.T + t_E
                valid = X_cam[:, 2] > 1e-3
                if not valid.any():
                    continue
                uv = get_model(model_name).project(gt.camera_params[cid], X_cam)
                inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                # A board is only usable if a decent chunk of it is visible.
                if inside.sum() < 8:
                    continue
                ids = np.flatnonzero(inside)
                pts = uv[ids] + rng.normal(scale=pixel_noise_std, size=(ids.size, 2))
                if outlier_fraction > 0:
                    n_out = rng.binomial(ids.size, outlier_fraction)
                    if n_out:
                        pick = rng.choice(ids.size, size=n_out, replace=False)
                        pts[pick] += (
                            rng.normal(scale=outlier_magnitude, size=(n_out, 2))
                            + np.sign(rng.standard_normal((n_out, 2))) * outlier_magnitude
                        )
                system.add_observation(
                    Observation(
                        frame=frame,
                        camera=cid,
                        board=bid,
                        point_ids=ids,
                        image_points=pts,
                        sigma=1.0,
                    )
                )
                n_obs_total += 1

    if n_obs_total == 0:
        raise RuntimeError("synthetic generator produced no visible observations")
    return system, gt


def seed_ground_truth_poses(system: CalibrationSystem, gt: GroundTruth) -> None:
    """Fill the system's board poses / extrinsics with the true values."""
    for cid, cam in system.cameras.items():
        cam.params = gt.camera_params[cid].copy()
        cam.extrinsic = gt.camera_extrinsics[cid].copy()
    for key, pose in gt.board_poses.items():
        if key in dict.fromkeys(system.frame_board_pairs):
            system.board_poses[key] = pose.copy()


def relative_extrinsic_error(estimated: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """
    (translation error in metres, rotation error in degrees) between two poses.

    Compared as a relative transform rather than componentwise, because
    componentwise quaternion differences are meaningless across the double
    cover.
    """
    from mlti_cal.models.manifolds import pose_compose, so3_log

    rel = pose_compose(pose_inverse(truth), estimated)
    t_err = float(np.linalg.norm(rel[0:3]))
    r_err = float(np.degrees(np.linalg.norm(so3_log(quat_to_matrix(rel[3:7])))))
    return t_err, r_err
