"""
The reprojection residual and the system -> Problem builder.

Residual chain, with every frame named explicitly:

    X_rig = T_rig_board  * X_board
    X_cam = T_cam_rig    * X_rig
    uv    = project(theta_cam, X_cam)
    r     = (uv - uv_observed) / sigma

One residual block covers all corners of one (frame, camera, board) triple.
Grouping this way keeps the number of Python-level callbacks proportional to
the number of images rather than the number of corners, which is roughly a
50x difference on a normal dataset -- and the robust loss is still applied
per corner, inside the block.
"""

from __future__ import annotations

import numpy as np

from mlti_cal.models.camera import CameraModel
from mlti_cal.models.manifolds import (
    IDENTITY_POSE,
    d_point_d_pose_tangent,
    quat_to_matrix,
)
from mlti_cal.problem.graph import EUCLIDEAN, POSE, ParameterBlock, Problem, ResidualBlock
from mlti_cal.problem.loss import Loss, TrivialLoss
from mlti_cal.problem.types import CalibrationSystem


def intr_key(camera_id: str) -> str:
    return f"intr:{camera_id}"


def extr_key(camera_id: str) -> str:
    return f"extr:{camera_id}"


def pose_key(frame: str, board: str) -> str:
    return f"pose:{frame}:{board}"


class ReprojectionResidual(ResidualBlock):
    """Corners of one board, in one camera, in one frame."""

    def __init__(
        self,
        model: CameraModel,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_id: str,
        frame: str,
        board: str,
        sigma: float = 1.0,
        loss: Loss | None = None,
    ):
        self.model = model
        self.X_board = np.asarray(object_points, dtype=float).reshape(-1, 3)
        self.uv_obs = np.asarray(image_points, dtype=float).reshape(-1, 2)
        self.sigma = float(sigma)
        self.loss = loss or TrivialLoss()
        self.camera_id = camera_id
        self.frame = frame
        self.board = board
        self.block_keys = (intr_key(camera_id), extr_key(camera_id), pose_key(frame, board))
        self.dim = 2 * self.X_board.shape[0]
        self.tag = f"{frame}|{camera_id}|{board}"
        #: filled in on each evaluate; the report engine reads them back
        self.last_weights: np.ndarray | None = None
        self.last_behind_camera: int = 0

    @property
    def num_points(self) -> int:
        return self.X_board.shape[0]

    def evaluate(
        self, values: list[np.ndarray], with_jacobians: bool = True
    ) -> tuple[np.ndarray, list[np.ndarray] | None]:
        theta, extr, pose = values
        R_P = quat_to_matrix(pose[3:7])
        t_P = pose[0:3]
        R_E = quat_to_matrix(extr[3:7])
        t_E = extr[0:3]

        X_rig = self.X_board @ R_P.T + t_P
        X_cam = X_rig @ R_E.T + t_E
        self.last_behind_camera = int(np.count_nonzero(~self.model.valid_mask(X_cam)))

        uv, J_theta, J_X = self.model.d_project(theta, X_cam)
        res2 = (uv - self.uv_obs) / self.sigma  # (K,2)

        # Robust weights, per corner, from the CURRENT residual.
        w = self.loss.weights(np.einsum("ij,ij->i", res2, res2))  # (K,)
        self.last_weights = w
        r = (res2 * w[:, None]).ravel()

        if not with_jacobians:
            return r, None

        s = w[:, None, None] / self.sigma
        # d(uv)/d(intrinsics)
        Ji = (J_theta * s).reshape(self.dim, -1)
        # d(uv)/d(extrinsic tangent):  nothing sits left of the extrinsic
        d_extr = d_point_d_pose_tangent(np.eye(3), R_E, X_rig)  # (K,3,6)
        Je = np.einsum("kij,kjl->kil", J_X, d_extr) * s
        # d(uv)/d(board-pose tangent): the extrinsic sits to its left
        d_pose = d_point_d_pose_tangent(R_E, R_P, self.X_board)  # (K,3,6)
        Jp = np.einsum("kij,kjl->kil", J_X, d_pose) * s
        return r, [Ji, Je.reshape(self.dim, 6), Jp.reshape(self.dim, 6)]

    def reprojection_errors(self, values: list[np.ndarray]) -> np.ndarray:
        """Per-corner Euclidean pixel error, UNWEIGHTED and UNWHITENED."""
        theta, extr, pose = values
        R_P, t_P = quat_to_matrix(pose[3:7]), pose[0:3]
        R_E, t_E = quat_to_matrix(extr[3:7]), extr[0:3]
        X_cam = (self.X_board @ R_P.T + t_P) @ R_E.T + t_E
        uv = self.model.project(theta, X_cam)
        return np.linalg.norm(uv - self.uv_obs, axis=1)

    def predicted(self, values: list[np.ndarray]) -> np.ndarray:
        theta, extr, pose = values
        R_P, t_P = quat_to_matrix(pose[3:7]), pose[0:3]
        R_E, t_E = quat_to_matrix(extr[3:7]), extr[0:3]
        X_cam = (self.X_board @ R_P.T + t_P) @ R_E.T + t_E
        return self.model.project(theta, X_cam)


def build_problem(
    system: CalibrationSystem,
    loss: Loss | None = None,
    optimize_intrinsics: bool = True,
    optimize_extrinsics: bool = True,
    optimize_poses: bool = True,
    fixed_intrinsic_components: dict[str, list[int]] | None = None,
) -> Problem:
    """
    Turn a `CalibrationSystem` into a solvable `Problem`.

    Gauge: the reference camera's extrinsic block is marked CONSTANT. That is
    the single gauge fixed here. Scale is metric already, from the board
    geometry, so no scale constraint is added -- adding one would over-constrain
    the problem and quietly bias the covariance.

    Args:
        fixed_intrinsic_components: camera id -> tangent indices to hold fixed,
            e.g. {"cam0": [8]} pins k3. This is the plumbing behind the GUI's
            per-parameter Free/Fixed control.
    """
    problem = Problem()
    fixed_intrinsic_components = fixed_intrinsic_components or {}

    if not system.cameras:
        raise ValueError("system has no cameras")
    if not system.observations:
        raise ValueError("system has no observations")

    ref = system.reference_camera

    for cid, cam in system.cameras.items():
        free_mask = np.ones(cam.model.num_params, dtype=bool)
        for idx in fixed_intrinsic_components.get(cid, []):
            free_mask[idx] = False
        problem.add_block(
            ParameterBlock(
                key=intr_key(cid),
                value=cam.params,
                kind=EUCLIDEAN,
                constant=not optimize_intrinsics,
                free_mask=free_mask,
            )
        )
        is_ref = cid == ref
        problem.add_block(
            ParameterBlock(
                key=extr_key(cid),
                value=IDENTITY_POSE.copy() if is_ref else cam.extrinsic,
                kind=POSE,
                # The reference extrinsic is constant no matter what the caller
                # asked for -- without it the whole rig can translate/rotate
                # freely and J^T J is rank deficient by 6.
                constant=is_ref or not optimize_extrinsics,
            )
        )

    for frame, board in system.frame_board_pairs:
        key = pose_key(frame, board)
        init = system.board_poses.get((frame, board))
        if init is None:
            raise KeyError(
                f"no initial pose for ({frame}, {board}); run "
                f"mlti_cal.problem.initialize.initialize_system first"
            )
        problem.add_block(
            ParameterBlock(key=key, value=init, kind=POSE, constant=not optimize_poses)
        )

    for obs in system.observations:
        cam = system.cameras[obs.camera]
        board = system.boards[obs.board]
        problem.add_residual(
            ReprojectionResidual(
                model=cam.model,
                object_points=board.object_points[obs.point_ids],
                image_points=obs.image_points,
                camera_id=obs.camera,
                frame=obs.frame,
                board=obs.board,
                sigma=obs.sigma,
                loss=loss,
            )
        )
    return problem


def write_back(system: CalibrationSystem, problem: Problem) -> None:
    """Copy optimised block values back into the system description."""
    for cid, cam in system.cameras.items():
        cam.params = problem.blocks[intr_key(cid)].value.copy()
        cam.extrinsic = problem.blocks[extr_key(cid)].value.copy()
    for frame, board in system.frame_board_pairs:
        system.board_poses[(frame, board)] = problem.blocks[pose_key(frame, board)].value.copy()


def snapshot_values(system: CalibrationSystem) -> dict[str, np.ndarray]:
    """
    Read the system's current values out into block-key form.

    The inverse of `write_back`, and the reason it exists: `initialize_system`
    and `write_back` mutate the same `cam.params` / `cam.extrinsic` /
    `board_poses` fields, so a solve destroys the bootstrap estimate unless it
    was copied out first. Copies are deep -- a snapshot that aliased the live
    arrays would be silently rewritten by the very solve it exists to survive.
    """
    out: dict[str, np.ndarray] = {}
    for cid, cam in system.cameras.items():
        out[intr_key(cid)] = np.asarray(cam.params, dtype=float).copy()
        out[extr_key(cid)] = np.asarray(cam.extrinsic, dtype=float).copy()
    for frame, board in system.frame_board_pairs:
        pose = system.board_poses.get((frame, board))
        if pose is not None:
            out[pose_key(frame, board)] = np.asarray(pose, dtype=float).copy()
    return out


def restore_values(system: CalibrationSystem, values: dict[str, np.ndarray]) -> int:
    """
    Push a `snapshot_values` dict back into the system. Returns blocks restored.

    Keys absent from `values` are left alone rather than zeroed: a snapshot
    taken before a cull is a superset, and one taken before a camera was added
    is a subset. Neither should corrupt the system it is restored into.
    """
    n = 0
    for cid, cam in system.cameras.items():
        if (v := values.get(intr_key(cid))) is not None:
            cam.params = np.asarray(v, dtype=float).copy()
            n += 1
        if (v := values.get(extr_key(cid))) is not None:
            cam.extrinsic = np.asarray(v, dtype=float).copy()
            n += 1
    for frame, board in system.frame_board_pairs:
        if (v := values.get(pose_key(frame, board))) is not None:
            system.board_poses[(frame, board)] = np.asarray(v, dtype=float).copy()
            n += 1
    return n
