"""
The calibration system description: cameras, boards, frames, observations.

Coordinate conventions -- stated explicitly:

  * Camera frame is OpenCV: +x right, +y down, +z forward into the scene.
  * A pose named `T_a_b` maps points from frame b into frame a:
        X_a = R(T_a_b) X_b + t(T_a_b)
  * `Camera.extrinsic` is  T_cam_rig  -- rig frame into that camera's frame.
  * `BoardPose` is  T_rig_board     -- board frame into the rig frame.
  * Therefore a board feature reaches the image as
        X_cam = T_cam_rig * T_rig_board * X_board
  * The reference camera holds T_cam_rig = identity and is CONSTANT. That is
    the only gauge fixed by construction; metric scale comes from the board's
    physical square size, so unlike SfM there is no scale gauge freedom.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlti_cal.models.camera import CameraModel, get_model
from mlti_cal.models.manifolds import IDENTITY_POSE


@dataclass
class Board:
    """A calibration object: rigid 3D features in its own frame."""

    id: str
    object_points: np.ndarray  # (M,3) float, indexed by point id
    kind: str = "charuco"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.object_points = np.asarray(self.object_points, dtype=float).reshape(-1, 3)

    @property
    def num_points(self) -> int:
        return self.object_points.shape[0]


@dataclass
class Camera:
    """One physical camera: a projection model plus its pose in the rig."""

    id: str
    model_name: str
    image_size: tuple[int, int]  # (width, height)
    params: np.ndarray | None = None
    extrinsic: np.ndarray | None = None  # T_cam_rig, 7-vector
    is_reference: bool = False

    def __post_init__(self) -> None:
        model = self.model
        if self.params is None:
            self.params = model.default_params(self.image_size)
        else:
            self.params = np.asarray(self.params, dtype=float).copy()
            if self.params.size != model.num_params:
                raise ValueError(
                    f"camera {self.id}: model {self.model_name} wants "
                    f"{model.num_params} params, got {self.params.size}"
                )
        if self.extrinsic is None:
            self.extrinsic = IDENTITY_POSE.copy()
        else:
            self.extrinsic = np.asarray(self.extrinsic, dtype=float).copy()

    @property
    def model(self) -> CameraModel:
        return get_model(self.model_name)


@dataclass
class Observation:
    """
    One board seen by one camera in one frame.

    `point_ids` index into `Board.object_points`; `image_points` are the
    matching detected pixels. `sigma` is the assumed per-coordinate pixel noise
    std used to whiten this observation's residual.
    """

    frame: str
    camera: str
    board: str
    point_ids: np.ndarray
    image_points: np.ndarray
    sigma: float = 1.0

    def __post_init__(self) -> None:
        self.point_ids = np.asarray(self.point_ids, dtype=int).ravel()
        self.image_points = np.asarray(self.image_points, dtype=float).reshape(-1, 2)
        if self.point_ids.size != self.image_points.shape[0]:
            raise ValueError(
                f"{self.frame}/{self.camera}/{self.board}: "
                f"{self.point_ids.size} ids vs {self.image_points.shape[0]} points"
            )

    @property
    def num_points(self) -> int:
        return self.point_ids.size


@dataclass
class CalibrationSystem:
    """Cameras + boards + observations, plus the estimated board poses."""

    cameras: dict[str, Camera] = field(default_factory=dict)
    boards: dict[str, Board] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    # (frame, board) -> T_rig_board 7-vector
    board_poses: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    def add_camera(self, camera: Camera) -> Camera:
        """
        Add a camera, maintaining the invariant that EXACTLY ONE is reference.

        The order of these branches matters. Unconditionally flagging the first
        camera added -- the obvious implementation -- silently discards the
        caller's choice: a config marking cam2 as reference would leave both
        cam0 and cam2 flagged, `reference_camera` would return cam0, and the
        rig origin would quietly be the wrong camera. No error, no rank
        deficiency, just extrinsics reported against something the user did not
        ask for.
        """
        if camera.id in self.cameras:
            raise ValueError(f"duplicate camera id {camera.id!r}")
        self.cameras[camera.id] = camera
        if camera.is_reference:
            self.set_reference_camera(camera.id)  # clears any previous one
        elif not any(c.is_reference for c in self.cameras.values()):
            camera.is_reference = True  # first camera defaults to the rig frame
        return camera

    def add_board(self, board: Board) -> Board:
        if board.id in self.boards:
            raise ValueError(f"duplicate board id {board.id!r}")
        self.boards[board.id] = board
        return board

    def add_observation(self, obs: Observation) -> Observation:
        if obs.camera not in self.cameras:
            raise KeyError(f"observation references unknown camera {obs.camera!r}")
        if obs.board not in self.boards:
            raise KeyError(f"observation references unknown board {obs.board!r}")
        max_id = self.boards[obs.board].num_points
        if obs.point_ids.size and (obs.point_ids.max() >= max_id or obs.point_ids.min() < 0):
            raise ValueError(
                f"observation {obs.frame}/{obs.camera}/{obs.board} has point ids "
                f"outside [0,{max_id})"
            )
        self.observations.append(obs)
        return obs

    # -- queries -----------------------------------------------------------
    @property
    def reference_camera(self) -> str:
        refs = [cid for cid, cam in self.cameras.items() if cam.is_reference]
        if not refs:
            raise ValueError("no reference camera set")
        if len(refs) > 1:
            # Never silently pick one: the gauge would be fixed on a camera the
            # caller did not choose, and every extrinsic would be reported
            # against the wrong origin.
            raise ValueError(
                f"{len(refs)} cameras are marked as reference ({', '.join(refs)}). "
                f"Exactly one camera defines the rig frame; use "
                f"set_reference_camera() to choose."
            )
        return refs[0]

    def set_reference_camera(self, camera_id: str) -> None:
        if camera_id not in self.cameras:
            raise KeyError(camera_id)
        for cid, cam in self.cameras.items():
            cam.is_reference = cid == camera_id

    @property
    def frames(self) -> list[str]:
        seen: dict[str, None] = {}
        for o in self.observations:
            seen.setdefault(o.frame, None)
        return list(seen)

    @property
    def frame_board_pairs(self) -> list[tuple[str, str]]:
        seen: dict[tuple[str, str], None] = {}
        for o in self.observations:
            seen.setdefault((o.frame, o.board), None)
        return list(seen)

    def observations_for(self, frame: str, camera: str | None = None) -> list[Observation]:
        return [
            o
            for o in self.observations
            if o.frame == frame and (camera is None or o.camera == camera)
        ]

    @property
    def num_observed_points(self) -> int:
        return int(sum(o.num_points for o in self.observations))

    def summary(self) -> dict:
        return {
            "cameras": len(self.cameras),
            "boards": len(self.boards),
            "frames": len(self.frames),
            "observations": len(self.observations),
            "observed_points": self.num_observed_points,
            "reference_camera": self.reference_camera if self.cameras else None,
        }
