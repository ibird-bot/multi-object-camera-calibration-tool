"""
Config loading and result export.

A calibration config declares the cameras (the `+` list the GUI edits) and the
boards, and points at the images. Everything else is derived.

Export writes OpenCV-compatible YAML so results drop straight into
`cv2.undistort` / `cv2.initUndistortRectifyMap` without anyone re-deriving the
parameter order.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mlti_cal.detectors.charuco import CharucoBoardSpec, MultiBoardDetector
from mlti_cal.models.manifolds import pose_to_rt, quat_to_matrix
from mlti_cal.problem.types import Board, CalibrationSystem, Camera, Observation

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class CameraConfig:
    id: str
    model: str = "pinhole_radtan"
    image_dir: str = ""
    image_size: list[int] | None = None
    is_reference: bool = False


@dataclass
class CalibrationConfig:
    """The whole project definition."""

    cameras: list[CameraConfig] = field(default_factory=list)
    boards: list[dict] = field(default_factory=list)
    pixel_noise_std: float = 0.3
    name: str = "calibration"

    @staticmethod
    def load(path: str | Path) -> CalibrationConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return CalibrationConfig(
            cameras=[CameraConfig(**c) for c in data.get("cameras", [])],
            boards=data.get("boards", []),
            pixel_noise_std=float(data.get("pixel_noise_std", 0.3)),
            name=data.get("name", "calibration"),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "pixel_noise_std": self.pixel_noise_std,
                    "cameras": [asdict(c) for c in self.cameras],
                    "boards": self.boards,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return p

    def board_specs(self) -> list[CharucoBoardSpec]:
        return [CharucoBoardSpec(**b) for b in self.boards]


def find_images(directory: str | Path) -> list[Path]:
    """Images in a directory, sorted by name -- the name IS the frame key."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"image directory not found: {d}")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def build_system_from_config(
    config: CalibrationConfig, verbose: bool = False
) -> tuple[CalibrationSystem, dict]:
    """
    Run detection over every camera's images and assemble a CalibrationSystem.

    Frames are matched across cameras BY FILENAME STEM. Multi-camera calibration
    requires synchronised capture, and the filename is the only synchronisation
    signal available from a directory of images -- so a mismatch is reported
    rather than silently producing a rig where cameras see different instants.
    """
    specs = config.board_specs()
    detector = MultiBoardDetector(specs)  # validates ID collisions
    system = CalibrationSystem()

    for spec in specs:
        system.add_board(
            Board(
                id=spec.id,
                object_points=detector.object_points(spec.id),
                kind="charuco",
                meta={
                    "squares_x": spec.squares_x,
                    "squares_y": spec.squares_y,
                    "square_length": spec.square_length,
                    "dictionary": spec.dictionary,
                },
            )
        )

    stats = {"per_camera": {}, "frames": {}, "skipped": []}
    frames_by_camera: dict[str, set[str]] = {}

    for cam_cfg in config.cameras:
        images = find_images(cam_cfg.image_dir)
        if not images:
            raise ValueError(f"camera {cam_cfg.id}: no images in {cam_cfg.image_dir}")
        first = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
        if first is None:
            raise ValueError(f"camera {cam_cfg.id}: cannot read {images[0]}")
        h, w = first.shape[:2]
        size = tuple(cam_cfg.image_size) if cam_cfg.image_size else (w, h)

        system.add_camera(
            Camera(
                id=cam_cfg.id,
                model_name=cam_cfg.model,
                image_size=size,
                is_reference=cam_cfg.is_reference,
            )
        )

        found_frames: set[str] = set()
        n_det = 0
        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                stats["skipped"].append(str(img_path))
                continue
            if (img.shape[1], img.shape[0]) != size:
                raise ValueError(
                    f"camera {cam_cfg.id}: {img_path.name} is "
                    f"{img.shape[1]}x{img.shape[0]} but the camera is declared "
                    f"{size[0]}x{size[1]}. Mixed resolutions cannot share intrinsics."
                )
            frame = img_path.stem
            for det in detector.detect_all(img):
                system.add_observation(
                    Observation(
                        frame=frame,
                        camera=cam_cfg.id,
                        board=det.board_id,
                        point_ids=det.point_ids,
                        image_points=det.image_points,
                        sigma=config.pixel_noise_std,
                    )
                )
                found_frames.add(frame)
                n_det += 1
            if verbose:
                print(f"  {cam_cfg.id}/{img_path.name}: {n_det} detections so far")
        frames_by_camera[cam_cfg.id] = found_frames
        stats["per_camera"][cam_cfg.id] = {
            "images": len(images),
            "detections": n_det,
            "frames_with_detections": len(found_frames),
        }

    if len(frames_by_camera) > 1:
        shared = set.intersection(*frames_by_camera.values())
        stats["shared_frames"] = len(shared)
        if not shared:
            raise ValueError(
                "No frame name is common to all cameras. Frames are matched by "
                "filename stem; rename synchronised captures to share a stem "
                "(e.g. cam0/0001.png and cam1/0001.png)."
            )
    if any(c.is_reference for c in config.cameras) is False and system.cameras:
        system.set_reference_camera(next(iter(system.cameras)))
    return system, stats


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_opencv_yaml(system: CalibrationSystem, path: str | Path) -> Path:
    """OpenCV FileStorage YAML: K, distCoeffs and extrinsics per camera."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_WRITE)
    try:
        fs.write("reference_camera", system.reference_camera)
        fs.write("num_cameras", len(system.cameras))
        for cid, cam in system.cameras.items():
            safe = cid.replace("-", "_").replace(" ", "_")
            fs.write(f"{safe}_model", cam.model_name)
            fs.write(f"{safe}_image_width", int(cam.image_size[0]))
            fs.write(f"{safe}_image_height", int(cam.image_size[1]))
            fs.write(f"{safe}_K", cam.model.matrix(cam.params))
            fs.write(f"{safe}_dist", cam.model.distortion(cam.params).reshape(1, -1))
            rvec, tvec = pose_to_rt(cam.extrinsic)
            fs.write(f"{safe}_rvec", rvec.reshape(3, 1))
            fs.write(f"{safe}_tvec", tvec.reshape(3, 1))
            fs.write(f"{safe}_R", quat_to_matrix(cam.extrinsic[3:7]))
    finally:
        fs.release()
    return p


def export_json(system: CalibrationSystem, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "reference_camera": system.reference_camera,
        "cameras": {
            cid: {
                "model": cam.model_name,
                "image_size": list(cam.image_size),
                "parameter_names": list(cam.model.param_names),
                "parameters": cam.params.tolist(),
                "extrinsic_T_cam_rig": cam.extrinsic.tolist(),
                "K": cam.model.matrix(cam.params).tolist(),
                "distortion": cam.model.distortion(cam.params).tolist(),
            }
            for cid, cam in system.cameras.items()
        },
        "boards": {
            bid: {"num_points": b.num_points, "kind": b.kind, "meta": b.meta}
            for bid, b in system.boards.items()
        },
        "board_poses": {
            f"{f}|{b}": np.asarray(p_).tolist() for (f, b), p_ in system.board_poses.items()
        },
    }
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p
