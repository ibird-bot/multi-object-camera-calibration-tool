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
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mlti_cal.detectors.base import Detection
from mlti_cal.detectors.pipeline import BoardDetectors
from mlti_cal.detectors.registry import (
    default_detector_settings,
    detector_settings_from_dict,
    detector_settings_to_dict,
    get_kind,
    implemented_kinds,
)
from mlti_cal.models.manifolds import pose_to_rt, quat_to_matrix
from mlti_cal.problem.settings import InitSettings
from mlti_cal.problem.types import Board, CalibrationSystem, Camera, Observation

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: See the note in `problem/initialize.py`: this is a library, so per-image
#: progress goes to a logger the caller can route, not to stdout.
log = logging.getLogger(__name__)


#: Bumped only when a config written by this version would be MISREAD by an
#: older one -- a renamed key, a changed unit, a changed default that alters a
#: result. Adding an optional key does not need a bump, because an old reader
#: ignoring it still reproduces the same calibration. Version 1 is the format as
#: it stood at 0.1.0; files written before the field existed are read as 1.
CONFIG_VERSION = 1

#: Every key `load` understands. `detection` is the pre-multi-detector spelling
#: of `detectors`, still read for backward compatibility, so it belongs here
#: even though nothing writes it any more.
KNOWN_TOP_LEVEL_KEYS = {
    "version",
    "name",
    "pixel_noise_std",
    "cameras",
    "boards",
    "detectors",
    "detection",
    "initialization",
}


class ConfigError(ValueError):
    """
    A config file that cannot be used, reported with the offending key.

    Its own type so callers can tell "your file is wrong" from "the calibration
    failed": the GUI shows the first as a fixable dialog, and a batch run should
    exit on it immediately rather than after decoding every image.
    """


@dataclass
class CameraConfig:
    id: str
    model: str = "pinhole_radtan"
    image_dir: str = ""
    image_size: list[int] | None = None
    is_reference: bool = False


def _load_cameras(data: dict) -> list[CameraConfig]:
    """
    The `cameras` list, validated by hand rather than by `CameraConfig(**c)`.

    Splatting raw JSON into the dataclass is what this replaces. A typo gave
    `TypeError: __init__() got an unexpected keyword argument 'image_dirs'` --
    which names neither the file nor the camera, and points at a constructor the
    user has never heard of. A missing `id` gave an even worse one. The failure
    is the same failure; only the message changes, and the message is the whole
    point.
    """
    raw = data.get("cameras", [])
    if not isinstance(raw, list):
        raise ConfigError(f"'cameras' must be a list, got {type(raw).__name__}")

    known = {f.name for f in fields(CameraConfig)}
    cameras: list[CameraConfig] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"camera #{i}: expected an object, got {type(entry).__name__}")
        where = entry.get("id") or f"#{i}"
        unknown = sorted(set(entry) - known)
        if unknown:
            raise ConfigError(
                f"camera {where!r}: unknown field(s) {unknown}; a camera takes {sorted(known)}"
            )
        if not entry.get("id"):
            raise ConfigError(f"camera #{i}: missing 'id', which every camera needs")
        if entry["id"] in seen:
            raise ConfigError(
                f"camera {entry['id']!r} is declared twice; ids identify the "
                f"camera in every result file, so they must be unique"
            )
        seen.add(entry["id"])
        cameras.append(CameraConfig(**entry))

    refs = [c.id for c in cameras if c.is_reference]
    if len(refs) > 1:
        # Caught here rather than in `CalibrationSystem.add_camera`, which would
        # raise the same objection several hundred decoded images later.
        raise ConfigError(
            f"{len(refs)} cameras are marked 'is_reference' ({', '.join(refs)}). "
            f"Exactly one camera defines the rig frame."
        )
    return cameras


def _validate_boards(boards: object) -> list[dict]:
    """
    The `boards` list, checked at LOAD time instead of at first detection.

    Every check here already existed somewhere -- `board_kind` knows the kinds,
    `DetectorKind.build_spec` knows the fields. What it did not have was a
    moment early enough to be useful: they ran from `detectors_for_run()`, so a
    misspelled `squares_x` surfaced only once the user pressed Run detection.
    Building each spec here is the cheapest way to reuse that validation rather
    than restate it, and it throws the same errors it always did.
    """
    if not isinstance(boards, list):
        raise ConfigError(f"'boards' must be a list, got {type(boards).__name__}")

    seen: set[str] = set()
    for i, board in enumerate(boards):
        if not isinstance(board, dict):
            raise ConfigError(f"board #{i}: expected an object, got {type(board).__name__}")
        if not board.get("id"):
            raise ConfigError(f"board #{i}: missing 'id', which every board needs")
        if board["id"] in seen:
            raise ConfigError(
                f"board {board['id']!r} is declared twice; ids name the board in "
                f"the report and in every exported pose, so they must be unique"
            )
        seen.add(board["id"])
        try:
            board_spec(board)  # validates kind and every field name
        except (ValueError, TypeError) as exc:
            raise ConfigError(str(exc)) from exc
    return boards


def _load_detectors(data: dict) -> dict:
    """
    Per-kind detector settings from raw JSON.

    Accepts the older flat `"detection"` block, which predates there being more
    than one detector kind, and files it under charuco -- which is what it
    always described.
    """
    if "detectors" in data:
        return detector_settings_from_dict(data["detectors"])
    if "detection" in data:
        return detector_settings_from_dict({"charuco": data["detection"]})
    return default_detector_settings()


def _load_noise(data: dict) -> float | None:
    """`pixel_noise_std` from raw JSON: absent -> default, null -> unknown."""
    if "pixel_noise_std" not in data:
        return 0.3
    value = data["pixel_noise_std"]
    return None if value is None else float(value)


def board_kind(board: dict) -> str:
    """
    Which detector a board dict asks for.

    Defaults to charuco, because every config written before there was a second
    kind describes a charuco board and must keep loading unchanged.

    Validated against the REGISTRY rather than a table kept here, so a board
    naming a third-party kind loads wherever that plugin is installed -- and
    fails with the list of what IS available where it is not.
    """
    kind = board.get("kind", "charuco")
    known = {k.id for k in implemented_kinds()}
    if kind not in known:
        raise ValueError(
            f"board {board.get('id', '?')!r}: unknown kind {kind!r}; "
            f"available kinds are {sorted(known)}"
        )
    return kind


def board_spec(board: dict):
    """One board dict as the spec its kind calls for."""
    return get_kind(board_kind(board)).build_spec(board)


def specs_by_kind(boards: list[dict]) -> dict[str, list]:
    """Every board grouped by kind id, in declaration order."""
    out: dict[str, list] = {}
    for board in boards:
        out.setdefault(board_kind(board), []).append(board_spec(board))
    return out


@dataclass
class CalibrationConfig:
    """The whole project definition."""

    cameras: list[CameraConfig] = field(default_factory=list)
    boards: list[dict] = field(default_factory=list)
    #: Assumed per-coordinate corner noise in pixels, or None for "unknown".
    #: None is a real answer, not a missing value: you often do not know it, and
    #: inventing 0.3 px would put a number you never measured into the whitening
    #: and into the covariance cross-check. When it is None the residuals are
    #: left unweighted and the covariance takes its sigma from the residuals,
    #: which it already does by default.
    pixel_noise_std: float | None = 0.3
    name: str = "calibration"
    #: Detector knobs, keyed by detector kind ("charuco", ...). Per KIND
    #: because the settings are properties of a detector, not of detection in
    #: general: `error_correction_rate` is meaningless to a checkerboard.
    #: Saved in full -- including values left at their defaults -- because a
    #: config recording only the overrides silently changes meaning when a
    #: default changes, and a calibration you cannot reproduce is not a result.
    detectors: dict = field(default_factory=default_detector_settings)
    initialization: InitSettings = field(default_factory=InitSettings)

    #: Format version of the file this config came from. Written back out
    #: unchanged on save so a round-trip does not silently relabel a file.
    version: int = CONFIG_VERSION

    @staticmethod
    def load(path: str | Path) -> CalibrationConfig:
        """
        Read and VALIDATE a config file.

        Validation happens here, not at first use. The cost of the old order was
        not the error itself but when it arrived: a mistyped field surfaced after
        detection had decoded every image of every camera, and a duplicate board
        id never surfaced at all -- the second board silently replaced the first
        in the detector map and the calibration ran on fewer targets than the
        file described. This module already says a calibration you cannot
        reproduce is not a result; a file that means something other than what it
        says is the same problem one step earlier.
        """
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{p}: not valid JSON -- {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{p}: expected a JSON object, got {type(data).__name__}")

        # Absent means a file written before the field existed, which is
        # format 1 by definition -- not an error.
        version = data.get("version", CONFIG_VERSION)
        if not isinstance(version, int) or version < 1:
            raise ConfigError(f"{p}: 'version' must be a positive integer, got {version!r}")
        if version > CONFIG_VERSION:
            raise ConfigError(
                f"{p}: config format version {version}, but this build understands "
                f"up to {CONFIG_VERSION}. Upgrade mlti-object-cal to read it -- "
                f"loading it here would silently ignore whatever is new in it."
            )

        # Top-level keys too, not just the ones inside `cameras` and `boards`.
        # A file saying "pixel_noise" instead of "pixel_noise_std" was accepted
        # in silence and calibrated against the 0.3 default -- the same failure
        # the per-camera check exists to stop, one level up, and harder to spot
        # because the run succeeds and only the uncertainty is wrong.
        unknown = sorted(set(data) - KNOWN_TOP_LEVEL_KEYS)
        if unknown:
            raise ConfigError(
                f"{p}: unknown top-level key(s) {unknown}; "
                f"a config takes {sorted(KNOWN_TOP_LEVEL_KEYS)}"
            )

        return CalibrationConfig(
            version=version,
            cameras=_load_cameras(data),
            boards=_validate_boards(data.get("boards", [])),
            # Three cases, and they are genuinely different: key absent means an
            # older config that predates the choice, so it keeps the old default;
            # an explicit null means the user said they do not know; a number
            # means they do.
            pixel_noise_std=_load_noise(data),
            name=data.get("name", "calibration"),
            detectors=_load_detectors(data),
            initialization=InitSettings.from_dict(data.get("initialization")),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    # First key on purpose: a reader that has to decide whether
                    # it can understand the file should not have to parse the
                    # whole thing to find out.
                    "version": self.version,
                    "name": self.name,
                    "pixel_noise_std": self.pixel_noise_std,
                    "cameras": [asdict(c) for c in self.cameras],
                    "boards": self.boards,
                    "detectors": detector_settings_to_dict(self.detectors),
                    "initialization": self.initialization.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return p

    def settings_for(self, kind_id: str):
        """This run's settings for one detector kind."""
        return self.detectors[kind_id]

    @property
    def charuco(self):
        """Charuco detector settings. Kept because the noise tools take one."""
        return self.detectors["charuco"]

    def specs_by_kind(self) -> dict[str, list]:
        """Board specs grouped by kind id, in declaration order."""
        return specs_by_kind(self.boards)

    def board_specs(self) -> list:
        """
        Every board spec, mixed kinds, in declaration order.

        Returning a mixed list keeps it from silently hiding a board it does not
        understand -- which is exactly what returning only the Charuco ones would
        have done once a second kind landed.
        """
        return [board_spec(b) for b in self.boards]

    def detectors_for_run(self) -> BoardDetectors:
        """
        Every configured kind's detector, built and cross-validated.

        Raises on any rig that cannot give an unambiguous answer -- Charuco ID
        collisions, two identically sized plain checkerboards -- so a bad setup
        fails here rather than after every image has been searched.
        """
        return BoardDetectors.build(self.specs_by_kind(), self.detectors)


@dataclass
class ImageProgress:
    """
    One detection event, reported per image as the run proceeds.

    Deliberately a plain dataclass and a plain callable: this is the core, and
    the core imports no Qt. The GUI adapts it to signals at its own boundary.

    `image` is the grayscale frame that was just searched, handed over so a
    viewer can draw the corners without decoding the file a second time. It is
    None only when the file could not be read.
    """

    camera: str
    path: Path
    stage: str  # "start" before the file is read, "done" after it is searched
    image: np.ndarray | None = None
    detections: tuple[Detection, ...] = ()

    @property
    def num_corners(self) -> int:
        return sum(d.num_points for d in self.detections)


def find_images(directory: str | Path) -> list[Path]:
    """Images in a directory, sorted by name -- the name IS the frame key."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"image directory not found: {d}")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def build_system_from_config(
    config: CalibrationConfig,
    on_image: Callable[[ImageProgress], None] | None = None,
) -> tuple[CalibrationSystem, dict]:
    """
    Run detection over every camera's images and assemble a CalibrationSystem.

    Frames are matched across cameras BY FILENAME STEM. Multi-camera calibration
    requires synchronised capture, and the filename is the only synchronisation
    signal available from a directory of images -- so a mismatch is reported
    rather than silently producing a rig where cameras see different instants.

    `on_image` is called twice per image, "start" then "done", so a caller can
    show which file is being searched and what was found in it. The pairing is
    unconditional within a run -- an unreadable file still gets its "done", with
    no image and no detections -- because a UI that marks an image as in-flight
    on "start" would otherwise leave it marked forever. The pairing is broken
    only by an exception, which aborts the whole run and is the caller's cue to
    clear the display.
    """
    # Specs grouped by kind, each group to its own detector with its own
    # settings. Building both here also validates the whole rig up front:
    # Charuco ID collisions and two same-sized plain checkerboards both raise
    # now rather than after every image has been searched.
    detector = config.detectors_for_run()
    system = CalibrationSystem()

    for board in config.boards:
        kind = board_kind(board)
        spec = board_spec(board)
        # The meta block describes the board in ITS OWN vocabulary. A dot grid
        # reported as having "squares" would be a lie in the one place a reader
        # goes to find out what was actually calibrated against.
        if kind == "circle_grid":
            meta = {
                "circles_x": spec.circles_x,
                "circles_y": spec.circles_y,
                "spacing": spec.spacing,
                "grid_type": spec.grid_type,
            }
        else:
            meta = {
                "squares_x": spec.squares_x,
                "squares_y": spec.squares_y,
                "square_length": spec.square_length,
            }
            if kind == "charuco":
                meta["dictionary"] = spec.dictionary
        system.add_board(
            Board(
                id=spec.id,
                object_points=detector.object_points(spec.id),
                kind=kind,
                meta=meta,
            )
        )

    # Annotated because the three values have three different shapes; without
    # it the inferred `dict[str, Collection[Any]]` makes every later mutation a
    # type error and hides any real one among them.
    stats: dict[str, Any] = {"per_camera": {}, "frames": {}, "skipped": []}
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
            if on_image is not None:
                on_image(ImageProgress(camera=cam_cfg.id, path=img_path, stage="start"))
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                stats["skipped"].append(str(img_path))
                if on_image is not None:
                    on_image(ImageProgress(camera=cam_cfg.id, path=img_path, stage="done"))
                continue
            if (img.shape[1], img.shape[0]) != size:
                raise ValueError(
                    f"camera {cam_cfg.id}: {img_path.name} is "
                    f"{img.shape[1]}x{img.shape[0]} but the camera is declared "
                    f"{size[0]}x{size[1]}. Mixed resolutions cannot share intrinsics."
                )
            frame = img_path.stem
            dets = detector.detect(img)
            for det in dets:
                system.add_observation(
                    Observation(
                        frame=frame,
                        camera=cam_cfg.id,
                        board=det.board_id,
                        point_ids=det.point_ids,
                        image_points=det.image_points,
                        sigma=1.0 if config.pixel_noise_std is None else config.pixel_noise_std,
                    )
                )
                found_frames.add(frame)
                n_det += 1
            if on_image is not None:
                on_image(
                    ImageProgress(
                        camera=cam_cfg.id,
                        path=img_path,
                        stage="done",
                        image=img,
                        detections=tuple(dets),
                    )
                )
            log.debug("%s/%s: %d detections so far", cam_cfg.id, img_path.name, n_det)
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
    # `add_camera` already maintains the one-reference invariant, including
    # honouring an explicitly flagged non-first camera. Only the all-unflagged
    # case needs a default.
    if system.cameras and not any(c.is_reference for c in system.cameras.values()):
        system.set_reference_camera(next(iter(system.cameras)))
    stats["reference_camera"] = system.reference_camera if system.cameras else None
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
