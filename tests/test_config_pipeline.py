"""
The real-image path: config -> detection -> bootstrap -> solve.

This is what `mlti-cal calibrate config.json` runs, and until this file existed
none of it had ever executed -- the synthetic generator bypasses detection,
image loading and frame matching entirely. `CharucoDetector.render()` gives us
genuine detectable images, so the whole path is testable without shipping
binary fixtures.

Images are rendered under a homography so the views differ, because a stack of
identical fronto-parallel boards is degenerate and would not calibrate.
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pytest

from mlti_cal.detectors.charuco import CharucoBoardSpec, CharucoDetector
from mlti_cal.io.config import (
    CalibrationConfig,
    CameraConfig,
    ImageProgress,
    build_system_from_config,
    export_json,
    export_opencv_yaml,
    find_images,
)
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import build_problem
from mlti_cal.solvers import SolveOptions, get_backend

BOARD = {
    "id": "board_A",
    "squares_x": 8,
    "squares_y": 6,
    "square_length": 0.030,
    "marker_length": 0.022,
    "dictionary": "DICT_4X4_250",
    "marker_id_offset": 0,
}
SIZE = (960, 720)


#: Intrinsics the rendered views are actually generated from.
TRUE_K = np.array([[820.0, 0.0, 470.0], [0.0, 815.0, 372.0], [0.0, 0.0, 1.0]])
PIXELS_PER_SQUARE = 70
MARGIN = 25


def _warped_views(n: int, seed: int = 0, size=SIZE, K: np.ndarray = TRUE_K) -> list[np.ndarray]:
    """
    Render `n` views of one board as seen by a SINGLE pinhole camera.

    Deliberately NOT arbitrary homographies. A random homography per view is
    not a valid calibration dataset: each is individually a legitimate
    perspective view of a plane, but there is generally no single intrinsic
    matrix consistent with all of them, so there is nothing correct for the
    solver to converge to. Measured with the naive version: initial RMS 91 px
    and no convergence -- which is the honest answer to an inconsistent
    problem, not a solver defect.

    Instead: fix a true K, pick a board pose per view, project the rendered
    image's four corners through that camera, and warp to the result. Every
    view is then explained by the same K and calibration has a right answer.

    A warp cannot produce lens distortion, so recovered distortion should be
    near zero.
    """
    det = CharucoDetector(CharucoBoardSpec(**BOARD))
    board = det.render(pixels_per_square=PIXELS_PER_SQUARE, margin=MARGIN)
    bh, bw = board.shape[:2]

    # Pixel <-> board-frame mapping of the rendered image: generateImage fits
    # the board inside the margin, so the drawable area spans the full board.
    sx = (bw - 2 * MARGIN) / (BOARD["squares_x"] * BOARD["square_length"])
    sy = (bh - 2 * MARGIN) / (BOARD["squares_y"] * BOARD["square_length"])
    corners_board = np.array(
        [
            [-MARGIN / sx, -MARGIN / sy, 0.0],
            [(bw - MARGIN) / sx, -MARGIN / sy, 0.0],
            [(bw - MARGIN) / sx, (bh - MARGIN) / sy, 0.0],
            [-MARGIN / sx, (bh - MARGIN) / sy, 0.0],
        ]
    )
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    extent = np.array(
        [
            BOARD["squares_x"] * BOARD["square_length"],
            BOARD["squares_y"] * BOARD["square_length"],
            0.0,
        ]
    )

    rng = np.random.default_rng(seed)
    w, h = size
    out: list[np.ndarray] = []
    attempts = 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        rvec = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4)])
        R, _ = cv2.Rodrigues(rvec)
        depth = rng.uniform(0.55, 1.0)
        tvec = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.06, 0.06), depth]) - R @ (
            extent * 0.5
        )
        pts, _ = cv2.projectPoints(corners_board, rvec, tvec, K, np.zeros(5))
        dst = pts.reshape(-1, 2).astype(np.float32)
        if dst.min() < 6 or dst[:, 0].max() > w - 6 or dst[:, 1].max() > h - 6:
            continue  # keep the whole board inside the frame
        H = cv2.getPerspectiveTransform(src, dst)
        out.append(cv2.warpPerspective(board, H, size, flags=cv2.INTER_LINEAR, borderValue=255))
    if len(out) < n:  # pragma: no cover - generator tuning failure
        raise RuntimeError(f"only produced {len(out)}/{n} in-frame views")
    return out


def _write_camera_dir(tmp_path, name: str, images, stems) -> str:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for img, stem in zip(images, stems, strict=True):
        cv2.imwrite(str(d / f"{stem}.png"), img)
    return str(d)


def test_find_images_sorted_and_filtered(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for name in ("b.png", "a.png", "c.jpg", "notes.txt"):
        (d / name).write_bytes(b"x")
    names = [p.name for p in find_images(d)]
    assert names == ["a.png", "b.png", "c.jpg"]
    with pytest.raises(NotADirectoryError):
        find_images(tmp_path / "nope")


def test_single_camera_config_pipeline_end_to_end(tmp_path):
    stems = [f"{i:03d}" for i in range(14)]
    folder = _write_camera_dir(tmp_path, "cam0", _warped_views(14, seed=1), stems)
    config = CalibrationConfig(
        cameras=[CameraConfig(id="cam0", image_dir=folder, is_reference=True)],
        boards=[BOARD],
    )
    system, stats = build_system_from_config(config)

    assert stats["per_camera"]["cam0"]["images"] == 14
    assert stats["per_camera"]["cam0"]["detections"] >= 10, stats
    assert system.cameras["cam0"].image_size == SIZE
    assert system.num_observed_points > 300

    initialize_system(system)
    problem = build_problem(system)
    result = get_backend("scipy").solve(problem, SolveOptions(max_iterations=2000))
    from mlti_cal.problem.reprojection import write_back

    write_back(system, problem)
    assert result.success, result.summary()
    # Rendered images are geometrically exact; what is left is detector and
    # resampling noise, which should stay well under a pixel.
    assert result.final_rms_px < 1.0, result.summary()

    # The views were generated from a known K, so it must be recovered. This is
    # the check that makes the whole real-image path meaningful rather than
    # merely "it ran".
    est = system.cameras["cam0"].params
    assert est[0] == pytest.approx(TRUE_K[0, 0], rel=0.05), f"fx {est[0]:.1f}"
    assert est[1] == pytest.approx(TRUE_K[1, 1], rel=0.05), f"fy {est[1]:.1f}"
    assert est[2] == pytest.approx(TRUE_K[0, 2], abs=40), f"cx {est[2]:.1f}"
    assert est[3] == pytest.approx(TRUE_K[1, 2], abs=40), f"cy {est[3]:.1f}"
    # A warp introduces no lens distortion, so the coefficients must stay small.
    assert abs(est[4]) < 0.15, f"k1 {est[4]:.4f} on distortion-free imagery"


def test_two_camera_pipeline_matches_frames_by_stem(tmp_path):
    stems = [f"{i:03d}" for i in range(12)]
    a = _write_camera_dir(tmp_path, "cam0", _warped_views(12, seed=2), stems)
    b = _write_camera_dir(tmp_path, "cam1", _warped_views(12, seed=3), stems)
    config = CalibrationConfig(
        cameras=[
            CameraConfig(id="cam0", image_dir=a, is_reference=True),
            CameraConfig(id="cam1", image_dir=b),
        ],
        boards=[BOARD],
    )
    system, stats = build_system_from_config(config)
    assert stats["shared_frames"] == 12
    assert set(system.frames) == set(stems)
    assert {o.camera for o in system.observations} == {"cam0", "cam1"}


def test_disjoint_frame_names_are_refused(tmp_path):
    """Cameras that share no frame stem cannot be a synchronised rig."""
    a = _write_camera_dir(tmp_path, "cam0", _warped_views(4, seed=4), ["a0", "a1", "a2", "a3"])
    b = _write_camera_dir(tmp_path, "cam1", _warped_views(4, seed=5), ["b0", "b1", "b2", "b3"])
    config = CalibrationConfig(
        cameras=[CameraConfig(id="cam0", image_dir=a), CameraConfig(id="cam1", image_dir=b)],
        boards=[BOARD],
    )
    with pytest.raises(ValueError, match="No frame name is common"):
        build_system_from_config(config)


def test_mixed_resolution_is_refused(tmp_path):
    """Images of different sizes cannot share one set of intrinsics."""
    imgs = _warped_views(4, seed=6)
    imgs[2] = cv2.resize(imgs[2], (640, 480))
    folder = _write_camera_dir(tmp_path, "cam0", imgs, ["0", "1", "2", "3"])
    config = CalibrationConfig(cameras=[CameraConfig(id="cam0", image_dir=folder)], boards=[BOARD])
    with pytest.raises(ValueError, match="Mixed resolutions|declared"):
        build_system_from_config(config)


def test_empty_image_directory_is_refused(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    config = CalibrationConfig(cameras=[CameraConfig(id="cam0", image_dir=str(d))], boards=[BOARD])
    with pytest.raises(ValueError, match="no images"):
        build_system_from_config(config)


def test_progress_is_reported_once_per_image_as_a_matched_pair(tmp_path):
    """
    Every "start" must get a "done", including for a file that cannot be read.

    This is not bookkeeping pedantry. The GUI tints the image it is working on
    and clears the tint on "done", so a start without its done leaves a frame
    marked as in-flight forever, on the one code path -- an unreadable file --
    that is least likely to be exercised by hand.
    """
    stems = [f"{i:03d}" for i in range(6)]
    folder = _write_camera_dir(tmp_path, "cam0", _warped_views(6, seed=11), stems)
    broken = pathlib.Path(folder) / "zz_broken.png"
    broken.write_bytes(b"this is not a PNG")

    config = CalibrationConfig(
        cameras=[CameraConfig(id="cam0", image_dir=folder, is_reference=True)], boards=[BOARD]
    )
    events: list[ImageProgress] = []
    system, stats = build_system_from_config(config, on_image=events.append)

    started = [e for e in events if e.stage == "start"]
    done = [e for e in events if e.stage == "done"]
    assert len(started) == len(done) == 7, f"{len(started)} starts, {len(done)} dones"
    assert [e.path for e in started] == [e.path for e in done], "events must pair up in order"
    assert stats["skipped"] == [str(broken)]

    # The payload has to carry what a viewer needs, or it re-decodes every file.
    readable = [e for e in done if e.image is not None]
    assert len(readable) == 6
    assert all(e.image.ndim == 2 for e in readable), "grayscale frames are what detection saw"
    assert sum(e.num_corners for e in readable) > 100
    assert max(e.num_corners for e in readable) == max(o.num_points for o in system.observations), (
        "reported corner counts must be the observations', not a running total"
    )

    unreadable = next(e for e in done if e.image is None)
    assert unreadable.path == broken
    assert unreadable.detections == ()
    assert unreadable.num_corners == 0


def test_detection_without_a_callback_is_unchanged(tmp_path):
    """The callback is optional and must not alter what gets built."""
    stems = [f"{i:03d}" for i in range(6)]
    folder = _write_camera_dir(tmp_path, "cam0", _warped_views(6, seed=11), stems)
    config = CalibrationConfig(
        cameras=[CameraConfig(id="cam0", image_dir=folder, is_reference=True)], boards=[BOARD]
    )
    quiet, quiet_stats = build_system_from_config(config)
    loud, loud_stats = build_system_from_config(config, on_image=lambda _p: None)

    assert quiet_stats == loud_stats
    assert len(quiet.observations) == len(loud.observations)
    assert quiet.num_observed_points == loud.num_observed_points


# ---------------------------------------------------------------------------
# Reference-camera selection
# ---------------------------------------------------------------------------


def test_explicitly_marked_non_first_camera_becomes_the_reference(tmp_path):
    """
    Regression: the first camera added used to be flagged unconditionally, so a
    config naming any other camera produced TWO references. `reference_camera`
    returned the first, the gauge was fixed on the wrong camera, and every
    extrinsic was reported against an origin nobody asked for -- with no error.
    """
    stems = [f"{i:03d}" for i in range(8)]
    a = _write_camera_dir(tmp_path, "cam0", _warped_views(8, seed=7), stems)
    b = _write_camera_dir(tmp_path, "cam1", _warped_views(8, seed=8), stems)
    config = CalibrationConfig(
        cameras=[
            CameraConfig(id="cam0", image_dir=a, is_reference=False),
            CameraConfig(id="cam1", image_dir=b, is_reference=True),
        ],
        boards=[BOARD],
    )
    system, stats = build_system_from_config(config)

    assert system.reference_camera == "cam1"
    assert stats["reference_camera"] == "cam1"
    assert sum(c.is_reference for c in system.cameras.values()) == 1

    initialize_system(system)
    problem = build_problem(system)
    from mlti_cal.problem.reprojection import extr_key

    assert problem.blocks[extr_key("cam1")].constant, "the gauge must be on cam1"
    assert not problem.blocks[extr_key("cam0")].constant


def test_first_camera_defaults_to_reference_when_none_marked():
    from mlti_cal.problem.types import CalibrationSystem, Camera

    s = CalibrationSystem()
    s.add_camera(Camera(id="a", model_name="pinhole_radtan", image_size=SIZE))
    s.add_camera(Camera(id="b", model_name="pinhole_radtan", image_size=SIZE))
    assert s.reference_camera == "a"
    assert sum(c.is_reference for c in s.cameras.values()) == 1


def test_two_reference_cameras_raise_rather_than_guessing():
    from mlti_cal.problem.types import CalibrationSystem, Camera

    s = CalibrationSystem()
    s.add_camera(Camera(id="a", model_name="pinhole_radtan", image_size=SIZE))
    s.add_camera(Camera(id="b", model_name="pinhole_radtan", image_size=SIZE))
    s.cameras["b"].is_reference = True  # corrupt the invariant directly
    with pytest.raises(ValueError, match="marked as reference"):
        _ = s.reference_camera


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_exports_are_readable_by_opencv(tmp_path):
    stems = [f"{i:03d}" for i in range(10)]
    folder = _write_camera_dir(tmp_path, "cam0", _warped_views(10, seed=9), stems)
    config = CalibrationConfig(cameras=[CameraConfig(id="cam0", image_dir=folder)], boards=[BOARD])
    system, _ = build_system_from_config(config)
    initialize_system(system)

    yaml_path = export_opencv_yaml(system, tmp_path / "calib.yaml")
    json_path = export_json(system, tmp_path / "calib.json")
    assert yaml_path.exists() and json_path.exists()

    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    try:
        K = fs.getNode("cam0_K").mat()
        dist = fs.getNode("cam0_dist").mat()
        assert K.shape == (3, 3) and K[2, 2] == pytest.approx(1.0)
        assert dist.size == 5
        # The exported values must actually drive OpenCV without reshaping.
        undistorted = cv2.undistort(np.zeros((720, 960), np.uint8), K, dist)
        assert undistorted.shape == (720, 960)
    finally:
        fs.release()

    import json

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["reference_camera"] == "cam0"
    assert len(data["cameras"]["cam0"]["parameters"]) == 9
