"""
Dot-grid detection, against rendered grids.

The geometry tests carry most of the weight here. Detection either finds the
whole grid or nothing, so it is hard to get subtly wrong -- but the SCALE
convention is very easy to get subtly wrong, and a spacing that is out by a
factor of two is invisible in reprojection error and in the intrinsics. It
shows up only in the metric extrinsics, as a baseline exactly twice or half
what it should be. `test_in_row_spacing_is_the_declared_spacing` is the guard.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mlti_cal.detectors.base import KIND_MARKER, Detection, draw_detections
from mlti_cal.detectors.charuco import CharucoBoardSpec, CharucoDetector
from mlti_cal.detectors.checkerboard import CheckerboardSpec, mask_boards
from mlti_cal.detectors.circle_grid import (
    GRID_TYPES,
    CircleGridDetector,
    CircleGridSpec,
    IndistinguishableGridsError,
    MultiCircleGridDetector,
)
from mlti_cal.detectors.pipeline import BoardDetectors
from mlti_cal.detectors.settings import CircleGridSettings

PX_PER_M = 2000.0


def render_grid(spec: CircleGridSpec, margin: int = 60, radius_frac: float = 0.25):
    """A grid drawn as dark dots on white. Returns the image and the true centres."""
    xy = spec.object_points[:, :2] * PX_PER_M
    radius = int(spec.spacing * PX_PER_M * radius_frac)
    w = int(xy[:, 0].max() + 2 * margin)
    h = int(xy[:, 1].max() + 2 * margin)
    img = np.full((h, w), 255, np.uint8)
    for x, y in xy:
        cv2.circle(img, (int(round(x)) + margin, int(round(y)) + margin), radius, 0, -1)
    return img, xy + margin


def scene_with(spec: CircleGridSpec, pad: int = 60):
    img, truth = render_grid(spec)
    scene = np.full((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad), 255, np.uint8)
    scene[pad : pad + img.shape[0], pad : pad + img.shape[1]] = img
    return scene, truth + pad


# -- geometry ---------------------------------------------------------------
@pytest.mark.parametrize("grid_type", GRID_TYPES)
def test_in_row_spacing_is_the_declared_spacing(grid_type):
    """
    `spacing` means centre-to-centre within a ROW, for both grid types.

    OpenCV's asymmetric sample is written against a half-pitch unit instead. A
    board entered under that reading is out by exactly 2x, which never shows in
    the RPE and lands entirely on the metric scale.
    """
    spec = CircleGridSpec("d", circles_x=4, circles_y=11, spacing=0.020, grid_type=grid_type)
    row0 = spec.object_points[: spec.circles_x, :2]
    assert np.allclose(np.diff(row0[:, 0]), 0.020)


def test_asymmetric_rows_are_staggered_and_symmetric_ones_are_not():
    """The stagger is the whole point: it is what resolves board orientation."""
    asym = CircleGridSpec("d", 4, 11, 0.020, "asymmetric").object_points
    sym = CircleGridSpec("d", 4, 11, 0.020, "symmetric").object_points
    # Row 1 starts half a pitch to the right of row 0, and half a pitch below.
    assert asym[4, 0] - asym[0, 0] == pytest.approx(0.010)
    assert asym[4, 1] - asym[0, 1] == pytest.approx(0.010)
    # A symmetric grid keeps rows aligned and a full pitch apart.
    assert sym[4, 0] - sym[0, 0] == pytest.approx(0.0)
    assert sym[4, 1] - sym[0, 1] == pytest.approx(0.020)


def test_point_count_and_ordering_are_row_major():
    spec = CircleGridSpec("d", 4, 11, 0.02)
    op = spec.object_points
    assert op.shape == (44, 3) and spec.num_points == 44
    assert (op[:, 2] == 0).all(), "a printed grid is planar"
    # Row-major: the first `circles_x` points share a y.
    assert len(set(np.round(op[:4, 1], 9))) == 1


def test_default_grid_type_is_asymmetric():
    """The default is the one that resolves orientation, in every entry point."""
    assert GRID_TYPES[0] == "asymmetric"
    assert CircleGridSpec("d", 4, 11, 0.02).grid_type == "asymmetric"


def test_degenerate_specs_are_refused():
    with pytest.raises(ValueError, match="at least 2"):
        CircleGridSpec("d", circles_x=1, circles_y=11, spacing=0.02)
    with pytest.raises(ValueError, match="spacing must be"):
        CircleGridSpec("d", 4, 11, 0.0)
    with pytest.raises(ValueError, match="unknown grid_type"):
        CircleGridSpec("d", 4, 11, 0.02, grid_type="staggered")


# -- detection --------------------------------------------------------------
@pytest.mark.parametrize("grid_type", GRID_TYPES)
def test_render_and_detect_roundtrip(grid_type):
    spec = CircleGridSpec("dots", 4, 11, 0.020, grid_type)
    scene, truth = scene_with(spec)
    det = CircleGridDetector(spec).detect(scene)
    assert det is not None
    assert det.num_points == spec.num_points
    assert det.kind == "circle_grid"
    assert np.abs(det.image_points - truth).max() < 1.0, "centres are off their dots"


def test_detect_returns_none_on_blank_image():
    scene = np.full((400, 400), 255, np.uint8)
    assert CircleGridDetector(CircleGridSpec("d", 4, 11, 0.02)).detect(scene) is None


def test_detection_is_all_or_nothing():
    """Half a grid is not a detection -- findCirclesGrid offers no partial answer."""
    spec = CircleGridSpec("d", 4, 11, 0.020)
    scene, _ = scene_with(spec)
    scene[: scene.shape[0] // 2, :] = 255  # erase the top half of the board
    assert CircleGridDetector(spec).detect(scene) is None


def test_blob_colour_setting_actually_reaches_the_detector():
    """Dark-on-light vs light-on-dark: getting it backwards finds nothing."""
    spec = CircleGridSpec("d", 4, 11, 0.020)
    scene, _ = scene_with(spec)
    inverted = 255 - scene
    assert CircleGridDetector(spec).detect(inverted) is None, "precondition"
    light = CircleGridSettings(blob_color="light")
    assert CircleGridDetector(spec, light).detect(inverted) is not None


def test_min_area_rejects_dots_below_it():
    """The knob is connected, not merely displayed."""
    spec = CircleGridSpec("d", 4, 11, 0.020)
    scene, _ = scene_with(spec)
    huge_floor = CircleGridSettings(min_area=200000.0, max_area=400000.0)
    assert CircleGridDetector(spec, huge_floor).detect(scene) is None


# -- ambiguity is refused, not guessed -------------------------------------
def test_two_identical_grids_are_refused():
    with pytest.raises(IndistinguishableGridsError, match="carries no identity"):
        MultiCircleGridDetector(
            [
                CircleGridSpec("a", 4, 11, 0.02, "asymmetric"),
                CircleGridSpec("b", 4, 11, 0.03, "asymmetric"),
            ]
        )


def test_same_size_but_different_type_is_accepted():
    """The finder is told which layout to expect, so these ARE distinguishable."""
    multi = MultiCircleGridDetector(
        [
            CircleGridSpec("a", 4, 11, 0.02, "asymmetric"),
            CircleGridSpec("b", 4, 11, 0.02, "symmetric"),
        ]
    )
    assert set(multi.detectors) == {"a", "b"}


def test_duplicate_board_id_is_refused():
    with pytest.raises(ValueError, match="duplicate board id"):
        MultiCircleGridDetector([CircleGridSpec("a", 4, 11, 0.02), CircleGridSpec("a", 5, 9, 0.02)])


# -- masking and the mixed pipeline ----------------------------------------
def test_a_grid_is_masked_out_before_the_checkerboard_searches():
    """
    A dot grid is uncoded too, so it must be removed before the plain
    chessboard search, not only the Charuco boards.
    """
    spec = CircleGridSpec("dots", 4, 11, 0.020)
    scene, _ = scene_with(spec)
    detector = CircleGridDetector(spec)
    hit = detector.detect(scene)
    assert hit is not None
    masked = mask_boards(scene, [hit], {"dots": detector.mask_geometry}, CircleGridSettings())
    assert CircleGridDetector(spec).detect(masked) is None, "the grid survived its own mask"


def test_grid_and_charuco_are_both_found_in_one_image():
    ch_spec = CharucoBoardSpec("ch", 8, 6, 0.030, 0.022, "DICT_4X4_250")
    ch = CharucoDetector(ch_spec)
    grid_spec = CircleGridSpec("dots", 4, 11, 0.020)
    grid_img, _ = render_grid(grid_spec)
    board = ch.render(pixels_per_square=40, margin=0)

    scene = np.full((900, 1400), 255, np.uint8)
    scene[60 : 60 + board.shape[0], 60 : 60 + board.shape[1]] = board
    scene[400 : 400 + grid_img.shape[0], 900 : 900 + grid_img.shape[1]] = grid_img

    detectors = BoardDetectors.build({"charuco": [ch_spec], "circle_grid": [grid_spec]})
    found = {d.board_id: d for d in detectors.detect(scene)}
    assert set(found) == {"ch", "dots"}
    assert found["dots"].kind == "circle_grid"
    assert found["dots"].num_points == grid_spec.num_points


def test_all_three_kinds_coexist_in_one_pipeline():
    detectors = BoardDetectors.build(
        {
            "charuco": [CharucoBoardSpec("ch", 8, 6, 0.03, 0.022, "DICT_4X4_250")],
            "checkerboard": [CheckerboardSpec("cb", 9, 7, 0.03)],
            "circle_grid": [CircleGridSpec("dots", 4, 11, 0.02)],
        }
    )
    # Coded first, then dot grid, then the plain chessboard -- each searching an
    # image with everything already found painted out.
    # A SET, not a sequence. `board_ids` follows registration order now, and the
    # pipeline no longer has a fixed detector order to encode: it runs the coded
    # kinds first and then iterates the uncoded ones to a fixed point. Asserting
    # a sequence here would be pinning an order nothing depends on -- and would
    # break the moment a plugin registered between two built-ins.
    assert set(detectors.board_ids) == {"ch", "dots", "cb"}
    # EVERY board, uncoded included: the uncoded detectors unblock each other by
    # masking, so a board with no geometry here is one that can never be painted
    # out and therefore blocks the others forever.
    assert set(detectors.mask_geometry) == {"ch", "dots", "cb"}
    for bid in detectors.board_ids:
        assert detectors.object_points(bid).shape[1] == 3
    with pytest.raises(KeyError, match="unknown board"):
        detectors.object_points("nope")


# -- overlay ----------------------------------------------------------------
def test_a_grid_does_not_share_the_charuco_glyph():
    """Colour says which board; the glyph says which detector. Both must differ."""
    assert KIND_MARKER["circle_grid"] != KIND_MARKER["charuco"]
    image = np.zeros((60, 60), np.uint8)
    pts, ids = np.array([[30.0, 30.0]]), np.array([0])
    a = draw_detections(image, [Detection("b", ids, pts, kind="charuco")], labels=False, radius=8)
    b = draw_detections(
        image, [Detection("b", ids, pts, kind="circle_grid")], labels=False, radius=8
    )
    assert not np.array_equal(a, b)


def render_checkerboard(squares_x: int, squares_y: int, px: int = 34) -> np.ndarray:
    img = np.zeros((squares_y * px, squares_x * px), np.uint8)
    for r in range(squares_y):
        for c in range(squares_x):
            if (r + c) % 2 == 0:
                img[r * px : (r + 1) * px, c * px : (c + 1) * px] = 255
    return img


def test_three_kinds_in_one_real_image_are_all_found():
    """
    The uncoded detectors block each other, and iterating is what breaks the tie.

    This is the case that caught two real bugs. The dot grid's 44 blobs are all
    found, but the Charuco board and the checkerboard contribute ~60 more, and
    `findCirclesGrid` fails on the cluttered set -- so the grid cannot go first.
    The checkerboard succeeds despite the dots, and masking it unblocks the grid
    on the next pass. That only works if the checkerboard HAS mask geometry,
    which it did not when it was always the last detector to run.
    """
    ch_spec = CharucoBoardSpec("board_A", 8, 6, 0.030, 0.022, "DICT_4X4_250")
    cb_spec = CheckerboardSpec("board_B", 9, 7, 0.030)
    grid_spec = CircleGridSpec("board_C", 4, 11, 0.020)

    board = CharucoDetector(ch_spec).render(pixels_per_square=34, margin=0)
    checker = render_checkerboard(9, 7)
    grid, _ = render_grid(grid_spec, margin=40, radius_frac=0.25)

    scene = np.full((900, 1500), 235, np.uint8)
    for img, (top, left) in ((board, (40, 30)), (checker, (40, 360)), (grid, (430, 700))):
        scene[top : top + img.shape[0], left : left + img.shape[1]] = img

    detectors = BoardDetectors.build(
        {"charuco": [ch_spec], "checkerboard": [cb_spec], "circle_grid": [grid_spec]}
    )
    assert set(detectors.mask_geometry) == {"board_A", "board_B", "board_C"}, (
        "every locatable board needs mask geometry, or it can never be painted out"
    )
    found = {d.board_id: d for d in detectors.detect(scene)}
    assert set(found) == {"board_A", "board_B", "board_C"}
    assert found["board_C"].num_points == grid_spec.num_points
    assert {d.kind for d in found.values()} == {"charuco", "checkerboard", "circle_grid"}
