"""
Checkerboard detection and the masking pass, against rendered images.

The load-bearing test here is `test_chessboard_finder_false_positives_on_a_bare
_charuco_board`. It is the whole reason masking exists, and it fails loudly if
a future OpenCV ever stops confusing the two -- at which point the masking pass
could be reconsidered rather than carried forever on a stale assumption.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlti_cal.detectors.base import BOARD_PALETTE, Detection, board_colours, draw_detections
from mlti_cal.detectors.charuco import CharucoBoardSpec, CharucoDetector
from mlti_cal.detectors.checkerboard import (
    CheckerboardDetector,
    CheckerboardSpec,
    IndistinguishableBoardsError,
    MaskGeometry,
    MultiCheckerboardDetector,
    mask_boards,
)
from mlti_cal.detectors.pipeline import BoardDetectors
from mlti_cal.detectors.settings import CheckerboardSettings

SQ = 40  # pixels per square in the rendered scenes


def render_checkerboard(squares_x: int, squares_y: int, px: int = SQ) -> np.ndarray:
    img = np.zeros((squares_y * px, squares_x * px), np.uint8)
    for r in range(squares_y):
        for c in range(squares_x):
            if (r + c) % 2 == 0:
                img[r * px : (r + 1) * px, c * px : (c + 1) * px] = 255
    return img


def charuco_spec(**kw) -> CharucoBoardSpec:
    base = dict(
        id="ch",
        squares_x=8,
        squares_y=6,
        square_length=0.030,
        marker_length=0.022,
        dictionary="DICT_4X4_250",
    )
    base.update(kw)
    return CharucoBoardSpec(**base)


def paste(scene: np.ndarray, img: np.ndarray, top: int, left: int) -> np.ndarray:
    scene[top : top + img.shape[0], left : left + img.shape[1]] = img
    return scene


# -- geometry ---------------------------------------------------------------
def test_squares_are_converted_to_opencv_interior_corners():
    """
    The spec counts SQUARES, exactly like the Charuco one; OpenCV wants interior
    corners. Getting this backwards is the classic chessboard off-by-one.
    """
    s = CheckerboardSpec("cb", squares_x=9, squares_y=7, square_length=0.03)
    assert s.pattern_size == (8, 6)
    assert s.num_corners == 48 == s.object_points.shape[0]


def test_origin_convention_matches_charuco():
    """
    Both kinds put the board origin at the printed sheet's outer corner, so the
    first interior corner sits one square in. Two conventions in one rig would
    make board poses from the two detectors incomparable.
    """
    cb = CheckerboardSpec("cb", 8, 6, 0.03)
    ch = CharucoDetector(charuco_spec(squares_x=8, squares_y=6, square_length=0.03))
    assert np.allclose(cb.object_points.min(axis=0), ch.object_points.min(axis=0))
    assert np.allclose(cb.object_points.max(axis=0), ch.object_points.max(axis=0))
    assert np.allclose(cb.outline_object_points.max(axis=0), [8 * 0.03, 6 * 0.03])


def test_a_board_of_fewer_than_three_squares_is_refused():
    with pytest.raises(ValueError, match="at least 3"):
        CheckerboardSpec("cb", squares_x=2, squares_y=7, square_length=0.03)


# -- detection --------------------------------------------------------------
def test_render_and_detect_roundtrip():
    spec = CheckerboardSpec("cb", 9, 7, 0.030)
    scene = np.full((600, 900), 200, np.uint8)
    paste(scene, render_checkerboard(9, 7), 100, 100)
    det = CheckerboardDetector(spec).detect(scene)
    assert det is not None
    assert det.num_points == spec.num_corners
    assert det.kind == "checkerboard"
    assert det.board_id == "cb"


def test_detection_is_all_or_nothing():
    """Half a board is not a detection -- OpenCV offers no partial answer."""
    spec = CheckerboardSpec("cb", 9, 7, 0.030)
    scene = np.full((600, 900), 200, np.uint8)
    board = render_checkerboard(9, 7)
    paste(scene, board[:, : board.shape[1] // 2], 100, 100)
    assert CheckerboardDetector(spec).detect(scene) is None


def test_detect_returns_none_on_blank_image():
    scene = np.full((400, 400), 200, np.uint8)
    assert CheckerboardDetector(CheckerboardSpec("cb", 9, 7, 0.03)).detect(scene) is None


def test_legacy_algorithm_also_detects():
    """The LEGACY path is offered in the catalog, so it has to actually work."""
    spec = CheckerboardSpec("cb", 9, 7, 0.030)
    scene = np.full((700, 1000), 255, np.uint8)
    paste(scene, render_checkerboard(9, 7), 120, 120)
    det = CheckerboardDetector(spec, CheckerboardSettings(algorithm="LEGACY")).detect(scene)
    assert det is not None and det.num_points == spec.num_corners


# -- the reason masking exists ---------------------------------------------
def test_chessboard_finder_false_positives_on_a_bare_charuco_board():
    """
    A Charuco board IS a chessboard with markers inked in, and the plain finder
    happily reports a full grid on one. There is no checkerboard in this image.
    """
    spec = charuco_spec()
    scene = np.full((700, 1000), 220, np.uint8)
    paste(scene, CharucoDetector(spec).render(pixels_per_square=60, margin=0), 80, 80)
    bogus = CheckerboardDetector(CheckerboardSpec("cb", 8, 6, 0.030)).detect(scene)
    assert bogus is not None, (
        "OpenCV no longer confuses a Charuco board for a plain chessboard. If "
        "this is genuinely fixed upstream, the masking pass can be revisited."
    )


def test_masking_removes_the_false_positive():
    spec = charuco_spec()
    ch = CharucoDetector(spec)
    scene = np.full((700, 1000), 220, np.uint8)
    paste(scene, ch.render(pixels_per_square=60, margin=0), 80, 80)
    hit = ch.detect(scene)
    assert hit is not None

    geometry = {
        "ch": MaskGeometry(ch.object_points, spec.outline_object_points, spec.square_length)
    }
    masked = mask_boards(scene, [hit], geometry, CheckerboardSettings())
    assert CheckerboardDetector(CheckerboardSpec("cb", 8, 6, 0.030)).detect(masked) is None


def test_mask_covers_the_printed_edge_not_just_the_interior_corners():
    """
    Charuco corners are interior, so a hull of them stops one square short of
    the printed edge -- the exact strip a chessboard detector feeds on.
    """
    spec = charuco_spec()
    ch = CharucoDetector(spec)
    scene = np.full((700, 1000), 220, np.uint8)
    board = ch.render(pixels_per_square=60, margin=0)
    paste(scene, board, 80, 80)
    hit = ch.detect(scene)
    geometry = {
        "ch": MaskGeometry(ch.object_points, spec.outline_object_points, spec.square_length)
    }
    masked = mask_boards(scene, [hit], geometry, CheckerboardSettings())
    region = masked[80 : 80 + board.shape[0], 80 : 80 + board.shape[1]]
    assert region.std() == 0.0, "board is still visible through the mask"
    assert region.flat[0] == CheckerboardSettings().mask_fill_value


def test_masking_leaves_the_real_checkerboard_alone():
    ch_spec = charuco_spec()
    ch = CharucoDetector(ch_spec)
    scene = np.full((700, 1200), 220, np.uint8)
    paste(scene, ch.render(pixels_per_square=SQ, margin=0), 60, 40)
    paste(scene, render_checkerboard(9, 7), 400, 700)

    detectors = BoardDetectors.build(
        {"charuco": [ch_spec], "checkerboard": [CheckerboardSpec("cb", 9, 7, 0.030)]}
    )
    found = {d.board_id: d for d in detectors.detect(scene)}
    assert set(found) == {"ch", "cb"}
    assert found["cb"].kind == "checkerboard"
    centroid = found["cb"].image_points.mean(axis=0)
    assert np.allclose(centroid, [700 + 9 * SQ / 2, 400 + 7 * SQ / 2], atol=2.0)


def test_mask_settings_can_turn_masking_off():
    """The knob is connected, not merely displayed."""
    ch_spec = charuco_spec()
    ch = CharucoDetector(ch_spec)
    scene = np.full((700, 1000), 220, np.uint8)
    paste(scene, ch.render(pixels_per_square=60, margin=0), 80, 80)
    detectors = BoardDetectors.build(
        {"charuco": [ch_spec], "checkerboard": [CheckerboardSpec("cb", 8, 6, 0.030)]},
        {"checkerboard": CheckerboardSettings(mask_other_boards=False)},
    )
    kinds = {d.board_id for d in detectors.detect(scene)}
    assert "cb" in kinds, "masking off should let the false positive back through"


# -- ambiguity is refused, not guessed -------------------------------------
def test_two_identically_sized_checkerboards_are_refused():
    with pytest.raises(IndistinguishableBoardsError, match="carries no identity"):
        MultiCheckerboardDetector(
            [CheckerboardSpec("a", 9, 7, 0.03), CheckerboardSpec("b", 9, 7, 0.04)]
        )


def test_differently_sized_checkerboards_are_accepted():
    multi = MultiCheckerboardDetector(
        [CheckerboardSpec("a", 9, 7, 0.03), CheckerboardSpec("b", 8, 6, 0.03)]
    )
    assert set(multi.detectors) == {"a", "b"}


def test_duplicate_board_id_is_refused():
    with pytest.raises(ValueError, match="duplicate board id"):
        MultiCheckerboardDetector(
            [CheckerboardSpec("a", 9, 7, 0.03), CheckerboardSpec("a", 8, 6, 0.03)]
        )


# -- colours ----------------------------------------------------------------
def test_every_board_gets_its_own_colour():
    colours = board_colours(["board_A", "board_B", "board_C"])
    assert len(set(colours.values())) == 3


def test_colour_depends_on_the_board_not_on_detection_order():
    """
    The bug this replaces: colour used to be indexed by position in the
    detection list, so a board missing from one frame recoloured all the others.
    """
    everyone = board_colours(["a", "b", "c"])
    assert board_colours(["c", "a", "b"]) == everyone
    # 'b' absent from this frame must not move 'c' onto b's colour.
    assert everyone["c"] != board_colours(["a", "b"])["b"]


def test_overlay_draws_a_different_glyph_per_kind():
    image = np.zeros((60, 60), np.uint8)
    pts = np.array([[30.0, 30.0]])
    ids = np.array([0])
    circle = draw_detections(
        image, [Detection("a", ids, pts, kind="charuco")], labels=False, radius=6
    )
    square = draw_detections(
        image, [Detection("a", ids, pts, kind="checkerboard")], labels=False, radius=6
    )
    assert not np.array_equal(circle, square)
    # A square corner is painted where a circle of the same radius is not.
    assert square[30 - 6, 30 - 6].any() and not circle[30 - 6, 30 - 6].any()


def test_overlay_uses_the_colour_mapping_it_is_given():
    image = np.zeros((60, 60), np.uint8)
    det = Detection("only_board", np.array([0]), np.array([[30.0, 30.0]]))
    out = draw_detections(image, [det], labels=False, colours={"only_board": BOARD_PALETTE[3]})
    assert tuple(int(v) for v in out[30, 30]) == BOARD_PALETTE[3]


def test_two_differently_sized_checkerboards_are_both_found():
    """
    Sequential masking, not just construction.

    The larger board is searched for first and painted out, because a smaller
    board's finder can otherwise lock onto part of a larger board and both
    detections then describe the same pixels.
    """
    scene = np.full((760, 1200), 215, np.uint8)
    paste(scene, render_checkerboard(9, 7), 60, 60)
    paste(scene, render_checkerboard(6, 5), 460, 760)

    multi = MultiCheckerboardDetector(
        [CheckerboardSpec("big", 9, 7, 0.030), CheckerboardSpec("small", 6, 5, 0.030)]
    )
    found = {d.board_id: d for d in multi.detect_all(scene)}
    assert set(found) == {"big", "small"}
    assert found["big"].num_points == 48
    assert found["small"].num_points == 20
    # Each detection sits on its own board, not twice on the same one.
    assert np.allclose(
        found["big"].image_points.mean(0), [60 + 9 * SQ / 2, 60 + 7 * SQ / 2], atol=2
    )
    assert np.allclose(
        found["small"].image_points.mean(0), [760 + 6 * SQ / 2, 460 + 5 * SQ / 2], atol=2
    )
