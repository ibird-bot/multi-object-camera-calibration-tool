"""
Detection tests (M1), against real rendered board images -- not mocks.

Rendering a board and detecting it back is the only check that actually
exercises the OpenCV 5 API surface, which is where the sibling project's
documentation turned out to be wrong.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mlti_cal.detectors.charuco import (
    CharucoBoardSpec,
    CharucoDetector,
    IdCollisionError,
    MultiBoardDetector,
    draw_detections,
)
from mlti_cal.io.synthetic import charuco_object_points


def spec(**kw):
    base = dict(
        id="board_A",
        squares_x=9,
        squares_y=7,
        square_length=0.03,
        marker_length=0.022,
        dictionary="DICT_4X4_250",
    )
    base.update(kw)
    return CharucoBoardSpec(**base)


def test_board_geometry_matches_our_generator():
    """
    The detector's object points and the synthetic generator's must agree, or
    solves on real data and on synthetic data are silently different problems.
    """
    s = spec()
    det = CharucoDetector(s)
    ours = charuco_object_points(s.squares_x, s.squares_y, s.square_length)
    theirs = det.object_points
    assert theirs.shape == ours.shape == (s.num_corners, 3)
    assert np.allclose(theirs, ours, atol=1e-9), (
        f"max diff {np.abs(theirs - ours).max():.2e}\nours[:3]={ours[:3]}\ntheirs[:3]={theirs[:3]}"
    )


def test_render_and_detect_roundtrip():
    det = CharucoDetector(spec())
    img = det.render(pixels_per_square=90, margin=30)
    found = det.detect(img)
    assert found is not None
    # A clean synthetic render should yield nearly every interior corner.
    assert found.num_points >= det.spec.num_corners - 2
    assert found.point_ids.max() < det.spec.num_corners
    assert found.image_points.shape == (found.num_points, 2)


def test_detected_corners_are_subpixel_accurate():
    """Corner ordering and geometry must match the board's own model."""
    det = CharucoDetector(spec())
    pps, margin = 100, 40
    img = det.render(pixels_per_square=pps, margin=margin)
    found = det.detect(img)
    assert found is not None
    # A flat board rendered orthographically is an exact AFFINE image of the
    # object points. Rather than hard-code OpenCV's internal margin/scale
    # layout (which is not part of its API contract), fit the affine and check
    # the residual: that is the property that actually matters -- corner
    # ORDERING and geometry agree with the board model. A single swapped or
    # mis-indexed corner blows this up immediately.
    obj = det.object_points[found.point_ids][:, :2]
    A = np.column_stack([obj, np.ones(len(obj))])  # (K,3)
    coeffs, *_ = np.linalg.lstsq(A, found.image_points, rcond=None)
    residual = np.linalg.norm(found.image_points - A @ coeffs, axis=1)
    assert np.median(residual) < 0.5, (
        f"median affine residual {np.median(residual):.3f} px, "
        f"max {residual.max():.3f} px -- corners do not match the board model"
    )
    # And the recovered scale must be sane and near-isotropic.
    sx = np.linalg.norm(coeffs[0])
    sy = np.linalg.norm(coeffs[1])
    assert 0.95 < sx / sy < 1.05, f"anisotropic scale {sx:.1f} vs {sy:.1f}"


def test_id_collision_is_refused():
    """Two boards, same dictionary, overlapping ids -> must not construct."""
    a = spec(id="A", marker_id_offset=0)
    b = spec(id="B", marker_id_offset=10)  # A uses 0..31, so this overlaps
    with pytest.raises(IdCollisionError, match="overlapping marker ids"):
        MultiBoardDetector([a, b])


def test_disjoint_id_ranges_are_accepted():
    a = spec(id="A", marker_id_offset=0)
    b = spec(id="B", marker_id_offset=a.num_markers)
    multi = MultiBoardDetector([a, b])
    assert set(multi.detectors) == {"A", "B"}
    assert a.marker_id_range[1] <= b.marker_id_range[0]


def test_different_dictionaries_may_share_ids():
    a = spec(id="A", dictionary="DICT_4X4_250", marker_id_offset=0)
    b = spec(id="B", dictionary="DICT_5X5_250", marker_id_offset=0)
    MultiBoardDetector([a, b])  # must not raise


def test_duplicate_board_id_is_refused():
    with pytest.raises(ValueError, match="duplicate board id"):
        MultiBoardDetector([spec(id="X"), spec(id="X", marker_id_offset=40)])


def test_board_too_big_for_dictionary_is_refused():
    with pytest.raises(ValueError, match="only holds"):
        CharucoDetector(spec(dictionary="DICT_4X4_50", marker_id_offset=40))


def test_marker_larger_than_square_is_refused():
    with pytest.raises(ValueError, match="must be"):
        spec(square_length=0.02, marker_length=0.03)


def test_multi_board_detects_two_boards_in_one_image():
    """Compose two rendered boards side by side and find both."""
    a = spec(id="A", squares_x=5, squares_y=5, marker_id_offset=0)
    b = spec(
        id="B",
        squares_x=5,
        squares_y=5,
        dictionary="DICT_5X5_250",
        marker_id_offset=0,
    )
    multi = MultiBoardDetector([a, b])
    ia = multi.detectors["A"].render(pixels_per_square=80, margin=25)
    ib = multi.detectors["B"].render(pixels_per_square=80, margin=25)
    h = max(ia.shape[0], ib.shape[0])
    canvas = np.full((h + 40, ia.shape[1] + ib.shape[1] + 60), 255, np.uint8)
    canvas[20 : 20 + ia.shape[0], 20 : 20 + ia.shape[1]] = ia
    x0 = 40 + ia.shape[1]
    canvas[20 : 20 + ib.shape[0], x0 : x0 + ib.shape[1]] = ib

    found = multi.detect_all(canvas)
    assert {d.board_id for d in found} == {"A", "B"}, [d.board_id for d in found]
    for d in found:
        assert d.num_points >= 8


def test_detect_returns_none_on_blank_image():
    det = CharucoDetector(spec())
    assert det.detect(np.full((480, 640), 255, np.uint8)) is None


def test_detect_accepts_colour_and_grayscale():
    det = CharucoDetector(spec())
    gray = det.render(pixels_per_square=80, margin=25)
    colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    a, b = det.detect(gray), det.detect(colour)
    assert a is not None and b is not None
    assert np.allclose(a.image_points, b.image_points)


def test_draw_detections_produces_an_overlay():
    det = CharucoDetector(spec())
    img = det.render(pixels_per_square=80, margin=25)
    found = det.detect(img)
    out = draw_detections(img, [found])
    assert out.ndim == 3 and out.shape[:2] == img.shape[:2]
    assert not np.array_equal(out, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
