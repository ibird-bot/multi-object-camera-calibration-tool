"""
Painting a located board out of an image, so the next detector cannot find it.

Lives apart from any one detector because it serves them ALL: a Charuco board
is masked for the checkerboard's sake, a dot grid for the checkerboard's, a
checkerboard for the dot grid's. It used to live in `checkerboard.py`, which
made `circle_grid.py` import its masking from a module named after a different
target -- a dependency that was never real.

Two ways to build a mask, and why both are needed
    Detected corners of a coded board are INTERIOR corners, so their convex hull
    stops one full repeat short of the printed edge on every side -- precisely
    the strip where a chessboard detector finds its false edges. Where enough of
    the board was detected, a homography (the board is planar, so this needs no
    intrinsics -- which do not exist yet at detection time) maps the board's full
    outline into the image and covers the whole sheet even where detection
    failed. Where only a small cluster of points was found, that extrapolation is
    least reliable exactly where it matters, so a padded hull of the points
    actually seen is used instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from mlti_cal.detectors.base import Detection


@runtime_checkable
class MaskSettings(Protocol):
    """
    The masking knobs, as any detector's settings object must expose them.

    A Protocol rather than a concrete class because these belong to the detector
    ABOUT to search, not to the board being painted out -- how much clutter has
    to be gone before a search is trustworthy is a property of the search. Each
    uncoded detector therefore owns its own copy, and a third-party detector
    supplies these four names to take part in masking at all.
    """

    #: Paint other located boards out before searching.
    mask_other_boards: bool
    #: How far past a masked board's printed edge to extend, in that board's own
    #: repeat length (`MaskGeometry.pitch`).
    mask_padding_squares: float
    #: Grey level painted over a masked board.
    mask_fill_value: int
    #: How much of a board must be located before its full outline is projected
    #: rather than a padded hull of the points actually seen.
    mask_min_coverage: float


@dataclass
class MaskGeometry:
    """What `mask_boards` needs to know about a board it is painting out."""

    object_points: np.ndarray  # (M,3) indexed by corner id
    outline: np.ndarray  # (4,2) printed sheet in board coords
    pitch: float


def _homography_polygon(
    det: Detection, geom: MaskGeometry, padding: float, min_coverage: float
) -> np.ndarray | None:
    """
    The board's full outline projected into the image, or None if not trustworthy.

    Trustworthy means: at least 4 corners, a homography that actually fits, and
    detected corners spanning at least `min_coverage` of the board's area. That
    last test is the important one -- a homography from a tight cluster in one
    quadrant extrapolates worst exactly where the mask is needed.
    """
    ids = np.asarray(det.point_ids, dtype=int)
    if ids.size < 4 or ids.max(initial=-1) >= geom.object_points.shape[0]:
        return None
    src = geom.object_points[ids][:, :2].astype(np.float64)
    dst = np.asarray(det.image_points, dtype=np.float64)

    full = geom.object_points[:, :2]
    full_area = np.prod(full.max(0) - full.min(0))
    seen_area = np.prod(src.max(0) - src.min(0))
    if full_area <= 0 or seen_area / full_area < min_coverage:
        return None

    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        return None

    # The outline is an axis-aligned rectangle in BOARD coordinates, so padding
    # is a plain expansion there. Doing it in board space rather than in pixels
    # is what makes the padding uniform after projection, whatever the tilt.
    pad = padding * geom.pitch
    lo = geom.outline.min(axis=0) - pad
    hi = geom.outline.max(axis=0) + pad
    outline = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    pts = cv2.perspectiveTransform(outline.reshape(-1, 1, 2).astype(np.float64), H)
    poly = pts.reshape(-1, 2)
    if not np.isfinite(poly).all():
        return None
    return poly


def _hull_polygon(det: Detection, geom: MaskGeometry, padding: float) -> np.ndarray | None:
    """
    Fallback: the convex hull of the corners actually seen, grown outward.

    Grown by one full square (the interior corners stop one square inside the
    printed edge) plus the padding, converted to pixels by measuring the
    board-to-image scale off this detection rather than assuming one.
    """
    pts = np.asarray(det.image_points, dtype=np.float64)
    if pts.shape[0] < 3:
        return None
    ids = np.asarray(det.point_ids, dtype=int)
    if ids.max(initial=-1) >= geom.object_points.shape[0]:
        return None
    board = geom.object_points[ids][:, :2]
    board_span = float(np.linalg.norm(board.max(0) - board.min(0)))
    image_span = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    if board_span <= 0:
        return None
    px_per_m = image_span / board_span
    grow = (1.0 + padding) * geom.pitch * px_per_m

    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    centre = hull.mean(axis=0)
    offsets = hull - centre
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    return hull + offsets / np.maximum(norms, 1e-9) * grow


def mask_polygons(
    detections: list[Detection],
    geometry: dict[str, MaskGeometry],
    settings: MaskSettings,
) -> list[np.ndarray]:
    """One polygon per detection that could be localised. Exposed for the GUI."""
    out = []
    for det in detections:
        geom = geometry.get(det.board_id)
        if geom is None:
            continue
        poly = _homography_polygon(
            det, geom, settings.mask_padding_squares, settings.mask_min_coverage
        )
        if poly is None:
            poly = _hull_polygon(det, geom, settings.mask_padding_squares)
        if poly is not None:
            out.append(poly)
    return out


def mask_boards(
    image: np.ndarray,
    detections: list[Detection],
    geometry: dict[str, MaskGeometry],
    settings: MaskSettings,
) -> np.ndarray:
    """
    A copy of `image` with every located board painted over in flat grey.

    Returns the image unchanged (well, copied) when there is nothing to mask, so
    a caller never has to branch on whether masking applied.
    """
    polys = mask_polygons(detections, geometry, settings)
    if not polys:
        return image.copy()
    out = image.copy()
    fill = int(settings.mask_fill_value)
    value = fill if out.ndim == 2 else (fill, fill, fill)
    cv2.fillPoly(out, [np.round(p).astype(np.int32) for p in polys], value)
    return out
