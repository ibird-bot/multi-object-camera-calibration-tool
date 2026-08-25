"""
Dot-grid (circle grid) detection.

On the registration question this detector deliberately does NOT answer
    A grid has no correspondence problem. `findCirclesGrid` returns the centres
    already ordered row by row, left to right, so dot k is dot k in every image
    that sees the board -- exactly the contract the checkerboard detector has.
    Nothing here matches points between images.

    That is worth stating because the obvious instinct for uncoded dots is to
    match them: nearest-neighbour between two frames, or ICP against a template.
    Both are wrong here and would be wrong even if this were an unstructured dot
    cloud. Nearest-neighbour in pixel coordinates only means anything when the
    board barely moved, and calibration deliberately demands large pose changes.
    ICP fits a RIGID transform, while a planar target reaches the image through a
    homography plus lens distortion -- the wrong model, and one that needs an
    initial alignment it cannot get. For an unstructured dot layout the right
    tool is RANSAC over homographies against the known layout, which is close to
    what `findCirclesGrid` already does internally for a grid.

Symmetric vs asymmetric
    Symmetric is a plain rectangular lattice. Asymmetric staggers every other
    row by half the row pitch, which is what lets the grid finder resolve the
    board's orientation instead of leaving the 180-degree flip a plain
    checkerboard suffers from. Prefer asymmetric.

Scale convention -- READ THIS
    `spacing` is the centre-to-centre distance between two adjacent dots in the
    SAME ROW, for both grid types. OpenCV's own asymmetric sample is written in
    terms of a half-pitch unit instead, so a spacing entered against that
    convention comes out a factor of two wrong. It is invisible in reprojection
    error and in the intrinsics -- it moves only the metric scale, which means a
    stereo baseline exactly twice or half what it should be. See
    `object_points` for the formula actually used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from mlti_cal.detectors.base import Detection
from mlti_cal.detectors.masking import MaskGeometry, mask_boards
from mlti_cal.detectors.registry import DetectorKind, register_detector
from mlti_cal.detectors.settings import (
    CIRCLE_GRID_CATALOG,
    CIRCLE_GRID_SUMMARY,
    CircleGridSettings,
)
from mlti_cal.options import Option

#: Recommended first -- `CircleGridSpec.grid_type` defaults to GRID_TYPES[0]
#: and the GUI offers them in this order, so the default is asymmetric in
#: every place a grid type is chosen.
GRID_TYPES = ("asymmetric", "symmetric")


@dataclass
class CircleGridSpec:
    """Everything needed to build a dot grid's 3D geometry."""

    id: str
    circles_x: int
    circles_y: int
    spacing: float
    grid_type: str = "asymmetric"

    def __post_init__(self) -> None:
        if self.grid_type not in GRID_TYPES:
            raise ValueError(
                f"board {self.id}: unknown grid_type {self.grid_type!r}; known: {list(GRID_TYPES)}"
            )
        if self.circles_x < 2 or self.circles_y < 2:
            raise ValueError(
                f"board {self.id}: circles_x/circles_y must be at least 2 each, "
                f"got {self.circles_x}x{self.circles_y}"
            )
        if self.spacing <= 0:
            raise ValueError(f"board {self.id}: spacing must be > 0")

    @property
    def pattern_size(self) -> tuple[int, int]:
        """(dots per row, number of rows) -- what OpenCV asks for."""
        return (self.circles_x, self.circles_y)

    @property
    def num_points(self) -> int:
        return self.circles_x * self.circles_y

    @property
    def object_points(self) -> np.ndarray:
        """
        (M,3) dot centres in the board frame, indexed by dot id.

        Row-major, matching the order OpenCV returns, origin at the first dot.

        Symmetric:   x = j * spacing,                 y = i * spacing
        Asymmetric:  x = (2j + i % 2) * spacing / 2,  y = i * spacing / 2

        Both give adjacent dots in the same row a centre distance of exactly
        `spacing`. The asymmetric form is OpenCV's own layout written in terms of
        that in-row pitch rather than its half-pitch unit; odd rows are offset
        sideways by half a pitch and rows are half a pitch apart vertically, so
        the nearest neighbour across rows sits at spacing/sqrt(2).
        """
        cols, rows = self.pattern_size
        j, i = np.meshgrid(np.arange(cols), np.arange(rows))
        j, i = j.ravel(), i.ravel()
        if self.grid_type == "asymmetric":
            x = (2 * j + (i % 2)) * (self.spacing / 2.0)
            y = i * (self.spacing / 2.0)
        else:
            x = j * self.spacing
            y = i * self.spacing
        return np.stack([x, y, np.zeros_like(x, dtype=float)], axis=1).astype(float)

    @property
    def outline_object_points(self) -> np.ndarray:
        """
        (4,2) the printed area, in board coordinates.

        Unlike a chessboard there is no square grid whose edge defines the sheet,
        so this is the bounding box of the dot centres grown by one full pitch --
        a proxy for the printed region, used only to paint this board out before
        another uncoded detector searches the image.
        """
        xy = self.object_points[:, :2]
        lo = xy.min(axis=0) - self.spacing
        hi = xy.max(axis=0) + self.spacing
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


class CircleGridDetector:
    """Wraps one blob detector plus `findCirclesGrid` and the board geometry."""

    def __init__(self, spec: CircleGridSpec, settings: CircleGridSettings | None = None):
        self.spec = spec
        self.settings = settings or CircleGridSettings()
        self._blob = self.settings.blob_detector()

    @property
    def object_points(self) -> np.ndarray:
        return self.spec.object_points

    @property
    def mask_geometry(self) -> MaskGeometry:
        return MaskGeometry(
            object_points=self.spec.object_points,
            outline=self.spec.outline_object_points,
            pitch=self.spec.spacing,
        )

    def detect(self, image: np.ndarray) -> Detection | None:
        """
        All-or-nothing: the complete grid, or None.

        `findCirclesGrid` returns True only when every centre was found AND
        ordered, so there is no partial answer to keep -- the same contract as
        the checkerboard, and for the same reason.
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.settings.invert_image:
            # Light dots on a dark sheet. See CircleGridSettings.invert_image for
            # the measurement: blobColor does not do this, inverting does.
            gray = cv2.bitwise_not(gray)
        found, centers = cv2.findCirclesGrid(
            gray,
            self.spec.pattern_size,
            flags=self.settings.flags(self.spec.grid_type),
            blobDetector=self._blob,
        )
        if not found or centers is None:
            return None
        pts = np.asarray(centers, dtype=float).reshape(-1, 2)
        if pts.shape[0] != self.spec.num_points:
            # Never observed with a fixed patternSize, but a silent size mismatch
            # would pair every dot with the wrong object point, which is worse
            # than no detection at all.
            return None
        return Detection(
            board_id=self.spec.id,
            point_ids=np.arange(pts.shape[0], dtype=int),
            image_points=pts,
            kind="circle_grid",
        )


class IndistinguishableGridsError(ValueError):
    """Two dot grids of identical size and type cannot be told apart."""


@dataclass
class MultiCircleGridDetector:
    """
    Detects several dot grids in one image, largest first, masking as it goes.

    Two grids with the same pattern size AND the same type are refused: nothing
    distinguishes them, so assigning a detection to one of them is a coin flip,
    and a coin flip written into a calibration is worse than an error. Two grids
    of the same size but different type ARE distinguishable, because the finder
    is told which layout to expect.
    """

    specs: list[CircleGridSpec]
    detectors: dict[str, CircleGridDetector] = field(default_factory=dict)
    settings: CircleGridSettings = field(default_factory=CircleGridSettings)

    def __post_init__(self) -> None:
        self.validate()
        self.specs = sorted(self.specs, key=lambda s: s.num_points, reverse=True)
        self.detectors = {s.id: CircleGridDetector(s, self.settings) for s in self.specs}

    def validate(self) -> None:
        seen: dict[tuple[int, int, str], str] = {}
        ids: set[str] = set()
        for spec in self.specs:
            if spec.id in ids:
                raise ValueError(f"duplicate board id {spec.id!r}")
            ids.add(spec.id)
            key = (*spec.pattern_size, spec.grid_type)
            if key in seen:
                raise IndistinguishableGridsError(
                    f"dot grids {seen[key]!r} and {spec.id!r} are both "
                    f"{spec.circles_x}x{spec.circles_y} {spec.grid_type} grids. A dot "
                    f"carries no identity, so a detection cannot be attributed to "
                    f"either one. Use grids of different sizes or types, or make one "
                    f"of them a Charuco board."
                )
            seen[key] = spec.id

    def object_points(self, board_id: str) -> np.ndarray:
        return self.detectors[board_id].object_points

    @property
    def mask_geometry(self) -> dict[str, MaskGeometry]:
        return {bid: d.mask_geometry for bid, d in self.detectors.items()}

    def detect_all(self, image: np.ndarray) -> list[Detection]:
        out: list[Detection] = []
        working = image
        geometry = self.mask_geometry
        for spec in self.specs:
            det = self.detectors[spec.id].detect(working)
            if det is None:
                continue
            out.append(det)
            if len(out) < len(self.specs) and self.settings.mask_other_boards:
                working = mask_boards(working, [det], geometry, self.settings)
        return out


register_detector(
    DetectorKind(
        id="circle_grid",
        label="Circle grid",
        description=(
            "Symmetric or asymmetric dot grid. Centroids survive defocus better "
            "than corners do, at the cost of a perspective bias: a circle images "
            "as an ellipse whose centroid is not the projected centre."
        ),
        coded=False,
        glyph="diamond",
        spec_cls=CircleGridSpec,
        group_cls=MultiCircleGridDetector,
        settings_cls=CircleGridSettings,
        catalog=CIRCLE_GRID_CATALOG,
        summary=CIRCLE_GRID_SUMMARY,
        board_fields=[
            Option(
                name="circles_x",
                kind="int",
                default=4,
                minimum=2,
                maximum=200,
                column="count_x",
                when="DOTS across the board, not squares. For an asymmetric grid "
                "this counts the dots in one row; the rows then interleave.",
            ),
            Option(
                name="circles_y",
                kind="int",
                default=11,
                minimum=2,
                maximum=200,
                column="count_y",
                when="Rows of dots. 4 x 11 asymmetric is the standard grid that ships with OpenCV.",
            ),
            Option(
                name="spacing",
                kind="float",
                default=0.020,
                minimum=1e-4,
                maximum=10.0,
                column="pitch",
                when="Centre-to-centre distance between two dots in the SAME ROW, "
                "in metres, for both grid types. OpenCV's own asymmetric sample is "
                "written against a half-pitch unit instead, so a value entered "
                "under that reading is out by exactly 2x -- invisible in "
                "reprojection error, and landing entirely on the metric scale.",
            ),
            Option(
                name="grid_type",
                kind="choice",
                default="asymmetric",
                choices=list(GRID_TYPES),
                when="Asymmetric staggers every other row by half a pitch, which is "
                "what lets the finder resolve the board's orientation; a symmetric "
                "grid leaves the same 180-degree flip a plain checkerboard has.",
                per_choice={
                    "asymmetric": "Preferred. Staggered rows, orientation resolved.",
                    "symmetric": "Plain rectangular lattice; orientation ambiguous.",
                },
            ),
        ],
    )
)
