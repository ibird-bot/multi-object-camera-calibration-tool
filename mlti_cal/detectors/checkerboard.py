"""
Plain checkerboard detection, with the coded boards masked out first.

Why masking exists
    A Charuco board IS a chessboard with Aruco markers inked into its white
    squares. `findChessboardCorners*` does not consume leftover corners some
    other detector declined -- it scans the WHOLE image for a complete grid of
    the declared size. So in a frame holding two Charuco boards and one plain
    checkerboard, the chessboard search is as likely to lock onto a Charuco
    board as onto the intended target, and when it does it reports a confident,
    fully-formed, entirely wrong detection.

    The fix is to run Charuco first, paint its boards out, and only then search
    for the plain board. That is what `mask_boards` does, and it is the step
    the intuition "the coded corners are taken, so the rest must be the
    chessboard" skips.

Two ways to build a mask, and why both are needed
    Detected Charuco corners are INTERIOR corners, so their convex hull stops
    one full square short of the printed edge on every side -- precisely the
    strip where a chessboard detector finds its false edges. Where enough of
    the board was detected, a homography (the board is planar, so this needs no
    intrinsics -- which do not exist yet at detection time) maps the board's
    full outline into the image and covers the whole sheet even where detection
    failed. Where only a small cluster of corners was found, that extrapolation
    is least reliable exactly where it matters, so a padded hull of the corners
    actually seen is used instead.

Orientation
    A plain checkerboard's corners carry no identity. OpenCV returns them in a
    consistent scan order but the end it starts from can flip between views, so
    the recovered board pose can differ by 180 degrees about the board normal.
    That is invisible in a single camera -- the pose absorbs it -- and NOT
    invisible in a rig, where one pose per (frame, board) is shared across all
    cameras, so two cameras that ordered the same board oppositely cannot both
    be fitted. It surfaces as huge residuals on SOME frames, which reads like
    an outlier problem rather than an ordering one.

    There is NO fix for it in this project today. `marker=True` looks like one
    and is not: measured against OpenCV 5.0.0, CALIB_CB_MARKER changes nothing
    about what `findChessboardCornersSB` returns, marked board or not. Use a
    Charuco board when orientation has to be unambiguous in a rig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from mlti_cal.detectors.base import Detection
from mlti_cal.detectors.masking import MaskGeometry, mask_boards
from mlti_cal.detectors.registry import DetectorKind, register_detector
from mlti_cal.detectors.settings import (
    CHECKERBOARD_CATALOG,
    CHECKERBOARD_SUMMARY,
    CheckerboardSettings,
)
from mlti_cal.options import Option


@dataclass
class CheckerboardSpec:
    """Everything needed to build a plain chessboard's 3D geometry."""

    id: str
    squares_x: int
    squares_y: int
    square_length: float

    def __post_init__(self) -> None:
        # SQUARES, not interior corners -- deliberately the same meaning as
        # CharucoBoardSpec. OpenCV's patternSize wants interior corners, which
        # is a classic off-by-one; converting once here is better than asking
        # the user to enter one board as squares and the next as corners.
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError(
                f"board {self.id}: squares_x/squares_y count SQUARES and must be "
                f"at least 3 each, got {self.squares_x}x{self.squares_y} "
                f"({self.squares_x - 1}x{self.squares_y - 1} interior corners)"
            )
        if self.square_length <= 0:
            raise ValueError(f"board {self.id}: square_length must be > 0")

    @property
    def pattern_size(self) -> tuple[int, int]:
        """(cols, rows) of INTERIOR corners -- what OpenCV asks for."""
        return (self.squares_x - 1, self.squares_y - 1)

    @property
    def num_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def object_points(self) -> np.ndarray:
        """
        (M,3) interior corners in the board frame, indexed by corner id.

        Row-major, matching the order OpenCV returns. The origin is the printed
        sheet's outer corner, so the first interior corner sits at
        (square_length, square_length) -- the same convention Charuco uses, which
        keeps `outline_object_points` correct and keeps the two kinds' board
        frames comparable.
        """
        cols, rows = self.pattern_size
        j, i = np.meshgrid(np.arange(cols), np.arange(rows))
        xy = np.stack([(j.ravel() + 1), (i.ravel() + 1)], axis=1) * self.square_length
        return np.hstack([xy, np.zeros((xy.shape[0], 1))])

    @property
    def outline_object_points(self) -> np.ndarray:
        """(4,2) outer corners of the printed sheet, in board coordinates."""
        w = self.squares_x * self.square_length
        h = self.squares_y * self.square_length
        return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=float)


class CheckerboardDetector:
    """Wraps one OpenCV chessboard finder plus its board geometry."""

    def __init__(self, spec: CheckerboardSpec, settings: CheckerboardSettings | None = None):
        self.spec = spec
        self.settings = settings or CheckerboardSettings()

    @property
    def object_points(self) -> np.ndarray:
        return self.spec.object_points

    @property
    def mask_geometry(self) -> MaskGeometry:
        return MaskGeometry(
            object_points=self.spec.object_points,
            outline=self.spec.outline_object_points,
            pitch=self.spec.square_length,
        )

    def detect(self, image: np.ndarray) -> Detection | None:
        """
        All-or-nothing: either the complete grid is found, or this returns None.

        There is no `min_corners` counterpart to the Charuco detector here
        because OpenCV offers no partial answer -- the grid matches at the
        declared size or it does not match at all.
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = self.settings.flags()
        if self.settings.algorithm == "SB":
            found, corners = cv2.findChessboardCornersSB(gray, self.spec.pattern_size, flags)
        else:
            found, corners = cv2.findChessboardCorners(gray, self.spec.pattern_size, flags)
            if found:
                # SB refines internally; the legacy finder returns quad-linked
                # corners that are integer-ish until this runs.
                win = int(self.settings.refine_win_size)
                corners = cv2.cornerSubPix(
                    gray,
                    np.asarray(corners, dtype=np.float32),
                    (win, win),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
                )
        if not found or corners is None:
            return None
        pts = np.asarray(corners, dtype=float).reshape(-1, 2)
        if pts.shape[0] != self.spec.num_corners:
            # Never observed with a fixed patternSize, but a silent size
            # mismatch here would misalign every corner against its object
            # point, which is worse than no detection.
            return None
        return Detection(
            board_id=self.spec.id,
            point_ids=np.arange(pts.shape[0], dtype=int),
            image_points=pts,
            kind="checkerboard",
        )


@dataclass
class MultiCheckerboardDetector:
    """
    Detects several plain checkerboards in one image.

    Each board found is masked out before the next is searched for, because two
    boards of different sizes still share their local structure: a 9x7 finder
    can lock onto part of an 11x8 board.

    Two boards with the SAME pattern size are refused. Nothing distinguishes
    them -- no markers, no identity -- so any assignment of one detection to one
    of them is a coin flip, and a coin flip written into a calibration is worse
    than an error.
    """

    specs: list[CheckerboardSpec]
    detectors: dict[str, CheckerboardDetector] = field(default_factory=dict)
    settings: CheckerboardSettings = field(default_factory=CheckerboardSettings)

    def __post_init__(self) -> None:
        self.validate()
        # Largest board first: it is the one most likely to be partly matched by
        # a smaller board's finder, so finding and masking it first removes the
        # ambiguity rather than leaving it to chance.
        self.specs = sorted(self.specs, key=lambda s: s.num_corners, reverse=True)
        self.detectors = {s.id: CheckerboardDetector(s, self.settings) for s in self.specs}

    def validate(self) -> None:
        seen: dict[tuple[int, int], str] = {}
        ids: set[str] = set()
        for spec in self.specs:
            if spec.id in ids:
                raise ValueError(f"duplicate board id {spec.id!r}")
            ids.add(spec.id)
            key = spec.pattern_size
            if key in seen:
                raise IndistinguishableBoardsError(
                    f"checkerboards {seen[key]!r} and {spec.id!r} are both "
                    f"{spec.squares_x}x{spec.squares_y} squares. A plain checkerboard "
                    f"carries no identity, so a detection cannot be attributed to "
                    f"either one. Use boards of different sizes, or make one of them "
                    f"a Charuco board."
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


class IndistinguishableBoardsError(ValueError):
    """Two plain checkerboards of identical size cannot be told apart."""


register_detector(
    DetectorKind(
        id="checkerboard",
        label="Checkerboard",
        description=(
            "Classic chessboard. The most accurate corners of any target here, "
            "and the least forgiving: the whole board must be visible and its "
            "corners carry no identity, so orientation is ambiguous. Coded "
            "boards in the same image are masked out before it searches."
        ),
        coded=False,
        glyph="square",
        spec_cls=CheckerboardSpec,
        group_cls=MultiCheckerboardDetector,
        settings_cls=CheckerboardSettings,
        catalog=CHECKERBOARD_CATALOG,
        summary=CHECKERBOARD_SUMMARY,
        board_fields=[
            Option(
                name="squares_x",
                kind="int",
                default=9,
                minimum=3,
                maximum=200,
                column="count_x",
                when="Squares across the board -- SQUARES, not interior corners, "
                "the same meaning a Charuco board gives it. OpenCV asks for "
                "interior corners and the conversion happens once, in the spec.",
            ),
            Option(
                name="squares_y",
                kind="int",
                default=7,
                minimum=3,
                maximum=200,
                column="count_y",
                when="Squares down the board. The whole board has to be visible for "
                "a frame to count at all -- there is no partial detection here.",
            ),
            Option(
                name="square_length",
                kind="float",
                default=0.030,
                minimum=1e-4,
                maximum=10.0,
                column="pitch",
                when="Side of one square, in metres. Sets the metric scale of the "
                "calibration: a wrong value leaves reprojection error untouched and "
                "scales the recovered baseline by the same factor.",
            ),
        ],
    )
)
