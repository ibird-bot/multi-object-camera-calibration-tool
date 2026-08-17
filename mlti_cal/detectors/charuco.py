"""
Charuco detection, multi-board aware.

Targets the OpenCV 5 API. Verified against the installed build rather than
assumed: `cv2.aruco.CharucoDetector.detectBoard` exists and
`interpolateCornersCharuco` does NOT (it was removed after OpenCV 4). The
sibling project's docstring still claims the latter -- it is stale, and code
written from it will not run here.

`detectBoard` returns squeezed arrays: charuco corners as (N,2) and ids as
(N,), not the (N,1,2) some 4.x code expects.

Multi-board
    Several boards can appear in one image, including boards drawn from the
    same ArUco dictionary. That only works if their marker ID ranges are
    disjoint -- otherwise a marker is genuinely ambiguous and the detector will
    happily assign it to the wrong board. `MultiBoardDetector` refuses to
    construct in that situation instead of producing plausible garbage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np

from mlti_cal.detectors.settings import CharucoSettings

DICTIONARIES = {
    name: getattr(cv2.aruco, name) for name in dir(cv2.aruco) if name.startswith("DICT_")
}


@dataclass
class Detection:
    """One board found in one image."""

    board_id: str
    point_ids: np.ndarray  # (K,) charuco corner ids
    image_points: np.ndarray  # (K,2) subpixel corners
    marker_ids: np.ndarray | None = None
    num_markers: int = 0

    @property
    def num_points(self) -> int:
        return int(self.point_ids.size)


@dataclass
class CharucoBoardSpec:
    """Everything needed to build the board and its 3D geometry."""

    id: str
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    dictionary: str = "DICT_4X4_50"
    marker_id_offset: int = 0
    #: Markers that must be decoded around a chessboard corner before it is
    #: interpolated. 2 is OpenCV's default and the right choice -- see the
    #: measurement in `CharucoDetector.__init__` before lowering it.
    min_markers: int = 2

    def __post_init__(self) -> None:
        if self.dictionary not in DICTIONARIES:
            raise KeyError(
                f"unknown dictionary {self.dictionary!r}; known: {sorted(DICTIONARIES)[:8]}..."
            )
        if self.marker_length >= self.square_length:
            raise ValueError(
                f"board {self.id}: marker_length ({self.marker_length}) must be "
                f"smaller than square_length ({self.square_length})"
            )
        if self.min_markers not in (1, 2):
            raise ValueError(f"board {self.id}: min_markers must be 1 or 2")

    @property
    def num_markers(self) -> int:
        """Charuco places a marker on every other square."""
        return (self.squares_x * self.squares_y) // 2

    @property
    def marker_id_range(self) -> tuple[int, int]:
        return self.marker_id_offset, self.marker_id_offset + self.num_markers

    @property
    def num_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)


class CharucoDetector:
    """Wraps one `cv2.aruco.CharucoDetector` plus its board geometry."""

    def __init__(self, spec: CharucoBoardSpec, settings: CharucoSettings | None = None):
        self.spec = spec
        self.settings = settings or CharucoSettings()
        self.dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[spec.dictionary])
        ids = np.arange(*spec.marker_id_range, dtype=np.int32)
        if ids.max(initial=-1) >= len(self.dictionary.bytesList):
            raise ValueError(
                f"board {spec.id}: needs marker ids up to {ids.max()} but "
                f"{spec.dictionary} only holds {len(self.dictionary.bytesList)}. "
                f"Use a larger dictionary or a smaller board."
            )
        self.board = cv2.aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y),
            spec.square_length,
            spec.marker_length,
            self.dictionary,
            ids,
        )
        # Every value here now comes from CharucoSettings, whose defaults
        # reproduce exactly what used to be hardcoded at this spot. The notes
        # that justified those choices live in CHARUCO_CATALOG, next to the
        # defaults themselves, so the tooltip and the code cannot disagree.
        params = cv2.aruco.DetectorParameters()
        self.settings.apply_to_detector_params(params)
        self._detector = cv2.aruco.CharucoDetector(self.board)
        self._detector.setDetectorParameters(params)

        # CharucoParameters defaults are wrong for multi-board frames and
        # leaving them alone silently costs most of the harder board. The
        # measured evidence for `try_refine_markers` is in CHARUCO_CATALOG.
        #
        # `minMarkers` is per BOARD, not per run, so it stays on the spec.
        # Its default of 2 is deliberate: dropping it to 1 looks attractive --
        # another 38 corners on the test set -- but those corners are
        # interpolated from a homography fitted to one marker, and the same
        # set showed RMS degrade 0.225 -> 0.330 px for that 4%. Coverage is
        # not the objective; accuracy is.
        charuco_params = cv2.aruco.CharucoParameters()
        self.settings.apply_to_charuco_params(charuco_params, spec.min_markers)
        self._detector.setCharucoParameters(charuco_params)
        self._warned_transposed = False

    @property
    def object_points(self) -> np.ndarray:
        """(M,3) chessboard corners in the board frame, indexed by corner id."""
        return np.asarray(self.board.getChessboardCorners(), dtype=float).reshape(-1, 3)

    def detect(self, image: np.ndarray, min_corners: int | None = None) -> Detection | None:
        """`min_corners` defaults to the value in `settings`; pass one to override."""
        if min_corners is None:
            min_corners = self.settings.min_corners
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, marker_corners, marker_ids = self._detector.detectBoard(gray)
        n_markers = 0 if marker_ids is None else int(np.asarray(marker_ids).size)
        n_corners = 0 if ids is None else int(np.asarray(ids).size)
        self._warn_if_grid_looks_transposed(n_markers, n_corners)
        if corners is None or ids is None:
            return None
        corners = np.asarray(corners, dtype=float).reshape(-1, 2)
        ids = np.asarray(ids, dtype=int).reshape(-1)
        if ids.size < min_corners:
            return None
        return Detection(
            board_id=self.spec.id,
            point_ids=ids,
            image_points=corners,
            marker_ids=None if marker_ids is None else np.asarray(marker_ids).reshape(-1),
            num_markers=0 if marker_ids is None else int(np.asarray(marker_ids).size),
        )

    def _warn_if_grid_looks_transposed(self, n_markers: int, n_corners: int) -> None:
        """
        Many markers decoded, zero corners interpolated -- almost always a wrong
        squares_x/squares_y.

        This exact failure is silent and cost real debugging time: a 12x9 board
        entered as 9x12 decodes all 54 of its markers perfectly and then yields
        0 of 88 corners, with no error anywhere. "No detections" reads as a bad
        image, so the search goes to lighting, focus and dictionary -- none of
        which are wrong. Marker IDs carry no orientation, so OpenCV cannot tell
        a transposed grid from a board that simply is not there.

        The signature is unambiguous: plenty of markers, no corners. A partial
        view at the frame edge shows few markers AND few corners, so it does not
        trip this. Warned once per detector -- per image would be 11 identical
        walls of text on this dataset alone.
        """
        if self._warned_transposed or n_corners > 0 or n_markers < 4:
            return
        self._warned_transposed = True
        s = self.spec
        warnings.warn(
            f"board {s.id!r}: decoded {n_markers} markers but interpolated 0 "
            f"chessboard corners. The board is in frame and the dictionary is "
            f"right, so the geometry is not: try squares_x={s.squares_y}, "
            f"squares_y={s.squares_x} (currently {s.squares_x}x{s.squares_y}). "
            f"If that is not it, the board may have been printed with the "
            f"pre-OpenCV-4.6 marker layout.",
            RuntimeWarning,
            stacklevel=3,
        )

    def render(self, pixels_per_square: int = 100, margin: int = 20) -> np.ndarray:
        """Render the board to an image -- used for printing and for tests."""
        w = self.spec.squares_x * pixels_per_square
        h = self.spec.squares_y * pixels_per_square
        return self.board.generateImage((w, h), marginSize=margin)


class IdCollisionError(ValueError):
    """Two boards share a dictionary and overlapping marker IDs."""


@dataclass
class MultiBoardDetector:
    """Detects several boards in one image, with collision validation."""

    specs: list[CharucoBoardSpec]
    detectors: dict[str, CharucoDetector] = field(default_factory=dict)
    #: One set of detector knobs for the whole run. Per-board geometry lives on
    #: the spec; these are properties of the images, not of the boards.
    settings: CharucoSettings = field(default_factory=CharucoSettings)

    def __post_init__(self) -> None:
        self.validate()
        self.detectors = {s.id: CharucoDetector(s, self.settings) for s in self.specs}

    def validate(self) -> None:
        seen: dict[str, str] = {}
        for spec in self.specs:
            if spec.id in seen:
                raise ValueError(f"duplicate board id {spec.id!r}")
            seen[spec.id] = spec.dictionary
        by_dict: dict[str, list[CharucoBoardSpec]] = {}
        for spec in self.specs:
            by_dict.setdefault(spec.dictionary, []).append(spec)
        for dict_name, group in by_dict.items():
            ranges = sorted((s.marker_id_range, s.id) for s in group)
            for (lo_a, hi_a), a in ranges:
                for (lo_b, hi_b), b in ranges:
                    if a >= b:
                        continue
                    if lo_a < hi_b and lo_b < hi_a:
                        raise IdCollisionError(
                            f"boards {a!r} and {b!r} both use {dict_name} with "
                            f"overlapping marker ids [{lo_a},{hi_a}) and "
                            f"[{lo_b},{hi_b}). A marker in the overlap cannot be "
                            f"attributed to a board. Set marker_id_offset so the "
                            f"ranges are disjoint, or use different dictionaries."
                        )

    def object_points(self, board_id: str) -> np.ndarray:
        return self.detectors[board_id].object_points

    def detect_all(self, image: np.ndarray, min_corners: int | None = None) -> list[Detection]:
        out = []
        for det in self.detectors.values():
            d = det.detect(image, min_corners=min_corners)
            if d is not None:
                out.append(d)
        return out


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    radius: int = 4,
    labels: bool = True,
    colour: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """
    Overlay detected corners -- used by the GUI detection view.

    `labels` and `colour` exist for the thumbnail-sized overlay. At 128 px the
    per-corner id text is an unreadable smear that hides the corners it
    annotates, and a fixed colour is wanted there so "has corners" reads at a
    glance; the full-size preview keeps the per-board palette, which is what
    tells two boards in one image apart.
    """
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    palette = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for i, det in enumerate(detections):
        c = colour if colour is not None else palette[i % len(palette)]
        for (x, y), pid in zip(det.image_points, det.point_ids, strict=True):
            cv2.circle(canvas, (int(round(x)), int(round(y))), radius, c, -1)
            if not labels:
                continue
            cv2.putText(
                canvas,
                str(int(pid)),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                c,
                1,
                cv2.LINE_AA,
            )
    return canvas
