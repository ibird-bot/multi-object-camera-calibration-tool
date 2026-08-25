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

from mlti_cal.detectors.base import Detection, draw_detections  # noqa: F401  (re-export)
from mlti_cal.detectors.masking import MaskGeometry
from mlti_cal.detectors.registry import DetectorKind, register_detector
from mlti_cal.detectors.settings import CHARUCO_CATALOG, CHARUCO_SUMMARY, CharucoSettings
from mlti_cal.options import Option

DICTIONARIES = {
    name: getattr(cv2.aruco, name) for name in dir(cv2.aruco) if name.startswith("DICT_")
}


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

    @property
    def outline_object_points(self) -> np.ndarray:
        """
        (4,2) outer corners of the PRINTED board, in board coordinates.

        Measured against the installed OpenCV rather than assumed:
        `getChessboardCorners` puts the board origin at the outer corner, so
        interior corners span [square_length, (squares-1) * square_length] while
        the printed sheet spans [0, squares * square_length]. The gap of one
        square on every side is exactly what a hull of detected corners misses,
        and it is where a chessboard detector finds its false positives.
        """
        w = self.squares_x * self.square_length
        h = self.squares_y * self.square_length
        return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=float)


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

    @property
    def mask_geometry(self) -> MaskGeometry:
        """What another detector needs to paint this board out of an image."""
        return MaskGeometry(
            object_points=self.object_points,
            outline=self.spec.outline_object_points,
            pitch=self.spec.square_length,
        )

    def detect(self, image: np.ndarray) -> Detection | None:
        """
        `detect(image)` and nothing else -- the same signature every detector has.

        `min_corners` used to be an override parameter here, threaded down from
        `BoardDetectors.detect` through `detect_all`. Nothing ever passed it, so
        all three layers only ever forwarded None to this line. It is read from
        the settings, where it is a documented catalog knob, and the uniform
        one-argument signature is what a third-party detector has to match.
        """
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
            kind="charuco",
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

    @property
    def mask_geometry(self) -> dict[str, MaskGeometry]:
        """
        Every board this group knows, keyed by id.

        Coded boards need this as much as uncoded ones do -- they are the FIRST
        thing painted out, because they are what an uncoded detector is most
        likely to mistake for its own target.
        """
        return {bid: d.mask_geometry for bid, d in self.detectors.items()}

    def detect_all(self, image: np.ndarray) -> list[Detection]:
        out = []
        for det in self.detectors.values():
            d = det.detect(image)
            if d is not None:
                out.append(d)
        return out


register_detector(
    DetectorKind(
        id="charuco",
        label="Charuco board",
        description=(
            "Chessboard corners interpolated from decoded ArUco markers. The "
            "markers give every corner an identity, so partial views and several "
            "boards in one image both work."
        ),
        coded=True,
        glyph="circle",
        spec_cls=CharucoBoardSpec,
        group_cls=MultiBoardDetector,
        settings_cls=CharucoSettings,
        catalog=CHARUCO_CATALOG,
        summary=CHARUCO_SUMMARY,
        board_fields=[
            Option(
                name="squares_x",
                kind="int",
                default=9,
                minimum=2,
                maximum=200,
                column="count_x",
                when="Squares across the board -- SQUARES, not interior corners. "
                "Entering this transposed decodes every marker and interpolates "
                "zero corners, with no error anywhere; see the warning the "
                "detector emits when it sees that signature.",
            ),
            Option(
                name="squares_y",
                kind="int",
                default=7,
                minimum=2,
                maximum=200,
                column="count_y",
                when="Squares down the board. If detection finds plenty of markers "
                "but no corners at all, try swapping this with squares_x before "
                "touching anything else.",
            ),
            Option(
                name="square_length",
                kind="float",
                default=0.030,
                minimum=1e-4,
                maximum=10.0,
                column="pitch",
                when="Side of one square, in metres. This is the only thing that "
                "sets the metric scale of the whole calibration: get it wrong and "
                "every intrinsic still fits perfectly while the recovered baseline "
                "is wrong by exactly the same factor.",
            ),
            Option(
                name="marker_length",
                kind="float",
                default=0.022,
                minimum=1e-4,
                maximum=10.0,
                when="Side of the ArUco marker printed inside a square, in metres. "
                "Must be smaller than square_length, or the board is refused.",
            ),
            Option(
                name="dictionary",
                kind="choice",
                default="DICT_4X4_250",
                choices=sorted(DICTIONARIES),
                when="Which ArUco dictionary the markers were printed from. It must "
                "hold at least as many markers as this board uses -- half its "
                "squares, starting at marker_id_offset.",
            ),
            Option(
                name="marker_id_offset",
                kind="int",
                default=0,
                minimum=0,
                maximum=100000,
                when="First marker id on this board. Boards sharing a dictionary "
                "must use DISJOINT id ranges, or a marker in the overlap cannot be "
                "attributed to a board and detection is refused outright.",
            ),
        ],
    )
)
