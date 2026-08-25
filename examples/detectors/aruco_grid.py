"""
A worked example of a third-party detector: a plain ArUco grid board.

Drop this file in

    ~/.mlti_cal/detectors/

and "Aruco grid board" appears in the + Object menu the next time the app
starts. Nothing in the installed package changes.

Why THIS example
    Plain ArUco was a built-in and was deliberately removed: for calibrating a
    planar target it is strictly worse than Charuco, whose markers give identity
    while the measurement comes from interpolated chessboard corners, which are
    markedly more accurate than the markers' own corners.

    But it is not worthless. Markers scattered over a rig, a board too large to
    hold a full chessboard, a target that must be readable at a glance -- those
    are real, and none of them justify carrying a second-best detector in the
    package for everyone. That is exactly the shape of thing a plugin is for,
    which is why it makes an honest example rather than a contrived one.

    Accuracy warning, because this is a real detector and someone will use it:
    marker corners are the quad-fit corners of a printed square, not corners
    interpolated from a saddle point. Expect worse reprojection error than a
    Charuco board of the same size. Prefer Charuco when the target is planar.

The contract, in full
    A detector is one `DetectorKind`, registered. It needs:

    spec_cls      A dataclass taking `id` plus exactly the `board_fields` below.
                  Must expose `object_points` -> (M,3) in the board frame,
                  indexed by point id, and `outline_object_points` -> (4,2), the
                  printed sheet's outline, used to paint this board out of the
                  image for the other detectors.

    group_cls     Built as `group_cls(specs, settings=settings)`. Must expose
                  `specs`, `detectors` (id -> object), `settings`,
                  `object_points(board_id)`, `mask_geometry` (id -> MaskGeometry)
                  and `detect_all(image) -> list[Detection]`.

    settings_cls  A dataclass with `to_dict()` / `from_dict()`. Every field must
                  appear in `catalog` or registration is refused -- a knob the
                  user can see but nothing reads is worse than no knob.

    board_fields  One `Option` per spec field, minus `id`. These become the GUI
                  table columns AND the JSON keys of a saved board, so they are
                  the target's own vocabulary. A field meaning the same thing as
                  another kind's shares a table column via `Option.column`.

    coded         True when every feature carries its own identity, as here.
                  Coded kinds run first and are painted out before the uncoded
                  ones search.

`Detection.point_ids` must index into `spec.object_points`. Pairing them wrongly
is the one error nothing downstream can catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from mlti_cal.detectors.base import Detection
from mlti_cal.detectors.masking import MaskGeometry, mask_boards
from mlti_cal.detectors.registry import DetectorKind, register_detector
from mlti_cal.options import Option

KIND_ID = "aruco_grid"
DICTIONARIES = {n: getattr(cv2.aruco, n) for n in dir(cv2.aruco) if n.startswith("DICT_")}


@dataclass
class ArucoGridSpec:
    """A grid of square markers, evenly separated. Four corners per marker."""

    id: str
    markers_x: int
    markers_y: int
    marker_length: float
    marker_separation: float
    dictionary: str = "DICT_4X4_250"
    marker_id_offset: int = 0

    def __post_init__(self) -> None:
        if self.dictionary not in DICTIONARIES:
            raise KeyError(f"unknown dictionary {self.dictionary!r}")
        if self.markers_x < 1 or self.markers_y < 1:
            raise ValueError(f"board {self.id}: needs at least one marker each way")
        if self.marker_length <= 0 or self.marker_separation < 0:
            raise ValueError(f"board {self.id}: marker_length must be > 0, separation >= 0")

    @property
    def num_markers(self) -> int:
        return self.markers_x * self.markers_y

    @property
    def num_points(self) -> int:
        return self.num_markers * 4

    def build_board(self):
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[self.dictionary])
        ids = np.arange(
            self.marker_id_offset, self.marker_id_offset + self.num_markers, dtype=np.int32
        )
        if ids.max(initial=-1) >= len(dictionary.bytesList):
            raise ValueError(
                f"board {self.id}: needs marker ids up to {ids.max()} but "
                f"{self.dictionary} only holds {len(dictionary.bytesList)}"
            )
        return cv2.aruco.GridBoard(
            (self.markers_x, self.markers_y),
            self.marker_length,
            self.marker_separation,
            dictionary,
            ids,
        )

    @property
    def object_points(self) -> np.ndarray:
        """
        (4*N, 3) marker corners, point id = marker index * 4 + corner index.

        The ordering is OpenCV's own `getObjPoints()`, flattened. Keeping the
        detector's id arithmetic identical to this flattening is the whole
        correctness requirement -- a corner paired with the wrong object point
        produces a calibration nothing downstream can flag as wrong.
        """
        pts = np.asarray(self.build_board().getObjPoints(), dtype=float)
        return pts.reshape(-1, 3)

    @property
    def outline_object_points(self) -> np.ndarray:
        xy = self.object_points[:, :2]
        lo = xy.min(axis=0) - self.marker_separation
        hi = xy.max(axis=0) + self.marker_separation
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


@dataclass
class ArucoGridSettings:
    """Every field here must appear in CATALOG, or registration is refused."""

    corner_refinement: str = "SUBPIX"
    error_correction_rate: float = 0.6
    mask_other_boards: bool = False
    mask_padding_squares: float = 0.6
    mask_fill_value: int = 128
    mask_min_coverage: float = 0.5

    REFINEMENT = {
        "NONE": cv2.aruco.CORNER_REFINE_NONE,
        "SUBPIX": cv2.aruco.CORNER_REFINE_SUBPIX,
        "CONTOUR": cv2.aruco.CORNER_REFINE_CONTOUR,
    }

    def __post_init__(self) -> None:
        if self.corner_refinement not in self.REFINEMENT:
            raise ValueError(f"unknown corner_refinement {self.corner_refinement!r}")

    def detector_params(self):
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = self.REFINEMENT[self.corner_refinement]
        p.errorCorrectionRate = float(self.error_correction_rate)
        return p

    def to_dict(self) -> dict:
        return {
            "corner_refinement": self.corner_refinement,
            "error_correction_rate": self.error_correction_rate,
            "mask_other_boards": self.mask_other_boards,
            "mask_padding_squares": self.mask_padding_squares,
            "mask_fill_value": self.mask_fill_value,
            "mask_min_coverage": self.mask_min_coverage,
        }

    @staticmethod
    def from_dict(data: dict | None) -> ArucoGridSettings:
        if not data:
            return ArucoGridSettings()
        known = set(ArucoGridSettings().to_dict())
        return ArucoGridSettings(**{k: v for k, v in data.items() if k in known})


CATALOG = [
    Option(
        name="corner_refinement",
        kind="choice",
        default="SUBPIX",
        choices=["NONE", "SUBPIX", "CONTOUR"],
        when="How marker corners are refined. Unlike Charuco these corners ARE the "
        "measurement -- nothing is interpolated from them -- so this setting bounds "
        "the accuracy of the whole calibration.",
        per_choice={
            "NONE": "Quad-fit corners, roughly 0.3-0.5 px of noise. Diagnostic only.",
            "SUBPIX": "Iterative subpixel fit against the local gradient. The default.",
            "CONTOUR": "Line fit to the marker contour; steadier on blurred images.",
        },
    ),
    Option(
        name="error_correction_rate",
        kind="float",
        default=0.6,
        minimum=0.0,
        maximum=1.0,
        when="Fraction of the dictionary's error-correction capability used when "
        "decoding an id. Higher reads more damaged markers and raises the chance of "
        "a WRONG id -- which with several boards means corners on the wrong board.",
    ),
    Option(
        name="mask_other_boards",
        kind="bool",
        default=False,
        when="Paint boards already located out of the image before searching. Off by "
        "default: marker decoding is not fooled by clutter the way a chessboard "
        "search is, so there is nothing to gain and a mask edge to lose.",
    ),
    Option(
        name="mask_padding_squares",
        kind="float",
        default=0.6,
        minimum=0.0,
        maximum=5.0,
        when="How far past a masked board's printed edge the mask extends, in that "
        "board's own repeat length. Only read when mask_other_boards is on.",
    ),
    Option(
        name="mask_fill_value",
        kind="int",
        default=128,
        minimum=0,
        maximum=255,
        when="Grey level painted over a masked board. Mid-grey rather than black, so "
        "the fill does not become the strongest edge in the picture.",
    ),
    Option(
        name="mask_min_coverage",
        kind="float",
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        when="How much of a board must be located before its mask is built from the "
        "projected outline rather than a padded hull of the points actually seen.",
    ),
]


class ArucoGridDetector:
    """One board: decode its markers, keep their corners."""

    def __init__(self, spec: ArucoGridSpec, settings: ArucoGridSettings | None = None):
        self.spec = spec
        self.settings = settings or ArucoGridSettings()
        self.board = spec.build_board()
        self._detector = cv2.aruco.ArucoDetector(
            self.board.getDictionary(), self.settings.detector_params()
        )
        self._first_id = spec.marker_id_offset

    @property
    def object_points(self) -> np.ndarray:
        return self.spec.object_points

    @property
    def mask_geometry(self) -> MaskGeometry:
        return MaskGeometry(
            object_points=self.spec.object_points,
            outline=self.spec.outline_object_points,
            pitch=self.spec.marker_length + self.spec.marker_separation,
        )

    def detect(self, image: np.ndarray) -> Detection | None:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None
        ids = np.asarray(ids).reshape(-1)
        keep = (ids >= self._first_id) & (ids < self._first_id + self.spec.num_markers)
        if not keep.any():
            return None
        point_ids: list[int] = []
        pts: list[np.ndarray] = []
        for marker_corners, marker_id in zip(np.asarray(corners)[keep], ids[keep], strict=True):
            index = int(marker_id) - self._first_id
            quad = np.asarray(marker_corners, dtype=float).reshape(-1, 2)
            for corner in range(4):
                point_ids.append(index * 4 + corner)
                pts.append(quad[corner])
        return Detection(
            board_id=self.spec.id,
            point_ids=np.array(point_ids, dtype=int),
            image_points=np.array(pts, dtype=float),
            marker_ids=ids[keep],
            num_markers=int(keep.sum()),
            kind=KIND_ID,
        )


@dataclass
class MultiArucoGridDetector:
    """Every board of this kind in one image."""

    specs: list[ArucoGridSpec]
    detectors: dict[str, ArucoGridDetector] = field(default_factory=dict)
    settings: ArucoGridSettings = field(default_factory=ArucoGridSettings)

    def __post_init__(self) -> None:
        by_dict: dict[str, list[ArucoGridSpec]] = {}
        ids: set[str] = set()
        for spec in self.specs:
            if spec.id in ids:
                raise ValueError(f"duplicate board id {spec.id!r}")
            ids.add(spec.id)
            by_dict.setdefault(spec.dictionary, []).append(spec)
        for name, group in by_dict.items():
            spans = sorted(
                (s.marker_id_offset, s.marker_id_offset + s.num_markers, s.id) for s in group
            )
            for (lo_a, hi_a, a), (lo_b, hi_b, b) in zip(spans, spans[1:], strict=False):
                if lo_a < hi_b and lo_b < hi_a:
                    raise ValueError(
                        f"boards {a!r} and {b!r} both use {name} with overlapping "
                        f"marker ids [{lo_a},{hi_a}) and [{lo_b},{hi_b}). A marker in "
                        f"the overlap cannot be attributed to a board."
                    )
        self.detectors = {s.id: ArucoGridDetector(s, self.settings) for s in self.specs}

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
            if self.settings.mask_other_boards and len(out) < len(self.specs):
                working = mask_boards(working, [det], geometry, self.settings)
        return out


register_detector(
    DetectorKind(
        id=KIND_ID,
        label="Aruco grid board",
        description=(
            "A grid of separated ArUco markers, using the marker corners directly "
            "with no chessboard interpolation. Every corner carries an identity, so "
            "partial views work -- but marker corners are markedly less accurate "
            "than the interpolated corners a Charuco board gives. Prefer Charuco "
            "for a planar target; this is for layouts a chessboard cannot cover."
        ),
        coded=True,
        glyph="triangle",
        origin="example plugin",
        spec_cls=ArucoGridSpec,
        group_cls=MultiArucoGridDetector,
        settings_cls=ArucoGridSettings,
        catalog=CATALOG,
        summary=(
            "Plain ArUco grid detection. The marker corners ARE the measurement "
            "here, so corner refinement bounds the accuracy of everything after it."
        ),
        board_fields=[
            Option(
                name="markers_x",
                kind="int",
                default=5,
                minimum=1,
                maximum=200,
                column="count_x",
                when="Markers across the board.",
            ),
            Option(
                name="markers_y",
                kind="int",
                default=7,
                minimum=1,
                maximum=200,
                column="count_y",
                when="Markers down the board.",
            ),
            Option(
                name="marker_length",
                kind="float",
                default=0.030,
                minimum=1e-4,
                maximum=10.0,
                column="pitch",
                when="Side of one printed marker, in metres. Together with the "
                "separation this sets the metric scale of the calibration.",
            ),
            Option(
                name="marker_separation",
                kind="float",
                default=0.006,
                minimum=0.0,
                maximum=10.0,
                when="Gap between neighbouring markers, in metres. Markers that "
                "touch cannot be segmented apart, so this must be greater than zero "
                "on a real printed board.",
            ),
            Option(
                name="dictionary",
                kind="choice",
                default="DICT_4X4_250",
                choices=sorted(DICTIONARIES),
                when="Which ArUco dictionary the markers were printed from. It must "
                "hold at least markers_x * markers_y ids beyond the offset.",
            ),
            Option(
                name="marker_id_offset",
                kind="int",
                default=0,
                minimum=0,
                maximum=100000,
                when="First marker id on this board. Boards sharing a dictionary "
                "must use disjoint id ranges, or a marker in the overlap cannot be "
                "attributed to a board.",
            ),
        ],
    )
)
