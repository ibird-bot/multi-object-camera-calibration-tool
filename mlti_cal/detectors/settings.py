"""
Every detector choice, in one place the user can see and change.

These used to be buried in `CharucoDetector.__init__`. Detection quality
dominates everything downstream -- a corner that is 0.5 px wrong is 0.5 px of
error no solver can remove -- so hiding these was the least defensible part of
the pipeline.

Defaults
    The aruco values below are OpenCV's own defaults, written out literally so
    they are readable here rather than only discoverable at runtime.
    `test_detection_settings_match_opencv_defaults` asserts they still match the
    installed build, so a future OpenCV cannot change them behind our back.

    TWO fields deliberately differ from OpenCV: `corner_refinement` is SUBPIX
    (OpenCV says NONE) and `try_refine_markers` is True (OpenCV says False).
    Both were already the behaviour of this project before these settings
    existed, and both are measured wins -- see the notes in the catalog.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2

from mlti_cal.options import Option

#: name -> cv2.aruco.CORNER_REFINE_* value.
CORNER_REFINEMENT = {
    "NONE": cv2.aruco.CORNER_REFINE_NONE,
    "SUBPIX": cv2.aruco.CORNER_REFINE_SUBPIX,
    "CONTOUR": cv2.aruco.CORNER_REFINE_CONTOUR,
    "APRILTAG": cv2.aruco.CORNER_REFINE_APRILTAG,
}


@dataclass
class CharucoSettings:
    """Detector knobs, applied to every board in a run."""

    # -- corner refinement -------------------------------------------------
    corner_refinement: str = "SUBPIX"
    corner_refinement_win_size: int = 5
    corner_refinement_max_iterations: int = 30
    corner_refinement_min_accuracy: float = 0.1
    # -- charuco interpolation --------------------------------------------
    try_refine_markers: bool = True
    check_markers: bool = True
    min_corners: int = 6
    # -- marker detection --------------------------------------------------
    adaptive_thresh_win_size_min: int = 3
    adaptive_thresh_win_size_max: int = 23
    adaptive_thresh_win_size_step: int = 10
    adaptive_thresh_constant: float = 7.0
    min_marker_perimeter_rate: float = 0.03
    max_marker_perimeter_rate: float = 4.0
    polygonal_approx_accuracy_rate: float = 0.03
    min_corner_distance_rate: float = 0.05
    min_distance_to_border: int = 3
    error_correction_rate: float = 0.6

    def __post_init__(self) -> None:
        if self.corner_refinement not in CORNER_REFINEMENT:
            raise ValueError(
                f"unknown corner_refinement {self.corner_refinement!r}; "
                f"known: {sorted(CORNER_REFINEMENT)}"
            )
        if self.adaptive_thresh_win_size_min > self.adaptive_thresh_win_size_max:
            raise ValueError(
                f"adaptive_thresh_win_size_min ({self.adaptive_thresh_win_size_min}) "
                f"exceeds max ({self.adaptive_thresh_win_size_max}); no window size "
                f"would ever be tried"
            )
        if self.min_marker_perimeter_rate >= self.max_marker_perimeter_rate:
            raise ValueError(
                f"min_marker_perimeter_rate ({self.min_marker_perimeter_rate}) must be "
                f"below max ({self.max_marker_perimeter_rate}); no marker could match"
            )

    # -- OpenCV boundary ---------------------------------------------------
    def apply_to_detector_params(self, params) -> None:
        """Write these onto a `cv2.aruco.DetectorParameters`."""
        params.cornerRefinementMethod = CORNER_REFINEMENT[self.corner_refinement]
        params.cornerRefinementWinSize = int(self.corner_refinement_win_size)
        params.cornerRefinementMaxIterations = int(self.corner_refinement_max_iterations)
        params.cornerRefinementMinAccuracy = float(self.corner_refinement_min_accuracy)
        params.adaptiveThreshWinSizeMin = int(self.adaptive_thresh_win_size_min)
        params.adaptiveThreshWinSizeMax = int(self.adaptive_thresh_win_size_max)
        params.adaptiveThreshWinSizeStep = int(self.adaptive_thresh_win_size_step)
        params.adaptiveThreshConstant = float(self.adaptive_thresh_constant)
        params.minMarkerPerimeterRate = float(self.min_marker_perimeter_rate)
        params.maxMarkerPerimeterRate = float(self.max_marker_perimeter_rate)
        params.polygonalApproxAccuracyRate = float(self.polygonal_approx_accuracy_rate)
        params.minCornerDistanceRate = float(self.min_corner_distance_rate)
        params.minDistanceToBorder = int(self.min_distance_to_border)
        params.errorCorrectionRate = float(self.error_correction_rate)

    def apply_to_charuco_params(self, params, min_markers: int) -> None:
        """Write these onto a `cv2.aruco.CharucoParameters`. `min_markers` is per board."""
        params.tryRefineMarkers = bool(self.try_refine_markers)
        params.checkMarkers = bool(self.check_markers)
        params.minMarkers = int(min_markers)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict | None) -> CharucoSettings:
        """Tolerant of unknown keys, so an older config still loads."""
        if not data:
            return CharucoSettings()
        known = {f for f in CharucoSettings().to_dict()}
        return CharucoSettings(**{k: v for k, v in data.items() if k in known})


CHARUCO_SUMMARY = (
    "Charuco detection. Runs before any calibration and bounds everything "
    "after it: the solver cannot recover accuracy the detector never found. "
    "Changing anything here invalidates existing detections."
)

CHARUCO_CATALOG: list[Option] = [
    Option(
        name="corner_refinement",
        kind="choice",
        default="SUBPIX",
        choices=["NONE", "SUBPIX", "CONTOUR", "APRILTAG"],
        when="How marker corners are refined before chessboard corners are "
        "interpolated from them. This is the single most consequential detection "
        "setting.",
        per_choice={
            "NONE": "Integer-ish corners straight from the quad fit. Corner noise "
            "~0.3-0.5 px, which then dominates the entire uncertainty story. "
            "Only for a quick 'is anything detected at all' pass.",
            "SUBPIX": "Default and the right answer here. Iterative subpixel fit "
            "against the local image gradient.",
            "CONTOUR": "Fits lines to the marker contour. Comparable to SUBPIX, "
            "sometimes steadier on blurred or low-contrast images.",
            "APRILTAG": "AprilTag's refinement. Strongest on small, distant markers; "
            "noticeably slower.",
        },
    ),
    Option(
        name="corner_refinement_win_size",
        kind="int",
        default=5,
        minimum=1,
        maximum=50,
        when="Half-width in pixels of the SUBPIX search window. Raise it for large, "
        "close markers; lowering it helps when adjacent corners are so close that "
        "the windows overlap.",
    ),
    Option(
        name="corner_refinement_max_iterations",
        kind="int",
        default=30,
        minimum=1,
        maximum=1000,
        when="Iteration cap for the refinement. Rarely binding -- the accuracy "
        "criterion below usually stops it first.",
    ),
    Option(
        name="corner_refinement_min_accuracy",
        kind="float",
        default=0.1,
        minimum=1e-4,
        maximum=10.0,
        when="Stop once the corner moves less than this many pixels. Lower means "
        "more iterations for a slightly steadier corner.",
    ),
    Option(
        name="try_refine_markers",
        kind="bool",
        default=True,
        when="Re-search for markers the first pass missed, using the board geometry "
        "to predict where they must be. MEASURED on this project's 11-image "
        "two-board set: board_5x5 546 -> 901 corners (+65%), board_4x4 924 -> 968, "
        "at a cost of 0.209 -> 0.225 px calibration RMS. The recovered corners are "
        "the tilted, small and half-lit ones -- exactly the views that constrain "
        "distortion.",
    ),
    Option(
        name="check_markers",
        kind="bool",
        default=True,
        when="Discard markers that do not fit the board layout. Turning it off "
        "admits spurious detections; there is no good reason to do so.",
    ),
    Option(
        name="min_corners",
        kind="int",
        default=6,
        minimum=4,
        maximum=1000,
        when="Fewest chessboard corners for a detection to be kept at all. Below 4 "
        "the view cannot even yield a pose, and a 4-5 corner view is almost pure "
        "noise in the intrinsics fit.",
    ),
    Option(
        name="adaptive_thresh_win_size_min",
        kind="int",
        default=3,
        minimum=3,
        maximum=99,
        when="Smallest adaptive-threshold window, in pixels. The binarisation step "
        "sweeps min -> max in steps; small windows find small/distant markers.",
    ),
    Option(
        name="adaptive_thresh_win_size_max",
        kind="int",
        default=23,
        minimum=3,
        maximum=299,
        when="Largest adaptive-threshold window. Raise it for big markers filling "
        "much of the frame, or for strong illumination gradients.",
    ),
    Option(
        name="adaptive_thresh_win_size_step",
        kind="int",
        default=10,
        minimum=1,
        maximum=100,
        when="Step between window sizes. Smaller means more thresholding passes "
        "tried -- better recall, proportionally slower detection.",
    ),
    Option(
        name="adaptive_thresh_constant",
        kind="float",
        default=7.0,
        minimum=-50.0,
        maximum=50.0,
        when="Constant subtracted from the local mean when binarising. Raise it on "
        "washed-out or glaring images where black and white blur together.",
    ),
    Option(
        name="min_marker_perimeter_rate",
        kind="float",
        default=0.03,
        minimum=0.001,
        maximum=1.0,
        when="Smallest accepted marker perimeter, as a fraction of the larger image "
        "dimension. LOWER it to detect far-away boards; raising it rejects small "
        "noisy blobs.",
    ),
    Option(
        name="max_marker_perimeter_rate",
        kind="float",
        default=4.0,
        minimum=0.01,
        maximum=10.0,
        when="Largest accepted marker perimeter, same units. Only binding when a "
        "board fills the frame.",
    ),
    Option(
        name="polygonal_approx_accuracy_rate",
        kind="float",
        default=0.03,
        minimum=0.001,
        maximum=1.0,
        when="How far a contour may deviate from a perfect quad, relative to its "
        "perimeter. RAISE it for fisheye lenses, where a real square's edges bow "
        "and a strict quad test rejects markers near the image border.",
    ),
    Option(
        name="min_corner_distance_rate",
        kind="float",
        default=0.05,
        minimum=0.0,
        maximum=1.0,
        when="Minimum spacing between a candidate's own corners, relative to its "
        "perimeter. Rejects degenerate near-collapsed quads.",
    ),
    Option(
        name="min_distance_to_border",
        kind="int",
        default=3,
        minimum=0,
        maximum=200,
        when="Reject markers whose corners sit within this many pixels of the image "
        "edge, where refinement has no neighbourhood to work with. Note the "
        "tension: border corners are also the ones that constrain distortion most.",
    ),
    Option(
        name="error_correction_rate",
        kind="float",
        default=0.6,
        minimum=0.0,
        maximum=1.0,
        when="Fraction of the dictionary's error-correction capability to actually "
        "use when decoding a marker ID. Higher decodes more damaged markers and "
        "raises the chance of a WRONG id -- which with multiple boards means a "
        "corner attributed to the wrong board.",
    ),
]


@dataclass
class CheckerboardSettings:
    """
    Plain-chessboard detector knobs, applied to every checkerboard in a run.

    A checkerboard is all-or-nothing: `findChessboardCorners*` searches the
    whole image for a COMPLETE grid of the declared size, so unlike Charuco
    there is no `min_corners` -- a partial view simply is not a detection.

    Masking
        The masking knobs live here, not on the Charuco settings, because
        masking exists to serve this detector. A Charuco board IS a chessboard
        with markers inked into it, so with both in one image the chessboard
        search happily locks onto the Charuco board. The Charuco boards are
        detected first and painted out before this detector ever sees the frame.
    """

    # -- algorithm ---------------------------------------------------------
    algorithm: str = "SB"
    normalize_image: bool = True
    exhaustive: bool = False
    accuracy: bool = True
    marker: bool = False
    # -- legacy algorithm only --------------------------------------------
    adaptive_thresh: bool = True
    filter_quads: bool = True
    fast_check: bool = False
    refine_win_size: int = 5
    # -- masking other boards out -----------------------------------------
    mask_other_boards: bool = True
    mask_padding_squares: float = 0.6
    mask_fill_value: int = 128
    mask_min_coverage: float = 0.5

    def __post_init__(self) -> None:
        if self.algorithm not in ("SB", "LEGACY"):
            raise ValueError(f"unknown algorithm {self.algorithm!r}; known: ['LEGACY', 'SB']")
        if self.marker and self.algorithm != "SB":
            raise ValueError(
                "marker=True needs algorithm='SB': the origin marker is only read "
                "by findChessboardCornersSB, the legacy detector cannot see it"
            )
        if self.refine_win_size < 1:
            raise ValueError(f"refine_win_size must be >= 1, got {self.refine_win_size}")
        if self.mask_padding_squares < 0:
            raise ValueError(f"mask_padding_squares must be >= 0, got {self.mask_padding_squares}")
        if not 0 <= self.mask_fill_value <= 255:
            raise ValueError(f"mask_fill_value must be in [0,255], got {self.mask_fill_value}")
        if not 0.0 <= self.mask_min_coverage <= 1.0:
            raise ValueError(f"mask_min_coverage must be in [0,1], got {self.mask_min_coverage}")

    # -- OpenCV boundary ---------------------------------------------------
    def flags(self) -> int:
        """The flag word for whichever algorithm is selected."""
        if self.algorithm == "SB":
            f = 0
            if self.normalize_image:
                f |= cv2.CALIB_CB_NORMALIZE_IMAGE
            if self.exhaustive:
                f |= cv2.CALIB_CB_EXHAUSTIVE
            if self.accuracy:
                f |= cv2.CALIB_CB_ACCURACY
            if self.marker:
                f |= cv2.CALIB_CB_MARKER
            return f
        f = 0
        if self.adaptive_thresh:
            f |= cv2.CALIB_CB_ADAPTIVE_THRESH
        if self.normalize_image:
            f |= cv2.CALIB_CB_NORMALIZE_IMAGE
        if self.filter_quads:
            f |= cv2.CALIB_CB_FILTER_QUADS
        if self.fast_check:
            f |= cv2.CALIB_CB_FAST_CHECK
        return f

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict | None) -> CheckerboardSettings:
        """Tolerant of unknown keys, so an older config still loads."""
        if not data:
            return CheckerboardSettings()
        known = set(CheckerboardSettings().to_dict())
        return CheckerboardSettings(**{k: v for k, v in data.items() if k in known})


CHECKERBOARD_SUMMARY = (
    "Plain checkerboard detection. The most accurate corners of any target "
    "here and the least forgiving: the whole board must be visible, and its "
    "corners carry no identity. Other boards in the same image are masked out "
    "first -- see the masking knobs. Changing anything here invalidates "
    "existing detections."
)

CHECKERBOARD_CATALOG: list[Option] = [
    Option(
        name="algorithm",
        kind="choice",
        default="SB",
        choices=["SB", "LEGACY"],
        when="Which OpenCV chessboard finder to use. Both are present in this "
        "build; SB is the better default by a wide margin and LEGACY is kept "
        "only for reproducing an older result.",
        per_choice={
            "SB": "Sector-based detector. Survives partial blur, uneven lighting "
            "and perspective that defeats the legacy one, and returns subpixel "
            "corners directly with no separate refinement step.",
            "LEGACY": "The original quad-linking finder. Needs a clean white "
            "margin around the board and fails outright on strong tilt. Its "
            "corners are integer-ish until refine_win_size polishes them.",
        },
    ),
    Option(
        name="normalize_image",
        kind="bool",
        default=True,
        when="Equalise image brightness before searching. Costs a little time and "
        "rescues boards lit unevenly across the frame, which is most handheld "
        "captures. Read by both algorithms.",
    ),
    Option(
        name="exhaustive",
        kind="bool",
        default=False,
        when="SB only. Run the slow, thorough search rather than stopping at the "
        "first plausible grid. Turn it on when a board you can clearly see is "
        "not being found; expect detection to take several times longer.",
    ),
    Option(
        name="accuracy",
        kind="bool",
        default=True,
        when="SB only. Run the extra subpixel refinement pass. This is the "
        "setting that makes checkerboard corners more accurate than interpolated "
        "Charuco ones -- there is no good reason to turn it off.",
    ),
    Option(
        name="marker",
        kind="bool",
        default=False,
        when="SB only. Sets CALIB_CB_MARKER for boards printed with OpenCV's "
        "origin marker. MEASURED on OpenCV 5.0.0: it changes nothing -- same "
        "corners, same order, marked board or not. It does NOT fix the "
        "180-degree ambiguity and does not reject an unmarked board; that needs "
        "findChessboardCornersSBWithMeta, which this project does not call. "
        "Kept in case the flag starts acting.",
    ),
    Option(
        name="adaptive_thresh",
        kind="bool",
        default=True,
        when="LEGACY only. Threshold locally rather than with one global level, "
        "which is what lets the legacy finder cope with a shadow across the "
        "board. Ignored entirely when algorithm is SB.",
    ),
    Option(
        name="filter_quads",
        kind="bool",
        default=True,
        when="LEGACY only. Reject quads whose shape does not fit a chessboard "
        "square, which is the main defence against a background that happens to "
        "look gridded. Ignored when algorithm is SB.",
    ),
    Option(
        name="fast_check",
        kind="bool",
        default=False,
        when="LEGACY only. Cheap early test that bails out on images with no "
        "board at all. Speeds up a set full of empty frames, at the cost of "
        "occasionally discarding a real but faint board. Ignored when SB.",
    ),
    Option(
        name="refine_win_size",
        kind="int",
        default=5,
        minimum=1,
        maximum=50,
        when="LEGACY only. Half-width in pixels of the cornerSubPix window used "
        "after detection. Lower it when adjacent corners are close enough that "
        "the windows overlap. SB refines internally and ignores this entirely.",
    ),
    Option(
        name="mask_other_boards",
        kind="bool",
        default=True,
        when="Paint the Charuco boards out of the image before searching for a "
        "plain chessboard. A Charuco board is a chessboard with markers inked "
        "in, so leaving this off in a mixed set means the chessboard detector "
        "locks onto the wrong board and reports confident nonsense.",
    ),
    Option(
        name="mask_padding_squares",
        kind="float",
        default=0.6,
        minimum=0.0,
        maximum=5.0,
        when="How far past the Charuco board's printed edge the mask extends, in "
        "board squares. Padding matters: a fill edge sitting exactly on real "
        "board structure is itself a straight edge the detector can latch onto. "
        "Raise it if a masked board is still being found.",
    ),
    Option(
        name="mask_fill_value",
        kind="int",
        default=128,
        minimum=0,
        maximum=255,
        when="Grey level painted over a masked board. Mid-grey rather than black "
        "on purpose -- a black block beside the real board creates the highest "
        "contrast edge in the picture. 0 and 255 are there for debugging, where "
        "seeing the mask matters more than not disturbing the search.",
    ),
    Option(
        name="mask_min_coverage",
        kind="float",
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        when="How much of a Charuco board must be detected before its mask is "
        "built by projecting the full board outline. Below this the detected "
        "corners are too clustered to extrapolate from safely, and a padded hull "
        "of the corners actually seen is used instead.",
    ),
]


@dataclass
class CircleGridSettings:
    """
    Dot-grid detector knobs: blob filtering, then grid extraction.

    Detection is two stages and the knobs split the same way. `SimpleBlobDetector`
    finds candidate dots by thresholding at a sweep of levels and filtering the
    resulting blobs by area, roundness and solidity; `findCirclesGrid` then tries
    to arrange the survivors into the declared grid. Almost every "it finds
    nothing" case is stage one -- the grid stage is only as good as the blobs
    handed to it, and it reports all-or-nothing.

    Accuracy note that no knob here fixes
        A circle images as an ELLIPSE whose centroid is not the projection of the
        circle's centre. The bias grows with tilt and with dot size relative to
        distance, and it is systematic, so it does not average out over frames.
        It is the reason a dot grid can lose to a checkerboard despite dots being
        far more robust to defocus. Removing it needs a conic fit corrected
        against the estimated pose, which is an after-initialisation step rather
        than a detection setting -- so it is documented here rather than offered.
    """

    # -- grid extraction ---------------------------------------------------
    clustering: bool = False
    # -- blob filtering ----------------------------------------------------
    blob_color: str = "dark"
    min_area: float = 25.0
    max_area: float = 5000.0
    min_circularity: float = 0.8
    min_convexity: float = 0.95
    min_inertia_ratio: float = 0.1
    min_dist_between_blobs: float = 10.0
    min_threshold: float = 50.0
    max_threshold: float = 220.0
    threshold_step: float = 10.0
    # -- masking other boards out -----------------------------------------
    mask_other_boards: bool = True
    mask_padding_squares: float = 0.6
    mask_fill_value: int = 128
    mask_min_coverage: float = 0.5

    def __post_init__(self) -> None:
        if self.blob_color not in ("dark", "light"):
            raise ValueError(f"unknown blob_color {self.blob_color!r}; known: ['dark', 'light']")
        if self.min_area >= self.max_area:
            raise ValueError(
                f"min_area ({self.min_area}) must be below max_area ({self.max_area}); "
                f"no blob could match"
            )
        if self.min_threshold >= self.max_threshold:
            raise ValueError(
                f"min_threshold ({self.min_threshold}) must be below max_threshold "
                f"({self.max_threshold}); no threshold level would ever be tried"
            )
        if self.threshold_step <= 0:
            raise ValueError(f"threshold_step must be > 0, got {self.threshold_step}")
        if self.mask_padding_squares < 0:
            raise ValueError(f"mask_padding_squares must be >= 0, got {self.mask_padding_squares}")
        if not 0 <= self.mask_fill_value <= 255:
            raise ValueError(f"mask_fill_value must be in [0,255], got {self.mask_fill_value}")
        if not 0.0 <= self.mask_min_coverage <= 1.0:
            raise ValueError(f"mask_min_coverage must be in [0,1], got {self.mask_min_coverage}")

    # -- OpenCV boundary ---------------------------------------------------
    @property
    def invert_image(self) -> bool:
        """
        Whether the image must be inverted before blobs are looked for.

        MEASURED against the installed OpenCV 5.0.0, not assumed: on a rendered
        grid of white dots on black, `SimpleBlobDetector` finds ZERO blobs with
        blobColor=255, with blobColor=0, and with filterByColor switched off
        entirely -- while the same grid inverted yields all 44. The detector
        binarises and then contours, so it only ever finds DARK blobs, and
        `blobColor` does not work as the switch its name suggests. Inverting the
        image is therefore the implementation of `blob_color`, not a workaround.
        """
        return self.blob_color == "light"

    def blob_detector(self):
        """A configured `cv2.SimpleBlobDetector`, ready for findCirclesGrid."""
        p = cv2.SimpleBlobDetector_Params()
        # Always dark: light targets are handled by inverting the image, for the
        # reason measured in `invert_image`.
        p.filterByColor = True
        p.blobColor = 0
        p.filterByArea = True
        p.minArea = float(self.min_area)
        p.maxArea = float(self.max_area)
        p.filterByCircularity = True
        p.minCircularity = float(self.min_circularity)
        p.filterByConvexity = True
        p.minConvexity = float(self.min_convexity)
        p.filterByInertia = True
        p.minInertiaRatio = float(self.min_inertia_ratio)
        p.minDistBetweenBlobs = float(self.min_dist_between_blobs)
        p.minThreshold = float(self.min_threshold)
        p.maxThreshold = float(self.max_threshold)
        p.thresholdStep = float(self.threshold_step)
        return cv2.SimpleBlobDetector_create(p)

    def flags(self, grid_type: str) -> int:
        """Grid flag word. `grid_type` is board geometry, so it comes from the spec."""
        f = (
            cv2.CALIB_CB_ASYMMETRIC_GRID
            if grid_type == "asymmetric"
            else cv2.CALIB_CB_SYMMETRIC_GRID
        )
        if self.clustering:
            f |= cv2.CALIB_CB_CLUSTERING
        return f

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict | None) -> CircleGridSettings:
        """Tolerant of unknown keys, so an older config still loads."""
        if not data:
            return CircleGridSettings()
        known = set(CircleGridSettings().to_dict())
        return CircleGridSettings(**{k: v for k, v in data.items() if k in known})


CIRCLE_GRID_SUMMARY = (
    "Dot-grid detection: blobs first, then the grid arrangement. Like a "
    "checkerboard it is all-or-nothing and carries no per-dot identity, so "
    "coded boards in the same image are masked out before it searches. Dot "
    "centroids survive defocus far better than corners, at the cost of a "
    "perspective bias -- see the note on the detector. Changing anything here "
    "invalidates existing detections."
)

CIRCLE_GRID_CATALOG: list[Option] = [
    Option(
        name="clustering",
        kind="bool",
        default=False,
        when="Use OpenCV's cluster-based grid finder instead of the default. It "
        "is markedly more robust to strong perspective, and markedly more "
        "sensitive to background clutter -- turn it on for steeply tilted views "
        "shot against a clean background, and off again if it starts finding "
        "grids in the scene behind the board.",
    ),
    Option(
        name="blob_color",
        kind="choice",
        default="dark",
        choices=["dark", "light"],
        when="Whether the dots are darker or lighter than the sheet they sit on. "
        "Getting this backwards finds exactly zero blobs, which then reads as a "
        "focus or lighting problem rather than a one-word setting. Light targets "
        "work by inverting the image: OpenCV's blob detector only ever finds dark "
        "blobs, whatever its blobColor parameter claims.",
        per_choice={
            "dark": "Black dots on a white sheet -- the usual printed target.",
            "light": "White dots on a dark sheet, including back-lit and "
            "retro-reflective targets shot with a ring light. The image is "
            "inverted before detection; nothing else changes.",
        },
    ),
    Option(
        name="min_area",
        kind="float",
        default=25.0,
        minimum=1.0,
        maximum=100000.0,
        when="Smallest accepted blob, in square pixels. This is the first knob to "
        "lower when a board held far from the camera stops being detected: at 25 "
        "px^2 a dot must be about 6 px across before it counts.",
    ),
    Option(
        name="max_area",
        kind="float",
        default=5000.0,
        minimum=2.0,
        maximum=10000000.0,
        when="Largest accepted blob, in square pixels. RAISE it when the board "
        "fills the frame -- at 5000 px^2 a dot wider than about 80 px is thrown "
        "away, which is easy to hit with a close-up target on a high-res sensor.",
    ),
    Option(
        name="min_circularity",
        kind="float",
        default=0.8,
        minimum=0.0,
        maximum=1.0,
        when="How round a blob must be, as 4*pi*area/perimeter^2 (1.0 is a perfect "
        "circle). Lower it for steeply tilted views, where a real dot images as a "
        "distinctly elongated ellipse and the default starts rejecting the very "
        "views that constrain distortion best.",
    ),
    Option(
        name="min_convexity",
        kind="float",
        default=0.95,
        minimum=0.0,
        maximum=1.0,
        when="Blob area divided by the area of its convex hull. Rejects dots "
        "merged with a neighbour or nicked by a shadow, both of which would "
        "otherwise contribute a badly displaced centre.",
    ),
    Option(
        name="min_inertia_ratio",
        kind="float",
        default=0.1,
        minimum=0.0,
        maximum=1.0,
        when="Ratio of the blob's minor to major axis -- how far from a line it "
        "is. The default is permissive on purpose, because it is circularity "
        "that should be doing the shape filtering here.",
    ),
    Option(
        name="min_dist_between_blobs",
        kind="float",
        default=10.0,
        minimum=0.0,
        maximum=1000.0,
        when="Minimum pixel separation between two accepted blob centres. It "
        "merges near-duplicate detections of one dot found at several threshold "
        "levels; raise it above the smallest real dot pitch you expect in the "
        "image and the grid loses whole rows.",
    ),
    Option(
        name="min_threshold",
        kind="float",
        default=50.0,
        minimum=0.0,
        maximum=255.0,
        when="First grey level in the binarisation sweep. The detector thresholds "
        "at every level from min to max and keeps blobs that persist across "
        "several, which is what makes it tolerant of uneven lighting.",
    ),
    Option(
        name="max_threshold",
        kind="float",
        default=220.0,
        minimum=1.0,
        maximum=255.0,
        when="Last grey level in the sweep. Widen the min/max span for images "
        "with a strong brightness gradient across the board; a wider span costs "
        "proportionally more passes.",
    ),
    Option(
        name="threshold_step",
        kind="float",
        default=10.0,
        minimum=0.1,
        maximum=128.0,
        when="Grey levels between successive passes. Smaller finds faint dots at "
        "the cost of speed, and it is the cheapest thing to try before loosening "
        "the shape filters, because it costs recall and not selectivity.",
    ),
    Option(
        name="mask_other_boards",
        kind="bool",
        default=True,
        when="Paint the coded boards out of the image before searching for the "
        "dot grid. Less critical than for a checkerboard -- a Charuco board looks "
        "nothing like a dot grid -- but it also removes the printed markers as a "
        "source of stray blobs, and costs nothing when there is no coded board.",
    ),
    Option(
        name="mask_padding_squares",
        kind="float",
        default=0.6,
        minimum=0.0,
        maximum=5.0,
        when="How far past a masked board's printed edge the mask extends, in "
        "that board's squares. Padding matters: a fill edge sitting exactly on "
        "real board structure is itself a hard edge that can survive as a blob.",
    ),
    Option(
        name="mask_fill_value",
        kind="int",
        default=128,
        minimum=0,
        maximum=255,
        when="Grey level painted over a masked board. Mid-grey rather than black "
        "or white on purpose -- either extreme is a high-contrast region that the "
        "threshold sweep will happily turn into one enormous blob.",
    ),
    Option(
        name="mask_min_coverage",
        kind="float",
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        when="How much of a coded board must be detected before its mask is built "
        "by projecting the full board outline. Below this the detected corners "
        "are too clustered to extrapolate from safely, and a padded hull of the "
        "corners actually seen is used instead.",
    ),
]
