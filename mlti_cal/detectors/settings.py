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
