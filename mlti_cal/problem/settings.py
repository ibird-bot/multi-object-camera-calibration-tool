"""
Every bootstrap choice, in one place the user can see and change.

The starting point decides which minimum the solver falls into, so these were
the second-least defensible set of hidden constants after the detector's. All
defaults reproduce exactly the behaviour that was hardcoded before.

Scope: these expose the VALUES the existing method uses. They do not add
alternative initialisation algorithms -- Bouguet vanishing-point init and
plumb-line distortion pre-estimation are separate pieces of work, not knobs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2

from mlti_cal.options import Option

#: name -> cv2.SOLVEPNP_* value. P3P/AP3P/IPPE_SQUARE are excluded on purpose:
#: they demand exactly 4 points (or a specific point ORDER), which a charuco
#: detection does not provide.
PNP_METHODS = {
    "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
    "EPNP": cv2.SOLVEPNP_EPNP,
    "SQPNP": cv2.SOLVEPNP_SQPNP,
    "IPPE": cv2.SOLVEPNP_IPPE,
}


@dataclass
class InitSettings:
    """Bootstrap knobs. Defaults are the previously hardcoded behaviour."""

    # -- PnP ---------------------------------------------------------------
    pnp_method: str = "ITERATIVE"
    pnp_ransac: bool = False
    ransac_reproj_threshold_px: float = 3.0
    ransac_confidence: float = 0.99
    ransac_max_iterations: int = 100
    # -- how much data a fit is allowed to run on -------------------------
    min_views_per_camera: int = 4
    min_points_for_calibration: int = 6
    min_points_for_pnp: int = 4
    # -- fisheye branch of the intrinsics fit ------------------------------
    fisheye_max_iterations: int = 60
    fisheye_epsilon: float = 1e-6
    fisheye_fix_skew: bool = True
    fisheye_recompute_extrinsic: bool = True
    # -- combining many estimates into one ---------------------------------
    translation_average: str = "median"
    board_pose_source: str = "max_corners"

    def __post_init__(self) -> None:
        if self.pnp_method not in PNP_METHODS:
            raise ValueError(
                f"unknown pnp_method {self.pnp_method!r}; known: {sorted(PNP_METHODS)}"
            )
        if self.translation_average not in ("median", "mean"):
            raise ValueError(
                f"translation_average must be 'median' or 'mean', got {self.translation_average!r}"
            )
        if self.board_pose_source not in ("max_corners", "average"):
            raise ValueError(
                f"board_pose_source must be 'max_corners' or 'average', "
                f"got {self.board_pose_source!r}"
            )
        # Hard geometric floors, not preferences. Below these OpenCV either
        # asserts or returns a pose fitted to fewer constraints than unknowns.
        if self.min_points_for_pnp < 4:
            raise ValueError("min_points_for_pnp cannot go below 4: a pose needs 4 points")
        if self.min_points_for_calibration < 4:
            raise ValueError(
                "min_points_for_calibration cannot go below 4: a view's homography needs 4 points"
            )
        if self.min_views_per_camera < 3:
            raise ValueError(
                "min_views_per_camera cannot go below 3: the linear solve for the "
                "image of the absolute conic needs 3 views"
            )

    @property
    def pnp_flag(self) -> int:
        return PNP_METHODS[self.pnp_method]

    @property
    def fisheye_criteria(self) -> tuple[int, int, float]:
        return (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            int(self.fisheye_max_iterations),
            float(self.fisheye_epsilon),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict | None) -> InitSettings:
        if not data:
            return InitSettings()
        known = {f for f in InitSettings().to_dict()}
        return InitSettings(**{k: v for k, v in data.items() if k in known})


INIT_SUMMARY = (
    "Bootstrap: OpenCV intrinsics per camera, then PnP per view, then camera "
    "extrinsics from co-observed boards, then board poses in the rig frame. "
    "These settings change the starting point, not the objective -- but a bad "
    "starting point is how a bundle adjustment converges to the wrong answer "
    "while reporting success."
)

INIT_CATALOG: list[Option] = [
    Option(
        name="pnp_method",
        kind="choice",
        default="ITERATIVE",
        choices=["ITERATIVE", "EPNP", "SQPNP", "IPPE"],
        when="How each board's pose is recovered from one view, once intrinsics are known.",
        per_choice={
            "ITERATIVE": "Default. DLT-style initial guess refined by "
            "Levenberg-Marquardt on reprojection error. Most accurate here, and "
            "charuco views give it plenty of points.",
            "EPNP": "Non-iterative O(n). Much faster on huge point counts, slightly "
            "less accurate. Useful if the bootstrap is the bottleneck.",
            "SQPNP": "Non-iterative and globally optimal for the algebraic cost. A "
            "good choice when ITERATIVE occasionally lands on a flipped pose.",
            "IPPE": "Planar-specific and fast. Valid here because a charuco board IS "
            "planar, but it is the most sensitive to a near-fronto-parallel view, "
            "where the two-fold planar pose ambiguity is genuinely hard.",
        },
    ),
    Option(
        name="pnp_ransac",
        kind="bool",
        default=False,
        when="Reject outlier corners inside PnP instead of trusting every detection. "
        "Off by default because charuco corners are interpolated from decoded "
        "markers and gross outliers are rare. Turn it ON if the report shows "
        "isolated huge residuals, or if you lowered min_markers or raised "
        "error_correction_rate -- both of which admit mis-attributed corners. "
        "Note it costs the exactness of a full fit on clean data.",
    ),
    Option(
        name="ransac_reproj_threshold_px",
        kind="float",
        default=3.0,
        minimum=0.1,
        maximum=100.0,
        when="Pixels of reprojection error above which a corner is called an outlier. "
        "Only used when pnp_ransac is on. Set it well above your expected corner "
        "noise (~0.2-0.5 px) but below the error you consider a real mistake; too "
        "tight and RANSAC discards good wide-angle corners.",
    ),
    Option(
        name="ransac_confidence",
        kind="float",
        default=0.99,
        minimum=0.5,
        maximum=0.9999,
        when="Probability that RANSAC finds an outlier-free sample. Higher means more iterations.",
    ),
    Option(
        name="ransac_max_iterations",
        kind="int",
        default=100,
        minimum=1,
        maximum=100000,
        when="Iteration cap for RANSAC. The default is OpenCV's and is ample for the "
        "low outlier rates seen with charuco.",
    ),
    Option(
        name="min_views_per_camera",
        kind="int",
        default=4,
        minimum=3,
        maximum=1000,
        when="Fewest views before a camera's intrinsics are fitted at all. Below this "
        "the camera keeps its default parameters and is reported with RMS = nan "
        "rather than looking calibrated. 3 is a hard floor -- the linear solve for "
        "the absolute conic needs three views -- and 3 is still badly conditioned.",
    ),
    Option(
        name="min_points_for_calibration",
        kind="int",
        default=6,
        minimum=4,
        maximum=1000,
        when="Fewest corners for a view to enter the intrinsics fit. 4 is the hard "
        "floor (a homography), 6 gives it some redundancy. Raising it drops your "
        "most oblique views, which are the ones that constrain distortion.",
    ),
    Option(
        name="min_points_for_pnp",
        kind="int",
        default=4,
        minimum=4,
        maximum=1000,
        when="Fewest corners for a view to get a pose. 4 is a hard floor for every "
        "PnP method here. Raise it to 8-10 if weak views are producing wild poses "
        "that pollute the extrinsic average.",
    ),
    Option(
        name="fisheye_max_iterations",
        kind="int",
        default=60,
        minimum=1,
        maximum=10000,
        when="Iteration cap for cv2.fisheye.calibrate. Fisheye fits are far more "
        "fragile than pinhole ones; raise this if the fit stops short.",
    ),
    Option(
        name="fisheye_epsilon",
        kind="float",
        default=1e-6,
        minimum=1e-12,
        maximum=1.0,
        when="Convergence epsilon for cv2.fisheye.calibrate.",
    ),
    Option(
        name="fisheye_fix_skew",
        kind="bool",
        default=True,
        when="Hold the skew term at zero. Keep it on: no modern sensor has meaningful "
        "skew, and freeing it lets the fit absorb real distortion error into a "
        "physically meaningless parameter.",
    ),
    Option(
        name="fisheye_recompute_extrinsic",
        kind="bool",
        default=True,
        when="Re-solve each view's pose at every iteration of the fisheye fit. "
        "Markedly more stable; the cost is speed.",
    ),
    Option(
        name="translation_average",
        kind="choice",
        default="median",
        choices=["median", "mean"],
        when="How the many per-frame estimates of one camera's translation are "
        "combined. Rotations always use the quaternion eigenvector mean -- there "
        "is no sensible componentwise median for a rotation.",
        per_choice={
            "median": "Default. One badly-conditioned PnP (board nearly edge-on, few "
            "corners) produces a wild outlier that a mean would absorb.",
            "mean": "Lower variance when every estimate is sound. Use it only if you "
            "have already established there are no outliers.",
        },
    ),
    Option(
        name="board_pose_source",
        kind="choice",
        default="max_corners",
        choices=["max_corners", "average"],
        when="Which camera's PnP defines a board's pose in the rig frame, for frames "
        "where several cameras saw it.",
        per_choice={
            "max_corners": "Default. Take the camera that detected the most corners -- "
            "its PnP is the best conditioned. No averaging, so one bad PnP is not "
            "diluted.",
            "average": "Transform every camera's estimate into the rig frame and "
            "average them (median translation, quaternion mean). Steadier, but it "
            "folds extrinsic error into the board pose, so a wrong extrinsic "
            "becomes harder to spot in the residuals.",
        },
    ),
]
