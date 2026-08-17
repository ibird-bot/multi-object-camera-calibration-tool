"""
Report thresholds: the numbers that decide what counts as a problem.

Everything here changes what the report SAYS, never what was solved. That is
the whole reason they belong in front of the user rather than inside the
functions: a warning that fires at a threshold nobody can see is an opinion
presented as a measurement.

Defaults reproduce the previously hardcoded values exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from mlti_cal.options import Option


@dataclass
class ReportSettings:
    """Thresholds for outliers, warnings and the uncertainty maps."""

    # -- residuals ---------------------------------------------------------
    outlier_k: float = 4.0
    # -- conditioning ------------------------------------------------------
    correlation_threshold: float = 0.98
    correlation_top_k: int = 3
    condition_number_warn: float = 1e8
    # -- data quantity -----------------------------------------------------
    min_frames_warn: int = 8
    min_occupied_fraction: float = 0.6
    max_low_tilt_fraction: float = 0.7
    max_outlier_fraction: float = 0.02
    # -- uncertainty maps --------------------------------------------------
    default_range_m: float = 1.5
    grid_step: int = 32
    # -- cross-validation --------------------------------------------------
    crossval_folds: int = 4

    def __post_init__(self) -> None:
        if self.crossval_folds < 2:
            raise ValueError("crossval_folds must be at least 2 to have a held-out set")
        if not 0.0 < self.correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be in (0, 1]")
        if self.outlier_k <= 0:
            raise ValueError("outlier_k must be > 0")
        if self.grid_step < 1:
            raise ValueError("grid_step must be at least 1 pixel")

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict | None) -> ReportSettings:
        if not data:
            return ReportSettings()
        known = {f for f in ReportSettings().to_dict()}
        return ReportSettings(**{k: v for k, v in data.items() if k in known})


REPORT_SUMMARY = (
    "Report thresholds. These change what the report calls a problem, not the "
    "calibration itself -- so loosening one makes warnings disappear without "
    "making anything better."
)

REPORT_CATALOG: list[Option] = [
    Option(
        name="outlier_k",
        kind="float",
        default=4.0,
        minimum=1.0,
        maximum=20.0,
        when="A corner is an outlier when its residual exceeds k robust standard "
        "deviations, where the robust sigma is 1.4826 x MAD -- so the threshold is "
        "not thrown off by the outliers it is trying to find. 4.0 is deliberately "
        "conservative; 3.0 flags roughly 0.3% of clean Gaussian data by chance.",
    ),
    Option(
        name="correlation_threshold",
        kind="float",
        default=0.98,
        minimum=0.5,
        maximum=1.0,
        when="Correlation above which two parameters are called inseparable. At 0.98 "
        "they trade off almost exactly and no amount of solving separates them; the "
        "classic pair is focal length against board distance.",
    ),
    Option(
        name="correlation_top_k",
        kind="int",
        default=3,
        minimum=1,
        maximum=50,
        when="How many of the worst correlated pairs to list in the warning.",
    ),
    Option(
        name="condition_number_warn",
        kind="float",
        default=1e8,
        minimum=1e2,
        maximum=1e16,
        when="Condition number of J^T J above which the problem is called "
        "ill-conditioned. 1e8 is roughly where double-precision starts losing half "
        "its significant digits.",
    ),
    Option(
        name="min_frames_warn",
        kind="int",
        default=8,
        minimum=1,
        maximum=1000,
        when="Warn below this many frames. Distortion coefficients need variety of "
        "pose to be identifiable, and few frames give them none.",
    ),
    Option(
        name="min_occupied_fraction",
        kind="float",
        default=0.6,
        minimum=0.0,
        maximum=1.0,
        when="Warn when less than this fraction of the image area ever saw a corner. "
        "Distortion is extrapolated wherever the board never went, and that "
        "extrapolation is not measured by the RMS.",
    ),
    Option(
        name="max_low_tilt_fraction",
        kind="float",
        default=0.7,
        minimum=0.0,
        maximum=1.0,
        when="Warn when more than this fraction of board views are tilted under 10 "
        "degrees. Fronto-parallel views barely constrain focal length -- this is "
        "the most common way a calibration looks good and is not.",
    ),
    Option(
        name="max_outlier_fraction",
        kind="float",
        default=0.02,
        minimum=0.0,
        maximum=1.0,
        when="Warn when more than this fraction of corners are outliers by the rule above.",
    ),
    Option(
        name="default_range_m",
        kind="float",
        default=1.5,
        minimum=0.01,
        maximum=1000.0,
        when="Depth in metres at which projection uncertainty is evaluated for the "
        "uncertainty maps. Set it to your actual working distance -- the numbers "
        "are meaningless at a range you never operate at.",
    ),
    Option(
        name="grid_step",
        kind="int",
        default=32,
        minimum=1,
        maximum=512,
        when="Pixel spacing of the uncertainty map grid. Smaller is a finer map and "
        "quadratically more covariance evaluations.",
    ),
    Option(
        name="crossval_folds",
        kind="int",
        default=4,
        minimum=2,
        maximum=50,
        when="Folds for held-out cross-validation. Frames are split k ways, the "
        "calibration is re-solved on each training split, and error is reported on "
        "the unseen frames -- the only number here that is not fitted on the data "
        "it judges. More folds means more re-solves.",
    ),
]
