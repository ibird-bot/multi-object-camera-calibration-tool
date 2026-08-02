"""
Plain-language warnings.

Every diagnostic elsewhere in `report/` produces numbers. This module turns
those numbers into sentences a user can act on, because "condition number
4.7e+9" tells almost nobody what to do next, whereas "your board is nearly
always fronto-parallel; tilt it 30-45 degrees in some shots" does.

Each warning states what was measured, why it matters, and what to change.
Severity is: info < warning < critical.
"""

from __future__ import annotations

from dataclasses import dataclass

INFO, WARNING, CRITICAL = "info", "warning", "critical"


@dataclass
class Warning_:
    severity: str
    code: str
    message: str
    action: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "action": self.action,
        }

    def __str__(self) -> str:
        s = f"[{self.severity.upper()}] {self.message}"
        return f"{s}\n    -> {self.action}" if self.action else s


def collect_warnings(
    covariance=None,
    residual_stats=None,
    normality_stats=None,
    outliers=None,
    coverage=None,
    crossval=None,
    solve_result=None,
    num_frames: int = 0,
) -> list[Warning_]:
    w: list[Warning_] = []

    # -- solver --------------------------------------------------------
    if solve_result is not None:
        if not solve_result.success:
            w.append(
                Warning_(
                    CRITICAL,
                    "solver_not_converged",
                    f"The solver stopped without converging: {solve_result.message}",
                    "Increase the iteration limit, or try a different backend. "
                    "Every number below is computed at a point that is not a minimum.",
                )
            )
        if solve_result.behind_camera_points:
            w.append(
                Warning_(
                    WARNING,
                    "points_behind_camera",
                    f"{solve_result.behind_camera_points} corner(s) projected from "
                    f"behind the camera during the solve.",
                    "Usually a bad initial pose. Check detection on the worst images.",
                )
            )

    # -- conditioning ---------------------------------------------------
    if covariance is not None:
        if covariance.rank_deficient:
            names = ", ".join(
                n
                for wd in covariance.weak_directions
                if wd.discarded
                for n, _ in wd.top_parameters[:2]
            )
            w.append(
                Warning_(
                    CRITICAL,
                    "rank_deficient",
                    f"The problem is rank deficient: {covariance.num_free_params} free "
                    f"parameters but only rank {covariance.rank}. Some combination of "
                    f"parameters is not observable from this data ({names}).",
                    "These parameters cannot be determined at all -- fix them, or "
                    "capture data that constrains them. Their reported uncertainty "
                    "is a lower bound only.",
                )
            )
        elif covariance.condition_number > 1e8:
            w.append(
                Warning_(
                    WARNING,
                    "ill_conditioned",
                    f"Condition number is {covariance.condition_number:.1e}: the "
                    f"weakest parameter direction is barely constrained.",
                    "Add pose variety -- more tilt, more depth range, more coverage "
                    "of the image corners.",
                )
            )
        if covariance.noise_disagreement_flagged:
            bigger = (
                "much larger than"
                if covariance.sigma2_residual > covariance.sigma2_assumed
                else "much smaller than"
            )
            w.append(
                Warning_(
                    WARNING,
                    "noise_model_mismatch",
                    f"Residual-derived noise is {bigger} the assumed pixel noise "
                    f"(sigma {covariance.sigma2_residual**0.5:.3f} px vs "
                    f"{covariance.sigma2_assumed**0.5:.3f} px, factor "
                    f"{covariance.noise_disagreement:.1f}).",
                    "Larger means the model cannot represent the data -- systematic "
                    "error is being absorbed as noise. Smaller means the model is "
                    "over-parameterised and is fitting the noise itself.",
                )
            )
        if num_frames and num_frames < 20 and covariance.condition_number > 1e5:
            w.append(
                Warning_(
                    WARNING,
                    "uncertainty_model_approximate",
                    f"With {num_frames} frames and condition number "
                    f"{covariance.condition_number:.1e}, the reported uncertainty is "
                    f"itself only approximate.",
                    "The covariance is a linearisation, sigma^2 (J^T J)^+, valid while "
                    "parameter errors stay small enough for the residual to be locally "
                    "linear. In this regime board-orientation errors get large enough "
                    "that the linearisation degrades -- measured on synthetic data, "
                    "reduced chi-squared scatters up to ~6x at 12 frames but settles "
                    "near 1 by 25. Treat these sigmas as order-of-magnitude, and add "
                    "frames if you need them to be tight.",
                )
            )

        strong = covariance.top_correlations(k=3, threshold=0.98)
        if strong:
            pairs = "; ".join(f"{a} vs {b} ({c:+.3f})" for a, b, c in strong)
            w.append(
                Warning_(
                    WARNING,
                    "extreme_correlation",
                    f"Parameters are almost perfectly correlated: {pairs}.",
                    "They trade off against each other and cannot be separated by "
                    "this data. Consider fixing one, or vary the capture geometry.",
                )
            )

    # -- data quantity / coverage ---------------------------------------
    if num_frames and num_frames < 8:
        w.append(
            Warning_(
                WARNING,
                "too_few_frames",
                f"Only {num_frames} frames. Distortion coefficients need variety to "
                f"be identifiable.",
                "Capture at least 15-20 views at varied tilt, distance and position.",
            )
        )
    if coverage is not None:
        for cid, cs in coverage.per_camera.items():
            if cs.occupied_fraction < 0.6:
                w.append(
                    Warning_(
                        WARNING,
                        "poor_coverage",
                        f"Camera {cid}: only {cs.occupied_fraction * 100:.0f}% of the "
                        f"image area contains any observation.",
                        "Move the board into the empty regions, especially the corners "
                        "where distortion is largest.",
                    )
                )
            empty_corners = [k for k, v in cs.corner_occupancy.items() if v == 0]
            if empty_corners:
                w.append(
                    Warning_(
                        WARNING,
                        "empty_image_corners",
                        f"Camera {cid}: no observations in {', '.join(empty_corners)}.",
                        "Distortion is extrapolated there. Projections near those "
                        "corners are not supported by data.",
                    )
                )
        if coverage.tilt and coverage.tilt.fraction_below_10deg > 0.7:
            w.append(
                Warning_(
                    WARNING,
                    "insufficient_tilt",
                    f"{coverage.tilt.fraction_below_10deg * 100:.0f}% of board views are "
                    f"within 10 degrees of fronto-parallel.",
                    "Tilt the board 30-45 degrees in some views. Without tilt, focal "
                    "length and board distance are nearly indistinguishable.",
                )
            )

    # -- residual structure ----------------------------------------------
    if normality_stats and normality_stats.get("n", 0) > 8:
        ek = normality_stats.get("excess_kurtosis", 0.0)
        tail = normality_stats.get("fraction_beyond_3_sigma", 0.0)
        if tail > 0.01:
            w.append(
                Warning_(
                    WARNING,
                    "heavy_tails",
                    f"{tail * 100:.1f}% of residual components exceed 3 sigma "
                    f"(expected 0.27%); excess kurtosis {ek:.1f}.",
                    "Either real outliers (bad detections) or unmodelled systematic "
                    "error. Check the residual quiver plot for structure.",
                )
            )
    if outliers and outliers.get("fraction", 0.0) > 0.02:
        w.append(
            Warning_(
                WARNING,
                "many_outliers",
                f"{outliers['count']} corners ({outliers['fraction'] * 100:.1f}%) exceed "
                f"the robust outlier threshold of {outliers['threshold_px']:.2f} px.",
                "Inspect those detections. Re-solve with a robust loss, or remove them.",
            )
        )

    # -- generalisation ---------------------------------------------------
    if crossval is not None and crossval.mean_train_rms > 0:
        ratio = crossval.mean_test_rms / crossval.mean_train_rms
        if ratio > 1.5:
            w.append(
                Warning_(
                    CRITICAL,
                    "overfitting",
                    f"Held-out error is {ratio:.2f}x the training error "
                    f"({crossval.mean_test_rms:.3f} vs {crossval.mean_train_rms:.3f} px).",
                    "The reported RMS is optimistic. Use the held-out number as the "
                    "honest accuracy, and consider a simpler distortion model.",
                )
            )
        elif ratio > 1.15:
            w.append(
                Warning_(
                    INFO,
                    "mild_optimism",
                    f"Held-out error is {ratio:.2f}x the training error. Some optimism "
                    f"is normal; quote {crossval.mean_test_rms:.3f} px, not "
                    f"{crossval.mean_train_rms:.3f} px.",
                    "",
                )
            )

    # -- absolute accuracy -------------------------------------------------
    if residual_stats is not None and residual_stats.per_camera:
        worst = max(residual_stats.per_camera.items(), key=lambda kv: kv[1]["rms_px"])
        best = min(residual_stats.per_camera.items(), key=lambda kv: kv[1]["rms_px"])
        if best[1]["rms_px"] > 0 and worst[1]["rms_px"] / best[1]["rms_px"] > 2.0:
            w.append(
                Warning_(
                    WARNING,
                    "uneven_cameras",
                    f"Camera {worst[0]} fits {worst[1]['rms_px'] / best[1]['rms_px']:.1f}x "
                    f"worse than {best[0]} ({worst[1]['rms_px']:.3f} vs "
                    f"{best[1]['rms_px']:.3f} px).",
                    "A global RMS averages this away. Check that camera's detections, "
                    "focus and model choice separately.",
                )
            )

    if not w:
        w.append(
            Warning_(
                INFO,
                "no_issues",
                "No conditioning, coverage or residual-structure problems detected.",
                "",
            )
        )
    return w


def summarise(warnings: list[Warning_]) -> dict:
    counts = {INFO: 0, WARNING: 0, CRITICAL: 0}
    for item in warnings:
        counts[item.severity] = counts.get(item.severity, 0) + 1
    return {
        "counts": counts,
        "worst_severity": (
            CRITICAL if counts[CRITICAL] else (WARNING if counts[WARNING] else INFO)
        ),
        "warnings": [item.to_dict() for item in warnings],
    }
