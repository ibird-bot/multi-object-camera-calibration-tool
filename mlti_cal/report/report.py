"""
The full calibration report -- one object that answers "can I trust this?".

Assembles covariance, projection uncertainty, residual structure, coverage,
outliers, optional cross-validation and (in synthetic mode) the ground-truth
coverage check, then renders warnings in plain language.

Everything is computed headlessly. The GUI renders this object; it does not
compute anything of its own, so a CLI run and a GUI run produce identical
numbers by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mlti_cal.problem.graph import Problem
from mlti_cal.problem.types import CalibrationSystem
from mlti_cal.report import coverage as coverage_mod
from mlti_cal.report import residuals as residual_mod
from mlti_cal.report import warnings as warn_mod
from mlti_cal.report.covariance import CovarianceResult, compute_covariance
from mlti_cal.report.uncertainty import (
    UncertaintyMap,
    parameter_error_vs_uncertainty,
    projection_uncertainty,
)
from mlti_cal.solvers.base import SolveResult


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


@dataclass
class CalibrationReport:
    system_summary: dict
    solve: dict | None
    covariance: CovarianceResult | None
    residuals: residual_mod.ResidualStats | None
    normality: dict = field(default_factory=dict)
    outliers: dict = field(default_factory=dict)
    error_vs_radius: dict = field(default_factory=dict)
    coverage: coverage_mod.CoverageReport | None = None
    uncertainty_maps: dict[str, UncertaintyMap] = field(default_factory=dict)
    crossval: dict | None = None
    gt_check: dict | None = None
    warnings: list = field(default_factory=list)
    camera_estimates: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "system": self.system_summary,
            "solve": self.solve,
            "cameras": self.camera_estimates,
            "covariance": self.covariance.to_dict() if self.covariance else None,
            "residuals": self.residuals.to_dict() if self.residuals else None,
            "normality": self.normality,
            "outliers": self.outliers,
            "error_vs_radius": self.error_vs_radius,
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "uncertainty": {k: v.to_dict() for k, v in self.uncertainty_maps.items()},
            "crossval": self.crossval,
            "ground_truth_check": self.gt_check,
            "warnings": warn_mod.summarise(self.warnings),
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, cls=NumpyEncoder), encoding="utf-8")
        return path

    def text_summary(self) -> str:
        lines: list[str] = []
        add = lines.append
        add("=" * 78)
        add("CALIBRATION REPORT")
        add("=" * 78)
        s = self.system_summary
        add(
            f"{s['cameras']} camera(s), {s['boards']} board(s), {s['frames']} frame(s), "
            f"{s['observed_points']} corners, reference = {s['reference_camera']}"
        )
        if self.solve:
            add(
                f"solver: {self.solve['backend']} | RMS "
                f"{self.solve['initial_rms_px']:.4f} -> {self.solve['final_rms_px']:.4f} px "
                f"in {self.solve['iterations']} it / {self.solve['time_seconds']:.2f}s"
            )

        add("")
        add("-- CAMERAS " + "-" * 66)
        for cid, info in self.camera_estimates.items():
            add(f"  {cid} [{info['model']}]")
            for name, val, sig in info["parameters"]:
                if sig is None or not np.isfinite(sig):
                    add(f"      {name:>4} = {val:14.6f}   (fixed)")
                else:
                    rel = abs(sig / val) * 100 if val else float("inf")
                    add(f"      {name:>4} = {val:14.6f}  +/- {sig:10.6f}  ({rel:6.2f}%)")

        if self.covariance:
            c = self.covariance
            add("")
            add("-- CONDITIONING " + "-" * 61)
            add(
                f"  free parameters : {c.num_free_params}   rank: {c.rank}"
                + ("   *** RANK DEFICIENT ***" if c.rank_deficient else "")
            )
            add(f"  condition number: {c.condition_number:.3e}")
            add(f"  dof             : {c.degrees_of_freedom}")
            add(
                f"  sigma (residual): {c.sigma2_residual**0.5:.4f} px   "
                f"sigma (assumed): {c.sigma2_assumed**0.5:.4f} px   "
                f"factor {c.noise_disagreement:.2f}"
                + ("   <-- MISMATCH" if c.noise_disagreement_flagged else "")
            )
            add("  weakest directions:")
            for wd in c.weak_directions[:4]:
                add(f"      {wd.describe()}")

        if self.residuals:
            add("")
            add("-- RESIDUALS " + "-" * 64)
            add(
                f"  overall RMS {self.residuals.overall_rms:.4f} px, "
                f"max {self.residuals.overall_max:.4f} px"
            )
            for cid, st in self.residuals.per_camera.items():
                add(f"      {cid}: RMS {st['rms_px']:.4f} px over {st['n']} corners")
            if self.outliers:
                add(
                    f"  outliers: {self.outliers['count']} "
                    f"({self.outliers['fraction'] * 100:.2f}%) beyond "
                    f"{self.outliers['threshold_px']:.2f} px"
                )

        if self.uncertainty_maps:
            add("")
            add("-- PROJECTION UNCERTAINTY " + "-" * 51)
            for cid, m in self.uncertainty_maps.items():
                add(
                    f"  {cid} @ {m.range_m:.1f} m: centre {m.at_centre():.4f} px, "
                    f"worst {m.worst():.4f} px"
                )

        if self.crossval:
            add("")
            add("-- GENERALISATION " + "-" * 59)
            add(
                f"  train RMS {self.crossval['mean_train_rms_px']:.4f} px  ->  "
                f"held-out RMS {self.crossval['mean_test_rms_px']:.4f} px "
                f"({self.crossval['optimism_ratio']:.2f}x)"
            )

        if self.gt_check:
            add("")
            add("-- GROUND-TRUTH CHECK (synthetic only) " + "-" * 38)
            g = self.gt_check
            add(f"  parameters within 1 sigma: {g['within_1_sigma'] * 100:.1f}%  (expect ~68%)")
            add(f"  parameters within 2 sigma: {g['within_2_sigma'] * 100:.1f}%  (expect ~95%)")
            add(f"  worst: {g['worst']['parameter']} at {g['worst']['n_sigma']:.2f} sigma")

        add("")
        add("-- WARNINGS " + "-" * 65)
        for item in self.warnings:
            add(f"  {item}")
        add("=" * 78)
        return "\n".join(lines)


def build_report(
    problem: Problem,
    system: CalibrationSystem,
    solve_result: SolveResult | None = None,
    pixel_noise_std: float = 0.3,
    uncertainty_ranges: dict[str, float] | None = None,
    default_range_m: float = 1.5,
    do_crossval: bool = False,
    crossval_folds: int = 4,
    truth_values: dict | None = None,
    grid_step: int = 32,
) -> CalibrationReport:
    """
    Assemble the full report.

    Args:
        uncertainty_ranges: camera id -> range in metres for its uncertainty
            map. Defaults to `default_range_m` for every camera.
        truth_values: block key -> true value. Enables the ground-truth
            coverage check; synthetic data only.
    """
    cov = None
    try:
        cov = compute_covariance(problem, pixel_noise_std=pixel_noise_std)
    except ValueError:
        pass  # too few degrees of freedom; reported via warnings below

    stats = residual_mod.compute_residual_stats(problem)
    norm = residual_mod.normality(stats)
    outl = residual_mod.find_outliers(stats)
    cov_report = coverage_mod.compute_coverage_report(system)

    ref_cam = next(iter(system.cameras.values()))
    evr = residual_mod.error_vs_radius(stats, (ref_cam.params[2], ref_cam.params[3]))

    maps: dict[str, UncertaintyMap] = {}
    if cov is not None:
        ranges = uncertainty_ranges or {}
        for cid in system.cameras:
            try:
                maps[cid] = projection_uncertainty(
                    problem,
                    system,
                    cov,
                    cid,
                    range_m=ranges.get(cid, default_range_m),
                    grid_step=grid_step,
                )
            except (ValueError, KeyError):
                continue

    cv_dict = None
    cv_obj = None
    if do_crossval:
        from mlti_cal.report.crossval import cross_validate

        try:
            cv_obj = cross_validate(system, k=crossval_folds)
            cv_dict = cv_obj.to_dict()
        except ValueError:
            cv_dict = None

    gt = None
    if truth_values and cov is not None:
        gt = parameter_error_vs_uncertainty(problem, cov, truth_values)

    # Per-camera parameter table with std errors attached by label.
    cameras: dict = {}
    for cid, cam in system.cameras.items():
        rows = []
        for i, name in enumerate(cam.model.param_names):
            label = f"intr:{cid}.{i}"
            sig = None
            if cov is not None and label in cov.labels:
                sig = float(cov.std_errors[cov.labels.index(label)])
            rows.append((name, float(cam.params[i]), sig))
        cameras[cid] = {
            "model": cam.model_name,
            "image_size": list(cam.image_size),
            "is_reference": cam.is_reference,
            "parameters": rows,
            "extrinsic": cam.extrinsic.tolist(),
        }

    warns = warn_mod.collect_warnings(
        covariance=cov,
        residual_stats=stats,
        normality_stats=norm,
        outliers=outl,
        coverage=cov_report,
        crossval=cv_obj,
        solve_result=solve_result,
        num_frames=len(system.frames),
    )

    return CalibrationReport(
        system_summary=system.summary(),
        solve=solve_result.to_dict() if solve_result else None,
        covariance=cov,
        residuals=stats,
        normality=norm,
        outliers=outl,
        error_vs_radius=evr,
        coverage=cov_report,
        uncertainty_maps=maps,
        crossval=cv_dict,
        gt_check=gt,
        warnings=warns,
        camera_estimates=cameras,
    )
