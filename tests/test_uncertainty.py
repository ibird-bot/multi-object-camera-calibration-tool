"""
The honesty tests (M4).

These are the ones that actually justify the product claim. It is easy to
report a covariance; it is the ground-truth coverage check that decides whether
that covariance means anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlti_cal.io.synthetic import generate_dataset
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import (
    build_problem,
    extr_key,
    intr_key,
    pose_key,
    write_back,
)
from mlti_cal.report.covariance import compute_covariance
from mlti_cal.report.report import build_report
from mlti_cal.report.uncertainty import parameter_error_vs_uncertainty, projection_uncertainty
from mlti_cal.solvers import SolveOptions, get_backend


def truth_blocks(gt) -> dict:
    """Ground truth keyed the way the Problem keys its blocks."""
    out = {}
    for cid, p in gt.camera_params.items():
        out[intr_key(cid)] = p
    for cid, e in gt.camera_extrinsics.items():
        out[extr_key(cid)] = e
    for (frame, board), pose in gt.board_poses.items():
        out[pose_key(frame, board)] = pose
    return out


def solved(num_cameras=2, num_frames=14, noise=0.3, seed=31, **kw):
    system, gt = generate_dataset(
        num_cameras=num_cameras,
        num_frames=num_frames,
        pixel_noise_std=noise,
        seed=seed,
        **kw,
    )
    initialize_system(system)
    problem = build_problem(system)
    result = get_backend("scipy").solve(problem, SolveOptions(max_iterations=400))
    write_back(system, problem)
    return system, gt, problem, result


# ---------------------------------------------------------------------------
# Covariance mechanics
# ---------------------------------------------------------------------------


def test_covariance_shape_and_labels():
    _, _, problem, _ = solved()
    cov = compute_covariance(problem, pixel_noise_std=0.3)
    n = problem.num_free_params
    assert cov.covariance.shape == (n, n)
    assert len(cov.labels) == n
    assert np.all(cov.std_errors >= 0)
    assert cov.degrees_of_freedom == problem.num_residuals - n


def test_covariance_is_symmetric_positive_semidefinite():
    _, _, problem, _ = solved()
    cov = compute_covariance(problem)
    C = cov.covariance
    assert np.allclose(C, C.T, atol=1e-12)
    assert np.linalg.eigvalsh(C).min() > -1e-10


def test_residual_sigma_matches_the_injected_noise():
    """The residual-derived noise estimate must recover the true noise level."""
    noise = 0.4
    _, _, problem, _ = solved(noise=noise, num_frames=18)
    cov = compute_covariance(problem, pixel_noise_std=noise)
    assert cov.sigma2_residual**0.5 == pytest.approx(noise, rel=0.15)
    assert not cov.noise_disagreement_flagged


def test_noise_mismatch_is_flagged():
    """Claiming an implausible noise floor must trip the flag, not pass quietly."""
    _, _, problem, _ = solved(noise=0.4)
    cov = compute_covariance(problem, pixel_noise_std=0.01)
    assert cov.noise_disagreement_flagged
    assert cov.noise_disagreement > 2.0


def test_rank_deficiency_is_detected_not_absorbed():
    """
    Freeing the reference extrinsic removes the gauge fix, making J^T J exactly
    6-deficient. A default pinv would return a confident-looking matrix; this
    must instead report the deficiency and name the directions.
    """
    system, _, _, _ = solved(num_cameras=2, num_frames=8)
    problem = build_problem(system)
    ref = system.reference_camera
    problem.blocks[extr_key(ref)].constant = False  # break the gauge on purpose

    cov = compute_covariance(problem, rank_threshold=1e-8)
    assert cov.rank_deficient, "a gauge-free rig must be reported rank deficient"
    assert cov.num_free_params - cov.rank >= 5
    discarded = [w for w in cov.weak_directions if w.discarded]
    assert discarded, "discarded directions must be reported, not silently dropped"
    assert all(w.top_parameters for w in discarded)


def test_gauge_fixed_problem_is_full_rank():
    _, _, problem, _ = solved(num_cameras=2, num_frames=10)
    cov = compute_covariance(problem)
    assert not cov.rank_deficient, (
        f"rank {cov.rank} of {cov.num_free_params}; "
        f"weakest {cov.weak_directions[0].describe() if cov.weak_directions else 'n/a'}"
    )


# ---------------------------------------------------------------------------
# THE honesty check
# ---------------------------------------------------------------------------


def test_gt_coverage_check():
    """
    Does the reported uncertainty actually bracket the true error?

    Judged by REDUCED CHI-SQUARED, not by counting parameters inside 1 sigma.
    Parameter errors are strongly correlated: a single weakly-constrained mode
    (cx <-> board-x-translation <-> board-tilt) slides as one unit and pushes
    dozens of parameters to the same n-sigma with the same sign. Measured here,
    all 20 pose.dtx errors land at 2.68-2.71 sigma with identical sign -- that
    is one draw, not twenty, and a marginal-fraction test misreads it as
    twenty independent failures.

    e^T Sigma^-1 e removes the correlation and is chi^2(rank) under a correct
    covariance, so chi2/rank ~ 1 is the honest single-realisation verdict.
    """
    system, gt, problem, _ = solved(num_cameras=2, num_frames=20, noise=0.3, seed=77)
    cov = compute_covariance(problem, pixel_noise_std=0.3)
    check = parameter_error_vs_uncertainty(problem, cov, truth_blocks(gt))

    assert check["num_parameters"] > 30
    assert 0.5 < check["reduced_chi2"] < 2.0, (
        f"reduced chi2 = {check['reduced_chi2']:.3f} (chi2={check['mahalanobis_chi2']:.1f}, "
        f"dof={check['chi2_dof']}). Far from 1 means the covariance is wrong "
        f"in scale or in basis."
    )
    # No individual parameter should be wildly outside its own uncertainty.
    assert check["max_n_sigma"] < 6.0, check["worst"]


def test_uncertainty_is_calibrated_across_noise_realisations():
    """
    THE ensemble honesty test.

    Repeat the whole calibration over independent noise draws and check that
    reduced chi-squared averages to 1. Chi2/n has mean 1 and standard deviation
    sqrt(2/n); with n ~ 100 free parameters and 10 realisations the mean is
    pinned to a few percent, so a covariance that is even 25% too small or too
    large fails this.

    This is the test that would catch a wrong tangent basis, a missing factor
    of 2, or sigma^2 taken from the wrong source -- none of which the
    single-realisation test can reliably distinguish from bad luck.
    """
    reduced = _reduced_chi2_ensemble(num_frames=25, seeds=range(40, 50))
    mean = float(reduced.mean())
    assert 0.8 < mean < 1.4, (
        f"mean reduced chi2 over {reduced.size} noise realisations = {mean:.3f} "
        f"(want ~1.0). Individual values: {np.round(reduced, 3).tolist()}"
    )
    assert reduced.max() < 2.5, np.round(reduced, 3).tolist()


def _reduced_chi2_ensemble(num_frames: int, seeds) -> np.ndarray:
    out = []
    for seed in seeds:
        _, gt, problem, result = solved(num_cameras=1, num_frames=num_frames, noise=0.3, seed=seed)
        assert result.success
        cov = compute_covariance(problem, pixel_noise_std=0.3)
        out.append(parameter_error_vs_uncertainty(problem, cov, truth_blocks(gt))["reduced_chi2"])
    return np.array(out)


def test_linear_uncertainty_model_degrades_on_weakly_conditioned_problems():
    """
    A documented LIMITATION, asserted so it cannot quietly change.

    Sigma = sigma^2 (J^T J)^+ is a linearisation, exact only while the
    parameter error is small enough for the residual to be locally linear in
    it. With few frames the board-orientation parameters are weakly
    constrained and their errors reach ~0.06 rad, where that assumption starts
    to fail: reduced chi-squared then scatters far above 1 even though the
    covariance formula is right.

    Verified directly rather than assumed: pushing the KNOWN noise realisation
    through the linear relation e = (J^T J)^-1 J^T eps reproduces reduced chi2
    ~ 1 (0.98 on the worst seed), while the ACTUAL error departs from that
    linear prediction substantially. The discrepancy is nonlinearity, not a
    wrong covariance.

    Measured: max reduced chi2 over 8 seeds is 6.32 at 12 frames, 1.57 at 25,
    1.46 at 40.
    """
    few = _reduced_chi2_ensemble(num_frames=12, seeds=range(40, 48))
    many = _reduced_chi2_ensemble(num_frames=25, seeds=range(40, 48))
    assert few.max() > many.max() * 1.8, (
        f"expected a visibly worse chi2 spread with fewer frames; "
        f"12 frames max={few.max():.2f}, 25 frames max={many.max():.2f}"
    )
    assert many.max() < 2.0, np.round(many, 3).tolist()


def test_deliberately_understated_sigma_fails_the_same_check():
    """
    Proves the check above has teeth. Halving sigma^2 quarters the covariance,
    which must quadruple reduced chi-squared and be caught. If this test ever
    fails, the honesty check is not actually measuring anything.
    """
    system, gt, problem, _ = solved(num_cameras=1, num_frames=14, noise=0.3, seed=77)
    honest = compute_covariance(problem, pixel_noise_std=0.3, sigma_source="residual")
    liar = compute_covariance(problem, pixel_noise_std=0.05, sigma_source="assumed")

    good = parameter_error_vs_uncertainty(problem, honest, truth_blocks(gt))
    bad = parameter_error_vs_uncertainty(problem, liar, truth_blocks(gt))

    assert 0.5 < good["reduced_chi2"] < 2.0
    assert bad["reduced_chi2"] > 4.0 * good["reduced_chi2"]
    assert liar.noise_disagreement_flagged


def test_principal_point_deviation_is_covered_by_its_uncertainty():
    """
    The 15 px cx deviation that test_problem.py declines to bound arbitrarily
    must be explained by cx's own reported sigma. This is the pair to that test.
    """
    system, gt, problem, _ = solved(num_cameras=1, num_frames=16, noise=0.3, seed=5)
    cov = compute_covariance(problem, pixel_noise_std=0.3)
    check = parameter_error_vs_uncertainty(problem, cov, truth_blocks(gt))
    by_name = {r["parameter"]: r for r in check["parameters"]}
    for comp, label in ((2, "cx"), (3, "cy")):
        row = by_name.get(f"intr:cam0.{comp}")
        assert row is not None
        assert row["n_sigma"] < 4.0, (
            f"{label} is off by {row['error']:.2f} px which is "
            f"{row['n_sigma']:.1f} sigma -- the uncertainty does NOT cover it"
        )


# ---------------------------------------------------------------------------
# Projection uncertainty maps
# ---------------------------------------------------------------------------


def test_projection_uncertainty_map_is_finite_and_positive():
    system, _, problem, _ = solved()
    cov = compute_covariance(problem)
    m = projection_uncertainty(problem, system, cov, "cam0", range_m=1.5, grid_step=48)
    assert np.all(np.isfinite(m.sigma_max))
    assert np.all(m.sigma_max >= 0)
    assert m.worst() >= m.at_centre()


def test_uncertainty_grows_away_from_the_covered_centre():
    """Corners have less data, so they must not report LOWER uncertainty."""
    system, _, problem, _ = solved(num_frames=16)
    cov = compute_covariance(problem)
    m = projection_uncertainty(problem, system, cov, "cam0", range_m=1.5, grid_step=32)
    h, w = m.sigma_max.shape
    centre = float(np.mean(m.sigma_max[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]))
    corners = float(
        np.mean([m.sigma_max[0, 0], m.sigma_max[0, -1], m.sigma_max[-1, 0], m.sigma_max[-1, -1]])
    )
    assert corners >= centre * 0.95, f"corners {corners:.4f} vs centre {centre:.4f}"


def test_uncertainty_requires_an_explicit_range():
    system, _, problem, _ = solved()
    cov = compute_covariance(problem)
    with pytest.raises(ValueError):
        projection_uncertainty(problem, system, cov, "cam0", range_m=0.0)


def test_more_data_lowers_uncertainty():
    """A basic sanity property: more frames must not increase uncertainty."""
    _, _, p_small, _ = solved(num_cameras=1, num_frames=8, seed=2)
    _, _, p_big, _ = solved(num_cameras=1, num_frames=24, seed=2)
    c_small = compute_covariance(p_small)
    c_big = compute_covariance(p_big)
    fx_small = c_small.std_error_of("intr:cam0.0")
    fx_big = c_big.std_error_of("intr:cam0.0")
    assert fx_big < fx_small, f"fx sigma {fx_big:.4f} with 24 frames vs {fx_small:.4f} with 8"


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def test_build_report_produces_a_complete_artifact(tmp_path):
    system, gt, problem, result = solved(num_cameras=2, num_frames=12)
    report = build_report(
        problem,
        system,
        solve_result=result,
        pixel_noise_std=0.3,
        truth_values=truth_blocks(gt),
        grid_step=64,
    )
    d = report.to_dict()
    for key in (
        "system",
        "solve",
        "cameras",
        "covariance",
        "residuals",
        "coverage",
        "uncertainty",
        "warnings",
        "ground_truth_check",
    ):
        assert key in d and d[key] is not None, f"missing report section {key}"

    text = report.text_summary()
    assert "CALIBRATION REPORT" in text
    assert "PROJECTION UNCERTAINTY" in text
    assert "GROUND-TRUTH CHECK" in text

    path = report.save_json(tmp_path / "report.json")
    assert path.exists() and path.stat().st_size > 2000


def test_report_warns_when_rank_deficient():
    system, _, _, _ = solved(num_cameras=2, num_frames=8)
    problem = build_problem(system)
    problem.blocks[extr_key(system.reference_camera)].constant = False
    report = build_report(problem, system, grid_step=64)
    codes = [w.code for w in report.warnings]
    assert "rank_deficient" in codes


def test_crossval_reports_optimism():
    from mlti_cal.report.crossval import cross_validate

    system, _, _, _ = solved(num_cameras=1, num_frames=16, noise=0.3)
    cv = cross_validate(system, k=4, max_iterations=200)
    assert len(cv.folds) == 4
    assert np.isfinite(cv.mean_train_rms) and np.isfinite(cv.mean_test_rms)
    # Held-out error is essentially never better than training error.
    assert cv.mean_test_rms > cv.mean_train_rms * 0.8
