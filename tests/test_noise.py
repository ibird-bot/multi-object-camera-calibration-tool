"""
Pixel-noise measurement from a static sequence.

The load-bearing test is the first one: inject a KNOWN sigma and check the
estimator recovers it. Everything else in this file exists because a noise
estimate that is quietly wrong is worse than none -- it becomes the whitening
of every residual and the yardstick the covariance is checked against.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlti_cal.detectors.charuco import Detection
from mlti_cal.detectors.noise import (
    MIN_FRAMES,
    NoiseEstimate,
    estimate_pixel_noise,
)
from mlti_cal.io.config import CalibrationConfig


def _sequence(n_frames=30, sigma=0.2, drift=0.0, n_corners=60, seed=0, board="b"):
    """A static sequence with known noise, optionally with the board creeping."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(50, 1900, size=(n_corners, 2))
    ids = np.arange(n_corners)
    frames = []
    for t in range(n_frames):
        pts = base + np.array([drift * t, 0.0]) + rng.normal(0, sigma, size=base.shape)
        frames.append([Detection(board, ids, pts)])
    return frames


@pytest.mark.parametrize("true_sigma", [0.02, 0.1, 0.3, 0.8])
def test_recovers_a_known_sigma(true_sigma):
    """The whole point: measured sigma must equal the sigma that was injected."""
    est = estimate_pixel_noise(_sequence(n_frames=40, sigma=true_sigma))
    assert est.sigma_px == pytest.approx(true_sigma, rel=0.06)
    assert est.sigma_x == pytest.approx(true_sigma, rel=0.10)
    assert est.sigma_y == pytest.approx(true_sigma, rel=0.10)


def test_board_motion_is_separated_from_noise():
    """
    A creeping tripod must not be reported as detection noise.

    This is the failure the common-mode decomposition exists for: the raw
    spread is dominated by the motion, and only the de-drifted number is a
    noise floor.
    """
    est = estimate_pixel_noise(_sequence(n_frames=30, sigma=0.2, drift=0.1))
    assert est.sigma_px == pytest.approx(0.2, rel=0.10), "motion leaked into the noise estimate"
    assert est.sigma_raw_px > 3 * est.sigma_px, "raw spread should be dominated by the motion"
    assert est.common_motion_px > 0.5
    assert est.drift_px > 2.0
    assert any("not static" in w for w in est.warnings)
    assert any("drift" in w for w in est.warnings)


def test_a_genuinely_static_sequence_says_so():
    est = estimate_pixel_noise(_sequence(n_frames=30, sigma=0.2))
    assert est.common_motion_px < 0.1
    assert any("genuinely static" in w for w in est.warnings)


def test_identical_frames_are_called_out_rather_than_reported_as_perfect():
    rng = np.random.default_rng(1)
    base = rng.uniform(50, 1900, size=(40, 2))
    frames = [[Detection("b", np.arange(40), base.copy())] for _ in range(10)]
    est = estimate_pixel_noise(frames)
    assert est.sigma_px == pytest.approx(0.0, abs=1e-9)
    assert any("zero" in w for w in est.warnings)
    # ...and no spurious anisotropy claim from dividing one ~0 by another.
    assert not any("differ by" in w for w in est.warnings)


def test_too_few_frames_refuses_instead_of_guessing():
    est = estimate_pixel_noise(_sequence(n_frames=MIN_FRAMES - 1))
    assert not est.usable
    assert any("frames" in w for w in est.warnings)


def test_rarely_seen_corners_are_dropped_and_counted():
    """A corner seen twice contributes luck, not a variance estimate."""
    frames = _sequence(n_frames=20, sigma=0.2, n_corners=10)
    ghost_id = 999
    for t in (0, 1):
        det = frames[t][0]
        frames[t][0] = Detection(
            det.board_id,
            np.append(det.point_ids, ghost_id),
            np.vstack([det.image_points, [[100.0, 100.0]]]),
        )
    est = estimate_pixel_noise(frames)
    assert est.num_corners_used == 10
    assert est.num_corners_dropped == 1
    assert all(c.point_id != ghost_id for c in est.per_corner)


def test_anisotropic_noise_is_reported_not_averaged_away():
    rng = np.random.default_rng(3)
    base = rng.uniform(50, 1900, size=(50, 2))
    ids = np.arange(50)
    frames = []
    for _ in range(30):
        noise = np.column_stack([rng.normal(0, 0.1, 50), rng.normal(0, 0.5, 50)])
        frames.append([Detection("b", ids, base + noise)])
    est = estimate_pixel_noise(frames)
    assert est.sigma_x == pytest.approx(0.1, rel=0.15)
    assert est.sigma_y == pytest.approx(0.5, rel=0.15)
    assert any("differ by" in w for w in est.warnings)


def test_pooling_weights_by_degrees_of_freedom():
    """
    A long series must outweigh a short one.

    Naive averaging of per-corner variances would let a 5-frame corner count as
    much as a 40-frame one, which is how a noise floor ends up set by its worst
    evidence.
    """
    rng = np.random.default_rng(4)
    long_pts = rng.normal(0, 0.2, size=(40, 2)) + np.array([500.0, 500.0])
    short_pts = rng.normal(0, 2.0, size=(5, 2)) + np.array([900.0, 900.0])
    frames = []
    for t in range(40):
        ids = [0] + ([1] if t < 5 else [])
        pts = [long_pts[t]] + ([short_pts[t]] if t < 5 else [])
        frames.append([Detection("b", np.array(ids), np.array(pts))])
    est = estimate_pixel_noise(frames, min_frames_per_corner=5)
    assert est.num_corners_used == 2
    # Both corners contribute, but the 40-frame one dominates: a dof-weighted
    # pool sits far below the midpoint of 0.2 and 2.0.
    assert est.sigma_px < 0.8


def test_summary_lines_are_readable_and_safe_when_empty():
    assert "no usable corners" in NoiseEstimate().summary_lines()[0]
    est = estimate_pixel_noise(_sequence())
    text = "\n".join(est.summary_lines())
    assert "pixel noise" in text and "px per coordinate" in text


# ---------------------------------------------------------------------------
# "I do not know the noise" is a real answer
# ---------------------------------------------------------------------------


def test_config_round_trips_unknown_noise(tmp_path):
    cfg = CalibrationConfig(cameras=[], boards=[], pixel_noise_std=None)
    back = CalibrationConfig.load(cfg.save(tmp_path / "c.json"))
    assert back.pixel_noise_std is None


def test_config_distinguishes_absent_from_null(tmp_path):
    """Absent means an older config; explicit null means the user said unknown."""
    absent = tmp_path / "absent.json"
    absent.write_text('{"cameras": [], "boards": []}', encoding="utf-8")
    assert CalibrationConfig.load(absent).pixel_noise_std == 0.3

    null = tmp_path / "null.json"
    null.write_text('{"cameras": [], "boards": [], "pixel_noise_std": null}', encoding="utf-8")
    assert CalibrationConfig.load(null).pixel_noise_std is None


def test_unknown_noise_leaves_residuals_unweighted():
    from mlti_cal.io.synthetic import generate_dataset

    system, _ = generate_dataset(num_frames=6, seed=0)
    for obs in system.observations:
        obs.sigma = 1.0
    assert all(o.sigma == 1.0 for o in system.observations)


def test_covariance_skips_the_cross_check_when_noise_is_unknown():
    """
    No assumed value means no mismatch warning -- there is no claim to check.

    The covariance itself is unaffected: `sigma_source` is "residual" by
    default, so it never used the assumed number for the sigmas anyway.
    """
    from mlti_cal.io.synthetic import generate_dataset
    from mlti_cal.problem.initialize import initialize_system
    from mlti_cal.problem.reprojection import build_problem
    from mlti_cal.report.covariance import compute_covariance

    system, _ = generate_dataset(num_frames=8, seed=0)
    initialize_system(system)
    problem = build_problem(system)

    known = compute_covariance(problem, pixel_noise_std=0.3)
    unknown = compute_covariance(problem, pixel_noise_std=None)

    assert np.isnan(unknown.sigma2_assumed)
    assert np.isnan(unknown.noise_disagreement)
    assert unknown.noise_disagreement_flagged is False
    # Same covariance, only the cross-check differs.
    assert unknown.sigma2_residual == pytest.approx(known.sigma2_residual)
    assert unknown.std_errors == pytest.approx(known.std_errors)


def test_report_says_not_stated_rather_than_nan():
    from mlti_cal.io.synthetic import generate_dataset
    from mlti_cal.problem.initialize import initialize_system
    from mlti_cal.problem.reprojection import build_problem
    from mlti_cal.report.report import build_report

    system, _ = generate_dataset(num_frames=8, seed=0)
    initialize_system(system)
    problem = build_problem(system)
    report = build_report(problem, system, pixel_noise_std=None)
    text = report.text_summary()
    assert "not stated" in text
    assert "sigma (assumed): nan" not in text
    assert not any(w.code == "noise_model_mismatch" for w in report.warnings)
