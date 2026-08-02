"""
Problem-graph assembly, the increment reparameterisation, and the first real
calibration (M3 vertical slice).

The Jacobian check here is deliberately done at a NON-ZERO increment. At x = 0
the SO(3) right-Jacobian correction is the identity, so a test that only probes
the base point cannot distinguish a correct implementation from one that omits
the correction entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlti_cal.io.synthetic import generate_dataset, relative_extrinsic_error
from mlti_cal.models.manifolds import so3_left_jacobian
from mlti_cal.problem.graph import POSE
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import build_problem, extr_key, intr_key, write_back
from mlti_cal.solvers import SolveOptions, get_backend

RNG = np.random.default_rng(4242)


def small_system(**kw):
    kw.setdefault("num_cameras", 1)
    kw.setdefault("num_frames", 8)
    kw.setdefault("pixel_noise_std", 0.0)
    system, gt = generate_dataset(seed=11, **kw)
    initialize_system(system)
    return system, gt


def right_jacobian_transforms(problem, deltas):
    out = {}
    for key, blk in problem.blocks.items():
        if blk.kind != POSE or not blk.num_free:
            continue
        phi = deltas[key][3:6]
        if not np.any(phi):
            continue
        M = np.eye(6)
        M[3:6, 3:6] = so3_left_jacobian(phi).T
        out[key] = M
    return out


def test_problem_builds_and_has_expected_blocks():
    system, _ = small_system(num_cameras=2)
    problem = build_problem(system)
    for cid in system.cameras:
        assert intr_key(cid) in problem.blocks
        assert extr_key(cid) in problem.blocks
    ref = system.reference_camera
    assert problem.blocks[extr_key(ref)].constant, "reference extrinsic must be fixed"
    assert not problem.blocks[extr_key("cam1")].constant
    assert problem.num_free_params > 0


def test_reference_extrinsic_contributes_no_columns():
    """The gauge fix must remove columns, not merely be 'left alone'."""
    system, _ = small_system(num_cameras=2)
    problem = build_problem(system)
    offsets, _ = problem.column_layout()
    assert extr_key(system.reference_camera) not in offsets


@pytest.mark.parametrize("at_zero", [True, False])
def test_problem_jacobian_matches_finite_difference(at_zero):
    system, _ = small_system(num_cameras=2, num_frames=4)
    problem = build_problem(system)
    n = problem.num_free_params
    base = problem.get_state()

    x = np.zeros(n) if at_zero else RNG.normal(scale=0.02, size=n)

    deltas = problem.set_from_base(base, x)
    _, J = problem.evaluate(
        with_jacobian=True, tangent_transform=right_jacobian_transforms(problem, deltas)
    )
    Ja = J.toarray()

    def f(xx):
        problem.set_from_base(base, xx)
        return problem.residuals_only()

    eps = 1e-6
    Jn = np.empty_like(Ja)
    for i in range(n):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        Jn[:, i] = (f(xp) - f(xm)) / (2 * eps)

    scale = np.maximum(np.abs(Jn).max(axis=0), 1e-8)
    err = (np.abs(Ja - Jn) / scale).max()
    assert err < 1e-4, f"worst relative column error {err:.3e}"


def test_omitting_right_jacobian_is_detectably_wrong():
    """
    Confirms the correction is load-bearing: without it the Jacobian at a
    non-zero increment must NOT match finite differences. If this test ever
    starts passing with the transform disabled, the check above is vacuous.
    """
    system, _ = small_system(num_cameras=1, num_frames=3)
    problem = build_problem(system)
    n = problem.num_free_params
    base = problem.get_state()
    x = RNG.normal(scale=0.25, size=n)  # large enough that J_r differs from I

    problem.set_from_base(base, x)
    _, J_wrong = problem.evaluate(with_jacobian=True)  # no transform
    Jw = J_wrong.toarray()

    def f(xx):
        problem.set_from_base(base, xx)
        return problem.residuals_only()

    eps = 1e-6
    cols = [i for i in range(n) if ".r" in problem.parameter_labels()[i]]
    assert cols, "expected some rotation columns"
    worst = 0.0
    for i in cols[:12]:
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        num = (f(xp) - f(xm)) / (2 * eps)
        denom = max(np.abs(num).max(), 1e-8)
        worst = max(worst, np.abs(Jw[:, i] - num).max() / denom)
    assert worst > 1e-3, "right-Jacobian correction appears to be a no-op"


# ---------------------------------------------------------------------------
# M3: first real calibration
# ---------------------------------------------------------------------------


def test_monocular_recovers_ground_truth_noise_free():
    system, gt = generate_dataset(num_cameras=1, num_frames=12, pixel_noise_std=0.0, seed=3)
    initialize_system(system)
    problem = build_problem(system)
    before = problem.rms()
    result = get_backend("scipy").solve(problem, SolveOptions(max_iterations=200))
    write_back(system, problem)

    assert result.final_cost <= result.initial_cost
    assert result.final_rms_px < 1e-3, result.summary()
    assert before >= problem.rms()

    est = system.cameras["cam0"].params
    true = gt.camera_params["cam0"]
    assert np.allclose(est[:4], true[:4], rtol=2e-3), f"\nest {est[:4]}\ntrue {true[:4]}"
    assert np.allclose(est[4:], true[4:], atol=5e-3), f"\nest {est[4:]}\ntrue {true[4:]}"


def test_monocular_with_noise_lands_near_noise_floor():
    noise = 0.3
    system, gt = generate_dataset(num_cameras=1, num_frames=16, pixel_noise_std=noise, seed=5)
    initialize_system(system)
    problem = build_problem(system)
    result = get_backend("scipy").solve(problem, SolveOptions(max_iterations=200))
    write_back(system, problem)

    # Per-corner Euclidean RMS of 2D Gaussian noise with per-axis std s is
    # s*sqrt(2); the fit absorbs a little, so allow a band rather than a point.
    expected = noise * np.sqrt(2.0)
    assert 0.5 * expected < result.final_rms_px < 1.5 * expected, result.summary()

    est = system.cameras["cam0"].params
    true = gt.camera_params["cam0"]

    # Focal length is well constrained by planar targets and must be close.
    assert np.allclose(est[:2], true[:2], rtol=0.02), f"focal {est[:2]} vs {true[:2]}"

    # The PRINCIPAL POINT deliberately gets no tight tolerance here. With
    # planar boards cx/cy are weakly observable and strongly correlated with
    # board tilt and with p1/p2; at 0.3 px noise this configuration moves cx by
    # ~15 px while RMS stays at the noise floor. The noise-free test above
    # pins cx to ~1 px, which proves the estimator is unbiased -- what is left
    # here is variance, not error.
    #
    # Asserting an arbitrary pixel tolerance would be exactly the dishonesty
    # this project exists to expose. The meaningful check is whether the
    # reported uncertainty BRACKETS this deviation, and that is what
    # tests/test_uncertainty.py::test_gt_coverage_check asserts.
    assert np.isfinite(est).all()


def test_multicamera_recovers_extrinsics():
    system, gt = generate_dataset(num_cameras=3, num_frames=16, pixel_noise_std=0.0, seed=9)
    initialize_system(system)
    problem = build_problem(system)
    result = get_backend("scipy").solve(problem, SolveOptions(max_iterations=300))
    write_back(system, problem)

    assert result.final_rms_px < 5e-3, result.summary()
    for cid in system.cameras:
        t_err, r_err = relative_extrinsic_error(
            system.cameras[cid].extrinsic, gt.camera_extrinsics[cid]
        )
        assert t_err < 1e-3, f"{cid}: translation error {t_err * 1000:.3f} mm"
        assert r_err < 0.05, f"{cid}: rotation error {r_err:.4f} deg"


def test_multiboard_multicamera_runs():
    system, gt = generate_dataset(
        num_cameras=2,
        num_frames=12,
        boards_spec=(("board_A", 9, 7, 0.03), ("board_B", 6, 5, 0.04)),
        pixel_noise_std=0.0,
        seed=13,
    )
    initialize_system(system)
    problem = build_problem(system)
    result = get_backend("scipy").solve(problem, SolveOptions(max_iterations=300))
    write_back(system, problem)
    assert result.final_rms_px < 1e-2, result.summary()
    assert len(system.boards) == 2


def test_fixed_intrinsic_component_stays_fixed():
    system, _ = small_system(num_cameras=1, num_frames=6)
    k3_before = system.cameras["cam0"].params[8]
    problem = build_problem(system, fixed_intrinsic_components={"cam0": [8]})
    get_backend("scipy").solve(problem, SolveOptions(max_iterations=40))
    write_back(system, problem)
    assert system.cameras["cam0"].params[8] == pytest.approx(k3_before, abs=1e-15)
