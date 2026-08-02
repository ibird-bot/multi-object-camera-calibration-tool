"""
Cross-backend agreement (M6).

The product claim is "pick your solver". That claim is only honest if every
backend is handed the identical problem and reaches the identical optimum. If
they disagree, either an adapter is translating wrongly or one of them is not
actually converging -- both are things a user must be told about, so they are
asserted here rather than hoped for.

Robust loss is left at trivial throughout: the core applies IRLS weights that
depend on the current residual, so a weighted comparison would be comparing two
slightly different objectives and would prove nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlti_cal.io.synthetic import generate_dataset
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import build_problem, intr_key
from mlti_cal.solvers import SolveOptions, backend_status, get_backend

CERES_OK, CERES_WHY = backend_status()["ceres"]
ceres_only = pytest.mark.skipif(not CERES_OK, reason=f"ceres unavailable: {CERES_WHY}")


def fresh_problem(**kw):
    kw.setdefault("num_cameras", 2)
    kw.setdefault("num_frames", 10)
    kw.setdefault("pixel_noise_std", 0.25)
    system, gt = generate_dataset(seed=21, **kw)
    initialize_system(system)
    return system, gt, build_problem(system)


def test_backend_status_reports_every_backend():
    status = backend_status()
    assert "scipy" in status and status["scipy"][0]
    assert "ceres" in status
    assert "gtsam" in status, "the gtsam adapter must be listed even when unusable"


@ceres_only
def test_ceres_solves_and_reduces_cost():
    _, _, problem = fresh_problem()
    result = get_backend("ceres").solve(problem, SolveOptions(max_iterations=100))
    assert result.final_cost < result.initial_cost
    assert result.final_rms_px < 1.0, result.summary()


@ceres_only
def test_scipy_and_ceres_reach_the_same_optimum():
    _, _, p_scipy = fresh_problem()
    base = p_scipy.get_state()

    # scipy's max_nfev counts FUNCTION EVALUATIONS, not outer iterations, and
    # trf needs far more of them than Ceres needs steps: at 300 it stops early
    # with "maximum number of function evaluations exceeded" and lands a hair
    # above the true optimum. Comparing a converged solver against a truncated
    # one would manufacture a disagreement that says nothing about the adapters.
    r_scipy = get_backend("scipy").solve(p_scipy, SolveOptions(max_iterations=5000))
    state_scipy = p_scipy.get_state()

    p_scipy.set_state(base)
    r_ceres = get_backend("ceres").solve(p_scipy, SolveOptions(max_iterations=300))
    state_ceres = p_scipy.get_state()

    assert r_scipy.success, f"scipy did not converge: {r_scipy.summary()}"
    assert r_ceres.success, f"ceres did not converge: {r_ceres.summary()}"
    assert r_scipy.final_cost == pytest.approx(r_ceres.final_cost, rel=1e-6), (
        f"\nscipy: {r_scipy.summary()}\nceres: {r_ceres.summary()}"
    )
    assert r_scipy.final_rms_px == pytest.approx(r_ceres.final_rms_px, rel=1e-6)

    # Intrinsics must agree to far better than their own uncertainty.
    for key in (k for k in state_scipy if k.startswith("intr:")):
        a, b = state_scipy[key], state_ceres[key]
        assert np.allclose(a[:4], b[:4], rtol=1e-5, atol=1e-4), f"{key}\n{a[:4]}\n{b[:4]}"


@ceres_only
@pytest.mark.parametrize(
    "linear_solver", ["SPARSE_SCHUR", "DENSE_SCHUR", "SPARSE_NORMAL_CHOLESKY", "ITERATIVE_SCHUR"]
)
def test_ceres_linear_solvers_agree(linear_solver):
    """
    Every linear solver solves the same normal equations, so the optimum must
    not depend on which one is chosen -- only the time taken should.
    """
    _, _, problem = fresh_problem(num_frames=8)
    base = problem.get_state()
    ref = get_backend("ceres").solve(
        problem, SolveOptions(max_iterations=200, extra={"linear_solver_type": "SPARSE_SCHUR"})
    )
    problem.set_state(base)
    got = get_backend("ceres").solve(
        problem,
        SolveOptions(max_iterations=200, extra={"linear_solver_type": linear_solver}),
    )
    assert got.final_cost == pytest.approx(ref.final_cost, rel=1e-4), (
        f"{linear_solver}: {got.final_cost} vs SPARSE_SCHUR {ref.final_cost}"
    )


@pytest.mark.parametrize("method", ["trf", "dogbox"])
def test_scipy_methods_agree(method):
    _, _, problem = fresh_problem(num_frames=8)
    base = problem.get_state()
    ref = get_backend("scipy").solve(problem, SolveOptions(max_iterations=400))
    problem.set_state(base)
    got = get_backend("scipy").solve(
        problem, SolveOptions(max_iterations=400, extra={"method": method})
    )
    assert got.final_cost == pytest.approx(ref.final_cost, rel=1e-4)


@ceres_only
def test_ceres_respects_fixed_intrinsic_components():
    system, _, _ = fresh_problem(num_cameras=1)
    problem = build_problem(system, fixed_intrinsic_components={"cam0": [8]})
    before = problem.blocks[intr_key("cam0")].value[8]
    get_backend("ceres").solve(problem, SolveOptions(max_iterations=60))
    after = problem.blocks[intr_key("cam0")].value[8]
    assert after == pytest.approx(before, abs=1e-15)


@ceres_only
def test_ceres_rejects_partial_rotation_fix_loudly():
    """A limitation must raise, not silently optimise what the user pinned."""
    from mlti_cal.solvers.ceres_backend import PartialRotationFixNotSupported

    system, _, _ = fresh_problem(num_cameras=2)
    problem = build_problem(system)
    blk = problem.blocks["extr:cam1"]
    blk.free_mask[3] = False  # pin one rotation component only
    with pytest.raises(PartialRotationFixNotSupported):
        get_backend("ceres").solve(problem, SolveOptions(max_iterations=5))


def test_scipy_supports_partial_rotation_fix():
    """The mask the Ceres backend rejects must genuinely work in scipy."""
    system, _, _ = fresh_problem(num_cameras=2)
    problem = build_problem(system)
    blk = problem.blocks["extr:cam1"]
    blk.free_mask[3] = False
    n_before = problem.num_free_params
    get_backend("scipy").solve(problem, SolveOptions(max_iterations=40))
    assert problem.num_free_params == n_before


# ---------------------------------------------------------------------------
# GTSAM convention conversion
#
# The adapter itself cannot run on this machine (no gtsam wheel for
# Windows/py3.12), but its two conversions are pure functions of numpy arrays,
# so the maths IS verifiable here. GTSAM's Pose3 retract is T * Exp([omega; v])
# with the FULL SE(3) exponential and rotation first; that is reproduced below
# using this repo's own se3_exp with the arguments swapped into our
# [translation; rotation] order.
# ---------------------------------------------------------------------------


def test_core_to_gtsam_jacobian_matches_finite_difference():
    from mlti_cal.models.manifolds import (
        d_point_d_pose_tangent,
        pose_to_matrix,
        quat_to_matrix,
        se3_exp,
    )
    from mlti_cal.solvers.gtsam_backend import core_to_gtsam_jacobian

    rng = np.random.default_rng(808)
    for _ in range(20):
        t = rng.normal(size=3)
        from mlti_cal.models.manifolds import matrix_to_quat, so3_exp

        q = matrix_to_quat(so3_exp(rng.normal(scale=0.7, size=3)))
        pose = np.concatenate([t, q])
        R = quat_to_matrix(q)
        X = rng.normal(size=(6, 3)) + np.array([0, 0, 3.0])

        J_core = d_point_d_pose_tangent(np.eye(3), R, X).reshape(-1, 6)
        J_gtsam = core_to_gtsam_jacobian(J_core, R)

        def fn(xi, pose=pose, X=X):
            # GTSAM: xi = [omega(3); v(3)], T_new = T * Exp_full([v; omega])
            T = pose_to_matrix(pose) @ se3_exp(np.concatenate([xi[3:6], xi[0:3]]))
            return ((T[:3, :3] @ X.T).T + T[:3, 3]).ravel()

        eps = 1e-7
        Jn = np.zeros_like(J_gtsam)
        for i in range(6):
            a, b = np.zeros(6), np.zeros(6)
            a[i], b[i] = eps, -eps
            Jn[:, i] = (fn(a) - fn(b)) / (2 * eps)
        assert np.allclose(Jn, J_gtsam, atol=1e-6), np.abs(Jn - J_gtsam).max()


def test_dropping_the_rotation_on_translation_columns_is_detectable():
    """The `@ R` in the conversion must matter, or the test above is vacuous."""
    from mlti_cal.models.manifolds import matrix_to_quat, quat_to_matrix, so3_exp
    from mlti_cal.solvers.gtsam_backend import core_to_gtsam_jacobian

    rng = np.random.default_rng(99)
    R = quat_to_matrix(matrix_to_quat(so3_exp(np.array([0.4, -0.3, 0.9]))))
    J_core = rng.normal(size=(12, 6))
    correct = core_to_gtsam_jacobian(J_core, R)
    naive = np.hstack([J_core[:, 3:6], J_core[:, 0:3]])  # permute only, no @R
    assert not np.allclose(correct, naive, atol=1e-6)


def test_gtsam_covariance_basis_roundtrip():
    from mlti_cal.models.manifolds import matrix_to_quat, quat_to_matrix, so3_exp
    from mlti_cal.solvers.gtsam_backend import (
        core_to_gtsam_jacobian,
        gtsam_to_core_covariance,
    )

    rng = np.random.default_rng(1234)
    R = quat_to_matrix(matrix_to_quat(so3_exp(np.array([0.2, 0.5, -0.1]))))
    J_core = rng.normal(size=(30, 6))
    # Sigma_core from the core Jacobian, Sigma_gtsam from the converted one;
    # mapping the latter back must reproduce the former exactly.
    cov_core = np.linalg.inv(J_core.T @ J_core)
    J_g = core_to_gtsam_jacobian(J_core, R)
    cov_gtsam = np.linalg.inv(J_g.T @ J_g)
    assert np.allclose(gtsam_to_core_covariance(cov_gtsam, R), cov_core, atol=1e-9)


def test_gtsam_reports_itself_unavailable_with_a_usable_message():
    ok, why = backend_status()["gtsam"]
    if ok:  # pragma: no cover - only on a conda/WSL machine
        pytest.skip("gtsam is installed here")
    assert "conda-forge" in why and "pygtsam" in why
