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
    # Registration is what is asserted, not usability: a backend whose optional
    # dependency is missing must still appear, with the reason, rather than
    # vanishing and reading as "never supported".
    assert set(status) == {"ceres", "scipy"}


@pytest.mark.parametrize("name", [n for n, (ok, _) in backend_status().items() if ok])
def test_every_backend_reports_its_descent(name):
    """
    A backend that solves silently cannot be compared to one that does not.

    pyceres 2.6 does not bind `Solver::Summary::iterations`, so reading the
    history off the summary produced an empty list and the Ceres progress plot
    collapsed to a before/after bar while scipy drew a curve -- the adapters
    looked different when only the reporting was. Ceres now collects the same
    numbers through an IterationCallback, and this asserts it keeps doing so.
    """
    _, _, problem = fresh_problem()
    result = get_backend(name).solve(problem, SolveOptions(max_iterations=100))
    costs = [h.cost for h in result.history if np.isfinite(h.cost)]
    assert len(costs) >= 2, f"{name} reported {len(costs)} usable cost record(s)"
    assert costs[-1] < costs[0], f"{name}: history does not descend"
    assert result.iterations == len(result.history)


@pytest.mark.parametrize("name", [n for n, (ok, _) in backend_status().items() if ok])
def test_on_iteration_streams_a_measured_rms(name):
    """
    The live view must report MEASURED pixels, not cost dressed up as pixels.

    Deriving RMS from cost is exact only under a trivial loss; under a robust
    one the cost is weighted and the derived number flatters the fit by exactly
    what the loss discounts. So each record carries an RMS the backend measured
    at that step, and the last one must equal the RMS of the returned solution
    -- if it does not, records are being emitted from a state the solver went on
    to abandon.
    """
    _, _, problem = fresh_problem()
    seen: list = []
    result = get_backend(name).solve(
        problem, SolveOptions(max_iterations=100, on_iteration=seen.append)
    )
    assert len(seen) >= 2, f"{name} streamed {len(seen)} record(s)"
    assert all(np.isfinite(r.rms_px) for r in seen), f"{name}: an RMS was not measured"
    assert seen[-1].rms_px == pytest.approx(result.final_rms_px, abs=1e-9)
    assert seen[0].rms_px == pytest.approx(result.initial_rms_px, rel=1e-6)


def test_no_watcher_means_no_rms_measurement_cost():
    """The extra projection pass is for the live view only; headless pays nothing."""
    _, _, problem = fresh_problem()
    result = get_backend("scipy").solve(problem, SolveOptions(max_iterations=30))
    assert result.history, "history is recorded either way"
    assert all(not np.isfinite(h.rms_px) for h in result.history)


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
