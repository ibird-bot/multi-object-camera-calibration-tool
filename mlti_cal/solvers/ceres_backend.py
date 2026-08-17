"""
Ceres adapter via pyceres.

Pose blocks are SPLIT
    pyceres 2.6 cannot express a custom manifold from Python -- `Plus`,
    `PlusJacobian`, `Minus` and `MinusJacobian` on `pyceres.Manifold` are
    throw-stubs, only `ambient_size`/`tangent_size` are wired to Python
    overrides. Verified against the installed binary, not assumed. So every
    pose becomes two Ceres parameter blocks:

        "<key>#t"  size 3, no manifold          -> t <- t + dt
        "<key>#q"  size 4, EigenQuaternionManifold -> q <- q (x) Exp(phi)

    which is precisely the repo's decoupled retraction (see models/manifolds).
    Core and Ceres therefore share one tangent basis and no covariance ever
    needs a basis correction.

Ambient vs tangent Jacobians
    A CostFunction is handed AMBIENT-sized buffers (4 columns for a
    quaternion) and Ceres reduces them itself. The core produces a 3-column
    tangent Jacobian J_phi. The exact bridge is

        J_ambient = J_phi @ pinv(PlusJacobian) = 2 * J_phi @ M(q)^T

    exact (not least-squares) because M(q) has orthonormal columns. Validated
    against Ceres' own gradient checker to ~1e-14.

Robust loss
    None is passed to Ceres. The core already applied per-corner IRLS weights;
    adding a Ceres loss on top would down-weight twice. See problem/loss.py.
"""

from __future__ import annotations

import time

import numpy as np

from mlti_cal.models.manifolds import quat_tangent_basis
from mlti_cal.problem.graph import POSE, ParameterBlock, Problem
from mlti_cal.solvers.base import (
    IterationRecord,
    SolveOptions,
    SolverBackend,
    SolveResult,
    count_behind_camera,
    per_corner_rms,
    register_backend,
)

try:
    import pyceres

    _PYCERES_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    pyceres = None
    _PYCERES_ERROR = str(exc)


LINEAR_SOLVERS = (
    "DENSE_QR",
    "DENSE_NORMAL_CHOLESKY",
    "SPARSE_NORMAL_CHOLESKY",
    "DENSE_SCHUR",
    "SPARSE_SCHUR",
    "ITERATIVE_SCHUR",
    "CGNR",
)
TRUST_REGION_STRATEGIES = ("LEVENBERG_MARQUARDT", "DOGLEG")
DOGLEG_TYPES = ("TRADITIONAL_DOGLEG", "SUBSPACE_DOGLEG")
PRECONDITIONERS = ("IDENTITY", "JACOBI", "SCHUR_JACOBI", "CLUSTER_JACOBI", "CLUSTER_TRIDIAGONAL")


class PartialRotationFixNotSupported(NotImplementedError):
    """Raised rather than silently optimising a component the user pinned."""


def _t_key(key: str) -> str:
    return f"{key}#t"


def _q_key(key: str) -> str:
    return f"{key}#q"


if pyceres is not None:

    class _CoreCostFunction(pyceres.CostFunction):
        """Wraps one core ResidualBlock as a Ceres CostFunction."""

        def __init__(self, residual, blocks: list[ParameterBlock]):
            super().__init__()
            self.residual = residual
            self.blocks = blocks
            self.set_num_residuals(residual.dim)
            sizes: list[int] = []
            for blk in blocks:
                if blk.kind == POSE:
                    sizes.extend([3, 4])
                else:
                    sizes.append(blk.value.size)
            self.set_parameter_block_sizes(sizes)

        def Evaluate(self, parameters, residuals, jacobians):  # noqa: N802 (Ceres API)
            # Rebuild core-shaped values from the split Ceres blocks.
            values: list[np.ndarray] = []
            i = 0
            for blk in self.blocks:
                if blk.kind == POSE:
                    t = np.asarray(parameters[i], dtype=float)
                    q = np.asarray(parameters[i + 1], dtype=float)
                    values.append(np.concatenate([t, q]))
                    i += 2
                else:
                    values.append(np.asarray(parameters[i], dtype=float))
                    i += 1

            want_jac = jacobians is not None
            r, jacs = self.residual.evaluate(values, with_jacobians=want_jac)
            residuals[:] = r
            if not want_jac:
                return True

            i = 0
            for bi, (blk, J) in enumerate(zip(self.blocks, jacs, strict=True)):
                if blk.kind == POSE:
                    if jacobians[i] is not None:
                        jacobians[i][:] = np.ascontiguousarray(J[:, 0:3]).ravel()
                    if jacobians[i + 1] is not None:
                        # Index by position, not list.index(): the same block
                        # object can legitimately appear twice in one residual.
                        q = values[bi][3:7]
                        M = quat_tangent_basis(q)  # (4,3), orthonormal columns
                        # PlusJacobian = 0.5*M  =>  pinv = 2*M^T. The factor 2
                        # is NOT optional; dropping it scales every rotation
                        # gradient by half, which still converges but reports
                        # a wrong Hessian and therefore a wrong covariance.
                        J_amb = 2.0 * (J[:, 3:6] @ M.T)  # (dim,4)
                        jacobians[i + 1][:] = np.ascontiguousarray(J_amb).ravel()
                    i += 2
                else:
                    if jacobians[i] is not None:
                        jacobians[i][:] = np.ascontiguousarray(J).ravel()
                    i += 1
            return True


def _write_state(problem: Problem, arrays: dict[str, np.ndarray]) -> None:
    """Copy Ceres' parameter buffers back into the core problem's blocks."""
    for key, blk in problem.blocks.items():
        if blk.kind == POSE:
            blk.value = np.concatenate([arrays[_t_key(key)], arrays[_q_key(key)]])
        else:
            blk.value = arrays[key].copy()


def _make_history_callback(history, problem, arrays, watcher):
    """
    Record Ceres' per-iteration progress, since the summary will not.

    pyceres 2.6 does not bind `Solver::Summary::iterations`, so the descent that
    Ceres certainly computed is simply not reachable from the summary object --
    checked against the installed binary, not assumed. `IterationCallback` is
    bound and carries the same numbers, so the history is collected as the solve
    runs instead. Without this the Ceres progress plot degenerates to a two-bar
    before/after chart while scipy draws a curve, which makes the two backends
    look different when only the reporting is.

    The RMS costs a state sync: Ceres owns its own buffers and the core blocks
    still hold the values from before the solve, so measuring without copying
    first would report the starting error at every iteration -- a flat line that
    looks like a solver doing nothing.
    """

    class _History(pyceres.IterationCallback):
        def __call__(self, summary) -> int:  # pragma: no cover - needs pyceres
            record = IterationRecord(
                iteration=int(summary.iteration),
                cost=float(summary.cost),
                gradient_norm=float(summary.gradient_max_norm),
                step_norm=float(summary.step_norm),
            )
            if watcher is not None:
                _write_state(problem, arrays)
                record.rms_px = per_corner_rms(problem)
            history.append(record)
            if watcher is not None:
                watcher(record)
            return pyceres.CallbackReturnType.SOLVER_CONTINUE

    return _History()


@register_backend
class CeresBackend(SolverBackend):
    name = "ceres"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if pyceres is None:
            return False, f"pyceres not importable: {_PYCERES_ERROR}"
        return True, ""

    def solve(self, problem: Problem, options: SolveOptions | None = None) -> SolveResult:
        ok, why = self.is_available()
        if not ok:
            raise RuntimeError(why)
        opts = options or SolveOptions()

        t0 = time.perf_counter()
        r0 = problem.residuals_only()
        initial_cost = 0.5 * float(r0 @ r0)
        initial_rms = per_corner_rms(problem)

        cp = pyceres.Problem()
        arrays: dict[str, np.ndarray] = {}

        # ---- parameter blocks ------------------------------------------
        for key, blk in problem.blocks.items():
            if blk.kind == POSE:
                arrays[_t_key(key)] = blk.value[0:3].copy()
                arrays[_q_key(key)] = blk.value[3:7].copy()
            else:
                arrays[key] = blk.value.copy()

        # ---- residuals --------------------------------------------------
        for res in problem.residuals:
            blocks = [problem.blocks[k] for k in res.block_keys]
            params: list[np.ndarray] = []
            for k, blk in zip(res.block_keys, blocks, strict=True):
                if blk.kind == POSE:
                    params.extend([arrays[_t_key(k)], arrays[_q_key(k)]])
                else:
                    params.append(arrays[k])
            cp.add_residual_block(_CoreCostFunction(res, blocks), None, params)

        # ---- manifolds, constants, per-component fixing -----------------
        for key, blk in problem.blocks.items():
            if blk.kind == POSE:
                ta, qa = arrays[_t_key(key)], arrays[_q_key(key)]
                if not cp.has_parameter_block(ta):
                    continue
                cp.set_manifold(qa, pyceres.EigenQuaternionManifold())
                if blk.constant or blk.num_free == 0:
                    cp.set_parameter_block_constant(ta)
                    cp.set_parameter_block_constant(qa)
                    continue
                mask = blk.free_mask
                if not mask[0:3].any():
                    cp.set_parameter_block_constant(ta)
                elif not mask[0:3].all():
                    fixed = [i for i in range(3) if not mask[i]]
                    cp.set_manifold(ta, pyceres.SubsetManifold(3, fixed))
                if not mask[3:6].any():
                    cp.set_parameter_block_constant(qa)
                elif not mask[3:6].all():
                    raise PartialRotationFixNotSupported(
                        f"block {key!r}: the Ceres backend cannot fix individual "
                        f"rotation components. EigenQuaternionManifold is "
                        f"all-or-nothing and pyceres 2.6 allows no custom "
                        f"manifold to express the subset. Fix all three "
                        f"rotation components or none, or use the scipy backend "
                        f"which supports arbitrary masks."
                    )
            else:
                arr = arrays[key]
                if not cp.has_parameter_block(arr):
                    continue
                if blk.constant or blk.num_free == 0:
                    cp.set_parameter_block_constant(arr)
                elif not blk.free_mask.all():
                    fixed = [int(i) for i in np.flatnonzero(~blk.free_mask)]
                    cp.set_manifold(arr, pyceres.SubsetManifold(blk.value.size, fixed))

        # ---- options ------------------------------------------------------
        so = pyceres.SolverOptions()
        so.max_num_iterations = opts.max_iterations
        so.function_tolerance = opts.function_tolerance
        so.gradient_tolerance = opts.gradient_tolerance
        so.parameter_tolerance = opts.parameter_tolerance
        so.num_threads = opts.num_threads
        so.minimizer_progress_to_stdout = bool(opts.verbose)

        linear_solver = opts.get("linear_solver_type", "SPARSE_SCHUR")
        _set_enum(so, "linear_solver_type", pyceres.LinearSolverType, linear_solver)
        strategy = opts.get("trust_region_strategy_type", "LEVENBERG_MARQUARDT")
        _set_enum(so, "trust_region_strategy_type", pyceres.TrustRegionStrategyType, strategy)
        if strategy == "DOGLEG":
            _set_enum(
                so, "dogleg_type", pyceres.DoglegType, opts.get("dogleg_type", "TRADITIONAL_DOGLEG")
            )
        precond = opts.get("preconditioner_type")
        if precond:
            _set_enum(so, "preconditioner_type", pyceres.PreconditionerType, precond)
        if opts.get("use_nonmonotonic_steps"):
            so.use_nonmonotonic_steps = True
        if opts.get("use_inner_iterations"):
            so.use_inner_iterations = True

        history: list[IterationRecord] = []
        # Held in a local: Ceres keeps a raw pointer, and letting Python collect
        # the callback mid-solve is a crash, not a missing log line.
        callback = _make_history_callback(history, problem, arrays, opts.on_iteration)
        so.callbacks = [callback]
        # Without this the parameter buffers are only written at the end, so the
        # per-iteration RMS would measure the starting point every time.
        so.update_state_every_iteration = True

        summary = pyceres.SolverSummary()
        pyceres.solve(so, cp, summary)

        # ---- write the solution back into the core problem ---------------
        _write_state(problem, arrays)

        r_final = problem.residuals_only()
        term = str(summary.termination_type)
        return SolveResult(
            backend=self.name,
            success=term.endswith("CONVERGENCE") or term.endswith("USER_SUCCESS"),
            message=(summary.message or "").strip()[:400],
            initial_cost=initial_cost,
            final_cost=0.5 * float(r_final @ r_final),
            initial_rms_px=initial_rms,
            final_rms_px=per_corner_rms(problem),
            iterations=len(history) or int(summary.num_successful_steps),
            time_seconds=time.perf_counter() - t0,
            num_residuals=problem.num_residuals,
            num_free_params=problem.num_free_params,
            options={
                "linear_solver_type": linear_solver,
                "trust_region_strategy_type": strategy,
                "preconditioner_type": precond,
                "max_num_iterations": opts.max_iterations,
                "termination": term,
            },
            history=history,
            behind_camera_points=count_behind_camera(problem),
        )


def _set_enum(target, attr: str, enum_cls, value: str) -> None:
    """Set a Ceres enum option by name, failing loudly on an unknown value."""
    if value is None:
        return
    if not hasattr(enum_cls, value):
        valid = [n for n in dir(enum_cls) if n.isupper()]
        raise ValueError(f"{attr}: {value!r} is not valid; choose from {valid}")
    setattr(target, attr, getattr(enum_cls, value))
