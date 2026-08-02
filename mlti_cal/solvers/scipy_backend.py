"""
scipy.optimize.least_squares adapter.

The manifold problem
    scipy has no concept of a manifold: it optimises a flat vector. The
    variable here is therefore an INCREMENT `x` from a fixed base state, with
    the state reconstructed as `base (+) x` on every evaluation.

    That reparameterisation changes the Jacobian, and the change is not
    negligible. Our analytic Jacobian is with respect to a local perturbation
    at the CURRENT point:

        q_current (x) Exp(delta)

    whereas scipy differentiates with respect to the accumulated increment:

        q_base (x) Exp(phi),     x_rot = phi

    Since  Exp(phi + dphi) = Exp(phi) Exp(J_r(phi) dphi),  the two are related
    by SO(3)'s RIGHT Jacobian:

        dr/dphi = dr/ddelta @ J_r(phi),      J_r(phi) = J_l(phi)^T

    Ignoring this (i.e. pretending J_r = I) still converges -- the correction
    is identity at phi = 0 and the gradient at the base point is exact -- but
    the Jacobian handed to the optimiser would be wrong everywhere else, which
    degrades the trust-region model and, worse, means the reported convergence
    diagnostics describe a function that is not the one being minimised. We
    apply the correction exactly.

Translation needs no correction: it is additive in both parameterisations.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares

from mlti_cal.models.manifolds import so3_left_jacobian
from mlti_cal.problem.graph import POSE, Problem
from mlti_cal.solvers.base import (
    IterationRecord,
    SolveOptions,
    SolveResult,
    SolverBackend,
    count_behind_camera,
    per_corner_rms,
    register_backend,
)

#: `method` values scipy exposes, with the trade-off that matters here.
METHODS = ("trf", "lm", "dogbox")


@register_backend
class ScipyBackend(SolverBackend):
    name = "scipy"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        return True, ""

    def solve(self, problem: Problem, options: SolveOptions | None = None) -> SolveResult:
        opts = options or SolveOptions()
        method = opts.get("method", "trf")
        if method not in METHODS:
            raise ValueError(f"scipy method must be one of {METHODS}, got {method!r}")
        tr_solver = opts.get("tr_solver", "lsmr" if method != "lm" else None)
        x_scale = opts.get("x_scale", "jac")

        base_state = problem.get_state()
        n_free = problem.num_free_params
        if n_free == 0:
            raise ValueError("problem has no free parameters")

        t0 = time.perf_counter()
        r0 = problem.residuals_only()
        initial_cost = 0.5 * float(r0 @ r0)
        initial_rms = per_corner_rms(problem)

        history: list[IterationRecord] = []
        pose_keys = [k for k, b in problem.blocks.items() if b.kind == POSE and b.num_free]

        def right_jacobian_transforms(deltas: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            """blkdiag(I_3, J_r(phi)) per pose block, for the current increment."""
            out: dict[str, np.ndarray] = {}
            for key in pose_keys:
                phi = deltas[key][3:6]
                if not np.any(phi):
                    continue  # J_r(0) = I, skip the multiply entirely
                M = np.eye(6)
                M[3:6, 3:6] = so3_left_jacobian(phi).T  # J_r = J_l^T
                out[key] = M
            return out

        state: dict = {"deltas": None}

        def fun(x: np.ndarray) -> np.ndarray:
            state["deltas"] = problem.set_from_base(base_state, x)
            r = problem.residuals_only()
            history.append(IterationRecord(len(history), 0.5 * float(r @ r)))
            return r

        def jac(x: np.ndarray):
            deltas = problem.set_from_base(base_state, x)
            _, J = problem.evaluate(
                with_jacobian=True, tangent_transform=right_jacobian_transforms(deltas)
            )
            # 'lm' is a dense-only method in scipy and rejects sparse input.
            return J.toarray() if method == "lm" else sp.csr_matrix(J)

        kwargs = dict(
            fun=fun,
            x0=np.zeros(n_free),
            jac=jac,
            method=method,
            max_nfev=opts.max_iterations,
            ftol=opts.function_tolerance,
            gtol=opts.gradient_tolerance,
            xtol=opts.parameter_tolerance,
            verbose=2 if opts.verbose else 0,
        )
        if method != "lm":
            kwargs["tr_solver"] = tr_solver
            kwargs["x_scale"] = x_scale
        loss_name = opts.get("scipy_loss")
        if loss_name and loss_name != "linear":
            # Exposed for completeness, but the core already applies a robust
            # loss per corner; stacking a second one here double-counts.
            kwargs["loss"] = loss_name
            kwargs["f_scale"] = opts.get("f_scale", 1.0)

        result = least_squares(**kwargs)

        # Leave the problem AT the solution, not wherever the last probe was.
        problem.set_from_base(base_state, result.x)
        r_final = problem.residuals_only()
        final_cost = 0.5 * float(r_final @ r_final)

        return SolveResult(
            backend=self.name,
            success=bool(result.success),
            message=str(result.message),
            initial_cost=initial_cost,
            final_cost=final_cost,
            initial_rms_px=initial_rms,
            final_rms_px=per_corner_rms(problem),
            iterations=int(result.nfev),
            time_seconds=time.perf_counter() - t0,
            num_residuals=problem.num_residuals,
            num_free_params=n_free,
            options={
                "method": method,
                "tr_solver": tr_solver,
                "x_scale": x_scale if method != "lm" else None,
                "max_nfev": opts.max_iterations,
            },
            history=history,
            behind_camera_points=count_behind_camera(problem),
        )
