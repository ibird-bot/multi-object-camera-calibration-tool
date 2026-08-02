"""
GTSAM adapter.

=============================================================================
STATUS: WRITTEN BUT NEVER EXECUTED. Read this before trusting it.
=============================================================================
There is no `gtsam` wheel for Windows / CPython 3.12, and `pygtsam` does not
exist on PyPI at all (both verified). Running this adapter requires either

    Miniforge + `conda install -c conda-forge gtsam`      (free, one-time)
    or WSL with a Linux Python

Until it runs somewhere, every line below is unvalidated. It is registered so
that `backend_status()` reports it as present-but-unusable rather than
pretending it does not exist -- but do NOT treat its numbers as verified the
first time it executes. Check it against the scipy backend on the same problem
first; `tests/test_backends.py::test_scipy_and_ceres_reach_the_same_optimum`
is the template.

-----------------------------------------------------------------------------
TWO convention mismatches, both handled explicitly here
-----------------------------------------------------------------------------
1. ORDERING. GTSAM's Pose3 tangent is [rotation(3); translation(3)] --
   rotation FIRST. This repo uses [translation(3); rotation(3)]. The columns
   must be permuted, not reinterpreted.

2. COUPLING. GTSAM's Pose3 retract (with the default Expmap) is the FULL SE(3)
   exponential:

       T_new = T * Exp([omega; v]),   t_new = t + R * V(omega) * v

   whereas this repo's retraction is decoupled:

       t_new = t + dt,   R_new = R * Exp(phi)

   Evaluated at zero the two relate by

       phi = omega                and        dt = R * v

   so the Jacobian conversion is

       dr/d(omega) = dr/d(phi)
       dr/d(v)     = dr/d(dt) @ R

   Note the R on the translation columns. Omitting it is the classic silent
   failure: the solve still converges (the gradient at the linearisation point
   is merely rotated, not wrong in direction for small motions) but the
   reported covariance is expressed in the wrong basis and every uncertainty
   number downstream is quietly incorrect.
"""

from __future__ import annotations

import time

import numpy as np

from mlti_cal.models.manifolds import quat_to_matrix
from mlti_cal.problem.graph import POSE, Problem
from mlti_cal.solvers.base import (
    SolveOptions,
    SolverBackend,
    SolveResult,
    count_behind_camera,
    per_corner_rms,
    register_backend,
)

try:
    import gtsam

    _GTSAM_ERROR = ""
except Exception as exc:  # pragma: no cover - expected on this machine
    gtsam = None
    _GTSAM_ERROR = str(exc)


OPTIMIZERS = ("LEVENBERG_MARQUARDT", "GAUSS_NEWTON", "DOGLEG")


def core_to_gtsam_jacobian(J_core: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Convert a (dim, 6) core pose Jacobian into GTSAM's tangent convention.

    Core columns:  [dt(3) | phi(3)]      (translation first, decoupled)
    GTSAM columns: [omega(3) | v(3)]     (rotation first, full exponential)

    Pure function, no gtsam import -- so it IS unit-testable on this machine
    even though the surrounding adapter is not. See
    tests/test_backends_gtsam_convention.py.
    """
    J_core = np.asarray(J_core, dtype=float)
    if J_core.shape[1] != 6:
        raise ValueError(f"expected 6 tangent columns, got {J_core.shape[1]}")
    J_dt = J_core[:, 0:3]
    J_phi = J_core[:, 3:6]
    return np.hstack([J_phi, J_dt @ np.asarray(R, dtype=float)])


def gtsam_to_core_covariance(cov_gtsam: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Map a 6x6 GTSAM pose covariance back into this repo's tangent basis.

    With  [omega; v] = A [dt; phi],  A = [[0, I], [R^T, 0]],  the covariance
    transforms as  Sigma_core = A^-1 Sigma_gtsam A^-T.
    """
    R = np.asarray(R, dtype=float)
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = R.T
    Ainv = np.linalg.inv(A)
    return Ainv @ np.asarray(cov_gtsam, dtype=float) @ Ainv.T


@register_backend
class GtsamBackend(SolverBackend):
    name = "gtsam"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if gtsam is None:
            return False, (
                "gtsam is not installed. No Windows/py3.12 pip wheel exists "
                "(and 'pygtsam' is not a real PyPI package). Install via "
                "Miniforge: `conda install -c conda-forge gtsam`, or use WSL. "
                f"Import error: {_GTSAM_ERROR}"
            )
        return True, ""

    def solve(self, problem: Problem, options: SolveOptions | None = None) -> SolveResult:
        ok, why = self.is_available()
        if not ok:
            raise RuntimeError(why)
        opts = options or SolveOptions()  # pragma: no cover - unreachable here

        t0 = time.perf_counter()
        r0 = problem.residuals_only()
        initial_cost = 0.5 * float(r0 @ r0)
        initial_rms = per_corner_rms(problem)

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        keys: dict[str, int] = {}
        next_key = 0

        for name, blk in problem.blocks.items():
            keys[name] = next_key
            if blk.kind == POSE:
                R = quat_to_matrix(blk.value[3:7])
                values.insert(next_key, gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*blk.value[0:3])))
            else:
                values.insert(next_key, blk.value.copy())
            next_key += 1

        noise = gtsam.noiseModel.Unit.Create(1)

        for res in problem.residuals:
            res_keys = [keys[k] for k in res.block_keys]
            blocks = [problem.blocks[k] for k in res.block_keys]

            def error_fn(_this, v, H=None, res=res, blocks=blocks, res_keys=res_keys):
                vals = []
                for bk, blk in zip(res_keys, blocks, strict=True):
                    if blk.kind == POSE:
                        p = v.atPose3(bk)
                        q = _rot_to_quat(p.rotation().matrix())
                        vals.append(np.concatenate([p.translation(), q]))
                    else:
                        vals.append(np.asarray(v.atVector(bk), dtype=float))
                r, jacs = res.evaluate(vals, with_jacobians=H is not None)
                if H is not None:
                    for i, (blk, J) in enumerate(zip(blocks, jacs, strict=True)):
                        if blk.kind == POSE:
                            R = quat_to_matrix(vals[i][3:7])
                            H[i] = core_to_gtsam_jacobian(J, R)
                        else:
                            H[i] = J
                return r

            graph.add(gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(res.dim), res_keys, error_fn))
            del noise

        # Gauge + fixed blocks: GTSAM has no set-constant, so pin with a
        # very tight prior. Documented as an approximation, unlike Ceres'
        # exact set_parameter_block_constant.
        for name, blk in problem.blocks.items():
            if not (blk.constant or blk.num_free == 0):
                continue
            tight = gtsam.noiseModel.Isotropic.Sigma(
                6 if blk.kind == POSE else blk.value.size, 1e-9
            )
            if blk.kind == POSE:
                R = quat_to_matrix(blk.value[3:7])
                graph.add(
                    gtsam.PriorFactorPose3(
                        keys[name], gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*blk.value[0:3])), tight
                    )
                )

        kind = opts.get("optimizer", "LEVENBERG_MARQUARDT")
        if kind == "GAUSS_NEWTON":
            params = gtsam.GaussNewtonParams()
            params.setMaxIterations(opts.max_iterations)
            optimizer = gtsam.GaussNewtonOptimizer(graph, values, params)
        elif kind == "DOGLEG":
            params = gtsam.DoglegParams()
            params.setMaxIterations(opts.max_iterations)
            optimizer = gtsam.DoglegOptimizer(graph, values, params)
        else:
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(opts.max_iterations)
            params.setRelativeErrorTol(opts.function_tolerance)
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)

        result_values = optimizer.optimize()

        for name, blk in problem.blocks.items():
            if blk.kind == POSE:
                p = result_values.atPose3(keys[name])
                blk.value = np.concatenate(
                    [np.asarray(p.translation()), _rot_to_quat(p.rotation().matrix())]
                )
            else:
                blk.value = np.asarray(result_values.atVector(keys[name]), dtype=float)

        r_final = problem.residuals_only()
        return SolveResult(
            backend=self.name,
            success=True,
            message=f"gtsam {kind}, {optimizer.iterations()} iterations",
            initial_cost=initial_cost,
            final_cost=0.5 * float(r_final @ r_final),
            initial_rms_px=initial_rms,
            final_rms_px=per_corner_rms(problem),
            iterations=int(optimizer.iterations()),
            time_seconds=time.perf_counter() - t0,
            num_residuals=problem.num_residuals,
            num_free_params=problem.num_free_params,
            options={"optimizer": kind, "max_iterations": opts.max_iterations},
            behind_camera_points=count_behind_camera(problem),
        )


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    from mlti_cal.models.manifolds import matrix_to_quat

    return matrix_to_quat(R)
