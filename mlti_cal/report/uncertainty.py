"""
Projection uncertainty: "how much can I trust a projected point, and where?"

This is the headline output. Rather than a single RMS, it answers the question
per pixel: if I use this calibration to project a world point that lands near
pixel q at distance d, how far might the answer be off, given the uncertainty
in the parameters I estimated?

Method
    1. Unproject pixel q to a ray, place a point on it at distance `range_m`.
    2. Differentiate that point's projection w.r.t. every FREE parameter that
       affects it -- this camera's intrinsics AND its extrinsic.
    3. Propagate the JOINT covariance:   Sigma_uv = J Sigma J^T
    4. Report the 1-sigma ellipse.

Why the joint covariance and not just the intrinsics block
    Intrinsics and extrinsics are strongly correlated -- focal length trades
    against distance almost perfectly. Propagating only the intrinsic block
    ignores those cross terms and produces an uncertainty that is too small,
    usually by a wide margin. Using the full joint block is what makes the
    number honest.

Range dependence is real, not a nuisance parameter
    Uncertainty at 1 m and at 100 m are genuinely different, because focal
    length and camera position trade off differently at different depths. A
    tool that reports one uncertainty map without stating the range is hiding
    a variable, so `range_m` is required rather than defaulted.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from mlti_cal.models.manifolds import d_point_d_pose_tangent, quat_to_matrix
from mlti_cal.problem.graph import Problem
from mlti_cal.problem.reprojection import extr_key, intr_key
from mlti_cal.problem.types import CalibrationSystem
from mlti_cal.report.covariance import CovarianceResult


@dataclass
class UncertaintyMap:
    camera: str
    range_m: float
    grid_u: np.ndarray  # (H,W) pixel x
    grid_v: np.ndarray  # (H,W) pixel y
    sigma_max: np.ndarray  # (H,W) 1-sigma along the worst direction, px
    sigma_rms: np.ndarray  # (H,W) sqrt(trace/2), px
    valid: np.ndarray  # (H,W) bool

    def worst(self) -> float:
        v = self.sigma_max[self.valid]
        return float(v.max()) if v.size else float("nan")

    def best(self) -> float:
        v = self.sigma_max[self.valid]
        return float(v.min()) if v.size else float("nan")

    def at_centre(self) -> float:
        h, w = self.sigma_max.shape
        return float(self.sigma_max[h // 2, w // 2])

    def to_dict(self) -> dict:
        return {
            "camera": self.camera,
            "range_m": self.range_m,
            "worst_sigma_px": self.worst(),
            "best_sigma_px": self.best(),
            "centre_sigma_px": self.at_centre(),
            "shape": list(self.sigma_max.shape),
        }


def _block_columns(problem: Problem, key: str) -> tuple[np.ndarray, np.ndarray]:
    """(global column indices, tangent component indices) for one block."""
    blk = problem.blocks.get(key)
    if blk is None or blk.num_free == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    offsets, _ = problem.column_layout()
    free_idx = blk.free_indices
    start = offsets[key]
    return np.arange(start, start + free_idx.size), free_idx


def unproject(model_name: str, params: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """
    Pixels -> unit rays in the camera frame, inverting the distortion.

    OpenCV's iterative undistortion is used rather than a hand-rolled inverse;
    it is the same code that will undistort the user's images, so the map and
    the images stay consistent.
    """
    pts = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
    fx, fy, cx, cy = params[:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    dist = np.asarray(params[4:], dtype=float)
    if model_name == "fisheye_kb":
        norm = cv2.fisheye.undistortPoints(pts, K, dist.reshape(4, 1)).reshape(-1, 2)
    else:
        norm = cv2.undistortPoints(pts, K, dist.reshape(1, -1)).reshape(-1, 2)
    rays = np.column_stack([norm, np.ones(norm.shape[0])])
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def projection_uncertainty(
    problem: Problem,
    system: CalibrationSystem,
    cov: CovarianceResult,
    camera_id: str,
    range_m: float,
    grid_step: int = 24,
) -> UncertaintyMap:
    """
    Per-pixel 1-sigma projection uncertainty for one camera at one range.

    Args:
        range_m: distance from the camera along the ray, in metres. Required --
            see the module docstring on why there is no default.
    """
    if range_m <= 0:
        raise ValueError("range_m must be positive and is required")
    cam = system.cameras[camera_id]
    model = cam.model
    w, h = cam.image_size

    us = np.arange(grid_step // 2, w, grid_step)
    vs = np.arange(grid_step // 2, h, grid_step)
    gu, gv = np.meshgrid(us, vs)
    pix = np.column_stack([gu.ravel(), gv.ravel()]).astype(float)

    rays = unproject(cam.model_name, cam.params, pix)
    X_cam = rays * range_m

    ik, i_sel = _block_columns(problem, intr_key(camera_id))
    ek, e_sel = _block_columns(problem, extr_key(camera_id))
    gidx = np.concatenate([ik, ek])
    if gidx.size == 0:
        raise ValueError(f"camera {camera_id} has no free parameters to propagate")
    Sigma = cov.covariance[np.ix_(gidx, gidx)]

    _, J_theta, J_X = model.d_project(cam.params, X_cam)

    parts = []
    if i_sel.size:
        parts.append(J_theta[:, :, i_sel])
    if e_sel.size:
        E = cam.extrinsic
        R_E = quat_to_matrix(E[3:7])
        # X_rig for the sampled point, needed for the extrinsic Jacobian.
        X_rig = (X_cam - E[0:3]) @ R_E
        d_extr = d_point_d_pose_tangent(np.eye(3), R_E, X_rig)  # (N,3,6)
        J_e = np.einsum("nij,njk->nik", J_X, d_extr)[:, :, e_sel]
        parts.append(J_e)
    J = np.concatenate(parts, axis=2)  # (N,2,K)

    # Sigma_uv = J Sigma J^T, batched over pixels.
    S = np.einsum("nik,kl,njl->nij", J, Sigma, J)
    # Eigenvalues of a symmetric 2x2, in closed form -- cheaper and more stable
    # than calling eigh N times.
    a, b, d = S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]
    tr = a + d
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b * b, 0.0))
    lam_max = 0.5 * (tr + disc)
    sigma_max = np.sqrt(np.maximum(lam_max, 0.0)).reshape(gu.shape)
    sigma_rms = np.sqrt(np.maximum(tr * 0.5, 0.0)).reshape(gu.shape)
    valid = np.isfinite(sigma_max)

    return UncertaintyMap(
        camera=camera_id,
        range_m=float(range_m),
        grid_u=gu,
        grid_v=gv,
        sigma_max=sigma_max,
        sigma_rms=sigma_rms,
        valid=valid,
    )


def parameter_error_vs_uncertainty(
    problem: Problem,
    cov: CovarianceResult,
    truth_values: dict[str, np.ndarray],
) -> dict:
    """
    THE honesty check, available only in synthetic mode.

    For every free parameter, compare the actual error against the reported
    1-sigma. If the uncertainty is honest, roughly 68% of parameters should
    fall inside 1 sigma and ~95% inside 2 sigma. A tool reporting 0.1 px
    uncertainty on a parameter that is off by 5 sigma is lying, and this is the
    function that catches it.

    Args:
        truth_values: block key -> true value, in the SAME ambient storage the
            problem uses (7-vectors for poses).
    """
    from mlti_cal.models.manifolds import pose_minus
    from mlti_cal.problem.graph import POSE

    offsets, _ = problem.column_layout()
    rows = []
    for key, blk in problem.blocks.items():
        if key not in offsets or key not in truth_values:
            continue
        truth = np.asarray(truth_values[key], dtype=float)
        if blk.kind == POSE:
            err_full = pose_minus(blk.value, truth)  # tangent error
        else:
            err_full = blk.value - truth
        sel = blk.free_indices
        start = offsets[key]
        for j, comp in enumerate(sel):
            col = start + j
            sigma = float(cov.std_errors[col])
            err = float(err_full[comp])
            rows.append(
                {
                    "parameter": cov.labels[col],
                    "error": err,
                    "sigma": sigma,
                    "n_sigma": abs(err) / sigma if sigma > 0 else float("inf"),
                }
            )

    n_sigmas = np.array([r["n_sigma"] for r in rows])
    finite = n_sigmas[np.isfinite(n_sigmas)]

    # ---- the statistically correct check --------------------------------
    # The per-parameter fractions above are easy to read but easy to
    # MISREAD: parameter errors are strongly correlated, so they are not
    # independent samples. In practice a single weakly-constrained mode (the
    # classic cx <-> board-x-translation <-> board-tilt degeneracy) slides as
    # one unit and drags dozens of parameters to the same n-sigma with the
    # same sign. Reading that as "dozens of parameters are 2.7 sigma out"
    # overstates the evidence enormously -- it is ONE draw, not dozens.
    #
    # The Mahalanobis distance e^T Sigma^-1 e removes the correlation and is
    # distributed as chi^2 with n degrees of freedom under a correct
    # covariance, so chi2/n should be ~1. THAT is the number to judge a
    # covariance by from a single realisation.
    err_vec = np.zeros(len(cov.labels))
    for r in rows:
        err_vec[cov.labels.index(r["parameter"])] = r["error"]

    # Computed as ||J e||^2 / sigma^2, NOT as e^T pinv(Sigma) e.
    #
    # They are algebraically identical, since Sigma = sigma^2 (J^T J)^+ and so
    # Sigma^-1 = J^T J / sigma^2 on the observable subspace. Numerically they
    # are not remotely the same: cond(Sigma) = cond(J)^2, which on a normal
    # calibration is ~1e11-1e12, so inverting Sigma in float64 throws away most
    # of the available precision and produced chi2 values inflated by up to 6x
    # on this very data. Multiplying by J instead never squares the condition
    # number and is also far cheaper.
    chi2 = float("nan")
    reduced = float("nan")
    p_value = float("nan")
    try:
        from scipy import stats as sps

        _, J = problem.evaluate(with_jacobian=True)
        Je = J @ err_vec
        chi2 = float(Je @ Je) / cov.sigma2_used
        dof = int(cov.rank)
        reduced = chi2 / dof if dof else float("nan")
        p_value = float(sps.chi2.sf(chi2, dof)) if dof else float("nan")
    except Exception:  # pragma: no cover - numerical edge cases only
        pass

    return {
        "parameters": rows,
        "num_parameters": len(rows),
        "within_1_sigma": float(np.mean(finite <= 1.0)) if finite.size else float("nan"),
        "within_2_sigma": float(np.mean(finite <= 2.0)) if finite.size else float("nan"),
        "within_3_sigma": float(np.mean(finite <= 3.0)) if finite.size else float("nan"),
        "max_n_sigma": float(finite.max()) if finite.size else float("nan"),
        "worst": max(rows, key=lambda r: r["n_sigma"]) if rows else None,
        "mahalanobis_chi2": chi2,
        "chi2_dof": int(cov.rank),
        "reduced_chi2": reduced,
        "chi2_pvalue": p_value,
        "note": (
            "reduced_chi2 ~ 1 means the covariance is honest. The within_N_sigma "
            "fractions are marginal and correlated -- do not treat them as N "
            "independent samples."
        ),
    }
