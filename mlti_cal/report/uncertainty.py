"""
Projection uncertainty: "how much can I trust a projected point, and where?"

This is the headline output. Rather than a single RMS, it answers the question
per pixel: if I use this calibration to project a world point that lands near
pixel q at distance d, how far might the answer be off, given the uncertainty
in the parameters I estimated?

Method
    1. Unproject pixel q to a ray, place a point on it at distance `range_m`.
    2. Hold that point fixed RELATIVE TO THE CALIBRATION BOARDS, not in the
       camera frame (see below), and differentiate its projection w.r.t. every
       FREE parameter that affects it -- this camera's intrinsics, its
       extrinsic, AND every board pose.
    3. Propagate the JOINT covariance:   Sigma_uv = J Sigma J^T
    4. Report the 1-sigma ellipse.

Why the point is fixed to the boards (`reference="boards"`)
    The camera frame is not a physical thing; it is defined by the solve. A
    noise draw that moves cx by 10 px also rotates every board pose to
    compensate, and a physical point keeps projecting to almost the same pixel.
    Holding the point fixed in camera coordinates ignores that compensation and
    reports sigma(cx) at the image centre -- measured: 10.35 px claimed where
    Monte Carlo re-solves spread by 0.84 px. So the point is carried into each
    board's frame with the solved pose, moved with the perturbed pose, and the
    results are averaged over all boards (mrcal's "mean-pcam" method):

        p_rig*  = mean_i  T_rig_board_i' * inv(T_rig_board_i) * p_rig
        q*      = project(intrinsics', T_cam_rig' * p_rig*)

    With this repo's retraction (t += dt, R <- R Exp(phi)) the mean is exactly
    one rigid motion  d(p_rig) = a - skew(p_rig) b,  with
        a = mean_i (dt_i + skew(t_i) R_i phi_i),    b = mean_i (R_i phi_i)
    so the whole set of pose blocks collapses into six numbers and the cost
    does not grow with the number of frames.

    `reference="camera"` keeps the old quantity. It answers "how loosely is
    the camera frame itself pinned down", which is a diagnostic, not a
    projection error.

Why the joint covariance and not just the intrinsics block
    Intrinsics, extrinsics and board poses are strongly correlated -- focal
    length trades against distance almost perfectly. Propagating only the
    intrinsic block ignores those cross terms. Using the full joint block is
    what makes the number honest.

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

from mlti_cal.models.camera import get_model
from mlti_cal.models.manifolds import d_point_d_pose_tangent, quat_to_matrix, skew, skew_stack
from mlti_cal.problem.graph import Problem
from mlti_cal.problem.reprojection import extr_key, intr_key, pose_key
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
    #: What the query point was held fixed to: "boards" (physical projection
    #: error) or "camera" (looseness of the camera frame). See module docstring.
    reference: str = "boards"

    def worst(self) -> float:
        v = self.sigma_max[self.valid]
        return float(v.max()) if v.size else float("nan")

    def best(self) -> float:
        v = self.sigma_max[self.valid]
        return float(v.min()) if v.size else float("nan")

    def at_centre(self) -> float:
        h, w = self.sigma_max.shape
        return float(self.sigma_max[h // 2, w // 2])

    @property
    def invalid_fraction(self) -> float:
        """Share of the sensor where the distortion model is not invertible."""
        return float(1.0 - self.valid.mean())

    def to_dict(self) -> dict:
        return {
            "camera": self.camera,
            "range_m": self.range_m,
            "reference": self.reference,
            "worst_sigma_px": self.worst(),
            "best_sigma_px": self.best(),
            "centre_sigma_px": self.at_centre(),
            "invalid_fraction": self.invalid_fraction,
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


def unproject(
    model_name: str,
    params: np.ndarray,
    pixels: np.ndarray,
    model=None,
    tolerance_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pixels -> unit rays in the camera frame, inverting the distortion.

    Returns (rays, valid). For OpenCV-compatible models OpenCV's own
    undistortion is used, so the map stays consistent with the code that will
    actually undistort the user's images. Every other model inverts its own
    projection (`undistort_to_normalised`): handing its coefficients to OpenCV
    would either raise or, worse, undistort by the wrong law.

    THE VALIDITY CHECK IS NOT OPTIONAL. A distortion polynomial fitted on the
    middle of the sensor is frequently non-invertible near the corners, and
    `cv2.undistortPoints` does not report failure -- its fixed-point iteration
    simply diverges and returns a huge normalised coordinate. Propagating that
    through the Jacobian produced a reported "uncertainty" of 1.8 MILLION
    pixels on this project's own demo data. Round-tripping each pixel and
    marking the failures invalid turns a nonsense number into an honest
    "outside the region this calibration can speak for".
    """
    if model is None:
        model = get_model(model_name)
    pts = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if model.opencv_compatible:
        K = model.matrix(params)
        dist = model.distortion(params)
        if model_name == "fisheye_kb":
            norm = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2), K, dist.reshape(4, 1))
        else:
            norm = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, dist.reshape(1, -1))
        norm = norm.reshape(-1, 2)
    else:
        norm = model.undistort_to_normalised(params, pts)
    rays = np.column_stack([norm, np.ones(norm.shape[0])])

    valid = np.all(np.isfinite(rays), axis=1)
    safe = np.where(valid[:, None], rays, np.array([0.0, 0.0, 1.0]))
    back = model.project(params, safe)
    err = np.linalg.norm(back - pts, axis=1)
    valid &= np.isfinite(err) & (err < tolerance_px)

    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rays / norms, valid


def projection_uncertainty(
    problem: Problem,
    system: CalibrationSystem,
    cov: CovarianceResult,
    camera_id: str,
    range_m: float,
    grid_step: int = 24,
    reference: str = "boards",
) -> UncertaintyMap:
    """
    Per-pixel 1-sigma projection uncertainty for one camera at one range.

    Args:
        range_m: distance from the camera along the ray, in metres. Required --
            see the module docstring on why there is no default.
        reference: "boards" (default) holds the query point fixed relative to
            the calibration boards -- the physical projection error. "camera"
            holds it fixed in camera coordinates -- a frame-looseness diagnostic.
    """
    if range_m <= 0:
        raise ValueError("range_m must be positive and is required")
    if reference not in ("boards", "camera"):
        raise ValueError(f"reference must be 'boards' or 'camera', got {reference!r}")
    cam = system.cameras[camera_id]
    model = cam.model
    w, h = cam.image_size

    def solved(key: str, fallback: np.ndarray) -> np.ndarray:
        # Linearise at the values the covariance was computed at. Reading them
        # from `system` would mix stale values with a fresh covariance whenever
        # a report is built before `write_back`.
        blk = problem.blocks.get(key)
        return np.asarray(blk.value if blk is not None else fallback, dtype=float)

    params = solved(intr_key(camera_id), cam.params)

    us = np.arange(grid_step // 2, w, grid_step)
    vs = np.arange(grid_step // 2, h, grid_step)
    gu, gv = np.meshgrid(us, vs)
    pix = np.column_stack([gu.ravel(), gv.ravel()]).astype(float)

    rays, ray_valid = unproject(cam.model_name, params, pix, model=model)
    X_cam = rays * range_m

    ik, i_sel = _block_columns(problem, intr_key(camera_id))
    ek, e_sel = _block_columns(problem, extr_key(camera_id))

    # M maps the full free-parameter vector onto the few numbers the projection
    # depends on: [this camera's intrinsics | its extrinsic | mean board motion].
    # Reducing the covariance through M first keeps the per-pixel Jacobian a
    # handful of columns wide however many frames there are.
    E = solved(extr_key(camera_id), cam.extrinsic)
    R_E = quat_to_matrix(E[3:7])
    X_rig = (X_cam - E[0:3]) @ R_E  # the sampled point in the rig frame
    mean_motion = np.zeros((6, cov.covariance.shape[0]))
    num_poses = 0
    if reference == "boards":
        for frame, board in system.board_poses:
            key = pose_key(frame, board)
            if key not in problem.blocks:
                continue
            num_poses += 1  # a pinned pose still counts: its copy of the point stays put
            cols, sel = _block_columns(problem, key)
            if cols.size == 0:
                continue
            pose = np.asarray(problem.blocks[key].value, dtype=float)
            R_i = quat_to_matrix(pose[3:7])
            A = np.zeros((6, 6))  # [a; b] contribution of [dt_i | phi_i]
            A[0:3, 0:3] = np.eye(3)
            A[0:3, 3:6] = skew(pose[0:3]) @ R_i
            A[3:6, 3:6] = R_i
            mean_motion[:, cols] = A[:, sel]
    has_motion = num_poses > 0 and bool(np.any(mean_motion))
    num_reduced = ik.size + ek.size + (6 if has_motion else 0)
    if num_reduced == 0:
        raise ValueError(f"camera {camera_id} has no free parameters to propagate")
    M = np.zeros((num_reduced, cov.covariance.shape[0]))
    own = np.concatenate([ik, ek])
    M[np.arange(own.size), own] = 1.0
    if has_motion:
        M[own.size :] = mean_motion / num_poses
    Sigma = M @ cov.covariance @ M.T

    _, J_theta, J_X = model.d_project(params, X_cam)

    parts = []
    if i_sel.size:
        parts.append(J_theta[:, :, i_sel])
    if e_sel.size:
        d_extr = d_point_d_pose_tangent(np.eye(3), R_E, X_rig)  # (N,3,6)
        J_e = np.einsum("nij,njk->nik", J_X, d_extr)[:, :, e_sel]
        parts.append(J_e)
    if has_motion:
        # d(p_cam)/d[a; b] = R_E [ I | -skew(p_rig) ]
        d_rig = np.concatenate(
            [np.broadcast_to(np.eye(3), (len(X_rig), 3, 3)), -skew_stack(X_rig)], axis=2
        )
        parts.append(np.einsum("nij,jk,nkl->nil", J_X, R_E, d_rig))
    J = np.concatenate(parts, axis=2)  # (N,2,K)

    # Sigma_uv = J Sigma J^T, batched over pixels.
    S = np.einsum("nik,kl,njl->nij", J, Sigma, J)
    # Eigenvalues of a symmetric 2x2, in closed form -- cheaper and more stable
    # than calling eigh N times.
    a, b, d = S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]
    tr = a + d
    disc = np.sqrt(np.maximum((a - d) ** 2 + 4 * b * b, 0.0))
    lam_max = 0.5 * (tr + disc)
    sigma_max = np.sqrt(np.maximum(lam_max, 0.0))
    sigma_rms = np.sqrt(np.maximum(tr * 0.5, 0.0))

    # Pixels whose distortion could not be inverted carry no meaningful
    # uncertainty. They are NaN'd rather than zeroed so they cannot be mistaken
    # for confident regions, and excluded from worst()/best().
    valid = ray_valid & np.isfinite(sigma_max)
    sigma_max = np.where(valid, sigma_max, np.nan).reshape(gu.shape)
    sigma_rms = np.where(valid, sigma_rms, np.nan).reshape(gu.shape)
    valid = valid.reshape(gu.shape)

    return UncertaintyMap(
        camera=camera_id,
        range_m=float(range_m),
        grid_u=gu,
        grid_v=gv,
        sigma_max=sigma_max,
        sigma_rms=sigma_rms,
        valid=valid,
        reference=reference,
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
