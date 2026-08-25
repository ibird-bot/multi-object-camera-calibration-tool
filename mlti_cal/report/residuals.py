"""
Residual analysis: everything a single RMS number throws away.

A low RMS is compatible with a badly wrong calibration. The diagnostics here
are the ones that actually distinguish "small random error" from "small
systematic error", which is the distinction that matters:

  * per-camera / per-board / per-image breakdown -- a global RMS averages a bad
    camera together with three good ones
  * the residual QUIVER FIELD -- systematic model inadequacy shows up as
    structured swirls or radial patterns, and is completely invisible to RMS
  * normality checks -- reprojection residuals should look Gaussian; skew,
    excess kurtosis and a heavy tail all mean something is unmodelled
  * error vs radius -- growth toward the image edge is the signature of a
    distortion model that cannot represent the lens
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlti_cal.problem.graph import Problem
from mlti_cal.problem.reprojection import ReprojectionResidual


@dataclass
class ResidualStats:
    errors: np.ndarray  # (N,) per-corner Euclidean px
    vectors: np.ndarray  # (N,2) predicted - observed, px
    points: np.ndarray  # (N,2) observed pixel location
    camera_ids: np.ndarray  # (N,) str
    board_ids: np.ndarray  # (N,) str
    frame_ids: np.ndarray  # (N,) str
    overall_rms: float = 0.0
    overall_mean: float = 0.0
    overall_max: float = 0.0
    per_camera: dict[str, dict] = field(default_factory=dict)
    per_board: dict[str, dict] = field(default_factory=dict)
    per_image: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall_rms_px": self.overall_rms,
            "overall_mean_px": self.overall_mean,
            "overall_max_px": self.overall_max,
            "num_corners": int(self.errors.size),
            "per_camera": self.per_camera,
            "per_board": self.per_board,
            "worst_images": sorted(
                ({"image": k, **v} for k, v in self.per_image.items()),
                key=lambda d: -d["rms_px"],
            )[:10],
        }


def _group_stats(errors: np.ndarray) -> dict:
    if errors.size == 0:
        return {"rms_px": float("nan"), "mean_px": float("nan"), "max_px": float("nan"), "n": 0}
    return {
        "rms_px": float(np.sqrt(np.mean(errors**2))),
        "mean_px": float(np.mean(errors)),
        "max_px": float(np.max(errors)),
        "n": int(errors.size),
    }


def compute_residual_stats(problem: Problem) -> ResidualStats:
    errs, vecs, pts, cams, boards, frames = [], [], [], [], [], []
    for res in problem.residuals:
        if not isinstance(res, ReprojectionResidual):
            continue
        values = [problem.blocks[k].value for k in res.block_keys]
        pred = res.predicted(values)
        d = pred - res.uv_obs
        errs.append(np.linalg.norm(d, axis=1))
        vecs.append(d)
        pts.append(res.uv_obs)
        n = res.num_points
        cams.append(np.full(n, res.camera_id))
        boards.append(np.full(n, res.board))
        frames.append(np.full(n, res.frame))

    if not errs:
        empty = np.zeros(0)
        return ResidualStats(
            empty, np.zeros((0, 2)), np.zeros((0, 2)), *(np.zeros(0, dtype=str),) * 3
        )

    stats = ResidualStats(
        errors=np.concatenate(errs),
        vectors=np.concatenate(vecs),
        points=np.concatenate(pts),
        camera_ids=np.concatenate(cams),
        board_ids=np.concatenate(boards),
        frame_ids=np.concatenate(frames),
    )
    stats.overall_rms = float(np.sqrt(np.mean(stats.errors**2)))
    stats.overall_mean = float(np.mean(stats.errors))
    stats.overall_max = float(np.max(stats.errors))
    for cid in np.unique(stats.camera_ids):
        stats.per_camera[str(cid)] = _group_stats(stats.errors[stats.camera_ids == cid])
    for bid in np.unique(stats.board_ids):
        stats.per_board[str(bid)] = _group_stats(stats.errors[stats.board_ids == bid])
    for res in problem.residuals:
        if isinstance(res, ReprojectionResidual):
            values = [problem.blocks[k].value for k in res.block_keys]
            stats.per_image[res.tag] = _group_stats(res.reprojection_errors(values))
    return stats


def normality(stats: ResidualStats) -> dict:
    """
    Are the residuals plausibly Gaussian?

    Reported as descriptive statistics plus a Kolmogorov-Smirnov p-value
    against a fitted normal. A tiny p-value on a large sample is expected even
    for good calibrations -- what matters is the SIZE of the departure (skew,
    excess kurtosis, tail ratio), so all three are reported rather than a
    single pass/fail.
    """
    from scipy import stats as sps

    x = stats.vectors.ravel()
    if x.size < 8:
        return {"n": int(x.size)}
    mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    # Standardise and test against the plain standard normal. scipy 1.18 no
    # longer accepts `args=(loc, scale)` alongside a string distribution name.
    ks_p = float(sps.kstest((x - mu) / sd, "norm").pvalue) if sd > 0 else float("nan")
    q = np.abs(x) / max(sd, 1e-12)
    return {
        "n": int(x.size),
        "mean": mu,
        "std": sd,
        "skew": float(sps.skew(x)),
        "excess_kurtosis": float(sps.kurtosis(x)),
        "ks_pvalue": ks_p,
        "fraction_beyond_3_sigma": float(np.mean(q > 3.0)),
        "expected_beyond_3_sigma": 0.0027,
    }


def qq_data(stats: ResidualStats, max_points: int = 2000) -> dict:
    """Theoretical vs observed quantiles, ready to plot."""
    from scipy import stats as sps

    x = np.sort(stats.vectors.ravel())
    if x.size == 0:
        return {"theoretical": [], "observed": []}
    if x.size > max_points:
        x = x[np.linspace(0, x.size - 1, max_points).astype(int)]
    p = (np.arange(x.size) + 0.5) / x.size
    sd = np.std(stats.vectors.ravel(), ddof=1)
    return {
        "theoretical": (sps.norm.ppf(p) * sd).tolist(),
        "observed": x.tolist(),
    }


def error_vs_radius(
    stats: ResidualStats, principal_point: tuple[float, float], num_bins: int = 12
) -> dict:
    """
    Reprojection error binned by distance from the principal point.

    Growth toward the edge means the distortion model cannot represent the
    lens. This is heteroscedasticity that a single RMS hides completely.
    """
    if stats.points.size == 0:
        return {"bin_centres": [], "rms_px": [], "counts": []}
    cx, cy = principal_point
    r = np.hypot(stats.points[:, 0] - cx, stats.points[:, 1] - cy)
    edges = np.linspace(0.0, float(r.max()) + 1e-9, num_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, num_bins - 1)
    centres, rms, counts = [], [], []
    for b in range(num_bins):
        sel = idx == b
        centres.append(float(0.5 * (edges[b] + edges[b + 1])))
        counts.append(int(sel.sum()))
        rms.append(float(np.sqrt(np.mean(stats.errors[sel] ** 2))) if sel.any() else float("nan"))
    return {"bin_centres": centres, "rms_px": rms, "counts": counts}


def find_outliers(stats: ResidualStats, k: float = 4.0) -> dict:
    """
    Corners whose error exceeds k robust standard deviations.

    The scale is estimated with the median absolute deviation, not the standard
    deviation: outliers inflate the std that is supposed to detect them, which
    is how gross errors routinely survive a 3-sigma rejection.
    """
    e = stats.errors
    if e.size == 0:
        return {"count": 0, "fraction": 0.0, "threshold_px": float("nan"), "indices": []}
    med = float(np.median(e))
    mad = float(np.median(np.abs(e - med)))
    robust_sd = 1.4826 * mad
    thr = med + k * robust_sd
    idx = np.flatnonzero(e > thr)
    per_cam: dict[str, int] = {}
    for cid in np.unique(stats.camera_ids[idx]) if idx.size else []:
        per_cam[str(cid)] = int(np.sum(stats.camera_ids[idx] == cid))
    return {
        "count": int(idx.size),
        "fraction": float(idx.size / e.size),
        "threshold_px": float(thr),
        "robust_sd_px": robust_sd,
        "median_px": med,
        "per_camera": per_cam,
        "indices": idx.tolist()[:500],
        "worst_images": sorted(
            {str(t) for t in np.unique(stats.frame_ids[idx])} if idx.size else set()
        )[:10],
    }


def quiver_field(stats: ResidualStats, camera_id: str, scale: float = 1.0) -> dict:
    """Per-camera residual vector field -- the plot that exposes structure."""
    sel = stats.camera_ids == camera_id
    return {
        "camera": camera_id,
        "x": stats.points[sel, 0].tolist(),
        "y": stats.points[sel, 1].tolist(),
        "u": (stats.vectors[sel, 0] * scale).tolist(),
        "v": (stats.vectors[sel, 1] * scale).tolist(),
        "magnitude": stats.errors[sel].tolist(),
    }


def residual_grid(
    stats: ResidualStats,
    camera_id: str,
    image_size: tuple[int, int],
    grid: tuple[int, int] | None = None,
) -> dict:
    """
    Residuals binned onto a spatial grid over the sensor.

    Two different questions, which the all-arrows quiver answers neither of
    cleanly:

      * `rms` -- how big is the error HERE. Cells with no corners are nan, not
        zero, so an unobserved region reads as "no data" rather than as a
        perfect fit. That distinction is the whole point on a capture that
        never covered the image corners.
      * `mean_u` / `mean_v` -- the MEAN residual vector per cell. Random error
        averages toward zero; whatever survives the averaging is systematic
        bias, i.e. model inadequacy. The raw quiver cannot show this because
        the random part dominates every individual arrow.
    """
    w, h = int(image_size[0]), int(image_size[1])
    if grid is None:
        cols = 20
        rows = max(4, int(round(cols * h / max(w, 1))))
    else:
        rows, cols = int(grid[0]), int(grid[1])

    shape = (rows, cols)
    counts = np.zeros(shape, dtype=int)
    rms = np.full(shape, np.nan)
    mean_u = np.full(shape, np.nan)
    mean_v = np.full(shape, np.nan)

    sel = stats.camera_ids == camera_id
    pts = stats.points[sel]
    if pts.size:
        # clip, not discard: a corner detected exactly on the far edge would
        # otherwise digitize into a column that does not exist.
        cx = np.clip((pts[:, 0] / max(w, 1) * cols).astype(int), 0, cols - 1)
        cy = np.clip((pts[:, 1] / max(h, 1) * rows).astype(int), 0, rows - 1)
        err = stats.errors[sel]
        vec = stats.vectors[sel]
        flat = cy * cols + cx
        counts = np.bincount(flat, minlength=rows * cols).reshape(shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            sq = np.bincount(flat, weights=err**2, minlength=rows * cols).reshape(shape)
            su = np.bincount(flat, weights=vec[:, 0], minlength=rows * cols).reshape(shape)
            sv = np.bincount(flat, weights=vec[:, 1], minlength=rows * cols).reshape(shape)
            occupied = counts > 0
            rms[occupied] = np.sqrt(sq[occupied] / counts[occupied])
            mean_u[occupied] = su[occupied] / counts[occupied]
            mean_v[occupied] = sv[occupied] / counts[occupied]

    return {
        "camera": camera_id,
        "grid": [rows, cols],
        "image_size": [w, h],
        "counts": counts.tolist(),
        "rms": rms.tolist(),
        "mean_u": mean_u.tolist(),
        "mean_v": mean_v.tolist(),
        "empty_cells": int((counts == 0).sum()),
    }
