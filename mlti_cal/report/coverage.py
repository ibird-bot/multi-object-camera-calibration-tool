"""
Observation coverage and geometric conditioning.

Uncertainty is worst where you have no data, and on a normal calibration that
means the image corners -- which is also where distortion is largest and where
people most often use the result. A coverage map answers "should I trust the
edge of this image?" before the uncertainty map has to.

Board tilt diversity gets the same treatment. If every board is fronto-parallel
the focal length and the board distance are nearly indistinguishable, and no
amount of data fixes it. That is a property of the capture, not the solver, so
it is worth telling the user plainly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlti_cal.models.manifolds import quat_to_matrix
from mlti_cal.problem.types import CalibrationSystem


@dataclass
class CoverageStats:
    camera: str
    histogram: np.ndarray  # (rows, cols) observation counts
    occupied_fraction: float
    corner_occupancy: dict[str, int]
    num_points: int
    grid: tuple[int, int]
    image_size: tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "camera": self.camera,
            "num_points": self.num_points,
            "occupied_fraction": self.occupied_fraction,
            "grid": list(self.grid),
            "corner_occupancy": self.corner_occupancy,
            "empty_cells": int((self.histogram == 0).sum()),
        }


@dataclass
class TiltStats:
    incidence_deg: np.ndarray
    mean_deg: float = 0.0
    max_deg: float = 0.0
    std_deg: float = 0.0
    fraction_below_10deg: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mean_incidence_deg": self.mean_deg,
            "max_incidence_deg": self.max_deg,
            "std_incidence_deg": self.std_deg,
            "fraction_nearly_frontoparallel": self.fraction_below_10deg,
            "n": int(self.incidence_deg.size),
        }


@dataclass
class CoverageReport:
    per_camera: dict[str, CoverageStats] = field(default_factory=dict)
    tilt: TiltStats | None = None

    def to_dict(self) -> dict:
        return {
            "per_camera": {k: v.to_dict() for k, v in self.per_camera.items()},
            "tilt": self.tilt.to_dict() if self.tilt else None,
        }


def compute_coverage(
    system: CalibrationSystem, grid: tuple[int, int] = (8, 10)
) -> dict[str, CoverageStats]:
    """2D occupancy histogram of observed corners, per camera."""
    rows, cols = grid
    out: dict[str, CoverageStats] = {}
    for cid, cam in system.cameras.items():
        w, h = cam.image_size
        hist = np.zeros((rows, cols), dtype=int)
        total = 0
        for obs in system.observations:
            if obs.camera != cid:
                continue
            p = obs.image_points
            c = np.clip((p[:, 0] / w * cols).astype(int), 0, cols - 1)
            r = np.clip((p[:, 1] / h * rows).astype(int), 0, rows - 1)
            np.add.at(hist, (r, c), 1)
            total += p.shape[0]
        corners = {
            "top_left": int(hist[0, 0]),
            "top_right": int(hist[0, -1]),
            "bottom_left": int(hist[-1, 0]),
            "bottom_right": int(hist[-1, -1]),
        }
        out[cid] = CoverageStats(
            camera=cid,
            histogram=hist,
            occupied_fraction=float((hist > 0).mean()),
            corner_occupancy=corners,
            num_points=total,
            grid=grid,
            image_size=(w, h),
        )
    return out


def compute_tilt(system: CalibrationSystem) -> TiltStats:
    """
    Angle between each board's normal and the reference camera's optical axis.

    0 degrees means fronto-parallel, which is the degenerate case for
    separating focal length from distance.
    """
    angles = []
    axis = np.array([0.0, 0.0, 1.0])
    for (_frame, _board), pose in system.board_poses.items():
        R = quat_to_matrix(np.asarray(pose)[3:7])
        normal = R @ np.array([0.0, 0.0, 1.0])
        c = abs(float(normal @ axis)) / (np.linalg.norm(normal) or 1.0)
        angles.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    a = np.asarray(angles, dtype=float)
    if a.size == 0:
        return TiltStats(a)
    return TiltStats(
        incidence_deg=a,
        mean_deg=float(a.mean()),
        max_deg=float(a.max()),
        std_deg=float(a.std()),
        fraction_below_10deg=float(np.mean(a < 10.0)),
    )


def compute_coverage_report(system: CalibrationSystem, grid=(8, 10)) -> CoverageReport:
    return CoverageReport(per_camera=compute_coverage(system, grid), tilt=compute_tilt(system))
