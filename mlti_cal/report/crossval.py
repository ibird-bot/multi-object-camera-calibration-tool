"""
Cross-validation and model comparison -- the honest generalisation error.

Training RMS is optimistic by construction: every board pose was fitted to the
very corners the RMS is measured on, so adding parameters can only make it go
down. That is why "our RMS is 0.12 px" means so little on its own.

What is measured here instead
    Hold out whole FRAMES. Re-fit only the held-out frames' board poses, with
    intrinsics and extrinsics FROZEN at their trained values, then measure
    reprojection on those frames. The board poses have to be re-fitted because
    they are nuisance parameters specific to each frame -- you could not know
    them in advance for a new image either. What is being tested is whether
    the CAMERA parameters generalise, which is the only part you keep.

    Holding out individual corners instead would leak badly: the remaining
    corners of the same board in the same image pin the pose almost exactly.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mlti_cal.problem.reprojection import build_problem, extr_key, intr_key, write_back
from mlti_cal.problem.types import CalibrationSystem
from mlti_cal.solvers.base import SolveOptions, per_corner_rms


@dataclass
class FoldResult:
    fold: int
    num_train_frames: int
    num_test_frames: int
    train_rms_px: float
    test_rms_px: float


@dataclass
class CrossValResult:
    folds: list[FoldResult] = field(default_factory=list)
    mean_train_rms: float = float("nan")
    mean_test_rms: float = float("nan")
    optimism: float = float("nan")

    @property
    def optimism_ratio(self) -> float:
        """Held-out RMS divided by training RMS. 1.0 means no optimism at all."""
        # Written to be nan-safe: `mean_train_rms` is nan before any fold has
        # run, and nan fails every comparison, so this returns nan rather than
        # dividing.
        if math.isnan(self.mean_train_rms) or self.mean_train_rms <= 0:
            return float("nan")
        return self.mean_test_rms / self.mean_train_rms

    def verdict(self) -> str:
        """
        What the ratio means, in a sentence.

        The bands are conventions, not measurements, and are stated as such.
        The number itself is what matters; this only stops a reader from having
        to know what a healthy ratio looks like.
        """
        r = self.optimism_ratio
        if math.isnan(r):
            return "training RMS is zero or undefined, so optimism cannot be measured."
        if r < 1.15:
            return (
                f"held-out error is {r:.2f}x the training error -- the camera "
                f"parameters generalise to frames they never saw."
            )
        if r < 1.5:
            return (
                f"held-out error is {r:.2f}x the training error -- mild optimism, "
                f"normal with few frames. More pose variety would tighten it."
            )
        return (
            f"held-out error is {r:.2f}x the training error -- the fit does NOT "
            f"generalise. The training RMS is flattering the calibration: suspect "
            f"too few frames, too little tilt, or too many free distortion terms "
            f"for this data."
        )

    def summary_lines(self) -> list[str]:
        """Column-aligned summary, for a monospace log or the terminal."""
        return [
            f"{'mean train RMS':<16}: {self.mean_train_rms:.4f} px  (fitted on these corners)",
            f"{'mean test RMS':<16}: {self.mean_test_rms:.4f} px  (frames the fit never saw)",
            f"{'optimism':<16}: {self.optimism:+.4f} px  (ratio {self.optimism_ratio:.2f})",
            f"-> {self.verdict()}",
        ]

    def to_dict(self) -> dict:
        return {
            "mean_train_rms_px": self.mean_train_rms,
            "mean_test_rms_px": self.mean_test_rms,
            "optimism_px": self.optimism,
            "optimism_ratio": self.optimism_ratio,
            "folds": [
                {
                    "fold": f.fold,
                    "train_frames": f.num_train_frames,
                    "test_frames": f.num_test_frames,
                    "train_rms_px": f.train_rms_px,
                    "test_rms_px": f.test_rms_px,
                }
                for f in self.folds
            ],
        }


def _subset(system: CalibrationSystem, frames: set[str]) -> CalibrationSystem:
    sub = CalibrationSystem()
    for cid, cam in system.cameras.items():
        sub.cameras[cid] = copy.deepcopy(cam)
    for bid, board in system.boards.items():
        sub.boards[bid] = board
    sub.observations = [o for o in system.observations if o.frame in frames]
    sub.board_poses = {k: v.copy() for k, v in system.board_poses.items() if k[0] in frames}
    return sub


def cross_validate(
    system: CalibrationSystem,
    backend_name: str = "scipy",
    k: int = 4,
    max_iterations: int = 200,
    seed: int = 0,
    on_fold: Callable[[FoldResult], None] | None = None,
) -> CrossValResult:
    """
    k-fold cross-validation over frames.

    `on_fold` is called with each `FoldResult` as it completes. Every fold is a
    full re-solve, so on a real rig this runs for minutes -- a caller that shows
    nothing until the last fold is indistinguishable from one that has hung.
    Plain callable, no Qt: the core stays headless and the GUI adapts it.

    The system is NOT modified. Each fold works on a deep copy, so the values
    on screen are still the ones the user solved for when this returns.
    """
    from mlti_cal.solvers import get_backend

    frames = list(system.frames)
    if len(frames) < k + 1:
        raise ValueError(f"need at least {k + 1} frames for {k}-fold CV, have {len(frames)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frames))
    folds = np.array_split(order, k)

    out = CrossValResult()
    for i, test_idx in enumerate(folds):
        test_frames = {frames[j] for j in test_idx}
        train_frames = set(frames) - test_frames

        train_sys = _subset(system, train_frames)
        train_problem = build_problem(train_sys)
        get_backend(backend_name).solve(train_problem, SolveOptions(max_iterations=max_iterations))
        write_back(train_sys, train_problem)
        train_rms = per_corner_rms(train_problem)

        # Test: same cameras, frozen; only the held-out board poses move.
        test_sys = _subset(system, test_frames)
        for cid, cam in test_sys.cameras.items():
            cam.params = train_sys.cameras[cid].params.copy()
            cam.extrinsic = train_sys.cameras[cid].extrinsic.copy()
        test_problem = build_problem(test_sys, optimize_intrinsics=False, optimize_extrinsics=False)
        for cid in test_sys.cameras:
            test_problem.blocks[intr_key(cid)].constant = True
            test_problem.blocks[extr_key(cid)].constant = True
        get_backend(backend_name).solve(test_problem, SolveOptions(max_iterations=max_iterations))
        test_rms = per_corner_rms(test_problem)

        fold = FoldResult(i, len(train_frames), len(test_frames), train_rms, test_rms)
        out.folds.append(fold)
        if on_fold is not None:
            on_fold(fold)

    out.mean_train_rms = float(np.mean([f.train_rms_px for f in out.folds]))
    out.mean_test_rms = float(np.mean([f.test_rms_px for f in out.folds]))
    out.optimism = out.mean_test_rms - out.mean_train_rms
    return out


def compare_models(
    system: CalibrationSystem,
    variants: dict[str, list[int]] | None = None,
    backend_name: str = "scipy",
    k: int = 3,
) -> dict:
    """
    Compare distortion-model complexity by HELD-OUT error, not training error.

    `variants` maps a label to the intrinsic component indices to hold fixed at
    zero, e.g. {"no_k3": [8], "no_tangential": [6, 7]}. The winner is the one
    with the lowest cross-validated test RMS -- which is frequently NOT the one
    with the most parameters.
    """
    variants = variants or {
        "full": [],
        "no_k3": [8],
        "no_tangential": [6, 7],
        "no_k3_no_tangential": [6, 7, 8],
    }
    results = {}
    for label, fixed in variants.items():
        sub = copy.deepcopy(system)
        for cam in sub.cameras.values():
            for idx in fixed:
                if idx < cam.params.size:
                    cam.params[idx] = 0.0
        try:
            cv = _cross_validate_fixed(sub, fixed, backend_name=backend_name, k=k)
            results[label] = {
                "fixed_components": fixed,
                "num_free_intrinsics": int(
                    next(iter(sub.cameras.values())).params.size - len(fixed)
                ),
                **cv.to_dict(),
            }
        except Exception as exc:  # pragma: no cover - variant may be infeasible
            results[label] = {"error": str(exc), "fixed_components": fixed}
    ranked = sorted(
        (k_ for k_, v in results.items() if "mean_test_rms_px" in v),
        key=lambda k_: results[k_]["mean_test_rms_px"],
    )
    return {"variants": results, "ranking": ranked, "best": ranked[0] if ranked else None}


def _cross_validate_fixed(
    system: CalibrationSystem, fixed: list[int], backend_name: str, k: int
) -> CrossValResult:
    from mlti_cal.solvers import get_backend

    frames = list(system.frames)
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(frames)), k)
    fixed_map = {cid: list(fixed) for cid in system.cameras}

    out = CrossValResult()
    for i, test_idx in enumerate(folds):
        test_frames = {frames[j] for j in test_idx}
        train_frames = set(frames) - test_frames
        train_sys = _subset(system, train_frames)
        tp = build_problem(train_sys, fixed_intrinsic_components=fixed_map)
        get_backend(backend_name).solve(tp, SolveOptions(max_iterations=200))
        write_back(train_sys, tp)

        test_sys = _subset(system, test_frames)
        for cid, cam in test_sys.cameras.items():
            cam.params = train_sys.cameras[cid].params.copy()
            cam.extrinsic = train_sys.cameras[cid].extrinsic.copy()
        ep = build_problem(test_sys, optimize_intrinsics=False, optimize_extrinsics=False)
        for cid in test_sys.cameras:
            ep.blocks[intr_key(cid)].constant = True
            ep.blocks[extr_key(cid)].constant = True
        get_backend(backend_name).solve(ep, SolveOptions(max_iterations=200))
        out.folds.append(
            FoldResult(
                i, len(train_frames), len(test_frames), per_corner_rms(tp), per_corner_rms(ep)
            )
        )

    out.mean_train_rms = float(np.mean([f.train_rms_px for f in out.folds]))
    out.mean_test_rms = float(np.mean([f.test_rms_px for f in out.folds]))
    out.optimism = out.mean_test_rms - out.mean_train_rms
    return out
