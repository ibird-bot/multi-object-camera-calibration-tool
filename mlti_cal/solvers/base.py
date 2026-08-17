"""
Backend-agnostic solver interface.

Every adapter receives the SAME `Problem` and returns the SAME `SolveResult`
shape. Adapters translate; they never differentiate. If an adapter needed its
own derivatives, the "which solver is better" comparison would be comparing
two different problems, which is the failure this design exists to prevent.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mlti_cal.problem.graph import Problem


@dataclass
class SolveOptions:
    """Options common to all backends; `extra` carries backend-specific keys."""

    max_iterations: int = 100
    function_tolerance: float = 1e-10
    gradient_tolerance: float = 1e-12
    parameter_tolerance: float = 1e-10
    num_threads: int = 1
    verbose: bool = False
    extra: dict = field(default_factory=dict)
    #: Called with each `IterationRecord` as the solve runs, for a live view.
    #: Setting it also asks the backend to MEASURE the per-corner RMS at every
    #: step, which costs an extra projection pass -- speed traded for visibility,
    #: and only when somebody is actually watching. Headless solves leave it None
    #: and pay nothing.
    on_iteration: Callable[[IterationRecord], None] | None = None

    def get(self, key: str, default=None):
        return self.extra.get(key, default)


@dataclass
class IterationRecord:
    iteration: int
    cost: float
    gradient_norm: float = float("nan")
    step_norm: float = float("nan")
    #: Per-corner reprojection RMS in PIXELS, measured -- never derived from
    #: `cost`. The two agree only under a trivial loss; under a robust one the
    #: cost is weighted and converting it would overstate the fit by exactly
    #: the amount the loss is discounting.
    rms_px: float = float("nan")


@dataclass
class SolveResult:
    """Outcome of one solve. `rms_px` fields are per-corner Euclidean pixels."""

    backend: str
    success: bool
    message: str
    initial_cost: float
    final_cost: float
    initial_rms_px: float
    final_rms_px: float
    iterations: int
    time_seconds: float
    num_residuals: int
    num_free_params: int
    options: dict = field(default_factory=dict)
    history: list[IterationRecord] = field(default_factory=list)
    behind_camera_points: int = 0

    def summary(self) -> str:
        return (
            f"[{self.backend}] {'OK' if self.success else 'FAILED'} in "
            f"{self.iterations} it / {self.time_seconds:.2f}s | "
            f"RMS {self.initial_rms_px:.4f} -> {self.final_rms_px:.4f} px | "
            f"cost {self.initial_cost:.6g} -> {self.final_cost:.6g} | "
            f"{self.num_residuals} residuals, {self.num_free_params} free params"
            f"{'' if not self.message else ' | ' + self.message}"
        )

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "success": self.success,
            "message": self.message,
            "initial_cost": self.initial_cost,
            "final_cost": self.final_cost,
            "initial_rms_px": self.initial_rms_px,
            "final_rms_px": self.final_rms_px,
            "iterations": self.iterations,
            "time_seconds": self.time_seconds,
            "num_residuals": self.num_residuals,
            "num_free_params": self.num_free_params,
            "options": self.options,
            "behind_camera_points": self.behind_camera_points,
            "history": [
                {"iteration": h.iteration, "cost": h.cost, "gradient_norm": h.gradient_norm}
                for h in self.history
            ],
        }


def per_corner_rms(problem: Problem) -> float:
    """
    RMS of the per-corner Euclidean reprojection error, in pixels.

    This is the number people quote, and it is deliberately NOT the same as
    `Problem.rms()` (which is per-coordinate and whitened). The two differ by
    about sqrt(2); quoting the smaller one as "reprojection error" is a common
    way to look more accurate than you are.
    """
    from mlti_cal.problem.reprojection import ReprojectionResidual

    errs = []
    for res in problem.residuals:
        if isinstance(res, ReprojectionResidual):
            values = [problem.blocks[k].value for k in res.block_keys]
            errs.append(res.reprojection_errors(values))
    if not errs:
        return float("nan")
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e**2)))


def count_behind_camera(problem: Problem) -> int:
    from mlti_cal.problem.reprojection import ReprojectionResidual

    return int(
        sum(r.last_behind_camera for r in problem.residuals if isinstance(r, ReprojectionResidual))
    )


class SolverBackend(ABC):
    """Base class for solver adapters."""

    name: str = "abstract"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """(usable here?, human-readable reason if not)."""
        return True, ""

    @abstractmethod
    def solve(self, problem: Problem, options: SolveOptions | None = None) -> SolveResult: ...

    # -- helpers for subclasses -------------------------------------------
    @staticmethod
    def _start(problem: Problem) -> tuple[float, float, float]:
        r = problem.residuals_only()
        return time.perf_counter(), 0.5 * float(r @ r), per_corner_rms(problem)


_BACKENDS: dict[str, type[SolverBackend]] = {}


def register_backend(cls: type[SolverBackend]) -> type[SolverBackend]:
    _BACKENDS[cls.name] = cls
    return cls


def get_backend(name: str) -> SolverBackend:
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(_BACKENDS)}")
    return _BACKENDS[name]()


def available_backends(only_usable: bool = False) -> list[str]:
    names = sorted(_BACKENDS)
    if not only_usable:
        return names
    return [n for n in names if _BACKENDS[n].is_available()[0]]


def backend_status() -> dict[str, tuple[bool, str]]:
    """Which backends can actually run in this environment, and why not."""
    return {n: c.is_available() for n, c in sorted(_BACKENDS.items())}
