"""Solver backends. Importing this package registers every available adapter."""

from mlti_cal.solvers.base import (  # noqa: F401
    IterationRecord,
    SolveOptions,
    SolveResult,
    SolverBackend,
    available_backends,
    backend_status,
    get_backend,
    register_backend,
)

# Imported for side effect: each module registers itself on import. Failures
# are swallowed per-backend so one missing optional dependency cannot make the
# package unimportable -- `backend_status()` reports what is actually usable.
from mlti_cal.solvers import scipy_backend  # noqa: F401,E402

try:  # pragma: no cover - depends on environment
    from mlti_cal.solvers import ceres_backend  # noqa: F401
except ImportError:  # pragma: no cover
    pass

try:  # pragma: no cover - depends on environment
    from mlti_cal.solvers import gtsam_backend  # noqa: F401
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "IterationRecord",
    "SolveOptions",
    "SolveResult",
    "SolverBackend",
    "available_backends",
    "backend_status",
    "get_backend",
    "register_backend",
]
