"""
Scoped cap on the BLAS/LAPACK thread pools behind numpy and scipy.

Why this exists
    Dense LAPACK calls on a calibration Jacobian ran about 10x slower with
    OpenBLAS's default threading than with a few threads. Measured on an
    i7-12700H (hybrid performance/efficiency cores, 20 logical processors),
    OpenBLAS 0.3.33, rig_4cam Jacobian 13338 x 573: one SVD took 17.3 s at the
    default 20 threads and 1.9 / 1.4 / 1.7 s at 1 / 4 / 8 threads. A dense
    factorisation of a matrix this shape cannot use 20 threads, and on a hybrid
    CPU the threads that land on efficiency cores make every other thread wait.

    numpy 2.x and scipy each bundle their own OpenBLAS; `threadpoolctl` finds
    and caps both. If it is not installed this does nothing, so the cap is a
    speed-up and never a requirement.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def blas_threads(limit: int | None) -> Iterator[None]:
    """Cap BLAS threads to `limit` inside the block; None or < 1 leaves them alone."""
    if limit is None or limit < 1:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - optional speed-up only
        yield
        return
    with threadpool_limits(limits=int(limit), user_api="blas"):
        yield
