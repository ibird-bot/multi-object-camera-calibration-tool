"""
Robust loss functions, applied as IRLS weights inside the core.

Why here and not in each backend
    A robust loss must act on the 2D residual of ONE corner. Ceres applies its
    loss to a whole residual block, and scipy applies its loss to each scalar
    element -- neither matches "per corner" unless the problem is reshaped to
    suit that backend specifically. Doing it once in the core means every
    backend optimises the identical objective, which is a precondition for the
    cross-backend agreement claim in PLAN.md.

Method
    For a corner with squared whitened residual s = ||r||^2, the weight is
    w = sqrt(rho'(s)), applied to both components of r and to the matching
    Jacobian rows. Within one linearisation w is treated as constant -- this is
    standard square-root IRLS.

Honest limitation
    This is NOT identical to Ceres' Triggs-corrected loss, which additionally
    modifies the Gauss-Newton Hessian. Near convergence with few outliers the
    difference is small, but it is a real difference and is reported as such
    rather than papered over. Backends therefore receive already-weighted
    residuals and are given no loss of their own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    name: str = "abstract"

    @abstractmethod
    def rho_prime(self, s: np.ndarray) -> np.ndarray:
        """First derivative of rho w.r.t. the squared residual s."""

    def weights(self, s: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(self.rho_prime(np.asarray(s, dtype=float)), 0.0))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}()"


class TrivialLoss(Loss):
    """Plain least squares. The default -- robustness is opt-in, not silent."""

    name = "trivial"

    def rho_prime(self, s: np.ndarray) -> np.ndarray:
        return np.ones_like(s)


class HuberLoss(Loss):
    """Quadratic within `delta` pixels, linear beyond. delta is in PIXELS."""

    name = "huber"

    def __init__(self, delta: float = 2.0):
        if delta <= 0:
            raise ValueError("huber delta must be > 0")
        self.delta = float(delta)

    def rho_prime(self, s: np.ndarray) -> np.ndarray:
        b = self.delta * self.delta
        return np.where(s <= b, 1.0, np.sqrt(b / np.maximum(s, 1e-300)))


class CauchyLoss(Loss):
    """Heavier downweighting than Huber; outliers are nearly ignored."""

    name = "cauchy"

    def __init__(self, scale: float = 2.0):
        if scale <= 0:
            raise ValueError("cauchy scale must be > 0")
        self.scale = float(scale)

    def rho_prime(self, s: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + s / (self.scale * self.scale))


class SoftLOneLoss(Loss):
    """Smooth approximation to L1."""

    name = "soft_l1"

    def __init__(self, scale: float = 2.0):
        if scale <= 0:
            raise ValueError("soft_l1 scale must be > 0")
        self.scale = float(scale)

    def rho_prime(self, s: np.ndarray) -> np.ndarray:
        b = self.scale * self.scale
        return 1.0 / np.sqrt(1.0 + s / b)


_LOSSES = {
    "trivial": TrivialLoss,
    "huber": HuberLoss,
    "cauchy": CauchyLoss,
    "soft_l1": SoftLOneLoss,
}


def make_loss(name: str | None, scale: float = 2.0) -> Loss:
    if name is None or name == "trivial":
        return TrivialLoss()
    if name not in _LOSSES:
        raise KeyError(f"unknown loss {name!r}; known: {sorted(_LOSSES)}")
    return _LOSSES[name](scale)


def available_losses() -> list[str]:
    return sorted(_LOSSES)
