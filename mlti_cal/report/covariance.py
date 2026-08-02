"""
Parameter covariance, with the rank decision made in the open.

Sigma_theta = sigma^2 * (J^T J)^+   on the observable subspace.

Two things here are deliberately not hidden:

1. THE RANK DECISION. `np.linalg.pinv` with its default rcond will quietly
   absorb near-null directions and hand back a confident-looking matrix. For a
   multi-camera / multi-board problem J^T J is routinely near-singular, so that
   default is the difference between "your calibration is weak in this
   direction" and a plausible number that is simply wrong. The threshold is
   explicit, configurable, and every discarded direction is reported together
   with the parameters that dominate it.

2. THE NOISE MODEL. sigma^2 is computed BOTH ways --
     * residual-derived:  r^T r / (m - n)
     * assumed:           the pixel-noise floor the user states
   The covariance uses the residual-derived value by default, but when the two
   disagree by more than `disagreement_factor` the result is flagged. That
   disagreement is itself the diagnostic: residual-derived >> assumed means the
   model cannot represent the data (systematic error masquerading as noise);
   residual-derived << assumed means the model is over-parameterised and is
   fitting the noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mlti_cal.problem.graph import Problem


@dataclass
class WeakDirection:
    """One (near-)unobservable combination of parameters."""

    index: int
    singular_value: float
    ratio_to_largest: float
    discarded: bool
    top_parameters: list[tuple[str, float]] = field(default_factory=list)

    def describe(self) -> str:
        parts = ", ".join(f"{n} ({w:+.2f})" for n, w in self.top_parameters)
        state = "DISCARDED" if self.discarded else "weak"
        inv_ratio = 1 / max(self.ratio_to_largest, 1e-300)
        return f"[{state}] s={self.singular_value:.3e} (1/{inv_ratio:.1e}): {parts}"


@dataclass
class CovarianceResult:
    covariance: np.ndarray
    std_errors: np.ndarray
    labels: list[str]
    sigma2_residual: float
    sigma2_assumed: float
    sigma2_used: float
    sigma_source: str
    noise_disagreement: float
    noise_disagreement_flagged: bool
    singular_values: np.ndarray
    rank: int
    num_free_params: int
    rank_threshold: float
    condition_number: float
    weak_directions: list[WeakDirection]
    degrees_of_freedom: int

    @property
    def rank_deficient(self) -> bool:
        return self.rank < self.num_free_params

    def std_error_of(self, label: str) -> float:
        return float(self.std_errors[self.labels.index(label)])

    def correlation(self) -> np.ndarray:
        d = np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))
        d[d == 0] = 1.0
        return self.covariance / np.outer(d, d)

    def top_correlations(self, k: int = 10, threshold: float = 0.9) -> list[tuple]:
        """Most strongly correlated parameter pairs -- the usual weak-geometry tell."""
        C = self.correlation()
        iu = np.triu_indices_from(C, k=1)
        vals = C[iu]
        order = np.argsort(-np.abs(vals))[:k]
        out = []
        for o in order:
            i, j = iu[0][o], iu[1][o]
            if abs(vals[o]) < threshold:
                break
            out.append((self.labels[i], self.labels[j], float(vals[o])))
        return out

    def to_dict(self) -> dict:
        return {
            "sigma2_residual": self.sigma2_residual,
            "sigma2_assumed": self.sigma2_assumed,
            "sigma2_used": self.sigma2_used,
            "sigma_source": self.sigma_source,
            "noise_disagreement": self.noise_disagreement,
            "noise_disagreement_flagged": self.noise_disagreement_flagged,
            "rank": self.rank,
            "num_free_params": self.num_free_params,
            "rank_deficient": self.rank_deficient,
            "rank_threshold": self.rank_threshold,
            "condition_number": self.condition_number,
            "degrees_of_freedom": self.degrees_of_freedom,
            "std_errors": {k: float(v) for k, v in zip(self.labels, self.std_errors, strict=True)},
            "weak_directions": [
                {
                    "index": w.index,
                    "singular_value": w.singular_value,
                    "ratio_to_largest": w.ratio_to_largest,
                    "discarded": w.discarded,
                    "top_parameters": w.top_parameters,
                }
                for w in self.weak_directions
            ],
            "top_correlations": self.top_correlations(),
        }


def compute_covariance(
    problem: Problem,
    pixel_noise_std: float = 0.3,
    rank_threshold: float = 1e-8,
    disagreement_factor: float = 2.0,
    sigma_source: str = "residual",
    weak_direction_report: int = 6,
    top_parameters: int = 4,
) -> CovarianceResult:
    """
    Args:
        pixel_noise_std: the assumed per-coordinate pixel noise floor. Used for
            the cross-check, and for the covariance itself when
            `sigma_source="assumed"`.
        rank_threshold: singular values below `rank_threshold * s_max` are
            treated as null directions and excluded from the pseudo-inverse.
        sigma_source: "residual" (default) or "assumed".
    """
    r, J = problem.evaluate(with_jacobian=True)
    m = r.size
    n = J.shape[1]
    dof = m - n
    if dof <= 0:
        raise ValueError(
            f"problem has {m} residuals and {n} free parameters; with "
            f"{dof} degrees of freedom no covariance is meaningful"
        )

    sigma2_residual = float(r @ r) / dof
    sigma2_assumed = float(pixel_noise_std) ** 2
    ratio = max(sigma2_residual, sigma2_assumed) / max(min(sigma2_residual, sigma2_assumed), 1e-300)
    sigma2_used = sigma2_residual if sigma_source == "residual" else sigma2_assumed

    # SVD of J rather than eigendecomposition of J^T J: squaring the matrix
    # squares the condition number too, and this problem is already
    # ill-conditioned enough that it costs real digits.
    Jd = J.toarray() if hasattr(J, "toarray") else np.asarray(J)
    _, s, Vt = np.linalg.svd(Jd, full_matrices=False)
    s_max = float(s[0]) if s.size else 0.0
    cutoff = rank_threshold * s_max
    keep = s > cutoff
    rank = int(keep.sum())

    s_inv2 = np.zeros_like(s)
    s_inv2[keep] = 1.0 / (s[keep] ** 2)
    cov = (Vt.T * s_inv2) @ Vt * sigma2_used

    labels = problem.parameter_labels()
    weak: list[WeakDirection] = []
    # Report the smallest directions whether or not they were discarded: a
    # direction just above the cutoff is exactly as much of a warning.
    for idx in range(len(s) - 1, max(len(s) - 1 - weak_direction_report, -1), -1):
        v = Vt[idx]
        order = np.argsort(-np.abs(v))[:top_parameters]
        weak.append(
            WeakDirection(
                index=int(idx),
                singular_value=float(s[idx]),
                ratio_to_largest=float(s[idx] / s_max) if s_max else 0.0,
                discarded=not bool(keep[idx]),
                top_parameters=[(labels[i], float(v[i])) for i in order],
            )
        )

    return CovarianceResult(
        covariance=cov,
        std_errors=np.sqrt(np.clip(np.diag(cov), 0.0, None)),
        labels=labels,
        sigma2_residual=sigma2_residual,
        sigma2_assumed=sigma2_assumed,
        sigma2_used=sigma2_used,
        sigma_source=sigma_source,
        noise_disagreement=float(ratio),
        noise_disagreement_flagged=bool(ratio > disagreement_factor),
        singular_values=s,
        rank=rank,
        num_free_params=n,
        rank_threshold=rank_threshold,
        condition_number=float(s_max / s[-1]) if s.size and s[-1] > 0 else float("inf"),
        weak_directions=weak,
        degrees_of_freedom=dof,
    )


def marginal_covariance(
    cov: CovarianceResult, prefix: str
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Sub-block of the covariance for every parameter whose label starts with
    `prefix` (e.g. "intr:cam0"), plus the column indices.

    Marginalising is just selection here -- the joint inverse already accounts
    for every nuisance parameter, so no Schur complement is needed.
    """
    idx = np.array([i for i, name in enumerate(cov.labels) if name.startswith(prefix)])
    if idx.size == 0:
        raise KeyError(f"no parameters match prefix {prefix!r}")
    return cov.covariance[np.ix_(idx, idx)], [cov.labels[i] for i in idx], idx
