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
    pixel_noise_std: float | None = 0.3,
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
    # `evaluate` returns None for the Jacobian only when it was not asked for.
    assert J is not None
    m = r.size
    n = J.shape[1]
    dof = m - n
    if dof <= 0:
        raise ValueError(
            f"problem has {m} residuals and {n} free parameters; with "
            f"{dof} degrees of freedom no covariance is meaningful"
        )

    sigma2_residual = float(r @ r) / dof
    # None means the user does not claim to know the noise. There is then
    # nothing to cross-check against, so the comparison is nan rather than a
    # comparison against an invented default -- and nan fails the `> factor`
    # test below, so no mismatch is reported for a claim nobody made.
    if pixel_noise_std is None:
        sigma2_assumed = float("nan")
        ratio = float("nan")
    else:
        sigma2_assumed = float(pixel_noise_std) ** 2
        ratio = max(sigma2_residual, sigma2_assumed) / max(
            min(sigma2_residual, sigma2_assumed), 1e-300
        )
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


_POSE_COMPONENT = {"dtx": "tx", "dty": "ty", "dtz": "tz", "rx": "rx", "ry": "ry", "rz": "rz"}


def split_label(label: str) -> tuple[str, str, str]:
    """
    Break a parameter label into (kind, owner, component).

    Labels are machine keys -- `intr:cam0.4`, `pose:<frame>:board_A.dtx`. The
    frame part is a filename stem and routinely contains dots, so the component
    is split off from the RIGHT and only then is the owner separated.
    """
    kind, _, rest = label.partition(":")
    owner, _, comp = rest.rpartition(".")
    return kind, owner, comp


def frame_aliases(labels: list[str]) -> dict[str, str]:
    """
    Short stand-in names for frames, `f00` upward in sorted order.

    Frame keys are capture filename stems -- 42 characters of timestamp is
    typical -- so printing them in a chart label is not an option. Dropping the
    frame instead is worse: several pose columns then share one name and a
    chart of the worst-correlated pairs reads `tx [board_A] vs tx [board_A]`,
    which names neither of them. The mapping is returned alongside the figures
    so a reader can get back to the actual image.
    """
    frames = set()
    for label in labels:
        kind, owner, _ = split_label(label)
        if kind == "pose":
            frame, _, _ = owner.rpartition(":")
            frames.add(frame)
    return {name: f"f{i:02d}" for i, name in enumerate(sorted(frames))}


def pretty_label(
    label: str,
    camera_param_names: dict[str, list[str]] | None = None,
    aliases: dict[str, str] | None = None,
) -> str:
    """`intr:cam0.4` -> `k1 (cam0)`. Falls back to the raw label if unparseable."""
    kind, owner, comp = split_label(label)
    if kind == "intr":
        names = (camera_param_names or {}).get(owner)
        if names is not None and comp.isdigit() and int(comp) < len(names):
            return f"{names[int(comp)]} ({owner})"
        return f"p{comp} ({owner})"
    if kind == "extr":
        return f"{_POSE_COMPONENT.get(comp, comp)} ({owner})"
    if kind == "pose":
        frame, _, board = owner.rpartition(":")
        tag = (aliases or {}).get(frame, frame)
        return f"{_POSE_COMPONENT.get(comp, comp)} [{board} {tag}]"
    return label


def correlation_blocks(
    cov: CovarianceResult,
    camera_param_names: dict[str, list[str]] | None = None,
    max_pose_columns: int = 240,
    top_k: int = 12,
) -> dict:
    """
    The correlation matrix sliced into the two parts worth looking at.

    The full matrix is square in the number of free parameters -- 141 on a
    small single-camera job, thousands on a real one -- so plotting it whole
    gives an unlabelled grey square. It also buries the finding: on a typical
    capture EVERY one of the strongest correlations is a camera parameter
    against a board pose, not a camera parameter against another camera
    parameter. So this returns both:

      * `camera`  -- the small labelled square block (intrinsics + extrinsics),
        where k1/k2 and fx/cx trade-offs live.
      * `cross`   -- camera parameters against pose parameters, the block the
        `extreme_correlation` warning is actually about.

    Pose columns are capped at `max_pose_columns`, keeping the columns with the
    strongest coupling to any camera parameter, because a legible strip matters
    more than completeness on a 500-frame job.
    """
    C = cov.correlation()
    labels = cov.labels
    cam_idx = [i for i, name in enumerate(labels) if name.startswith(("intr:", "extr:"))]
    pose_idx = [i for i, name in enumerate(labels) if name.startswith("pose:")]

    aliases = frame_aliases(labels)
    pretty = [pretty_label(name, camera_param_names, aliases) for name in labels]

    cross_idx = pose_idx
    if cam_idx and len(pose_idx) > max_pose_columns:
        strength = np.abs(C[np.ix_(cam_idx, pose_idx)]).max(axis=0)
        keep = np.sort(np.argsort(-strength)[:max_pose_columns])
        cross_idx = [pose_idx[k] for k in keep]

    # Poses arrive interleaved frame-by-frame, which would band the strip into
    # dozens of alternating one-pose groups. Ordering by board first makes each
    # board one contiguous, labellable block.
    def _board_of(i: int) -> str:
        _, owner, _ = split_label(labels[i])
        frame, _, board = owner.rpartition(":")
        return board or owner

    cross_idx = sorted(cross_idx, key=lambda i: (_board_of(i), labels[i]))

    # Column group boundaries, so the strip can be ticked by board rather than
    # by 132 unreadable per-parameter labels.
    groups: list[dict] = []
    for pos, i in enumerate(cross_idx):
        _, owner, _ = split_label(labels[i])
        frame, _, board = owner.rpartition(":")
        name = board or owner
        if groups and groups[-1]["name"] == name:
            groups[-1]["end"] = pos + 1
        else:
            groups.append({"name": name, "start": pos, "end": pos + 1})

    return {
        "camera_labels": [pretty[i] for i in cam_idx],
        "camera_matrix": C[np.ix_(cam_idx, cam_idx)].tolist() if cam_idx else [],
        "cross_row_labels": [pretty[i] for i in cam_idx],
        "cross_matrix": C[np.ix_(cam_idx, cross_idx)].tolist() if cam_idx and cross_idx else [],
        "cross_column_groups": groups,
        "num_pose_columns": len(pose_idx),
        "shown_pose_columns": len(cross_idx),
        "frame_aliases": aliases,
        "top_pairs": [
            (
                pretty_label(a, camera_param_names, aliases),
                pretty_label(b, camera_param_names, aliases),
                r,
            )
            for a, b, r in cov.top_correlations(k=top_k, threshold=0.0)
        ],
    }
