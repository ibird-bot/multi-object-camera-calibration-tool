"""
Backend-agnostic optimisation problem: parameter blocks, residual blocks, and
sparse Jacobian assembly.

The core owns the maths. A backend adapter's only job is to translate this
structure into its own API and hand the answer back -- no adapter may compute
derivatives of its own, because the point of the tool is that the SAME problem
goes to every solver.

Free / fixed / deactivated
    Each block carries a per-TANGENT-component boolean mask. `constant=True` is
    shorthand for an all-false mask. This is how the GUI's Free/Fixed/
    Deactivated tri-state is expressed all the way down to the linear algebra:
    fixed components simply never get a column.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from mlti_cal.models.manifolds import POSE_DIM, pose_plus

EUCLIDEAN = "euclidean"
POSE = "pose"


@dataclass
class ParameterBlock:
    """One optimisable quantity."""

    key: str
    value: np.ndarray
    kind: str = EUCLIDEAN
    constant: bool = False
    free_mask: np.ndarray | None = None  # (tangent_size,) bool; None => all free

    def __post_init__(self) -> None:
        self.value = np.asarray(self.value, dtype=float).ravel().copy()
        if self.kind == POSE and self.value.size != POSE_DIM:
            raise ValueError(f"{self.key}: pose block must store {POSE_DIM} values")
        if self.free_mask is None:
            self.free_mask = np.ones(self.tangent_size, dtype=bool)
        else:
            self.free_mask = np.asarray(self.free_mask, dtype=bool).ravel().copy()
            if self.free_mask.size != self.tangent_size:
                raise ValueError(
                    f"{self.key}: free_mask has {self.free_mask.size} entries, "
                    f"tangent size is {self.tangent_size}"
                )

    @property
    def tangent_size(self) -> int:
        return 6 if self.kind == POSE else self.value.size

    @property
    def num_free(self) -> int:
        return 0 if self.constant else int(self.free_mask.sum())

    @property
    def free_indices(self) -> np.ndarray:
        if self.constant:
            return np.zeros(0, dtype=int)
        return np.flatnonzero(self.free_mask)

    def plus(self, delta_full: np.ndarray) -> None:
        """Retract this block in place by a FULL-tangent-size delta."""
        if self.kind == POSE:
            self.value = pose_plus(self.value, delta_full)
        else:
            self.value = self.value + delta_full

    def set_free(self, names_or_idx=None) -> None:
        """Mark all components free (default) or only the given tangent indices."""
        self.constant = False
        if names_or_idx is None:
            self.free_mask[:] = True
        else:
            self.free_mask[:] = False
            self.free_mask[np.asarray(names_or_idx, dtype=int)] = True


class ResidualBlock(ABC):
    """A group of residuals depending on a fixed list of parameter blocks."""

    block_keys: tuple[str, ...] = ()
    dim: int = 0
    #: free-text tag used by the report engine to group residuals
    tag: str = ""

    @abstractmethod
    def evaluate(
        self, values: list[np.ndarray], with_jacobians: bool = True
    ) -> tuple[np.ndarray, list[np.ndarray] | None]:
        """
        Returns (residual(dim,), [J_k (dim, tangent_size_k)] or None).

        Jacobians are TANGENT-space, matching this repo's retraction.
        """


@dataclass
class Problem:
    """A set of parameter blocks plus the residuals that couple them."""

    blocks: dict[str, ParameterBlock] = field(default_factory=dict)
    residuals: list[ResidualBlock] = field(default_factory=list)

    # -- construction ------------------------------------------------------
    def add_block(self, block: ParameterBlock) -> ParameterBlock:
        if block.key in self.blocks:
            raise ValueError(f"duplicate parameter block {block.key!r}")
        self.blocks[block.key] = block
        return block

    def add_residual(self, residual: ResidualBlock) -> ResidualBlock:
        for k in residual.block_keys:
            if k not in self.blocks:
                raise KeyError(f"residual references unknown block {k!r}")
        self.residuals.append(residual)
        return residual

    # -- layout ------------------------------------------------------------
    @property
    def num_residuals(self) -> int:
        return int(sum(r.dim for r in self.residuals))

    def column_layout(self) -> tuple[dict[str, int], int]:
        """
        Map block key -> first global column, plus the total free column count.

        Blocks with no free components are absent from the mapping. Order is
        insertion order, which is what makes an ordering hint meaningful to
        Ceres later.
        """
        offsets: dict[str, int] = {}
        col = 0
        for key, blk in self.blocks.items():
            n = blk.num_free
            if n:
                offsets[key] = col
                col += n
        return offsets, col

    @property
    def num_free_params(self) -> int:
        return self.column_layout()[1]

    def parameter_labels(self) -> list[str]:
        """Human-readable name for every free column -- used in reports."""
        labels: list[str] = []
        for key, blk in self.blocks.items():
            if not blk.num_free:
                continue
            comp = (
                ["dtx", "dty", "dtz", "rx", "ry", "rz"]
                if blk.kind == POSE
                else [str(i) for i in range(blk.tangent_size)]
            )
            labels.extend(f"{key}.{comp[i]}" for i in blk.free_indices)
        return labels

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        with_jacobian: bool = True,
        tangent_transform: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, sp.csr_matrix | None]:
        """
        Residual vector and sparse Jacobian over FREE columns only.

        The Jacobian is assembled in COO triplets and converted once; building
        it row-block by row-block into a dense array would be O(m*n) memory and
        is the usual reason naive bundle adjusters fall over at scale.

        Args:
            tangent_transform: optional block key -> (tangent, tangent) matrix
                M, applied as `J_block @ M` BEFORE free-column selection. This
                is how a backend that parameterises by an increment from a
                fixed base state (scipy, which has no manifolds) supplies the
                SO(3) right-Jacobian correction. Applying it before selection
                is what makes it correct when only some rotation components are
                free -- transforming afterwards would fold a fixed component
                into a free one.
        """
        offsets, ncols = self.column_layout()
        r_all = np.empty(self.num_residuals)
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        row0 = 0
        for res in self.residuals:
            values = [self.blocks[k].value for k in res.block_keys]
            r, jacs = res.evaluate(values, with_jacobians=with_jacobian)
            r_all[row0 : row0 + res.dim] = r
            if with_jacobian and jacs is not None:
                for key, J in zip(res.block_keys, jacs, strict=True):
                    blk = self.blocks[key]
                    if not blk.num_free or J is None:
                        continue
                    if tangent_transform is not None and key in tangent_transform:
                        J = J @ tangent_transform[key]
                    sel = blk.free_indices
                    Jf = J[:, sel]  # (dim, n_free_k)
                    nz = np.flatnonzero(Jf)
                    if nz.size == 0:
                        continue
                    rr, cc = np.unravel_index(nz, Jf.shape)
                    rows.append(rr + row0)
                    cols.append(cc + offsets[key])
                    vals.append(Jf.ravel()[nz])
            row0 += res.dim

        if not with_jacobian:
            return r_all, None
        if rows:
            J = sp.coo_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.num_residuals, ncols),
            ).tocsr()
        else:
            J = sp.csr_matrix((self.num_residuals, ncols))
        return r_all, J

    def residuals_only(self) -> np.ndarray:
        return self.evaluate(with_jacobian=False)[0]

    # -- state -------------------------------------------------------------
    def apply_delta(self, delta_free: np.ndarray) -> None:
        """Retract every block by its slice of a free-column delta vector."""
        offsets, ncols = self.column_layout()
        if delta_free.size != ncols:
            raise ValueError(f"delta has {delta_free.size} entries, expected {ncols}")
        for key, blk in self.blocks.items():
            if key not in offsets:
                continue
            sel = blk.free_indices
            full = np.zeros(blk.tangent_size)
            full[sel] = delta_free[offsets[key] : offsets[key] + sel.size]
            blk.plus(full)

    def scatter(self, delta_free: np.ndarray) -> dict[str, np.ndarray]:
        """Free-column vector -> per-block FULL tangent vectors (zeros for fixed)."""
        offsets, ncols = self.column_layout()
        if delta_free.size != ncols:
            raise ValueError(f"delta has {delta_free.size} entries, expected {ncols}")
        out: dict[str, np.ndarray] = {}
        for key, blk in self.blocks.items():
            full = np.zeros(blk.tangent_size)
            if key in offsets:
                sel = blk.free_indices
                full[sel] = delta_free[offsets[key] : offsets[key] + sel.size]
            out[key] = full
        return out

    def set_from_base(
        self, base_state: dict[str, np.ndarray], delta_free: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Set every block to `base (+) delta`, NOT `current (+) delta`.

        A backend whose optimisation variable is an increment from a fixed
        linearisation point must re-derive the state from that base on every
        evaluation; retracting incrementally instead would make the residual
        depend on the order the optimiser happened to probe points in.

        Returns the scattered per-block tangents, which the caller needs to
        build the right-Jacobian correction.
        """
        deltas = self.scatter(delta_free)
        for key, blk in self.blocks.items():
            blk.value = np.asarray(base_state[key], dtype=float).copy()
            if np.any(deltas[key]):
                blk.plus(deltas[key])
        return deltas

    def get_state(self) -> dict[str, np.ndarray]:
        return {k: b.value.copy() for k, b in self.blocks.items()}

    def set_state(self, state: dict[str, np.ndarray]) -> None:
        for k, v in state.items():
            self.blocks[k].value = np.asarray(v, dtype=float).copy()

    # -- diagnostics -------------------------------------------------------
    def cost(self) -> float:
        """0.5 * ||r||^2, the Ceres convention."""
        r = self.residuals_only()
        return 0.5 * float(r @ r)

    def rms(self) -> float:
        """
        RMS of the residual vector, in whitened units.

        For unit sigma and 2D reprojection residuals this is RMS pixels per
        coordinate. `report.residuals` computes the per-corner Euclidean RMS
        that people actually quote; the two differ by sqrt(2) and conflating
        them is a classic way to report an error that looks better than it is.
        """
        r = self.residuals_only()
        return float(np.sqrt(r @ r / max(r.size, 1)))
