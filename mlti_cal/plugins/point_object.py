"""
A calibration object described by its points: `id, x, y, z` in metres.

This is the object half of the plug-in contract. It works for any target whose
points carry an identity -- coded ellipses, retro-reflective targets, a 3D jig
measured on a CMM -- because nothing here knows how the points look, only
where they are.

File format (CSV or whitespace separated, `#` starts a comment)::

    id, x, y, z            # header optional
    7,   0.000, 0.000, 0.0
    103, 0.120, 0.000, 0.0
    ...

- `id` is the target's identity as the detector reports it: a whole number,
  unique within the object. Ids need not be contiguous or sorted.
- x, y, z are in METRES in the object's own right-handed frame.
- A flat object must have z = 0 for every point. Points that are coplanar but
  not on z = 0 are refused: re-expressing them in another frame would silently
  change what the reported object pose means.
- Three optional columns `sx, sy, sz` (per-point standard deviation, metres)
  are read and kept, for a later version that refines point positions. They do
  not change the calibration today.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

#: Below this ratio of the smallest to the largest spread, the points are one plane.
_PLANAR_RATIO = 1e-6


class PointFileError(ValueError):
    """A points file or table that cannot describe a calibration object."""


def parse_points(text: str, source: str = "points") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    `id,x,y,z[,sx,sy,sz]` text -> (ids (M,), xyz (M,3), sigma (M,3) or NaN).

    Raises PointFileError naming the line, so a typo in row 412 of a CMM export
    is found by reading the message rather than by bisecting the file.
    """
    rows: list[list[float]] = []
    lines = [ln.split("#", 1)[0] for ln in text.splitlines()]
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        cells = next(csv.reader(io.StringIO(line))) if "," in line else line.split()
        cells = [c.strip() for c in cells if c.strip()]
        try:
            values = [float(c) for c in cells]
        except ValueError:
            if not rows and cells and cells[0].lower() == "id":
                continue  # header
            raise PointFileError(f"{source}, line {lineno}: not numbers: {line!r}") from None
        if len(values) not in (4, 7):
            raise PointFileError(
                f"{source}, line {lineno}: expected 4 values (id, x, y, z) or 7 "
                f"(id, x, y, z, sx, sy, sz), got {len(values)}"
            )
        rows.append(values if len(values) == 7 else [*values, np.nan, np.nan, np.nan])
    if not rows:
        raise PointFileError(f"{source}: no points")
    data = np.asarray(rows, dtype=float)
    ids = data[:, 0]
    if not np.isfinite(data[:, :4]).all():
        raise PointFileError(f"{source}: ids and coordinates must be finite numbers")
    if not np.array_equal(ids, np.round(ids)):
        raise PointFileError(f"{source}: ids must be whole numbers")
    unique, counts = np.unique(ids, return_counts=True)
    if (counts > 1).any():
        raise PointFileError(
            f"{source}: id(s) {unique[counts > 1][:5].astype(int).tolist()} appear more than once"
        )
    return ids.astype(np.int64), data[:, 1:4], data[:, 4:7]


def format_points(ids, xyz, sigma=None) -> str:
    """
    The inverse of `parse_points`, for writing a table out to a file.

    `sigma` (M, 3) is written as `sx, sy, sz` on the rows that have it, so a
    file read with uncertainty columns is written back with them.
    """
    xyz = np.asarray(xyz, dtype=float)
    sig = np.full_like(xyz, np.nan) if sigma is None else np.asarray(sigma, dtype=float)
    has = np.isfinite(sig).all(axis=1)
    lines = ["id,x,y,z,sx,sy,sz" if has.any() else "id,x,y,z"]
    for i, (x, y, z), s, ok in zip(ids, xyz, sig, has, strict=True):
        row = f"{int(i)},{x:.9g},{y:.9g},{z:.9g}"
        lines.append(row + (f",{s[0]:.9g},{s[1]:.9g},{s[2]:.9g}" if ok else ""))
    return "\n".join(lines) + "\n"


def is_planar(xyz: np.ndarray) -> bool:
    """True when every point lies on one plane (any plane)."""
    centred = xyz - xyz.mean(axis=0)
    s = np.linalg.svd(centred, compute_uv=False)
    return s[0] == 0 or s[-1] / s[0] < _PLANAR_RATIO


@dataclass
class PointObjectSpec:
    """
    One object described by its points.

    `points_file` is read relative to the working directory, like a camera's
    `image_dir`. `points` (a list of [id, x, y, z] rows) is the alternative for
    a config written by hand or by a script; exactly one of the two is used.
    """

    id: str
    points_file: str = ""
    points: list | None = None

    target_ids: np.ndarray = field(init=False, repr=False, compare=False)
    xyz: np.ndarray = field(init=False, repr=False, compare=False)
    sigma: np.ndarray = field(init=False, repr=False, compare=False)
    planar: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.points is not None:
            source = f"object {self.id!r} (inline points)"
            text = "\n".join(",".join(str(v) for v in row) for row in self.points)
        elif self.points_file:
            path = Path(self.points_file)
            source = f"object {self.id!r} ({path})"
            if not path.is_file():
                raise PointFileError(f"{source}: file not found")
            text = path.read_text(encoding="utf-8-sig")
        else:
            raise PointFileError(f"object {self.id!r}: give a points_file (id,x,y,z per line)")
        self.target_ids, self.xyz, self.sigma = parse_points(text, source)
        self.planar = is_planar(self.xyz)
        if self.planar and not np.allclose(self.xyz[:, 2], 0.0, atol=1e-12):
            raise PointFileError(
                f"{source}: the points are all on one plane, but not on z = 0. Express "
                f"them in a frame whose x-y plane is the object's surface (z = 0 for "
                f"every point); the tool does not choose a frame for you, because that "
                f"would change what the reported object pose means."
            )
        need = 4 if self.planar else 6
        if len(self.target_ids) < need:
            raise PointFileError(
                f"{source}: a {'flat' if self.planar else '3D'} object needs at least "
                f"{need} points to be located in an image, got {len(self.target_ids)}"
            )

    @property
    def object_points(self) -> np.ndarray:
        """(M, 3), row i belongs to `target_ids[i]`."""
        return self.xyz

    @cached_property
    def pitch(self) -> float:
        """Median distance to the nearest neighbour: the object's repeat length."""
        from scipy.spatial import cKDTree

        d, _ = cKDTree(self.xyz).query(self.xyz, k=2)
        return float(np.median(d[:, 1]))

    @property
    def outline_object_points(self) -> np.ndarray:
        """(4, 2) bounding rectangle of the points in x-y, grown by one pitch."""
        lo = self.xyz[:, :2].min(axis=0) - self.pitch
        hi = self.xyz[:, :2].max(axis=0) + self.pitch
        return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
