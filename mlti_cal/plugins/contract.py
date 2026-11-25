"""
The gate between third-party code and the calibration.

Everything a plug-in hands back passes through here before the rest of the
tool sees it. The rules are the documented contract (`docs/extending.md`); a
result that breaks one is refused with a message naming the plug-in and the
rule, never repaired. Repairing would mean guessing, and a guessed
correspondence written into a calibration is the one error nothing downstream
can detect.

What happens on a violation is decided by the caller, not here: a detector
result that breaks the contract costs that one image, a camera model that
breaks it at load time is not registered. Both are reported.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np

#: The contract version this build implements. A plug-in may declare
#: `PLUGIN_API = 1`; one declaring a version this build does not know is
#: refused at load instead of being run under rules it was not written for.
PLUGIN_API = 1


class ContractError(ValueError):
    """A plug-in, or something it returned, breaks the documented contract."""


def check_api_version(module, where: str) -> None:
    version = getattr(module, "PLUGIN_API", PLUGIN_API)
    if version != PLUGIN_API:
        raise ContractError(
            f"{where}: written for PLUGIN_API = {version!r}, this build implements "
            f"PLUGIN_API = {PLUGIN_API}"
        )


def check_name(name, where: str) -> str:
    if not isinstance(name, str) or not name.isidentifier():
        raise ContractError(
            f"{where}: NAME must be a string usable as an identifier "
            f"(letters, digits, underscores), got {name!r}"
        )
    return name


def check_callable(func, n_args: int, label: str, where: str) -> Callable:
    """`func` must accept exactly `n_args` positional arguments."""
    if not callable(func):
        raise ContractError(f"{where}: {label} is not a function")
    try:
        inspect.signature(func).bind(*range(n_args))
    except TypeError as exc:
        raise ContractError(
            f"{where}: {label} must take {n_args} positional argument(s) ({exc})"
        ) from None
    except ValueError:  # builtins without a signature: accepted, checked at call time
        pass
    return func


def setting_kind(value) -> str:
    """The settings-panel type of one SETTINGS default. bool before int: bool IS an int."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    raise TypeError(type(value).__name__)


def check_settings(settings, where: str) -> dict:
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise ContractError(f"{where}: SETTINGS must be a dict, got {type(settings).__name__}")
    for key, value in settings.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise ContractError(f"{where}: SETTINGS key {key!r} is not a valid name")
        try:
            setting_kind(value)
        except TypeError:
            raise ContractError(
                f"{where}: SETTINGS[{key!r}] is a {type(value).__name__}; only bool, "
                f"int and float values can be edited in the settings window"
            ) from None
    return dict(settings)


# -- detector results ---------------------------------------------------------
def validate_detections(raw, image_shape: tuple[int, ...], name: str) -> np.ndarray:
    """
    One `detect()` return value -> a checked (N, 3) float array [id, x, y].

    Pixel convention is OpenCV's: (0, 0) is the CENTRE of the top-left pixel,
    so the image spans [-0.5, w - 0.5] x [-0.5, h - 0.5]. Raises ContractError
    naming the rule broken.
    """
    where = f"detector {name!r}"
    if not isinstance(raw, np.ndarray):
        raise ContractError(
            f"{where}: detect() must return a numpy array, got {type(raw).__name__}"
        )
    if not np.issubdtype(raw.dtype, np.floating):
        raise ContractError(
            f"{where}: detect() must return a FLOATING array (sub-pixel x, y), got dtype "
            f"{raw.dtype}. Convert with points.astype(float)."
        )
    if raw.size == 0:
        return np.zeros((0, 3))
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ContractError(
            f"{where}: detect() must return shape (N, 3) as [target_id, x, y], got {raw.shape}"
        )
    pts = np.asarray(raw, dtype=float)
    if not np.isfinite(pts).all():
        raise ContractError(f"{where}: detect() returned NaN or infinite values")
    ids = pts[:, 0]
    if not np.array_equal(ids, np.round(ids)):
        raise ContractError(f"{where}: target ids (column 0) must be whole numbers")
    unique, counts = np.unique(ids, return_counts=True)
    if (counts > 1).any():
        dup = unique[counts > 1][:5].astype(int).tolist()
        raise ContractError(
            f"{where}: target id(s) {dup} returned more than once in one image; "
            f"an id must identify one point"
        )
    h, w = image_shape[:2]
    x, y = pts[:, 1], pts[:, 2]
    outside = (x < -0.5) | (x > w - 0.5) | (y < -0.5) | (y > h - 0.5)
    if outside.any():
        i = int(np.flatnonzero(outside)[0])
        raise ContractError(
            f"{where}: point for target {int(ids[i])} at ({x[i]:.2f}, {y[i]:.2f}) is outside "
            f"the {w}x{h} image. Columns must be [target_id, x, y] with x across, y down."
        )
    return pts


# -- camera model results -----------------------------------------------------
def validate_projection(raw, n: int, name: str) -> np.ndarray:
    """One `project()` return value -> a checked (N, 2) float array."""
    where = f"camera model {name!r}"
    if not isinstance(raw, np.ndarray):
        raise ContractError(f"{where}: project() must return a numpy array")
    out = np.asarray(raw, dtype=float)
    if out.shape != (n, 2):
        raise ContractError(f"{where}: project() must return shape ({n}, 2), got {out.shape}")
    return out


def validate_jacobian(raw, n: int, n_params: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    """One `jacobian()` return value -> (d_uv/d_params (N,2,P), d_uv/d_points (N,2,3))."""
    where = f"camera model {name!r}"
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise ContractError(
            f"{where}: jacobian() must return a pair (d_pixels/d_params, d_pixels/d_points)"
        )
    J_p, J_X = (np.asarray(a, dtype=float) for a in raw)
    if J_p.shape != (n, 2, n_params):
        raise ContractError(
            f"{where}: d_pixels/d_params must have shape ({n}, 2, {n_params}), got {J_p.shape}"
        )
    if J_X.shape != (n, 2, 3):
        raise ContractError(
            f"{where}: d_pixels/d_points must have shape ({n}, 2, 3), got {J_X.shape}"
        )
    return J_p, J_X
