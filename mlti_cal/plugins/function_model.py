"""
A camera model written as ONE function, adapted to the model registry.

The plug-in supplies::

    NAME = "my_lens"
    PARAM_NAMES = ("fx", "fy", "cx", "cy", "a", "b")
    def project(params, points):        # points: (N,3), camera frame, OpenCV axes
        return pixels                   # (N,2)

and optionally `jacobian(params, points) -> (d_pixels/d_params (N,2,P),
d_pixels/d_points (N,2,3))` and `default_params(image_size) -> (P,)`.

Without a Jacobian one is computed by central differences: two extra
projections per parameter and per point coordinate, so slower, but never
wrong. A Jacobian that IS supplied is compared against the numerical one when
the plug-in loads and refused if they disagree -- a wrong analytic derivative
does not crash a solve, it quietly stops it short of the minimum.

The first four parameters are always fx, fy, cx, cy: the bootstrap seeds them
from a pinhole fit, and every report that speaks of "focal length" reads them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mlti_cal.models.camera import CameraModel
from mlti_cal.plugins.contract import (
    ContractError,
    check_callable,
    check_name,
    validate_jacobian,
    validate_projection,
)

CORE = ("fx", "fy", "cx", "cy")
#: Relative step for the numerical Jacobian; central differences make the
#: truncation error O(h^2), so 1e-6 relative leaves ~1e-12 -- far below what
#: any solver tolerance can see.
_STEP = 1e-6
#: Agreement required between a supplied Jacobian and the numerical one.
_JACOBIAN_TOLERANCE = 1e-4


def numerical_jacobian(project, params: np.ndarray, X: np.ndarray):
    """(uv, d_uv/d_params (N,2,P), d_uv/d_X (N,2,3)) by central differences."""
    p = np.asarray(params, dtype=float)
    X = np.asarray(X, dtype=float)
    uv = project(p, X)
    n = X.shape[0]
    J_p = np.empty((n, 2, p.size))
    for k in range(p.size):
        h = _STEP * max(1.0, abs(p[k]))
        dp = np.zeros_like(p)
        dp[k] = h
        J_p[:, :, k] = (project(p + dp, X) - project(p - dp, X)) / (2 * h)
    J_X = np.empty((n, 2, 3))
    for k in range(3):
        h = _STEP * np.maximum(1.0, np.abs(X[:, k]))
        dX = np.zeros_like(X)
        dX[:, k] = h
        J_X[:, :, k] = (project(p, X + dX) - project(p, X - dX)) / (2 * h[:, None])
    return uv, J_p, J_X


def _test_points(rng: np.random.Generator, n: int = 40) -> np.ndarray:
    """Points in front of a camera, inside a ~60 degree cone."""
    z = rng.uniform(0.8, 3.0, n)
    xy = rng.uniform(-0.5, 0.5, (n, 2)) * z[:, None]
    return np.column_stack([xy, z])


def model_class_from_module(module, path: Path | None = None) -> type[CameraModel]:
    """
    Check a plug-in module against the camera-model contract, build its class.

    The model's own function IS run here: projecting 40 points is pure maths
    and cheap, and a model that cannot project its own default parameters, or
    whose Jacobian is wrong, must not reach a solve.
    """
    where = f"plug-in {path.name if path else module.__name__}"
    name = check_name(getattr(module, "NAME", None), where)
    names = tuple(getattr(module, "PARAM_NAMES", ()) or ())
    if names[:4] != CORE or not all(isinstance(n, str) for n in names):
        raise ContractError(f"{where}: PARAM_NAMES must start with {CORE}, got {names}")
    if len(set(names)) != len(names):
        raise ContractError(f"{where}: PARAM_NAMES has a repeated name")
    proj = check_callable(getattr(module, "project", None), 2, "project(params, points)", where)
    jac = getattr(module, "jacobian", None)
    if jac is not None:
        check_callable(jac, 2, "jacobian(params, points)", where)
    defaults_fn = getattr(module, "default_params", None)
    n_params = len(names)

    def project(self, params, X):
        X = np.asarray(X, dtype=float).reshape(-1, 3)
        return validate_projection(proj(np.asarray(params, dtype=float), X), len(X), name)

    def raw_project(params, X):
        return validate_projection(proj(params, X), len(X), name)

    def d_project(self, params, X):
        X = np.asarray(X, dtype=float).reshape(-1, 3)
        p = np.asarray(params, dtype=float)
        if jac is None:
            return numerical_jacobian(raw_project, p, X)
        J_p, J_X = validate_jacobian(jac(p, X), len(X), n_params, name)
        return raw_project(p, X), J_p, J_X

    def default_params(self, image_size):
        if defaults_fn is None:
            return CameraModel.default_params(self, image_size)
        out = np.asarray(defaults_fn(tuple(image_size)), dtype=float).ravel()
        if out.size != n_params:
            raise ContractError(f"{where}: default_params() must return {n_params} values")
        return out

    cls = type(
        f"PluginModel_{name}",
        (CameraModel,),
        {
            "name": name,
            "param_names": names,
            "label": str(getattr(module, "LABEL", name)),
            "origin": f"plug-in: {path.name}" if path else "plug-in",
            "project": project,
            "d_project": d_project,
            "default_params": default_params,
            "__doc__": (getattr(module, "__doc__", None) or f"Camera model plug-in {name}."),
        },
    )
    _self_test(cls(), where, has_jacobian=jac is not None)
    return cls


def _self_test(model: CameraModel, where: str, has_jacobian: bool) -> None:
    params = model.default_params((1280, 800))
    X = _test_points(np.random.default_rng(0))
    try:
        uv = model.project(params, X)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(
            f"{where}: project() raised {type(exc).__name__} on the default parameters: {exc}"
        ) from None
    if not np.isfinite(uv).all():
        raise ContractError(f"{where}: project() gives non-finite pixels for points in front")
    if not has_jacobian:
        return
    _, J_p, J_X = model.d_project(params, X)
    _, N_p, N_X = numerical_jacobian(lambda p, x: model.project(p, x), params, X)
    for label, got, want in (("d_pixels/d_params", J_p, N_p), ("d_pixels/d_points", J_X, N_X)):
        scale = max(1.0, float(np.abs(want).max()))
        err = float(np.abs(got - want).max()) / scale
        if err > _JACOBIAN_TOLERANCE:
            raise ContractError(
                f"{where}: jacobian() disagrees with the numerical derivative of project() "
                f"({label}, relative error {err:.1e}). Fix it or delete it -- without one, "
                f"a numerical Jacobian is used."
            )
