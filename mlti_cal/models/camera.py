"""
Camera projection models: value + analytic Jacobians.

Every model implements

    project(params, X_cam)   -> uv                       (N,2)
    d_project(params, X_cam) -> (uv, J_params, J_point)   (N,2), (N,2,P), (N,2,3)

X_cam is (N,3) in the camera frame, OpenCV convention: +x right, +y down,
+z forward (into the scene). Points with z <= Z_MIN are not projectable; use
`valid_mask` and drop them, do not let them silently produce garbage.

The analytic Jacobians here are the whole ballgame -- the solver comparison and
every uncertainty number downstream inherit their correctness. They are gated
by finite-difference tests in tests/test_jacobians.py, which must pass before
any solver is allowed to run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

Z_MIN = 1e-6


class CameraModel(ABC):
    """Base class for projection models."""

    name: str = "abstract"
    param_names: tuple[str, ...] = ()

    @property
    def num_params(self) -> int:
        return len(self.param_names)

    # -- required ----------------------------------------------------------
    @abstractmethod
    def project(self, params: np.ndarray, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def d_project(
        self, params: np.ndarray, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    # -- shared ------------------------------------------------------------
    def valid_mask(self, X: np.ndarray) -> np.ndarray:
        """Points strictly in front of the camera."""
        return np.asarray(X, dtype=float)[:, 2] > Z_MIN

    def default_params(self, image_size: tuple[int, int]) -> np.ndarray:
        """
        A neutral starting guess: focal ~ image width, principal point centred,
        zero distortion. Deliberately crude -- the solver's job is to move it,
        and a too-clever initialisation hides observability problems.
        """
        w, h = image_size
        p = np.zeros(self.num_params)
        p[0] = float(w)
        p[1] = float(w)
        p[2] = 0.5 * float(w)
        p[3] = 0.5 * float(h)
        return p

    def matrix(self, params: np.ndarray) -> np.ndarray:
        """3x3 pinhole intrinsic matrix K (ignores distortion)."""
        fx, fy, cx, cy = np.asarray(params, dtype=float)[:4]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    def distortion(self, params: np.ndarray) -> np.ndarray:
        return np.asarray(params, dtype=float)[4:].copy()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({', '.join(self.param_names)})"


# ---------------------------------------------------------------------------
# Pinhole + Brown-Conrady ("radtan"), OpenCV-compatible
# ---------------------------------------------------------------------------


class PinholeRadTan(CameraModel):
    """
    OpenCV `cv2.calibrateCamera` model.

        x  = X/Z,  y = Y/Z,  r2 = x^2 + y^2
        D  = 1 + k1 r2 + k2 r2^2 + k3 r2^3
        x' = x D + 2 p1 x y + p2 (r2 + 2 x^2)
        y' = y D + p1 (r2 + 2 y^2) + 2 p2 x y
        u  = fx x' + cx,   v = fy y' + cy

    Parameter order matches OpenCV's distCoeffs layout (k1,k2,p1,p2,k3) so
    exported YAML drops straight into cv2.undistort without reshuffling.
    """

    name = "pinhole_radtan"
    param_names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")

    def _normalised(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        z = np.maximum(X[:, 2], Z_MIN)
        return X[:, 0] / z, X[:, 1] / z, z

    def project(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = np.asarray(params, dtype=float)
        x, y, _ = self._normalised(X)
        r2 = x * x + y * y
        d = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xp = x * d + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yp = y * d + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.stack([fx * xp + cx, fy * yp + cy], axis=1)

    def d_project(
        self, params: np.ndarray, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = np.asarray(params, dtype=float)
        Xa = np.asarray(X, dtype=float)
        x, y, z = self._normalised(Xa)
        n = x.shape[0]

        r2 = x * x + y * y
        d = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dd_dr2 = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2)

        xp = x * d + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yp = y * d + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        uv = np.stack([fx * xp + cx, fy * yp + cy], axis=1)

        # ---- d(uv)/d(params) ------------------------------------------
        J_p = np.zeros((n, 2, 9))
        J_p[:, 0, 0] = xp  # du/dfx
        J_p[:, 0, 2] = 1.0  # du/dcx
        J_p[:, 1, 1] = yp  # dv/dfy
        J_p[:, 1, 3] = 1.0  # dv/dcy
        r4, r6 = r2 * r2, r2 * r2 * r2
        J_p[:, 0, 4] = fx * x * r2  # du/dk1
        J_p[:, 0, 5] = fx * x * r4  # du/dk2
        J_p[:, 0, 6] = fx * 2.0 * x * y  # du/dp1
        J_p[:, 0, 7] = fx * (r2 + 2.0 * x * x)  # du/dp2
        J_p[:, 0, 8] = fx * x * r6  # du/dk3
        J_p[:, 1, 4] = fy * y * r2
        J_p[:, 1, 5] = fy * y * r4
        J_p[:, 1, 6] = fy * (r2 + 2.0 * y * y)
        J_p[:, 1, 7] = fy * 2.0 * x * y
        J_p[:, 1, 8] = fy * y * r6

        # ---- d(x',y')/d(x,y) ------------------------------------------
        # The cross terms are provably equal; computing once is both faster and
        # a standing check on the derivation.
        cross = 2.0 * x * y * dd_dr2 + 2.0 * p1 * x + 2.0 * p2 * y
        dxp_dx = d + 2.0 * x * x * dd_dr2 + 2.0 * p1 * y + 6.0 * p2 * x
        dxp_dy = cross
        dyp_dx = cross
        dyp_dy = d + 2.0 * y * y * dd_dr2 + 6.0 * p1 * y + 2.0 * p2 * x

        # ---- d(x,y)/d(X,Y,Z) ------------------------------------------
        inv_z = 1.0 / z
        # dx/dX = 1/z, dx/dY = 0, dx/dZ = -x/z ; dy/dY = 1/z, dy/dZ = -y/z
        J_x = np.zeros((n, 2, 3))
        J_x[:, 0, 0] = inv_z
        J_x[:, 0, 2] = -x * inv_z
        J_x[:, 1, 1] = inv_z
        J_x[:, 1, 2] = -y * inv_z

        # chain: d(uv)/d(X) = diag(fx,fy) @ d(x'y')/d(xy) @ d(xy)/d(X)
        A = np.empty((n, 2, 2))
        A[:, 0, 0] = fx * dxp_dx
        A[:, 0, 1] = fx * dxp_dy
        A[:, 1, 0] = fy * dyp_dx
        A[:, 1, 1] = fy * dyp_dy
        J_X = np.einsum("nij,njk->nik", A, J_x)
        return uv, J_p, J_X


# ---------------------------------------------------------------------------
# Fisheye / Kannala-Brandt, OpenCV `cv2.fisheye` compatible
# ---------------------------------------------------------------------------


class FisheyeKB(CameraModel):
    """
    OpenCV `cv2.fisheye` equidistant model.

        a = X/Z, b = Y/Z,  r = hypot(a,b),  theta = atan(r)
        theta_d = theta (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)
        x' = (theta_d / r) a,   y' = (theta_d / r) b
        u  = fx x' + cx,        v  = fy y' + cy

    The r -> 0 limit (theta_d/r -> 1) is handled explicitly; without it the
    optical-centre pixel produces 0/0 and poisons the whole Jacobian row.
    """

    name = "fisheye_kb"
    param_names = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")

    _R_EPS = 1e-9

    def project(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        fx, fy, cx, cy, k1, k2, k3, k4 = np.asarray(params, dtype=float)
        Xa = np.asarray(X, dtype=float)
        z = np.maximum(Xa[:, 2], Z_MIN)
        a, b = Xa[:, 0] / z, Xa[:, 1] / z
        r = np.hypot(a, b)
        th = np.arctan(r)
        th2 = th * th
        thd = th * (1.0 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4))))
        s = np.where(r < self._R_EPS, 1.0, thd / np.where(r < self._R_EPS, 1.0, r))
        return np.stack([fx * s * a + cx, fy * s * b + cy], axis=1)

    def d_project(
        self, params: np.ndarray, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fx, fy, cx, cy, k1, k2, k3, k4 = np.asarray(params, dtype=float)
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        z = np.maximum(Xa[:, 2], Z_MIN)
        a, b = Xa[:, 0] / z, Xa[:, 1] / z
        r = np.hypot(a, b)
        small = r < self._R_EPS
        r_safe = np.where(small, 1.0, r)

        th = np.arctan(r)
        th2 = th * th
        poly = 1.0 + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4)))
        thd = th * poly
        s = np.where(small, 1.0, thd / r_safe)

        uv = np.stack([fx * s * a + cx, fy * s * b + cy], axis=1)

        # ---- d/d(params) ----------------------------------------------
        J_p = np.zeros((n, 2, 8))
        J_p[:, 0, 0] = s * a
        J_p[:, 0, 2] = 1.0
        J_p[:, 1, 1] = s * b
        J_p[:, 1, 3] = 1.0
        # d(theta_d)/dk_i = theta^(2i+1); ds/dk_i = that / r
        for i, kpow in enumerate((3, 5, 7, 9)):
            dthd = th**kpow
            ds = np.where(small, 0.0, dthd / r_safe)
            J_p[:, 0, 4 + i] = fx * ds * a
            J_p[:, 1, 4 + i] = fy * ds * b

        # ---- d/d(a,b) ---------------------------------------------------
        dthd_dth = 1.0 + th2 * (3.0 * k1 + th2 * (5.0 * k2 + th2 * (7.0 * k3 + th2 * 9.0 * k4)))
        dth_dr = 1.0 / (1.0 + r * r)
        dthd_dr = dthd_dth * dth_dr
        ds_dr = np.where(small, 0.0, (dthd_dr * r_safe - thd) / (r_safe * r_safe))
        # dr/da = a/r, dr/db = b/r
        da = np.where(small, 0.0, a / r_safe)
        db = np.where(small, 0.0, b / r_safe)
        dxp_da = s + a * ds_dr * da
        dxp_db = a * ds_dr * db
        dyp_da = b * ds_dr * da
        dyp_db = s + b * ds_dr * db

        # ---- d(a,b)/d(X,Y,Z) --------------------------------------------
        inv_z = 1.0 / z
        J_ab = np.zeros((n, 2, 3))
        J_ab[:, 0, 0] = inv_z
        J_ab[:, 0, 2] = -a * inv_z
        J_ab[:, 1, 1] = inv_z
        J_ab[:, 1, 2] = -b * inv_z

        A = np.empty((n, 2, 2))
        A[:, 0, 0] = fx * dxp_da
        A[:, 0, 1] = fx * dxp_db
        A[:, 1, 0] = fy * dyp_da
        A[:, 1, 1] = fy * dyp_db
        J_X = np.einsum("nij,njk->nik", A, J_ab)
        return uv, J_p, J_X


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MODELS: dict[str, type[CameraModel]] = {
    PinholeRadTan.name: PinholeRadTan,
    FisheyeKB.name: FisheyeKB,
}


def register_model(cls: type[CameraModel]) -> type[CameraModel]:
    """Decorator to add a third-party model to the registry."""
    _MODELS[cls.name] = cls
    return cls


def get_model(name: str) -> CameraModel:
    if name not in _MODELS:
        raise KeyError(f"unknown camera model {name!r}; known: {sorted(_MODELS)}")
    return _MODELS[name]()


def available_models() -> list[str]:
    return sorted(_MODELS)
