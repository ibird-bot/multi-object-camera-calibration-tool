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
    #: How many leading parameters define the projection itself. Every model
    #: here starts with (fx, fy, cx, cy); everything after is distortion.
    num_core_params: int = 4

    @property
    def num_params(self) -> int:
        return len(self.param_names)

    def can_deactivate(self, index: int) -> bool:
        """
        May this parameter be forced to zero and held there?

        Only distortion terms. Zeroing a distortion coefficient states "this
        lens has no such term", which is a modelling choice. Zeroing fx, fy,
        cx or cy is not a simpler camera, it is a camera that projects every
        point onto one pixel -- the residuals stop meaning anything and the
        solve fails in a way that looks like bad data.
        """
        return int(index) >= self.num_core_params

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

    #: True when this model's `distortion()` is a valid OpenCV distCoeffs
    #: vector, so `cv2.undistortPoints` and friends can be handed it directly.
    #: False everywhere else -- passing e.g. Double Sphere's (xi, alpha) to
    #: OpenCV would be read as (k1, k2) and silently undistort by the wrong law.
    opencv_compatible: bool = False

    def undistort_to_normalised(
        self, params: np.ndarray, uv: np.ndarray, iterations: int = 20, tol: float = 1e-10
    ) -> np.ndarray:
        """
        Pixels -> normalised (x, y) on the z = 1 plane, by Newton on `project`.

        Generic: every model already supplies an exact d(uv)/d(point), so the
        inverse comes free rather than needing a hand-derived per-model formula
        (which is where undistortion code usually goes wrong). Two to four
        iterations is typical.

        Only valid for rays with z > 0, since the z = 1 plane is where the
        answer is expressed -- fine for bootstrapping from a board in front of
        the camera, and not a substitute for a proper unprojection to a bearing
        vector on a lens past 180 degrees.
        """
        p = np.asarray(params, dtype=float)
        target = np.asarray(uv, dtype=float).reshape(-1, 2)
        fx, fy, cx, cy = p[:4]
        # Pinhole inverse as the starting point; it is exact when there is no
        # distortion and close enough to converge when there is.
        xy = np.column_stack([(target[:, 0] - cx) / fx, (target[:, 1] - cy) / fy])

        for _ in range(iterations):
            X = np.column_stack([xy, np.ones(len(xy))])
            uv_now, _, J_X = self.d_project(p, X)
            residual = uv_now - target
            if np.max(np.abs(residual)) < tol:
                break
            # Only the x,y columns matter: z is pinned at 1.
            A = J_X[:, :, :2]
            det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            dx = (A[:, 1, 1] * residual[:, 0] - A[:, 0, 1] * residual[:, 1]) / det
            dy = (-A[:, 1, 0] * residual[:, 0] + A[:, 0, 0] * residual[:, 1]) / det
            xy = xy - np.column_stack([dx, dy])
        return xy

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
    opencv_compatible = True

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
    opencv_compatible = True

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
# Projective models -- these work on the 3D point directly rather than on a
# normalised plane, which is what lets them represent fields of view at and
# beyond 180 degrees, where X/Z does not exist.
# ---------------------------------------------------------------------------


class DoubleSphere(CameraModel):
    """
    Double Sphere (Usenko, Demmel & Cremers, 3DV 2018).

        d1  = |X|
        k   = xi d1 + z
        d2  = sqrt(x^2 + y^2 + k^2)
        den = alpha d2 + (1 - alpha) k
        u   = fx x/den + cx,   v = fy y/den + cy

    Two parameters for the whole distortion, no polynomial, and a closed-form
    inverse. On wide fisheyes it typically matches a 4-term Kannala-Brandt fit
    while being far better conditioned: there are no high-order coefficients to
    trade off against each other, which is the usual source of the near-perfect
    correlations the report warns about.

    At xi = 0, alpha = 0 it IS the pinhole model, which is what makes a pinhole
    bootstrap a valid starting point.
    """

    name = "double_sphere"
    param_names = ("fx", "fy", "cx", "cy", "xi", "alpha")

    _DEN_EPS = 1e-9

    def valid_mask(self, X: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """
        Overridden: z > 0 is the WRONG test for this model.

        A 200-degree lens genuinely images points behind the pinhole plane, and
        the base-class front-of-camera test would silently discard exactly the
        observations that constrain the wide end of the distortion. What
        actually matters is that the projection denominator stays positive.
        """
        Xa = np.asarray(X, dtype=float)
        if params is None:
            return Xa[:, 2] > Z_MIN
        _, _, _, _, xi, alpha = np.asarray(params, dtype=float)
        d1 = np.linalg.norm(Xa, axis=1)
        k = xi * d1 + Xa[:, 2]
        d2 = np.sqrt(Xa[:, 0] ** 2 + Xa[:, 1] ** 2 + k * k)
        return alpha * d2 + (1.0 - alpha) * k > self._DEN_EPS

    def _terms(self, params, Xa):
        _, _, _, _, xi, alpha = params
        x, y, z = Xa[:, 0], Xa[:, 1], Xa[:, 2]
        d1 = np.sqrt(x * x + y * y + z * z)
        d1 = np.maximum(d1, Z_MIN)
        k = xi * d1 + z
        d2 = np.sqrt(x * x + y * y + k * k)
        d2 = np.maximum(d2, Z_MIN)
        den = alpha * d2 + (1.0 - alpha) * k
        den = np.where(np.abs(den) < self._DEN_EPS, self._DEN_EPS, den)
        return x, y, z, d1, k, d2, den

    def project(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy = p[:4]
        x, y, _, _, _, _, den = self._terms(p, np.asarray(X, dtype=float))
        return np.stack([fx * x / den + cx, fy * y / den + cy], axis=1)

    def d_project(self, params, X):
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy, xi, alpha = p
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        x, y, z, d1, k, d2, den = self._terms(p, Xa)

        inv = 1.0 / den
        inv2 = inv * inv
        uv = np.stack([fx * x * inv + cx, fy * y * inv + cy], axis=1)

        # ---- d(den)/d(X,Y,Z) ------------------------------------------
        dd1 = Xa / d1[:, None]  # (n,3)
        dk = xi * dd1
        dk[:, 2] += 1.0
        dd2 = np.empty((n, 3))
        dd2[:, 0] = (x + k * dk[:, 0]) / d2
        dd2[:, 1] = (y + k * dk[:, 1]) / d2
        dd2[:, 2] = (k * dk[:, 2]) / d2
        dden = alpha * dd2 + (1.0 - alpha) * dk

        J_X = np.empty((n, 2, 3))
        J_X[:, 0, :] = -fx * x[:, None] * dden * inv2[:, None]
        J_X[:, 0, 0] += fx * inv
        J_X[:, 1, :] = -fy * y[:, None] * dden * inv2[:, None]
        J_X[:, 1, 1] += fy * inv

        # ---- d/d(params) ----------------------------------------------
        J_p = np.zeros((n, 2, 6))
        J_p[:, 0, 0] = x * inv
        J_p[:, 0, 2] = 1.0
        J_p[:, 1, 1] = y * inv
        J_p[:, 1, 3] = 1.0
        # dk/dxi = d1, dd2/dxi = k d1 / d2
        dden_dxi = alpha * (k * d1 / d2) + (1.0 - alpha) * d1
        dden_dalpha = d2 - k
        J_p[:, 0, 4] = -fx * x * dden_dxi * inv2
        J_p[:, 1, 4] = -fy * y * dden_dxi * inv2
        J_p[:, 0, 5] = -fx * x * dden_dalpha * inv2
        J_p[:, 1, 5] = -fy * y * dden_dalpha * inv2
        return uv, J_p, J_X


class EnhancedUnified(CameraModel):
    """
    Enhanced Unified Camera Model (Khomutenko, Garcia & Martinet, RA-L 2016).

        rho = sqrt(beta (x^2 + y^2) + z^2)
        den = alpha rho + (1 - alpha) z
        u   = fx x/den + cx,   v = fy y/den + cy

    `alpha` blends pinhole (0) toward the unified projection (1); `beta` shapes
    the quadric the ray is projected onto, so the pair covers ellipsoidal,
    spherical and hyperboloidal mirrors as well as fisheye lenses. Two
    parameters again, and again exactly pinhole at alpha = 0.
    """

    name = "eucm"
    param_names = ("fx", "fy", "cx", "cy", "alpha", "beta")

    _DEN_EPS = 1e-9

    def valid_mask(self, X: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Positive denominator, not z > 0 -- see DoubleSphere.valid_mask."""
        Xa = np.asarray(X, dtype=float)
        if params is None:
            return Xa[:, 2] > Z_MIN
        _, _, _, _, alpha, beta = np.asarray(params, dtype=float)
        s = beta * (Xa[:, 0] ** 2 + Xa[:, 1] ** 2) + Xa[:, 2] ** 2
        rho = np.sqrt(np.maximum(s, Z_MIN))
        den = alpha * rho + (1.0 - alpha) * Xa[:, 2]
        return (s > Z_MIN) & (den > self._DEN_EPS)

    def _terms(self, params, Xa):
        """
        Returns (x, y, z, rho, den, inside) where `inside` is the domain mask.

        beta > 0 is a DOMAIN condition, not a preference: the radicand
        beta(x^2+y^2) + z^2 goes negative for a wide enough ray once beta is
        negative, and numpy's sqrt then returns NaN. This project has no bounded
        solver, so nothing stops a step from putting beta there -- and a NaN
        residual propagates into the Jacobian, the cost and the covariance,
        turning a recoverable bad step into a dead solve with no error message.

        The radicand is therefore floored, and every derivative is zeroed on the
        floored entries: a clamped value has no gradient, and reporting the
        unclamped one would push the solver further into the region that is not
        a camera.
        """
        _, _, _, _, alpha, beta = params
        x, y, z = Xa[:, 0], Xa[:, 1], Xa[:, 2]
        s = beta * (x * x + y * y) + z * z
        inside = s > Z_MIN
        rho = np.sqrt(np.maximum(s, Z_MIN))
        den = alpha * rho + (1.0 - alpha) * z
        den = np.where(np.abs(den) < self._DEN_EPS, self._DEN_EPS, den)
        return x, y, z, rho, den, inside

    def project(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy = p[:4]
        x, y, _, _, den, _ = self._terms(p, np.asarray(X, dtype=float))
        return np.stack([fx * x / den + cx, fy * y / den + cy], axis=1)

    def d_project(self, params, X):
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy, alpha, beta = p
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        x, y, z, rho, den, inside = self._terms(p, Xa)

        inv = 1.0 / den
        inv2 = inv * inv
        uv = np.stack([fx * x * inv + cx, fy * y * inv + cy], axis=1)

        # Zero gradient wherever the radicand was floored -- see `_terms`.
        g = np.where(inside, 1.0, 0.0)
        drho = np.empty((n, 3))
        drho[:, 0] = g * beta * x / rho
        drho[:, 1] = g * beta * y / rho
        drho[:, 2] = g * z / rho
        dden = alpha * drho
        dden[:, 2] += 1.0 - alpha

        J_X = np.empty((n, 2, 3))
        J_X[:, 0, :] = -fx * x[:, None] * dden * inv2[:, None]
        J_X[:, 0, 0] += fx * inv
        J_X[:, 1, :] = -fy * y[:, None] * dden * inv2[:, None]
        J_X[:, 1, 1] += fy * inv

        J_p = np.zeros((n, 2, 6))
        J_p[:, 0, 0] = x * inv
        J_p[:, 0, 2] = 1.0
        J_p[:, 1, 1] = y * inv
        J_p[:, 1, 3] = 1.0
        dden_dalpha = rho - z
        dden_dbeta = g * alpha * (x * x + y * y) / (2.0 * rho)
        J_p[:, 0, 4] = -fx * x * dden_dalpha * inv2
        J_p[:, 1, 4] = -fy * y * dden_dalpha * inv2
        J_p[:, 0, 5] = -fx * x * dden_dbeta * inv2
        J_p[:, 1, 5] = -fy * y * dden_dbeta * inv2
        return uv, J_p, J_X


# ---------------------------------------------------------------------------
# Radial models on the normalised plane
# ---------------------------------------------------------------------------


class _NormalisedRadial(CameraModel):
    """
    Shared machinery for models of the form  (x, y) -> g(r) (x, y).

    A subclass supplies `_scale`, returning the radial factor g and its
    derivatives with respect to r^2 and to each of its own parameters. Doing
    the chain rule once here is not just less code: it is one place for the
    r -> 0 limit, which is where these models produce NaNs when each is
    written out separately.
    """

    def _scale(self, params, r2):
        """-> (g, dg/dr2, dg/dparam for the distortion params only)."""
        raise NotImplementedError

    def _normalised(self, X):
        X = np.asarray(X, dtype=float)
        z = np.maximum(X[:, 2], Z_MIN)
        return X[:, 0] / z, X[:, 1] / z, z

    def project(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy = p[:4]
        x, y, _ = self._normalised(X)
        g, _, _ = self._scale(p, x * x + y * y)
        return np.stack([fx * g * x + cx, fy * g * y + cy], axis=1)

    def d_project(self, params, X):
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy = p[:4]
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        x, y, z = self._normalised(Xa)
        r2 = x * x + y * y
        g, dg_dr2, dg_dp = self._scale(p, r2)

        uv = np.stack([fx * g * x + cx, fy * g * y + cy], axis=1)

        J_p = np.zeros((n, 2, self.num_params))
        J_p[:, 0, 0] = g * x
        J_p[:, 0, 2] = 1.0
        J_p[:, 1, 1] = g * y
        J_p[:, 1, 3] = 1.0
        for i, dg in enumerate(dg_dp):
            J_p[:, 0, 4 + i] = fx * dg * x
            J_p[:, 1, 4 + i] = fy * dg * y

        # d(g x)/dx = g + 2 x^2 dg/dr2, etc.
        dxp_dx = g + 2.0 * x * x * dg_dr2
        dxp_dy = 2.0 * x * y * dg_dr2
        dyp_dx = dxp_dy
        dyp_dy = g + 2.0 * y * y * dg_dr2

        inv_z = 1.0 / z
        J_xy = np.zeros((n, 2, 3))
        J_xy[:, 0, 0] = inv_z
        J_xy[:, 0, 2] = -x * inv_z
        J_xy[:, 1, 1] = inv_z
        J_xy[:, 1, 2] = -y * inv_z

        A = np.empty((n, 2, 2))
        A[:, 0, 0] = fx * dxp_dx
        A[:, 0, 1] = fx * dxp_dy
        A[:, 1, 0] = fy * dyp_dx
        A[:, 1, 1] = fy * dyp_dy
        return uv, J_p, np.einsum("nij,njk->nik", A, J_xy)


class FOV(_NormalisedRadial):
    """
    FOV / field-of-view model (Devernay & Faugeras, MVA 2001).

        r  = hypot(x, y)
        rd = atan(2 r tan(w/2)) / w
        u  = fx (rd/r) x + cx

    One parameter, and it is the one photographers would name: `w` is the
    ideal lens field of view in radians. Cheap and surprisingly good on real
    wide lenses.

    `w = 0` is a genuine degeneracy, not merely a numerical one: expanding the
    factor gives 1 + w^2 (1/12 - r^2/3), so its derivative with respect to w
    vanishes at zero and a solver started there can never move it. The default
    is therefore 0.5 rad rather than the usual zero-distortion start.
    """

    name = "fov"
    param_names = ("fx", "fy", "cx", "cy", "w")

    _W_EPS = 1e-8
    _R2_EPS = 1e-20

    def default_params(self, image_size: tuple[int, int]) -> np.ndarray:
        p = super().default_params(image_size)
        p[4] = 0.5  # see the class docstring: w = 0 is unrecoverable
        return p

    def _scale(self, params, r2):
        w = float(params[4])
        r = np.sqrt(np.maximum(r2, self._R2_EPS))
        if abs(w) < self._W_EPS:
            # Pinhole limit. dg/dw is exactly 0 here, which is the degeneracy
            # named in the docstring rather than an approximation.
            return np.ones_like(r2), np.zeros_like(r2), (np.zeros_like(r2),)
        t = np.tan(0.5 * w)
        a = 2.0 * r * t
        atan_a = np.arctan(a)
        g = atan_a / (w * r)

        # dg/dr, then convert to dg/dr2 = (dg/dr)/(2r)
        datan_dr = 2.0 * t / (1.0 + a * a)
        dg_dr = (datan_dr * r - atan_a) / (w * r * r)
        dg_dr2 = dg_dr / (2.0 * r)

        dt_dw = 0.5 * (1.0 + t * t)
        datan_dw = 2.0 * r * dt_dw / (1.0 + a * a)
        dg_dw = datan_dw / (w * r) - atan_a / (w * w * r)
        return g, dg_dr2, (dg_dw,)


class HalconDivision(CameraModel):
    """
    Division model (Fitzgibbon, CVPR 2001), as HALCON's `division` distortion.

    HALCON states the distortion as undistorted-from-distorted,

        r_u = r_d / (1 + kappa r_d^2)

    which has to be inverted to PROJECT. The inverse is closed form, and in the
    numerically stable arrangement

        g = r_d / r_u = 2 / (1 + sqrt(1 - 4 kappa r_u^2))

    it needs no small-kappa special case at all: the naive
    (1 - sqrt(...)) / (2 kappa r_u^2) form is 0/0 at kappa = 0 and loses most of
    its significant digits just above it.

    One coefficient, invertible in closed form both ways, and stable -- which
    is why it survives in machine vision long after polynomial models took over
    elsewhere.
    """

    name = "halcon_division"
    param_names = ("fx", "fy", "cx", "cy", "kappa")

    _DISC_MIN = 1e-12

    def _normalised(self, X):
        X = np.asarray(X, dtype=float)
        z = np.maximum(X[:, 2], Z_MIN)
        return X[:, 0] / z, X[:, 1] / z, z

    def _factor(self, kappa, r2):
        s = 4.0 * kappa * r2
        # s > 1 means the point is outside the radius this kappa can represent;
        # clamping keeps the solver's step finite instead of returning NaN and
        # poisoning the whole Jacobian.
        disc = np.maximum(1.0 - s, self._DISC_MIN)
        q = np.sqrt(disc)
        g = 2.0 / (1.0 + q)
        # dg/ds = 1 / (q (1+q)^2)
        dg_ds = 1.0 / (q * (1.0 + q) ** 2)
        return g, dg_ds, q

    def project(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy, kappa = p
        x, y, _ = self._normalised(X)
        g, _, _ = self._factor(kappa, x * x + y * y)
        return np.stack([fx * g * x + cx, fy * g * y + cy], axis=1)

    def d_project(self, params, X):
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy, kappa = p
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        x, y, z = self._normalised(Xa)
        r2 = x * x + y * y
        g, dg_ds, _ = self._factor(kappa, r2)

        uv = np.stack([fx * g * x + cx, fy * g * y + cy], axis=1)

        dg_dr2 = dg_ds * 4.0 * kappa
        dg_dkappa = dg_ds * 4.0 * r2

        J_p = np.zeros((n, 2, 5))
        J_p[:, 0, 0] = g * x
        J_p[:, 0, 2] = 1.0
        J_p[:, 1, 1] = g * y
        J_p[:, 1, 3] = 1.0
        J_p[:, 0, 4] = fx * dg_dkappa * x
        J_p[:, 1, 4] = fy * dg_dkappa * y

        dxp_dx = g + 2.0 * x * x * dg_dr2
        cross = 2.0 * x * y * dg_dr2
        dyp_dy = g + 2.0 * y * y * dg_dr2

        inv_z = 1.0 / z
        J_xy = np.zeros((n, 2, 3))
        J_xy[:, 0, 0] = inv_z
        J_xy[:, 0, 2] = -x * inv_z
        J_xy[:, 1, 1] = inv_z
        J_xy[:, 1, 2] = -y * inv_z

        A = np.empty((n, 2, 2))
        A[:, 0, 0] = fx * dxp_dx
        A[:, 0, 1] = fx * cross
        A[:, 1, 0] = fy * cross
        A[:, 1, 1] = fy * dyp_dy
        return uv, J_p, np.einsum("nij,njk->nik", A, J_xy)


# ---------------------------------------------------------------------------
# Full Brown-Conrady variants
# ---------------------------------------------------------------------------


class ThinPrism(CameraModel):
    """
    OpenCV's extended model: rational radial + tangential + thin prism.

        radial = (1 + k1 r2 + k2 r4 + k3 r6) / (1 + k4 r2 + k5 r4 + k6 r6)
        x' = x radial + 2 p1 x y + p2 (r2 + 2x^2) + s1 r2 + s2 r4
        y' = y radial + p1 (r2 + 2y^2) + 2 p2 x y + s3 r2 + s4 r4

    The rational form fits wider lenses than the plain polynomial; the thin
    prism terms s1..s4 absorb the slight non-radial asymmetry left by a tilted
    or decentred element.

    Twelve distortion coefficients is a lot, and they correlate hard. Fit it
    only when the report shows structure the simpler models leave in the
    residuals, and check the held-out error rather than the training RMS --
    this is the model most able to fit noise.

    Parameter order matches OpenCV's 12-element distCoeffs exactly.
    """

    name = "thin_prism"
    param_names = (
        "fx", "fy", "cx", "cy",
        "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
        "s1", "s2", "s3", "s4",
    )  # fmt: skip

    def _normalised(self, X):
        X = np.asarray(X, dtype=float)
        z = np.maximum(X[:, 2], Z_MIN)
        return X[:, 0] / z, X[:, 1] / z, z

    def _distort(self, p, x, y):
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = p[4:]
        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2
        num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        rad = num / den
        xp = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + s1 * r2 + s2 * r4
        yp = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y + s3 * r2 + s4 * r4
        return xp, yp, r2, r4, r6, num, den, rad

    def project(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy = p[:4]
        x, y, _ = self._normalised(X)
        xp, yp, *_ = self._distort(p, x, y)
        return np.stack([fx * xp + cx, fy * yp + cy], axis=1)

    def d_project(self, params, X):
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy = p[:4]
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = p[4:]
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        x, y, z = self._normalised(Xa)
        xp, yp, r2, r4, r6, num, den, rad = self._distort(p, x, y)
        uv = np.stack([fx * xp + cx, fy * yp + cy], axis=1)

        J_p = np.zeros((n, 2, 16))
        J_p[:, 0, 0] = xp
        J_p[:, 0, 2] = 1.0
        J_p[:, 1, 1] = yp
        J_p[:, 1, 3] = 1.0
        inv_den = 1.0 / den
        # d(rad)/dk1..k3 = r^(2i)/den ; d(rad)/dk4..k6 = -num r^(2i)/den^2
        for slot, rp in ((4, r2), (5, r4), (8, r6)):
            J_p[:, 0, slot] = fx * x * rp * inv_den
            J_p[:, 1, slot] = fy * y * rp * inv_den
        for slot, rp in ((9, r2), (10, r4), (11, r6)):
            d = -num * rp * inv_den * inv_den
            J_p[:, 0, slot] = fx * x * d
            J_p[:, 1, slot] = fy * y * d
        J_p[:, 0, 6] = fx * 2.0 * x * y
        J_p[:, 0, 7] = fx * (r2 + 2.0 * x * x)
        J_p[:, 1, 6] = fy * (r2 + 2.0 * y * y)
        J_p[:, 1, 7] = fy * 2.0 * x * y
        J_p[:, 0, 12] = fx * r2
        J_p[:, 0, 13] = fx * r4
        J_p[:, 1, 14] = fy * r2
        J_p[:, 1, 15] = fy * r4

        # d(rad)/dr2 via the quotient rule
        dnum = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4
        dden = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4
        drad = (dnum * den - num * dden) * inv_den * inv_den

        dxp_dx = (
            rad + 2.0 * x * x * drad + 2.0 * p1 * y + 6.0 * p2 * x + 2.0 * x * (s1 + 2.0 * s2 * r2)
        )
        dxp_dy = 2.0 * x * y * drad + 2.0 * p1 * x + 2.0 * p2 * y + 2.0 * y * (s1 + 2.0 * s2 * r2)
        dyp_dx = 2.0 * x * y * drad + 2.0 * p1 * x + 2.0 * p2 * y + 2.0 * x * (s3 + 2.0 * s4 * r2)
        dyp_dy = (
            rad + 2.0 * y * y * drad + 6.0 * p1 * y + 2.0 * p2 * x + 2.0 * y * (s3 + 2.0 * s4 * r2)
        )

        inv_z = 1.0 / z
        J_xy = np.zeros((n, 2, 3))
        J_xy[:, 0, 0] = inv_z
        J_xy[:, 0, 2] = -x * inv_z
        J_xy[:, 1, 1] = inv_z
        J_xy[:, 1, 2] = -y * inv_z

        A = np.empty((n, 2, 2))
        A[:, 0, 0] = fx * dxp_dx
        A[:, 0, 1] = fx * dxp_dy
        A[:, 1, 0] = fy * dyp_dx
        A[:, 1, 1] = fy * dyp_dy
        return uv, J_p, np.einsum("nij,njk->nik", A, J_xy)


class MatlabStandard(CameraModel):
    """
    MATLAB Computer Vision Toolbox model: Brown-Conrady plus a SKEW term.

        x' = x (1 + k1 r2 + k2 r4 + k3 r6) + tangential
        u  = fx x' + skew y' + cx
        v  = fy y' + cy

    The distortion is the ordinary three-radial two-tangential Brown-Conrady,
    so what actually distinguishes this from the OpenCV model is `skew`, which
    OpenCV fixes at zero. Skew is non-orthogonality of the sensor axes; on any
    modern digital sensor it is zero to well within measurement, so freeing it
    usually just gives the fit another way to absorb error. It is here because
    MATLAB estimates it, and because comparing against a MATLAB calibration
    requires being able to represent what MATLAB produced.

    Note the OTHER MATLAB convention this does NOT adopt: MATLAB indexes pixels
    from 1, so its reported principal point is 1 larger in each axis than the
    0-indexed value used here and by OpenCV. Subtract 1 from each when
    importing numbers from MATLAB.
    """

    name = "matlab"
    param_names = ("fx", "fy", "cx", "cy", "skew", "k1", "k2", "k3", "p1", "p2")
    #: `distortion()` reorders to OpenCV's (k1,k2,p1,p2,k3) and drops skew
    #: into K, so the pair really is OpenCV-compatible.
    opencv_compatible = True

    def matrix(self, params: np.ndarray) -> np.ndarray:
        """K WITH the skew term -- the base class assumes it is zero."""
        p = np.asarray(params, dtype=float)
        return np.array([[p[0], p[4], p[2]], [0.0, p[1], p[3]], [0.0, 0.0, 1.0]])

    def distortion(self, params: np.ndarray) -> np.ndarray:
        """
        (k1,k2,p1,p2,k3) in OpenCV order, skew excluded.

        Skew lives in K, not in distCoeffs; leaving it in this vector would put
        it in the exported `_dist` field, where `cv2.undistort` would read it as
        k1 and silently ruin the undistortion.
        """
        p = np.asarray(params, dtype=float)
        return np.array([p[5], p[6], p[8], p[9], p[7]])

    def _normalised(self, X):
        X = np.asarray(X, dtype=float)
        z = np.maximum(X[:, 2], Z_MIN)
        return X[:, 0] / z, X[:, 1] / z, z

    def _distort(self, p, x, y):
        _, _, _, _, _, k1, k2, k3, p1, p2 = p
        r2 = x * x + y * y
        rad = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xp = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yp = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return xp, yp, r2, rad

    def project(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy, skew = p[:5]
        x, y, _ = self._normalised(X)
        xp, yp, _, _ = self._distort(p, x, y)
        return np.stack([fx * xp + skew * yp + cx, fy * yp + cy], axis=1)

    def d_project(self, params, X):
        p = np.asarray(params, dtype=float)
        fx, fy, cx, cy, skew, k1, k2, k3, p1, p2 = p
        Xa = np.asarray(X, dtype=float)
        n = Xa.shape[0]
        x, y, z = self._normalised(Xa)
        xp, yp, r2, rad = self._distort(p, x, y)
        r4, r6 = r2 * r2, r2 * r2 * r2
        uv = np.stack([fx * xp + skew * yp + cx, fy * yp + cy], axis=1)

        J_p = np.zeros((n, 2, 10))
        J_p[:, 0, 0] = xp
        J_p[:, 0, 2] = 1.0
        J_p[:, 0, 4] = yp  # du/dskew
        J_p[:, 1, 1] = yp
        J_p[:, 1, 3] = 1.0
        # A skew term makes u depend on y' too, so every distortion column has
        # a skew contribution the OpenCV model does not.
        for slot, rp in ((5, r2), (6, r4), (7, r6)):
            J_p[:, 0, slot] = fx * x * rp + skew * y * rp
            J_p[:, 1, slot] = fy * y * rp
        dxp_dp1, dyp_dp1 = 2.0 * x * y, r2 + 2.0 * y * y
        dxp_dp2, dyp_dp2 = r2 + 2.0 * x * x, 2.0 * x * y
        J_p[:, 0, 8] = fx * dxp_dp1 + skew * dyp_dp1
        J_p[:, 1, 8] = fy * dyp_dp1
        J_p[:, 0, 9] = fx * dxp_dp2 + skew * dyp_dp2
        J_p[:, 1, 9] = fy * dyp_dp2

        drad = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2)
        cross = 2.0 * x * y * drad + 2.0 * p1 * x + 2.0 * p2 * y
        dxp_dx = rad + 2.0 * x * x * drad + 2.0 * p1 * y + 6.0 * p2 * x
        dyp_dy = rad + 2.0 * y * y * drad + 6.0 * p1 * y + 2.0 * p2 * x

        inv_z = 1.0 / z
        J_xy = np.zeros((n, 2, 3))
        J_xy[:, 0, 0] = inv_z
        J_xy[:, 0, 2] = -x * inv_z
        J_xy[:, 1, 1] = inv_z
        J_xy[:, 1, 2] = -y * inv_z

        A = np.empty((n, 2, 2))
        A[:, 0, 0] = fx * dxp_dx + skew * cross
        A[:, 0, 1] = fx * cross + skew * dyp_dy
        A[:, 1, 0] = fy * cross
        A[:, 1, 1] = fy * dyp_dy
        return uv, J_p, np.einsum("nij,njk->nik", A, J_xy)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MODELS: dict[str, type[CameraModel]] = {
    PinholeRadTan.name: PinholeRadTan,
    FisheyeKB.name: FisheyeKB,
    DoubleSphere.name: DoubleSphere,
    EnhancedUnified.name: EnhancedUnified,
    ThinPrism.name: ThinPrism,
    MatlabStandard.name: MatlabStandard,
    FOV.name: FOV,
    HalconDivision.name: HalconDivision,
}

#: Human-readable labels, matching the names these models are known by
#: elsewhere so a user coming from another tool can find the one they mean.
MODEL_LABELS: dict[str, str] = {
    "pinhole_radtan": "OpenCV",
    "fisheye_kb": "OpenCV Fisheye",
    "double_sphere": "Double Sphere",
    "eucm": "Enhanced Unified",
    "thin_prism": "Thin Prism",
    "matlab": "Matlab",
    "fov": "FOV",
    "halcon_division": "Halcon Division",
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
