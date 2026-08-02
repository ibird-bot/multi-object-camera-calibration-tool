"""
SO(3) / SE(3) manifold utilities.

=============================================================================
CONVENTION -- fixed repo-wide. EVERY solver adapter must conform to this.
=============================================================================

Storage
    A pose is a 7-vector  p = [tx, ty, tz, qx, qy, qz, qw]
    translation first, then a unit quaternion in (x, y, z, w) order -- scalar
    component LAST. It represents the rigid transform  X_parent = R(q) X_child + t.

Tangent
    delta in R^6  =  [dt(3) ; phi(3)]   -- TRANSLATION FIRST, then rotation.

Retraction -- the DECOUPLED form (not the full SE(3) exponential):

    t  <-  t + dt                (additive, in the PARENT frame)
    q  <-  q (x) Exp(phi)        (right / BODY-frame rotation)

Why this exact choice -- empirically forced, not aesthetic
    pyceres 2.6 CANNOT express a custom manifold from Python. Verified against
    the installed binary: only `ambient_size` and `tangent_size` are wired to
    Python overrides; `Plus`, `PlusJacobian`, `Minus` and `MinusJacobian` are
    throw-stubs ("<PlusJacobian> not implemented."). So the Ceres adapter is
    obliged to use BUILT-IN manifolds, which means every pose must be split
    into two parameter blocks:

        translation (3, Euclidean, no manifold)  ->  t <- t + dt
        quaternion  (4, EigenQuaternionManifold) ->  q <- q (x) Exp(phi)

    That split *is* the decoupled retraction above. Adopting it as the core
    convention makes the core tangent basis and the Ceres tangent basis
    identical, so no change-of-basis is ever applied to Jacobians or to
    covariance. Had the core used the full SE(3) exponential (translation
    coupled through the V matrix), every covariance reported by the Ceres
    backend would have needed a silent basis correction -- exactly the kind of
    invisible discrepancy this project exists to expose.

    `EigenQuaternionManifold` stores (x, y, z, w) and right-multiplies, which
    matches the storage above. This was confirmed by feeding analytic Jacobians
    to Ceres' own gradient checker and getting agreement to ~1e-14.

    GTSAM's Pose3 tangent is rotation-first AND uses the coupled exponential,
    so `solvers/gtsam_backend.py` converts explicitly rather than assuming.

Ambient vs tangent Jacobians
    A Ceres CostFunction is handed AMBIENT-sized Jacobian buffers (7 columns
    per pose: 3 translation + 4 quaternion) and Ceres reduces them to local
    size itself via the manifold's PlusJacobian. The core computes tangent
    Jacobians (3 + 3). The exact bridge, used by the Ceres adapter, is

        PlusJacobian(q) = 0.5 * M(q)      (4x3)   with  M(q)^T M(q) = I
        J_ambient_q     = J_phi @ pinv(0.5*M) = 2 * J_phi @ M(q)^T

    which is exact, not an approximation, because M has orthonormal columns.
    See `quat_plus_jacobian`.

Consistency note
    The full SE(3) exp/log (`se3_exp`, `se3_log`) are retained below because
    they are the right tool for interpolation and for comparing against GTSAM,
    but they are NOT the retraction. `pose_plus` is the single source of truth;
    finite-difference tests must perturb through it.
"""

from __future__ import annotations

import numpy as np

# Below this angle (radians) the small-angle Taylor expansions are used instead
# of the trigonometric closed forms, which lose precision as theta -> 0.
_EPS = 1e-10

POSE_DIM = 7  # ambient storage size
TANGENT_DIM = 6  # local/tangent size

IDENTITY_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of a 3-vector: skew(a) @ b == cross(a, b)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# --------------------------------------------------------------------------
# SO(3)
# --------------------------------------------------------------------------


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rotation-vector -> 3x3 rotation matrix (Rodrigues)."""
    phi = np.asarray(phi, dtype=float)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < _EPS:
        # R = I + K + K^2/2 + ...
        return np.eye(3) + K + 0.5 * (K @ K)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> rotation vector."""
    R = np.asarray(R, dtype=float)
    # cos(theta) = (trace(R) - 1) / 2, clipped against round-off.
    c = (np.trace(R) - 1.0) * 0.5
    c = min(1.0, max(-1.0, c))
    theta = float(np.arccos(c))
    if theta < _EPS:
        # R ~ I + skew(phi)
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if np.pi - theta < 1e-6:
        # Near pi the antisymmetric part vanishes; recover the axis from the
        # symmetric part  R + I = 2 * outer(axis, axis)  (up to sign).
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > _EPS:
            axis = A[:, k] / axis[k]
        axis = axis / max(np.linalg.norm(axis), _EPS)
        # Fix the sign using whatever is left of the antisymmetric part.
        anti = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if float(anti @ axis) < 0.0:
            axis = -axis
        return axis * theta
    s = theta / (2.0 * np.sin(theta))
    return s * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def so3_left_jacobian(phi: np.ndarray) -> np.ndarray:
    """
    Left Jacobian V of SO(3).

    This is the matrix that couples rotation into translation inside the SE(3)
    exponential:  Exp([rho; phi]).translation == V(phi) @ rho.
    """
    phi = np.asarray(phi, dtype=float)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < _EPS:
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    t2 = theta * theta
    a = (1.0 - np.cos(theta)) / t2
    b = (theta - np.sin(theta)) / (t2 * theta)
    return np.eye(3) + a * K + b * (K @ K)


def so3_left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse of `so3_left_jacobian`."""
    phi = np.asarray(phi, dtype=float)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < _EPS:
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * (K @ K)
    half = 0.5 * theta
    c = (1.0 / (theta * theta)) * (1.0 - (half * np.cos(half) / np.sin(half)))
    return np.eye(3) - 0.5 * K + c * (K @ K)


# --------------------------------------------------------------------------
# Quaternions, (x, y, z, w) order
# --------------------------------------------------------------------------


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < _EPS:
        return np.array([0.0, 0.0, 0.0, 1.0])
    q = q / n
    # Canonical sign: keep the scalar part non-negative so that two numerically
    # equal rotations always compare equal componentwise.
    return -q if q[3] < 0.0 else q


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return quat_normalize(np.array([x, y, z, w]))


# --------------------------------------------------------------------------
# SE(3): 7-vector <-> 4x4, exp/log, retraction
# --------------------------------------------------------------------------


def pose_to_matrix(p: np.ndarray) -> np.ndarray:
    """[t(3); q(4)] -> 4x4 homogeneous transform."""
    p = np.asarray(p, dtype=float)
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(p[3:7])
    T[:3, 3] = p[0:3]
    return T


def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform -> [t(3); q(4)]."""
    T = np.asarray(T, dtype=float)
    return np.concatenate([T[:3, 3], matrix_to_quat(T[:3, :3])])


def se3_exp(delta: np.ndarray) -> np.ndarray:
    """
    Full SE(3) exponential.  delta = [rho(3); phi(3)]  ->  4x4 transform.

    Note the V-matrix coupling: translation is V(phi) @ rho, not rho.
    """
    delta = np.asarray(delta, dtype=float)
    rho, phi = delta[0:3], delta[3:6]
    T = np.eye(4)
    T[:3, :3] = so3_exp(phi)
    T[:3, 3] = so3_left_jacobian(phi) @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """Inverse of `se3_exp`.  4x4 transform -> delta = [rho(3); phi(3)]."""
    T = np.asarray(T, dtype=float)
    phi = so3_log(T[:3, :3])
    rho = so3_left_jacobian_inv(phi) @ T[:3, 3]
    return np.concatenate([rho, phi])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a (x) b, both in (x, y, z, w) order."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_exp(phi: np.ndarray) -> np.ndarray:
    """Rotation vector -> unit quaternion (x, y, z, w)."""
    phi = np.asarray(phi, dtype=float)
    theta = float(np.linalg.norm(phi))
    if theta < _EPS:
        # sin(t/2)/t -> 1/2 as t -> 0
        return quat_normalize(np.array([*(0.5 * phi), 1.0]))
    half = 0.5 * theta
    return np.concatenate([phi * (np.sin(half) / theta), [np.cos(half)]])


def quat_plus_jacobian(q: np.ndarray) -> np.ndarray:
    """
    d(q (x) Exp(phi)) / d(phi) evaluated at phi = 0.  Shape (4, 3).

    Equals 0.5 * M(q), where M has orthonormal columns (M^T M = I). That
    property is what makes `2 * J_phi @ M.T` an exact ambient Jacobian rather
    than a least-squares approximation -- see the module docstring.
    """
    qx, qy, qz, qw = np.asarray(q, dtype=float)
    M = np.array(
        [
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw],
            [-qx, -qy, -qz],
        ]
    )
    return 0.5 * M


def quat_tangent_basis(q: np.ndarray) -> np.ndarray:
    """M(q) itself (4x3, orthonormal columns). `2 * M.T` is pinv(PlusJacobian)."""
    return 2.0 * quat_plus_jacobian(q)


def pose_plus(p: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    THE retraction. Every backend must reproduce exactly this.

        t <- t + dt              (parent frame, additive)
        q <- q (x) Exp(phi)      (body frame, right-multiplied)

    with delta = [dt(3); phi(3)]. Decoupled on purpose -- see module docstring.
    """
    p = np.asarray(p, dtype=float)
    delta = np.asarray(delta, dtype=float)
    t = p[0:3] + delta[0:3]
    q = quat_normalize(quat_mul(p[3:7], quat_exp(delta[3:6])))
    return np.concatenate([t, q])


def pose_minus(p2: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Inverse of `pose_plus`:  the delta with  pose_plus(p1, delta) == p2."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    dt = p2[0:3] - p1[0:3]
    R1 = quat_to_matrix(p1[3:7])
    R2 = quat_to_matrix(p2[3:7])
    phi = so3_log(R1.T @ R2)
    return np.concatenate([dt, phi])


def pose_inverse(p: np.ndarray) -> np.ndarray:
    """Pose of the inverse transform."""
    R = quat_to_matrix(np.asarray(p, dtype=float)[3:7])
    t = np.asarray(p, dtype=float)[0:3]
    Ri = R.T
    return np.concatenate([-Ri @ t, matrix_to_quat(Ri)])


def pose_compose(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Pose of  T(pa) @ T(pb)."""
    return matrix_to_pose(pose_to_matrix(pa) @ pose_to_matrix(pb))


def transform_points(p: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply pose `p` to an (N,3) array of points."""
    T = pose_to_matrix(p)
    return X @ T[:3, :3].T + T[:3, 3]


def skew_stack(X: np.ndarray) -> np.ndarray:
    """Vectorised skew: (N,3) -> (N,3,3)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    S = np.zeros((n, 3, 3))
    S[:, 0, 1] = -X[:, 2]
    S[:, 0, 2] = X[:, 1]
    S[:, 1, 0] = X[:, 2]
    S[:, 1, 2] = -X[:, 0]
    S[:, 2, 0] = -X[:, 1]
    S[:, 2, 1] = X[:, 0]
    return S


def d_point_d_pose_tangent(
    R_left: np.ndarray, R_pose: np.ndarray, X_child: np.ndarray
) -> np.ndarray:
    """
    Jacobian of a point w.r.t. the DECOUPLED tangent of one pose in a chain.

    Consider a point carried through a pose P and then through everything
    sitting to its left, L:

        Y = L * (P * X_child)

    Perturb only P, using this repo's retraction (t += dt, R <- R Exp(phi)):

        d(P X)/d(dt)  =  I
        d(P X)/d(phi) = -R_P skew(X_child)

    and therefore

        dY/d(dt)  =  R_L
        dY/d(phi) = -R_L R_P skew(X_child)

    Note the asymmetry: the translation columns do NOT involve R_P. That is
    precisely what distinguishes this from the coupled/body-frame convention,
    and getting it backwards produces a Jacobian that is wrong only when
    R_P != I -- i.e. it passes a lazy identity-pose unit test and then quietly
    degrades every real solve.

    Args:
        R_left: (3,3) rotation of the accumulated transform LEFT of this pose.
            Identity when the pose is outermost (e.g. a camera extrinsic when
            the rig frame is the reference).
        R_pose: (3,3) rotation of the pose being perturbed.
        X_child: (N,3) points in the frame to the RIGHT of this pose.

    Returns:
        (N, 3, 6) array, columns ordered [dt(3) | phi(3)].
    """
    X_child = np.atleast_2d(np.asarray(X_child, dtype=float))
    n = X_child.shape[0]
    J = np.empty((n, 3, 6))
    J[:, :, 0:3] = np.asarray(R_left, dtype=float)[None, :, :]
    RLRP = np.asarray(R_left, dtype=float) @ np.asarray(R_pose, dtype=float)
    J[:, :, 3:6] = -np.einsum("ij,njk->nik", RLRP, skew_stack(X_child))
    return J


# --------------------------------------------------------------------------
# Convenience for building poses from human-readable inputs
# --------------------------------------------------------------------------


def pose_from_rt(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """OpenCV-style (rotation vector, translation vector) -> pose 7-vector."""
    return np.concatenate(
        [np.asarray(tvec, dtype=float).ravel(), matrix_to_quat(so3_exp(np.asarray(rvec).ravel()))]
    )


def pose_to_rt(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pose 7-vector -> OpenCV-style (rvec, tvec)."""
    p = np.asarray(p, dtype=float)
    return so3_log(quat_to_matrix(p[3:7])), p[0:3].copy()
