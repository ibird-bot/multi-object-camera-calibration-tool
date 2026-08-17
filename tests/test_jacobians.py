"""
THE GATE. Analytic Jacobians vs central finite differences.

Nothing downstream is trustworthy if this file fails: the solver comparison,
every covariance, and every uncertainty map inherit these derivatives. Per
PLAN.md this is a hard M2 gate -- it exists before any solver adapter does.

Manifold-valued blocks are perturbed THROUGH `pose_plus`, never through the raw
7-vector storage. Differencing the 7-vector would produce a 2x7 that cannot be
compared to the 2x6 analytic block without dragging in the plus-Jacobian, which
is exactly the mistake this file is here to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlti_cal.models.camera import available_models, get_model
from mlti_cal.models.manifolds import (
    d_point_d_pose_tangent,
    matrix_to_quat,
    pose_minus,
    pose_plus,
    pose_to_matrix,
    quat_plus_jacobian,
    quat_to_matrix,
    so3_exp,
    so3_log,
)

RNG = np.random.default_rng(20260802)


def central_diff(fn, x, eps=1e-6):
    """Central-difference Jacobian of fn at x. Returns (out_dim, len(x))."""
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(fn(x))
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        dp, dm = x.copy(), x.copy()
        dp[i] += eps
        dm[i] -= eps
        J[:, i] = (np.atleast_1d(fn(dp)) - np.atleast_1d(fn(dm))).ravel() / (2.0 * eps)
    return J


def random_pose(rng, t_scale=1.0, r_scale=0.8):
    t = rng.normal(scale=t_scale, size=3)
    q = matrix_to_quat(so3_exp(rng.normal(scale=r_scale, size=3)))
    return np.concatenate([t, q])


def random_points(rng, n=12, z_lo=1.0, z_hi=4.0):
    return np.column_stack(
        [rng.uniform(-1.0, 1.0, n), rng.uniform(-1.0, 1.0, n), rng.uniform(z_lo, z_hi, n)]
    )


# ---------------------------------------------------------------------------
# Manifold self-consistency
# ---------------------------------------------------------------------------


def test_pose_plus_minus_roundtrip():
    for _ in range(50):
        p1, p2 = random_pose(RNG), random_pose(RNG)
        d = pose_minus(p2, p1)
        back = pose_plus(p1, d)
        # compare as transforms; quaternion sign is canonicalised but compare
        # the matrices to be safe against any double-cover surprise.
        assert np.allclose(pose_to_matrix(back), pose_to_matrix(p2), atol=1e-10)


def test_pose_plus_zero_is_identity():
    for _ in range(20):
        p = random_pose(RNG)
        assert np.allclose(pose_plus(p, np.zeros(6)), p, atol=1e-12)


def test_so3_log_exp_roundtrip_principal_branch():
    """log(exp(phi)) == phi only for |phi| < pi -- log returns the principal branch."""
    for _ in range(200):
        phi = RNG.normal(scale=1.2, size=3)
        n = np.linalg.norm(phi)
        if n >= np.pi - 1e-6:  # rescale into the branch instead of skipping
            phi = phi * ((np.pi - 1e-3) / n)
        assert np.allclose(so3_log(so3_exp(phi)), phi, atol=1e-9)


def test_so3_exp_log_roundtrip_any_rotation():
    """exp(log(R)) == R holds for every rotation, including |phi| > pi inputs."""
    for _ in range(200):
        phi = RNG.normal(scale=2.5, size=3)  # deliberately beyond pi
        R = so3_exp(phi)
        assert np.allclose(so3_exp(so3_log(R)), R, atol=1e-9)


def test_so3_log_near_pi():
    """The theta -> pi branch is the one that silently returns garbage."""
    for axis in np.eye(3):
        for theta in (np.pi - 1e-7, np.pi - 1e-9):
            R = so3_exp(axis * theta)
            rec = so3_log(R)
            assert np.allclose(so3_exp(rec), R, atol=1e-7), f"axis={axis} theta={theta}"


def test_quat_plus_jacobian_matches_fd():
    """d(q (x) Exp(phi))/d(phi) at phi=0, and orthonormality of M."""
    for _ in range(30):
        p = random_pose(RNG)
        q = p[3:7]
        Jn = central_diff(
            lambda d, q=q: pose_plus(np.concatenate([np.zeros(3), q]), d)[3:7], np.zeros(6)
        )[:, 3:6]
        assert np.allclose(Jn, quat_plus_jacobian(q), atol=1e-8)
        M = 2.0 * quat_plus_jacobian(q)
        assert np.allclose(M.T @ M, np.eye(3), atol=1e-12)


# ---------------------------------------------------------------------------
# Pose-chain Jacobian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("left_is_identity", [True, False])
def test_d_point_d_pose_tangent(left_is_identity):
    """
    Perturb the pose through the real retraction and difference the resulting
    transformed point. Runs with a NON-identity left transform, because the
    translation columns are R_left (not R_left @ R_pose) and an identity-only
    test cannot tell those apart.
    """
    for _ in range(25):
        p_pose = random_pose(RNG)
        p_left = np.array([0.0, 0, 0, 0, 0, 0, 1.0]) if left_is_identity else random_pose(RNG)
        X = random_points(RNG, 8)
        T_left = pose_to_matrix(p_left)
        R_left = T_left[:3, :3]
        R_pose = quat_to_matrix(p_pose[3:7])

        def fn(delta, p_pose=p_pose, T_left=T_left, X=X):
            pp = pose_plus(p_pose, delta)
            T = T_left @ pose_to_matrix(pp)
            return ((T[:3, :3] @ X.T).T + T[:3, 3]).ravel()

        Jn = central_diff(fn, np.zeros(6))
        Ja = d_point_d_pose_tangent(R_left, R_pose, X).reshape(-1, 6)
        assert np.allclose(Jn, Ja, atol=1e-7), np.abs(Jn - Ja).max()


def test_translation_columns_are_not_r_left_r_pose():
    """
    Guard against the coupled/decoupled mix-up. With a non-identity pose
    rotation the two candidate answers must actually differ, otherwise the test
    above proves nothing.
    """
    p_left, p_pose = random_pose(RNG), random_pose(RNG)
    R_left = pose_to_matrix(p_left)[:3, :3]
    R_pose = quat_to_matrix(p_pose[3:7])
    assert not np.allclose(R_left, R_left @ R_pose, atol=1e-3)


# ---------------------------------------------------------------------------
# Camera models
# ---------------------------------------------------------------------------


#: Plausible values per model, in the model's OWN parameter order.
#:
#: These are deliberately not near-zero. A Jacobian bug in a distortion term is
#: invisible when that term is 0 -- its column is then correct by accident --
#: so every coefficient here is far enough from zero to exercise its own
#: derivative and its coupling into the others.
REALISTIC_PARAMS = {
    "pinhole_radtan": [900.0, 905.0, 640.0, 360.0, -0.28, 0.11, 1e-3, -8e-4, -0.02],
    "fisheye_kb": [420.0, 421.0, 640.0, 360.0, -0.02, 3e-3, -1e-3, 2e-4],
    # xi and alpha both well inside (0,1): a real fisheye sits around here.
    "double_sphere": [350.0, 351.0, 640.0, 360.0, -0.18, 0.58],
    "eucm": [360.0, 362.0, 640.0, 360.0, 0.62, 1.15],
    "thin_prism": [
        900.0,
        905.0,
        640.0,
        360.0,
        -0.28,
        0.11,
        1e-3,
        -8e-4,
        -0.02,
        1e-3,
        -5e-4,
        2e-4,
        3e-4,
        -1e-4,
        2e-4,
        -1.5e-4,
    ],  # fmt: skip
    "matlab": [900.0, 905.0, 640.0, 360.0, 0.7, -0.28, 0.11, -0.02, 1e-3, -8e-4],
    "fov": [500.0, 502.0, 640.0, 360.0, 0.92],
    "halcon_division": [800.0, 802.0, 640.0, 360.0, -0.11],
}


def realistic_params(model):
    if model.name not in REALISTIC_PARAMS:
        raise AssertionError(
            f"no test parameters for {model.name!r}. Every registered model must "
            f"have them, or it is silently untested."
        )
    p = np.array(REALISTIC_PARAMS[model.name], dtype=float)
    assert p.size == model.num_params, (
        f"{model.name}: {p.size} test parameters for {model.num_params} model parameters"
    )
    return p


def test_every_registered_model_is_covered():
    """A model with no entry above would be parametrized but never exercised."""
    assert set(available_models()) == set(REALISTIC_PARAMS)


@pytest.mark.parametrize("name", available_models())
def test_camera_project_matches_d_project_value(name):
    model = get_model(name)
    p = realistic_params(model)
    X = random_points(RNG, 20)
    uv_a = model.project(p, X)
    uv_b, _, _ = model.d_project(p, X)
    assert np.allclose(uv_a, uv_b, atol=1e-12)


@pytest.mark.parametrize("name", available_models())
def test_camera_jacobian_wrt_params(name):
    model = get_model(name)
    p = realistic_params(model)
    X = random_points(RNG, 15)
    _, J_p, _ = model.d_project(p, X)
    Jn = central_diff(lambda q, X=X: model.project(q, X).ravel(), p, eps=1e-6)
    Ja = J_p.reshape(-1, model.num_params)
    # Focal-length columns are O(1) while k-columns are O(1e-3); compare with a
    # per-column relative tolerance rather than one global atol.
    scale = np.maximum(np.abs(Jn).max(axis=0), 1e-6)
    err = np.abs(Jn - Ja) / scale
    assert err.max() < 1e-5, f"{name}: worst rel err {err.max():.3e}"


@pytest.mark.parametrize("name", available_models())
def test_camera_jacobian_wrt_point(name):
    model = get_model(name)
    p = realistic_params(model)
    X = random_points(RNG, 15)
    _, _, J_X = model.d_project(p, X)
    for i in range(X.shape[0]):
        Jn = central_diff(lambda x, p=p: model.project(p, x[None, :]).ravel(), X[i], eps=1e-7)
        assert np.allclose(Jn, J_X[i], rtol=1e-5, atol=1e-6), f"{name} point {i}"


@pytest.mark.parametrize("name", available_models())
def test_camera_jacobian_at_optical_centre(name):
    """r = 0 is the singular case in the fisheye model; must stay finite."""
    model = get_model(name)
    p = realistic_params(model)
    X = np.array([[0.0, 0.0, 2.0], [1e-12, -1e-12, 1.5]])
    uv, J_p, J_X = model.d_project(p, X)
    for arr in (uv, J_p, J_X):
        assert np.all(np.isfinite(arr)), f"{name} produced non-finite values at r=0"
    assert np.allclose(uv[0], p[2:4], atol=1e-9)


@pytest.mark.parametrize("name", available_models())
def test_camera_jacobian_wide_angle(name):
    """Large incidence angles are where a wrong distortion derivative shows."""
    model = get_model(name)
    p = realistic_params(model)
    X = np.column_stack(
        [
            RNG.uniform(-2.5, 2.5, 15),
            RNG.uniform(-2.5, 2.5, 15),
            RNG.uniform(1.0, 2.0, 15),
        ]
    )
    _, J_p, _ = model.d_project(p, X)
    Jn = central_diff(lambda q, X=X: model.project(q, X).ravel(), p, eps=1e-6)
    Ja = J_p.reshape(-1, model.num_params)
    scale = np.maximum(np.abs(Jn).max(axis=0), 1e-6)
    assert (np.abs(Jn - Ja) / scale).max() < 1e-5
