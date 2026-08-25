"""
Behaviour every camera model must have, beyond its Jacobians.

The finite-difference gate in test_jacobians.py proves the derivatives match
the projection. It cannot prove the projection is the RIGHT one -- a model with
a sign error is perfectly self-consistent. These tests check the properties
that pin down the intended geometry: that each model degenerates to the pinhole
where its own theory says it should, that its inverse really inverts it, and
that a full calibration recovers the parameters it was generated from.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlti_cal.io.synthetic import generate_dataset, seed_ground_truth_poses
from mlti_cal.models.camera import MODEL_LABELS, available_models, get_model
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import build_problem, write_back
from mlti_cal.solvers import SolveOptions, backend_status, get_backend
from mlti_cal.solvers.base import per_corner_rms
from tests.test_jacobians import REALISTIC_PARAMS

#: Models added on top of the two OpenCV ones.
NEW_MODELS = ["double_sphere", "eucm", "thin_prism", "matlab", "fov", "halcon_division"]

RNG = np.random.default_rng(11)


def points(n=200, spread=1.1):
    return np.column_stack(
        [
            RNG.uniform(-spread, spread, n),
            RNG.uniform(-spread, spread, n),
            RNG.uniform(1.5, 4.0, n),
        ]
    )


def test_the_requested_models_are_all_registered():
    expected = {
        "pinhole_radtan",
        "fisheye_kb",
        "double_sphere",
        "eucm",
        "thin_prism",
        "matlab",
        "fov",
        "halcon_division",
    }
    assert expected == set(available_models())
    assert set(MODEL_LABELS) == expected, "every model needs a human-readable label"


@pytest.mark.parametrize("name", available_models())
def test_undistortion_inverts_projection(name):
    """
    The generic Newton inverse must return the ray the projection came from.

    Used by the bootstrap for every model OpenCV cannot undistort, so an error
    here becomes a wrong PnP pose and a wrong starting point.
    """
    model = get_model(name)
    p = np.array(REALISTIC_PARAMS[name], dtype=float)
    X = points()
    uv = model.project(p, X)
    xy = model.undistort_to_normalised(p, uv)
    back = model.project(p, np.column_stack([xy, np.ones(len(xy))]))
    assert np.abs(back - uv).max() < 1e-8, f"{name}: inverse does not invert"
    # ...and it recovers the actual normalised coordinates, not just any
    # pre-image that happens to reproject correctly.
    assert np.abs(xy - X[:, :2] / X[:, 2:3]).max() < 1e-8


@pytest.mark.parametrize("name", NEW_MODELS)
def test_reduces_to_the_pinhole_at_zero_distortion(name):
    """
    Each new model contains the pinhole as a special case.

    This is not a nicety: `initialize_intrinsics` bootstraps these models from
    an OpenCV PINHOLE fit, which is only a legitimate starting point because
    zero distortion puts them exactly on the pinhole.
    """
    model = get_model(name)
    p = np.array(REALISTIC_PARAMS[name], dtype=float)
    p[4:] = 0.0
    if name == "fov":
        # w = 0 is a removable singularity, approached rather than evaluated.
        p[4] = 1e-9
    X = points()
    fx, fy, cx, cy = p[:4]
    pinhole = np.column_stack([fx * X[:, 0] / X[:, 2] + cx, fy * X[:, 1] / X[:, 2] + cy])
    assert np.abs(model.project(p, X) - pinhole).max() < 1e-7


def test_fov_does_not_default_to_its_own_degeneracy():
    """
    FOV's factor is 1 + w^2(1/12 - r^2/3), so d/dw vanishes at w = 0.

    Starting there leaves the parameter permanently unidentifiable -- the
    solver sees a zero column and never moves it -- so the default must not be
    the usual zero-distortion start.
    """
    model = get_model("fov")
    p = model.default_params((1280, 720))
    assert p[4] != 0.0

    X = points()
    _, J_p, _ = model.d_project(p, X)
    assert np.abs(J_p[:, :, 4]).max() > 1e-6, "w has no gradient at the default"

    # ...and at w = 0 it genuinely has none, which is what the default avoids.
    p0 = p.copy()
    p0[4] = 0.0
    _, J0, _ = model.d_project(p0, X)
    assert np.abs(J0[:, :, 4]).max() == 0.0


def test_matlab_keeps_skew_out_of_the_opencv_distortion_vector():
    """
    Skew belongs in K. In distCoeffs it would be read as k1.

    `export_opencv_yaml` writes `distortion()` straight out, so a skew term
    leaking into that vector would silently corrupt anyone's `cv2.undistort`.
    """
    model = get_model("matlab")
    p = np.array(REALISTIC_PARAMS["matlab"], dtype=float)
    dist = model.distortion(p)
    assert dist.size == 5
    # OpenCV order is (k1, k2, p1, p2, k3); ours is (skew, k1, k2, k3, p1, p2).
    assert dist == pytest.approx([p[5], p[6], p[8], p[9], p[7]])
    assert model.matrix(p)[0, 1] == pytest.approx(p[4]), "skew must appear in K"


def test_opencv_compatible_flag_is_honest():
    """
    The flag decides whether OpenCV is handed our coefficients directly.

    Marking a model compatible when it is not means OpenCV reads, say, Double
    Sphere's xi as k1 and undistorts by a law the camera does not obey -- which
    produces plausible, wrong poses rather than an error.
    """
    for name in ("pinhole_radtan", "fisheye_kb", "matlab"):
        assert get_model(name).opencv_compatible
    for name in ("double_sphere", "eucm", "fov", "halcon_division", "thin_prism"):
        assert not get_model(name).opencv_compatible


@pytest.mark.parametrize("name", ["double_sphere", "eucm"])
def test_projective_models_can_see_past_ninety_degrees(name):
    """
    These exist for lenses wider than 180 degrees, where X/Z does not exist.

    The base-class `z > 0` test would discard exactly the observations that
    constrain the wide end, so both override it with the real condition: a
    positive projection denominator.
    """
    model = get_model(name)
    p = np.array(REALISTIC_PARAMS[name], dtype=float)
    behind = np.array([[0.9, 0.0, -0.05], [0.0, 0.9, -0.05]])
    assert np.all(model.valid_mask(behind, p)), "a wide lens must accept these"
    assert np.all(np.isfinite(model.project(p, behind)))
    # Without params there is nothing to evaluate the condition with, so it
    # falls back to the conservative front-of-camera test.
    assert not np.any(model.valid_mask(behind))


#: Same models as `NEW_MODELS`, but Thin Prism carries the `slow` marker. With
#: 16 intrinsics per camera it needs thousands of scipy iterations and takes
#: ~150s of a ~250s suite -- 60% of the total runtime for one parameter set.
#: Marked rather than deleted or shrunk: it is the only end-to-end check that
#: the widest parameterisation converges, so it still runs on every full local
#: run and on the nightly job, just not on the per-push one.
END_TO_END_MODELS = [
    pytest.param(name, marks=pytest.mark.slow) if name == "thin_prism" else name
    for name in NEW_MODELS
]


@pytest.mark.parametrize("name", END_TO_END_MODELS)
def test_calibrates_end_to_end_to_the_noise_floor(name):
    """
    The real check: generate from known parameters, calibrate, hit the floor.

    Per-corner RMS should approach sigma*sqrt(2) = 0.424 px for sigma = 0.3 px
    per coordinate. Falling short means the model, its Jacobians, its bootstrap
    or its PnP path is wrong -- this is the test that covers the wiring the
    unit tests cannot.
    """
    system, gt = generate_dataset(num_frames=14, num_cameras=2, seed=3, model_name=name)
    report = initialize_system(system)
    assert report["pnp_solved"] > 0, f"{name}: bootstrap produced no poses"

    problem = build_problem(system)
    # Ceres where it exists: Thin Prism has 16 intrinsics per camera and scipy's
    # trf spends several function evaluations per step, so the iteration cap
    # rather than the objective decides when it stops.
    backend = "ceres" if backend_status().get("ceres", (False, ""))[0] else "scipy"
    result = get_backend(backend).solve(problem, SolveOptions(max_iterations=4000))
    write_back(system, problem)
    rms = per_corner_rms(problem)

    assert result.success, f"{name}: {result.message}"
    assert rms < 0.55, f"{name}: converged to {rms:.4f} px, above the 0.424 px noise floor"

    # Deliberately NOT asserted: that the recovered parameters equal the true
    # ones. Intrinsics and board poses jointly explain the images, and in the
    # projective models the trade-off is severe -- measured here, Double Sphere
    # lands on fx 448 against a true 375, with xi absorbing the difference,
    # while sitting exactly on the noise floor. Both describe the same camera
    # over the field the board covered. Pinning the numbers would pin an
    # arbitrary point along that valley; `test_projection_reproduces_its_own_
    # data` is what actually proves the projection is right, and the report's
    # correlation warning is what tells a user the valley is there.


@pytest.mark.parametrize("name", NEW_MODELS)
def test_projection_reproduces_its_own_data(name):
    """
    The decisive correctness check, and the only non-circular one.

    Generate observations from known parameters with NO noise, then evaluate
    the residual AT those exact parameters. It must be zero to machine
    precision. Finite differences only prove d_project matches project; this
    proves `project` is the function the generator used -- a sign error or a
    transposed term would sail through the Jacobian gate and die here.
    """
    system, gt = generate_dataset(
        num_frames=10,
        num_cameras=2,
        seed=5,
        model_name=name,
        pixel_noise_std=0.0,
        init_perturbation=False,  # intrinsics and extrinsics at ground truth
    )
    seed_ground_truth_poses(system, gt)  # ...and the board poses too
    rms = per_corner_rms(build_problem(system))
    assert rms < 1e-9, f"{name}: {rms:.3e} px residual at the parameters that made the data"


@pytest.mark.parametrize("name", NEW_MODELS)
def test_bootstrap_reports_nan_rather_than_opencvs_own_rms(name):
    """
    OpenCV fits a PINHOLE to get K for these models.

    Reporting that fit's RMS as the camera's initialisation error would credit
    this model with a number produced by a different one.
    """
    system, _ = generate_dataset(num_frames=8, num_cameras=1, seed=1, model_name=name)
    report = initialize_system(system)
    assert all(np.isnan(v) for v in report["intrinsics_rms"].values())
