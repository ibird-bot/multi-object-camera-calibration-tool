"""
Calibrating from a 3D object: points on three faces of a box, no flat board.

Corner-level synthetic data from known cameras and object poses, so the result
can be compared with the truth rather than with itself. Conventions as
everywhere: OpenCV camera axes, T_cam_rig extrinsics, T_rig_board object
poses, [t, q] with q scalar-last, metres.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from mlti_cal.io.synthetic import relative_extrinsic_error
from mlti_cal.models import camera as camera_models
from mlti_cal.models.camera import get_model
from mlti_cal.models.manifolds import (
    matrix_to_quat,
    pose_compose,
    pose_inverse,
    quat_to_matrix,
    so3_exp,
)
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import build_problem, write_back
from mlti_cal.problem.types import Board, CalibrationSystem, Camera, Observation
from mlti_cal.solvers import SolveOptions, backend_status, get_backend
from mlti_cal.solvers.base import per_corner_rms

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "plugins"
SIZE = (1280, 800)


def box_points(side: float = 0.3, n: int = 5) -> np.ndarray:
    """A grid on each of three faces meeting at the origin: x=0, y=0 and z=0."""
    g = np.linspace(0.03, side - 0.03, n)
    a, b = (m.ravel() for m in np.meshgrid(g, g))
    zero = np.zeros_like(a)
    return np.vstack(
        [
            np.column_stack([a, b, zero]),
            np.column_stack([a, zero, b]),
            np.column_stack([zero, a, b]),
        ]
    )


def _pose(rotvec, t) -> np.ndarray:
    return np.concatenate(
        [np.asarray(t, float), matrix_to_quat(so3_exp(np.asarray(rotvec, float)))]
    )


def make_system(models: dict[str, tuple[str, np.ndarray]], frames: int = 16, seed: int = 0):
    """Two cameras 0.2 m apart, a box tumbling 1-1.6 m in front of them."""
    rng = np.random.default_rng(seed)
    extrinsics = {
        "cam0": _pose([0, 0, 0], [0, 0, 0]),
        "cam1": pose_inverse(_pose([0, np.radians(12), 0], [0.2, 0, 0])),
    }
    system = CalibrationSystem()
    for cid, (model_name, _) in models.items():
        system.add_camera(Camera(cid, model_name, SIZE, is_reference=cid == "cam0"))
    pts = box_points()
    system.add_board(Board("box", pts, kind="test_3d"))
    truth_poses = {}
    for f in range(frames):
        # The box's open corner faces the cameras, so all three faces show,
        # then a random tumble of up to 20 degrees per axis.
        tumble = rng.uniform(-0.35, 0.35, 3)
        R = so3_exp(tumble) @ so3_exp(np.array([np.radians(-35), np.radians(35), 0.0]))
        t = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.2, 0.2), rng.uniform(1.0, 1.6)])
        T_rig_box = np.concatenate([t - R @ np.full(3, 0.15), matrix_to_quat(R)])
        truth_poses[f"{f:03d}"] = T_rig_box
        for cid, (model_name, params) in models.items():
            T = pose_compose(extrinsics[cid], T_rig_box)
            X = pts @ quat_to_matrix(T[3:7]).T + T[0:3]
            uv = get_model(model_name).project(params, X)
            w, h = SIZE
            ok = (
                (X[:, 2] > 0.1)
                & (uv[:, 0] > 2)
                & (uv[:, 0] < w - 2)
                & (uv[:, 1] > 2)
                & (uv[:, 1] < h - 2)
            )
            if ok.sum() >= 12:
                noisy = uv[ok] + rng.normal(scale=0.2, size=(ok.sum(), 2))
                system.add_observation(
                    Observation(f"{f:03d}", cid, "box", np.flatnonzero(ok), noisy)
                )
    return system, extrinsics


def rms_at_truth(system, models, truth) -> float:
    """
    The per-corner RMS with every camera at its TRUE value (object poses refit).

    A correct solve must reach at least this low: the truth is not the minimum
    of noisy data, the optimum is. Measured: the solve lands 0.0004-0.0009 px
    below it. Comparing with this, rather than demanding the truth back to
    some arbitrary tolerance, separates "the solver is wrong" from "one 0.3 m
    object pins the principal point only to ~1-2 px" -- which alone moves the
    extrinsic rotation by ~0.05-0.2 degrees.
    """
    import copy

    gt = copy.deepcopy(system)
    for cid, (_, params) in models.items():
        gt.cameras[cid].params = params.copy()
        gt.cameras[cid].extrinsic = truth[cid].copy()
    problem = build_problem(gt, optimize_intrinsics=False, optimize_extrinsics=False)
    backend = "ceres" if backend_status().get("ceres", (False, ""))[0] else "scipy"
    get_backend(backend).solve(problem, SolveOptions(max_iterations=100))
    return per_corner_rms(problem)


def calibrate(system):
    report = initialize_system(system)
    start = {cid: c.extrinsic.copy() for cid, c in system.cameras.items()}
    start_params = {cid: c.params.copy() for cid, c in system.cameras.items()}
    problem = build_problem(system)
    backend = "ceres" if backend_status().get("ceres", (False, ""))[0] else "scipy"
    options = SolveOptions(max_iterations=300, extra={} if backend == "ceres" else {"method": "lm"})
    result = get_backend(backend).solve(problem, options)
    write_back(system, problem)
    return report, start, start_params, result, per_corner_rms(problem)


PINHOLE = np.array([900.0, 902.0, 646.0, 396.0, -0.25, 0.08, 2e-4, -3e-4, -0.01])
DOUBLE_SPHERE = np.array([520.0, 521.0, 636.0, 404.0, -0.15, 0.58])


def test_a_rig_calibrates_from_a_3d_object_alone():
    models = {"cam0": ("pinhole_radtan", PINHOLE), "cam1": ("double_sphere", DOUBLE_SPHERE)}
    system, truth = make_system(models)
    assert not system.boards["box"].is_flat
    report, start, start_params, result, rms = calibrate(system)

    assert report["pnp_solved"] > 20
    # The START is graded loosely -- it only has to be in the basin. fx only
    # for the pinhole: Double Sphere's fx trades off against xi and alpha
    # (correlation ~0.999 on this kind of data), so its fx alone says little;
    # its extrinsic below is the meaningful check.
    assert abs(start_params["cam0"][0] / PINHOLE[0] - 1) < 0.05, start_params["cam0"][0]
    t0, r0 = relative_extrinsic_error(start["cam1"], truth["cam1"])
    assert t0 < 0.05 and r0 < 3.0, f"start: {t0 * 1000:.1f} mm, {r0:.2f} deg"
    # The RESULT: at the optimum (no worse than the truth), and near the truth.
    assert rms <= rms_at_truth(system, models, truth) + 1e-4, f"final RMS {rms:.4f} px"
    t1, r1 = relative_extrinsic_error(system.cameras["cam1"].extrinsic, truth["cam1"])
    assert t1 < 0.002 and r1 < 0.3, f"final: {t1 * 1000:.2f} mm, {r1:.3f} deg"
    assert abs(system.cameras["cam0"].params[0] - PINHOLE[0]) < 3.0


@pytest.fixture
def my_lens(tmp_path, monkeypatch):
    from mlti_cal.detectors.registry import load_plugins

    models, labels = dict(camera_models._MODELS), dict(camera_models.MODEL_LABELS)
    shutil.copy(EXAMPLES / "my_lens.py", tmp_path / "my_lens.py")
    monkeypatch.setenv("MLTI_CAL_PLUGIN_PATH", str(tmp_path))
    load_plugins()
    yield
    camera_models._MODELS.clear()
    camera_models._MODELS.update(models)
    camera_models.MODEL_LABELS.clear()
    camera_models.MODEL_LABELS.update(labels)


def test_a_plugin_camera_model_calibrates_like_a_builtin(my_lens):
    """my_lens has no analytic Jacobian: the numerical one must carry the solve."""
    truth_params = np.array([880.0, 884.0, 650.0, 402.0, -0.21])
    models = {"cam0": ("pinhole_radtan", PINHOLE), "cam1": ("my_lens", truth_params)}
    system, truth = make_system(models, seed=4)
    _, _, _, _, rms = calibrate(system)
    assert rms <= rms_at_truth(system, models, truth) + 1e-4, f"final RMS {rms:.4f} px"
    got = system.cameras["cam1"].params
    assert np.allclose(got[:4], truth_params[:4], atol=3.0), got
    assert abs(got[4] - truth_params[4]) < 0.01, got
    t1, r1 = relative_extrinsic_error(system.cameras["cam1"].extrinsic, truth["cam1"])
    assert t1 < 0.002 and r1 < 0.3, f"final: {t1 * 1000:.2f} mm, {r1:.3f} deg"
