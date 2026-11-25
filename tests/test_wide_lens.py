"""
A lens that sees past 90 degrees off-axis must bootstrap, not only solve.

Found on the BabelCalib fisheye benchmark (TUM VI, 190 degrees; ENTANIYA, 250):
board poses came from a PnP on (x/z, y/z), which is wrong for rays at or past
90 degrees, so Double Sphere and EUCM started the bundle adjustment from corrupt
poses and stopped at 23-95 px while Kannala-Brandt reached 0.25 px.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mlti_cal.problem import initialize as init
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import build_problem, write_back
from mlti_cal.problem.types import Board, CalibrationSystem, Camera, Observation
from mlti_cal.solvers import SolveOptions, get_backend

SIZE = (512, 512)
TRUTH = {
    "double_sphere": [105.0, 105.0, 256.0, 256.0, -0.18, 0.59],
    "eucm": [112.0, 112.0, 256.0, 256.0, 0.62, 1.05],
}


def _wrapped_views(
    camera: Camera, n_views: int, rng, wrap: float
) -> tuple[Board, list[Observation], int]:
    """Views of a board that wraps around the camera: some corners lie past 90 degrees."""
    cols, rows, pitch = 12, 8, 0.04
    j, i = np.meshgrid(np.arange(cols), np.arange(rows))
    obj = np.column_stack([j.ravel() * pitch, i.ravel() * pitch, np.zeros(j.size)])
    board = Board(id="grid", object_points=obj, kind="checkerboard")
    # Board x -> camera z, board y -> camera y, board normal -> -camera x.
    base = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    views, beyond = [], 0
    for k in range(n_views * 6):
        if len(views) == n_views:
            break
        yaw, roll = rng.uniform(-0.3, 0.3, 2)
        R = cv2.Rodrigues(np.array([0.0, yaw, roll]))[0] @ base
        t = np.array(
            [
                rng.uniform(0.16, 0.34),
                -0.14 + rng.uniform(-0.04, 0.04),
                -wrap + rng.uniform(-0.1, 0.15),
            ]
        )
        X = obj @ R.T + t
        uv = camera.model.project(camera.params, X)
        theta = np.arctan2(np.hypot(X[:, 0], X[:, 1]), X[:, 2])
        ok = np.isfinite(uv).all(1) & (uv[:, 0] > 2) & (uv[:, 0] < SIZE[0] - 2)
        ok &= (uv[:, 1] > 2) & (uv[:, 1] < SIZE[1] - 2)
        if ok.sum() < 40:
            continue
        beyond += int((theta[ok] > np.pi / 2).sum())
        noisy = uv[ok] + rng.normal(0, 0.05, (ok.sum(), 2))
        views.append(Observation(f"f{k}", "cam", "grid", np.nonzero(ok)[0], noisy, sigma=1.0))
    return board, views, beyond


def _build(model: str, wrap: float = 0.12):
    rng = np.random.default_rng(7)
    truth = Camera(id="cam", model_name=model, image_size=SIZE, params=np.array(TRUTH[model]))
    board, views, beyond = _wrapped_views(truth, 24, rng, wrap)
    system = CalibrationSystem()
    system.add_camera(Camera(id="cam", model_name=model, image_size=SIZE))
    system.set_reference_camera("cam")
    system.add_board(board)
    for o in views:
        system.add_observation(o)
    return system, beyond


def _solve(system: CalibrationSystem) -> float:
    initialize_system(system)
    problem = build_problem(system)
    result = get_backend("scipy").solve(problem, SolveOptions(max_iterations=300))
    write_back(system, problem)
    return float(result.final_rms_px)


@pytest.mark.slow
@pytest.mark.parametrize("model", ["double_sphere", "eucm"])
def test_a_lens_past_90_degrees_bootstraps_and_solves(model):
    system, beyond = _build(model)
    assert beyond > 100, "the fixture must put corners past 90 degrees off-axis"
    rms = _solve(system)
    assert rms < 0.2, f"{model}: {rms:.3f} px, the noise is 0.05 px per axis"


@pytest.mark.slow
def test_the_fix_is_what_makes_the_difference(monkeypatch):
    """Without the near-axis PnP and the pose refinement this fixture fails."""
    monkeypatch.setattr(
        init, "_near_axis_mask", lambda cam, img, cfg: (np.ones(len(img), bool), False)
    )
    system, _ = _build("double_sphere")
    assert _solve(system) > 1.0


def test_a_narrow_lens_is_untouched():
    """Inside 85 degrees the mask is all True and the flag is False: the old path."""
    cam = Camera(
        id="cam",
        model_name="double_sphere",
        image_size=SIZE,
        params=np.array([350.0, 350.0, 256.0, 256.0, -0.18, 0.58]),
    )
    img = np.array([[256.0, 256.0], [400.0, 300.0], [100.0, 90.0]])
    mask, wide = init._near_axis_mask(cam, img, init.InitSettings())
    assert mask.all() and not wide
