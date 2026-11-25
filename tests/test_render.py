"""
The image renderer: what it draws must be where the ground truth says.

Everything downstream -- the example rig, the demo video -- compares a
calibration against this ground truth, so a renderer that is off by a fraction
of a pixel would show up as a calibration error that is not there.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlti_cal.detectors.charuco import CharucoDetector
from mlti_cal.io.config import board_spec
from mlti_cal.io.render import (
    SceneCamera,
    _project,
    board_points,
    board_texture,
    pixel_rays,
    render_view,
)
from mlti_cal.models.manifolds import IDENTITY_POSE, matrix_to_quat, so3_exp

BOARD = {
    "id": "b",
    "kind": "charuco",
    "squares_x": 6,
    "squares_y": 5,
    "square_length": 0.04,
    "marker_length": 0.03,
    "dictionary": "DICT_4X4_50",
}


@pytest.mark.parametrize(
    ("model_name", "params"),
    [
        ("pinhole_radtan", [420.0, 420.0, 322.0, 238.0, -0.25, 0.08, 1e-4, -2e-4, -0.01]),
        ("double_sphere", [260.0, 260.0, 318.0, 242.0, -0.15, 0.58]),
    ],
)
def test_detected_corners_land_on_the_ground_truth(model_name, params):
    """Regression: drawn squares were half a texel off, ~0.3 px in the image."""
    cam = SceneCamera("c", model_name, (640, 480), np.array(params), IDENTITY_POSE.copy())
    pts = board_points(BOARD)
    R = so3_exp(np.array([0.35, -0.3, 0.1]))
    centre = np.array([0.12, 0.10, 0.0])
    pose = np.concatenate([np.array([0.03, -0.02, 0.55]) - R @ centre, matrix_to_quat(R)])

    rays, valid = pixel_rays(cam)
    rng = np.random.default_rng(0)
    background = np.full((480, 640), 100.0, np.float32)
    img = render_view(cam, rays, valid, [("b", board_texture(BOARD), pose)], rng, background)

    det = CharucoDetector(board_spec(BOARD)).detect(img)
    assert det is not None and len(det.point_ids) >= 15
    truth = _project(cam, pts, pose)[det.point_ids]
    err = np.asarray(det.image_points) - truth
    # Median, not RMS: on the wide lens a few corners refine to ~0.5 px, which
    # is the detector on small curved squares, not the renderer. What the
    # renderer can get wrong is a systematic offset, checked strictly below.
    assert np.median(np.linalg.norm(err, axis=1)) < 0.15
    assert np.abs(err.mean(axis=0)).max() < 0.08, f"systematic offset {err.mean(axis=0)} px"


def test_a_board_seen_from_behind_is_not_drawn():
    cam = SceneCamera(
        "c", "pinhole_radtan", (320, 240), np.array([300.0, 300, 160, 120, 0, 0, 0, 0, 0]),
        IDENTITY_POSE.copy(),
    )  # fmt: skip
    flipped = so3_exp(np.array([np.pi, 0.0, 0.0]))  # print facing away
    pose = np.concatenate([[-0.1, 0.08, 0.6], matrix_to_quat(flipped)])
    rays, valid = pixel_rays(cam, supersample=1)
    background = np.full((240, 320), 100.0, np.float32)
    img = render_view(
        cam, rays, valid, [("b", board_texture(BOARD), pose)], np.random.default_rng(0),
        background, noise_std=0.0, blur_sigma=0.3,
    )  # fmt: skip
    assert img.std() < 5, "a back-facing board must leave the background untouched"
