"""
Render the custom-object example: a 3D target, a plug-in detector and a
plug-in camera model, with ground truth.

    python examples/custom_object/make_dataset.py
    mlti-cal --plugins examples/plugins calibrate examples/custom_object/config.json \
        --backend ceres -o Output/custom_object

Run both from the repository root.

What it shows -- the three plug-in contracts of docs/extending.md together:

- the OBJECT is three faces of a 24 cm cube covered in ArUco markers -- a 3D
  target, not a flat board. `cube.csv` lists one point per marker: its id and
  its centre in metres, in the cube's frame (written by this script);
- the DETECTOR is `examples/plugins/aruco_centres.py`, one function that
  returns [marker id, x, y] per marker found;
- cam1's LENS is `examples/plugins/my_lens.py`, a camera model given only as a
  projection function -- its derivatives are computed numerically.

`images/truth.json` holds the true camera parameters, extrinsics (T_cam_rig)
and cube poses (T_rig_object), as [t, q] with q scalar-last, metres; camera
axes are OpenCV's (x right, y down, z forward).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from mlti_cal.detectors.registry import PLUGIN_ERRORS, load_plugin_dir
from mlti_cal.io.render import (
    SceneCamera,
    _background,
    _yaw_extrinsic,
    aruco_cube,
    pixel_rays,
    render_view,
)
from mlti_cal.models.camera import get_model
from mlti_cal.models.manifolds import (
    matrix_to_quat,
    pose_compose,
    pose_inverse,
    quat_to_matrix,
    so3_exp,
)
from mlti_cal.plugins.point_object import format_points

HERE = Path(__file__).resolve().parent
SIZE = (1280, 800)
CUBE = 0.24


def cameras() -> list[SceneCamera]:
    """Three cameras 25 cm apart, the outer two turned 12 degrees inwards; cam0 is the rig frame."""
    cams = [
        SceneCamera(
            "cam0",
            "pinhole_radtan",
            SIZE,
            np.array([900.0, 901.0, 646.0, 396.0, -0.25, 0.08, 2e-4, -3e-4, -0.01]),
            _yaw_extrinsic(-0.25, -12.0),
        ),
        SceneCamera(
            "cam1",
            "my_lens",
            SIZE,
            np.array([860.0, 862.0, 634.0, 405.0, -0.18]),
            _yaw_extrinsic(0.0, 0.0),
        ),
        SceneCamera(
            "cam2",
            "pinhole_radtan",
            SIZE,
            np.array([885.0, 884.0, 638.0, 402.0, -0.27, 0.09, -1e-4, 2e-4, -0.015]),
            _yaw_extrinsic(0.25, 12.0),
        ),
    ]
    # Express the rig in cam0's frame, as the calibration does (cam0 is the
    # reference), so truth.json compares with calibration.json directly.
    rig_from_cam0 = pose_inverse(cams[0].extrinsic)
    for c in cams:
        c.extrinsic = pose_compose(c.extrinsic, rig_from_cam0)
    return cams


def cube_poses(cams: list[SceneCamera], n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """
    T_rig_object for each frame: the cube's marked corner turned towards the
    cameras, tumbled, and centred on a random point of one camera's image --
    cycling through the cameras, so every one gets its corners covered.
    """
    diagonal = np.ones(3) / np.sqrt(3)
    axis = np.cross(diagonal, [0.0, 0.0, 1.0])
    towards = so3_exp(axis / np.linalg.norm(axis) * np.arccos(diagonal[2]))
    poses = []
    for f in range(n):
        cam = cams[f % len(cams)]
        model = get_model(cam.model_name)
        target = np.array([[rng.uniform(0.15, 0.85) * SIZE[0], rng.uniform(0.15, 0.85) * SIZE[1]]])
        xy = model.undistort_to_normalised(cam.params, target)[0]
        centre_cam = np.array([xy[0], xy[1], 1.0]) * rng.uniform(0.75, 1.3)
        centre = transform(pose_inverse(cam.extrinsic), centre_cam)
        R_cam = so3_exp(rng.uniform(-0.5, 0.5, 3)) @ towards
        R = quat_to_matrix(pose_inverse(cam.extrinsic)[3:7]) @ R_cam
        poses.append(np.concatenate([centre - R @ np.full(3, CUBE / 2), matrix_to_quat(R)]))
    return poses


def transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return quat_to_matrix(T[3:7]) @ p + T[0:3]


def main(num_frames: int = 36, seed: int = 5) -> None:
    load_plugin_dir(HERE.parent / "plugins")  # cam1 uses the my_lens plug-in
    if PLUGIN_ERRORS:
        raise SystemExit(f"plug-ins failed to load: {list(PLUGIN_ERRORS)}")
    rng = np.random.default_rng(seed)
    faces, points = aruco_cube(side=CUBE)
    ids = sorted(points)
    (HERE / "cube.csv").write_text(
        "# Marker centres of the example cube: id, x, y, z in metres, cube frame.\n"
        + format_points(ids, [points[i] for i in ids]),
        encoding="utf-8",
    )
    cams = cameras()
    poses = cube_poses(cams, num_frames, rng)
    out = HERE / "images"
    for cam in cams:
        rays, valid = pixel_rays(cam)
        background = _background(SIZE, rng)
        (out / cam.id).mkdir(parents=True, exist_ok=True)
        for f, T_obj in enumerate(poses):
            boards = [(fid, tex, pose_compose(T_obj, T_face)) for fid, tex, T_face in faces]
            image = render_view(cam, rays, valid, boards, rng, background)
            cv2.imwrite(str(out / cam.id / f"{f:04d}.png"), image)
        print(f"  {cam.id}: {num_frames} images")
    truth = {
        "cameras": {
            c.id: {
                "model": c.model_name,
                "image_size": list(c.image_size),
                "parameters": c.params.tolist(),
                "extrinsic_T_cam_rig": c.extrinsic.tolist(),
            }
            for c in cams
        },
        "object_poses_T_rig_object": {f"{f:04d}": p.tolist() for f, p in enumerate(poses)},
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(
        f"{num_frames} frames x {len(cams)} cameras written to {out}; object: {HERE / 'cube.csv'}"
    )


if __name__ == "__main__":
    main()
