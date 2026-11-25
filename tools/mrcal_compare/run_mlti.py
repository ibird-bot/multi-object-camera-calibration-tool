"""
mlti_cal side of the mrcal comparison: calibrate one camera from a corners file.

Input is a mrgingham `corners.vnl` (lines `filename x y level`, `- - -` where a
board was not found), e.g. the public mrcal sample set. Corners are stored
row-major, so corner k = i * gridn + j sits at (j * spacing, i * spacing, 0) in
the board frame -- the same layout mrcal's `ref_calibration_object` uses.

Conventions written to the JSON: OpenCV camera frame (+x right, +y down, +z
forward); `frames_rt_toref` rows are [rodrigues(3), translation(3)] mapping
board -> camera, which is what mrcal calls rt_toref for a single camera.

    python tools/mrcal_compare/run_mlti.py CORNERS OUT.json --model thin_prism
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from mlti_cal.models.manifolds import pose_to_matrix, pose_to_rt
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import build_problem, write_back
from mlti_cal.problem.types import Board, CalibrationSystem, Camera, Observation
from mlti_cal.report.covariance import compute_covariance
from mlti_cal.report.uncertainty import projection_uncertainty
from mlti_cal.solvers import SolveOptions, backend_status, get_backend


def read_corners(path: Path, gridn: int) -> OrderedDict[str, np.ndarray]:
    """filename -> (gridn*gridn, 2) pixels, complete boards only, file order."""
    rows: OrderedDict[str, list] = OrderedDict()
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, x, y = line.split()[:3]
        if x != "-":
            rows.setdefault(name, []).append((float(x), float(y)))
    full = gridn * gridn
    return OrderedDict((k, np.array(v)) for k, v in rows.items() if len(v) == full)


def rms_px(system: CalibrationSystem) -> float:
    """sqrt(mean(dx^2 + dy^2)) over every observed corner."""
    cam = next(iter(system.cameras.values()))
    sq = []
    for o in system.observations:
        T = pose_to_matrix(system.board_poses[(o.frame, o.board)])
        X = system.boards[o.board].object_points[o.point_ids]
        uv = cam.model.project(cam.params, X @ T[:3, :3].T + T[:3, 3])
        sq.append(np.sum((uv - o.image_points) ** 2, axis=1))
    return float(np.sqrt(np.mean(np.concatenate(sq))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("corners", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="thin_prism")
    ap.add_argument("--gridn", type=int, default=14)
    ap.add_argument("--spacing", type=float, default=0.0588, help="corner spacing, metres")
    ap.add_argument("--size", type=int, nargs=2, default=(6000, 3376), metavar=("W", "H"))
    ap.add_argument("--grid-step", type=int, default=400)
    args = ap.parse_args()

    corners = read_corners(args.corners, args.gridn)
    ij = np.indices((args.gridn, args.gridn)).reshape(2, -1)
    points = np.column_stack([ij[1], ij[0], np.zeros(ij.shape[1])]) * args.spacing

    system = CalibrationSystem()
    system.add_camera(Camera(id="cam0", model_name=args.model, image_size=tuple(args.size)))
    system.add_board(Board(id="board", object_points=points, kind="checkerboard"))
    ids = np.arange(len(points))
    for name, uv in corners.items():
        system.add_observation(Observation(name, "cam0", "board", ids, uv, sigma=1.0))

    initialize_system(system)
    rms_init = rms_px(system)
    backend = "ceres" if backend_status()["ceres"][0] else "scipy"
    problem = build_problem(system)
    result = get_backend(backend).solve(problem, SolveOptions(max_iterations=500))
    write_back(system, problem)

    cov = compute_covariance(problem, pixel_noise_std=None)
    cam = system.cameras["cam0"]
    frames = [f for f in corners if (f, "board") in system.board_poses]
    dist = [float(np.linalg.norm(system.board_poses[(f, "board")][:3])) for f in frames]
    ranges = [round(float(np.median(dist)), 2), 100.0]
    maps = {}
    for rng in ranges:
        m = projection_uncertainty(problem, system, cov, "cam0", rng, grid_step=args.grid_step)
        old = projection_uncertainty(
            problem, system, cov, "cam0", rng, grid_step=args.grid_step, reference="camera"
        )
        ok = m.valid.ravel()
        maps[str(rng)] = {
            "pixels": np.column_stack([m.grid_u.ravel(), m.grid_v.ravel()])[ok].tolist(),
            "sigma_max": m.sigma_max.ravel()[ok].tolist(),
            # The pre-fix quantity (point fixed in camera coordinates), kept so
            # the comparison shows what the old map would have claimed.
            "sigma_max_camera_reference": old.sigma_max.ravel()[ok].tolist(),
        }

    n_intr = cam.model.num_params
    out = {
        "model": args.model,
        "backend": backend,
        "success": bool(result.success),
        "iterations": int(result.iterations),
        "image_size": list(args.size),
        "gridn": args.gridn,
        "spacing": args.spacing,
        "param_names": list(cam.model.param_names),
        "params": cam.params.tolist(),
        "param_sigma": [cov.std_error_of(f"intr:cam0.{i}") for i in range(n_intr)],
        "rms_init_px": rms_init,
        "rms_px": rms_px(system),
        "sigma_residual_px": float(np.sqrt(cov.sigma2_residual)),
        "rank_deficient": bool(cov.rank_deficient),
        "frames": frames,
        "frames_rt_toref": [
            np.concatenate(pose_to_rt(system.board_poses[(f, "board")])).tolist() for f in frames
        ],
        "board_distance_m": [min(dist), float(np.median(dist)), max(dist)],
        "uncertainty": maps,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(
        f"{args.model} via {backend}: {len(frames)} frames, RMS {rms_init:.3f} -> "
        f"{out['rms_px']:.4f} px, success={result.success}, {result.iterations} iterations"
    )
    print("params:", np.array2string(cam.params, precision=5, suppress_small=True))


if __name__ == "__main__":
    main()
