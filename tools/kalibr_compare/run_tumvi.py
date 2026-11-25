"""
mlti_cal vs Kalibr on a real fisheye stereo rig: the TUM VI calibration data.

Dataset: TUM VI `dataset-calib-cam8_512_16` (EuRoC export, CC BY 4.0), two
512x512 fisheye cameras and a 6x6 Kalibr AprilGrid (tag 0.088 m, spacing 0.3).
Reference: the Kalibr pinhole-equidistant calibration TUM publishes for that
same sequence (`pinhole-equi-512.zip`, camchain-imucam-imucalib.yaml, which
was run with --cam .../dataset-calib-cam8_512_16/pinhole-equi/camchain-camcalib.yaml).
Kalibr's pinhole + equidistant model is exactly this package's `fisheye_kb`.

What is compared:
  * intrinsics, as differences in mlti_cal 1-sigma units;
  * projection difference in pixels over the region the board covered -- once
    directly and once after the best-fit rotation of the camera frame (the
    two tools may place the camera frame differently by a small rotation that
    the board poses absorb; mrcal's differencing makes the same allowance);
  * T_cam1_cam0: baseline length and rotation.

Conventions: OpenCV camera frame (+x right, +y down, +z forward); cam0 is the
reference; Kalibr's T_cn_cnm1 = T_cam1_cam0 maps cam0 points into cam1, which
is this package's `Camera.extrinsic` for cam1 when cam0 is the reference.

    python tools/kalibr_compare/run_tumvi.py DATASET_MAV0_DIR OUT.json [--edge-scale 0.93]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from aprilgrid import AprilGrid, make_detector  # noqa: E402

from mlti_cal.models.manifolds import matrix_to_pose, pose_to_matrix  # noqa: E402
from mlti_cal.problem.initialize import initialize_system  # noqa: E402
from mlti_cal.problem.reprojection import build_problem, write_back  # noqa: E402
from mlti_cal.problem.types import Board, CalibrationSystem, Camera, Observation  # noqa: E402
from mlti_cal.report.covariance import compute_covariance, marginal_covariance  # noqa: E402
from mlti_cal.solvers import SolveOptions, backend_status, get_backend  # noqa: E402

# Copied from TUM's pinhole-equi-512/camchain-imucam-imucalib.yaml.
KALIBR = {
    "cam0": [
        190.97847715128717,
        190.9733070521226,
        254.93170605935475,
        256.8974428996504,
        0.0034823894022493434,
        0.0007150348452162257,
        -0.0020532361418706202,
        0.00020293673591811182,
    ],
    "cam1": [
        190.44236969414825,
        190.4344384721956,
        252.59949716835982,
        254.91723064636983,
        0.0034003170790442797,
        0.001766278153469831,
        -0.00266312569781606,
        0.0003299517423931039,
    ],
    "T_cam1_cam0": [
        [0.9999994457734953, -0.0008233639921576076, -0.0006561436136444361, -0.10106110275180535],
        [0.0007916877528168117, 0.9988994619156757, -0.04689603624058988, -0.0019764575873431013],
        [0.0006940340102242531, 0.04689549078870055, 0.9988995601463054, -0.0011756424802046581],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "reprojection_mean_px": {"cam0": 0.0939591806842, "cam1": 0.0951201111853},
}
CAMERAS = ("cam0", "cam1")


def load8(path: Path) -> np.ndarray:
    """TUM VI PNGs are 16-bit; the top byte is the 8-bit image."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return (im >> 8).astype(np.uint8) if im.dtype == np.uint16 else im


def detect_all(mav0: Path, grid: AprilGrid, min_tags: int) -> dict:
    det = make_detector()
    p = det.getDetectorParameters()
    # Measured on this data with Kalibr's own intrinsics: SUBPIX leaves 1.16 px
    # of misfit against the grid, CONTOUR 0.46 px.
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    det.setDetectorParameters(p)
    n_tags = grid.cols * grid.rows
    out: dict = {c: {} for c in CAMERAS}
    for cam in CAMERAS:
        for f in sorted((mav0 / cam / "data").glob("*.png")):
            corners, ids, _ = det.detectMarkers(load8(f))
            if ids is None:
                continue
            keep = [(c, i) for c, i in zip(corners, ids.ravel(), strict=True) if 0 <= i < n_tags]
            if len(keep) < min_tags:
                continue
            pid = np.concatenate([4 * int(i) + np.arange(4) for _, i in keep])
            uv = np.concatenate([c.reshape(4, 2) for c, _ in keep]).astype(float)
            out[cam][f.stem] = (pid, uv)
    return out


def rms_per_camera(system: CalibrationSystem) -> dict:
    sq: dict = {c: [] for c in system.cameras}
    for o in system.observations:
        cam = system.cameras[o.camera]
        T = pose_to_matrix(cam.extrinsic) @ pose_to_matrix(system.board_poses[(o.frame, o.board)])
        X = system.boards[o.board].object_points[o.point_ids]
        uv = cam.model.project(cam.params, X @ T[:3, :3].T + T[:3, 3])
        sq[o.camera].append(np.sum((uv - o.image_points) ** 2, axis=1))
    return {c: float(np.sqrt(np.mean(np.concatenate(v)))) for c, v in sq.items()}


def kb_rays(params, uv: np.ndarray) -> np.ndarray:
    """Unit rays for pixels under a KB / OpenCV-fisheye model (theta < 90 deg)."""
    fx, fy, cx, cy, *d = params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    xy = cv2.fisheye.undistortPoints(uv[None].astype(np.float64), K, np.array(d))[0]
    v = np.column_stack([xy, np.ones(len(xy))])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def projection_difference(model, ours, ref, observed_uv: np.ndarray) -> dict:
    """Pixel difference ours vs Kalibr over the 32-px cells the board covered."""
    cell = 32
    occupied = {(int(u // cell), int(v // cell)) for u, v in observed_uv}
    q = np.array([[(i + 0.5) * cell, (j + 0.5) * cell] for i, j in sorted(occupied)])
    rays = kb_rays(ref, q)
    direct = np.linalg.norm(model.project(np.array(ours), rays) - q, axis=1)
    # Best-fit rotation between the two camera frames (Kabsch on the rays).
    ours_rays = kb_rays(ours, q)
    U, _, Vt = np.linalg.svd(ours_rays.T @ rays)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    aligned = np.linalg.norm(model.project(np.array(ours), rays @ R.T) - q, axis=1)
    return {
        "cells": len(q),
        "direct_median_px": float(np.median(direct)),
        "direct_max_px": float(direct.max()),
        "aligned_median_px": float(np.median(aligned)),
        "aligned_max_px": float(aligned.max()),
        "aligned_rotation_deg": float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))),
    }


def rms_at_kalibr(system: CalibrationSystem) -> dict:
    """
    Kalibr's cameras and T_cam1_cam0, frozen, on OUR corners: only the board
    poses are re-fitted. If this is clearly above our own RMS, the difference
    between the two calibrations is what our corners ask for, not a solver miss.
    """
    sub = copy.deepcopy(system)
    for c in CAMERAS:
        sub.cameras[c].params = np.array(KALIBR[c], dtype=float)
    sub.cameras["cam1"].extrinsic = matrix_to_pose(np.array(KALIBR["T_cam1_cam0"]))
    problem = build_problem(sub, optimize_intrinsics=False, optimize_extrinsics=False)
    get_backend("scipy").solve(problem, SolveOptions(max_iterations=200))
    write_back(sub, problem)
    return rms_per_camera(sub)


def solve(
    dets: dict, grid: AprilGrid, model_name: str
) -> tuple[CalibrationSystem, object, object, dict]:
    system = CalibrationSystem()
    for c in CAMERAS:
        system.add_camera(Camera(id=c, model_name=model_name, image_size=(512, 512)))
    system.set_reference_camera("cam0")
    system.add_board(Board(id="grid", object_points=grid.object_points(), kind="aprilgrid"))
    for cam in CAMERAS:
        for frame, (pid, uv) in dets[cam].items():
            system.add_observation(Observation(frame, cam, "grid", pid, uv, sigma=1.0))
    initialize_system(system)
    backend = "ceres" if backend_status()["ceres"][0] else "scipy"
    problem = build_problem(system)
    t0 = time.perf_counter()
    result = get_backend(backend).solve(problem, SolveOptions(max_iterations=500))
    info = {
        "backend": backend,
        "success": bool(result.success),
        "iterations": int(result.iterations),
        "solve_s": time.perf_counter() - t0,
    }
    write_back(system, problem)
    cov = compute_covariance(problem, pixel_noise_std=None)
    return system, problem, cov, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("mav0", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--edge-scale",
        type=float,
        default=1.0,
        help="detected tag edge / printed tag edge (detector bias; 1.0 = nominal)",
    )
    ap.add_argument("--min-tags", type=int, default=4)
    ap.add_argument("--models", nargs="+", default=["fisheye_kb", "double_sphere", "eucm"])
    args = ap.parse_args()

    grid = AprilGrid(edge_scale=args.edge_scale)
    dets = detect_all(args.mav0, grid, args.min_tags)
    both = sorted(set(dets["cam0"]) & set(dets["cam1"]))
    n_corners = sum(len(v[0]) for c in CAMERAS for v in dets[c].values())
    print(
        f"detections: cam0 {len(dets['cam0'])}, cam1 {len(dets['cam1'])} images, "
        f"{len(both)} seen by both; corners {n_corners}"
    )

    T_ref = np.array(KALIBR["T_cam1_cam0"])
    results: dict = {
        "dataset": str(args.mav0),
        "edge_scale": args.edge_scale,
        "images_with_detections": {c: len(dets[c]) for c in CAMERAS},
        "frames_seen_by_both": len(both),
        "kalibr": KALIBR,
        "models": {},
    }
    for model_name in args.models:
        system, problem, cov, info = solve(dets, grid, model_name)
        rms = rms_per_camera(system)
        entry: dict = {
            "solve": info,
            "rms_px": rms,
            "sigma_residual_px": float(np.sqrt(cov.sigma2_residual)),
        }
        T = pose_to_matrix(system.cameras["cam1"].extrinsic)
        t_cov, _, _ = marginal_covariance(cov, "extr:cam1")
        u = T[:3, 3] / np.linalg.norm(T[:3, 3])
        entry["baseline_m"] = float(np.linalg.norm(T[:3, 3]))
        entry["baseline_sigma_m"] = float(np.sqrt(u @ t_cov[:3, :3] @ u))
        entry["rotation_sigma_deg"] = float(np.degrees(np.sqrt(np.trace(t_cov[3:, 3:]))))
        entry["intrinsics"] = {c: system.cameras[c].params.tolist() for c in CAMERAS}
        entry["intrinsics_sigma"] = {
            c: [
                cov.std_error_of(f"intr:{c}.{i}") for i in range(system.cameras[c].model.num_params)
            ]
            for c in CAMERAS
        }
        entry["param_names"] = list(system.cameras["cam0"].model.param_names)
        if model_name == "fisheye_kb":
            dR = T[:3, :3] @ T_ref[:3, :3].T
            entry["vs_kalibr"] = {
                "baseline_diff_m": entry["baseline_m"] - float(np.linalg.norm(T_ref[:3, 3])),
                "camera_centre_diff_m": float(
                    np.linalg.norm(-T[:3, :3].T @ T[:3, 3] + T_ref[:3, :3].T @ T_ref[:3, 3])
                ),
                "rotation_diff_deg": float(np.degrees(np.linalg.norm(cv2.Rodrigues(dR)[0]))),
                "intrinsics_diff_sigma": {
                    c: (
                        (np.array(entry["intrinsics"][c]) - np.array(KALIBR[c]))
                        / np.array(entry["intrinsics_sigma"][c])
                    ).tolist()
                    for c in CAMERAS
                },
                "projection": {},
            }
            entry["vs_kalibr"]["rms_at_kalibr_px"] = rms_at_kalibr(system)
            for c in CAMERAS:
                uv = np.concatenate([v[1] for v in dets[c].values()])
                entry["vs_kalibr"]["projection"][c] = projection_difference(
                    system.cameras[c].model, entry["intrinsics"][c], KALIBR[c], uv
                )
        results["models"][model_name] = entry
        print(
            f"{model_name:14s} RMS cam0 {rms['cam0']:.4f} cam1 {rms['cam1']:.4f} px, "
            f"baseline {entry['baseline_m'] * 1000:.3f} "
            f"+- {entry['baseline_sigma_m'] * 1000:.3f} mm, "
            f"{info['iterations']} it, {info['solve_s']:.1f} s, success={info['success']}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
