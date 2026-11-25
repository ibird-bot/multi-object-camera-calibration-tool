"""
mlti_cal side of the MULTI-camera mrcal comparison, on real images.

1. Detects a chessboard in every image with this package's own detector.
2. Writes the corners to CORNERS.json -- the exact input run_mrcal_multicam.py
   reads, so both tools solve from identical pixels.
3. Solves the rig with mlti_cal (cameras + extrinsics + board poses jointly).
4. Solves it again with OpenCV (calibrateCamera per camera, then
   stereoCalibrate with intrinsics free) as a second, independent reference.

Board frame: corner k = i * cols + j sits at (j * spacing, i * spacing, 0) --
mrcal's `ref_calibration_object` layout, so board poses transfer unchanged.

Conventions written to the JSON (identical to mrcal's):
  * camera frame OpenCV: +x right, +y down, +z forward;
  * camera 0 is the reference (rig) frame;
  * `extrinsics_rt_fromref[i-1]` = [rodrigues, t] of T_cam_i<-cam0;
  * `frames_rt_toref[f]` = [rodrigues, t] of T_cam0<-board for frame f.

    python tools/mrcal_compare/run_mlti_multicam.py --images DIR \
        --camera left "left*.jpg" --camera right "right*.jpg" \
        --cols 9 --rows 6 --spacing 0.025 OUTDIR
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from mlti_cal.detectors.checkerboard import CheckerboardDetector, CheckerboardSpec
from mlti_cal.models.manifolds import pose_to_matrix, pose_to_rt
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import build_problem, write_back
from mlti_cal.problem.types import Board, CalibrationSystem, Camera, Observation
from mlti_cal.report.covariance import compute_covariance
from mlti_cal.report.uncertainty import projection_uncertainty
from mlti_cal.solvers import SolveOptions, backend_status, get_backend


def frame_key(path: Path) -> str:
    """The digits in a file name, so left07.jpg and right07.jpg pair up."""
    digits = re.findall(r"\d+", path.stem)
    return digits[-1] if digits else path.stem


def detect(args) -> dict:
    spec = CheckerboardSpec(
        id="board",
        squares_x=args.cols + 1,
        squares_y=args.rows + 1,
        square_length=args.spacing,
    )
    detector = CheckerboardDetector(spec)
    cameras, observations = [], []
    for name, pattern in args.camera:
        files = sorted(Path(args.images).glob(pattern))
        if not files:
            raise SystemExit(f"no images match {pattern!r} in {args.images}")
        size = None
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            size = (img.shape[1], img.shape[0])
            det = detector.detect(img)
            if det is not None:
                observations.append(
                    {"frame": frame_key(f), "camera": name, "corners": det.image_points.tolist()}
                )
        cameras.append({"name": name, "image_size": size, "images": len(files)})
    return {
        "cameras": cameras,
        "cols": args.cols,
        "rows": args.rows,
        "spacing": args.spacing,
        "observations": observations,
    }


def object_points(cols: int, rows: int, spacing: float) -> np.ndarray:
    j, i = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.column_stack([j.ravel(), i.ravel(), np.zeros(j.size)]) * spacing


def rms_px(system: CalibrationSystem) -> tuple[float, dict]:
    """Overall and per-camera sqrt(mean(dx^2 + dy^2))."""
    sq: dict[str, list] = {c: [] for c in system.cameras}
    for o in system.observations:
        cam = system.cameras[o.camera]
        T = pose_to_matrix(cam.extrinsic) @ pose_to_matrix(system.board_poses[(o.frame, o.board)])
        X = system.boards[o.board].object_points[o.point_ids]
        uv = cam.model.project(cam.params, X @ T[:3, :3].T + T[:3, 3])
        sq[o.camera].append(np.sum((uv - o.image_points) ** 2, axis=1))
    per = {c: float(np.sqrt(np.mean(np.concatenate(v)))) for c, v in sq.items()}
    allv = np.concatenate([np.concatenate(v) for v in sq.values()])
    return float(np.sqrt(np.mean(allv))), per


def solve_mlti(data: dict, model: str, grid_step: int) -> dict:
    names = [c["name"] for c in data["cameras"]]
    system = CalibrationSystem()
    for c in data["cameras"]:
        system.add_camera(Camera(id=c["name"], model_name=model, image_size=tuple(c["image_size"])))
    system.set_reference_camera(names[0])
    pts = object_points(data["cols"], data["rows"], data["spacing"])
    system.add_board(Board(id="board", object_points=pts, kind="checkerboard"))
    ids = np.arange(len(pts))
    for o in data["observations"]:
        system.add_observation(
            Observation(o["frame"], o["camera"], "board", ids, np.array(o["corners"]), sigma=1.0)
        )

    initialize_system(system)
    rms_init, _ = rms_px(system)
    backend = "ceres" if backend_status()["ceres"][0] else "scipy"
    problem = build_problem(system)
    t0 = time.perf_counter()
    result = get_backend(backend).solve(problem, SolveOptions(max_iterations=500))
    solve_s = time.perf_counter() - t0
    write_back(system, problem)
    rms, rms_per = rms_px(system)

    cov = compute_covariance(problem, pixel_noise_std=None)
    frames = sorted({f for (f, _b) in system.board_poses})
    dist = [float(np.linalg.norm(system.board_poses[(f, "board")][:3])) for f in frames]
    ranges = [round(float(np.median(dist)), 3), 100.0 * float(np.median(dist))]
    unc: dict = {}
    for name in names:
        unc[name] = {}
        for rng in ranges:
            m = projection_uncertainty(problem, system, cov, name, rng, grid_step=grid_step)
            ok = m.valid.ravel()
            unc[name][repr(rng)] = {
                "pixels": np.column_stack([m.grid_u.ravel(), m.grid_v.ravel()])[ok].tolist(),
                "sigma_max": m.sigma_max.ravel()[ok].tolist(),
            }
    return {
        "backend": backend,
        "success": bool(result.success),
        "iterations": int(result.iterations),
        "solve_s": solve_s,
        "rms_init_px": rms_init,
        "rms_px": rms,
        "rms_per_camera_px": rms_per,
        "param_names": list(system.cameras[names[0]].model.param_names),
        "intrinsics": {n: system.cameras[n].params.tolist() for n in names},
        "intrinsics_sigma": {
            n: [
                cov.std_error_of(f"intr:{n}.{i}") for i in range(system.cameras[n].model.num_params)
            ]
            for n in names
        },
        "extrinsics_rt_fromref": [
            np.concatenate(pose_to_rt(system.cameras[n].extrinsic)).tolist() for n in names[1:]
        ],
        "frames": frames,
        "frames_rt_toref": [
            np.concatenate(pose_to_rt(system.board_poses[(f, "board")])).tolist() for f in frames
        ],
        "board_distance": [min(dist), float(np.median(dist)), max(dist)],
        "uncertainty": unc,
    }


def solve_opencv(data: dict) -> dict | None:
    """calibrateCamera per camera, then stereoCalibrate on shared frames (2 cameras only)."""
    names = [c["name"] for c in data["cameras"]]
    if len(names) != 2:
        return None
    pts = object_points(data["cols"], data["rows"], data["spacing"]).astype(np.float32)
    by = {
        (o["frame"], o["camera"]): np.array(o["corners"], np.float32) for o in data["observations"]
    }
    size = tuple(data["cameras"][0]["image_size"])
    mono = {}
    for n in names:
        imgp = [v for (f, c), v in sorted(by.items()) if c == n]
        rms, K, d, _, _ = cv2.calibrateCamera([pts] * len(imgp), imgp, size, None, None)
        mono[n] = (K, d, rms)
    shared = sorted({f for (f, c) in by if c == names[0]} & {f for (f, c) in by if c == names[1]})
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-12)
    rms, K0, d0, K1, d1, R, T, _, _ = cv2.stereoCalibrate(
        [pts] * len(shared),
        [by[(f, names[0])] for f in shared],
        [by[(f, names[1])] for f in shared],
        mono[names[0]][0].copy(),
        mono[names[0]][1].copy(),
        mono[names[1]][0].copy(),
        mono[names[1]][1].copy(),
        size,
        flags=flags,
        criteria=crit,
    )

    def as_params(K, d):
        d = d.ravel()
        return [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *d[:5].tolist()]

    return {
        "shared_frames": shared,
        # As OpenCV returns it, over the shared frames only -- not directly
        # comparable with the other tools' RMS, which use every observation.
        "stereo_rms_px": float(rms),
        "intrinsics": {names[0]: as_params(K0, d0), names[1]: as_params(K1, d1)},
        "extrinsics_rt_fromref": [
            np.concatenate([cv2.Rodrigues(R)[0].ravel(), T.ravel()]).tolist()
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--images", required=True)
    ap.add_argument("--camera", nargs=2, action="append", metavar=("NAME", "GLOB"), required=True)
    ap.add_argument("--cols", type=int, required=True, help="inner corners per row")
    ap.add_argument("--rows", type=int, required=True, help="inner corners per column")
    ap.add_argument("--spacing", type=float, required=True, help="corner spacing, metres")
    ap.add_argument("--model", default="pinhole_radtan")
    ap.add_argument("--grid-step", type=int, default=40)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    data = detect(args)
    (args.outdir / "corners.json").write_text(json.dumps(data))
    for c in data["cameras"]:
        n = sum(o["camera"] == c["name"] for o in data["observations"])
        print(f"{c['name']}: board found in {n}/{c['images']} images")

    out = {
        "model": args.model,
        "dataset": data["cameras"],
        "mlti": solve_mlti(data, args.model, args.grid_step),
    }
    out["opencv"] = solve_opencv(data) if args.model == "pinhole_radtan" else None
    (args.outdir / "mlti.json").write_text(json.dumps(out, indent=1))
    m = out["mlti"]
    print(
        f"mlti_cal ({m['backend']}): RMS {m['rms_init_px']:.3f} -> {m['rms_px']:.4f} px "
        f"{m['rms_per_camera_px']}, success={m['success']}, {m['solve_s']:.2f} s"
    )


if __name__ == "__main__":
    main()
