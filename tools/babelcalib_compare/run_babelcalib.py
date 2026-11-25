"""
mlti_cal on BabelCalib's public fisheye benchmark (the `Kalibr` aggregate).

Data: https://github.com/ylochman/babelcalib/releases/tag/datasets (Kalibr.zip).
Eight cameras (EuRoC, TUM VI, GoPro, ENTANIYA 250 degrees, four Basler/Kalibr
lenses), each a 6 x 6 AprilGrid (tag 0.088 m) with the corners already
detected and split into `train/` and `test/`. Detection is therefore not part of
this comparison: every method sees the same corners.

Protocol, as in the paper (Lochman et al., ICCV 2021, section 5.1): calibrate on
`train`, freeze the intrinsics, fit only the board pose of each `test` image,
report the reprojection RMS. The paper's number is a robust, weighted RMS whose
scale is not stated, so two numbers are reported here: the plain RMS over all
test corners, and a Huber-weighted RMS (scale 1 px). They are not the paper's
estimator, so the paper's table is a reference, not a like-for-like score.

Conventions: pixel (0, 0) is the corner of the image as in the corner files,
OpenCV camera frame, target points from `aprilgrid_6x6.dsc` (corner ids are the
running index of the corner lines), z = 0.

    python tools/babelcalib_compare/run_babelcalib.py Output/babelcalib/Kalibr OUT.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from mlti_cal.models.manifolds import pose_to_matrix
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import build_problem, write_back
from mlti_cal.problem.types import Board, CalibrationSystem, Camera, Observation
from mlti_cal.solvers import SolveOptions, backend_status, get_backend

MODELS = ["fisheye_kb", "double_sphere", "eucm", "fov"]
# Table 5 of the paper, row "Kalibr" (mean test RMS over the eight cameras, px).
PAPER = {
    "fisheye_kb": {"Kalibr": 0.194, "OpenCV": 2.291, "BabelCalib": 0.173},
    "eucm": {"Kalibr": 0.250, "BabelCalib": 0.230},
    "fov": {"Kalibr": 0.257, "BabelCalib": 0.237},
    "double_sphere": {"Kalibr": 0.264, "BabelCalib": 0.326},
}


def load_target(dsc: Path) -> np.ndarray:
    """Corner coordinates in file order; the line index is the corner id."""
    pts = []
    for line in dsc.read_text().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 6 and f[0] == "-1":
            pts.append([float(f[3]), float(f[4]), float(f[5])])
    return np.array(pts)


def load_corners(path: Path) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    head = {k: v.strip() for k, v in (ln.split(":", 1) for ln in lines[:5])}
    ids, uv = [], []
    for ln in lines[5:]:
        f = ln.split(",")
        if len(f) >= 5:
            ids.append(int(f[1]))
            uv.append([float(f[3]), float(f[4])])
    return (int(head["width"]), int(head["height"])), np.array(ids), np.array(uv)


def load_split(folder: Path, n_points: int) -> tuple[tuple[int, int] | None, dict]:
    size, frames = None, {}
    for f in sorted(folder.glob("*.orpc")):
        size, ids, uv = load_corners(f)
        keep = ids < n_points
        if keep.sum() >= 8:
            frames[f.stem] = (ids[keep], uv[keep])
    return size, frames


def make_system(model: str, size, pts: np.ndarray, frames: dict) -> CalibrationSystem:
    system = CalibrationSystem()
    system.add_camera(Camera(id="cam", model_name=model, image_size=size))
    system.set_reference_camera("cam")
    system.add_board(Board(id="grid", object_points=pts, kind="aprilgrid"))
    for name, (ids, uv) in frames.items():
        system.add_observation(Observation(name, "cam", "grid", ids, uv, sigma=1.0))
    return system


def corner_errors(system: CalibrationSystem) -> np.ndarray:
    out = []
    cam = system.cameras["cam"]
    for o in system.observations:
        T = pose_to_matrix(cam.extrinsic) @ pose_to_matrix(system.board_poses[(o.frame, o.board)])
        X = system.boards[o.board].object_points[o.point_ids]
        uv = cam.model.project(cam.params, X @ T[:3, :3].T + T[:3, 3])
        out.append(np.linalg.norm(uv - o.image_points, axis=1))
    return np.concatenate(out)


def summarise(e: np.ndarray) -> dict:
    w = np.minimum(1.0, 1.0 / np.maximum(e, 1e-12))
    return {
        "rms_px": float(np.sqrt(np.mean(e**2))),
        "huber_rms_px": float(np.sqrt(np.sum(w * e**2) / np.sum(w))),
        "over_2px_pct": float(100.0 * np.mean(e > 2.0)),
        "corners": int(e.size),
    }


def solve_ours(model: str, size, pts, train: dict, test: dict, backend: str) -> dict:
    tr = make_system(model, size, pts, train)
    t0 = time.perf_counter()
    initialize_system(tr)
    problem = build_problem(tr)
    res = get_backend(backend).solve(problem, SolveOptions(max_iterations=500))
    write_back(tr, problem)
    secs = time.perf_counter() - t0
    train_stats = summarise(corner_errors(tr))

    te = make_system(model, size, pts, test)
    te.cameras["cam"].params = tr.cameras["cam"].params.copy()
    initialize_system(te, do_intrinsics=False)
    prob = build_problem(te, optimize_intrinsics=False, optimize_extrinsics=False)
    get_backend("scipy").solve(prob, SolveOptions(max_iterations=200))
    write_back(te, prob)
    return {
        "success": bool(res.success),
        "seconds": secs,
        "params": tr.cameras["cam"].params.tolist(),
        "train": train_stats,
        "test": summarise(corner_errors(te)),
    }


def solve_opencv_kb(size, pts, train: dict, test: dict) -> dict:
    """OpenCV's fisheye (Kannala-Brandt) calibration, same corners, same protocol."""
    obj = [pts[ids][None].astype(np.float32) for ids, _ in train.values()]
    img = [uv[None].astype(np.float32) for _, uv in train.values()]
    K, D = np.eye(3), np.zeros((4, 1))
    flags = cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10)
    rms, K, D, _, _ = cv2.fisheye.calibrate(obj, img, size, K, D, flags=flags, criteria=crit)
    errs = []
    for ids, uv in test.values():
        n = cv2.fisheye.undistortPoints(uv[None].astype(np.float64), K, D).reshape(-1, 2)
        ok, rv, tv = cv2.solvePnP(pts[ids], n, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.fisheye.projectPoints(pts[ids][None], rv, tv, K, D)
        errs.append(np.linalg.norm(proj.reshape(-1, 2) - uv, axis=1))
    return {"train_rms_opencv_px": float(rms), "test": summarise(np.concatenate(errs))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--lenses", nargs="+", default=None)
    args = ap.parse_args()

    pts = load_target(args.root / "aprilgrid_6x6.dsc")
    backend = "ceres" if backend_status()["ceres"][0] else "scipy"
    lenses = args.lenses or sorted(p.name for p in args.root.iterdir() if p.is_dir())
    print(f"{len(pts)} target corners, backend {backend}, lenses {lenses}")
    results: dict = {"backend": backend, "target_corners": len(pts), "lenses": {}}
    for lens in lenses:
        size, train = load_split(args.root / lens / "train", len(pts))
        _, test = load_split(args.root / lens / "test", len(pts))
        entry: dict = {"size": size, "train_images": len(train), "test_images": len(test)}
        try:
            entry["opencv_kb"] = solve_opencv_kb(size, pts, train, test)
        except cv2.error as exc:
            entry["opencv_kb"] = {"failed": str(exc).splitlines()[0][:120]}
        for model in args.models:
            try:
                entry[model] = solve_ours(model, size, pts, train, test, backend)
            except Exception as exc:  # recorded, not hidden: a failure is a result
                entry[model] = {"failed": f"{type(exc).__name__}: {str(exc)[:160]}"}
        results["lenses"][lens] = entry
        row = [f"{lens:14s} {size[0]}x{size[1]} {len(train)}+{len(test)} img"]
        for key in ["opencv_kb", *args.models]:
            r = entry[key]
            row.append(f"{key} {r['test']['rms_px']:.3f}" if "test" in r else f"{key} FAIL")
        print(" | ".join(row), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
