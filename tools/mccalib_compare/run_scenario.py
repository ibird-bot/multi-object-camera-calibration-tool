"""
mlti_cal vs MC-Calib's published synthetic multi-camera data, against ground truth.

Dataset: the `Blender_Images` set released with MC-Calib (Rameau et al., CVIU
2022, https://github.com/rameau-fr/MC-Calib). Scenario_2 is five cameras fanned
by 15 degrees steps, so only neighbouring cameras share the single 5 x 5 Charuco
board: the rig is solvable only by chaining. Every scenario ships its
`GroundTruth.yml` and the result MC-Calib itself produced (`Results/`).

Compared, per camera:
  * intrinsics against the ground truth, in pixels and in this tool's 1-sigma;
  * T_cam_i<-cam0 against the ground truth: rotation, camera-centre distance,
    and the 6-dof Mahalanobis distance of the error under this tool's own
    covariance (about 6 per camera if the covariance is honest);
  * the same pose and intrinsics numbers for MC-Calib's saved result;
  * the RMS of each calibration on identical corners (this tool's detections),
    re-fitting only the board poses.

Conventions
  * Blender camera frame is +x right, +y up, camera looks down -z. The ground
    truth `P_i` is the transform the script applies as
    `world_from_cam_i = world_from_cam_1 * inverse(P_i)`, so `P_i` is
    T_cam_i<-cam_1 in the Blender frame. It becomes OpenCV's frame (+y down,
    +z forward) by F P F with F = diag(1, -1, -1, 1).
  * MC-Calib's `camera_pose_matrix` is T_cam0<-cam_i. Checked, not assumed: the
    camera centre it gives, (0.9917, 0, -0.1626) m for camera 1 of Scenario 2, is
    exactly -R^T t of the converted ground truth.
  * This tool: camera 0 is the reference, `Camera.extrinsic` = T_cam_i<-cam0.

Board (not published; measured from the images): 5 x 5 squares of 0.192 m (the
corner spacing in MC-Calib's saved object), DICT_6X6, marker ids 0..11, marker
side 0.75 x square (measured as the ratio of marker side to square pitch over 10
fully-seen frames, median 0.744).

    python tools/mccalib_compare/run_scenario.py SCENARIO_DIR OUT.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mrcal_compare"))
from run_mlti_multicam import rms_px  # noqa: E402

from mlti_cal.detectors.charuco import CharucoBoardSpec, CharucoDetector  # noqa: E402
from mlti_cal.models.manifolds import (  # noqa: E402
    matrix_to_pose,
    pose_to_matrix,
    so3_log,
)
from mlti_cal.problem.initialize import initialize_system  # noqa: E402
from mlti_cal.problem.reprojection import build_problem, write_back  # noqa: E402
from mlti_cal.problem.types import Board, CalibrationSystem, Camera, Observation  # noqa: E402
from mlti_cal.report.covariance import compute_covariance, marginal_covariance  # noqa: E402
from mlti_cal.solvers import SolveOptions, backend_status, get_backend  # noqa: E402

F = np.diag([1.0, -1.0, -1.0, 1.0])


def read_mat(fs: cv2.FileStorage, name: str) -> np.ndarray:
    return np.asarray(fs.getNode(name).mat(), dtype=float)


def load_truth(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(K_i, T_cam_i<-cam_1 in the OpenCV frame) for every camera."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    n = int(fs.getNode("nb_camera").real())
    K = [read_mat(fs, f"K_{i + 1}") for i in range(n)]
    T = [F @ read_mat(fs, f"P_{i + 1}") @ F for i in range(n)]
    fs.release()
    return K, T


def load_mccalib(path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """(K_i, distortion_i, T_cam_i<-cam0) from MC-Calib's calibrated_cameras_data.yml."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    n = int(fs.getNode("nb_camera").real())
    K, D, T = [], [], []
    for i in range(n):
        node = fs.getNode(f"camera_{i}")
        K.append(np.asarray(node.getNode("camera_matrix").mat(), dtype=float))
        D.append(np.asarray(node.getNode("distortion_vector").mat(), dtype=float).ravel())
        T.append(np.linalg.inv(np.asarray(node.getNode("camera_pose_matrix").mat(), dtype=float)))
    fs.release()
    return K, D, T


def params_of(K: np.ndarray, d: np.ndarray) -> np.ndarray:
    """pinhole_radtan layout: fx fy cx cy k1 k2 p1 p2 k3 (OpenCV's order)."""
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *d[:5]], dtype=float)


def detect(
    root: Path, specs: list[CharucoBoardSpec], stride: int = 1
) -> tuple[dict, tuple[int, int], dict]:
    detectors = [CharucoDetector(s) for s in specs]
    base = root / "Images" if (root / "Images").is_dir() else root
    cams = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("Cam_"))
    obs: dict[str, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]] = {}
    size = (0, 0)
    counts = {}
    for k, cam in enumerate(cams):
        files = sorted(cam.glob("*.png"))[::stride]
        found = 0
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            size = (img.shape[1], img.shape[0])
            hit = False
            for det_obj in detectors:
                det = det_obj.detect(img)
                if det is not None:
                    key = (f.stem, det.board_id)
                    obs.setdefault(f"cam{k}", {})[key] = (det.point_ids, det.image_points)
                    hit = True
            found += hit
        counts[f"cam{k}"] = {"images": len(files), "detected": found}
    return obs, size, counts


def build(obs: dict, size: tuple[int, int], object_points: dict) -> CalibrationSystem:
    names = sorted(obs)
    system = CalibrationSystem()
    for n in names:
        system.add_camera(Camera(id=n, model_name="pinhole_radtan", image_size=size))
    system.set_reference_camera(names[0])
    for bid, pts in object_points.items():
        system.add_board(Board(id=bid, object_points=pts, kind="charuco"))
    for n in names:
        for (frame, bid), (pid, uv) in obs[n].items():
            system.add_observation(Observation(frame, n, bid, pid, uv, sigma=1.0))
    return system


def rms_with_cameras(
    system: CalibrationSystem, params: list[np.ndarray], poses: list[np.ndarray]
) -> dict:
    """The given cameras, frozen, on these corners; only the board poses are re-fitted."""
    sub = copy.deepcopy(system)
    for i, n in enumerate(sorted(sub.cameras)):
        sub.cameras[n].params = params[i].copy()
        sub.cameras[n].extrinsic = matrix_to_pose(poses[i])
    problem = build_problem(sub, optimize_intrinsics=False, optimize_extrinsics=False)
    get_backend("scipy").solve(problem, SolveOptions(max_iterations=200))
    write_back(sub, problem)
    total, per = rms_px(sub)
    return {"rms_px": total, "per_camera": per}


def pose_error(T_hat: np.ndarray, T_ref: np.ndarray) -> dict:
    """Rotation (deg) and camera-centre (m) difference between two T_cam<-cam0."""
    Rh, th, Rr, tr = T_hat[:3, :3], T_hat[:3, 3], T_ref[:3, :3], T_ref[:3, 3]
    phi = so3_log(Rr.T @ Rh)  # right perturbation, the retraction this tool uses
    return {
        "rotation_deg": float(np.degrees(np.linalg.norm(phi))),
        "centre_m": float(np.linalg.norm(-Rh.T @ th + Rr.T @ tr)),
        "_dt": (th - tr).tolist(),
        "_phi": phi.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("scenario", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--squares", type=int, nargs=2, default=[5, 5])
    ap.add_argument("--square-length", type=float, default=0.192)
    ap.add_argument("--boards", type=int, default=1, help="boards, marker ids in blocks")
    ap.add_argument("--marker-ratio", type=float, default=0.75)
    ap.add_argument("--dictionary", default="DICT_6X6_250")
    args = ap.parse_args()

    stride = args.squares[0] * args.squares[1] // 2  # markers per board
    specs = [
        CharucoBoardSpec(
            id=f"b{k}",
            squares_x=args.squares[0],
            squares_y=args.squares[1],
            square_length=args.square_length,
            marker_length=args.square_length * args.marker_ratio,
            dictionary=args.dictionary,
            marker_id_offset=k * stride,
        )
        for k in range(args.boards)
    ]
    obs, size, counts = detect(args.scenario, specs)
    print("detections:", counts)
    object_points = {s.id: CharucoDetector(s).object_points for s in specs}

    system = build(obs, size, object_points)
    initialize_system(system)
    rms_init, _ = rms_px(system)
    backend = "ceres" if backend_status()["ceres"][0] else "scipy"
    problem = build_problem(system)
    result = get_backend(backend).solve(problem, SolveOptions(max_iterations=500))
    write_back(system, problem)
    rms, rms_per = rms_px(system)
    print(f"mlti_cal ({backend}): RMS {rms_init:.3f} -> {rms:.4f} px, success={result.success}")

    cov = compute_covariance(problem, pixel_noise_std=None)
    names = sorted(system.cameras)
    K_gt, T_gt1 = load_truth(args.scenario / "GroundTruth.yml")
    # ground truth relative to camera 0 is already T_cam_i<-cam_1
    K_mc, D_mc, T_mc = load_mccalib(args.scenario / "Results" / "calibrated_cameras_data.yml")

    out: dict = {
        "backend": backend,
        "detections": counts,
        "rms_init_px": rms_init,
        "rms_px": rms,
        "rms_per_camera_px": rms_per,
        "sigma_residual_px": float(np.sqrt(cov.sigma2_residual)),
        "cameras": {},
    }
    for i, n in enumerate(names):
        cam = system.cameras[n]
        p = cam.params
        sig = [cov.std_error_of(f"intr:{n}.{j}") for j in range(cam.model.num_params)]
        entry: dict = {
            "params": p.tolist(),
            "sigma": sig,
            "truth_fx_fy_cx_cy": [K_gt[i][0, 0], K_gt[i][1, 1], K_gt[i][0, 2], K_gt[i][1, 2]],
            "mccalib_fx_fy_cx_cy": [K_mc[i][0, 0], K_mc[i][1, 1], K_mc[i][0, 2], K_mc[i][1, 2]],
            "mccalib_distortion": D_mc[i].tolist(),
        }
        truth4 = np.array(entry["truth_fx_fy_cx_cy"])
        mc4 = np.array(entry["mccalib_fx_fy_cx_cy"])
        entry["mlti_err_px"] = (p[:4] - truth4).tolist()
        entry["mlti_err_sigma"] = ((p[:4] - truth4) / np.array(sig[:4])).tolist()
        entry["mccalib_err_px"] = (mc4 - truth4).tolist()
        if i > 0:
            T_hat = pose_to_matrix(cam.extrinsic)
            e = pose_error(T_hat, T_gt1[i])
            t_cov, _, _ = marginal_covariance(cov, f"extr:{n}")
            err6 = np.array(e["_dt"] + e["_phi"])
            e["mahalanobis2_6dof"] = float(err6 @ np.linalg.solve(t_cov, err6))
            e["sigma_centre_like_t_m"] = np.sqrt(np.diag(t_cov)[:3]).tolist()
            e["rotation_sigma_deg"] = float(np.degrees(np.sqrt(np.trace(t_cov[3:, 3:]))))
            entry["mlti_pose"] = {k: v for k, v in e.items() if not k.startswith("_")}
            m = pose_error(T_mc[i], T_gt1[i])
            entry["mccalib_pose"] = {k: v for k, v in m.items() if not k.startswith("_")}
        out["cameras"][n] = entry

    # RMS of three calibrations on identical corners; only board poses re-fitted.
    gt_params = [params_of(K_gt[i], np.zeros(5)) for i in range(len(names))]
    mc_params = [params_of(K_mc[i], D_mc[i]) for i in range(len(names))]
    out["rms_at_truth"] = rms_with_cameras(system, gt_params, T_gt1)
    out["rms_at_mccalib"] = rms_with_cameras(system, mc_params, T_mc)
    # Truth with only fy replaced by this tool's estimate: if that alone closes
    # the gap, the published fy is the one inconsistent number in the truth.
    gt_fy = []
    for i, n in enumerate(names):
        q = gt_params[i].copy()
        q[1] = system.cameras[n].params[1]
        gt_fy.append(q)
    out["rms_at_truth_with_our_fy"] = rms_with_cameras(system, gt_fy, T_gt1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(
        f"\nRMS on these corners: mlti {rms:.4f}  truth {out['rms_at_truth']['rms_px']:.4f}  "
        f"MC-Calib {out['rms_at_mccalib']['rms_px']:.4f}  "
        f"truth with our fy {out['rms_at_truth_with_our_fy']['rms_px']:.4f}"
    )
    for n, c in out["cameras"].items():
        print(
            f"{n}: fx fy cx cy err mlti {np.round(c['mlti_err_px'], 3)}  "
            f"(sigma units {np.round(c['mlti_err_sigma'], 1)})  "
            f"MC-Calib {np.round(c['mccalib_err_px'], 3)}"
        )
        if "mlti_pose" in c:
            a, b = c["mlti_pose"], c["mccalib_pose"]
            print(
                f"     pose: mlti {a['rotation_deg']:.4f} deg {a['centre_m'] * 1000:.2f} mm "
                f"(Mahalanobis^2 {a['mahalanobis2_6dof']:.1f}) | MC-Calib "
                f"{b['rotation_deg']:.4f} deg {b['centre_m'] * 1000:.2f} mm"
            )


if __name__ == "__main__":
    main()
