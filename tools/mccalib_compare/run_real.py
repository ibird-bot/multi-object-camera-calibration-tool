"""
mlti_cal vs MC-Calib on MC-Calib's REAL multi-camera sequence (no ground truth).

Dataset: `Real_Images/Seq02_Overlapping_multicamera` from the MC-Calib release
(Rameau et al., CVIU 2022): six cameras, 1280 x 720, one 5 x 5 Charuco board
(6 x 6 dictionary, marker ids from 144, square 9.15 in the dataset's own unit),
917 frames per camera, and the calibration MC-Calib produced for it.

There is no ground truth, so the comparison is between the two tools:
  * RMS of each calibration on identical corners (this tool's detections; the
    other calibration's cameras are frozen, only board poses re-fitted);
  * intrinsics difference in units of this tool's 1-sigma;
  * T_cam_i<-cam0 difference: rotation, and camera-centre distance, the latter
    also in units of this tool's sigma (6-dof Mahalanobis distance).

Conventions as in run_scenario.py. Translations are in the board's unit.

    python tools/mccalib_compare/run_real.py SEQUENCE_DIR OUT.json [--unit-scale 1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scenario import (  # noqa: E402
    build,
    detect,
    load_mccalib,
    params_of,
    pose_error,
    rms_px,
    rms_with_cameras,
)

from mlti_cal.detectors.charuco import CharucoBoardSpec, CharucoDetector  # noqa: E402
from mlti_cal.models.manifolds import pose_to_matrix  # noqa: E402
from mlti_cal.problem.initialize import initialize_system  # noqa: E402
from mlti_cal.problem.reprojection import build_problem, write_back  # noqa: E402
from mlti_cal.report.covariance import compute_covariance, marginal_covariance  # noqa: E402
from mlti_cal.solvers import SolveOptions, backend_status, get_backend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("sequence", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--square", type=float, default=9.15)
    ap.add_argument("--marker-ratio", type=float, default=0.75)
    ap.add_argument("--first-marker", type=int, default=144)
    ap.add_argument("--stride", type=int, default=1, help="use every n-th frame")
    args = ap.parse_args()

    spec = CharucoBoardSpec(
        id="b0",
        squares_x=5,
        squares_y=5,
        square_length=args.square,
        marker_length=args.square * args.marker_ratio,
        dictionary="DICT_6X6_250",
        marker_id_offset=args.first_marker,
    )
    obs, size, counts = detect(args.sequence, [spec], stride=args.stride)
    print("detections:", counts)
    system = build(obs, size, {"b0": CharucoDetector(spec).object_points})
    initialize_system(system)
    rms_init, _ = rms_px(system)
    backend = "ceres" if backend_status()["ceres"][0] else "scipy"
    problem = build_problem(system)
    result = get_backend(backend).solve(problem, SolveOptions(max_iterations=500))
    write_back(system, problem)
    rms, per = rms_px(system)
    print(f"mlti_cal ({backend}): RMS {rms_init:.3f} -> {rms:.4f} px, success={result.success}")

    cov = compute_covariance(problem, pixel_noise_std=None)
    names = sorted(system.cameras)
    K_mc, D_mc, T_mc = load_mccalib(args.sequence / "calibrated_cameras_data.yml")
    out: dict = {
        "detections": counts,
        "rms_init_px": rms_init,
        "rms_px": rms,
        "rms_per_camera_px": per,
        "sigma_residual_px": float(np.sqrt(cov.sigma2_residual)),
        "cameras": {},
    }
    mc_params = [params_of(K_mc[i], D_mc[i]) for i in range(len(names))]
    out["rms_at_mccalib"] = rms_with_cameras(system, mc_params, T_mc)
    for i, n in enumerate(names):
        cam = system.cameras[n]
        sig = np.array([cov.std_error_of(f"intr:{n}.{j}") for j in range(cam.model.num_params)])
        diff = cam.params[:4] - mc_params[i][:4]
        entry = {
            "params": cam.params.tolist(),
            "mccalib_params": mc_params[i].tolist(),
            "intr_diff_px": diff.tolist(),
            "intr_diff_sigma": (diff / sig[:4]).tolist(),
        }
        if i > 0:
            T = pose_to_matrix(cam.extrinsic)
            e = pose_error(T, T_mc[i])
            t_cov, _, _ = marginal_covariance(cov, f"extr:{n}")
            err6 = np.array(e["_dt"] + e["_phi"])
            e["mahalanobis2_6dof"] = float(err6 @ np.linalg.solve(t_cov, err6))
            entry["pose_vs_mccalib"] = {k: v for k, v in e.items() if not k.startswith("_")}
        out["cameras"][n] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(
        f"RMS on these corners: mlti {rms:.4f}  "
        f"MC-Calib cameras {out['rms_at_mccalib']['rms_px']:.4f}"
    )
    for n, c in out["cameras"].items():
        line = (
            f"{n}: fx fy cx cy diff px {np.round(c['intr_diff_px'], 2)}  "
            f"sigma {np.round(c['intr_diff_sigma'], 1)}"
        )
        if "pose_vs_mccalib" in c:
            p = c["pose_vs_mccalib"]
            line += (
                f" | pose {p['rotation_deg']:.4f} deg {p['centre_m']:.3f} units"
                f" (Mahalanobis^2 {p['mahalanobis2_6dof']:.1f})"
            )
        print(line)


if __name__ == "__main__":
    main()
