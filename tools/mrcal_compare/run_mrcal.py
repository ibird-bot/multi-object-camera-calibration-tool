"""
mrcal side of the comparison. Runs under Linux/WSL: `apt install python3-mrcal`.

Solves the SAME problem as run_mlti.py -- same corners, same lens model, unit
weights, no outlier rejection, no board-warp, no regularisation -- three ways:

  independent  mrcal's own seed (stereographic) and staged solve, never seeing
               the mlti_cal result. Agreement here means two unrelated
               pipelines reach the same answer.
  at_mlti      mrcal's cost evaluated AT the mlti_cal solution, unoptimised.
               Agreement means the two projection functions and conventions
               are identical.
  from_mlti    mrcal optimising from the mlti_cal solution. If it does not
               move, mlti_cal had found the minimum of mrcal's cost.

Then mrcal's projection uncertainty (mean-pcam) at the pixels and ranges the
mlti_cal map was evaluated at.

    python3 tools/mrcal_compare/run_mrcal.py CORNERS MLTI.json OUT.json
"""

import argparse
import json
from pathlib import Path

import mrcal
import numpy as np

MODELS = {"thin_prism": "LENSMODEL_OPENCV12", "pinhole_radtan": "LENSMODEL_OPENCV5"}


def read_observations(path, frames, gridn):
    """(Nframes, gridn, gridn, 3) of (x, y, weight=1), in the given frame order."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, x, y = line.split()[:3]
        if x != "-":
            rows.setdefault(name, []).append((float(x), float(y)))
    obs = np.ones((len(frames), gridn, gridn, 3))
    for i, name in enumerate(frames):
        obs[i, :, :, :2] = np.array(rows[name]).reshape(gridn, gridn, 2)
    return obs


def inputs(lensmodel, intrinsics, frames_rt, obs, size, spacing):
    n = len(obs)
    idx = np.column_stack([np.arange(n), np.zeros(n), -np.ones(n)]).astype(np.int32)
    return dict(
        intrinsics=np.array(intrinsics, dtype=float).reshape(1, -1),
        extrinsics_rt_fromref=np.zeros((0, 6)),
        frames_rt_toref=np.array(frames_rt, dtype=float),
        points=None,
        observations_board=obs.copy(),
        indices_frame_camintrinsics_camextrinsics=idx,
        observations_point=None,
        indices_point_camintrinsics_camextrinsics=None,
        lensmodel=lensmodel,
        imagersizes=np.array([size], dtype=np.int32),
        calibration_object_spacing=spacing,
        # Spelled out, not left to default: mrcal's uncertainty code reads these
        # keys from the dict and treats a missing one as False.
        do_optimize_intrinsics_core=True,
        do_optimize_intrinsics_distortions=True,
        do_optimize_extrinsics=True,
        do_optimize_frames=True,
        do_optimize_calobject_warp=False,
        do_apply_outlier_rejection=False,
        do_apply_regularization=False,
    )


def rms_px(oi):
    """sqrt(mean(dx^2 + dy^2)) over every corner, from mrcal's own residuals."""
    x = mrcal.optimizer_callback(**oi, no_jacobian=True, no_factorization=True)[1]
    n = oi["observations_board"][..., 0].size
    return float(np.sqrt(np.sum(x[: 2 * n] ** 2) / n))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("corners")
    ap.add_argument("mlti_json")
    ap.add_argument("out")
    ap.add_argument("--focal", type=float, default=1900.0, help="seed focal length, px")
    args = ap.parse_args()

    ref = json.loads(Path(args.mlti_json).read_text())
    lensmodel = MODELS[ref["model"]]
    size, spacing, gridn = ref["image_size"], ref["spacing"], ref["gridn"]
    obs = read_observations(args.corners, ref["frames"], gridn)
    n_dist = len(ref["params"]) - 4

    # ---- independent: mrcal's own seed and staging --------------------------
    idx_fc = np.column_stack([np.arange(len(obs)), np.zeros(len(obs))]).astype(np.int32)
    seed_intr, _, seed_frames = mrcal.seed_stereographic(
        np.array([size], dtype=np.int32), args.focal, idx_fc, obs, spacing
    )
    oi = inputs("LENSMODEL_STEREOGRAPHIC", seed_intr, seed_frames, obs, size, spacing)
    mrcal.optimize(**oi)
    core, frames = oi["intrinsics"][0, :4], oi["frames_rt_toref"]
    rng = np.random.default_rng(0)
    start = np.concatenate([core, rng.normal(scale=1e-5, size=n_dist)])
    # A regularised pass first, as mrcal's own tool does, then the bare problem.
    oi = inputs(lensmodel, start, frames, obs, size, spacing)
    oi["do_apply_regularization"] = True
    mrcal.optimize(**oi)
    oi = inputs(lensmodel, oi["intrinsics"], oi["frames_rt_toref"], obs, size, spacing)
    mrcal.optimize(**oi)
    independent = {"params": oi["intrinsics"][0].tolist(), "rms_px": rms_px(oi)}

    # ---- at / from the mlti_cal solution ------------------------------------
    at = inputs(lensmodel, ref["params"], ref["frames_rt_toref"], obs, size, spacing)
    rms_at = rms_px(at)
    mrcal.optimize(**at)
    from_mlti = {"params": at["intrinsics"][0].tolist(), "rms_px": rms_px(at)}

    # ---- projection uncertainty at the mlti_cal grid ------------------------
    # mrcal 2.4's uncertainty code requires its regularisation terms to exist
    # (it indexes into them unconditionally), so this one model is re-solved
    # with them on. The shift this causes is reported as rms_regularised_px.
    reg = inputs(lensmodel, at["intrinsics"], at["frames_rt_toref"], obs, size, spacing)
    reg["do_apply_regularization"] = True
    mrcal.optimize(**reg)
    model = mrcal.cameramodel(optimization_inputs=reg, icam_intrinsics=0)
    unc = {}
    for rng_m, grid in ref["uncertainty"].items():
        q = np.array(grid["pixels"], dtype=float)
        v = mrcal.unproject(q, *model.intrinsics(), normalize=True)
        unc[rng_m] = mrcal.projection_uncertainty(
            v * float(rng_m), model, what="worstdirection-stdev"
        ).tolist()

    out = {
        "mrcal_lensmodel": lensmodel,
        "independent": independent,
        "rms_at_mlti_px": rms_at,
        "from_mlti": from_mlti,
        "regularised": {"params": reg["intrinsics"][0].tolist(), "rms_px": rms_px(reg)},
        "uncertainty": unc,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"independent: RMS {independent['rms_px']:.4f} px")
    print(f"at mlti    : RMS {rms_at:.4f} px   (mlti_cal reported {ref['rms_px']:.4f})")
    print(f"from mlti  : RMS {from_mlti['rms_px']:.4f} px")
    print(f"regularised: RMS {out['regularised']['rms_px']:.4f} px (used for uncertainty only)")


if __name__ == "__main__":
    main()
