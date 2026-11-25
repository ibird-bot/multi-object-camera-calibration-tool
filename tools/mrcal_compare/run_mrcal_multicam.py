"""
mrcal side of the MULTI-camera comparison. Runs under Linux/WSL: `apt install python3-mrcal`.

Reads the corners.json and mlti.json written by run_mlti_multicam.py and solves
the same rig -- same corners, same lens model, unit weights, no outlier
rejection, no board warp -- three ways, like run_mrcal.py does for one camera:

  independent  mrcal's own stereographic seed and staged solve
  at_mlti      mrcal's cost evaluated AT the mlti_cal solution, unoptimised
  from_mlti    mrcal optimising from the mlti_cal solution

then mrcal's projection uncertainty for every camera at the pixels and ranges
the mlti_cal maps were evaluated at.

    python3 tools/mrcal_compare/run_mrcal_multicam.py OUTDIR
"""

import json
import sys
from pathlib import Path

import mrcal
import numpy as np

MODELS = {"thin_prism": "LENSMODEL_OPENCV12", "pinhole_radtan": "LENSMODEL_OPENCV5"}


def observations(data, frames, names):
    """mrcal arrays, sorted by frame then camera as mrcal requires."""
    fidx = {f: i for i, f in enumerate(frames)}
    cidx = {n: i for i, n in enumerate(names)}
    rows = sorted(
        (o for o in data["observations"] if o["frame"] in fidx),
        key=lambda o: (fidx[o["frame"]], cidx[o["camera"]]),
    )
    obs = np.ones((len(rows), data["rows"], data["cols"], 3))
    idx = np.zeros((len(rows), 3), dtype=np.int32)
    for k, o in enumerate(rows):
        obs[k, :, :, :2] = np.array(o["corners"]).reshape(data["rows"], data["cols"], 2)
        ic = cidx[o["camera"]]
        idx[k] = (fidx[o["frame"]], ic, ic - 1)  # camera 0 is the reference: -1
    return obs, idx


def inputs(lensmodel, intr, extr, frames_rt, obs, idx, sizes, spacing):
    return dict(
        intrinsics=np.array(intr, dtype=float),
        extrinsics_rt_fromref=np.array(extr, dtype=float).reshape(-1, 6),
        frames_rt_toref=np.array(frames_rt, dtype=float),
        points=None,
        observations_board=obs.copy(),
        indices_frame_camintrinsics_camextrinsics=idx,
        observations_point=None,
        indices_point_camintrinsics_camextrinsics=None,
        lensmodel=lensmodel,
        imagersizes=np.array(sizes, dtype=np.int32),
        calibration_object_spacing=spacing,
        do_optimize_intrinsics_core=True,
        do_optimize_intrinsics_distortions=True,
        do_optimize_extrinsics=True,
        do_optimize_frames=True,
        do_optimize_calobject_warp=False,
        do_apply_outlier_rejection=False,
        do_apply_regularization=False,
    )


def rms_px(oi):
    x = mrcal.optimizer_callback(**oi, no_jacobian=True, no_factorization=True)[1]
    n = oi["observations_board"][..., 0].size
    return float(np.sqrt(np.sum(x[: 2 * n] ** 2) / n))


def summary(oi):
    return {
        "intrinsics": oi["intrinsics"].tolist(),
        "extrinsics_rt_fromref": oi["extrinsics_rt_fromref"].tolist(),
        "rms_px": rms_px(oi),
    }


def main():
    outdir = Path(sys.argv[1])
    data = json.loads((outdir / "corners.json").read_text())
    ref = json.loads((outdir / "mlti.json").read_text())
    m = ref["mlti"]
    lensmodel = MODELS[ref["model"]]
    names = [c["name"] for c in data["cameras"]]
    sizes = [c["image_size"] for c in data["cameras"]]
    spacing = data["spacing"]
    obs, idx = observations(data, m["frames"], names)
    n_dist = len(m["intrinsics"][names[0]]) - 4
    args = (obs, idx, sizes, spacing)

    # ---- independent ------------------------------------------------------
    focal = 0.8 * max(sizes[0])
    s_intr, s_extr, s_frames = mrcal.seed_stereographic(
        np.array(sizes, dtype=np.int32), focal, idx[:, :2].copy(), obs, spacing
    )
    oi = inputs("LENSMODEL_STEREOGRAPHIC", s_intr, s_extr, s_frames, *args)
    mrcal.optimize(**oi)
    rng = np.random.default_rng(0)
    start = np.column_stack(
        [oi["intrinsics"][:, :4], rng.normal(scale=1e-5, size=(len(names), n_dist))]
    )
    oi = inputs(lensmodel, start, oi["extrinsics_rt_fromref"], oi["frames_rt_toref"], *args)
    oi["do_apply_regularization"] = True
    mrcal.optimize(**oi)
    oi = inputs(
        lensmodel, oi["intrinsics"], oi["extrinsics_rt_fromref"], oi["frames_rt_toref"], *args
    )
    mrcal.optimize(**oi)
    independent = summary(oi)

    # ---- at / from the mlti_cal solution ------------------------------------
    intr = [m["intrinsics"][n] for n in names]
    at = inputs(lensmodel, intr, m["extrinsics_rt_fromref"], m["frames_rt_toref"], *args)
    rms_at = rms_px(at)
    mrcal.optimize(**at)
    from_mlti = summary(at)

    # ---- projection uncertainty (needs regularisation terms in mrcal 2.4) ---
    reg = inputs(
        lensmodel, at["intrinsics"], at["extrinsics_rt_fromref"], at["frames_rt_toref"], *args
    )
    reg["do_apply_regularization"] = True
    mrcal.optimize(**reg)
    unc = {}
    for ic, name in enumerate(names):
        model = mrcal.cameramodel(optimization_inputs=reg, icam_intrinsics=ic)
        unc[name] = {}
        for rng_m, grid in m["uncertainty"][name].items():
            q = np.array(grid["pixels"], dtype=float)
            v = mrcal.unproject(q, *model.intrinsics(), normalize=True)
            unc[name][rng_m] = mrcal.projection_uncertainty(
                v * float(rng_m), model, what="worstdirection-stdev"
            ).tolist()

    out = {
        "mrcal_lensmodel": lensmodel,
        "observations": int(len(obs)),
        "independent": independent,
        "rms_at_mlti_px": rms_at,
        "from_mlti": from_mlti,
        "regularised": summary(reg),
        "uncertainty": unc,
    }
    (outdir / "mrcal.json").write_text(json.dumps(out, indent=1))
    print(f"independent: RMS {independent['rms_px']:.4f} px")
    print(f"at mlti    : RMS {rms_at:.4f} px   (mlti_cal reported {m['rms_px']:.4f})")
    print(f"from mlti  : RMS {from_mlti['rms_px']:.4f} px")
    print(f"regularised: RMS {out['regularised']['rms_px']:.4f} px (uncertainty only)")


if __name__ == "__main__":
    main()
