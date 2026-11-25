"""
Print the mlti_cal vs mrcal comparison from the two JSON files.

    python tools/mrcal_compare/compare.py MLTI.json MRCAL.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from mlti_cal.models.camera import get_model
from mlti_cal.report.uncertainty import unproject


def main() -> None:
    a = json.loads(Path(sys.argv[1]).read_text())
    b = json.loads(Path(sys.argv[2]).read_text())
    model = get_model(a["model"])
    p_mlti = np.array(a["params"])
    sigma = np.array(a["param_sigma"])

    print(f"model {a['model']} = {b['mrcal_lensmodel']}, {len(a['frames'])} frames, "
          f"image {a['image_size'][0]}x{a['image_size'][1]}, "
          f"boards at {a['board_distance_m'][0]:.2f}-{a['board_distance_m'][2]:.2f} m")  # fmt: skip

    print("\nRMS reprojection error (px), sqrt(mean(dx^2+dy^2)) over all corners")
    print(f"  mlti_cal, own pipeline            {a['rms_px']:.6f}")
    print(f"  mrcal, own seed and solve         {b['independent']['rms_px']:.6f}")
    print(f"  mrcal cost AT mlti_cal solution   {b['rms_at_mlti_px']:.6f}")
    print(f"  mrcal optimising FROM it          {b['from_mlti']['rms_px']:.6f}")

    # Rational-model coefficients are not unique enough to compare one by one;
    # what matters is whether the two parameter sets project the same.
    w, h = a["image_size"]
    us, vs = np.meshgrid(np.arange(100, w, 200), np.arange(100, h, 200))
    pix = np.column_stack([us.ravel(), vs.ravel()]).astype(float)
    rays, ok = unproject(a["model"], p_mlti, pix, model=model)
    print("\nSame rays projected with each parameter set, difference in px")
    for label, key in (("mrcal independent", "independent"), ("mrcal from mlti", "from_mlti")):
        d = np.linalg.norm(model.project(np.array(b[key]["params"]), rays[ok]) - pix[ok], axis=1)
        print(f"  {label:<18} median {np.median(d):.4f}   max {d.max():.4f}   "
              f"({ok.mean() * 100:.0f}% of the sensor)")  # fmt: skip

    print("\nIntrinsics core")
    print(f"  {'':4} {'mlti_cal':>12} {'mrcal indep.':>13} {'difference':>11} {'mlti sigma':>11}")
    p_ind = np.array(b["independent"]["params"])
    for i, name in enumerate(a["param_names"][:4]):
        print(f"  {name:4} {p_mlti[i]:12.4f} {p_ind[i]:13.4f} {p_ind[i] - p_mlti[i]:11.4f} "
              f"{sigma[i]:11.4f}")  # fmt: skip

    print("\nProjection uncertainty, worst-direction 1-sigma (px), same pixels and range")
    print(f"  {'range':>8} {'pixels':>7} {'mlti median':>12} {'mrcal median':>13} "
          f"{'ratio median':>13} {'ratio min':>10} {'ratio max':>10} {'old map':>15}")  # fmt: skip
    for rng, grid in a["uncertainty"].items():
        mine, theirs = np.array(grid["sigma_max"]), np.array(b["uncertainty"][rng])
        ratio = mine / theirs
        old = np.median(grid.get("sigma_max_camera_reference", [np.nan]))
        print(f"  {float(rng):7.2f}m {mine.size:7d} {np.median(mine):12.4f} "
              f"{np.median(theirs):13.4f} {np.median(ratio):13.3f} {ratio.min():10.3f} "
              f"{ratio.max():10.3f} {old:15.4f}")  # fmt: skip


if __name__ == "__main__":
    main()
