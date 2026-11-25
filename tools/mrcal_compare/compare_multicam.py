"""
Print the multi-camera comparison: mlti_cal vs mrcal (vs OpenCV for stereo).

    python tools/mrcal_compare/compare_multicam.py OUTDIR [--markdown]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def rel(rt_a, rt_b) -> tuple[float, float, float]:
    """(rotation difference deg, translation difference, baseline of a)."""
    Ra = cv2.Rodrigues(np.array(rt_a[:3]))[0]
    Rb = cv2.Rodrigues(np.array(rt_b[:3]))[0]
    ang = np.degrees(np.linalg.norm(cv2.Rodrigues(Ra @ Rb.T)[0]))
    # Camera centres in the reference frame: c = -R^T t.
    ca = -Ra.T @ np.array(rt_a[3:])
    cb = -Rb.T @ np.array(rt_b[3:])
    return float(ang), float(np.linalg.norm(ca - cb)), float(np.linalg.norm(ca))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("outdir", type=Path)
    args = ap.parse_args()
    ref = json.loads((args.outdir / "mlti.json").read_text())
    mr = json.loads((args.outdir / "mrcal.json").read_text())
    m, cv = ref["mlti"], ref.get("opencv")
    names = list(m["intrinsics"])
    unit = "m"  # board spacing is given in metres

    print(f"## {args.outdir}  ({ref['model']}, {mr['observations']} board observations)\n")
    print("| run | RMS px |")
    print("|---|---|")
    print(f"| mlti_cal ({m['backend']}) | {m['rms_px']:.6f} |")
    print(f"| mrcal independent | {mr['independent']['rms_px']:.6f} |")
    print(f"| mrcal evaluated at mlti_cal solution | {mr['rms_at_mlti_px']:.6f} |")
    print(f"| mrcal optimised from mlti_cal solution | {mr['from_mlti']['rms_px']:.6f} |")
    print()

    pn = m["param_names"]
    for ic, n in enumerate(names):
        print(f"### intrinsics: {n}\n")
        hdr = "| param | mlti_cal | 1-sigma | mrcal indep. | diff / sigma |"
        if cv:
            hdr += " OpenCV | diff / sigma |"
        print(hdr)
        print("|---" * (hdr.count("|") - 1) + "|")
        for k, name in enumerate(pn):
            a = m["intrinsics"][n][k]
            s = m["intrinsics_sigma"][n][k]
            b = mr["independent"]["intrinsics"][ic][k]
            row = f"| {name} | {a:.6g} | {s:.3g} | {b:.6g} | {(b - a) / s:+.3f} |"
            if cv:
                c = cv["intrinsics"][n][k]
                row += f" {c:.6g} | {(c - a) / s:+.3f} |"
            print(row)
        print()

    print("### extrinsics (camera i relative to the reference camera)\n")
    print(
        "| camera | baseline (mlti_cal) | vs mrcal: rotation deg | centre diff "
        "| vs OpenCV: rotation deg | centre diff |"
    )
    print("|---|---|---|---|---|---|")
    for i, n in enumerate(names[1:]):
        a = m["extrinsics_rt_fromref"][i]
        r1, t1, base = rel(a, mr["independent"]["extrinsics_rt_fromref"][i])
        row = f"| {n} | {base:.6g} {unit} | {r1:.2e} | {t1:.2e} {unit} |"
        if cv:
            r2, t2, _ = rel(a, cv["extrinsics_rt_fromref"][i])
            row += f" {r2:.2e} | {t2:.2e} {unit} |"
        else:
            row += " - | - |"
        print(row)
    print()

    print("### projection uncertainty, worst direction, 1-sigma px (mlti_cal / mrcal)\n")
    print(
        "| camera | range | mlti_cal median | mrcal median | ratio min | ratio median | ratio max |"
    )
    print("|---|---|---|---|---|---|---|")
    for n in names:
        for rng, grid in m["uncertainty"][n].items():
            a = np.array(grid["sigma_max"])
            b = np.array(mr["uncertainty"][n][rng])
            r = a / b
            print(
                f"| {n} | {float(rng):.3g} | {np.median(a):.4f} | {np.median(b):.4f} | "
                f"{r.min():.3f} | {np.median(r):.3f} | {r.max():.3f} |"
            )


if __name__ == "__main__":
    main()
