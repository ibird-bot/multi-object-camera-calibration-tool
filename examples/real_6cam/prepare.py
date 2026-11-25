"""
Six real cameras: a subsample of MC-Calib's `Seq02_Overlapping_multicamera`.

The data is not shipped here. Download `Real_Images` from the MC-Calib
repository (https://github.com/rameau-fr/MC-Calib, "Datasets"; Rameau et al.,
"MC-Calib: A generic and robust calibration toolbox for multi-camera
systems", CVIU 2022) and extract `Seq02_Overlapping_multicamera`. Then:

    python examples/real_6cam/prepare.py PATH/TO/Seq02_Overlapping_multicamera
    mlti-cal calibrate examples/real_6cam/config.json --backend ceres --crossval \
        -o Output/real_6cam_result

The rig is three Intel RealSense infrared stereo pairs in a row, 1280 x 720,
looking at one 5 x 5 Charuco board. Board values come from MC-Calib's config
for this sequence: squares of 9.15 (centimetres: the pairs come out 4.96 apart,
the RealSense 50 mm baseline), markers 0.75 of a square, 6 x 6 dictionary,
board index 12, so marker ids 144 to 155.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "Output" / "real_6cam"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "sequence", type=Path, help="the extracted Seq02_Overlapping_multicamera folder"
    )
    ap.add_argument("--every", type=int, default=10, help="keep one frame in N (default 10)")
    args = ap.parse_args()
    for cam in sorted(args.sequence.glob("Cam_00*")):
        dst = OUT / cam.name
        dst.mkdir(parents=True, exist_ok=True)
        frames = sorted(cam.glob("*.png"))[:: args.every]
        for f in frames:
            shutil.copy2(f, dst / f.name)
        print(f"{cam.name}: {len(frames)} frames -> {dst}")


if __name__ == "__main__":
    main()
