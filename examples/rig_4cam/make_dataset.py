"""
Render the four-camera example rig into `examples/rig_4cam/images/`.

    python examples/rig_4cam/make_dataset.py
    mlti-cal calibrate examples/rig_4cam/config.json --crossval -o Output/rig_4cam

Run both from the repository root: `config.json` points at the image folders
by paths relative to it. Rendering takes a few minutes and is deterministic,
so the images are generated rather than committed.

The rig: four cameras fanned over a 100-degree arc, one of them a wide-angle
Double Sphere lens, and three boards -- two Charuco boards sharing one
dictionary by marker-id offset, and an asymmetric dot grid. cam0 and cam3
never see the same board, so cam3 is calibrated through the cameras between
them. `images/truth.json` holds the true parameters and poses to compare with.
"""

from pathlib import Path

from mlti_cal.io.render import plan_scene, write_dataset

if __name__ == "__main__":
    out = Path(__file__).parent / "images"
    scene = plan_scene(num_frames=30)
    counts = write_dataset(scene, out)
    print(f"{len(scene.frames)} frames written to {out}")
    for cam, n in counts.items():
        print(f"  {cam}: {n} boards fully in view")
