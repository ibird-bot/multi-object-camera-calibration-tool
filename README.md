# Multi Object Camera Calibration Tool

[![Latest release](https://img.shields.io/github/v/release/ibird-bot/multi-object-camera-calibration-tool)](https://github.com/ibird-bot/multi-object-camera-calibration-tool/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

An open-source multi-camera calibration tool for calibrating **any number of
cameras** against **any number of calibration objects** at the same time.
Charuco board detection, eight camera models, and a desktop GUI.

You can add as many calibration objects as you need and as many cameras as you
like. Boards and cameras are declared in the setup tab, detected, then solved
together in a single bundle adjustment, so every camera pose and every board
pose is estimated in one consistent frame. Alongside the calibration it reports
how much the result can be trusted: parameter covariance, per-pixel projection
uncertainty, residual maps, coverage and cross-validated held-out error.

![Projection uncertainty map: a 1-sigma error heatmap over the sensor, clean in the centre and rising toward the edges](docs/images/uncertainty-map.png)

*A 1-sigma projection uncertainty map from `mlti-cal demo` -- where this
calibration can be trusted, not just its RMS.*

### What it does

- **Multi-camera calibration** for rigs of any size, not just stereo pairs
- **Multi-object calibration**: several targets in one session, with marker ID
  collision checking and refusal of any layout that cannot be told apart
- **Three target types**: Charuco boards, plain checkerboards and dot grids,
  mixable in the same images -- coded boards are detected first and masked out
  before the uncoded detectors search
- **Intrinsic and extrinsic calibration** solved jointly in one optimisation
- **Eight camera models**, including pinhole, fisheye, double sphere and FOV
- **Two solver backends**: scipy and Ceres
- **Uncertainty reporting** instead of a single RMS reprojection error
- **Desktop GUI** built with PySide6, plus a headless command line

## Try it in 30 seconds

No data required -- generates synthetic cameras and boards, calibrates, and
checks the result against known ground truth:

```bash
pip install -e .
mlti-cal demo --cameras 3 --frames 20
```

This is also the fastest way to see whether an install is healthy, and it is
the same code path the GUI runs, so a CLI result and a GUI result never
disagree.

## A full example: four cameras, three boards, rendered images

`examples/rig_4cam` renders a complete capture -- 120 images -- of a four-camera
rig fanned over 100 degrees, one of them a wide-angle Double Sphere lens, with
two Charuco boards and an asymmetric dot grid. The two outer cameras never see
the same board, so the rig is only solvable by chaining through the middle
cameras. Ground truth is written next to the images.

![The GUI on the four-camera example: detection with per-board colours, the starting point, and a solve to 0.13 px](docs/images/rig4cam.gif)

```bash
python examples/rig_4cam/make_dataset.py            # ~3 minutes, deterministic
mlti-cal calibrate examples/rig_4cam/config.json --backend ceres --crossval -o Output/rig_4cam
```

Run both from the repository root. Measured on this example: 0.131 px RMS,
every extrinsic within 0.5 mm and 0.03 degrees of the truth, held-out error
0.99x the training error. The report also flags that the wide lens's focal
length and `xi` cannot be separated by this data (correlation +0.999) -- which
is true, and is why their individual values differ from the truth while the
lens still projects to within 0.3 px of it.

Without Ceres, `--backend scipy` gives the same result, final cost and RMS
identical to Ceres, in 46 s (Ceres: 2.7 s; measured on an i7-12700H, solve
only). Most of the old gap was not the algorithm: OpenBLAS's default thread
pool made each dense SVD ~10x slower on hybrid CPUs, and the scipy backend now
caps it at `num_threads` (default 1).

## Requirements

Python 3.12.

## Install

Clone the repository, create a virtual environment and install the
dependencies:

```bash
git clone https://github.com/ibird-bot/multi-object-camera-calibration-tool.git
cd multi-object-camera-calibration-tool

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -e ".[gui]"
```

The extras decide what you get:

| Command | What it installs |
|---|---|
| `pip install -e .` | Headless core: detection, bootstrap, the scipy solver, the report engine, the CLI. No Qt. |
| `pip install -e ".[gui]"` | The above plus the desktop GUI (PySide6, pyqtgraph, matplotlib). |
| `pip install -e ".[ceres]"` | Adds the Ceres backend. Needs a compiled Ceres, so it is deliberately optional -- without it `mlti-cal backends` lists Ceres as unusable and the scipy backend solves the identical problem. |
| `pip install -e ".[dev]"` | pytest, ruff, mypy, coverage. |

## Running the GUI

With the virtual environment active:

```bash
python -m mlti_cal.gui
```

or, since the package installs an entry point:

```bash
mlti-cal-gui
```

The application opens on the "Setup & Detection" tab, where you declare your
cameras and calibration boards and run detection over their images.
Optimisation and the report follow in their own tabs.

## The report

A low RMS is compatible with a badly wrong calibration, so the report tab
answers "can I trust this?" rather than printing one number. Seven figures,
each also written to disk by **Export...**:

| Figure | What it answers |
|---|---|
| Projection uncertainty | How far a point fixed relative to the boards could project wrong at a stated range, per pixel |
| Residual field | Per-corner error arrows; structure here is model inadequacy RMS cannot see |
| Residual map | Per-cell RMS across the sensor, and the per-cell **mean** vector -- averaging cancels random error, so what survives is systematic bias |
| Correlations | Which parameters the data cannot tell apart: the labelled camera block, the worst pairs, and camera parameters against board poses |
| Coverage | Where corners were actually observed, and how varied the board tilts were |
| Residual distribution | Histogram and Q-Q against a normal; heavy tails mean outliers or unmodelled error |
| Error vs radius | Growth toward the edge means the distortion model cannot represent the lens |

![Correlation matrices: camera parameters, the least-separable pairs, and camera parameters against board poses](docs/images/correlation-matrix.png)

*Which parameters the data cannot tell apart -- including camera intrinsics
against board poses, the block a camera-only matrix would hide.*

Two conventions worth knowing:

- **Unobserved regions are masked, never zero.** A sensor cell with no corners
  is drawn grey. Zero would read as a perfect fit in exactly the area the
  calibration knows nothing about.
- **Correlations include board poses.** On a typical capture the strongest
  correlations are a camera parameter against a board pose -- a static board
  makes the principal point indistinguishable from a board translation -- so a
  camera-only matrix would hide the finding. Hover any cell for the exact
  coefficient.

Everything is computed headlessly and stored in `report.json`, so a CLI run and
a GUI run produce identical numbers. Export writes `report.json`,
`report.txt`, `calibration.json`, `calibration.yaml` (OpenCV `FileStorage`) and
one PNG per figure, each rendered at a size chosen for its content rather than
inherited from the window.

## Validation against other tools

Checked on public data against independent calibration tools. Where both tools
are given the same corners, differences come from the solvers alone.

| Compared with | Data | Result |
|---|---|---|
| [mrcal](https://mrcal.secretsauce.net/) 2.4.1 | 1 camera, 84 real images | Identical RMS (0.524893 px); projection uncertainty within 1.5% at every pixel |
| mrcal and OpenCV | Real stereo pair | Identical RMS to mrcal, uncertainty ratio 1.021 (exactly the degrees-of-freedom factor); OpenCV's intrinsics within 1.1 sigma |
| [Kalibr](https://github.com/ethz-asl/kalibr) (published) | TUM VI fisheye stereo, real | Rotation within 0.024 deg, baseline within 0.5 mm of 101 mm |
| [MC-Calib](https://github.com/rameau-fr/MC-Calib) | 6 real cameras, no ground truth | 0.3736 vs 0.3754 px on identical corners; poses within 0.022 deg |
| MC-Calib and ground truth | 5 rendered cameras | Poses within 1.3 mm and 0.033 deg of the truth; MC-Calib's own about twice as close |
| [BabelCalib](https://github.com/ylochman/babelcalib) benchmark | 8 real fisheye lenses, up to 250 deg | Held-out 0.25 px with Double Sphere or EUCM, level with Kalibr; Kannala-Brandt and FOV fail past 90 deg off-axis |

Method, limits and how to reproduce every row: [`docs/validation.md`](docs/validation.md).

## Supported camera models

Eight camera models, each with analytic Jacobians checked against finite
differences.

| Shown as | id | params | For |
|---|---|---|---|
| OpenCV | `pinhole_radtan` | 9 | Normal and wide lenses. Start here. |
| OpenCV Fisheye | `fisheye_kb` | 8 | Kannala-Brandt equidistant fisheye, up to 180 degrees in total |
| Double Sphere | `double_sphere` | 6 | Fisheye, 2 distortion params, well conditioned |
| Enhanced Unified | `eucm` | 6 | Fisheye and catadioptric, 2 params |
| Thin Prism | `thin_prism` | 16 | Rational radial + tangential + prism |
| Matlab | `matlab` | 10 | OpenCV plus a skew term |
| FOV | `fov` | 5 | One parameter: the lens field of view, up to 180 degrees in total |
| Halcon Division | `halcon_division` | 5 | One parameter, closed-form inverse |

## Bring your own detector, object and lens

Use your own target detection and decoding, your own calibration object (flat
or 3D) and your own camera model, without modifying this package. Adapt them
to three fixed contracts:

```python
def detect(image, settings):  # -> (N, 3) float array: [target_id, x, y]
    ...
```
```
id, x, y, z                        # the object: one line per target, metres
```
```python
def project(params, points):  # -> (N, 2) pixels; derivatives optional
    ...
```

Each is one Python file dropped in `~/.mlti_cal/plugins/`. Every result is
checked before it is used, and a plug-in that breaks its contract is reported
and skipped, never patched up. `mlti-cal plugins check FILE` runs the same
checks for the team writing it.

Contracts, conventions and guarantees: [`docs/extending.md`](docs/extending.md).
Templates: [`examples/plugins/`](examples/plugins/). For a working example of
all three together (a 3D marker cube, a short ArUco detector and a lens given
only as a projection function), see `examples/custom_object`. It calibrates to within
0.34 mm of the ground truth. A detector needing more control than one function
can still register a full kind ([`docs/writing-a-detector.md`](docs/writing-a-detector.md)).

## Coordinate conventions

Stated explicitly, because getting these wrong is silent:

- Camera frame is OpenCV: +x right, +y down, +z into the scene.
- `T_a_b` maps points from frame `b` into frame `a`.
- `Camera.extrinsic` is `T_cam_rig`; board poses are `T_rig_board`.
- Poses store `[tx,ty,tz, qx,qy,qz,qw]` (scalar **last**).
- The tangent is `[dt(3); phi(3)]`, translation first, with the **decoupled**
  retraction `t <- t + dt`, `q <- q (x) Exp(phi)`.
- The reference camera's extrinsic is fixed at identity. That is the only gauge
  fixed; scale is already metric from the board.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The end-to-end solves marked `slow` take minutes. To skip them:

```bash
pytest -m "not slow"
```

That is what CI runs on every push; the full suite runs nightly.

## Versioning

[Semantic versioning](https://semver.org/) from 1.0.0 onward: a breaking
change to the config file format, the CLI, or the public Python API
(`mlti_cal.problem`, `mlti_cal.models`, `mlti_cal.solvers`, `mlti_cal.report`)
is a major version bump, not a silent change. See [CHANGELOG.md](CHANGELOG.md)
for what changed in each release.

## License

MIT. See [LICENSE](LICENSE).
