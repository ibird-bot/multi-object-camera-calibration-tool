# mlti_object_cal

A multi-camera / multi-object calibration workbench that tells you how much to
trust the answer.

Most calibration tools report one number — RMS reprojection error — and a low
RMS routinely hides a badly-constrained or overfit calibration. This tool
answers the useful question instead: **how much can I trust a projected point,
and where?**

Two differentiators:

1. **Solver choice is first class.** The same problem goes to scipy, Ceres or
   GTSAM through one backend-agnostic `Problem`. Every option carries a
   "when to use this" note, shared by the CLI and the GUI tooltips.
2. **Truth-telling reports.** Parameter covariance with the rank decision made
   in the open, per-pixel projection-uncertainty maps, residual quiver fields,
   coverage, outliers, cross-validated held-out error, and — on synthetic data
   — a check of whether the claimed uncertainty actually brackets the true
   error.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[gui,dev]"
```

Requires Python 3.12. `pyceres` installs from a wheel on Windows; GTSAM does
not (see Limitations).

## Use

```bash
mlti-cal backends                    # what can actually run here
mlti-cal backends --describe ceres   # the option catalog, with guidance
mlti-cal demo --cameras 2 --frames 22 --backend ceres --crossval
mlti-cal compare                     # every backend on one problem
mlti-cal calibrate config.json       # real images
python -m mlti_cal.gui               # the three-window GUI
```

A config looks like:

```json
{
  "name": "my_rig",
  "pixel_noise_std": 0.3,
  "cameras": [
    {"id": "cam0", "model": "pinhole_radtan", "image_dir": "Input/cam0", "is_reference": true},
    {"id": "cam1", "model": "pinhole_radtan", "image_dir": "Input/cam1"}
  ],
  "boards": [
    {"id": "board_A", "squares_x": 9, "squares_y": 7, "square_length": 0.030,
     "marker_length": 0.022, "dictionary": "DICT_4X4_250", "marker_id_offset": 0}
  ]
}
```

Frames are matched across cameras **by filename stem**, so synchronised
captures must share a name (`cam0/0001.png`, `cam1/0001.png`).

## Conventions

Stated explicitly, because getting these wrong is silent:

- Camera frame is OpenCV: +x right, +y down, +z into the scene.
- `T_a_b` maps points from frame `b` into frame `a`.
- `Camera.extrinsic` is `T_cam_rig`; board poses are `T_rig_board`.
- Poses store `[tx,ty,tz, qx,qy,qz,qw]` (scalar **last**).
- The tangent is `[dt(3); phi(3)]`, translation first, with the **decoupled**
  retraction `t <- t + dt`, `q <- q (x) Exp(phi)`.
- The reference camera's extrinsic is fixed at identity. That is the only gauge
  fixed; scale is already metric from the board.

## Architecture

```
mlti_cal/
  models/      camera models + analytic Jacobians, SE(3) manifolds
  problem/     Problem graph, reprojection residual, robust loss, bootstrap
  solvers/     scipy / ceres / gtsam adapters + the option catalog
  report/      covariance, projection uncertainty, residuals, coverage, cross-val
  detectors/   Charuco detection, multi-board with ID-collision validation
  io/          config, synthetic generation, export
  cli/  gui/   headless entry point; PySide6 views
```

**The core imports zero Qt** and `tests/test_gui.py` enforces it in a fresh
interpreter. The GUI computes nothing — a CLI run and a GUI run cannot produce
different numbers.

## Verification

85 tests. The ones that matter:

- **Jacobian gate** — analytic vs finite differences for every camera model and
  the SE(3) manifold, perturbed *through the retraction*. Nothing downstream is
  trustworthy without it, so it runs first.
- **Cross-backend** — scipy and Ceres reach the same optimum to rel 1e-6, and
  all four Ceres linear solvers agree.
- **Ground-truth coverage** — reduced chi-squared ≈ 1 over an ensemble of noise
  realisations. A deliberately understated sigma is asserted to *fail* the same
  check, so the check demonstrably has teeth.

## Limitations

Stated plainly rather than buried.

- **GTSAM is written but has never been executed.** No `gtsam` wheel exists for
  Windows/CPython 3.12, and `pygtsam` is not a real PyPI package. Use Miniforge
  (`conda install -c conda-forge gtsam`) or WSL. Its two convention conversions
  (rotation-first ordering, coupled exponential) *are* unit-tested here as pure
  functions, but the adapter as a whole is unvalidated — check it against scipy
  before trusting it.
- **The uncertainty is a linearisation.** `Sigma = sigma^2 (J^T J)^+` is exact
  only while parameter errors stay small enough for local linearity. On
  weakly-conditioned problems it degrades: measured over 8 noise realisations,
  max reduced chi-squared was 6.32 at 12 frames, 1.57 at 25, 1.46 at 40. The
  report emits `uncertainty_model_approximate` in that regime. Use ≥20 frames
  if you need the sigmas to be tight.
- **Robust loss is core IRLS, not Ceres' Triggs correction.** Done so every
  backend optimises an identical objective; near convergence with few outliers
  the difference is small, but it is real.
- **Ceres cannot fix individual rotation components.** `EigenQuaternionManifold`
  is all-or-nothing and pyceres 2.6 permits no custom manifold. The adapter
  raises rather than silently optimising a pinned component; the scipy backend
  supports arbitrary masks.
- **Where the distortion model is not invertible, no uncertainty is reported.**
  Those pixels are masked and the report says what fraction of the sensor is
  affected, rather than printing the divergent number OpenCV's undistortion
  produces there.
