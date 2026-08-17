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
mlti-cal backends --describe ceres   # the solver option catalog, with guidance
mlti-cal settings                    # every detector/bootstrap/report knob
mlti-cal settings detection          # just one topic
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

A saved config also carries `"detection"` and `"initialization"` blocks holding
every detector and bootstrap setting, written out in full so a calibration can
be reproduced exactly later.

`pixel_noise_std` may be `null`, meaning "I do not know". Residuals are then
left unweighted and the covariance takes its sigma from the residuals, which it
does by default anyway; the only thing lost is the cross-check between assumed
and achieved noise. Note that a robust loss's scale is in *whitened* units, so
switching the assumption off changes where the loss cuts in.

## Measuring the pixel noise

Setup tab -> **Measure...** next to the assumed pixel noise. Point a fixed
camera at a fixed board, take 20+ pictures without touching anything, and every
difference between one frame's corner and the next frame's same corner is
detection noise.

It reports the per-coordinate sigma separately for x and y, and — because
"static" is a claim about the world, not a property of the files — it measures
whole-board motion separately and removes it. If your tripod crept 0.8 px
during the sequence, you are told that instead of being handed a 0.9 px "noise
floor" that is mostly a moving tripod. Duplicate files, drift, anisotropy and
too-rarely-seen corners are each called out.

This measures **repeatability, not accuracy**: a refinement biased by half a
pixel in a consistent direction is perfectly repeatable and looks excellent
here. Systematic error shows up in the residual pattern, not in this number.

## Settings

Nothing that changes a number is hidden. Every knob -- detector, bootstrap,
solver, report thresholds -- is declared once as an `Option` with its default,
its bounds and a note on when to reach for it. That single declaration drives
the GUI widget, its tooltip and the `mlti-cal settings` output, so what you read
in the terminal is what the panel does.

| Group | Count | Examples |
|---|---|---|
| Charuco detector | 17 | corner refinement method and window, aruco thresholds, error correction rate |
| Bootstrap | 14 | PnP method, RANSAC + threshold, point/view floors, pose averaging |
| Solver | 2-9 | linear solver, tolerances, thread count, dense-solve limit |
| Report | 11 | outlier k, correlation threshold, coverage/tilt limits, cross-val folds |

Detector settings belong to a **detector**, not to "detection" — every knob
above is an ArUco/Charuco parameter and always was, and a checkerboard detector
would share none of them. They live in **Detectors → Detector settings…**
(Ctrl+D), one tab per kind, with the kinds that are not written yet listed and
disabled rather than omitted. `mlti-cal settings charuco` prints the same
catalog. In a config they are keyed by kind under `"detectors"`; the older flat
`"detection"` block still loads and is filed under charuco.

Defaults reproduce the previously hardcoded behaviour exactly, and
`tests/test_settings_catalogs.py` asserts that every catalog default still
equals the value the code actually uses.

The solver panel is built from the selected backend and shows **only what that
backend reads** — its own options plus the common ones its adapter actually
uses. scipy has no thread count and the GTSAM adapter sets only a relative
error tolerance, so neither is offered there; a control that silently does
nothing is the failure this panel exists to remove. The declaration is checked
against the adapter source in `tests/test_settings_catalogs.py`, so it cannot
drift.

Starting-point settings open in their own window with an explicit **Apply**.
Nothing takes effect until you apply, so a half-typed threshold is never
briefly the live setting, and the cross-field rules (PnP needs 4 points; a
threshold window's minimum cannot exceed its maximum) are validated while the
window is still open to correct.

Two things stay fixed on purpose. The reference camera's extrinsic is the gauge
(see Conventions), and rotation averaging always uses the quaternion
eigenvector mean, since a componentwise median of a rotation is meaningless.

Changing a detector setting marks existing detections stale; the app asks
before estimating or solving on corners the previous settings produced.

## Camera models

Eight, each with analytic Jacobians gated by finite differences in
`tests/test_jacobians.py`.

| Shown as | id | params | For |
|---|---|---|---|
| OpenCV | `pinhole_radtan` | 9 | Normal and wide lenses. Start here. |
| OpenCV Fisheye | `fisheye_kb` | 8 | Kannala-Brandt equidistant fisheye |
| Double Sphere | `double_sphere` | 6 | Fisheye, 2 distortion params, well conditioned |
| Enhanced Unified | `eucm` | 6 | Fisheye and catadioptric, 2 params |
| Thin Prism | `thin_prism` | 16 | Rational radial + tangential + prism |
| Matlab | `matlab` | 10 | OpenCV plus a skew term |
| FOV | `fov` | 5 | One parameter: the lens field of view |
| Halcon Division | `halcon_division` | 5 | One parameter, closed-form inverse |

Everything except the two fisheye-family models degenerates to the pinhole at
zero distortion, which is what lets all of them be bootstrapped from an OpenCV
pinhole fit — OpenCV can only *fit* its own two. For the rest, `mlti_cal` takes
K from a pinhole fit, discards OpenCV's distortion rather than reinterpreting
its coefficients as the new model's, and reports `nan` for the initialisation
RMS because that number describes OpenCV's model and not this one. PnP then
inverts the model's own projection by Newton on its analytic Jacobian instead
of handing OpenCV coefficients it would misread.

Two things worth knowing before choosing:

- **Double Sphere and Enhanced Unified trade off hard against focal length**
  unless the board covers a genuinely wide field. Measured on the synthetic
  set: Double Sphere sits exactly on the noise floor with `fx = 448` against a
  true `375`, with `xi` absorbing the difference. Both describe the same camera
  over the field observed. The report's correlation warning is what tells you.
- **FOV's `w = 0` is a real degeneracy**, not a numerical one — the factor
  expands as `1 + w²(1/12 − r²/3)`, so the gradient vanishes there and a solver
  started at zero can never move it. Its default is `0.5 rad` for that reason.

Not implemented: **CentralBSpline** and **OCamCalib**. Both are larger than a
new formula — the first needs a variable-length control-point vector, which the
fixed `param_names` contract does not currently allow; the second needs a
polynomial root-find inside `project`, making its Jacobian implicit. Say the
word and they can be done properly.

## The log

One console, docked at the bottom of the window and shared by every tab
(**View → Show log**, Ctrl+L). Each line is tagged with the stage that produced
it — `setup`, `detect`, `noise`, `init`, `problem`, `solve`, `crossval`,
`report` — and the dropdown filters to one stage without discarding the rest.
Copy, save and clear are on the same bar.

Before this, the transcript was split across three widgets: detection wrote to
the status bar, the bootstrap and solve wrote into a box inside the
Optimization tab, and the noise measurement wrote only into its own dialog — so
what a session actually did depended on which tab you were looking at.

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
