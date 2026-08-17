# mlti_object_cal

A multi-camera, multi-object calibration workbench.

You can add as many calibration objects as you need and as many cameras as you
like. Boards and cameras are declared in the setup tab, detected, then solved
together in a single optimisation, so every camera pose and every board pose is
estimated in one consistent frame. Alongside the calibration it reports how much
the result can be trusted: parameter covariance, per-pixel projection
uncertainty, residual maps, coverage and cross-validated held-out error.

## Requirements

Python 3.12.

## Install

Clone the repository, create a virtual environment and install the
dependencies:

```bash
git clone https://github.com/ibird-bot/multi-object-calibration-tool.git
cd multi-object-calibration-tool

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
pip install -e .
```

## Running the GUI

With the virtual environment active:

```bash
python -m mlti_cal.gui
```

or, since the package installs an entry point:

```bash
mlti-cal-gui
```

The application opens on the Setup tab, where you declare your cameras and
calibration boards. Detection, optimisation and the report follow in their own
tabs.

## Camera models

Eight models, each with analytic Jacobians checked against finite differences.

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

## Tests

```bash
pip install pytest
pytest
```
