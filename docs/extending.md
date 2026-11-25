# Extending mlti_object_cal: your detector, your object, your lens

You can calibrate with your own target detector, your own calibration object
(flat or 3D) and your own camera model **without changing mlti_object_cal**.
You adapt your code to a fixed input and output; the tool does the rest:
bootstrap, bundle adjustment, uncertainty, report, GUI.

| You provide | As | Contract |
|---|---|---|
| A detector / decoder | one function `detect(image, settings)` | [A](#a-detector) |
| A calibration object | a points file, one `id, x, y, z` line per target | [B](#b-object) |
| A camera model | one function `project(params, points)` | [C](#c-camera-model) |

Each plug-in is **one Python file**. Templates to copy: [`examples/plugins/`](../examples/plugins/).

Contract version: **`PLUGIN_API = 1`**. Put that line in your file. A later
version of the tool that changes a contract will refuse a file written for
another version, by name, rather than run it under rules it was not written for.

## Installing a plug-in

Put the file in a plug-in folder and restart the application:

| Folder | When |
|---|---|
| `~/.mlti_cal/plugins/` (on Windows `C:\Users\<you>\.mlti_cal\plugins\`) | always searched |
| any folder named in the `MLTI_CAL_PLUGIN_PATH` environment variable (`;`-separated on Windows) | a shared or project folder |
| `mlti-cal --plugins FOLDER ...` | one command-line run |

In the GUI, **Detectors > Plug-ins...** lists the folders searched, what loaded
and what did not, with the reason. Files whose name starts with `_` are not
loaded, so `_helpers.py` next to your plug-in is a support module.

## Checking a plug-in

```
mlti-cal plugins check my_detector.py
mlti-cal plugins check my_detector.py --image capture_0001.png --points my_object.csv
```

This runs every check the application runs, and also calls your `detect` on a
blank image and, with `--image`, on a real one. It prints what it checked,
what failed and what to change, and returns a non-zero exit code on failure,
so it can run in your own CI. `mlti-cal plugins list` shows what a normal run
would load.

---

## Conventions (all three contracts)

| | |
|---|---|
| Units | metres for 3D, pixels for 2D |
| Pixel coordinates | x to the right, y down; **(0, 0) is the centre of the top-left pixel** (OpenCV) |
| Camera frame | OpenCV axes: x right, y down, z forward along the optical axis |
| Object frame | your own, right-handed; a flat object lies on z = 0 |
| Poses (in outputs) | `[tx, ty, tz, qx, qy, qz, qw]`, quaternion scalar **last**. `T_cam_rig` maps rig to camera coordinates; `T_rig_board` maps object to rig coordinates |

---

## A. Detector

```python
import numpy as np

PLUGIN_API = 1
NAME = "my_detector"  # unique; also the `kind` of its objects
LABEL = "My detector"  # optional, shown in the GUI
DESCRIPTION = "What it finds."  # optional, shown in the GUI
SETTINGS = {"min_diameter_px": 6.0}  # optional: bool, int or float values
SETTINGS_HELP = {"min_diameter_px": "Smallest target accepted, in pixels."}  # optional


def detect(image, settings):
    ...
    return points
```

**Input**

- `image`: a 2D `numpy.ndarray`, `uint8`, grey. It is a copy, so you may modify it.
- `settings`: a `dict` with the keys of `SETTINGS` and the values the user set
  in **Detectors > Detector settings**.

**Output**: a **floating-point** `numpy.ndarray` of shape `(N, 3)`, one row per target:

```
[target_id, x, y]
```

- `target_id` is the identity your decoder read: a whole number, matching an
  `id` in the object's points file.
- `x, y` is the target's measured centre, in pixels, with the convention above.
- Nothing found: return `np.zeros((0, 3))`.

**Rules.** Every return value is checked. A result that breaks one of these
rules costs **that image**: it is skipped, and the log names your plug-in
and the rule. The log is the GUI's Log panel (source `plugins`) or the
terminal. After detection, the number of skipped images is reported too. A
result is never repaired and used.

| Rule | Why |
|---|---|
| a `numpy.ndarray`, floating dtype, shape `(N, 3)` | sub-pixel positions; integer pixels are a 0.3 px error |
| all values finite | |
| ids are whole numbers, **no id twice in one image** | an id must name one point |
| every point inside the image, `-0.5 <= x <= w - 0.5`, `-0.5 <= y <= h - 0.5` | catches swapped `x, y` and `[x, y, id]` column order |
| `detect` does not raise | an exception is caught and costs that image only |

Ids that belong to **no** configured object are ignored, not treated as errors:
other targets in the scene, such as scale bars or a second jig, do no harm.
They are counted and reported after detection, because a large count usually
means the detector and the points file number the targets differently.

**Calling conditions.** `detect` is called once per image, from a background
thread, **never concurrently**. Your module is imported once at start-up.
Expensive set-up such as loading a library or checking a licence can therefore
happen at import time or lazily on the first call. Start-up itself does **not**
call `detect`; only `mlti-cal plugins check` does.

**Your detector in a config**: its objects use `NAME` as their `kind`:

```json
{"id": "my_object", "kind": "my_detector", "points_file": "my_object.csv"}
```

Several objects can share one detector. Each target id must then belong to
exactly one of them: two objects listing the same id are refused at setup,
naming both.

---

## B. Object

A text file, one line per target, comma or whitespace separated, `#` starts a comment:

```
id, x, y, z          # header line optional
7,   0.000, 0.000, 0.0
103, 0.120, 0.000, 0.0
5000, 0.000, 0.080, 0.0
```

- `id`: the id your detector reports; whole numbers, unique, in any order and
  not necessarily contiguous.
- `x, y, z`: metres, in the object's frame.
- **Flat object**: z = 0 on every line. Points that lie on one plane but not on
  z = 0 are refused. Re-express them in a frame whose x-y plane is the surface:
  choosing that frame silently would change what the reported object pose
  means.
- **3D object**: any layout that is not on one plane. At least 6 points must
  be seen in an image to locate it there; at least 4 for a flat one.
- Optional columns `sx, sy, sz` hold each point's standard deviation, in metres.
  They are read and kept but not used in this version; they are reserved for
  refining the object's points in a later version without a format change.

In the GUI: add the object with **+ Object** and choose your detector. Then
select its row and press **Points...** to type the table, paste it from a
spreadsheet, or import a CSV, and save it as the object's points file.

**What the tool does with a 3D object.** A camera that sees only flat objects is
started exactly as before (OpenCV planar calibration). A camera that sees a 3D
object is started differently:

1. A focal-length sweep scored by pose fit.
2. A pose (PnP) per view.
3. A fit of that camera alone in its own model.
4. The full bundle adjustment.

Verified on the example below, and in `tests/test_object_3d.py` against
ground truth.

---

## C. Camera model

```python
import numpy as np

PLUGIN_API = 1
NAME = "my_lens"
LABEL = "My lens"  # optional
PARAM_NAMES = ("fx", "fy", "cx", "cy", "k1")  # first four always fx, fy, cx, cy


def project(params, points):
    ...
    return pixels


# optional
def jacobian(params, points):
    return d_pixels_d_params, d_pixels_d_points


# optional
def default_params(image_size):
    w, h = image_size
    return np.array([w, w, w / 2, h / 2, 0.0])
```

| | |
|---|---|
| `params` | 1D array, one value per `PARAM_NAMES` entry |
| `points` | `(N, 3)` in the camera frame, metres, OpenCV axes, z > 0 |
| `project` returns | `(N, 2)` float pixels, convention above; vectorised (no per-point loop); must not modify its inputs |
| `jacobian` returns | `(N, 2, P)` d(pixels)/d(params) and `(N, 2, 3)` d(pixels)/d(points) |
| `default_params` | a starting guess; without it fx = fy = image width, principal point at the centre, the rest 0 |

- **Without `jacobian`**, derivatives are computed numerically (central
  differences). That costs 2 × (P + 3) extra projections per evaluation, so it
  is slower, but it is correct.
- **With `jacobian`**, it is compared with the numerical derivative when the file
  loads, and a model whose `jacobian` disagrees is **not registered**. A wrong
  derivative does not crash a solve; it quietly stops it short of the minimum.
- At load, the model must also project its default parameters to finite pixels.

Then use `NAME` as a camera's `model` in a config, or pick it in the GUI's camera table.

---

## Worked example: all three together

[`examples/custom_object/`](../examples/custom_object/) uses:

- a **3D object**: three faces of a 24 cm cube covered in ArUco markers, with
  one target per marker (`cube.csv`);
- a **detector plug-in**: [`examples/plugins/aruco_centres.py`](../examples/plugins/aruco_centres.py),
  a short function that returns `[marker id, x, y]` per ArUco marker;
- a **camera-model plug-in** on one of the three cameras:
  [`examples/plugins/my_lens.py`](../examples/plugins/my_lens.py), which has
  no Jacobian.

```
python examples/custom_object/make_dataset.py        # renders 108 images, ~2 minutes
mlti-cal --plugins examples/plugins calibrate examples/custom_object/config.json --backend ceres --crossval -o Output/custom_object
```

Measured against the ground truth in `examples/custom_object/images/truth.json`:

| | |
|---|---|
| reprojection RMS | 0.524 px at the start, 0.320 px after the solve (Ceres, 7 iterations, 0.3 s) |
| held-out error | 1.01 × the training error |
| extrinsics | cam1 (the plug-in lens) 0.27 mm / 0.055°, cam2 0.34 mm / 0.079° |
| focal lengths | within 1.3 px of the truth |
| principal points | within 1.2 px, about one reported standard deviation |

---

## What is protected, and what is not

**Protected.**

- A plug-in that fails to import, breaks a contract or declares another
  `PLUGIN_API` is listed with its error and skipped. The application still
  starts.
- A bad `detect` result or an exception costs one image.
- A config naming a plug-in that is not installed fails with the list of
  available kinds or models.

**Not protected.** Plug-ins run inside the application's process. A crash
inside native code your plug-in calls, such as a C or C++ library that
segfaults, takes the application down with it. If your library can do that,
isolate it yourself, for example by running it in a subprocess from `detect`.
Built-in process isolation is possible in a later version.

## The advanced route

The function contracts above are adapters over the detector registry. A
detector that needs more control than one function gives (its own board
geometry fields, masking behaviour, partial-board logic) can register a full
`DetectorKind`, described in [writing-a-detector.md](writing-a-detector.md).
Both kinds of plug-in live in the same folders.
