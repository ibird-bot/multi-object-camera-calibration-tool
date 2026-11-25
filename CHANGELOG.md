# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/);
versioning follows [SemVer](https://semver.org/) from 1.0.0 onward.

## [Unreleased]

## [1.1.0] - 2026-11-25

### Added
- Validation against mrcal 2.4.1 on a public real dataset (`docs/validation.md`,
  `tools/mrcal_compare`). Same corners, same lens model, same problem: RMS
  0.524893 px in both tools, projections within 0.0007 px, projection
  uncertainty within 1.5%.
- Multi-camera validation on real images (`docs/validation.md`). Against
  mrcal and OpenCV on OpenCV's stereo samples (`tools/mrcal_compare/
  *_multicam.py`): identical RMS (0.267679 px), intrinsics within 0.001 sigma,
  camera centres within 1e-9 m, uncertainty ratio 1.021 on both cameras, which
  is exactly the degrees-of-freedom factor. Against Kalibr's published
  calibration of the TUM VI fisheye stereo rig (`tools/kalibr_compare`):
  Kalibr's cameras fit `mlti_cal`'s corners within 0.003 px of `mlti_cal`'s
  own solution; projection differs by ~1 px and baseline by 0.5 mm, set by the
  OpenCV AprilTag corner bias rather than the solver.
- Validation against MC-Calib's rendered multi-camera data and BabelCalib's
  public fisheye benchmark (`docs/validation.md`, `tools/mccalib_compare`,
  `tools/babelcalib_compare`). Five chained cameras: 0.0574 px against 0.0578 px
  for MC-Calib's saved calibration on identical corners; poses within 0.033
  degrees and 1.3 mm of the ground truth, and the reported covariance matches the
  observed error (squared error 6.3 on average, 6 expected). MC-Calib's own poses
  are closer to the truth, by up to 0.7 mm. On MC-Calib's real six-camera
  sequence (no ground truth) the fit on identical corners is 0.3736 px against
  0.3754 px for MC-Calib's saved cameras, with poses within 0.022 degrees. Eight
  real fisheye lenses on held-out
  images: Double Sphere 0.252 px and EUCM 0.247 px on all eight, level with the
  published Kalibr numbers; Kannala-Brandt and FOV fail (2 to 6 px) on the two
  widest lenses, where BabelCalib reports no failures. The cause is that both
  models are written on the normalised plane with z clamped, so every ray at or
  past 90 degrees projects to the same radius (Kannala-Brandt: 303.0 px at 95,
  110 and 125 degrees); use Double Sphere or EUCM beyond 180 degrees in total.
- Report warning `intrinsic_sigma_unreliable`: when two parameters of one
  camera correlate beyond 0.999, that pair and every parameter of the camera
  correlated above 0.98 with them have their individual sigmas called
  untrustworthy. On the four-camera example it names the Double Sphere fx, fy,
  xi and alpha, which sit 30-49 of their own sigmas from the truth while their
  projection is within 0.3 px of it. It stays silent on four healthy synthetic
  sets whose strongest intrinsic correlation is 0.9955. The covariance itself
  is unchanged.
- A Monte Carlo test of the projection-uncertainty map: 80 re-solves under
  fresh pixel noise must reproduce what the map predicts (marked `slow`).
- Plug-in contracts (`docs/extending.md`, `mlti_cal.plugins`): a detector is
  one function `detect(image, settings) -> [target_id, x, y]` rows, an object
  is an `id, x, y, z` points file, and a camera model is one `project(params,
  points)` function, with a numerical Jacobian when none is given. Every
  plug-in result passes a validation gate. A violation skips that image, or
  refuses that model, and names the plug-in and the rule broken.
  `PLUGIN_API = 1`.
- 3D calibration objects. A camera that sees one is started with a focal sweep,
  PnP and a fit in its own model. Cameras that only see flat targets are
  bootstrapped exactly as before.
- `mlti-cal plugins list | check FILE [--image --points]`, and a global
  `--plugins FOLDER` option.
- GUI: **Detectors > Plug-ins...** (what loaded, what failed and why) and a
  **Points...** editor for plug-in objects (type, paste or import CSV).
- Plug-ins are searched in `~/.mlti_cal/plugins/` and `MLTI_CAL_PLUGIN_PATH`.
  The `detectors/` folder and `MLTI_CAL_DETECTOR_PATH` keep working.
- `examples/custom_object`: a 3D marker cube, a detector plug-in and a lens
  plug-in calibrated together against ground truth.
- `mlti_cal.io.render`: renders board images through the true lens models,
  with ground truth, so the whole pipeline can be tested from pixels.
- `examples/rig_4cam`: a four-camera, three-board example built with it.
- `examples/real_6cam`: six real RealSense infrared cameras from MC-Calib's
  public data (prepared by its `prepare.py`; the images are not shipped). The
  README GIF and uncertainty figure now come from this real rig, the figure via
  `tools/make_readme_figure.py`.
- "Generate printables" in the GUI: one A4 PDF per Charuco board, at true
  physical scale where it fits, with a ruled bar to check the print.

### Changed
- The GUI was redesigned:
  - **Look:** light and dark themes that follow Windows (View > Theme), and
    a step rail (Setup & detection, Solve, Report) with a done or needs-a-look
    mark on each step.
  - **Status:** a thin progress bar while work runs. The log is a drawer,
    with the newest line in the status bar and a warning count on its toggle.
  - **Solve:** result cards (error before and after, iterations and time,
    held-out error). The convergence chart is in real pixel values, with
    the solved value drawn in. On a log axis a 0.64 to 0.13 px solve looked
    like a drop to nothing. When the first step takes most of the drop, the
    axis is split and the lower part is zoomed onto the remaining steps. Backend options are folded away.
  - **Report:** a plain-language verdict with four numbers and a per-camera
    table first. Warnings are grouped by topic, one figure is shown at a time
    and the text report is a separate page.
  - **Setup:** tables have no row numbers or cell grid, and inputs inside a
    row show their box only on hover or focus. The detections list sits under
    the form, and the image preview has the right side. Configs are opened and
    saved from a new **File** menu (Ctrl+O, Ctrl+S).
  - **Correlation figure:** labels each camera's block once when there are
    more than 24 parameters, and shows the 8 least separable pairs.
- `calibration.json` board metadata lists every field the board's kind declares.
  For example, Charuco boards now also record `marker_length` and
  `marker_id_offset`.

### Fixed
- **Lenses seeing more than 90 degrees off-axis started the solve from corrupt
  board poses.** For every non-pinhole model the pose of each view came from a
  PnP on (x/z, y/z), which is wrong for rays at or past 90 degrees (z = 0, then
  the point flips through the origin), so a view whose outer corners were that
  wide started the bundle adjustment in a bad minimum, and the solve reported
  success at 25 to 200 px of training error. Found on BabelCalib's benchmark:
  held-out RMS on TUM VI (190 degrees) went from 27.3 px (Double Sphere) and
  22.8 px (EUCM) to 0.195 px for both; on BF2M2020S23 from 53.5 and 75.3 px to
  0.212 and 0.209 px; on the 250 degree ENTANIYA from 98.9 and 95.9 px to 0.519
  and 0.542 px. PnP now uses only the corners inside 85 degrees, and the pose is
  refined against the camera's own projection with all corners. A lens that
  stays inside 85 degrees takes the old path unchanged. Re-run on the new code
  with identical results: TUM VI against Kalibr (0.3411 / 0.3516 px, baseline
  100.588 mm), the four-camera example (final cost 57.5406, RMS 0.1314 px) and
  the mrcal single-camera comparison (0.5249 px). Kannala-Brandt on the two
  widest benchmark lenses moves by a few percent (2.008 to 2.124 px) because its
  failure there is the model limit above. Still open: on a synthetic lens with
  most corners past 90 degrees, EUCM's own-model start can still end at 12 px.
- **EUCM could not leave its starting point.** It started at alpha = beta = 0,
  where both distortion gradients are exactly zero (beta = 0 makes the model a
  pinhole for every alpha), so every solve stayed a pinhole. Found on the real
  TUM VI fisheye rig: 3.36 px RMS, against 0.34 px for Double Sphere and
  Kannala-Brandt on the same corners. It now starts at beta = 1, still exactly
  a pinhole, and reaches 0.341 px with alpha 0.63, beta 1.04. The synthetic
  end-to-end test had passed because a pinhole fits that narrow synthetic lens
  under its threshold; a new test checks the start is not a stationary point.
  Configs that set EUCM's starting parameters explicitly are unaffected.
  Visible change: on a narrow lens, where only the product of alpha and beta
  is constrained, the two can now slide apart along that valley (alpha 3441,
  beta ~0 was seen on a synthetic set) instead of sitting at 0 / 0. The
  projection is still right; the individual values are meaningless there,
  which the correlation warnings report.
- **The scipy backend and the covariance were ~10x slower than they need be
  on hybrid CPUs.** OpenBLAS's default thread pool oversubscribes them; the
  dense SVD that `trf` runs every iteration took 17 s on the four-camera
  Jacobian (13338 x 573). Both now cap BLAS threads with `threadpoolctl`
  (a new core dependency, pure Python), the solve at `num_threads` (default
  1). Measured on an i7-12700H with OpenBLAS 0.3.33, same start, same final
  cost 57.540568418 and RMS 0.131363 px before and after: rig_4cam scipy
  solve 349 s -> 45.6 s, covariance 17.6 s -> 2.0 s, demo solve 1.71 s ->
  0.98 s. Other machines will differ. Ceres remains much faster (2.7 s on
  rig_4cam) and stays optional.
- **The projection-uncertainty map reported the wrong quantity.** It held the
  query point fixed in camera coordinates, so it showed how loosely the camera
  frame is defined (about sigma(cx) at the image centre), not how far a
  physical point projects wrong. The point is now held fixed relative to the
  calibration boards and averaged over all board poses (mrcal's "mean-pcam"
  method). Measured on a 1-camera, 25-frame synthetic set at 1.0 m against 200
  Monte Carlo re-solves: centre 10.35 px before, 0.81 px after, 0.83 px
  measured. **Every number in the map changes**, usually downward near the
  covered centre. `projection_uncertainty(..., reference="camera")` returns
  the old quantity, and `UncertaintyMap` and `report.json` carry a
  `reference` field saying which one was computed.
- The projection-uncertainty map now linearises at the values in the solved
  problem, not at the copies in the system description. A report built before
  `write_back` used stale poses with a fresh covariance. Checked on a
  non-reference camera of a 2-camera rig against 200 Monte Carlo re-solves:
  the map is within 5% at every sampled pixel (0.0300 px vs 0.0289 px measured
  at the centre, 1.796 vs 1.782 px off-axis, at 0.03 px noise).
- Cross-validation re-solved every fold with a plain squared loss and every
  camera parameter free, whatever the user had solved with. Folds now take
  the solved problem as `template=` and keep its robust loss, its pinned
  intrinsics and its frozen camera blocks; the GUI, the report and
  `--crossval` all pass it. Measured on the 2-camera demo with 3% outliers,
  a Huber loss and k3 pinned: training RMS 3.867 -> 3.935 px, held-out
  3.925 -> 3.920 px, ratio 1.015 -> 0.996 (unweighted RMS, outliers
  included). Called without `template`, behaviour is unchanged.
- The Charuco "board geometry is wrong" warning fired on correctly configured
  boards seen at steep angles: a 9x7 board tilted 75-80 degrees decodes 5-15
  markers and interpolates no corners, and was told to swap squares_x and
  squares_y. The marker/corner counts now only trigger a test: the same image
  is re-detected as the transposed board and with the pre-OpenCV-4.6 layout,
  and the warning names whichever one recovers corners. If neither does,
  nothing is said.
- If a report figure failed to draw, the GUI's error path itself raised
  `AttributeError`: `SourceLog.warning` was called but never existed.
- With a larger font, image thumbnails were clipped to a corner. Qt sized
  every cell from the first item, which has no image yet.
- Calibrating from a config whose objects use a detector plug-in raised
  `AttributeError`, because the board metadata assumed every board had
  `squares_x`. This affected the shipped `aruco_grid` example too.
- The projection-uncertainty map raised for five of the eight camera models
  (Double Sphere, EUCM, FOV, Halcon division, Matlab): their parameters were
  handed to OpenCV as OpenCV distortion coefficients.
- Rigs where a camera never shares a board with the reference camera could
  not be bootstrapped, and configs whose cameras share frame names only
  pairwise were refused. Cameras are now chained through their neighbours.
- Non-OpenCV lens models started from a pinhole K with zero distortion; on a
  wide Double Sphere lens that put its extrinsic 0.8 m and 27 degrees off.
  They are now fitted in their own model before the rig is assembled.
- `--crossval` on the command line always re-solved with scipy, whatever
  `--backend` was chosen.
- `CharucoDetector.render` drew squares smaller than asked when a margin was
  set (12% on a 210 px/square board). The image is now `2 * margin` larger.

## [1.0.0] - 2026-09-27

First stable release.

### Added
- Multi-camera calibration for rigs of any size; multi-object calibration for
  several targets in one session, with marker ID collision checking.
- Three target types -- Charuco, checkerboard, dot grid -- mixable in the same
  images.
- Eight camera models with analytic Jacobians: pinhole, fisheye
  (Kannala-Brandt), double sphere, EUCM, thin prism, Matlab, FOV, Halcon
  division.
- Joint intrinsic/extrinsic bundle adjustment, scipy and Ceres solver
  backends.
- Uncertainty reporting: parameter covariance, per-pixel projection
  uncertainty, residual and correlation maps, coverage, cross-validated
  held-out error -- instead of a single RMS number.
- Desktop GUI (PySide6) and a headless CLI on the same code path, so a CLI run
  and a GUI run cannot disagree.
- Pluggable detectors: a third-party detector reaches detection through the
  same path as the built-ins, no changes to the installed package needed.

### Removed
- The GTSAM solver backend, which was implemented and then dropped: GTSAM has
  no Windows wheel, and scipy + Ceres already cover the multi-solver
  requirement natively.
