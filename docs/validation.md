# Validation against mrcal, Kalibr, MC-Calib and BabelCalib

`mlti_cal` is checked against independent calibration toolkits and published
benchmarks on public image data:

| Dataset | Cameras | Reference | What it checks |
|---|---|---|---|
| mrcal sample set | 1, wide lens | [mrcal](https://mrcal.secretsauce.net/) 2.4.1 | solver, projection, uncertainty |
| OpenCV stereo samples | 2 | mrcal 2.4.1 and OpenCV `stereoCalibrate` | multi-camera solve, extrinsics, uncertainty on a non-reference camera |
| TUM VI `calib-cam8` | 2, fisheye (~190 degrees) | [Kalibr](https://github.com/ethz-asl/kalibr), as published by TUM | fisheye models, real stereo extrinsics, end to end with detection |
| MC-Calib `Seq02` (real photographs) | 6, 1280 x 720 | MC-Calib's saved calibration (no ground truth) | real rig of 3+ cameras, detection to solve |
| MC-Calib `Blender_Images` (rendered) | 2 and 5, rendered | [MC-Calib](https://github.com/rameau-fr/MC-Calib) and the dataset's ground truth | chained multi-camera rig against known poses |
| BabelCalib `Kalibr` benchmark | 8 lenses, one camera each, up to 250 degrees | [BabelCalib](https://github.com/ylochman/babelcalib), Kalibr, OpenCV (published results) | fisheye and FOV models on held-out images, wide-lens bootstrap |

In the mrcal comparisons both tools get the same corner detections, the same
lens model and the same least-squares problem, so any difference comes from
the implementations. The Kalibr comparison cannot do that (Kalibr's own
detections are not published), so it measures the whole pipeline.

## Single camera: mrcal sample set

Dataset: mrcal's public sample set, a 12 mm fisheye lens on a full-frame camera.
84 images of a 14 x 14 chessboard with 58.8 mm spacing, 6000 x 3376 px, boards
0.61 to 1.60 m from the camera. Corners come from the dataset's own
`corners.vnl`. Lens model: `thin_prism`, which is mrcal's `LENSMODEL_OPENCV12`.
mrcal version 2.4.1.

**Reprojection error** (RMS of the corner error distance, in pixels):

| Run | RMS (px) |
|---|---|
| `mlti_cal`, its own bootstrap and Ceres solve | 0.524893 |
| mrcal, its own seed and solve, never shown the `mlti_cal` result | 0.524893 |
| mrcal's cost evaluated at the `mlti_cal` solution, not optimised | 0.524893 |
| mrcal optimising from the `mlti_cal` solution | 0.524893 |

**Projection.** The same rays projected with the `mlti_cal` parameters and
with mrcal's independent parameters differ by at most 0.0007 px over the whole
sensor. fx, fy, cx and cy agree to four decimals.

**Projection uncertainty** (worst-direction 1-sigma, at the same 120 pixels):

| Range | `mlti_cal` median | mrcal median | Ratio, median | Ratio, min to max |
|---|---|---|---|---|
| 1.05 m | 0.0297 px | 0.0295 px | 1.008 | 0.992 to 1.015 |
| 100 m | 0.0775 px | 0.0769 px | 1.008 | 0.994 to 1.009 |

Before the fix described in the changelog, the map held the query point fixed
in camera coordinates and reported a median of 0.155 px at both ranges, five
times mrcal's value at 1.05 m.

## Two cameras: OpenCV stereo samples vs mrcal and OpenCV

Dataset: the `left01..14.jpg` / `right01..14.jpg` pairs shipped in OpenCV's
`samples/data` (13 pairs, 640 x 480, a 9 x 6 inner-corner chessboard). The
square size is not published; 25 mm is assumed, which scales translations
only. Corners are found by `mlti_cal`'s own checkerboard detector (11 of 13
images per camera) and handed to all three tools. Lens model `pinhole_radtan`
= mrcal `LENSMODEL_OPENCV5` = OpenCV's default 5 coefficients.

| Run | RMS (px) |
|---|---|
| `mlti_cal`, its own bootstrap and Ceres solve | 0.267679 |
| mrcal, its own seed and solve | 0.267679 |
| mrcal's cost at the `mlti_cal` solution | 0.267679 |
| mrcal optimising from the `mlti_cal` solution | 0.267679 |

- **Intrinsics**, both cameras: every parameter equals mrcal's within 0.001
  of its 1-sigma. OpenCV's `stereoCalibrate` lands within 1.1 sigma (it uses
  only the frames both cameras saw, so it solves a slightly different problem).
- **Extrinsics**: the right camera's centre differs from mrcal's by 7e-10 m;
  from OpenCV's by 0.04 mm and 0.08 degrees.
- **Projection uncertainty**, both cameras including the non-reference one, at
  the median board range and at 100x it: the ratio to mrcal is 1.021 at the
  median pixel. That is exactly sqrt(N / (N - p)) = 1.0208 for N = 2376
  residuals and p = 96 parameters: `mlti_cal` estimates the noise from the
  residuals with the degrees-of-freedom correction, mrcal without it. On the
  84-image single-camera set above the same factor is 1.008. Maximum ratio
  1.137, at a few pixels at the edge of the left image.

## Fisheye stereo: TUM VI vs Kalibr

Dataset: TUM VI `dataset-calib-cam8_512_16` (EuRoC export, CC BY 4.0): 202
synchronised image pairs from two 512 x 512 fisheye cameras looking at a 6 x 6
Kalibr AprilGrid (tag 88 mm, gap 0.3 x tag). Reference: the Kalibr
pinhole-equidistant calibration TUM publishes for this same sequence
(`pinhole-equi-512.zip`). Kalibr's pinhole + equidistant model is exactly
`fisheye_kb`.

`mlti_cal` has no AprilGrid detector, so `tools/kalibr_compare/aprilgrid.py`
reads the grid with OpenCV's AprilTag detector (2-bit border, contour corner
refinement). It found the grid in 158 (cam0) and 163 (cam1) of the 202 images per camera,
17 032 corners in all. The tag
layout was measured, not assumed: of the 16 possible corner orders only one
fits images undistorted with Kalibr's intrinsics.

**Detector bias.** OpenCV's corners sit inside the printed black squares. With
the nominal tag size the fit is 1.27 px RMS. Shrinking the modelled tag about
its centre (the tag spacing, which sets the metric scale, is untouched) gives
the lowest RMS at 0.93 x nominal, chosen by `mlti_cal`'s own RMS, not by
agreement with Kalibr:

| Tag edge used | RMS cam0 / cam1 (px) | Kalibr's parameters on the same corners | Baseline | Projection vs Kalibr, median / max |
|---|---|---|---|---|
| 1.00 x 88 mm | 1.266 / 1.257 | 1.274 / 1.261 | 99.82 +- 0.29 mm | 3.3 / 5.2 px |
| 0.93 x 88 mm | **0.341 / 0.352** | 0.344 / 0.353 | 100.59 +- 0.08 mm | 1.1 / 1.7 px |
| 0.90 x 88 mm | 0.556 / 0.561 | 0.557 / 0.562 | 100.91 +- 0.13 mm | 0.2 / 0.7 px |

Kalibr: baseline 101.09 mm, reported mean reprojection error 0.094 px with its
own detector.

What the table shows:

- **The solver is not the limit.** In every row, Kalibr's published cameras,
  frozen, fit `mlti_cal`'s corners only 0.001 to 0.008 px worse than
  `mlti_cal`'s own solution. Both solutions explain these corners almost
  equally well; they differ along a weakly constrained direction (focal length
  against board distance) that the detector's corner bias decides.
- **The result depends on the detector.** Projection agreement with Kalibr
  moves from 3.3 px to 0.2 px with the corner-bias setting. With OpenCV's
  AprilTag corners, expect about 1 px of projection and 0.5% of baseline
  difference from Kalibr's published calibration on this rig.
- **Rotation between the cameras** agrees with Kalibr within 0.024 degrees
  (1-sigma 0.019 degrees) at 0.93.
- **Three fisheye models** -- `fisheye_kb`, `double_sphere`, `eucm` -- reach
  the same RMS to 0.0002 px and the same baseline to 0.03 mm.

This comparison found a bug: EUCM started at alpha = beta = 0, a point where
both of its distortion gradients are exactly zero, so it never left the
pinhole and stopped at 3.36 px. Fixed (see the changelog); the synthetic
end-to-end test had passed because a pinhole fits that narrow synthetic lens
under its threshold.

## Five cameras with ground truth: MC-Calib's rendered rig

Dataset: `Blender_Images` from the MC-Calib repository (Rameau et al., CVIU
2022): Blender renders with ground-truth intrinsics and camera poses, and the
calibration MC-Calib itself produced. Scenario 2 has five 1824 x 1376 cameras
fanned in 15 degree steps and a single 5 x 5 Charuco board (0.192 m squares, 6 x 6
ArUco dictionary), so only neighbouring cameras ever see the board together and
the rig can only be solved by chaining. These are renders, not photographs.
The board's marker size is not published; 0.75 of the square was measured from
the images, and 0.70 and 0.80 change the RMS by 0.0003 px and the poses by less
than 0.01 degrees and 0.04 mm. The board was found in 36, 40, 20, 44 and 19 of
the 99 frames per camera (MC-Calib's own result uses 30, 34, 17, 40 and 16).

**Reprojection error** on identical corners (this tool's detections; for the
last three rows only the board poses are re-fitted):

| Cameras | RMS (px) |
|---|---|
| `mlti_cal` solution | 0.0574 |
| MC-Calib's saved calibration | 0.0578 |
| Published ground truth | 0.668 |
| Ground truth with only `fy` replaced by this tool's value | 0.0687 |

The published `fy` (2093.9 px) is inconsistent with the images: this tool and
MC-Calib both find `fy` of about 2106 px, 11 to 13 px higher, and replacing that
one number brings the ground-truth cameras to 0.069 px. In Scenario 1 (two
cameras) the published `K` of camera 0 is 55 px off in `fx` and 86 px in `fy` from
both tools' fits, whose `fx` and `fy` agree with each other to 0.1 px (RMS 0.0554 px
and 0.0556 px on identical corners; the published cameras give 4.96 px). The cause is
not known. Intrinsics are therefore judged by RMS, and poses against the ground
truth.

**Poses**, T_cam_i<-cam0 against the ground truth, four non-reference cameras:

| Camera | `mlti_cal` rotation / centre | MC-Calib rotation / centre | `mlti_cal` Mahalanobis^2 (6 dof) |
|---|---|---|---|
| 1 | 0.026 deg / 0.48 mm | 0.008 deg / 0.38 mm | 3.4 |
| 2 | 0.029 deg / 0.75 mm | 0.029 deg / 0.49 mm | 4.6 |
| 3 | 0.024 deg / 1.11 mm | 0.003 deg / 0.61 mm | 10.3 |
| 4 | 0.033 deg / 1.30 mm | 0.026 deg / 0.62 mm | 7.0 |

- **MC-Calib's saved poses are closer to the truth**: in centre distance on
  every camera, in rotation on three of four (camera 2 ties), by up to 0.7 mm
  and 0.03 degrees. Both are small against the 0.5 to 1 m baselines.
  Scenario 1 shows the same ordering (0.012 degrees and 0.25 mm against 0.004
  degrees and 0.03 mm). This tool is not better here; the cause was not found.
- **The reported uncertainty is consistent with the error.** The squared error
  of each pose, weighted by this tool's own covariance, should average 6 for 6
  degrees of freedom. It averages 6.3 over the four poses of Scenario 2 and 6.6
  over the five poses of both scenarios (sum 33.0, expected 30). Five poses are
  a consistency check, not a calibration of the covariance.

Scenarios 3 to 5 (8 boards in four groups, and a cube) were not run. Their
boards form one rigid object whose layout MC-Calib estimates itself. This tool
treats boards as independent, so cameras that never share a board cannot be
linked, and it refuses to start and says so: "camera(s) never observe a board at
the same time as the reference camera". A rigid multi-board object of unknown
layout is a feature MC-Calib has and this tool does not.

## Six real cameras: MC-Calib's `Seq02`

Dataset: `Real_Images/Seq02_Overlapping_multicamera` from the MC-Calib release:
six cameras (1280 x 720) photographing one 5 x 5 Charuco board (6 x 6
dictionary, marker ids from 144, squares of 9.15 in the dataset's own unit),
917 frames per camera, and the calibration MC-Calib produced. There is **no
ground truth**, so this shows agreement between two tools, not accuracy. Every
4th frame was used (229 per camera); the board was found in 215, 218, 215, 210,
207 and 202 of them. The marker size is not published; 0.75 of the square was
measured on frontal views.

- **Reprojection error** on identical corners: 0.3736 px for `mlti_cal`; 0.3754 px
  for MC-Calib's saved cameras with only the board poses re-fitted.
- **Intrinsics** (fx, fy, cx, cy): within 0.4 px of MC-Calib's on every camera,
  and within 1.4 of this tool's 1-sigma except the principal-point y of cameras
  1 and 2 (2.2 and 1.5 sigma).
- **Poses** T_cam_i<-cam0 against MC-Calib's: rotation within 0.022 degrees,
  camera centre within 0.024 units (0.26% of one board square). The squared
  difference weighted by this tool's covariance is 2.9, 28.9, 13.5, 12.0 and 8.4
  for the five non-reference cameras (6 expected if MC-Calib were exact; it is
  not, so this does not test the covariance).

## Fisheye lenses: BabelCalib's benchmark

Dataset: the `Kalibr` split of BabelCalib's public benchmark (Lochman et al.,
ICCV 2021): eight cameras (EuRoC, TUM VI, GoPro, four Basler lenses, and the
250 degree ENTANIYA), a 6 x 6 AprilGrid, with the corners already detected and
split into 35 train and 15 test images per camera. Detection is not part of
this comparison. Protocol, as in the paper: calibrate on `train`, freeze the
intrinsics, fit only the pose of each `test` image, report the reprojection RMS.

Test RMS in px, plain RMS over all test corners:

| Lens | Kannala-Brandt | Double Sphere | EUCM | FOV |
|---|---|---|---|---|
| BF2M2020S23 | 2.124 | 0.212 | 0.209 | 2.006 |
| BF5M13720 | 0.196 | 0.202 | 0.198 | 0.398 |
| BM2820 | 0.201 | 0.229 | 0.201 | 0.245 |
| BM4018S118 | 0.280 | 0.283 | 0.280 | 0.306 |
| ENTANIYA (250 deg) | 5.785 | 0.519 | 0.542 | 6.451 |
| EUROC | 0.158 | 0.162 | 0.159 | 0.163 |
| GOPRO | 0.192 | 0.213 | 0.187 | 0.193 |
| TUMVI | 0.257 | 0.195 | 0.195 | 0.306 |
| **Mean over lenses under 1 px** | 0.214 (6 of 8) | 0.252 (8 of 8) | 0.247 (8 of 8) | 0.268 (6 of 8) |

Against the paper's Table 5, row `Kalibr` (mean test RMS over the eight cameras,
failures excluded, failures in brackets):

| | KB | EUCM | FOV | Double Sphere |
|---|---|---|---|---|
| Kalibr | 0.194 (3 of 8) | 0.250 | 0.257 (3 of 8) | 0.264 |
| BabelCalib | 0.173 | 0.230 | 0.237 | 0.326 |
| `mlti_cal` | 0.214 (2 of 8) | 0.247 | 0.268 (2 of 8) | 0.252 |

The paper's number is a robust, weighted RMS whose scale it does not state, so
this is a reference, not a like-for-like score. The Huber-weighted version
(scale 1 px) gives KB 0.205, Double Sphere 0.246, EUCM 0.240 and FOV 0.253 px,
and `mlti_cal`'s plain RMS is the larger of the two.

- **Double Sphere and EUCM** reach 0.25 px on all eight lenses, including the
  250 degree one, level with Kalibr's published numbers and within 0.02 px of
  BabelCalib's EUCM.
- **Kannala-Brandt and FOV fail on the two widest lenses** (BF2M2020S23 and
  ENTANIYA: 2 to 6 px) although Double Sphere and EUCM fit the same corners at
  0.2 and 0.5 px. BabelCalib reports no failures for either, so `mlti_cal` is
  behind it here. The cause is in the model code, not the optimiser: both are
  written on the normalised plane (x/z, y/z) with z clamped, as in OpenCV's
  fisheye model, so every ray at or past 90 degrees off-axis projects to the same
  radius. Measured on one lens: Kannala-Brandt gives 303.0 px at 95, 110 and 125
  degrees alike, FOV 235.6 px, while Double Sphere and EUCM keep growing. For a
  lens seeing more than 180 degrees in total, use Double Sphere or EUCM.
- **OpenCV's own fisheye calibration**, run on the same corners, is above 1 px or
  raises on 4 of 8 lenses (the paper reports 3 of 8).

This benchmark found a bug. Board poses for non-pinhole lenses came from a PnP
on (x/z, y/z), which is wrong for rays at or past 90 degrees, so on lenses seeing
more than that every outer corner corrupted its view's pose and the bundle
adjustment started in a bad minimum, reporting success at 25 to 200 px of training error. Poses are
now computed from the corners inside 85 degrees and refined against the camera's
own projection (changelog). Test RMS before and after:

| Lens, model | Before | After |
|---|---|---|
| TUMVI, Double Sphere / EUCM / FOV | 27.3 / 22.8 / 46.1 | 0.195 / 0.195 / 0.306 |
| BF2M2020S23, Double Sphere / EUCM | 53.5 / 75.3 | 0.212 / 0.209 |
| ENTANIYA, Double Sphere / EUCM | 98.9 / 95.9 | 0.519 / 0.542 |
| ENTANIYA, Kannala-Brandt / FOV | 234.5 / 115.9 | 5.785 / 6.451 |
| BF2M2020S23, Kannala-Brandt / FOV | 2.008 / 127.2 | 2.124 / 2.006 |
| TUMVI, Kannala-Brandt | 0.255 | 0.257 |

Kannala-Brandt moves by a few percent on the two lenses where it already could
not fit (2.008 to 2.124 px, 0.255 to 0.257 px on TUM VI): its failure there is the
model limit above, not the starting poses.

Lenses within 85 degrees take the old path. Re-run on the new code, nothing else
moved: TUM VI against Kalibr gives the same RMS (0.3411 / 0.3516 px) and baseline
(100.588 +- 0.080 mm), the four-camera example the same final cost (57.5406, RMS
0.1314 px), and the mrcal single-camera comparison the same 0.5249 px. One case remains: on a synthetic Double
Sphere-like lens with most corners (870) past 90 degrees, EUCM's own start still
ends at 12 px; with 484 corners past 90 degrees both models reach the noise floor.

## What this does and does not show

- The projection functions, conventions, the multi-camera solve and the
  uncertainty propagation match mrcal to the reported precision, on one
  single-camera and one two-camera real dataset.
- The fisheye models and the stereo extrinsics agree with Kalibr's published
  calibration of a real rig to within what the corner detector explains.
- The five-camera MC-Calib rig is rendered. Its ground truth has a wrong `fy`;
  poses are the check, not intrinsics. MC-Calib's saved poses are closer to the
  truth than this tool's, by about 0.5 mm.
- The six-camera real sequence has no ground truth: it shows agreement with
  MC-Calib, not accuracy.
- Not covered: the Halcon division model against any external tool, and rigid
  multi-board objects of unknown layout (not supported). The other MC-Calib real
  sequences (non-overlapping cameras, hybrid, converging) were not run.
- The uncertainty is a linearisation in both tools. On weakly constrained
  problems it understates the true spread away from the covered area. On a
  25-frame synthetic set with 0.3 px noise, Monte Carlo re-solves gave 4.1 px
  where the map predicted 1.8 px at a quarter of the image width. At 0.03 px
  noise the two agree. `tests/test_uncertainty.py` checks the map against Monte
  Carlo re-solves.
- mrcal 2.4.1 computes uncertainty only with its regularisation terms on, so
  its uncertainty numbers come from a re-solve with them enabled. That re-solve
  has an RMS of 0.5250 px.

## Reproduce

mrcal runs on Linux. On Windows, use WSL with Ubuntu 24.04:

```
sudo apt install mrcal python3-mrcal
```

Download the corners file, then run the three steps from the repository root.
The first and third run in the project environment, the second in Linux.

```
curl -o Output/mrcal_compare/data/corners.vnl https://mrcal.secretsauce.net/external/2022-11-05--dtla-overpass--samyang--alpha7/2-f22-infinity/corners.vnl

python  tools/mrcal_compare/run_mlti.py  Output/mrcal_compare/data/corners.vnl Output/mrcal_compare/mlti.json --model thin_prism
python3 tools/mrcal_compare/run_mrcal.py Output/mrcal_compare/data/corners.vnl Output/mrcal_compare/mlti.json Output/mrcal_compare/mrcal.json
python  tools/mrcal_compare/compare.py   Output/mrcal_compare/mlti.json Output/mrcal_compare/mrcal.json
```

Both sides use unit weights, no outlier rejection, no board-warp term and no
regularisation in the solve. Conventions: OpenCV camera frame (+x right, +y
down, +z forward), board poses map board to camera, corner `k = i * 14 + j` is
at `(j, i, 0) * spacing`.

Two cameras (images from OpenCV's GitHub repository, `samples/data`, saved as
`Output/opencv_stereo/images/left01.jpg` etc.):

```
python  tools/mrcal_compare/run_mlti_multicam.py Output/opencv_stereo/result --images Output/opencv_stereo/images --camera left "left*.jpg" --camera right "right*.jpg" --cols 9 --rows 6 --spacing 0.025
python3 tools/mrcal_compare/run_mrcal_multicam.py Output/opencv_stereo/result
python  tools/mrcal_compare/compare_multicam.py Output/opencv_stereo/result
```

Extrinsics use mrcal's convention: camera 0 is the reference,
`extrinsics_rt_fromref[i-1]` maps camera-0 points into camera i.

TUM VI (download `dataset-calib-cam8_512_16.tar` from
`https://vision.in.tum.de/tumvi/exported/euroc/512_16/` and extract it to
`Output/tumvi/calib-cam8/`):

```
python tools/kalibr_compare/run_tumvi.py Output/tumvi/calib-cam8/dataset-calib-cam8_512_16/mav0 Output/tumvi/result.json --edge-scale 0.93
```

Kalibr's numbers are copied into the script from TUM's
`camchain-imucam-imucalib.yaml`; its `T_cn_cnm1` is `T_cam1_cam0`, the same
quantity as `mlti_cal`'s `extrinsic` of cam1 with cam0 as reference.

MC-Calib (download `Blender_Images` and `Real_Images` from the MC-Calib README;
extract to `Output/mccalib/synth/` and `Output/mccalib/real/`; `Real_Images` is
45 GB, only `Seq02_Overlapping_multicamera/Cam_00*` and the `.yml` files are needed):

```
python tools/mccalib_compare/run_scenario.py Output/mccalib/synth/Blender_Images/Scenario_2 Output/mccalib/scenario2.json
python tools/mccalib_compare/run_scenario.py Output/mccalib/synth/Blender_Images/Scenario_1 Output/mccalib/scenario1.json
python tools/mccalib_compare/run_real.py Output/mccalib/real/Real_Images/Seq02_Overlapping_multicamera Output/mccalib/real_seq02.json --stride 4
```

Ground-truth `P_i` is T_cam_i<-cam_1 in Blender's frame (+y up, looks down -z);
the script converts it with diag(1, -1, -1, 1). MC-Calib's `camera_pose_matrix`
is T_cam0<-cam_i; the script inverts it.

BabelCalib (download `Kalibr.zip` from the `datasets` release of
`github.com/ylochman/babelcalib`, extract to `Output/babelcalib/`):

```
python tools/babelcalib_compare/run_babelcalib.py Output/babelcalib/Kalibr Output/babelcalib/kalibr_results.json
```
