# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/);
versioning follows [SemVer](https://semver.org/) from 1.0.0 onward.

## [Unreleased]

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
