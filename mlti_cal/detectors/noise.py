"""
Measure corner detection noise from a STATIC sequence.

The idea
    Point a fixed camera at a fixed board and take N pictures without touching
    anything. Nothing in the world changed, so every difference between one
    frame's corner and the next frame's SAME corner is detection noise: sensor
    read noise, quantisation, and the subpixel refinement reacting to both. The
    per-corner standard deviation across the sequence IS the pixel noise, in
    the same per-coordinate units the reprojection residual is whitened by.

What this does NOT measure
    Repeatability, not accuracy. A refinement that is biased by half a pixel in
    a consistent direction is perfectly repeatable and this reports it as
    excellent. Systematic error shows up in the residual pattern, not here.

The failure mode that matters
    "Static" is a claim about the world, not a property of the files. If the
    board sags, the tripod creeps, or the room warms up, every corner moves
    TOGETHER and a naive standard deviation reports that motion as noise. So
    the common-mode part is measured separately and subtracted: if the whole
    board translated by 0.8 px over the sequence, you are told that, rather
    than being handed a 0.9 px "noise floor" that is mostly a moving tripod.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mlti_cal.detectors.charuco import CharucoBoardSpec, Detection, MultiBoardDetector
from mlti_cal.detectors.settings import CharucoSettings

#: Fewest frames a corner must appear in before its variance means anything.
#: A corner seen 3 times has 2 degrees of freedom and its "std" is mostly luck.
MIN_FRAMES_PER_CORNER = 5

#: Fewest frames in the sequence overall.
MIN_FRAMES = 5


@dataclass
class CornerNoise:
    """One tracked corner's spread across the sequence."""

    board_id: str
    point_id: int
    num_frames: int
    std_x: float
    std_y: float
    mean_x: float
    mean_y: float

    @property
    def std(self) -> float:
        """Per-coordinate std, pooling this corner's two axes."""
        return float(np.sqrt(0.5 * (self.std_x**2 + self.std_y**2)))


@dataclass
class NoiseEstimate:
    """What a static sequence says the per-coordinate pixel noise is."""

    #: The headline: pooled per-coordinate sigma AFTER common motion is removed.
    #: This is the number to type into `pixel_noise_std`.
    sigma_px: float = float("nan")
    sigma_x: float = float("nan")
    sigma_y: float = float("nan")
    #: Before removing common motion. Equals `sigma_px` on a truly static rig.
    sigma_raw_px: float = float("nan")
    #: RMS of the per-frame whole-board translation. This is the part that is
    #: NOT detection noise.
    common_motion_px: float = float("nan")
    #: Straight-line component of that motion, first frame to last.
    drift_px: float = float("nan")

    num_frames: int = 0
    num_corners_used: int = 0
    num_corners_dropped: int = 0
    num_samples: int = 0
    duplicate_images: int = 0

    per_corner: list[CornerNoise] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.num_corners_used > 0 and self.sigma_px == self.sigma_px

    def to_dict(self) -> dict:
        return {
            "sigma_px": self.sigma_px,
            "sigma_x": self.sigma_x,
            "sigma_y": self.sigma_y,
            "sigma_raw_px": self.sigma_raw_px,
            "common_motion_px": self.common_motion_px,
            "drift_px": self.drift_px,
            "num_frames": self.num_frames,
            "num_corners_used": self.num_corners_used,
            "num_corners_dropped": self.num_corners_dropped,
            "num_samples": self.num_samples,
            "duplicate_images": self.duplicate_images,
            "warnings": list(self.warnings),
        }

    def summary_lines(self) -> list[str]:
        """Column-aligned summary for a monospace log."""
        if not self.usable:
            return ["no usable corners -- " + (self.warnings[0] if self.warnings else "unknown")]
        lines = [
            f"{'pixel noise':<18}: {self.sigma_px:.4f} px per coordinate",
            f"{'  x / y':<18}: {self.sigma_x:.4f} / {self.sigma_y:.4f} px",
            f"{'before de-drift':<18}: {self.sigma_raw_px:.4f} px",
            f"{'whole-board motion':<18}: {self.common_motion_px:.4f} px RMS"
            f"  (drift {self.drift_px:.4f} px end to end)",
            f"{'measured on':<18}: {self.num_corners_used} corners x {self.num_frames} frames "
            f"= {self.num_samples} samples"
            + (f", {self.num_corners_dropped} corners dropped" if self.num_corners_dropped else ""),
        ]
        lines += [f"-> {w}" for w in self.warnings]
        return lines


def _pooled_std(values: list[np.ndarray]) -> float:
    """
    Pooled standard deviation over series of differing length.

    sqrt( sum (n_i - 1) var_i / sum (n_i - 1) ). Weighting by degrees of
    freedom, not by count: a corner seen 30 times should count for far more
    than one seen 6 times, and using n instead of n-1 quietly biases short
    series upward.
    """
    num = 0.0
    den = 0.0
    for v in values:
        n = v.size
        if n < 2:
            continue
        num += (n - 1) * float(np.var(v, ddof=1))
        den += n - 1
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def estimate_pixel_noise(
    frames: Sequence[Sequence[Detection]],
    min_frames_per_corner: int = MIN_FRAMES_PER_CORNER,
) -> NoiseEstimate:
    """
    Per-coordinate pixel noise from detections over a static sequence.

    Args:
        frames: one list of `Detection` per image, in capture order.

    Takes detections rather than images so the estimator can be tested against
    a known injected sigma with no image I/O -- which is the only way to show
    it recovers the right answer.
    """
    out = NoiseEstimate(num_frames=len(frames))
    if len(frames) < MIN_FRAMES:
        out.warnings.append(
            f"only {len(frames)} frames; at least {MIN_FRAMES} are needed before a "
            f"standard deviation means anything. Capture more of the same scene."
        )
        return out

    # (board, corner id) -> {frame index: (x, y)}
    series: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    for t, dets in enumerate(frames):
        for det in dets:
            for pid, xy in zip(det.point_ids, det.image_points, strict=True):
                series.setdefault((det.board_id, int(pid)), {})[t] = np.asarray(xy, dtype=float)

    kept = {k: v for k, v in series.items() if len(v) >= min_frames_per_corner}
    out.num_corners_dropped = len(series) - len(kept)
    out.num_corners_used = len(kept)
    if not kept:
        out.warnings.append(
            f"no corner was detected in at least {min_frames_per_corner} frames, so "
            f"nothing can be measured. Either the board is not reliably visible or "
            f"the frames are not of the same scene."
        )
        return out

    # -- common motion: does the whole board move together? ---------------
    # Each corner's offset from its own mean position; averaged over corners
    # this is the rigid translation of the pattern in that frame. Real
    # detection noise is independent per corner and averages toward zero, so
    # what survives the average is motion, not noise.
    means = {k: np.mean(np.stack(list(v.values())), axis=0) for k, v in kept.items()}
    offsets_by_frame: dict[int, list[np.ndarray]] = {}
    for key, per_frame in kept.items():
        for t, xy in per_frame.items():
            offsets_by_frame.setdefault(t, []).append(xy - means[key])

    frame_ids = sorted(offsets_by_frame)
    common = {t: np.mean(np.stack(offsets_by_frame[t]), axis=0) for t in frame_ids}
    common_stack = np.stack([common[t] for t in frame_ids])
    out.common_motion_px = float(np.sqrt(np.mean(np.sum(common_stack**2, axis=1))))

    # Straight-line part of that motion: fit each axis against frame index and
    # report the fitted change from first frame to last. Distinguishes a slow
    # thermal creep (a trend) from a bump halfway through (no trend, big RMS).
    t_arr = np.asarray(frame_ids, dtype=float)
    if t_arr.size >= 2 and np.ptp(t_arr) > 0:
        drift = []
        for axis in (0, 1):
            slope, _ = np.polyfit(t_arr, common_stack[:, axis], 1)
            drift.append(slope * (t_arr[-1] - t_arr[0]))
        out.drift_px = float(np.hypot(*drift))

    # -- the noise itself, with and without the common part ---------------
    raw_x, raw_y, jit_x, jit_y = [], [], [], []
    for key, per_frame in kept.items():
        ts = sorted(per_frame)
        pts = np.stack([per_frame[t] for t in ts])
        centred = pts - means[key]
        raw_x.append(centred[:, 0])
        raw_y.append(centred[:, 1])
        moved = np.stack([common[t] for t in ts])
        jitter = centred - moved
        jit_x.append(jitter[:, 0])
        jit_y.append(jitter[:, 1])
        out.per_corner.append(
            CornerNoise(
                board_id=key[0],
                point_id=key[1],
                num_frames=len(ts),
                std_x=float(np.std(jitter[:, 0], ddof=1)) if len(ts) > 1 else float("nan"),
                std_y=float(np.std(jitter[:, 1], ddof=1)) if len(ts) > 1 else float("nan"),
                mean_x=float(means[key][0]),
                mean_y=float(means[key][1]),
            )
        )

    out.num_samples = int(sum(v.size for v in raw_x))
    out.sigma_raw_px = _pooled_std(raw_x + raw_y)
    out.sigma_x = _pooled_std(jit_x)
    out.sigma_y = _pooled_std(jit_y)
    out.sigma_px = _pooled_std(jit_x + jit_y)

    out.warnings.extend(_interpret(out))
    return out


def _interpret(est: NoiseEstimate) -> list[str]:
    """Plain-language reading of the numbers, in this project's idiom."""
    w: list[str] = []
    if est.duplicate_images:
        w.append(
            f"{est.duplicate_images} image(s) are byte-identical to another in the "
            f"set. Duplicates contribute zero variance and drag the estimate down; "
            f"they are not extra evidence."
        )
    if est.sigma_px < 1e-6:
        w.append(
            "measured noise is essentially zero, which no real sensor produces. "
            "Either the same file was supplied repeatedly, or the images are "
            "identical copies rather than separate exposures."
        )
    elif est.sigma_px < 0.01:
        w.append(
            f"{est.sigma_px:.4f} px is below what a real camera achieves. Check the "
            f"frames are separate exposures rather than copies."
        )
    if est.common_motion_px > 0.5 * est.sigma_raw_px and est.sigma_raw_px > 0:
        w.append(
            f"the whole board moved {est.common_motion_px:.3f} px RMS, which is a "
            f"large part of the {est.sigma_raw_px:.3f} px raw spread. The scene was "
            f"not static; {est.sigma_px:.3f} px is what remains after removing that "
            f"motion and is the number to trust."
        )
    if est.drift_px > est.sigma_px:
        w.append(
            f"there is a steady {est.drift_px:.3f} px drift from the first frame to "
            f"the last -- something is creeping (tripod settling, thermal expansion, "
            f"focus breathing). Let the rig settle before capturing."
        )
    if est.num_corners_dropped > 2 * max(est.num_corners_used, 1):
        w.append(
            f"{est.num_corners_dropped} corners were seen too rarely to use against "
            f"{est.num_corners_used} kept. The estimate describes the corners that "
            f"detect reliably -- typically the central, well-lit ones -- and not the "
            f"border corners that constrain distortion most."
        )
    # Guarded above a real floor: a ratio computed from two numbers that are
    # both ~1e-9 is arithmetic, not anisotropy, and reporting it next to the
    # "essentially zero" warning above is just noise about noise.
    if est.sigma_x > 0.01 and est.sigma_y > 0.01:
        ratio = max(est.sigma_x, est.sigma_y) / min(est.sigma_x, est.sigma_y)
        if ratio > 1.5:
            w.append(
                f"x and y noise differ by {ratio:.1f}x ({est.sigma_x:.3f} vs "
                f"{est.sigma_y:.3f} px). A single per-coordinate sigma is then an "
                f"approximation; suspect rolling shutter, motion blur along one axis, "
                f"or anisotropic refinement."
            )
    if not w:
        w.append(
            f"the sequence looks genuinely static: whole-board motion "
            f"{est.common_motion_px:.3f} px against {est.sigma_px:.3f} px of noise."
        )
    return w


def estimate_from_images(
    paths: Sequence[str | Path],
    spec: CharucoBoardSpec,
    settings: CharucoSettings | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> NoiseEstimate:
    """
    Run detection over a static sequence and estimate the noise from it.

    Only the ONE board being measured is given to the detector. A multi-board
    config validates marker-ID collisions across every board it holds, and an
    unrelated collision must not block a noise measurement that never touches
    that board.

    `on_progress` receives (index, total, corners found in this image).
    """
    import cv2

    detector = MultiBoardDetector([spec], settings=settings or CharucoSettings())
    frames: list[list[Detection]] = []
    digests: set[str] = set()
    duplicates = 0
    unreadable: list[str] = []

    for i, path in enumerate(paths):
        raw = Path(path).read_bytes()
        digest = hashlib.sha1(raw, usedforsecurity=False).hexdigest()
        if digest in digests:
            duplicates += 1
        digests.add(digest)

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            unreadable.append(str(path))
            if on_progress is not None:
                on_progress(i, len(paths), 0)
            continue
        dets = detector.detect_all(image)
        frames.append(dets)
        if on_progress is not None:
            on_progress(i, len(paths), sum(d.num_points for d in dets))

    est = estimate_pixel_noise(frames)
    est.duplicate_images = duplicates
    if duplicates:
        # Recompute the interpretation now that the duplicate count is known --
        # it is discovered here, not by the estimator, which never sees files.
        est.warnings = _interpret(est)
    if unreadable:
        est.warnings.append(f"{len(unreadable)} file(s) could not be read and were skipped.")
    return est
