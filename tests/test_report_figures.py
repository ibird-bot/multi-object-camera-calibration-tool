"""
Report figures: the parts a passing import does not check.

These are regression tests for a specific failure. Export used to call
`savefig` on the live Qt canvas, so every PNG came out at whatever size the
window happened to be -- 5115x1084 on a maximised window, the same absurd shape
for a square correlation matrix and for a line plot, with aspect-locked content
stranded in a corner of a field of white. Nothing raised, nothing failed to
import, and the test suite was entirely happy. So the checks here are about
SHAPE and about the numbers behind the pictures, because those are what broke.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# matplotlib is a `gui` extra, and the headless CI job installs without it.
# figures.py imports it lazily so the core stays importable there, but these
# tests actually render, so the whole module has nothing to say without it.
pytest.importorskip("matplotlib")

from mlti_cal.io.synthetic import generate_dataset
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import build_problem
from mlti_cal.report.covariance import correlation_blocks, pretty_label
from mlti_cal.report.figures import FIGURES, render, spec_for
from mlti_cal.report.report import build_report
from mlti_cal.report.residuals import compute_residual_stats, residual_grid
from mlti_cal.solvers import SolveOptions, get_backend


@pytest.fixture(scope="module")
def solved_report():
    system, _gt = generate_dataset(num_cameras=2, num_frames=10, seed=7)
    initialize_system(system)
    problem = build_problem(system)
    result = get_backend("scipy").solve(problem, SolveOptions())
    report = build_report(problem, system, solve_result=result, pixel_noise_std=0.3, grid_step=64)
    return report, system


# ---------------------------------------------------------------------------
# Sizing -- the actual bug
# ---------------------------------------------------------------------------


def test_every_figure_has_its_own_size(solved_report):
    """
    No two-figure-fits-all. A single shared figsize reproduces the original
    bug at a smaller scale: whatever suits the correlation matrix starves the
    sensor-shaped plots, and vice versa.
    """
    report, system = solved_report
    sizes = {spec.key: spec.size_for(system, "cam0") for spec in FIGURES}
    for key, (w, h) in sizes.items():
        assert 5.0 < w < 20.0, f"{key} width {w}"
        assert 3.0 < h < 12.0, f"{key} height {h}"
        # The pathological export was 34:7. Nothing should come close.
        assert w / h < 4.0, f"{key} aspect {w / h:.1f} is a letterbox"
    assert len(set(sizes.values())) > 1, "all figures share one size"


def test_sensor_shaped_figures_follow_the_camera(solved_report):
    """
    A figure that draws the image plane must be shaped like the image plane,
    or `imshow` keeps the data aspect and matplotlib pads the difference with
    blank paper -- which is exactly why the old export looked half empty.
    """
    report, system = solved_report
    w_px, h_px = system.cameras["cam0"].image_size
    for key in ("uncertainty", "quiver"):
        fw, fh = spec_for(key).size_for(system, "cam0")
        # Height is plot height plus a fixed allowance for title and labels.
        plot_h = fh - 1.4
        plot_w = fw - 1.4
        assert plot_h / plot_w == pytest.approx(h_px / w_px, rel=0.05), key


def test_rendering_writes_a_png_of_the_declared_size(solved_report, tmp_path):
    report, system = solved_report
    for spec in FIGURES:
        fig = render(spec, report, system, "cam0", dpi=100)
        assert fig.get_size_inches() == pytest.approx(spec.size_for(system, "cam0"))
        path = tmp_path / f"{spec.key}.png"
        fig.savefig(path, dpi=100)
        assert path.stat().st_size > 5000, f"{spec.key} rendered essentially blank"


def test_every_figure_draws_something(solved_report):
    """A drawer that silently bails leaves a placeholder, not a plot."""
    report, system = solved_report
    for spec in FIGURES:
        fig = render(spec, report, system, "cam0", dpi=72)
        assert fig.axes, f"{spec.key} drew no axes"
        blank = [
            t.get_text()
            for ax in fig.axes
            for t in ax.texts
            if "no " in t.get_text() or "could not" in t.get_text()
        ]
        assert not blank, f"{spec.key} fell back to a placeholder: {blank}"


def test_a_drawer_never_needs_pyplot():
    """
    Drawing through pyplot would tie export to whichever backend imported
    first, and pyplot state is not safe to touch off the main thread.
    """
    import mlti_cal.report.figures as figures

    source = Path(figures.__file__).read_text(encoding="utf-8")
    for forbidden in ("import matplotlib.pyplot", "from matplotlib import pyplot"):
        assert forbidden not in source, forbidden


# ---------------------------------------------------------------------------
# The numbers behind the new figures
# ---------------------------------------------------------------------------


def test_unobserved_cells_are_nan_not_zero(solved_report):
    """
    Zero would render as "perfect fit" in exactly the corners the calibration
    knows nothing about. That is the opposite of the truth, and it is the
    reading a viewer takes from a dark cell on a hot colormap.
    """
    report, system = solved_report
    grid = report.residual_grids["cam0"]
    rms = np.asarray(grid["rms"], dtype=float)
    counts = np.asarray(grid["counts"], dtype=int)
    assert np.all(np.isnan(rms[counts == 0]))
    assert np.all(np.isfinite(rms[counts > 0]))
    assert grid["empty_cells"] == int((counts == 0).sum())


def test_grid_rms_agrees_with_the_raw_residuals(solved_report):
    """The picture and the headline number must come from the same data."""
    report, system = solved_report
    grid = report.residual_grids["cam0"]
    counts = np.asarray(grid["counts"], dtype=int)
    rms = np.asarray(grid["rms"], dtype=float)
    assert counts.sum() == report.residuals.per_camera["cam0"]["n"]
    pooled = np.sqrt(np.nansum(rms[counts > 0] ** 2 * counts[counts > 0]) / counts.sum())
    assert pooled == pytest.approx(report.residuals.per_camera["cam0"]["rms_px"], rel=1e-9)


def test_mean_vector_is_smaller_than_rms(solved_report):
    """
    The point of the second panel: averaging inside a cell cancels the random
    part, so the mean must sit below the RMS wherever the error is mostly
    noise. If they were equal the averaging would be doing nothing.
    """
    report, system = solved_report
    grid = report.residual_grids["cam0"]
    counts = np.asarray(grid["counts"], dtype=int)
    rms = np.asarray(grid["rms"], dtype=float)
    mag = np.hypot(np.asarray(grid["mean_u"], dtype=float), np.asarray(grid["mean_v"], dtype=float))
    busy = counts >= 5
    assert busy.any()
    assert np.all(mag[busy] <= rms[busy] + 1e-9)


def test_residual_grid_handles_a_camera_with_no_corners(solved_report):
    report, system = solved_report
    stats = compute_residual_stats(build_problem(system))
    grid = residual_grid(stats, "nosuchcam", (1920, 1080))
    assert grid["empty_cells"] == grid["grid"][0] * grid["grid"][1]
    assert np.all(np.isnan(np.asarray(grid["rms"], dtype=float)))


# ---------------------------------------------------------------------------
# Correlation blocks
# ---------------------------------------------------------------------------


def test_correlation_blocks_cover_what_the_warning_talks_about(solved_report):
    """
    A camera-only matrix is the tempting shape and the wrong one: the
    strongest correlations on a real capture are camera parameters against
    board poses, and the warning text names exactly those. A map that replaces
    the log has to contain what the log says.
    """
    report, _system = solved_report
    blocks = report.correlations
    assert blocks is not None
    camera = np.asarray(blocks["camera_matrix"], dtype=float)
    assert camera.shape[0] == camera.shape[1] == len(blocks["camera_labels"])
    assert np.allclose(np.diag(camera), 1.0)
    assert np.allclose(camera, camera.T)

    cross = np.asarray(blocks["cross_matrix"], dtype=float)
    assert cross.shape == (len(blocks["cross_row_labels"]), blocks["shown_pose_columns"])
    assert np.abs(cross).max() <= 1.0 + 1e-9

    # Column bands must tile the strip exactly, or the board labels sit under
    # the wrong columns.
    groups = blocks["cross_column_groups"]
    assert groups[0]["start"] == 0
    assert groups[-1]["end"] == cross.shape[1]
    for a, b in zip(groups[:-1], groups[1:], strict=True):
        assert a["end"] == b["start"]


def test_pose_labels_name_their_frame(solved_report):
    """
    Dropping the frame collapses distinct pose columns onto one name, and the
    worst-pairs chart then reads `tx [board] vs tx [board]` -- which names
    neither parameter and cannot be acted on.
    """
    report, _system = solved_report
    pairs = report.correlations["top_pairs"]
    assert pairs
    rendered = [f"{a} vs {b}" for a, b, _ in pairs]
    assert len(set(rendered)) == len(rendered), f"ambiguous labels: {rendered}"
    aliases = report.correlations["frame_aliases"]
    assert aliases and len(set(aliases.values())) == len(aliases)


def test_pretty_label_survives_a_frame_name_full_of_dots():
    """
    Capture stems carry timestamps, so the component has to be split off from
    the right. Splitting from the left silently mislabels every pose column.
    """
    label = "pose:image-S0-D1-I0-2026-07-06-17-40-00.6750762:board_A.dtx"
    assert pretty_label(label, aliases={"image-S0-D1-I0-2026-07-06-17-40-00.6750762": "f00"}) == (
        "tx [board_A f00]"
    )
    assert pretty_label("intr:cam0.4", {"cam0": ["fx", "fy", "cx", "cy", "k1"]}) == "k1 (cam0)"


def test_pose_columns_are_capped_and_kept_strongest(solved_report):
    """A 500-frame job must still produce a legible strip."""
    report, system = solved_report
    problem = build_problem(system)
    from mlti_cal.report.covariance import compute_covariance

    cov = compute_covariance(problem)
    capped = correlation_blocks(cov, max_pose_columns=8)
    assert capped["shown_pose_columns"] == 8
    assert capped["num_pose_columns"] > 8
    full = correlation_blocks(cov)
    kept = np.abs(np.asarray(capped["cross_matrix"], dtype=float)).max(axis=0)
    everything = np.abs(np.asarray(full["cross_matrix"], dtype=float)).max(axis=0)
    assert kept.min() >= np.sort(everything)[-8] - 1e-9
