"""
Every report figure, drawn into a caller-supplied `Figure`.

Two callers share this module: the GUI, which draws into a live Qt canvas, and
the exporter, which draws into a fresh off-screen figure. That sharing is the
point. Before it existed, export called `savefig` on the on-screen canvas, so a
PNG inherited whatever size Qt had stretched the widget to -- a maximised window
produced 5115x1084 files, 34 inches by 7, in which a square heatmap occupied a
tenth of the width and a line plot smeared across the rest. The figures were not
inconsistent by accident; they had no size of their own at all.

So size belongs to the figure, not to the window. Each `FigureSpec` carries a
`figsize` keyed to its content: image-domain plots get the aspect of the sensor,
the correlation matrix gets a square, line plots get something close to golden.

No pyplot here. `Figure` is constructed directly and the backend is chosen by
the caller, so drawing does not depend on which backend imported first.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Shared look. Applied per-axes rather than through global rcParams, which are
# process-wide and would leak into any other matplotlib user in the same
# interpreter -- including the detection views.
TITLE_SIZE = 10.5
LABEL_SIZE = 9
TICK_SIZE = 8
NOTE_SIZE = 8
NOTE_COLOUR = "#555555"
ACCENT = "#3b7dd8"
GRID = {"alpha": 0.25, "linewidth": 0.6}


def style_axes(ax, *, grid: bool = False, spines: bool = True) -> None:
    ax.tick_params(labelsize=TICK_SIZE)
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)
    if grid:
        ax.grid(**GRID)
        ax.set_axisbelow(True)
    if not spines:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def set_title(ax, title: str, note: str | None = None) -> None:
    """Title in one weight, the explanation under it in another."""
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=14 if note else 6)
    if note:
        ax.text(
            0.5,
            1.008,
            note,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=NOTE_SIZE,
            color=NOTE_COLOUR,
        )


def add_colorbar(fig, mappable, ax, label: str, attached: bool = False):
    """
    Colorbar for `ax`.

    `attached=True` hangs it off the axes itself rather than off the gridspec
    cell. That distinction matters for any aspect-locked plot: an image shrinks
    inside its allocated cell to keep its shape, but a cell-anchored colorbar
    keeps the cell height and ends up conspicuously taller than the image it
    describes. An inset tracks the drawn axes, so the two always line up.
    """
    if attached:
        cax = ax.inset_axes([1.015, 0.0, 0.022, 1.0])
        cb = fig.colorbar(mappable, cax=cax)
    else:
        cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)
    return cb


def message(fig, text: str) -> None:
    """Placeholder for a figure whose data is missing."""
    fig.clear()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=10, color=NOTE_COLOUR)
    ax.set_axis_off()


def sensor_figsize(
    image_size, width: float = 9.0, colorbar: float = 1.4, panels: int = 1
) -> tuple[float, float]:
    """
    A figure shaped like the sensor, with room for a colorbar and axis labels.

    Getting this wrong is exactly the original bug: `imshow` locks the data
    aspect, so a figure wider than its content pads the difference with blank
    paper instead of drawing the content bigger.
    """
    w, h = float(image_size[0]), float(image_size[1])
    plot_width = (width - colorbar * panels) / panels
    return (width, plot_width * (h / max(w, 1.0)) + 1.4)


# ----------------------------------------------------------------------
# drawers -- each takes (fig, report, system, camera_id) and draws in place
# ----------------------------------------------------------------------
def draw_uncertainty(fig, report, system, cam: str) -> None:
    m = report.uncertainty_maps.get(cam)
    if m is None:
        message(fig, f"no uncertainty map for {cam}")
        return
    ax = fig.add_subplot(111)
    w, h = system.cameras[cam].image_size
    im = ax.imshow(
        m.sigma_max,
        origin="upper",
        extent=[0, w, h, 0],
        cmap="viridis",
        interpolation="bilinear",
    )
    add_colorbar(fig, im, ax, "1-sigma projection error (px)", attached=True)
    note = "how far a projected point could be wrong, given the parameter covariance"
    if m.invalid_fraction > 0.001:
        note = f"{m.invalid_fraction * 100:.0f}% masked: distortion not invertible there"
    set_title(
        ax,
        f"{cam} @ {m.range_m:g} m -- centre {m.at_centre():.3f} px, worst {m.worst():.3f} px",
        note,
    )
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    style_axes(ax)


def draw_quiver(fig, report, system, cam: str) -> None:
    from mlti_cal.report.residuals import quiver_field

    if report.residuals is None or report.residuals.errors.size == 0:
        message(fig, "no residuals")
        return
    q = quiver_field(report.residuals, cam)
    if not q["x"]:
        message(fig, f"no residuals for {cam}")
        return
    ax = fig.add_subplot(111)
    w, h = system.cameras[cam].image_size
    sc = ax.quiver(
        q["x"],
        q["y"],
        q["u"],
        q["v"],
        np.array(q["magnitude"]),
        cmap="plasma",
        angles="xy",
        width=0.0022,
    )
    add_colorbar(fig, sc, ax, "error (px)", attached=True)
    # Equal aspect is not cosmetic here: both axes are pixels, so anisotropic
    # scaling would rotate every arrow and invent structure that is not there.
    ax.set_aspect("equal")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    set_title(
        ax,
        f"{cam} residual field -- one arrow per corner, autoscaled",
        "structure here is model inadequacy that RMS cannot see",
    )
    style_axes(ax)


def draw_coverage(fig, report, system, cam: str) -> None:
    if report.coverage is None or cam not in report.coverage.per_camera:
        message(fig, "no coverage data")
        return
    cs = report.coverage.per_camera[cam]
    axes = fig.subplots(1, 2)
    im = axes[0].imshow(cs.histogram, cmap="magma", interpolation="nearest")
    add_colorbar(fig, im, axes[0], "corners per cell", attached=True)
    set_title(
        axes[0],
        f"{cam}: {cs.occupied_fraction * 100:.0f}% of cells occupied",
        "empty cells mean distortion is extrapolated there",
    )
    axes[0].set_xlabel("grid column")
    axes[0].set_ylabel("grid row")
    style_axes(axes[0])

    tilt = report.coverage.tilt
    if tilt is not None and tilt.incidence_deg.size:
        axes[1].hist(tilt.incidence_deg, bins=18, color=ACCENT, edgecolor="white", linewidth=0.5)
        axes[1].axvline(10, color="crimson", ls="--", lw=1.2)
        axes[1].set_xlabel("board incidence angle (deg)")
        axes[1].set_ylabel("observations")
        set_title(
            axes[1],
            f"tilt diversity -- {tilt.fraction_below_10deg * 100:.0f}% under 10 deg",
            "frontoparallel boards cannot separate focal length from distance",
        )
        style_axes(axes[1], grid=True, spines=False)
    else:
        axes[1].set_axis_off()


def draw_distribution(fig, report, system, cam: str) -> None:
    from mlti_cal.report.residuals import qq_data

    if report.residuals is None or report.residuals.errors.size == 0:
        message(fig, "no residuals")
        return
    axes = fig.subplots(1, 2)
    n = report.normality
    axes[0].hist(
        report.residuals.vectors.ravel(),
        bins=60,
        color=ACCENT,
        edgecolor="white",
        linewidth=0.4,
    )
    set_title(
        axes[0],
        "residual components",
        f"skew {n.get('skew', float('nan')):.2f}, "
        f"excess kurtosis {n.get('excess_kurtosis', float('nan')):.2f}",
    )
    axes[0].set_xlabel("residual (px)")
    axes[0].set_ylabel("count")
    style_axes(axes[0], grid=True, spines=False)

    q = qq_data(report.residuals)
    if q["theoretical"]:
        axes[1].plot(q["theoretical"], q["observed"], ".", ms=2.5, color=ACCENT)
        lim = max(abs(min(q["theoretical"])), abs(max(q["theoretical"])))
        axes[1].plot([-lim, lim], [-lim, lim], "--", lw=1.2, color="crimson")
    set_title(axes[1], "Q-Q against a normal", "curvature at the ends means heavy tails")
    axes[1].set_xlabel("theoretical quantile (px)")
    axes[1].set_ylabel("observed quantile (px)")
    style_axes(axes[1], grid=True, spines=False)


def draw_radius(fig, report, system, cam: str) -> None:
    evr = report.error_vs_radius
    if not evr.get("bin_centres"):
        message(fig, "no data")
        return
    ax = fig.add_subplot(111)
    ax.plot(evr["bin_centres"], evr["rms_px"], "o-", color=ACCENT, lw=1.8, ms=5)
    ax.set_xlabel("radius from principal point (px)")
    ax.set_ylabel("RMS error (px)")
    set_title(
        ax,
        "error vs radius",
        "growth toward the edge means the distortion model cannot represent the lens",
    )
    style_axes(ax, grid=True, spines=False)


@dataclass
class HoverGrid:
    """
    A heatmap whose cells can be read back by pointing at them.

    The GUI needs three things to turn a cursor position into a sentence --
    which axes, what the cell holds, and what the row and column are called.
    Bundling them here keeps the drawer as the single place that knows the
    layout, so the view never re-derives an index and never disagrees with the
    picture. Ignored entirely by the exporter, which has no cursor.
    """

    ax: object
    matrix: object
    row_labels: list
    col_labels: list
    value_fmt: str = "{:+.3f}"
    value_name: str = "correlation"

    def describe(self, row: int, col: int) -> str | None:
        if not (0 <= row < len(self.row_labels) and 0 <= col < len(self.col_labels)):
            return None
        value = self.matrix[row][col]
        return (
            f"{self.row_labels[row]}\n{self.col_labels[col]}\n"
            f"{self.value_name} {self.value_fmt.format(value)}"
        )


def _diverging(matrix, ax, labels_x=None, labels_y=None):
    """Correlation heatmap on a fixed -1..1 scale, so colour means magnitude."""
    im = ax.imshow(
        np.asarray(matrix, dtype=float),
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    if labels_y is not None:
        ax.set_yticks(range(len(labels_y)))
        ax.set_yticklabels(labels_y, fontsize=TICK_SIZE)
    if labels_x is not None:
        ax.set_xticks(range(len(labels_x)))
        ax.set_xticklabels(labels_x, fontsize=TICK_SIZE, rotation=45, ha="right")
    return im


def draw_correlation(fig, report, system, cam: str) -> None:
    """
    Which parameters the data cannot tell apart.

    Replaces a line of log text with the whole picture. Two heatmaps, because
    the finding does not live in one of them: the square block is the familiar
    fx/cx and k1/k2 trade-off, but on a typical capture the strongest
    correlations are camera parameters against BOARD POSES, which a
    camera-only matrix cannot show at all.
    """
    blocks = report.correlations
    if not blocks or not blocks.get("camera_matrix"):
        message(fig, "no covariance -- too few degrees of freedom for a correlation matrix")
        return

    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3], height_ratios=[1.0, 0.6])
    hovers: list[HoverGrid] = []

    # -- camera block ---------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    labels = blocks["camera_labels"]
    matrix = blocks["camera_matrix"]
    im = _diverging(matrix, ax, labels, labels)
    ax.set_aspect("equal")
    add_colorbar(fig, im, ax, "correlation", attached=True)
    set_title(ax, "camera parameters", "red = move together, blue = trade off")
    # Annotating is only legible on a small block, and only off the diagonal
    # where every value is 1.00 by construction and says nothing.
    if len(labels) <= 14:
        arr = np.asarray(matrix, dtype=float)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j:
                    continue
                v = arr[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="white" if abs(v) > 0.55 else "#333333",
                )
    style_axes(ax)
    hovers.append(HoverGrid(ax, matrix, labels, labels))

    # -- worst pairs ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    pairs = blocks.get("top_pairs", [])[:12][::-1]
    if pairs:
        values = [r for _, _, r in pairs]
        names = [f"{a}  vs  {b}" for a, b, _ in pairs]
        colours = ["#c0392b" if v > 0 else "#2c6fbb" for v in values]
        ax2.barh(range(len(values)), values, color=colours, height=0.72)
        ax2.set_yticks(range(len(values)))
        ax2.set_yticklabels(names, fontsize=TICK_SIZE)
        ax2.set_xlim(-1.08, 1.08)
        ax2.axvline(0, color="#888888", lw=0.8)
        for x in (-0.99, 0.99):
            ax2.axvline(x, color="crimson", ls=":", lw=1.0)
        ax2.set_xlabel("correlation coefficient")
    set_title(
        ax2,
        "least separable pairs",
        "past the dotted lines the two cannot be estimated independently",
    )
    style_axes(ax2, grid=True, spines=False)

    # -- camera against board poses -------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    cross = blocks.get("cross_matrix")
    if cross:
        im3 = _diverging(cross, ax3, None, blocks["cross_row_labels"])
        add_colorbar(fig, im3, ax3, "correlation", attached=True)
        groups = blocks.get("cross_column_groups", [])
        ticks, names = [], []
        for g in groups:
            ticks.append(0.5 * (g["start"] + g["end"] - 1))
            names.append(g["name"])
            if g["start"] > 0:
                ax3.axvline(g["start"] - 0.5, color="#222222", lw=1.2)
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(names, fontsize=TICK_SIZE)
        shown, total = blocks["shown_pose_columns"], blocks["num_pose_columns"]
        extra = "" if shown == total else f" (strongest {shown} of {total})"
        set_title(
            ax3,
            f"camera parameters against board pose parameters{extra}",
            "this is the block the extreme-correlation warning is about",
        )
        cols = [f"pose column {i}" for i in range(len(cross[0]))]
        for g in groups:
            name = g["name"]
            for i in range(g["start"], g["end"]):
                cols[i] = f"{name} pose"
        hovers.append(HoverGrid(ax3, cross, blocks["cross_row_labels"], cols))
        style_axes(ax3)
    else:
        ax3.set_axis_off()

    fig.hover_grids = hovers


def draw_residual_map(fig, report, system, cam: str) -> None:
    """
    Where the error is, and which part of it is systematic.

    The all-arrows quiver draws every corner, so the random component
    dominates and hides the trend. Averaging inside a cell cancels the random
    part; whatever survives is bias the model failed to represent. Cells with
    no corners stay masked rather than reading as a perfect fit.
    """
    from matplotlib import colormaps

    grid = (report.residual_grids or {}).get(cam)
    if not grid:
        message(fig, f"no residual grid for {cam}")
        return

    w, h = grid["image_size"]
    extent = [0, w, h, 0]
    rms = np.ma.masked_invalid(np.asarray(grid["rms"], dtype=float))
    mu = np.ma.masked_invalid(np.asarray(grid["mean_u"], dtype=float))
    mv = np.ma.masked_invalid(np.asarray(grid["mean_v"], dtype=float))
    rows, cols = grid["grid"]

    axes = fig.subplots(1, 2)

    hot = colormaps["inferno"].with_extremes(bad="#d9d9d9")
    im = axes[0].imshow(rms, origin="upper", extent=extent, cmap=hot, interpolation="nearest")
    add_colorbar(fig, im, axes[0], "RMS error in cell (px)", attached=True)
    empty = grid["empty_cells"]
    set_title(
        axes[0],
        f"{cam}: error by sensor region",
        f"grey = never observed ({empty} of {rows * cols} cells), unsupported by data",
    )

    magnitude = np.ma.sqrt(mu**2 + mv**2)
    cool = colormaps["viridis"].with_extremes(bad="#d9d9d9")
    im2 = axes[1].imshow(
        magnitude, origin="upper", extent=extent, cmap=cool, interpolation="nearest"
    )
    add_colorbar(fig, im2, axes[1], "mean residual in cell (px)", attached=True)
    # Cell centres in pixel coordinates, so each arrow sits on its own cell.
    xc = (np.arange(cols) + 0.5) * (w / cols)
    yc = (np.arange(rows) + 0.5) * (h / rows)
    gx, gy = np.meshgrid(xc, yc)
    # Autoscale would size the arrows against the whole axes, so a 0.5 px bias
    # is drawn hundreds of pixels long and spills across neighbouring cells --
    # and off the plot entirely near the edges. Pin the longest arrow to one
    # cell instead, and say by how much it was exaggerated, so the arrows stay
    # comparable between cells and between cameras.
    cell = w / cols
    peak = float(magnitude.max()) if magnitude.count() else 0.0
    exaggeration = (0.9 * cell / peak) if peak > 0 else 1.0
    axes[1].quiver(
        gx,
        gy,
        mu,
        mv,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1.0 / exaggeration,
        width=0.005,
        edgecolor="black",
        linewidth=0.4,
    )
    set_title(
        axes[1],
        f"{cam}: systematic component (arrows {exaggeration:.0f}x)",
        "averaging cancels random error; what is left is model bias",
    )

    for ax in axes:
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_aspect("equal")
        style_axes(ax)


@dataclass(frozen=True)
class FigureSpec:
    key: str
    title: str
    draw: Callable
    #: None means "shaped like the sensor" -- resolved per camera at draw time,
    #: because the right height depends on the aspect of that camera.
    figsize: tuple[float, float] | None = None
    #: Total width, and how many sensor-shaped panels share it.
    sensor_width: float = 9.0
    panels: int = 1

    def size_for(self, system, cam: str) -> tuple[float, float]:
        if self.figsize is not None:
            return self.figsize
        camera = system.cameras.get(cam) if system is not None else None
        return sensor_figsize(
            camera.image_size if camera else (16, 9),
            width=self.sensor_width,
            panels=self.panels,
        )


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec("uncertainty", "Projection uncertainty", draw_uncertainty),
    FigureSpec("quiver", "Residual field", draw_quiver),
    FigureSpec("coverage", "Coverage", draw_coverage, figsize=(11.5, 4.8)),
    FigureSpec("distribution", "Residual distribution", draw_distribution, figsize=(11.5, 4.8)),
    FigureSpec("radius", "Error vs radius", draw_radius, figsize=(8.5, 5.2)),
    FigureSpec("residual_map", "Residual map", draw_residual_map, sensor_width=15.0, panels=2),
    FigureSpec("correlation", "Correlations", draw_correlation, figsize=(13.5, 9.0)),
)


def spec_for(key: str) -> FigureSpec:
    for spec in FIGURES:
        if spec.key == key:
            return spec
    raise KeyError(key)


def render(spec: FigureSpec, report, system, cam: str, dpi: int = 150):
    """Draw one figure into a fresh off-screen `Figure`, sized by its spec."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=spec.size_for(system, cam), dpi=dpi, layout="constrained")
    FigureCanvasAgg(fig)
    spec.draw(fig, report, system, cam)
    return fig
