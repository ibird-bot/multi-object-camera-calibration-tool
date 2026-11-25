"""
A4 printables for Charuco boards.

Charuco only: its dictionary and marker-id offset make each board a specific
target, while any checkerboard of the same size is interchangeable.

A page is only correct printed at 100%: `square_length` is the metric scale of
the whole calibration, and a printer's "fit to page" silently changes it. Each
page carries a ruled 50 mm bar to check against, and a board too large for A4 is
scaled down and labelled as unusable for calibration.
"""

from __future__ import annotations

from pathlib import Path

from mlti_cal.report.figures import NOTE_COLOUR

A4_MM = (210.0, 297.0)  # portrait width, height
MARGIN_MM = 12.0
TOP_LABEL_MM = 10.0
BOTTOM_LABEL_MM = 16.0
#: ~305 dpi once placed on the page -- independent of `scale`, since the pixel
#: count and the display size it is stretched to shrink together.
PIXELS_PER_MM = 12.0
#: Text colour of the "scaled, do not use" warning.
WARNING_COLOUR = "#b00020"


def _printable_area(page_w_mm: float, page_h_mm: float) -> tuple[float, float]:
    """Width and height left for the board once margins and labels are taken."""
    return page_w_mm - 2 * MARGIN_MM, page_h_mm - TOP_LABEL_MM - BOTTOM_LABEL_MM


def _page_and_scale(board_w_mm: float, board_h_mm: float) -> tuple[tuple[float, float], float]:
    """
    The page orientation and scale factor that get this board onto one A4 sheet.

    Scale is 1.0 whenever the board fits; portrait is tried first because it is
    the default tray orientation. A board too big for either is scaled down onto
    landscape, the wider of the two.
    """
    portrait = A4_MM
    landscape = (A4_MM[1], A4_MM[0])
    for page in (portrait, landscape):
        avail_w, avail_h = _printable_area(*page)
        if board_w_mm <= avail_w and board_h_mm <= avail_h:
            return page, 1.0
    avail_w, avail_h = _printable_area(*landscape)
    return landscape, min(avail_w / board_w_mm, avail_h / board_h_mm)


def render_charuco_printable(spec):
    """
    One board, laid out on one A4 page at true physical scale where it fits.

    Lazy matplotlib import for the same reason as `report/figures.py`: this
    module lives in the core, and the core must stay importable with no
    plotting stack present.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from mlti_cal.detectors.charuco import CharucoDetector

    detector = CharucoDetector(spec)  # validates dictionary size, marker fit, etc.

    board_w_mm = spec.squares_x * spec.square_length * 1000.0
    board_h_mm = spec.squares_y * spec.square_length * 1000.0
    (page_w_mm, page_h_mm), scale = _page_and_scale(board_w_mm, board_h_mm)

    fig = Figure(figsize=(page_w_mm / 25.4, page_h_mm / 25.4), dpi=150)
    FigureCanvasAgg(fig)

    disp_w_mm = board_w_mm * scale
    disp_h_mm = board_h_mm * scale
    _, avail_h_mm = _printable_area(page_w_mm, page_h_mm)
    y0_mm = BOTTOM_LABEL_MM + (avail_h_mm - disp_h_mm) / 2
    x0_mm = (page_w_mm - disp_w_mm) / 2

    ax = fig.add_axes(
        (x0_mm / page_w_mm, y0_mm / page_h_mm, disp_w_mm / page_w_mm, disp_h_mm / page_h_mm)
    )
    img_w_px = max(1, round(disp_w_mm * PIXELS_PER_MM))
    img_h_px = max(1, round(disp_h_mm * PIXELS_PER_MM))
    # marginSize=0: the page margin is the quiet zone; a margin inside the
    # image would shrink the squares below the physical size laid out above.
    image = detector.board.generateImage((img_w_px, img_h_px), marginSize=0)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    title = (
        f"{spec.id}  --  Charuco {spec.squares_x}x{spec.squares_y} squares, "
        f"{spec.square_length * 1000:.2f} mm/square, "
        f"marker {spec.marker_length * 1000:.2f} mm, {spec.dictionary}"
    )
    fig.text(
        0.5,
        1 - (TOP_LABEL_MM * 0.5) / page_h_mm,
        title,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

    if scale < 1.0:
        fig.text(
            0.5,
            (BOTTOM_LABEL_MM * 0.55) / page_h_mm,
            f"SCALED TO {scale * 100:.1f}% TO FIT A4 -- true size is "
            f"{board_w_mm:.0f} x {board_h_mm:.0f} mm. Print full scale on "
            f"larger media or through a print shop; do not use this page as-is "
            f"for calibration.",
            ha="center",
            va="center",
            fontsize=7,
            color=WARNING_COLOUR,
            fontweight="bold",
        )
    else:
        _draw_scale_bar(fig, page_w_mm, page_h_mm)

    return fig


def _draw_scale_bar(fig, page_w_mm: float, page_h_mm: float) -> None:
    """A ruled 50 mm bar, so a printer's 'fit to page' is caught with a ruler."""
    bar_mm = min(50.0, page_w_mm - 2 * MARGIN_MM)
    rx0 = MARGIN_MM / page_w_mm
    ry0 = (BOTTOM_LABEL_MM * 0.15) / page_h_mm
    rw = bar_mm / page_w_mm
    rh = 3.0 / page_h_mm
    rax = fig.add_axes((rx0, ry0, rw, rh))
    rax.set_xlim(0, bar_mm)
    rax.set_ylim(0, 1)
    rax.plot([0, bar_mm], [0.5, 0.5], color="black", linewidth=1.0)
    for tick_mm in range(0, int(bar_mm) + 1, 10):
        rax.plot([tick_mm, tick_mm], [0.15, 0.85], color="black", linewidth=0.8)
    rax.axis("off")
    fig.text(
        rx0 + rw + 0.01,
        ry0 + rh / 2,
        f"{bar_mm:.0f} mm -- check with a ruler after printing; "
        f"printed short means the printer rescaled the page (use 100% / actual size)",
        fontsize=6,
        va="center",
        color=NOTE_COLOUR,
    )


def generate_charuco_printables(specs, out_dir: str | Path) -> list[Path]:
    """One PDF per Charuco board spec, written to `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in specs:
        fig = render_charuco_printable(spec)
        path = out / f"printable_{spec.id}.pdf"
        fig.savefig(path)
        written.append(path)
    return written
