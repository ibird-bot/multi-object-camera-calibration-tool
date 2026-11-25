"""
One look for the whole application, in light and dark.

Design rules, kept deliberately small so the tool stays calm to work in:
- one accent colour (blue), used for the primary action and for selection;
- three status colours (good / check / problem), used ONLY for status;
- flat surfaces, one border level, 4 px corners; a group is a heading, not a box;
- colour means status, never decoration: numbers are plain text unless they
  need attention;
- one type scale: body 10 pt, captions 9 pt, metric values 18 pt.

The palette follows the Windows light/dark setting unless the user picked one
in View > Theme (remembered between runs).

Matplotlib follows the theme on screen only, through `mpl_context`, never by
changing `rcParams` globally: `report/figures.py` also renders the exported
PNGs, and an export from a dark session must still produce figures fit for a
white page.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str  # window background
    surface: str  # cards, tables, plots
    surface_alt: str  # hover, headers, inputs
    border: str
    text: str
    muted: str
    accent: str
    accent_text: str  # text on the accent colour
    accent_soft: str  # selection background
    good: str
    warn: str
    bad: str


LIGHT = Palette(
    name="light",
    bg="#f4f5f7",
    surface="#ffffff",
    surface_alt="#eef0f3",
    border="#d9dde3",
    text="#1d2127",
    muted="#5d6571",
    accent="#2f6fde",
    accent_text="#ffffff",
    accent_soft="#dfe9fb",
    good="#2e7d4f",
    warn="#b26b00",
    bad="#c0392b",
)

DARK = Palette(
    name="dark",
    bg="#15171c",
    surface="#1d2027",
    surface_alt="#262a33",
    border="#343a46",
    text="#e4e7eb",
    muted="#98a1ad",
    accent="#5b8def",
    accent_text="#0d1117",
    accent_soft="#223352",
    good="#4caf7a",
    warn="#e0a43c",
    bad="#e5675a",
)

MODES = ("system", "light", "dark")
FONT = "Segoe UI"


def system_is_dark() -> bool:
    app = QGuiApplication.instance()
    if app is None:
        return False
    try:
        return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:  # Qt older than 6.5
        return QGuiApplication.palette().window().color().lightness() < 128


def arrow_images(p: Palette) -> dict[str, str]:
    """
    Up/down arrow PNGs in the palette's muted colour, for the stylesheet.

    Qt stops drawing the native arrows of a styled spin box or combo box, and
    a stylesheet can only replace them with an image file -- so the two
    triangles are painted once per palette into the temp folder.
    """
    import tempfile
    from pathlib import Path

    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QImage, QPainter, QPolygonF

    folder = Path(tempfile.gettempdir()) / "mlti_cal_theme"
    folder.mkdir(exist_ok=True)
    out = {}
    for direction in ("up", "down"):
        path = folder / f"arrow_{direction}_{p.name}.png"
        if not path.exists():
            img = QImage(18, 12, QImage.Format.Format_ARGB32)
            img.fill(Qt.GlobalColor.transparent)
            painter = QPainter(img)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(p.muted))
            pts = [(1, 11), (17, 11), (9, 1)] if direction == "up" else [(1, 1), (17, 1), (9, 11)]
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
            painter.end()
            img.save(str(path))
        out[direction] = path.as_posix()
    return out


def stylesheet(p: Palette, scale: float = 1.0) -> str:
    """The whole look; `scale` enlarges every type size (presentation mode)."""
    arrows = arrow_images(p)

    def pt(size: float) -> str:
        return f"{size * scale:.1f}pt"

    return f"""
* {{ font-family: "{FONT}"; font-size: {pt(10)}; }}
QWidget {{ color: {p.text}; }}
QMainWindow, QDialog, QStackedWidget > QWidget,
QScrollArea > QWidget > QWidget {{ background: {p.bg}; }}
QToolTip {{ background: {p.surface}; color: {p.text}; border: 1px solid {p.border};
            padding: 6px; border-radius: 4px; }}

/* cards */
/* A group is a heading over its content, not a box: the tables and inputs
   inside already have an edge, and a box around boxes is only noise. */
QGroupBox {{ background: transparent; border: none; margin-top: 22px; padding: 4px 0 0 0; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 0; top: 2px; padding: 0;
                    color: {p.muted}; font-weight: 600; }}
QFrame#Card, QFrame#Banner {{ background: {p.surface}; border: 1px solid {p.border};
    border-radius: 4px; }}
QFrame#Row {{ border: none; border-bottom: 1px solid {p.border}; }}
QLabel[role="muted"] {{ color: {p.muted}; font-size: {pt(9)}; }}
QLabel[role="metric"] {{ font-size: {pt(18)}; font-weight: 600; }}
QLabel[role="caption"] {{ color: {p.muted}; font-size: {pt(9)}; }}
QLabel[role="heading"] {{ font-size: {pt(13)}; font-weight: 600; }}

/* buttons */
QPushButton, QToolButton {{ background: {p.surface}; border: 1px solid {p.border};
    border-radius: 4px; padding: 5px 12px; }}
QPushButton:hover, QToolButton:hover {{ background: {p.surface_alt}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {p.accent_soft}; }}
QPushButton:disabled {{ color: {p.muted}; }}
QPushButton[primary="true"] {{ background: {p.accent}; color: {p.accent_text};
    border: 1px solid {p.accent}; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {p.accent}; border-color: {p.text}; }}
QPushButton[primary="true"]:disabled {{ background: {p.surface_alt}; color: {p.muted};
    border-color: {p.border}; }}
QPushButton[segment="true"] {{ border: none; border-bottom: 2px solid transparent;
    border-radius: 0; background: transparent; color: {p.muted}; padding: 6px 12px; }}
QPushButton[segment="true"]:hover {{ color: {p.text}; }}
QPushButton[segment="true"]:checked {{ color: {p.text}; border-bottom-color: {p.accent}; }}
QToolButton#SectionToggle {{ border: none; background: transparent; font-weight: 600;
    color: {p.muted}; padding: 4px 2px; }}

/* inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {p.surface}; border: 1px solid {p.border}; border-radius: 4px;
    padding: 3px 6px; selection-background-color: {p.accent_soft}; selection-color: {p.text}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus {{ border-color: {p.accent}; }}
QComboBox QAbstractItemView {{ background: {p.surface}; border: 1px solid {p.border};
    selection-background-color: {p.accent_soft}; selection-color: {p.text}; }}
/* Once a spin box or combo box has a styled border, Qt stops drawing its
   arrows natively -- they must be drawn here (as border triangles) or they
   render as stray lines. */
QComboBox {{ padding-right: 22px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right;
    width: 20px; border: none; }}
QComboBox::down-arrow {{ image: url("{arrows["down"]}"); width: 9px; height: 6px; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 20px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border;
    subcontrol-position: top right; width: 18px; border: none;
    border-left: 1px solid {p.border}; border-top-right-radius: 4px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border;
    subcontrol-position: bottom right; width: 18px; border: none;
    border-left: 1px solid {p.border}; border-bottom-right-radius: 4px; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {p.surface_alt}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{arrows["up"]}");
    width: 8px; height: 5px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{arrows["down"]}");
    width: 8px; height: 5px; }}
QListWidget#figure_list::item {{ padding: 6px 8px; border-radius: 4px; }}
QListWidget#figure_list::item:selected {{ background: {p.accent_soft}; color: {p.text}; }}

/* tables and trees */
QTableWidget, QTableView, QTreeWidget, QTreeView, QListWidget {{
    background: {p.surface}; alternate-background-color: {p.surface_alt};
    border: 1px solid {p.border}; border-radius: 4px; gridline-color: {p.border};
    selection-background-color: {p.accent_soft}; selection-color: {p.text}; }}
QHeaderView::section {{ background: {p.surface}; color: {p.muted}; border: none;
    border-bottom: 1px solid {p.border}; padding: 6px 8px; font-size: {pt(9)}; }}
QTableCornerButton::section {{ background: {p.surface}; border: none; }}
QTableWidget::item {{ border-bottom: 1px solid {p.surface_alt}; padding: 0 6px; }}
/* An input inside a table or tree row has the row as its edge: its own box
   appears only on hover or focus. */
QAbstractItemView QLineEdit, QAbstractItemView QComboBox, QAbstractItemView QSpinBox,
QAbstractItemView QDoubleSpinBox {{ border-color: transparent; border-radius: 2px; }}
QAbstractItemView QLineEdit:hover, QAbstractItemView QComboBox:hover,
QAbstractItemView QSpinBox:hover, QAbstractItemView QDoubleSpinBox:hover {{
    border-color: {p.border}; }}
QAbstractItemView QLineEdit:focus, QAbstractItemView QComboBox:focus,
QAbstractItemView QSpinBox:focus, QAbstractItemView QDoubleSpinBox:focus {{
    border-color: {p.accent}; }}
QAbstractItemView QPushButton {{ padding: 2px 8px; color: {p.muted}; }}

/* step rail */
QListWidget#StepRail {{ background: {p.bg}; border: none; border-right: 1px solid {p.border};
    border-radius: 0; padding: 8px 6px; outline: 0; }}
QListWidget#StepRail::item {{ padding: 8px 10px; border-radius: 4px; margin: 1px 0;
    color: {p.muted}; }}
QListWidget#StepRail::item:hover {{ background: {p.surface_alt}; }}
QListWidget#StepRail::item:selected {{ background: {p.accent_soft}; color: {p.text}; }}

/* tabs (inner ones only; the main tab bar is replaced by the rail) */
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; padding: 6px 12px; color: {p.muted};
    border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {p.accent}; }}

/* busy bar */
QProgressBar#BusyBar {{ border: none; background: transparent; max-height: 3px; }}
QProgressBar#BusyBar::chunk {{ background: {p.accent}; }}
QProgressBar[role="bar"] {{ border: none; background: {p.surface_alt}; border-radius: 3px;
    max-height: 6px; }}
QProgressBar[role="bar"]::chunk {{ background: {p.accent}; border-radius: 3px; }}

/* chrome */
QMenuBar {{ background: {p.bg}; }}
QMenuBar::item:selected {{ background: {p.surface_alt}; border-radius: 4px; }}
QMenu {{ background: {p.surface}; border: 1px solid {p.border}; padding: 4px; }}
QMenu::item {{ padding: 5px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {p.accent_soft}; color: {p.text}; }}
QStatusBar {{ background: {p.bg}; color: {p.muted}; border-top: 1px solid {p.border}; }}
QStatusBar::item {{ border: none; }}
QLabel#status_line {{ color: {p.muted}; padding-left: 10px; }}
QDockWidget::title {{ background: {p.surface_alt}; padding: 4px 8px; }}
QSplitter::handle {{ background: {p.bg}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {p.border}; border-radius: 4px;
    min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QCheckBox::indicator {{ width: 15px; height: 15px; }}
"""


def qt_palette(p: Palette) -> QPalette:
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: p.bg,
        QPalette.ColorRole.WindowText: p.text,
        QPalette.ColorRole.Base: p.surface,
        QPalette.ColorRole.AlternateBase: p.surface_alt,
        QPalette.ColorRole.Text: p.text,
        QPalette.ColorRole.Button: p.surface,
        QPalette.ColorRole.ButtonText: p.text,
        QPalette.ColorRole.Highlight: p.accent,
        QPalette.ColorRole.HighlightedText: p.accent_text,
        QPalette.ColorRole.ToolTipBase: p.surface,
        QPalette.ColorRole.ToolTipText: p.text,
        QPalette.ColorRole.PlaceholderText: p.muted,
        QPalette.ColorRole.Link: p.accent,
    }
    for role, colour in roles.items():
        pal.setColor(role, QColor(colour))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(p.muted))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(p.muted))
    return pal


class ThemeManager(QObject):
    """The current palette, applied app-wide; `changed` tells plots to redraw."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self._mode = str(QSettings("mlti_object_cal", "gui").value("theme", "system"))
        if self._mode not in MODES:
            self._mode = "system"
        self.palette = LIGHT
        #: Type and chart scale. 1.0 normally; larger while a demo video is
        #: recorded, so text is legible on a phone. Never saved.
        self.scale = 1.0
        hints = QGuiApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(lambda *_: self._follow_system())

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"theme mode must be one of {MODES}, got {mode!r}")
        self._mode = mode
        QSettings("mlti_object_cal", "gui").setValue("theme", mode)
        self.apply()

    def _follow_system(self) -> None:
        if self._mode == "system":
            self.apply()

    def resolved(self) -> Palette:
        if self._mode == "dark" or (self._mode == "system" and system_is_dark()):
            return DARK
        return LIGHT

    def apply(self) -> None:
        app = QApplication.instance()
        if not isinstance(app, QApplication):
            return
        self.palette = self.resolved()
        app.setStyle("Fusion")
        app.setPalette(qt_palette(self.palette))
        app.setStyleSheet(stylesheet(self.palette, self.scale))
        self.changed.emit()

    def set_presentation(self, scale: float) -> None:
        """Enlarge text and charts by `scale` (1.0 = normal). Not remembered."""
        self.scale = float(scale)
        self.apply()


_MANAGER: ThemeManager | None = None


def manager() -> ThemeManager:
    """The application's theme; created (and applied) on first use."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ThemeManager()
        _MANAGER.apply()
    return _MANAGER


def current() -> Palette:
    return manager().palette if _MANAGER is not None else LIGHT


def current_scale() -> float:
    return _MANAGER.scale if _MANAGER is not None else 1.0


def mpl_rc(p: Palette) -> dict:
    return {
        "figure.facecolor": p.surface,
        "axes.facecolor": p.surface,
        "savefig.facecolor": p.surface,
        "axes.edgecolor": p.border,
        "axes.labelcolor": p.text,
        "axes.titlecolor": p.text,
        "text.color": p.text,
        "xtick.color": p.muted,
        "ytick.color": p.muted,
        "grid.color": p.border,
        "grid.alpha": 0.7,
        "legend.facecolor": p.surface,
        "legend.edgecolor": p.border,
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "axes.prop_cycle": __import__("cycler").cycler(
            color=[p.accent, p.warn, p.good, p.bad, p.muted]
        ),
    }


@contextmanager
def mpl_context(figure=None):
    """Matplotlib styled like the window, for one on-screen draw; exports untouched."""
    import matplotlib

    p = current()
    with matplotlib.rc_context(mpl_rc(p)):
        if figure is not None:
            figure.set_facecolor(p.surface)
            _scale_figure(figure, current_scale())
        yield p


BASE_DPI = 100.0


def _scale_figure(figure, scale: float) -> None:
    """
    Draw an on-screen figure `scale` times larger without changing its widget.

    Raising the dpi makes the same widget hold fewer "inches", so every font
    and line comes out bigger -- the same effect the presentation scale has
    on the Qt widgets. The Qt canvas reads `figure.dpi` on every resize, so
    the size is recomputed here once and stays consistent afterwards.
    """
    canvas = getattr(figure, "canvas", None)
    dpi = BASE_DPI * scale * getattr(canvas, "device_pixel_ratio", 1.0)
    if abs(figure.dpi - dpi) < 1e-6 or canvas is None or not hasattr(canvas, "width"):
        return
    figure.set_dpi(dpi)
    w = canvas.width() * getattr(canvas, "device_pixel_ratio", 1.0)
    h = canvas.height() * getattr(canvas, "device_pixel_ratio", 1.0)
    if w > 0 and h > 0:
        figure.set_size_inches(w / dpi, h / dpi, forward=False)
