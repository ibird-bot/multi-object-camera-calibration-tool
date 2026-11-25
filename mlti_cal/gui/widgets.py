"""Reusable widgets: an embedded matplotlib canvas and an image viewer."""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mlti_cal.gui import theme


def to_qimage(array: np.ndarray) -> QImage:
    """
    Wrap a numpy image as a QImage.

    `.copy()` is not optional: QImage does not take ownership of the buffer, so
    without it the pixels are freed the moment the numpy array goes out of
    scope and the widget paints garbage or crashes.
    """
    arr = np.ascontiguousarray(array)
    if arr.ndim == 2:
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, ch = arr.shape
    # OpenCV hands us BGR; Qt wants RGB.
    rgb = np.ascontiguousarray(arr[:, :, ::-1]) if ch == 3 else arr
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class PlotCanvas(QWidget):
    """A matplotlib figure with a toolbar, sized to fit its dock."""

    def __init__(self, parent=None, toolbar: bool = True, figsize=(5, 4)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # A splitter will happily squeeze a widget to a few pixels. At that
        # point the axes have less room than their own labels need, constrained
        # layout bails out, and the plot is unreadable anyway -- so refuse to go
        # below a size where a figure can still be drawn.
        self.canvas.setMinimumSize(240, 180)
        self._tooltip = None
        self._hovered = None
        self._hover_connected = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

    def clear(self):
        self.figure.clear()
        # Cleared along with the figure: the annotation is an artist on it, and
        # keeping a stale reference would redraw a tooltip onto axes that no
        # longer exist.
        self._tooltip = None
        self._hovered = None
        return self.figure

    def draw(self):
        self.canvas.draw_idle()

    # -- cell inspection ------------------------------------------------
    def enable_hover(self) -> None:
        """
        Read heatmap cells back by pointing at them.

        A correlation matrix answers "which parameters are entangled" at a
        glance but not "entangled by how much"; the number matters and there is
        no room to print several hundred of them. Any figure whose drawer left
        `hover_grids` on it becomes inspectable -- the canvas asks the grid to
        describe the cell rather than working out what it contains, so the
        readout cannot disagree with the picture.
        """
        if getattr(self, "_hover_connected", False):
            return
        self._hover_connected = True
        self._tooltip = None
        self._hovered = None
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def _on_hover(self, event):
        grids = getattr(self.figure, "hover_grids", None)
        cell = None
        if grids and event.inaxes is not None and event.xdata is not None:
            for grid in grids:
                if grid.ax is event.inaxes:
                    cell = (grid, int(round(event.ydata)), int(round(event.xdata)))
                    break

        key = None if cell is None else (id(cell[0]), cell[1], cell[2])
        if key == self._hovered:
            return  # same cell: repainting on every mouse move is what makes
            # a hover readout feel sluggish on a large figure.
        self._hovered = key

        if self._tooltip is not None:
            self._tooltip.remove()
            self._tooltip = None

        if cell is not None:
            grid, row, col = cell
            text = grid.describe(row, col)
            if text:
                self._tooltip = grid.ax.annotate(
                    text,
                    xy=(col, row),
                    xytext=(12, 12),
                    textcoords="offset points",
                    fontsize=8,
                    zorder=100,
                    annotation_clip=False,
                    bbox={"boxstyle": "round,pad=0.4", "fc": "#ffffe0", "ec": "#888888"},
                )
        self.canvas.draw_idle()

    def message(self, text: str):
        """Placeholder text for a plot that has no data yet, in the window's theme."""
        with theme.mpl_context(self.figure) as p:
            fig = self.clear()
            fig.set_facecolor(p.surface)
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=10, color=p.muted)
            ax.set_axis_off()
            self.draw()


class ImageView(QScrollArea):
    """Scrollable image display with fit-to-window toggling, wheel zoom and drag-to-pan."""

    #: Multiplier applied per wheel notch, and the range zoom is clamped to.
    ZOOM_STEP = 1.25
    MIN_ZOOM = 0.1
    MAX_ZOOM = 20.0
    #: Longest side of the zoomed pixmap. The pixmap is rebuilt at full zoomed
    #: size, so without a cap a 4000 px image at 20x would need ~25 GB.
    MAX_ZOOMED_PX = 16000

    #: Emitted when a wheel-zoom turns fit-to-window off on its own, so an
    #: external "fit" checkbox can follow instead of silently going stale.
    fit_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        # State before `setWidget`: it installs QScrollArea's own event filter,
        # which calls the `eventFilter` override below while __init__ is running.
        self._pixmap: QPixmap | None = None
        self._fit = True
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPoint()
        self._pan_scroll_start = QPoint()
        self._label = QLabel("no image")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(1, 1)
        self.setWidget(self._label)
        # Not resizable: the label is sized in `_rescale` to the picture, or to
        # the window when the picture is smaller. A resizable label stayed at
        # window size (its minimum size is 1x1), so a zoomed picture was clipped
        # around its centre and its edges could not be scrolled to.
        self.setWidgetResizable(False)
        # Mouse presses land on the label, which fills the viewport, and do not
        # propagate up the way wheel events do -- so panning is caught there.
        self._label.installEventFilter(self)

    def set_image(self, array: np.ndarray | None):
        if array is None:
            self._pixmap = None
            self._label.setText("no image")
            self._label.resize(self.viewport().size())
            return
        self._pixmap = QPixmap.fromImage(to_qimage(array))
        self._zoom = 1.0
        self._rescale()

    def set_fit(self, fit: bool):
        self._fit = fit
        if fit:
            self._zoom = 1.0
        self._rescale()

    def _fit_scale(self) -> float:
        if self._pixmap is None or self._pixmap.isNull():
            return 1.0
        vp, pm = self.viewport().size(), self._pixmap.size()
        return min(vp.width() / pm.width(), vp.height() / pm.height())

    def _rescale(self):
        if self._pixmap is None:
            self._label.resize(self.viewport().size())
            return
        if self._fit:
            target = self.viewport().size()
        else:
            target = self._pixmap.size() * self._zoom
        shown = self._pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        view = self.viewport().size()
        self._label.setPixmap(shown)
        self._label.resize(max(shown.width(), view.width()), max(shown.height(), view.height()))

    def resizeEvent(self, event):  # Qt API requires this name
        super().resizeEvent(event)
        self._rescale()

    def wheelEvent(self, event):  # Qt API requires this name
        delta = event.angleDelta().y()
        if self._pixmap is None or self._pixmap.isNull() or delta == 0:
            super().wheelEvent(event)
            return
        if self._fit:
            # Start from the size on screen, so the image does not jump.
            self._zoom = self._fit_scale()
            self._fit = False
            self.fit_changed.emit(False)
        factor = self.ZOOM_STEP if delta > 0 else 1 / self.ZOOM_STEP
        longest = max(self._pixmap.width(), self._pixmap.height())
        max_zoom = min(self.MAX_ZOOM, self.MAX_ZOOMED_PX / longest)
        # The point of the image under the mouse must stay under the mouse.
        # Without this the scroll position never moves, so every zoom grows the
        # picture away from the top-left or the centre and the edges cannot be
        # reached with the cursor.
        cursor = event.position().toPoint()
        before = self._image_fraction_at(cursor)
        self._zoom = min(max_zoom, max(self.MIN_ZOOM, self._zoom * factor))
        self._rescale()
        self._scroll_to_fraction(before, cursor)
        event.accept()

    def _shown_pixmap_size(self) -> QSize:
        pm = self._label.pixmap()
        return pm.size() if pm is not None and not pm.isNull() else QSize(1, 1)

    def _content_offset(self) -> QPoint:
        """Where the picture starts inside the label: it is centred when smaller."""
        pm, lab = self._shown_pixmap_size(), self._label.size()
        return QPoint(
            max(0, (lab.width() - pm.width()) // 2), max(0, (lab.height() - pm.height()) // 2)
        )

    def _image_fraction_at(self, viewport_pos: QPoint) -> tuple[float, float]:
        """The picture point under a viewport position, as 0..1 of its width and height."""
        pm, off = self._shown_pixmap_size(), self._content_offset()
        x = self.horizontalScrollBar().value() + viewport_pos.x() - off.x()
        y = self.verticalScrollBar().value() + viewport_pos.y() - off.y()
        return (min(1.0, max(0.0, x / pm.width())), min(1.0, max(0.0, y / pm.height())))

    def _scroll_to_fraction(self, fraction: tuple[float, float], viewport_pos: QPoint) -> None:
        """Scroll so that picture point `fraction` sits under `viewport_pos`."""
        pm = self._shown_pixmap_size()
        hbar, vbar = self.horizontalScrollBar(), self.verticalScrollBar()
        off = self._content_offset()
        hbar.setValue(round(fraction[0] * pm.width() + off.x() - viewport_pos.x()))
        vbar.setValue(round(fraction[1] * pm.height() + off.y() - viewport_pos.y()))

    def eventFilter(self, obj, event):
        if obj is self._label and not self._fit:
            etype = event.type()
            if etype == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._panning = True
                self._pan_start = event.globalPosition().toPoint()
                self._pan_scroll_start = QPoint(
                    self.horizontalScrollBar().value(), self.verticalScrollBar().value()
                )
                self._label.setCursor(Qt.ClosedHandCursor)
                return True
            if etype == QEvent.MouseMove and self._panning:
                delta = event.globalPosition().toPoint() - self._pan_start
                self.horizontalScrollBar().setValue(self._pan_scroll_start.x() - delta.x())
                self.verticalScrollBar().setValue(self._pan_scroll_start.y() - delta.y())
                return True
            if etype == QEvent.MouseButtonRelease and self._panning:
                self._panning = False
                self._label.setCursor(Qt.ArrowCursor)
                return True
        return super().eventFilter(obj, event)


class ThumbnailStrip(QListWidget):
    """
    A wrapping grid of image thumbnails, one per capture.

    Items are addressed by an opaque string key rather than by row, because
    detection reports progress out of the order the rows were built and rows
    get culled underneath us. A key keeps "which picture is this" stable.
    """

    THUMB = 128
    picked = Signal(str)  # key of the clicked thumbnail

    #: Filled in while an image is being processed, then cleared.
    ACTIVE = QColor(56, 108, 176)
    #: Kept after the run on images where detection found nothing -- the whole
    #: point of showing captures as pictures is spotting the bad ones.
    EMPTY = QColor(140, 45, 45)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        # Camera images are wider than tall. A square icon cell left a band of
        # empty space between each picture and its caption.
        icon_h = self.THUMB * 3 // 4
        self.setIconSize(QSize(self.THUMB, icon_h))
        # Two caption lines, at the presentation scale when a demo is recorded:
        # the widget's own font is not styled yet at this point.
        caption_h = int(2 * self.fontMetrics().lineSpacing() * max(1.0, theme.current_scale())) + 10
        self.setGridSize(QSize(self.THUMB + 16, icon_h + caption_h))
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setUniformItemSizes(True)
        self.setSpacing(4)
        self.setWordWrap(True)
        self._items: dict[str, QListWidgetItem] = {}
        self._empty: set[str] = set()
        self._active: str | None = None
        self.itemSelectionChanged.connect(self._emit_picked)

    # -- contents ------------------------------------------------------
    def set_entries(self, keys_and_labels=()) -> None:
        """
        Replace the whole strip. Icons are filled in afterwards.

        Not named `reset`: QAbstractItemView already defines `reset()` and Qt
        calls it itself whenever the model changes, so a `reset()` that calls
        `clear()` recurses until the process dies. Verified -- it did.
        """
        self.clear()
        self._items.clear()
        self._empty.clear()
        self._active = None
        for key, label in keys_and_labels:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            item.setToolTip(key)
            # Explicit, because uniform item sizes are taken from the FIRST item,
            # which has no icon yet: with a larger font that size came out
            # smaller than the icon and every thumbnail was clipped to a corner.
            item.setSizeHint(self.gridSize() - QSize(4, 4))
            self.addItem(item)
            self._items[key] = item

    def set_image(self, key: str, array: np.ndarray) -> None:
        item = self._items.get(key)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(to_qimage(array))))

    def set_caption(self, key: str, text: str) -> None:
        item = self._items.get(key)
        if item is not None:
            item.setText(text)

    def item_for(self, key: str) -> QListWidgetItem | None:
        """Look an item up by key. Not `item`: QListWidget already takes a row."""
        return self._items.get(key)

    def keys(self) -> list[str]:
        return list(self._items)

    # -- progress ------------------------------------------------------
    @property
    def active_key(self) -> str | None:
        return self._active

    @property
    def empty_keys(self) -> set[str]:
        return set(self._empty)

    def mark_active(self, key: str | None) -> None:
        """Tint the image currently being processed and scroll it into view."""
        previous, self._active = self._active, key
        # Only the two items whose state actually changed are repainted. The
        # obvious loop over every item runs once per image, so a few hundred
        # captures turn into tens of thousands of pointless repaints during the
        # one operation that has to stay responsive.
        for k in (previous, key):
            if k is not None:
                self._paint(k)
        if key is not None and key in self._items:
            self.scrollToItem(self._items[key])

    def mark_empty(self, key: str, empty: bool = True) -> None:
        """Flag an image that yielded no corners."""
        if empty:
            self._empty.add(key)
        else:
            self._empty.discard(key)
        self._paint(key)

    def _paint(self, key: str) -> None:
        item = self._items.get(key)
        if item is None:
            return
        if key == self._active:
            colour = self.ACTIVE
        elif key in self._empty:
            colour = self.EMPTY
        else:
            item.setBackground(QBrush())
            item.setForeground(QBrush())
            return
        item.setBackground(QBrush(colour))
        # Both tints are dark; the default text colour is near-black under a
        # light theme, which would leave the caption unreadable exactly on the
        # items the user needs to read.
        item.setForeground(QBrush(QColor(Qt.white)))

    def _emit_picked(self):
        for item in self.selectedItems():
            self.picked.emit(item.data(Qt.UserRole))
            return
