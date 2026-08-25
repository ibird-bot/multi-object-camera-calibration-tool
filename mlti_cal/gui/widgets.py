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
from PySide6.QtCore import QSize, Qt, Signal
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
        """Placeholder text for a plot that has no data yet."""
        fig = self.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=10)
        ax.set_axis_off()
        self.draw()


class ImageView(QScrollArea):
    """Scrollable image display with fit-to-window toggling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = QLabel("no image")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(1, 1)
        self.setWidget(self._label)
        self.setWidgetResizable(True)
        self._pixmap: QPixmap | None = None
        self._fit = True

    def set_image(self, array: np.ndarray | None):
        if array is None:
            self._pixmap = None
            self._label.setText("no image")
            return
        self._pixmap = QPixmap.fromImage(to_qimage(array))
        self._rescale()

    def set_fit(self, fit: bool):
        self._fit = fit
        self._rescale()

    def _rescale(self):
        if self._pixmap is None:
            return
        if self._fit:
            target = self.viewport().size()
            self._label.setPixmap(
                self._pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self._label.setPixmap(self._pixmap)

    def resizeEvent(self, event):  # Qt API requires this name
        super().resizeEvent(event)
        self._rescale()


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
        self.setIconSize(QSize(self.THUMB, self.THUMB))
        self.setGridSize(QSize(self.THUMB + 20, self.THUMB + 32))
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
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
            item.setToolTip(key)
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
