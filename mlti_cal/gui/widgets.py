"""Reusable widgets: an embedded matplotlib canvas and an image viewer."""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QSize, Qt, Signal  # noqa: E402
from PySide6.QtGui import QBrush, QColor, QIcon, QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

    def clear(self):
        self.figure.clear()
        return self.figure

    def draw(self):
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

    def resizeEvent(self, event):  # noqa: N802 (Qt API)
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
