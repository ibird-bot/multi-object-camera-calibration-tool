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
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


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
        arr = np.ascontiguousarray(array)
        if arr.ndim == 2:
            h, w = arr.shape
            img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, ch = arr.shape
            # OpenCV hands us BGR; Qt wants RGB.
            rgb = np.ascontiguousarray(arr[:, :, ::-1]) if ch == 3 else arr
            img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img.copy())
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
