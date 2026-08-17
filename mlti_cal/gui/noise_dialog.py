"""
Measure the pixel noise from a static sequence, in its own window.

Separate from the Setup tab on purpose. This is a different capture: the main
config wants the board MOVED between shots, and this wants it not moved at all.
Sharing one image list would invite pointing this at the calibration set, where
every corner genuinely moves and the "noise" measured would be the board's
motion.

The detector settings default to the ones the Setup tab is using, because
measuring the noise of a detector you are not going to run is measuring the
wrong thing -- but they stay editable here, since comparing refinement methods
on the same static set is exactly how you choose one.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from mlti_cal.detectors.settings import CHARUCO_CATALOG, CHARUCO_SUMMARY, CharucoSettings
from mlti_cal.gui.session import NoiseWorker, run_in_thread
from mlti_cal.gui.settings_panel import SettingsPanel
from mlti_cal.io.config import find_images


class NoiseDialog(QDialog):
    """
    Pick a static sequence and a board, run detection, report the noise.

    Emits `accepted_value` with the measured sigma when the user chooses to
    use it. Nothing is written back unless they do -- a measurement they
    disagree with should not silently become the assumption.
    """

    accepted_value = Signal(float)
    #: Every line of the finished measurement, emitted whether or not the user
    #: goes on to adopt the value -- a measurement that was taken and rejected
    #: is still part of what happened in this session.
    measured = Signal(list)

    def __init__(self, board_specs: list, settings: CharucoSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measure pixel noise from a static sequence")
        self.resize(760, 640)
        self._specs = list(board_specs)
        self._estimate = None
        self._build(settings)

    def _build(self, settings: CharucoSettings) -> None:
        root = QVBoxLayout(self)

        blurb = QLabel(
            "Point a fixed camera at a fixed board and take several pictures "
            "<b>without touching anything</b>. Every difference between one frame's "
            "corner and the next frame's same corner is detection noise.\n\n"
            "This measures repeatability, not accuracy: a refinement biased by half "
            "a pixel in a consistent direction is perfectly repeatable and will look "
            "excellent here. If the rig moves during the sequence, that motion is "
            "detected and reported separately rather than counted as noise."
        )
        blurb.setWordWrap(True)
        blurb.setTextFormat(Qt.RichText)
        root.addWidget(blurb)

        form = QFormLayout()
        row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("folder of static images")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._pick_folder)
        row.addWidget(self.folder_edit, 1)
        row.addWidget(browse)
        form.addRow("images", row)

        self.board_combo = QComboBox()
        for spec in self._specs:
            self.board_combo.addItem(f"{spec.id}  ({spec.squares_x}x{spec.squares_y})", spec)
        self.board_combo.setToolTip(
            "Which board is in the static pictures. Only this board is given to "
            "the detector, so a marker-ID collision between other configured "
            "boards cannot block a measurement that never touches them."
        )
        form.addRow("board", self.board_combo)
        root.addLayout(form)

        self.panel = SettingsPanel("Detector settings", CHARUCO_SUMMARY, CHARUCO_CATALOG)
        self.panel.set_values(settings.to_dict())
        root.addWidget(self.panel)

        bar = QHBoxLayout()
        self.run_btn = QPushButton("Run detection and measure")
        self.run_btn.clicked.connect(self._run)
        bar.addWidget(self.run_btn)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        bar.addWidget(self.progress, 1)
        root.addLayout(bar)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.output.setPlainText("no measurement yet")
        root.addWidget(self.output, 1)

        foot = QHBoxLayout()
        self.use_btn = QPushButton("Use this value")
        self.use_btn.setEnabled(False)
        self.use_btn.setToolTip(
            "Write the measured sigma into the assumed pixel noise on the Setup "
            "tab, and switch it on."
        )
        self.use_btn.clicked.connect(self._use)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        foot.addStretch()
        foot.addWidget(self.use_btn)
        foot.addWidget(close)
        root.addLayout(foot)

    # ------------------------------------------------------------------
    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder of static images")
        if path:
            self.folder_edit.setText(path)

    def _run(self) -> None:
        folder = self.folder_edit.text().strip()
        if not folder:
            QMessageBox.information(self, "No images", "Choose a folder of static images first.")
            return
        try:
            paths = find_images(folder)
        except NotADirectoryError as exc:
            QMessageBox.critical(self, "Bad folder", str(exc))
            return
        if len(paths) < 5:
            QMessageBox.information(
                self,
                "Too few images",
                f"Found {len(paths)} image(s). At least 5 of the same static scene "
                f"are needed before a standard deviation means anything; 20 or more "
                f"is better.",
            )
            return
        spec = self.board_combo.currentData()
        if spec is None:
            QMessageBox.information(
                self, "No board", "Define a calibration board on the Setup tab first."
            )
            return
        try:
            settings = CharucoSettings(**self.panel.values())
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid detector settings", str(exc))
            return

        self.run_btn.setEnabled(False)
        self.use_btn.setEnabled(False)
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        self.output.setPlainText(f"detecting in {len(paths)} image(s)...")
        worker = NoiseWorker([str(p) for p in paths], spec, settings)
        # Bound methods of this dialog, never lambdas -- see run_in_thread.
        worker.image_done.connect(self._on_image)
        run_in_thread(self, worker, self._on_done, self._on_failed)

    def _on_image(self, index: int, total: int, corners: int) -> None:
        self.progress.setValue(index + 1)
        self.progress.setFormat(f"%v / %m   ({corners} corners)")

    def _on_done(self, estimate) -> None:
        self.run_btn.setEnabled(True)
        self._estimate = estimate
        lines = [
            f"images   : {Path(self.folder_edit.text()).name}",
            f"board    : {self.board_combo.currentText()}",
            f"detector : {self.panel.summary_line()}",
            "",
            *estimate.summary_lines(),
        ]
        self.output.setPlainText("\n".join(lines))
        self.measured.emit(lines)
        self.use_btn.setEnabled(bool(estimate.usable))

    def _on_failed(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.output.setPlainText(f"FAILED: {message}")
        QMessageBox.critical(self, "Measurement failed", message)

    def _use(self) -> None:
        if self._estimate is not None and self._estimate.usable:
            self.accepted_value.emit(float(self._estimate.sigma_px))
            self.accept()
