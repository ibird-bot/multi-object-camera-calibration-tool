"""
Main window: the three views as tabs over one shared Session.

Run with:  python -m mlti_cal.gui
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QTabWidget

from mlti_cal.gui.detection_view import DetectionView
from mlti_cal.gui.optimization_view import OptimizationView
from mlti_cal.gui.report_view import ReportView
from mlti_cal.gui.session import Session
from mlti_cal.solvers import backend_status


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mlti_object_cal -- calibration workbench")
        self.resize(1500, 950)
        self.session = Session()

        self.tabs = QTabWidget()
        self.detection = DetectionView(self.session)
        self.optimization = OptimizationView(self.session)
        self.report = ReportView(self.session)
        self.tabs.addTab(self.detection, "1. Setup && Detection")
        self.tabs.addTab(self.optimization, "2. Optimization")
        self.tabs.addTab(self.report, "3. Report")
        self.setCentralWidget(self.tabs)

        self.status_label = QLabel("ready")
        self.statusBar().addWidget(self.status_label)
        self._show_backend_status()

        for view in (self.detection, self.optimization, self.report):
            view.status.connect(self._set_status)
        self.detection.system_changed.connect(self._on_system_changed)
        self.optimization.solved.connect(lambda: self.tabs.setCurrentWidget(self.report))

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _on_system_changed(self):
        self.optimization.refresh()
        self.tabs.setCurrentWidget(self.optimization)

    def _show_backend_status(self):
        usable = [n for n, (ok, _) in backend_status().items() if ok]
        blocked = {n: why for n, (ok, why) in backend_status().items() if not ok}
        msg = f"backends available: {', '.join(usable)}"
        if blocked:
            msg += f"  |  unavailable: {', '.join(blocked)}"
        self.statusBar().addPermanentWidget(QLabel(msg))


def main() -> int:
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
    except Exception as exc:  # pragma: no cover - startup failure path
        QMessageBox.critical(None, "Startup failed", f"{type(exc).__name__}: {exc}")
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
