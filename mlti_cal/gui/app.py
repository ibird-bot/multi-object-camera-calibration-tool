"""
Main window: the three views as tabs over one shared Session.

Run with:  python -m mlti_cal.gui
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

from mlti_cal.gui.detection_view import DetectionView
from mlti_cal.gui.log_console import LogConsole
from mlti_cal.gui.optimization_view import OptimizationView
from mlti_cal.gui.report_view import ReportView
from mlti_cal.gui.session import Session


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mlti_object_cal -- calibration workbench")
        self.resize(1500, 950)
        self.session = Session()

        # One console for the whole app, built before the views so each can be
        # handed it. Docked rather than embedded in a tab: the transcript is
        # about the session, not about whichever tab is in front.
        self.console = LogConsole()
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setObjectName("log_dock")
        self.log_dock.setWidget(self.console)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)

        self.tabs = QTabWidget()
        self.detection = DetectionView(self.session, console=self.console)
        self.optimization = OptimizationView(self.session, console=self.console)
        self.report = ReportView(self.session, console=self.console)
        self.tabs.addTab(self.detection, "1. Setup && Detection")
        self.tabs.addTab(self.optimization, "2. Optimization")
        self.tabs.addTab(self.report, "3. Report")
        self.setCentralWidget(self.tabs)

        self._build_menus()

        self.status_label = QLabel("ready")
        self.statusBar().addWidget(self.status_label)

        for view in (self.detection, self.optimization, self.report):
            view.status.connect(self._set_status)
        # Everything worth keeping goes to the shared console -- a status bar
        # holds one line for a second, which is no use when you want to see what
        # a run actually did. Optimization is not connected here: its messages
        # already go straight to the console, and echoing them would double
        # every line. `logged`, not `status`: detection ticks the status bar once
        # per image, which would evict the whole transcript from a bounded log.
        self.detection.logged.connect(self._log_setup)
        self.report.status.connect(self._log_report)
        self.detection.system_changed.connect(self._on_system_changed)
        # No tab switch on a finished solve: the numbers worth reading first --
        # the RMS, the iteration log, the parameters -- are all in the
        # Optimization tab, and being thrown into Report hides the one place
        # that says whether the solve was any good.

    def _build_menus(self) -> None:
        """
        Top-level menus.

        Detector settings live here and ONLY here, because they belong to the
        detector rather than to any one board or run: the same parameters drive
        the calibration detection pass and the pixel noise measurement, which
        are in different windows. One way in means no second control that can
        fall out of step with the first. What they were set to is recorded in
        the log whenever they change, so a run stays reproducible without a
        second copy of the controls on a tab.

        The noise measurement itself is not here -- it belongs next to the
        assumed pixel noise it exists to fill in, which is on the Setup tab.
        """
        detectors = self.menuBar().addMenu("&Detectors")

        settings_action = QAction("Detector &settings...", self)
        settings_action.setShortcut(QKeySequence("Ctrl+D"))
        settings_action.setStatusTip("Parameters for each detector, one tab per kind")
        settings_action.triggered.connect(self.detection.open_detector_settings)
        detectors.addAction(settings_action)

        view = self.menuBar().addMenu("&View")
        show_log = self.log_dock.toggleViewAction()
        show_log.setText("Show &log")
        show_log.setShortcut(QKeySequence("Ctrl+L"))
        view.addAction(show_log)

    def _log_setup(self, text: str) -> None:
        self.console.write("setup", text)

    def _log_report(self, text: str) -> None:
        self.console.write("report", text)

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _on_system_changed(self):
        # Refresh only -- no tab switch. Culling and re-detection happen while
        # the user is still working in the Setup tab; yanking them to
        # Optimization mid-edit loses their place.
        self.optimization.refresh()


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
