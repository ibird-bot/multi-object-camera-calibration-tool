"""
Main window: the workflow as steps down the left, one shared Session.

Run with:  python -m mlti_cal.gui

Layout
    A step rail (Setup & detection, Solve, Report) replaces the tab bar; the
    views still live in a QTabWidget whose bar is hidden, so everything that
    drives `window.tabs` keeps working and the rail and the tabs cannot
    disagree. Each step shows whether it is done or needs a look.

    The log is a drawer: hidden until asked for, with the newest line always
    in the status bar and a count of warnings on its toggle, so a warning is
    never silent but the transcript does not take a third of the screen.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mlti_cal.gui import theme
from mlti_cal.gui.components import StepRail
from mlti_cal.gui.detection_view import DetectionView
from mlti_cal.gui.log_console import LogConsole, attach_console
from mlti_cal.gui.optimization_view import OptimizationView
from mlti_cal.gui.report_view import ReportView
from mlti_cal.gui.session import Session, busy

STEPS = ("Setup & detection", "Solve", "Report")
#: Sources whose lines count as warnings on the log toggle.
WARNING_SOURCES = ("plugins", "warning")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.theme = theme.manager()
        self.setWindowTitle("mlti_object_cal -- calibration workbench")
        self.resize(1500, 950)
        self.session = Session()

        # One console for the whole app, built before the views so each can be
        # handed it. Docked rather than embedded in a page: the transcript is
        # about the session, not about whichever page is in front.
        self.console = LogConsole()
        # Library and plug-in warnings (e.g. a detect() result refused by the
        # contract gate) are logged, not raised; this puts them in front of the user.
        self.log_handler = attach_console(self.console)
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setObjectName("log_dock")
        self.log_dock.setWidget(self.console)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()

        self.tabs = QTabWidget()
        self.detection = DetectionView(self.session, console=self.console)
        self.optimization = OptimizationView(self.session, console=self.console)
        self.report = ReportView(self.session, console=self.console)
        self.tabs.addTab(self.detection, "1. Setup && Detection")
        self.tabs.addTab(self.optimization, "2. Optimization")
        self.tabs.addTab(self.report, "3. Report")
        self.tabs.tabBar().hide()

        self.rail = StepRail(list(STEPS))
        self.rail.currentRowChanged.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self.rail.setCurrentRow)
        self.rail.setCurrentRow(0)

        self.busy_bar = QProgressBar()
        self.busy_bar.setObjectName("BusyBar")
        self.busy_bar.setRange(0, 0)  # indeterminate: a job is running
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setFixedHeight(3)
        self.busy_bar.setVisible(False)

        page = QWidget()
        pv = QVBoxLayout(page)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(0)
        pv.addWidget(self.busy_bar)
        pv.addWidget(self.tabs, 1)
        central = QWidget()
        cl = QHBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self.rail)
        cl.addWidget(page, 1)
        self.setCentralWidget(central)

        self._build_menus()
        self._build_status_bar()

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
        self.theme.changed.connect(self._on_theme_changed)
        # No page switch on a finished solve: the numbers worth reading first --
        # the RMS, the iteration log, the parameters -- are on the Solve page,
        # and being thrown into Report hides the one place that says whether
        # the solve was any good.

        # Busy bar and step marks follow the session; polling is cheap and
        # cannot miss a state change the way a forgotten signal could.
        self._tick = QTimer(self)
        self._tick.setInterval(200)
        self._tick.timeout.connect(self._update_progress)
        self._tick.start()

    # ------------------------------------------------------------------
    def _build_menus(self) -> None:
        """
        Top-level menus.

        Detector settings live here and ONLY here, because they belong to the
        detector rather than to any one board or run: the same parameters drive
        the calibration detection pass and the pixel noise measurement, which
        are in different windows. One way in means no second control that can
        fall out of step with the first. What they were set to is recorded in
        the log whenever they change, so a run stays reproducible without a
        second copy of the controls on a page.

        The noise measurement itself is not here -- it belongs next to the
        assumed pixel noise it exists to fill in, which is on the Setup page.
        """
        file_menu = self.menuBar().addMenu("&File")
        load_action = QAction("&Open config...", self)
        load_action.setShortcut(QKeySequence("Ctrl+O"))
        load_action.triggered.connect(self.detection._load_config)
        save_action = QAction("&Save config...", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.detection._save_config)
        file_menu.addAction(load_action)
        file_menu.addAction(save_action)

        detectors = self.menuBar().addMenu("&Detectors")

        settings_action = QAction("Detector &settings...", self)
        settings_action.setShortcut(QKeySequence("Ctrl+D"))
        settings_action.setStatusTip("Parameters for each detector, one tab per kind")
        settings_action.triggered.connect(self.detection.open_detector_settings)
        detectors.addAction(settings_action)

        plugins_action = QAction("&Plug-ins...", self)
        plugins_action.setStatusTip("What loaded from the plug-in folders, and what failed and why")
        plugins_action.triggered.connect(self._show_plugins)
        detectors.addSeparator()
        detectors.addAction(plugins_action)

        view = self.menuBar().addMenu("&View")
        show_log = self.log_dock.toggleViewAction()
        show_log.setText("Show &log")
        show_log.setShortcut(QKeySequence("Ctrl+L"))
        view.addAction(show_log)

        themes = view.addMenu("&Theme")
        group = QActionGroup(self)
        for mode, label in (("system", "Follow &Windows"), ("light", "&Light"), ("dark", "&Dark")):
            action = QAction(label, self, checkable=True)
            action.setChecked(self.theme.mode == mode)
            action.triggered.connect(lambda _=False, m=mode: self.theme.set_mode(m))
            group.addAction(action)
            themes.addAction(action)

    def _build_status_bar(self) -> None:
        self.status_label = QLabel("ready")
        self.status_label.setObjectName("status_line")
        self.statusBar().addWidget(self.status_label, 1)
        self._warnings_seen = 0
        self.log_button = QPushButton("Log")
        self.log_button.setObjectName("log_button")
        self.log_button.setToolTip("Show or hide the full log (Ctrl+L)")
        self.log_button.setFlat(True)
        self.log_button.clicked.connect(
            lambda: self.log_dock.setVisible(not self.log_dock.isVisible())
        )
        self.log_dock.visibilityChanged.connect(self._on_log_visibility)
        self.statusBar().addPermanentWidget(self.log_button)
        self.console.appended.connect(self._on_log_line)

    def _show_plugins(self) -> None:
        from mlti_cal.gui.plugin_dialogs import PluginsDialog

        PluginsDialog(self).exec()

    # -- status and log --------------------------------------------------
    def _log_setup(self, text: str) -> None:
        self.console.write("setup", text)

    def _log_report(self, text: str) -> None:
        self.console.write("report", text)

    def _set_status(self, text: str):
        self.status_label.setText(text)

    def _on_log_line(self, source: str, line: str) -> None:
        self.status_label.setText(f"{source}: {line.strip()}")
        if source in WARNING_SOURCES and not self.log_dock.isVisible():
            self._warnings_seen += 1
            self.log_button.setText(f"Log  ({self._warnings_seen} warning(s))")

    def _on_log_visibility(self, visible: bool) -> None:
        if visible:
            self._warnings_seen = 0
            self.log_button.setText("Log")

    # -- focus mode (presenting one panel) ---------------------------------
    def focus_mode(self, *widgets: QWidget) -> None:
        """
        Show only `widgets` -- everything else in the window is hidden.

        For presenting and for filming: the chosen panel (say the result cards
        and the convergence chart) then fills the window, and the window can be
        sized to the shape of the picture it is filmed for, instead of a small
        region being cropped out of a dense full layout afterwards.

        Every widget on the path from a chosen one up to the window stays; every
        sibling of that path is hidden, and remembered so `end_focus` restores
        exactly what was visible before.
        """
        self.end_focus()
        path: set[QWidget] = set()
        for widget in widgets:
            node: QWidget | None = widget
            while node is not None and node is not self:
                path.add(node)
                node = node.parentWidget()
        hidden: list[QWidget] = []
        for node in path:
            parent = node.parentWidget()
            if parent is None:
                continue
            for sibling in parent.findChildren(
                QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly
            ):
                if sibling in path or sibling.isWindow() or not sibling.isVisibleTo(parent):
                    continue
                sibling.hide()
                hidden.append(sibling)
        for chrome in (self.menuBar(), self.statusBar(), self.log_dock):
            if chrome.isVisible():
                chrome.hide()
                hidden.append(chrome)
        self._focus_hidden = hidden

    def end_focus(self) -> None:
        for widget in reversed(getattr(self, "_focus_hidden", [])):
            widget.show()
        self._focus_hidden = []

    # -- progress --------------------------------------------------------
    def _update_progress(self) -> None:
        views = (self.detection, self.optimization, self.report)
        if not getattr(self, "_focus_hidden", None):  # no layout jumps while presenting
            self.busy_bar.setVisible(any(busy(v) for v in views))
        s = self.session
        has_data = s.system is not None and bool(s.system.observations)
        self.rail.set_state(0, "done" if has_data else "todo")
        result = s.result
        if result is None:
            self.rail.set_state(1, "todo")
        else:
            self.rail.set_state(1, "done" if result.success else "attention")
        report = s.report
        if report is None:
            self.rail.set_state(2, "todo")
        else:
            critical = any(w.severity == "critical" for w in report.warnings)
            self.rail.set_state(2, "attention" if critical else "done")

    def _on_theme_changed(self) -> None:
        self.rail.refresh_marks()
        for view in (self.optimization, self.report):
            restyle = getattr(view, "on_theme_changed", None)
            if restyle is not None:
                restyle()

    def _on_system_changed(self):
        # Refresh only -- no page switch. Culling and re-detection happen while
        # the user is still working on the Setup page; yanking them to Solve
        # mid-edit loses their place.
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
