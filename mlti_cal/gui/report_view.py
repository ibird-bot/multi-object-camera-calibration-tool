"""
Window 3 -- Report.

The payoff view: the warnings panel in plain language, the (synthetic only)
ground-truth honesty verdict, and one tab per figure in `report.figures`.

Nothing is computed here, and nothing is drawn here either. The tab list, the
drawing and the exported PNGs all come from the figure registry, so a tab
cannot exist without its export and neither can drift from the other. Export
re-renders through that registry rather than saving the on-screen canvas: the
live figure is whatever size Qt stretched the widget to, which is how the
exports used to come out 34 inches wide with the plot marooned in a corner.

Every figure reads a `CalibrationReport` produced by the headless engine.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mlti_cal.gui.session import ReportWorker, Session, refuse_if_busy, run_in_thread
from mlti_cal.gui.settings_panel import SettingsPanel
from mlti_cal.gui.widgets import PlotCanvas
from mlti_cal.io.config import export_json, export_opencv_yaml
from mlti_cal.report.figures import FIGURES, render
from mlti_cal.report.settings import REPORT_CATALOG, REPORT_SUMMARY, ReportSettings

SEVERITY_COLOUR = {"critical": "#c0392b", "warning": "#c87f0a", "info": "#2d7d46"}


class ReportView(QWidget):
    status = Signal(str)

    def __init__(self, session: Session, console=None, parent=None):
        super().__init__(parent)
        self.session = session
        # Shared with every other tab; a private one when built standalone.
        from mlti_cal.gui.log_console import LogConsole, SourceLog

        self.console = console if console is not None else LogConsole()
        self.log = SourceLog(self.console, "report")
        self._build()

    def _build(self):
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.build_btn = QPushButton("Build report")
        self.build_btn.clicked.connect(self._build_report)
        bar.addWidget(self.build_btn)

        bar.addWidget(QLabel("uncertainty range (m):"))
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(0.05, 1000.0)
        self.range_spin.setValue(1.5)
        self.range_spin.setToolTip(
            "Projection uncertainty depends on range: focal length and camera "
            "position trade off differently with depth. No single value is "
            "correct, so it must be stated."
        )
        bar.addWidget(self.range_spin)

        bar.addWidget(QLabel("camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._redraw)
        bar.addWidget(self.camera_combo)

        bar.addStretch()
        self.settings_btn = QPushButton("Thresholds...")
        self.settings_btn.setCheckable(True)
        self.settings_btn.setToolTip(
            "Thresholds that decide which warnings fire. They change what the "
            "report SAYS, never what was solved."
        )
        self.settings_btn.toggled.connect(self._toggle_settings)
        bar.addWidget(self.settings_btn)
        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self._export)
        bar.addWidget(export_btn)
        root.addLayout(bar)

        self.report_panel = SettingsPanel("Report thresholds", REPORT_SUMMARY, REPORT_CATALOG)
        self.report_panel.setVisible(False)
        root.addWidget(self.report_panel)

        split = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.plots: dict[str, PlotCanvas] = {}
        # One tab per registered figure. Adding a figure to the registry adds
        # its tab and its exported PNG at once, so the two cannot drift.
        for spec in FIGURES:
            # Start each canvas at the size its figure was designed for. Only
            # the visible tab gets resized by Qt, so the others would otherwise
            # draw at the 5x4in constructor default -- on which a three-panel
            # figure with rotated tick labels has no room left for the axes
            # themselves, and constrained layout gives up with a warning.
            c = PlotCanvas(figsize=spec.size_for(None, ""))
            c.message("no report yet -- solve, then Build report")
            c.enable_hover()
            self.plots[spec.key] = c
            self.tabs.addTab(c, spec.title)
        split.addWidget(self.tabs)

        lower = QWidget()
        lv = QVBoxLayout(lower)
        lv.setContentsMargins(0, 0, 0, 0)
        self.warnings_label = QLabel("no report yet")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setTextFormat(Qt.RichText)
        lv.addWidget(self.warnings_label)
        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        lv.addWidget(self.summary_text)
        split.addWidget(lower)
        split.setSizes([520, 380])
        root.addWidget(split)

    # ------------------------------------------------------------------
    def _build_report(self):
        if refuse_if_busy(self, "a report build"):
            return
        if self.session.problem is None:
            QMessageBox.information(self, "Nothing to report", "Solve first.")
            return
        self.build_btn.setEnabled(False)
        # The range spinbox in the toolbar and `default_range_m` in the panel
        # are the same quantity; the toolbar one is what the user just touched,
        # so it wins and the panel is kept in step rather than silently ignored.
        settings = ReportSettings(**self.report_panel.values())
        settings.default_range_m = self.range_spin.value()
        worker = ReportWorker(self.session, self.range_spin.value(), settings=settings)
        run_in_thread(self, worker, self._on_report, self._on_failed, self.status.emit)

    def _toggle_settings(self, shown: bool) -> None:
        self.report_panel.setVisible(shown)

    def _on_report(self, report):
        self.build_btn.setEnabled(True)
        self.session.report = report
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItems(list(self.session.system.cameras))
        self.camera_combo.blockSignals(False)
        self._redraw()
        self.status.emit("report built")

    def _on_failed(self, message):
        self.build_btn.setEnabled(True)
        QMessageBox.critical(self, "Report failed", message)
        self.status.emit("report failed")

    # ------------------------------------------------------------------
    def _redraw(self):
        report = self.session.report
        if report is None:
            return
        cam = self.camera_combo.currentText() or next(iter(self.session.system.cameras))
        for spec in FIGURES:
            canvas = self.plots[spec.key]
            fig = canvas.clear()
            try:
                spec.draw(fig, report, self.session.system, cam)
            except Exception as exc:  # one broken figure must not blank the rest
                from mlti_cal.report.figures import message

                self.log.warning(f"figure {spec.key!r} failed to draw: {exc}")
                message(fig, f"{spec.key} could not be drawn:\n{exc}")
            canvas.draw()
        self._render_warnings(report)
        self.summary_text.setPlainText(report.text_summary())

    def _render_warnings(self, report):
        parts = []
        gt = report.gt_check
        if gt:
            rchi2 = gt.get("reduced_chi2", float("nan"))
            good = 0.5 < rchi2 < 2.0
            verdict = (
                "the reported uncertainty is honest"
                if good
                else "THE REPORTED UNCERTAINTY IS NOT TRUSTWORTHY"
            )
            parts.append(
                f"<div style='padding:4px;background:{'#e6f4ea' if good else '#fdecea'};'>"
                f"<b>Ground-truth check:</b> reduced chi-squared "
                f"<b>{rchi2:.3f}</b> (want ~1.0) &mdash; "
                f"{verdict}"
                f"</div>"
            )
        for item in report.warnings:
            colour = SEVERITY_COLOUR.get(item.severity, "#333")
            action = f"<br><i style='color:#555'>&rarr; {item.action}</i>" if item.action else ""
            parts.append(
                f"<div style='margin:3px 0'><b style='color:{colour}'>"
                f"[{item.severity.upper()}]</b> {item.message}{action}</div>"
            )
        self.warnings_label.setText("".join(parts))

    def _export(self):
        if self.session.report is None:
            QMessageBox.information(self, "Nothing to export", "Build the report first.")
            return
        d = QFileDialog.getExistingDirectory(self, "Export to folder")
        if not d:
            return
        from pathlib import Path

        out = Path(d)
        self.session.report.save_json(out / "report.json")
        export_json(self.session.system, out / "calibration.json")
        export_opencv_yaml(self.session.system, out / "calibration.yaml")
        (out / "report.txt").write_text(self.session.report.text_summary(), encoding="utf-8")
        # Deliberately NOT `canvas.figure.savefig`. The on-screen figure has
        # been stretched to whatever size the Qt widget happens to be, so
        # saving it produced one 34-inch-wide PNG per tab, each a different
        # shape, with the content marooned in a corner. Re-render instead, at
        # the size the figure was designed for.
        cam = self.camera_combo.currentText() or next(iter(self.session.system.cameras))
        for spec in FIGURES:
            try:
                fig = render(spec, self.session.report, self.session.system, cam)
                fig.savefig(out / f"{spec.key}.png", dpi=150)
            except Exception as exc:
                self.log.warning(f"figure {spec.key!r} not exported: {exc}")
        self.status.emit(f"exported to {out}")
        QMessageBox.information(self, "Exported", f"Written to {out}")
