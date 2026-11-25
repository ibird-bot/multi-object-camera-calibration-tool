"""
Window 3 -- Report.

Verdict first, detail on demand. Four pages behind one switch:

  Overview   one sentence on whether the calibration can be used, four
             numbers (error, held-out error, worst camera, coverage) and a
             per-camera table.
  Findings   the warnings, grouped by topic and ordered by severity: four
             cameras with empty corners are ONE finding with four lines, not
             four identical paragraphs. Each says what to do.
  Figures    one figure at a time, at full size, picked from a list.
  Text       the full text report, as exported.

The previous page showed all of it at once -- seven figure tabs, every warning
as its own paragraph and the raw text -- which put the one thing a user needs
first ("can I use this?") nowhere in particular.

Nothing is computed here. The figure list, the drawing and the exported PNGs
all come from the figure registry, so a figure cannot exist without its
export and neither can drift from the other. Export re-renders through that
registry rather than saving the on-screen canvas, and outside the GUI theme:
an export from a dark session still produces figures for a white page.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mlti_cal.gui import theme
from mlti_cal.gui.components import (
    Banner,
    MetricCard,
    Segmented,
    dot_icon,
    plain_table,
    primary,
    set_role,
)
from mlti_cal.gui.session import ReportWorker, Session, refuse_if_busy, run_in_thread
from mlti_cal.gui.settings_panel import SettingsPanel
from mlti_cal.gui.widgets import PlotCanvas
from mlti_cal.io.config import export_json, export_opencv_yaml
from mlti_cal.models.camera import MODEL_LABELS
from mlti_cal.report.figures import FIGURES, render
from mlti_cal.report.settings import REPORT_CATALOG, REPORT_SUMMARY, ReportSettings

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}
PAGES = ("Overview", "Findings", "Figures", "Text report")

#: One plain title per warning code; the individual messages go underneath.
FINDING_TITLES = {
    "solver_not_converged": "The solver did not converge",
    "points_behind_camera": "Points were projected from behind a camera",
    "rank_deficient": "Some parameters cannot be determined from this data",
    "ill_conditioned": "The problem is poorly conditioned",
    "noise_model_mismatch": "The assumed pixel noise disagrees with the fit",
    "uncertainty_model_approximate": "The uncertainty estimate is approximate here",
    "extreme_correlation": "Some parameters cannot be told apart",
    "intrinsic_sigma_unreliable": "Some camera parameter uncertainties are not trustworthy",
    "too_few_frames": "Too few frames",
    "poor_coverage": "Parts of the image have no data",
    "empty_image_corners": "Image corners without data",
    "insufficient_tilt": "Not enough tilt between the views",
    "heavy_tails": "More large errors than the noise explains",
    "many_outliers": "Many outlier corners",
    "overfitting": "Held-out frames fit clearly worse (overfitting)",
    "mild_optimism": "Held-out frames fit slightly worse",
    "uneven_cameras": "One camera fits worse than the others",
    "no_issues": "No issues found",
}


def _severity_colour(severity: str) -> str:
    p = theme.current()
    return {"critical": p.bad, "warning": p.warn}.get(severity, p.good)


def _discard(widget: QWidget) -> None:
    """
    Remove a row NOW. `deleteLater` alone leaves the widget painted until the
    event loop gets round to it -- and a rebuilt list then draws over the old
    one (seen: two generations of findings printed on top of each other).
    """
    widget.hide()
    widget.setParent(None)
    widget.deleteLater()


class FindingCard(QFrame):
    """
    One topic as a flat row: a status dot and a title, the measured facts, then
    what to do. No box and no fold -- a finding is a few lines, and hiding
    them behind a "Details" toggle made every one a click.
    """

    def __init__(self, code: str, items: list, parent=None):
        super().__init__(parent)
        self.setObjectName("Row")
        self.setProperty("finding", code)
        severity = min((w.severity for w in items), key=lambda s: SEVERITY_ORDER.get(s, 3))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 10, 4, 12)
        lay.setSpacing(3)
        head = QHBoxLayout()
        head.setSpacing(8)
        mark = QLabel()
        mark.setPixmap(dot_icon(_severity_colour(severity), 7).pixmap(14, 14))
        head.addWidget(mark)
        title = FINDING_TITLES.get(code, items[0].message)
        heading = QLabel(f"<b>{title}</b>")
        heading.setWordWrap(True)
        head.addWidget(heading, 1)
        lay.addLayout(head)
        for w in items:
            line = QLabel(w.message)
            line.setWordWrap(True)
            line.setIndent(22)
            line.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            lay.addWidget(line)
        for action in dict.fromkeys(w.action for w in items if w.action):
            a = set_role(QLabel(action), "muted")
            a.setWordWrap(True)
            a.setIndent(22)
            lay.addWidget(a)


class ReportView(QWidget):
    status = Signal(str)

    def __init__(self, session: Session, console=None, parent=None):
        super().__init__(parent)
        self.session = session
        # Shared with every other page; a private one when built standalone.
        from mlti_cal.gui.log_console import LogConsole, SourceLog

        self.console = console if console is not None else LogConsole()
        self.log = SourceLog(self.console, "report")
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        bar = QHBoxLayout()
        self.build_btn = primary(QPushButton("Build report"))
        self.build_btn.clicked.connect(self._build_report)
        bar.addWidget(self.build_btn)
        bar.addSpacing(12)
        self.switch = Segmented(list(PAGES))
        self.switch.setObjectName("report_switch")
        bar.addWidget(self.switch)
        bar.addStretch()

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

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_overview())
        self.pages.addWidget(self._build_findings())
        self.pages.addWidget(self._build_figures())
        self.pages.addWidget(self._build_text())
        self.switch.changed.connect(self.pages.setCurrentIndex)
        root.addWidget(self.pages, 1)
        self._show_empty()

    # -- pages -----------------------------------------------------------
    def _build_overview(self) -> QWidget:
        page = QWidget()
        page.setObjectName("report_overview")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(10)
        self.banner = Banner()
        self.banner.setObjectName("Banner")
        lay.addWidget(self.banner)
        self.gt_label = set_role(QLabel(""), "muted")
        self.gt_label.setWordWrap(True)
        self.gt_label.setVisible(False)
        lay.addWidget(self.gt_label)

        self.overview_cards = QWidget()
        self.overview_cards.setObjectName("report_cards")
        grid = QGridLayout(self.overview_cards)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        self.rms_card = MetricCard("Reprojection error", "rms_card")
        self.heldout_card = MetricCard("Held-out frames", "report_heldout_card")
        self.worst_card = MetricCard("Worst camera", "worst_card")
        self.coverage_card = MetricCard("Image coverage", "coverage_card")
        for i, card in enumerate(
            (self.rms_card, self.heldout_card, self.worst_card, self.coverage_card)
        ):
            grid.addWidget(card, 0, i)
        lay.addWidget(self.overview_cards)

        # The three most important findings, so the answer to "what next?" is
        # on the first page and the full list is one click away.
        self.top_findings = QFrame()
        self.top_findings.setObjectName("Card")
        self.top_findings.setProperty("card", "top_findings")
        tl = QVBoxLayout(self.top_findings)
        tl.setContentsMargins(14, 10, 14, 10)
        tl.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(set_role(QLabel("Check first"), "caption"))
        head.addStretch()
        self.all_findings_btn = QPushButton("All findings")
        self.all_findings_btn.setFlat(True)
        self.all_findings_btn.clicked.connect(lambda: self.show_page("Findings"))
        head.addWidget(self.all_findings_btn)
        tl.addLayout(head)
        self.top_findings_rows = QVBoxLayout()
        self.top_findings_rows.setSpacing(6)
        tl.addLayout(self.top_findings_rows)
        self.top_findings.setVisible(False)
        lay.addWidget(self.top_findings)

        self.camera_table = QTableWidget(0, 5)
        self.camera_table.setObjectName("camera_table")
        self.camera_table.setHorizontalHeaderLabels(
            ["camera", "model", "corners", "RMS (px)", "coverage"]
        )
        header = self.camera_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        plain_table(self.camera_table)
        for col in (2, 3):  # number columns: the heading sits over the digits
            self.camera_table.horizontalHeaderItem(col).setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
        self.camera_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.camera_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.camera_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        lay.addWidget(self.camera_table)
        lay.addStretch()
        return page

    def _build_findings(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName("report_findings")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        holder = QWidget()
        self.findings_layout = QVBoxLayout(holder)
        self.findings_layout.setContentsMargins(0, 8, 8, 0)
        self.findings_layout.setSpacing(8)
        self.findings_layout.addStretch()
        scroll.setWidget(holder)
        return scroll

    def _build_figures(self) -> QWidget:
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        self.figure_list = QListWidget()
        self.figure_list.setObjectName("figure_list")
        self.figure_list.setFixedWidth(200)
        self.tabs = QTabWidget()
        self.tabs.tabBar().hide()
        self.plots: dict[str, PlotCanvas] = {}
        # One entry per registered figure. Adding a figure to the registry adds
        # its entry and its exported PNG at once, so the two cannot drift.
        for spec in FIGURES:
            # Start each canvas at the size its figure was designed for. Only
            # the visible one gets resized by Qt, so the others would otherwise
            # draw at the 5x4in constructor default -- on which a three-panel
            # figure with rotated tick labels has no room left for the axes
            # themselves, and constrained layout gives up with a warning.
            c = PlotCanvas(figsize=spec.size_for(None, ""))
            c.setObjectName(f"figure_{spec.key}")
            c.message("no report yet -- solve, then Build report")
            c.enable_hover()
            self.plots[spec.key] = c
            self.tabs.addTab(c, spec.title)
            self.figure_list.addItem(spec.title)
        self.figure_list.currentRowChanged.connect(self.tabs.setCurrentIndex)
        self.figure_list.setCurrentRow(0)
        lay.addWidget(self.figure_list)
        lay.addWidget(self.tabs, 1)
        return page

    def _build_text(self) -> QWidget:
        self.summary_text = QPlainTextEdit()
        self.summary_text.setObjectName("report_text")
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        return self.summary_text

    def show_page(self, name: str) -> None:
        """Switch to one of PAGES by name (also used by the demo-video director)."""
        self.switch.set_index(PAGES.index(name))

    def show_figure(self, key: str) -> None:
        self.figure_list.setCurrentRow([s.key for s in FIGURES].index(key))

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
        self.switch.set_index(0)
        self.status.emit("report built")

    def _on_failed(self, message):
        self.build_btn.setEnabled(True)
        QMessageBox.critical(self, "Report failed", message)
        self.status.emit("report failed")

    def on_theme_changed(self) -> None:
        self.banner.restyle()
        if self.session.report is not None:
            self._redraw()

    # ------------------------------------------------------------------
    def _redraw(self):
        report = self.session.report
        if report is None:
            return
        cam = self.camera_combo.currentText() or next(iter(self.session.system.cameras))
        for spec in FIGURES:
            canvas = self.plots[spec.key]
            with theme.mpl_context(canvas.figure):
                fig = canvas.clear()
                try:
                    spec.draw(fig, report, self.session.system, cam)
                except Exception as exc:  # one broken figure must not blank the rest
                    from mlti_cal.report.figures import message

                    self.log.warning(f"figure {spec.key!r} failed to draw: {exc}")
                    message(fig, f"{spec.key} could not be drawn:\n{exc}")
                canvas.draw()
        self._render_overview(report)
        self._render_findings(report)
        self.summary_text.setPlainText(report.text_summary())

    def _show_empty(self) -> None:
        self.banner.set(
            "No report yet",
            "Solve first, then press Build report.",
            "neutral",
        )
        for card in (self.rms_card, self.heldout_card, self.worst_card, self.coverage_card):
            card.set("--", "")

    def _render_overview(self, report) -> None:
        warnings = report.warnings
        critical = [w for w in warnings if w.severity == "critical"]
        to_check = [w for w in warnings if w.severity == "warning"]
        rms = report.residuals.overall_rms if report.residuals else math.nan
        if critical:
            self.banner.set(
                "Do not use this calibration yet",
                critical[0].message
                + (f"  ({len(critical)} problems)" if len(critical) > 1 else ""),
                "bad",
            )
        elif to_check:
            topics = len({w.code for w in to_check})
            self.banner.set(
                f"Usable, with {topics} finding{'s' if topics != 1 else ''} to check",
                f"{rms:.3f} px per corner. Findings are listed most important first.",
                "warn",
            )
        else:
            self.banner.set(
                "Looks good",
                f"{rms:.3f} px per corner. No check raised a concern.",
                "good",
            )

        gt = report.gt_check
        if gt:
            rchi2 = gt.get("reduced_chi2", float("nan"))
            honest = 0.5 < rchi2 < 2.0
            self.gt_label.setText(
                f"Ground-truth check: reduced chi-squared {rchi2:.3f} (want about 1.0) -- "
                + (
                    "the reported uncertainty is honest."
                    if honest
                    else "THE REPORTED UNCERTAINTY IS NOT TRUSTWORTHY."
                )
            )
        self.gt_label.setVisible(bool(gt))

        solve = report.solve or {}
        start = solve.get("initial_rms_px")
        self.rms_card.set(
            f"{rms:.3f} px",
            f"per corner, {start:.3f} at the start" if start else "per corner",
            tone="good" if not critical else "bad",
        )
        cv = report.crossval
        if cv and not math.isnan(cv.get("optimism_ratio", math.nan)):
            ratio = cv["optimism_ratio"]
            tone = "good" if ratio < 1.2 else ("warn" if ratio < 1.5 else "bad")
            self.heldout_card.set(
                f"{ratio:.2f}× training",
                f"{cv['mean_test_rms_px']:.3f} px on unseen frames",
                tone,
            )
        else:
            self.heldout_card.set("not measured", "run Cross-validate on the Solve page")

        per_cam = report.residuals.per_camera if report.residuals else {}
        if per_cam:
            worst = max(per_cam, key=lambda c: per_cam[c]["rms_px"])
            best = min(per_cam[c]["rms_px"] for c in per_cam)
            ratio = per_cam[worst]["rms_px"] / best if best > 0 else 1.0
            self.worst_card.set(
                worst,
                f"{per_cam[worst]['rms_px']:.3f} px, {ratio:.1f}× the best",
                tone="warn" if ratio > 2.0 else None,
            )
        coverage = report.coverage.per_camera if report.coverage else {}
        if coverage:
            low = min(coverage, key=lambda c: coverage[c].occupied_fraction)
            frac = coverage[low].occupied_fraction
            self.coverage_card.set(
                f"{frac * 100:.0f}%",
                f"of the image in {low}, the least covered",
                tone="good" if frac > 0.7 else "warn",
            )
        self._fill_camera_table(per_cam, coverage)

    def _fill_camera_table(self, per_cam: dict, coverage: dict) -> None:
        cams = list(self.session.system.cameras.items())
        self.camera_table.setRowCount(len(cams))
        # Sized to its rows (up to eight), not stretched over the page.
        row_h = self.camera_table.verticalHeader().defaultSectionSize()
        header_h = self.camera_table.horizontalHeader().sizeHint().height()
        self.camera_table.setFixedHeight(header_h + row_h * min(len(cams), 8) + 4)
        top = max((v["rms_px"] for v in per_cam.values()), default=1.0) or 1.0
        for r, (cid, cam) in enumerate(cams):
            stats = per_cam.get(cid, {})
            cells = [
                cid,
                MODEL_LABELS.get(cam.model_name, cam.model_name),
                str(stats.get("n", "")),
                f"{stats.get('rms_px', math.nan):.3f}",
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                side = Qt.AlignmentFlag.AlignLeft if c < 2 else Qt.AlignmentFlag.AlignRight
                item.setTextAlignment(side | Qt.AlignmentFlag.AlignVCenter)
                self.camera_table.setItem(r, c, item)
            cov = coverage.get(cid)
            bar = QProgressBar()
            bar.setProperty("role", "bar")
            bar.setRange(0, 100)
            bar.setValue(int(round(100 * cov.occupied_fraction)) if cov else 0)
            bar.setFormat(f"{bar.value()}%")
            bar.setTextVisible(False)
            bar.setToolTip(f"{bar.value()}% of the image grid has observations")
            holder = QWidget()
            hl = QHBoxLayout(holder)
            hl.setContentsMargins(10, 0, 10, 0)
            hl.addWidget(bar, 1)
            hl.addWidget(QLabel(f"{bar.value()}%"))
            self.camera_table.setCellWidget(r, 4, holder)
            # The RMS cell carries the share of the worst camera as its tooltip.
            self.camera_table.item(r, 3).setToolTip(
                f"{stats.get('rms_px', math.nan) / top:.0%} of the worst camera's error"
            )

    def _render_findings(self, report) -> None:
        while self.findings_layout.count() > 1:
            item = self.findings_layout.takeAt(0)
            if item.widget() is not None:
                _discard(item.widget())
        groups: dict[str, list] = {}
        for w in report.warnings:
            groups.setdefault(w.code, []).append(w)
        ordered = sorted(
            groups.items(),
            key=lambda kv: min(SEVERITY_ORDER.get(w.severity, 3) for w in kv[1]),
        )
        for i, (code, items) in enumerate(ordered):
            self.findings_layout.insertWidget(i, FindingCard(code, items))
        n = len([c for c, items in groups.items() if any(w.severity != "info" for w in items)])
        self.switch.set_label(1, f"Findings ({n})" if n else "Findings")
        self._fill_top_findings(ordered[:3])

    def _fill_top_findings(self, ordered: list) -> None:
        while self.top_findings_rows.count():
            item = self.top_findings_rows.takeAt(0)
            if item.widget() is not None:
                _discard(item.widget())
        shown = [(code, items) for code, items in ordered if code != "no_issues"]
        for code, items in shown:
            severity = min((w.severity for w in items), key=lambda s: SEVERITY_ORDER.get(s, 3))
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            mark = QLabel()
            mark.setPixmap(dot_icon(_severity_colour(severity), 6).pixmap(12, 12))
            rl.addWidget(mark)
            title = FINDING_TITLES.get(code, items[0].message)
            action = next((w.action for w in items if w.action), "")
            text = QLabel(
                f"<b>{title}</b>"
                + (
                    f"  <span style='color:{theme.current().muted}'>{action}</span>"
                    if action
                    else ""
                )
            )
            text.setWordWrap(True)
            rl.addWidget(text, 1)
            self.top_findings_rows.addWidget(row)
        self.top_findings.setVisible(bool(shown))

    # ------------------------------------------------------------------
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
        # saving it produced one 34-inch-wide PNG per figure, each a different
        # shape, with the content marooned in a corner. Re-render instead, at
        # the size the figure was designed for -- and in the default (light)
        # style, whatever the window looks like.
        cam = self.camera_combo.currentText() or next(iter(self.session.system.cameras))
        for spec in FIGURES:
            try:
                fig = render(spec, self.session.report, self.session.system, cam)
                fig.savefig(out / f"{spec.key}.png", dpi=150)
            except Exception as exc:
                self.log.warning(f"figure {spec.key!r} not exported: {exc}")
        self.status.emit(f"exported to {out}")
        QMessageBox.information(self, "Exported", f"Written to {out}")
