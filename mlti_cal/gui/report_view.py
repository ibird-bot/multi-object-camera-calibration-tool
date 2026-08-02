"""
Window 3 -- Report.

The payoff view: projection-uncertainty heatmap, residual quiver field,
coverage, normality, error-vs-radius, the warnings panel in plain language,
and (synthetic only) the ground-truth honesty verdict.

Nothing is computed here. Every figure reads a `CalibrationReport` produced by
the headless engine.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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

from mlti_cal.gui.session import ReportWorker, Session, run_in_thread
from mlti_cal.gui.widgets import PlotCanvas
from mlti_cal.io.config import export_json, export_opencv_yaml

SEVERITY_COLOUR = {"critical": "#c0392b", "warning": "#c87f0a", "info": "#2d7d46"}


class ReportView(QWidget):
    status = Signal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
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
            "Projection uncertainty genuinely depends on range -- focal length "
            "and camera position trade off differently at different depths. "
            "There is no single correct value, which is why it must be stated."
        )
        bar.addWidget(self.range_spin)

        self.crossval_check = QCheckBox("cross-validate (slow)")
        self.crossval_check.setToolTip(
            "Hold out whole frames, re-fit only their board poses with the "
            "camera parameters frozen, and measure reprojection there. This is "
            "the honest generalisation error; training RMS is optimistic."
        )
        bar.addWidget(self.crossval_check)

        bar.addWidget(QLabel("camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._redraw)
        bar.addWidget(self.camera_combo)

        bar.addStretch()
        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self._export)
        bar.addWidget(export_btn)
        root.addLayout(bar)

        split = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.plots: dict[str, PlotCanvas] = {}
        for key, title in (
            ("uncertainty", "Projection uncertainty"),
            ("quiver", "Residual field"),
            ("coverage", "Coverage"),
            ("distribution", "Residual distribution"),
            ("radius", "Error vs radius"),
        ):
            c = PlotCanvas()
            c.message("no report yet -- solve, then Build report")
            self.plots[key] = c
            self.tabs.addTab(c, title)
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
        if self.session.problem is None:
            QMessageBox.information(self, "Nothing to report", "Solve first.")
            return
        self.build_btn.setEnabled(False)
        worker = ReportWorker(
            self.session, self.range_spin.value(), self.crossval_check.isChecked()
        )
        run_in_thread(self, worker, self._on_report, self._on_failed, self.status.emit)

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
        self._plot_uncertainty(report, cam)
        self._plot_quiver(report, cam)
        self._plot_coverage(report, cam)
        self._plot_distribution(report)
        self._plot_radius(report)
        self._render_warnings(report)
        self.summary_text.setPlainText(report.text_summary())

    def _plot_uncertainty(self, report, cam):
        canvas = self.plots["uncertainty"]
        m = report.uncertainty_maps.get(cam)
        if m is None:
            canvas.message(f"no uncertainty map for {cam}")
            return
        fig = canvas.clear()
        ax = fig.add_subplot(111)
        w, h = self.session.system.cameras[cam].image_size
        im = ax.imshow(
            m.sigma_max,
            origin="upper",
            extent=[0, w, h, 0],
            cmap="viridis",
            interpolation="bilinear",
        )
        fig.colorbar(im, ax=ax, label="1-sigma projection error (px)")
        title = (
            f"{cam} @ {m.range_m:g} m -- centre {m.at_centre():.3f} px, worst {m.worst():.3f} px"
        )
        if m.invalid_fraction > 0.001:
            title += f"\n{m.invalid_fraction * 100:.0f}% masked: distortion not invertible there"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        canvas.draw()

    def _plot_quiver(self, report, cam):
        from mlti_cal.report.residuals import quiver_field

        canvas = self.plots["quiver"]
        if report.residuals is None or report.residuals.errors.size == 0:
            canvas.message("no residuals")
            return
        q = quiver_field(report.residuals, cam)
        if not q["x"]:
            canvas.message(f"no residuals for {cam}")
            return
        fig = canvas.clear()
        ax = fig.add_subplot(111)
        w, h = self.session.system.cameras[cam].image_size
        mag = np.array(q["magnitude"])
        sc = ax.quiver(
            q["x"], q["y"], q["u"], q["v"], mag, cmap="plasma", angles="xy", width=0.0025
        )
        fig.colorbar(sc, ax=ax, label="error (px)")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(
            f"{cam} residual field (arrows exaggerated by autoscale)\n"
            f"structure here = model inadequacy that RMS cannot see",
            fontsize=9,
        )
        canvas.draw()

    def _plot_coverage(self, report, cam):
        canvas = self.plots["coverage"]
        if report.coverage is None or cam not in report.coverage.per_camera:
            canvas.message("no coverage data")
            return
        cs = report.coverage.per_camera[cam]
        fig = canvas.clear()
        ax = fig.add_subplot(121)
        im = ax.imshow(cs.histogram, cmap="magma", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="corners per cell")
        ax.set_title(f"{cam}: {cs.occupied_fraction * 100:.0f}% of cells occupied", fontsize=9)
        ax2 = fig.add_subplot(122)
        tilt = report.coverage.tilt
        if tilt is not None and tilt.incidence_deg.size:
            ax2.hist(tilt.incidence_deg, bins=18, color="#3b7dd8")
            ax2.axvline(10, color="crimson", ls="--", lw=1)
            ax2.set_xlabel("board incidence angle (deg)")
            ax2.set_title(
                f"tilt diversity -- {tilt.fraction_below_10deg * 100:.0f}% under 10 deg",
                fontsize=9,
            )
        canvas.draw()

    def _plot_distribution(self, report):
        from mlti_cal.report.residuals import qq_data

        canvas = self.plots["distribution"]
        if report.residuals is None or report.residuals.errors.size == 0:
            canvas.message("no residuals")
            return
        fig = canvas.clear()
        ax = fig.add_subplot(121)
        ax.hist(report.residuals.vectors.ravel(), bins=60, color="#3b7dd8")
        n = report.normality
        ax.set_title(
            f"residual components\nskew {n.get('skew', float('nan')):.2f}, "
            f"excess kurtosis {n.get('excess_kurtosis', float('nan')):.2f}",
            fontsize=9,
        )
        ax.set_xlabel("px")
        ax2 = fig.add_subplot(122)
        q = qq_data(report.residuals)
        if q["theoretical"]:
            ax2.plot(q["theoretical"], q["observed"], ".", ms=2)
            lim = max(abs(min(q["theoretical"])), abs(max(q["theoretical"])))
            ax2.plot([-lim, lim], [-lim, lim], "r--", lw=1)
        ax2.set_title("Q-Q vs normal (curvature = heavy tails)", fontsize=9)
        ax2.set_xlabel("theoretical")
        ax2.set_ylabel("observed")
        canvas.draw()

    def _plot_radius(self, report):
        canvas = self.plots["radius"]
        evr = report.error_vs_radius
        if not evr.get("bin_centres"):
            canvas.message("no data")
            return
        fig = canvas.clear()
        ax = fig.add_subplot(111)
        ax.plot(evr["bin_centres"], evr["rms_px"], "o-")
        ax.set_xlabel("radius from principal point (px)")
        ax.set_ylabel("RMS error (px)")
        ax.set_title(
            "error vs radius -- growth toward the edge means the distortion "
            "model cannot represent the lens",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        canvas.draw()

    def _render_warnings(self, report):
        parts = []
        gt = report.gt_check
        if gt:
            rchi2 = gt.get("reduced_chi2", float("nan"))
            good = 0.5 < rchi2 < 2.0
            parts.append(
                f"<div style='padding:4px;background:{'#e6f4ea' if good else '#fdecea'};'>"
                f"<b>Ground-truth check:</b> reduced chi-squared "
                f"<b>{rchi2:.3f}</b> (want ~1.0) &mdash; "
                f"{'the reported uncertainty is honest' if good else 'THE REPORTED UNCERTAINTY IS NOT TRUSTWORTHY'}"
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
        for key, canvas in self.plots.items():
            canvas.figure.savefig(out / f"{key}.png", dpi=150)
        self.status.emit(f"exported to {out}")
        QMessageBox.information(self, "Exported", f"Written to {out}")
