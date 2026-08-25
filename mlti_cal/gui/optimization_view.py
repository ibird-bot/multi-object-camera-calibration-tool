"""
Window 2 -- Optimization.

Three things live here:

  * the parameter tree, where every quantity is Free / Fixed / Deactivated.
    This maps straight onto `ParameterBlock.free_mask`, so what the tree shows
    is literally which columns exist in the Jacobian.
  * the backend picker with its option controls, built from
    `solvers/catalog.py` -- the same source the CLI prints, with each option's
    "when to use this" text as the tooltip.
  * a live progress plot, driven from a worker thread.
"""

from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mlti_cal.gui.log_console import with_source
from mlti_cal.gui.session import (
    CrossValWorker,
    InitWorker,
    Session,
    SolveWorker,
    refuse_if_busy,
    run_in_thread,
)
from mlti_cal.gui.settings_panel import (
    fill_form,
    make_option_widget,
    option_tooltip,
)
from mlti_cal.gui.widgets import PlotCanvas
from mlti_cal.problem.initialize import unsupported_pins
from mlti_cal.problem.loss import available_losses
from mlti_cal.problem.reprojection import extr_key, intr_key, pose_key
from mlti_cal.problem.settings import INIT_CATALOG, INIT_SUMMARY, InitSettings
from mlti_cal.report.settings import REPORT_CATALOG
from mlti_cal.solvers import SolveOptions, backend_status
from mlti_cal.solvers.base import IterationRecord, per_corner_rms
from mlti_cal.solvers.catalog import CATALOG, COMMON_OPTIONS, options_for

#: The fold count, taken from the report catalog rather than redeclared, so the
#: spinbox next to the button carries the same bounds and guidance as the one in
#: the Report tab and the `mlti-cal settings report` output.
CROSSVAL_FOLDS_OPTION = next(o for o in REPORT_CATALOG if o.name == "crossval_folds")

#: Longest per-evaluation table printed after a solve. Longer histories are
#: decimated: the shape of the descent is the point, not every row.
MAX_ITERATION_LINES = 30
#: Minimum gap between live repaints while a solve runs. Steps arrive faster
#: than a window can draw them, so this bounds the work rather than the rate --
#: without it the queued repaints outlive the solve and the "live" view finishes
#: after the solve does.
LIVE_REFRESH_SECONDS = 0.2


class OptimizationView(QWidget):
    solved = Signal()
    status = Signal(str)

    def __init__(self, session: Session, console=None, parent=None):
        super().__init__(parent)
        self.session = session
        # The transcript is shared with every other tab. A private console is
        # made when none is given so this view can still be built standalone.
        from mlti_cal.gui.log_console import LogConsole, SourceLog

        self.console = console if console is not None else LogConsole()
        self.log = SourceLog(self.console, "solve")
        self._option_widgets: dict[str, QWidget] = {}
        #: (camera id, parameter index) -> value the USER typed. Pinned means
        #: known, not estimated: held through the bootstrap and the solve.
        self._pinned: dict[tuple[str, int], float] = {}
        #: (camera id, parameter index) the user switched off: forced to zero.
        self._inactive: set[tuple[str, int]] = set()
        #: The tree is a VIEW of the two above. Reading state back out of
        #: widgets loses it on every rebuild, which is how a "fixed" parameter
        #: silently becomes free again after a solve repopulates the tree.
        self._value_edits: dict[tuple[str, int], QLineEdit] = {}
        self._populating = False
        #: Records streamed by the solve currently running. Reset per solve, so
        #: the fallback table cannot double-print the second solve of a session.
        self._live: list = []
        self._last_live_paint = 0.0
        #: Bootstrap settings as a plain dict, edited in their own window and
        #: only replaced when the user applies. Held here rather than read off
        #: widgets so a half-typed value is never briefly the live setting.
        self._init_values: dict = InitSettings().to_dict()
        self._build()
        self._refresh_init_button()

    def _build(self):
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)

        # ---------------- left: parameters -----------------------------
        left = QWidget()
        lv = QVBoxLayout(left)
        head = QHBoxLayout()
        head.addWidget(
            QLabel("<b>Parameters</b> -- type a value to hold it, untick to switch it off")
        )
        head.addStretch()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip(
            "Drop every value you typed and switch every parameter back on.\nEstimated values stay."
        )
        self.clear_btn.clicked.connect(self._clear_values)
        head.addWidget(self.clear_btn)
        lv.addLayout(head)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["parameter", "value", "active"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        lv.addWidget(self.tree)
        note = QLabel(
            "Empty: estimated. Number: held fixed, in the estimate and the solve. "
            "Untick: forced to zero -- not available for fx, fy, cx, cy, without "
            "which a camera cannot project. The reference extrinsic is always "
            "fixed; it is the gauge."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        lv.addWidget(note)
        split.addWidget(left)

        # ---------------- right: solver --------------------------------
        right = QWidget()
        rv = QVBoxLayout(right)

        solver_box = QGroupBox("Solver")
        sf = QFormLayout(solver_box)
        self.backend_combo = QComboBox()
        for name, (ok, why) in backend_status().items():
            self.backend_combo.addItem(f"{name}{'' if ok else '  (unavailable)'}", name)
            idx = self.backend_combo.count() - 1
            self.backend_combo.setItemData(
                idx, CATALOG.get(name, {}).get("summary", "") if ok else why, Qt.ToolTipRole
            )
            if not ok:
                self.backend_combo.model().item(idx).setEnabled(False)
        self.backend_combo.currentIndexChanged.connect(self._rebuild_options)
        sf.addRow("backend", self.backend_combo)

        self.iterations = QSpinBox()
        self.iterations.setRange(1, 100000)
        self.iterations.setValue(300)
        sf.addRow("max iterations", self.iterations)

        self.loss_combo = QComboBox()
        self.loss_combo.addItems(available_losses())
        # Explicit, because `available_losses()` is sorted alphabetically and
        # "cauchy" wins that sort: the form otherwise defaults every solve to a
        # hard robust loss at 2 px. Cauchy's weighted cost SATURATES, so a
        # corner 100 px out costs the same as one 6 px out, and the solver can
        # lower the reported cost by abandoning a chunk of the geometry --
        # a solve that looks converged and is wrong. Robustness is opt-in.
        self.loss_combo.setCurrentText("trivial")
        self.loss_combo.setToolTip(
            "Per-corner IRLS weights, applied in the core so every backend "
            "optimises the same objective. Not Ceres' Triggs correction.\n\n"
            "trivial -- least squares. Start here.\n"
            "huber -- outliers cost less, never nothing.\n"
            "cauchy/soft_l1 -- outliers nearly ignored; the cost then stops "
            "measuring fit quality."
        )
        sf.addRow("robust loss", self.loss_combo)
        self.loss_scale = QDoubleSpinBox()
        self.loss_scale.setRange(0.1, 100.0)
        self.loss_scale.setValue(2.0)
        self.loss_scale.setToolTip("Loss threshold/scale in PIXELS.")
        sf.addRow("loss scale (px)", self.loss_scale)
        rv.addWidget(solver_box)

        self.options_box = QGroupBox("Backend options")
        self.options_form = QFormLayout(self.options_box)
        rv.addWidget(self.options_box)

        # ---- starting point -------------------------------------------
        init_box = QGroupBox("Starting point")
        iv = QVBoxLayout(init_box)
        ib = QHBoxLayout()
        self.init_btn = QPushButton("Estimate starting point")
        self.init_btn.setToolTip(
            "Intrinsics via cv2.calibrateCamera, poses via cv2.solvePnP, "
            "extrinsics from boards seen by two cameras at once.\n"
            "Nothing is estimated until you click this."
        )
        self.init_btn.clicked.connect(self._estimate)
        self.restore_btn = QPushButton("Revert to estimate")
        self.restore_btn.setToolTip(
            "Put the saved starting values back, discarding the solve's output."
        )
        self.restore_btn.clicked.connect(self._restore_initial)
        self.init_settings_btn = QPushButton("Settings...")
        self.init_settings_btn.setToolTip(
            "PnP method, RANSAC, point and view floors, averaging of repeated "
            "estimates. Nothing changes until you apply."
        )
        self.init_settings_btn.clicked.connect(self.open_init_settings)
        ib.addWidget(self.init_btn)
        ib.addWidget(self.restore_btn)
        ib.addWidget(self.init_settings_btn)
        ib.addStretch()
        iv.addLayout(ib)
        self.init_summary = QLabel()
        self.init_summary.setWordWrap(True)
        self.init_summary.setTextFormat(Qt.RichText)
        iv.addWidget(self.init_summary)
        rv.addWidget(init_box)

        btns = QHBoxLayout()
        self.solve_btn = QPushButton("Solve")
        self.solve_btn.clicked.connect(self._solve)
        self.build_btn = QPushButton("Build problem")
        self.build_btn.setToolTip(
            "Assemble the problem from the current tree and loss without solving, "
            "and print it to the log. Solve does this for you; this is for "
            "looking first."
        )
        self.build_btn.clicked.connect(self._build_problem)
        self.crossval_btn = QPushButton("Cross-validate")
        self.crossval_btn.setToolTip(
            "Hold out whole frames, re-solve on the rest, measure reprojection "
            "on the frames the fit never saw.\n\n"
            "The only error here not measured on its own training data. "
            "Training RMS always drops when you add parameters; this does "
            "not.\n\n"
            "One full solve per fold. Result goes to the log; your parameters "
            "are untouched."
        )
        self.crossval_btn.clicked.connect(self._cross_validate)
        self.folds_spin = make_option_widget(CROSSVAL_FOLDS_OPTION)
        self.folds_spin.setToolTip(option_tooltip(CROSSVAL_FOLDS_OPTION))
        btns.addWidget(self.solve_btn)
        btns.addWidget(self.build_btn)
        btns.addWidget(self.crossval_btn)
        btns.addWidget(QLabel("folds"))
        btns.addWidget(self.folds_spin)
        btns.addStretch()
        rv.addLayout(btns)

        self.progress_plot = PlotCanvas(figsize=(4, 2.4), toolbar=False)
        self.progress_plot.message("no solve yet")
        rv.addWidget(self.progress_plot, 1)

        # No log widget here any more: it is the shared console at the bottom
        # of the window, visible from every tab. The progress plot takes the
        # space it used to occupy.
        split.addWidget(right)
        split.setSizes([560, 640])

        self._rebuild_options()

    # ------------------------------------------------------------------
    def _rebuild_options(self):
        while self.options_form.rowCount():
            self.options_form.removeRow(0)
        self._option_widgets.clear()
        backend = self.backend_combo.currentData()
        if backend not in CATALOG:
            return
        # `options_for` returns the backend's own options plus only those COMMON
        # options this backend actually reads -- scipy has no thread count, for
        # instance, so rendering one beside it would show a control that
        # silently does nothing.
        #
        # One shared builder, so bounds and decimal places come from the catalog
        # rather than being retyped here. The old inline version fixed every
        # float spinbox at 2 decimals, which rounded a 1e-10 tolerance to zero
        # and made scipy reject every tolerance it was given.
        self._option_widgets = fill_form(
            self.options_form,
            options_for(backend),
            skip=("max_iterations",),  # already exposed above
        )
        self.options_box.setTitle(f"Options for {backend}")
        self.options_box.setToolTip(CATALOG[backend]["summary"])

    def _collect_options(self) -> dict:
        out = {}
        for name, w in self._option_widgets.items():
            if isinstance(w, QComboBox):
                out[name] = w.currentText()
            elif isinstance(w, QCheckBox):
                out[name] = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                out[name] = w.value()
        return out

    def _solve_options(self) -> SolveOptions:
        """
        Widget values as a SolveOptions.

        The common options are FIELDS on SolveOptions, not `extra` keys -- both
        adapters read them from the fields, so leaving them in `extra` would
        render four spinboxes that silently do nothing.
        """
        values = self._collect_options()
        common = {o.name: values.pop(o.name) for o in COMMON_OPTIONS if o.name in values}
        return SolveOptions(
            max_iterations=self.iterations.value(),
            function_tolerance=float(common.get("function_tolerance", 1e-10)),
            gradient_tolerance=float(common.get("gradient_tolerance", 1e-12)),
            parameter_tolerance=float(common.get("parameter_tolerance", 1e-10)),
            num_threads=int(common.get("num_threads", 1)),
            extra=values,
        )

    # ------------------------------------------------------------------
    def _cross_validate(self):
        """
        Held-out error, on demand, printed into the log.

        Requires a solved (or at least initialised) system: cross-validating
        default parameters would measure how badly an unfitted model
        generalises, which is a number about nothing.
        """
        if refuse_if_busy(self, "cross-validation"):
            return
        if not self.session.is_ready_to_solve:
            QMessageBox.information(self, "No data", "Run detection first.")
            return
        if not self.session.is_initialized:
            QMessageBox.information(
                self,
                "No starting point",
                "Estimate the starting point first -- each fold re-solves from "
                "the current values, and the defaults do not converge.",
            )
            return
        if not self._warn_if_detections_are_stale("cross-validation"):
            return

        folds = self.folds_spin.value()
        n_frames = len(self.session.system.frames)
        if n_frames < folds + 1:
            QMessageBox.information(
                self,
                "Not enough frames",
                f"{folds}-fold cross-validation needs at least {folds + 1} frames; "
                f"this session has {n_frames}.\n\n"
                f"Lower the fold count or capture more frames.",
            )
            return

        backend = self.backend_combo.currentData()
        ok, why = backend_status()[backend]
        if not ok:
            QMessageBox.warning(self, "Backend unavailable", why)
            return

        self.crossval_btn.setEnabled(False)
        with_source(self.log, "crossval")
        self.log.appendPlainText(
            f"--- cross-validation: {folds} folds, {backend}, "
            f"{n_frames} frames ({folds} full re-solves) ---"
        )
        self.log.appendPlainText(
            f"{'fold':>4}  {'train':>6}  {'test':>5}  {'train RMS':>10}  {'test RMS':>9}"
        )
        worker = CrossValWorker(
            self.session.system,
            backend=backend,
            folds=folds,
            max_iterations=self.iterations.value(),
        )
        # Bound methods of this widget, never lambdas -- see run_in_thread.
        worker.fold_done.connect(self._on_fold_done)
        run_in_thread(self, worker, self._on_crossval, self._on_crossval_failed, self._log)

    def _on_fold_done(self, fold) -> None:
        """One row per fold, as it lands rather than all at the end."""
        self.log.appendPlainText(
            f"{fold.fold:>4}  {fold.num_train_frames:>6}  {fold.num_test_frames:>5}  "
            f"{fold.train_rms_px:>10.4f}  {fold.test_rms_px:>9.4f}"
        )

    def _on_crossval(self, result) -> None:
        self.crossval_btn.setEnabled(True)
        self.session.crossval = result
        for line in result.summary_lines():
            self.log.appendPlainText(line)
        self.status.emit(
            f"cross-validation: held-out RMS {result.mean_test_rms:.4f} px "
            f"vs training {result.mean_train_rms:.4f} px"
        )

    def _on_crossval_failed(self, message: str) -> None:
        self.crossval_btn.setEnabled(True)
        self.log.appendPlainText(f"cross-validation FAILED: {message}")
        QMessageBox.critical(self, "Cross-validation failed", message)
        self.status.emit("cross-validation failed")

    def _warn_if_detections_are_stale(self, action: str) -> bool:
        """
        Ask before computing on corners the current detector settings did not produce.

        Returning True means the user chose to continue anyway. A warning
        rather than a block: sometimes you genuinely want to see what the old
        detections give. Silence is the only unacceptable option, because the
        result would carry the new settings' label and the old settings' data.
        """
        if not self.session.detection_stale:
            return True
        answer = QMessageBox.warning(
            self,
            "Detections are out of date",
            "The detector settings were changed after these corners were found, "
            f"so {action} would use detections produced by the PREVIOUS settings.\n\n"
            "Re-run detection in the Setup tab, or continue and treat the result "
            "as belonging to the old settings.",
            QMessageBox.Cancel | QMessageBox.Ignore,
            QMessageBox.Cancel,
        )
        return answer == QMessageBox.Ignore

    def open_init_settings(self) -> None:
        """Bootstrap settings in their own window, applied explicitly."""
        from mlti_cal.gui.settings_dialog import CatalogSettingsDialog

        dialog = CatalogSettingsDialog(
            "Starting point settings",
            INIT_SUMMARY,
            INIT_CATALOG,
            values=self._init_values,
            # Built through InitSettings so its cross-field rules run here --
            # the geometric floors, and the fact that min cannot exceed max --
            # rather than at estimate time with the dialog already closed.
            build=lambda raw: InitSettings(**raw),
            parent=self,
        )
        dialog.applied.connect(self._on_init_settings_applied)
        dialog.exec()

    def _on_init_settings_applied(self, settings: InitSettings) -> None:
        values = settings.to_dict()
        if values != self._init_values:
            self._init_values = values
            with_source(self.log, "init")
            self.log.appendPlainText(f"starting point settings: {self._init_summary_line()}")
        self._refresh_init_button()

    def _init_summary_line(self) -> str:
        """Only what the user moved; "all defaults" when they moved nothing."""
        defaults = InitSettings().to_dict()
        changed = {k: v for k, v in self._init_values.items() if v != defaults[k]}
        return ", ".join(f"{k}={v}" for k, v in changed.items()) if changed else "all defaults"

    def _refresh_init_button(self) -> None:
        """Mark the button when the settings are no longer the defaults."""
        changed = self._init_summary_line() != "all defaults"
        self.init_settings_btn.setText("Settings*..." if changed else "Settings...")

    def _init_settings(self) -> InitSettings:
        """The applied bootstrap settings as an InitSettings."""
        return InitSettings(**self._init_values)

    # ------------------------------------------------------------------
    def log_line(self, message: str) -> None:
        """Append one line to the log box. Public: the other tabs report here."""
        self.log.appendPlainText(message)

    def _log(self, message: str) -> None:
        """One line, to the log box and the status bar both."""
        self.log_line(message)
        self.status.emit(message)

    def _log_pins(self, for_bootstrap: bool = False) -> dict[str, dict[int, float]]:
        """
        Say which values are being held before using them, and return them.

        `for_bootstrap` gates the OpenCV caveat, which belongs to the bootstrap
        alone: the solve never calls OpenCV, and printing it there says a tool
        is involved that is not.
        """
        pins = self._pins_for_bootstrap()
        for cid, per_cam in sorted(pins.items()):
            cam = self.session.system.cameras.get(cid)
            if cam is None:
                continue
            names = cam.model.param_names
            held = ", ".join(f"{names[i]}={v:g}" for i, v in sorted(per_cam.items()))
            suffix = "" if for_bootstrap else "  (no Jacobian column: the solver cannot move it)"
            self.log.appendPlainText(f"holding {cid}: {held}{suffix}")
            if not for_bootstrap:
                continue
            blind = unsupported_pins(cam.model_name, per_cam)
            if blind:
                # A real limitation, worth one line: the value is still forced,
                # but OpenCV had no flag to hold it DURING the fit, so the rest
                # of that camera's intrinsics were fitted against a value that
                # then changed under them.
                held_names = ", ".join(names[i] for i in blind)
                self.log.appendPlainText(
                    f"  note: OpenCV has no flag to hold {held_names} on its own, so the "
                    "bootstrap fitted the other intrinsics with it free and it was forced "
                    "back afterwards. The solve holds it properly."
                )
        return pins

    def _estimate(self):
        """Run the bootstrap, on demand, in a worker thread."""
        if refuse_if_busy(self, "the starting estimate"):
            return
        if not self.session.is_ready_to_solve:
            QMessageBox.information(self, "No data", "Run detection first.")
            return
        if not self._warn_if_detections_are_stale("the starting estimate"):
            return
        with_source(self.log, "init")
        self.log.appendPlainText("--- estimating starting point ---")
        self.log.appendPlainText(f"starting point settings: {self._init_summary_line()}")
        self._apply_user_values()
        pins = self._log_pins(for_bootstrap=True)
        self.init_btn.setEnabled(False)
        worker = InitWorker(
            self.session.system,
            fixed_intrinsics=pins or None,
            settings=self._init_settings(),
        )
        run_in_thread(self, worker, self._on_estimated, self._on_estimate_failed, self._log)

    def _on_estimated(self, report: dict):
        self.init_btn.setEnabled(True)
        try:
            self.session.capture_initial(report)
        except Exception as exc:  # a bootstrap that cannot be measured is not usable
            QMessageBox.critical(self, "Cannot measure the estimate", str(exc))
            self.session.clear_initial()
            self._update_init_summary()
            return
        self.log.appendPlainText(
            f"start: per-corner RMS {self.session.initial_rms:.3f} px, "
            f"{report.get('pnp_solved', 0)} PnP poses"
        )
        self.refresh()
        self._update_init_summary()
        self._log(f"starting point ready -- RMS {self.session.initial_rms:.3f} px")

    def _on_estimate_failed(self, message: str):
        self.init_btn.setEnabled(True)
        self.session.clear_initial()
        self._update_init_summary()
        self.log.appendPlainText(f"ESTIMATE FAILED: {message}")
        QMessageBox.critical(self, "Estimate failed", message)
        self.status.emit("starting estimate failed")

    def _restore_initial(self):
        n = self.session.restore_initial()
        if not n:
            QMessageBox.information(self, "No estimate", "Run the starting estimate first.")
            return
        self.refresh()
        self._log(f"reverted {n} block(s) to the starting estimate")

    def _update_init_summary(self):
        """Spell out the bootstrap's own quality signals, not just its RMS."""
        s = self.session
        self.restore_btn.setEnabled(s.is_initialized)
        if not s.is_initialized:
            self.init_summary.setText(
                "<span style='color:#b36b00;'><b>No starting point.</b> "
                "Click <i>Estimate starting point</i>.</span>"
            )
            return
        rep = s.initial_report or {}
        lines = [
            f"<b>Per-corner reprojection RMS: {s.initial_rms:.3f} px</b> "
            f"({rep.get('pnp_solved', 0)} PnP poses, {rep.get('board_poses', 0)} board poses)"
        ]
        # Per-camera OpenCV RMS: nan means too few views, so that camera still
        # holds default parameters and is NOT actually calibrated.
        for cid, rms in (rep.get("intrinsics_rms") or {}).items():
            if math.isnan(rms):
                lines.append(
                    f"<span style='color:#c00;'>{cid}: intrinsics NOT estimated "
                    f"(under 4 usable views) -- still at defaults</span>"
                )
            else:
                lines.append(f"{cid}: OpenCV intrinsics RMS {rms:.3f} px")
        # Extrinsic support = frames where this camera and the reference saw the
        # same board. Low support is the quiet failure the bootstrap warns about.
        for cid, n in (rep.get("extrinsic_support") or {}).items():
            if n < 0:
                lines.append(f"{cid}: reference camera (extrinsic fixed to identity)")
            elif n < 5:
                lines.append(
                    f"<span style='color:#c00;'>{cid}: extrinsic from only {n} shared "
                    f"view(s) -- weakly observable</span>"
                )
            else:
                lines.append(f"{cid}: extrinsic from {n} shared views")
        self.init_summary.setText("<br>".join(lines))

    def refresh(self) -> bool:
        """
        Rebuild the problem from the current tree state and repopulate.

        Returns True only when a problem was actually built. The callers that
        report on it need that: `session.problem` keeps its previous value when
        `build_problem` raises or when we bail out early, so "not None" does not
        mean "current", and summarising a stale problem as if it were fresh is
        exactly the lie this view exists to prevent.
        """
        self._update_init_summary()
        # Before anything is assembled, not only on Estimate and Solve:
        # `build_problem` copies cam.params into the blocks, so a held parameter
        # whose value had not been written yet would be frozen at the value it
        # is replacing -- the column correctly gone, the number silently wrong.
        self._apply_user_values()
        built = False
        # Without a starting point `build_problem` raises on the first missing
        # board pose, so the problem waits -- but the parameter tree does not.
        # It is populated either way: typing a value you already know is most
        # useful BEFORE the bootstrap, which is the one thing the old
        # clear-the-tree behaviour made impossible.
        if self.session.is_ready_to_solve and self.session.is_initialized:
            try:
                self.session.rebuild_problem(
                    loss_name=self.loss_combo.currentText(),
                    loss_scale=self.loss_scale.value(),
                    fixed_intrinsics=self._fixed_intrinsics() or None,
                )
                built = True
            except Exception as exc:
                QMessageBox.critical(self, "Cannot build problem", str(exc))
        self._populate_tree(self.session.problem if built else None)
        return built

    # ------------------------------------------------------------------
    def _build_problem(self):
        """
        The "Build problem" button: assemble, then say what was assembled.

        The numbers go to the log, not the status bar: they are the whole point
        of pressing this, and a status line cannot be compared against the
        previous build once it has been overwritten.
        """
        with_source(self.log, "problem")
        self.log.appendPlainText("--- build problem ---")
        if not self.refresh():
            if not self.session.is_ready_to_solve:
                why = "no data -- run detection first"
            elif not self.session.is_initialized:
                why = "no starting point -- click 'Estimate starting point' first"
            else:
                why = "build failed, see the dialog"
            self.log.appendPlainText(f"not built: {why}")
            return
        for line in self._problem_lines():
            self.log.appendPlainText(line)

    def _problem_lines(self) -> list[str]:
        """Size of the problem, what is free in it, and what is being held."""
        system = self.session.system
        problem = self.session.problem
        corners = sum(r.num_points for r in problem.residuals)
        n_res, n_free = problem.num_residuals, problem.num_free_params

        intr_free = intr_total = 0
        held: list[str] = []
        for cid, cam in system.cameras.items():
            blk = problem.blocks.get(intr_key(cid))
            if blk is None:
                continue
            intr_free += blk.num_free
            intr_total += blk.tangent_size
            if blk.constant:
                held.append(f"{cid}: all")
                continue
            # Values, not just names. "Deactivated" is only zeroed on Solve, so
            # at build time such a component still carries its old value and is
            # in the residual with it -- printing it stops that being a surprise.
            names = [
                f"{name}={blk.value[i]:.6g}"
                for i, name in enumerate(cam.model.param_names)
                if not blk.free_mask[i]
            ]
            if names:
                held.append(f"{cid}: {', '.join(names)}")

        extr_free = extr_total = 0
        extr_held: list[str] = []
        for cid, cam in system.cameras.items():
            blk = problem.blocks.get(extr_key(cid))
            if blk is None:
                continue
            extr_free += blk.num_free
            extr_total += blk.tangent_size
            if blk.num_free == 0:
                extr_held.append(f"{cid} (gauge)" if cam.is_reference else cid)

        poses = [
            problem.blocks[pose_key(f, b)]
            for f, b in system.frame_board_pairs
            if pose_key(f, b) in problem.blocks
        ]
        pose_free = sum(b.num_free for b in poses)

        intr_line = f"  intrinsics: {intr_free} of {intr_total} free"
        if held:
            intr_line += "   held: " + "; ".join(held)
        extr_line = f"  extrinsics: {extr_free} of {extr_total} free"
        if extr_held:
            extr_line += "   fixed: " + ", ".join(extr_held)

        lines = [
            f"data        : cameras {len(system.cameras)} | frames {len(system.frames)} | "
            f"boards {len(system.boards)} | views {len(problem.residuals)} | corners {corners}",
            f"residuals   : {n_res} rows  ({corners} corners x 2)",
            f"free params : {n_free} columns",
            intr_line,
            extr_line,
            f"  poses     : {pose_free} of {6 * len(poses)} free  ({len(poses)} x 6 DOF)",
            f"loss        : {self.loss_combo.currentText()}, scale {self.loss_scale.value():g} px",
            f"RMS now     : {per_corner_rms(problem):.3f} px per corner (unweighted)",
        ]
        if n_free == 0:
            lines.append("WARNING: nothing is free -- Solve has nothing to estimate")
        elif n_res <= n_free:
            lines.append(
                f"WARNING: {n_res} rows for {n_free} columns -- underdetermined, "
                "the solution will not be unique"
            )
        else:
            lines.append(f"redundancy  : {n_res - n_free} rows more than columns")
        return lines

    # ---- parameter tree ----------------------------------------------
    def _fixed_intrinsics(self) -> dict[str, list[int]]:
        """Camera id -> indices with no Jacobian column: pinned or switched off."""
        out: dict[str, list[int]] = {}
        for cid, idx in (*self._pinned, *self._inactive):
            out.setdefault(cid, []).append(idx)
        return {cid: sorted(set(v)) for cid, v in out.items()}

    def _pins_for_bootstrap(self) -> dict[str, dict[int, float]]:
        """Camera id -> {index: value} the bootstrap must not estimate."""
        out: dict[str, dict[int, float]] = {}
        for (cid, idx), value in self._pinned.items():
            out.setdefault(cid, {})[idx] = value
        for cid, idx in self._inactive:
            out.setdefault(cid, {})[idx] = 0.0
        return out

    def _apply_user_values(self) -> None:
        """
        Push what the user typed and unticked into the live system.

        Called before every estimate and every solve, so the values in the
        system are the ones the tree shows -- an unticked parameter that is
        still carrying its old value would be reported as "0" by the UI and
        used as non-zero by the maths.
        """
        cameras = self.session.system.cameras
        for (cid, idx), value in self._pinned.items():
            if cid in cameras:
                cameras[cid].params[idx] = value
        for cid, idx in self._inactive:
            if cid in cameras:
                cameras[cid].params[idx] = 0.0

    def _populate_tree(self, problem=None):
        """
        Rebuild the tree. `problem` is optional: the parameters exist before any
        problem does, and hiding them until after the bootstrap left this panel
        blank at exactly the moment a user wants to type a value they know.
        """
        self._populating = True
        try:
            self.tree.clear()
            self._value_edits.clear()
            system = self.session.system
            for cid, cam in system.cameras.items():
                top = QTreeWidgetItem([f"{cid}  [{cam.model_name}]", "", ""])
                top.setData(0, Qt.UserRole, cid)
                self.tree.addTopLevelItem(top)
                for i, name in enumerate(cam.model.param_names):
                    self._add_param_row(top, cid, cam, i, name)
                self._add_extrinsic_row(top, cid, cam, problem)
                top.setExpanded(True)
            if system.cameras:
                self.tree.addTopLevelItem(
                    QTreeWidgetItem(
                        [f"board poses ({len(system.frame_board_pairs)} x 6 DOF)", "", "free"]
                    )
                )
        finally:
            self._populating = False
        if problem is not None:
            self.status.emit(
                f"problem: {problem.num_residuals} residuals, "
                f"{problem.num_free_params} free parameters"
            )

    def _add_param_row(self, parent, cid: str, cam, index: int, name: str) -> None:
        key = (cid, index)
        child = QTreeWidgetItem([name, "", ""])
        child.setData(0, Qt.UserRole + 1, index)
        parent.addChild(child)

        edit = QLineEdit()
        edit.setPlaceholderText("estimated")
        edit.setToolTip(
            "Empty: estimated.\n"
            "A number: held at exactly that, in the estimate and the solve.\n"
            "Clear the box to estimate it again."
        )
        edit.editingFinished.connect(lambda k=key: self._value_edited(k))
        self._value_edits[key] = edit
        self.tree.setItemWidget(child, 1, edit)

        check = QCheckBox()
        check.setChecked(key not in self._inactive)
        if cam.model.can_deactivate(index):
            check.setToolTip("Untick to force this parameter to zero and hold it there.")
            check.toggled.connect(lambda on, k=key: self._active_toggled(k, on))
        else:
            check.setEnabled(False)
            check.setToolTip(
                f"{name} defines the projection itself -- zeroing it does not "
                "simplify the model, it breaks it."
            )
        self.tree.setItemWidget(child, 2, check)
        self._show_value(key, cam)

    def _add_extrinsic_row(self, parent, cid: str, cam, problem) -> None:
        ext = QTreeWidgetItem(["extrinsic T_cam_rig", "", ""])
        parent.addChild(ext)
        if cam.is_reference:
            ext.setText(2, "reference (gauge, always fixed)")
            return
        blk = None if problem is None else problem.blocks.get(extr_key(cid))
        if blk is None:
            ext.setText(2, "free")
        else:
            ext.setText(2, "fixed" if blk.constant else "free")

    def _show_value(self, key: tuple[str, int], cam) -> None:
        """Put the right text and styling in one value box."""
        edit = self._value_edits.get(key)
        if edit is None:
            return
        _, index = key
        pinned = key in self._pinned
        inactive = key in self._inactive
        if inactive:
            text = "0"
        elif pinned:
            text = f"{self._pinned[key]:.6g}"
        elif self.session.is_initialized:
            text = f"{cam.params[index]:.6g}"
        else:
            # Before the bootstrap the stored value is the crude default
            # (focal = image width, centred). Showing it would read as an
            # estimate; the box stays empty so "estimated" means estimated.
            text = ""
        edit.setText(text)
        edit.setEnabled(not inactive)
        font = edit.font()
        font.setBold(pinned)
        edit.setFont(font)

    def _clear_values(self) -> None:
        """
        Drop every pin and switch everything back on.

        Values that were forced to zero stay at zero rather than being restored:
        nothing remembers what they held before, and zero is the honest default
        for a distortion term. The next estimate moves them.
        """
        if not self._pinned and not self._inactive:
            self._log("nothing to clear -- no values typed, nothing switched off")
            return
        n_held, n_off = len(self._pinned), len(self._inactive)
        self._pinned.clear()
        self._inactive.clear()
        self.refresh()
        self._log(f"cleared {n_held} typed value(s), switched {n_off} parameter(s) back on")

    def _value_edited(self, key: tuple[str, int]) -> None:
        """
        A value box lost focus or took Enter.

        Deliberately does NOT rebuild anything: `editingFinished` also fires
        while the tree is being torn down, and rebuilding from inside it
        destroys the widget mid-signal. The dicts are the state; the problem is
        reassembled on the next Build, Estimate or Solve.
        """
        if self._populating:
            return
        edit = self._value_edits.get(key)
        if edit is None or key in self._inactive:
            return
        cid, index = key
        cam = self.session.system.cameras.get(cid)
        name = cam.model.param_names[index] if cam else str(index)
        text = edit.text().strip()
        if not text:
            if self._pinned.pop(key, None) is not None:
                self._log(f"{cid}.{name} released -- it will be estimated again")
                if cam is not None:
                    self._show_value(key, cam)
            return
        try:
            value = float(text)
        except ValueError:
            self._log(f"{cid}.{name}: '{text}' is not a number -- ignored")
            if cam is not None:
                self._show_value(key, cam)
            return
        if self._pinned.get(key) == value:
            return
        self._pinned[key] = value
        self._log(f"{cid}.{name} held at {value:g} -- not estimated, not solved for")
        if cam is not None:
            self._show_value(key, cam)

    def _active_toggled(self, key: tuple[str, int], active: bool) -> None:
        if self._populating:
            return
        cid, index = key
        cam = self.session.system.cameras.get(cid)
        name = cam.model.param_names[index] if cam else str(index)
        if active:
            self._inactive.discard(key)
            self._log(f"{cid}.{name} switched on -- estimated again")
        else:
            self._inactive.add(key)
            # Off beats held: a parameter cannot be zero and 0.12 at once, and
            # leaving the pin would put a value back the moment it is reticked.
            if self._pinned.pop(key, None) is not None:
                self._log(f"{cid}.{name} switched off -- the value you typed is dropped")
            else:
                self._log(f"{cid}.{name} switched off -- forced to zero")
        if cam is None:
            return
        if not active:
            cam.params[index] = 0.0
        self._show_value(key, cam)

    # ------------------------------------------------------------------
    def _solve(self):
        if refuse_if_busy(self, "a solve"):
            return
        if not self.session.is_ready_to_solve:
            QMessageBox.information(self, "No data", "Run detection first.")
            return
        if not self.session.is_initialized:
            # Solving from default parameters (focal = image width, zero
            # distortion, identity extrinsics) converges to nonsense that still
            # looks like a result. Refuse rather than produce it.
            QMessageBox.information(
                self,
                "No starting point",
                "Click 'Estimate starting point' first.\n\n"
                "The defaults (focal = image width, identity extrinsics) do not "
                "converge to anything trustworthy.",
            )
            return
        if not self._warn_if_detections_are_stale("this solve"):
            return
        backend = self.backend_combo.currentData()
        ok, why = backend_status()[backend]
        if not ok:
            QMessageBox.warning(self, "Backend unavailable", why)
            return

        with_source(self.log, "solve")
        self.log.appendPlainText(f"--- solving with {backend} ---")
        self.log_line(self._iteration_header(IterationRecord(0, 0.0, rms_px=0.0)))
        # Held and switched-off parameters go into the system BEFORE the problem
        # is assembled, so `_fixed_intrinsics` drops exactly those columns and
        # the solver has no way to move them.
        self._apply_user_values()
        self._log_pins()
        if not self.refresh():
            return

        self.solve_btn.setEnabled(False)
        self._live = []
        self._last_live_paint = 0.0
        worker = SolveWorker(
            self.session.problem,
            backend,
            self._solve_options(),
            live=True,
        )
        # Connected here, not through run_in_thread, and to a bound method:
        # a lambda has no thread affinity, so Qt would call it ON the worker
        # thread, where touching the log and the canvas is undefined behaviour.
        worker.iteration.connect(self._on_iteration)
        run_in_thread(self, worker, self._on_solved, self._on_failed, self._log)

    def _on_iteration(self, record) -> None:
        """
        One solver step, live, on the GUI thread.

        Throttled by TIME rather than by count: appending text and redrawing the
        canvas both force a repaint, and a fast problem produces steps quicker
        than the window can draw them -- at which point the queue grows without
        bound and the "live" view ends up lagging further behind than no live
        view at all.
        """
        self._live.append(record)
        now = time.monotonic()
        # The first step always paints: it is the one that tells the user the
        # click did something, and waiting a throttle interval to say so is the
        # exact complaint this exists to answer.
        if len(self._live) > 1 and now - self._last_live_paint < LIVE_REFRESH_SECONDS:
            return
        self._last_live_paint = now
        self.log_line(self._iteration_row(record, self._live[0]))
        self._plot_live()

    def _on_solved(self, result):
        self.solve_btn.setEnabled(True)
        self.session.result = result
        self.session.commit()
        # `write_back` has just copied the solved blocks into the system. Held
        # components had no column, so they arrive unchanged -- reasserting them
        # costs nothing and makes that guarantee independent of the solver.
        self._apply_user_values()
        if self._live:
            # Already watched step by step; only the last row is missing,
            # because the throttle skipped it.
            self.log_line(self._iteration_row(self._live[-1], self._live[0]))
            self.log_line(f"{len(self._live)} step(s), rejected steps included")
        else:
            self._log_iterations(result)
        self.log.appendPlainText(result.summary())
        if result.behind_camera_points:
            self.log.appendPlainText(
                f"WARNING: {result.behind_camera_points} corners projected from "
                f"behind the camera during the solve."
            )
        self._plot_progress(result)
        self._populate_tree()
        self.status.emit(result.summary())
        self.solved.emit()

    def _on_failed(self, message: str):
        self.solve_btn.setEnabled(True)
        self.log.appendPlainText(f"FAILED: {message}")
        QMessageBox.critical(self, "Solve failed", message)
        self.status.emit("solve failed")

    @staticmethod
    def _iteration_row(record, first) -> str:
        """
        One line of the descent: RMS first, because that is the honest number.

        `rms_px` is MEASURED by the backend at each step, never derived from the
        cost. Under a robust loss the cost is weighted and would flatter the fit
        by exactly what the loss is discounting -- which is how a solve can
        report its cost halving while the pixels get twenty times worse.
        """
        has_rms = record.rms_px == record.rms_px
        base = first.rms_px if has_rms else first.cost
        now = record.rms_px if has_rms else record.cost
        change = 100.0 * (now - base) / base if base else float("nan")
        unit = "px" if has_rms else "  "
        line = f"  {record.iteration:6d}   {now:11.5g} {unit}  {change:+9.3f}%"
        if record.gradient_norm == record.gradient_norm:
            line += f"   {record.gradient_norm:11.4g}"
        return line

    def _iteration_header(self, sample) -> str:
        has_rms = sample.rms_px == sample.rms_px
        head = "    step   " + ("     RMS px" if has_rms else "        cost")
        head += "     vs start"
        if sample.gradient_norm == sample.gradient_norm:
            head += "     grad norm"
        return head

    def _log_iterations(self, result) -> None:
        """Print the descent after the fact, for a solve nothing watched live."""
        usable = [h for h in result.history if h.cost == h.cost]
        if len(usable) < 2:
            self.log.appendPlainText("(this backend reported no per-iteration history)")
            return
        step = max(1, len(usable) // MAX_ITERATION_LINES)
        rows = sorted({*range(0, len(usable), step), len(usable) - 1})
        self.log.appendPlainText(
            f"per iteration ({len(usable)} recorded"
            + (f", 1 row per {step}" if step > 1 else "")
            + ", rejected steps included):"
        )
        self.log.appendPlainText(self._iteration_header(usable[0]))
        for i in rows:
            self.log.appendPlainText(self._iteration_row(usable[i], usable[0]))

    @staticmethod
    def _series(records) -> tuple[list[float], str]:
        """The curve to draw: measured RMS when the backend gave one, else cost."""
        rms = [h.rms_px for h in records if h.rms_px == h.rms_px]
        if len(rms) > 1:
            return rms, "per-corner RMS, px (log)"
        return [h.cost for h in records if h.cost == h.cost], "cost (log)"

    def _plot_live(self) -> None:
        """Redraw the descent so far. Same axes as the final plot, no title yet."""
        values, label = self._series(self._live)
        if len(values) < 2:
            return
        fig = self.progress_plot.clear()
        ax = fig.add_subplot(111)
        ax.plot(values, lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(label)
        ax.set_title(f"solving... {len(self._live)} steps", fontsize=9)
        ax.grid(alpha=0.3)
        self.progress_plot.draw()

    def _plot_progress(self, result):
        fig = self.progress_plot.clear()
        ax = fig.add_subplot(111)
        values, label = self._series(result.history)
        if len(values) > 1:
            ax.plot(values, lw=1.2)
            ax.set_yscale("log")
            ax.set_xlabel("step")
            ax.set_ylabel(label)
        else:
            ax.bar(["initial", "final"], [result.initial_rms_px, result.final_rms_px])
            ax.set_ylabel("per-corner RMS, px")
        ax.set_title(
            f"{result.backend}: RMS {result.initial_rms_px:.3f} -> {result.final_rms_px:.3f} px",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        self.progress_plot.draw()
