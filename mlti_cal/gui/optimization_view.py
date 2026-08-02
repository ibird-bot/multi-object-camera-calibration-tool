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
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mlti_cal.gui.session import Session, SolveWorker, run_in_thread
from mlti_cal.gui.widgets import PlotCanvas
from mlti_cal.problem.loss import available_losses
from mlti_cal.problem.reprojection import extr_key
from mlti_cal.solvers import SolveOptions, backend_status
from mlti_cal.solvers.catalog import CATALOG

FREE, FIXED, DEACTIVATED = "Free", "Fixed", "Deactivated"
STATES = (FREE, FIXED, DEACTIVATED)


class OptimizationView(QWidget):
    solved = Signal()
    status = Signal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._option_widgets: dict[str, QWidget] = {}
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)

        # ---------------- left: parameters -----------------------------
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>Parameters</b> -- Free / Fixed / Deactivated"))
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["parameter", "value", "state"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        lv.addWidget(self.tree)
        note = QLabel(
            "Fixed keeps the value but removes its Jacobian column. Deactivated "
            "additionally zeroes it. The reference camera's extrinsic is always "
            "fixed: it is the gauge, and freeing it makes the covariance rank "
            "deficient by 6."
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
        self.loss_combo.setToolTip(
            "Robust loss, applied per corner inside the core as IRLS weights so "
            "every backend optimises the identical objective. Note this is not "
            "identical to Ceres' Triggs-corrected loss."
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

        btns = QHBoxLayout()
        self.solve_btn = QPushButton("Solve")
        self.solve_btn.clicked.connect(self._solve)
        rebuild = QPushButton("Rebuild problem")
        rebuild.clicked.connect(self.refresh)
        btns.addWidget(self.solve_btn)
        btns.addWidget(rebuild)
        btns.addStretch()
        rv.addLayout(btns)

        self.progress_plot = PlotCanvas(figsize=(4, 2.4), toolbar=False)
        self.progress_plot.message("no solve yet")
        rv.addWidget(self.progress_plot, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        rv.addWidget(self.log, 1)
        split.addWidget(right)
        split.setSizes([560, 640])

        self._rebuild_options()

    # ------------------------------------------------------------------
    def _rebuild_options(self):
        while self.options_form.rowCount():
            self.options_form.removeRow(0)
        self._option_widgets.clear()
        backend = self.backend_combo.currentData()
        entry = CATALOG.get(backend)
        if not entry:
            return
        for opt in entry["options"]:
            if opt.name == "max_iterations":
                continue  # already exposed above
            tip = opt.when
            if opt.per_choice:
                tip += "\n\n" + "\n".join(f"* {k}: {v}" for k, v in opt.per_choice.items())
            if opt.kind == "choice":
                w = QComboBox()
                w.addItems(opt.choices)
                default = str(opt.default).split(" ")[0]
                if default in opt.choices:
                    w.setCurrentText(default)
            elif opt.kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(opt.default))
            elif opt.kind == "int":
                w = QSpinBox()
                w.setRange(1, 100000)
                w.setValue(int(opt.default))
            else:
                w = QDoubleSpinBox()
                w.setRange(-1e6, 1e6)
                w.setValue(float(opt.default))
            w.setToolTip(tip)
            label = QLabel(opt.name)
            label.setToolTip(tip)
            self.options_form.addRow(label, w)
            self._option_widgets[opt.name] = w

    def _collect_options(self) -> dict:
        out = {}
        for name, w in self._option_widgets.items():
            if isinstance(w, QComboBox):
                out[name] = w.currentText()
            elif isinstance(w, QCheckBox):
                out[name] = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                out[name] = w.value()
        # scipy's tr_solver default is size-dependent; only send an override.
        if out.get("tr_solver") == "exact" and self.backend_combo.currentData() == "scipy":
            pass
        return out

    # ------------------------------------------------------------------
    def refresh(self):
        """Rebuild the problem from the current tree state and repopulate."""
        if not self.session.is_ready_to_solve:
            self.tree.clear()
            return
        fixed = self._fixed_intrinsics_from_tree()
        try:
            self.session.rebuild_problem(
                loss_name=self.loss_combo.currentText(),
                loss_scale=self.loss_scale.value(),
                fixed_intrinsics=fixed or None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Cannot build problem", str(exc))
            return
        self._populate_tree()

    def _fixed_intrinsics_from_tree(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            cam_item = root.child(i)
            cid = cam_item.data(0, Qt.UserRole)
            if not cid:
                continue
            for j in range(cam_item.childCount()):
                child = cam_item.child(j)
                idx = child.data(0, Qt.UserRole + 1)
                combo = self.tree.itemWidget(child, 2)
                if idx is None or combo is None:
                    continue
                if combo.currentText() != FREE:
                    out.setdefault(cid, []).append(int(idx))
        return out

    def _populate_tree(self):
        prev = self._current_states()
        self.tree.clear()
        system = self.session.system
        problem = self.session.problem
        if problem is None:
            return
        for cid, cam in system.cameras.items():
            top = QTreeWidgetItem([f"{cid}  [{cam.model_name}]", "", ""])
            top.setData(0, Qt.UserRole, cid)
            self.tree.addTopLevelItem(top)
            for i, name in enumerate(cam.model.param_names):
                child = QTreeWidgetItem([name, f"{cam.params[i]:.6g}", ""])
                child.setData(0, Qt.UserRole + 1, i)
                top.addChild(child)
                combo = QComboBox()
                combo.addItems(STATES)
                combo.setCurrentText(prev.get((cid, i), FREE))
                combo.setToolTip(
                    "Free: estimated.\nFixed: held at its current value, no "
                    "Jacobian column.\nDeactivated: forced to zero and held."
                )
                self.tree.setItemWidget(child, 2, combo)
            ext = QTreeWidgetItem(["extrinsic T_cam_rig", "", ""])
            top.addChild(ext)
            blk = problem.blocks.get(extr_key(cid))
            state = (
                "reference (gauge, always fixed)"
                if cam.is_reference
                else ("fixed" if blk is not None and blk.constant else "free")
            )
            ext.setText(2, state)
            top.setExpanded(True)

        poses = QTreeWidgetItem(
            [f"board poses ({len(system.frame_board_pairs)} x 6 DOF)", "", "free"]
        )
        self.tree.addTopLevelItem(poses)
        self.status.emit(
            f"problem: {problem.num_residuals} residuals, {problem.num_free_params} free parameters"
        )

    def _current_states(self) -> dict:
        out = {}
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            cam_item = root.child(i)
            cid = cam_item.data(0, Qt.UserRole)
            if not cid:
                continue
            for j in range(cam_item.childCount()):
                child = cam_item.child(j)
                idx = child.data(0, Qt.UserRole + 1)
                combo = self.tree.itemWidget(child, 2)
                if idx is not None and combo is not None:
                    out[(cid, int(idx))] = combo.currentText()
        return out

    # ------------------------------------------------------------------
    def _solve(self):
        if not self.session.is_ready_to_solve:
            QMessageBox.information(self, "No data", "Run detection or load the demo first.")
            return
        # Deactivated parameters are zeroed before the problem is rebuilt.
        for (cid, idx), state in self._current_states().items():
            if state == DEACTIVATED:
                self.session.system.cameras[cid].params[idx] = 0.0
        self.refresh()
        if self.session.problem is None:
            return

        backend = self.backend_combo.currentData()
        ok, why = backend_status()[backend]
        if not ok:
            QMessageBox.warning(self, "Backend unavailable", why)
            return

        self.solve_btn.setEnabled(False)
        self.log.appendPlainText(f"--- solving with {backend} ---")
        worker = SolveWorker(
            self.session.problem,
            backend,
            SolveOptions(
                max_iterations=self.iterations.value(),
                extra=self._collect_options(),
            ),
        )
        run_in_thread(self, worker, self._on_solved, self._on_failed, self.status.emit)

    def _on_solved(self, result):
        self.solve_btn.setEnabled(True)
        self.session.result = result
        self.session.commit()
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

    def _plot_progress(self, result):
        fig = self.progress_plot.clear()
        ax = fig.add_subplot(111)
        costs = [h.cost for h in result.history if h.cost == h.cost]
        if len(costs) > 1:
            ax.plot(costs, lw=1.2)
            ax.set_yscale("log")
            ax.set_xlabel("evaluation")
            ax.set_ylabel("cost (log)")
        else:
            ax.bar(["initial", "final"], [result.initial_cost, result.final_cost])
            ax.set_ylabel("cost")
        ax.set_title(
            f"{result.backend}: RMS {result.initial_rms_px:.3f} -> {result.final_rms_px:.3f} px",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        self.progress_plot.draw()
