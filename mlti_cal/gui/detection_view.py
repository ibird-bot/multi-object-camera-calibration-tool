"""
Window 1 -- Setup & Detection.

Cameras are added with a `+` exactly like calibration objects, 1..N with a
minimum of one, so monocular is just the N=1 case of the same UI. The image
list shows per-image detection counts so a bad capture is visible before it
poisons a solve, and images can be culled without touching the files on disk.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mlti_cal.detectors.charuco import (
    DICTIONARIES,
    CharucoBoardSpec,
    MultiBoardDetector,
    draw_detections,
)
from mlti_cal.gui.session import Session
from mlti_cal.gui.widgets import ImageView
from mlti_cal.io.config import CalibrationConfig, CameraConfig, build_system_from_config
from mlti_cal.models.camera import available_models
from mlti_cal.problem.initialize import initialize_system


class DetectionView(QWidget):
    system_changed = Signal()
    status = Signal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.session = session
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QWidget()
        lv = QVBoxLayout(left)

        # ---- cameras ---------------------------------------------------
        cam_box = QGroupBox("Cameras  (add with +, minimum 1)")
        cl = QVBoxLayout(cam_box)
        self.camera_table = QTableWidget(0, 4)
        self.camera_table.setHorizontalHeaderLabels(["id", "model", "image folder", "ref"])
        self.camera_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.camera_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        cl.addWidget(self.camera_table)
        row = QHBoxLayout()
        add_cam = QPushButton("+ Camera")
        add_cam.clicked.connect(self._add_camera)
        rm_cam = QPushButton("- Remove")
        rm_cam.clicked.connect(lambda: self._remove_row(self.camera_table))
        set_ref = QPushButton("Set reference")
        set_ref.setToolTip(
            "The reference camera defines the rig frame. Its extrinsic is held "
            "fixed at identity -- this is the gauge fix, without which the whole "
            "rig can drift and the covariance is rank deficient by 6."
        )
        set_ref.clicked.connect(self._set_reference)
        for b in (add_cam, rm_cam, set_ref):
            row.addWidget(b)
        row.addStretch()
        cl.addLayout(row)
        lv.addWidget(cam_box)

        # ---- boards ----------------------------------------------------
        board_box = QGroupBox("Calibration objects (Charuco boards)")
        bl = QVBoxLayout(board_box)
        self.board_table = QTableWidget(0, 7)
        self.board_table.setHorizontalHeaderLabels(
            ["id", "squares x", "squares y", "square (m)", "marker (m)", "dictionary", "id offset"]
        )
        self.board_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        bl.addWidget(self.board_table)
        brow = QHBoxLayout()
        add_b = QPushButton("+ Board")
        add_b.clicked.connect(self._add_board)
        rm_b = QPushButton("- Remove")
        rm_b.clicked.connect(lambda: self._remove_row(self.board_table))
        for b in (add_b, rm_b):
            brow.addWidget(b)
        brow.addStretch()
        bl.addLayout(brow)
        lv.addWidget(board_box)

        # ---- run -------------------------------------------------------
        run_box = QGroupBox("Detection")
        rl = QFormLayout(run_box)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.01, 10.0)
        self.noise_spin.setSingleStep(0.05)
        self.noise_spin.setValue(0.3)
        self.noise_spin.setToolTip(
            "Assumed per-coordinate corner detection noise, in pixels.\n\n"
            "Used to whiten residuals and as the cross-check against the "
            "noise level implied by the fit. When the two disagree by more "
            "than 2x the report says so: much larger means systematic error is "
            "being absorbed as noise, much smaller means the model is fitting "
            "the noise itself."
        )
        rl.addRow("assumed pixel noise", self.noise_spin)

        btns = QHBoxLayout()
        self.detect_btn = QPushButton("Run detection")
        self.detect_btn.clicked.connect(self._run_detection)
        demo_btn = QPushButton("Load synthetic demo")
        demo_btn.setToolTip(
            "Generate a synthetic multi-camera dataset with known ground truth. "
            "This is the only mode where the ground-truth honesty check can run."
        )
        demo_btn.clicked.connect(self._load_demo)
        load_btn = QPushButton("Load config...")
        load_btn.clicked.connect(self._load_config)
        save_btn = QPushButton("Save config...")
        save_btn.clicked.connect(self._save_config)
        for b in (self.detect_btn, demo_btn, load_btn, save_btn):
            btns.addWidget(b)
        rl.addRow(btns)
        lv.addWidget(run_box)

        self.summary = QLabel("no data loaded")
        self.summary.setWordWrap(True)
        lv.addWidget(self.summary)
        lv.addStretch()
        splitter.addWidget(left)

        # ---- right: per-image list + preview ---------------------------
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel("Detections per image  (uncheck to cull)"))
        self.image_table = QTableWidget(0, 4)
        self.image_table.setHorizontalHeaderLabels(["use", "frame", "camera", "corners"])
        self.image_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.image_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.image_table.itemSelectionChanged.connect(self._preview_selected)
        rv.addWidget(self.image_table, 1)
        cull = QHBoxLayout()
        apply_cull = QPushButton("Apply culling")
        apply_cull.setToolTip("Drop unchecked observations from the problem.")
        apply_cull.clicked.connect(self._apply_culling)
        self.fit_check = QCheckBox("fit image to window")
        self.fit_check.setChecked(True)
        self.fit_check.toggled.connect(lambda v: self.image_view.set_fit(v))
        cull.addWidget(apply_cull)
        cull.addWidget(self.fit_check)
        cull.addStretch()
        rv.addLayout(cull)
        self.image_view = ImageView()
        rv.addWidget(self.image_view, 2)
        splitter.addWidget(right)
        splitter.setSizes([560, 640])

        self._add_camera()
        self._add_board()

    # ------------------------------------------------------------------
    def _add_camera(self):
        t = self.camera_table
        r = t.rowCount()
        t.insertRow(r)
        t.setItem(r, 0, QTableWidgetItem(f"cam{r}"))
        combo = QComboBox()
        combo.addItems(available_models())
        t.setCellWidget(r, 1, combo)
        folder = QWidget()
        fl = QHBoxLayout(folder)
        fl.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText("folder of images for this camera")
        browse = QPushButton("...")
        browse.setMaximumWidth(30)
        browse.clicked.connect(lambda _, e=edit: self._browse_into(e))
        fl.addWidget(edit)
        fl.addWidget(browse)
        t.setCellWidget(r, 2, folder)
        t.setItem(r, 3, QTableWidgetItem("yes" if r == 0 else ""))
        t.item(r, 3).setFlags(Qt.ItemIsEnabled)

    def _browse_into(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Image folder")
        if d:
            edit.setText(d)

    def _add_board(self):
        t = self.board_table
        r = t.rowCount()
        t.insertRow(r)
        defaults = [f"board_{chr(ord('A') + r)}", "9", "7", "0.030", "0.022"]
        for c, v in enumerate(defaults):
            t.setItem(r, c, QTableWidgetItem(v))
        combo = QComboBox()
        combo.addItems(sorted(DICTIONARIES))
        combo.setCurrentText("DICT_4X4_250")
        combo.setToolTip(
            "Boards sharing a dictionary must use disjoint marker ID ranges, "
            "otherwise a detected marker cannot be attributed to a board. Set "
            "the ID offset accordingly, or give each board its own dictionary."
        )
        t.setCellWidget(r, 5, combo)
        spin = QSpinBox()
        spin.setRange(0, 10000)
        spin.setValue(r * 40)
        t.setCellWidget(r, 6, spin)

    def _remove_row(self, table: QTableWidget):
        rows = sorted({i.row() for i in table.selectedItems()}, reverse=True)
        minimum = 1 if table is self.camera_table else 0
        for r in rows:
            if table.rowCount() <= minimum:
                QMessageBox.information(self, "Minimum reached", "At least one camera is required.")
                break
            table.removeRow(r)

    def _set_reference(self):
        rows = {i.row() for i in self.camera_table.selectedItems()}
        if not rows:
            return
        pick = min(rows)
        for r in range(self.camera_table.rowCount()):
            self.camera_table.item(r, 3).setText("yes" if r == pick else "")

    # ------------------------------------------------------------------
    def _collect_config(self) -> CalibrationConfig:
        cams = []
        for r in range(self.camera_table.rowCount()):
            folder_widget = self.camera_table.cellWidget(r, 2)
            edit = folder_widget.findChild(QLineEdit)
            cams.append(
                CameraConfig(
                    id=self.camera_table.item(r, 0).text().strip(),
                    model=self.camera_table.cellWidget(r, 1).currentText(),
                    image_dir=edit.text().strip(),
                    is_reference=self.camera_table.item(r, 3).text() == "yes",
                )
            )
        boards = []
        for r in range(self.board_table.rowCount()):
            boards.append(
                {
                    "id": self.board_table.item(r, 0).text().strip(),
                    "squares_x": int(self.board_table.item(r, 1).text()),
                    "squares_y": int(self.board_table.item(r, 2).text()),
                    "square_length": float(self.board_table.item(r, 3).text()),
                    "marker_length": float(self.board_table.item(r, 4).text()),
                    "dictionary": self.board_table.cellWidget(r, 5).currentText(),
                    "marker_id_offset": self.board_table.cellWidget(r, 6).value(),
                }
            )
        return CalibrationConfig(
            cameras=cams, boards=boards, pixel_noise_std=self.noise_spin.value()
        )

    def _run_detection(self):
        try:
            config = self._collect_config()
            MultiBoardDetector(config.board_specs())  # fail fast on ID collisions
            self.status.emit("detecting...")
            system, stats = build_system_from_config(config)
            initialize_system(system)
        except Exception as exc:
            QMessageBox.critical(self, "Detection failed", f"{type(exc).__name__}:\n{exc}")
            self.status.emit("detection failed")
            return
        self.session.system = system
        self.session.pixel_noise_std = config.pixel_noise_std
        self.session.truth_values = None
        self.session.source = "images"
        self._config_for_preview = config
        self.refresh()
        self.system_changed.emit()

    def _load_demo(self):
        from mlti_cal.io.synthetic import generate_dataset
        from mlti_cal.problem.reprojection import extr_key, intr_key, pose_key

        n_cams = max(1, self.camera_table.rowCount())
        system, gt = generate_dataset(
            num_cameras=n_cams, num_frames=20, pixel_noise_std=self.noise_spin.value()
        )
        initialize_system(system)
        truth = {}
        for cid, p in gt.camera_params.items():
            truth[intr_key(cid)] = p
        for cid, e in gt.camera_extrinsics.items():
            truth[extr_key(cid)] = e
        for (f, b), pose in gt.board_poses.items():
            truth[pose_key(f, b)] = pose
        self.session.system = system
        self.session.truth_values = truth
        self.session.pixel_noise_std = self.noise_spin.value()
        self.session.source = "synthetic"
        self.refresh()
        self.system_changed.emit()
        self.status.emit("synthetic dataset loaded -- ground-truth check available")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", "", "JSON (*.json)")
        if not path:
            return
        try:
            config = CalibrationConfig.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.camera_table.setRowCount(0)
        self.board_table.setRowCount(0)
        for cam in config.cameras:
            self._add_camera()
            r = self.camera_table.rowCount() - 1
            self.camera_table.item(r, 0).setText(cam.id)
            self.camera_table.cellWidget(r, 1).setCurrentText(cam.model)
            self.camera_table.cellWidget(r, 2).findChild(QLineEdit).setText(cam.image_dir)
            self.camera_table.item(r, 3).setText("yes" if cam.is_reference else "")
        for b in config.boards:
            self._add_board()
            r = self.board_table.rowCount() - 1
            self.board_table.item(r, 0).setText(str(b["id"]))
            self.board_table.item(r, 1).setText(str(b["squares_x"]))
            self.board_table.item(r, 2).setText(str(b["squares_y"]))
            self.board_table.item(r, 3).setText(str(b["square_length"]))
            self.board_table.item(r, 4).setText(str(b["marker_length"]))
            self.board_table.cellWidget(r, 5).setCurrentText(b.get("dictionary", "DICT_4X4_250"))
            self.board_table.cellWidget(r, 6).setValue(int(b.get("marker_id_offset", 0)))
        self.noise_spin.setValue(config.pixel_noise_std)
        self.status.emit(f"loaded {path}")

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save config", "config.json", "JSON (*.json)")
        if path:
            self._collect_config().save(path)
            self.status.emit(f"saved {path}")

    # ------------------------------------------------------------------
    def refresh(self):
        system = self.session.system
        s = system.summary()
        self.summary.setText(
            f"<b>{s['cameras']}</b> camera(s), <b>{s['boards']}</b> board(s), "
            f"<b>{s['frames']}</b> frame(s), <b>{s['observations']}</b> observations, "
            f"<b>{s['observed_points']}</b> corners. Reference: <b>{s['reference_camera']}</b>. "
            f"Source: {self.session.source}."
        )
        t = self.image_table
        t.setRowCount(0)
        for obs in system.observations:
            r = t.rowCount()
            t.insertRow(r)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked)
            t.setItem(r, 0, chk)
            t.setItem(r, 1, QTableWidgetItem(obs.frame))
            t.setItem(r, 2, QTableWidgetItem(obs.camera))
            item = QTableWidgetItem(str(obs.num_points))
            if obs.num_points < 12:
                item.setForeground(Qt.red)
                item.setToolTip("Few corners -- this view contributes little and may be noisy.")
            t.setItem(r, 3, item)

    def _apply_culling(self):
        keep = []
        t = self.image_table
        for r in range(t.rowCount()):
            if t.item(r, 0).checkState() == Qt.Checked:
                keep.append(self.session.system.observations[r])
        removed = len(self.session.system.observations) - len(keep)
        if removed and not keep:
            QMessageBox.warning(self, "Nothing left", "Culling would remove every observation.")
            return
        self.session.system.observations = keep
        try:
            initialize_system(self.session.system)
        except Exception as exc:
            QMessageBox.warning(self, "Re-initialisation failed", str(exc))
        self.refresh()
        self.system_changed.emit()
        self.status.emit(f"culled {removed} observation(s)")

    def _preview_selected(self):
        rows = {i.row() for i in self.image_table.selectedItems()}
        if not rows or self.session.source != "images":
            if self.session.source == "synthetic":
                self._preview_synthetic(min(rows) if rows else 0)
            return
        obs = self.session.system.observations[min(rows)]
        config = getattr(self, "_config_for_preview", None)
        if config is None:
            return
        cam_cfg = next((c for c in config.cameras if c.id == obs.camera), None)
        if cam_cfg is None:
            return
        from pathlib import Path

        for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            p = Path(cam_cfg.image_dir) / f"{obs.frame}{suffix}"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return
                from mlti_cal.detectors.charuco import Detection

                overlay = draw_detections(
                    img,
                    [
                        Detection(
                            board_id=obs.board,
                            point_ids=obs.point_ids,
                            image_points=obs.image_points,
                        )
                    ],
                )
                self.image_view.set_image(overlay)
                return

    def _preview_synthetic(self, row: int):
        """Synthetic data has no images; plot the detected corners instead."""
        if not self.session.system.observations:
            return
        obs = self.session.system.observations[min(row, len(self.session.system.observations) - 1)]
        cam = self.session.system.cameras[obs.camera]
        w, h = cam.image_size
        canvas = np.full((h, w, 3), 32, np.uint8)
        for x, y in obs.image_points:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1)
        cv2.putText(
            canvas,
            f"{obs.frame} / {obs.camera} / {obs.board} (synthetic, no image)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        self.image_view.set_image(canvas)
