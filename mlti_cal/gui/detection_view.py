"""
Window 1 -- Setup & Detection.

Cameras are added with a `+` exactly like calibration objects, 1..N with a
minimum of one, so monocular is just the N=1 case of the same UI. The image
list shows per-image detection counts so a bad capture is visible before it
poisons a solve, and images can be culled without touching the files on disk.
"""

from __future__ import annotations

from pathlib import Path

import cv2
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
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlti_cal.detectors.charuco import (
    DICTIONARIES,
    Detection,
    MultiBoardDetector,
    draw_detections,
)
from mlti_cal.gui.session import DetectionWorker, Session, run_in_thread
from mlti_cal.gui.widgets import ImageView, ThumbnailStrip
from mlti_cal.io.config import CalibrationConfig, CameraConfig, find_images
from mlti_cal.models.camera import MODEL_LABELS, available_models

#: Taken from the config dataclass rather than restated, so a rig built in the
#: GUI and one written as JSON by hand cannot disagree about the camera model.
DEFAULT_CAMERA_MODEL = CameraConfig.model


class DetectionView(QWidget):
    system_changed = Signal()
    status = Signal(str)
    #: Outcomes worth keeping, for the Optimization log. Separate from `status`
    #: on purpose: detection emits one status line PER IMAGE, and a few hundred
    #: of those would push every other line out of a bounded log widget.
    logged = Signal(str)

    def __init__(self, session: Session, console=None, parent=None):
        super().__init__(parent)
        self.session = session
        # Shared with every other tab; a private one when built standalone.
        from mlti_cal.gui.log_console import LogConsole, SourceLog

        self.console = console if console is not None else LogConsole()
        self.log = SourceLog(self.console, "detect")
        self._thumb_paths: dict[str, Path] = {}
        self._detect_total = 0
        self._detect_done = 0
        #: Camera id flagged as the rig frame by a loaded config. None means
        #: "nobody said", and `build_system_from_config` falls back to the first
        #: camera -- the same default as before this window stopped asking.
        self._reference_id: str | None = None
        #: True once a detector setting has been changed while detections from
        #: the PREVIOUS settings are still loaded. Read by the Optimization tab.
        self._detection_stale = False
        self._build()

    @property
    def detection_is_stale(self) -> bool:
        """Detections on screen were produced with different detector settings."""
        return self._detection_stale

    def _announce(self, text: str) -> None:
        """Status bar AND log: something finished, rather than something ticked."""
        self.status.emit(text)
        self.logged.emit(text)

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
        self.camera_table = QTableWidget(0, 3)
        self.camera_table.setHorizontalHeaderLabels(["id", "model", "image folder"])
        self.camera_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.camera_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        cl.addWidget(self.camera_table)
        row = QHBoxLayout()
        add_cam = QPushButton("+ Camera")
        add_cam.clicked.connect(self._add_camera)
        rm_cam = QPushButton("- Remove")
        rm_cam.clicked.connect(lambda: self._remove_row(self.camera_table))
        for b in (add_cam, rm_cam):
            row.addWidget(b)
        row.addStretch()
        cl.addLayout(row)
        lv.addWidget(cam_box)

        # ---- boards ----------------------------------------------------
        board_box = QGroupBox("Calibration objects")
        bl = QVBoxLayout(board_box)
        self.board_table = QTableWidget(0, 7)
        self.board_table.setHorizontalHeaderLabels(
            ["id", "squares x", "squares y", "square (m)", "marker (m)", "dictionary", "id offset"]
        )
        self.board_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        bl.addWidget(self.board_table)
        brow = QHBoxLayout()
        add_b = self._build_add_object_button()
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
        # 4 decimals, not Qt's default 2: this box receives MEASURED values from
        # the static-sequence tool, and rounding 0.1234 px to 0.12 throws away
        # 3% of a number whose whole purpose was to be measured rather than
        # guessed. The floor is likewise below any real sensor rather than at a
        # tidy 0.01.
        self.noise_spin.setRange(0.0001, 10.0)
        self.noise_spin.setDecimals(4)
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
        # Deactivatable, because "I do not know the corner noise" is the honest
        # answer far more often than a typed 0.3 is. Unticked, residuals are
        # left unweighted and the covariance takes its sigma from the residuals
        # -- which it does by default anyway -- and no noise cross-check is
        # reported, since there is no claim to check.
        self.noise_check = QCheckBox()
        self.noise_check.setChecked(True)
        self.noise_check.setToolTip(
            "Untick if you do not know the corner detection noise.\n\n"
            "Unticked: residuals are not whitened, the covariance derives sigma "
            "from the residuals themselves, and the report stops cross-checking "
            "the assumed noise against the achieved one -- that comparison is one "
            "of the better lies detectors, so switching it off costs something.\n\n"
            "CAREFUL with a robust loss: its scale is in WHITENED units, so the "
            "same 'loss scale 2.0' cuts in at 2.0 x sigma when this is on and at "
            "2.0 px when it is off. Re-tune it if you change this."
        )
        self.noise_check.toggled.connect(self._on_noise_toggled)
        self.measure_noise_btn = QPushButton("Measure...")
        self.measure_noise_btn.setToolTip(
            "Measure it from a static sequence: a fixed camera and a fixed board, "
            "photographed several times without touching anything."
        )
        self.measure_noise_btn.clicked.connect(self._measure_noise)
        noise_row = QHBoxLayout()
        noise_row.addWidget(self.noise_check)
        noise_row.addWidget(self.noise_spin, 1)
        noise_row.addWidget(self.measure_noise_btn)
        rl.addRow("assumed pixel noise", noise_row)

        btns = QHBoxLayout()
        self.detect_btn = QPushButton("Run detection")
        self.detect_btn.clicked.connect(self._run_detection)
        load_btn = QPushButton("Load config...")
        load_btn.clicked.connect(self._load_config)
        save_btn = QPushButton("Save config...")
        save_btn.clicked.connect(self._save_config)
        for b in (self.detect_btn, load_btn, save_btn):
            btns.addWidget(b)
        rl.addRow(btns)
        lv.addWidget(run_box)

        self.summary = QLabel("no data loaded")
        self.summary.setWordWrap(True)
        lv.addWidget(self.summary)
        lv.addStretch()
        splitter.addWidget(left)

        # ---- right: thumbnails + per-image list + preview ---------------
        right = QWidget()
        rv = QVBoxLayout(right)

        # The strip sits above the table so a bad capture is visible as a
        # picture, not as a row of numbers.
        rv.addWidget(QLabel("Images"))
        self.thumbnails = ThumbnailStrip()
        self.thumbnails.setToolTip(
            "Every image found in the configured folders. During detection the "
            "image being processed is highlighted; afterwards each thumbnail "
            "shows its detected corners in red."
        )
        self.thumbnails.picked.connect(self._preview_key)
        rv.addWidget(self.thumbnails, 3)

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
        # Shown under the names these models are known by elsewhere, with the
        # internal id as the item DATA -- someone arriving from another tool
        # looks for "Double Sphere", not for "double_sphere".
        for model_name in available_models():
            combo.addItem(MODEL_LABELS.get(model_name, model_name), model_name)
        # Explicit, because `available_models()` is sorted alphabetically and
        # "double_sphere" now wins that sort: the form would otherwise default
        # every camera to a fisheye-oriented model, which is wrong for most
        # rigs and starts them from a bootstrap they did not ask for.
        combo.setCurrentIndex(combo.findData(DEFAULT_CAMERA_MODEL))
        combo.setToolTip(
            "OpenCV for normal and wide lenses -- start here.\n\n"
            "OpenCV Fisheye / Double Sphere / Enhanced Unified for genuine "
            "fisheye optics, where a radial-tangential polynomial cannot "
            "represent the projection at all. Double Sphere and Enhanced "
            "Unified use two parameters instead of four and are far better "
            "conditioned, but their parameters trade off against focal length "
            "unless the board covers a wide field.\n\n"
            "Thin Prism adds rational radial and prism terms for lenses with "
            "residual asymmetry -- twelve coefficients that correlate hard, so "
            "check the held-out error, not the training RMS.\n\n"
            "Matlab is OpenCV plus a skew term. FOV and Halcon Division are "
            "one-parameter models: less flexible, much harder to overfit."
        )
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

    def _browse_into(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Image folder")
        if d:
            edit.setText(d)
            self.reload_thumbnails()

    # ------------------------------------------------------------------
    def _scan_images(self) -> tuple[list[tuple[str, str]], dict[str, Path]]:
        """List what the configured folders hold. Reads no pixels."""
        entries: list[tuple[str, str]] = []
        paths: dict[str, Path] = {}
        for cam_cfg in self._collect_config().cameras:
            if not cam_cfg.image_dir:
                continue
            try:
                images = find_images(cam_cfg.image_dir)
            except (NotADirectoryError, FileNotFoundError):
                continue
            for p in images:
                key = f"{cam_cfg.id}/{p.stem}"
                entries.append((key, f"{p.stem}\n{cam_cfg.id}"))
                paths[key] = p
        return entries, paths

    def reload_thumbnails(self) -> None:
        """
        Show every image the configured folders contain, before any detection.

        Decoded at 1/8 scale via IMREAD_REDUCED_*: OpenCV skips most of the
        work inside the codec rather than shrinking afterwards, which is what
        makes showing a whole capture folder viable at all. Full-resolution
        decoding here would take seconds per hundred images for pixels that get
        thrown away by the 128 px icon anyway.
        """
        entries, paths = self._scan_images()
        self._thumb_paths = paths
        self.thumbnails.set_entries(entries)
        if not entries:
            self._announce("no images found -- set an image folder for each camera")
            return
        for key, path in paths.items():
            img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
            if img is not None:
                self.thumbnails.set_image(key, img)
        self._announce(f"{len(entries)} image(s) found -- run detection to find corners")

    def _sync_thumbnail_entries(self) -> dict[str, Path]:
        """
        Guarantee a row exists for every image detection is about to report on.

        Compares the KEY SET, not merely "is the strip empty": a folder typed
        straight into the table never went through the browse handler, so with
        an emptiness check a second camera added that way would have no rows,
        and `set_image` would silently drop every thumbnail for it -- blank
        pictures, no error. Listing directories is cheap; decoding is not, and
        this deliberately does neither more than it must.
        """
        entries, paths = self._scan_images()
        if set(paths) != set(self._thumb_paths):
            self._thumb_paths = paths
            self.thumbnails.set_entries(entries)
        return paths

    #: Object types the picker offers. Only the first is implemented; the rest
    #: are listed, disabled, so the shape of the app is honest about where they
    #: will go rather than pretending a Charuco board is the only thing a
    #: calibration target can be.
    OBJECT_TYPES = (
        ("Charuco board", True),
        ("Aruco board", False),
        ("Checkerboard", False),
        ("Circle grid", False),
    )

    def _build_add_object_button(self) -> QToolButton:
        btn = QToolButton()
        btn.setText("+ Object")
        btn.setPopupMode(QToolButton.InstantPopup)
        # The types are named in the tooltip as well as the menu, so hovering
        # answers "what can this thing calibrate against" without a click.
        implemented = [name for name, ok in self.OBJECT_TYPES if ok]
        pending = [name for name, ok in self.OBJECT_TYPES if not ok]
        btn.setToolTip(
            "Add a calibration object. Click to pick the type.\n"
            f"Available: {', '.join(implemented)}.\n"
            f"Not implemented yet: {', '.join(pending)}."
        )
        menu = QMenu(btn)
        # QMenu drops action tooltips unless asked, which would hide the only
        # place the "not implemented yet" reason is written.
        menu.setToolTipsVisible(True)
        for label, implemented in self.OBJECT_TYPES:
            action = menu.addAction(label)
            if implemented:
                action.triggered.connect(self._add_board)
                action.setToolTip("Chessboard corners refined from Aruco markers.")
            else:
                action.setEnabled(False)
                action.setToolTip("Not implemented yet -- detector not written.")
        btn.setMenu(menu)
        return btn

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

    # ------------------------------------------------------------------
    def _collect_config(self) -> CalibrationConfig:
        """
        The table as a config.

        There is no reference-camera control here on purpose: the gauge belongs
        with the other free/fixed choices in the Optimization window. A
        reference read from a loaded JSON is still carried through
        (`_reference_id`), so loading and saving a config does not quietly move
        the rig frame to the first camera.
        """
        cams = []
        for r in range(self.camera_table.rowCount()):
            folder_widget = self.camera_table.cellWidget(r, 2)
            edit = folder_widget.findChild(QLineEdit)
            cam_id = self.camera_table.item(r, 0).text().strip()
            cams.append(
                CameraConfig(
                    id=cam_id,
                    # currentData, not currentText: the combo now shows a
                    # label and carries the model id as its data.
                    model=self.camera_table.cellWidget(r, 1).currentData(),
                    image_dir=edit.text().strip(),
                    is_reference=cam_id == self._reference_id,
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
            cameras=cams,
            boards=boards,
            pixel_noise_std=(self.noise_spin.value() if self.noise_check.isChecked() else None),
            detectors=dict(self.session.detector_settings),
        )

    def _on_noise_toggled(self, known: bool) -> None:
        """Grey the box out rather than hide it: the value is remembered."""
        self.noise_spin.setEnabled(known)
        self.status.emit(
            f"pixel noise: {self.noise_spin.value():.3f} px assumed"
            if known
            else "pixel noise: unknown -- residuals unweighted, sigma taken from the fit"
        )

    def _measure_noise(self) -> None:
        """
        Open the static-sequence measurement window.

        Needs a board: the measurement tracks corners by charuco ID across
        frames, so it has to know which board is in the pictures.
        """
        from mlti_cal.gui.noise_dialog import NoiseDialog

        try:
            specs = self._collect_config().board_specs()
        except Exception as exc:
            QMessageBox.critical(self, "Board definition invalid", f"{type(exc).__name__}:\n{exc}")
            return
        if not specs:
            QMessageBox.information(
                self,
                "No board defined",
                "Add a calibration object first -- the measurement tracks each "
                "corner by its id across the static frames, so it needs to know "
                "which board it is looking at.",
            )
            return
        dialog = NoiseDialog(specs, self.session.detector_settings["charuco"], self)
        dialog.accepted_value.connect(self._on_noise_measured)
        dialog.measured.connect(self._on_noise_report)
        dialog.exec()

    def _on_noise_report(self, lines: list) -> None:
        """The measurement's full output, kept in the session transcript."""
        self.console.write("noise", "--- pixel noise measurement ---")
        for line in lines:
            self.console.write("noise", line)

    def _on_noise_measured(self, sigma: float) -> None:
        self.noise_check.setChecked(True)
        self.noise_spin.setValue(sigma)
        self._announce(f"measured pixel noise: {sigma:.4f} px per coordinate")

    def open_detector_settings(self) -> None:
        """
        Open the per-detector settings window. Public: the menu bar calls it too.
        """
        from mlti_cal.gui.detector_dialog import DetectorSettingsDialog

        dialog = DetectorSettingsDialog(
            self.session.detector_settings, in_use=self._kinds_in_use(), parent=self
        )
        dialog.settings_applied.connect(self._on_detector_settings_applied)
        dialog.exec()

    def _kinds_in_use(self) -> set[str]:
        """Detector kinds the configured objects actually need."""
        # Every object type the picker can add today is a charuco board, so this
        # is charuco whenever a board exists. It reads the table rather than
        # assuming, so it keeps telling the truth once a second kind lands.
        return {"charuco"} if self.board_table.rowCount() else set()

    def _on_detector_settings_applied(self, settings: dict) -> None:
        changed = [
            kind
            for kind, value in settings.items()
            if value != self.session.detector_settings.get(kind)
        ]
        self.session.detector_settings = dict(settings)
        if changed:
            # The settings are edited in their own window and reported nowhere
            # on this tab, so the log is where the record of what they became
            # lives. Values, not just "something changed": a transcript saying
            # only that a setting moved cannot be used to reproduce a run.
            self._announce(f"detector settings changed -- {self.detector_summary_text()}")
            self._on_detection_setting_changed()

    def detector_summary_text(self) -> str:
        """What each detector is set to, defaults named as such."""
        from mlti_cal.detectors.registry import DETECTOR_KINDS

        parts = []
        for kind_id, value in sorted(self.session.detector_settings.items()):
            label = DETECTOR_KINDS[kind_id].label
            defaults = type(value)().to_dict()
            changed = {k: v for k, v in value.to_dict().items() if v != defaults[k]}
            detail = (
                ", ".join(f"{k}={v}" for k, v in changed.items()) if changed else "all defaults"
            )
            parts.append(f"{label}: {detail}")
        return "   |   ".join(parts)

    def _on_detection_setting_changed(self) -> None:
        """
        A changed detector setting invalidates every corner already found.

        Silently keeping them would be the worst failure this app can produce:
        the user changes corner refinement, hits solve, and gets a result
        computed from corners detected under the OLD setting, with the new one
        displayed next to it. So the existing detections are marked stale and
        the run button says so until detection is re-run.
        """
        if self.session.source == "none":
            return
        self._detection_stale = True
        self.session.detection_stale = True
        self.detect_btn.setText("Run detection  (settings changed)")
        self.summary.setText(
            "Detector settings changed since these corners were found. "
            "Re-run detection before initialising or solving -- the current "
            "detections were produced with the previous settings."
        )

    def _run_detection(self):
        """
        Start detection on a background thread.

        Everything that can be checked instantly is checked instantly, before
        the thread exists, so a typo in a board spec still fails as a dialog on
        the click rather than a second later through the failure path.
        """
        try:
            config = self._collect_config()
            MultiBoardDetector(
                config.board_specs(), settings=config.charuco
            )  # fail fast on ID collisions
        except Exception as exc:
            QMessageBox.critical(self, "Detection failed", f"{type(exc).__name__}:\n{exc}")
            self._announce("detection failed")
            return

        paths = self._sync_thumbnail_entries()
        self._config_for_preview = config
        self._detect_total = len(paths)
        self._detect_done = 0
        self.detect_btn.setEnabled(False)
        self.status.emit("detecting...")

        worker = DetectionWorker(config)
        # Bound methods of this widget, never lambdas -- see run_in_thread.
        worker.image_started.connect(self._on_image_started)
        worker.image_done.connect(self._on_image_done)
        run_in_thread(self, worker, self._on_detected, self._on_detect_failed, self.status.emit)

    def _on_image_started(self, key: str):
        self.thumbnails.mark_active(key)

    def _on_image_done(self, key: str, thumb, corners: int):
        if thumb is not None:
            self.thumbnails.set_image(key, thumb)
        cam, _, stem = key.partition("/")
        self.thumbnails.set_caption(key, f"{stem}\n{cam}  {corners} pts")
        self.thumbnails.mark_empty(key, corners == 0)
        self._detect_done += 1
        self.status.emit(f"detecting... {self._detect_done}/{self._detect_total}  {key}")

    def _on_detected(self, payload):
        system, stats = payload
        self.thumbnails.mark_active(None)
        self.detect_btn.setEnabled(True)
        self._detection_stale = False
        self.session.detection_stale = False
        self.detect_btn.setText("Run detection")

        if not system.observations:
            # A run that searched every image and found nothing is a FAILURE to
            # the user, whatever the return type says. Committing it would
            # replace a working session with an empty one and hand the
            # Optimization tab a problem with no residuals; the mismatched
            # board that causes this is by far the most common setup error, so
            # it gets named. The red thumbnails are left on screen as evidence.
            n = stats.get("per_camera", {})
            searched = sum(c.get("images", 0) for c in n.values())
            QMessageBox.warning(
                self,
                "No detections",
                f"Searched {searched} image(s) and found no corners.\n\n"
                "The board description almost certainly does not match the "
                "printed board. Check the squares x / squares y counts and the "
                "dictionary against the board actually in the pictures.",
            )
            self._announce(f"detection found nothing in {searched} image(s)")
            return

        self.session.system = system
        self.session.pixel_noise_std = self._config_for_preview.pixel_noise_std
        self.session.truth_values = None
        self.session.source = "images"
        self.session.clear_initial()  # a new system has no estimate yet
        self.refresh()
        self.system_changed.emit()
        blank = len(self.thumbnails.empty_keys)
        note = f"  ({blank} image(s) with no corners)" if blank else ""
        self._announce(f"detection done -- {len(system.observations)} observation(s){note}")

    def _on_detect_failed(self, message: str):
        # Re-enable BEFORE the modal dialog: a failure that leaves the button
        # dead locks the user out of retrying for the rest of the session.
        self.detect_btn.setEnabled(True)
        self.thumbnails.mark_active(None)
        self._announce("detection failed")
        QMessageBox.critical(self, "Detection failed", message)

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load config", "", "JSON (*.json)")
        if not path:
            return
        try:
            config = CalibrationConfig.load(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._apply_config(config)
        self._announce(f"loaded {path}")

    def _apply_config(self, config: CalibrationConfig) -> None:
        """
        Put a config into the form. Separate from `_load_config` so loading can
        be exercised without a file dialog standing in the way.
        """
        self.camera_table.setRowCount(0)
        self.board_table.setRowCount(0)
        self._reference_id = next((c.id for c in config.cameras if c.is_reference), None)
        # Settings first: filling the tables triggers change signals, and the
        # panel must already hold the loaded values when they fire.
        self.session.detector_settings = dict(config.detectors)
        for cam in config.cameras:
            self._add_camera()
            r = self.camera_table.rowCount() - 1
            self.camera_table.item(r, 0).setText(cam.id)
            combo = self.camera_table.cellWidget(r, 1)
            combo.setCurrentIndex(combo.findData(cam.model))
            self.camera_table.cellWidget(r, 2).findChild(QLineEdit).setText(cam.image_dir)
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
        known = config.pixel_noise_std is not None
        self.noise_check.setChecked(known)
        if known:
            self.noise_spin.setValue(config.pixel_noise_std)

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save config", "config.json", "JSON (*.json)")
        if path:
            self._collect_config().save(path)
            self._announce(f"saved {path}")

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
        # The estimate was fitted to the observations that just went away, so it
        # is dropped rather than kept and quietly presented as current. The user
        # re-runs "Estimate starting point" when they are done culling.
        self.session.clear_initial()
        self.refresh()
        self.system_changed.emit()
        self._announce(f"culled {removed} observation(s) -- re-run the starting estimate")

    def _detections_for(self, key: str) -> list[Detection]:
        """Rebuild drawable detections for one thumbnail from the observations."""
        cam, _, frame = key.partition("/")
        return [
            Detection(board_id=o.board, point_ids=o.point_ids, image_points=o.image_points)
            for o in self.session.system.observations
            if o.camera == cam and o.frame == frame
        ]

    def _preview_key(self, key: str):
        """A thumbnail was clicked: show it full size, with corners if we have them."""
        path = self._thumb_paths.get(key)
        if path is None:
            return
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.status.emit(f"cannot read {path}")
            return
        dets = self._detections_for(key)
        self.image_view.set_image(draw_detections(img, dets) if dets else img)
        self.status.emit(f"{key}: {sum(len(d.point_ids) for d in dets)} corner(s)")

    def _preview_selected(self):
        rows = {i.row() for i in self.image_table.selectedItems()}
        if not rows or self.session.source != "images":
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
