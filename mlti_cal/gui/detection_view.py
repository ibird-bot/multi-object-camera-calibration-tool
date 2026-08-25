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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlti_cal.detectors import DETECTOR_KINDS, PLUGIN_ERRORS
from mlti_cal.detectors.base import Detection, board_colours, draw_detections
from mlti_cal.detectors.registry import get_kind, implemented_kinds
from mlti_cal.gui.session import DetectionWorker, Session, refuse_if_busy, run_in_thread
from mlti_cal.gui.widgets import ImageView, ThumbnailStrip
from mlti_cal.io.config import CalibrationConfig, CameraConfig, board_kind, find_images
from mlti_cal.models.camera import MODEL_LABELS, available_models
from mlti_cal.options import Option

#: Taken from the config dataclass rather than restated, so a rig built in the
#: GUI and one written as JSON by hand cannot disagree about the camera model.
DEFAULT_CAMERA_MODEL = CameraConfig.model


def default_board_kind() -> str:
    """
    Which kind a new board starts as.

    Charuco when it is installed, because it is the one that needs no masking
    and the one most rigs use; otherwise simply the first registered kind, so a
    build with only plugins still has a sensible starting point.
    """
    kinds = [k.id for k in implemented_kinds()]
    return "charuco" if "charuco" in kinds else kinds[0]


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
        headers = list(self.FIXED_COLUMNS) + self.board_column_labels()
        self.board_table = QTableWidget(0, len(headers))
        # Headers come from the registry: `id`, `kind`, then one column per
        # distinct board field across every installed detector. Installing a
        # plugin that declares a new field adds a column here on the next run.
        self.board_table.setHorizontalHeaderLabels(headers)
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
            "Assumed per-coordinate corner noise, in pixels.\n\n"
            "Whitens residuals; cross-checked against the noise the fit "
            "implies. Over 2x apart the report says so -- larger means "
            "systematic error absorbed as noise, smaller means the model is "
            "fitting noise."
        )
        # Deactivatable, because "I do not know the corner noise" is the honest
        # answer far more often than a typed 0.3 is. Unticked, residuals are
        # left unweighted and the covariance takes its sigma from the residuals
        # -- which it does by default anyway -- and no noise cross-check is
        # reported, since there is no claim to check.
        self.noise_check = QCheckBox()
        self.noise_check.setChecked(True)
        self.noise_check.setToolTip(
            "Untick if you do not know the corner noise.\n\n"
            "Unticked: residuals unweighted, covariance takes sigma from the "
            "residuals, no noise cross-check -- which costs you one of the "
            "better lies detectors.\n\n"
            "Robust loss scale is in WHITENED units: 2.0 means 2.0 x sigma "
            "when ticked, 2.0 px when not. Re-tune it."
        )
        self.noise_check.toggled.connect(self._on_noise_toggled)
        self.measure_noise_btn = QPushButton("Measure...")
        self.measure_noise_btn.setToolTip(
            "Measure it from a static sequence: fixed camera, fixed board, "
            "several shots, nothing touched."
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
            "Every image in the configured folders. The one being processed is "
            "highlighted; afterwards each thumbnail shows its corners in red."
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
            "OpenCV -- normal and wide lenses. Start here.\n"
            "Fisheye / Double Sphere / Enhanced Unified -- true fisheye, where "
            "radial-tangential cannot fit. The last two: 2 params not 4, better "
            "conditioned, but trade off against focal unless the board covers a "
            "wide field.\n"
            "Thin Prism -- adds rational radial and prism terms. 12 correlated "
            "coefficients: judge on held-out error, not training RMS.\n"
            "Matlab -- OpenCV plus skew.\n"
            "FOV / Halcon Division -- 1 param, harder to overfit."
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

    #: Board-table columns, derived from the REGISTRY rather than listed here.
    #: Six hardcoded per-kind dicts used to live at this spot -- which columns a
    #: kind reads, what the shared combo offers, what it falls back to, why a
    #: cell is greyed, what the shared geometry columns mean. Every one of them
    #: had to be edited to add a target type, and a third party could not edit
    #: them at all. They are all now consequences of `DetectorKind.board_fields`.
    #:
    #: Fields of different kinds that mean the same thing share a column by
    #: declaring the same `Option.column` -- a Charuco board's `squares_x` and a
    #: dot grid's `circles_x` are both `count_x`. A field with no `column` gets
    #: one of its own. The table is therefore as wide as the union of what the
    #: installed kinds declare, which stays small because kinds overlap.
    FIXED_COLUMNS = ("id", "kind")

    @classmethod
    def board_columns(cls) -> list[tuple[str, list[Option]]]:
        """(column key, the fields of each kind that live in it), in a stable order."""
        columns: dict[str, list[Option]] = {}
        for kind in implemented_kinds():
            for opt in kind.board_fields:
                columns.setdefault(opt.column or opt.name, []).append(opt)
        return list(columns.items())

    @classmethod
    def board_column_labels(cls) -> list[str]:
        """
        Header text per column.

        A shared column is headed by its key (`count x`), because the fields in
        it are named differently by each kind and picking one name would make
        the header lie for every other row. A column belonging to a single field
        is headed by that field's name, which is exact.
        """
        labels = []
        for key, opts in cls.board_columns():
            names = {o.name for o in opts}
            labels.append((names.pop() if len(names) == 1 else key).replace("_", " "))
        return labels

    def _column_index(self, key: str) -> int:
        keys = [k for k, _ in self.board_columns()]
        return len(self.FIXED_COLUMNS) + keys.index(key)

    def _field_for(self, kind_id: str, column_key: str) -> Option | None:
        """The field this kind puts in that column, or None if it uses none."""
        for opt in get_kind(kind_id).board_fields:
            if (opt.column or opt.name) == column_key:
                return opt
        return None

    def _board_kind(self, row: int) -> str:
        widget = self.board_table.cellWidget(row, 1)
        return default_board_kind() if widget is None else widget.currentData()

    def _on_board_kind_changed(self) -> None:
        """Find the row the sending combo currently sits in, then sync it."""
        sender = self.sender()
        for row in range(self.board_table.rowCount()):
            if self.board_table.cellWidget(row, 1) is sender:
                self._sync_board_row(row)
                return

    def _sync_board_row(self, row: int) -> None:
        """
        Point every geometry column at what THIS row's kind declares for it.

        A column the kind does not use is greyed rather than cleared, so
        switching the kind back does not lose what was typed. A choice column is
        repopulated, because the same column can offer an ArUco dictionary on one
        row and a grid type on the next.
        """
        if row >= self.board_table.rowCount():
            return
        kind_id = self._board_kind(row)
        label = get_kind(kind_id).label
        for key, _ in self.board_columns():
            col = self._column_index(key)
            opt = self._field_for(kind_id, key)
            note = "" if opt is not None else f"A {label} has no {key.replace('_', ' ')}."
            widget = self.board_table.cellWidget(row, col)
            if widget is not None:
                self._sync_choice_cell(widget, opt)
                widget.setEnabled(opt is not None)
                widget.setToolTip(opt.when if opt is not None else note)
                continue
            item = self.board_table.item(row, col)
            if item is None:
                continue
            flags = item.flags()
            item.setFlags(
                flags | Qt.ItemIsEditable if opt is not None else flags & ~Qt.ItemIsEditable
            )
            item.setToolTip(opt.when if opt is not None else note)

    @staticmethod
    def _sync_choice_cell(combo, opt: Option | None) -> None:
        """
        Repopulate a choice cell for the field now occupying it.

        Signals are blocked while it happens: clearing a combo emits
        currentIndexChanged, which would re-enter `_sync_board_row`. The default
        matters too -- falling back to index 0 once left DICT_4X4_100 selected on
        a row that had asked for nothing of the sort.
        """
        if opt is None or opt.kind != "choice":
            return
        existing = [combo.itemText(i) for i in range(combo.count())]
        if existing == list(opt.choices):
            return
        blocked = combo.blockSignals(True)
        previous = combo.currentText()
        combo.clear()
        combo.addItems(list(opt.choices))
        combo.setCurrentText(previous if previous in opt.choices else str(opt.default))
        combo.blockSignals(blocked)

    def _build_add_object_button(self) -> QToolButton:
        """
        The picker is built FROM the registry, not from a list kept beside it.

        A hardcoded copy is how "checkerboard: not implemented yet" survives in
        the menu for a release after the detector lands. The registry already
        knows which kinds work and why the others do not, so asking it means the
        menu cannot disagree with the code.
        """
        btn = QToolButton()
        btn.setText("+ Object")
        btn.setPopupMode(QToolButton.InstantPopup)
        kinds = list(DETECTOR_KINDS.values())
        implemented = [k.label for k in kinds if k.implemented]
        pending = [k.label for k in kinds if not k.implemented]
        btn.setToolTip(
            "Add a calibration object. Click to pick the type.\n"
            f"Available: {', '.join(implemented)}.\n"
            f"Not implemented yet: {', '.join(pending) or 'none'}."
        )
        menu = QMenu(btn)
        # QMenu drops action tooltips unless asked, which would hide the only
        # place the "not implemented yet" reason is written.
        menu.setToolTipsVisible(True)
        # A plugin that failed to import is reported HERE, in the menu where a
        # user goes looking for it. Silence would be indistinguishable from
        # never having installed it, which is the failure this guards against.
        if PLUGIN_ERRORS:
            broken = menu.addAction(f"{len(PLUGIN_ERRORS)} detector plugin(s) failed to load")
            broken.setEnabled(False)
            broken.setToolTip(
                "\n\n".join(f"{where}\n{detail}" for where, detail in PLUGIN_ERRORS.items())
            )
            menu.addSeparator()
        for kind in kinds:
            label = kind.label if kind.origin == "built-in" else f"{kind.label}  [{kind.origin}]"
            action = menu.addAction(label)
            if kind.implemented:
                action.triggered.connect(lambda _=False, k=kind.id: self._add_board(k))
                action.setToolTip(kind.description)
            else:
                action.setEnabled(False)
                action.setToolTip(f"Not implemented yet -- {kind.unavailable_reason}.")
        btn.setMenu(menu)
        return btn

    def _add_board(self, kind_id: str | None = None):
        """
        A new row, with a cell per registry column at this kind's defaults.

        Nothing here knows what a Charuco board is. The cells come from
        `DetectorKind.board_fields`, so a plugin's board is entered through the
        same table as a built-in one the first time the app is run after it is
        installed.
        """
        kind_id = kind_id or default_board_kind()
        t = self.board_table
        r = t.rowCount()
        t.insertRow(r)
        t.setItem(r, 0, QTableWidgetItem(f"board_{chr(ord('A') + r)}"))

        kind_combo = QComboBox()
        for kind in implemented_kinds():
            suffix = "" if kind.origin == "built-in" else f"  [{kind.origin}]"
            kind_combo.addItem(kind.label + suffix, kind.id)
        kind_combo.setCurrentIndex(max(0, kind_combo.findData(kind_id)))
        kind_combo.setToolTip(
            "Which detector looks for this object. Coded targets are found first "
            "and painted out before the uncoded ones search."
        )
        # The row is looked up when the signal fires, never captured here:
        # removing an earlier row shifts every row below it, and a captured index
        # would then grey out the wrong board's columns.
        kind_combo.currentIndexChanged.connect(self._on_board_kind_changed)
        t.setCellWidget(r, 1, kind_combo)

        for key, opts in self.board_columns():
            col = self._column_index(key)
            mine = self._field_for(kind_id, key)
            if all(o.kind == "choice" for o in opts):
                combo = QComboBox()
                source = mine or opts[0]
                combo.addItems(list(source.choices))
                combo.setCurrentText(str(source.default))
                t.setCellWidget(r, col, combo)
            else:
                # A text cell, because this column holds numbers for at least one
                # kind. Its starting value is this kind's default where it has
                # one, and the first declaring kind's otherwise -- so switching
                # the row's kind later finds something sensible already there.
                source = mine or opts[0]
                t.setItem(r, col, QTableWidgetItem(str(source.default)))
        self._sync_board_row(r)

    def _read_cell(self, row: int, opt: Option):
        """One cell as the type its field declares."""
        col = self._column_index(opt.column or opt.name)
        widget = self.board_table.cellWidget(row, col)
        if widget is not None:
            return widget.currentText()
        text = self.board_table.item(row, col).text().strip()
        if opt.kind == "int":
            return int(text)
        if opt.kind == "float":
            return float(text)
        if opt.kind == "bool":
            return text.lower() in ("1", "true", "yes")
        return text

    def _write_cell(self, row: int, opt: Option, value) -> None:
        col = self._column_index(opt.column or opt.name)
        widget = self.board_table.cellWidget(row, col)
        if widget is not None:
            widget.setCurrentText(str(value))
        else:
            self.board_table.item(row, col).setText(str(value))

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
            kind = get_kind(self._board_kind(r))
            # Each board is written out in ITS OWN vocabulary and carries only
            # the fields its kind declares. A dot grid recorded as having
            # "squares", or a checkerboard carrying a dictionary, would
            # misdescribe the rig in the one file someone opens to find out what
            # was calibrated against.
            board = {"id": self.board_table.item(r, 0).text().strip(), "kind": kind.id}
            for opt in kind.board_fields:
                board[opt.name] = self._read_cell(r, opt)
            boards.append(board)
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
            # Charuco specifically, not "the coded kinds": the noise estimator
            # tracks a corner by its id across a static sequence, and it is
            # written against CharucoBoardSpec. A plugin's coded board would
            # need its own estimator, not this one pointed at it.
            specs = self._collect_config().specs_by_kind().get("charuco", [])
        except Exception as exc:
            QMessageBox.critical(self, "Board definition invalid", f"{type(exc).__name__}:\n{exc}")
            return
        if not specs:
            QMessageBox.information(
                self,
                "No Charuco board defined",
                "The noise measurement needs a Charuco board: it tracks corners "
                "by id across frames, and only a coded board gives an id that "
                "survives. A checkerboard's corner 0 is whichever end OpenCV "
                "started from.",
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
        return {self._board_kind(r) for r in range(self.board_table.rowCount())}

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
            "Re-run detection before initialising or solving."
        )

    def _run_detection(self):
        """
        Start detection on a background thread.

        Everything that can be checked instantly is checked instantly, before
        the thread exists, so a typo in a board spec still fails as a dialog on
        the click rather than a second later through the failure path.
        """
        if refuse_if_busy(self, "another detection run"):
            return
        try:
            config = self._collect_config()
            # Fail fast on charuco ID collisions AND on two same-sized plain
            # checkerboards, both of which make a detection unattributable.
            config.detectors_for_run()
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
            kind = get_kind(board_kind(b))
            self._add_board(kind.id)
            r = self.board_table.rowCount() - 1
            self.board_table.item(r, 0).setText(str(b["id"]))
            for opt in kind.board_fields:
                # Defaulted, not required: a hand-written board may leave a
                # field with a spec-level default off entirely.
                self._write_cell(r, opt, b.get(opt.name, opt.default))
            self._sync_board_row(r)
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
        boards = self.session.system.boards
        return [
            Detection(
                board_id=o.board,
                point_ids=o.point_ids,
                image_points=o.image_points,
                # Carried through so the overlay can draw a checkerboard corner
                # as a square and a charuco one as a circle. An observation does
                # not itself record which detector found it -- the board does,
                # and the board is what the observation names.
                kind=boards[o.board].kind if o.board in boards else "charuco",
            )
            for o in self.session.system.observations
            if o.camera == cam and o.frame == frame
        ]

    def _board_colours(self) -> dict:
        """
        One colour per CONFIGURED board, not per board found in this frame.

        Deriving it from the frame is the bug this replaces: a board missing
        from one image would shift every other board's colour in that image
        alone, so the same board changed colour as you clicked through the strip.
        """
        return board_colours(self.session.system.boards)

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
        self.image_view.set_image(
            draw_detections(img, dets, colours=self._board_colours()) if dets else img
        )
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
                boards = self.session.system.boards
                overlay = draw_detections(
                    img,
                    [
                        Detection(
                            board_id=obs.board,
                            point_ids=obs.point_ids,
                            image_points=obs.image_points,
                            kind=boards[obs.board].kind if obs.board in boards else "charuco",
                        )
                    ],
                    colours=self._board_colours(),
                )
                self.image_view.set_image(overlay)
                return
