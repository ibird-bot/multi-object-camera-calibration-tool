"""
The two windows a plug-in user needs: the object's point list, and what loaded.

`PointsDialog` edits an object's `id, x, y, z` table -- typed, pasted from a
spreadsheet or imported from a CSV -- and saves it as the points file the
object row points at. It validates with the same parser the calibration uses,
so a table it accepts is a table the run accepts.

`PluginsDialog` shows the folders searched, every plug-in that loaded and every
one that did not, with the reason: a plug-in that fails silently looks exactly
like one that was never installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from mlti_cal.plugins.point_object import PointFileError, format_points, is_planar, parse_points

#: The last three are optional per-point standard deviations, kept in the file
#: for a later version that refines point positions; empty means "not given".
COLUMNS = ("id", "x (m)", "y (m)", "z (m)", "sx (m)", "sy (m)", "sz (m)")


class PointsDialog(QDialog):
    """Edit an object's points; `saved_path` is set when the user saves."""

    def __init__(self, object_id: str, path: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Points of {object_id}")
        self.resize(560, 520)
        self.object_id = object_id
        self.saved_path: str | None = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            "One row per target: its id as your detector reports it, and its position "
            "in metres in the object's own frame. A flat object has z = 0 everywhere."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setObjectName("points_table")
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        self.status = QLabel("")
        self.status.setObjectName("points_status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        row = QHBoxLayout()
        for label, slot in (
            ("+ Row", self._add_row),
            ("- Remove", self._remove_rows),
            ("Import CSV...", self._import),
            ("Paste", self._paste),
        ):
            b = QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch()
        layout.addLayout(row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if path and Path(path).is_file():
            self._load_text(Path(path).read_text(encoding="utf-8-sig"), str(path))
        self.table.itemChanged.connect(lambda _item: self._validate())
        self._validate()

    # -- table <-> text ---------------------------------------------------
    def _rows(self) -> list[list[str]]:
        out = []
        for r in range(self.table.rowCount()):
            cells = [self.table.item(r, c) for c in range(len(COLUMNS))]
            out.append([c.text().strip() if c is not None else "" for c in cells])
        return [row for row in out if any(row)]

    def text(self) -> str:
        # Empty sigma cells are dropped, so a row is 4 values or 7, never 5.
        return "\n".join(",".join(c for c in row if c) for row in self._rows())

    def _load_text(self, text: str, source: str) -> bool:
        try:
            ids, xyz, sigma = parse_points(text, source)
        except PointFileError as exc:
            QMessageBox.warning(self, "Cannot read points", str(exc))
            return False
        blocked = self.table.blockSignals(True)
        self.table.setRowCount(0)
        for i, p, s in zip(ids, xyz, sigma, strict=True):
            r = self.table.rowCount()
            self.table.insertRow(r)
            values = [str(int(i)), *(f"{v:.9g}" for v in p)]
            values += [f"{v:.9g}" for v in s] if np.isfinite(s).all() else ["", "", ""]
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(value))
        self.table.blockSignals(blocked)
        self._validate()
        return True

    def _validate(self) -> bool:
        if not self._rows():
            self.status.setText("No points yet.")
            return False
        try:
            ids, xyz, _ = parse_points(self.text(), "table")
        except PointFileError as exc:
            self.status.setText(f"<span style='color:#c0392b'>{exc}</span>")
            return False
        flat = is_planar(xyz)
        if flat and not np.allclose(xyz[:, 2], 0.0):
            self.status.setText(
                "<span style='color:#c0392b'>These points are on one plane but not on "
                "z = 0. Express them in a frame where the object's surface is z = 0.</span>"
            )
            return False
        self.status.setText(f"{len(ids)} points, {'flat' if flat else '3D'} object.")
        return True

    # -- actions -----------------------------------------------------------
    def _add_row(self) -> None:
        self.table.insertRow(self.table.rowCount())

    def _remove_rows(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedItems()}, reverse=True):
            self.table.removeRow(r)
        self._validate()

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import points", "", "Points (*.csv *.txt);;All files (*)"
        )
        if path:
            self._load_text(Path(path).read_text(encoding="utf-8-sig"), path)

    def _paste(self) -> None:
        """Rows copied from a spreadsheet arrive tab-separated; the parser takes whitespace."""
        self._load_text(QGuiApplication.clipboard().text(), "clipboard")

    def _save(self) -> None:
        if not self._validate():
            QMessageBox.warning(self, "Not saved", self.status.text())
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save points", f"{self.object_id}.csv", "Points (*.csv)"
        )
        if not path:
            return
        ids, xyz, sigma = parse_points(self.text(), "table")
        Path(path).write_text(format_points(ids, xyz, sigma), encoding="utf-8")
        self.saved_path = path
        self.accept()


class PluginsDialog(QDialog):
    """Folders searched, what loaded, what failed and why."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from mlti_cal.plugins.check import list_plugins

        self.setWindowTitle("Plug-ins")
        self.resize(720, 460)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Drop a plug-in file in a plug-in folder and restart. The contracts are in "
            "docs/extending.md; <code>mlti-cal plugins check FILE</code> tests one file."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        text = QPlainTextEdit("\n".join(list_plugins()))
        text.setObjectName("plugins_text")
        text.setReadOnly(True)
        layout.addWidget(text)
        row = QHBoxLayout()
        open_btn = QPushButton("Open plug-in folder")
        open_btn.setToolTip("Creates ~/.mlti_cal/plugins if it does not exist yet.")
        open_btn.clicked.connect(self._open_folder)
        row.addWidget(open_btn)
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)

    @staticmethod
    def _open_folder() -> None:
        folder = Path.home() / ".mlti_cal" / "plugins"
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
