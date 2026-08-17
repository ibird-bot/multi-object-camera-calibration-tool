"""
One catalog in its own window, with an explicit Apply.

The counterpart to `DetectorSettingsDialog`, which needs a tab per detector
kind. This one shows a single catalog and is what everything else uses.

Apply is the point. An inline panel takes effect the instant a spinbox ticks
past a value on its way somewhere else, so a half-typed threshold is briefly
the live setting. Here nothing changes until the user says so, and what they
said is validated as a whole -- which is the only way to check the constraints
that span two fields, like a window minimum that must stay below its maximum.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox, QVBoxLayout

from mlti_cal.gui.settings_panel import SettingsPanel
from mlti_cal.options import Option


class CatalogSettingsDialog(QDialog):
    """
    Edit one `Option` catalog. Emits `applied` with the validated values.

    `build` turns the raw widget values into whatever settings object the
    caller wants, and is where validation happens: it is handed the values and
    may raise ValueError, which is shown to the user instead of being applied.
    """

    applied = Signal(object)

    def __init__(
        self,
        title: str,
        summary: str,
        catalog: list[Option],
        values: dict | None = None,
        build: Callable[[dict], Any] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 700)
        self._build_fn = build
        self.panel = SettingsPanel(title, summary, catalog, expanded=True)
        if values:
            self.panel.set_values(values)

        root = QVBoxLayout(self)
        root.addWidget(self.panel, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        root.addWidget(buttons)

    def values(self) -> dict:
        return self.panel.values()

    def apply(self) -> bool:
        raw = self.panel.values()
        try:
            payload = self._build_fn(raw) if self._build_fn else raw
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return False
        self.applied.emit(payload)
        return True

    def _ok(self) -> None:
        if self.apply():
            self.accept()
