"""
Detector settings, one tab per detector kind.

Reached from the menu bar rather than the Setup tab, because these belong to
the DETECTOR and not to any one calibration object: two charuco boards in the
same session are found by the same detector with the same parameters, so a
per-board copy of these controls would be three copies of one truth.

Kinds that are not implemented get a tab too, holding the reason. A settings
window that silently lists only charuco would leave the user unable to tell
"this detector has no options" from "this detector does not exist yet".
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mlti_cal.detectors.registry import DETECTOR_KINDS, DetectorKind
from mlti_cal.gui.settings_panel import SettingsPanel


class DetectorSettingsDialog(QDialog):
    """
    Edit every detector's settings. Emits `settings_applied` on OK/Apply.

    Nothing is written back until the user applies: a half-typed threshold
    should not silently become the setting that the next detection run uses.
    """

    settings_applied = Signal(dict)  # kind id -> settings object

    def __init__(self, settings: dict[str, Any], in_use: set[str] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detector settings")
        self.resize(820, 720)
        self._in_use = set(in_use or ())
        self.panels: dict[str, SettingsPanel] = {}
        self._build(settings)

    def _build(self, settings: dict[str, Any]) -> None:
        root = QVBoxLayout(self)
        blurb = QLabel(
            "Settings belong to a detector, not to a board: every Charuco board in "
            "this session is found by the same detector with the same parameters.\n\n"
            "Detection quality bounds everything downstream -- the solver cannot "
            "recover accuracy the detector never found -- so changing anything here "
            "invalidates detections already on screen."
        )
        blurb.setWordWrap(True)
        root.addWidget(blurb)

        self.tabs = QTabWidget()
        for kind in DETECTOR_KINDS.values():
            self.tabs.addTab(self._page(kind, settings.get(kind.id)), self._tab_label(kind))
            if not kind.implemented:
                self.tabs.setTabEnabled(self.tabs.count() - 1, False)
                self.tabs.setTabToolTip(self.tabs.count() - 1, kind.unavailable_reason)
        root.addWidget(self.tabs, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        root.addWidget(buttons)

    def _tab_label(self, kind: DetectorKind) -> str:
        if not kind.implemented:
            return f"{kind.label} (not available)"
        return f"{kind.label} • in use" if kind.id in self._in_use else kind.label

    def _page(self, kind: DetectorKind, current: Any) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        description = QLabel(kind.description)
        description.setWordWrap(True)
        description.setStyleSheet("color: gray;")
        layout.addWidget(description)

        if not kind.implemented:
            note = QLabel(
                f"<b>Not available:</b> {kind.unavailable_reason}.<br><br>"
                f"It is listed here so the set of targets this tool intends to "
                f"support is visible, rather than implied by its absence."
            )
            note.setWordWrap(True)
            note.setTextFormat(Qt.RichText)
            layout.addWidget(note)
            layout.addStretch()
            return page

        panel = SettingsPanel(kind.label, kind.summary, kind.catalog, expanded=True)
        if current is not None:
            panel.set_values(current.to_dict())
        self.panels[kind.id] = panel
        layout.addWidget(panel, 1)
        return page

    # ------------------------------------------------------------------
    def values(self) -> dict[str, Any]:
        """Current widget values as settings objects. Raises ValueError if invalid."""
        out: dict[str, Any] = {}
        for kind_id, panel in self.panels.items():
            out[kind_id] = DETECTOR_KINDS[kind_id].settings_cls(**panel.values())
        return out

    def apply(self) -> bool:
        try:
            values = self.values()
        except ValueError as exc:
            # The dataclass validates combinations a spinbox range cannot, such
            # as a threshold window whose minimum exceeds its maximum.
            QMessageBox.critical(self, "Invalid detector settings", str(exc))
            return False
        self.settings_applied.emit(values)
        return True

    def _ok(self) -> None:
        if self.apply():
            self.accept()
