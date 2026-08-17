"""
One log for the whole application.

Every stage used to report somewhere different: detection wrote to the status
bar, the bootstrap and the solve wrote into a box inside the Optimization tab,
the report wrote into that same box through a forwarding hook, and the noise
measurement wrote only into its own dialog. So the transcript of a session was
split across three widgets, two of which you had to be on the right tab to see.

This is the single destination. Each line carries the stage that produced it,
so the log can be read whole or filtered down to one stage, and it survives
switching tabs because it does not live on one.

Entries are kept as data, not as text, which is what makes the filter possible:
re-rendering from the stored entries shows exactly the lines a stage produced
rather than grepping formatted output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

#: Stages that write here, in pipeline order rather than alphabetical: the
#: filter should read like the workflow.
SOURCES = ("setup", "detect", "noise", "init", "problem", "solve", "crossval", "report")

#: Lines kept. Bounded because a live solve emits one per iteration and an
#: unbounded log turns a long run into a memory leak.
MAX_ENTRIES = 5000


@dataclass(frozen=True)
class LogEntry:
    when: float
    source: str
    text: str

    def render(self, show_time: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.when)) if show_time else ""
        head = f"{stamp} " if stamp else ""
        return f"{head}{self.source:<8} | {self.text}"


class LogConsole(QWidget):
    """
    The shared transcript, with a stage filter.

    Exposes `appendPlainText` and `toPlainText` so it can stand in for the
    QPlainTextEdit this replaced without every call site being rewritten.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries: list[LogEntry] = []
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 4)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("show:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("everything", None)
        for source in SOURCES:
            self.filter_combo.addItem(source, source)
        self.filter_combo.setToolTip(
            "Narrow the log to one stage. Nothing is discarded -- the other "
            "lines are still there when you switch back."
        )
        self.filter_combo.currentIndexChanged.connect(self._rerender)
        bar.addWidget(self.filter_combo)

        self.time_check = QCheckBox("timestamps")
        self.time_check.setChecked(True)
        self.time_check.toggled.connect(self._rerender)
        bar.addWidget(self.time_check)

        self.follow_check = QCheckBox("follow")
        self.follow_check.setChecked(True)
        self.follow_check.setToolTip(
            "Scroll to the newest line. Untick to read the middle of a running "
            "solve without being dragged to the bottom every iteration."
        )
        bar.addWidget(self.follow_check)

        bar.addStretch()
        for label, slot, tip in (
            ("Copy", self.copy_all, "Copy the visible lines to the clipboard"),
            ("Save...", self.save_as, "Write the visible lines to a text file"),
            ("Clear", self.clear, "Discard every line"),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        root.addLayout(bar)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(MAX_ENTRIES)
        # Fixed width: the iteration table and the build summary are column
        # aligned and only line up in a monospace face.
        self.text.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        root.addWidget(self.text, 1)

    # -- writing ----------------------------------------------------------
    def write(self, source: str, message: str) -> None:
        """One line per line of `message`, all tagged with `source`."""
        now = time.time()
        for line in str(message).split("\n"):
            entry = LogEntry(now, source, line)
            self.entries.append(entry)
            if self._visible(entry):
                self._emit(entry)
        if len(self.entries) > MAX_ENTRIES:
            del self.entries[: len(self.entries) - MAX_ENTRIES]

    def appendPlainText(self, message: str) -> None:  # noqa: N802 (QPlainTextEdit API)
        """Untagged write, for code that has not been given a stage."""
        self.write("app", message)

    def toPlainText(self) -> str:  # noqa: N802 (QPlainTextEdit API)
        return self.text.toPlainText()

    # -- actions ----------------------------------------------------------
    def clear(self) -> None:
        self.entries.clear()
        self.text.clear()

    def copy_all(self) -> None:
        QGuiApplication.clipboard().setText(self.toPlainText())

    def save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save log", "log.txt", "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.toPlainText())

    # -- internals --------------------------------------------------------
    def _visible(self, entry: LogEntry) -> bool:
        wanted = self.filter_combo.currentData()
        return wanted is None or entry.source == wanted

    def _emit(self, entry: LogEntry) -> None:
        self.text.appendPlainText(entry.render(self.time_check.isChecked()))
        if self.follow_check.isChecked():
            bar = self.text.verticalScrollBar()
            bar.setValue(bar.maximum())

    def _rerender(self) -> None:
        self.text.clear()
        show_time = self.time_check.isChecked()
        for entry in self.entries:
            if self._visible(entry):
                self.text.appendPlainText(entry.render(show_time))


class SourceLog:
    """
    A console bound to one stage.

    Lets a view keep calling `self.log.appendPlainText(...)` while every line it
    writes is tagged, so the stage a message came from is recorded at the point
    it is written rather than guessed from its wording later.
    """

    def __init__(self, console: LogConsole, source: str = "app"):
        self.console = console
        self.source = source

    def appendPlainText(self, message: str) -> None:  # noqa: N802 (QPlainTextEdit API)
        self.console.write(self.source, message)

    def write(self, source: str, message: str) -> None:
        self.console.write(source, message)

    def toPlainText(self) -> str:  # noqa: N802 (QPlainTextEdit API)
        return self.console.toPlainText()

    def setMaximumBlockCount(self, _n: int) -> None:  # noqa: N802 (QPlainTextEdit API)
        """Accepted and ignored: the console bounds itself."""

    def setStyleSheet(self, _s: str) -> None:  # noqa: N802 (QWidget API)
        """Accepted and ignored: the console owns its own appearance."""


def with_source(log, source: str):
    """Point a `SourceLog` at a stage. Safe when handed a plain widget."""
    if isinstance(log, SourceLog):
        log.source = source
    return log
