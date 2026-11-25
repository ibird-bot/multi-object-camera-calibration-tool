"""
Small reusable pieces of the interface, styled by `theme.py`.

Each one is a plain widget with an object name, so the theme can style it and
the demo-video director can find and frame it on its own.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mlti_cal.gui import theme


def dot_icon(colour: str, size: int = 10) -> QIcon:
    """A filled circle, for status marks."""
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(colour))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(2, 2, size * 2 - 4, size * 2 - 4)
    painter.end()
    return QIcon(pm)


def set_role(label: QLabel, role: str) -> QLabel:
    label.setProperty("role", role)
    return label


def primary(button: QPushButton) -> QPushButton:
    """The one action a page is for."""
    button.setProperty("primary", True)
    return button


def plain_table(table: QTableWidget) -> QTableWidget:
    """
    A table read as rows: no row numbers, no cell grid, left-aligned headers.

    The row numbers repeated what the first column already names, and a grid
    around cells that hold bordered inputs drew every box twice.
    """
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(32)
    table.setShowGrid(False)
    table.horizontalHeader().setDefaultAlignment(
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )
    table.horizontalHeader().setHighlightSections(False)
    return table


class StepRail(QListWidget):
    """
    The workflow as a list down the left: Setup, Solve, Report.

    Each step carries a status mark -- not started, done, or needs attention --
    so where the session stands is readable without visiting each page.
    """

    STATES = ("todo", "done", "attention")

    def __init__(self, steps: list[str], parent=None):
        super().__init__(parent)
        self.setObjectName("StepRail")
        self.setFixedWidth(196)
        self.setIconSize(QSize(10, 10))
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._states = ["todo"] * len(steps)
        for text in steps:  # the dot is the marker; a number beside it was a second one
            self.addItem(QListWidgetItem(text))
        self.refresh_marks()

    def set_state(self, index: int, state: str) -> None:
        if state not in self.STATES:
            raise ValueError(state)
        if self._states[index] != state:
            self._states[index] = state
            self.refresh_marks()

    def step_state(self, index: int) -> str:
        return self._states[index]

    def refresh_marks(self) -> None:
        p = theme.current()
        colour = {"todo": p.border, "done": p.good, "attention": p.warn}
        tips = {"todo": "not done yet", "done": "done", "attention": "needs a look"}
        for i, state in enumerate(self._states):
            item = self.item(i)
            item.setIcon(dot_icon(colour[state]))
            item.setToolTip(tips[state])


class MetricCard(QFrame):
    """A titled number with a caption: the shape of every result worth a glance."""

    def __init__(self, title: str, name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setAccessibleName(name)
        self.setProperty("card", name)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        self.title = set_role(QLabel(title), "caption")
        self.value = set_role(QLabel("--"), "metric")
        self.value.setObjectName(f"{name}_value")
        self.caption = set_role(QLabel(""), "caption")
        self.caption.setWordWrap(True)
        for w in (self.title, self.value, self.caption):
            lay.addWidget(w)

    def set(self, value: str, caption: str = "", tone: str | None = None) -> None:
        """
        `tone` "warn" or "bad" colours the value; "good" and None leave it plain.
        Colour is kept for what needs attention -- a page of green numbers
        says nothing.
        """
        self.value.setText(value)
        self.caption.setText(caption)
        p = theme.current()
        colour = {"warn": p.warn, "bad": p.bad}.get(tone or "", "")
        self.value.setStyleSheet(f"color: {colour};" if colour else "")


class Banner(QFrame):
    """A one-line verdict in plain words, with a status dot for its severity."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Banner")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        head = QHBoxLayout()
        head.setSpacing(8)
        self.mark = QLabel()
        self.headline = set_role(QLabel(""), "heading")
        head.addWidget(self.mark)
        head.addWidget(self.headline, 1)
        self.detail = set_role(QLabel(""), "muted")
        self.detail.setWordWrap(True)
        lay.addLayout(head)
        lay.addWidget(self.detail)
        self._tone = "neutral"

    def set(self, headline: str, detail: str, tone: str) -> None:
        self.headline.setText(headline)
        self.detail.setText(detail)
        self._tone = tone
        self.restyle()

    def restyle(self) -> None:
        p = theme.current()
        colour = {"good": p.good, "warn": p.warn, "bad": p.bad}.get(self._tone)
        self.mark.setVisible(colour is not None)
        if colour is not None:
            self.mark.setPixmap(dot_icon(colour, 7).pixmap(14, 14))


class Section(QWidget):
    """A titled block that folds away: advanced settings start folded."""

    toggled = Signal(bool)

    def __init__(self, title: str, content: QWidget, expanded: bool = False, parent=None):
        super().__init__(parent)
        self.button = QToolButton()
        self.button.setObjectName("SectionToggle")
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.content = content
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.button)
        lay.addWidget(content)
        self.button.toggled.connect(self._toggle)
        self._toggle(expanded)

    def _toggle(self, expanded: bool) -> None:
        self.button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.content.setVisible(expanded)
        self.toggled.emit(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.button.setChecked(expanded)


class Segmented(QWidget):
    """A row of mutually exclusive buttons: the page switcher inside a view."""

    changed = Signal(int)

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: list[QPushButton] = []
        for i, label in enumerate(labels):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setProperty("segment", True)
            self.group.addButton(b, i)
            self.buttons.append(b)
            lay.addWidget(b)
        lay.addStretch()
        self.buttons[0].setChecked(True)
        self.group.idClicked.connect(self.changed.emit)

    def set_index(self, index: int) -> None:
        self.buttons[index].setChecked(True)
        self.changed.emit(index)

    def set_label(self, index: int, label: str) -> None:
        self.buttons[index].setText(label)
