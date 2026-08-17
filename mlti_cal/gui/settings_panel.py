"""
One widget that renders any `Option` catalog, and reads the values back.

Every knob in this application -- detection, initialisation, solver, report --
is described by the same `Option` record, so there is exactly one place that
turns those records into widgets. That is what keeps the tooltip a user reads
identical to the text `--describe` prints: both come from the catalog, and
neither is retyped here.

Bounds come from the catalog too. `Option.minimum` on `min_points_for_pnp` is 4
because a pose genuinely needs 4 points; wiring it into `setRange` means the
spinbox cannot produce an OpenCV assertion failure in the first place.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from mlti_cal.options import Option


def option_tooltip(opt: Option) -> str:
    """The catalog's guidance, with per-choice notes appended."""
    tip = opt.when
    if opt.per_choice:
        tip += "\n\n" + "\n".join(f"* {k}: {v}" for k, v in opt.per_choice.items())
    if opt.minimum is not None or opt.maximum is not None:
        tip += f"\n\nAllowed range: {opt.minimum} to {opt.maximum}."
    return tip


def make_option_widget(opt: Option) -> QWidget:
    """One widget for one option, bounded and defaulted from the catalog."""
    if opt.kind == "choice":
        w = QComboBox()
        w.addItems(opt.choices)
        # Some defaults carry an explanation ("exact (auto below 2000 ...)");
        # the first token is the value.
        default = str(opt.default).split(" ")[0]
        if default in opt.choices:
            w.setCurrentText(default)
    elif opt.kind == "bool":
        w = QCheckBox()
        w.setChecked(bool(opt.default))
    elif opt.kind == "int":
        w = QSpinBox()
        w.setRange(
            int(opt.minimum) if opt.minimum is not None else 0,
            int(opt.maximum) if opt.maximum is not None else 100000,
        )
        w.setValue(int(opt.default))
    else:
        w = QDoubleSpinBox()
        w.setRange(
            float(opt.minimum) if opt.minimum is not None else -1e12,
            float(opt.maximum) if opt.maximum is not None else 1e12,
        )
        # Small thresholds (1e-10 tolerances, 1e-6 epsilons) need real decimals
        # and a step that is not larger than the value itself.
        magnitude = abs(float(opt.default)) or 1.0
        if magnitude < 0.01:
            w.setDecimals(14)
            w.setSingleStep(magnitude / 10.0)
        else:
            w.setDecimals(4)
            w.setSingleStep(max(magnitude / 20.0, 0.001))
        w.setValue(float(opt.default))
    return w


def widget_value(w: QWidget):
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
        return w.value()
    raise TypeError(f"no value reader for {type(w).__name__}")


def set_widget_value(w: QWidget, value) -> None:
    if isinstance(w, QComboBox):
        w.setCurrentText(str(value))
    elif isinstance(w, QCheckBox):
        w.setChecked(bool(value))
    elif isinstance(w, QSpinBox):
        w.setValue(int(value))
    elif isinstance(w, QDoubleSpinBox):
        w.setValue(float(value))


def fill_form(form: QFormLayout, options: list[Option], skip: tuple[str, ...] = ()) -> dict:
    """Add a row per option. Returns name -> widget."""
    widgets: dict = {}
    for opt in options:
        if opt.name in skip:
            continue
        w = make_option_widget(opt)
        tip = option_tooltip(opt)
        w.setToolTip(tip)
        label = QLabel(opt.name.replace("_", " "))
        label.setToolTip(tip)
        form.addRow(label, w)
        widgets[opt.name] = w
    return widgets


class SettingsPanel(QGroupBox):
    """
    A collapsible group box holding one catalog.

    Collapsed by default. These are advanced knobs: showing fifteen spinboxes
    the moment the window opens would bury the three controls most runs
    actually need, which is its own kind of opacity.
    """

    def __init__(
        self,
        title: str,
        summary: str,
        options: list[Option],
        parent=None,
        expanded: bool = False,
        on_change: Callable[[], None] | None = None,
    ):
        super().__init__(title, parent)
        self.options = options
        self._on_change = on_change

        outer = QFormLayout(self)
        blurb = QLabel(summary)
        blurb.setWordWrap(True)
        outer.addRow(blurb)

        self._toggle = QPushButton("Show settings")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.toggled.connect(self._on_toggled)
        self._reset = QPushButton("Reset to defaults")
        self._reset.clicked.connect(self.reset)
        outer.addRow(self._toggle, self._reset)

        self.body = QWidget(self)
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.widgets = fill_form(self.form, options)
        outer.addRow(self.body)

        for w in self.widgets.values():
            self._connect_change(w)
        self.body.setVisible(expanded)
        self._sync_toggle_text()

    # -- state ------------------------------------------------------------
    def values(self) -> dict:
        return {name: widget_value(w) for name, w in self.widgets.items()}

    def set_values(self, values: dict) -> None:
        for name, value in (values or {}).items():
            w = self.widgets.get(name)
            if w is not None:
                set_widget_value(w, value)

    def reset(self) -> None:
        self.set_values(
            {
                o.name: str(o.default).split(" ")[0] if o.kind == "choice" else o.default
                for o in self.options
            }
        )

    def changed_from_default(self) -> dict:
        """Only the options the user actually moved. Used for the summary line."""
        out = {}
        for opt in self.options:
            if opt.name not in self.widgets:
                continue
            current = widget_value(self.widgets[opt.name])
            default = str(opt.default).split(" ")[0] if opt.kind == "choice" else opt.default
            if opt.kind == "float":
                if abs(float(current) - float(default)) > 1e-15:
                    out[opt.name] = current
            elif current != default:
                out[opt.name] = current
        return out

    def summary_line(self) -> str:
        changed = self.changed_from_default()
        if not changed:
            return "all defaults"
        return ", ".join(f"{k}={v}" for k, v in changed.items())

    # -- internals --------------------------------------------------------
    def _connect_change(self, w: QWidget) -> None:
        if self._on_change is None:
            return
        if isinstance(w, QComboBox):
            w.currentTextChanged.connect(lambda *_: self._on_change())
        elif isinstance(w, QCheckBox):
            w.toggled.connect(lambda *_: self._on_change())
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.valueChanged.connect(lambda *_: self._on_change())

    def _on_toggled(self, shown: bool) -> None:
        self.body.setVisible(shown)
        self._sync_toggle_text()

    def _sync_toggle_text(self) -> None:
        n = len(self.widgets)
        self._toggle.setText(
            f"Hide {n} settings" if self._toggle.isChecked() else f"Show {n} settings"
        )
