"""
The `Option` record: one exposed knob, its default, and when to reach for it.

Lives here rather than in `solvers/` because it is not a solver concept. The
detector, the initialiser and the report engine all make choices that used to
be hardcoded, and every one of them is described by the same four things: a
name, a type, a default, and guidance. Sharing the record means the GUI can
render any catalog with one widget builder, and `--describe` can print any
catalog with one renderer.

The `when` text is the reason this exists. A dropdown of seven enum names helps
nobody; the note that says which one to pick, and why, is the product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Option:
    name: str
    kind: str  # "choice" | "int" | "float" | "bool"
    default: Any
    when: str
    choices: list[str] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    per_choice: dict[str, str] = field(default_factory=dict)
    #: Optional grouping key, read only when a catalog is rendered as TABLE
    #: COLUMNS rather than as a form -- which today is the board-geometry
    #: catalog on each detector kind. Two kinds whose fields mean the same thing
    #: under different names (a Charuco board's `squares_x`, a dot grid's
    #: `circles_x`) share a column by declaring the same `column`. Left None the
    #: field gets a column of its own, named after it.
    column: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "when": self.when,
            "choices": self.choices,
            "min": self.minimum,
            "max": self.maximum,
            "per_choice": self.per_choice,
            "column": self.column,
        }


def render_options(title: str, summary: str, options: list[Option]) -> str:
    """One catalog as text. Shared by every `--describe` path in the CLI."""
    lines = [f"=== {title} ===", summary, ""] if summary else [f"=== {title} ===", ""]
    for opt in options:
        bounds = ""
        if opt.minimum is not None or opt.maximum is not None:
            bounds = f"  range=[{opt.minimum}, {opt.maximum}]"
        lines.append(f"  {opt.name}  [{opt.kind}]  default={opt.default!r}{bounds}")
        lines.append(f"      {opt.when}")
        for choice, note in opt.per_choice.items():
            lines.append(f"        - {choice}: {note}")
        lines.append("")
    return "\n".join(lines)
