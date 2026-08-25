"""
Which detectors exist, what each one needs, and how a third party adds one.

Settings belong to a DETECTOR, not to "detection". This registry is the list of
detector kinds, each carrying its own board-geometry fields, its own settings
class and its own catalog.

One declaration, not nine
    A detector used to be spelled out in nine places: the registry, a config key
    table, a spec-builder function, per-kind branches in `_collect_config` and
    `_apply_config`, four hardcoded per-kind dicts in the GUI table, and the
    pipeline's ordering tuple. Adding a target type meant editing all of them,
    and a third party could not add one at all without patching the package.

    A `DetectorKind` is now the single declaration. Everything downstream reads
    it: the config parses boards through `spec_cls`, the GUI builds its table
    columns from `board_fields`, the pipeline orders detectors by `coded`, and
    the overlay picks its glyph from `glyph`. Nothing else knows the id of any
    particular detector.

Registration mirrors the solver backends
    Each built-in detector module ends with a `register_detector(...)` call, and
    `mlti_cal/detectors/__init__.py` imports those modules for the side effect --
    the same idiom `solvers/__init__.py` already uses. There is deliberately not
    a second plugin convention in this codebase.

A kind may be listed with `implemented=False` and a reason, which greys it out
in the picker rather than hiding it. Every built-in is implemented today, so
nothing uses that path; it is kept because the alternative is silently dropping
a target type from the UI when one is next started.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from mlti_cal.options import Option

#: Directories scanned for third-party detectors. A `.py` file dropped in one of
#: these is imported at startup and any `register_detector` call it makes takes
#: effect -- which is what makes a new detector appear in the GUI on the next
#: run without touching the installed package.
PLUGIN_DIRS: list[Path] = [Path.home() / ".mlti_cal" / "detectors"]

#: Extra plugin directories, os.pathsep-separated. For tests, and for running a
#: detector out of a working tree without installing it.
PLUGIN_PATH_ENV = "MLTI_CAL_DETECTOR_PATH"

#: Entry-point group for detectors shipped as installed packages.
ENTRY_POINT_GROUP = "mlti_cal.detectors"

#: Plugins that raised while loading: source -> formatted traceback. Recorded
#: rather than raised, so one broken plugin cannot stop the app from starting --
#: and recorded rather than swallowed, because a plugin that silently fails to
#: load looks exactly like one the user never installed.
PLUGIN_ERRORS: dict[str, str] = {}


@dataclass(frozen=True)
class DetectorKind:
    """One kind of calibration target, and everything the app needs to use it."""

    id: str
    label: str
    description: str

    #: The board's GEOMETRY as the user enters it: one Option per field of
    #: `spec_cls`, excluding `id`. Rendered as table columns by the GUI and used
    #: as the JSON keys of a saved board, so these names are user-visible and are
    #: the detector's own vocabulary -- a dot grid says `circles_x`, never
    #: `squares_x`. Fields of different kinds that mean the same thing share a
    #: table column by declaring the same `Option.column`.
    board_fields: list[Option] = field(default_factory=list)

    #: Dataclass describing one board. Built as `spec_cls(id=..., **fields)`.
    spec_cls: type | None = None

    #: Detects every board of this kind in one image. Built as
    #: `group_cls(specs, settings=settings)`; must expose `specs`, `detectors`,
    #: `settings`, `object_points(board_id)`, `mask_geometry` and
    #: `detect_all(image)`.
    group_cls: type | None = None

    settings_cls: type | None = None
    catalog: list[Option] = field(default_factory=list)
    summary: str = ""

    #: True when every feature carries its own identity, as Charuco's markers do.
    #: Coded kinds run FIRST and are never blocked by clutter; uncoded kinds run
    #: afterwards, on an image with the located boards painted out. This is the
    #: only thing the pipeline needs in order to order them.
    coded: bool = False

    #: Overlay marker shape: "circle" | "square" | "diamond" | "triangle".
    #: Colour identifies the BOARD, so the glyph is what identifies the detector.
    glyph: str = "circle"

    implemented: bool = True
    #: Why this kind cannot be used yet. Empty when it can.
    unavailable_reason: str = ""

    #: Where this kind came from, shown in the GUI so a user can tell a plugin
    #: from a built-in.
    origin: str = "built-in"

    def default_settings(self) -> Any:
        if self.settings_cls is None:
            raise ValueError(f"detector {self.id!r} has no settings: {self.unavailable_reason}")
        return self.settings_cls()

    def default_board(self, board_id: str) -> dict:
        """A new board of this kind, at its declared defaults."""
        return {"id": board_id, "kind": self.id, **{o.name: o.default for o in self.board_fields}}

    def build_spec(self, board: dict):
        """
        One board dict -> one spec, validated against THIS kind's fields.

        Unknown keys are rejected by name rather than ignored: a config saying
        `squares_x` for a dot grid is a real mistake, and silently dropping it
        would calibrate against a board the user did not describe.
        """
        if self.spec_cls is None:
            raise ValueError(f"detector {self.id!r} cannot build specs: {self.unavailable_reason}")
        declared = {opt.name for opt in self.board_fields}
        # Spec fields carrying a default are accepted from a config without
        # being shown in the GUI -- `min_markers` is one. Expert knobs stay
        # editable by hand without adding a column nobody touches.
        optional = _defaulted_fields(self.spec_cls) - {"id"}
        values = {k: v for k, v in board.items() if k not in ("kind", "id")}
        unknown = set(values) - declared - optional
        if unknown:
            raise TypeError(
                f"board {board.get('id', '?')!r}: a {self.label} has no field(s) "
                f"{sorted(unknown)}; it takes {sorted(declared)}"
            )
        # A field the spec gives a default may be omitted from a config -- the
        # GUI always writes it, but a hand-written board that leaves `grid_type`
        # off should get the spec's own default rather than an error.
        missing = [k for k in declared if k not in values and k not in optional]
        if missing:
            raise TypeError(f"board {board.get('id', '?')!r}: a {self.label} needs {missing}")
        return self.spec_cls(id=board["id"], **values)

    def build_group(self, specs: list, settings: Any = None):
        """Every board of this kind, in one detector."""
        if self.group_cls is None:
            raise ValueError(f"detector {self.id!r} has no detector: {self.unavailable_reason}")
        return self.group_cls(list(specs), settings=settings or self.default_settings())


def _defaulted_fields(spec_cls: type) -> set[str]:
    if not is_dataclass(spec_cls):
        return set()
    return {
        f.name
        for f in dataclasses.fields(spec_cls)
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING  # type: ignore[misc]
    }


DETECTOR_KINDS: dict[str, DetectorKind] = {}


def register_detector(kind: DetectorKind) -> DetectorKind:
    """
    Add a detector kind to the registry.

    Re-registering an id REPLACES it, which is what lets a plugin deliberately
    override a built-in -- a better Charuco detector, say -- instead of having
    to pick another name and sit beside the original in the menu.
    """
    _validate(kind)
    DETECTOR_KINDS[kind.id] = kind
    return kind


def _validate(kind: DetectorKind) -> None:
    """
    Reject a malformed kind at registration, where the traceback names the
    plugin, rather than at first use somewhere deep in the GUI.
    """
    if not kind.id or not kind.id.isidentifier():
        raise ValueError(f"detector id {kind.id!r} must be a valid Python identifier")
    if not kind.label:
        raise ValueError(f"detector {kind.id!r} has no label")
    if not kind.description:
        raise ValueError(f"detector {kind.id!r} does not say what it is")
    if not kind.implemented:
        if not kind.unavailable_reason:
            raise ValueError(f"detector {kind.id!r} is disabled with no reason given")
        return
    for attr in ("spec_cls", "group_cls", "settings_cls"):
        if getattr(kind, attr) is None:
            raise ValueError(f"detector {kind.id!r} is implemented but has no {attr}")
    if not kind.board_fields:
        raise ValueError(f"detector {kind.id!r} declares no board_fields")
    names = [opt.name for opt in kind.board_fields]
    if len(names) != len(set(names)):
        raise ValueError(f"detector {kind.id!r} declares a board field twice")
    if "id" in names:
        raise ValueError(f"detector {kind.id!r}: every board has an 'id', do not declare it")
    documented = {o.name for o in kind.catalog}
    settings_fields = set(kind.settings_cls().to_dict())
    if documented != settings_fields:
        raise ValueError(
            f"detector {kind.id!r}: catalog and settings disagree -- undocumented "
            f"{sorted(settings_fields - documented)}, documented but absent "
            f"{sorted(documented - settings_fields)}"
        )


# -- lookups ----------------------------------------------------------------
def implemented_kinds() -> list[DetectorKind]:
    return [k for k in DETECTOR_KINDS.values() if k.implemented]


def coded_kinds() -> list[DetectorKind]:
    return [k for k in implemented_kinds() if k.coded]


def uncoded_kinds() -> list[DetectorKind]:
    return [k for k in implemented_kinds() if not k.coded]


def get_kind(kind_id: str) -> DetectorKind:
    if kind_id not in DETECTOR_KINDS:
        raise KeyError(f"unknown detector kind {kind_id!r}; known: {sorted(DETECTOR_KINDS)}")
    return DETECTOR_KINDS[kind_id]


def default_detector_settings() -> dict[str, Any]:
    """One settings object per implemented kind, all at their defaults."""
    return {k.id: k.default_settings() for k in implemented_kinds()}


def detector_settings_from_dict(data: dict | None) -> dict[str, Any]:
    """
    Rebuild per-kind settings from JSON.

    Unknown kinds are dropped rather than raising: a config written on a machine
    where a plugin was installed must still load here, minus the part this build
    cannot use.
    """
    out = default_detector_settings()
    for kind_id, values in (data or {}).items():
        kind = DETECTOR_KINDS.get(kind_id)
        if kind is None or not kind.implemented:
            continue
        out[kind_id] = kind.settings_cls.from_dict(values)
    return out


def detector_settings_to_dict(settings: dict[str, Any]) -> dict:
    return {kind_id: value.to_dict() for kind_id, value in sorted(settings.items())}


# -- plugin discovery -------------------------------------------------------
def plugin_dirs() -> list[Path]:
    dirs = list(PLUGIN_DIRS)
    extra = os.environ.get(PLUGIN_PATH_ENV, "")
    return dirs + [Path(p) for p in extra.split(os.pathsep) if p.strip()]


def load_plugins(on_error: Callable[[str, str], None] | None = None) -> list[str]:
    """
    Import every third-party detector: `.py` files in the plugin directories,
    then installed packages advertising the `mlti_cal.detectors` entry point.

    Returns the ids newly registered. Failures are recorded in `PLUGIN_ERRORS`
    and handed to `on_error` rather than raised -- one broken plugin must not
    stop the application from starting, and must not be invisible either.
    """
    before = set(DETECTOR_KINDS)
    for directory in plugin_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            _load_plugin_file(path, on_error)
    _load_entry_points(on_error)
    return sorted(set(DETECTOR_KINDS) - before)


def _record_error(where: str, on_error: Callable[[str, str], None] | None) -> None:
    detail = traceback.format_exc()
    PLUGIN_ERRORS[where] = detail
    if on_error is not None:
        on_error(where, detail)


def _load_plugin_file(path: Path, on_error: Callable[[str, str], None] | None) -> None:
    module_name = f"mlti_cal_plugin_{path.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        _record_error(str(path), on_error)


def _load_entry_points(on_error: Callable[[str, str], None] | None) -> None:
    try:
        from importlib.metadata import entry_points

        found = list(entry_points(group=ENTRY_POINT_GROUP))
    except Exception:  # pragma: no cover - importlib.metadata is stdlib
        return
    for ep in found:
        try:
            loaded = ep.load()
            # An entry point may register on import, or be a callable that does.
            if callable(loaded):
                loaded()
        except Exception:
            _record_error(f"entry point {ep.name}", on_error)
