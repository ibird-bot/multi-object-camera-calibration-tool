"""
A detector written as ONE function, adapted to the detector registry.

The plug-in supplies::

    NAME = "my_detector"
    SETTINGS = {"min_diameter_px": 6.0}       # optional
    def detect(image, settings):              # image: 2D uint8, settings: dict
        return points                         # (N, 3) float: [target_id, x, y]

and this module builds everything else the registry asks of a detector: a
`DetectorKind` whose objects are point lists (`PointObjectSpec`), a settings
class and catalog generated from `SETTINGS`, and a group detector that calls the
function once per image and hands each row to the object owning that id.

Two things here are the whole correctness argument:

- Target ids are NOT point ids. `Detection.point_ids` index rows of the
  object's point array; a plug-in reports the ids its decoder read. They are
  translated through a per-object id -> row map, so ids may be sparse,
  unsorted and shared out over several objects.
- Every return value goes through `contract.validate_detections`. A result
  that breaks the contract costs that image, with a message naming the
  plug-in and the rule -- it is never patched up and used.
"""

from __future__ import annotations

import logging
from dataclasses import field, make_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mlti_cal.detectors.base import Detection
from mlti_cal.detectors.masking import MaskGeometry
from mlti_cal.detectors.registry import DetectorKind
from mlti_cal.options import Option
from mlti_cal.plugins.contract import (
    ContractError,
    check_callable,
    check_name,
    check_settings,
    setting_kind,
    validate_detections,
)
from mlti_cal.plugins.point_object import PointObjectSpec

log = logging.getLogger("mlti_cal.plugins")

#: Contract violations logged per detector per run before the rest are only
#: counted, so a systematic mistake does not bury the log under 1000 lines.
_LOG_LIMIT = 5

POINTS_FILE_FIELD = Option(
    name="points_file",
    kind="str",
    default="",
    column="points_file",
    when="CSV of the object's points: one 'id, x, y, z' line per target, in metres "
    "in the object's own frame (z = 0 on a flat object). Relative paths are read "
    "from the working directory.",
)


def settings_class(name: str, defaults: dict[str, Any]) -> type:
    """A settings dataclass with `to_dict`/`from_dict`, one field per SETTINGS key."""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in defaults}

    def from_dict(data: dict | None):
        values = {}
        for key, value in (data or {}).items():
            if key not in defaults:
                continue  # a setting the plug-in no longer declares
            kind = setting_kind(defaults[key])
            values[key] = {"bool": bool, "int": int, "float": float}[kind](value)
        return cls(**values)

    cls = make_dataclass(
        f"{name}_Settings",
        [(k, type(v), field(default=v)) for k, v in defaults.items()],
        namespace={"to_dict": to_dict, "from_dict": staticmethod(from_dict)},
    )
    return cls


def settings_catalog(name: str, defaults: dict[str, Any], help_text: dict) -> list[Option]:
    out = []
    for key, value in defaults.items():
        kind = setting_kind(value)
        out.append(
            Option(
                name=key,
                kind=kind,
                default=value,
                when=str(help_text.get(key, f"Passed to {name}.detect() as settings[{key!r}].")),
                minimum=-1e9 if kind != "bool" else None,
                maximum=1e9 if kind != "bool" else None,
            )
        )
    return out


class FunctionDetectorGroup:
    """Every object detected by one plug-in function, in one image at a time."""

    def __init__(self, specs: list[PointObjectSpec], settings=None, *, name: str, func, cls):
        self.name = name
        self._func = func
        self.specs = list(specs)
        self.settings = settings if settings is not None else cls()
        self.detectors: dict[str, PointObjectSpec] = {}
        #: target id -> (object id, row in that object's points)
        self._owner: dict[int, tuple[str, int]] = {}
        for spec in self.specs:
            if spec.id in self.detectors:
                raise ValueError(f"duplicate object id {spec.id!r}")
            self.detectors[spec.id] = spec
            for row, tid in enumerate(spec.target_ids.tolist()):
                if tid in self._owner:
                    raise ValueError(
                        f"objects {self._owner[tid][0]!r} and {spec.id!r} both contain "
                        f"target id {tid}; detector {name!r} could not tell which one it "
                        f"saw. Give every target a unique id across the objects it detects."
                    )
                self._owner[tid] = (spec.id, row)
        self.rejected_images = 0
        self.unknown_ids = 0
        self._logged = 0

    def object_points(self, board_id: str) -> np.ndarray:
        return self.detectors[board_id].object_points

    @property
    def mask_geometry(self) -> dict[str, MaskGeometry]:
        return {
            s.id: MaskGeometry(
                object_points=s.object_points,
                outline=s.outline_object_points,
                pitch=s.pitch,
                planar=s.planar,
            )
            for s in self.specs
        }

    def _reject(self, message: str) -> list[Detection]:
        self.rejected_images += 1
        if self._logged < _LOG_LIMIT:
            log.warning("%s -- image skipped", message)
        elif self._logged == _LOG_LIMIT:
            log.warning(
                "detector %r: further contract violations are counted, not logged", self.name
            )
        self._logged += 1
        return []

    def detect_all(self, image: np.ndarray) -> list[Detection]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Copies: the same image goes on to the other detectors, and the
        # settings object is the one the GUI edits.
        try:
            raw = self._func(np.array(gray, copy=True), self.settings.to_dict())
        except Exception as exc:  # third-party code: any failure is that image's
            return self._reject(
                f"detector {self.name!r}: detect() raised {type(exc).__name__}: {exc}"
            )
        try:
            pts = validate_detections(raw, gray.shape, self.name)
        except ContractError as exc:
            return self._reject(str(exc))

        rows: dict[str, list[int]] = {}
        xy: dict[str, list[np.ndarray]] = {}
        for tid, x, y in pts:
            owner = self._owner.get(int(tid))
            if owner is None:
                # A target that belongs to no configured object -- a scale bar,
                # another jig in the scene. Not an error: it is just not ours.
                self.unknown_ids += 1
                continue
            obj, row = owner
            rows.setdefault(obj, []).append(row)
            xy.setdefault(obj, []).append(np.array([x, y]))
        return [
            Detection(
                board_id=obj,
                point_ids=np.asarray(rows[obj], dtype=int),
                image_points=np.asarray(xy[obj], dtype=float),
                num_markers=len(rows[obj]),
                kind=self.name,
            )
            for obj in rows
        ]


def detector_kind_from_module(module, path: Path | None = None) -> DetectorKind:
    """
    Check a plug-in module against the detector contract and build its kind.

    Only cheap, offline checks happen here, because this runs at every start of
    the application: names, signature, settings types. The plug-in's code is
    NOT called -- a vendor SDK may take seconds to initialise or check a
    licence. `mlti-cal plugins check` runs the full dry run on demand.
    """
    where = f"plug-in {path.name if path else module.__name__}"
    name = check_name(getattr(module, "NAME", None), where)
    func = check_callable(getattr(module, "detect", None), 2, "detect(image, settings)", where)
    defaults = check_settings(getattr(module, "SETTINGS", None), where)
    help_text = getattr(module, "SETTINGS_HELP", {}) or {}
    cls = settings_class(name, defaults)

    def group(specs, settings=None):
        return FunctionDetectorGroup(specs, settings, name=name, func=func, cls=cls)

    description = (getattr(module, "DESCRIPTION", "") or "").strip() or (
        f"Detector plug-in {name} ({path.name if path else module.__name__}): "
        f"finds identified targets and returns [target_id, x, y] per target."
    )
    return DetectorKind(
        id=name,
        label=str(getattr(module, "LABEL", name)),
        description=description,
        board_fields=[POINTS_FILE_FIELD],
        spec_cls=PointObjectSpec,
        group_cls=group,  # type: ignore[arg-type]
        settings_cls=cls,
        catalog=settings_catalog(name, defaults, help_text),
        summary=f"Settings handed to {name}.detect() as a dict.",
        coded=True,
        glyph=str(getattr(module, "GLYPH", "triangle")),
        origin=f"plug-in: {path.name}" if path else "plug-in",
    )
