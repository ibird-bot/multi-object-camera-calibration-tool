"""
Which detectors exist, and what each one's settings are.

Until now there was one settings object called `DetectionSettings`, and every
field in it was an ArUco/Charuco parameter. That name claimed to speak for
detection in general while describing exactly one detector, so a checkerboard
detector would have had nowhere to put `flags` and no way to say that
`error_correction_rate` means nothing to it.

Settings belong to a DETECTOR, not to "detection". This registry is the list of
detector kinds, each carrying its own settings class and its own catalog. Kinds
that are not written yet are listed anyway, disabled, with the reason -- the
shape of the app stays honest about where they go instead of pretending a
Charuco board is the only thing a calibration target can be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlti_cal.detectors.settings import CHARUCO_CATALOG, CHARUCO_SUMMARY, CharucoSettings
from mlti_cal.options import Option


@dataclass(frozen=True)
class DetectorKind:
    """One kind of calibration target and the detector that finds it."""

    id: str
    label: str
    description: str
    implemented: bool
    settings_cls: type | None = None
    catalog: list[Option] = field(default_factory=list)
    summary: str = ""
    #: Why this kind cannot be used yet. Empty when it can.
    unavailable_reason: str = ""

    def default_settings(self) -> Any:
        if self.settings_cls is None:
            raise ValueError(f"detector {self.id!r} has no settings: {self.unavailable_reason}")
        return self.settings_cls()


DETECTOR_KINDS: dict[str, DetectorKind] = {
    "charuco": DetectorKind(
        id="charuco",
        label="Charuco board",
        description=(
            "Chessboard corners interpolated from decoded ArUco markers. The "
            "markers give every corner an identity, so partial views and several "
            "boards in one image both work."
        ),
        implemented=True,
        settings_cls=CharucoSettings,
        catalog=CHARUCO_CATALOG,
        summary=CHARUCO_SUMMARY,
    ),
    "aruco": DetectorKind(
        id="aruco",
        label="Aruco board",
        description=(
            "Marker corners used directly, with no chessboard interpolation. "
            "Faster and works on sparse layouts, but marker corners are markedly "
            "less accurate than interpolated chessboard corners."
        ),
        implemented=False,
        unavailable_reason="detector not written yet",
    ),
    "checkerboard": DetectorKind(
        id="checkerboard",
        label="Checkerboard",
        description=(
            "Classic chessboard. The most accurate corners of any target here, "
            "and the least forgiving: the whole board must be visible and its "
            "corners carry no identity, so orientation is ambiguous."
        ),
        implemented=False,
        unavailable_reason="detector not written yet",
    ),
    "circle_grid": DetectorKind(
        id="circle_grid",
        label="Circle grid",
        description=(
            "Symmetric or asymmetric dot grid. Centroids survive defocus better "
            "than corners do, at the cost of a perspective bias: a circle images "
            "as an ellipse whose centroid is not the projected centre."
        ),
        implemented=False,
        unavailable_reason="detector not written yet",
    ),
}


def implemented_kinds() -> list[DetectorKind]:
    return [k for k in DETECTOR_KINDS.values() if k.implemented]


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

    Unknown kinds are dropped rather than raising: a config written by a newer
    version that knows about `checkerboard` must still load here, minus the part
    this build cannot use.
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
