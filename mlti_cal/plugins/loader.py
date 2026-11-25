"""
Turning an imported plug-in file into registered detectors and camera models.

Discovery itself -- which folders, which files, how a broken file is recorded
-- stays in `detectors/registry.py`, so there is one plug-in mechanism, not
two. After a file has been imported there, `adopt` looks at what it defines:

    detect + NAME     -> a function detector (contract A, objects are point lists)
    project + NAME    -> a camera model (contract C)
    neither           -> an "advanced" plug-in that registered itself directly
                         with `register_detector` / `register_model`

A file may define both a detector and a camera model. Anything that breaks a
contract raises `ContractError`; the registry records it against the file and
the application carries on without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedPlugin:
    name: str
    role: str  # "detector" | "camera model" | "advanced"
    path: str


#: What loaded in this process, for the GUI's plug-in list and `mlti-cal plugins list`.
LOADED: list[LoadedPlugin] = []


def contributions(module) -> list[str]:
    """Which contracts `module` takes part in, judged by the names it defines."""
    roles = []
    if callable(getattr(module, "detect", None)):
        roles.append("detector")
    if callable(getattr(module, "project", None)):
        roles.append("camera model")
    return roles


def build(module, path: Path | None = None):
    """
    (DetectorKind or None, model class or None) for one module, unregistered.

    Shared by `adopt` and `mlti-cal plugins check`, so the check a developer
    runs is exactly the one the application applies.
    """
    from mlti_cal.plugins.contract import check_api_version

    check_api_version(module, f"plug-in {path.name if path else module.__name__}")
    roles = contributions(module)
    kind = model = None
    if "detector" in roles:
        from mlti_cal.plugins.function_detector import detector_kind_from_module

        kind = detector_kind_from_module(module, path)
    if "camera model" in roles:
        from mlti_cal.plugins.function_model import model_class_from_module

        model = model_class_from_module(module, path)
    return kind, model


def adopt(module, path: Path) -> None:
    """Register what `module` provides. Raises ContractError if it breaks a contract."""
    from mlti_cal.detectors.registry import register_detector
    from mlti_cal.models.camera import register_model

    kind, model = build(module, path)
    if kind is not None:
        register_detector(kind)
        LOADED.append(LoadedPlugin(kind.id, "detector", str(path)))
    if model is not None:
        register_model(model)
        LOADED.append(LoadedPlugin(model.name, "camera model", str(path)))
    if kind is None and model is None:
        LOADED.append(LoadedPlugin(path.stem, "advanced", str(path)))
