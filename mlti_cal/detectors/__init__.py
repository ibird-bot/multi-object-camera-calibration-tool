"""
Detectors. Importing this package registers every available one.

The three built-ins are imported for their side effect -- each module ends in a
`register_detector(...)` call -- exactly as `solvers/__init__.py` imports its
backends. Third-party detectors are then loaded from the plugin directories and
from any installed package advertising the `mlti_cal.detectors` entry point.

Discovery runs HERE, at package import, and not later on demand. `board_kind()`
validates a config's boards against the registry, so a config naming a plugin
kind has to find that kind already registered by the time the file is read --
and every path into this application imports this package before it reads a
config.

A broken plugin is recorded in `registry.PLUGIN_ERRORS` rather than raised: one
bad file must not stop the app from starting. It is not swallowed either -- the
GUI reports what failed to load, because a plugin that silently does nothing is
indistinguishable from one that was never installed.
"""

from __future__ import annotations

import os

from mlti_cal.detectors import charuco, checkerboard, circle_grid  # noqa: F401
from mlti_cal.detectors.registry import (  # noqa: F401
    DETECTOR_KINDS,
    PLUGIN_ERRORS,
    DetectorKind,
    coded_kinds,
    default_detector_settings,
    detector_settings_from_dict,
    detector_settings_to_dict,
    get_kind,
    implemented_kinds,
    load_plugins,
    register_detector,
    uncoded_kinds,
)

#: Set to "0" to skip third-party discovery -- for reproducing a result without
#: whatever happens to be installed in the user's plugin directory.
PLUGINS_ENV = "MLTI_CAL_LOAD_PLUGINS"

#: Ids contributed by plugins in this process, for the GUI to label.
LOADED_PLUGINS: list[str] = []

if os.environ.get(PLUGINS_ENV, "1") != "0":
    LOADED_PLUGINS = load_plugins()

__all__ = [
    "DETECTOR_KINDS",
    "LOADED_PLUGINS",
    "PLUGIN_ERRORS",
    "DetectorKind",
    "coded_kinds",
    "default_detector_settings",
    "detector_settings_from_dict",
    "detector_settings_to_dict",
    "get_kind",
    "implemented_kinds",
    "load_plugins",
    "register_detector",
    "uncoded_kinds",
]
