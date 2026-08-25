"""
Session-wide test setup: make the suite hermetic.

Two things here have to happen before any test module is imported, which is why
they live in a conftest rather than at the top of a test file.

Qt runs offscreen
    `test_gui.py` sets this too, but only when it is the module being imported
    first. Setting it here covers the case where something else constructs a
    QApplication earlier -- on a headless CI runner a real platform plugin does
    not exist and the process aborts rather than failing a test.

Plugin discovery cannot see the developer's home directory
    `registry.PLUGIN_DIRS` contains `~/.mlti_cal/detectors`, and `load_plugins()`
    scans it in ADDITION to whatever `MLTI_CAL_DETECTOR_PATH` names. The plugin
    tests monkeypatch the env var, so they were already pointing at a tmp dir --
    but the home directory was still being scanned alongside it. A developer with
    one real detector installed would see it registered mid-suite, and tests that
    assert on the exact set of known kinds would fail for reasons that have
    nothing to do with the change under test. Pointing the list at an empty tmp
    directory for the whole session makes the result depend only on the repo.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True, scope="session")
def _isolate_plugin_dirs(tmp_path_factory):
    from mlti_cal.detectors import registry

    empty = tmp_path_factory.mktemp("no_plugins")
    original = list(registry.PLUGIN_DIRS)
    registry.PLUGIN_DIRS[:] = [empty]
    yield
    registry.PLUGIN_DIRS[:] = original
