"""
GUI tests, run offscreen.

Two things are checked, and the first matters more than the second:

  1. THE CORE IMPORTS ZERO QT. PLAN.md makes this a hard rule: the maths must
     be importable and runnable headless, so it can be tested, scripted and run
     in CI, and so the GUI cannot hide bugs in it. A rule like that decays the
     moment it stops being enforced, so it is enforced here.

  2. The window actually builds, loads data, solves and reports end to end.
     A GUI that constructs but falls over on the first real action is not
     working software.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

CORE_PACKAGES = [
    "mlti_cal.models",
    "mlti_cal.problem",
    "mlti_cal.solvers",
    "mlti_cal.report",
    "mlti_cal.detectors",
    "mlti_cal.io",
    "mlti_cal.cli",
]


def test_core_is_free_of_qt():
    """Import every core module in a fresh interpreter and assert no Qt."""
    import subprocess

    script = (
        "import importlib, pkgutil, sys\n"
        f"pkgs = {CORE_PACKAGES!r}\n"
        "for p in pkgs:\n"
        "    m = importlib.import_module(p)\n"
        "    for mod in pkgutil.iter_modules(m.__path__):\n"
        "        importlib.import_module(p + '.' + mod.name)\n"
        "bad = [n for n in sys.modules if n.split('.')[0] in "
        "('PySide6', 'PyQt5', 'PyQt6', 'shiboken6')]\n"
        "print('QT_LEAK:' + ','.join(sorted(bad)) if bad else 'CLEAN')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "CLEAN" in out.stdout, (
        f"core imported Qt -- the headless rule is broken: {out.stdout.strip()}"
    )


def test_every_core_module_imports():
    for pkg in CORE_PACKAGES:
        m = importlib.import_module(pkg)
        for mod in pkgutil.iter_modules(m.__path__):
            importlib.import_module(f"{pkg}.{mod.name}")


# ---------------------------------------------------------------------------
# Offscreen GUI
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def test_main_window_builds(qapp):
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    assert w.tabs.count() == 3
    assert w.session.system.cameras == {}
    w.close()


def test_full_gui_flow_synthetic(qapp):
    """Load demo -> solve -> report, driving the widgets as a user would."""
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    w.detection._load_demo()
    assert w.session.system.cameras, "demo did not populate the system"
    assert w.session.system.observations
    assert w.session.truth_values, "synthetic mode must supply ground truth"
    assert w.detection.image_table.rowCount() > 0

    w.optimization.refresh()
    problem = w.session.problem
    assert problem is not None and problem.num_free_params > 0

    # Solve synchronously -- the worker's run() is a plain method, so this
    # exercises the same code path without racing a QThread in a test.
    from mlti_cal.gui.session import SolveWorker
    from mlti_cal.solvers import SolveOptions

    captured = {}
    worker = SolveWorker(problem, "scipy", SolveOptions(max_iterations=200))
    worker.finished.connect(lambda r: captured.setdefault("result", r))
    worker.failed.connect(lambda m: captured.setdefault("error", m))
    worker.run()
    assert "error" not in captured, captured.get("error")
    result = captured["result"]
    assert result.final_cost <= result.initial_cost

    w.optimization._on_solved(result)
    assert w.session.result is result

    report = w.session.make_report(range_m=1.5)
    w.report._on_report(report)
    assert w.session.report is not None
    assert report.gt_check is not None, "ground-truth check must run in synthetic mode"
    assert "CALIBRATION REPORT" in w.report.summary_text.toPlainText()
    w.close()


def test_parameter_tree_fixing_removes_columns(qapp):
    """The Free/Fixed control must really change the Jacobian, not just the UI."""
    from PySide6.QtWidgets import QComboBox

    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    w.detection._load_demo()
    w.optimization.refresh()
    before = w.session.problem.num_free_params

    tree = w.optimization.tree
    cam_item = tree.topLevelItem(0)
    fixed = 0
    for j in range(cam_item.childCount()):
        child = cam_item.child(j)
        combo = tree.itemWidget(child, 2)
        if isinstance(combo, QComboBox):
            combo.setCurrentText("Fixed")
            fixed += 1
            if fixed == 3:
                break
    assert fixed == 3
    w.optimization.refresh()
    after = w.session.problem.num_free_params
    assert after == before - 3, f"expected {before - 3} free params, got {after}"
    w.close()


def test_report_view_handles_no_data(qapp):
    """Building a report with nothing loaded must warn, not crash."""
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    assert w.session.problem is None
    w.report._redraw()  # must be a no-op, not an exception
    w.close()


def test_unavailable_backend_is_disabled_in_the_picker(qapp):
    from mlti_cal.gui.app import MainWindow
    from mlti_cal.solvers import backend_status

    w = MainWindow()
    combo = w.optimization.backend_combo
    for i in range(combo.count()):
        name = combo.itemData(i)
        ok, _ = backend_status()[name]
        enabled = combo.model().item(i).isEnabled()
        assert enabled == ok, f"{name}: enabled={enabled} but usable={ok}"
    w.close()
