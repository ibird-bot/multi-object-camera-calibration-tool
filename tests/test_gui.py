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


def test_every_catalog_option_round_trips_through_its_widget(qapp):
    """
    Widget -> value must return exactly what the catalog declared.

    This caught a real bug: the float spinboxes were built with Qt's default of
    2 decimal places, so a 1e-10 solver tolerance became 0.00 and scipy
    rejected every solve with "at least one of the tolerances must be higher
    than machine epsilon". A knob that is displayed and then silently discarded
    is worse than one that was never exposed.
    """
    from mlti_cal.detectors.settings import CHARUCO_CATALOG
    from mlti_cal.gui.settings_panel import make_option_widget, widget_value
    from mlti_cal.problem.settings import INIT_CATALOG
    from mlti_cal.report.settings import REPORT_CATALOG
    from mlti_cal.solvers.catalog import CATALOG, COMMON_OPTIONS

    catalogs = [CHARUCO_CATALOG, INIT_CATALOG, REPORT_CATALOG, COMMON_OPTIONS]
    catalogs += [entry["options"] for entry in CATALOG.values()]
    for catalog in catalogs:
        for opt in catalog:
            got = widget_value(make_option_widget(opt))
            if opt.kind == "float":
                assert got == pytest.approx(float(opt.default), rel=1e-12), (
                    f"{opt.name}: widget returned {got!r}, catalog says {opt.default!r}"
                )
            elif opt.kind == "choice":
                assert got == str(opt.default).split(" ")[0], opt.name
            else:
                assert got == opt.default, opt.name


def test_changing_a_detector_setting_marks_existing_detections_stale(qapp):
    """
    Corners found under the old settings must not be silently reused.

    Without this, a user changes corner refinement, presses Solve, and gets a
    calibration computed from the PREVIOUS detections while the new setting is
    displayed next to it.
    """
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    load_synthetic(w)
    assert not w.detection.detection_is_stale, "precondition: freshly loaded"

    from mlti_cal.detectors.settings import CharucoSettings

    w.detection._on_detector_settings_applied(
        {"charuco": CharucoSettings(corner_refinement="CONTOUR")}
    )
    assert w.detection.detection_is_stale, "a changed detector setting was not flagged"
    assert "settings changed" in w.detection.detect_btn.text()

    # ...and the new value is what a detection run would actually use.
    assert w.detection._collect_config().charuco.corner_refinement == "CONTOUR"

    # The flag reaches the session, which is what the Optimization tab reads
    # before it computes anything on those corners.
    assert w.session.detection_stale is True


def test_a_stale_solve_asks_before_using_old_corners(qapp, monkeypatch):
    """The guard must actually gate the solve, not merely exist."""
    from mlti_cal.gui import optimization_view as ov
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    load_synthetic(w)
    bootstrap(w)
    w.session.detection_stale = True

    asked = {}

    def fake_warning(*args, **kwargs):
        asked["shown"] = True
        return ov.QMessageBox.Cancel

    monkeypatch.setattr(ov.QMessageBox, "warning", fake_warning)
    started = {}
    monkeypatch.setattr(ov, "SolveWorker", lambda *a, **k: started.setdefault("built", True))

    w.optimization._solve()
    assert asked.get("shown"), "a stale solve was not questioned"
    assert "built" not in started, "Cancel did not stop the solve"


def test_cross_validate_button_reports_held_out_error(qapp, monkeypatch):
    """
    The Cross-validate button runs real folds and prints them in the log.

    Held-out error is the one number in this window that is not measured on the
    data it was fitted to, so the button has to actually run folds -- not read a
    cached figure -- and it must leave the user's own solved parameters alone.
    """
    import time

    from PySide6.QtWidgets import QApplication

    from mlti_cal.gui import optimization_view as ov
    from mlti_cal.gui.app import MainWindow

    seen: dict[str, object] = {}

    def record_dialog(*args, **kwargs):
        seen["dialog"] = args[2] if len(args) > 2 else "?"

    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(ov.QMessageBox, name, record_dialog)

    w = MainWindow()
    load_synthetic(w)
    bootstrap(w)
    idx = w.optimization.backend_combo.findData("scipy")
    w.optimization.backend_combo.setCurrentIndex(idx)
    w.optimization.iterations.setValue(60)
    w.optimization.folds_spin.setValue(2)

    before = {cid: cam.params.copy() for cid, cam in w.session.system.cameras.items()}
    w.optimization._cross_validate()
    assert "dialog" not in seen, f"cross-validation refused to start: {seen['dialog']}"

    deadline = time.monotonic() + 120
    while w.session.crossval is None and time.monotonic() < deadline:
        QApplication.processEvents()
    assert w.session.crossval is not None, "cross-validation never finished"
    # Keep pumping until the thread actually stops. `thread.quit` is a QUEUED
    # connection on the same signal as the result handler, so blocking the GUI
    # thread in wait() here would stop the quit from ever being delivered --
    # a deadlock in the test, not in the app, which returns to its event loop.
    deadline = time.monotonic() + 10
    while w.optimization._active_thread.isRunning() and time.monotonic() < deadline:
        QApplication.processEvents()
    assert w.optimization._active_thread.wait(5000)

    result = w.session.crossval
    assert len(result.folds) == 2
    assert result.mean_test_rms > 0 and result.mean_train_rms > 0

    log = w.optimization.log.toPlainText()
    assert "cross-validation" in log
    assert "mean test RMS" in log, "the held-out number never reached the text zone"
    assert "optimism" in log
    for fold in result.folds:
        assert f"{fold.test_rms_px:.4f}" in log, "a fold row is missing from the log"

    # The user's own parameters must survive: every fold works on a copy, and
    # silently replacing the solved values with a fold's would be a data loss
    # the user never asked for.
    for cid, params in before.items():
        assert w.session.system.cameras[cid].params == pytest.approx(params)


def test_cross_validate_refuses_more_folds_than_frames(qapp, monkeypatch):
    """k+1 frames are needed; asking for more must explain, not crash."""
    from mlti_cal.gui import optimization_view as ov
    from mlti_cal.gui.app import MainWindow

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        ov.QMessageBox,
        "information",
        lambda *a, **k: seen.setdefault("dialog", a[2] if len(a) > 2 else "?"),
    )

    w = MainWindow()
    load_synthetic(w)
    bootstrap(w)
    w.optimization.folds_spin.setValue(50)
    w.optimization._cross_validate()
    assert "folds" in str(seen.get("dialog", "")).lower() or "frames" in str(
        seen.get("dialog", "")
    ), f"unhelpful refusal: {seen.get('dialog')}"


def test_pixel_noise_can_be_switched_off(qapp):
    """
    "I do not know the corner noise" must be expressible, not approximated.

    Typing a plausible 0.3 the user never measured puts an invented number into
    the whitening and into the covariance cross-check, and the report then
    compares the fit against a claim nobody made.
    """
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    view = w.detection
    assert view.noise_check.isChecked(), "precondition: on by default"
    assert view._collect_config().pixel_noise_std == pytest.approx(0.3)

    view.noise_check.setChecked(False)
    assert not view.noise_spin.isEnabled(), "the value box should grey out"
    assert view._collect_config().pixel_noise_std is None

    view.noise_check.setChecked(True)
    assert view.noise_spin.isEnabled()
    assert view._collect_config().pixel_noise_std == pytest.approx(0.3)


def test_unknown_noise_survives_a_config_round_trip(qapp, tmp_path):
    from mlti_cal.gui.app import MainWindow
    from mlti_cal.io.config import CalibrationConfig

    w = MainWindow()
    w.detection.noise_check.setChecked(False)
    path = w.detection._collect_config().save(tmp_path / "c.json")

    w2 = MainWindow()
    w2.detection._apply_config(CalibrationConfig.load(path))
    assert not w2.detection.noise_check.isChecked()
    assert w2.detection._collect_config().pixel_noise_std is None


def test_a_measured_noise_value_switches_the_assumption_back_on(qapp):
    """The measurement is only useful if it can become the assumption."""
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    w.detection.noise_check.setChecked(False)
    w.detection._on_noise_measured(0.1234)
    assert w.detection.noise_check.isChecked()
    assert w.detection.noise_spin.value() == pytest.approx(0.1234, abs=1e-4)
    assert w.detection._collect_config().pixel_noise_std == pytest.approx(0.1234, abs=1e-4)


def test_measure_dialog_needs_a_board_and_says_so(qapp, monkeypatch):
    from mlti_cal.gui import detection_view as dv
    from mlti_cal.gui.app import MainWindow

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        dv.QMessageBox,
        "information",
        lambda *a, **k: seen.setdefault("dialog", a[2] if len(a) > 2 else "?"),
    )
    w = MainWindow()
    w.detection.board_table.setRowCount(0)
    w.detection._measure_noise()
    assert "board" in str(seen.get("dialog", "")).lower(), f"unhelpful: {seen.get('dialog')}"


def test_noise_dialog_builds_with_the_current_detector_settings(qapp):
    """The dialog must measure the detector you are actually going to run."""
    from mlti_cal.detectors.settings import CharucoSettings
    from mlti_cal.gui.app import MainWindow
    from mlti_cal.gui.noise_dialog import NoiseDialog

    w = MainWindow()
    w.detection._on_detector_settings_applied(
        {"charuco": CharucoSettings(corner_refinement="CONTOUR")}
    )
    specs = w.detection._collect_config().board_specs()
    dialog = NoiseDialog(specs, w.session.detector_settings["charuco"])
    assert dialog.panel.widgets["corner_refinement"].currentText() == "CONTOUR"
    assert dialog.board_combo.count() == len(specs)
    assert not dialog.use_btn.isEnabled(), "nothing measured yet"


def test_detector_settings_dialog_has_a_tab_per_kind(qapp):
    """
    Every kind gets a tab, implemented or not.

    Showing only charuco would leave the user unable to tell "this detector has
    no options" from "this detector does not exist yet".
    """
    from mlti_cal.detectors.registry import DETECTOR_KINDS, implemented_kinds
    from mlti_cal.gui.detector_dialog import DetectorSettingsDialog
    from mlti_cal.gui.session import Session

    session = Session()
    dialog = DetectorSettingsDialog(session.detector_settings, in_use={"charuco"})
    assert dialog.tabs.count() == len(DETECTOR_KINDS)
    assert set(dialog.panels) == {k.id for k in implemented_kinds()}

    labels = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
    assert any("in use" in t for t in labels), "the kind actually in use is not marked"
    for i, label in enumerate(labels):
        if "not available" in label:
            assert not dialog.tabs.isTabEnabled(i)
            assert dialog.tabs.tabToolTip(i), "a disabled tab must say why"


def test_detector_settings_apply_reaches_the_config_and_marks_stale(qapp):
    from mlti_cal.gui.app import MainWindow
    from mlti_cal.gui.detector_dialog import DetectorSettingsDialog

    w = MainWindow()
    load_synthetic(w)
    assert not w.session.detection_stale

    dialog = DetectorSettingsDialog(w.session.detector_settings, in_use={"charuco"})
    dialog.settings_applied.connect(w.detection._on_detector_settings_applied)
    dialog.panels["charuco"].widgets["error_correction_rate"].setValue(0.45)
    assert dialog.apply()

    assert w.session.detector_settings["charuco"].error_correction_rate == pytest.approx(0.45)
    assert w.detection._collect_config().charuco.error_correction_rate == pytest.approx(0.45)
    assert w.session.detection_stale, "changing a detector setting must invalidate detections"
    # No control or label for these on the tab any more, so the log is the
    # record of what they became -- with values, not just "something changed".
    assert "0.45" in w.console.toPlainText()


def test_invalid_detector_settings_are_refused_not_applied(qapp, monkeypatch):
    """The dataclass catches combinations no spinbox range can."""
    from mlti_cal.gui import detector_dialog as dd
    from mlti_cal.gui.session import Session

    seen = {}
    monkeypatch.setattr(
        dd.QMessageBox,
        "critical",
        lambda *a, **k: seen.setdefault("dialog", a[2] if len(a) > 2 else "?"),
    )
    session = Session()
    dialog = dd.DetectorSettingsDialog(session.detector_settings)
    panel = dialog.panels["charuco"]
    panel.widgets["adaptive_thresh_win_size_min"].setValue(40)
    panel.widgets["adaptive_thresh_win_size_max"].setValue(10)

    assert dialog.apply() is False
    assert "window size" in str(seen.get("dialog", "")), f"unhelpful: {seen.get('dialog')}"
    assert session.detector_settings["charuco"].adaptive_thresh_win_size_min == 3


def test_detector_settings_are_reachable_only_from_the_menu_bar(qapp):
    """
    One way in. A second control on the Setup tab could fall out of step with
    the menu's, which is how two views of one setting start disagreeing.
    """
    from PySide6.QtWidgets import QPushButton

    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    actions = [
        a.text() for m in w.menuBar().actions() for a in (m.menu().actions() if m.menu() else [])
    ]
    assert any("Detector" in a and "settings" in a for a in actions)

    buttons = [b.text() for b in w.detection.findChildren(QPushButton)]
    assert not any("Detector settings" in b for b in buttons), (
        "detector settings must be reachable only from the menu bar"
    )


def test_the_noise_measurement_stays_on_the_main_ui(qapp):
    """It belongs beside the assumed pixel noise it exists to fill in."""
    from PySide6.QtWidgets import QPushButton

    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    actions = [
        a.text() for m in w.menuBar().actions() for a in (m.menu().actions() if m.menu() else [])
    ]
    assert not any("pixel noise" in a for a in actions), "measurement should not be in the menu bar"

    buttons = [b.text() for b in w.detection.findChildren(QPushButton)]
    assert any("Measure" in b for b in buttons)
    assert w.detection.measure_noise_btn.isEnabled()


def test_changed_detector_settings_are_recorded_in_the_log(qapp):
    """
    Moving the knobs into a window must not lose the record of their values.

    Nothing on the tab reports them now, so a transcript saying only that
    "something changed" would leave a run impossible to reproduce.
    """
    from mlti_cal.detectors.settings import CharucoSettings
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    assert "all defaults" in w.detection.detector_summary_text()
    w.detection._on_detector_settings_applied(
        {"charuco": CharucoSettings(corner_refinement="APRILTAG")}
    )
    logged = w.console.toPlainText()
    assert "APRILTAG" in logged and "Charuco" in logged


def test_one_console_is_shared_by_every_tab(qapp):
    """
    The transcript belongs to the session, not to a tab.

    It used to live inside the Optimization tab, so detection and report
    messages were only visible if you happened to be on the right tab -- and
    the noise measurement wrote nowhere but its own dialog.
    """
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    assert w.detection.console is w.console
    assert w.optimization.console is w.console
    assert w.report.console is w.console
    # Docked, so it is on screen whichever tab is in front.
    assert w.log_dock.widget() is w.console


def test_each_stage_tags_its_own_lines(qapp):
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    w.console.write("detect", "found 42 corners")
    w.console.write("solve", "converged")
    assert {e.source for e in w.console.entries} == {"detect", "solve"}
    text = w.console.toPlainText()
    assert "detect" in text and "found 42 corners" in text


def test_the_filter_hides_lines_without_discarding_them(qapp):
    from mlti_cal.gui.log_console import LogConsole

    console = LogConsole()
    console.write("detect", "corners")
    console.write("solve", "converged")

    console.filter_combo.setCurrentText("solve")
    assert "converged" in console.toPlainText()
    assert "corners" not in console.toPlainText()

    console.filter_combo.setCurrentText("everything")
    assert "corners" in console.toPlainText(), "filtering must not destroy lines"
    assert len(console.entries) == 2


def test_a_multi_line_message_becomes_one_tagged_line_each(qapp):
    """Otherwise a filtered view shows a block whose middle lines lost their tag."""
    from mlti_cal.gui.log_console import LogConsole

    console = LogConsole()
    console.write("report", "line one\nline two\nline three")
    assert len(console.entries) == 3
    assert all(e.source == "report" for e in console.entries)


def test_the_console_is_bounded(qapp):
    from mlti_cal.gui.log_console import MAX_ENTRIES, LogConsole

    console = LogConsole()
    for i in range(MAX_ENTRIES + 250):
        console.write("solve", f"iteration {i}")
    assert len(console.entries) == MAX_ENTRIES
    assert "iteration 0" not in console.toPlainText(), "oldest lines should age out"


def test_setup_and_report_messages_reach_the_shared_console(qapp):
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    w.detection._announce("detection done -- 12 observations")
    w.report.status.emit("report built")
    sources = {e.source for e in w.console.entries}
    assert {"setup", "report"} <= sources
    assert "12 observations" in w.console.toPlainText()


def test_solver_options_follow_the_selected_backend(qapp):
    """
    Each backend shows its own options and only the common ones it READS.

    scipy has no thread count and the GTSAM adapter sets only a relative error
    tolerance -- rendering the rest beside them would be controls that silently
    do nothing, which is the failure mode this whole panel exists to remove.
    """
    from mlti_cal.gui.app import MainWindow
    from mlti_cal.solvers.catalog import options_for

    w = MainWindow()
    combo = w.optimization.backend_combo
    seen = {}
    for i in range(combo.count()):
        combo.setCurrentIndex(i)
        backend = combo.currentData()
        seen[backend] = set(w.optimization._option_widgets)
        expected = {o.name for o in options_for(backend)} - {"max_iterations"}
        assert seen[backend] == expected, backend
        assert backend in w.optimization.options_box.title()

    assert "num_threads" in seen["ceres"]
    assert "num_threads" not in seen["scipy"], "scipy ignores it"
    assert "gradient_tolerance" not in seen["gtsam"], "the GTSAM adapter ignores it"
    assert seen["scipy"] != seen["ceres"], "the panel must actually change"


def test_solver_option_values_reach_the_solve_options(qapp):
    """A rendered knob has to arrive where the backend reads it."""
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    combo = w.optimization.backend_combo
    combo.setCurrentIndex(combo.findData("ceres"))
    w.optimization._option_widgets["num_threads"].setValue(4)
    w.optimization._option_widgets["function_tolerance"].setValue(1e-8)

    opts = w.optimization._solve_options()
    assert opts.num_threads == 4
    assert opts.function_tolerance == pytest.approx(1e-8)
    # Backend-specific ones travel in `extra`, not as fields.
    assert "linear_solver_type" in opts.extra


def test_starting_point_settings_open_in_their_own_window(qapp):
    """No inline panel: the knobs live behind a button, applied explicitly."""
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    assert not hasattr(w.optimization, "init_panel"), "the inline panel should be gone"
    assert w.optimization.init_settings_btn.text() == "Settings..."


def test_starting_point_settings_take_effect_only_on_apply(qapp):
    """
    Editing a widget must not change the live setting until Apply.

    An inline panel takes effect as a spinbox ticks past values on its way
    somewhere else, so a half-typed threshold is briefly the live setting.
    """
    from mlti_cal.gui.app import MainWindow
    from mlti_cal.gui.settings_dialog import CatalogSettingsDialog
    from mlti_cal.problem.settings import INIT_CATALOG, INIT_SUMMARY, InitSettings

    w = MainWindow()
    view = w.optimization
    dialog = CatalogSettingsDialog(
        "Starting point settings",
        INIT_SUMMARY,
        INIT_CATALOG,
        values=view._init_values,
        build=lambda raw: InitSettings(**raw),
    )
    dialog.applied.connect(view._on_init_settings_applied)

    dialog.panel.widgets["pnp_ransac"].setChecked(True)
    dialog.panel.widgets["ransac_reproj_threshold_px"].setValue(1.5)
    assert view._init_settings().pnp_ransac is False, "changed before Apply"

    assert dialog.apply()
    assert view._init_settings().pnp_ransac is True
    assert view._init_settings().ransac_reproj_threshold_px == pytest.approx(1.5)
    # Non-default settings are visible without opening the window again.
    assert view.init_settings_btn.text().startswith("Settings*")
    assert "pnp_ransac=True" in w.console.toPlainText()


def test_starting_point_settings_refuse_an_invalid_combination(qapp, monkeypatch):
    """
    Validation runs on Apply, while the window is still open to be corrected.

    The floors are geometric, not preferences: PnP genuinely needs 4 points.
    """
    from mlti_cal.gui import settings_dialog as sd
    from mlti_cal.gui.app import MainWindow
    from mlti_cal.problem.settings import INIT_CATALOG, INIT_SUMMARY, InitSettings

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        sd.QMessageBox,
        "critical",
        lambda *a, **k: seen.setdefault("dialog", a[2] if len(a) > 2 else "?"),
    )
    w = MainWindow()
    dialog = sd.CatalogSettingsDialog(
        "Starting point settings",
        INIT_SUMMARY,
        INIT_CATALOG,
        values=w.optimization._init_values,
        build=lambda raw: InitSettings(**raw),
    )
    dialog.applied.connect(w.optimization._on_init_settings_applied)
    # Below the spinbox floor is impossible, so reach the cross-field rule the
    # dataclass enforces instead.
    dialog.panel.widgets["translation_average"].setCurrentText("median")
    raw = dialog.panel.values()
    raw["min_points_for_pnp"] = 3
    monkeypatch.setattr(dialog.panel, "values", lambda: raw)

    assert dialog.apply() is False
    assert "4 points" in str(seen.get("dialog", "")), f"unhelpful: {seen.get('dialog')}"
    assert w.optimization._init_settings().min_points_for_pnp == 4, "invalid value was applied"


def test_main_window_builds(qapp):
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    assert w.tabs.count() == 3
    assert w.session.system.cameras == {}
    w.close()


def test_full_gui_flow_synthetic(qapp):
    """Synthetic data -> solve -> report, driving the widgets as a user would."""
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    load_synthetic(w)
    assert w.session.system.cameras, "the synthetic loader did not populate the system"
    assert w.session.system.observations
    assert w.session.truth_values, "synthetic mode must supply ground truth"
    assert w.detection.image_table.rowCount() > 0

    bootstrap(w)
    problem = w.session.problem
    assert problem is not None and problem.num_free_params > 0

    # Direct call: checks the worker body. The THREADED path is covered
    # separately by test_solve_through_run_in_thread_completes_on_gui_thread,
    # which is the one that matters -- bypassing the thread here is precisely
    # what once hid a crash-on-first-Solve bug.
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


def test_solve_through_run_in_thread_completes_on_gui_thread(qapp, monkeypatch):
    """
    Press Solve for real and let it go through the actual QThread.

    Every other GUI test calls `worker.run()` directly, which never exercises
    `run_in_thread` -- and that is exactly how a crash on the very first Solve
    survived a green suite. The load-bearing assertion is the second one: the
    completion handler must run on the GUI THREAD. Connecting the completion
    signals to a lambda (a non-QObject receiver, so Qt cannot queue and falls
    back to a direct call) makes `_on_solved` run on the worker thread, where
    it touches widgets, matplotlib and the tree -- undefined behaviour.

    Asserting the solve itself ran off the GUI thread is kept for intent, but
    it is not the regression check: the broken version moved the worker to a
    thread perfectly well. It only got the callbacks wrong.

    HOW THIS IS INSTRUMENTED, because the obvious way silently breaks it:
    assigning a spy function onto `OptimizationView._on_solved` (or onto
    `SolveWorker.run`) makes PySide6 unable to resolve the bound method back to
    its QObject, so it stops queueing and direct-connects -- the exact defect
    under test. The spy then reports a bug that only its own presence created.
    Measured: patching `_on_solved` that way reports the handler on the worker
    thread even against the fixed code. So the worker is instrumented by
    SUBCLASSING (a real class-body method keeps its affinity) and the handler
    thread is observed from outside, via a genuine slot on `solved` forced to
    DirectConnection so it runs in whatever thread the handler is already on.
    `_on_solved` itself is never touched. Verified against the pre-fix
    implementation: it reports the handler off the GUI thread, as it must.
    """
    import time

    from PySide6.QtCore import QObject, Qt, QThread
    from PySide6.QtWidgets import QApplication

    from mlti_cal.gui import optimization_view as ov
    from mlti_cal.gui.app import MainWindow
    from mlti_cal.solvers import backend_status

    seen: dict[str, object] = {}
    gui_thread = QThread.currentThread()

    class SpyWorker(ov.SolveWorker):
        def run(self):
            seen["worker_thread"] = QThread.currentThread()
            super().run()

    class Recorder(QObject):
        def note(self):
            seen["handler_thread"] = QThread.currentThread()
            seen["done"] = True

    # A modal box in _on_failed would block the event loop we are about to
    # spin, turning any solve failure into a hung suite instead of a report.
    # This is a plain call, not a signal connection, so patching it is safe.
    def record_dialog(*args, **kwargs):
        seen["dialog"] = args[2] if len(args) > 2 else "?"
        seen["done"] = True

    monkeypatch.setattr(ov.QMessageBox, "critical", record_dialog)
    monkeypatch.setattr(ov, "SolveWorker", SpyWorker)

    w = MainWindow()
    recorder = Recorder()
    w.optimization.solved.connect(recorder.note, Qt.DirectConnection)
    load_synthetic(w)
    bootstrap(w)
    w.optimization.iterations.setValue(60)

    # Pin the backend rather than trusting the combo's default: the point here
    # is the threading contract, and a machine without ceres would otherwise
    # sit on a disabled entry and time out instead of failing meaningfully.
    combo = w.optimization.backend_combo
    names = [combo.itemData(i) for i in range(combo.count())]
    pick = next(n for n in ("scipy", "ceres", *names) if backend_status()[n][0])
    combo.setCurrentIndex(names.index(pick))
    assert combo.currentData() == pick, "the backend pin did not take"

    w.optimization._solve()  # the real slot the Solve button is wired to

    deadline = time.monotonic() + 90
    while "done" not in seen and time.monotonic() < deadline:
        QApplication.processEvents()

    assert "dialog" not in seen, f"solve failed: {seen['dialog']}"
    assert "done" in seen, "the solve never completed -- no result reached the GUI thread"

    assert seen["worker_thread"] is not gui_thread, (
        "the solve ran on the GUI thread; the window would freeze for its whole duration"
    )
    assert seen["handler_thread"] is gui_thread, (
        f"_on_solved ran on {seen['handler_thread']}, not the GUI thread -- run_in_thread "
        f"made a direct connection, so widget code executes on the worker thread"
    )

    thread = w.optimization._active_thread
    assert thread.wait(5000), "the worker thread never exited its event loop"
    assert thread.isFinished()
    assert w.session.result is not None, "the session did not receive the result"
    w.close()


def _detection_folder(tmp_path):
    """A folder of genuinely detectable images, borrowed from the pipeline test."""
    from test_config_pipeline import _warped_views, _write_camera_dir

    stems = [f"{i:03d}" for i in range(4)]
    return _write_camera_dir(tmp_path, "cam0", _warped_views(4, seed=21), stems), stems


def _point_camera_at(view, folder: str):
    """Type the path in, exactly as a user who does not use the browse button."""
    from PySide6.QtWidgets import QLineEdit

    view.camera_table.cellWidget(0, 2).findChild(QLineEdit).setText(folder)


def _describe_board(view):
    """
    Fill the board row to match the rendered fixtures.

    Necessary, not incidental: the form defaults to 9x7 and the fixtures are
    8x6, so leaving it alone detects exactly nothing. That is the realistic
    user error, and it has its own test below.
    """
    from test_config_pipeline import BOARD

    t = view.board_table
    for col, key in enumerate(("id", "squares_x", "squares_y", "square_length", "marker_length")):
        t.item(0, col).setText(str(BOARD[key]))
    t.cellWidget(0, 5).setCurrentText(BOARD["dictionary"])
    t.cellWidget(0, 6).setValue(BOARD["marker_id_offset"])


def test_detection_runs_off_the_gui_thread_and_draws_on_the_thumbnails(qapp, tmp_path, monkeypatch):
    """
    Run detection for real, through the real thread, on real images.

    Detection used to run inline, freezing the window for the whole run with no
    repaint -- which is why it looked like the button did nothing. Three things
    are asserted, and the first two are the ones a green suite could otherwise
    hide: the work leaves the GUI thread, the results come BACK on the GUI
    thread (a lambda connection would run widget code on the worker thread, see
    run_in_thread), and every thumbnail actually receives an icon.

    The folder is typed in rather than browsed, so the strip has no rows when
    detection starts: that is the path where a missing key makes `set_image` a
    silent no-op and every thumbnail stays blank without any error.
    """
    import time

    from PySide6.QtCore import QObject, Qt, QThread
    from PySide6.QtWidgets import QApplication

    from mlti_cal.gui import detection_view as dv
    from mlti_cal.gui.app import MainWindow

    folder, stems = _detection_folder(tmp_path)
    seen: dict[str, object] = {}
    gui_thread = QThread.currentThread()

    class SpyWorker(dv.DetectionWorker):
        def run(self):
            seen["worker_thread"] = QThread.currentThread()
            super().run()

    class Recorder(QObject):
        def note(self):
            seen["handler_thread"] = QThread.currentThread()
            seen["done"] = True

    def record_dialog(*args, **kwargs):
        seen["dialog"] = args[2] if len(args) > 2 else "?"
        seen["done"] = True

    monkeypatch.setattr(dv, "DetectionWorker", SpyWorker)
    monkeypatch.setattr(dv.QMessageBox, "critical", record_dialog)

    w = MainWindow()
    recorder = Recorder()
    w.detection.system_changed.connect(recorder.note, Qt.DirectConnection)
    _point_camera_at(w.detection, folder)
    _describe_board(w.detection)
    assert w.detection.thumbnails.keys() == [], "precondition: nothing scanned yet"

    w.detection._run_detection()  # the real slot behind the button
    assert not w.detection.detect_btn.isEnabled(), "a second run must not be startable"

    deadline = time.monotonic() + 120
    while "done" not in seen and time.monotonic() < deadline:
        QApplication.processEvents()

    assert "dialog" not in seen, f"detection failed: {seen['dialog']}"
    assert "done" in seen, "detection never completed"
    assert seen["worker_thread"] is not gui_thread, (
        "detection ran on the GUI thread; the window freezes for the whole run"
    )
    assert seen["handler_thread"] is gui_thread, (
        f"the completion handler ran on {seen['handler_thread']}, not the GUI thread"
    )

    strip = w.detection.thumbnails
    assert sorted(strip.keys()) == [f"cam0/{s}" for s in stems]
    for key in strip.keys():
        item = strip.item_for(key)
        assert item is not None and not item.icon().isNull(), f"{key} has no thumbnail"
        assert "pts" in item.text(), f"{key} caption never updated: {item.text()!r}"
    assert strip.active_key is None, "the in-progress highlight was never cleared"
    assert strip.empty_keys == set(), "every rendered view should detect corners"

    assert w.session.system.observations, "the session did not receive the detections"
    assert w.session.source == "images"
    assert w.detection.detect_btn.isEnabled()
    assert w.detection._active_thread.wait(5000)
    w.close()


def test_failed_detection_re_enables_the_button_and_clears_the_highlight(
    qapp, tmp_path, monkeypatch
):
    """
    A failure must leave the UI usable.

    The button is disabled at launch to stop a double-start, so a failure path
    that forgets to re-enable it locks the user out of retrying for the rest of
    the session -- with a dialog that says the run failed but no way to act on
    it. An empty folder is the realistic trigger: it is what you get by picking
    the parent of the captures.
    """
    import time

    from PySide6.QtCore import QObject, Qt
    from PySide6.QtWidgets import QApplication

    from mlti_cal.gui import detection_view as dv
    from mlti_cal.gui.app import MainWindow

    empty = tmp_path / "no_captures_here"
    empty.mkdir()
    seen: dict[str, object] = {}

    def record_dialog(*args, **kwargs):
        seen["dialog"] = args[2] if len(args) > 2 else "?"
        seen["done"] = True

    class Recorder(QObject):
        def note(self):
            seen["unexpected_success"] = True
            seen["done"] = True

    monkeypatch.setattr(dv.QMessageBox, "critical", record_dialog)

    w = MainWindow()
    recorder = Recorder()
    w.detection.system_changed.connect(recorder.note, Qt.DirectConnection)
    _point_camera_at(w.detection, str(empty))
    w.detection._run_detection()

    deadline = time.monotonic() + 60
    while "done" not in seen and time.monotonic() < deadline:
        QApplication.processEvents()

    assert "unexpected_success" not in seen, "an empty folder cannot produce a calibration"
    assert "dialog" in seen, "the failure was never reported to the user"
    assert "no images" in str(seen["dialog"]), f"unhelpful message: {seen['dialog']!r}"
    assert w.detection.detect_btn.isEnabled(), "the user is locked out of retrying"
    assert w.detection.thumbnails.active_key is None
    assert w.detection._active_thread.wait(5000)
    w.close()


def test_a_board_that_matches_nothing_is_reported_instead_of_looking_successful(
    qapp, tmp_path, monkeypatch
):
    """
    Detection that finds nothing must say so and keep the session intact.

    Found while writing the test above, which used the form's default 9x7 board
    against 8x6 fixtures: every image detected zero corners and the app still
    reported success, committed an observation-free system and jumped to the
    Optimization tab. That is the same "detection does nothing" experience the
    thread work set out to fix, so the empty result gets named as the setup
    error it is -- and the red thumbnails stay on screen as the evidence.
    """
    import time

    from PySide6.QtCore import QObject, Qt
    from PySide6.QtWidgets import QApplication

    from mlti_cal.gui import detection_view as dv
    from mlti_cal.gui.app import MainWindow

    folder, stems = _detection_folder(tmp_path)
    seen: dict[str, object] = {}

    def record(*args, **kwargs):
        seen["title"] = args[1] if len(args) > 1 else "?"
        seen["text"] = args[2] if len(args) > 2 else "?"
        seen["done"] = True

    class Recorder(QObject):
        def note(self):
            seen["committed"] = True
            seen["done"] = True

    monkeypatch.setattr(dv.QMessageBox, "warning", record)
    monkeypatch.setattr(dv.QMessageBox, "critical", record)

    w = MainWindow()
    recorder = Recorder()
    w.detection.system_changed.connect(recorder.note, Qt.DirectConnection)
    _point_camera_at(w.detection, folder)  # board left at the 9x7 default on purpose
    w.detection._run_detection()

    deadline = time.monotonic() + 120
    while "done" not in seen and time.monotonic() < deadline:
        QApplication.processEvents()

    assert "committed" not in seen, "an observation-free system must not reach the session"
    assert w.session.system.observations == [], "the session was overwritten with nothing"
    assert seen.get("title") == "No detections", f"wrong dialog: {seen.get('title')!r}"
    assert "squares" in str(seen["text"]), "the message must point at the likely cause"

    strip = w.detection.thumbnails
    assert strip.empty_keys == {f"cam0/{s}" for s in stems}, "empty frames must be flagged"
    assert strip.active_key is None
    assert w.detection.detect_btn.isEnabled()
    assert w.detection._active_thread.wait(5000)
    w.close()


def load_synthetic(w) -> None:
    """
    Put a synthetic dataset with ground truth into the session.

    The GUI used to offer this as a "Load synthetic demo" button; that button is
    gone, but the tests still need a dataset that arrives without images on disk,
    and ground truth is the only way to check the honesty verdict end to end. So
    the loading now happens here, straight through the headless generator, with
    no widget involved.
    """
    from mlti_cal.io.synthetic import generate_dataset
    from mlti_cal.problem.reprojection import extr_key, intr_key, pose_key

    noise = w.detection.noise_spin.value()
    n_cams = max(1, w.detection.camera_table.rowCount())
    system, gt = generate_dataset(num_cameras=n_cams, num_frames=20, pixel_noise_std=noise)
    truth = {intr_key(cid): p for cid, p in gt.camera_params.items()}
    truth.update({extr_key(cid): e for cid, e in gt.camera_extrinsics.items()})
    truth.update({pose_key(f, b): pose for (f, b), pose in gt.board_poses.items()})

    w.session.system = system
    w.session.truth_values = truth
    w.session.pixel_noise_std = noise
    w.session.source = "synthetic"
    w.session.clear_initial()
    w.detection.refresh()
    w.detection.system_changed.emit()


def bootstrap(w) -> None:
    """
    Give the Optimization window the starting point it now insists on.

    Solve and the problem build are both gated on `is_initialized`, deliberately
    -- solving from focal = image width converges to plausible nonsense. Tests
    that drove the pre-gate flow got a `None` problem and, in the threaded case,
    hung on the modal that says so.
    """
    from mlti_cal.problem.initialize import initialize_system

    report = initialize_system(w.session.system)
    w.session.capture_initial(report)
    w.optimization.refresh()


def test_typed_value_removes_columns_and_is_never_moved(qapp):
    """
    Typing a value must change the Jacobian, not just the UI.

    A held parameter is the user saying "I know this one". If it kept its
    column the solver would quietly move it, which is worse than not offering
    the control at all.
    """
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    load_synthetic(w)
    bootstrap(w)
    before = w.session.problem.num_free_params

    opt = w.optimization
    cam_id = next(iter(w.session.system.cameras))
    for index in (2, 3, 4):  # cx, cy, k1
        edit = opt._value_edits[(cam_id, index)]
        edit.setText("111.5")
        opt._value_edited((cam_id, index))
    assert len(opt._pinned) == 3

    assert opt.refresh()
    after = w.session.problem.num_free_params
    assert after == before - 3, f"expected {before - 3} free params, got {after}"
    assert list(w.session.system.cameras[cam_id].params[2:5]) == [111.5, 111.5, 111.5]
    w.close()


def test_switching_a_parameter_off_zeroes_and_holds_it(qapp):
    """Unticking means zero, and zero it must stay -- with no column either."""
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    load_synthetic(w)
    bootstrap(w)
    before = w.session.problem.num_free_params

    opt = w.optimization
    cam_id = next(iter(w.session.system.cameras))
    k3 = w.session.system.cameras[cam_id].model.param_names.index("k3")
    opt._active_toggled((cam_id, k3), False)

    assert opt.refresh()
    assert w.session.problem.num_free_params == before - 1
    assert w.session.system.cameras[cam_id].params[k3] == 0.0
    assert not opt._value_edits[(cam_id, k3)].isEnabled()
    w.close()


def test_clear_drops_every_typed_value_and_switch(qapp):
    """Clear is the one-click undo for this panel: every column comes back."""
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    load_synthetic(w)
    bootstrap(w)
    before = w.session.problem.num_free_params

    opt = w.optimization
    cam_id = next(iter(w.session.system.cameras))
    k3 = w.session.system.cameras[cam_id].model.param_names.index("k3")
    opt._value_edits[(cam_id, 2)].setText("640")
    opt._value_edited((cam_id, 2))
    opt._active_toggled((cam_id, k3), False)
    assert opt.refresh()
    assert w.session.problem.num_free_params == before - 2

    opt._clear_values()
    assert not opt._pinned and not opt._inactive
    assert w.session.problem.num_free_params == before
    assert opt._value_edits[(cam_id, k3)].isEnabled()
    w.close()


def test_live_iterations_are_throttled_but_never_silent(qapp):
    """
    The first step always paints; the rest are rate limited.

    Steps can arrive faster than the window redraws. Painting every one lets the
    queue outlive the solve, so the "live" view finishes after the solve does --
    which is worse than no live view. The first record is exempt: it is what
    tells the user the click did something.
    """
    from mlti_cal.gui.app import MainWindow
    from mlti_cal.solvers.base import IterationRecord

    w = MainWindow()
    opt = w.optimization
    opt._live = []
    opt._last_live_paint = 0.0

    painted: list[str] = []
    opt.log_line = painted.append
    for i in range(50):
        opt._on_iteration(IterationRecord(i, 100.0 - i, rms_px=2.0 - i * 0.01))

    assert len(opt._live) == 50, "every record must be kept, painted or not"
    assert len(painted) == 1, f"only the first step should have painted, got {len(painted)}"
    assert "2" in painted[0]
    w.close()


def test_a_live_solve_prints_rms_not_cost(qapp):
    """RMS is the number that means something; cost only appears without one."""
    from mlti_cal.gui.app import MainWindow
    from mlti_cal.solvers.base import IterationRecord

    w = MainWindow()
    opt = w.optimization
    first = IterationRecord(0, 1000.0, rms_px=4.0)
    row = opt._iteration_row(IterationRecord(7, 250.0, rms_px=1.0), first)
    assert "px" in row and "1" in row
    assert "-75.000%" in row, row  # 4.0 -> 1.0 in RMS, not 1000 -> 250 in cost
    assert "RMS px" in opt._iteration_header(first)

    # A backend that measures no RMS still gets a readable table.
    bare = IterationRecord(0, 1000.0)
    assert "cost" in opt._iteration_header(bare)
    assert "px" not in opt._iteration_row(IterationRecord(7, 250.0), bare)
    w.close()


def test_a_finished_solve_does_not_switch_tabs(qapp):
    """Staying put: the RMS and the iteration log are what to read first."""
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    w.tabs.setCurrentWidget(w.optimization)
    w.optimization.solved.emit()
    assert w.tabs.currentWidget() is w.optimization
    w.close()


def test_core_parameters_cannot_be_switched_off(qapp):
    """fx, fy, cx, cy have no off switch: a camera without them cannot project."""
    from PySide6.QtWidgets import QCheckBox

    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    load_synthetic(w)
    bootstrap(w)
    tree = w.optimization.tree
    cam_item = tree.topLevelItem(0)
    names = w.session.system.cameras[next(iter(w.session.system.cameras))].model.param_names
    for j, name in enumerate(names):
        check = tree.itemWidget(cam_item.child(j), 2)
        assert isinstance(check, QCheckBox)
        assert check.isEnabled() == (j >= 4), f"{name}: enabled={check.isEnabled()}"
    w.close()


def test_parameters_are_visible_before_the_estimate(qapp):
    """The tree is the place to type a value you already know -- before the fit."""
    from mlti_cal.gui.app import MainWindow

    w = MainWindow()
    load_synthetic(w)
    assert not w.session.is_initialized
    assert not w.optimization.refresh(), "no problem can be built without a starting point"

    tree = w.optimization.tree
    assert tree.topLevelItemCount() > 0, "parameters must be listed before the estimate"
    cam_id = next(iter(w.session.system.cameras))
    edits = [w.optimization._value_edits[(cam_id, i)] for i in range(4)]
    assert all(e.text() == "" for e in edits), "values must be blank, not crude defaults"
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
