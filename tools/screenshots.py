"""
Screenshots of every page of the GUI, in light and dark, on the rig_4cam example.

    python tools/screenshots.py Output/screens

Drives the real application -- detection, starting point, Ceres solve,
cross-validation, report -- then grabs each page. For reviewing the look of
the interface after a change; needs `examples/rig_4cam/images` and Ceres.
The theme is switched for the grab only: the user's saved choice is untouched.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples" / "rig_4cam" / "config.json"
SIZE = (1600, 960)


def pump(app, seconds: float = 0.0) -> None:
    end = time.monotonic() + seconds
    while True:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if time.monotonic() >= end:
            return
        time.sleep(0.01)


def wait_for(app, done, timeout: float, label: str) -> None:
    start = time.monotonic()
    while not done():
        if time.monotonic() - start > timeout:
            raise TimeoutError(label)
        pump(app, 0.05)
    pump(app, 0.4)


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "Output" / "screens")
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv[:1])
    from mlti_cal.gui import theme
    from mlti_cal.gui.app import MainWindow
    from mlti_cal.io.config import CalibrationConfig

    w = MainWindow()
    w.resize(*SIZE)
    w.show()
    pump(app, 0.5)
    det, opt, rep, s = w.detection, w.optimization, w.report, w.session

    det._apply_config(CalibrationConfig.load(CONFIG))
    det._run_detection()
    wait_for(app, lambda: det.detect_btn.isEnabled() and s.system.observations, 300, "detect")
    strip = det.thumbnails
    strip.setCurrentItem(strip.item_for("cam2/0005"))
    w.tabs.setCurrentWidget(opt)
    opt._estimate()
    wait_for(app, lambda: s.is_initialized and opt.init_btn.isEnabled(), 300, "estimate")
    opt.backend_combo.setCurrentIndex(opt.backend_combo.findData("ceres"))
    opt._solve()
    wait_for(app, lambda: s.result is not None and opt.solve_btn.isEnabled(), 300, "solve")
    opt._cross_validate()
    wait_for(app, lambda: s.crossval is not None and opt.crossval_btn.isEnabled(), 300, "crossval")
    w.tabs.setCurrentWidget(rep)
    rep._build_report()
    wait_for(app, lambda: s.report is not None and rep.build_btn.isEnabled(), 600, "report")

    manager = theme.manager()
    saved = manager._mode
    shots = [
        ("1_setup", lambda: w.tabs.setCurrentWidget(det)),
        ("2_solve", lambda: w.tabs.setCurrentWidget(opt)),
        ("3_report_overview", lambda: (w.tabs.setCurrentWidget(rep), rep.show_page("Overview"))),
        ("4_report_findings", lambda: rep.show_page("Findings")),
        (
            "5_report_correlation",
            lambda: (rep.show_page("Figures"), rep.show_figure("correlation")),
        ),
        ("6_report_coverage", lambda: rep.show_figure("coverage")),
    ]
    try:
        for mode in ("light", "dark"):
            manager._mode = mode  # not set_mode: that would save the choice
            manager.apply()
            pump(app, 0.6)
            for name, show in shots:
                show()
                pump(app, 0.8)
                path = out / f"{name}_{mode}.png"
                w.grab().save(str(path))
                print("wrote", path)
    finally:
        manager._mode = saved
        manager.apply()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
