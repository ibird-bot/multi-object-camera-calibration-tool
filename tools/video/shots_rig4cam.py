"""
The shot list for the rig_4cam demo: the real GUI, start to finish, STAGED.

    python tools/video/shots_rig4cam.py Output/video/take_vertical --format vertical
    python tools/video/shots_rig4cam.py Output/video/take_landscape --format landscape

Needs the example images (`python examples/rig_4cam/make_dataset.py`) and the
Ceres backend.

Every shot is staged: `d.stage(...)` shows only the shot's subject and the
window has the shape of the output picture (square for the vertical cut,
2:1 for the landscape one), at a presentation text scale chosen so the
smallest text lands at >= 22 px in the output. The editor then uses whole
frames -- nothing is cropped out of a dense layout, which is what made the
first cut show thin strips and half-cut lines.

Marks logged here are what `edit.py` cuts on:

    shot:<id>       a shot starts (its subject is staged and settled)
    end:<id>        it ends
    wait:<id> / waited:<id>   a stretch the editor may speed up
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
from pathlib import Path

from director import Director, Script, Until, Wait  # noqa: E402  (run as a script)
from PySide6.QtWidgets import QApplication, QScrollArea

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples" / "rig_4cam" / "config.json"
#: A frame of the wide-angle camera with two boards in view.
PREVIEW_KEY = "cam2/0005"

#: Per format: window size (logical px), capture scale, UI text scale.
#: 720 x 1.5 = 1080 output px: frames come out at the output size, no resampling.
FORMATS = {
    "vertical": {"window": (720, 720), "capture": 1.5, "text": 1.4, "thumb": 196},
    "landscape": {"window": (1280, 640), "capture": 1.5, "text": 1.25, "thumb": 210},
}

#: Window size per shot (logical px). The width is the format's; the height
#: fits the subject, so a short panel is not filmed with empty space below it
#: and a tall one is not cut off. Checked on contact sheets of every take.
SHOT_SIZE = {
    "vertical": {
        "setup": (720, 720),
        "detect": (720, 720),
        "preview": (720, 450),
        "estimate": (720, 430),
        "solve": (720, 720),
        "crossval": (720, 720),
        "report": (720, 470),
        "coverage": (720, 470),
        "findings": (720, 400),
    },
    "landscape": {
        "setup": (1280, 640),
        "detect": (1280, 640),
        "preview": (1280, 640),
        "estimate": (1280, 420),
        "solve": (1280, 640),
        "crossval": (1280, 640),
        "report": (1280, 440),
        "coverage": (1280, 400),
        "findings": (900, 330),
    },
}


def script(d: Director, w, fmt: str) -> Script:
    from mlti_cal.io.config import CalibrationConfig

    det, opt, rep, s = w.detection, w.optimization, w.report, w.session
    vertical = fmt == "vertical"
    size = SHOT_SIZE[fmt]

    # -- 1. setup: four cameras, three boards, already described ---------------
    det._apply_config(CalibrationConfig.load(CONFIG))
    # Only the columns the caption talks about: wider ones would be cut off.
    keep = 4 if vertical else 5
    for col in range(keep, det.board_table.columnCount()):
        det.board_table.setColumnHidden(col, True)
    if vertical:  # folder paths would be cut short at this width; the model column is the point
        for col in (2, 3):
            det.camera_table.setColumnHidden(col, True)
    yield d.stage(
        det.camera_table.parentWidget(), det.board_table.parentWidget(), size=size["setup"]
    )
    d.mark("shot:setup")
    yield d.move_to(det.camera_table, 0.9)
    yield Wait(1.2)
    yield d.move_to(det.board_table, 0.8)
    yield Wait(1.6)
    d.mark("end:setup")
    for col in range(det.board_table.columnCount()):
        det.board_table.setColumnHidden(col, False)
    for col in range(det.camera_table.columnCount()):
        det.camera_table.setColumnHidden(col, False)

    # -- 2. detection: the capture lights up, image by image -------------------
    yield d.stage(det.thumbnails, size=size["detect"])
    d.mark("shot:detect")
    det._run_detection()
    d.mark("wait:detect")
    yield Until(lambda: det.detect_btn.isEnabled() and s.system.observations,
                timeout=240, label="detection")  # fmt: skip
    d.mark("waited:detect")
    yield Wait(0.6)
    d.mark("end:detect")

    # -- 3. one frame of the wide-angle camera: corners per board ----------------
    strip = det.thumbnails
    strip.setCurrentItem(strip.item_for(PREVIEW_KEY))
    yield d.stage(det.image_view, size=size["preview"])
    d.mark("shot:preview")
    yield Wait(3.0)
    d.mark("end:preview")

    # -- 4. starting point: the chained camera ---------------------------------
    w.tabs.setCurrentWidget(opt)
    scroll = opt.findChild(QScrollArea, "init_summary_scroll")
    scroll.setMaximumHeight(16777215)  # the whole summary, no inner scrolling
    yield d.stage(opt.init_box, size=size["estimate"])
    d.mark("shot:estimate")
    yield d.move_to(opt.init_btn, 0.8)
    yield d.click(opt.init_btn)
    d.mark("wait:estimate")
    yield Until(lambda: s.is_initialized and opt.init_btn.isEnabled(),
                timeout=300, label="starting point")  # fmt: skip
    d.mark("waited:estimate")
    yield Wait(0.4)
    yield d.move_to(opt.init_summary, 0.7)
    yield Wait(2.6)
    d.mark("end:estimate")

    # -- 5. solve: cards and the live descent -------------------------------------
    combo = opt.backend_combo
    combo.setCurrentIndex(combo.findData("ceres"))
    if vertical:
        opt.steps_card.hide()  # two cards fit a square picture; the caption says "Ceres"
    yield d.stage(opt.solve_btn, opt.results, opt.progress_plot, size=size["solve"])
    d.mark("shot:solve")
    yield d.move_to(opt.solve_btn, 0.8)
    yield d.click(opt.solve_btn)
    d.mark("wait:solve")
    yield Until(lambda: s.result is not None and opt.solve_btn.isEnabled(),
                timeout=300, label="solve")  # fmt: skip
    d.mark("waited:solve")
    yield d.move_to(opt.error_card, 0.7)
    yield Wait(2.6)
    d.mark("end:solve")

    # -- 6. held-out error, on the same panel ------------------------------------
    yield d.stage(opt.crossval_btn, opt.results, opt.progress_plot, size=size["crossval"])
    d.mark("shot:crossval")
    yield d.move_to(opt.crossval_btn, 0.8)
    yield d.click(opt.crossval_btn)
    d.mark("wait:crossval")
    yield Until(lambda: s.crossval is not None and opt.crossval_btn.isEnabled(),
                timeout=300, label="cross-validation")  # fmt: skip
    d.mark("waited:crossval")
    yield d.move_to(opt.heldout_card, 0.7)
    yield Wait(2.6)
    d.mark("end:crossval")
    opt.steps_card.show()

    # -- 7. report: build, then where the data is thin ----------------------------
    w.tabs.setCurrentWidget(rep)
    yield d.stage(rep.build_btn, rep.banner, rep.top_findings, size=size["report"])
    d.mark("shot:report")
    yield d.move_to(rep.build_btn, 0.8)
    yield d.click(rep.build_btn)
    d.mark("wait:report")
    d.paused = True  # see Director._grab: grabbing during this build has hung twice
    yield Until(lambda: s.report is not None and rep.build_btn.isEnabled(),
                timeout=600, label="report")  # fmt: skip
    yield Wait(0.4)  # let the verdict paint before grabbing again
    d.paused = False
    d.mark("waited:report")
    yield Wait(1.2)  # the verdict, briefly
    d.mark("end:report")
    # Not the uncertainty map: its centre value is dominated by how loosely the
    # camera frame is defined (see the audit), not by a physical projection
    # error, so it is not a number to put on screen.
    # The coverage NUMBERS rather than the coverage figure: the figure is two
    # panels designed for a page and collapses in a phone-shaped picture.
    rep.show_page("Overview")
    yield d.stage(rep.coverage_card, rep.camera_table, size=size["coverage"], settle=0.8)
    d.mark("shot:coverage")
    yield Wait(3.0)
    d.mark("end:coverage")

    # -- 8. ...and what it cannot tell apart ---------------------------------------
    rep.show_page("Findings")
    cards = [
        rep.findings_layout.itemAt(i).widget()
        for i in range(rep.findings_layout.count())
        if rep.findings_layout.itemAt(i).widget() is not None
        and rep.findings_layout.itemAt(i).widget().property("finding")
    ]
    # The correlation finding when there is one; otherwise the first finding.
    card = next((c for c in cards if c.property("finding") == "extreme_correlation"), cards[0])
    yield d.stage(card, size=size["findings"], settle=0.6)
    d.mark("shot:findings")
    yield Wait(3.2)
    d.mark("end:findings")
    w.end_focus()


def main() -> int:
    global CONFIG, PREVIEW_KEY
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("take", nargs="?", default=str(ROOT / "Output" / "video" / "take"))
    p.add_argument("--format", choices=sorted(FORMATS), default="vertical")
    p.add_argument("--config", default=str(CONFIG), help="calibration config to film")
    p.add_argument("--preview", default=PREVIEW_KEY, help="camera/frame shown with its corners")
    a = p.parse_args()
    CONFIG, PREVIEW_KEY = Path(a.config).resolve(), a.preview
    spec = FORMATS[a.format]
    take = Path(a.take).resolve()
    # A take is ~1 minute. If it hangs, say where -- every thread's stack --
    # and stop, instead of sitting there with the window open.
    faulthandler.dump_traceback_later(300, exit=True)
    # The example config names its image folders relative to the repository
    # root; started from anywhere else, detection finds no images at all.
    os.chdir(ROOT)

    _app = QApplication(sys.argv[:1])  # must stay referenced while Qt runs
    from mlti_cal.gui import theme
    from mlti_cal.gui.app import MainWindow

    manager = theme.manager()
    manager._mode = "dark"  # for this recording only: not saved as the user's choice
    manager.set_presentation(spec["text"])
    from mlti_cal.gui.widgets import ThumbnailStrip

    # Bigger thumbnails for the film; they are generated at this size, not upscaled.
    ThumbnailStrip.THUMB = spec["thumb"]
    from mlti_cal.gui.session import DetectionWorker

    DetectionWorker.REDUCTION = 4  # thumbnails drawn from 320 px images, not upscaled
    w = MainWindow()
    # Fixed for the whole take: every frame has the output picture's shape.
    w.setFixedSize(*spec["window"])
    w.show()
    director = Director(w, take, scale=spec["capture"])
    director.run(script(director, w, a.format))
    print("take written to", take, "| failed:", director.failed)
    return 0 if director.failed is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
