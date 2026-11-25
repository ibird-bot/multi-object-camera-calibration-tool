"""
Film the GUI: drive it through its real slots and record what it draws.

Frames are grabbed by Qt itself (`QWidget.grab`), not by a screen recorder:
nothing takes over the mouse, other windows can never appear in the footage,
and every frame is pixel-exact even when the window is covered. The cursor is
NOT in the frames. The director logs where it would be and when it clicks, and
`edit.py` draws it afterwards -- smooth at any output frame rate, whatever rate
the grabs happened at.

A script is a generator that yields commands (`Wait`, `Until`); the helpers on
`Director` log timeline events and return those commands, so a shot reads as

    yield d.move_to(view.detect_btn, 0.8)
    yield d.click(view.detect_btn)
    yield Until(lambda: view.detect_btn.isEnabled(), timeout=120)

Shots are STAGED, not cropped: `stage(widget, ...)` shows only the subject and
the window has the shape of the output picture, so the subject fills it.

Outputs, in the take directory:
    frames/NNNNNN.jpg   window grabs
    frames.csv          index, seconds since start
    events.json         window size, device pixel ratio, timeline events
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QWidget


def _ui_scale() -> float:
    from mlti_cal.gui import theme

    return theme.current_scale()


@dataclass
class Wait:
    seconds: float


@dataclass
class Until:
    predicate: Callable[[], bool]
    timeout: float = 60.0
    label: str = ""


Script = Generator[Wait | Until, None, None]


class Director:
    def __init__(
        self, window: QWidget, take_dir: str | Path, fps: float = 20.0, scale: float = 2.0
    ):
        self.window = window
        self.scale = scale
        self.take = Path(take_dir)
        (self.take / "frames").mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.events: list[dict] = []
        self._t0 = time.perf_counter()
        self._frames: list[tuple[int, float]] = []
        self._cursor = (window.width() / 2, window.height() / 2)
        self._grab_timer = QTimer()
        self._grab_timer.timeout.connect(self._grab)
        self.failed: str | None = None

    # -- timeline ----------------------------------------------------------
    def now(self) -> float:
        return time.perf_counter() - self._t0

    def log(self, kind: str, **data) -> None:
        self.events.append({"t": round(self.now(), 4), "type": kind, **data})

    def _point(self, target: QWidget | QPoint | tuple[float, float]) -> tuple[float, float]:
        if isinstance(target, QWidget):
            c = target.mapTo(self.window, target.rect().center())
            return (c.x(), c.y())
        if isinstance(target, QPoint):
            return (target.x(), target.y())
        return (float(target[0]), float(target[1]))

    def rect_of(self, *widgets: QWidget, pad: int = 8) -> list[int]:
        """The rectangle covering `widgets`, in window coordinates, for the editor to zoom on."""
        r = QRect()
        for widget in widgets:
            r = r.united(QRect(widget.mapTo(self.window, QPoint(0, 0)), widget.size()))
        r = r.adjusted(-pad, -pad, pad, pad)
        return [r.x(), r.y(), r.width(), r.height()]

    # -- commands ------------------------------------------------------------
    def move_to(self, target, seconds: float = 0.7) -> Wait:
        to = self._point(target)
        self.log("move", t1=round(self.now() + seconds, 4), frm=self._cursor, to=to)
        self._cursor = to
        return Wait(seconds)

    def click(
        self, widget: QWidget | None = None, action: Callable[[], None] | None = None
    ) -> Wait:
        """Log a click at the cursor, then fire the widget's real `click()`."""
        self.log("click", at=self._cursor)
        fire = action or (widget.click if widget is not None else None)
        if fire is not None:
            QTimer.singleShot(150, fire)
        return Wait(0.35)

    def mark(self, name: str, **data) -> None:
        """Shot boundaries, captions and focus regions, for the editor."""
        self.log("mark", name=name, **data)

    def stage(
        self, *widgets: QWidget, size: tuple[int, int] | None = None, settle: float = 0.5
    ) -> Wait:
        """
        Show only `widgets` in the window (`MainWindow.focus_mode`), size the
        window to the shot (`size`, logical px), then let the layout settle.
        The subject then fills the picture instead of being cropped out of the
        full interface; a short subject gets a short window, not empty space.
        """
        self.window.focus_mode(*widgets)
        if size is not None:
            self.window.setFixedSize(*size)
        self.log("stage", window=list(size or (self.window.width(), self.window.height())))
        return Wait(settle)

    # -- recording -----------------------------------------------------------
    #: While True, no frames are grabbed (the editor shows the last one).
    paused = False

    def _grab(self) -> None:
        # Paused around the report build: twice, a grab during that build froze
        # the GUI thread inside QWidget.render (stack dumped by the watchdog),
        # while the worker was computing the report. Not grabbing there avoids
        # it; that wait is fast-forwarded to about a second of output anyway.
        if self.paused:
            return
        # Rendered into a pixmap `scale` times larger than the window, so text
        # and vector drawing are sharp when the editor zooms in on a region.
        index = len(self._frames)
        t = self.now()
        pix = QPixmap(self.window.size() * self.scale)
        pix.setDevicePixelRatio(self.scale)
        pix.fill(Qt.white)
        self.window.render(pix)
        pix.save(str(self.take / "frames" / f"{index:06d}.jpg"), "JPG", 92)
        self._frames.append((index, round(t, 4)))

    def run(self, script: Script) -> None:
        """Run `script` inside the Qt event loop, recording until it ends."""
        app = QApplication.instance()
        dpr = self.window.devicePixelRatioF()

        def finish() -> None:
            self._grab_timer.stop()
            with open(self.take / "frames.csv", "w", encoding="utf-8") as f:
                f.write("index,t\n")
                f.writelines(f"{i},{t}\n" for i, t in self._frames)
            (self.take / "events.json").write_text(
                json.dumps(
                    {
                        "window": [self.window.width(), self.window.height()],
                        "device_pixel_ratio": dpr,
                        "frame_scale": self.scale,
                        # Smallest UI text (9 pt captions) in WINDOW pixels, at the
                        # presentation scale in force: the editor's legibility check.
                        "min_font_px": 9 * 96 / 72 * _ui_scale(),
                        "failed": self.failed,
                        "events": self.events,
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
            app.quit()

        def step() -> None:
            try:
                cmd = next(script)
            except StopIteration:
                finish()
                return
            except Exception as exc:  # a broken script must still save the take
                self.failed = f"{type(exc).__name__}: {exc}"
                finish()
                return
            if isinstance(cmd, Wait):
                QTimer.singleShot(int(cmd.seconds * 1000), step)
            else:
                deadline = time.perf_counter() + cmd.timeout

                def poll(cmd=cmd, deadline=deadline) -> None:
                    if cmd.predicate():
                        step()
                    elif time.perf_counter() > deadline:
                        self.failed = f"timed out waiting for {cmd.label or 'a condition'}"
                        finish()
                    else:
                        QTimer.singleShot(100, poll)

                poll()

        self._t0 = time.perf_counter()
        self._grab_timer.start(int(1000 / self.fps))
        QTimer.singleShot(300, step)
        app.exec()
