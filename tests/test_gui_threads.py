"""
Worker-thread lifetime: a second job must not kill the first.

`run_in_thread` used to stash the running job in a single attribute --
`parent._active_thread = thread`. `OptimizationView` starts four different
workers with itself as parent, and each completion handler re-enables only its
OWN button, so nothing stopped a user from pressing "Estimate starting point"
while a solve was running. The second call rebound the attribute, dropped the
last Python reference to the solve's QThread, and PySide6 collected a thread
that was still executing.

These tests are deliberately about the mechanism rather than about any one
button: the failure mode is generic to the helper, and a test per button would
pass while the next worker added reintroduced it.
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from mlti_cal.gui.session import busy, run_in_thread  # noqa: E402


class SlowWorker(QObject):
    """Blocks until `release()` is called, so a second job can start mid-run."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self):
        super().__init__()
        self._released = False
        self.ran = False

    def release(self) -> None:
        self._released = True

    def run(self) -> None:
        self.ran = True
        deadline = time.monotonic() + 10.0
        while not self._released and time.monotonic() < deadline:
            time.sleep(0.005)
        self.finished.emit("done")


class Receiver(QObject):
    """Bound-method slots, as `run_in_thread` requires."""

    def __init__(self):
        super().__init__()
        self.done = []
        self.failures = []

    def on_done(self, payload) -> None:
        self.done.append(payload)

    def on_fail(self, message: str) -> None:
        self.failures.append(message)


def pump(app, predicate, timeout=10.0) -> bool:
    """Spin the event loop until `predicate` holds, without blocking on wait()."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication

    existing = QApplication.instance()
    yield existing or QApplication([])


def test_a_second_job_does_not_drop_the_first_threads_reference(app):
    """
    The regression itself.

    Both threads must still be held, and BOTH must run to completion -- under
    the single-slot version the first thread's only reference was gone the
    moment the second started.
    """
    parent = QWidget()
    first, second = SlowWorker(), SlowWorker()
    rx = Receiver()

    t1 = run_in_thread(parent, first, rx.on_done, rx.on_fail)
    assert pump(app, lambda: first.ran), "first worker never started"

    # The second job starts while the first is still inside `run`.
    t2 = run_in_thread(parent, second, rx.on_done, rx.on_fail)
    assert pump(app, lambda: second.ran), "second worker never started"

    assert len(parent._live_jobs) == 2, "starting a second job dropped the first"
    assert t1.isRunning() and t2.isRunning()

    first.release()
    second.release()
    assert pump(app, lambda: len(rx.done) == 2), f"only {len(rx.done)} of 2 jobs completed"
    assert pump(app, lambda: not t1.isRunning() and not t2.isRunning())
    assert t1.wait(5000) and t2.wait(5000)


def test_a_finished_job_is_released(app):
    """The set must not grow without bound over a long session."""
    parent = QWidget()
    worker = SlowWorker()
    rx = Receiver()

    thread = run_in_thread(parent, worker, rx.on_done, rx.on_fail)
    assert pump(app, lambda: worker.ran)
    assert busy(parent)

    worker.release()
    # Pumped, never `thread.wait()` here: `worker.finished -> thread.quit` is a
    # QUEUED connection, so blocking the GUI thread on wait() is what stops the
    # quit from ever being delivered. Same trap the helper's docstring names.
    assert pump(app, lambda: not thread.isRunning()), "thread never stopped"
    assert pump(app, lambda: not parent._live_jobs), "finished job was never released"
    assert not busy(parent)


def test_busy_is_false_before_anything_starts(app):
    assert not busy(QWidget())
