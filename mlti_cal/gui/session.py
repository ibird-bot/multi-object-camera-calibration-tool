"""
Shared GUI state and the background solve worker.

The session owns the CalibrationSystem, the built Problem and the last report.
It computes nothing itself -- every number comes from the headless core, so a
GUI run and a CLI run cannot disagree. Views read from here and emit signals;
they never touch numpy directly beyond formatting for display.

Solving runs on a QThread. A calibration solve is seconds to minutes; doing it
on the GUI thread would freeze the window and, worse, make the progress display
lie about what the solver is doing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QObject, QThread, Signal

from mlti_cal.problem.graph import Problem
from mlti_cal.problem.loss import make_loss
from mlti_cal.problem.reprojection import build_problem, write_back
from mlti_cal.problem.types import CalibrationSystem
from mlti_cal.report.report import CalibrationReport, build_report
from mlti_cal.solvers import SolveOptions, SolveResult, get_backend


@dataclass
class Session:
    """Everything the three views share."""

    system: CalibrationSystem = field(default_factory=CalibrationSystem)
    problem: Problem | None = None
    result: SolveResult | None = None
    report: CalibrationReport | None = None
    truth_values: dict | None = None  # populated only in synthetic mode
    pixel_noise_std: float = 0.3
    images: dict[tuple[str, str], object] = field(default_factory=dict)
    source: str = "none"

    def rebuild_problem(
        self,
        loss_name: str | None = None,
        loss_scale: float = 2.0,
        fixed_intrinsics: dict[str, list[int]] | None = None,
        optimize_intrinsics: bool = True,
        optimize_extrinsics: bool = True,
        optimize_poses: bool = True,
    ) -> Problem:
        self.problem = build_problem(
            self.system,
            loss=make_loss(loss_name, loss_scale),
            fixed_intrinsic_components=fixed_intrinsics,
            optimize_intrinsics=optimize_intrinsics,
            optimize_extrinsics=optimize_extrinsics,
            optimize_poses=optimize_poses,
        )
        return self.problem

    def commit(self) -> None:
        if self.problem is not None:
            write_back(self.system, self.problem)

    def make_report(self, range_m: float = 1.5, do_crossval: bool = False) -> CalibrationReport:
        if self.problem is None:
            raise RuntimeError("no problem built")
        self.report = build_report(
            self.problem,
            self.system,
            solve_result=self.result,
            pixel_noise_std=self.pixel_noise_std,
            default_range_m=range_m,
            do_crossval=do_crossval,
            truth_values=self.truth_values,
        )
        return self.report

    @property
    def is_ready_to_solve(self) -> bool:
        return bool(self.system.cameras) and bool(self.system.observations)


class SolveWorker(QObject):
    """Runs one solve off the GUI thread."""

    finished = Signal(object)  # SolveResult
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, problem: Problem, backend: str, options: SolveOptions):
        super().__init__()
        self.problem = problem
        self.backend = backend
        self.options = options

    def run(self) -> None:
        try:
            self.progress.emit(f"solving with {self.backend}...")
            result = get_backend(self.backend).solve(self.problem, self.options)
            self.finished.emit(result)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ReportWorker(QObject):
    """Builds the report off the GUI thread (cross-validation is slow)."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, session: Session, range_m: float, do_crossval: bool):
        super().__init__()
        self.session = session
        self.range_m = range_m
        self.do_crossval = do_crossval

    def run(self) -> None:
        try:
            self.progress.emit(
                "building report" + (" with cross-validation..." if self.do_crossval else "...")
            )
            self.finished.emit(self.session.make_report(self.range_m, do_crossval=self.do_crossval))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def run_in_thread(parent, worker: QObject, on_done, on_fail, on_progress=None):
    """
    Wire a worker onto a QThread and start it.

    The thread and worker are stashed on `parent` because PySide6 will garbage
    collect a QThread that nothing references, killing the job mid-flight -- a
    classic and very confusing crash.
    """
    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    def cleanup():
        thread.quit()
        thread.wait()

    worker.finished.connect(lambda r: (cleanup(), on_done(r)))
    worker.failed.connect(lambda m: (cleanup(), on_fail(m)))
    if on_progress is not None:
        worker.progress.connect(on_progress)
    parent._active_thread = thread
    parent._active_worker = worker
    thread.start()
    return thread
