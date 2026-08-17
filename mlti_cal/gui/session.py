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

from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from mlti_cal.detectors.charuco import draw_detections
from mlti_cal.detectors.registry import default_detector_settings
from mlti_cal.io.config import CalibrationConfig, build_system_from_config
from mlti_cal.problem.graph import Problem
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.loss import make_loss
from mlti_cal.problem.reprojection import (
    build_problem,
    restore_values,
    snapshot_values,
    write_back,
)
from mlti_cal.problem.settings import InitSettings
from mlti_cal.problem.types import CalibrationSystem
from mlti_cal.report.report import CalibrationReport, build_report
from mlti_cal.report.settings import ReportSettings
from mlti_cal.solvers import SolveOptions, SolveResult, get_backend
from mlti_cal.solvers.base import per_corner_rms


@dataclass
class Session:
    """Everything the three views share."""

    system: CalibrationSystem = field(default_factory=CalibrationSystem)
    problem: Problem | None = None
    result: SolveResult | None = None
    report: CalibrationReport | None = None
    truth_values: dict | None = None  # populated only in synthetic mode
    #: None means the user does not claim to know it -- see CalibrationConfig.
    pixel_noise_std: float | None = 0.3
    images: dict[tuple[str, str], object] = field(default_factory=dict)
    source: str = "none"
    #: Detector knobs per kind, edited from the Detectors menu. Shared state
    #: rather than a widget's property: the settings window is opened from the
    #: menu bar, the Setup tab writes them into the config, and the noise
    #: measurement reads them -- three owners, so none of them owns it.
    detector_settings: dict = field(default_factory=default_detector_settings)

    #: True when a detector setting changed after these observations were found.
    #: The corners in `system` were produced by the PREVIOUS settings, so any
    #: solve run now reports a result computed from data the displayed settings
    #: did not produce. Set by the setup view, read wherever a solve begins.
    detection_stale: bool = False

    #: Last cross-validation result, if the user asked for one. Kept on the
    #: session so the Report tab can show the same numbers the Optimization
    #: tab printed, instead of re-running four solves to get them again.
    crossval: object | None = None

    #: Bootstrap output, kept because `write_back` overwrites the live fields.
    initial_values: dict | None = None
    initial_report: dict | None = None
    initial_rms: float | None = None  # per-corner reprojection RMS, pixels

    @property
    def is_initialized(self) -> bool:
        """True once the user has run the bootstrap on the current system."""
        return self.initial_values is not None

    def capture_initial(self, report: dict | None = None) -> None:
        """Freeze the current (just-bootstrapped) system as the initial estimate."""
        self.initial_values = snapshot_values(self.system)
        self.initial_report = report
        self.initial_rms = per_corner_rms(build_problem(self.system))

    def restore_initial(self) -> int:
        """Put the bootstrap values back into the system. Returns blocks restored."""
        if not self.initial_values:
            return 0
        return restore_values(self.system, self.initial_values)

    def clear_initial(self) -> None:
        """Drop the estimate. Called when the observations underneath it change."""
        self.initial_values = None
        self.initial_report = None
        self.initial_rms = None

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

    def make_report(
        self,
        range_m: float = 1.5,
        do_crossval: bool = False,
        settings: ReportSettings | None = None,
    ) -> CalibrationReport:
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
            settings=settings,
        )
        return self.report

    @property
    def is_ready_to_solve(self) -> bool:
        return bool(self.system.cameras) and bool(self.system.observations)


class InitWorker(QObject):
    """
    Runs the bootstrap off the GUI thread.

    Off-thread because `initialize_system` calls `cv2.calibrateCamera` once per
    camera over every view; on a real dataset that is seconds of solid compute
    and would freeze the window if run from the button's click handler.
    """

    finished = Signal(object)  # report dict from initialize_system
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        system: CalibrationSystem,
        do_intrinsics: bool = True,
        fixed_intrinsics: dict[str, dict[int, float]] | None = None,
        settings: InitSettings | None = None,
    ):
        super().__init__()
        self.system = system
        self.do_intrinsics = do_intrinsics
        #: Intrinsic components the user typed in: known, so not estimated.
        self.fixed_intrinsics = fixed_intrinsics
        #: Bootstrap knobs from the Starting point panel.
        self.settings = settings

    def run(self) -> None:
        try:
            self.progress.emit("estimating starting point...")
            report = initialize_system(
                self.system,
                do_intrinsics=self.do_intrinsics,
                fixed_intrinsics=self.fixed_intrinsics,
                on_progress=self.progress.emit,
                settings=self.settings,
            )
            self.finished.emit(report)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class SolveWorker(QObject):
    """Runs one solve off the GUI thread."""

    finished = Signal(object)  # SolveResult
    failed = Signal(str)
    progress = Signal(str)
    #: One per solver step, emitted while the solve is still running. Its own
    #: channel rather than `progress`: hundreds of these must not crowd out the
    #: handful of one-line status messages.
    iteration = Signal(object)  # IterationRecord

    def __init__(self, problem: Problem, backend: str, options: SolveOptions, live: bool = False):
        super().__init__()
        self.problem = problem
        self.backend = backend
        self.options = options
        if live:
            # Emitting from the worker thread is the point: the receiver is a
            # GUI-thread QObject, so Qt queues each record and the window
            # repaints between steps instead of after the whole solve.
            self.options.on_iteration = self.iteration.emit

    def run(self) -> None:
        try:
            self.progress.emit(f"solving with {self.backend}...")
            result = get_backend(self.backend).solve(self.problem, self.options)
            self.finished.emit(result)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DetectionWorker(QObject):
    """
    Runs detection over every configured image off the GUI thread.

    Detection is the longest blocking operation in the app -- every file of
    every camera is decoded and searched -- so running it inline froze the
    window for the entire run with no repaint, which is indistinguishable from
    the button doing nothing.

    The per-image thumbnail is built HERE, on the worker thread, and only the
    128 px-scale result is emitted. Handing the full-resolution frame to the
    GUI instead would be simpler and wrong: queued signals buffer, so a fast
    detector feeding a busy GUI thread would pile up decoded full-size frames
    until memory ran out. Downscaling first bounds the payload at a few tens of
    kilobytes no matter how far behind the GUI falls.
    """

    #: Same 1/8 rule the folder-pick path decodes at, so icons do not visibly
    #: change sharpness as detection overwrites them.
    REDUCTION = 8
    #: Corners are drawn in one fixed red here rather than the per-board
    #: palette: on an icon the question is "did this frame work at all".
    CORNER_BGR = (0, 0, 255)

    finished = Signal(object)  # (CalibrationSystem, stats)
    failed = Signal(str)
    progress = Signal(str)
    image_started = Signal(str)  # thumbnail key
    image_done = Signal(str, object, int)  # key, BGR thumbnail or None, corner count

    def __init__(self, config: CalibrationConfig):
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            self.progress.emit("detecting...")
            system, stats = build_system_from_config(self.config, on_image=self._on_image)
            # Detection stops at detection. The bootstrap is a separate, explicit
            # user action (InitWorker) so its estimate is seen and accepted rather
            # than silently produced and then overwritten by the first solve.
            self.finished.emit((system, stats))
        except Exception as exc:  # surfaced in the UI, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    # -- core -> Qt boundary, called on THIS thread -----------------------
    def _on_image(self, prog) -> None:
        key = f"{prog.camera}/{prog.path.stem}"
        if prog.stage == "start":
            self.image_started.emit(key)
            return
        thumb = None if prog.image is None else self._thumbnail(prog.image, prog.detections)
        self.image_done.emit(key, thumb, prog.num_corners)

    @classmethod
    def _thumbnail(cls, image, detections):
        """Downscale, then draw the corners at the SCALED coordinates."""
        h, w = image.shape[:2]
        small = cv2.resize(
            image,
            (max(1, w // cls.REDUCTION), max(1, h // cls.REDUCTION)),
            interpolation=cv2.INTER_AREA,
        )
        # Integer division makes the true factor differ from 1/8; measure it off
        # the result rather than assuming, or the dots land off their corners.
        scale = small.shape[1] / w
        scaled = [
            replace(d, image_points=np.asarray(d.image_points, dtype=float) * scale)
            for d in detections
        ]
        return draw_detections(small, scaled, radius=2, labels=False, colour=cls.CORNER_BGR)


class NoiseWorker(QObject):
    """
    Runs detection over a static sequence and estimates the pixel noise.

    Off-thread for the same reason as DetectionWorker: every file is decoded
    and searched. Only the corner count is emitted per image -- the frames
    themselves are held in the core until the estimate is complete, and
    shipping thumbnails here would buy nothing, since the whole point is that
    every picture looks the same.
    """

    finished = Signal(object)  # NoiseEstimate
    failed = Signal(str)
    progress = Signal(str)
    image_done = Signal(int, int, int)  # index, total, corners

    def __init__(self, paths: list[str], spec, settings):
        super().__init__()
        self.paths = paths
        self.spec = spec
        self.settings = settings

    def run(self) -> None:
        try:
            from mlti_cal.detectors.noise import estimate_from_images

            self.progress.emit(f"measuring noise over {len(self.paths)} image(s)...")
            estimate = estimate_from_images(
                self.paths,
                self.spec,
                settings=self.settings,
                on_progress=self.image_done.emit,
            )
            self.finished.emit(estimate)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CrossValWorker(QObject):
    """
    Runs k-fold cross-validation off the GUI thread.

    The slowest thing this application does: every fold is a full re-solve, so
    k=4 costs roughly four solves. Each finished fold is emitted as it lands,
    because a progress bar that only moves at the end is not progress.

    The system is not modified -- `cross_validate` copies per fold -- so the
    parameter tree still shows the user's own solve when this finishes.
    """

    finished = Signal(object)  # CrossValResult
    failed = Signal(str)
    progress = Signal(str)
    fold_done = Signal(object)  # FoldResult

    def __init__(
        self,
        system: CalibrationSystem,
        backend: str = "scipy",
        folds: int = 4,
        max_iterations: int = 200,
    ):
        super().__init__()
        self.system = system
        self.backend = backend
        self.folds = folds
        self.max_iterations = max_iterations

    def run(self) -> None:
        try:
            from mlti_cal.report.crossval import cross_validate

            self.progress.emit(f"cross-validating over {self.folds} folds...")
            result = cross_validate(
                self.system,
                backend_name=self.backend,
                k=self.folds,
                max_iterations=self.max_iterations,
                on_fold=self.fold_done.emit,
            )
            self.finished.emit(result)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ReportWorker(QObject):
    """Builds the report off the GUI thread (cross-validation is slow)."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        session: Session,
        range_m: float,
        do_crossval: bool,
        settings: ReportSettings | None = None,
    ):
        super().__init__()
        self.session = session
        self.range_m = range_m
        self.do_crossval = do_crossval
        self.settings = settings

    def run(self) -> None:
        try:
            self.progress.emit(
                "building report" + (" with cross-validation..." if self.do_crossval else "...")
            )
            self.finished.emit(
                self.session.make_report(
                    self.range_m, do_crossval=self.do_crossval, settings=self.settings
                )
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def run_in_thread(parent, worker: QObject, on_done, on_fail, on_progress=None):
    """
    Wire a worker onto a QThread and start it.

    Two things here are load-bearing and both are easy to get wrong.

    1. NEVER connect the completion signals to a lambda. A lambda is not a
       QObject, so Qt has no receiver thread affinity to queue against and
       AutoConnection degrades to a DIRECT call -- meaning the handler would
       run on the WORKER thread. Since these handlers touch widgets (and pop
       QMessageBox on failure), that is undefined behaviour and a routine hard
       crash. `on_done`/`on_fail` are bound methods of GUI-thread QObjects, so
       connecting them directly gives a proper queued connection.

    2. NEVER call `thread.wait()` from a slot the thread itself invoked -- that
       is a thread waiting on itself. Use `thread.quit` as its own connection
       and let the event loop unwind.

    The thread and worker are stashed on `parent` because PySide6 garbage
    collects a QThread that nothing references, killing the job mid-flight.
    """
    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    worker.finished.connect(on_done)
    worker.finished.connect(thread.quit)
    worker.failed.connect(on_fail)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    if on_progress is not None:
        worker.progress.connect(on_progress)

    parent._active_thread = thread
    parent._active_worker = worker
    thread.start()
    return thread
