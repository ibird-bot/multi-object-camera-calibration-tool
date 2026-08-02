"""
Headless entry point.

Every number the GUI shows comes from this same code path, so a CLI run and a
GUI run cannot disagree. This is also what makes batch/CI calibration possible.

    mlti-cal backends                      what can actually run here
    mlti-cal backends --describe ceres     the option catalog with guidance
    mlti-cal demo                          synthetic end-to-end + honesty check
    mlti-cal calibrate config.json         real images
    mlti-cal compare                       every backend on one problem
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from mlti_cal.io.config import (
    CalibrationConfig,
    build_system_from_config,
    export_json,
    export_opencv_yaml,
)
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.loss import make_loss
from mlti_cal.problem.reprojection import build_problem, extr_key, intr_key, pose_key, write_back
from mlti_cal.report.report import build_report
from mlti_cal.solvers import SolveOptions, backend_status, get_backend
from mlti_cal.solvers.catalog import CATALOG, describe


def _truth_blocks(gt) -> dict:
    out = {}
    for cid, p in gt.camera_params.items():
        out[intr_key(cid)] = p
    for cid, e in gt.camera_extrinsics.items():
        out[extr_key(cid)] = e
    for (frame, board), pose in gt.board_poses.items():
        out[pose_key(frame, board)] = pose
    return out


def cmd_backends(args) -> int:
    if args.describe:
        print(describe(args.describe))
        return 0
    print(f"{'backend':<10} {'usable':<8} notes")
    print("-" * 78)
    for name, (ok, why) in backend_status().items():
        summary = CATALOG.get(name, {}).get("summary", "").split(".")[0]
        print(f"{name:<10} {'yes' if ok else 'NO':<8} {summary if ok else why[:120]}")
    return 0


def _solve_and_report(system, args, truth=None) -> int:
    loss = make_loss(args.loss, args.loss_scale)
    problem = build_problem(system, loss=loss)
    print(f"problem: {problem.num_residuals} residuals, {problem.num_free_params} free params")

    ok, why = backend_status().get(args.backend, (False, "unknown backend"))
    if not ok:
        print(f"ERROR: backend {args.backend!r} cannot run here:\n  {why}", file=sys.stderr)
        return 2

    result = get_backend(args.backend).solve(
        problem,
        SolveOptions(
            max_iterations=args.max_iterations,
            verbose=args.verbose,
            extra=dict(args.option or []),
        ),
    )
    write_back(system, problem)
    print(result.summary())

    report = build_report(
        problem,
        system,
        solve_result=result,
        pixel_noise_std=args.pixel_noise,
        default_range_m=args.range,
        do_crossval=args.crossval,
        truth_values=truth,
    )
    print()
    print(report.text_summary())

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report.save_json(out / "report.json")
    export_json(system, out / "calibration.json")
    export_opencv_yaml(system, out / "calibration.yaml")
    print(
        f"\nwritten: {out / 'report.json'}, {out / 'calibration.json'}, {out / 'calibration.yaml'}"
    )
    return 0


def cmd_demo(args) -> int:
    from mlti_cal.io.synthetic import generate_dataset

    print(
        f"generating synthetic data: {args.cameras} camera(s), {args.frames} frames, "
        f"noise {args.noise} px"
    )
    system, gt = generate_dataset(
        num_cameras=args.cameras,
        num_frames=args.frames,
        pixel_noise_std=args.noise,
        outlier_fraction=args.outliers,
        seed=args.seed,
    )
    initialize_system(system, verbose=args.verbose)
    args.pixel_noise = args.noise
    return _solve_and_report(system, args, truth=_truth_blocks(gt))


def cmd_calibrate(args) -> int:
    config = CalibrationConfig.load(args.config)
    print(
        f"config: {config.name} -- {len(config.cameras)} camera(s), {len(config.boards)} board(s)"
    )
    system, stats = build_system_from_config(config, verbose=args.verbose)
    for cid, s in stats["per_camera"].items():
        print(f"  {cid}: {s['detections']} detections over {s['images']} images")
    initialize_system(system, verbose=args.verbose)
    args.pixel_noise = config.pixel_noise_std
    return _solve_and_report(system, args)


def cmd_compare(args) -> int:
    """Run every usable backend on the identical problem and tabulate."""
    from mlti_cal.io.synthetic import generate_dataset

    system, _ = generate_dataset(
        num_cameras=args.cameras,
        num_frames=args.frames,
        pixel_noise_std=args.noise,
        seed=args.seed,
    )
    initialize_system(system)
    problem = build_problem(system)
    base = problem.get_state()

    print(f"{problem.num_residuals} residuals, {problem.num_free_params} free params\n")
    print(f"{'backend':<10} {'ok':<5} {'iters':>7} {'time_s':>9} {'final_cost':>14} {'rms_px':>9}")
    print("-" * 78)
    rows = []
    for name, (ok, why) in backend_status().items():
        if not ok:
            print(f"{name:<10} {'-':<5} {'-':>7} {'-':>9} {'unavailable':>14} {'-':>9}")
            continue
        problem.set_state(base)
        try:
            r = get_backend(name).solve(problem, SolveOptions(max_iterations=args.max_iterations))
        except Exception as exc:
            print(f"{name:<10} FAIL  {exc}")
            continue
        rows.append((name, r))
        print(
            f"{name:<10} {'yes' if r.success else 'NO':<5} {r.iterations:>7} "
            f"{r.time_seconds:>9.3f} {r.final_cost:>14.6f} {r.final_rms_px:>9.4f}"
        )
    if len(rows) > 1:
        costs = np.array([r.final_cost for _, r in rows])
        spread = float(costs.max() - costs.min()) / max(abs(costs.min()), 1e-12)
        print(f"\nrelative spread in final cost across backends: {spread:.3e}")
        print(
            "backends agree"
            if spread < 1e-6
            else "WARNING: backends disagree -- at least one has not converged"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlti-cal", description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--backend", default="scipy", help="scipy | ceres | gtsam")
        sp.add_argument("--max-iterations", type=int, default=300, dest="max_iterations")
        sp.add_argument("--loss", default=None, help="trivial | huber | cauchy | soft_l1")
        sp.add_argument("--loss-scale", type=float, default=2.0, dest="loss_scale")
        sp.add_argument(
            "--range",
            type=float,
            default=1.5,
            help="range in metres for the projection-uncertainty map",
        )
        sp.add_argument("--crossval", action="store_true", help="k-fold held-out error")
        sp.add_argument("-o", "--output", default="Output", help="output directory")
        sp.add_argument("-v", "--verbose", action="store_true")
        sp.add_argument(
            "--option",
            nargs=2,
            action="append",
            metavar=("KEY", "VALUE"),
            help="backend-specific option, repeatable",
        )

    b = sub.add_parser("backends", help="which solvers can run here")
    b.add_argument("--describe", metavar="BACKEND", help="print the option catalog")
    b.set_defaults(func=cmd_backends)

    d = sub.add_parser("demo", help="synthetic end-to-end run with honesty check")
    d.add_argument("--cameras", type=int, default=2)
    d.add_argument("--frames", type=int, default=20)
    d.add_argument("--noise", type=float, default=0.3)
    d.add_argument("--outliers", type=float, default=0.0)
    d.add_argument("--seed", type=int, default=7)
    common(d)
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser("calibrate", help="calibrate from a config + real images")
    c.add_argument("config")
    common(c)
    c.set_defaults(func=cmd_calibrate)

    m = sub.add_parser("compare", help="every backend on one problem")
    m.add_argument("--cameras", type=int, default=2)
    m.add_argument("--frames", type=int, default=12)
    m.add_argument("--noise", type=float, default=0.3)
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--max-iterations", type=int, default=500, dest="max_iterations")
    m.set_defaults(func=cmd_compare)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
