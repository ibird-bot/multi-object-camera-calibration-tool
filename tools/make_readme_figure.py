"""
Render one report figure from a real calibration, for the README.

Runs the same pipeline as `mlti-cal calibrate` (detect, bootstrap, solve,
report) and draws the figure with the report's own exporter, so the PNG is
exactly what Export writes from the GUI.

    python tools/make_readme_figure.py examples/real_6cam/config.json \
        docs/images/uncertainty-map.png --figure uncertainty --camera cam2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mlti_cal.io.config import CalibrationConfig, build_system_from_config
from mlti_cal.problem.initialize import initialize_system
from mlti_cal.problem.reprojection import build_problem, write_back
from mlti_cal.report.figures import render, spec_for
from mlti_cal.report.report import build_report
from mlti_cal.solvers import SolveOptions, backend_status, get_backend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("config", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--figure", default="uncertainty", help="uncertainty, quiver, coverage, ...")
    ap.add_argument("--camera", default="cam0")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    config = CalibrationConfig.load(args.config)
    system, _ = build_system_from_config(config)
    initialize_system(system)
    problem = build_problem(system)
    backend = "ceres" if backend_status()["ceres"][0] else "scipy"
    result = get_backend(backend).solve(problem, SolveOptions(max_iterations=500))
    write_back(system, problem)
    report = build_report(
        problem, system, solve_result=result, pixel_noise_std=config.pixel_noise_std
    )
    fig = render(spec_for(args.figure), report, system, args.camera, dpi=args.dpi)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(
        f"{args.out}: {args.figure} of {args.camera}, RMS {result.final_rms_px:.4f} px ({backend})"
    )


if __name__ == "__main__":
    main()
