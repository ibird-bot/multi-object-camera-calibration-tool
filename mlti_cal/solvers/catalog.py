"""
Solver option catalog: what each knob does and WHEN to reach for it.

Single source of truth shared by the CLI and the GUI, so the guidance a user
reads in a tooltip is the same text the `--help` prints, and neither can drift
from what the adapter actually accepts.

The "when" notes are the point. Exposing `linear_solver_type` as a dropdown of
seven enum names helps nobody; saying "SPARSE_SCHUR -- the right default for
bundle adjustment: exploits the fact that pose blocks vastly outnumber camera
blocks" is the difference between a configurable tool and a usable one.

Guidance is condensed from each backend's own documentation, plus behaviour
measured on this project's own problems -- where that is the case the note says
so, because a measured claim and a quoted claim deserve different trust.
"""

from __future__ import annotations

from mlti_cal.options import Option, render_options

__all__ = [
    "CATALOG",
    "COMMON_OPTIONS",
    "LOSS_CATALOG",
    "Option",
    "describe",
    "options_for",
    "to_dict",
]

#: Options every backend honours, held on `SolveOptions` itself rather than in
#: `extra`. They were wired to both adapters from the start but appeared in no
#: catalog, so nothing rendered them and they could not be changed from the GUI.
COMMON_OPTIONS: list[Option] = [
    Option(
        name="function_tolerance",
        kind="float",
        default=1e-10,
        minimum=1e-16,
        maximum=1e-2,
        when="Stop when the relative cost decrease falls below this. The usual reason "
        "a solve stops 'early'. The default is deliberately tight -- calibration is "
        "cheap and a premature stop hides in the report as a plausible RMS.",
    ),
    Option(
        name="gradient_tolerance",
        kind="float",
        default=1e-12,
        minimum=1e-20,
        maximum=1e-2,
        when="Stop when the largest gradient component falls below this -- i.e. the "
        "point is stationary. Loosen it if the solver grinds through iterations "
        "that no longer change the RMS.",
    ),
    Option(
        name="parameter_tolerance",
        kind="float",
        default=1e-10,
        minimum=1e-16,
        maximum=1e-2,
        when="Stop when the step becomes this small relative to the parameters. "
        "Beware: focal lengths are O(1000) and distortion coefficients O(0.01), so "
        "one shared relative tolerance means very different absolute precision.",
    ),
    Option(
        name="num_threads",
        kind="int",
        default=1,
        minimum=1,
        maximum=64,
        when="Threads for Jacobian evaluation and the linear solve. Honoured by Ceres "
        "only; scipy ignores it. 1 by default so a run is bit-"
        "for-bit reproducible -- raise it for large rigs and accept that "
        "floating-point summation order, and so the last digits, may change.",
    ),
]


CATALOG: dict[str, dict] = {
    "scipy": {
        "summary": (
            "scipy.optimize.least_squares. Always available, pure Python/NumPy. "
            "No manifold support, so poses are handled by an increment "
            "parameterisation with an SO(3) right-Jacobian correction."
        ),
        "options": [
            Option(
                name="method",
                kind="choice",
                default="trf",
                choices=["trf", "lm", "dogbox"],
                when="Trust Region Reflective is the general-purpose choice.",
                per_choice={
                    "trf": "Default. Handles large sparse problems and bounds. "
                    "Robust on ill-conditioned calibration problems.",
                    "lm": "Classic Levenberg-Marquardt (MINPACK). Dense only, no "
                    "bounds, no sparse Jacobian. Fast on small problems; will "
                    "exhaust memory on a large rig.",
                    "dogbox": "Rectangular trust region. Occasionally better than trf "
                    "when many parameters sit at bounds; rarely relevant here.",
                },
            ),
            Option(
                name="tr_solver",
                kind="choice",
                default="exact (auto below 2000 free params, else lsmr)",
                choices=["exact", "lsmr"],
                when="How each trust-region subproblem is solved. This one matters.",
                per_choice={
                    "exact": "Direct factorisation, dense Jacobian. MEASURED on this "
                    "project: an 84-parameter problem converges in ~2.5 s with "
                    "'exact' versus not converging at all within 5000 function "
                    "evaluations (145 s) with 'lsmr'. Prefer it whenever the "
                    "problem fits in memory.",
                    "lsmr": "Iterative, sparse-friendly. Necessary once the free "
                    "parameter count makes a dense factorisation impractical, "
                    "but its approximate steps stall small problems.",
                },
            ),
            Option(
                name="x_scale",
                kind="choice",
                default="jac",
                choices=["jac", "1.0"],
                when="'jac' rescales variables by Jacobian magnitude. Keep it: focal "
                "lengths are O(1000) and distortion coefficients O(0.01), and "
                "without scaling the trust region is dominated by the focal terms.",
            ),
            Option(
                name="dense_column_limit",
                kind="int",
                default=2000,
                minimum=1,
                maximum=1000000,
                when="Free-parameter count below which 'exact' is chosen automatically "
                "when tr_solver is left at its default. Raise it to force a dense "
                "factorisation on a bigger rig -- worth trying if a large solve "
                "stalls, since 'lsmr' stalling is the failure measured above -- and "
                "lower it if a dense Jacobian exhausts memory.",
            ),
            Option(
                name="max_iterations",
                kind="int",
                default=100,
                minimum=1,
                when="Maps to scipy's max_nfev, which counts FUNCTION EVALUATIONS, "
                "not outer iterations -- trf uses several per step, so this needs "
                "to be considerably larger than an equivalent Ceres iteration cap.",
            ),
        ],
    },
    "ceres": {
        "summary": (
            "Google Ceres via pyceres. Sparse Schur, proper manifolds, and "
            "substantially faster than scipy on the same problem (measured: 14 "
            "iterations / 0.26 s versus scipy's 300+ evaluations / 6.9 s)."
        ),
        "options": [
            Option(
                name="linear_solver_type",
                kind="choice",
                default="SPARSE_SCHUR",
                choices=[
                    "SPARSE_SCHUR",
                    "DENSE_SCHUR",
                    "ITERATIVE_SCHUR",
                    "SPARSE_NORMAL_CHOLESKY",
                    "DENSE_NORMAL_CHOLESKY",
                    "DENSE_QR",
                    "CGNR",
                ],
                when="The single most consequential Ceres option for bundle "
                "adjustment. All of them find the same optimum -- they differ "
                "only in speed and memory.",
                per_choice={
                    "SPARSE_SCHUR": "Default and the right answer for calibration. "
                    "Exploits the structure where many board-pose blocks pair "
                    "with few camera blocks.",
                    "DENSE_SCHUR": "Same trick, dense factorisation. Faster for small "
                    "problems (a handful of frames), impractical beyond ~1000 poses.",
                    "ITERATIVE_SCHUR": "Preconditioned CG on the Schur complement. For "
                    "very large problems where even sparse factorisation is too "
                    "expensive. Pair with a preconditioner.",
                    "SPARSE_NORMAL_CHOLESKY": "Ignores Schur structure. Fine for small "
                    "problems and for structureless graphs.",
                    "DENSE_NORMAL_CHOLESKY": "Small dense problems only.",
                    "DENSE_QR": "Most numerically stable, slowest. Useful when a "
                    "problem is so ill-conditioned that Cholesky fails.",
                    "CGNR": "Conjugate gradients on the normal equations. Rarely the "
                    "best choice for this problem shape.",
                },
            ),
            Option(
                name="trust_region_strategy_type",
                kind="choice",
                default="LEVENBERG_MARQUARDT",
                choices=["LEVENBERG_MARQUARDT", "DOGLEG"],
                when="How a step is chosen once the linear system is solved.",
                per_choice={
                    "LEVENBERG_MARQUARDT": "Default. Solves a new linear system per "
                    "trust-region radius trial.",
                    "DOGLEG": "Reuses one factorisation across radius trials, so it can "
                    "be faster when factorisation dominates -- typically large "
                    "problems with a direct solver.",
                },
            ),
            Option(
                name="preconditioner_type",
                kind="choice",
                default="JACOBI",
                choices=[
                    "IDENTITY",
                    "JACOBI",
                    "SCHUR_JACOBI",
                    "CLUSTER_JACOBI",
                    "CLUSTER_TRIDIAGONAL",
                ],
                when="Only used by ITERATIVE_SCHUR/CGNR. SCHUR_JACOBI is the usual "
                "choice for bundle adjustment; the CLUSTER_* variants are stronger "
                "but cost more to build.",
            ),
            Option(
                name="use_nonmonotonic_steps",
                kind="bool",
                default=False,
                when="Allows the cost to rise temporarily to escape a narrow valley. "
                "Worth trying when a solve stalls short of convergence.",
            ),
            Option(
                name="use_inner_iterations",
                kind="bool",
                default=False,
                when="Optimises the pose blocks exactly at each step given the camera "
                "blocks. Can speed up badly initialised problems; adds per-iteration "
                "cost.",
            ),
        ],
    },
}


LOSS_CATALOG = [
    Option(
        name="trivial",
        kind="choice",
        default=True,
        when="Plain least squares. Correct when detections are clean -- a robust "
        "loss on clean data only discards good information.",
    ),
    Option(
        name="huber",
        kind="float",
        default=2.0,
        when="Quadratic inside the threshold, linear outside. The default choice when "
        "a few bad detections are suspected. Threshold is in PIXELS; set it to "
        "roughly 3x your expected corner noise.",
    ),
    Option(
        name="cauchy",
        kind="float",
        default=2.0,
        when="Downweights hard -- far outliers are effectively ignored. Use when "
        "gross mis-detections are present, but be aware it can also discard "
        "genuine wide-angle corners.",
    ),
    Option(
        name="soft_l1",
        kind="float",
        default=2.0,
        when="Smooth L1. Gentler than Cauchy, less abrupt than Huber.",
    ),
]


#: Which COMMON options each backend actually READS, verified against the
#: adapters rather than assumed. scipy passes ftol/gtol/xtol to
#: `least_squares` but has no thread count; Ceres reads all four.
#:
#: This is not tidiness. Rendering `num_threads` next to scipy shows a control
#: that silently does nothing, which is precisely the kind of quiet lie the
#: rest of this project exists to remove.
HONOURED_COMMON: dict[str, tuple[str, ...]] = {
    "scipy": ("function_tolerance", "gradient_tolerance", "parameter_tolerance"),
    "ceres": (
        "function_tolerance",
        "gradient_tolerance",
        "parameter_tolerance",
        "num_threads",
    ),
}


def options_for(backend: str) -> list[Option]:
    """Every option this backend reads: its own, then the common ones it honours."""
    if backend not in CATALOG:
        raise KeyError(f"unknown backend {backend!r}; known: {sorted(CATALOG)}")
    honoured = HONOURED_COMMON.get(backend, ())
    return list(CATALOG[backend]["options"]) + [o for o in COMMON_OPTIONS if o.name in honoured]


def describe(backend: str) -> str:
    """Human-readable rendering, used by `mlti-cal backends --describe`."""
    if backend not in CATALOG:
        raise KeyError(f"unknown backend {backend!r}; known: {sorted(CATALOG)}")
    return render_options(backend, CATALOG[backend]["summary"], options_for(backend))


def to_dict() -> dict:
    return {
        name: {
            "summary": entry["summary"],
            "options": [o.to_dict() for o in options_for(name)],
        }
        for name, entry in CATALOG.items()
    }
