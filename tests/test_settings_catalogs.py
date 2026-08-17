"""
The catalogs must not drift from the code they describe.

These settings exist so a user can see what the pipeline is doing. A catalog
whose default differs from the dataclass, or whose tooltip describes a knob
nothing reads, is worse than no catalog at all -- it is a confident lie. Every
test here checks that correspondence rather than any numerical result.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mlti_cal.detectors.charuco import CharucoBoardSpec, CharucoDetector, MultiBoardDetector
from mlti_cal.detectors.registry import (
    DETECTOR_KINDS,
    default_detector_settings,
    detector_settings_from_dict,
    implemented_kinds,
)
from mlti_cal.detectors.settings import CHARUCO_CATALOG, CharucoSettings
from mlti_cal.io.config import CalibrationConfig
from mlti_cal.problem.settings import INIT_CATALOG, InitSettings
from mlti_cal.report.settings import REPORT_CATALOG, ReportSettings
from mlti_cal.solvers.base import SolveOptions
from mlti_cal.solvers.catalog import CATALOG, COMMON_OPTIONS, describe

CASES = [
    ("charuco", CHARUCO_CATALOG, CharucoSettings),
    ("initialization", INIT_CATALOG, InitSettings),
    ("report", REPORT_CATALOG, ReportSettings),
]


@pytest.mark.parametrize("name,catalog,cls", CASES)
def test_catalog_covers_every_field(name, catalog, cls):
    """No field may be settable in code but invisible in the catalog."""
    documented = {o.name for o in catalog}
    fields = set(cls().to_dict())
    assert fields == documented, (
        f"{name}: undocumented fields {sorted(fields - documented)}, "
        f"catalog entries with no field {sorted(documented - fields)}"
    )


@pytest.mark.parametrize("name,catalog,cls", CASES)
def test_catalog_defaults_match_the_dataclass(name, catalog, cls):
    """A tooltip promising default=X while the code uses Y is a lie."""
    actual = cls().to_dict()
    for opt in catalog:
        expected = actual[opt.name]
        if opt.kind == "float":
            assert float(opt.default) == pytest.approx(float(expected)), f"{name}.{opt.name}"
        else:
            assert opt.default == expected, f"{name}.{opt.name}"


@pytest.mark.parametrize("name,catalog,cls", CASES)
def test_catalog_bounds_admit_the_default(name, catalog, cls):
    """A spinbox whose range excludes its own default cannot round-trip."""
    for opt in catalog:
        if opt.kind in ("choice", "bool"):
            continue
        if opt.minimum is not None:
            assert float(opt.default) >= float(opt.minimum), f"{name}.{opt.name}"
        if opt.maximum is not None:
            assert float(opt.default) <= float(opt.maximum), f"{name}.{opt.name}"


@pytest.mark.parametrize("name,catalog,cls", CASES)
def test_every_option_carries_guidance(name, catalog, cls):
    """The `when` text is the product; an option without it is just a widget."""
    for opt in catalog:
        assert len(opt.when) > 40, f"{name}.{opt.name} has no real guidance"
        if opt.kind == "choice":
            assert set(opt.per_choice) <= set(opt.choices), f"{name}.{opt.name}"


def test_choice_options_accept_every_value_they_offer():
    """A dropdown must not offer a value the dataclass rejects."""
    for opt in CHARUCO_CATALOG:
        if opt.kind == "choice":
            for choice in opt.choices:
                CharucoSettings(**{opt.name: choice})
    for opt in INIT_CATALOG:
        if opt.kind == "choice":
            for choice in opt.choices:
                InitSettings(**{opt.name: choice})


def test_detection_settings_match_opencv_defaults():
    """
    Our written-out aruco defaults still equal the installed OpenCV's.

    If OpenCV changes one, this fails rather than the project silently
    documenting a default it no longer uses.
    """
    p = cv2.aruco.DetectorParameters()
    s = CharucoSettings()
    assert s.adaptive_thresh_win_size_min == p.adaptiveThreshWinSizeMin
    assert s.adaptive_thresh_win_size_max == p.adaptiveThreshWinSizeMax
    assert s.adaptive_thresh_win_size_step == p.adaptiveThreshWinSizeStep
    assert s.adaptive_thresh_constant == pytest.approx(p.adaptiveThreshConstant)
    assert s.min_marker_perimeter_rate == pytest.approx(p.minMarkerPerimeterRate)
    assert s.max_marker_perimeter_rate == pytest.approx(p.maxMarkerPerimeterRate)
    assert s.polygonal_approx_accuracy_rate == pytest.approx(p.polygonalApproxAccuracyRate)
    assert s.min_corner_distance_rate == pytest.approx(p.minCornerDistanceRate)
    assert s.min_distance_to_border == p.minDistanceToBorder
    assert s.error_correction_rate == pytest.approx(p.errorCorrectionRate)
    assert s.corner_refinement_win_size == p.cornerRefinementWinSize
    assert s.corner_refinement_max_iterations == p.cornerRefinementMaxIterations
    assert s.corner_refinement_min_accuracy == pytest.approx(p.cornerRefinementMinAccuracy)


def test_settings_actually_reach_opencv():
    """The knob is connected, not merely displayed."""
    spec = CharucoBoardSpec(
        id="b", squares_x=5, squares_y=4, square_length=0.03, marker_length=0.022
    )
    settings = CharucoSettings(corner_refinement="NONE", corner_refinement_win_size=11)
    det = CharucoDetector(spec, settings)
    params = det._detector.getDetectorParameters()
    assert params.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_NONE
    assert params.cornerRefinementWinSize == 11
    # ...and the default detector still asks for subpixel refinement, which is
    # what this project relied on before the setting existed.
    assert (
        CharucoDetector(spec)._detector.getDetectorParameters().cornerRefinementMethod
        == cv2.aruco.CORNER_REFINE_SUBPIX
    )


def test_multi_board_detector_passes_settings_to_every_board():
    specs = [
        CharucoBoardSpec("a", 5, 4, 0.03, 0.022, marker_id_offset=0),
        CharucoBoardSpec("b", 5, 4, 0.03, 0.022, marker_id_offset=20),
    ]
    settings = CharucoSettings(corner_refinement="CONTOUR")
    multi = MultiBoardDetector(specs, settings=settings)
    for det in multi.detectors.values():
        assert det.settings.corner_refinement == "CONTOUR"


def test_min_corners_default_comes_from_settings():
    spec = CharucoBoardSpec("a", 5, 4, 0.03, 0.022)
    det = CharucoDetector(spec, CharucoSettings(min_corners=999))
    blank = np.zeros((200, 200), dtype=np.uint8)
    assert det.detect(blank) is None


def test_config_round_trips_both_settings(tmp_path):
    cfg = CalibrationConfig(
        cameras=[],
        boards=[],
        detectors={
            "charuco": CharucoSettings(corner_refinement="APRILTAG", error_correction_rate=0.4)
        },
        initialization=InitSettings(pnp_ransac=True, ransac_reproj_threshold_px=1.25),
    )
    path = cfg.save(tmp_path / "c.json")
    back = CalibrationConfig.load(path)
    assert back.charuco.corner_refinement == "APRILTAG"
    assert back.charuco.error_correction_rate == pytest.approx(0.4)
    assert back.initialization.pnp_ransac is True
    assert back.initialization.ransac_reproj_threshold_px == pytest.approx(1.25)


def test_old_config_without_settings_still_loads(tmp_path):
    """Configs written before these settings existed must keep working."""
    path = tmp_path / "old.json"
    path.write_text('{"name": "x", "cameras": [], "boards": []}', encoding="utf-8")
    cfg = CalibrationConfig.load(path)
    assert cfg.charuco == CharucoSettings()
    assert cfg.initialization == InitSettings()


def test_floors_are_enforced_not_merely_documented():
    with pytest.raises(ValueError, match="4 points"):
        InitSettings(min_points_for_pnp=3)
    with pytest.raises(ValueError, match="absolute conic"):
        InitSettings(min_views_per_camera=2)
    with pytest.raises(ValueError, match="unknown pnp_method"):
        InitSettings(pnp_method="P3P")
    with pytest.raises(ValueError, match="held-out"):
        ReportSettings(crossval_folds=1)
    with pytest.raises(ValueError, match="no window size"):
        CharucoSettings(adaptive_thresh_win_size_min=30, adaptive_thresh_win_size_max=10)


def test_common_solver_options_exist_on_solve_options():
    """Rendering a knob that SolveOptions does not carry would do nothing."""
    fields = set(SolveOptions().__dict__)
    for opt in COMMON_OPTIONS:
        assert opt.name in fields, opt.name


def test_describe_lists_only_the_options_a_backend_reads():
    """
    `--describe` and the GUI panel come from the same `options_for`.

    A backend must not advertise a knob it ignores: scipy has no thread count,
    and the GTSAM adapter sets only a relative error tolerance.
    """
    from mlti_cal.solvers.catalog import HONOURED_COMMON, options_for

    assert set(CATALOG) == {"scipy", "ceres", "gtsam"}
    assert "dense_column_limit" in describe("scipy")

    for backend in CATALOG:
        names = {o.name for o in options_for(backend)}
        text = describe(backend)
        for opt in COMMON_OPTIONS:
            if opt.name in HONOURED_COMMON[backend]:
                assert opt.name in text, f"{backend} reads {opt.name} but does not list it"
            else:
                assert opt.name not in names, f"{backend} lists {opt.name} but ignores it"

    assert "num_threads" in describe("ceres")
    assert "num_threads" not in {o.name for o in options_for("scipy")}


def test_honoured_common_options_match_what_the_adapters_read():
    """
    The declaration is checked against the source, not trusted.

    If an adapter starts or stops reading one of these, this fails rather than
    letting the GUI quietly show a dead control or hide a live one.
    """
    import inspect

    from mlti_cal.solvers import ceres_backend, gtsam_backend, scipy_backend
    from mlti_cal.solvers.catalog import HONOURED_COMMON

    sources = {
        "scipy": inspect.getsource(scipy_backend),
        "ceres": inspect.getsource(ceres_backend),
        "gtsam": inspect.getsource(gtsam_backend),
    }
    for backend, source in sources.items():
        for opt in COMMON_OPTIONS:
            reads = f"opts.{opt.name}" in source
            declared = opt.name in HONOURED_COMMON[backend]
            assert reads == declared, (
                f"{backend}: HONOURED_COMMON says {declared} for {opt.name}, "
                f"but the adapter {'reads' if reads else 'does not read'} it"
            )


# ---------------------------------------------------------------------------
# Settings belong to a detector, not to "detection"
# ---------------------------------------------------------------------------


def test_every_implemented_kind_has_settings_and_a_catalog():
    for kind in implemented_kinds():
        assert kind.settings_cls is not None, kind.id
        assert kind.catalog, f"{kind.id} has no catalog"
        assert kind.summary, f"{kind.id} has no summary"
        kind.default_settings()


def test_unimplemented_kinds_say_why_rather_than_vanishing():
    """
    Listing them is the point: absence would read as "not supported, ever".
    """
    pending = [k for k in DETECTOR_KINDS.values() if not k.implemented]
    assert pending, "the registry should still name the kinds that are planned"
    for kind in pending:
        assert kind.unavailable_reason, f"{kind.id} is disabled with no reason given"
        assert kind.description, f"{kind.id} does not say what it is"
        assert kind.settings_cls is None
        with pytest.raises(ValueError, match="no settings"):
            kind.default_settings()


def test_default_settings_cover_exactly_the_implemented_kinds():
    assert set(default_detector_settings()) == {k.id for k in implemented_kinds()}


def test_settings_from_a_newer_config_do_not_break_this_build():
    """A config naming a detector this build lacks must still load."""
    loaded = detector_settings_from_dict(
        {"charuco": {"corner_refinement": "CONTOUR"}, "checkerboard": {"flags": 7}}
    )
    assert loaded["charuco"].corner_refinement == "CONTOUR"
    assert "checkerboard" not in loaded


def test_legacy_flat_detection_block_still_loads(tmp_path):
    """Configs written before detectors were per-kind kept a flat block."""
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"cameras": [], "boards": [], "detection": {"corner_refinement": "APRILTAG"}}',
        encoding="utf-8",
    )
    cfg = CalibrationConfig.load(path)
    assert cfg.charuco.corner_refinement == "APRILTAG"


def test_cli_topics_are_named_after_detectors():
    from mlti_cal.cli.main import SETTINGS_TOPICS

    assert "charuco" in SETTINGS_TOPICS
    assert "detection" not in SETTINGS_TOPICS, (
        "a topic called 'detection' claims to speak for every detector while "
        "listing only ArUco parameters"
    )
