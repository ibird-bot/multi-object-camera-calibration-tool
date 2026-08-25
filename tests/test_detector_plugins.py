"""
The third-party detector contract.

These tests are the contract. A detector supplied by someone else has to reach
the config, the pipeline and the GUI through exactly the same path a built-in
does -- if any of these start needing a special case for a plugin, the
abstraction has failed and the next person has to patch the package again.

Every test that registers something restores the registry afterwards. The
registry is process-global by design (a plugin registers at import), so a test
that leaked a kind would silently change the GUI's column count for every test
that ran after it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlti_cal.detectors.checkerboard import CheckerboardSpec
from mlti_cal.detectors.registry import (
    DETECTOR_KINDS,
    PLUGIN_ERRORS,
    DetectorKind,
    coded_kinds,
    get_kind,
    implemented_kinds,
    load_plugins,
    register_detector,
    uncoded_kinds,
)
from mlti_cal.io.config import CalibrationConfig, board_kind
from mlti_cal.options import Option

EXAMPLE_DIR = "examples/detectors"


def render_aruco_grid(spec, px: float = 4000.0, pad: int = 80) -> np.ndarray:
    """The example plugin's board, rendered at its true aspect ratio."""
    xy = spec.object_points[:, :2]
    extent = xy.max(axis=0) - xy.min(axis=0)
    size = (int(round(extent[0] * px)), int(round(extent[1] * px)))
    img = spec.build_board().generateImage(size, marginSize=0)
    h, w = img.shape
    scene = np.full((h + 2 * pad, w + 2 * pad), 220, np.uint8)
    scene[pad : pad + h, pad : pad + w] = img
    return scene


@pytest.fixture
def clean_registry():
    """Snapshot the registry, restore it afterwards."""
    before = dict(DETECTOR_KINDS)
    errors_before = dict(PLUGIN_ERRORS)
    yield
    DETECTOR_KINDS.clear()
    DETECTOR_KINDS.update(before)
    PLUGIN_ERRORS.clear()
    PLUGIN_ERRORS.update(errors_before)


def minimal_settings_cls():
    from dataclasses import dataclass

    @dataclass
    class S:
        threshold: float = 0.5

        def to_dict(self):
            return {"threshold": self.threshold}

        @staticmethod
        def from_dict(d):
            return S(**(d or {}))

    return S


def minimal_kind(**over) -> DetectorKind:
    cls = minimal_settings_cls()
    base = dict(
        id="toy",
        label="Toy target",
        description="A target that exists only in this test.",
        spec_cls=CheckerboardSpec,
        group_cls=object,
        settings_cls=cls,
        catalog=[Option(name="threshold", kind="float", default=0.5, when="x" * 60)],
        board_fields=[
            Option(name="squares_x", kind="int", default=9, when="x" * 60, column="count_x"),
            Option(name="squares_y", kind="int", default=7, when="x" * 60, column="count_y"),
            Option(name="square_length", kind="float", default=0.03, when="x" * 60, column="pitch"),
        ],
    )
    base.update(over)
    return DetectorKind(**base)


# -- what registration refuses ---------------------------------------------
def test_a_kind_whose_catalog_and_settings_disagree_is_refused(clean_registry):
    """
    The drift rule applies to plugins too, at registration.

    A knob the user can see but nothing reads is worse than no knob -- the whole
    reason the built-in catalogs are tested against their dataclasses. A plugin
    gets the same check, and gets it where the traceback names the plugin.
    """
    with pytest.raises(ValueError, match="catalog and settings disagree"):
        register_detector(minimal_kind(catalog=[]))


def test_a_kind_with_no_board_fields_is_refused(clean_registry):
    with pytest.raises(ValueError, match="declares no board_fields"):
        register_detector(minimal_kind(board_fields=[]))


def test_declaring_id_as_a_board_field_is_refused(clean_registry):
    """Every board has an id; a second one would shadow it."""
    fields = [Option(name="id", kind="int", default=1, when="x" * 60)]
    with pytest.raises(ValueError, match="every board has an 'id'"):
        register_detector(minimal_kind(board_fields=fields))


def test_an_implemented_kind_missing_its_classes_is_refused(clean_registry):
    with pytest.raises(ValueError, match="has no group_cls"):
        register_detector(minimal_kind(group_cls=None))


def test_a_bad_id_is_refused(clean_registry):
    with pytest.raises(ValueError, match="valid Python identifier"):
        register_detector(minimal_kind(id="not an identifier"))


# -- what registration accepts ---------------------------------------------
def test_registering_makes_the_kind_visible_everywhere(clean_registry):
    register_detector(minimal_kind(coded=False))
    assert get_kind("toy").label == "Toy target"
    assert "toy" in {k.id for k in implemented_kinds()}
    assert "toy" in {k.id for k in uncoded_kinds()}
    assert "toy" not in {k.id for k in coded_kinds()}


def test_re_registering_an_id_replaces_it(clean_registry):
    """A plugin may deliberately override a built-in rather than sit beside it."""
    register_detector(minimal_kind())
    register_detector(minimal_kind(label="Toy target v2"))
    assert get_kind("toy").label == "Toy target v2"
    assert [k.id for k in implemented_kinds()].count("toy") == 1


def test_a_registered_kind_parses_and_rejects_boards_by_its_own_vocabulary(clean_registry):
    register_detector(minimal_kind())
    kind = get_kind("toy")
    board = kind.default_board("board_A")
    assert board == {
        "id": "board_A",
        "kind": "toy",
        "squares_x": 9,
        "squares_y": 7,
        "square_length": 0.03,
    }
    assert isinstance(kind.build_spec(board), CheckerboardSpec)

    with pytest.raises(TypeError, match="has no field"):
        kind.build_spec({"id": "b", "kind": "toy", "circles_x": 4})
    with pytest.raises(TypeError, match="needs"):
        kind.build_spec({"id": "b", "kind": "toy", "squares_x": 9})


def test_a_config_can_name_a_plugin_kind(clean_registry):
    """The config validates against the registry, not a table kept beside it."""
    register_detector(minimal_kind())
    board = get_kind("toy").default_board("board_A")
    assert board_kind(board) == "toy"
    assert isinstance(CalibrationConfig(boards=[board]).board_specs()[0], CheckerboardSpec)


def test_an_unknown_kind_is_reported_with_what_is_available():
    with pytest.raises(ValueError, match="available kinds are"):
        board_kind({"id": "b", "kind": "no_such_detector"})


# -- discovery --------------------------------------------------------------
def test_a_dropped_in_file_is_discovered(clean_registry, monkeypatch):
    monkeypatch.setenv("MLTI_CAL_DETECTOR_PATH", EXAMPLE_DIR)
    loaded = load_plugins()
    assert "aruco_grid" in loaded
    kind = get_kind("aruco_grid")
    assert kind.origin == "example plugin", "the GUI labels a plugin by its origin"
    assert kind.implemented
    # ArUco markers carry their own identity, so this plugin declares itself
    # coded -- and that is enough to put it in the pipeline's FIRST pass,
    # alongside Charuco, with nothing in the package naming it.
    assert kind.coded
    assert "aruco_grid" in {k.id for k in coded_kinds()}


def test_a_broken_plugin_is_recorded_not_raised(clean_registry, monkeypatch, tmp_path):
    """
    One bad file must not stop the app from starting -- and must not be silent.

    A plugin that fails to load invisibly is indistinguishable from one that was
    never installed, which is the worst of both.
    """
    (tmp_path / "broken.py").write_text("raise RuntimeError('boom')", encoding="utf-8")
    monkeypatch.setenv("MLTI_CAL_DETECTOR_PATH", str(tmp_path))
    before = set(DETECTOR_KINDS)
    load_plugins()  # must not raise
    assert set(DETECTOR_KINDS) == before
    assert any("broken.py" in where for where in PLUGIN_ERRORS)


def test_a_plugin_error_reaches_the_callback(clean_registry, monkeypatch, tmp_path):
    (tmp_path / "alsobroken.py").write_text("import nonexistent_module_xyz", encoding="utf-8")
    monkeypatch.setenv("MLTI_CAL_DETECTOR_PATH", str(tmp_path))
    seen: list[tuple[str, str]] = []
    load_plugins(on_error=lambda where, detail: seen.append((where, detail)))
    assert seen and "alsobroken.py" in seen[0][0]
    assert "nonexistent_module_xyz" in seen[0][1]


def test_underscore_files_are_skipped(clean_registry, monkeypatch, tmp_path):
    """`__init__.py` and `_helpers.py` are support files, not detectors."""
    (tmp_path / "_helper.py").write_text("raise RuntimeError('should not run')", encoding="utf-8")
    monkeypatch.setenv("MLTI_CAL_DETECTOR_PATH", str(tmp_path))
    load_plugins()
    assert not any("_helper.py" in where for where in PLUGIN_ERRORS)


# -- the example plugin, end to end ----------------------------------------
def test_the_example_plugin_runs_through_the_real_pipeline(clean_registry, monkeypatch):
    """
    A plugin reaches detection through exactly the path a built-in uses.

    This is the test that fails if any layer grows a special case: the config
    parses its board, the pipeline builds its group, and it finds a real target
    in a real image, without anything in `mlti_cal` naming it.
    """
    monkeypatch.setenv("MLTI_CAL_DETECTOR_PATH", EXAMPLE_DIR)
    load_plugins()

    config = CalibrationConfig(
        boards=[
            {
                "id": "grid",
                "kind": "aruco_grid",
                "markers_x": 5,
                "markers_y": 7,
                "marker_length": 0.030,
                "marker_separation": 0.006,
                "dictionary": "DICT_4X4_250",
                "marker_id_offset": 0,
            }
        ]
    )
    detectors = config.detectors_for_run()
    assert list(detectors.groups) == ["aruco_grid"]
    assert detectors.board_ids == ["grid"]
    assert "grid" in detectors.mask_geometry, "a plugin board must be maskable too"
    assert detectors.object_points("grid").shape == (140, 3), "4 corners x 35 markers"

    spec = config.board_specs()[0]
    scene = render_aruco_grid(spec)
    found = detectors.detect(scene)
    assert [d.board_id for d in found] == ["grid"]
    assert found[0].kind == "aruco_grid"
    assert found[0].num_markers == 35
    # Every point id must index the object points, or corners are paired with
    # the wrong 3D point and nothing downstream can tell.
    ids = np.sort(found[0].point_ids)
    assert ids.min() == 0 and ids.max() == 139 and len(ids) == 140


def test_a_plugin_mixes_with_the_built_ins(clean_registry, monkeypatch):
    monkeypatch.setenv("MLTI_CAL_DETECTOR_PATH", EXAMPLE_DIR)
    load_plugins()
    config = CalibrationConfig(
        boards=[
            {
                "id": "A",
                "squares_x": 8,
                "squares_y": 6,
                "square_length": 0.03,
                "marker_length": 0.022,
                "dictionary": "DICT_4X4_250",
                "marker_id_offset": 0,
            },
            {
                "id": "M",
                "kind": "aruco_grid",
                "markers_x": 3,
                "markers_y": 3,
                "marker_length": 0.02,
                "marker_separation": 0.005,
                "dictionary": "DICT_5X5_250",
                "marker_id_offset": 0,
            },
        ]
    )
    detectors = config.detectors_for_run()
    assert set(detectors.groups) == {"charuco", "aruco_grid"}
    assert set(detectors.mask_geometry) == {"A", "M"}


def test_a_plugin_board_round_trips_through_a_saved_config(clean_registry, monkeypatch, tmp_path):
    monkeypatch.setenv("MLTI_CAL_DETECTOR_PATH", EXAMPLE_DIR)
    load_plugins()
    board = get_kind("aruco_grid").default_board("board_A")
    config = CalibrationConfig(boards=[board])
    path = config.save(tmp_path / "plugin.json")
    assert CalibrationConfig.load(path).boards == [board]
