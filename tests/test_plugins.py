"""
The function contracts in docs/extending.md: what a plug-in may hand back,
and what happens when it hands back something else.

The rule under test throughout: a plug-in result that breaks the contract is
REFUSED (that image is skipped, that model is not registered) with a message
naming the plug-in and the rule -- never patched up and used.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from mlti_cal.detectors import registry
from mlti_cal.detectors.pipeline import BoardDetectors
from mlti_cal.detectors.registry import DETECTOR_KINDS, PLUGIN_ERRORS, get_kind, load_plugins
from mlti_cal.io.config import board_spec
from mlti_cal.models import camera as camera_models
from mlti_cal.plugins import loader
from mlti_cal.plugins.contract import ContractError, validate_detections
from mlti_cal.plugins.function_detector import FunctionDetectorGroup, settings_class
from mlti_cal.plugins.point_object import PointFileError, PointObjectSpec, parse_points

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "plugins"


@pytest.fixture
def clean_registries():
    """Plug-ins register process-wide; every test puts everything back."""
    kinds, errors = dict(DETECTOR_KINDS), dict(PLUGIN_ERRORS)
    models, labels = dict(camera_models._MODELS), dict(camera_models.MODEL_LABELS)
    loaded = list(loader.LOADED)
    yield
    DETECTOR_KINDS.clear()
    DETECTOR_KINDS.update(kinds)
    PLUGIN_ERRORS.clear()
    PLUGIN_ERRORS.update(errors)
    camera_models._MODELS.clear()
    camera_models._MODELS.update(models)
    camera_models.MODEL_LABELS.clear()
    camera_models.MODEL_LABELS.update(labels)
    loader.LOADED[:] = loaded


def load_from(folder: Path, monkeypatch) -> list[str]:
    monkeypatch.setenv("MLTI_CAL_PLUGIN_PATH", str(folder))
    return load_plugins()


# -- the result gate ------------------------------------------------------------
GOOD = np.array([[7.0, 10.5, 20.25], [103.0, 30.0, 40.0]])


def test_a_valid_result_passes_unchanged():
    assert np.array_equal(validate_detections(GOOD, (100, 200), "d"), GOOD)
    assert validate_detections(np.zeros((0, 3)), (100, 200), "d").shape == (0, 3)


def test_the_image_edge_follows_the_opencv_pixel_centre_convention():
    edge = np.array([[1.0, -0.5, -0.5], [2.0, 199.5, 99.5]])
    validate_detections(edge, (100, 200), "d")
    with pytest.raises(ContractError, match="outside"):
        validate_detections(np.array([[1.0, 199.6, 5.0]]), (100, 200), "d")


@pytest.mark.parametrize(
    ("raw", "rule"),
    [
        ([[1.0, 2.0, 3.0]], "numpy array"),
        (np.array([[1, 2, 3]]), "FLOATING"),
        (np.array([[1.0, 2.0]]), r"shape \(N, 3\)"),
        (np.array([[1.0, np.nan, 3.0]]), "NaN"),
        (np.array([[1.5, 2.0, 3.0]]), "whole numbers"),
        (np.array([[4.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), "more than once"),
        (np.array([[4.0, 20.0, 300.0]]), "outside"),
    ],
)
def test_each_contract_rule_is_enforced_by_name(raw, rule):
    with pytest.raises(ContractError, match=rule) as err:
        validate_detections(raw, (100, 200), "acme")
    assert "'acme'" in str(err.value), "the message must name the plug-in"


# -- the object -----------------------------------------------------------------
def test_points_files_accept_headers_comments_and_either_separator():
    ids, xyz, sigma = parse_points(
        "# measured 2026-11-11\nid,x,y,z\n5000, 0.1, 0.2, 0.0\n7 0.0 0.0 0.0  # origin\n"
    )
    assert ids.tolist() == [5000, 7]
    assert np.allclose(xyz, [[0.1, 0.2, 0.0], [0.0, 0.0, 0.0]])
    assert np.isnan(sigma).all()


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("1,0,0,0\n1,1,0,0\n", "more than once"),
        ("1,0,0\n", "expected 4 values"),
        ("1,0,0,0\n2,a,0,0\n", "line 2"),
        ("1.5,0,0,0\n", "whole numbers"),
    ],
)
def test_bad_points_files_are_refused_with_the_reason(text, rule):
    with pytest.raises(PointFileError, match=rule):
        parse_points(text)


def test_a_flat_object_off_the_z0_plane_is_refused_not_moved():
    tilted = [[i, x, y, 0.5 * x] for i, (x, y) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)])]
    with pytest.raises(PointFileError, match="z = 0"):
        PointObjectSpec(id="o", points=tilted)


def test_flat_and_3d_objects_are_told_apart():
    flat = PointObjectSpec(id="f", points=[[i, i % 2, i // 2, 0.0] for i in range(4)])
    cube = PointObjectSpec(
        id="c", points=[[i, i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)]
    )
    assert flat.planar and not cube.planar


# -- the group: ids, ownership, failures ---------------------------------------
def _objects():
    # Sparse ids, rows deliberately out of id order.
    a = PointObjectSpec(
        id="A", points=[[5000, 0.3, 0.0, 0], [7, 0.0, 0.0, 0], [103, 0.0, 0.3, 0], [9, 0.3, 0.3, 0]]
    )
    b = PointObjectSpec(
        id="B", points=[[40, 0.0, 0.0, 0], [41, 0.2, 0.0, 0], [42, 0.0, 0.2, 0], [43, 0.2, 0.2, 0]]
    )
    return [a, b]


def _group(func):
    cls = settings_class("t", {})
    return FunctionDetectorGroup(_objects(), name="t", func=func, cls=cls)


def test_target_ids_are_mapped_to_the_rows_of_the_object_that_owns_them():
    found = np.array([[103, 10.0, 11.0], [41, 20.0, 21.0], [5000, 30.0, 31.0], [999, 1.0, 1.0]])
    group = _group(lambda image, settings: found)
    dets = {d.board_id: d for d in group.detect_all(np.zeros((50, 50), np.uint8))}
    # 103 is row 2 of A, 5000 is row 0 of A, 41 is row 1 of B.
    assert dets["A"].point_ids.tolist() == [2, 0]
    assert np.allclose(dets["A"].image_points, [[10, 11], [30, 31]])
    assert dets["B"].point_ids.tolist() == [1]
    # Each pairing is the object point that id names -- the property that matters.
    obj = group.object_points("A")[dets["A"].point_ids]
    assert np.allclose(obj, [[0.0, 0.3, 0], [0.3, 0.0, 0]])
    assert group.unknown_ids == 1, "an id no object owns is dropped and counted"


def test_two_objects_claiming_one_id_are_refused_naming_both():
    a, b = _objects()
    b = PointObjectSpec(id="B", points=[[7, 0, 0, 0], [8, 1, 0, 0], [10, 0, 1, 0], [11, 1, 1, 0]])
    with pytest.raises(ValueError, match="'A' and 'B' both contain target id 7"):
        FunctionDetectorGroup([a, b], name="t", func=None, cls=settings_class("t", {}))


def test_a_raising_or_lying_detector_costs_one_image_not_the_run(caplog):
    def explodes(image, settings):
        raise RuntimeError("licence server unreachable")

    group = _group(explodes)
    with caplog.at_level(logging.WARNING, logger="mlti_cal.plugins"):
        assert group.detect_all(np.zeros((50, 50), np.uint8)) == []
    assert "licence server unreachable" in caplog.text and "'t'" in caplog.text

    group = _group(lambda image, settings: np.array([[1, 2, 3]]))  # integers
    assert group.detect_all(np.zeros((50, 50), np.uint8)) == []
    assert group.rejected_images == 1


def test_the_plugin_gets_a_copy_of_the_image():
    image = np.full((20, 20), 7, np.uint8)

    def scribbles(img, settings):
        img[:] = 0
        return np.zeros((0, 3))

    _group(scribbles).detect_all(image)
    assert (image == 7).all(), "the next detector must see the image unmodified"


# -- discovery and registration -------------------------------------------------
def test_the_example_plugins_load_and_register(tmp_path, monkeypatch, clean_registries):
    for name in ("aruco_centres.py", "my_lens.py", "my_detector.py"):
        shutil.copy(EXAMPLES / name, tmp_path / name)
    loaded = load_from(tmp_path, monkeypatch)
    assert {"aruco_centres", "my_detector"} <= set(loaded)
    assert "my_lens" in camera_models.available_models()
    assert camera_models.MODEL_LABELS["my_lens"].startswith("My lens")
    kind = get_kind("aruco_centres")
    assert kind.coded and kind.origin == "plug-in: aruco_centres.py"
    assert set(kind.default_settings().to_dict()) == {"min_perimeter_rate", "subpixel"}
    assert not PLUGIN_ERRORS


def test_a_broken_plugin_is_reported_and_the_rest_still_load(
    tmp_path, monkeypatch, clean_registries
):
    shutil.copy(EXAMPLES / "my_lens.py", tmp_path / "my_lens.py")
    (tmp_path / "bad_return.py").write_text("NAME = 'bad'\ndef detect(image):\n    pass\n")
    (tmp_path / "syntax.py").write_text("def detect(:\n")
    (tmp_path / "bad_jac.py").write_text(
        "import numpy as np\n"
        "NAME = 'bad_jac'\nPARAM_NAMES = ('fx', 'fy', 'cx', 'cy')\n"
        "def project(p, X):\n"
        "    return np.column_stack([p[0]*X[:,0]/X[:,2]+p[2], p[1]*X[:,1]/X[:,2]+p[3]])\n"
        "def jacobian(p, X):\n    n = len(X)\n    return np.zeros((n, 2, 4)), np.zeros((n, 2, 3))\n"
    )
    load_from(tmp_path, monkeypatch)
    assert "my_lens" in camera_models.available_models()
    errors = "\n".join(f"{k}\n{v}" for k, v in PLUGIN_ERRORS.items())
    assert "bad_return.py" in errors and "2 positional argument" in errors
    assert "syntax.py" in errors
    assert "bad_jac.py" in errors and "disagrees" in errors
    assert "bad_jac" not in camera_models.available_models()


def test_a_plugin_for_a_future_contract_version_is_refused(tmp_path, monkeypatch, clean_registries):
    (tmp_path / "future.py").write_text(
        "import numpy as np\nPLUGIN_API = 2\nNAME = 'future'\n"
        "def detect(image, settings):\n    return np.zeros((0, 3))\n"
    )
    load_from(tmp_path, monkeypatch)
    assert "future" not in DETECTOR_KINDS
    assert any("PLUGIN_API" in v for v in PLUGIN_ERRORS.values())


def test_a_plugin_model_projects_and_differentiates_like_a_builtin(
    tmp_path, monkeypatch, clean_registries
):
    shutil.copy(EXAMPLES / "my_lens.py", tmp_path / "my_lens.py")
    load_from(tmp_path, monkeypatch)
    mine = camera_models.get_model("my_lens")
    ref = camera_models.get_model("pinhole_radtan")
    p = np.array([900.0, 905.0, 640.0, 400.0, -0.2])
    X = np.array([[0.1, -0.2, 1.5], [-0.4, 0.3, 2.0]])
    p_ref = np.array([900.0, 905.0, 640.0, 400.0, -0.2, 0, 0, 0, 0])
    assert np.allclose(mine.project(p, X), ref.project(p_ref, X))
    _, J_p, J_X = mine.d_project(p, X)
    _, R_p, R_X = ref.d_project(p_ref, X)
    assert np.allclose(J_p, R_p[:, :, :5], atol=1e-5)
    assert np.allclose(J_X, R_X, atol=1e-5)


def test_a_config_object_of_a_plugin_kind_builds_and_detects(
    tmp_path, monkeypatch, clean_registries
):
    """End to end through the registry: real ArUco decoding, centres to 0.05 px."""
    shutil.copy(EXAMPLES / "aruco_centres.py", tmp_path / "aruco_centres.py")
    load_from(tmp_path, monkeypatch)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    image = np.full((400, 600), 255, np.uint8)
    truth = {}
    for marker_id, (x0, y0) in {3: (50, 60), 17: (300, 80), 42: (150, 250)}.items():
        image[y0 : y0 + 80, x0 : x0 + 80] = cv2.aruco.generateImageMarker(dictionary, marker_id, 80)
        truth[marker_id] = (x0 + 39.5, y0 + 39.5)  # centre of an 80 px block
    lines = "\n".join(f"{i},{x / 1000},{y / 1000},0" for i, (x, y) in truth.items())
    (tmp_path / "obj.csv").write_text("id,x,y,z\n" + lines + "\n99,0.9,0.9,0\n")

    board = {"id": "plate", "kind": "aruco_centres", "points_file": str(tmp_path / "obj.csv")}
    spec = board_spec(board)
    detectors = BoardDetectors.build({"aruco_centres": [spec]})
    (det,) = detectors.detect(image)
    found = dict(zip(spec.target_ids[det.point_ids].tolist(), det.image_points, strict=True))
    assert set(found) == set(truth)
    for marker_id, centre in truth.items():
        assert np.allclose(found[marker_id], centre, atol=0.05), marker_id


@pytest.mark.slow  # renders a 1280x800 view at 2x supersampling: ~13 s
def test_a_3d_object_is_found_by_the_example_detector(tmp_path, monkeypatch, clean_registries):
    """Rendered marker cube, 3 faces, through a distorted lens: centres to ~0.1 px."""
    from mlti_cal.io.render import SceneCamera, aruco_cube, pixel_rays, render_view
    from mlti_cal.models.manifolds import matrix_to_quat, pose_compose, quat_to_matrix, so3_exp
    from mlti_cal.plugins.point_object import format_points

    shutil.copy(EXAMPLES / "aruco_centres.py", tmp_path / "aruco_centres.py")
    load_from(tmp_path, monkeypatch)
    faces, points = aruco_cube()
    ids = sorted(points)
    (tmp_path / "cube.csv").write_text(format_points(ids, [points[i] for i in ids]))
    spec = board_spec(
        {"id": "cube", "kind": "aruco_centres", "points_file": str(tmp_path / "cube.csv")}
    )
    assert not spec.planar

    params = np.array([900.0, 902, 646, 396, -0.25, 0.08, 2e-4, -3e-4, -0.01])
    cam = SceneCamera("c", "pinhole_radtan", (1280, 800), params, np.array([0, 0, 0, 0, 0, 0, 1.0]))
    # The cube's marked corner (diagonal 1,1,1) turned to point away from the
    # camera, so its three outward faces are all seen; then a small tumble.
    d = np.ones(3) / np.sqrt(3)
    axis = np.cross(d, [0.0, 0.0, 1.0])
    towards = so3_exp(axis / np.linalg.norm(axis) * np.arccos(d[2]))
    R = so3_exp(np.array([0.1, -0.15, 0.05])) @ towards
    T = np.concatenate([np.array([0.05, 0.0, 1.0]) - R @ np.full(3, 0.12), matrix_to_quat(R)])
    rays, valid = pixel_rays(cam)
    image = render_view(
        cam, rays, valid, [(f, tex, pose_compose(T, Tf)) for f, tex, Tf in faces],
        np.random.default_rng(0), np.full((800, 1280), 100.0, np.float32),
    )  # fmt: skip
    (det,) = BoardDetectors.build({"aruco_centres": [spec]}).detect(image)
    assert det.num_points >= 15, "at least two faces' markers must be read"
    X = spec.object_points[det.point_ids] @ quat_to_matrix(T[3:7]).T + T[0:3]
    truth = camera_models.get_model("pinhole_radtan").project(params, X)
    err = np.linalg.norm(det.image_points - truth, axis=1)
    assert np.median(err) < 0.15, f"median {np.median(err):.3f} px"


def test_the_check_command_passes_the_templates_and_names_what_is_wrong(tmp_path, clean_registries):
    from mlti_cal.plugins.check import check_file

    for name in ("my_detector.py", "aruco_centres.py", "my_lens.py"):
        passed, lines = check_file(EXAMPLES / name)
        assert passed, "\n".join(lines)
    bad = tmp_path / "ints.py"
    bad.write_text(
        "import numpy as np\nNAME = 'ints'\n"
        "def detect(image, settings):\n    return np.array([[1, 10, 10]])\n"
    )
    passed, lines = check_file(bad)
    assert not passed
    assert any("FLOATING" in line for line in lines)
    assert "ints" not in DETECTOR_KINDS, "checking must not register anything"


def test_the_gui_carries_a_plugin_object_through_the_points_editor(
    tmp_path, monkeypatch, clean_registries
):
    """+ Object offers the plug-in kind; Points... fills its file; the config carries it."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    shutil.copy(EXAMPLES / "aruco_centres.py", tmp_path / "aruco_centres.py")
    load_from(tmp_path, monkeypatch)
    from mlti_cal.gui.detection_view import DetectionView
    from mlti_cal.gui.plugin_dialogs import PluginsDialog, PointsDialog
    from mlti_cal.gui.session import Session

    csv = tmp_path / "plate.csv"
    csv.write_text("id,x,y,z\n3,0,0,0\n17,0.1,0,0\n42,0,0.1,0\n50,0.1,0.1,0\n")
    dialog = PointsDialog("plate", str(csv))
    assert dialog.table.rowCount() == 4
    assert "4 points, flat object" in dialog.status.text()
    dialog.table.item(1, 0).setText("3")  # a duplicate id
    assert "more than once" in dialog.status.text()
    dialog.close()

    view = DetectionView(Session())
    view.board_table.setRowCount(0)
    view._add_board("aruco_centres")
    field = get_kind("aruco_centres").board_fields[0]
    view._write_cell(0, field, str(csv))
    board = view._collect_config().boards[0]
    assert board == {"id": "board_A", "kind": "aruco_centres", "points_file": str(csv)}
    assert board_spec(board).target_ids.tolist() == [3, 17, 42, 50]
    from PySide6.QtWidgets import QPlainTextEdit

    listing = PluginsDialog()
    text = listing.findChild(QPlainTextEdit, "plugins_text").toPlainText()
    assert "aruco_centres" in text and "aruco_centres.py" in text
    listing.close()
    view.close()
    app.processEvents()


def test_point_uncertainties_survive_a_round_trip():
    from mlti_cal.plugins.point_object import format_points

    text = "id,x,y,z,sx,sy,sz\n1,0,0,0,1e-5,1e-5,2e-5\n2,0.1,0,0\n"
    ids, xyz, sigma = parse_points(format_points(*parse_points(text)))
    assert ids.tolist() == [1, 2]
    assert np.allclose(sigma[0], [1e-5, 1e-5, 2e-5])
    assert np.isnan(sigma[1]).all()


def test_plugin_warnings_from_a_worker_thread_reach_the_gui_log():
    """Detection runs off the GUI thread; its contract violations must still be shown."""
    import threading

    from PySide6.QtWidgets import QApplication

    from mlti_cal.gui.log_console import LogConsole, attach_console

    app = QApplication.instance() or QApplication([])
    console = LogConsole()
    handler = attach_console(console)
    try:
        group = _group(lambda image, settings: np.array([[1, 2, 3]]))  # integers
        worker = threading.Thread(target=group.detect_all, args=(np.zeros((50, 50), np.uint8),))
        worker.start()
        worker.join()
        app.processEvents()  # the queued signal is delivered here, on the GUI thread
        assert "FLOATING" in console.toPlainText()
        assert "plugins" in console.toPlainText()
        assert handler.count == 1
    finally:
        logging.getLogger("mlti_cal").removeHandler(handler)
        console.close()


def test_skipped_images_and_foreign_ids_are_reported():
    from mlti_cal.io.config import plugin_stat_lines

    lines = plugin_stat_lines({"plugins": {"d": {"rejected_images": 3, "unknown_ids": 12}}})
    assert "3 image(s) skipped" in lines[0]
    assert "12 detected target(s)" in lines[1]
    assert plugin_stat_lines({"plugins": {"d": {"rejected_images": 0, "unknown_ids": 0}}}) == []


def test_plugin_folders_cannot_shadow_each_other(tmp_path, monkeypatch, clean_registries):
    """Two files with one name in two folders are two modules, not one."""
    one, two = tmp_path / "one", tmp_path / "two"
    for folder, name in ((one, "first"), (two, "second")):
        folder.mkdir()
        (folder / "det.py").write_text(
            f"import numpy as np\nNAME = '{name}'\n"
            "def detect(image, settings):\n    return np.zeros((0, 3))\n"
        )
    monkeypatch.setattr(registry, "PLUGIN_DIRS", [one, two])
    monkeypatch.delenv("MLTI_CAL_PLUGIN_PATH", raising=False)
    monkeypatch.delenv("MLTI_CAL_DETECTOR_PATH", raising=False)
    assert {"first", "second"} <= set(load_plugins())
