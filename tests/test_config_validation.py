"""
Config files are validated when they are LOADED, not at first use.

The point of every test here is timing as much as correctness. Each of these
files was already doomed before this validation existed -- what changed is that
the complaint now arrives on `load()`, naming the key, instead of arriving after
detection has decoded every image of every camera (or, for a duplicate board id,
not arriving at all).
"""

from __future__ import annotations

import json

import pytest

from mlti_cal.io.config import CONFIG_VERSION, CalibrationConfig, ConfigError


def write(tmp_path, data: dict):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def good_board(board_id="board_A", **over):
    board = {
        "id": board_id,
        "kind": "charuco",
        "squares_x": 12,
        "squares_y": 9,
        "square_length": 0.03,
        "marker_length": 0.022,
        "dictionary": "DICT_4X4_50",
        "marker_id_offset": 0,
    }
    board.update(over)
    return board


def test_a_valid_config_still_loads(tmp_path):
    path = write(
        tmp_path,
        {
            "name": "rig",
            "cameras": [{"id": "cam0", "image_dir": "images"}],
            "boards": [good_board()],
        },
    )
    config = CalibrationConfig.load(path)
    assert config.name == "rig"
    assert [c.id for c in config.cameras] == ["cam0"]
    assert config.version == CONFIG_VERSION


# -- cameras ----------------------------------------------------------------
def test_a_mistyped_camera_field_names_the_camera_and_the_field(tmp_path):
    """
    The whole reason `_load_cameras` exists instead of `CameraConfig(**c)`.

    The old failure was `TypeError: __init__() got an unexpected keyword
    argument 'image_dirs'` -- which does not say which camera, which file, or
    what the valid fields are.
    """
    path = write(tmp_path, {"cameras": [{"id": "cam0", "image_dirs": "images"}]})
    with pytest.raises(ConfigError) as exc:
        CalibrationConfig.load(path)
    message = str(exc.value)
    assert "cam0" in message
    assert "image_dirs" in message
    assert "image_dir" in message  # tells you what it SHOULD have been


def test_a_camera_without_an_id_is_refused(tmp_path):
    path = write(tmp_path, {"cameras": [{"image_dir": "images"}]})
    with pytest.raises(ConfigError, match="missing 'id'"):
        CalibrationConfig.load(path)


def test_duplicate_camera_ids_are_refused(tmp_path):
    path = write(tmp_path, {"cameras": [{"id": "cam0"}, {"id": "cam0"}]})
    with pytest.raises(ConfigError, match="declared twice"):
        CalibrationConfig.load(path)


def test_two_reference_cameras_are_refused_at_load(tmp_path):
    """
    `CalibrationSystem.add_camera` already refuses this -- but only once the
    system is built, which is after detection has run over every image.
    """
    path = write(
        tmp_path,
        {
            "cameras": [
                {"id": "cam0", "is_reference": True},
                {"id": "cam1", "is_reference": True},
            ]
        },
    )
    with pytest.raises(ConfigError, match="reference"):
        CalibrationConfig.load(path)


# -- boards -----------------------------------------------------------------
def test_a_misspelled_board_field_is_caught_at_load(tmp_path):
    board = good_board()
    board["squares_z"] = board.pop("squares_x")
    path = write(tmp_path, {"boards": [board]})
    with pytest.raises(ConfigError, match="squares_z"):
        CalibrationConfig.load(path)


def test_an_unknown_board_kind_lists_what_is_available(tmp_path):
    path = write(tmp_path, {"boards": [{"id": "b", "kind": "not_a_detector"}]})
    with pytest.raises(ConfigError) as exc:
        CalibrationConfig.load(path)
    assert "charuco" in str(exc.value)


def test_duplicate_board_ids_are_refused(tmp_path):
    """
    This one used to fail SILENTLY. Boards are keyed by id downstream, so the
    second declaration replaced the first and the calibration ran against fewer
    targets than the file described -- with no warning and a plausible result.
    """
    path = write(tmp_path, {"boards": [good_board("dup"), good_board("dup", squares_x=7)]})
    with pytest.raises(ConfigError, match="declared twice"):
        CalibrationConfig.load(path)


def test_a_board_without_an_id_is_refused(tmp_path):
    board = good_board()
    del board["id"]
    path = write(tmp_path, {"boards": [board]})
    with pytest.raises(ConfigError, match="missing 'id'"):
        CalibrationConfig.load(path)


# -- file-level -------------------------------------------------------------
def test_malformed_json_names_the_file(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        CalibrationConfig.load(p)


def test_a_newer_format_version_is_refused_rather_than_half_read(tmp_path):
    """
    Refusing beats best-effort here: a future key this build does not know about
    could be the one that changes what the numbers mean, and a calibration run
    from a file that was only partly understood is exactly the unreproducible
    result the format version exists to prevent.
    """
    path = write(tmp_path, {"version": CONFIG_VERSION + 1, "cameras": [{"id": "cam0"}]})
    with pytest.raises(ConfigError, match="Upgrade"):
        CalibrationConfig.load(path)


def test_a_config_written_before_versioning_is_read_as_version_1(tmp_path):
    path = write(tmp_path, {"cameras": [{"id": "cam0"}], "boards": [good_board()]})
    assert CalibrationConfig.load(path).version == 1


def test_version_round_trips_through_save(tmp_path):
    config = CalibrationConfig(cameras=[], boards=[good_board()])
    saved = config.save(tmp_path / "out.json")
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert data["version"] == CONFIG_VERSION
    assert CalibrationConfig.load(saved).version == CONFIG_VERSION


def test_a_mistyped_top_level_key_is_refused(tmp_path):
    """
    The quietest failure of the lot, and the reason this check is not just on
    cameras and boards: `pixel_noise` instead of `pixel_noise_std` used to load
    without complaint, silently calibrate against the 0.3 default, and produce a
    result whose uncertainty was wrong in a way nothing reported.
    """
    path = write(tmp_path, {"cameras": [{"id": "cam0"}], "pixel_noise": 0.8})
    with pytest.raises(ConfigError) as exc:
        CalibrationConfig.load(path)
    assert "pixel_noise" in str(exc.value)
    assert "pixel_noise_std" in str(exc.value)


def test_the_pre_multi_detector_detection_key_still_loads(tmp_path):
    """`detection` predates `detectors`; old files must keep working."""
    path = write(
        tmp_path,
        {"cameras": [{"id": "cam0"}], "boards": [good_board()], "detection": {}},
    )
    assert CalibrationConfig.load(path).detectors["charuco"] is not None
