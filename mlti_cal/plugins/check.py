"""
`mlti-cal plugins check FILE`: the contract checks, run on demand.

The same checks the application applies when it loads a plug-in, plus the one
it deliberately does not run at every start-up: actually CALLING `detect`, on
a blank image and optionally on a real one, through the same result gate.
Meant for the team writing a plug-in, so every line says what was checked and
what to change.
"""

from __future__ import annotations

import importlib.util
import time
import traceback
from pathlib import Path

import numpy as np


def _import(path: Path):
    spec = importlib.util.spec_from_file_location(f"mlti_cal_check_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    return image


def check_file(
    path: str | Path, image: str | Path | None = None, points: str | Path | None = None
) -> tuple[bool, list[str]]:
    """(passed, report lines) for one plug-in file."""
    from mlti_cal.plugins.contract import ContractError, validate_detections
    from mlti_cal.plugins.loader import build, contributions

    path = Path(path)
    lines: list[str] = []
    ok = True

    def fail(text: str) -> None:
        nonlocal ok
        ok = False
        lines.append(f"FAIL  {text}")

    try:
        module = _import(path)
    except Exception:
        fail(f"importing {path.name} raised:\n{traceback.format_exc()}")
        return ok, lines
    lines.append(f"ok    imported {path.name}")

    roles = contributions(module)
    if not roles:
        fail("defines neither detect(image, settings) nor project(params, points)")
        return ok, lines
    try:
        kind, model = build(module, path)
    except ContractError as exc:
        fail(str(exc))
        return ok, lines
    if model is not None:
        lines.append(
            f"ok    camera model {model.name!r}: {len(model.param_names)} parameters "
            f"{model.param_names}; projects its default parameters; "
            + (
                "jacobian() agrees with the numerical derivative"
                if getattr(module, "jacobian", None)
                else "no jacobian(): numerical derivatives will be used"
            )
        )
    if kind is None:
        return ok, lines

    defaults = kind.default_settings().to_dict()
    lines.append(f"ok    detector {kind.id!r}: settings {defaults or '(none)'}")
    func = module.detect

    def run(img: np.ndarray, label: str):
        start = time.perf_counter()
        try:
            raw = func(np.array(img, copy=True), dict(defaults))
        except Exception:
            fail(f"detect() on {label} raised:\n{traceback.format_exc()}")
            return None
        took = time.perf_counter() - start
        try:
            pts = validate_detections(raw, img.shape, kind.id)
        except ContractError as exc:
            fail(f"detect() on {label}: {exc}")
            return None
        lines.append(f"ok    detect() on {label}: {len(pts)} target(s) in {took * 1000:.0f} ms")
        return pts

    run(np.full((480, 640), 128, np.uint8), "a blank 640x480 image")
    if image is None:
        lines.append("note  give --image to run detect() on a real picture of your targets")
        return ok, lines
    try:
        picture = _read_image(Path(image))
    except ValueError as exc:
        fail(str(exc))
        return ok, lines
    pts = run(picture, Path(image).name)
    if pts is None or points is None:
        return ok, lines

    from mlti_cal.plugins.point_object import PointObjectSpec

    try:
        spec = PointObjectSpec(id="check", points_file=str(points))
    except ValueError as exc:
        fail(str(exc))
        return ok, lines
    known = set(spec.target_ids.tolist())
    found = {int(i) for i in pts[:, 0]}
    matched = found & known
    lines.append(
        f"ok    object {Path(points).name}: {len(known)} points "
        f"({'flat' if spec.planar else '3D'}); {len(matched)} of the {len(found)} "
        f"target(s) found belong to it"
    )
    if found and not matched:
        fail(
            "none of the ids detect() returned is in the points file -- check that both "
            "use the same numbering"
        )
    return ok, lines


def list_plugins() -> list[str]:
    """What loaded in this process and what did not, for `mlti-cal plugins list`."""
    from mlti_cal.detectors.registry import PLUGIN_ERRORS, plugin_dirs
    from mlti_cal.plugins.loader import LOADED

    lines = ["plug-in folders searched:"]
    lines += [f"  {d}{'' if d.is_dir() else '   (does not exist)'}" for d in plugin_dirs()]
    lines.append("loaded:" if LOADED else "loaded: none")
    lines += [f"  {p.role:<13} {p.name:<24} {p.path}" for p in LOADED]
    if PLUGIN_ERRORS:
        lines.append("failed to load:")
        for where, detail in PLUGIN_ERRORS.items():
            last = detail.strip().splitlines()[-1]
            lines.append(f"  {where}\n      {last}")
    return lines
