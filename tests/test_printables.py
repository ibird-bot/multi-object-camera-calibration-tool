"""The printed board must come out at its physical size."""

import pytest

pytest.importorskip("matplotlib")

from mlti_cal.detectors.charuco import CharucoBoardSpec  # noqa: E402
from mlti_cal.report.printables import (  # noqa: E402
    generate_charuco_printables,
    render_charuco_printable,
)


def _board_mm_on_page(fig) -> tuple[float, float]:
    page_w_mm, page_h_mm = (v * 25.4 for v in fig.get_size_inches())
    pos = fig.axes[0].get_position()
    return pos.width * page_w_mm, pos.height * page_h_mm


def test_a_board_that_fits_is_printed_at_true_size():
    spec = CharucoBoardSpec("A", 7, 5, 0.025, 0.018, "DICT_4X4_50")
    w, h = _board_mm_on_page(render_charuco_printable(spec))
    assert w == pytest.approx(7 * 25.0, abs=1e-6)
    assert h == pytest.approx(5 * 25.0, abs=1e-6)


def test_a_board_too_large_for_a4_is_scaled_down_and_labelled():
    spec = CharucoBoardSpec("big", 12, 9, 0.065, 0.048, "DICT_4X4_250")
    fig = render_charuco_printable(spec)
    w, h = _board_mm_on_page(fig)
    assert w < 12 * 65.0 and h < 9 * 65.0
    assert w / h == pytest.approx(12 / 9, rel=1e-6)
    assert any("SCALED TO" in t.get_text() for t in fig.texts)


def test_one_pdf_per_board(tmp_path):
    specs = [
        CharucoBoardSpec("A", 5, 4, 0.03, 0.02, "DICT_4X4_50"),
        CharucoBoardSpec("B", 5, 4, 0.03, 0.02, "DICT_5X5_50"),
    ]
    written = generate_charuco_printables(specs, tmp_path)
    assert [p.name for p in written] == ["printable_A.pdf", "printable_B.pdf"]
    assert all(p.stat().st_size > 0 for p in written)
