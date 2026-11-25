"""
The cut for the real six-camera demo (MC-Calib's public Seq02 sequence).

Record the take with

    python tools/video/shots_rig4cam.py Output/video/take_real_l --format landscape \
        --config examples/real_6cam/config.json --preview cam2/000050

then cut it from `tools/video/` with `--cut cut_real6cam`. Every number in a
caption was read off that take (solve card, report) or the matching CLI run,
not written from memory.
"""

from __future__ import annotations

from edit import Segment, Take

BRAND = "mlti_object_cal  ·  open source, MIT"
REPO = "github.com/ibird-bot/multi-object-camera-calibration-tool"


def gif_window(edit: list[Segment]) -> tuple[float, float]:
    """Output seconds for the README GIF: the corner preview through the solve."""
    t, start, end = 0.0, None, None
    for seg in edit:
        if start is None and seg.caption.startswith("Every corner"):
            start = t
        t += seg.duration
        if seg.caption.startswith("0.85"):
            end = t  # keeps moving to the last solve segment, the hold
    assert start is not None and end is not None, "the cut has no preview or solve shot"
    return start, end


def build_edit(take: Take, fmt: str) -> list[Segment]:
    slow = 1.0 if fmt == "vertical" else 1.5
    t = take.t

    def hold(shot: str, seconds: float, caption: str, sub: str = ""):
        return Segment("take", seconds * slow, t(f"shot:{shot}"), t(f"end:{shot}"), caption, sub)

    def staged(shot: str, seconds: float, caption: str, sub: str = ""):
        """Click at normal speed, the wait fast-forwarded, then the result held."""
        start, wait, waited, end = (t(f"{k}:{shot}") for k in ("shot", "wait", "waited", "end"))
        return [
            Segment("take", min(wait - start, 1.2) * slow, start, wait, caption, sub),
            Segment("take", 1.2 * slow, wait, waited, caption, sub),
            Segment("take", seconds * slow, waited, end, caption, sub),
        ]

    # No "hook" segment: edit.py draws it from the four synthetic rig_4cam cameras.
    edit = [
        hold(
            "setup",
            2.6,
            "6 real cameras, one Charuco board",
            "three RealSense stereo pairs, MC-Calib's Seq02",
        ),
        Segment("take", 1.6 * slow, t("wait:detect"), t("waited:detect"), "Detect all 552 images"),
        Segment("take", 0.5 * slow, t("waited:detect"), t("end:detect"), "Detect all 552 images"),
        hold("preview", 2.4, "Every corner found", "cam2: a real 1280 × 720 infrared frame"),
    ]
    edit += staged("estimate", 2.6, "Every camera gets a starting point",
                   "each from 79 to 86 shared views")  # fmt: skip
    edit += staged("solve", 2.4, "0.85 → 0.38 px", "Ceres, 7 iterations")
    edit += staged("crossval", 2.4, "Unseen frames fit almost as well",
                   "held-out error 1.06× training")  # fmt: skip
    edit += staged("report", 1.4, "Then it tells you what to check")
    edit += [
        hold(
            "coverage",
            2.6,
            "It shows where your data is thin",
            "cam2: board seen in 62% of the image",
        ),
        hold(
            "findings",
            3.2,
            "…and what it can't tell apart",
            "cam0's cx vs where the board is: correlation −0.99",
        ),  # fmt: skip
        Segment(
            "card",
            3.5 * slow,
            lines=[
                "Matches MC-Calib on this data",
                "3 stereo baselines: 49.6 to 49.7 mm",
                "8 lens models · bring your own detector",
                REPO,
            ],
        ),
    ]
    return edit
