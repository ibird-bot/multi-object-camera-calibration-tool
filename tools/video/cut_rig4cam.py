"""
The cut for the rig_4cam demo, variation A of `scripts_rig4cam.md`.

Every shot of the take was STAGED by the director: the subject alone, in a
window shaped for the picture. So a shot here is just a stretch of time --
the click (normal speed), the wait (fast-forwarded) and the hold (normal speed)
-- and the editor shows the whole frame. Durations are output seconds; the
landscape cut holds everything longer.

Every number in a caption was measured on this example (Ceres solve, the
report's own figures), not written from memory.
"""

from __future__ import annotations

from edit import Segment, Take

BRAND = "mlti_object_cal  ·  open source, MIT"
REPO = "github.com/ibird-bot/multi-object-camera-calibration-tool"


def gif_window(edit: list[Segment]) -> tuple[float, float]:
    """Output seconds for the README GIF: the corner preview through the solve."""
    t, start, end = 0.0, None, None
    for seg in edit:
        if start is None and seg.caption.startswith("Every board"):
            start = t
        t += seg.duration
        if seg.caption.startswith("0.64"):
            end = t  # keeps moving to the last solve segment, the hold
    assert start is not None and end is not None, "the cut has no preview or solve shot"
    return start, end


def build_edit(take: Take, fmt: str) -> list[Segment]:
    slow = 1.0 if fmt == "vertical" else 1.5
    t = take.t

    def hold(shot: str, seconds: float, caption: str, sub: str = "", end: str = "end"):
        return Segment("take", seconds * slow, t(f"shot:{shot}"), t(f"{end}:{shot}"), caption, sub)

    def staged(shot: str, seconds: float, caption: str, sub: str = ""):
        """Click at normal speed, the wait fast-forwarded, then the result held."""
        start, wait, waited, end = (t(f"{k}:{shot}") for k in ("shot", "wait", "waited", "end"))
        return [
            Segment("take", min(wait - start, 1.2) * slow, start, wait, caption, sub),
            Segment("take", 1.2 * slow, wait, waited, caption, sub),
            Segment("take", seconds * slow, waited, end, caption, sub),
        ]

    edit = [
        Segment("hook", 1.4 * slow, caption="4 cameras. 3 boards. One solve."),
        Segment(
            "hook",
            1.8 * slow,
            t0=1.4 * slow,
            flag=True,
            caption="The outer two never see the same board.",
        ),  # fmt: skip
        hold("setup", 2.6, "Two Charuco boards + a dot grid", "one wide-angle lens in the mix"),
        Segment("take", 1.6 * slow, t("wait:detect"), t("waited:detect"), "Detect all 120 images"),
        Segment("take", 0.5 * slow, t("waited:detect"), t("end:detect"), "Detect all 120 images"),
        hold("preview", 2.4, "Every board, its own colour", "cam2: Double Sphere wide-angle lens"),
    ]
    # Measured: cam0-cam3 and cam1-cam3 share 0 views, cam2-cam3 share 26 --
    # and the summary on screen says "cam3: placed through cam2".
    edit += staged("estimate", 2.6, "Every camera gets a starting point",
                   "cam3 via cam2 — it shares no view with cam0")  # fmt: skip
    edit += staged("solve", 2.4, "0.64 → 0.13 px", "Ceres, under 3 seconds")
    edit += staged("crossval", 2.4, "Unseen frames fit just as well",
                   "held-out error 0.99× training")  # fmt: skip
    edit += staged("report", 1.4, "Then it tells you what to check")
    edit += [
        hold(
            "coverage", 2.6, "It shows where your data is thin", "cam2: data in 75% of the image"
        ),  # fmt: skip
        hold(
            "findings",
            3.2,
            "…and what it can't tell apart",
            "cam2: focal length vs ξ, correlation 0.999",
        ),  # fmt: skip
        Segment(
            "card",
            3.5 * slow,
            lines=[
                "Checked against ground truth",
                "every camera within 0.5 mm",
                "8 lens models · bring your own detector",
                REPO,
            ],
        ),
    ]
    return edit
