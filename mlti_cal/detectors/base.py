"""
What every detector shares: the detection record, and how detections are drawn.

`Detection` and `draw_detections` used to live in `charuco.py`, which was fine
while Charuco was the only kind. It stopped being fine the moment a
checkerboard detector existed: a chessboard importing its own result type from
a module named after Aruco-coded boards states a dependency that is not real.
Nothing here knows what a marker is.

Colour is per BOARD, and stable
    A previous version picked `palette[i]` by the detection's position in the
    list. That is wrong in a way that is easy to miss: if board A is not found
    in a frame, board B moves to index 0 and silently takes A's colour, so the
    same board changes colour from image to image and two different boards look
    like one. Colour is now assigned from the SORTED set of board ids, and
    callers that know the full set (the config does) pass it in so a board
    missing from one frame does not renumber the others.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

#: BGR, ordered for distinguishability against grey board images and against
#: each other -- including for the common red/green deficiencies, which is why
#: green and orange are not adjacent and pure red is not first.
BOARD_PALETTE: list[tuple[int, int, int]] = [
    (0, 255, 0),  # green
    (0, 165, 255),  # orange
    (255, 128, 0),  # azure
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (255, 255, 0),  # cyan
    (0, 0, 255),  # red
    (128, 0, 255),  # pink
]

#: Corner glyph per detector kind. The kind is drawn as a SHAPE rather than
#: given its own colour, because colour is already spent on board identity --
#: with two Charuco boards and one checkerboard the user needs three colours
#: AND still needs to see which two came from the same detector.
KIND_MARKER = {
    "charuco": "circle",
    "checkerboard": "square",
    # NOT "circle" -- a dot grid drawn with the same glyph as a Charuco board is
    # the one pair a user cannot tell apart at a glance, and a dot overlaid on a
    # printed dot is exactly where you most want to see which is which.
    "circle_grid": "diamond",
}


@dataclass
class Detection:
    """One board found in one image."""

    board_id: str
    point_ids: np.ndarray  # (K,) corner ids, indexing the board's object points
    image_points: np.ndarray  # (K,2) subpixel corners
    marker_ids: np.ndarray | None = None
    num_markers: int = 0
    #: Which detector produced this. Drives the overlay glyph and lets a caller
    #: tell a full-board chessboard hit from a partial Charuco one.
    kind: str = "charuco"

    @property
    def num_points(self) -> int:
        return int(self.point_ids.size)


def board_colours(board_ids: Iterable[str]) -> dict[str, tuple[int, int, int]]:
    """
    A stable BGR colour per board id.

    Sorted, so the mapping depends only on WHICH boards exist and not on the
    order they were detected, declared or loaded in. Beyond `BOARD_PALETTE` the
    colours repeat; that is a legibility limit, not a correctness one.
    """
    return {
        bid: BOARD_PALETTE[i % len(BOARD_PALETTE)] for i, bid in enumerate(sorted(set(board_ids)))
    }


def _draw_glyph(canvas, x: int, y: int, radius: int, colour, marker: str) -> None:
    if marker == "square":
        cv2.rectangle(canvas, (x - radius, y - radius), (x + radius, y + radius), colour, -1)
    elif marker == "diamond":
        pts = np.array(
            [[x, y - radius], [x + radius, y], [x, y + radius], [x - radius, y]], dtype=np.int32
        )
        cv2.fillConvexPoly(canvas, pts, colour)
    else:
        cv2.circle(canvas, (x, y), radius, colour, -1)


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    radius: int = 4,
    labels: bool = True,
    colour: tuple[int, int, int] | None = None,
    colours: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """
    Overlay detected corners -- used by the GUI detection view.

    `colours` maps board id -> BGR. Pass the mapping for EVERY configured board,
    not just the ones in this image, so a board missing from one frame does not
    recolour the rest; when it is None the mapping is derived from the
    detections present, which is right for a one-off preview and not for a
    sequence.

    `labels` and `colour` exist for the thumbnail-sized overlay. At 128 px the
    per-corner id text is an unreadable smear that hides the corners it
    annotates, and a fixed colour is wanted there so "has corners" reads at a
    glance; the full-size preview keeps the per-board palette, which is what
    tells two boards in one image apart.
    """
    canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if colours is None:
        colours = board_colours(d.board_id for d in detections)
    for det in detections:
        c = colour if colour is not None else colours.get(det.board_id, BOARD_PALETTE[0])
        marker = KIND_MARKER.get(det.kind, "circle")
        for (x, y), pid in zip(det.image_points, det.point_ids, strict=True):
            _draw_glyph(canvas, int(round(x)), int(round(y)), radius, c, marker)
            if not labels:
                continue
            cv2.putText(
                canvas,
                str(int(pid)),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                c,
                1,
                cv2.LINE_AA,
            )
    return canvas
