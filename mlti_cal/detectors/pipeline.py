"""
Running every registered detector over one image, in the order that keeps them
honest.

The order is not arbitrary and it is not a performance choice. Kinds whose
features carry their own identity (`DetectorKind.coded` -- Charuco's markers)
run FIRST: clutter cannot fool them, and once located they can be painted out.
Uncoded kinds run afterwards, on an image with everything already found removed.

Measured, on a synthetic scene holding a bare 8x6 Charuco board and no plain
checkerboard at all: `findChessboardCornersSB` for an 8x6 board reports a
confident full-grid detection on it. After masking it correctly reports nothing.
That is the whole reason this module exists, and it is also where "the coded
corners are taken, so what is left must be the chessboard" gets made real -- the
leftovers are not what OpenCV works from, so the located boards have to be
physically removed from the pixels first.

Nothing here names a detector. A kind contributed by a plugin takes part on
exactly the same terms as the built-ins: coded ones join the first pass, uncoded
ones join the fixed-point loop, and any kind exposing `mask_geometry` can be
painted out for the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mlti_cal.detectors.base import Detection
from mlti_cal.detectors.masking import MaskGeometry, mask_boards
from mlti_cal.detectors.registry import DetectorKind, get_kind


@dataclass
class BoardDetectors:
    """Every configured board's detector, and the one entry point that runs them."""

    #: kind id -> that kind's group detector.
    groups: dict[str, Any] = field(default_factory=dict)
    #: board id -> geometry for every board that can be painted out. EVERY
    #: locatable board belongs here, uncoded ones included: the uncoded
    #: detectors unblock each other by masking, so a board with no geometry is
    #: one that can never be removed, and it then blocks the others forever.
    mask_geometry: dict[str, MaskGeometry] = field(default_factory=dict)

    @staticmethod
    def build(specs_by_kind: dict[str, list], settings: dict[str, Any] | None = None):
        """
        One group detector per kind that actually has boards.

        Construction is where an ambiguous rig is refused -- Charuco marker-ID
        collisions, two identically sized checkerboards -- so a bad setup fails
        here rather than after every image has been searched.
        """
        settings = settings or {}
        groups: dict[str, Any] = {}
        geometry: dict[str, MaskGeometry] = {}
        for kind_id, specs in specs_by_kind.items():
            if not specs:
                continue
            group = get_kind(kind_id).build_group(specs, settings.get(kind_id))
            groups[kind_id] = group
            geometry.update(getattr(group, "mask_geometry", {}) or {})
        return BoardDetectors(groups=groups, mask_geometry=geometry)

    # -- lookups -----------------------------------------------------------
    def kinds(self) -> list[DetectorKind]:
        return [get_kind(k) for k in self.groups]

    @property
    def board_ids(self) -> list[str]:
        return [spec.id for group in self.groups.values() for spec in group.specs]

    def object_points(self, board_id: str) -> np.ndarray:
        for group in self.groups.values():
            if board_id in group.detectors:
                return group.object_points(board_id)
        raise KeyError(f"unknown board {board_id!r}; known: {self.board_ids}")

    # -- the run -----------------------------------------------------------
    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Every board found in one image: coded kinds first, then the uncoded ones
        repeatedly, until a whole pass turns up nothing new.

        Why a fixed point rather than a fixed order
            The uncoded detectors block each other and no single ordering
            resolves it. On a scene holding a Charuco board, a checkerboard and a
            4x11 dot grid, the grid's 44 dots are all found -- but the other two
            boards contribute some 60 further blobs, and `findCirclesGrid` then
            fails on the cluttered candidate set. Running the grid first
            therefore fails; running the checkerboard first succeeds, and masking
            it unblocks the grid on the next pass. Reverse the boards and the
            argument reverses with them.

            Order is not the answer -- iterating is. Each pass masks everything
            found so far and retries whatever is still missing, which converges
            regardless of which detector is blocked by which, and costs one
            wasted sweep in the common case where every board is found at once.
            It also means a plugin cannot be starved by where it happens to sit
            in the registry.
        """
        detections: list[Detection] = []
        for kind_id, group in self.groups.items():
            if get_kind(kind_id).coded:
                detections.extend(group.detect_all(image))

        uncoded = [g for k, g in self.groups.items() if not get_kind(k).coded]
        found: set[str] = {d.board_id for d in detections}
        # One pass per uncoded group is the most that can ever help: a pass that
        # changes anything unblocks at least one group, and there are only so
        # many groups to unblock.
        for _ in range(len(uncoded)):
            if not self._sweep(image, uncoded, detections, found):
                break
        return detections

    def _sweep(
        self,
        image: np.ndarray,
        uncoded: list,
        detections: list[Detection],
        found: set[str],
    ) -> bool:
        """
        One pass over the uncoded groups. True if anything new was located.

        Masking is governed by the settings of the detector ABOUT to search, not
        of the board being painted out: how much clutter has to be gone before a
        search is trustworthy is a property of the search.
        """
        progressed = False
        for group in uncoded:
            if all(spec.id in found for spec in group.specs):
                continue
            working = image
            if getattr(group.settings, "mask_other_boards", False) and detections:
                working = mask_boards(image, detections, self.mask_geometry, group.settings)
            for det in group.detect_all(working):
                if det.board_id in found:
                    continue
                detections.append(det)
                found.add(det.board_id)
                progressed = True
        return progressed
