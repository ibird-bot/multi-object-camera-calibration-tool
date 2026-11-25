"""
Kalibr AprilGrid detection with OpenCV, for the Kalibr comparison only.

Kalibr's AprilGrid is tag36h11 with a 2-bit black border (`blackTagBorder: 2`),
which OpenCV's ArUco module reads with `markerBorderBits = 2`.

Board frame used here (stated because a mirrored layout is silently wrong):
+x along a tag row, +y down the sheet, z into the board, origin at the corner
of the tag in column 0 / top row. Kalibr numbers tags from the BOTTOM-left
upward, so tag `i` sits in column `i % cols` and row `rows - 1 - i // cols`
counted from the top. Point id `4 * tag + k`, k = OpenCV's corner index. The
layout was not assumed: all 8 in-tag corner orders x 2 row orders were fitted
against TUM VI images undistorted with Kalibr's intrinsics, and only this one
fits (see `object_points`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AprilGrid:
    cols: int = 6
    rows: int = 6
    tag_size: float = 0.088
    spacing_ratio: float = 0.3
    ids_from_bottom: bool = True
    # Detected tag edge / printed tag edge. OpenCV's corners sit slightly inside
    # the printed black square (threshold + blur bias); 1.0 = nominal geometry.
    # Shrinks each tag about its centre and leaves the pitch -- the metric
    # scale -- unchanged.
    edge_scale: float = 1.0

    def object_points(self) -> np.ndarray:
        """(4 * cols * rows, 3), indexed by point id."""
        pitch = self.tag_size * (1.0 + self.spacing_ratio)
        s = self.tag_size * self.edge_scale
        off = (self.tag_size - s) / 2
        # In OpenCV order for a tag as it appears on this board: measured on
        # TUM VI (median 0.46 px misfit vs 27+ px for the other 7 orders), the
        # first returned corner is the bottom-right of the square in this
        # y-down frame -- Kalibr's tags are printed upside down relative to it.
        corner = np.array([[s, s], [0, s], [0, 0], [s, 0]], dtype=float) + off
        pts = []
        for tag in range(self.cols * self.rows):
            col = tag % self.cols
            row = tag // self.cols
            if self.ids_from_bottom:
                row = self.rows - 1 - row
            pts.append(corner + [col * pitch, row * pitch])
        xy = np.concatenate(pts)
        return np.column_stack([xy, np.zeros(len(xy))])


def make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = 2
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 3
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.001
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect(detector: cv2.aruco.ArucoDetector, gray: np.ndarray, grid: AprilGrid):
    """(point_ids, pixels) or None; only tags with ids on the grid are kept."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    n_tags = grid.cols * grid.rows
    pid, uv = [], []
    for c, i in zip(corners, ids.ravel(), strict=True):
        if 0 <= i < n_tags:
            pid.extend(4 * int(i) + np.arange(4))
            uv.append(c.reshape(4, 2))
    if not uv:
        return None
    return np.array(pid), np.concatenate(uv).astype(float)
