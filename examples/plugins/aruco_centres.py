"""
A working detector plug-in: the CENTRES of ArUco markers, decoded by OpenCV.

Each marker is one target: its id is the marker id, its position the marker's
centre. The centre is taken as the intersection of the two diagonals rather
than the mean of the four corners: a perspective projection maps the centre of
a square to the intersection of the imaged diagonals, while the corner mean
drifts as soon as the marker is tilted.

Pair it with an object file listing each marker centre:

    id, x, y, z          # metres, in the object's frame
    0,  0.030, 0.030, 0.0
    1,  0.110, 0.030, 0.0
    ...

Accuracy note: this example is about the contract, not a recommendation. One
point per marker carries less information than the 4 corners, and lens
distortion bends the diagonals slightly; real coded-target detectors fit the
target's own shape instead.
"""

import cv2
import numpy as np

PLUGIN_API = 1
NAME = "aruco_centres"
LABEL = "ArUco marker centres (example)"
DESCRIPTION = (
    "Example plug-in: one target per ArUco marker (DICT_4X4_250), located at the "
    "intersection of the marker's diagonals."
)
SETTINGS = {"min_perimeter_rate": 0.02, "subpixel": True}
SETTINGS_HELP = {
    "min_perimeter_rate": "Smallest marker kept, as a fraction of the image's larger side.",
    "subpixel": "Refine the marker corners to sub-pixel before taking the centre.",
}

_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)


def _diagonal_intersection(quad: np.ndarray) -> np.ndarray:
    """Intersection of the lines c0-c2 and c1-c3, in homogeneous coordinates."""
    h = np.column_stack([quad, np.ones(4)])
    p = np.cross(np.cross(h[0], h[2]), np.cross(h[1], h[3]))
    return p[:2] / p[2]


def detect(image: np.ndarray, settings: dict) -> np.ndarray:
    params = cv2.aruco.DetectorParameters()
    params.minMarkerPerimeterRate = float(settings["min_perimeter_rate"])
    if settings["subpixel"]:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(_DICTIONARY, params).detectMarkers(image)
    if ids is None:
        return np.zeros((0, 3))
    rows = [
        (int(i), *_diagonal_intersection(np.asarray(c, dtype=float).reshape(4, 2)))
        for c, i in zip(corners, np.asarray(ids).ravel(), strict=True)
    ]
    return np.asarray(rows, dtype=float).reshape(-1, 3)
