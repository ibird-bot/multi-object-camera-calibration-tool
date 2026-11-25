"""
Template: a detector plug-in. Copy this file to ~/.mlti_cal/plugins/ and fill
in `detect`. The full contract is in docs/extending.md.

Your code receives one image and returns the targets it found and decoded.
Nothing else is required: no classes, no registration call.
"""

import numpy as np

PLUGIN_API = 1  # the contract version this file was written for

#: Unique name. It is also the `kind` of the objects this detector finds, in
#: the config and in the GUI's object table.
NAME = "my_detector"

#: Optional. Shown in the GUI.
LABEL = "My detector"
DESCRIPTION = "Finds my coded targets and returns their sub-pixel centres."

#: Optional. Every value here becomes an editable field in the Detector
#: settings window; your `detect` receives the current values as a dict.
#: Values must be bool, int or float.
SETTINGS = {
    "min_diameter_px": 6.0,
    "max_diameter_px": 200.0,
}
SETTINGS_HELP = {
    "min_diameter_px": "Smallest target to accept, in pixels.",
}


def detect(image: np.ndarray, settings: dict) -> np.ndarray:
    """
    image     2D numpy array, uint8, grey. A copy: you may modify it.
    settings  dict with the keys of SETTINGS and their current values.

    Return a FLOAT numpy array of shape (N, 3), one row per target found:

        [target_id, x, y]

    target_id  the identity your decoder read (a whole number). It must match
               an id in the object's points file; ids that match no object are
               ignored, so other targets in the scene do no harm.
    x, y       the target's centre in pixels, x to the right, y down. (0, 0) is
               the CENTRE of the top-left pixel (OpenCV convention).

    Nothing found: return np.zeros((0, 3)).
    Never return the same id twice in one image.
    """
    found = []  # e.g. [(17, 812.37, 455.02), (23, 901.11, 460.80)]
    # ... call your own detection and decoding code here ...
    return np.asarray(found, dtype=float).reshape(-1, 3)
