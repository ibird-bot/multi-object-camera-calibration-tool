"""
Template and working example: a camera-model plug-in.

A pinhole camera with one radial distortion term,

    x' = x (1 + k1 r^2),   r^2 = x^2 + y^2,   (x, y) = (X/Z, Y/Z)
    u  = fx x' + cx,       v = fy y' + cy

Copy to ~/.mlti_cal/plugins/ and replace `project` with your own lens. The
full contract is in docs/extending.md.
"""

import numpy as np

PLUGIN_API = 1
NAME = "my_lens"
LABEL = "My lens (pinhole + k1)"

#: The first four are always fx, fy, cx, cy. Add your own after them.
PARAM_NAMES = ("fx", "fy", "cx", "cy", "k1")


def project(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    params  1D array, one value per PARAM_NAMES entry.
    points  (N, 3) points in the CAMERA frame, metres, OpenCV axes:
            x right, y down, z forward (along the optical axis).
    return  (N, 2) float pixels, (0, 0) = centre of the top-left pixel.

    Must work on many points at once (use numpy, no per-point loop) and must
    not modify its inputs.
    """
    fx, fy, cx, cy, k1 = params
    z = np.maximum(points[:, 2], 1e-9)
    x, y = points[:, 0] / z, points[:, 1] / z
    d = 1.0 + k1 * (x * x + y * y)
    return np.column_stack([fx * x * d + cx, fy * y * d + cy])


# Optional. Without it, derivatives are computed numerically (slower, exact
# enough). If you supply one it is checked against the numerical one at load.
#
# def jacobian(params, points):
#     return d_pixels_d_params,  # (N, 2, len(PARAM_NAMES))
#            d_pixels_d_points   # (N, 2, 3)
#
# Optional. A sensible starting guess for your extra parameters; the default
# is fx = fy = image width, principal point at the centre, the rest zero.
#
# def default_params(image_size):
#     w, h = image_size
#     return np.array([w, w, w / 2, h / 2, 0.0])
