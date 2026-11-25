"""
Synthetic captures as IMAGES: printed boards seen through true lens models.

`io/synthetic.py` produces corner coordinates directly, which tests the solver
but skips detection. This module renders the pictures instead, so the whole
pipeline -- detection, bootstrap, solve, report -- runs on data whose ground
truth is known exactly.

Method
    For every output pixel, invert the camera model to a ray (the model's own
    Newton inverse, exact to ~1e-10 px -- OpenCV's undistortPoints is off by up
    to ~0.8 px near the corners, which would bias the ground truth), intersect
    the ray with each board plane, and sample that board's printed texture.
    Rendering at 2x and area-averaging down gives anti-aliased edges; a little
    blur, vignetting and sensor noise follow.

    Unlike a homography warp, this produces real lens distortion: straight
    board edges come out curved exactly as the camera model says they should.

Conventions are the project's own: OpenCV camera axes, `T_cam_rig` extrinsics,
`T_rig_board` board poses, poses stored as [t, q] with q scalar-last. A board's
printed face is its -z side, as with OpenCV's boards: a camera sees the print
when it sits at negative z in the board frame.

Known limit: the inverse is expressed on the z = 1 plane, so pixels whose rays
are more than 90 degrees off-axis render as background.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mlti_cal.io.config import board_kind, board_spec
from mlti_cal.models.camera import get_model
from mlti_cal.models.manifolds import (
    matrix_to_quat,
    pose_compose,
    pose_inverse,
    quat_to_matrix,
    so3_exp,
)

#: Texture resolution. 2000 px/m is 0.5 mm per texel: at 1 m with f = 900 a
#: texel is about one supersampled pixel, so the print is neither blocky nor
#: aliased at working distances.
TEXTURE_PX_PER_M = 2000.0
SUPERSAMPLE = 2
#: Printed black and white are not 0 and 255 in a photograph.
INK, PAPER = 30.0, 225.0


@dataclass
class SceneCamera:
    id: str
    model_name: str
    image_size: tuple[int, int]
    params: np.ndarray
    extrinsic: np.ndarray  # T_cam_rig


@dataclass
class Scene:
    cameras: list[SceneCamera]
    #: Board definitions exactly as they appear in a config file.
    boards: list[dict]
    #: (frame, board id) -> T_rig_board.
    poses: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    @property
    def frames(self) -> list[str]:
        return sorted({f for f, _ in self.poses})

    def truth_dict(self) -> dict:
        return {
            "cameras": {
                c.id: {
                    "model": c.model_name,
                    "image_size": list(c.image_size),
                    "parameters": c.params.tolist(),
                    "extrinsic_T_cam_rig": c.extrinsic.tolist(),
                }
                for c in self.cameras
            },
            "board_poses_T_rig_board": {f"{f}|{b}": p.tolist() for (f, b), p in self.poses.items()},
        }


# ---------------------------------------------------------------------------
# Board textures
# ---------------------------------------------------------------------------


@dataclass
class Texture:
    """A board's print as an image, and where its board frame sits on it."""

    image: np.ndarray  # float32, 0 = ink, 1 = paper
    origin_px: tuple[float, float]  # texel of board (0, 0)
    px_per_m: float


def _edge(margin: int) -> tuple[float, float]:
    """
    Texel coordinate of the board origin when squares are drawn from `margin`.

    Texel i is centred on coordinate i, so a square filling texels margin.. has
    its edge half a texel earlier. Measured: without this every Charuco corner
    came out 0.25 mm (half a texel) off, about 0.3 px in the images.
    """
    return (margin - 0.5, margin - 0.5)


def board_texture(board: dict, px_per_m: float = TEXTURE_PX_PER_M) -> Texture:
    """Render one config board dict to its printed appearance."""
    kind = board_kind(board)
    spec = board_spec(board)
    if kind == "charuco":
        from mlti_cal.detectors.charuco import CharucoDetector

        pps = int(round(spec.square_length * px_per_m))
        margin = pps // 2
        img = CharucoDetector(spec).render(pixels_per_square=pps, margin=margin)
        # `render` scales by whole pixels per square; use the scale it really drew at.
        return Texture(img.astype(np.float32) / 255.0, _edge(margin), pps / spec.square_length)
    if kind == "checkerboard":
        pps = int(round(spec.square_length * px_per_m))
        margin = pps // 2
        h, w = spec.squares_y * pps, spec.squares_x * pps
        img = np.ones((h + 2 * margin, w + 2 * margin), np.float32)
        for i in range(spec.squares_y):
            for j in range(spec.squares_x):
                if (i + j) % 2 == 0:
                    y0, x0 = margin + i * pps, margin + j * pps
                    img[y0 : y0 + pps, x0 : x0 + pps] = 0.0
        return Texture(img, _edge(margin), pps / spec.square_length)
    if kind == "circle_grid":
        xy = spec.object_points[:, :2] * px_per_m
        margin = spec.spacing * px_per_m
        radius = 0.25 * spec.spacing * px_per_m
        w = int(np.ceil(xy[:, 0].max() + 2 * margin))
        h = int(np.ceil(xy[:, 1].max() + 2 * margin))
        img = np.full((h, w), 255, np.uint8)
        shift = 4  # sub-pixel accurate circle centres
        for x, y in xy + margin:
            centre = (int(round(x * 2**shift)), int(round(y * 2**shift)))
            cv2.circle(img, centre, int(round(radius * 2**shift)), 0, -1, cv2.LINE_AA, shift)
        return Texture(img.astype(np.float32) / 255.0, (margin, margin), px_per_m)
    raise ValueError(f"no renderer for board kind {kind!r}")


def aruco_face_texture(
    first_id: int,
    grid: int = 3,
    side: float = 0.24,
    marker: float = 0.05,
    px_per_m: float = TEXTURE_PX_PER_M,
) -> tuple[Texture, dict[int, np.ndarray]]:
    """
    One square face covered by a `grid` x `grid` block of ArUco markers.

    Returns the face's texture and {marker id: centre (x, y) on the face, m}.
    The face frame has its origin at the texture's top-left EDGE, so a marker
    filling texels [m, m+n) is centred at (m + n/2) texels -- the same
    half-texel bookkeeping as `_edge`, done once here for the whole face.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    size = int(round(side * px_per_m))
    n = int(round(marker * px_per_m))
    img = np.full((size, size), 255, np.uint8)
    centres = {}
    pitch = size / grid
    for k in range(grid * grid):
        r, c = divmod(k, grid)
        x0 = int(round((c + 0.5) * pitch - n / 2))
        y0 = int(round((r + 0.5) * pitch - n / 2))
        img[y0 : y0 + n, x0 : x0 + n] = cv2.aruco.generateImageMarker(dictionary, first_id + k, n)
        centres[first_id + k] = np.array([x0 + n / 2, y0 + n / 2]) / px_per_m
    return Texture(img.astype(np.float32) / 255.0, (-0.5, -0.5), px_per_m), centres


#: Face frames of a cube's three faces meeting at the origin, as T_object_face.
#: Each face's printed (-z) side faces OUTWARD, so the three faces are seen
#: from the (-1, -1, -1) side of the corner, as on a real box.
CUBE_FACE_ROTATIONS = {
    "z0": np.eye(3),  # the x-y face, outward normal -z
    "y0": np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(float),  # outward -y
    "x0": np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(float),  # outward -x
}


def aruco_cube(side: float = 0.24, grid: int = 3, marker: float = 0.05):
    """
    A 3D object: three faces of a cube, each with its own block of markers.

    Returns (faces, points): faces is [(face id, Texture, T_object_face)] for
    `render_view`, points is {marker id: (x, y, z) in the object frame} --
    exactly what a points file for this object contains.
    """
    faces, points = [], {}
    for i, (fid, R) in enumerate(CUBE_FACE_ROTATIONS.items()):
        tex, centres = aruco_face_texture(i * grid * grid, grid, side, marker)
        faces.append((fid, tex, np.concatenate([np.zeros(3), matrix_to_quat(R)])))
        for mid, xy in centres.items():
            points[mid] = R @ np.array([xy[0], xy[1], 0.0])
    return faces, points


# ---------------------------------------------------------------------------
# Rays and rendering
# ---------------------------------------------------------------------------


def pixel_rays(cam: SceneCamera, supersample: int = SUPERSAMPLE, chunk: int = 200_000):
    """
    Camera-frame rays (x, y, 1) for every supersampled pixel, and a valid mask.

    Sub-sample k of pixel u sits at u + (k + 0.5) / s - 0.5, so area-averaging
    the s x s block lands back on the OpenCV pixel centre u.
    """
    model = get_model(cam.model_name)
    w, h = cam.image_size
    s = supersample
    us = (np.arange(w * s) + 0.5) / s - 0.5
    vs = (np.arange(h * s) + 0.5) / s - 0.5
    gu, gv = np.meshgrid(us, vs)
    pix = np.column_stack([gu.ravel(), gv.ravel()])
    xy = np.empty_like(pix)
    for start in range(0, len(pix), chunk):
        part = pix[start : start + chunk]
        xy[start : start + chunk] = model.undistort_to_normalised(cam.params, part)
    rays = np.column_stack([xy, np.ones(len(xy))])
    back = model.project(cam.params, rays)
    valid = np.isfinite(back).all(axis=1) & (np.linalg.norm(back - pix, axis=1) < 1e-3)
    return rays.reshape(h * s, w * s, 3), valid.reshape(h * s, w * s)


def _background(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A soft, uneven backdrop, so boards are not floating on flat grey."""
    w, h = size
    small = rng.normal(0.0, 1.0, (max(2, h // 64), max(2, w // 64))).astype(np.float32)
    blobs = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (xx / w - 0.5) * 18.0 + (yy / h - 0.5) * 10.0
    return (95.0 + 14.0 * blobs + ramp).astype(np.float32)


def render_view(
    cam: SceneCamera,
    rays: np.ndarray,
    valid: np.ndarray,
    boards: list[tuple[str, Texture, np.ndarray]],
    rng: np.random.Generator,
    background: np.ndarray,
    noise_std: float = 1.6,
    blur_sigma: float = 0.6,
) -> np.ndarray:
    """
    One 8-bit grey image of `boards` (id, texture, T_rig_board) seen by `cam`.

    Nearest board wins where two overlap, and a board seen from behind is not
    drawn -- the print is on one side only.
    """
    hs, ws = rays.shape[:2]
    R_E, t_E = quat_to_matrix(cam.extrinsic[3:7]), cam.extrinsic[0:3]
    centre_rig = -R_E.T @ t_E
    d_rig = rays.reshape(-1, 3) @ R_E  # (R_E^T d) for every ray

    out = cv2.resize(background, (ws, hs), interpolation=cv2.INTER_LINEAR).ravel()
    depth = np.full(hs * ws, np.inf)
    for _bid, tex, T_rig_board in boards:
        R_P, t_P = quat_to_matrix(T_rig_board[3:7]), T_rig_board[0:3]
        o_b = R_P.T @ (centre_rig - t_P)
        if o_b[2] >= 0:  # camera behind the print
            continue
        d_b = d_rig @ R_P
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = -o_b[2] / d_b[:, 2]
        hit = valid.ravel() & np.isfinite(lam) & (lam > 0) & (lam < depth)
        X = o_b[None, :2] + lam[:, None] * d_b[:, :2]
        mx = (tex.origin_px[0] + X[:, 0] * tex.px_per_m).astype(np.float32)
        my = (tex.origin_px[1] + X[:, 1] * tex.px_per_m).astype(np.float32)
        sample = cv2.remap(
            tex.image,
            mx.reshape(hs, ws),
            my.reshape(hs, ws),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1.0,
        ).ravel()
        on_board = hit & (sample >= 0)
        # Lambert-ish shading: a board turned away from the camera is darker.
        normal_rig = R_P[:, 2]
        cos = np.abs(d_rig[on_board] @ normal_rig) / np.linalg.norm(d_rig[on_board], axis=1)
        shade = 0.72 + 0.28 * cos
        out[on_board] = (INK + (PAPER - INK) * sample[on_board]) * shade
        depth[on_board] = lam[on_board]

    img = cv2.resize(out.reshape(hs, ws), cam.image_size, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    w, h = cam.image_size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    img *= 1.0 - 0.18 * r2 / 2.0
    img += np.asarray(rng.normal(0.0, noise_std, img.shape), dtype=np.float32)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# The example rig
# ---------------------------------------------------------------------------


def _yaw_extrinsic(x_m: float, yaw_deg: float) -> np.ndarray:
    """T_cam_rig for a camera at (x, 0, 0) in the rig, turned `yaw` about rig y."""
    R_rig_cam = so3_exp(np.array([0.0, np.radians(yaw_deg), 0.0]))
    T_rig_cam = np.concatenate([[x_m, 0.0, 0.0], matrix_to_quat(R_rig_cam)])
    return pose_inverse(T_rig_cam)


def example_rig() -> tuple[list[SceneCamera], list[dict]]:
    """
    Four cameras fanned out in a 100-degree arc, three boards of two kinds.

    cam0 and cam3 point 100 degrees apart and never see the same board, so
    cam3 is only reachable through a camera in between (with the default
    scene, cam2: cam1 and cam3 share no view either). cam2 is a wide-angle Double
    Sphere lens; the others are OpenCV pinhole with real barrel distortion.
    """
    size = (1280, 800)

    def pinhole(f, k1, k2, k3, p1=2e-4, p2=-3e-4):
        return np.array([f, f * 1.001, 640.0 + 6, 400.0 - 4, k1, k2, p1, p2, k3])

    cameras = [
        SceneCamera(
            "cam0",
            "pinhole_radtan",
            size,
            pinhole(880.0, -0.27, 0.095, -0.018),
            _yaw_extrinsic(-0.30, -50.0),
        ),  # fmt: skip
        SceneCamera(
            "cam1",
            "pinhole_radtan",
            size,
            pinhole(905.0, -0.24, 0.080, -0.012),
            _yaw_extrinsic(-0.10, -17.0),
        ),  # fmt: skip
        SceneCamera(
            "cam2",
            "double_sphere",
            size,
            np.array([520.0, 521.0, 636.0, 404.0, -0.15, 0.58]),
            _yaw_extrinsic(0.10, 17.0),
        ),  # fmt: skip
        SceneCamera(
            "cam3",
            "pinhole_radtan",
            size,
            pinhole(890.0, -0.29, 0.110, -0.025),
            _yaw_extrinsic(0.30, 50.0),
        ),  # fmt: skip
    ]
    # Reference first: cam0 defines the rig frame, as in a config file.
    rig_from_cam0 = pose_inverse(cameras[0].extrinsic)
    for c in cameras:
        c.extrinsic = pose_compose(c.extrinsic, rig_from_cam0)

    boards = [
        {
            "id": "charuco_A",
            "kind": "charuco",
            "squares_x": 10,
            "squares_y": 7,
            "square_length": 0.04,
            "marker_length": 0.03,
            "dictionary": "DICT_4X4_100",
            "marker_id_offset": 0,
        },  # fmt: skip
        {
            "id": "charuco_B",
            "kind": "charuco",
            "squares_x": 8,
            "squares_y": 6,
            "square_length": 0.045,
            "marker_length": 0.034,
            "dictionary": "DICT_4X4_100",
            "marker_id_offset": 40,
        },  # fmt: skip
        {
            "id": "dots_C",
            "kind": "circle_grid",
            "circles_x": 4,
            "circles_y": 11,
            "spacing": 0.05,
            "grid_type": "asymmetric",
        },  # fmt: skip
    ]
    return cameras, boards


def board_points(board: dict) -> np.ndarray:
    """(M,3) feature points of a config board dict, in its board frame."""
    if board_kind(board) == "charuco":
        from mlti_cal.detectors.charuco import CharucoDetector

        return CharucoDetector(board_spec(board)).object_points
    return board_spec(board).object_points


def _place(cam: SceneCamera, pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    A random board pose (T_rig_board) aimed somewhere in `cam`'s view.

    Targets are spread over the WHOLE sensor, corners included: a capture that
    only fills the middle of the image leaves the distortion unconstrained
    exactly where it is largest.
    """
    model = get_model(cam.model_name)
    w, h = cam.image_size
    target = np.array([[rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h]])
    xy = model.undistort_to_normalised(cam.params, target)[0]
    ray = np.array([xy[0], xy[1], 1.0])
    ray /= np.linalg.norm(ray)
    depth = rng.uniform(0.75, 1.7)

    # Board z along the viewing ray (print faces the camera), then tilted.
    z = ray
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_align = np.column_stack([x, y, z])
    tilt = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.35, 0.35)])
    R_cam_board = R_align @ so3_exp(tilt)
    centre_board = np.array([*(pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0)) / 2, 0.0])
    t_cam_board = ray * depth - R_cam_board @ centre_board
    T_cam_board = np.concatenate([t_cam_board, matrix_to_quat(R_cam_board)])
    return pose_compose(pose_inverse(cam.extrinsic), T_cam_board)


def _project(
    cam: SceneCamera, pts: np.ndarray, T_rig_board: np.ndarray, exact: bool = True
) -> np.ndarray:
    """
    Pixel positions of `pts`; NaN for points behind the camera or seen from the back.

    `exact=False` skips the fold-back check below, for rough uses (a bounding
    box) where its Newton inverse would dominate the cost of planning a scene.
    """
    T = pose_compose(cam.extrinsic, T_rig_board)
    R = quat_to_matrix(T[3:7])
    X = pts @ R.T + T[0:3]
    uv = np.full((len(pts), 2), np.nan)
    # The print is visible only from the board's -z side (see module docstring).
    facing = (R.T @ -T[0:3])[2] < 0
    front = X[:, 2] > 0.05
    if facing and front.any():
        model = get_model(cam.model_name)
        uv_front = model.project(cam.params, X[front])
        if not exact:
            uv[front] = uv_front
            return uv
        # A polynomial distortion folds back far outside the field of view, so
        # a point well outside it can "project" into the image. Such ghosts
        # never appear in a rendered image (rays are traced from the pixels),
        # so they are dropped here: the pixel must invert to the same ray.
        xy = model.undistort_to_normalised(cam.params, uv_front)
        real = np.linalg.norm(xy - X[front, :2] / X[front, 2:3], axis=1) < 1e-6
        uv[np.flatnonzero(front)[real]] = uv_front[real]
    return uv


def _visible_fraction(cam: SceneCamera, pts: np.ndarray, T_rig_board: np.ndarray) -> float:
    """Share of the board's features that land inside `cam`'s image."""
    uv = _project(cam, pts, T_rig_board)
    w, h = cam.image_size
    with np.errstate(invalid="ignore"):
        inside = (uv[:, 0] > 4) & (uv[:, 0] < w - 4) & (uv[:, 1] > 4) & (uv[:, 1] < h - 4)
    return float(inside.sum()) / len(pts)


def _box(uv: np.ndarray) -> np.ndarray | None:
    """(xmin, ymin, xmax, ymax) of the finite points, or None if there are none."""
    ok = np.isfinite(uv).all(axis=1)
    return np.concatenate([uv[ok].min(axis=0), uv[ok].max(axis=0)]) if ok.any() else None


def _overlaps(a, b, pad: float = 20.0) -> bool:
    return (
        a is not None
        and b is not None
        and a[0] < b[2] + pad
        and b[0] < a[2] + pad
        and a[1] < b[3] + pad
        and b[1] < a[3] + pad
    )


def plan_scene(num_frames: int = 30, seed: int = 3) -> Scene:
    """
    Board poses for every frame, ground truth included.

    Each board in each frame is aimed at one camera, cycling through them, and
    kept only if that camera sees all of it and it does not overlap a board
    already placed in that frame, in any camera. Neighbouring cameras pick it
    up wherever their views overlap -- which is what chains cam3 to the
    reference.
    """
    rng = np.random.default_rng(seed)
    cameras, boards = example_rig()
    points = {b["id"]: board_points(b) for b in boards}
    scene = Scene(cameras=cameras, boards=boards)
    for f in range(num_frames):
        frame = f"{f:04d}"
        placed: list[tuple[str, np.ndarray]] = []
        for bi, board in enumerate(boards):
            pts = points[board["id"]]
            cam = cameras[(f + bi) % len(cameras)]
            for _ in range(300):
                pose = _place(cam, pts, rng)
                if _visible_fraction(cam, pts, pose) < 1.0:
                    continue
                boxes = [_box(_project(c, pts, pose, exact=False)) for c in cameras]
                clash = any(
                    _overlaps(boxes[ci], _box(_project(c, points[bid], other, exact=False)))
                    for bid, other in placed
                    for ci, c in enumerate(cameras)
                )
                if not clash:
                    scene.poses[(frame, board["id"])] = pose
                    placed.append((board["id"], pose))
                    break
    return scene


def write_dataset(scene: Scene, out_dir: str | Path, seed: int = 11) -> dict:
    """
    Render every view to `out_dir/<camera>/<frame>.png`, plus `truth.json`.

    Returns, per camera, how many boards are fully in view across all frames
    according to the ground truth -- an upper bound on what detection can find.
    """
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    textures = {b["id"]: board_texture(b) for b in scene.boards}
    points = {b["id"]: board_points(b) for b in scene.boards}
    counts: dict[str, int] = {}
    for cam in scene.cameras:
        rays, valid = pixel_rays(cam)
        background = _background(cam.image_size, rng)
        (out / cam.id).mkdir(parents=True, exist_ok=True)
        counts[cam.id] = 0
        for frame in scene.frames:
            in_frame = [
                (b["id"], textures[b["id"]], scene.poses[(frame, b["id"])])
                for b in scene.boards
                if (frame, b["id"]) in scene.poses
            ]
            counts[cam.id] += sum(
                _visible_fraction(cam, points[bid], pose) == 1.0 for bid, _, pose in in_frame
            )
            img = render_view(cam, rays, valid, in_frame, rng, background)
            cv2.imwrite(str(out / cam.id / f"{frame}.png"), img)
    (out / "truth.json").write_text(json.dumps(scene.truth_dict(), indent=1), encoding="utf-8")
    return counts
