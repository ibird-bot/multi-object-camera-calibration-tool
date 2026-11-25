"""
Cut a recorded take into finished videos.

    cd tools/video
    python edit.py ../../Output/video/take --format vertical -o out_vertical.mp4
    python edit.py ../../Output/video/take --format landscape -o out.mp4 --gif out.gif
    python edit.py ... --music track.mp3      # optional audio bed

The cut itself -- which stretch of the take, how fast, framed on what, with
which caption -- is built by `cut_rig4cam.py` from the marks `director.py`
logged. This file only renders it: each staged frame whole, the cursor and its
click ripples (drawn here, not recorded), captions hugging the picture, a speed
badge on fast-forwarded waits, and the encode through ffmpeg.

Before a single frame is encoded, `Renderer.check` applies the framing rules
(picture height, enlargement, smallest text size) to every shot and refuses
the cut, naming each shot that breaks one.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 30
FONT_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_SEMI = r"C:\Windows\Fonts\seguisb.ttf"
BG = (24, 20, 16)  # BGR, near-black
ACCENT = (79, 195, 247)  # RGB, for PIL
FORMATS = {"vertical": (1080, 1920), "landscape": (1920, 1080)}


# ---------------------------------------------------------------------------
# The take
# ---------------------------------------------------------------------------


class Take:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        ev = json.loads((self.path / "events.json").read_text(encoding="utf-8"))
        self.scale = float(ev.get("frame_scale", 1.0))
        #: Smallest UI text in window pixels, recorded by the director.
        self.min_font_px = float(ev.get("min_font_px", 12.0))
        self.window = ev["window"]
        self.events = ev["events"]
        self.marks = {e["name"]: e for e in self.events if e["type"] == "mark"}
        rows = list(csv.DictReader(open(self.path / "frames.csv", encoding="utf-8")))
        self.index = np.array([int(r["index"]) for r in rows])
        self.times = np.array([float(r["t"]) for r in rows])
        self.moves = [e for e in self.events if e["type"] == "move"]
        self.clicks = [e for e in self.events if e["type"] == "click"]
        self._cache: dict[int, np.ndarray] = {}

    def t(self, name: str) -> float:
        return float(self.marks[name]["t"])

    def frame_at(self, t: float) -> np.ndarray:
        k = int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 1))
        i = int(self.index[k])
        if i not in self._cache:
            if len(self._cache) > 6:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = cv2.imread(str(self.path / "frames" / f"{i:06d}.jpg"))
        return self._cache[i]

    def cursor_at(self, t: float) -> tuple[float, float]:
        pos = (self.window[0] / 2, self.window[1] / 2)
        for m in self.moves:
            if m["t"] > t:
                break
            if t < m["t1"]:
                u = _ease((t - m["t"]) / max(m["t1"] - m["t"], 1e-6))
                return (
                    m["frm"][0] + (m["to"][0] - m["frm"][0]) * u,
                    m["frm"][1] + (m["to"][1] - m["frm"][1]) * u,
                )
            pos = tuple(m["to"])
        return pos

    def clicks_at(self, t: float) -> list[tuple[tuple[float, float], float]]:
        """(position, age in seconds) of clicks younger than half a second."""
        return [(tuple(c["at"]), t - c["t"]) for c in self.clicks if 0.0 <= t - c["t"] < 0.5]


def _ease(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------------------
# The cut
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """One stretch of output: take time t0..t1 played over `duration` seconds."""

    kind: str  # "take", "hook" or "card"
    duration: float
    t0: float = 0.0
    t1: float = 0.0
    caption: str = ""
    sub: str = ""
    lines: list[str] = field(default_factory=list)  # end-card text
    flag: bool = False  # hook: highlight the outer camera pair
    text: bool = True  # the shot shows UI text, so the legibility rule applies

    @property
    def speed(self) -> float:
        return (self.t1 - self.t0) / self.duration if self.kind == "take" else 1.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def text_layer(size, items) -> np.ndarray:
    """
    An RGBA overlay with text blocks. `items` are dicts with text, font, xy,
    width, fill, anchor ("top" or "bottom" for the block), box (draw a plate).
    """
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for it in items:
        if not it["text"]:
            continue
        lines = _wrap(draw, it["text"], it["font"], it["width"])
        asc, desc = it["font"].getmetrics()
        line_h = int((asc + desc) * 1.12)
        block_h = line_h * len(lines)
        x, y = it["xy"]
        if it.get("anchor") == "bottom":
            y -= block_h
        if it.get("box"):
            pad = int(line_h * 0.35)
            w = max(draw.textlength(s, font=it["font"]) for s in lines)
            x0 = x - pad if not it.get("center") else (size[0] - w) / 2 - pad
            draw.rounded_rectangle(
                (x0, y - pad, x0 + w + 2 * pad, y + block_h + pad * 0.6),
                radius=pad,
                fill=(0, 0, 0, 175),
            )
        for k, s in enumerate(lines):
            lx = x if not it.get("center") else (size[0] - draw.textlength(s, font=it["font"])) / 2
            draw.text(
                (lx, y + k * line_h), s, font=it["font"], fill=it.get("fill", (255, 255, 255, 255))
            )
    return np.asarray(img)


def blend(canvas: np.ndarray, rgba: np.ndarray) -> None:
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    if not a.any():
        return
    rgb = rgba[:, :, 2::-1].astype(np.float32)  # RGB -> BGR
    canvas[:] = (canvas * (1 - a) + rgb * a).astype(np.uint8)


def draw_cursor(img: np.ndarray, p: tuple[float, float], size: float) -> None:
    x, y = p
    s = size
    pts = np.array(
        [(0, 0), (0, 1.0), (0.27, 0.76), (0.45, 1.13), (0.6, 1.06), (0.42, 0.7), (0.75, 0.7)]
    )
    poly = (pts * s + (x, y)).astype(np.int32)
    cv2.fillPoly(img, [poly], (255, 255, 255), cv2.LINE_AA)
    cv2.polylines(img, [poly], True, (20, 20, 20), max(1, int(s / 14)), cv2.LINE_AA)


def draw_ripple(img: np.ndarray, p, age: float, size: float) -> None:
    u = age / 0.5
    r = int(size * (0.3 + 1.2 * u))
    overlay = img.copy()
    cv2.circle(
        overlay, (int(p[0]), int(p[1])), r, (247, 195, 79), max(2, int(size / 8)), cv2.LINE_AA
    )
    cv2.addWeighted(overlay, 1 - u, img, u, 0, img)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


#: The framing rules every recorded shot must pass before anything is encoded.
#: They are the reasons the first cut failed: subjects shown as thin strips,
#: UI text too small to read on a phone, regions blown up past their pixels.
MIN_TEXT_PX = 22  # smallest UI text, in output pixels
MAX_ZOOM = 1.6  # output pixels per captured pixel
MIN_PICTURE_H = {"vertical": 480, "landscape": 560}  # output pixels


class FramingError(ValueError):
    """A shot breaks a framing rule; the message lists every one that does."""


class Renderer:
    def __init__(self, take: Take, fmt: str, images_dir: Path, brand: str):
        self.take, self.fmt = take, fmt
        self.W, self.H = FORMATS[fmt]
        self.images = images_dir
        self.brand = brand
        if fmt == "vertical":
            # Picture full width, centred a little above the middle; captions
            # hug it from above and below rather than sitting at the screen edges.
            self.centre_y, self.max_h = 900, 1180
            self.cap_font, self.sub_font = _font(FONT_BOLD, 70), _font(FONT_SEMI, 44)
        else:
            # Picture in the top 960 px, captions in the band below it: they
            # never cover the interface.
            self.band = 960
            self.cap_font, self.sub_font = _font(FONT_BOLD, 44), _font(FONT_SEMI, 32)
        self._overlays: dict[int, np.ndarray] = {}
        self._hook_cache: dict[str, np.ndarray] = {}

    # -- where a picture goes ------------------------------------------------------
    def place(self, w: int, h: int) -> tuple[int, int, int, int]:
        """Output rectangle (x, y, w, h) for a picture of w x h pixels."""
        if self.fmt == "vertical":
            dw, dh = self.W, round(self.W * h / w)
            if dh > self.max_h:
                dh, dw = self.max_h, round(self.max_h * w / h)
            return ((self.W - dw) // 2, self.centre_y - dh // 2, dw, dh)
        k = min(self.W / w, self.band / h)
        dw, dh = round(w * k), round(h * k)
        return ((self.W - dw) // 2, (self.band - dh) // 2, dw, dh)

    def check(self, edit: list[Segment]) -> None:
        """Refuse to render a cut whose shots break the framing rules."""
        problems = []
        for i, seg in enumerate(edit):
            if seg.kind != "take":
                continue
            frame = self.take.frame_at(seg.t0 + 1e-3)
            fh, fw = frame.shape[:2]
            _, _, dw, dh = self.place(fw, fh)
            zoom = dw / fw
            window_w = fw / self.take.scale
            text_px = self.take.min_font_px * dw / window_w
            name = f"segment {i} ({seg.caption or 'no caption'}, {fw}x{fh} captured)"
            if dh < MIN_PICTURE_H[self.fmt]:
                problems.append(f"{name}: picture only {dh} px tall -- a strip, not a shot")
            if zoom > MAX_ZOOM:
                problems.append(f"{name}: enlarged {zoom:.2f}x (max {MAX_ZOOM}), record it larger")
            if seg.text and text_px < MIN_TEXT_PX:
                problems.append(
                    f"{name}: smallest text {text_px:.1f} px (min {MIN_TEXT_PX}) -- raise "
                    f"the presentation scale or record a narrower window"
                )
        if problems:
            raise FramingError("framing rules broken:\n  " + "\n  ".join(problems))

    # -- per-segment overlays -------------------------------------------------------
    def overlay(self, i: int, seg: Segment, dest) -> np.ndarray:
        if i in self._overlays:
            return self._overlays[i]
        W, H = self.W, self.H
        x, y, w, h = dest
        if self.fmt == "vertical":
            items = [
                dict(text=seg.caption, font=self.cap_font, xy=(60, y - 40), width=W - 120,
                     anchor="bottom"),
                dict(text=seg.sub, font=self.sub_font, xy=(60, y + h + 36), width=W - 120,
                     fill=(*ACCENT, 255)),
                dict(text=self.brand, font=_font(FONT_SEMI, 34), xy=(60, H - 110),
                     width=W - 120, fill=(150, 150, 150, 255)),
            ]  # fmt: skip
        else:
            items = [
                dict(text=seg.caption, font=self.cap_font, xy=(60, self.band + 10), width=W - 120),
                dict(text=seg.sub, font=self.sub_font, xy=(60, self.band + 66), width=W - 120,
                     fill=(*ACCENT, 255)),
            ]  # fmt: skip
        layer = text_layer((W, H), items)
        self._overlays[i] = layer
        return layer

    # -- pieces ------------------------------------------------------------------
    def take_frame(self, canvas, seg: Segment, t: float):
        """The whole staged frame at time `t`, placed, with cursor and clicks."""
        tk = self.take
        frame = tk.frame_at(t)
        fh, fw = frame.shape[:2]
        x, y, w, h = self.place(fw, fh)
        canvas[y : y + h, x : x + w] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        k = w / (fw / tk.scale)  # output pixels per window pixel

        def to_out(p):
            return (x + p[0] * k, y + p[1] * k)

        size = float(np.clip(24 * k, 26, 54))
        for p, age in tk.clicks_at(t):
            draw_ripple(canvas, to_out(p), age, size)
        c = to_out(tk.cursor_at(t))
        if x <= c[0] < x + w and y <= c[1] < y + h:
            draw_cursor(canvas, c, size)
        if seg.speed > 1.6:
            label = f"x{seg.speed:.0f}"
            cv2.putText(canvas, label, (x + w - 140, y + 64), cv2.FONT_HERSHEY_DUPLEX, 1.5,
                        (255, 255, 255), 3, cv2.LINE_AA)  # fmt: skip
        return (x, y, w, h)

    def _cam_image(self, cam: str, frame: int) -> np.ndarray:
        key = f"{cam}/{frame:04d}"
        if key not in self._hook_cache:
            if len(self._hook_cache) > 40:
                self._hook_cache.clear()
            self._hook_cache[key] = cv2.imread(str(self.images / cam / f"{frame:04d}.png"))
        return self._hook_cache[key]

    def hook_rect(self):
        return self.place(1280 * 2, 800 * 2)

    def hook_frame(self, canvas, seg: Segment, u: float) -> None:
        """Four camera views flicking through the capture; the outer pair flagged."""
        x, y, w, h = self.hook_rect()
        tw, th = w // 2, h // 2
        # t0 carries on the flicker across consecutive hook segments.
        frame = int((seg.t0 + u * seg.duration) / 0.18) % 30
        flag = seg.flag
        for k, cam in enumerate(("cam0", "cam1", "cam2", "cam3")):
            im = cv2.resize(
                self._cam_image(cam, frame), (tw - 8, th - 8), interpolation=cv2.INTER_AREA
            )
            if flag and cam in ("cam1", "cam2"):
                im = (im * 0.35).astype(np.uint8)
            tx, ty = x + (k % 2) * tw + 4, y + (k // 2) * th + 4
            canvas[ty : ty + th - 8, tx : tx + tw - 8] = im
            colour = (60, 60, 230) if flag and cam in ("cam0", "cam3") else (90, 90, 90)
            cv2.rectangle(
                canvas, (tx - 3, ty - 3), (tx + tw - 6, ty + th - 6), colour, 5 if flag else 2
            )
            cv2.putText(canvas, cam, (tx + 14, ty + 46), cv2.FONT_HERSHEY_DUPLEX, 1.3,
                        (255, 255, 255), 2, cv2.LINE_AA)  # fmt: skip

    def card_frame(self, canvas, seg: Segment, u: float) -> None:
        fade = min(1.0, u * seg.duration / 0.4)
        big, small = _font(FONT_BOLD, 76 if self.fmt == "vertical" else 64), _font(FONT_SEMI, 40)
        items, y = [], self.H * (0.30 if self.fmt == "vertical" else 0.26)
        url = _font(FONT_SEMI, 32 if self.fmt == "vertical" else 40)
        for k, line in enumerate(seg.lines):
            # The repository address has no space to wrap at: smaller, not cut.
            font = big if k == 0 else (url if "/" in line else small)
            fill = (255, 255, 255, int(255 * fade)) if k != 1 else (*ACCENT, int(255 * fade))
            items.append(
                dict(text=line, font=font, xy=(80, y), width=self.W - 160, center=True, fill=fill)
            )
            y += (210 if self.fmt == "vertical" else 150) if k == 0 else 80
        blend(canvas, text_layer((self.W, self.H), items))

    # -- the whole cut -----------------------------------------------------------
    def frames(self, edit: list[Segment]):
        self.check(edit)
        for i, seg in enumerate(edit):
            n = max(1, int(round(seg.duration * FPS)))
            for f in range(n):
                u = f / n
                canvas = np.empty((self.H, self.W, 3), np.uint8)
                canvas[:] = BG
                if seg.kind == "take":
                    dest = self.take_frame(canvas, seg, seg.t0 + (seg.t1 - seg.t0) * u)
                elif seg.kind == "hook":
                    self.hook_frame(canvas, seg, u)
                    dest = self.hook_rect()
                else:
                    self.card_frame(canvas, seg, u)
                    dest = (0, 0, self.W, self.H)
                blend(canvas, self.overlay(i, seg, dest))
                yield canvas


def encode(frames, size, out: Path, music: Path | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-"]  # fmt: skip
    if music is not None:
        cmd += ["-i", str(music), "-shortest", "-c:a", "aac", "-b:a", "192k"]  # fmt: skip
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out)]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def make_gif(video: Path, gif: Path, start: float, length: float, width: int = 960) -> None:
    vf = (f"fps=12,scale={width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=sierra2_4a")  # fmt: skip
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(length),
                    "-i", str(video), "-vf", vf, str(gif)], check=True)  # fmt: skip


def main() -> int:
    import importlib

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("take")
    p.add_argument("--cut", default="cut_rig4cam", help="caption module (cut_*.py here)")
    p.add_argument("--format", choices=sorted(FORMATS), default="vertical")
    p.add_argument("-o", "--output", required=True)
    p.add_argument(
        "--images", default=str(Path(__file__).resolve().parents[2] / "examples/rig_4cam/images")
    )
    p.add_argument("--music")
    p.add_argument("--gif", help="also write a GIF excerpt (landscape cut only)")
    a = p.parse_args()
    cut = importlib.import_module(a.cut)
    BRAND, build_edit, gif_window = cut.BRAND, cut.build_edit, cut.gif_window

    take = Take(a.take)
    edit = build_edit(take, a.format)
    r = Renderer(take, a.format, Path(a.images), BRAND)
    out = Path(a.output)
    encode(r.frames(edit), FORMATS[a.format], out, Path(a.music) if a.music else None)
    total = sum(s.duration for s in edit)
    print(f"{out}: {total:.1f} s, {len(edit)} segments")
    if a.gif:
        start, end = gif_window(edit)
        make_gif(out, Path(a.gif), start, end - start)
        print(f"{a.gif}: {start:.1f} s to {end:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
