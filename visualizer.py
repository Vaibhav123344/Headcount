"""
PIL-based live showcase (replaces the Streamlit dashboard).

Reads everything from Redis:
  - frame:{cam_id}   -> annotated JPEG per camera
  - state:gallery    -> {unique_people, next_id, entries[], zone_counts}

Composites a single frame:
  [ header: total people + zone occupancy ]
  [ camera column ][      BEV layout with tracked people + trails      ]

Writes the composite to live_view.png every tick, and shows a live OpenCV
window when a display is available (press 'q' to quit).
"""
import io
import os
import json
import time

import cv2
import numpy as np
import redis
from PIL import Image, ImageDraw, ImageFont

CAM_W          = 420          # camera thumbnail width
GAP            = 12
HEADER_H       = 78
OUT_PATH       = "live_view.png"
EMA_ALPHA      = 0.3
TRAIL_LEN      = 25
BG             = (24, 26, 32)
FG             = (235, 235, 240)


def _load_config():
    with open("config.json") as f:
        return json.load(f)


def _font(size, bold=True):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for p in (f"/usr/share/fonts/truetype/dejavu/{name}", name):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = _font(30)
F_LABEL = _font(18)
F_SMALL = _font(15)
F_DOT   = _font(16)


def _palette(n=200):
    rng = np.random.default_rng(42)
    return [tuple(int(c) for c in rng.integers(60, 255, size=3)) for _ in range(n)]


COLORS = _palette()


def _load_map(path):
    """Load map.png as an RGB PIL image (alpha flattened onto white)."""
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def _decode_jpeg(raw):
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def _fit_width(img, width):
    w, h = img.size
    return img.resize((width, max(1, round(h * width / w))))


def _fit_box(img, max_w, max_h):
    """Scale to fit inside (max_w, max_h) preserving aspect ratio."""
    w, h = img.size
    s = min(max_w / w, max_h / h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))))


class Visualizer:
    def __init__(self):
        self.cfg = _load_config()
        self.r = redis.Redis(host=self.cfg["redis"]["host"], port=self.cfg["redis"]["port"], db=0)
        self.cams = list(self.cfg.get("cameras", {}).keys())
        layout = self.cfg.get("layout", {})
        self.map_bg = _load_map(layout.get("map_path", "map.png"))
        self.zones = layout.get("zones", [])
        self.smoothed = {}   # gid -> (x, y)
        self.trails = {}     # gid -> [(x, y), ...]
        self.has_display = os.environ.get("DISPLAY") is not None

    @staticmethod
    def _zone_points(z):
        """Return polygon vertices for a zone (supports legacy rectangle format)."""
        if "points" in z:
            return [tuple(p) for p in z["points"]]
        if "x1" in z:
            return [(z["x1"], z["y1"]), (z["x2"], z["y1"]), (z["x2"], z["y2"]), (z["x1"], z["y2"])]
        return []

    # ---- BEV panel -------------------------------------------------------
    def _draw_bev(self, state):
        bev = self.map_bg.copy()
        d = ImageDraw.Draw(bev, "RGBA")

        zone_counts = state.get("zone_counts", {}) if state else {}

        # Zone polygons + live per-zone count
        for z in self.zones:
            pts = self._zone_points(z)
            if not pts:
                continue
            d.polygon(pts, outline=(90, 140, 200, 220), fill=(90, 140, 200, 40))
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            cnt = zone_counts.get(z["name"], 0)
            tag = f"{z['name']}: {cnt}"
            d.text((cx - 40, cy - 8), tag, font=F_LABEL, fill=(0, 0, 0))
            d.text((cx - 41, cy - 9), tag, font=F_LABEL, fill=(40, 80, 150))

        entries = state.get("entries", []) if state else []
        active = set()
        for e in entries:
            gid = e["global_id"]
            active.add(gid)
            rx, ry = e["last_x"], e["last_y"]
            if gid in self.smoothed:
                ox, oy = self.smoothed[gid]
                sx = ox * (1 - EMA_ALPHA) + rx * EMA_ALPHA
                sy = oy * (1 - EMA_ALPHA) + ry * EMA_ALPHA
            else:
                sx, sy = rx, ry
            self.smoothed[gid] = (sx, sy)
            self.trails.setdefault(gid, []).append((sx, sy))
            if len(self.trails[gid]) > TRAIL_LEN:
                self.trails[gid].pop(0)

            color = COLORS[gid % len(COLORS)]
            pts = self.trails[gid]
            for k in range(1, len(pts)):
                d.line([pts[k - 1], pts[k]], fill=color + (int(255 * k / len(pts)),), width=3)

            x, y = int(sx), int(sy)
            d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=color, outline=(0, 0, 0), width=2)
            label = f"ID:{gid}"
            d.text((x + 13, y - 10), label, font=F_DOT, fill=(0, 0, 0))
            d.text((x + 12, y - 11), label, font=F_DOT, fill=color)

        # Drop trails for people who left
        for gid in list(self.smoothed):
            if gid not in active:
                self.smoothed.pop(gid, None)
                self.trails.pop(gid, None)
        return bev

    # ---- Camera column ---------------------------------------------------
    def _draw_cams(self, height):
        col = Image.new("RGB", (CAM_W, height), BG)
        d = ImageDraw.Draw(col)
        n = max(1, len(self.cams))
        # Split the available height into one equal slot per camera so 2, 3 or 4+
        # feeds all fit without the bottom ones being clipped off-canvas.
        slot_h = max(60, (height - GAP * (n + 1)) // n)
        max_w = CAM_W - 2 * GAP
        y = GAP
        for cam in self.cams:
            raw = self.r.get(f"frame:{cam}")
            img = _decode_jpeg(raw) if raw else None
            if img is not None:
                thumb = _fit_box(img, max_w, slot_h)
                col.paste(thumb, (GAP, y))
                d.rectangle([GAP, y, GAP + thumb.width, y + thumb.height], outline=(80, 84, 96), width=1)
                d.text((GAP + 6, y + 4), cam, font=F_LABEL, fill=FG)
            else:
                d.rectangle([GAP, y, CAM_W - GAP, y + slot_h], outline=(80, 84, 96), width=1)
                d.text((GAP + 8, y + slot_h // 2), f"waiting for {cam}...", font=F_SMALL, fill=(150, 150, 160))
            y += slot_h + GAP
        return col

    # ---- Compose ---------------------------------------------------------
    def compose(self, state):
        bev = self._draw_bev(state)
        body_h = max(bev.height, 400)
        cams = self._draw_cams(body_h)

        canvas = Image.new("RGB", (cams.width + GAP + bev.width, HEADER_H + body_h), BG)
        d = ImageDraw.Draw(canvas)

        total = state.get("unique_people", 0) if state else 0
        d.text((GAP, 14), f"HEADCOUNT   Total People: {total}", font=F_TITLE, fill=FG)

        zone_counts = state.get("zone_counts", {}) if state else {}
        parts = [f"{z}: {c}" for z, c in zone_counts.items() if z != "Unknown" or c > 0]
        d.text((GAP, 52), "   |   ".join(parts) if parts else "no active zones",
               font=F_SMALL, fill=(160, 190, 230))

        canvas.paste(cams, (0, HEADER_H))
        canvas.paste(bev, (cams.width + GAP, HEADER_H))
        return canvas

    def run(self):
        print(f"[visualizer] cams={self.cams} display={self.has_display}. Writing {OUT_PATH}.")
        win = "Headcount Live"
        if self.has_display:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while True:
            raw = self.r.get("state:gallery")
            state = json.loads(raw.decode("utf-8")) if raw else None
            canvas = self.compose(state)
            canvas.save(OUT_PATH)
            if self.has_display:
                cv2.imshow(win, cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(0.3)


if __name__ == "__main__":
    Visualizer().run()
