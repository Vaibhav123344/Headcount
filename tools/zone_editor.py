"""
Interactive zone editor.

Open the floorplan (map.png) and draw named areas by clicking polygon
vertices. Any number of zones, any number of points per zone (3+).
Zones are written to config.json under layout.zones as:

    {"name": "Area A", "points": [[x, y], [x, y], ...]}

Every add / delete auto-saves to config.json, so the file always matches
what you see on screen. The global_matcher does point-in-polygon tests to
report per-zone headcount; the visualizer draws them.

Controls
--------
  Left click       add a vertex to the current polygon
  Right click      close current polygon  ->  type its name, Enter to confirm
  u                undo last vertex of the current polygon
  d                delete the zone under the mouse cursor
  s                force re-save to config.json
  r                reload zones from config.json (discard current drawing)
  q / Esc          quit

While naming: type letters/numbers/spaces, Backspace edits,
Enter confirms (empty -> auto name), Esc cancels the new polygon.

Run:
  python tools/zone_editor.py
  python tools/zone_editor.py --map map.png --config config.json
"""
import argparse
import json
import string

import cv2
import numpy as np

PALETTE = [
    (66, 135, 245), (46, 204, 113), (231, 76, 60), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (52, 152, 219),
]
TYPEABLE = set(string.ascii_letters + string.digits + " _-")


def load_map(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not load map image: {path}")
    return img


def load_zones(config_path):
    with open(config_path) as f:
        cfg = json.load(f)
    out = []
    for z in cfg.get("layout", {}).get("zones", []):
        if "points" in z:
            out.append({"name": z["name"], "points": [[int(p[0]), int(p[1])] for p in z["points"]]})
        elif "x1" in z:  # migrate old rectangle format
            out.append({"name": z["name"], "points": [
                [int(z["x1"]), int(z["y1"])], [int(z["x2"]), int(z["y1"])],
                [int(z["x2"]), int(z["y2"])], [int(z["x1"]), int(z["y2"])],
            ]})
    return out


def save_zones(config_path, zones):
    with open(config_path) as f:
        cfg = json.load(f)
    cfg.setdefault("layout", {})["zones"] = [
        {"name": z["name"], "points": [[int(x), int(y)] for x, y in z["points"]]}
        for z in zones
    ]
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[saved] {len(zones)} zone(s) -> {config_path}")


def point_in_poly(x, y, pts):
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


class ZoneEditor:
    def __init__(self, map_path, config_path):
        self.map_path = map_path
        self.config_path = config_path
        self.base = load_map(map_path)
        self.zones = load_zones(config_path)
        self.current = []            # vertices of the in-progress polygon
        self.mode = "draw"           # "draw" | "name"
        self.name_buf = ""
        self.mouse = (0, 0)
        self.win = "Zone Editor"
        print(f"[loaded] {len(self.zones)} existing zone(s) from {config_path}")

    def _color(self, i):
        return PALETTE[i % len(PALETTE)]

    def _autosave(self):
        save_zones(self.config_path, self.zones)

    def _on_mouse(self, event, x, y, flags, _):
        self.mouse = (x, y)
        if self.mode == "name":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._start_naming()

    def _start_naming(self):
        if len(self.current) < 3:
            print("[!] need at least 3 points before closing a zone")
            return
        self.mode = "name"
        self.name_buf = ""

    def _commit_name(self):
        name = self.name_buf.strip() or self._auto_name()
        self.zones.append({"name": name, "points": self.current})
        print(f"[+] added '{name}' ({len(self.current)} pts)")
        self.current = []
        self.mode = "draw"
        self.name_buf = ""
        self._autosave()

    def _auto_name(self):
        letters = string.ascii_uppercase
        return f"Area {letters[len(self.zones) % 26]}"

    def _delete_under_mouse(self):
        mx, my = self.mouse
        for i in range(len(self.zones) - 1, -1, -1):  # topmost first
            if point_in_poly(mx, my, self.zones[i]["points"]):
                dropped = self.zones.pop(i)
                print(f"[-] deleted '{dropped['name']}'")
                self._autosave()
                return
        print("[!] no zone under cursor to delete")

    def _render(self):
        img = self.base.copy()
        overlay = img.copy()
        mx, my = self.mouse

        for i, z in enumerate(self.zones):
            c = self._color(i)
            pts = np.array(z["points"], dtype=np.int32)
            hot = point_in_poly(mx, my, z["points"])           # highlight zone under cursor
            cv2.fillPoly(overlay, [pts], (0, 0, 255) if hot else c)
            cv2.polylines(img, [pts], True, (0, 0, 255) if hot else c, 3 if hot else 2)
            cx, cy = pts.mean(axis=0).astype(int)
            tag = z["name"] + ("  [d=delete]" if hot else "")
            cv2.putText(img, tag, (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(img, tag, (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 1)
        cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

        if self.current:
            for p in self.current:
                cv2.circle(img, tuple(p), 5, (0, 0, 255), -1)
            poly = np.array(self.current, np.int32)
            closed = self.mode == "name"
            cv2.polylines(img, [poly], closed, (0, 0, 255), 2)

        # HUD
        if self.mode == "name":
            line1 = f"Name this zone: {self.name_buf}_"
            line2 = "Enter=confirm  Backspace=edit  Esc=cancel"
        else:
            line1 = f"zones:{len(self.zones)}  current pts:{len(self.current)}"
            line2 = "L-click=add  R-click=close  u=undo  d=delete  s=save  r=reload  q=quit"
        for j, txt in enumerate((line1, line2)):
            y = 26 + j * 26
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    def _handle_name_key(self, k):
        if k in (13, 10):            # Enter
            self._commit_name()
        elif k == 27:                # Esc -> cancel this polygon
            self.current = []
            self.mode = "draw"
            self.name_buf = ""
            print("[x] cancelled zone")
        elif k in (8, 127):          # Backspace / Delete
            self.name_buf = self.name_buf[:-1]
        elif 0 <= k < 256 and chr(k) in TYPEABLE:
            self.name_buf += chr(k)

    def _handle_draw_key(self, k):
        if k in (ord("q"), 27):
            return False
        elif k == ord("u"):
            if self.current:
                self.current.pop()
        elif k == ord("d"):
            self._delete_under_mouse()
        elif k == ord("s"):
            self._autosave()
        elif k == ord("r"):
            self.zones = load_zones(self.config_path)
            self.current = []
            print("[reloaded] zones from config")
        return True

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)
        while True:
            cv2.imshow(self.win, self._render())
            k = cv2.waitKey(20) & 0xFF
            if k == 255:             # no key
                continue
            if self.mode == "name":
                self._handle_name_key(k)
            else:
                if not self._handle_draw_key(k):
                    break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--map", default=None, help="floorplan image (default: layout.map_path from config)")
    args = ap.parse_args()

    map_path = args.map
    if map_path is None:
        with open(args.config) as f:
            map_path = json.load(f).get("layout", {}).get("map_path", "map.png")

    ZoneEditor(map_path, args.config).run()
