# Headcount — Operator Guide (Linux)

Full walkthrough: setup → calibrate → draw zones → tune ReID → run. Each step says
**what it does**, **when you need it**, and **how to run it**.

Everything talks over Redis. `config.json` is the single source of truth (cameras,
zones, matcher knobs) — edit it, don't hardcode.

---

## 0. One-time setup

**What:** start Redis and the Python env. Nothing runs without these.

```bash
# Redis (queues + shared state) — must be up first
docker compose up -d            # starts redis:7.2-alpine on localhost:6379

# Python environment
source .venv/bin/activate       # do this in EVERY new terminal
```

Check Redis is alive:
```bash
redis-cli ping                  # -> PONG
```

**Assets that must exist at repo root:** `map.png` (floorplan), `yolo26l-pose.pt`
(detector), and — only if using OSNet ReID — `osnet_x1_0_msmt17.pth`.

---

## 1. Extract a calibration frame  (`tools/extract_frame.py`)

**What:** grabs the first frame of a video as a still image.
**When:** only for the *manual* calibration options (2B / 2C use the video directly,
so you can skip this if you use Option A).

```bash
python tools/extract_frame.py --video videos/cam2.mp4 --output calibration/cam2_frame.jpg
python tools/extract_frame.py --video videos/cam3.mp4 --output calibration/cam3_frame.jpg
```

---

## 2. Calibrate cameras → floorplan  (REQUIRED, once per camera)

**What:** builds each camera's homography matrix (`calibration/<cam>_matrix.npy`) that
maps camera pixels → floorplan pixels on `map.png`.
**When:** always — tracking is meaningless without it. Redo if a camera physically moves.
Pick **one** option per camera.

### Option A — Person-track calibration  (RECOMMENDED)
**What:** a person walks; the tool auto-collects their foot points across two cameras and
fits the homography with MAGSAC++. Most accurate, least clicking.
```bash
python tools/calibrate_person_track.py --video1 videos/cam2.mp4 --video2 videos/cam3.mp4
```
*Controls:*
- **Paused by default** — press `1` to advance Cam 1, `2` to advance Cam 2 until their footsteps are in sync.
- **Space** — play / pause.
- **Click** — click a person in the LEFT panel, then the SAME person in the RIGHT panel to pair them. Let them walk to auto-collect points.
- **f** — fit homography (MAGSAC++).
- **s** — save matrices to `calibration/`.

### Option B — Manual floorplan calibration
**What:** click matching landmarks between a camera still and `map.png`; RANSAC fit.
**When:** no walking person available; static scene with clear landmarks.
```bash
python tools/calibrate_with_floorplan.py --image calibration/cam2_frame.jpg --floor-plan map.png --output calibration/cam2_matrix.npy
python tools/calibrate_with_floorplan.py --image calibration/cam3_frame.jpg --floor-plan map.png --output calibration/cam3_matrix.npy
```

### Option C — Layout calibrator
**What:** same goal (camera → BEV layout), click ≥4 landmarks between a camera frame and the layout.
```bash
python tools/layout_calibrator.py --video videos/cam2.mp4 --layout map.png --output calibration/cam2_matrix.npy
python tools/layout_calibrator.py --video videos/cam3.mp4 --layout map.png --output calibration/cam3_matrix.npy
```

**Verify:** each configured camera in `config.json` has a matching `matrix_path` file.

---

## 3. Draw zones for per-area headcount  (`tools/zone_editor.py`)

**What:** click polygons on the floorplan to define named areas (Counter, Entrance, …).
The matcher counts how many people are inside each; the visualizer shows each area's live count.
**When:** whenever you want per-area counts, or the layout / area definitions change.

```bash
python tools/zone_editor.py            # opens map.png from config.json layout.map_path
```
*Controls:*
- **Left-click** — add a polygon vertex.
- **Right-click** — close the current polygon, then type its name on-screen and press **Enter** (empty = auto "Area A/B/C…").
- **u** — undo the last vertex.
- **d** — hover over an existing zone (it highlights red, shows `[d=delete]`) and press `d` to delete that specific zone.
- **s** — force save. **r** — reload from config. **q / Esc** — quit.

Any number of zones, any number of points per zone (3+). **Every add/delete auto-saves**
to `config.json` → `layout.zones` (format `{"name","points":[[x,y],...]}`), so the file
always matches the screen.

---

## 4. Choose the ReID model  (optional — default is fine)

**What:** the visual re-identification backbone that distinguishes people by appearance.
Used as a **tiebreaker** (spatial position is the primary signal for overlapping cameras).
**When:** only if you want to A/B a more discriminative / faster model.

Switch in `workers/reid_worker.py`, top of `ModernReIDWorker` — **exactly one line active:**
```python
BACKBONE_TYPE = "siglip"      # default: generic ViT, robust to lighting/color
# BACKBONE_TYPE = "osnet"     # person-ReID (osnet_x1_0_msmt17.pth): more discriminative + faster
```
- **To use OSNet:** comment the `siglip` line, uncomment the `osnet` line.
- **To go back:** reverse it.
- **After any switch you MUST re-tune** (Step 5) — the similarity scale differs, so the old
  `reid_sim_threshold` is invalid. OSNet needs its weights file at repo root.

---

## 5. Tune the ReID similarity threshold  (`tools/redis_reid_tuner.py`)

**What:** finds the cosine-similarity cutoff that says "same person" vs "different person".
**When:** first deployment, after changing lighting/camera angles, or after switching the ReID model (Step 4).

1. Start the pipeline in **tuning mode** (SCT trackers + ReID worker, matcher OFF):
   ```bash
   ./run_pipeline.sh --tune
   ```
2. In a **new terminal**:
   ```bash
   source .venv/bin/activate
   python tools/redis_reid_tuner.py
   ```
3. A similarity-matrix grid appears. Slide the threshold until the **same** person across
   cameras is Green and **different** people are Blue. Aim for the widest clean gap. (Stale IDs >10s auto-evict.)
4. Note the "Final Tuned Threshold", press `q`.
5. Put that value in `config.json` → `matcher.reid_sim_threshold`.
6. `Ctrl+C` the tuning pipeline.

**A/B two models:** run this once on `siglip`, once on `osnet`; keep whichever gives the
wider same/different gap, and save its threshold.

---

## 6. Run the full pipeline  (`run_pipeline.sh`)

**What:** flushes Redis, then launches everything wired from `config.json`.
```bash
./run_pipeline.sh
```
Startup order:
1. `workers/init_db.py` — clears queues + stale state in Redis.
2. `workers/global_matcher.py` — assigns global IDs (BEV spatial priority + ReID tiebreak), per-zone counts.
3. `workers/reid_worker.py` — loads the ReID model; **blocks the launch until `reid_worker:ready=1`** (first-run model download can take a while).
4. One `workers/sct_worker.py` per camera in `config.json` (YOLO-pose + BoT-SORT + homography + Kalman).
5. `visualizer.py` — PIL live view (see Step 7).

**Stop:** `Ctrl+C` — traps and kills all child workers.

---

## 7. View the output  (`visualizer.py`)

**What:** single PIL composite — header with **Total People** + per-zone occupancy, a column
of live camera thumbnails, and the BEV `map.png` with tracked people (colored dots + IDs +
trails) and each zone's live count drawn on it.
- Writes `live_view.png` every tick (open it on a headless server).
- Opens a live OpenCV window when `$DISPLAY` is set (press `q` in the window to close it).

Runs automatically from `run_pipeline.sh`. To run it alone against a live pipeline:
```bash
python visualizer.py
```

---

## `config.json` reference — matcher knobs

| Key | Meaning |
|---|---|
| `reid_sim_threshold` | Cosine cutoff for "same person" (set by Step 5; **model-specific**). |
| `spatial_match_radius_m` | Max BEV distance to attach a detection to an existing track. |
| `dedup_spatial_radius_m` | Cross-camera duplicates within this distance get merged. |
| `recover_radius_m` | Distance to recover a briefly-lost track. |
| `max_speed_mps` | Speed gate — rejects impossible jumps. |
| `lost_short_term_sec` / `lost_mid_term_sec` | Track kept as lost, then deleted, after these. |
| `feature_bank_size` | How many ReID embeddings retained per person. |
| `time_buffer_ms` | Jitter-buffer window aligning multi-camera frames. |
| `stale_position_sec` | Under backlog, drop position updates older than this (self-healing latency). |
| `stale_reid_sec` | Drop ReID crops/embeddings older than this (don't waste compute). |
| `reid_queue_max` | Hard cap on the ReID crop queue length (burst protection). |

---

## Quick reference

| Task | Command |
|---|---|
| Start Redis | `docker compose up -d` |
| Activate env | `source .venv/bin/activate` |
| Extract a frame | `python tools/extract_frame.py --video <v> --output <img>` |
| Calibrate (auto) | `python tools/calibrate_person_track.py --video1 <a> --video2 <b>` |
| Draw zones | `python tools/zone_editor.py` |
| Tune ReID | `./run_pipeline.sh --tune` + `python tools/redis_reid_tuner.py` |
| Run everything | `./run_pipeline.sh` |
| Headless output | open `live_view.png` |
| Stop | `Ctrl+C` in the pipeline terminal |
