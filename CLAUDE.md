# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Headcount is a real-time, multi-camera pedestrian tracking and headcount system. It maps multiple overlapping camera feeds onto a unified 2D Bird's-Eye-View (BEV) floorplan (`map.png`), tracks unique individuals across cameras, and reports zone occupancy. It runs as several decoupled processes that communicate exclusively through **Redis** (queues + key/value state). There is no single monolithic entrypoint — the pipeline is a set of workers wired together by `run_pipeline.sh`.

## Running

Redis must be up first (`docker-compose up -d` starts `redis:7.2-alpine` on `localhost:6379`). Then, inside the venv:

```bash
source .venv/bin/activate
./run_pipeline.sh              # full pipeline: init_db → global_matcher → reid_worker → per-camera sct_workers → visualizer
./run_pipeline.sh --tune       # everything EXCEPT global_matcher, for ReID threshold tuning
```

`run_pipeline.sh` flushes Redis, runs `workers/init_db.py`, starts the ReID worker and **blocks until it sets `reid_worker:ready=1`** (model load / first-run download can take a while), then launches one `sct_worker.py` per camera listed in `config.json`, and finally the PIL visualizer. Ctrl+C traps and `pkill`s all children.

There is no test suite, linter, or build step. `run_pipeline.bat` is the Windows equivalent. `commands.md` has the full calibrate → tune → run walkthrough; `project_walkthourgh.txt` has an in-depth architecture + math writeup.

## Architecture: the fast/slow path split

Everything hinges on splitting spatial tracking (fast) from visual re-ID (slow) so tracking stays high-FPS while the heavy ViT runs asynchronously.

- **`workers/sct_worker.py`** — one process per camera. Runs YOLO-pose (`yolo26l-pose.pt`) + BoT-SORT (`botsort.yaml`) for local tracking, extracts a foot ground-contact point (ankles → knees → bbox-bottom fallback), projects it to floorplan pixels via the camera's homography matrix, and smooths it with a `Kalman2D` filter.
  - **Dynamic Kalman measurement noise (R)**: trust the point less as its source degrades (`NOISE_BY_QUALITY` ankle<knee<bbox) **and** as it projects farther from the camera (distance-based confidence — see `cam_position` below). Both stack into the per-update `meas_noise`.
  - **BoT-SORT is tuned for FIXED cameras**: `botsort.yaml` sets `gmc_method: none` — global motion compensation is for moving cameras; on static crowded scenes it mis-reads moving people as camera motion, so it's off.
  - **Fast path**: every frame, pushes position payloads to `warehouse:queue:matcher`.
  - **Slow path**: bbox crops to `warehouse:queue:reid` — every ~10th frame for new/unmatched tracks (`REID_INTERVAL`), plus a slow refresh (`REID_REFRESH_INTERVAL`) for matched tracks so feature banks don't go stale. The crop queue is capped (`reid_queue_max`) so bursts can't flood Redis.
  - Also writes annotated JPEGs to `frame:{cam_id}` for the visualizer.
- **`workers/reid_worker.py`** — the ViT feature extractor. Loads a `timm` backbone (default `BACKBONE_TYPE = "siglip"` → `vit_base_patch16_siglip_224`; `"swin"` also available), or **OSNet** person-ReID (`osnet_x1_0_msmt17.pth`, torchreid) via the commented `BACKBONE_TYPE = "osnet"` toggle at the top of the class — A/B them with the tuner; **switching the backbone changes the similarity scale, so you must re-tune `reid_sim_threshold`**. Consumes `warehouse:queue:reid`, runs fp16-autocast inference, L2-normalizes embeddings, pushes them to `warehouse:queue:reid_result`, and drops crops older than `stale_reid_sec`. Sets `reid_worker:ready`.
- **`workers/global_matcher.py`** — the coordinator. Consumes fast-path positions and assigns global IDs (Hungarian / `scipy.linear_sum_assignment` over BEV distance, with a speed gate); consumes late-arriving embeddings to fill each track's `feature_bank`; dedupes cross-camera duplicates by spatial proximity + visual similarity; recovers briefly-lost tracks spatially; ages tracks (short-term lost → mid-term lost → deleted). Writes live state to `state:gallery` and mappings to `global_id_map`. `GalleryEntry` is the per-person state object.
  - **Headcount stability**: a new track is **tentative** and NOT counted until it survives `confirm_hits` position updates — this lets cross-camera dedup merge a person's duplicate copies before either is counted, and discards 1–2 frame ghosts. The count only includes **confirmed** tracks seen within `count_hold_sec` (steady through brief detection gaps). `unique_people`, `entries`, and `zone_counts` are all computed over this same confirmed set, so the BEV dots and the number always agree.
  - **N cameras**: everything is keyed per-`cam_id` (`active_local_ids`, `last_cam_update`, `cam_history`); cross-camera dedup runs pairwise over primary-camera groups with a shared `merged` set, so 3+ cameras collapse transitively into one global ID. Stale per-camera bindings are pruned each cleanup so a camera that lost someone can't block re-binding later. Add a camera = add a `config.json` `cameras.<id>` entry (with its matrix); launcher, matcher, and visualizer pick it up automatically.
- **`visualizer.py`** — PIL showcase (replaced the old Streamlit `dashboard.py`). Reads `frame:{cam_id}` for live camera thumbnails and `state:gallery` to render EMA-smoothed dots + trails on native-resolution `map.png` (world coords are native map pixels — do not rescale the BEV or dots misalign), with total headcount + per-zone occupancy in the header. Writes `live_view.png` each tick; opens an OpenCV window when `$DISPLAY` is set.

Messages on all queues are **msgpack-encoded** (`msgpack.packb`), not JSON. Redis is `db=0`, `socket_timeout=None`.

## config.json is the single source of truth

`run_pipeline.sh`, `global_matcher.py`, `sct_worker.py`, and `visualizer.py` all read `config.json` at runtime. To add/remove a camera or change matcher behavior, edit this file — do not hardcode. Key sections:
- `cameras.<id>` — `video_path` + `matrix_path` (homography `.npy`); optional `cam_position: [x, y]` (the camera's location in `map.png` pixels — enables distance-based confidence; omit to disable). Each entry spawns one `sct_worker`.
- `layout` — `map_path`, `pixels_per_meter` (50.0), and `zones` (polygons `{"name","points":[[x,y],...]}`, legacy rects still read).
- `matcher` — speed/distance gates and lost-track decay times, plus:
  - `reid_sim_threshold` — cosine cutoff for "same person"; **model-specific, re-tune after any backbone switch**.
  - `spatial_match_radius_m`, `dedup_spatial_radius_m`, `recover_radius_m` — BEV match / cross-cam merge / lost-recovery distances.
  - `confirm_hits`, `count_hold_sec` — headcount-stability knobs (see global_matcher above).
  - `stale_position_sec`, `stale_reid_sec`, `reid_queue_max` — backlog/latency self-healing (drop only stale data, never live).

## Calibration (tools/)

Homography matrices in `calibration/*.npy` map each camera's pixels to floorplan pixels; tracking is meaningless without them. Regenerate with one of:
- `tools/layout_calibrator.py` — click ≥4 landmarks between a camera frame and `map.png` (required for BEV output).
- `tools/calibrate_person_track.py` — pair a walking person across two cameras; auto-collects foot points, fits with MAGSAC++.
- `tools/calibrate_with_floorplan.py` — manual point-click, RANSAC fit. (`tools/extract_frame.py` grabs a first frame for these.)
- `tools/redis_reid_tuner.py` — run against `--tune` mode to pick `reid_sim_threshold`, then write it back to `config.json`.
- `tools/zone_editor.py` — open `map.png`, click polygon vertices to define named areas (any count, 3+ points each), save into `config.json` `layout.zones`. Zones are stored as `{"name", "points": [[x,y],...]}`; the matcher does point-in-polygon tests for per-zone headcount (legacy `{x1,y1,x2,y2}` rect format still supported on read).

## Notes

- `torchreid` (OSNet) must be installed manually from the vendored `deep-person-reid/` (`python setup.py install`) per `requirements.txt`; the active default ReID path uses `timm`/SigLIP, not OSNet.
- Model weights (`*.pt`, `*.pth`), `*.npy`, and `videos/` are gitignored — treat them as local assets, not tracked source.
- `calibration_old/` is stale; the live matrices live in `calibration/`.
