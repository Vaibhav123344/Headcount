# Headcount: Real-Time Multi-Camera Pedestrian Tracking & Analytics

Headcount is a highly scalable, asynchronous multi-camera tracking system designed to monitor large physical spaces. It detects people, maps their positions onto a unified 2D Bird's-Eye-View (BEV) floorplan, tracks unique individuals seamlessly across overlapping camera feeds, and reports real-time zone occupancy.

By decoupling high-FPS spatial tracking from heavy visual Re-Identification (ReID) through Redis message queues, Headcount maintains real-time performance even in dense crowds.

## ✨ Features

- **Unified 2D Bird's-Eye-View (BEV):** Uses homography projection to map foot-contact points (ankles/knees) from multiple camera feeds onto a single metric floorplan (`map.png`).
- **Cross-Camera Re-Identification (ReID):** Automatically merges tracks when a person moves between cameras using spatial proximity and deep visual embeddings (SigLIP or OSNet).
- **Asynchronous Architecture:** Fast-path spatial tracking (YOLO + BotSORT) runs at camera frame rates, while slow-path ReID (Vision Transformers) runs asynchronously without blocking the pipeline.
- **Robust Trajectory Smoothing:** Implements distance-aware 2D Kalman filtering to smooth projected pixel jitter and estimate pedestrian velocity.
- **Zone Occupancy Analytics:** Defines polygonal zones on the floorplan to provide instant, precise headcount metrics per area.
- **Lost Track Recovery:** Re-identifies and recovers global IDs for individuals who temporarily leave the monitored area and re-enter later.

## 🏗️ Architecture

The pipeline consists of several decoupled worker processes communicating exclusively via **Redis** (queues and key/value state):

1. **`sct_worker.py` (Single Camera Tracker):**
   - Runs one process per camera.
   - Detects people and poses using **YOLOv8-pose** (`yolo26l-pose.pt`).
   - Tracks locally using **BotSORT**.
   - Extracts ground-contact points, projects them to the BEV map, and smooths them with a **Kalman Filter**.
   - Pushes high-frequency position data to the fast-path queue.
   - Pushes low-frequency cropped person images to the slow-path queue.
2. **`reid_worker.py` (Re-Identification):**
   - Consumes person crops from the slow-path queue.
   - Extracts high-dimensional visual embeddings using **SigLIP** (ViT) or **OSNet**.
   - Returns normalized embeddings to the Matcher.
3. **`global_matcher.py` (The Coordinator):**
   - Consumes fast-path positions and slow-path embeddings.
   - Uses the Hungarian algorithm for spatial matching and cross-camera deduplication.
   - Maintains the `state:gallery` and calculates zone occupancies.
4. **`visualizer.py`:**
   - Subscribes to the global state and renders live camera thumbnails, BEV dots, trajectory trails, and headcount metrics on the floorplan.

## 🛠️ Installation & Requirements

### Prerequisites
- Python 3.9+
- Redis Server (Can be run via Docker)
- NVIDIA GPU (Recommended for real-time YOLO and ReID inference)

### Setup
1. Clone the repository and navigate to the directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Redis (if using Docker):
   ```bash
   docker-compose up -d
   ```
*Note: If you wish to use the OSNet backbone instead of SigLIP, you must manually install `torchreid` from the vendored `deep-person-reid/` directory via `python setup.py install`.*

## ⚙️ Configuration (`config.json`)

`config.json` is the single source of truth for the entire pipeline. It controls:
- **`cameras`**: Define your RTSP/Video paths and the path to their respective homography matrices (`.npy`).
- **`layout`**: Specify your map image, `pixels_per_meter`, and named polygonal `zones`.
- **`matcher`**: Tune matching thresholds, speed gates, track decay times, and the `reid_sim_threshold` (cosine similarity cutoff for matching identities).

## 🎯 Calibration Tools

Tracking is meaningless without accurate camera-to-floorplan homography matrices. Use the included tools in the `tools/` directory to calibrate your setup:

- `tools/layout_calibrator.py`: Click ≥4 landmarks between a camera frame and `map.png` to generate the matrix.
- `tools/calibrate_person_track.py`: Automatically collects foot points of a person walking across two cameras and fits the matrix using MAGSAC++.
- `tools/zone_editor.py`: Interactive GUI to click and draw named polygonal zones directly onto your floorplan.
- `tools/redis_reid_tuner.py`: Helper script to automatically tune your `reid_sim_threshold` when you switch ReID backbones.

## 🚀 Running the Pipeline

To launch the full system (Matcher, ReID Worker, per-camera SCT Workers, and the Visualizer):

```bash
# Linux/macOS
./run_pipeline.sh

# Windows
run_pipeline.bat
```

To run in calibration/tuning mode (skips the Global Matcher so you can tune ReID thresholds):
```bash
./run_pipeline.sh --tune
```

## 📝 License

[MIT License](LICENSE)
