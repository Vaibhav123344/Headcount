# Headcount Project Commands (Linux)

Here is the complete step-by-step guide to calibrating the cameras, tuning ReID, and running your pipeline.

## Step 1: Extract Calibration Frames

We need the first frame of each video to calibrate the 3D ground plane (only required for manual floorplan calibration or layout mapping).

```bash
# Activate your virtual environment first!
source .venv/bin/activate

# Extract frames for all cameras
python tools/extract_frame.py --video videos/cam2.mp4 --output calibration/cam2_frame.jpg
python tools/extract_frame.py --video videos/cam3.mp4 --output calibration/cam3_frame.jpg
# ... repeat for any other cameras
```

## Step 2: Calibrate the Cameras

**Option A: Person-Track Calibration (RECOMMENDED)**
Use a walking person's feet to build cross-camera homography automatically.
```bash
python tools/calibrate_person_track.py --video1 videos/cam2.mp4 --video2 videos/cam3.mp4
```
*Controls:*
- **PAUSED BY DEFAULT:** Use `1` to advance Cam 1, or `2` to advance Cam 2 to perfectly sync their footsteps before playing!
- **SPACE:** Play/Pause.
- **Click:** Click a person in LEFT panel, then SAME person in RIGHT panel to pair. Let them walk to auto-collect points.
- **f:** Fit homography using MAGSAC++.
- **s:** Save matrices to `calibration/`.

**Option B: Manual Floorplan Calibration**
```bash
python tools/calibrate_with_floorplan.py --image calibration/cam2_frame.jpg --floor-plan map.png --output calibration/cam2_matrix.npy
python tools/calibrate_with_floorplan.py --image calibration/cam3_frame.jpg --floor-plan map.png --output calibration/cam3_matrix.npy
```

**Option C: Map Master Camera to 2D BEV Layout (NEW)**
If you want the final tracking to output coordinates on a 2D floorplan layout instead of relative to Camera 2.
```bash
python tools/layout_calibrator.py --video videos/cam3.mp4 --layout map.png --matrix1 calibration/cam2_matrix.npy --matrix2 calibration/cam3_matrix.npy
```

## Step 3: Tune ReID Similarity Threshold (Optional but Recommended)

ReID thresholds depend heavily on your lighting and camera angles. Tune the similarity threshold before running the matcher:

1. Start the pipeline in **Tuning Mode** (starts SCT trackers + ViT ReID worker, but stops matcher):
   ```bash
   ./run_pipeline.sh --tune
   ```
2. Open a new terminal window and run the tuner tool:
   ```bash
   source .venv/bin/activate
   python tools/redis_reid_tuner.py
   ```
3. A visual matrix will appear. Adjust the slider until the same person across cameras is Green, and different people are Blue. (Note: stale IDs older than 10s are automatically evicted).
4. Note the "Final Tuned Threshold" value and press `q` to exit.
5. Open `config.json` and update `"reid_sim_threshold"` with this value.
6. Stop the tuning pipeline (`Ctrl+C` in the first terminal).

## Step 4: Run the Full Pipeline

Everything is configured. The pipeline reads camera setups and matcher configuration directly from `config.json`.

```bash
# Start the full pipeline
./run_pipeline.sh
```

This will automatically:
1. Initialize the Database mapping structures in Redis
2. Start the Global Matcher (using your tuned threshold and Bayesian Probabilistic Fusion)
3. Start the ReID Middleware Worker (runs heavy ViT `timm` models like SigLIP or Swin)
4. Dynamically start Pure-Spatial Object Trackers (`sct_worker.py`) for every camera defined in `config.json`
5. Launch the Streamlit Live Dashboard in your web browser!