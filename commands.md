# Headcount Project Commands (Linux)

Here is the complete step-by-step guide to calibrating the cameras and running your new Inline-ReID pipeline on Linux.

Open your terminal in the `Headcount` directory and run the following:

## Step 1: Extract Calibration Frames
We need the first frame of each video to calibrate the 3D ground plane.

```bash
# Activate your virtual environment first!
source .venv/bin/activate

# Extract frame for Camera 1 (camera2_20260627_103326.mp4)
python tools/extract_frame.py --video videos/camera2_20260627_103326.mp4 --output calibration/cam1_frame.jpg

# Extract frame for Camera 2 (camera3_20260627_103325.mp4)
python tools/extract_frame.py --video videos/camera3_20260627_103325.mp4 --output calibration/cam2_frame.jpg
```

## Step 2: Calibrate the Cameras (Homography)
Run the calibration tool on the extracted frames. A window will pop up. 
Click **exactly 4 points** on the ground that form a rectangle in the real world (e.g., corners of a pallet, tiles on the floor).
Order matters: **Top-Left, Top-Right, Bottom-Right, Bottom-Left**.

**For Camera 1:**
```bash
python tools/calibrate_camera.py --image calibration/cam1_frame.jpg --output calibration/cam1_matrix.npy
```
*(Press any key when it shows the Top-Down preview to close and save it)*

**For Camera 2:**
```bash
python tools/calibrate_camera.py --image calibration/cam2_frame.jpg --output calibration/cam2_matrix.npy
```

## Step 3: Run the Pipeline!
Everything is configured. `run_pipeline.sh` has been updated to use our new zero-latency architecture.

```bash
# First, ensure you have a clean Redis state
python -c "import redis; r = redis.Redis(); r.flushdb(); print('Redis flushed')"

# Start the pipeline
./run_pipeline.sh
```

This will automatically:
1. Initialize the Database mapping structures
2. Start the Global Matcher
3. Start the Object Tracker with Inline-ReID for `cam1`
4. Start the Object Tracker with Inline-ReID for `cam2`
5. Launch the Streamlit Live Dashboard in your web browser!








## Step 2b: Person-Track Calibration (RECOMMENDED — Much More Accurate)
Instead of clicking floor corners manually, use a walking person's feet to build
the cross-camera homography. This gives much better accuracy because it collects
hundreds of ground-truth foot correspondences automatically.

```bash
python tools/calibrate_person_track.py \
    --video1 videos/camera2_20260627_103326.mp4 \
    --video2 videos/camera3_20260627_103325.mp4
```

**Instructions:**
1. Click a person visible in the **left panel** (cam1)
2. Click the **same person** in the **right panel** (cam2)
3. Let them walk around — foot points auto-collect as they move
4. Pair additional people for better floor coverage
5. Press **'f'** to fit the homography and see the reprojection error
6. Press **'s'** to save → `calibration/cam1_matrix.npy` and `calibration/cam2_matrix.npy`

**Controls:** `SPACE`=play/pause, `c`=toggle collection, `x`=clear pair,
`u`=undo, `f`=fit, `s`=save, `r`=reset all, `q`=quit


Quick Summary of the Commands
1. Extract a single frame for calibration:

bash
source .venv/bin/activate
python tools/extract_frame.py --video videos/cam2.mp4 --output calibration/cam1_frame.jpg
python tools/extract_frame.py --video videos/cam3.mp4 --output calibration/cam2_frame.jpg
2. Perform the Dual-Window Floor Plan Calibration:

```bash
python tools/calibrate_with_floorplan.py --image calibration/cam1_frame.jpg --floor-plan map.png --output calibration/cam1_matrix.npy
python tools/calibrate_with_floorplan.py --image calibration/cam2_frame.jpg --floor-plan map.png --output calibration/cam2_matrix.npy
```
3. Or use Person-Track Calibration (recommended):

```bash
python tools/calibrate_person_track.py --video1 videos/cam2.mp4 --video2 videos/cam3.mp4
```
4. Run the optimized pipeline:

bash
python -c "import redis; r = redis.Redis(); r.flushdb(); print('Redis flushed')"
./run_pipeline.sh