# Headcount Project Commands

Here is the complete step-by-step guide to setting up and running your pipeline on your new videos (`cam1.mp4` and `cam2.mp4`).

Open your PowerShell terminal in the `C:\Users\HP\Documents\Projects\Headcount` directory and run the following:

## Step 1: Extract Calibration Frames
We need the first frame of each video to calibrate the 3D ground plane. I've created a helper script for this.
```powershell
call .venv\Scripts\activate.bat
python tools\extract_frame.py --video videos\cam1.mp4 --output calibration\cam1_frame.jpg
python tools\extract_frame.py --video videos\cam2.mp4 --output calibration\cam2_frame.jpg
```

## Step 2: Calibrate the Cameras (Homography)
Run the calibration tool on the extracted frames. A window will pop up. 
Click **exactly 4 points** on the ground that form a rectangle in the real world (e.g., corners of a pallet, tiles on the floor).
Order matters: Top-Left, Top-Right, Bottom-Right, Bottom-Left.

**For Camera 1:**
```powershell
python tools\calibrate_camera.py --image calibration\cam1_frame.jpg --output calibration\cam1_matrix.npy
```
*(Press any key when it shows the Top-Down preview to close and save it)*

**For Camera 2:**
```powershell
python tools\calibrate_camera.py --image calibration\cam2_frame.jpg --output calibration\cam2_matrix.npy
```

## Step 3: Run the Pipeline!
Everything is configured. `run_pipeline.bat` and `dashboard.py` have been automatically updated to read from your new `cam1.mp4` and `cam2.mp4` files.

```powershell
.\run_pipeline.bat
```

This will automatically:
1. Start the Redis container
2. Wipe any old tracking data
3. Open the Batched Global Matcher
4. Open the Re-ID Worker
5. Open the Object Tracker for `cam1`
6. Open the Object Tracker for `cam2`
7. Launch the Streamlit Live Dashboard in your web browser!