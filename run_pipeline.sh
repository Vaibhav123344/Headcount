#!/bin/bash
# Bash script to start the entire pipeline

echo "Starting Database Initialization..."
python workers/init_db.py

echo "Starting Global Matcher..."
python workers/global_matcher.py &

echo "Starting Re-ID Feature Extractor..."
python workers/reid_worker.py &

echo "Starting Single Camera Trackers (Pose + Homography)..."
python workers/sct_worker.py --cam_id cam_00 --source videos/cam_00.mp4 --homography calibration/cam_00_matrix.npy &
python workers/sct_worker.py --cam_id cam_01 --source videos/cam_01.mp4 --homography calibration/cam_01_matrix.npy &
python workers/sct_worker.py --cam_id cam_02 --source videos/cam_02.mp4 --homography calibration/cam_02_matrix.npy &

echo "Pipeline is running. Press Ctrl+C to stop."
wait
