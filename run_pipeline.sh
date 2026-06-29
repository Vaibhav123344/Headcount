#!/bin/bash
# Bash script to start the entire pipeline

echo "Starting Database Initialization..."
python workers/init_db.py

echo "Starting Global Matcher..."
python workers/global_matcher.py &


echo "Starting Single Camera Trackers (Pose + Homography)..."
python workers/sct_worker.py --cam_id cam1 --source videos/cam2.mp4 --homography calibration/cam1_matrix.npy &
python workers/sct_worker.py --cam_id cam2 --source videos/cam3.mp4 --homography calibration/cam2_matrix.npy &

echo "Starting Streamlit Dashboard..."
python -m streamlit run dashboard.py --server.headless true &

echo "Pipeline is running. Press Ctrl+C to stop."
wait
