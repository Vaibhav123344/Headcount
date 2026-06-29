@echo off
echo Starting Infrastructure (Redis)...
docker compose up -d

echo Starting Database Initialization...
call .venv\Scripts\activate.bat
python workers\init_db.py

echo Starting Global Matcher...
start "Global Matcher" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate.bat && python workers\global_matcher.py && pause"

echo Starting Re-ID Feature Extractor...
start "Re-ID Worker" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate.bat && python workers\reid_worker.py && pause"

echo Starting Single Camera Trackers (Pose + Homography)...
start "Camera 1" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate.bat && python workers\sct_worker.py --cam_id cam1 --source videos\camera2_20260627_103326.mp4 --homography calibration\cam1_matrix.npy && pause"
start "Camera 2" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate.bat && python workers\sct_worker.py --cam_id cam2 --source videos\camera3_20260627_103325.mp4 --homography calibration\cam2_matrix.npy && pause"

echo Starting Streamlit Dashboard...
start "Dashboard" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate.bat && streamlit run dashboard.py"

echo Pipeline is running in separate windows. Close those windows to stop.
