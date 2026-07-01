#!/bin/bash
# Bash script to start the entire pipeline

echo "Starting Database Initialization..."
python workers/init_db.py

if [ "$1" == "--tune" ]; then
    echo "Running in TUNING MODE (Global Matcher disabled)."
else
    echo "Starting Global Matcher..."
    python workers/global_matcher.py &
fi

echo "Starting Single Camera Trackers based on config.json..."
python -c '
import json, os
cfg = json.load(open("config.json"))
for cid, cinfo in cfg.get("cameras", {}).items():
    vpath = cinfo["video_path"]
    mpath = cinfo["matrix_path"]
    cmd = f"python workers/sct_worker.py --cam_id {cid} --source {vpath} --homography {mpath} &"
    print("Launching:", cmd)
    os.system(cmd)
'

echo "Starting Streamlit Dashboard..."
python -m streamlit run dashboard.py --server.headless true &

echo "Pipeline is running. Press Ctrl+C to stop."
wait
