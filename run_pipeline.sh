#!/bin/bash
# Bash script to start the entire pipeline

# --- CLEANUP: Kill any leftover workers from previous crashed runs ---
echo "Cleaning up leftover processes from previous runs..."
pkill -f "workers/sct_worker.py" 2>/dev/null
pkill -f "workers/reid_worker.py" 2>/dev/null
pkill -f "workers/global_matcher.py" 2>/dev/null
pkill -f "streamlit run dashboard.py" 2>/dev/null
sleep 1  # Give them time to die

echo "Starting Database Initialization..."
python workers/init_db.py

if [ "$1" == "--tune" ]; then
    echo "Running in TUNING MODE (Global Matcher disabled)."
else
    echo "Starting Global Matcher..."
    python workers/global_matcher.py &
fi

echo "Starting ReID Middleware Worker (ViT/timm)..."
echo "  (Waiting for model to load before starting trackers...)"
python workers/reid_worker.py &

# Wait for ReID worker to signal readiness (it sets a Redis key after model load)
MAX_WAIT=120  # seconds (model download can take a while on first run)
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    READY=$(python -c "import redis; r=redis.Redis(); print(r.get('reid_worker:ready') or b'')" 2>/dev/null)
    if [ "$READY" = "b'1'" ]; then
        echo "  ✓ ReID Worker is READY."
        break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
    # Show progress every 10 seconds
    if [ $((WAITED % 10)) -eq 0 ]; then
        echo "  ... still waiting for ReID model ($WAITED/${MAX_WAIT}s)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "  ⚠ WARNING: Timed out waiting for ReID worker. Starting trackers anyway."
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

# Trap Ctrl+C to kill all child processes cleanly
trap "echo '  Stopping all workers...'; pkill -P $$; pkill -f 'workers/sct_worker.py'; pkill -f 'workers/reid_worker.py'; pkill -f 'workers/global_matcher.py'; pkill -f 'streamlit run dashboard.py'; exit 0" SIGINT SIGTERM

wait
