import redis
import msgpack
import numpy as np
import cv2
import time
import json
import sys

def load_config():
    with open("config.json") as f:
        return json.load(f)

cfg = load_config()
host = cfg.get("redis", {}).get("host", "localhost")
port = cfg.get("redis", {}).get("port", 6379)
cams = list(cfg.get("cameras", {}).keys())

if len(cams) < 2:
    print("Error: Need at least 2 cameras in config.json to run the tuner.")
    sys.exit(1)

cam_a = cams[0]
cam_b = cams[1]

# --- Setup Redis ---
r = redis.Redis(host=host, port=port, db=0)
queue_name = "warehouse:queue:matcher"

print("Listening to Redis stream for ReID tuning...")
print("Make sure SCT workers & ReID workers are running, and Global Matcher is STOPPED.")

threshold = cfg.get("matcher", {}).get("reid_sim_threshold", 0.60)
win = "Redis ReID Tuner Matrix"
cv2.namedWindow(win)
cv2.createTrackbar("Threshold x100", win, int(threshold * 100), 95, lambda v: None)

# IMPROVEMENT: Track timestamps to evict stale IDs from the UI
cam_data = {cam_a: {}, cam_b: {}} 
last_seen = {cam_a: {}, cam_b: {}} 
TRACK_TTL = 10.0 # Remove IDs from matrix if not seen in 10 seconds

while True:
    tb = cv2.getTrackbarPos("Threshold x100", win) / 100.0
    threshold = max(0.05, tb)
    
    # IMPROVEMENT: Use blpop to prevent 100% CPU core lockup
    res = r.blpop(queue_name, timeout=1)
    
    if res:
        data = msgpack.unpackb(res[1], strict_map_key=False)
        cam = data['cam_id']
        local_id = data['local_track_id']
        
        # Ensure we only process data if the embedding actually exists
        emb_key = b"embedding" if b"embedding" in data else "embedding"
        if emb_key in data:
            emb = np.array(data[emb_key], dtype=np.float32)
            if np.linalg.norm(emb) > 0:
                emb = emb / np.linalg.norm(emb)
            
            if cam in cam_data:
                cam_data[cam][local_id] = emb
                last_seen[cam][local_id] = time.time()

    # IMPROVEMENT: Evict stale tracks so the matrix only shows live people
    current_time = time.time()
    for cam in [cam_a, cam_b]:
        stale_ids = [lid for lid, ts in last_seen[cam].items() if current_time - ts > TRACK_TTL]
        for lid in stale_ids:
            del cam_data[cam][lid]
            del last_seen[cam][lid]

    # Sort keys by most recently updated to keep the UI relevant
    c0_keys = sorted(cam_data[cam_a].keys(), key=lambda k: last_seen[cam_a][k], reverse=True)
    c1_keys = sorted(cam_data[cam_b].keys(), key=lambda k: last_seen[cam_b][k], reverse=True)
    
    if not c0_keys or not c1_keys:
        # Blank waiting screen if no overlapping people are currently live
        panel = np.full((400, 600, 3), 30, np.uint8)
        cv2.putText(panel, "Waiting for live overlapping detections...", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        cv2.imshow(win, panel)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        continue

    # Draw matrix
    panel = np.full((400, 600, 3), 30, np.uint8)
    cv2.putText(panel, f"Similarity Threshold: {threshold:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    x0, y0, cw, ch = 120, 80, 80, 40
    
    # Headers
    for j, id1 in enumerate(c1_keys[:5]): 
        cv2.putText(panel, f"C1:L{id1}", (x0 + j * cw, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)
        
    for i, id0 in enumerate(c0_keys[:5]):
        cv2.putText(panel, f"C0:L{id0}", (10, y0 + i * ch + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
        
        for j, id1 in enumerate(c1_keys[:5]):
            sim = float(np.dot(cam_data[cam_a][id0], cam_data[cam_b][id1]))
            color = (0, 200, 0) if sim >= threshold else (40, 40, 90)
            
            px, py = x0 + j * cw, y0 + i * ch
            cv2.rectangle(panel, (px, py), (px + cw - 4, py + ch - 4), color, -1)
            cv2.putText(panel, f"{sim:.2f}", (px + 10, py + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(win, panel)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"Final Tuned Threshold: {threshold:.2f}")