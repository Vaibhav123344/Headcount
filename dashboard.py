import streamlit as st
import redis
import json
import time
import cv2
import numpy as np

st.set_page_config(page_title="Headcount Dashboard", layout="wide")

@st.cache_resource
def load_config():
    with open("config.json") as f:
        return json.load(f)

@st.cache_resource
def get_redis(host, port):
    return redis.Redis(host=host, port=port, db=0)

cfg = load_config()
r = get_redis(cfg["redis"]["host"], cfg["redis"]["port"])

st.title("👥 Headcount - Live Dashboard")
st.caption("Pulling live tracking data and video frames directly from Redis")

CAMS = list(cfg.get("cameras", {}).keys())

st.subheader("📹 Live Camera Previews")
cols = st.columns(len(CAMS))
img_placeholders = {cam_id: cols[i].empty() for i, cam_id in enumerate(CAMS)}

st.subheader("🗺️ BEV Layout & Zones")
bev_col, zone_col = st.columns([2, 1])

with bev_col:
    bev_placeholder = st.empty()
    
with zone_col:
    zone_placeholder = st.empty()

st.subheader("📊 Global Tracking Gallery")
stats_placeholder = st.empty()

# Load map image
map_path = cfg.get("layout", {}).get("map_path", "map.png")
try:
    map_bg_raw = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
    if map_bg_raw is None:
        st.error(f"Failed to load layout map: {map_path}")
        map_bg = np.zeros((800, 800, 3), dtype=np.uint8)
    else:
        if len(map_bg_raw.shape) == 3 and map_bg_raw.shape[2] == 4:
            alpha = map_bg_raw[:, :, 3] / 255.0
            white_bg = np.ones_like(map_bg_raw[:, :, :3]) * 255
            map_bg = np.empty_like(map_bg_raw[:, :, :3])
            for i in range(3):
                map_bg[:, :, i] = map_bg_raw[:, :, i] * alpha + white_bg[:, :, i] * (1 - alpha)
            map_bg = map_bg.astype(np.uint8)
        else:
            map_bg = map_bg_raw
except Exception as e:
    st.error(f"Error loading map: {e}")
    map_bg = np.zeros((800, 800, 3), dtype=np.uint8)

np.random.seed(42)
colors = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(50, 255, size=(200, 3))]

# EMA smoothing for BEV dot positions
smoothed_positions = {}  # gid -> (x, y)
EMA_ALPHA = 0.3  # Lower = smoother but more lag
trails = {}  # gid -> list of recent (x, y) positions

while True:
    for cam_id in CAMS:
        frame_bytes = r.get(f"frame:{cam_id}")
        if frame_bytes:
            img_placeholders[cam_id].image(frame_bytes, caption=cam_id, use_container_width=True)
        else:
            img_placeholders[cam_id].info(f"Waiting for {cam_id} stream...")

    state_bytes = r.get("state:gallery")
    
    # Process BEV Map
    disp_map = map_bg.copy()
    
    with stats_placeholder.container():
        if state_bytes:
            state = json.loads(state_bytes.decode('utf-8'))
            
            # Draw dots on map with smoothing
            active_gids = set()
            for entry in state.get("entries", []):
                raw_x, raw_y = entry["last_x"], entry["last_y"]
                gid = entry["global_id"]
                active_gids.add(gid)
                color = colors[gid % len(colors)]
                
                # Apply EMA smoothing
                if gid in smoothed_positions:
                    old_x, old_y = smoothed_positions[gid]
                    sx = old_x * (1 - EMA_ALPHA) + raw_x * EMA_ALPHA
                    sy = old_y * (1 - EMA_ALPHA) + raw_y * EMA_ALPHA
                else:
                    sx, sy = raw_x, raw_y
                smoothed_positions[gid] = (sx, sy)
                
                # Update trail
                if gid not in trails:
                    trails[gid] = []
                trails[gid].append((int(sx), int(sy)))
                if len(trails[gid]) > 20:
                    trails[gid].pop(0)
                
                # Draw trail line
                pts = trails[gid]
                for k in range(1, len(pts)):
                    alpha_fade = k / len(pts)
                    thick = max(1, int(alpha_fade * 3))
                    cv2.line(disp_map, pts[k-1], pts[k], color, thick)
                
                x, y = int(sx), int(sy)
                cv2.circle(disp_map, (x, y), 12, color, -1)
                cv2.circle(disp_map, (x, y), 12, (0,0,0), 2)
                cv2.putText(disp_map, f"ID:{gid}", (x - 20, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
                cv2.putText(disp_map, f"ID:{gid}", (x - 20, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Clean up stale smoothed positions and trails
            for stale_gid in list(smoothed_positions.keys()):
                if stale_gid not in active_gids:
                    del smoothed_positions[stale_gid]
                    trails.pop(stale_gid, None)
                
            bev_placeholder.image(cv2.cvtColor(disp_map, cv2.COLOR_BGR2RGB), caption="Live BEV Tracker", use_container_width=True)
            
            # Draw zone stats
            with zone_placeholder.container():
                st.write("**Zone Occupancy**")
                zone_counts = state.get("zone_counts", {})
                for z, count in zone_counts.items():
                    if z != "Unknown" or count > 0:
                        st.metric(z, count)
                
            # Table stats
            c1, c2 = st.columns(2)
            c1.metric("Active People in Gallery", state.get("unique_people", 0))
            c2.metric("Next ID to Assign", state.get("next_id", 1))
            
            if state.get("entries"):
                st.dataframe(state["entries"], use_container_width=True)
            else:
                st.info("No active tracks.")
        else:
            st.info("Waiting for Matcher data...")
            bev_placeholder.image(cv2.cvtColor(disp_map, cv2.COLOR_BGR2RGB), caption="Live BEV Tracker", use_container_width=True)

    time.sleep(0.5)