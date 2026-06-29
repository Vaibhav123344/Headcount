import streamlit as st
import redis
import json
import time

st.set_page_config(page_title="Headcount Dashboard", layout="wide")

@st.cache_resource
def get_redis():
    return redis.Redis(host='localhost', port=6379, db=0)

r = get_redis()

st.title("👥 Headcount - Live Dashboard")
st.caption("Pulling live tracking data and video frames directly from Redis")

CAMS = ["cam1", "cam2"]

st.subheader("📹 Live Camera Previews")
cols = st.columns(len(CAMS))
img_placeholders = {cam_id: cols[i].empty() for i, cam_id in enumerate(CAMS)}

st.subheader("📊 Global Tracking Gallery")
stats_placeholder = st.empty()

# Infinite loop to update placeholders with live data from Redis
while True:
    # 1. Update Video Frames
    for cam_id in CAMS:
        frame_bytes = r.get(f"frame:{cam_id}")
        if frame_bytes:
            # frame_bytes is a JPEG encoded string from OpenCV
            img_placeholders[cam_id].image(frame_bytes, caption=cam_id, width="stretch")
        else:
            img_placeholders[cam_id].info(f"Waiting for {cam_id} stream...")

    # 2. Update Gallery Stats
    state_bytes = r.get("state:gallery")
    with stats_placeholder.container():
        if state_bytes:
            state = json.loads(state_bytes.decode('utf-8'))
            c1, c2 = st.columns(2)
            c1.metric("Active People in Gallery", state.get("unique_people", 0))
            c2.metric("Next ID to Assign", state.get("next_id", 1))
            
            if state.get("entries"):
                st.dataframe(state["entries"], width="stretch")
            else:
                st.info("No active tracks.")
        else:
            st.info("Waiting for Matcher data...")

    # Sleep briefly to control refresh rate
    time.sleep(0.5)
