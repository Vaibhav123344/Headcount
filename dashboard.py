import streamlit as st
import redis
import json
import time

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

# Dynamically load camera names from config.json
CAMS = list(cfg["cameras"].keys())

st.subheader("📹 Live Camera Previews")
cols = st.columns(len(CAMS))
img_placeholders = {cam_id: cols[i].empty() for i, cam_id in enumerate(CAMS)}

st.subheader("📊 Global Tracking Gallery")
stats_placeholder = st.empty()

while True:
    for cam_id in CAMS:
        frame_bytes = r.get(f"frame:{cam_id}")
        if frame_bytes:
            img_placeholders[cam_id].image(frame_bytes, caption=cam_id, use_container_width=True)
        else:
            img_placeholders[cam_id].info(f"Waiting for {cam_id} stream...")

    state_bytes = r.get("state:gallery")
    with stats_placeholder.container():
        if state_bytes:
            state = json.loads(state_bytes.decode('utf-8'))
            c1, c2 = st.columns(2)
            c1.metric("Active People in Gallery", state.get("unique_people", 0))
            c2.metric("Next ID to Assign", state.get("next_id", 1))
            
            if state.get("entries"):
                st.dataframe(state["entries"], use_container_width=True)
            else:
                st.info("No active tracks.")
        else:
            st.info("Waiting for Matcher data...")

    time.sleep(0.5)