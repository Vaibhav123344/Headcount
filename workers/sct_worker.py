import cv2
import numpy as np
from ultralytics import YOLO
import redis
import json
import base64
import argparse
import time
import torch
import os

try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    FeatureExtractor = None

# Fix for PyTorch 2.6+ blocking ultralytics models
original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)
torch.load = _safe_load

class SCTWorker:
    # ---------- Quality / EMA constants ----------
    MIN_CROP_HEIGHT   = 64      # pixels – skip tiny crops for ReID
    MIN_CROP_WIDTH    = 32
    MIN_CONF_FOR_REID = 0.45    # YOLO confidence gate for embedding extraction
    ANKLE_CONF_THRESH = 0.3     # minimum keypoint confidence to trust an ankle
    NORMAL_AR_LOW     = 1.5     # aspect-ratio below this → occluded
    EMA_ALPHA         = 0.15    # embedding exponential moving average weight
    VELOCITY_WINDOW   = 5       # number of frames for velocity estimation
    REID_INTERVAL     = 10      # extract embedding every N frames per track

    def __init__(self, cam_id, source, homography_path):
        self.cam_id = cam_id
        self.source = source
        self.H = np.load(homography_path)
        self.r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=None)
        self.queue_name = "warehouse:queue:matcher" # Inline ReID, send directly to matcher

        # ── Upgrade 1: Use YOLO-Pose for ankle keypoints ──
        # BoT-SORT only receives bounding boxes; pose data is extracted separately.
        self.model = YOLO(r"/home/yc12214/vaibhav/Headcount/yolo26l-pose.pt")
        
        self.global_id_cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if FeatureExtractor is not None:
            try:
                self.extractor = FeatureExtractor(
                    model_name='osnet_x1_0',
                    model_path='osnet_x1_0_msmt17.pth',
                    device=str(self.device)
                )
            except Exception as e:
                print(f"Failed to load torchreid extractor: {e}")
                self.extractor = None
        else:
            self.extractor = None
        
        self.track_history = {}       # local_id -> [(world_x, world_y), ...]
        self.track_frame_count = {}   # local_id -> int
        self.track_timestamps = {}    # local_id -> [timestamp, ...]   ── Upgrade 2
        self.track_embeddings = {}    # local_id -> np.array (512D EMA) ── Upgrade 4

        # Color palette for drawing different IDs
        np.random.seed(42)
        self.colors = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(50, 255, size=(200, 3))]

        # Check if we have a display (X11) available
        self.has_display = os.environ.get('DISPLAY') is not None

    def update_global_id_cache(self):
        """Fetch the entire map once per frame to avoid blocking network I/O."""
        try:
            raw_map = self.r.hgetall("global_id_map")
            self.global_id_cache = {
                k.decode('utf-8'): int(v.decode('utf-8')) 
                for k, v in raw_map.items()
            }
        except Exception:
            self.global_id_cache = {}

    def get_global_id(self, local_track_id):
        """Fast, non-blocking local dictionary lookup."""
        key = f"{self.cam_id}:{int(local_track_id)}"
        return self.global_id_cache.get(key, None)

    # ── Upgrade 1: Ankle keypoint extraction ──────────────────────────
    @staticmethod
    def _extract_ankle_point(keypoints_data, idx):
        """Return (foot_x, foot_y, True) from ankles, or (None, None, False)."""
        if keypoints_data is None:
            return None, None, False
        kpts = keypoints_data.data.cpu().numpy()
        if idx >= len(kpts):
            return None, None, False

        person_kpts = kpts[idx]           # shape (17, 3) → [x, y, confidence]
        left_ankle  = person_kpts[15]
        right_ankle = person_kpts[16]

        visible = []
        if left_ankle[2]  > SCTWorker.ANKLE_CONF_THRESH:
            visible.append(left_ankle[:2])
        if right_ankle[2] > SCTWorker.ANKLE_CONF_THRESH:
            visible.append(right_ankle[:2])

        if visible:
            avg = np.mean(visible, axis=0)
            return float(avg[0]), float(avg[1]), True
        return None, None, False

    # ── Upgrade 2: Velocity vector computation ────────────────────────
    def _compute_velocity(self, local_id):
        """Return (vx, vy) in meters/sec from the position+timestamp history."""
        history   = self.track_history.get(local_id, [])
        ts_hist   = self.track_timestamps.get(local_id, [])
        if len(history) < 2 or len(ts_hist) < 2:
            return 0.0, 0.0
        dt = ts_hist[-1] - ts_hist[0]
        if dt < 0.001:
            return 0.0, 0.0
        vx = (history[-1][0] - history[0][0]) / dt
        vy = (history[-1][1] - history[0][1]) / dt
        return float(vx), float(vy)

    def run(self):
        print(f"[{self.cam_id}] Starting SCT Worker (Pose + Ankle + EMA) on {self.source}")
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[{self.cam_id}] Error opening video source.")
            return

        window_name = f"Camera: {self.cam_id}"
        if self.has_display:
            try:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 960, 540)
            except cv2.error:
                print(f"[{self.cam_id}] No GUI available, running headless.")
                self.has_display = False

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{self.cam_id}] End of video stream.")
                break

            current_time = time.time()
            
            # Fetch the global-ID map once per frame
            self.update_global_id_cache()

            # BoT-SORT receives only bounding boxes (pose data is stripped by
            # Ultralytics internally — BoT-SORT uses IoU/motion, not keypoints).
            results = self.model.track(frame, persist=True, tracker="botsort.yaml",
                                         conf=0.35, iou=0.5, classes=[0], verbose=False)

            if results[0].boxes.id is not None:
                boxes     = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                confs     = results[0].boxes.conf.cpu().numpy()
                keypoints = results[0].keypoints  # may be None if model lacks kpt head

                for idx, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confs)):
                    x1, y1, x2, y2 = map(int, box)
                    local_id   = int(track_id)
                    confidence = float(conf)
                    
                    if local_id not in self.track_frame_count:
                        self.track_frame_count[local_id]  = 0
                        self.track_history[local_id]      = []
                        self.track_timestamps[local_id]   = []
                    self.track_frame_count[local_id] += 1
                    
                    # ── Upgrade 5: Occlusion detection via aspect ratio ──
                    frame_h, frame_w = frame.shape[:2]
                    bbox_h = y2 - y1
                    bbox_w = max(x2 - x1, 1)
                    aspect_ratio = bbox_h / bbox_w
                    occluded = aspect_ratio < self.NORMAL_AR_LOW

                    # Skip very close-up crops for ReID (person filling >70% of frame)
                    skip_reid = bbox_h / frame_h > 0.7

                    # ── Upgrade 1: Ankle-based foot localization ──
                    ankle_x, ankle_y, used_ankle = self._extract_ankle_point(keypoints, idx)
                    if used_ankle:
                        foot_x, foot_y = ankle_x, ankle_y
                    else:
                        # Fallback: bottom-center of bounding box
                        foot_x = (x1 + x2) / 2.0
                        foot_y = float(y2)

                    # perspectiveTransform expects shape (N, 1, 2)
                    bottom_center = np.array([[[foot_x, foot_y]]], dtype=np.float32)

                    # Transform to world coordinate using Floor Plan Homography matrix
                    world_coord = cv2.perspectiveTransform(bottom_center, self.H)
                    floor_plan_x = world_coord[0][0][0]
                    floor_plan_y = world_coord[0][0][1]

                    # Convert floor plan pixels to physical meters
                    SCALE_PIXELS_PER_METER = 100.0  # Adjust based on floor plan scale
                    raw_world_x = floor_plan_x / SCALE_PIXELS_PER_METER
                    raw_world_y = floor_plan_y / SCALE_PIXELS_PER_METER
                    
                    # Coordinate Smoothing (Moving Average over last N frames)
                    self.track_history[local_id].append((raw_world_x, raw_world_y))
                    self.track_timestamps[local_id].append(current_time)
                    if len(self.track_history[local_id]) > self.VELOCITY_WINDOW:
                        self.track_history[local_id].pop(0)
                        self.track_timestamps[local_id].pop(0)
                        
                    avg_world = np.mean(self.track_history[local_id], axis=0)
                    world_x, world_y = float(avg_world[0]), float(avg_world[1])

                    # ── Upgrade 2: Velocity vector ──
                    velocity_x, velocity_y = self._compute_velocity(local_id)

                    # ── Upgrade 4: Quality-gated EMA embedding extraction ──
                    meets_quality = (
                        bbox_h >= self.MIN_CROP_HEIGHT and
                        bbox_w >= self.MIN_CROP_WIDTH and
                        confidence >= self.MIN_CONF_FOR_REID and
                        not skip_reid
                    )

                    if meets_quality and self.track_frame_count[local_id] % self.REID_INTERVAL == 1:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0 and self.extractor is not None:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            raw_features = self.extractor([crop_rgb]).cpu().numpy()[0]
                            norm = np.linalg.norm(raw_features)
                            if norm > 0:
                                raw_features = raw_features / norm

                            # EMA update: smooth embedding over time
                            if local_id in self.track_embeddings:
                                ema = ((1.0 - self.EMA_ALPHA) * self.track_embeddings[local_id]
                                       + self.EMA_ALPHA * raw_features)
                                n = np.linalg.norm(ema)
                                if n > 0:
                                    ema = ema / n
                                self.track_embeddings[local_id] = ema
                            else:
                                self.track_embeddings[local_id] = raw_features

                            # Send the STABLE EMA embedding, not the noisy raw frame embedding
                            payload = {
                                "cam_id":           self.cam_id,
                                "local_track_id":   local_id,
                                "timestamp":        current_time,
                                "world_x":          float(world_x),
                                "world_y":          float(world_y),
                                "velocity_x":       float(velocity_x),
                                "velocity_y":       float(velocity_y),
                                "occluded":         occluded,
                                "bbox_aspect_ratio": float(aspect_ratio),
                                "embedding":        self.track_embeddings[local_id].tolist()
                            }
                            self.r.rpush(self.queue_name, json.dumps(payload))

                    # --- Draw bounding box and ID on frame ---
                    global_id = self.get_global_id(local_id)
                    if global_id is not None:
                        label = f"G:{global_id} L:{local_id} {confidence*100:.0f}%"
                        color = self.colors[global_id % len(self.colors)]
                    else:
                        label = f"L:{local_id} {confidence*100:.0f}%"
                        color = (150, 150, 150)

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label background
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Draw foot point (ankle or fallback)
                    foot_color = (0, 255, 0) if used_ankle else (0, 165, 255)  # green=ankle, orange=fallback
                    cv2.circle(frame, (int(foot_x), int(foot_y)), 5, foot_color, -1)

                    # Draw occlusion indicator
                    if occluded:
                        cv2.putText(frame, "OCC", (x2 + 5, y1 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw header bar with camera info and track count
            h, w = frame.shape[:2]
            track_count = len(track_ids) if results[0].boxes.id is not None else 0
            header = f"{self.cam_id}   tracks: {track_count}"
            cv2.rectangle(frame, (0, 0), (w, 32), (30, 30, 30), -1)
            cv2.putText(frame, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Push annotated frame to Redis for Streamlit dashboard
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            self.r.set(f"frame:{self.cam_id}", jpeg.tobytes())

            # Show the frame only if display is available
            if self.has_display:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"[{self.cam_id}] Quit signal received.")
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[{self.cam_id}] Processed {frame_count} frames, {track_count} tracks")

        cap.release()
        if self.has_display:
            cv2.destroyAllWindows()
        print(f"[{self.cam_id}] SCT Worker finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--homography", required=True)
    args = parser.parse_args()

    worker = SCTWorker(args.cam_id, args.source, args.homography)
    worker.run()
