import cv2
import numpy as np
from ultralytics import YOLO
import redis
import msgpack # ── Upgrade 4: Binary Serialization
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

# ── Upgrade 3: World-Space Kalman Filter (No more DBSCAN jitter) ──
class Kalman2D:
    def __init__(self, dt=0.033):
        self.dt = dt
        self.x = np.zeros((4, 1))  # State: [x, y, vx, vy]
        self.P = np.eye(4) * 1000  # Uncertainty covariance
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]) # Transition matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]]) # Measurement matrix
        self.R = np.eye(2) * 0.1   # Measurement noise (Pose Jitter)
        self.Q = np.eye(4) * 0.01  # Process noise (Movement unpredictability)
        self.initialized = False

    def update(self, meas_x, meas_y):
        z = np.array([[meas_x], [meas_y]])
        if not self.initialized:
            self.x = np.array([[meas_x], [meas_y], [0], [0]])
            self.initialized = True
            return meas_x, meas_y, 0, 0

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Update
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return float(self.x[0][0]), float(self.x[1][0]), float(self.x[2][0]), float(self.x[3][0])

class SCTWorker:
    MIN_CROP_HEIGHT   = 64
    MIN_CROP_WIDTH    = 32
    MIN_CONF_FOR_REID = 0.45
    ANKLE_CONF_THRESH = 0.3
    NORMAL_AR_LOW     = 1.5
    REID_INTERVAL     = 10

    def __init__(self, cam_id, source, homography_path):
        self.cam_id = cam_id
        self.source = source
        self.H = np.load(homography_path)
        self.r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=None)
        self.queue_name = "warehouse:queue:matcher"

        self.model = YOLO(r"/home/yc12214/vaibhav/Headcount/yolo26l-pose.pt")
        self.global_id_cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if FeatureExtractor is not None:
            self.extractor = FeatureExtractor(
                model_name='osnet_x1_0', model_path='osnet_x1_0_msmt17.pth', device=str(self.device)
            )
        
        self.track_frame_count = {}   
        self.kalman_filters = {}     # local_id -> Kalman2D

        np.random.seed(42)
        self.colors = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(50, 255, size=(200, 3))]
        self.has_display = os.environ.get('DISPLAY') is not None

    def update_global_id_cache(self):
        try:
            raw_map = self.r.hgetall("global_id_map")
            self.global_id_cache = {k.decode('utf-8'): int(v.decode('utf-8')) for k, v in raw_map.items()}
        except Exception:
            self.global_id_cache = {}

    def get_global_id(self, local_track_id):
        return self.global_id_cache.get(f"{self.cam_id}:{int(local_track_id)}", None)

    @staticmethod
    def _extract_foot_point(keypoints_data, idx, box):
        x1, y1, x2, y2 = box
        fallback_x, fallback_y = (x1 + x2) / 2.0, float(y2)
        if keypoints_data is None: return fallback_x, fallback_y, 'low'
        
        kpts = keypoints_data.data.cpu().numpy()
        if idx >= len(kpts): return fallback_x, fallback_y, 'low'
        person_kpts = kpts[idx]
        
        # Ankles
        visible_ankles = [kp[:2] for kp in person_kpts[15:17] if kp[2] > SCTWorker.ANKLE_CONF_THRESH]
        if visible_ankles: return float(np.mean(visible_ankles, axis=0)[0]), float(np.mean(visible_ankles, axis=0)[1]), 'high'
        # Knees
        visible_knees = [kp[:2] for kp in person_kpts[13:15] if kp[2] > SCTWorker.ANKLE_CONF_THRESH]
        if visible_knees: return float(np.mean(visible_knees, axis=0)[0]), float(np.mean(visible_knees, axis=0)[1]), 'high'
        
        return fallback_x, fallback_y, 'low'

    def run(self):
        print(f"[{self.cam_id}] Starting SCT Worker...")
        cap = cv2.VideoCapture(self.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_frame_time = 1.0 / fps

        window_name = f"Camera: {self.cam_id}"
        if self.has_display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret: break

            current_time = time.time()  # High accuracy Sync Timestamp
            self.update_global_id_cache()

            results = self.model.track(frame, persist=True, tracker="botsort.yaml", conf=0.35, iou=0.5, classes=[0], verbose=False)

            if results[0].boxes.id is not None:
                boxes, track_ids, confs = results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.cpu().numpy(), results[0].boxes.conf.cpu().numpy()
                keypoints = results[0].keypoints

                for idx, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confs)):
                    x1, y1, x2, y2 = map(int, box)
                    local_id, confidence = int(track_id), float(conf)
                    
                    if local_id not in self.kalman_filters:
                        self.kalman_filters[local_id] = Kalman2D(dt=target_frame_time)
                        self.track_frame_count[local_id] = 0
                    self.track_frame_count[local_id] += 1
                    
                    occluded = ((y2 - y1) / max(x2 - x1, 1)) < self.NORMAL_AR_LOW
                    skip_reid = (y2 - y1) / frame.shape[0] > 0.7

                    foot_x, foot_y, kp_quality = self._extract_foot_point(keypoints, idx, (x1, y1, x2, y2))
                    world_coord = cv2.perspectiveTransform(np.array([[[foot_x, foot_y]]], dtype=np.float32), self.H)
                    raw_x, raw_y = world_coord[0][0][0] / 100.0, world_coord[0][0][1] / 100.0
                    
                    # Apply Kalman Filter to get smooth coordinates and instant velocity
                    world_x, world_y, vel_x, vel_y = self.kalman_filters[local_id].update(raw_x, raw_y)

                    if not skip_reid and confidence >= self.MIN_CONF_FOR_REID and self.track_frame_count[local_id] % self.REID_INTERVAL == 1:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            raw_features = self.extractor([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)]).cpu().numpy()[0]
                            if np.linalg.norm(raw_features) > 0:
                                raw_features = raw_features / np.linalg.norm(raw_features)

                            payload = {
                                "cam_id": self.cam_id,
                                "local_track_id": local_id,
                                "timestamp": current_time,
                                "world_x": world_x,
                                "world_y": world_y,
                                "velocity_x": vel_x,
                                "velocity_y": vel_y,
                                "occluded": occluded,
                                "keypoint_quality": kp_quality,
                                "embedding": raw_features.tolist()
                            }
                            # ── Upgrade 4: MessagePack pushes binary at high speed ──
                            self.r.rpush(self.queue_name, msgpack.packb(payload))

                    # Drawing logic
                    gid = self.get_global_id(local_id)
                    color = self.colors[gid % len(self.colors)] if gid else (150, 150, 150)
                    label = f"G:{gid} L:{local_id}" if gid else f"L:{local_id}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            self.r.set(f"frame:{self.cam_id}", jpeg.tobytes())

            if self.has_display:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            elapsed = time.time() - loop_start
            if elapsed < target_frame_time: time.sleep(target_frame_time - elapsed)
            frame_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam_id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--homography", required=True)
    args = parser.parse_args()
    SCTWorker(args.cam_id, args.source, args.homography).run()











