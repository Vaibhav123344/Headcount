import cv2
import numpy as np
from ultralytics import YOLO
import redis
import msgpack
import argparse
import time
import torch
import os

# Fix for PyTorch 2.6+ blocking ultralytics models
original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)
torch.load = _safe_load

class Kalman2D:
    def __init__(self, dt=0.033):
        self.dt = dt
        self.x = np.zeros((4, 1))  
        self.P = np.eye(4) * 1000  
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]) 
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]]) 
        self.R = np.eye(2) * 5.0    # High: homography projection is noisy
        self.Q = np.eye(4) * 0.005  # Low: people move slowly on BEV
        self.initialized = False

    def update(self, meas_x, meas_y, meas_noise=None):
        z = np.array([[meas_x], [meas_y]])
        if not self.initialized:
            self.x = np.array([[meas_x], [meas_y], [0], [0]])
            self.initialized = True
            return meas_x, meas_y, 0, 0

        # Dynamic measurement noise: trust clean ankle points, distrust fallbacks.
        R = self.R if meas_noise is None else np.eye(2) * meas_noise

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return float(self.x[0][0]), float(self.x[1][0]), float(self.x[2][0]), float(self.x[3][0])

class SCTWorker:
    MIN_CROP_HEIGHT   = 64
    MIN_CROP_WIDTH    = 32
    MIN_CONF_FOR_REID = 0.50 # Increased strictness to avoid blurry ReID poisoning
    ANKLE_CONF_THRESH = 0.3
    NORMAL_AR_LOW     = 1.5
    REID_INTERVAL     = 10
    REID_REFRESH_INTERVAL = 60  # Re-ReID already-matched tracks to keep feature bank fresh
    # Kalman measurement noise per foot-point quality (higher = trust less)
    NOISE_BY_QUALITY  = {'ankle': 5.0, 'knee': 8.0, 'bbox': 20.0}

    def __init__(self, cam_id, source, homography_path):
        self.cam_id = cam_id
        self.source = source
        self.H = np.load(homography_path)
        self.r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=None)
        
        # We will use two queues: fast path (matcher) and slow path (reid)
        # self.out_queue is removed

        # Bound the slow-path ReID queue so crop bursts can't flood Redis / go stale.
        try:
            import json as _json
            self.reid_queue_max = _json.load(open("config.json")).get("matcher", {}).get("reid_queue_max", 300)
        except Exception:
            self.reid_queue_max = 300

        self.model = YOLO(r"/home/yc12214/vaibhav/Headcount/yolo26l-pose.pt")
        self.global_id_cache = {}
        
        self.track_frame_count = {}   
        self.kalman_filters = {}     

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
        if keypoints_data is None: return fallback_x, fallback_y, 'bbox'

        kpts = keypoints_data.data.cpu().numpy()
        if idx >= len(kpts): return fallback_x, fallback_y, 'bbox'
        person_kpts = kpts[idx]

        visible_ankles = [kp[:2] for kp in person_kpts[15:17] if kp[2] > SCTWorker.ANKLE_CONF_THRESH]
        if visible_ankles: return float(np.mean(visible_ankles, axis=0)[0]), float(np.mean(visible_ankles, axis=0)[1]), 'ankle'
        visible_knees = [kp[:2] for kp in person_kpts[13:15] if kp[2] > SCTWorker.ANKLE_CONF_THRESH]
        if visible_knees: return float(np.mean(visible_knees, axis=0)[0]), float(np.mean(visible_knees, axis=0)[1]), 'knee'

        return fallback_x, fallback_y, 'bbox'

    def run(self):
        print(f"[{self.cam_id}] Starting Pure-Spatial SCT Worker...")
        cap = cv2.VideoCapture(self.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_frame_time = 1.0 / fps

        window_name = f"Camera: {self.cam_id}"
        if self.has_display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret: break

            current_time = time.time() 
            self.update_global_id_cache()

            results = self.model.track(frame, persist=True, tracker="botsort.yaml", conf=0.3, iou=0.5, classes=[0], imgsz=1024 ,verbose=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                keypoints = results[0].keypoints

                for idx, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confs)):
                    local_id, confidence = int(track_id), float(conf)
                    
                    # Ensure bounding box is clamped within frame dimensions
                    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                    x2, y2 = min(frame_width, int(box[2])), min(frame_height, int(box[3]))
                    
                    if local_id not in self.kalman_filters:
                        self.kalman_filters[local_id] = Kalman2D(dt=target_frame_time)
                        self.track_frame_count[local_id] = 0
                    self.track_frame_count[local_id] += 1
                    
                    foot_x, foot_y, kp_quality = self._extract_foot_point(keypoints, idx, (x1, y1, x2, y2))
                    world_coord = cv2.perspectiveTransform(np.array([[[foot_x, foot_y]]], dtype=np.float32), self.H)
                    raw_x, raw_y = float(world_coord[0][0][0]), float(world_coord[0][0][1]) # Layout pixels directly

                    meas_noise = self.NOISE_BY_QUALITY.get(kp_quality, 5.0)
                    world_x, world_y, vel_x, vel_y = self.kalman_filters[local_id].update(raw_x, raw_y, meas_noise)

                    payload = {
                        "cam_id": self.cam_id,
                        "local_track_id": local_id,
                        "timestamp": current_time,
                        "world_x": world_x,
                        "world_y": world_y,
                        "velocity_x": vel_x,
                        "velocity_y": vel_y,
                    }

                    # FAST PATH: Push position-only updates directly to matcher every frame
                    self.r.rpush("warehouse:queue:matcher", msgpack.packb(payload))

                    # SLOW PATH (ReID): extract crops for UNMATCHED tracks (fast), plus
                    # a slow periodic refresh for MATCHED tracks so feature banks don't go stale.
                    is_unmatched = self.get_global_id(local_id) is None

                    h, w = y2 - y1, x2 - x1
                    good_ar = h / max(w, 1) >= self.NORMAL_AR_LOW
                    good_size = h >= self.MIN_CROP_HEIGHT and w >= self.MIN_CROP_WIDTH
                    is_full_body = h / float(frame_height) <= 0.7
                    good_crop = good_ar and good_size and is_full_body and confidence >= self.MIN_CONF_FOR_REID

                    fcount = self.track_frame_count[local_id]
                    if is_unmatched:
                        due = fcount % self.REID_INTERVAL == 1
                    else:
                        due = fcount % self.REID_REFRESH_INTERVAL == 1
                    needs_reid = good_crop and due

                    if needs_reid:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            _, img_encoded = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                            reid_payload = payload.copy()
                            reid_payload["image_bytes"] = img_encoded.tobytes()
                            # Push then cap: keep only the newest N crops so a burst of
                            # new tracks can't back up the ViT with stale images.
                            pipe = self.r.pipeline(transaction=False)
                            pipe.rpush("warehouse:queue:reid", msgpack.packb(reid_payload))
                            pipe.ltrim("warehouse:queue:reid", -self.reid_queue_max, -1)
                            pipe.execute()

                    if self.has_display:
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