import cv2
import numpy as np
from ultralytics import YOLO
import redis
import json
import base64
import argparse
import time

class SCTWorker:
    def __init__(self, cam_id, source, homography_path):
        self.cam_id = cam_id
        self.source = source
        self.H = np.load(homography_path)
        self.r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=None)
        self.queue_name = "warehouse:queue:reid"
        # Use the local yolo26l-pose model
        self.model = YOLO('yolo26l-pose.pt')

        # Color palette for drawing different IDs
        np.random.seed(42)
        self.colors = [(int(c[0]), int(c[1]), int(c[2])) for c in np.random.randint(50, 255, size=(200, 3))]

    def get_global_id(self, local_track_id):
        """Look up the global ID from Redis for this camera's local track."""
        gid = self.r.hget("global_id_map", f"{self.cam_id}:{int(local_track_id)}")
        if gid is not None:
            return int(gid.decode('utf-8'))
        return None

    def run(self):
        print(f"[{self.cam_id}] Starting SCT Worker on {self.source}")
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[{self.cam_id}] Error opening video source.")
            return

        window_name = f"Camera: {self.cam_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[{self.cam_id}] End of video stream.")
                break

            current_time = time.time()
            results = self.model.track(frame, persist=True, tracker="botsort.yaml",
                                         conf=0.35, iou=0.5, classes=[0], verbose=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

                for idx, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confs)):
                    x1, y1, x2, y2 = map(int, box)
                    local_id = int(track_id)
                    confidence = float(conf)

                    # Determine foot coordinates using ankles (keypoints 15 and 16) if available
                    foot_x, foot_y = None, None
                    if keypoints is not None:
                        kpts = keypoints[idx]
                        left_ankle = kpts[15]
                        right_ankle = kpts[16]
                        
                        valid_ankles = []
                        if left_ankle[0] != 0 and left_ankle[1] != 0:
                            valid_ankles.append(left_ankle)
                        if right_ankle[0] != 0 and right_ankle[1] != 0:
                            valid_ankles.append(right_ankle)
                            
                        if len(valid_ankles) > 0:
                            avg_kpt = np.mean(valid_ankles, axis=0)
                            foot_x, foot_y = float(avg_kpt[0]), float(avg_kpt[1])
                            
                    # Fallback to bounding box bottom center if ankles aren't visible
                    if foot_x is None or foot_y is None:
                        foot_x = (x1 + x2) / 2.0
                        foot_y = float(y2)

                    # perspectiveTransform expects shape (N, 1, 2)
                    bottom_center = np.array([[[foot_x, foot_y]]], dtype=np.float32)

                    # Transform to world coordinate using Homography matrix
                    world_coord = cv2.perspectiveTransform(bottom_center, self.H)
                    world_x = world_coord[0][0][0]
                    world_y = world_coord[0][0][1]

                    # Crop image for Re-ID
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # Encode to jpg base64
                    _, buffer = cv2.imencode('.jpg', crop)
                    crop_b64 = base64.b64encode(buffer).decode('utf-8')

                    # Payload for Redis
                    payload = {
                        "cam_id": self.cam_id,
                        "local_track_id": local_id,
                        "timestamp": current_time,
                        "world_x": float(world_x),
                        "world_y": float(world_y),
                        "image_crop": crop_b64
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

                    # Draw foot point
                    cv2.circle(frame, (int(foot_x), int(foot_y)), 5, (0, 255, 0), -1)

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

            # Show the frame
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"[{self.cam_id}] Quit signal received.")
                break

        cap.release()
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
