"""Interactive Multi-Person Track-based cross-camera calibration for the Headcount pipeline.

Instead of manually clicking floor corners, this tool builds the cam1→cam2
ground-plane homography from the FEET of multiple paired people as they walk.

NEW FEATURES:
  - MAGSAC++ Solver: Vastly superior to RANSAC for noisy foot tracking.
  - EMA Smoothing: Raw YOLO keypoints are smoothed before calibration.
  - Video Sync Controls: Advance frames independently to fix desynced videos.
  - Multi-person support: Track up to 3-4 people at the same time!
  - DYNAMIC RE-PAIRING: Pairs persist even when tracks are lost. 

Controls:
    click cam1 then cam2   pair the same person across views
    SPACE                  play / pause
    1                      advance ONLY Cam 1 by 1 frame (use while paused to sync)
    2                      advance ONLY Cam 2 by 1 frame (use while paused to sync)
    x                      clear active pairings (keep collected points)
    u                      undo the last collected point
    f                      fit homography from collected points (USAC_MAGSAC)
    s                      save homographies to calibration/
    r                      reset all collected points and start over
    q                      quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

# Ensure project root is on sys.path for botsort.yaml discovery
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

from ultralytics import YOLO

# Fix for PyTorch 2.6+ blocking ultralytics models
original_load = torch.load
def _safe_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)
torch.load = _safe_load

# ─── Layout constants ────────────────────────────────────────────────
PANEL_W, PANEL_H = 800, 450
MIN_SPREAD_PX    = 15       # Increased slightly to prevent micro-clustering
MIN_POINTS       = 8        
GOOD_POINTS      = 35       
ANKLE_CONF_THRESH = 0.35    
EMA_ALPHA        = 0.6      # Smoothing factor for foot jitter

def foot_of_bbox(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2.0, float(y2)], dtype=np.float64)

def foot_from_ankles(keypoints_data, idx):
    if keypoints_data is None:
        return None, False
    kpts = keypoints_data.data.cpu().numpy()
    if idx >= len(kpts):
        return None, False

    person_kpts = kpts[idx]           
    left_ankle  = person_kpts[15]
    right_ankle = person_kpts[16]

    visible = []
    if left_ankle[2] > ANKLE_CONF_THRESH:
        visible.append(left_ankle[:2])
    if right_ankle[2] > ANKLE_CONF_THRESH:
        visible.append(right_ankle[:2])

    if visible:
        avg = np.mean(visible, axis=0)
        return np.array([float(avg[0]), float(avg[1])], dtype=np.float64), True
    return None, False

def find_pair_by_tid(pairs, ci, tid):
    for idx, pair in enumerate(pairs):
        if pair["cam_tids"][ci] == tid:
            return idx, pair
    return None, None

def main():
    ap = argparse.ArgumentParser(description="Interactive Multi-Person Track Calibration")
    ap.add_argument("--video1", required=True, help="Path to cam1 video")
    ap.add_argument("--video2", required=True, help="Path to cam2 video")
    ap.add_argument("--model", default="yolo26l-pose.pt", help="YOLO-Pose model")
    ap.add_argument("--output-dir", default="calibration", help="Output directory")
    args = ap.parse_args()

    names = ["cam1", "cam2"]
    videos = [args.video1, args.video2]

    for v in videos:
        if not os.path.isfile(v):
            print(f"Error: Video not found: {v}")
            sys.exit(1)

    model_path = os.path.join(PROJECT_ROOT, args.model)
    if not os.path.isfile(model_path):
        model_path = args.model
    print(f"Loading YOLO-Pose model: {model_path}")

    model_cam1 = YOLO(model_path)
    model_cam2 = YOLO(model_path)
    models = [model_cam1, model_cam2]

    caps = [cv2.VideoCapture(v) for v in videos]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Cannot open video {videos[i]}")
            sys.exit(1)

    # ─── State ────────────────────────────────────────────────────────
    pts_a, pts_b      = [], []          
    pairs             = []              
    staged_click      = None            
    H                 = None            
    playing           = False # Start paused so user can sync frames!
    collecting        = True
    cur_frames        = [None, None]
    reproj_err        = None
    
    # EMA smoothing dictionaries {cam_id: {track_id: np.array([x, y])}}
    foot_ema          = [{}.copy(), {}.copy()] 

    win = "Person-Track Calibration  [Headcount]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, PANEL_W * 2, PANEL_H + 60)

    scale = [1.0, 1.0]
    click = {"pos": None}
    
    # Pre-load first frames
    _, cur_frames[0] = caps[0].read()
    _, cur_frames[1] = caps[1].read()

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ci = 0 if x < PANEL_W else 1
        ox = (x - ci * PANEL_W) * scale[ci]
        oy = y * scale[ci]
        click["pos"] = (ci, np.array([ox, oy]))

    cv2.setMouseCallback(win, on_mouse)

    def fit():
        nonlocal H, reproj_err
        if len(pts_a) < MIN_POINTS:
            print(f"  ✗ Need >= {MIN_POINTS} points, have {len(pts_a)}.")
            return
        a = np.array(pts_a, dtype=np.float64)
        b = np.array(pts_b, dtype=np.float64)
        
        # UPGRADE: USAC_MAGSAC is vastly superior to standard RANSAC for homography
        H_fit, mask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 3.0)
        
        if H_fit is None:
            print("  ✗ Homography fit failed. Walk more area.")
            return
        H = H_fit
        inl = mask.ravel().astype(bool)
        proj = cv2.perspectiveTransform(
            a[inl].reshape(-1, 1, 2).astype(np.float64), H
        ).reshape(-1, 2)
        reproj_err = float(np.mean(np.linalg.norm(proj - b[inl], axis=1)))
        quality = "EXCELLENT" if reproj_err < 1.5 else ("GOOD" if reproj_err < 4 else "needs more spread")
        print(f"  ✓ Fitted H from {len(pts_a)} points | inliers {int(inl.sum())} | error {reproj_err:.1f} px | {quality}")

    print("\n" + "=" * 70)
    print("  PERSON-TRACK CALIBRATION")
    print("=" * 70)
    print("  ** START PAUSED: Use '1' and '2' to perfectly sync footsteps before playing **")
    print("  1. Click a person in LEFT panel, then SAME person in RIGHT panel to pair.")
    print("  2. Press SPACE to let them walk — points auto-collect concurrently.")
    print("  3. Press 'f' to fit (MAGSAC++), 's' to save, 'q' to quit.")
    print("=" * 70 + "\n")

    advance_cams = [False, False]

    while True:
        # ── Read frames ───────────────────────────────────────────────
        frames_ok = True
        for ci, cap in enumerate(caps):
            if playing or advance_cams[ci]:
                ok, fr = cap.read()
                if not ok:
                    frames_ok = False
                    break
                cur_frames[ci] = fr
                advance_cams[ci] = False
                
        if not frames_ok:
            playing = False
            print("  End of video reached. Press 'f' to fit, 's' to save.")

        if cur_frames[0] is None or cur_frames[1] is None:
            continue

        # ── Detect + track on both cameras ────────────────────────────
        tracks_per = []  
        for ci, fr in enumerate(cur_frames):
            scale[ci] = fr.shape[1] / PANEL_W
            results = models[ci].track(fr, persist=True, tracker="botsort.yaml", conf=0.35, iou=0.5, classes=[0], verbose=False)

            cam_tracks = []
            if results[0].boxes.id is not None:
                boxes     = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                keypoints = results[0].keypoints

                for idx, (box, tid) in enumerate(zip(boxes, track_ids)):
                    tid = int(tid)
                    foot_pt, used_ankle = foot_from_ankles(keypoints, idx)
                    if not used_ankle:
                        foot_pt = foot_of_bbox(box)

                    # UPGRADE: EMA Smoothing for foot coordinates to kill jitter
                    if tid in foot_ema[ci]:
                        foot_pt = (EMA_ALPHA * foot_pt) + ((1 - EMA_ALPHA) * foot_ema[ci][tid])
                    foot_ema[ci][tid] = foot_pt

                    cam_tracks.append({
                        "box":   box,
                        "tid":   tid,
                        "foot":  foot_pt,
                        "ankle": used_ankle,
                        "idx":   idx,
                    })
            tracks_per.append(cam_tracks)

        # ── Handle click → arm a track ─────────────────────────
        if click["pos"] is not None:
            ci, p = click["pos"]
            click["pos"] = None
            best_tid, best_area = None, 1e18
            for t in tracks_per[ci]:
                x1, y1, x2, y2 = t["box"]
                if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if area < best_area:
                        best_tid, best_area = t["tid"], area
            
            if best_tid is not None:
                if staged_click is None:
                    staged_click = (ci, best_tid)
                    pidx, _ = find_pair_by_tid(pairs, ci, best_tid)
                    if pidx is not None:
                        print(f"  → Staged {names[ci]} ID={best_tid} (Pair {pairs[pidx]['id']}). Click other cam.")
                    else:
                        print(f"  → Staged {names[ci]} ID={best_tid}. Click same person in other panel.")
                else:
                    sci, stid = staged_click
                    if sci == ci:
                        staged_click = (ci, best_tid)
                        print(f"  → Swapped staged selection to {names[ci]} ID={best_tid}.")
                    else:
                        pidx_s, pair_s = find_pair_by_tid(pairs, sci, stid)
                        pidx_n, pair_n = find_pair_by_tid(pairs, ci, best_tid)

                        if pidx_s is not None and pidx_n is not None and pidx_s == pidx_n:
                            print(f"  ✓ Already paired together in Pair {pairs[pidx_s]['id']}")
                        elif pidx_s is not None:
                            if pidx_n is not None and pidx_n != pidx_s:
                                pairs[pidx_n]["cam_tids"][ci] = None
                            pair_s["cam_tids"][ci] = best_tid
                        elif pidx_n is not None:
                            pair_n["cam_tids"][sci] = stid
                        else:
                            color = (int(np.random.randint(50, 255)), int(np.random.randint(50, 255)), int(np.random.randint(50, 255)))
                            pairs.append({"id": len(pairs) + 1, "color": color, "cam_tids": [stid if sci==0 else best_tid, best_tid if sci==0 else stid], "last_recorded": None})
                        staged_click = None

        # ── Auto-collect points ───
        if collecting and pairs and playing:
            for pair in pairs:
                cam1_tid, cam2_tid = pair["cam_tids"]
                if cam1_tid is None or cam2_tid is None: continue

                foot1 = next((t["foot"] for t in tracks_per[0] if t["tid"] == cam1_tid), None)
                foot2 = next((t["foot"] for t in tracks_per[1] if t["tid"] == cam2_tid), None)
                
                if foot1 is not None and foot2 is not None:
                    last_pt = pair["last_recorded"]
                    if last_pt is None or np.linalg.norm(foot1 - last_pt[0]) > MIN_SPREAD_PX:
                        pts_a.append(foot1.copy())
                        pts_b.append(foot2.copy())
                        pair["last_recorded"] = (foot1.copy(), foot2.copy())

        # ── Draw ──────────────────────────────────────────────────────
        panels = []
        for ci, fr in enumerate(cur_frames):
            disp = cv2.resize(fr, (PANEL_W, PANEL_H))
            sx, sy = PANEL_W / fr.shape[1], PANEL_H / fr.shape[0]

            for t in tracks_per[ci]:
                x1, y1, x2, y2 = (t["box"] * [sx, sy, sx, sy]).astype(int)
                tid = t["tid"]
                is_staged = (staged_click == (ci, tid))
                pidx, matched_pair = find_pair_by_tid(pairs, ci, tid)
                
                col, thickness = ((0, 255, 255), 3) if is_staged else (matched_pair["color"], 2) if pidx is not None else ((180, 180, 180), 1)
                cv2.rectangle(disp, (x1, y1), (x2, y2), col, thickness)

                fpx, fpy = int(t["foot"][0] * sx), int(t["foot"][1] * sy)
                cv2.circle(disp, (fpx, fpy), 5, (0, 255, 0) if t["ankle"] else (0, 165, 255), -1)

                label = f"ID: {tid}"
                if pidx is not None:
                    other_tid = matched_pair["cam_tids"][1 - ci]
                    status = "OK" if other_tid and any(t2["tid"] == other_tid for t2 in tracks_per[1 - ci]) else "LOST" if other_tid else "HALF"
                    label += f" [P{matched_pair['id']}:{status}]"
                cv2.putText(disp, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            for p in (pts_a if ci == 0 else pts_b):
                cv2.circle(disp, (int(p[0] * sx), int(p[1] * sy)), 3, (255, 200, 0) if ci == 0 else (0, 165, 255), -1)

            cv2.rectangle(disp, (0, 0), (PANEL_W, 30), (0, 0, 0), -1)
            cv2.putText(disp, f"{names[ci].upper()}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            panels.append((disp, sx, sy))

        if H is not None and len(pts_a):
            proj = cv2.perspectiveTransform(np.array(pts_a, dtype=np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)
            for q in proj:
                cv2.circle(panels[1][0], (int(q[0] * panels[1][1]), int(q[1] * panels[1][2])), 4, (0, 0, 255), 1)

        row = np.hstack([p[0] for p in panels])
        bar = np.full((55, row.shape[1], 3), 20, np.uint8)
        cv2.putText(bar, f"Points: {len(pts_a)}/{GOOD_POINTS}   H-Status: {'FITTED ✓' if H is not None else 'not fitted'}   Player: {'PLAYING' if playing else 'PAUSED'}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(bar, "SPACE=play/pause  1=adv Cam1  2=adv Cam2  f=fit (MAGSAC)  s=save  x=clear pairs  q=quit", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.rectangle(bar, (10, 48), (10 + int(min(1.0, len(pts_a) / GOOD_POINTS) * (row.shape[1] - 20)), 52), (0, 255, 0) if len(pts_a) >= GOOD_POINTS else (0, 180, 255), -1)
        cv2.imshow(win, np.vstack([row, bar]))

        # ── Keyboard ──────────────────────────────────────────────────
        key = cv2.waitKey(15 if playing else 0) & 0xFF
        if key == ord("q"): break
        elif key == ord(" "): playing = not playing
        elif key == ord("1") and not playing: advance_cams[0] = True
        elif key == ord("2") and not playing: advance_cams[1] = True
        elif key == ord("c"): collecting = not collecting
        elif key == ord("x"): pairs.clear(); staged_click = None
        elif key == ord("u") and pts_a: pts_a.pop(); pts_b.pop()
        elif key == ord("r"): pts_a.clear(); pts_b.clear(); pairs.clear(); staged_click = None; H = None; foot_ema = [{}.copy(), {}.copy()]
        elif key == ord("f"): fit()
        elif key == ord("s") and H is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            np.save(os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.video1))[0]}_matrix.npy"), H)
            np.save(os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.video2))[0]}_matrix.npy"), np.eye(3))
            print(f"  ✓ Saved matrices to {args.output_dir}/")

    for cap in caps: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()








# """Interactive Multi-Person Track-based cross-camera calibration for the Headcount pipeline.

# Instead of manually clicking floor corners, this tool builds the cam1→cam2
# ground-plane homography from the FEET of multiple paired people as they walk.

# NEW FEATURES:
#   - Multi-person support: Track up to 3-4 people at the same time!
#   - Color-coded pairs: Each paired person gets a unique random bounding box color.
#   - GLOWING TRAILS: Glowing path coordinates drawn on the screen so you know 
#     which areas have already been covered.
#   - MODERN HUD PANEL: A semi-transparent overlay shows points count, active
#     pairs list, RANSAC fit error, and staging status.
#   - DYNAMIC RE-PAIRING: Pairs persist even when tracks are lost. Click to
#     reassign track IDs to existing pairs when people reappear with new IDs.

# Controls:
#     click cam1 then cam2   pair the same person across views
#     SPACE                  play / pause
#     x                      clear active pairings (keep collected points)
#     u                      undo the last collected point
#     f                      fit homography from collected points (RANSAC) + show error
#     s                      save homographies to calibration/
#     r                      reset all collected points and start over
#     q                      quit
# """
# from __future__ import annotations

# import argparse
# import os
# import sys
# import time

# import cv2
# import numpy as np
# import torch

# # Ensure project root is on sys.path for botsort.yaml discovery
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(PROJECT_ROOT)

# from ultralytics import YOLO

# # Fix for PyTorch 2.6+ blocking ultralytics models
# original_load = torch.load
# def _safe_load(*args, **kwargs):
#     kwargs.setdefault('weights_only', False)
#     return original_load(*args, **kwargs)
# torch.load = _safe_load

# # ─── Layout constants ────────────────────────────────────────────────
# PANEL_W, PANEL_H = 800, 450
# MIN_SPREAD_PX    = 12       # only record a new point once foot has moved this far
# MIN_POINTS       = 8        # absolute minimum to fit a homography
# GOOD_POINTS      = 30       # recommended minimum for a reliable fit
# ANKLE_CONF_THRESH = 0.3     # minimum keypoint confidence to trust an ankle


# def foot_of_bbox(xyxy):
#     """Fallback: bottom-center of a bounding box."""
#     x1, y1, x2, y2 = xyxy
#     return np.array([(x1 + x2) / 2.0, float(y2)], dtype=np.float64)


# def foot_from_ankles(keypoints_data, idx):
#     """Extract averaged ankle point from YOLO-Pose keypoints (indices 15, 16).
#     Returns (foot_point, used_ankle_bool)."""
#     if keypoints_data is None:
#         return None, False
#     kpts = keypoints_data.data.cpu().numpy()
#     if idx >= len(kpts):
#         return None, False

#     person_kpts = kpts[idx]           # shape (17, 3) → [x, y, confidence]
#     left_ankle  = person_kpts[15]
#     right_ankle = person_kpts[16]

#     visible = []
#     if left_ankle[2] > ANKLE_CONF_THRESH:
#         visible.append(left_ankle[:2])
#     if right_ankle[2] > ANKLE_CONF_THRESH:
#         visible.append(right_ankle[:2])

#     if visible:
#         avg = np.mean(visible, axis=0)
#         return np.array([float(avg[0]), float(avg[1])], dtype=np.float64), True
#     return None, False


# def find_pair_by_tid(pairs, ci, tid):
#     """Return (index, pair_dict) if tid is paired on camera ci, else (None, None)."""
#     for idx, pair in enumerate(pairs):
#         if pair["cam_tids"][ci] == tid:
#             return idx, pair
#     return None, None


# def main():
#     ap = argparse.ArgumentParser(
#         description="Interactive Multi-Person Track Calibration for Headcount.")
#     ap.add_argument("--video1", required=True,
#                     help="Path to cam1 video (e.g. videos/cam2.mp4)")
#     ap.add_argument("--video2", required=True,
#                     help="Path to cam2 video (e.g. videos/cam3.mp4)")
#     ap.add_argument("--model", default="yolo26l-pose.pt",
#                     help="YOLO-Pose model file (default: yolo26l-pose.pt)")
#     ap.add_argument("--output-dir", default="calibration",
#                     help="Directory to save homography matrices")
#     args = ap.parse_args()

#     names = ["cam1", "cam2"]
#     videos = [args.video1, args.video2]

#     # Verify videos exist
#     for v in videos:
#         if not os.path.isfile(v):
#             print(f"Error: Video not found: {v}")
#             sys.exit(1)

#     # Load YOLO-Pose model (shared for both cameras)
#     model_path = os.path.join(PROJECT_ROOT, args.model)
#     if not os.path.isfile(model_path):
#         model_path = args.model
#     print(f"Loading YOLO-Pose model: {model_path}")

#     # We need TWO separate YOLO instances because BoTSORT persist=True
#     # maintains internal tracker state per model instance.
#     model_cam1 = YOLO(model_path)
#     model_cam2 = YOLO(model_path)
#     models = [model_cam1, model_cam2]

#     # Open video captures
#     caps = [cv2.VideoCapture(v) for v in videos]
#     for i, cap in enumerate(caps):
#         if not cap.isOpened():
#             print(f"Error: Cannot open video {videos[i]}")
#             sys.exit(1)

#     # ─── State ────────────────────────────────────────────────────────
#     pts_a, pts_b      = [], []          # global list of correspondences
#     # Dynamic pair system: each pair is a persistent slot with:
#     #   "id": int, "color": (B,G,R), "cam_tids": [cam1_tid|None, cam2_tid|None],
#     #   "last_recorded": (foot1, foot2)|None
#     # Pairs survive track-ID changes — click to reassign when tracks reappear.
#     pairs             = []              # list of pair dicts
#     staged_click      = None            # (ci, tid) | None
#     H                 = None            # fitted homography
#     playing           = True
#     collecting        = True
#     cur_frames        = [None, None]
#     frame_count       = 0
#     reproj_err        = None

#     win = "Person-Track Calibration  [Headcount]"
#     cv2.namedWindow(win, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(win, PANEL_W * 2, PANEL_H + 60)

#     scale = [1.0, 1.0]
#     click = {"pos": None}

#     def on_mouse(event, x, y, flags, param):
#         if event != cv2.EVENT_LBUTTONDOWN:
#             return
#         ci = 0 if x < PANEL_W else 1
#         ox = (x - ci * PANEL_W) * scale[ci]
#         oy = y * scale[ci]
#         click["pos"] = (ci, np.array([ox, oy]))

#     cv2.setMouseCallback(win, on_mouse)

#     def fit():
#         nonlocal H, reproj_err
#         if len(pts_a) < MIN_POINTS:
#             print(f"  ✗ Need >= {MIN_POINTS} points, have {len(pts_a)}.")
#             return
#         a = np.array(pts_a, dtype=np.float64)
#         b = np.array(pts_b, dtype=np.float64)
#         H_fit, mask = cv2.findHomography(a, b, cv2.RANSAC, 5.0)
#         if H_fit is None:
#             print("  ✗ Homography fit failed (points may be collinear). Walk more area.")
#             return
#         H = H_fit
#         inl = mask.ravel().astype(bool)
#         proj = cv2.perspectiveTransform(
#             a[inl].reshape(-1, 1, 2).astype(np.float64), H
#         ).reshape(-1, 2)
#         reproj_err = float(np.mean(np.linalg.norm(proj - b[inl], axis=1)))
#         quality = "EXCELLENT" if reproj_err < 2 else ("GOOD" if reproj_err < 5 else "needs more points/spread")
#         print(f"  ✓ Fitted H from {len(pts_a)} points | "
#               f"inliers {int(inl.sum())} | "
#               f"mean reproj error {reproj_err:.1f} px | {quality}")

#     print("\n" + "=" * 70)
#     print("  PERSON-TRACK CALIBRATION")
#     print("=" * 70)
#     print("  1. Click a person in the LEFT panel (cam1)")
#     print("  2. Click the SAME person in the RIGHT panel (cam2) to PAIR them.")
#     print("  3. Repeat for more walking people (multi-person supported).")
#     print("  4. Let them walk — points auto-collect concurrently.")
#     print("  5. If a track is LOST, click the person again to reassign the pair.")
#     print("  6. Press 'f' to fit,  's' to save,  'q' to quit.")
#     print("=" * 70 + "\n")

#     while True:
#         # ── Read frames ───────────────────────────────────────────────
#         if playing:
#             frames_ok = True
#             new_frames = []
#             for cap in caps:
#                 ok, fr = cap.read()
#                 if not ok:
#                     frames_ok = False
#                     break
#                 new_frames.append(fr)
#             if not frames_ok:
#                 playing = False
#                 print("  End of video reached. Press 'f' to fit, 's' to save.")
#                 continue
#             cur_frames = new_frames
#             frame_count += 1

#         if cur_frames[0] is None:
#             continue

#         # ── Detect + track on both cameras ────────────────────────────
#         tracks_per = []  # list of [(box, tid, keypoints_idx), ...]
#         for ci, fr in enumerate(cur_frames):
#             scale[ci] = fr.shape[1] / PANEL_W

#             results = models[ci].track(
#                 fr, persist=True, tracker="botsort.yaml",
#                 conf=0.35, iou=0.5, classes=[0], verbose=False
#             )

#             cam_tracks = []
#             if results[0].boxes.id is not None:
#                 boxes     = results[0].boxes.xyxy.cpu().numpy()
#                 track_ids = results[0].boxes.id.cpu().numpy()
#                 keypoints = results[0].keypoints

#                 for idx, (box, tid) in enumerate(zip(boxes, track_ids)):
#                     # Try ankle-based foot point (same as sct_worker)
#                     foot_pt, used_ankle = foot_from_ankles(keypoints, idx)
#                     if not used_ankle:
#                         foot_pt = foot_of_bbox(box)

#                     cam_tracks.append({
#                         "box":   box,
#                         "tid":   int(tid),
#                         "foot":  foot_pt,
#                         "ankle": used_ankle,
#                         "idx":   idx,
#                     })
#             tracks_per.append(cam_tracks)

#         # ── Handle click → arm a track (dynamic re-pairing) ─────────
#         if click["pos"] is not None:
#             ci, p = click["pos"]
#             click["pos"] = None
#             # Find which track box the click landed in
#             best_tid, best_area = None, 1e18
#             for t in tracks_per[ci]:
#                 x1, y1, x2, y2 = t["box"]
#                 if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
#                     area = (x2 - x1) * (y2 - y1)
#                     if area < best_area:
#                         best_tid, best_area = t["tid"], area
            
#             if best_tid is not None:
#                 if staged_click is None:
#                     # Stage the selection
#                     staged_click = (ci, best_tid)
#                     pidx, _ = find_pair_by_tid(pairs, ci, best_tid)
#                     if pidx is not None:
#                         print(f"  → Staged {names[ci]} track id={best_tid} (Pair {pairs[pidx]['id']}). "
#                               f"Click other cam to reassign.")
#                     else:
#                         print(f"  → Staged {names[ci]} track id={best_tid}. "
#                               f"Click same person in the other panel.")
#                 else:
#                     sci, stid = staged_click
#                     if sci == ci:
#                         # Same camera — replace staged selection
#                         staged_click = (ci, best_tid)
#                         print(f"  → Swapped staged selection to {names[ci]} "
#                               f"track id={best_tid}.")
#                     else:
#                         # Different cameras — form or update a pair!
#                         # Look up existing pair membership
#                         pidx_s, pair_s = find_pair_by_tid(pairs, sci, stid)
#                         pidx_n, pair_n = find_pair_by_tid(pairs, ci, best_tid)

#                         if (pidx_s is not None and pidx_n is not None
#                                 and pidx_s == pidx_n):
#                             # Both already in the same pair
#                             print(f"  ✓ Already paired together in "
#                                   f"Pair {pairs[pidx_s]['id']}")
#                         elif pidx_s is not None:
#                             # Staged person owns a pair → update other cam
#                             if pidx_n is not None and pidx_n != pidx_s:
#                                 pairs[pidx_n]["cam_tids"][ci] = None
#                             pair_s["cam_tids"][ci] = best_tid
#                             print(f"  ✓ Updated Pair {pair_s['id']}: "
#                                   f"cam1:{pair_s['cam_tids'][0]} ↔ "
#                                   f"cam2:{pair_s['cam_tids'][1]}")
#                         elif pidx_n is not None:
#                             # Clicked person owns a pair → update other cam
#                             pair_n["cam_tids"][sci] = stid
#                             print(f"  ✓ Updated Pair {pair_n['id']}: "
#                                   f"cam1:{pair_n['cam_tids'][0]} ↔ "
#                                   f"cam2:{pair_n['cam_tids'][1]}")
#                         else:
#                             # Neither has a pair → create new
#                             color = (int(np.random.randint(50, 255)),
#                                      int(np.random.randint(50, 255)),
#                                      int(np.random.randint(50, 255)))
#                             new_pair = {
#                                 "id": len(pairs) + 1,
#                                 "color": color,
#                                 "cam_tids": [None, None],
#                                 "last_recorded": None,
#                             }
#                             new_pair["cam_tids"][sci] = stid
#                             new_pair["cam_tids"][ci] = best_tid
#                             pairs.append(new_pair)
#                             print(f"  ✓ NEW Pair {new_pair['id']}: "
#                                   f"cam1:{new_pair['cam_tids'][0]} ↔ "
#                                   f"cam2:{new_pair['cam_tids'][1]}")

#                         staged_click = None

#         # ── Auto-collect points for all active pairs concurrently ───
#         if collecting and pairs:
#             for pair in pairs:
#                 cam1_tid = pair["cam_tids"][0]
#                 cam2_tid = pair["cam_tids"][1]
#                 if cam1_tid is None or cam2_tid is None:
#                     continue

#                 foot1 = None
#                 foot2 = None
                
#                 # Look up coordinates in current frames
#                 for t in tracks_per[0]:
#                     if t["tid"] == cam1_tid:
#                         foot1 = t["foot"]
#                         break
#                 for t in tracks_per[1]:
#                     if t["tid"] == cam2_tid:
#                         foot2 = t["foot"]
#                         break
                
#                 if foot1 is not None and foot2 is not None:
#                     # Enforce spread distance per pair
#                     last_pt = pair["last_recorded"]
#                     if (last_pt is None or 
#                         np.linalg.norm(foot1 - last_pt[0]) > MIN_SPREAD_PX):
#                         pts_a.append(foot1.copy())
#                         pts_b.append(foot2.copy())
#                         pair["last_recorded"] = (foot1.copy(), foot2.copy())

#         # ── Draw ──────────────────────────────────────────────────────
#         panels = []
#         for ci, fr in enumerate(cur_frames):
#             disp = cv2.resize(fr, (PANEL_W, PANEL_H))
#             sx = PANEL_W / fr.shape[1]
#             sy = PANEL_H / fr.shape[0]

#             for t in tracks_per[ci]:
#                 x1, y1, x2, y2 = (t["box"] * [sx, sy, sx, sy]).astype(int)
#                 tid = t["tid"]
                
#                 # Check status of the track for coloring
#                 is_staged = (staged_click is not None and staged_click == (ci, tid))
                
#                 # Dynamic pair lookup
#                 pidx, matched_pair = find_pair_by_tid(pairs, ci, tid)
                
#                 if is_staged:
#                     col = (0, 255, 255)  # Yellow for staged
#                     thickness = 3
#                 elif pidx is not None:
#                     col = matched_pair["color"]
#                     thickness = 2
#                 else:
#                     col = (180, 180, 180)
#                     thickness = 1

#                 cv2.rectangle(disp, (x1, y1), (x2, y2), col, thickness)

#                 # Draw foot point
#                 fp = t["foot"]
#                 fpx, fpy = int(fp[0] * sx), int(fp[1] * sy)
#                 foot_col = (0, 255, 0) if t["ankle"] else (0, 165, 255)
#                 cv2.circle(disp, (fpx, fpy), 5, foot_col, -1)

#                 # Draw track label with dynamic pair status
#                 label = f"ID: {tid}"
#                 if pidx is not None:
#                     other_ci = 1 - ci
#                     other_tid = matched_pair["cam_tids"][other_ci]
#                     if other_tid is not None:
#                         other_visible = any(
#                             t2["tid"] == other_tid for t2 in tracks_per[other_ci])
#                         status = "OK" if other_visible else "LOST"
#                     else:
#                         status = "HALF"
#                     label += f" [P{matched_pair['id']}:{status}]"
#                 if is_staged:
#                     label += " [STAGED]"
#                 cv2.putText(disp, label, (x1, y1 - 4),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

#             # Draw glowing paths (collected points) on this camera panel
#             pts = pts_a if ci == 0 else pts_b
#             for p in pts:
#                 px, py = int(p[0] * sx), int(p[1] * sy)
#                 # Cyan for cam1, Orange for cam2
#                 c = (255, 200, 0) if ci == 0 else (0, 165, 255)
#                 cv2.circle(disp, (px, py), 3, c, -1)

#             # Camera panel top header bar
#             cv2.rectangle(disp, (0, 0), (PANEL_W, 30), (0, 0, 0), -1)
#             cv2.putText(disp, f"{names[ci].upper()}", (8, 22),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#             # Draw a modern, semi-transparent HUD overlay in top-right
#             # Build per-pair status indicators
#             pair_indicators = []
#             for p in pairs[:4]:
#                 c1_ok = p["cam_tids"][0] is not None and any(
#                     t["tid"] == p["cam_tids"][0] for t in tracks_per[0])
#                 c2_ok = p["cam_tids"][1] is not None and any(
#                     t["tid"] == p["cam_tids"][1] for t in tracks_per[1])
#                 s1 = "+" if c1_ok else "-"
#                 s2 = "+" if c2_ok else "-"
#                 pair_indicators.append(f"P{p['id']}:{s1}|{s2}")

#             n_extra = min(len(pairs), 4)
#             hud_w = 270
#             hud_h = 100 + n_extra * 18
#             hud_x = PANEL_W - hud_w - 10
#             hud_y = 40
            
#             # Semi-transparent HUD rectangle
#             hud_overlay = disp.copy()
#             cv2.rectangle(hud_overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 20, 20), -1)
#             cv2.addWeighted(hud_overlay, 0.75, disp, 0.25, 0, disp)
            
#             # Draw border
#             cv2.rectangle(disp, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (80, 80, 80), 1)
            
#             # HUD text lines
#             hud_lines = [
#                 f"Pairs: {len(pairs)}  {' '.join(pair_indicators)}",
#                 f"Points: {len(pts_a)} / {GOOD_POINTS}",
#                 "RANSAC: " + (f"{reproj_err:.2f} px" if reproj_err else "not fitted"),
#                 "Staged: " + (f"Cam{staged_click[0]+1} ID {staged_click[1]}" if staged_click else "None"),
#             ]
#             # Per-pair detail lines
#             for p in pairs[:4]:
#                 c1t = p["cam_tids"][0]
#                 c2t = p["cam_tids"][1]
#                 hud_lines.append(f"  P{p['id']}: L={c1t if c1t else '-'}  R={c2t if c2t else '-'}")

#             for i, line in enumerate(hud_lines):
#                 cv2.putText(disp, line, (hud_x + 8, hud_y + 18 + i*18),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

#             panels.append((disp, sx, sy))

#         # Overlay: project cam1 feet into cam2 via H
#         if H is not None and len(pts_a):
#             a_arr = np.array(pts_a, dtype=np.float64).reshape(-1, 1, 2)
#             proj = cv2.perspectiveTransform(a_arr, H).reshape(-1, 2)
#             d1, sx1, sy1 = panels[1]
#             for q in proj:
#                 # Red rings on Cam 2 indicating projection matching
#                 cv2.circle(d1, (int(q[0] * sx1), int(q[1] * sy1)), 4, (0, 0, 255), 1)

#         # Stitch panels horizontally
#         row = np.hstack([p[0] for p in panels])

#         # Bottom control status bar
#         bar = np.full((55, row.shape[1], 3), 20, np.uint8)
#         n_pts = len(pts_a)
#         h_status = "FITTED ✓" if H is not None else "not fitted"
#         state = "PLAYING" if playing else "PAUSED"
#         coll = "ON" if collecting else "OFF"
        
#         # Build status strings
#         msg_l1 = (f"Points: {n_pts}/{GOOD_POINTS}   "
#                   f"Auto-Collect: {coll}   H-Status: {h_status}   Player: {state}")
#         msg_l2 = "SPACE=pause  f=fit homography  s=save matrices  x=clear pairings  u=undo last point  r=reset all  q=quit"
        
#         cv2.putText(bar, msg_l1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
#         cv2.putText(bar, msg_l2, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

#         # Progress bar at very bottom of screen
#         progress = min(1.0, n_pts / GOOD_POINTS)
#         bar_w = int(progress * (row.shape[1] - 20))
#         bar_col = (0, 255, 0) if n_pts >= GOOD_POINTS else (0, 180, 255)
#         cv2.rectangle(bar, (10, 48), (10 + bar_w, 52), bar_col, -1)

#         cv2.imshow(win, np.vstack([row, bar]))

#         # ── Keyboard ──────────────────────────────────────────────────
#         key = cv2.waitKey(15 if playing else 0) & 0xFF
#         if key == ord("q"):
#             break
#         elif key == ord(" "):
#             playing = not playing
#         elif key == ord("c"):
#             collecting = not collecting
#             print(f"  Auto-collection {'ON' if collecting else 'OFF'}")
#         elif key == ord("x"):
#             # Clear all pairs (keeps accumulated calibration points)
#             pairs.clear()
#             staged_click = None
#             print("  ✓ Cleared all pairs (collected points are KEPT)")
#         elif key == ord("u") and pts_a:
#             pts_a.pop()
#             pts_b.pop()
#             print(f"  Undone last point ({len(pts_a)} remaining)")
#         elif key == ord("r"):
#             pts_a.clear()
#             pts_b.clear()
#             pairs.clear()
#             staged_click = None
#             H = None
#             reproj_err = None
#             print("  ✗ Reset all collected points and pairs")
#         elif key == ord("f"):
#             fit()
#         elif key == ord("s"):
#             if H is None:
#                 fit()
#             if H is not None:
#                 os.makedirs(args.output_dir, exist_ok=True)
#                 base1 = os.path.splitext(os.path.basename(args.video1))[0]
#                 base2 = os.path.splitext(os.path.basename(args.video2))[0]
#                 path_cam1 = os.path.join(args.output_dir, f"{base1}_matrix.npy")
#                 path_cam2 = os.path.join(args.output_dir, f"{base2}_matrix.npy")
#                 np.save(path_cam1, H)
#                 np.save(path_cam2, np.eye(3))
#                 print(f"\n  ✓ Saved {path_cam1}  (cam1 → cam2 homography)")
#                 print(f"  ✓ Saved {path_cam2}  (identity — cam2 is reference)")
#                 print(f"  Pipeline is ready: ./run_pipeline.sh\n")

#     for cap in caps:
#         cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()