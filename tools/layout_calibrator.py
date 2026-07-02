import cv2
import numpy as np
import argparse
import os

def main():
    ap = argparse.ArgumentParser(description="Map Camera to 2D BEV Layout")
    ap.add_argument("--video", required=True, help="Path to camera video (e.g., videos/cam2.mp4)")
    ap.add_argument("--layout", required=True, help="Path to 2D layout map (e.g., map.png)")
    ap.add_argument("--output", required=True, help="Path to save output matrix (e.g., calibration/cam2_matrix.npy)")
    args = ap.parse_args()

    # Load 1st frame of the video
    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()
    if not ret: return print("Failed to read video.")
    
    layout = cv2.imread(args.layout)
    if layout is None: return print("Failed to load layout image.")

    pts_cam, pts_layout = [], []
    
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if param == "cam":
                pts_cam.append([x, y])
                print(f"Cam point added: {x}, {y}")
            else:
                pts_layout.append([x, y])
                print(f"Layout point added: {x}, {y}")

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2D Layout Map", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Camera", on_mouse, "cam")
    cv2.setMouseCallback("2D Layout Map", on_mouse, "layout")

    print("Click at least 4 matching floor points on BOTH windows (e.g., corners of tables/room).")
    print("Press 'f' to fit the layout. Press 'q' to quit.")

    while True:
        disp_cam = frame.copy()
        disp_lay = layout.copy()

        for p in pts_cam: cv2.circle(disp_cam, tuple(p), 5, (0, 255, 255), -1)
        for p in pts_layout: cv2.circle(disp_lay, tuple(p), 5, (0, 255, 0), -1)

        cv2.imshow("Camera", disp_cam)
        cv2.imshow("2D Layout Map", disp_lay)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('f'):
            if len(pts_cam) >= 4 and len(pts_layout) >= 4:
                # Map Camera to Layout directly
                H_cam_to_layout, status = cv2.findHomography(
                    np.array(pts_cam, dtype=np.float32), 
                    np.array(pts_layout, dtype=np.float32)
                )

                # Save the matrix
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                np.save(args.output, H_cam_to_layout)
                
                print(f"✓ Successfully mapped camera to layout! Matrix saved to {args.output}")
                break
            else:
                print("Need at least 4 points on both!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()