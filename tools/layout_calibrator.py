import cv2
import numpy as np
import argparse
import os

def main():
    ap = argparse.ArgumentParser(description="Map Master Camera to 2D BEV Layout")
    ap.add_argument("--video", required=True, help="Path to master video (e.g., videos/cam5.mp4)")
    ap.add_argument("--layout", required=True, help="Path to 2D layout map (e.g., layout.png)")
    ap.add_argument("--matrix1", required=True, help="Path to existing cam1 matrix (e.g., calibration/cam2_matrix.npy)")
    ap.add_argument("--matrix2", required=True, help="Path to existing master matrix (e.g., calibration/cam3_matrix.npy)")
    args = ap.parse_args()

    # Load 1st frame of the master video
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

    cv2.namedWindow("Master Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2D Layout Map", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Master Camera", on_mouse, "cam")
    cv2.setMouseCallback("2D Layout Map", on_mouse, "layout")

    print("Click at least 4 matching floor points on BOTH windows (corners of a room/zone).")
    print("Press 'f' to fit the layout. Press 'q' to quit.")

    while True:
        disp_cam = frame.copy()
        disp_lay = layout.copy()

        for p in pts_cam: cv2.circle(disp_cam, tuple(p), 5, (0, 255, 255), -1)
        for p in pts_layout: cv2.circle(disp_lay, tuple(p), 5, (0, 255, 0), -1)

        cv2.imshow("Master Camera", disp_cam)
        cv2.imshow("2D Layout Map", disp_lay)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('f'):
            if len(pts_cam) >= 4 and len(pts_layout) >= 4:
                # 1. Map Master Camera to Layout
                H_cam_to_layout, _ = cv2.findHomography(
                    np.array(pts_cam, dtype=np.float32), 
                    np.array(pts_layout, dtype=np.float32)
                )

                # 2. Load existing tracking matrices
                H_cam1_to_master = np.load(args.matrix1)
                
                # 3. MATRIX CHAINING (The Magic Step)
                H_cam1_to_layout = np.dot(H_cam_to_layout, H_cam1_to_master)
                H_master_to_layout = H_cam_to_layout 

                # 4. Overwrite edge matrices
                np.save(args.matrix1, H_cam1_to_layout)
                np.save(args.matrix2, H_master_to_layout)
                
                print("✓ Successfully Chained Matrices! Edge workers will now output Layout coordinates directly.")
                break
            else:
                print("Need exactly 4 points on both!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()