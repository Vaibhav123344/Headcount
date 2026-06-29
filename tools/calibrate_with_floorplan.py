import cv2
import numpy as np
import sys
import argparse
import os

cam_pts = []
floor_pts = []
cam_disp = None
floor_disp = None

def on_cam_click(event, x, y, flags, param):
    global cam_pts, cam_disp
    if event == cv2.EVENT_LBUTTONDOWN:
        cam_pts.append([x, y])
        cv2.circle(cam_disp, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(cam_disp, str(len(cam_pts)), (x + 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("1. Camera Frame (Click Landmark)", cam_disp)
        print(f"Cam Point {len(cam_pts)} registered: ({x}, {y})")

def on_floor_click(event, x, y, flags, param):
    global floor_pts, floor_disp
    if event == cv2.EVENT_LBUTTONDOWN:
        floor_pts.append([x, y])
        cv2.circle(floor_disp, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(floor_disp, str(len(floor_pts)), (x + 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("2. Floor Plan (Click Same Landmark)", floor_disp)
        print(f"Floor Point {len(floor_pts)} registered: ({x}, {y})")

def main():
    parser = argparse.ArgumentParser(description="Calibrate camera with a Floor Plan using RANSAC.")
    parser.add_argument("--image", required=True, help="Path to camera frame image.")
    parser.add_argument("--floor-plan", required=True, help="Path to floor plan image.")
    parser.add_argument("--output", default="camera_matrix.npy", help="Output path for the matrix.")
    args = parser.parse_args()

    global cam_disp, floor_disp

    if not os.path.exists(args.image) or not os.path.exists(args.floor_plan):
        print("Error: Could not load camera image or floor plan. Check your paths!")
        sys.exit()

    cam_img = cv2.imread(args.image)
    floor_img = cv2.imread(args.floor_plan)

    cam_disp = cam_img.copy()
    floor_disp = floor_img.copy()

    # Setup GUI Windows
    cv2.namedWindow("1. Camera Frame (Click Landmark)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2. Floor Plan (Click Same Landmark)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("1. Camera Frame (Click Landmark)", 800, 600)
    cv2.resizeWindow("2. Floor Plan (Click Same Landmark)", 800, 600)

    cv2.setMouseCallback("1. Camera Frame (Click Landmark)", on_cam_click)
    cv2.setMouseCallback("2. Floor Plan (Click Same Landmark)", on_floor_click)

    cv2.imshow("1. Camera Frame (Click Landmark)", cam_disp)
    cv2.imshow("2. Floor Plan (Click Same Landmark)", floor_disp)

    print("--- DUAL-WINDOW FLOOR PLAN CALIBRATION ---")
    print("INSTRUCTIONS:")
    print("1. Click a landmark (e.g., corner of table) in the Camera Frame window.")
    print("2. Click the exact same landmark on the Floor Plan window.")
    print("3. Repeat this process for at least 8 distinct points across the room.")
    print("4. Press 'S' on your keyboard to save the Homography Matrix, or 'Q' to quit.")

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            print("Calibration canceled.")
            break
        elif key in [ord('s'), ord('S'), 13]:
            if len(cam_pts) != len(floor_pts) or len(cam_pts) < 4:
                print(f"Error: Point count mismatch or too few points! Cam: {len(cam_pts)}, Floor: {len(floor_pts)}. Need at least 4.")
                continue
            
            # Convert list to numpy arrays
            src_arr = np.array(cam_pts, dtype=np.float32)
            dst_arr = np.array(floor_pts, dtype=np.float32)
            
            # Compute Homography using RANSAC for robust math
            # Threshold increased to 50.0 because dst_arr is now in pixels, not meters!
            H, status = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 50.0)
            
            if H is not None:
                # Save computed matrix
                np.save(args.output, H)
                print(f"\nSuccess! Robust matrix saved to: {args.output}")
                print(f"Total verified points mapped (Inliers): {np.sum(status)}")
            else:
                print("Failed to compute Homography matrix. Try clicking more spread-out points.")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
