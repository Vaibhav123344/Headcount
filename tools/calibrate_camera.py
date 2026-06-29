# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import argparse
import os

# Global variables for the mouse callback
src_pts = []
img_display = None
window_name = 'Calibration - Click points (Press "c" or "Enter" to finish)'

def click_event(event, x, y, flags, param):
    global src_pts, img_display, window_name
    if event == cv2.EVENT_LBUTTONDOWN:
        src_pts.append([x, y])
        pt_idx = len(src_pts)
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img_display, str(pt_idx), (x+10, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow(window_name, img_display)
        print(f"Point {pt_idx} recorded at ({x}, {y}).")

def main():
    parser = argparse.ArgumentParser(description="Calibrate camera with Homography.")
    parser.add_argument("--image", required=True, help="Path to the sample camera frame image.")
    parser.add_argument("--output", default="camera_matrix.npy", help="Output path for the matrix (.npy).")
    args = parser.parse_args()

    global img_display, src_pts, window_name
    
    if not os.path.exists(args.image):
        print(f"Error: Could not find image at {args.image}")
        return

    img = cv2.imread(args.image)
    img_display = img.copy()

    print("\n" + "="*60)
    print("EASY MODE: Click exactly 4 points that form a RECTANGLE")
    print("           (Top-Left, Top-Right, Bottom-Right, Bottom-Left).")
    print("           The system will auto-calculate physical coordinates.")
    print("\nADVANCED MODE: Click 5+ points. You will be asked to manually")
    print("               type in the physical measurements for each.")
    print("="*60 + "\n")
    print("IMPORTANT: When finished clicking points, MAKE SURE THE IMAGE WINDOW")
    print("IS IN FOCUS (click on it), then press 'c' or 'ENTER' on your keyboard.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.imshow(window_name, img_display)
    cv2.setMouseCallback(window_name, click_event)

    # Wait until 'c', Enter, Esc, or 'q' is pressed
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key in [ord('c'), ord('C'), ord('q'), ord('Q'), 13, 27]:
            if len(src_pts) >= 4:
                break
            else:
                print(f"You only selected {len(src_pts)} points. Please select at least 4 points.")

    cv2.destroyAllWindows()
    
    dst_pts = []
    
    if len(src_pts) == 4:
        print("\n--- AUTO-GENERATING COORDINATES (4 Points) ---")
        print("Assuming you clicked a perfect rectangle.")
        width, height = 10.0, 10.0 # Arbitrary 10x10 meter square
        dst_pts = [
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height]
        ]
        for i in range(4):
            print(f"Point {i+1} automatically mapped to {dst_pts[i]}")
    else:
        print("\n--- Physical Coordinate Mapping (5+ Points) ---")
        print("Now, enter the real-world measurements (X Y) in meters for each point.")
        for i in range(len(src_pts)):
            while True:
                try:
                    coords = input(f"Enter world coordinates (X Y) for Point {i+1}: ")
                    x, y = map(float, coords.strip().split())
                    dst_pts.append([x, y])
                    break
                except ValueError:
                    print("Invalid input! Please enter two numbers separated by a space (e.g., 5.0 10.0)")

    src_pts_arr = np.array(src_pts, dtype=np.float32)
    dst_pts_arr = np.array(dst_pts, dtype=np.float32)

    print("\nCalculating Homography matrix...")
    if len(src_pts) == 4:
        H, status = cv2.findHomography(src_pts_arr, dst_pts_arr)
    else:
        H, status = cv2.findHomography(src_pts_arr, dst_pts_arr, cv2.RANSAC, 5.0)

    if H is not None:
        print("Homography Matrix successfully calculated!")
        np.save(args.output, H)
        print(f"Matrix saved to {args.output}")
        
        print("\n--- Calibration Reprojection Errors ---")
        projected_pts = cv2.perspectiveTransform(src_pts_arr.reshape(-1, 1, 2), H).reshape(-1, 2)
        total_error = 0
        for i in range(len(src_pts_arr)):
            err = np.linalg.norm(dst_pts_arr[i] - projected_pts[i])
            total_error += err
            outlier_status = "INLIER" if (status is not None and status[i][0] == 1) else ""
            print(f"Point {i+1}: Measured {dst_pts_arr[i]} -> Calculated {projected_pts[i].round(2)} | Error: {err:.2f}m {outlier_status}")
        print(f"Average Error: {total_error/len(src_pts_arr):.2f}m")
    else:
        print("Failed to compute Homography matrix.")

if __name__ == '__main__':
    main()
