import cv2
import numpy as np
import argparse
import os

# Global variables for the mouse callback
src_pts = []
img_display = None
window_name = 'Calibration - Click 4 points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)'

def click_event(event, x, y, flags, param):
    global src_pts, img_display, window_name
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_pts) < 4:
            src_pts.append([x, y])
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img_display, str(len(src_pts)), (x+10, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow(window_name, img_display)
            
            if len(src_pts) == 4:
                print("4 points selected. Press any key to continue...")

def main():
    parser = argparse.ArgumentParser(description="Calibrate camera and generate Homography matrix.")
    parser.add_argument("--image", required=True, help="Path to the sample camera frame image.")
    parser.add_argument("--output", default="camera_matrix.npy", help="Output path for the matrix (.npy).")
    parser.add_argument("--width", type=int, default=500, help="Real-world width (or arbitrary scale) of the selected area.")
    parser.add_argument("--height", type=int, default=500, help="Real-world height (or arbitrary scale) of the selected area.")
    args = parser.parse_args()

    global img_display, src_pts, window_name
    
    if not os.path.exists(args.image):
        print(f"Error: Could not find image at {args.image}")
        return

    img = cv2.imread(args.image)
    img_display = img.copy()

    print("--- Camera Calibration ---")
    print("Please click exactly 4 points on the floor that form a rectangle in the real world.")
    print("Order of clicks: 1. Top-Left, 2. Top-Right, 3. Bottom-Right, 4. Bottom-Left")

    # Create a resizable window and set a moderate default size
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    cv2.imshow(window_name, img_display)
    cv2.setMouseCallback(window_name, click_event)

    print("Waiting for user to click 4 points...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(src_pts) != 4:
        print("Error: You must select exactly 4 points. Exiting.")
        return

    src_pts_arr = np.array(src_pts, dtype=np.float32)
    
    # Destination points form a rectangle starting from (0,0)
    dst_pts_arr = np.array([
        [0, 0],
        [args.width, 0],
        [args.width, args.height],
        [0, args.height]
    ], dtype=np.float32)

    print("\nCalculating Homography matrix...")
    H, status = cv2.findHomography(src_pts_arr, dst_pts_arr)

    if H is not None:
        print("Homography Matrix successfully calculated:")
        print(H)
        np.save(args.output, H)
        print(f"Matrix saved to {args.output}")
        
        # Show a quick warped preview
        preview_win = "Top-Down Preview (Press any key to close)"
        cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_win, 800, 800)
        warped = cv2.warpPerspective(img, H, (args.width, args.height))
        cv2.imshow(preview_win, warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to compute Homography matrix.")

if __name__ == '__main__':
    main()
