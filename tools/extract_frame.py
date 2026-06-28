import cv2
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Extract the first frame from a video for calibration.")
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--output", required=True, help="Path to save the output image (e.g. frame.jpg)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(args.output, frame)
        print(f"Successfully saved first frame to {args.output}")
    else:
        print("Error: Could not read frame from video.")
        sys.exit(1)
        
    cap.release()

if __name__ == "__main__":
    main()
