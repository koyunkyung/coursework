import cv2
import os
import sys

def get_box_from_video(video_path, skip_seconds=3.0):
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Calculate frame to skip to
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps * skip_seconds)
    
    # Jump to the start of the actual content
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame from video.")
        return

    # Select ROI
    rect = cv2.selectROI("Select Object from Video", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    
    x_min, y_min, w, h = rect
    box_coords = [x_min, y_min, x_min + w, y_min + h]

    print(f"\n Box Coordinates for {os.path.basename(video_path)}:")
    print(f"box = {box_coords}")

if __name__ == "__main__":
    video_path = "./data/raw/teapot.mp4" 
    get_box_from_video(video_path, skip_seconds=5.0)