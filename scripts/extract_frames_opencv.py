"""
extract_frames_opencv.py

Purpose:
    Extracts frames from a video file using OpenCV and saves them as JPEG images.

Usage:
    python scripts/extract_frames_opencv.py --video path/to/video.mp4 --out_dir path/to/output_frames --every_n 1

    --video:   Path to the input video file.
    --out_dir: Directory to save extracted frames.
    --every_n: (Optional) Save every n-th frame (default: 1, saves all frames).

    Extracted frames will be saved as frame_000000.jpg, frame_000001.jpg, etc. in the output directory.
"""

import cv2
import os

def extract_frames(video_path, output_dir, every_n=1):
    """
    Extract frames from a video using OpenCV.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
        every_n (int): Save every n-th frame (default: 1, save all).
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved_idx} frames to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract frames from a video using OpenCV.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save frames")
    parser.add_argument("--every_n", type=int, default=1, help="Save every n-th frame (default: 1)")
    args = parser.parse_args()

    extract_frames(args.video, args.out_dir, args.every_n)
