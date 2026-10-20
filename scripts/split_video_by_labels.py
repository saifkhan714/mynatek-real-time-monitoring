"""
split_video_by_labels.py

Purpose:
    Splits a video into multiple chunks based on label changes in a provided annotation JSON file.
    Each chunk is saved as a separate video file, grouped into subdirectories by label (e.g., 'cleaning', 'noncleaning').

Usage:
    python scripts/split_video_by_labels.py --video path/to/video.mp4 --annotations path/to/annotations.json --outdir path/to/output_dir

    --video:       Path to the input video file.
    --annotations: Path to the annotation JSON file (must contain 'frame_num' and 'label' for each frame).
    --outdir:      Output directory (default: current directory).

    The script will create subdirectories for each label and save video chunks corresponding to contiguous label segments.
"""

import os
import json
import cv2
import argparse
from pathlib import Path

def parse_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Only keep entries with a label
    labeled = [entry for entry in data if entry.get('label') is not None]
    # If label is missing for some frames, fill forward
    for i in range(1, len(labeled)):
        if labeled[i]['label'] is None:
            labeled[i]['label'] = labeled[i-1]['label']
    return labeled

def get_chunks(labeled):
    """
    Returns a list of (start_frame, end_frame, label) tuples.
    """
    if not labeled:
        return []
    chunks = []
    prev_label = labeled[0]['label']
    start_frame = labeled[0]['frame_num']
    for i in range(1, len(labeled)):
        curr_label = labeled[i]['label']
        if curr_label != prev_label:
            end_frame = labeled[i-1]['frame_num']
            chunks.append((start_frame, end_frame, prev_label))
            start_frame = labeled[i]['frame_num']
            prev_label = curr_label
    # Add last chunk
    chunks.append((start_frame, labeled[-1]['frame_num'], prev_label))
    return chunks

def save_chunk(video_path, out_dir, chunk_idx, start_frame, end_frame, label, fps):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'chunk_{chunk_idx:03d}_{label}_{start_frame:06d}_{end_frame:06d}.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for f in range(start_frame, end_frame+1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    out.release()
    cap.release()
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Split video into chunks based on label changes.")
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--annotations', type=str, required=True, help='Path to the annotation JSON file')
    parser.add_argument('--outdir', type=str, default='.', help='Output directory')
    args = parser.parse_args()

    labeled = parse_annotations(args.annotations)
    if not labeled:
        print("No labeled data found.")
        return

    chunks = get_chunks(labeled)
    print(f"Found {len(chunks)} chunks.")

    # Prepare output folders
    cleaning_dir = os.path.join(args.outdir, 'cleaning')
    noncleaning_dir = os.path.join(args.outdir, 'noncleaning')

    # Get video info
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    for idx, (start, end, label) in enumerate(chunks):
        if label == "Cleaning":
            out_dir = cleaning_dir
        else:
            out_dir = noncleaning_dir
        save_chunk(args.video, out_dir, idx+1, start, end, label, fps)

if __name__ == "__main__":
    main()
