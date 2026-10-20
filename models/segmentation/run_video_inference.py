import argparse
import os

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from models.segmentation.inference import SegmentationInference

def main():
    parser = argparse.ArgumentParser(description="Run segmentation inference on a video using a trained YOLO model.")
    parser.add_argument('--model', type=str, required=True, help='Path to best.pt weights file')
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, required=True, help='Path to save output video')
    args = parser.parse_args()

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    inference = SegmentationInference(args.model)
    inference.infer_video(args.input, output_path=args.output)
    print(f"Inference complete. Output saved to {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
