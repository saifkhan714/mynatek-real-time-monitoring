"""
seg_to_box.py

Purpose:
    Converts segmentation polygon annotations to bounding box annotations.
    For each polygon in the annotation file, computes the tightest bounding box and writes it as (class_id, x_min, y_min, x_max, y_max).

Usage:
    python scripts/seg_to_box.py --ann_dir path/to/polygon_annotations --img_dir path/to/images --out_dir path/to/bbox_annotations

    --ann_dir: Directory with .txt annotation files (polygon format: class_id x1 y1 x2 y2 ...).
    --img_dir: Directory with images (used to scale normalized coordinates).
    --out_dir: Directory to save bbox annotation files (same base name as input, .txt extension).

    Each output .txt file will have lines in the format:
        <class_id> <x_min> <y_min> <x_max> <y_max>
    with coordinates in absolute pixel values.
"""

import os
import cv2
import argparse

def parse_annotation_line(line):
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
    return class_id, points

def polygon_to_bbox(points, img_w, img_h):
    xs = [x * img_w for x, y in points]
    ys = [y * img_h for x, y in points]
    x_min = max(0, min(xs))
    y_min = max(0, min(ys))
    x_max = min(img_w - 1, max(xs))
    y_max = min(img_h - 1, max(ys))
    return x_min, y_min, x_max, y_max

def process_annotation_file(txt_path, img_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        return
    img_h, img_w = img.shape[:2]
    with open(txt_path, "r") as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        class_id, points = parse_annotation_line(line)
        x_min, y_min, x_max, y_max = polygon_to_bbox(points, img_w, img_h)
        bboxes.append(f"{class_id} {x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f}\n")
    with open(out_path, "w") as f:
        f.writelines(bboxes)

def main():
    parser = argparse.ArgumentParser(description="Convert segmentation polygons to bounding boxes.")
    parser.add_argument("--ann_dir", type=str, required=True, help="Directory with .txt annotation files")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save bbox annotation files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for fname in os.listdir(args.ann_dir):
        if fname.endswith(".txt"):
            txt_path = os.path.join(args.ann_dir, fname)
            img_name = os.path.splitext(fname)[0] + ".jpg"
            img_path = os.path.join(args.img_dir, img_name)
            out_path = os.path.join(args.out_dir, fname)
            process_annotation_file(txt_path, img_path, out_path)
            print(f"Processed {fname}")

if __name__ == "__main__":
    main()
