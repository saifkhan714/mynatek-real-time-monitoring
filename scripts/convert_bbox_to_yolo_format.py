"""
convert_bbox_to_yolo_format.py

Purpose:
    Converts bounding box label files from absolute coordinates (x_min, y_min, x_max, y_max) to YOLO format
    (x_center, y_center, width, height), normalized to [0,1] using the corresponding image dimensions.

Usage:
    python scripts/convert_bbox_to_yolo_format.py --label_dir path/to/abs_bbox_labels --image_dir path/to/images --out_dir path/to/yolo_labels

    This will create YOLO-format label files in the specified output directory, ready for YOLOv8 training.
    Each output label file will have lines in the format:
        <class_id> <x_center> <y_center> <width> <height>
    where all values are normalized to [0,1].
"""

import os
import cv2

def convert_bbox_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height

def process_label_file(label_path, image_path, out_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return
    img_h, img_w = img.shape[:2]
    new_lines = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_min, y_min, x_max, y_max = map(float, parts[1:])
            x_c, y_c, w, h = convert_bbox_to_yolo((x_min, y_min, x_max, y_max), img_w, img_h)
            # Clamp values to [0,1]
            x_c = min(max(x_c, 0), 1)
            y_c = min(max(y_c, 0), 1)
            w = min(max(w, 0), 1)
            h = min(max(h, 0), 1)
            new_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
    with open(out_path, "w") as f:
        f.writelines(new_lines)

def convert_dir(label_dir, image_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(label_dir):
        if fname.endswith(".txt"):
            label_path = os.path.join(label_dir, fname)
            image_path = os.path.join(image_dir, os.path.splitext(fname)[0] + ".jpg")
            out_path = os.path.join(out_dir, fname)
            process_label_file(label_path, image_path, out_path)
            print(f"Converted {fname}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert bbox label files from absolute to YOLO format.")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory with bbox label .txt files")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save YOLO-format label files")
    args = parser.parse_args()

    convert_dir(args.label_dir, args.image_dir, args.out_dir)
