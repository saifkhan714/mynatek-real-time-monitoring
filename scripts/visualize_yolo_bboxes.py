"""
visualize_yolo_bboxes.py

Purpose:
    Visualizes YOLO-format bounding boxes on 10 random images from a directory, displaying them with class names.
    The images are shown in a window (not saved). Press any key to advance, or ESC to exit early.

Usage:
    python scripts/visualize_yolo_bboxes.py --img_dir path/to/images --yaml path/to/data.yaml --bbox_dir path/to/yolo_labels

    --img_dir:  Directory containing images.
    --yaml:     YAML file with class names (must have a 'names' field).
    --bbox_dir: Directory with YOLO-format label files (same base name as images, .txt extension).

    The script will display each image with bounding boxes and class labels overlaid.
"""

import os
import random
import yaml
import cv2

def load_class_names(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data["names"] if "names" in data else []

def draw_yolo_bboxes(img, label_path, class_names, color=(0,255,0)):
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_c, y_c, bw, bh = parts
            class_id = int(class_id)
            x_c, y_c, bw, bh = map(float, (x_c, y_c, bw, bh))
            x1 = int((x_c - bw/2) * w)
            y1 = int((y_c - bh/2) * h)
            x2 = int((x_c + bw/2) * w)
            y2 = int((y_c + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = class_names[class_id] if class_id < len(class_names) else str(class_id)
            cv2.putText(img, label, (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize YOLO-format bounding boxes on 10 random images (display only, do not save).")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--yaml", type=str, required=True, help="YAML file with class names")
    parser.add_argument("--bbox_dir", type=str, required=True, help="Directory with YOLO-format bbox label files")
    args = parser.parse_args()

    class_names = load_class_names(args.yaml)
    img_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(img_files) == 0:
        print("No images found in", args.img_dir)
        return
    sample_imgs = random.sample(img_files, min(10, len(img_files)))
    for fname in sample_imgs:
        img_path = os.path.join(args.img_dir, fname)
        label_path = os.path.join(args.bbox_dir, os.path.splitext(fname)[0] + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        img_viz = draw_yolo_bboxes(img, label_path, class_names)
        cv2.imshow("YOLO BBoxes", img_viz)
        print(f"Showing: {img_path} (press any key for next, or ESC to exit)")
        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
