import os
import cv2
import argparse
import yaml
import random

def load_label_names(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # Try common keys for label names
    if "names" in data:
        return data["names"]
    elif "labels" in data:
        return data["labels"]
    else:
        raise ValueError("No 'names' or 'labels' field found in data.yaml")

def get_class_colors(num_classes):
    random.seed(42)
    colors = []
    for _ in range(num_classes):
        colors.append(tuple([int(x) for x in random.sample(range(50, 256), 3)]))
    return colors

def draw_bboxes(image, bbox_file, label_names, class_colors):
    with open(bbox_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_min, y_min, x_max, y_max = parts
        class_id_int = int(class_id)
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        pt1 = (int(x_min), int(y_min))
        pt2 = (int(x_max), int(y_max))
        color = class_colors[class_id_int % len(class_colors)]
        label = label_names[class_id_int] if class_id_int < len(label_names) else str(class_id)
        cv2.rectangle(image, pt1, pt2, color, 2)
        cv2.putText(image, label, (pt1[0], pt1[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image

def main():
    parser = argparse.ArgumentParser(description="Visualize 10 images with bounding boxes and label names.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--bbox_dir", type=str, required=True, help="Directory with bbox annotation files")
    parser.add_argument("--yaml", type=str, required=True, help="Path to data.yaml with label names")
    args = parser.parse_args()

    label_names = load_label_names(args.yaml)
    class_colors = get_class_colors(len(label_names))

    img_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    img_files = sorted(img_files)[:10]

    for img_name in img_files:
        img_path = os.path.join(args.img_dir, img_name)
        bbox_path = os.path.join(args.bbox_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(bbox_path):
            print(f"No bbox file for {img_name}, skipping.")
            continue
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read {img_path}, skipping.")
            continue
        image = draw_bboxes(image, bbox_path, label_names, class_colors)
        cv2.imshow("Bounding Boxes", image)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit early
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
