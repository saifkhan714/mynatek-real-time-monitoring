import os
import argparse
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm

def load_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:5])
            labels.append((cls, x_center, y_center, w, h))
    return labels

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return [x1, y1, x2, y2]

def compute_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 validation inference and metrics.")
    parser.add_argument("--val_txt", type=str, required=True, help="Path to val.txt with image paths")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained YOLOv8 weights (e.g., best.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory with ground truth label .txt files")
    parser.add_argument("--iou_thresh", type=float, default=0.7, help="IoU threshold for TP/FP")
    args = parser.parse_args()

    model = YOLO(args.weights)
    with open(args.val_txt, "r") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_list = []

    for img_path in tqdm(image_paths, desc="Validating"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}, skipping.")
            continue
        img_h, img_w = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(args.label_dir, base + ".txt")
        gt_labels = load_labels(label_path)
        gt_boxes = [yolo_to_xyxy(xc, yc, w, h, img_w, img_h) for (_, xc, yc, w, h) in gt_labels]
        gt_classes = [cls for (cls, _, _, _, _) in gt_labels]

        results = model.predict(source=img, data=args.data, save=False, show=False, conf=0.25)
        pred_boxes = []
        pred_classes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pred_boxes.append([x1, y1, x2, y2])
            pred_classes.append(int(box.cls[0]))

        matched_gt = set()
        matched_pred = set()
        for i, pb in enumerate(pred_boxes):
            found_match = False
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou >= args.iou_thresh and pred_classes[i] == gt_classes[j]:
                    total_tp += 1
                    iou_list.append(iou)
                    matched_gt.add(j)
                    matched_pred.add(i)
                    found_match = True
                    break
            if not found_match:
                total_fp += 1
        total_fn += len(gt_boxes) - len(matched_gt)
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    mean_iou = np.mean(iou_list) if iou_list else 0.0

    print(f"Total GT: {total_gt}, Total Pred: {total_pred}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")

if __name__ == "__main__":
    main()
