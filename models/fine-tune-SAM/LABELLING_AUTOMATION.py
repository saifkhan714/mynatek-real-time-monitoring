import argparse
import os
import cv2
import torch
import numpy as np
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "sam2")))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def run_yolo_detection(yolo_model, image):
    # Use Ultralytics YOLOv8 detection model for inference
    results = yolo_model.predict(source=image, save=False, show=False, conf=0.25)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id,
                "class_name": r.names[class_id] if hasattr(r, "names") else str(class_id),
                "conf": conf
            })
    return detections

def run_sam_on_bbox(predictor, image, bbox, grid_size=5, min_area=500):
    x_min, y_min, x_max, y_max = bbox
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return []
    # Always set the image before each predict call to avoid state issues
    predictor.set_image(crop)
    height, width, _ = crop.shape
    N = grid_size
    ys = np.linspace(0, height - 1, N, dtype=int)
    xs = np.linspace(0, width - 1, N, dtype=int)
    all_masks = []
    for y in ys:
        for x in xs:
            predictor.set_image(crop)  # Ensure image is set before each predict
            input_point = [[[x, y]]]
            input_label = [[1]]
            masks, _, _ = predictor.predict(
                point_coords=np.array(input_point),
                point_labels=np.array(input_label),
                multimask_output=True
            )
            for m in masks:
                all_masks.append(m)
    # Remove duplicate masks and filter by area
    unique_masks = []
    mask_hashes = set()
    for m in all_masks:
        mask_bin = (m > 0.5).astype(np.uint8)
        mask_hash = hash(mask_bin.tobytes())
        area = mask_bin.sum()
        if mask_hash not in mask_hashes and area >= min_area:
            unique_masks.append(mask_bin)
            mask_hashes.add(mask_hash)
    # Map masks back to original image coordinates
    mapped_masks = []
    for mask in unique_masks:
        full_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # Resize mask to fit the bbox region if needed
        h_box = y_max - y_min
        w_box = x_max - x_min
        if mask.shape != (h_box, w_box):
            mask_resized = cv2.resize(mask, (w_box, h_box), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        full_mask[y_min:y_max, x_min:x_max] = mask_resized
        mapped_masks.append(full_mask)
    return mapped_masks

def extract_bbox_and_polygon(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    bbox = [x_min, y_min, x_max, y_max]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) >= 3:
            polygons.append(cnt.squeeze(1).tolist())
    return bbox, polygons

def save_mask_and_metadata(mask, bbox, polygons, frame_id, mask_id, output_dir, metadata_list, class_name="object"):
    mask_filename = f"frame_{frame_id}_mask_{mask_id}.png"
    mask_path = os.path.join(output_dir, mask_filename)
    cv2.imwrite(mask_path, mask * 255)
    metadata = {
        "frame_id": frame_id,
        "mask_id": mask_id,
        "mask_path": mask_filename,
        "bbox": bbox,
        "polygons": polygons,
        "class": class_name
    }
    metadata_list.append(metadata)

def process_bbox_with_sam(predictor, image, bbox, class_name, grid_size, min_area, frame_id, mask_id, output_dir):
    masks = run_sam_on_bbox(predictor, image, bbox, grid_size, min_area)
    if not masks:
        print(f"[INFO] No valid masks found for class '{class_name}' at {bbox}.")
        return None
    # Select the mask with the largest area
    areas = [mask.sum() for mask in masks]
    max_idx = int(np.argmax(areas))
    best_mask = masks[max_idx]
    bbox2, polygons = extract_bbox_and_polygon(best_mask)
    if bbox2 is not None and polygons:
        metadata_list = []
        save_mask_and_metadata(best_mask, bbox2, polygons, frame_id=frame_id, mask_id=mask_id, output_dir=output_dir, metadata_list=metadata_list, class_name=class_name)
        # Save overlay for this object
        overlay = image.copy()
        color_overlay = np.zeros_like(overlay)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        color_overlay[best_mask > 0] = color
        alpha = 0.5
        overlay_img = cv2.addWeighted(image, 1 - alpha, color_overlay, alpha, 0)
        overlay_path = os.path.join(output_dir, f"overlay_{class_name}_mask_{mask_id}.png")
        cv2.imwrite(overlay_path, overlay_img[..., ::-1])
        print(f"[INFO] Saved overlay for class '{class_name}' mask {mask_id} to {overlay_path}")
        print(f"[INFO] Saved best mask for class '{class_name}' at {bbox} (area={areas[max_idx]}).")
        return metadata_list[0]
    return None

def process_image_with_yolo_async(image_path, yolo_model, predictor, output_dir, grid_size=5, min_area=500):
    print(f"\n[INFO] Processing image with YOLO+SAM (async): {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image {image_path}")
        return []
    detections = run_yolo_detection(yolo_model, image)
    print(f"[INFO] YOLO detected {len(detections)} objects.")
    # Draw and save YOLO predicted bounding boxes on the image
    yolo_bb_img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        conf = det["conf"]
        color = (0, 255, 0)
        cv2.rectangle(yolo_bb_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(yolo_bb_img, f"{class_name} {conf:.2f}", (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    yolo_bb_path = os.path.join(output_dir, "yolo_pred_bboxes.png")
    cv2.imwrite(yolo_bb_path, yolo_bb_img)
    print(f"[INFO] Saved YOLO predicted bounding boxes to {yolo_bb_path}")

    metadata_list = []
    mask_id = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=min(8, len(detections))) as executor:
        futures = []
        for det in detections:
            bbox = det["bbox"]
            class_name = det["class_name"]
            futures.append(executor.submit(
                process_bbox_with_sam,
                predictor, image, bbox, class_name, grid_size, min_area, 0, mask_id, output_dir
            ))
            mask_id += 1
        for future in as_completed(futures):
            result = future.result()
            if result:
                metadata_list.append(result)
    print(f"[INFO] Saved {len(metadata_list)} best masks and metadata for image.")
    print(f"[INFO] Image processing time: {time.time() - start:.2f} seconds")
    return metadata_list

def main():
    parser = argparse.ArgumentParser(description="Automated labelling with YOLO+SAM2 segmentation model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or video")
    parser.add_argument("--output", type=str, default="labelling_output", help="Directory to save masks and metadata")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu"], help="Device to run inference")
    parser.add_argument("--grid_size", type=int, default=5, help="Grid size for prompt points in SAM (default: 5)")
    parser.add_argument("--min_area", type=int, default=500, help="Minimum mask area to keep (default: 500 pixels)")
    parser.add_argument("--sam_checkpoint", type=str, default="checkpoints/model_full_large_final.pth", help="Path to SAM segmentation model checkpoint")
    parser.add_argument("--yolo_checkpoint", type=str, default="checkpoints/yolo_best.pt", help="Path to YOLO detection model checkpoint")
    args = parser.parse_args()

    # Use user-specified config for SAM2
    model_cfg = r"D:\work\mynatek\mynatek\models\fine-tune-SAM\sam2_configs\sam2_hiera_l.yaml"
    model_ckpt = args.sam_checkpoint

    # Device selection
    import torch
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device in ["cuda", "gpu"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # Load YOLO model (ultralytics)
    try:
        from ultralytics import YOLO
        print(f"[INFO] Loading YOLO detection model: {args.yolo_checkpoint}")
        yolo_model = YOLO(args.yolo_checkpoint)
    except ImportError:
        print("[ERROR] ultralytics package not installed. Please install with 'pip install ultralytics'.")
        return

    # Build and load SAM2 model
    print(f"[INFO] Loading SAM2 segmentation model: {model_ckpt} with config: {model_cfg}")
    # Handle full model .pth or state_dict
    import torch
    if model_ckpt.endswith(".pth"):
        model = torch.load(model_ckpt, map_location=device)
    else:
        model = build_sam2(model_cfg, model_ckpt, device=device)
    model.eval()
    predictor = SAM2ImagePredictor(model)

    os.makedirs(args.output, exist_ok=True)
    metadata_list = []
    if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("[ERROR] Video + YOLO+SAM pipeline not implemented in this script. Please use image input for now.")
        return
    else:
        print("[INFO] Detected image input.")
        metadata_list = process_image_with_yolo_async(args.input, yolo_model, predictor, args.output, args.grid_size, args.min_area)

    # Save metadata as JSON
    json_path = os.path.join(args.output, "labels.json")
    with open(json_path, "w") as f:
        json.dump(metadata_list, f, indent=2)
    print(f"[INFO] ✅ Saved mask metadata to {json_path}")
    print("[INFO] Labelling automation complete.")

if __name__ == "__main__":
    main()

# ---------------------------
# Example Command:
#
# python LABELLING_AUTOMATION.py --input sam_test2.png --output labelling_output --sam_checkpoint checkpoints/model_full_large_final.pth --yolo_checkpoint checkpoints/yolo_best.pt
# ---------------------------
