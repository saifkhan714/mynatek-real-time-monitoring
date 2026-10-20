import argparse
import os
import torch
import numpy as np
import logging
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

parser = argparse.ArgumentParser(description="Run inference with fine-tuned SAM2 model")
parser.add_argument("--input", type=str, required=True, help="Path to input image or directory")
parser.add_argument("--output", type=str, required=True, help="Path to save output mask/image or directory")
parser.add_argument("--log", type=str, default="inference.log", help="Path to log file")
parser.add_argument("--type", type=str, default="image", choices=["image"], help="Type of input (only 'image' supported currently)")
parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda", "gpu"], help="Device to run inference")
parser.add_argument("--model_size", type=str, default="large", choices=["small", "large"], help="Model size")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to fine-tuned model checkpoint (e.g., checkpoints/training_current_exp/model_full_large_final.pth)")
parser.add_argument("--grid_size", type=int, default=10, help="Grid size for prompt points (default: 10)")
parser.add_argument("--min_area", type=int, default=500, help="Minimum mask area to keep (default: 500 pixels)")
args = parser.parse_args()

# Determine device
device = torch.device("cuda" if args.device in ["cuda", "gpu"] and torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Select config based on model size
if args.model_size == "large":
    model_cfg = "sam2_hiera_l.yaml"
elif args.model_size == "small":
    model_cfg = "sam2_hiera_s.yaml"
else:
    raise ValueError("Invalid model size")

# Load model architecture and weights
if args.checkpoint.endswith(".pth"):
    print(f"Loading full model object from {args.checkpoint}")
    model = torch.load(args.checkpoint, map_location=device)
    print("✅ Full model loaded from .pth file.")
elif args.checkpoint.endswith(".pt"):
    print(f"Loading model state_dict from {args.checkpoint}")
    # Build model architecture
    if args.model_size == "large":
        model_cfg = "sam2_hiera_l.yaml"
    elif args.model_size == "small":
        model_cfg = "sam2_hiera_s.yaml"
    else:
        raise ValueError("Invalid model size")
    # Build model architecture without loading weights
    model = build_sam2(model_cfg, None, device=device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    print("✅ Model architecture built and state_dict loaded from .pt file.")
else:
    raise ValueError("Please provide a .pth (full model) or .pt (state_dict) file as the checkpoint.")

model.eval()
predictor = SAM2ImagePredictor(model)


# Setup logging
logging.basicConfig(filename=args.log, level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
def log(msg):
    print(msg)
    logging.info(msg)

def process_image(image_path, output_dir, grid_size=10, min_area=500):
    try:
        image = cv2.imread(image_path)
        if image is None:
            log(f"ERROR: Could not read image {image_path}")
            return
        image = image[..., ::-1]
        image = image.copy()
        predictor.set_image(image)

        # Predict all masks for a grid of points
        height, width, _ = image.shape
        N = grid_size  # grid size
        ys = np.linspace(0, height - 1, N, dtype=int)
        xs = np.linspace(0, width - 1, N, dtype=int)
        all_masks = []
        for y in ys:
            for x in xs:
                input_point = [[[x, y]]]
                input_label = [[1]]
                masks, _, _ = predictor.predict(
                    point_coords=np.array(input_point),
                    point_labels=np.array(input_label),
                    multimask_output=True
                )
                for m in masks:
                    all_masks.append(m)

        # Remove duplicate masks (by pixel content) and filter by area
        unique_masks = []
        mask_hashes = set()
        for m in all_masks:
            mask_bin = (m > 0.5).astype(np.uint8)
            mask_hash = hash(mask_bin.tobytes())
            area = mask_bin.sum()
            if mask_hash not in mask_hashes and area >= min_area:
                unique_masks.append(mask_bin)
                mask_hashes.add(mask_hash)

        # Create a subdirectory for this image's results
        img_base = os.path.splitext(os.path.basename(image_path))[0]
        img_out_dir = os.path.join(output_dir, img_base)
        os.makedirs(img_out_dir, exist_ok=True)

        # Save all unique masks in the subdirectory
        for i, mask in enumerate(unique_masks):
            cv2.imwrite(os.path.join(img_out_dir, f"mask_{i}.png"), mask * 255)

        log(f"✅ Saved {len(unique_masks)} unique masks (area >= {min_area}) for {image_path} to {img_out_dir}")

        # Create and save overlay with all masks colored (non-overlapping, each mask gets a unique color)
        overlay = image.copy()
        color_overlay = np.zeros_like(overlay)
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(unique_masks))]
        for i, mask in enumerate(unique_masks):
            color_overlay[mask > 0] = colors[i]
        # Overlay all masks on the image with alpha blending
        alpha = 0.5
        overlay_all = cv2.addWeighted(image, 1 - alpha, color_overlay, alpha, 0)
        cv2.imwrite(os.path.join(img_out_dir, "overlay_all.png"), overlay_all[..., ::-1])
        log(f"✅ Saved overlay with all masks for {image_path} to {img_out_dir}")

        # Save overlay of all masks on the original image (redundant, but for clarity)
        cv2.imwrite(os.path.join(img_out_dir, "overlay_on_image.png"), overlay_all[..., ::-1])
        log(f"✅ Saved overlay on original image for {image_path} to {img_out_dir}")

    except Exception as e:
        log(f"ERROR processing {image_path}: {e}")

# Batch or single image inference
if os.path.isdir(args.input):
    os.makedirs(args.output, exist_ok=True)
    image_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_path in image_files:
        process_image(img_path, args.output)
else:
    os.makedirs(args.output, exist_ok=True)
    process_image(args.input, args.output, grid_size=args.grid_size, min_area=args.min_area)

# ==========================
# Example CLI commands:
#
# Inference with small model on GPU (if available):
# python INFER.py --input sample_image.jpg --output inference_output --type image --device auto --model_size small --checkpoint checkpoints/training_current_exp/model_full_small_final.pth
#
# Inference with large model on GPU:
# python INFER.py --input sample_image.jpg --output inference_output --type image --device auto --model_size large --checkpoint checkpoints/training_current_exp/model_full_large_final.pth
#
# Inference with a fine-tuned large model:
# python INFER.py --input sample_image.jpg --output inference_output --type image --device cpu --model_size large --checkpoint checkpoints/training_current_exp/model_full_large_final.pth
#
# Inference on CPU:
# python INFER.py --input sample_image.jpg --output inference_output --type image --device cpu --model_size small --checkpoint checkpoints/training_current_exp/model_full_small_final.pth
#
# (Adjust --input, --output, --type, --device, --model_size, and --checkpoint as needed)
# ==========================
