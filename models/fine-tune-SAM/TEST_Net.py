# segment image region using  fine tune model
# See Train.py on how to fine tune/train the model
import numpy as np
import torch
import cv2
import os
import argparse
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def get_device(device_arg):
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg in ["cuda", "gpu"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def get_model_config_and_ckpt(model_size):
    if model_size == "small":
        return "sam2_hiera_s.yaml", "checkpoints/sam2_hiera_small.pt"
    elif model_size == "large":
        print("⚠️  Large model selected. Ensure your GPU has at least 24GB memory. Consider lowering batch size if you encounter OOM errors.")
        return "sam2_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"
    else:
        raise ValueError("Invalid model_size. Choose 'small' or 'large'.")

def read_image(image_path, mask_path): # read and resize image and mask
    img = cv2.imread(image_path)[...,::-1]  # read image as rgb
    mask = cv2.imread(mask_path,0) # mask of the region we want to segment
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)),interpolation=cv2.INTER_NEAREST)
    return img, mask

def get_points(mask,num_points): # Sample points inside the input mask
    points=[]
    for i in range(num_points):
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)

import urllib.request

def download_checkpoint_if_missing(checkpoint, model_size):
    if not os.path.exists(checkpoint):
        print(f"Checkpoint file not found: {checkpoint}")
        # Placeholder URLs - replace with actual URLs for your models
        urls = {
            "small": "https://your-model-server/checkpoints/sam2_hiera_small.pt",
            "large": "https://your-model-server/checkpoints/sam2.1_hiera_large.pt"
        }
        url = urls.get(model_size)
        if url:
            print(f"Attempting to download {model_size} model checkpoint from {url} ...")
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            try:
                urllib.request.urlretrieve(url, checkpoint)
                print(f"Downloaded checkpoint to {checkpoint}")
            except Exception as e:
                print(f"ERROR: Failed to download checkpoint: {e}")
                print(f"Please manually download the checkpoint from {url} and place it at {checkpoint}")
                exit(1)
        else:
            print(f"No download URL available for model size '{model_size}'. Please manually place the checkpoint at {checkpoint}")
            exit(1)

def main():
    parser = argparse.ArgumentParser(description="SAM2 Test Script (CPU/GPU, Small/Large Model)")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned model weights (e.g., model.torch)")
    parser.add_argument("--output_dir", type=str, default="test_output", help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu"], help="Device to use: auto, cpu, cuda/gpu")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "large"], help="Model size: small or large")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of points/segment to sample")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device(args.device)
    model_cfg, sam2_checkpoint = get_model_config_and_ckpt(args.model_size)
    # Removed download_checkpoint_if_missing; only user-provided model will be tested

    image, mask = read_image(args.image, args.mask)
    input_points = get_points(mask, args.num_samples)

    # Only load model if checkpoint is provided
    if args.checkpoint is None:
        print("No checkpoint provided. Skipping model loading and inference.")
        return
    if not args.checkpoint.endswith(".pth"):
        raise ValueError("Please provide a .pth file (full model object) as the checkpoint.")
    print(f"Loading full model object from {args.checkpoint}")
    sam2_model = torch.load(args.checkpoint, map_location=device)
    print("✅ Full model loaded from .pth file.")
    predictor = SAM2ImagePredictor(sam2_model)

    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0],1])
        )

    masks = masks[:,0].astype(bool)
    shorted_masks = masks[np.argsort(scores[:,0])][::-1].astype(bool)

    seg_map = np.zeros_like(shorted_masks[0],dtype=np.uint8)
    occupancy_mask = np.zeros_like(shorted_masks[0],dtype=bool)
    for i in range(shorted_masks.shape[0]):
        mask = shorted_masks[i]
        if (mask*occupancy_mask).sum()/mask.sum()>0.15: continue
        mask[occupancy_mask]=0
        seg_map[mask]=i+1
        occupancy_mask[mask]=1

    rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for id_class in range(1,seg_map.max()+1):
        rgb_image[seg_map == id_class] = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]

    cv2.imwrite(os.path.join(args.output_dir, "annotation.png"), rgb_image)
    cv2.imwrite(os.path.join(args.output_dir, "mix.png"), (rgb_image/2+image/2).astype(np.uint8))

if __name__ == "__main__":
    main()

# ==========================
# Example CLI commands:
#
# Test with small model on GPU (if available):
# python TEST_Net.py --device auto --model_size small --checkpoint model.torch --image sample_image.jpg --mask sample_mask.png --output_dir test_output/
#
# Test with large model on GPU:
# python TEST_Net.py --device auto --model_size large --checkpoint model.torch --image sample_image.jpg --mask sample_mask.png --output_dir test_output/
#
# Test on CPU:
# python TEST_Net.py --device cpu --model_size small --checkpoint model.torch --image sample_image.jpg --mask sample_mask.png --output_dir test_output/
#
# (Adjust --device, --model_size, --checkpoint, --image, --mask, --output_dir as needed)
# ==========================
