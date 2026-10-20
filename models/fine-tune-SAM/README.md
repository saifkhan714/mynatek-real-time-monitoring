# Fine-Tune Segment Anything 2 (SAM2) - Project Guide

## Overview

This project provides a streamlined pipeline for fine-tuning and running inference with the Segment Anything 2 (SAM2) model on custom datasets. It includes scripts for training (single and multi-image batch), inference, and evaluation, with all model/config paths hardcoded for ease of use.

---

## Environment Setup

1. **Clone the repository**  
   ```bash
   git clone <your-repo-url>
   cd fine-tune-SAM
   ```

2. **Install dependencies**  
   Make sure you have Python 3.8+ and a CUDA-enabled GPU (recommended).  
   ```bash
   pip install -r requirements.txt
   ```

3. **Install SAM2 (if not already)**  
   ```bash
   pip install -e .
   ```

---

## Directory Structure

- `TRAIN.py` — Main training script (single image per batch)
- `TRAIN_multi_image_batch.py` — Training with multi-image batches
- `INFER.py` — Inference script for generating masks from images
- `TEST_Net.py` — Script for evaluating a trained model
- `sam2_hiera_large.pt`, `sam2_hiera_small.pt` — Pretrained model weights (place in project root)
- `sam2_configs/` — Directory containing YAML config files (do not move)
- `checkpoints/` — Directory where training checkpoints and final models are saved
- `debug_data/`, `shuffled_data/`, etc. — Example data directories
- `inference_output/` — Output directory for inference results

---

## Data Preparation

- Place your training images in `debug_data/Train/Image/` and corresponding masks in `debug_data/Train/Instance/`.
- For validation and test splits, use `debug_data/Val/` and `debug_data/Test/` with the same structure.

---

## Training

### Single Image Batch Training

```bash
python TRAIN.py --device auto --model_size large --epochs 50 --batch_size 4 --steps_per_epoch 100 --data_dir debug_data/ --exp_name training_current_exp
```

- **Model/config loading is hardcoded**:  
  - Large model: `sam2_hiera_large.pt` (root), config: `sam2_configs/sam2_hiera_l.yaml`
  - Small model: `sam2_hiera_small.pt` (root), config: `sam2_configs/sam2_hiera_s.yaml`
- Checkpoints and final models are saved in `checkpoints/<exp_name>/`.

### Multi-Image Batch Training

```bash
python TRAIN_multi_image_batch.py --device auto --model_size large --epochs 50 --batch_size 4 --steps_per_epoch 100 --data_dir debug_data/ --exp_name training_current_exp
```

---

## Inference

Run inference on a single image:

```bash
python INFER.py --input sam_test2.png --output inference_output --device auto --model_size large --checkpoint checkpoints/training_current_exp/model_full_large_final.pth --grid_size 10 --min_area 500
```

- `--grid_size`: Number of prompt points per axis (higher = more masks, slower)
- `--min_area`: Minimum mask area to keep (in pixels)
- Output masks and overlays are saved in `inference_output/<image_name>/`

---

## Evaluation

Evaluate a trained model on a test image and mask:

```bash
python TEST_Net.py --image sample_image.jpg --mask sample_mask.png --checkpoint checkpoints/training_current_exp/model_full_large_final.pth --output_dir test_output/ --device auto --model_size large
```

---

## Notes

- **Model/config loading is hardcoded**: You do not need to specify config file paths; the code will always load from `sam2_configs/` and the project root.
- **Checkpoints**: All training outputs are saved in `checkpoints/<exp_name>/` (default: `training_current_exp`).
- **Data**: Place your data in the expected directory structure for training/validation/testing.

---

## Troubleshooting

- **Config not found**: Ensure `sam2_configs/sam2_hiera_l.yaml` and `sam2_hiera_large.pt` are in the correct locations.
- **CUDA errors**: Make sure you have a compatible GPU and CUDA drivers.
- **Too many/small masks in inference**: Adjust `--grid_size` and `--min_area` in INFER.py.
- **Slow training**: Comment out validation or reduce steps/epochs for quick tests.

---

## Original SAM2 README and Citation

<!-- (Keep the rest of the original README here, including license, contributors, and citation) -->
