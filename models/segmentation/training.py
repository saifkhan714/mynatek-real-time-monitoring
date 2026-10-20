import os
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
from utils import load_data_yaml

class YOLOSegmentationTrainer:
    def __init__(self, weights: str = 'yolo11l-seg', device: str = None):
        """Initialize the YOLO segmentation trainer
        
        Args:
            weights: Path to pretrained weights or model name
            device: Device to use for training (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weights)
        
    def train(self, data_yaml: str, project_dir: str, processed_frames: int, labels_dir: str = None, resume: bool = False):
        """Train the segmentation model on video frames
        
        Args:
            data_yaml: Path to data.yaml file
            project_dir: Directory to save training outputs
            processed_frames: Number of frames to use for training
            labels_dir: Directory containing original label .txt files
            resume: Whether to resume training from last checkpoint
        """
        # Load and validate data.yaml
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            
        if not isinstance(data, dict) or 'names' not in data:
            raise ValueError("Invalid data.yaml format. Must contain 'names' mapping.")
            
        # Create train/val split based on processed frames
        frames_dir = Path(data_yaml).parent
        all_frames = sorted(frames_dir.glob('frame_*.jpg'))
        
        if not all_frames:
            raise ValueError(f"No frames found in {frames_dir}")
            
        # Use first processed_frames for training
        train_frames = all_frames[:processed_frames]
        
        # Create train/val split (80/20)
        split_idx = int(len(train_frames) * 0.8)
        train_split = train_frames[:split_idx]
        val_split = train_frames[split_idx:]
        
        # Create train/val directories
        train_dir = frames_dir / 'train'
        val_dir = frames_dir / 'val'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # Move frames and copy corresponding label files from labels_dir to train/val directories
        import shutil
        for frame in train_split:
            frame.rename(train_dir / frame.name)
            if labels_dir is not None:
                label_file = Path(labels_dir) / frame.with_suffix('.txt').name
                if label_file.exists():
                    shutil.copy(label_file, train_dir / label_file.name)
        for frame in val_split:
            frame.rename(val_dir / frame.name)
            if labels_dir is not None:
                label_file = Path(labels_dir) / frame.with_suffix('.txt').name
                if label_file.exists():
                    shutil.copy(label_file, val_dir / label_file.name)
            
        # Create data.yaml in frames directory with absolute paths
        frames_data_yaml = frames_dir / 'data.yaml'
        frames_data = {
            'path': str(frames_dir.absolute()),
            'train': str(train_dir.absolute()),
            'val': str(val_dir.absolute()),
            'names': data['names']
        }
        
        with open(frames_data_yaml, 'w') as f:
            yaml.dump(frames_data, f, default_flow_style=False)
            
        # Set up checkpoint directory
        checkpoint_dir = Path(project_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Find latest checkpoint if resuming
        latest_checkpoint = None
        if resume:
            checkpoints = sorted(checkpoint_dir.glob('*.pt'))
            if checkpoints:
                latest_checkpoint = str(checkpoints[-1])
                print(f"Resuming from checkpoint: {latest_checkpoint}")
        
        # Train the model
        # === AUGMENTATION CONFIGURATION START ===
        # To add data augmentation, pass an 'augmentations' dictionary to the train() method as shown below.
        # Example (customize as needed for your requirements):
        augmentations = {
            'rotation': 15,  # ±15 degrees
            'scale': [0.5, 1.5],
            'translate': 0.1,  # ±10%
            'perspective': 0.001,
            'crop': 0.1,
            'hsv_h': 0.015,  # color jitter
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'brightness': 0.2,
            'contrast': 0.2,
            'gaussian_noise': 0.01,
            'blur': 1.0
        }
        # Then, add 'augmentations=augmentations' to the train() call below.
        # === AUGMENTATION CONFIGURATION END ===

        results = self.model.train(
            data=str(frames_data_yaml.absolute()),
            epochs=100,  # Number of epochs
            imgsz=640,
            batch=16,  # Batch size
            lr0=0.001,  # Initial learning rate
            lrf=0.01,  # Final learning rate (for cosine schedule, adjust as needed)
            weight_decay=0.0005,  # Weight decay
            patience=10,  # Early stopping patience
            label_smoothing=0.1,  # Label smoothing (adjust as needed)
            dropout=0.3,  # Dropout (if supported by model/config)
            project=str(Path(project_dir).absolute()),
            name='train',
            device=self.device,
            exist_ok=True,
            resume=resume and latest_checkpoint is not None,
            pretrained=latest_checkpoint if resume and latest_checkpoint else None,
            save_period=10,  # Save checkpoint every 10 epochs
            save_dir=str(checkpoint_dir.absolute()),
            degrees=15.0,         # ±15 degrees
            scale=0.5,     # Random scaling
            translate=0.1,
            shear=2.0,            # Random shear
            perspective=0.001,    # Perspective transform
            hsv_h=0.02,           # Hue jitter
            hsv_s=0.7,            # Saturation jitter
            hsv_v=0.4,            # Value jitter
            flipud=0.1,           # Vertical flip probability
            fliplr=0.5,           # Horizontal flip probability
            # <-- UNCOMMENT AND CONFIGURE THIS LINE TO ENABLE AUGMENTATION
            # Note: L2 regularization is typically set via weight_decay. For custom L2, check model config.
            # Note: Learning rate schedule is controlled by lr0/lrf and scheduler type (default is cosine).
            # Note: mAP and AP are reported by YOLO automatically in results.
        )
        
        return results

    def plot_training_curves(self, run_dir: Path):
        """Plot training curves from results"""
        import pandas as pd
        import matplotlib.pyplot as plt
        
        hist = pd.read_csv(run_dir/'results.csv')
        fig, ax = plt.subplots(1, 3, figsize=(18,4))
        
        #plot losses
        ax[0].plot(hist['epoch'], hist['train/box_loss'], label='box')
        ax[0].plot(hist['epoch'], hist['train/mask_loss'], label='mask')
        ax[0].set_title('Train loss')
        ax[0].legend()
        
        #plot precision/recall
        ax[1].plot(hist['epoch'], hist['metrics/precision(B)'], label='P')
        ax[1].plot(hist['epoch'], hist['metrics/recall(B)'], label='R')
        ax[1].set_title('Precision / Recall')
        ax[1].legend()
        
        #plot mAP
        ax[2].plot(hist['epoch'], hist['metrics/mAP50(M)'])
        ax[2].set_title('mAP-50 (mask)')
        plt.show()
