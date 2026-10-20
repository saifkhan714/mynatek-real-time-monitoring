import os
import sys
import torch
import pickle
import numpy as np
from torch.utils import data
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
import random
from torchvision import transforms
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

# Ensure project root is in sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stgcn.Models import TwoStreamSpatialTemporalGraph

# ========== ENHANCED CONFIGURATION ==========
# Paths to your pre-split datasets
TRAIN_DATA_PATH = "data_preparation/custom_data/cleaning_dataset_train.pkl"
VAL_DATA_PATH = "data_preparation/custom_data/cleaning_dataset_val.pkl"
TEST_DATA_PATH = "data_preparation/custom_data/cleaning_dataset_test.pkl"

# List your action classes here
CLASS_NAMES = ["Cleaning", "Not-Cleaning"]

# Number of keypoints in your skeleton (17 for YOLOv11)
NUM_KEYPOINTS = 17

# Enhanced Training parameters
EPOCHS = 10  # Increased epochs for better convergence
BATCH_SIZE = 64  # Increased batch size
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # L2 regularization
MOMENTUM = 0.9  # For SGD optimizer

# Class imbalance handling
USE_CLASS_WEIGHTS = True
USE_DATA_BALANCING = False  # Disable oversampling
USE_DATA_AUGMENTATION = True  # Enable data augmentation
MINORITY_CLASS_WEIGHT = 2.0  # Give more weight to minority class

# Learning rate scheduler parameters
LR_PATIENCE = 5  # Reduced patience for faster learning
LR_FACTOR = 0.5
MIN_LR = 1e-6

# Data augmentation parameters
ROTATION_RANGE = 15  # degrees
SCALE_RANGE = 0.1  # 10% scale variation
TEMPORAL_JITTER = 2  # frames
NOISE_LEVEL = 0.05  # Add small random noise to keypoints

# Model architecture parameters
DROPOUT_RATE = 0.5  # Increased dropout to prevent overfitting
HIDDEN_CHANNELS = [64, 128, 256, 512]  # Increased network capacity
USE_ATTENTION = True  # Enable attention mechanism

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 10  # Reduced patience for faster convergence
EARLY_STOPPING_MIN_DELTA = 0.001

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu or cuda')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer choice')
parser.add_argument('--scheduler', type=str, default='cosine', choices=['reduce_on_plateau', 'cosine'], help='learning rate scheduler')
args, _ = parser.parse_known_args()
DEVICE = args.device
print(f"Using device: {DEVICE}")

SAVE_DIR = "stgcn_custom_saved"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "tsstg-model-enhanced.pth")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "tsstg-model-best.pth")

# ========== Data Augmentation ==========
class PoseAugmentation:
    def __init__(self, rotation_range, scale_range, temporal_jitter, noise_level):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.temporal_jitter = temporal_jitter
        self.noise_level = noise_level

    def random_rotation(self, x):
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        rad = np.radians(angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Apply rotation to x,y coordinates
        x_rot = x.copy()
        # Reshape for rotation: (T, V, C) -> (T*V, 2)
        coords = x_rot[:, :, :2].reshape(-1, 2)
        # Apply rotation
        rotated_coords = np.dot(coords, rotation_matrix.T)
        # Reshape back: (T*V, 2) -> (T, V, 2)
        x_rot[:, :, :2] = rotated_coords.reshape(x_rot.shape[0], x_rot.shape[1], 2)
        return x_rot

    def random_scale(self, x):
        scale = random.uniform(1 - self.scale_range, 1 + self.scale_range)
        x_scaled = x.copy()
        x_scaled[:, :, :2] *= scale
        return x_scaled

    def temporal_jittering(self, x):
        if self.temporal_jitter > 0:
            jitter = random.randint(-self.temporal_jitter, self.temporal_jitter)
            if jitter != 0:  # Only apply jitter if it's not zero
                if jitter > 0:
                    # Pad at the end and take from the beginning
                    x = np.pad(x, ((0, jitter), (0, 0), (0, 0)), mode='edge')[:x.shape[0]]
                else:
                    # Pad at the beginning and take from the end
                    x = np.pad(x, ((abs(jitter), 0), (0, 0), (0, 0)), mode='edge')[abs(jitter):]
        return x

    def add_noise(self, x):
        noise = np.random.normal(0, self.noise_level, x.shape)
        return x + noise

    def __call__(self, x):
        x = self.random_rotation(x)
        x = self.random_scale(x)
        x = self.temporal_jittering(x)
        x = self.add_noise(x)
        return x

# ========== Enhanced Model Architecture ==========
class EnhancedTwoStreamSpatialTemporalGraph(TwoStreamSpatialTemporalGraph):
    def __init__(self, graph_args, num_class, edge_importance_weighting=True, **kwargs):
        super().__init__(graph_args, num_class, edge_importance_weighting, **kwargs)
        
        # Add attention mechanism
        if USE_ATTENTION:
            self.attention = nn.Sequential(
                nn.Linear(256 * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 256 * 2),
                nn.Sigmoid()
            )
        
        # Enhanced classifier
        self.fcn = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, num_class)
        )

    def forward(self, inputs):
        out1 = self.pts_stream(inputs[0])
        out2 = self.mot_stream(inputs[1])
        
        concat = torch.cat([out1, out2], dim=-1)
        
        if USE_ATTENTION:
            attention_weights = self.attention(concat)
            concat = concat * attention_weights
        
        out = self.fcn(concat)
        return torch.sigmoid(out)

def calculate_class_weights(labels):
    """Calculate class weights based on class distribution with emphasis on minority class"""
    class_counts = Counter(np.argmax(labels, axis=1))
    total_samples = len(labels)
    
    # Calculate base weights
    weights = {i: total_samples / (len(class_counts) * count) 
              for i, count in class_counts.items()}
    
    # Identify minority class
    minority_class = min(class_counts.items(), key=lambda x: x[1])[0]
    
    # Apply additional weight to minority class
    weights[minority_class] *= MINORITY_CLASS_WEIGHT
    
    print(f"\nClass weights before minority boost:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {total_samples / (len(class_counts) * class_counts[i]):.2f}")
    
    print(f"\nClass weights after minority boost:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {weights[i]:.2f}")
    
    return torch.tensor([weights[i] for i in range(len(CLASS_NAMES))], dtype=torch.float32)

def load_custom_dataset(train_path, val_path, test_path, batch_size):
    """
    Load pre-split datasets from separate .pkl files
    """
    # Load training data
    with open(train_path, "rb") as f:
        train_features, train_labels = pickle.load(f)
    
    # Load validation data
    with open(val_path, "rb") as f:
        val_features, val_labels = pickle.load(f)
    
    # Load test data
    with open(test_path, "rb") as f:
        test_features, test_labels = pickle.load(f)
    
    # Print class distribution
    print("\nClass distribution in training set:")
    class_dist = Counter(np.argmax(train_labels, axis=1))
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {class_dist[i]} samples")
    
    # Convert to torch tensors and permute to (N, C, T, V)
    train_features = torch.tensor(train_features, dtype=torch.float32).permute(0, 3, 1, 2)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    
    val_features = torch.tensor(val_features, dtype=torch.float32).permute(0, 3, 1, 2)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)
    
    test_features = torch.tensor(test_features, dtype=torch.float32).permute(0, 3, 1, 2)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)
    
    # Check keypoint dimension
    if train_features.shape[-1] != NUM_KEYPOINTS:
        raise ValueError(f"Your data has {train_features.shape[-1]} keypoints, but NUM_KEYPOINTS is set to {NUM_KEYPOINTS}")
    
    print(f"\nDataset sizes:")
    print(f"Training set: {len(train_features)} samples")
    print(f"Validation set: {len(val_features)} samples")
    print(f"Test set: {len(test_features)} samples")
    
    # Apply data augmentation to training set
    if USE_DATA_AUGMENTATION:
        aug = PoseAugmentation(ROTATION_RANGE, SCALE_RANGE, TEMPORAL_JITTER, NOISE_LEVEL)
        train_features_np = train_features.numpy()
        
        # Print shape for debugging
        print(f"\nTraining features shape before augmentation: {train_features_np.shape}")
        
        # Apply augmentation to each sample
        augmented_features = []
        for x in train_features_np:
            # Reshape to (T, V, C) for augmentation
            x_reshaped = x.transpose(1, 2, 0)  # (C, T, V) -> (T, V, C)
            x_aug = aug(x_reshaped)
            # Reshape back to (C, T, V)
            x_aug = x_aug.transpose(2, 0, 1)
            augmented_features.append(x_aug)
        
        train_features = np.array(augmented_features)
        print(f"Training features shape after augmentation: {train_features.shape}")
        
        train_features = torch.tensor(train_features, dtype=torch.float32)
    
    # Create datasets
    train_set = data.TensorDataset(train_features, train_labels)
    val_set = data.TensorDataset(val_features, val_labels)
    test_set = data.TensorDataset(test_features, test_labels)
    
    # Create data loaders
    train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size)
    test_loader = data.DataLoader(test_set, batch_size)
    
    return train_loader, val_loader, test_loader, train_labels

def evaluate_model(model, val_loader, criterion, device):
    """Evaluate model and return detailed metrics"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for pts, lbs in val_loader:
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            mot, pts, lbs = mot.to(device), pts.to(device), lbs.to(device)
            
            out = model((pts, mot))
            loss = criterion(out, lbs)
            
            val_loss += loss.item()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(lbs.argmax(1).cpu().numpy())
    
    val_loss /= len(val_loader)
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    
    return val_loss, cm, report

def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).float().mean().item()

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Loading data from pre-split files...")
    train_loader, val_loader, test_loader, train_labels = load_custom_dataset(
        TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE
    )
    num_class = len(CLASS_NAMES)
    print(f"Number of classes: {num_class}")
    print(f"Number of keypoints: {NUM_KEYPOINTS}")

    # Enhanced Model
    graph_args = {'strategy': 'spatial', 'layout': 'coco'}
    model = EnhancedTwoStreamSpatialTemporalGraph(graph_args, num_class).to(DEVICE)
    
    # Optimizer selection
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:  # sgd
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    if args.scheduler == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=LR_PATIENCE, factor=LR_FACTOR, min_lr=MIN_LR)
    else:  # cosine
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=MIN_LR)
    
    # Calculate class weights if enabled
    if USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(train_labels.numpy())
        print("\nClass weights:", class_weights)
        criterion = torch.nn.BCELoss(weight=class_weights.to(DEVICE))
    else:
        criterion = torch.nn.BCELoss()
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for pts, lbs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            mot, pts, lbs = mot.to(DEVICE), pts.to(DEVICE), lbs.to(DEVICE)
            
            out = model((pts, mot))
            loss = criterion(out, lbs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += accuracy_batch(out.detach().cpu(), lbs.detach().cpu())
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation with detailed metrics
        val_loss, cm, report = evaluate_model(model, val_loader, criterion, DEVICE)
        
        # Learning rate scheduling
        if args.scheduler == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Early stopping check
        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"\nNew best model saved with validation loss: {best_val_loss:.4f}")
        else:
            early_stopping_counter += 1
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, MODEL_SAVE_PATH)
        
        # Early stopping
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    print("\nTraining complete.")
    print(f"Best model saved to {BEST_MODEL_PATH}")

if __name__ == "__main__":
    main() 