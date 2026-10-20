import os
import sys
import torch
import pickle
import numpy as np
from torch.utils import data
from torch.optim import Adam
from tqdm import tqdm

# Ensure project root is in sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stgcn.Models import TwoStreamSpatialTemporalGraph

# ========== USER CONFIGURATION ==========
# Paths to your custom dataset splits (should be .pkl files with (features, labels))
TRAIN_PATH = "data_preparation/custom_data/cleaning_dataset_train.pkl"
VAL_PATH = "data_preparation/custom_data/cleaning_dataset_val.pkl"

# List your action classes here
CLASS_NAMES = ["Cleaning", "Not-Cleaning"]

# Number of keypoints in your skeleton (17 for YOLOv11, change if needed)
NUM_KEYPOINTS = 17

# Training parameters
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu or cuda')
args, _ = parser.parse_known_args()
DEVICE = args.device
print(f"Using device: {DEVICE}")
SAVE_DIR = "stgcn_custom_saved"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "tsstg-model.pth")
# ========================================

def load_custom_dataset(train_path, val_path, batch_size):
    with open(train_path, "rb") as f:
        train_features, train_labels = pickle.load(f)
    with open(val_path, "rb") as f:
        val_features, val_labels = pickle.load(f)
    train_features = torch.tensor(train_features, dtype=torch.float32).permute(0, 3, 1, 2)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    val_features = torch.tensor(val_features, dtype=torch.float32).permute(0, 3, 1, 2)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)
    # Check keypoint dimension
    if train_features.shape[-1] != NUM_KEYPOINTS or val_features.shape[-1] != NUM_KEYPOINTS:
        raise ValueError(f"Your data has {train_features.shape[-1]} keypoints, but NUM_KEYPOINTS is set to {NUM_KEYPOINTS}. Please check your data and configuration.")
    train_set = data.TensorDataset(train_features, train_labels)
    val_set = data.TensorDataset(val_features, val_labels)
    train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size)
    return train_loader, val_loader

def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).float().mean().item()

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Loading training data from {TRAIN_PATH} ...")
    print(f"Loading validation data from {VAL_PATH} ...")
    train_loader, val_loader = load_custom_dataset(TRAIN_PATH, VAL_PATH, BATCH_SIZE)
    num_class = len(CLASS_NAMES)
    print(f"Number of classes: {num_class}")
    print(f"Number of keypoints: {NUM_KEYPOINTS}")

    # Model
    graph_args = {'strategy': 'spatial', 'layout': 'coco'}
    model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    # Training loop
    for epoch in range(EPOCHS):
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

        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for pts, lbs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
                mot, pts, lbs = mot.to(DEVICE), pts.to(DEVICE), lbs.to(DEVICE)
                out = model((pts, mot))
                loss = criterion(out, lbs)
                val_loss += loss.item()
                val_acc += accuracy_batch(out.detach().cpu(), lbs.detach().cpu())
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save model after each epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

    print("Training complete.")

if __name__ == "__main__":
    main()

"""
INSTRUCTIONS:

1. Prepare your custom dataset as a .pkl file with two objects: (features, labels)
   - features: numpy array of shape (N, T, V, C)
       N = number of samples (sequences)
       T = number of frames per sequence (e.g., 30)
       V = number of keypoints (e.g., 17 for YOLOv11, or your skeleton)
       C = number of channels (2 for x,y or 3 for x,y,score)
   - labels: numpy array of shape (N, num_classes)
       For binary classification, num_classes=2, use one-hot encoding (e.g., [1,0] for not_cleaning, [0,1] for cleaning)

2. Set NUM_KEYPOINTS at the top of this script to match your data (default 17 for YOLOv11).

3. Place your .pkl file in a directory (e.g., custom_data/cleaning_dataset.pkl).

4. Edit DATA_PATH and CLASS_NAMES at the top of this script to match your dataset.

5. Run the script:
   python stgcn/train_custom.py

6. The trained model will be saved to stgcn_custom_saved/tsstg-model.pth

You can now use this model for inference in your pipeline. Make sure to use a model trained with the same number of keypoints as your inference data.
"""
