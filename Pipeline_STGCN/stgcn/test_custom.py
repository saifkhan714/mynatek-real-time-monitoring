import os
import sys
import torch
import pickle
import numpy as np
from torch.utils import data
from tqdm import tqdm

# Ensure project root is in sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stgcn.Models import TwoStreamSpatialTemporalGraph

# ========== USER CONFIGURATION ==========
TEST_PATH = "data_preparation/custom_data/cleaning_dataset_test.pkl"  # Path to your test .pkl file
MODEL_PATH = "stgcn_custom_saved/tsstg-model.pth"          # Path to trained model
CLASS_NAMES = ["Cleaning", "Not-Cleaning"]
NUM_KEYPOINTS = 17
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='cpu or cuda')
args, _ = parser.parse_known_args()
DEVICE = args.device
print(f"Using device: {DEVICE}")
BATCH_SIZE = 32
# ========================================

def load_dataset(test_path, batch_size):
    with open(test_path, "rb") as f:
        features, labels = pickle.load(f)
    features = torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2)
    labels = torch.tensor(labels, dtype=torch.float32)
    test_set = data.TensorDataset(features, labels)
    test_loader = data.DataLoader(test_set, batch_size)
    return test_loader

def accuracy_batch(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).float().mean().item()

def main():
    print(f"Loading test data from {TEST_PATH} ...")
    test_loader = load_dataset(TEST_PATH, BATCH_SIZE)
    num_class = len(CLASS_NAMES)
    print(f"Number of classes: {num_class}")
    print(f"Number of keypoints: {NUM_KEYPOINTS}")

    # Model
    graph_args = {'strategy': 'spatial', 'layout': 'coco'}
    model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_trues = []
    test_loss, test_acc = 0.0, 0.0
    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for pts, lbs in tqdm(test_loader, desc="Testing"):
            mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            mot, pts, lbs = mot.to(DEVICE), pts.to(DEVICE), lbs.to(DEVICE)
            out = model((pts, mot))
            loss = criterion(out, lbs)
            test_loss += loss.item()
            test_acc += accuracy_batch(out.detach().cpu(), lbs.detach().cpu())
            all_preds.append(out.cpu().numpy())
            all_trues.append(lbs.cpu().numpy())
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Per-class accuracy
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    pred_labels = preds.argmax(1)
    true_labels = trues.argmax(1)
    for i, cname in enumerate(CLASS_NAMES):
        mask = true_labels == i
        if mask.sum() > 0:
            acc = (pred_labels[mask] == i).mean()
            print(f"Class '{cname}': {mask.sum()} samples, Accuracy: {acc:.4f}")
        else:
            print(f"Class '{cname}': 0 samples")

    # Confusion matrix
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, pred_labels)
        print("Confusion Matrix:")
        print(cm)
    except ImportError:
        print("scikit-learn not installed, skipping confusion matrix.")

if __name__ == "__main__":
    main()

"""
INSTRUCTIONS:
1. Ensure your trained model is at MODEL_PATH and your test data is at DATA_PATH.
2. Run:
   python stgcn/test_custom.py
3. The script will print overall and per-class accuracy, and a confusion matrix if scikit-learn is installed.
"""
