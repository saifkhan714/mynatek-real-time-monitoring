import os
import time
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import traceback
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from stgcn.Models import TwoStreamSpatialTemporalGraph

# Enhanced Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

save_folder = 'output/models'
epochs = 100  # Increased epochs
batch_size = 16  # Adjusted batch size
learning_rate = 0.001
weight_decay = 1e-4  # L2 regularization

# Updated class names for your specific case
class_names = ['Cleaning', 'Not-Cleaning']
num_class = len(class_names)

class DataAugmentation:
    """Data augmentation for skeleton sequences."""
    @staticmethod
    def random_rotation(x, angle_range=(-10, 10)):
        """Random rotation of skeleton joints."""
        angle = np.random.uniform(angle_range[0], angle_range[1])
        rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]
        ])
        x_rotated = np.zeros_like(x)
        for i in range(x.shape[0]):
            x_rotated[i, :2] = np.dot(x[i, :2].T, rotation_matrix).T
            x_rotated[i, 2] = x[i, 2]  # Keep z coordinate unchanged
        return x_rotated

    @staticmethod
    def random_scale(x, scale_range=(0.9, 1.1)):
        """Random scaling of skeleton joints."""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return x * scale

    @staticmethod
    def random_jitter(x, sigma=0.01):
        """Add random noise to skeleton joints."""
        noise = np.random.normal(0, sigma, x.shape)
        return x + noise

    @staticmethod
    def temporal_jitter(x, max_shift=2):
        """Random temporal shift of the sequence."""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return x
            
        # Get the original shape
        frames, joints, channels = x.shape
        
        if shift > 0:
            # Shift forward: pad at the beginning, crop from the end
            padded = np.pad(x, ((shift, 0), (0, 0), (0, 0)), mode='edge')
            return padded[:-shift, :, :]
        else:
            # Shift backward: pad at the end, crop from the beginning
            padded = np.pad(x, ((0, -shift), (0, 0), (0, 0)), mode='edge')
            return padded[-shift:, :, :]

def augment_data(features, labels):
    """Apply data augmentation to the dataset."""
    augmented_features = []
    augmented_labels = []
    
    for i in range(len(features)):
        # Original data
        augmented_features.append(features[i])
        augmented_labels.append(labels[i])
        
        # Apply augmentations
        x = features[i]
        
        # Rotation
        x_rot = DataAugmentation.random_rotation(x)
        augmented_features.append(x_rot)
        augmented_labels.append(labels[i])
        
        # Scale
        x_scale = DataAugmentation.random_scale(x)
        augmented_features.append(x_scale)
        augmented_labels.append(labels[i])
        
        # Jitter
        x_jitter = DataAugmentation.random_jitter(x)
        augmented_features.append(x_jitter)
        augmented_labels.append(labels[i])
        
        # Temporal jitter
        x_temp = DataAugmentation.temporal_jitter(x)
        augmented_features.append(x_temp)
        augmented_labels.append(labels[i])
    
    return np.array(augmented_features), np.array(augmented_labels)

def load_data(data_files, batch_size=1, split_size=0.2):
    """Load dataset and create data loaders with data augmentation."""
    try:
        features, labels = [], []

        for file in data_files:
            print(f"Attempting to load data from: {file}")
            if not os.path.exists(file):
                print(f"ERROR: File not found: {file}")
                continue

            with open(file, 'rb') as f:
                try:
                    fts, lbs = pickle.load(f)
                    print(f"Successfully loaded data from {file}")
                    print(f"  - Features shape: {fts.shape}")
                    print(f"  - Labels shape: {lbs.shape}")
                    features.append(fts)
                    labels.append(lbs)
                except Exception as e:
                    print(f"ERROR loading data from {file}: {str(e)}")
                    traceback.print_exc()

        if not features or not labels:
            print("ERROR: No data loaded from any file")
            return None, None

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Apply data augmentation
        print("Applying data augmentation...")
        features, labels = augment_data(features, labels)
        print(f"Augmented data shape: {features.shape}")
        print(f"Augmented labels shape: {labels.shape}")

        # Split data
        if split_size > 0:
            from sklearn.model_selection import train_test_split
            x_train, x_valid, y_train, y_valid = train_test_split(
                features, labels, test_size=split_size, random_state=42, stratify=labels
            )

            # Create datasets
            train_set = data.TensorDataset(
                torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2),
                torch.tensor(y_train, dtype=torch.float32)
            )

            valid_set = data.TensorDataset(
                torch.tensor(x_valid, dtype=torch.float32).permute(0, 3, 1, 2),
                torch.tensor(y_valid, dtype=torch.float32)
            )

            train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
            valid_loader = data.DataLoader(valid_set, batch_size)

            print(f"Created data loaders: {len(train_loader)} training batches, {len(valid_loader)} validation batches")
            return train_loader, valid_loader
        else:
            train_set = data.TensorDataset(
                torch.tensor(features, dtype=torch.float32).permute(0, 3, 1, 2),
                torch.tensor(labels, dtype=torch.float32)
            )
            train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
            return train_loader, None

    except Exception as e:
        print(f"ERROR in load_data: {str(e)}")
        traceback.print_exc()
        return None, None

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (pts, labels) in enumerate(train_loader):
        # Create motion input
        motion = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        
        # Move data to device
        motion = motion.to(device)
        pts = pts.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model((pts, motion))
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        
    return total_loss / len(train_loader), correct / total

def validate(model, valid_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for pts, labels in valid_loader:
            motion = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
            
            motion = motion.to(device)
            pts = pts.to(device)
            labels = labels.to(device)
            
            outputs = model((pts, motion))
            loss = criterion(outputs, labels)
            
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
    
    return total_loss / len(valid_loader), correct / total

def main():
    try:
        os.makedirs(save_folder, exist_ok=True)

        # Load data
        print("Loading data...")
        train_file = 'data/processed/train_data.pkl'
        test_file = 'data/processed/test_data.pkl'

        train_loader, valid_loader = load_data([train_file], batch_size, split_size=0.2)
        if train_loader is None:
            print("Failed to load training data. Exiting.")
            return

        test_loader, _ = load_data([test_file], batch_size)
        if test_loader is None:
            print("Failed to load test data. Exiting.")
            return

        # Initialize model
        print(f"Initializing model on {device}...")
        graph_args = {'strategy': 'spatial'}
        model = TwoStreamSpatialTemporalGraph(graph_args, num_class).to(device)

        # Define optimizer and loss function
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate schedulers
        scheduler1 = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        scheduler2 = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        criterion = torch.nn.BCELoss()

        # Training loop
        best_val_acc = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        print("Starting training...")
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            if valid_loader is not None:
                val_loss, val_acc = validate(model, valid_loader, criterion, device)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Update learning rate
                scheduler1.step(val_loss)
                scheduler2.step()
                
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(save_folder, 'best_model.pth'))
                    print(f'New best model saved with validation accuracy: {val_acc:.4f}')
            else:
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        # Save final model
        torch.save(model.state_dict(), os.path.join(save_folder, 'final_model.pth'))
        print("Training complete!")

        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        if valid_loader is not None:
            plt.plot(val_losses, label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        if valid_loader is not None:
            plt.plot(val_accs, label='Val Acc')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, 'training_history.png'))
        plt.close()

    except Exception as e:
        print(f"ERROR in main: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main()