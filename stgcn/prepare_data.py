import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
import alphaction
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Constants
SEQUENCE_LENGTH = 30  # Number of frames to use for each action sequence
NUM_JOINTS = 14  # Number of joints in skeleton
NUM_CHANNELS = 3  # x, y, confidence score

def extract_poses_from_video(video_path, pose_detector):
    """Extract poses from a video file using AlphaPose."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    all_poses = []

    for i in tqdm(range(frame_count), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Get pose estimation from frame
        results = pose_detector.detect(frame)

        if len(results) > 0:
            # Take the pose with highest confidence if multiple detected
            pose = results[0]['keypoints']
            all_poses.append(pose)
        else:
            # If no pose detected, add zeros
            all_poses.append(np.zeros((NUM_JOINTS, NUM_CHANNELS)))

    cap.release()

    return np.array(all_poses), (width, height)

def create_action_sequences(poses, labels, sequence_length=SEQUENCE_LENGTH, overlap=0.5):
    """
    Create fixed-length sequences from pose data with specified overlap.

    Args:
        poses: List of pose arrays from videos
        labels: List of labels corresponding to each video
        sequence_length: Length of each sequence
        overlap: Fraction of overlap between consecutive sequences

    Returns:
        sequences: Array of shape (n_sequences, sequence_length, joints, channels)
        sequence_labels: Array of one-hot encoded labels for each sequence
    """
    sequences = []
    sequence_labels = []

    stride = int(sequence_length * (1 - overlap))

    for pose_data, label in zip(poses, labels):
        if len(pose_data) < sequence_length:
            # If video is shorter than sequence length, pad with zeros
            padding = np.zeros((sequence_length - len(pose_data), NUM_JOINTS, NUM_CHANNELS))
            pose_data = np.concatenate([pose_data, padding], axis=0)
            sequences.append(pose_data)
            sequence_labels.append(label)
        else:
            # Create overlapping sequences
            for i in range(0, len(pose_data) - sequence_length + 1, stride):
                seq = pose_data[i:i+sequence_length]
                sequences.append(seq)
                sequence_labels.append(label)

    # Convert to one-hot encoded labels
    class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down', 'Stand up', 'Sit down', 'Fall Down']
    num_classes = len(class_names)

    one_hot_labels = np.zeros((len(sequence_labels), num_classes))
    for i, label in enumerate(sequence_labels):
        one_hot_labels[i, class_names.index(label)] = 1

    return np.array(sequences), one_hot_labels

def load_dataset(data_dir, pose_detector):
    """
    Load videos from the data directory, extract poses, and create sequences.

    Args:
        data_dir: Directory containing video files organized in class folders
        pose_detector: Pose detection model

    Returns:
        sequences: Processed pose sequences
        labels: One-hot encoded labels
    """
    all_poses = []
    all_labels = []

    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for class_name in class_dirs:
        class_dir = os.path.join(data_dir, class_name)
        video_files = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

        for video_file in video_files:
            video_path = os.path.join(class_dir, video_file)
            poses, frame_size = extract_poses_from_video(video_path, pose_detector)

            if poses is not None:
                all_poses.append(poses)
                all_labels.append(class_name)

    sequences, labels = create_action_sequences(all_poses, all_labels)
    return sequences, labels

def main():
    # Initialize pose detector
    pose_detector = alphaction.models.get_pose_model()

    # Define data directories
    train_dir = 'data/train'
    test_dir = 'data/test'

    # Create output directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)

    # Process training data
    print("Processing training data...")
    train_sequences, train_labels = load_dataset(train_dir, pose_detector)

    # Process test data
    print("Processing test data...")
    test_sequences, test_labels = load_dataset(test_dir, pose_detector)

    # Save processed data
    with open('data/processed/train_data.pkl', 'wb') as f:
        pickle.dump((train_sequences, train_labels), f)

    with open('data/processed/test_data.pkl', 'wb') as f:
        pickle.dump((test_sequences, test_labels), f)

    print("Data preparation complete!")
    print(f"Training sequences: {train_sequences.shape}")
    print(f"Testing sequences: {test_sequences.shape}")

if __name__ == "__main__":
    main()