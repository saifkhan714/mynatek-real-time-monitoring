import os
import json
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def convert_json_to_stgcn_format(json_files_dir, output_train_file, output_test_file, sequence_length=30, test_size=0.2):
    """
    Convert JSON keypoint files from your labeling tool to ST-GCN compatible format
    """
    all_sequences = []
    all_labels = []
    
    # Define class mapping based on your labels
    class_names = ["Cleaning", "Not-Cleaning"]
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    
    # List all JSON files
    json_files = glob(os.path.join(json_files_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        with open(json_file, 'r') as f:
            sequences = json.load(f)
        
        for sequence in sequences:
            label = sequence["label"]
            if label not in class_mapping:
                print(f"Skipping unknown label: {label}")
                continue
                
            skeletons = sequence["skeletons"]
            
            # Process each sequence in chunks of sequence_length with overlap
            stride = sequence_length // 2  # 50% overlap
            for i in range(0, max(1, len(skeletons) - sequence_length + 1), stride):
                chunk = skeletons[i:i+sequence_length]
                
                # Skip if chunk is too short and pad if needed
                if len(chunk) < sequence_length:
                    if len(chunk) < sequence_length // 2:
                        continue  # Skip if too short
                    
                    # Pad with the last frame repeated
                    pad_length = sequence_length - len(chunk)
                    last_frame = chunk[-1]
                    padding = [last_frame] * pad_length
                    chunk.extend(padding)
                
                # Convert to numpy array [sequence_length, num_joints, channels(x,y,score)]
                sequence_array = np.array(chunk, dtype=np.float32)
                
                # Create one-hot encoded label
                num_classes = len(class_mapping)
                label_one_hot = np.zeros(num_classes, dtype=np.float32)
                label_one_hot[class_mapping[label]] = 1
                
                all_sequences.append(sequence_array)
                all_labels.append(label_one_hot)
    
    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)
    
    print(f"Converted {len(all_sequences)} sequences with shape {all_sequences.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        all_sequences, all_labels, test_size=test_size, random_state=42, stratify=all_labels
    )
    
    # Save training data
    with open(output_train_file, 'wb') as f:
        pickle.dump((X_train, y_train), f)
    print(f"Training data saved to {output_train_file} with {len(X_train)} sequences")
    
    # Save test data
    with open(output_test_file, 'wb') as f:
        pickle.dump((X_test, y_test), f)
    print(f"Test data saved to {output_test_file} with {len(X_test)} sequences")
    
    return class_names

def main():
    # Set your directories
    json_files_dir = "./json_data"  # Directory with your JSON files
    output_train_file = "./data/processed/train_data.pkl"
    output_test_file = "./data/processed/test_data.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
    
    # Convert files
    class_names = convert_json_to_stgcn_format(json_files_dir, output_train_file, output_test_file)
    
    # Save class names for reference
    with open("./data/class_names.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")

if __name__ == "__main__":
    main()