import os
import json
import pickle
import numpy as np

# USER CONFIGURATION
JSON_DIR = "/home/developer/STGCN_data"  # Directory containing your JSON files
OUT_DIR = "custom_data"  # Output directory for PKL files
SEQUENCE_LENGTH = 30  # Number of frames per sequence
CLASS_NAMES = ["Not-Cleaning", "Cleaning"]  # List your classes here (case-sensitive to match your JSON)
NUM_KEYPOINTS = 17  # Number of keypoints per frame

# Split ratios (should sum to 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def label_to_onehot(label, class_names):
    onehot = np.zeros(len(class_names), dtype=np.float32)
    idx = class_names.index(label)
    onehot[idx] = 1.0
    return onehot

def process_json_file(json_path, class_names, seq_len):
    with open(json_path, "r") as f:
        data = json.load(f)
    frames = [frame for frame in data if "joints" in frame and frame["joints"]]
    features = []
    labels = []
    # Slide a window of seq_len over the frames
    for i in range(0, len(frames) - seq_len + 1):
        seq = frames[i:i+seq_len]
        joints_seq = [np.array(seq[j]["joints"][0], dtype=np.float32) for j in range(seq_len)]
        # Accept (NUM_KEYPOINTS, 2) or (NUM_KEYPOINTS, 3)
        if any(j.shape not in [(NUM_KEYPOINTS, 2), (NUM_KEYPOINTS, 3)] for j in joints_seq):
            continue
        # If (x, y), add dummy score; if (x, y, score), keep as is
        for idx, j in enumerate(joints_seq):
            if j.shape == (NUM_KEYPOINTS, 2):
                j = np.concatenate([j, np.ones((NUM_KEYPOINTS, 1), dtype=np.float32)], axis=1)
                joints_seq[idx] = j
        label = seq[-1]["label"]
        if label not in class_names:
            continue
        features.append(np.stack(joints_seq, axis=0))  # (seq_len, 17, 3)
        labels.append(label_to_onehot(label, class_names))
    return features, labels

def split_data(features, labels, train_ratio, val_ratio, test_ratio, seed=42):
    n = len(features)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]
    return (
        [features[i] for i in train_idx], [labels[i] for i in train_idx],
        [features[i] for i in val_idx], [labels[i] for i in val_idx],
        [features[i] for i in test_idx], [labels[i] for i in test_idx]
    )

def main():
    print("=== JSON to ST-GCN PKL Data Conversion with Splitting ===")
    print(f"JSON_DIR: {JSON_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")
    print(f"SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
    print(f"CLASS_NAMES: {CLASS_NAMES}")
    print(f"NUM_KEYPOINTS: {NUM_KEYPOINTS}")
    print(f"Split ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    all_features = []
    all_labels = []
    json_files = [fname for fname in os.listdir(JSON_DIR) if fname.endswith(".json")]
    print(f"Found {len(json_files)} JSON files to process.")
    for fname in json_files:
        fpath = os.path.join(JSON_DIR, fname)
        print(f"Processing {fname} ...")
        feats, labs = process_json_file(fpath, CLASS_NAMES, SEQUENCE_LENGTH)
        all_features.extend(feats)
        all_labels.extend(labs)
        print(f"  {len(feats)} valid sequences extracted from {fname}")
    if not all_features:
        print("No valid sequences found. Check your data and configuration.")
        return
    print("All files processed. Splitting data into train/val/test...")
    (train_feats, train_labs,
     val_feats, val_labs,
     test_feats, test_labs) = split_data(all_features, all_labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    os.makedirs(OUT_DIR, exist_ok=True)
    # Save each split
    splits = [
        ("train", train_feats, train_labs),
        ("val", val_feats, val_labs),
        ("test", test_feats, test_labs)
    ]
    for split_name, feats, labs in splits:
        features = np.stack(feats, axis=0)
        labels = np.stack(labs, axis=0)
        out_path = os.path.join(OUT_DIR, f"cleaning_dataset_{split_name}.pkl")
        print(f"Saving {split_name} set: {features.shape[0]} sequences to {out_path} ...")
        with open(out_path, "wb") as f:
            pickle.dump((features, labels), f)
    print("=== Conversion and splitting complete. ===")
    print("NOTE: If your JSON data already contains (x, y, score), no dummy score is added. If only (x, y), a dummy score of 1.0 is added for compatibility.")

if __name__ == "__main__":
    main()

"""
INSTRUCTIONS:
1. Edit JSON_DIR to point to your folder of JSON pose files.
2. Edit OUT_DIR to where you want the .pkl files saved.
3. Edit CLASS_NAMES to match the labels in your JSON files (case-sensitive).
4. Optionally adjust SEQUENCE_LENGTH and split ratios.
5. Run this script:
   python data_preparation/json_to_stgcn_pkl.py
6. Use the resulting *_train.pkl, *_val.pkl, *_test.pkl files for training, validation, and testing.

NOTE: If your JSON data already contains (x, y, score) for each keypoint, the score is preserved. If only (x, y), a dummy score of 1.0 is added for compatibility.
"""
