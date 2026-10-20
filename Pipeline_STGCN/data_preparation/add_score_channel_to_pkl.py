import pickle
import numpy as np
import os

SPLITS = ["train", "val", "test"]
IN_DIR = "custom_data"
OUT_DIR = "custom_data"

print("=== Add Score Channel to PKL Data (All Splits) ===")
for split in SPLITS:
    in_path = os.path.join(IN_DIR, f"cleaning_dataset_{split}.pkl")
    out_path = os.path.join(OUT_DIR, f"cleaning_dataset_{split}_with_score.pkl")
    print(f"Processing split: {split}")
    print(f"IN_PATH: {in_path}")
    print(f"OUT_PATH: {out_path}")

    with open(in_path, "rb") as f:
        features, labels = pickle.load(f)

    print(f"Original features shape: {features.shape}")
    # features: (N, T, V, 2) -> (N, T, V, 3)
    if features.shape[-1] == 2:
        N, T, V, _ = features.shape
        print("Adding dummy score channel (all 1.0) to features...")
        score = np.ones((N, T, V, 1), dtype=features.dtype)
        features_new = np.concatenate([features, score], axis=-1)
        print(f"New features shape: {features_new.shape}")
    else:
        features_new = features
        print("Features already have 3 channels. No changes made.")

    print(f"Saving new .pkl with score channel to {out_path} ...")
    with open(out_path, "wb") as f:
        pickle.dump((features_new, labels), f)
print("=== Score channel addition for all splits complete. ===")
