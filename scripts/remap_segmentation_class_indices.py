"""
remap_segmentation_class_indices.py

Purpose:
    Remaps non-contiguous or non-zero-based class indices in YOLO segmentation label files (polygon format) to contiguous indices starting from 0.
    Also updates the data.yaml file to reflect the new class index mapping.

Usage:
    python scripts/remap_segmentation_class_indices.py --yaml_in path/to/original_data.yaml --yaml_out path/to/new_data.yaml --label_dir path/to/segmentation_labels

    --yaml_in:   Path to the original data.yaml (with possibly non-contiguous class indices in 'names').
    --yaml_out:  Path to save the new data.yaml (with contiguous class indices in 'names').
    --label_dir: Directory with YOLO segmentation label .txt files to update in-place (polygon format).

    The script will:
      - Remap all segmentation label files in-place to use contiguous class indices.
      - Write a new data.yaml with the updated class index mapping.
"""

import os
import yaml

def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data

def save_yaml(data, yaml_path):
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

def build_index_mapping(names_dict):
    # names_dict: {old_idx: class_name, ...}
    sorted_items = sorted(names_dict.items(), key=lambda x: int(x[0]))
    old_to_new = {}
    new_names = []
    for new_idx, (old_idx, class_name) in enumerate(sorted_items):
        old_to_new[int(old_idx)] = new_idx
        new_names.append(class_name)
    return old_to_new, new_names

def remap_segmentation_label_file(label_path, old_to_new):
    with open(label_path, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        old_idx = int(parts[0])
        if old_idx not in old_to_new:
            continue  # skip invalid class
        new_idx = old_to_new[old_idx]
        new_line = " ".join([str(new_idx)] + parts[1:]) + "\n"
        new_lines.append(new_line)
    with open(label_path, "w") as f:
        f.writelines(new_lines)

def remap_all_segmentation_labels(label_dir, old_to_new):
    for fname in os.listdir(label_dir):
        if fname.endswith(".txt"):
            remap_segmentation_label_file(os.path.join(label_dir, fname), old_to_new)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Remap non-contiguous class indices to contiguous for YOLO segmentation labels.")
    parser.add_argument("--yaml_in", type=str, required=True, help="Path to original data.yaml")
    parser.add_argument("--yaml_out", type=str, required=True, help="Path to save new data.yaml")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory with YOLO segmentation label .txt files to update")
    args = parser.parse_args()

    data = load_yaml(args.yaml_in)
    names_dict = data["names"]
    old_to_new, new_names = build_index_mapping(names_dict)
    print("Remapping indices:", old_to_new)

    # Remap all segmentation label files in-place
    remap_all_segmentation_labels(args.label_dir, old_to_new)

    # Write new data.yaml with contiguous list
    data["names"] = new_names
    save_yaml(data, args.yaml_out)
    print(f"Saved new data.yaml with contiguous class indices to {args.yaml_out}")

if __name__ == "__main__":
    main()
