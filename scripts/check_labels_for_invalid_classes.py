"""
check_labels_for_invalid_classes.py

Purpose:
    Checks all YOLO label files in a directory for invalid class indices (i.e., class indices not in the range 0 to num_classes-1).

Usage:
    python scripts/check_labels_for_invalid_classes.py --lbl_dir path/to/labels --num_classes 44

    Prints the file name, line number, and offending line for any label file with an out-of-range class index.
    If all labels are valid, it will report that as well.
"""

import os

def find_invalid_labels(lbl_dir, num_classes):
    invalid_files = []
    for fname in os.listdir(lbl_dir):
        if fname.endswith(".txt"):
            fpath = os.path.join(lbl_dir, fname)
            with open(fpath, "r") as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        class_idx = int(parts[0])
                        if class_idx >= num_classes or class_idx < 0:
                            invalid_files.append((fpath, i+1, line.strip()))
                    except Exception:
                        invalid_files.append((fpath, i+1, line.strip()))
    return invalid_files

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check YOLO label files for invalid class indices.")
    parser.add_argument("--lbl_dir", type=str, required=True, help="Directory with label .txt files")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes (max valid class index is num_classes-1)")
    args = parser.parse_args()

    invalids = find_invalid_labels(args.lbl_dir, args.num_classes)
    if invalids:
        print("Invalid class indices found in label files:")
        for fpath, lineno, line in invalids:
            print(f"{fpath} (line {lineno}): {line}")
    else:
        print("No invalid class indices found in label files.")
