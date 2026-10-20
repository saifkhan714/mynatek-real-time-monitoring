import os
import argparse
import random
import yaml

def split_files(file_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    random.shuffle(file_list)
    n = len(file_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:]
    return train_files, val_files, test_files

import shutil

def write_split_files_and_copy(split_files, img_dir, lbl_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in split_files:
        img_src = os.path.join(img_dir, fname + ".jpg")
        lbl_src = os.path.join(lbl_dir, fname + ".txt")
        img_dst = os.path.join(out_dir, fname + ".jpg")
        lbl_dst = os.path.join(out_dir, fname + ".txt")
        if os.path.exists(img_src):
            shutil.copy2(img_src, img_dst)
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, lbl_dst)

def generate_classic_yaml(train_dir, val_dir, test_dir, class_names, out_yaml_path):
    data = {
        "train": os.path.abspath(train_dir),
        "val": os.path.abspath(val_dir),
        "test": os.path.abspath(test_dir),
        "nc": len(class_names),
        "names": class_names
    }
    with open(out_yaml_path, "w") as f:
        yaml.dump(data, f)
    print(f"Classic YOLO data.yaml generated at {out_yaml_path}")

def prepare_splits_and_yaml(img_dir, lbl_dir, orig_yaml, out_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    img_files = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg")]
    lbl_files = [os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt")]
    base_files = sorted(list(set(img_files) & set(lbl_files)))

    train_files, val_files, test_files = split_files(base_files, train_ratio, val_ratio, test_ratio)

    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    test_dir = os.path.join(out_dir, "test")
    write_split_files_and_copy(train_files, img_dir, lbl_dir, train_dir)
    write_split_files_and_copy(val_files, img_dir, lbl_dir, val_dir)
    write_split_files_and_copy(test_files, img_dir, lbl_dir, test_dir)

    # Read class names from orig_yaml
    with open(orig_yaml, "r") as f:
        orig_data = yaml.safe_load(f)
    class_names = orig_data["names"] if "names" in orig_data else []

    custom_yaml = os.path.join(out_dir, "custom_data.yaml")
    generate_classic_yaml(
        train_dir, val_dir, test_dir,
        class_names,
        custom_yaml
    )
    return custom_yaml

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on custom bounding box data with automatic data split.")
    parser.add_argument("--data", type=str, required=True, help="Path to original data.yaml")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--lbl_dir", type=str, required=True, help="Directory with labels")
    parser.add_argument("--project", type=str, default="yolov8_train", help="Project directory for results")
    args = parser.parse_args()

    # Hardcoded training parameters
    epochs = 100
    imgsz = 640
    batch = 16
    weights = "yolo11x.pt"
    mosaic = 1.0
    close_mosaic = 10
    patience = 20
    lr0 = 0.001

    # Prepare splits and custom yaml
    split_dir = os.path.join(args.project, "splits")
    custom_yaml = prepare_splits_and_yaml(
        img_dir=args.img_dir,
        lbl_dir=args.lbl_dir,
        orig_yaml=args.data,
        out_dir=split_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )

    # Import here to avoid import error if ultralytics is not installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install the Ultralytics YOLOv8 package with: pip install ultralytics")
        return

    model = YOLO(weights)
    model.train(
        data=custom_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=args.project,
        lr0=lr0,                # Set learning rate as requested
        cache=False,            # Disable caching as requested
        degrees=15.0,           # ±15 degrees (default)
        scale=0.5,              # Random scaling (default)
        translate=0.1,          # Random translation (default)
        shear=2.0,              # Random shear (default)
        perspective=0.001,      # Perspective transform (default)
        hsv_h=0.02,             # Hue jitter (default)
        hsv_s=0.7,              # Saturation jitter (default)
        hsv_v=0.4,              # Value jitter (default)
        flipud=0.1,             # Vertical flip probability (default)
        fliplr=0.5,             # Horizontal flip probability (default)
        mosaic=mosaic,          # Mosaic augmentation probability
        close_mosaic=close_mosaic, # Epoch to stop using mosaic
        patience=patience,          # Early stopping patience
        save_period=10,             # Save checkpoint every 10 epochs
        mixup=0.2,                  # Enable mixup augmentation (default 0, set to 0.2)
        cutmix=0.2,                 # Enable cutmix augmentation (default 0, set to 0.2)
        auto_augment="augmix"  # Enable auto_augment policy (e.g., "randaugment")
    )

if __name__ == "__main__":
    main()
