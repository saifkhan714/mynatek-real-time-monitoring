import os
import argparse
import random
import yaml
import shutil

def split_files(file_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    random.shuffle(file_list)
    n = len(file_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:]
    return train_files, val_files, test_files

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

def generate_segmentation_yaml(train_dir, val_dir, test_dir, class_names, out_yaml_path):
    data = {
        "task": "segment",
        "train": os.path.abspath(train_dir),
        "val": os.path.abspath(val_dir),
        "test": os.path.abspath(test_dir),
        "nc": len(class_names),
        "names": class_names
    }
    with open(out_yaml_path, "w") as f:
        yaml.dump(data, f)
    print(f"Segmentation YOLO data.yaml generated at {out_yaml_path}")

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
    generate_segmentation_yaml(
        train_dir, val_dir, test_dir,
        class_names,
        custom_yaml
    )
    return custom_yaml

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation model with automatic data split.")
    parser.add_argument("--data", type=str, required=True, help="Path to original data.yaml")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--lbl_dir", type=str, required=True, help="Directory with segmentation labels (YOLO polygon format)")
    parser.add_argument("--project", type=str, default="yolov8_seg_train", help="Project directory for results")
    parser.add_argument("--weights", type=str, default="yolov8x-seg.pt", help="YOLOv8 segmentation model weights")
    args = parser.parse_args()

    # Hardcoded training parameters (can be made configurable)
    epochs = 100
    imgsz = 640
    batch = 16
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

    model = YOLO("yolo11x-seg.pt")
    model.train(
        data=custom_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=args.project,
        lr0=lr0,
        cache=False,
        degrees=15.0,
        scale=0.5,
        translate=0.1,
        shear=2.0,
        perspective=0.001,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        mosaic=mosaic,
        close_mosaic=close_mosaic,
        patience=patience,
        save_period=10,
        copy_paste=0.4  # Enable copy-paste augmentation (default 0, set to 0.2)
    )

if __name__ == "__main__":
    main()
