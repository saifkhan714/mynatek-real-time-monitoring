import os
import random
import shutil
import argparse

def split_dataset(
    image_dir,
    mask_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    move_files=False
):
    """
    Splits a dataset of images and masks into train, val, and test folders.

    Args:
        image_dir (str): Path to images.
        mask_dir (str): Path to masks (must match image names).
        output_dir (str): Output root directory for splits.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
        seed (int): Random seed for reproducibility.
        move_files (bool): If True, move files instead of copying.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.seed(seed)
    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train, val, test = images[:n_train], images[n_train:n_train+n_val], images[n_train+n_val:]

    for split, split_list in zip(['Train', 'Val', 'Test'], [train, val, test]):
        img_out = os.path.join(output_dir, split, 'Image')
        mask_out = os.path.join(output_dir, split, 'Instance')
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)
        for img_name in split_list:
            src_img = os.path.join(image_dir, img_name)
            dst_img = os.path.join(img_out, img_name)
            mask_name = img_name[:-4] + '.png'
            src_mask = os.path.join(mask_dir, mask_name)
            dst_mask = os.path.join(mask_out, mask_name)
            if os.path.exists(src_img) and os.path.exists(src_mask):
                if move_files:
                    shutil.move(src_img, dst_img)
                    shutil.move(src_mask, dst_mask)
                else:
                    shutil.copy(src_img, dst_img)
                    shutil.copy(src_mask, dst_mask)
            else:
                print(f"WARNING: Missing image or mask for {img_name}, skipping.")
    print(f"✅ Split complete: {len(train)} train, {len(val)} val, {len(test)} test images.")

    # Optionally, write file lists
    for split, split_list in zip(['train', 'val', 'test'], [train, val, test]):
        with open(os.path.join(output_dir, f"{split}_file_list.txt"), "w") as f:
            for img_name in split_list:
                f.write(f"{img_name}\n")
    print("✅ File lists saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test folders and file lists.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--move_files", action="store_true", help="Move files instead of copying")
    args = parser.parse_args()

    split_dataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        move_files=args.move_files
    )

# ==========================
# Example CLI command:
#
# Split debug_data/Train/Image and debug_data/Train/Instance into train/val/test in a new directory:
# python DATA.py --image_dir debug_data/Train/Image --mask_dir debug_data/Train/Instance --output_dir shuffled_data_split
#
# (Adjust --train_ratio, --val_ratio, --test_ratio, --move_files as needed)
# ==========================
