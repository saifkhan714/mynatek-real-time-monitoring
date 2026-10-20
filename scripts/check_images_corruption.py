"""
check_images_corruption_dir.py

Purpose:
    Checks all images in a specified directory for corruption using the Pillow library.

Usage:
    python scripts/check_images_corruption_dir.py --img_dir path/to/image_directory

    Prints the paths of any corrupted images and a summary count.
    If no corrupted images are found, it will report that as well.
"""

from PIL import Image
import os

def is_image_corrupted(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except Exception as e:
        print(f"Corrupted: {image_path} ({e})")
        return True

def check_images_in_directory(img_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    corrupted = []
    for fname in os.listdir(img_dir):
        if fname.lower().endswith(extensions):
            fpath = os.path.join(img_dir, fname)
            if is_image_corrupted(fpath):
                corrupted.append(fpath)
    return corrupted

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check all images in a directory for corruption.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images to check")
    args = parser.parse_args()

    corrupted = check_images_in_directory(args.img_dir)
    if corrupted:
        print("\nSummary: Corrupted images found:")
        for f in corrupted:
            print(f)
        print(f"\nTotal corrupted: {len(corrupted)}")
    else:
        print("No corrupted images found.")
