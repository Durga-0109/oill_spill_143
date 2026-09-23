"""
Dataset preparation, validation, and split utility for SAR Oil Spill Segmentation.
"""

import os
import shutil
import random
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def validate_and_split(
    raw_images_dir: str,
    raw_masks_dir: str,
    output_dir: str = "data",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    random.seed(seed)
    raw_img_p = Path(raw_images_dir)
    raw_msk_p = Path(raw_masks_dir)
    out_p = Path(output_dir)

    if not raw_img_p.exists() or not raw_msk_p.exists():
        print(f"Error: Missing input folder: {raw_img_p} or {raw_msk_p}")
        return

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    all_imgs = sorted([p for p in raw_img_p.iterdir() if p.is_file() and p.suffix.lower() in exts])

    valid_pairs = []
    for img in all_imgs:
        mask_candidates = [
            raw_msk_p / f"{img.stem}{img.suffix}",
            raw_msk_p / f"{img.stem}.png",
            raw_msk_p / f"{img.stem}_mask.png",
        ]
        found_mask = None
        for cand in mask_candidates:
            if cand.exists():
                found_mask = cand
                break
        if found_mask:
            valid_pairs.append((img, found_mask))

    print(f"Found {len(valid_pairs)} matching image-mask pairs.")
    if len(valid_pairs) == 0:
        print("No valid image-mask pairs found to split.")
        return

    random.shuffle(valid_pairs)
    n_total = len(valid_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": valid_pairs[:n_train],
        "validation": valid_pairs[n_train:n_train + n_val],
        "test": valid_pairs[n_train + n_val:]
    }

    for split_name, pairs in splits.items():
        img_dest = out_p / split_name / "images"
        msk_dest = out_p / split_name / "masks"
        img_dest.mkdir(parents=True, exist_ok=True)
        msk_dest.mkdir(parents=True, exist_ok=True)

        for img_file, msk_file in pairs:
            shutil.copy2(img_file, img_dest / img_file.name)
            shutil.copy2(msk_file, msk_dest / msk_file.name)

        print(f"  {split_name.capitalize()}: {len(pairs)} pairs copied to {out_p / split_name}")

    print("Dataset preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="raw_data/images", help="Folder containing raw SAR images")
    parser.add_argument("--masks", default="raw_data/masks", help="Folder containing raw ground truth masks")
    parser.add_argument("--output", default="data", help="Target data folder")
    args = parser.parse_args()

    if Path(args.images).exists() and Path(args.masks).exists():
        validate_and_split(args.images, args.masks, args.output)
    else:
        print("Usage: python ml/prepare_dataset.py --images <path_to_images> --masks <path_to_masks>")
        print("Both source directories must exist.")
