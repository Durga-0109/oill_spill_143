"""
Dataset loader for SAR Oil Spill Pixel-Level Segmentation.

Expects directory structure:
  data/
    train/
      images/
      masks/
    validation/
      images/
      masks/
    test/
      images/
      masks/

Each mask must contain:
  0 = background / non-oil
  1 = oil spill (values in 255 are normalized to 1)
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms.functional as TF
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    Dataset = object
    TORCH_AVAILABLE = False


class SAROilSpillDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-1 / SAR Oil Spill Segmentation.
    Pairs SAR scene images with binary ground-truth masks.
    """

    SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 256,
        augment: bool = False
    ):
        """
        Args:
            root_dir: Base directory containing train/validation/test folders.
            split: Subfolder name ('train', 'validation', or 'test').
            img_size: Target square resolution (e.g. 256 or 512).
            augment: Whether to apply data augmentations (flips, rotations).
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.images_dir = self.root_dir / split / "images"
        self.masks_dir = self.root_dir / split / "masks"

        self.image_paths: List[Path] = []
        self.mask_paths: List[Path] = []

        self._scan_dataset()

    def _scan_dataset(self):
        if not self.images_dir.exists() or not self.masks_dir.exists():
            return

        all_imgs = sorted([
            p for p in self.images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_IMAGE_EXTS
        ])

        for img_path in all_imgs:
            # Match mask by exact stem or common mask naming conventions
            candidates = [
                self.masks_dir / f"{img_path.stem}{img_path.suffix}",
                self.masks_dir / f"{img_path.stem}.png",
                self.masks_dir / f"{img_path.stem}_mask.png",
                self.masks_dir / f"{img_path.stem}_mask{img_path.suffix}",
            ]
            matched_mask = None
            for cand in candidates:
                if cand.exists():
                    matched_mask = cand
                    break

            if matched_mask:
                self.image_paths.append(img_path)
                self.mask_paths.append(matched_mask)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to load SAROilSpillDataset.")

        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load SAR image
        image = Image.open(img_path).convert("RGB")
        # Load ground truth mask in grayscale mode
        mask = Image.open(mask_path).convert("L")

        # Resize to standard model input dimensions
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Apply spatial data augmentations synchronously to image and mask
        if self.augment:
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if torch.rand(1).item() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            rot = torch.randint(0, 4, (1,)).item()
            if rot > 0:
                angle = rot * 90
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # Convert image to FloatTensor in [0, 1]
        img_np = np.array(image, dtype=np.float32) / 255.0
        # Shape: (3, H, W)
        image_tensor = torch.from_numpy(img_np.transpose((2, 0, 1)))

        # Convert mask to binary {0.0, 1.0} FloatTensor
        mask_np = np.array(mask, dtype=np.float32)
        # Normalize: if values are 0-255, threshold at 128
        mask_binary = (mask_np > 127).astype(np.float32)
        # Shape: (1, H, W)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)

        return image_tensor, mask_tensor
