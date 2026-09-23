"""
Evaluation script for Attention U-Net SAR Oil Spill Segmentation on Test Split.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
    from torch.utils.data import DataLoader
    from ml.model import AttentionUNet
    from ml.dataset import SAROilSpillDataset
    from ml.train import compute_metrics
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False


def evaluate_test_split(
    data_dir: str = "data",
    model_path: str = "models/oil_spill_attention_unet.pth",
    img_size: int = 256,
    batch_size: int = 8
):
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not available.")
        return

    test_dataset = SAROilSpillDataset(data_dir, split="test", img_size=img_size, augment=False)
    if len(test_dataset) == 0:
        print("No test samples found in data/test/.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Attention U-Net on {len(test_dataset)} test samples (Device: {device.type.upper()})...")

    model = AttentionUNet(in_channels=3, out_channels=1).to(device)
    if not Path(model_path).exists():
        print(f"Error: Model weights not found at {model_path}.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(masks.numpy())

    cat_preds = np.concatenate(all_preds, axis=0)
    cat_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_metrics(cat_preds, cat_targets)
    print("=" * 60)
    print("TEST EVALUATION RESULTS:")
    print(f"  Dice Score (F1): {metrics['dice']:.4f}")
    print(f"  IoU (Jaccard):   {metrics['iou']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print("=" * 60)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="models/oil_spill_attention_unet.pth")
    args = parser.parse_args()
    evaluate_test_split(args.data_dir, args.model_path)
