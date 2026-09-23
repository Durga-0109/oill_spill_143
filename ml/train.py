"""
Training script for Attention U-Net SAR Oil Spill Segmentation.

Implements:
- Loss: Combined Dice Loss + BCEWithLogitsLoss
- Optimizer: AdamW with Cosine Annealing Learning Rate Scheduler
- Metrics Tracking: Dice Score, IoU (Jaccard), Precision, Recall, F1 Score
- Checkpoints: Best model checkpoint saved to 'models/oil_spill_attention_unet.pth'
- Real Metadata: 'models/model_metadata.json' with exact computed metrics

If dataset is not provided in data/train:
- Clearly reports: "REAL LABELED SAR OIL-SPILL DATASET REQUIRED"
"""

import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path
import numpy as np

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from ml.model import AttentionUNet
    from ml.dataset import SAROilSpillDataset
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        total = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """BCEWithLogitsLoss + DiceLoss for robust edge and area optimization."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)


def compute_metrics(preds_binary: np.ndarray, targets_binary: np.ndarray) -> dict:
    """
    Computes exact pixel-level evaluation metrics:
    Dice, IoU, Precision, Recall, F1 Score.
    """
    preds = preds_binary.astype(bool)
    targets = targets_binary.astype(bool)

    tp = np.logical_and(preds, targets).sum()
    fp = np.logical_and(preds, np.logical_not(targets)).sum()
    fn = np.logical_and(np.logical_not(preds), targets).sum()
    tn = np.logical_and(np.logical_not(preds), np.logical_not(targets)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def train_pipeline(
    data_dir: str = "data",
    output_model_path: str = "models/oil_spill_attention_unet.pth",
    output_meta_path: str = "models/model_metadata.json",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    img_size: int = 256,
    patience: int = 10,
    dataset_name: str = "SAR_Oil_Spill_Benchmark"
):
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is not available in the current environment.")
        return

    data_path = Path(data_dir)
    train_dataset = SAROilSpillDataset(data_dir, split="train", img_size=img_size, augment=True)
    val_dataset = SAROilSpillDataset(data_dir, split="validation", img_size=img_size, augment=False)

    print("=" * 70)
    print("SPILLWATCH ATTENTION U-NET TRAINING PIPELINE")
    print("=" * 70)
    print(f"Data root directory:       {data_path.resolve()}")
    print(f"Train samples found:       {len(train_dataset)}")
    print(f"Validation samples found:  {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("\n" + "!" * 70)
        print("REAL LABELED SAR OIL-SPILL DATASET REQUIRED")
        print("!" * 70)
        print("No training images and masks were found in:")
        print(f"  Images: {data_path / 'train' / 'images'}")
        print(f"  Masks:  {data_path / 'train' / 'masks'}")
        print("\nTo train the model, place your labeled SAR dataset in the above folders:")
        print("  - Each SAR scene (.png/.jpg/.tif) in images/")
        print("  - Corresponding ground-truth binary mask (0=non-oil, 1=oil spill) in masks/")
        print("The pipeline is fully constructed and will execute automatically once images are supplied.")
        print("=" * 70)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference & Training Device: {device.type.upper()}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if len(val_dataset) > 0 else train_loader

    model = AttentionUNet(in_channels=3, out_channels=1).to(device)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_dice = -1.0
    best_metrics = {}
    epochs_no_improve = 0

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)
        scheduler.step()

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(masks.cpu().numpy())

        val_loss /= len(val_dataset) if len(val_dataset) > 0 else len(train_dataset)
        cat_preds = np.concatenate(all_preds, axis=0)
        cat_targets = np.concatenate(all_targets, axis=0)

        metrics = compute_metrics(cat_preds, cat_targets)
        val_dice = metrics["dice"]
        val_iou = metrics["iou"]

        print(f"Epoch [{epoch:03d}/{epochs:03d}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | "
              f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

        # Best Model Checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            best_metrics = metrics
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_model_path)
            print(f"  --> Checkpoint saved to {output_model_path} (Best Dice: {best_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs without improvement.")
                break

    total_duration = time.time() - start_time
    print("=" * 70)
    print(f"Training completed in {total_duration / 60:.2f} minutes.")
    print(f"Best Validation Dice Score: {best_dice:.4f}")

    # Save real model metadata
    metadata = {
        "model_name": "Attention U-Net",
        "model_architecture": "AttentionUNet-4Encoder-AttentionGates-1x1Conv",
        "model_version": "v1.0.0",
        "training_date": datetime.datetime.utcnow().isoformat() + "Z",
        "input_resolution": [img_size, img_size],
        "input_channels": 3,
        "dataset_name": dataset_name,
        "total_train_samples": len(train_dataset),
        "total_val_samples": len(val_dataset),
        "best_metrics": best_metrics,
        "loss_function": "Combined DiceLoss + BCEWithLogitsLoss",
        "optimizer": "AdamW",
        "learning_rate": lr,
        "epochs_trained": epoch,
        "checkpoint_path": output_model_path,
        "status": "trained_and_validated"
    }

    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model metadata written to: {output_meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Attention U-Net on SAR Oil Spill Dataset")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data folder")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=256, help="Image resolution")
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size
    )
