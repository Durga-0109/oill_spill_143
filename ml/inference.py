"""
Standalone CLI inference for Attention U-Net SAR Oil Spill Segmentation.
Generates:
  - mask.png (0 = non-oil, 255 = oil)
  - overlay.png (SAR image + transparent red highlight on detected oil pixels)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
    import cv2
    from ml.model import AttentionUNet
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False


def run_inference(
    image_path: str,
    model_path: str = "models/oil_spill_attention_unet.pth",
    output_dir: str = "results",
    threshold: float = 0.5,
    img_size: int = 256
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_p = Path(image_path)
    if not img_p.exists():
        print(f"Error: Image not found at {image_path}")
        return

    # Load original image
    raw_img = Image.open(img_p).convert("RGB")
    orig_w, orig_h = raw_img.size

    # Resize for model input
    resized = raw_img.resize((img_size, img_size), Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet(in_channels=3, out_channels=1).to(device)

    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained Attention U-Net from {model_path} onto {device.type.upper()}.")
    else:
        print(f"Warning: Checkpoint not found at {model_path}. Using initialized architecture.")

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Threshold probability map to binary mask
    binary_mask_small = (prob_map >= threshold).astype(np.uint8)

    # Resize binary mask back to original image size
    binary_mask = cv2.resize(binary_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 1. Save binary mask PNG
    mask_png_path = out_dir / f"{img_p.stem}_mask.png"
    Image.fromarray((binary_mask * 255).astype(np.uint8)).save(mask_png_path)

    # 2. Save composite overlay (original image with semi-transparent red on detected oil pixels)
    orig_np = np.array(raw_img)
    overlay_np = orig_np.copy()
    red_mask = np.zeros_like(orig_np)
    red_mask[:, :, 0] = 239  # Red channel
    red_mask[:, :, 1] = 68   # Green
    red_mask[:, :, 2] = 68   # Blue

    oil_pixels = (binary_mask == 1)
    alpha = 0.5
    overlay_np[oil_pixels] = (
        (1 - alpha) * orig_np[oil_pixels] + alpha * red_mask[oil_pixels]
    ).astype(np.uint8)

    overlay_png_path = out_dir / f"{img_p.stem}_overlay.png"
    Image.fromarray(overlay_np).save(overlay_png_path)

    oil_pixel_count = int(np.sum(binary_mask))
    total_pixels = binary_mask.size
    oil_ratio = oil_pixel_count / total_pixels

    classification = "OIL SPILL DETECTED" if oil_pixel_count > 50 else "NO OIL SPILL DETECTED"
    confidence = float(np.mean(prob_map[binary_mask_small == 1])) if oil_pixel_count > 0 else float(1.0 - np.mean(prob_map))

    print(f"Classification:   {classification}")
    print(f"Confidence:       {confidence * 100:.2f}%")
    print(f"Oil Pixels:       {oil_pixel_count:,} ({oil_ratio * 100:.2f}% of scene)")
    print(f"Mask saved to:    {mask_png_path}")
    print(f"Overlay saved to: {overlay_png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input SAR scene image")
    parser.add_argument("--model", default="models/oil_spill_attention_unet.pth", help="Model weights path")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    args = parser.parse_args()
    run_inference(args.image, args.model, args.output_dir, args.threshold)
