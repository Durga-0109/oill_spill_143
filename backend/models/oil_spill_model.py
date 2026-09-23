"""
Production Attention U-Net Model & Inference Engine for SAR Oil Spill Detection.

Implements:
  - Single-load architecture at backend startup
  - GPU (CUDA) auto-detection with graceful CPU fallback
  - Pixel-level SAR oil spill segmentation
  - Real sigmoid probability map and configurable thresholding (OIL_THRESHOLD=0.5)
  - Mathematical post-processing (morphology, bounding boxes, oil pixel count, oil ratio)
  - Physical area calculation (only when GeoTIFF/geospatial resolution is present)
  - Pixel-level semi-transparent red overlay generation (only highlighting actual oil pixels)
  - Two strict classifications: "OIL SPILL DETECTED" vs "NO OIL SPILL DETECTED"
  - Real confidence calculation without hardcoded or fake numbers
"""

import os
import io
import time
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from PIL import Image

try:
    import torch
    from ml.model import AttentionUNet
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    AttentionUNet = None
    TORCH_AVAILABLE = False

try:
    import cv2
except ImportError:
    cv2 = None


class OilSpillModel:
    """
    Singleton-style inference manager for Attention U-Net SAR Oil Spill Segmentation.
    Loads once on startup, processes incoming scenes, and returns verified metrics.
    """

    def __init__(
        self,
        weights_path: str = "models/oil_spill_attention_unet.pth",
        metadata_path: str = "models/model_metadata.json",
        img_size: int = 256,
        oil_threshold: float = 0.5,
        min_oil_pixels: int = 50
    ):
        self.weights_path = Path(weights_path)
        self.metadata_path = Path(metadata_path)
        self.img_size = img_size
        self.oil_threshold = float(os.getenv("OIL_THRESHOLD", oil_threshold))
        self.min_oil_pixels = min_oil_pixels

        self.model_name = "Attention U-Net"
        self.model_version = "v1.0.0"
        self.device = "cpu"
        self.model = None
        self.is_loaded = False
        self.metadata = {}
        self.status = "NOT CONFIGURED"
        self.status_message = "No trained model weights found yet."

        self.load_model()

    def load_model(self) -> bool:
        """Loads trained Attention U-Net model ONCE onto active device."""
        if not TORCH_AVAILABLE:
            self.status = "UNAVAILABLE"
            self.status_message = "PyTorch is not available in environment."
            return False

        # Detect GPU or CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Initialize network
        try:
            self.model = AttentionUNet(in_channels=3, out_channels=1).to(self.device)
        except Exception as e:
            self.status = "INITIALIZATION_FAILED"
            self.status_message = str(e)
            return False

        # Load weights if present
        if self.weights_path.exists() and self.weights_path.stat().st_size > 1000:
            try:
                state_dict = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_loaded = True
                self.status = "CONNECTED"
                self.status_message = f"Model loaded on {self.device.upper()}."

                if self.metadata_path.exists():
                    import json
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                    self.model_version = self.metadata.get("model_version", "v1.0.0")

                return True
            except Exception as e:
                self.status = "LOAD_ERROR"
                self.status_message = f"Failed loading weights: {e}"
                return False
        else:
            self.is_loaded = False
            self.status = "REAL LABELED SAR OIL-SPILL DATASET REQUIRED"
            self.status_message = (
                "Attention U-Net architecture is ready. Place labeled SAR dataset in "
                "data/train/ and run 'python ml/train.py' to generate weights."
            )
            return False

    def get_status(self) -> Dict[str, Any]:
        """Returns verified real model status for GET /api/model/status."""
        return {
            "loaded": self.is_loaded,
            "model_name": self.model_name,
            "version": self.model_version,
            "device": self.device,
            "threshold": self.oil_threshold,
            "status": self.status,
            "message": self.status_message,
            "metrics": self.metadata.get("best_metrics", None)
        }

    def preprocess_image(self, image_bytes: bytes) -> Tuple[Image.Image, Optional[torch.Tensor], Tuple[int, int]]:
        """
        Validates and converts image bytes into normalized tensor.
        Returns: (PIL.Image, torch.Tensor or None, (orig_width, orig_height))
        """
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = raw_image.size

        if not TORCH_AVAILABLE:
            return raw_image, None, (orig_w, orig_h)

        resized = raw_image.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_np = np.array(resized, dtype=np.float32) / 255.0
        # Shape: (1, 3, H, W)
        tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0)
        return raw_image, tensor, (orig_w, orig_h)

    def extract_geospatial(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts geospatial metadata and pixel resolution if GeoTIFF.
        Does NOT invent coordinates or pixel sizes.
        """
        geo_info = {
            "has_geospatial": False,
            "pixel_resolution_m": None,
            "latitude": None,
            "longitude": None,
            "crs": None,
            "message": "Geolocation metadata unavailable."
        }

        try:
            # Try reading with PIL / TiffTags
            pil_img = Image.open(io.BytesIO(image_bytes))
            # Check GeoTIFF tags (34735 = GeoKeyDirectoryTag, 33550 = ModelPixelScaleTag, 33922 = ModelTiepointTag)
            tags = getattr(pil_img, "tag_v2", {})
            if 33550 in tags:
                # Pixel scale tag: [ScaleX, ScaleY, ScaleZ]
                scale = tags[33550]
                if len(scale) >= 2:
                    geo_info["pixel_resolution_m"] = float(scale[0])
                    geo_info["has_geospatial"] = True

            if 33922 in tags:
                # Tiepoint tag
                tp = tags[33922]
                if len(tp) >= 6:
                    geo_info["longitude"] = float(tp[3])
                    geo_info["latitude"] = float(tp[4])
                    geo_info["has_geospatial"] = True
                    geo_info["message"] = "GeoTIFF coordinate reference extracted."
        except Exception:
            pass

        return geo_info

    def predict(
        self,
        image_bytes: bytes,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes full inference pipeline:
        1. Preprocess SAR image
        2. Run Attention U-Net
        3. Extract pixel probability map
        4. Apply threshold -> Binary mask
        5. Post-process contours and bounding boxes
        6. Calculate physical area (if geospatial)
        7. Generate pixel-level semi-transparent overlay
        8. Return validated metrics
        """
        t_start = time.perf_counter()
        th = threshold if threshold is not None else self.oil_threshold

        raw_image, img_tensor, (orig_w, orig_h) = self.preprocess_image(image_bytes)
        geo_info = self.extract_geospatial(image_bytes)

        if not self.is_loaded or self.model is None or img_tensor is None:
            # Model weights not trained or PyTorch unavailable
            # Strictly report unconfigured state without inventing fake data
            elapsed = round((time.perf_counter() - t_start), 3)
            return {
                "classification": "NO OIL SPILL DETECTED",
                "oil_detected": False,
                "confidence": 0.0,
                "raw_probability": 0.0,
                "oil_pixel_count": 0,
                "oil_ratio": 0.0,
                "spill_area": "Physical spill area unavailable — image georeferencing/resolution required.",
                "spill_area_km2": None,
                "bounding_boxes": [],
                "primary_bounding_box": None,
                "processing_time": elapsed,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "inference_device": self.device.upper(),
                "status": self.status,
                "message": self.status_message,
                "binary_mask": np.zeros((orig_h, orig_w), dtype=np.uint8),
                "geospatial": geo_info
            }

        # 1. Forward Pass
        self.model.eval()
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            logits = self.model(img_tensor)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        # 2. Binary Thresholding
        binary_mask_small = (prob_map >= th).astype(np.uint8)

        # Resize binary mask to original image dimensions using Nearest Neighbor
        if cv2 is not None:
            binary_mask = cv2.resize(binary_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_pil = Image.fromarray(binary_mask_small * 255).resize((orig_w, orig_h), Image.NEAREST)
            binary_mask = (np.array(mask_pil) > 127).astype(np.uint8)

        # 3. Post-Processing & Connected Components
        oil_pixel_count = int(np.sum(binary_mask))
        total_pixels = binary_mask.size
        oil_ratio = round(oil_pixel_count / total_pixels, 6)

        # Extract bounding boxes
        bounding_boxes = []
        if cv2 is not None and oil_pixel_count > 0:
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) >= 10:  # Noise threshold
                    x, y, w, h = cv2.boundingRect(cnt)
                    bounding_boxes.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    })

        primary_bbox = bounding_boxes[0] if bounding_boxes else None

        # 4. Strict Oil Detection Decision
        oil_detected = (oil_pixel_count >= self.min_oil_pixels) and (oil_ratio > 0.0005)
        classification = "OIL SPILL DETECTED" if oil_detected else "NO OIL SPILL DETECTED"

        # 5. Verified Confidence & Raw Probability
        raw_probability = float(np.mean(prob_map))
        if oil_detected:
            oil_probs = prob_map[binary_mask_small == 1]
            confidence = float(np.mean(oil_probs)) if len(oil_probs) > 0 else raw_probability
        else:
            confidence = float(1.0 - raw_probability)

        # 6. Physical Area Calculation
        if geo_info.get("pixel_resolution_m") and oil_detected:
            res_m = geo_info["pixel_resolution_m"]
            area_m2 = oil_pixel_count * (res_m ** 2)
            area_km2 = round(area_m2 / 1_000_000.0, 4)
            spill_area_display = f"{area_km2} km² ({round(area_m2):,} m²)"
        elif oil_detected:
            spill_area_display = "Physical spill area unavailable — image georeferencing/resolution required."
            area_km2 = None
        else:
            spill_area_display = "0.00 km²"
            area_km2 = 0.0

        elapsed = round((time.perf_counter() - t_start), 3)

        return {
            "classification": classification,
            "oil_detected": oil_detected,
            "confidence": round(confidence, 4),
            "raw_probability": round(raw_probability, 6),
            "oil_pixel_count": oil_pixel_count,
            "oil_ratio": oil_ratio,
            "spill_area": spill_area_display,
            "spill_area_km2": area_km2,
            "bounding_boxes": bounding_boxes,
            "primary_bounding_box": primary_bbox,
            "processing_time": elapsed,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "inference_device": self.device.upper(),
            "status": "COMPLETED",
            "message": "Pixel-level Attention U-Net segmentation complete.",
            "binary_mask": binary_mask,
            "geospatial": geo_info
        }

    def generate_mask_image(self, binary_mask: np.ndarray) -> Image.Image:
        """
        Creates the black and white mask image.
        0 = black (non-oil), 255 = white (oil spill).
        """
        return Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")

    def create_overlay_image(self, original_image: Image.Image, binary_mask: np.ndarray) -> Image.Image:
        """
        Generates pixel-level composite overlay.
        Only pixels predicted as oil by the mask receive a semi-transparent red highlight.
        Does NOT color the whole image or draw fake rectangles.
        """
        orig_rgb = original_image.convert("RGB")
        orig_np = np.array(orig_rgb)

        if orig_np.shape[:2] != binary_mask.shape:
            binary_mask = cv2.resize(
                binary_mask,
                (orig_np.shape[1], orig_np.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        overlay_np = orig_np.copy()
        oil_indices = (binary_mask == 1)

        if np.any(oil_indices):
            # Target vivid warning red: RGB(239, 68, 68)
            red_layer = np.zeros_like(orig_np)
            red_layer[:, :, 0] = 239
            red_layer[:, :, 1] = 68
            red_layer[:, :, 2] = 68

            alpha = 0.52
            overlay_np[oil_indices] = (
                (1.0 - alpha) * orig_np[oil_indices] + alpha * red_layer[oil_indices]
            ).astype(np.uint8)

        return Image.fromarray(overlay_np, mode="RGB")


# Global singleton instance loaded once
oil_spill_model = OilSpillModel()
