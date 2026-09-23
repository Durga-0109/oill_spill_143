import os
import io
import hashlib
import datetime
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from PIL import Image

class EvidenceService:
    """
    Manages forensic evidence integrity:
    1. Server-side SHA-256 cryptographic hashing of uploaded image bytes.
    2. Immutable storage of original evidence files (never overwritten).
    3. Storage of processed inference images, masks, overlays, and generated reports.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(base_dir)

        self.uploads_dir = self.base_dir / "uploads"
        self.evidence_dir = self.uploads_dir / "evidence"
        self.processed_dir = self.uploads_dir / "processed"
        self.results_dir = self.base_dir / "results"
        self.masks_dir = self.results_dir / "masks"
        self.overlays_dir = self.results_dir / "overlays"
        self.reports_dir = self.results_dir / "reports"

        for directory in [
            self.uploads_dir, self.evidence_dir, self.processed_dir,
            self.results_dir, self.masks_dir, self.overlays_dir, self.reports_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def calculate_sha256(self, file_bytes: bytes) -> str:
        """Computes server-side SHA-256 cryptographic hash."""
        return hashlib.sha256(file_bytes).hexdigest()

    def store_evidence(
        self,
        file_bytes: bytes,
        original_filename: str,
        operator_id: str = "OPERATOR-01"
    ) -> Dict[str, Any]:
        """
        Stores original immutable evidence file and returns evidence record with hash.
        """
        sha256_hash = self.calculate_sha256(file_bytes)
        file_id = f"EV-{datetime.datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        
        # Clean file extension
        ext = os.path.splitext(original_filename)[1].lower() or ".jpg"
        saved_name = f"{file_id}_{sha256_hash[:10]}{ext}"
        saved_path = self.evidence_dir / saved_name

        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        # Create normalized processed copy (PNG 512x512)
        processed_name = f"{file_id}_proc.png"
        processed_path = self.processed_dir / processed_name
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            img.save(processed_path, format="PNG")
        except Exception:
            with open(processed_path, "wb") as f:
                f.write(file_bytes)

        return {
            "file_id": file_id,
            "original_filename": original_filename,
            "saved_filename": saved_name,
            "sha256_hash": sha256_hash,
            "file_size_bytes": len(file_bytes),
            "evidence_path": str(saved_path),
            "evidence_url": f"/uploads/evidence/{saved_name}",
            "processed_url": f"/uploads/processed/{processed_name}",
            "stored_at": datetime.datetime.utcnow().isoformat() + "Z",
            "operator_id": operator_id
        }

    def save_mask_and_overlay(
        self,
        analysis_id: str,
        mask_array: Any,
        overlay_array: Any
    ) -> Dict[str, str]:
        """Saves generated mask and overlay images to disk."""
        mask_name = f"{analysis_id}_mask.png"
        overlay_name = f"{analysis_id}_overlay.png"

        mask_path = self.masks_dir / mask_name
        overlay_path = self.overlays_dir / overlay_name

        try:
            mask_img = Image.fromarray(mask_array)
            mask_img.save(mask_path, format="PNG")
        except Exception:
            pass

        try:
            overlay_img = Image.fromarray(overlay_array)
            overlay_img.save(overlay_path, format="PNG")
        except Exception:
            pass

        return {
            "mask_url": f"/results/masks/{mask_name}",
            "overlay_url": f"/results/overlays/{overlay_name}",
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path)
        }

evidence_service = EvidenceService()
