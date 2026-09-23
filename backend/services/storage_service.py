"""
Storage Service for SpillWatch.
Manages file uploads to Supabase Storage buckets:
  - sar-originals
  - sar-masks
  - sar-overlays
  - reports

Includes automatic local filesystem fallback if Supabase credentials are not yet configured.
"""

import os
import io
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

try:
    from supabase import create_client, Client
    SUPABASE_LIB_AVAILABLE = True
except ImportError:
    Client = None
    create_client = None
    SUPABASE_LIB_AVAILABLE = False


class StorageService:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        self.service_role_key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or
            os.getenv("SUPABASE_ANON_KEY", "").strip()
        )
        self.client: Optional[Client] = None
        self.is_connected = False

        self.local_base_dir = ROOT_DIR / "backend"
        self.local_uploads_dir = self.local_base_dir / "uploads"
        self.local_results_dir = self.local_base_dir / "results"

        self._ensure_local_dirs()
        self._init_supabase()

    def _ensure_local_dirs(self):
        for d in [
            self.local_uploads_dir / "evidence",
            self.local_results_dir / "masks",
            self.local_results_dir / "overlays",
            self.local_results_dir / "reports",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _init_supabase(self):
        if not SUPABASE_LIB_AVAILABLE or not self.supabase_url or not self.service_role_key:
            self.is_connected = False
            return

        try:
            self.client = create_client(self.supabase_url, self.service_role_key)
            self.is_connected = True
        except Exception as e:
            print(f"[StorageService] Supabase init warning: {e}")
            self.is_connected = False

    def upload_file(
        self,
        bucket_name: str,
        destination_path: str,
        file_bytes: bytes,
        content_type: str = "image/png"
    ) -> str:
        """
        Uploads a file to Supabase Storage bucket.
        Falls back to local filesystem if Supabase is unconfigured or returns an error.
        Returns the accessible URL.
        """
        if self.is_connected and self.client:
            try:
                # Upload or upsert
                self.client.storage.from_(bucket_name).upload(
                    path=destination_path,
                    file=file_bytes,
                    file_options={"content-type": content_type, "upsert": "true"}
                )
                # Get public URL
                public_url = self.client.storage.from_(bucket_name).get_public_url(destination_path)
                return public_url
            except Exception as e:
                print(f"[StorageService] Supabase bucket upload failed ({bucket_name}/{destination_path}): {e}")

        # Local fallback
        bucket_folder_map = {
            "sar-originals": self.local_uploads_dir / "evidence",
            "sar-masks": self.local_results_dir / "masks",
            "sar-overlays": self.local_results_dir / "overlays",
            "reports": self.local_results_dir / "reports"
        }

        folder = bucket_folder_map.get(bucket_name, self.local_results_dir)
        local_file = folder / destination_path
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_bytes(file_bytes)

        if bucket_name == "sar-originals":
            return f"/uploads/evidence/{destination_path}"
        elif bucket_name == "sar-masks":
            return f"/results/masks/{destination_path}"
        elif bucket_name == "sar-overlays":
            return f"/results/overlays/{destination_path}"
        elif bucket_name == "reports":
            return f"/results/reports/{destination_path}"
        else:
            return f"/results/{destination_path}"


storage_service = StorageService()
