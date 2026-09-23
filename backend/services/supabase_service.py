"""
Supabase Service for SpillWatch.
Manages database operations with Supabase PostgreSQL:
  - users
  - analyses
  - segmentation_results
  - vessel_records
  - drift_predictions
  - audit_logs

Provides dynamic KPI calculations and historical analysis retrieval.
Includes automatic local database fallback when Supabase is offline or unconfigured.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
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


class SupabaseService:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        self.service_role_key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip() or
            os.getenv("SUPABASE_ANON_KEY", "").strip()
        )
        self.client: Optional[Client] = None
        self.is_connected = False
        self.local_memory_analyses: List[Dict[str, Any]] = []
        self.local_audit_logs: List[Dict[str, Any]] = []

        self._init_supabase()

    def _init_supabase(self):
        if not SUPABASE_LIB_AVAILABLE or not self.supabase_url or not self.service_role_key:
            self.is_connected = False
            return

        try:
            self.client = create_client(self.supabase_url, self.service_role_key)
            # Ping connection
            res = self.client.table("analyses").select("id").limit(1).execute()
            self.is_connected = True
        except Exception as e:
            print(f"[SupabaseService] Notice: Supabase not currently reachable ({e}). Using local store.")
            self.is_connected = False

    def get_or_create_user(self, name: str = "Durga C", email: str = "durga@spillwatch.maritime.gov", role: str = "lead_operator") -> Dict[str, Any]:
        """Retrieves or provisions the active operator profile."""
        if self.is_connected and self.client:
            try:
                res = self.client.table("users").select("*").eq("email", email).execute()
                if res.data and len(res.data) > 0:
                    return res.data[0]

                # Create if not exists
                insert_res = self.client.table("users").insert({
                    "name": name,
                    "email": email,
                    "role": role
                }).execute()
                if insert_res.data and len(insert_res.data) > 0:
                    return insert_res.data[0]
            except Exception as e:
                print(f"[SupabaseService] get_or_create_user error: {e}")

        return {
            "id": "00000000-0000-0000-0000-000000000001",
            "name": name,
            "email": email,
            "role": role,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z"
        }

    def save_analysis(
        self,
        analysis_id: str,
        user_id: Optional[str],
        image_name: str,
        original_image_url: str,
        classification: str,
        confidence: float,
        raw_probability: float,
        oil_pixel_count: int,
        oil_ratio: float,
        spill_area: str,
        spill_area_unit: str = "km²",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        processing_time: float = 0.0,
        model_name: str = "Attention U-Net",
        model_version: str = "v1.0.0",
        inference_device: str = "CPU",
        image_hash: Optional[str] = None,
        status: str = "completed"
    ) -> Dict[str, Any]:
        """Saves a completed analysis record to public.analyses."""
        record = {
            "analysis_id": analysis_id,
            "user_id": user_id,
            "image_name": image_name,
            "original_image_url": original_image_url,
            "classification": classification,
            "confidence": float(confidence),
            "raw_probability": float(raw_probability),
            "oil_pixel_count": int(oil_pixel_count),
            "oil_ratio": float(oil_ratio),
            "spill_area": spill_area,
            "spill_area_unit": spill_area_unit,
            "latitude": latitude,
            "longitude": longitude,
            "processing_time": float(processing_time),
            "model_name": model_name,
            "model_version": model_version,
            "inference_device": inference_device,
            "image_hash": image_hash,
            "status": status,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z"
        }

        if self.is_connected and self.client:
            try:
                res = self.client.table("analyses").insert(record).execute()
                if res.data and len(res.data) > 0:
                    return res.data[0]
            except Exception as e:
                print(f"[SupabaseService] save_analysis error: {e}")

        self.local_memory_analyses.insert(0, record)
        return record

    def save_segmentation_result(
        self,
        analysis_id: str,
        mask_url: str,
        overlay_url: str,
        bounding_box: Optional[Dict[str, Any]],
        oil_pixel_count: int,
        oil_ratio: float
    ) -> Dict[str, Any]:
        """Saves segmentation mask/overlay and bounding boxes."""
        record = {
            "analysis_id": analysis_id,
            "mask_url": mask_url,
            "overlay_url": overlay_url,
            "bounding_box": bounding_box,
            "oil_pixel_count": int(oil_pixel_count),
            "oil_ratio": float(oil_ratio),
            "created_at": datetime.datetime.utcnow().isoformat() + "Z"
        }

        if self.is_connected and self.client:
            try:
                self.client.table("segmentation_results").insert(record).execute()
            except Exception as e:
                print(f"[SupabaseService] save_segmentation_result error: {e}")

        return record

    def save_vessel_records(self, analysis_id: str, vessels: List[Dict[str, Any]]):
        """Saves correlated AIS vessel records."""
        if not vessels:
            return

        records = []
        for v in vessels:
            records.append({
                "analysis_id": analysis_id,
                "vessel_id": str(v.get("mmsi") or v.get("vessel_id", "")),
                "vessel_name": v.get("vessel_name") or v.get("name", "Unknown"),
                "latitude": v.get("latitude") or v.get("lat"),
                "longitude": v.get("longitude") or v.get("lon"),
                "speed": v.get("speed_kn") or v.get("speed"),
                "course": v.get("heading_deg") or v.get("course"),
                "distance_from_spill": v.get("distance_km"),
                "correlation_score": v.get("attribution_score") or v.get("score")
            })

        if self.is_connected and self.client:
            try:
                self.client.table("vessel_records").insert(records).execute()
            except Exception as e:
                print(f"[SupabaseService] save_vessel_records error: {e}")

    def save_drift_prediction(self, analysis_id: str, drift_data: Dict[str, Any]):
        """Saves 1-hour and 2-hour drift coordinates."""
        if not drift_data:
            return

        trajectories = drift_data.get("trajectories", [])
        p1 = trajectories[1] if len(trajectories) > 1 else {}
        p2 = trajectories[2] if len(trajectories) > 2 else {}

        record = {
            "analysis_id": analysis_id,
            "current_latitude": drift_data.get("initial_position", {}).get("latitude"),
            "current_longitude": drift_data.get("initial_position", {}).get("longitude"),
            "predicted_latitude_1h": p1.get("latitude"),
            "predicted_longitude_1h": p1.get("longitude"),
            "predicted_latitude_2h": p2.get("latitude"),
            "predicted_longitude_2h": p2.get("longitude"),
            "wind_speed": drift_data.get("weather_input", {}).get("wind_speed_kn"),
            "wind_direction": drift_data.get("weather_input", {}).get("wind_direction_deg"),
            "current_speed": drift_data.get("weather_input", {}).get("current_speed_kn"),
            "current_direction": drift_data.get("weather_input", {}).get("current_direction_deg"),
            "created_at": datetime.datetime.utcnow().isoformat() + "Z"
        }

        if self.is_connected and self.client:
            try:
                self.client.table("drift_predictions").insert(record).execute()
            except Exception as e:
                print(f"[SupabaseService] save_drift_prediction error: {e}")

    def log_audit(self, user_id: Optional[str], analysis_id: Optional[str], action: str, status: str = "SUCCESS"):
        """Saves immutable pipeline audit log entries."""
        entry = {
            "user_id": user_id,
            "analysis_id": analysis_id,
            "action": action,
            "status": status,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }

        if self.is_connected and self.client:
            try:
                self.client.table("audit_logs").insert(entry).execute()
            except Exception as e:
                print(f"[SupabaseService] log_audit error: {e}")

        self.local_audit_logs.insert(0, entry)

    def get_kpi_statistics(self) -> Dict[str, Any]:
        """
        Calculates dynamic KPIs directly from Supabase:
        - TOTAL ANALYSES
        - OIL SPILLS DETECTED
        - NO OIL SPILL
        - TODAY'S ANALYSES
        - AVERAGE RESPONSE TIME
        """
        today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"

        if self.is_connected and self.client:
            try:
                # 1. Total Analyses
                total_res = self.client.table("analyses").select("id", count="exact").execute()
                total = total_res.count if total_res.count is not None else 0

                # 2. Spills Detected
                spills_res = self.client.table("analyses").select("id", count="exact").eq("classification", "OIL SPILL DETECTED").execute()
                spills = spills_res.count if spills_res.count is not None else 0

                # 3. Clean Oceans
                clean_res = self.client.table("analyses").select("id", count="exact").eq("classification", "NO OIL SPILL DETECTED").execute()
                clean = clean_res.count if clean_res.count is not None else 0

                # 4. Today's Analyses
                today_res = self.client.table("analyses").select("id", count="exact").gte("created_at", today_start).execute()
                today_count = today_res.count if today_res.count is not None else 0

                # 5. Average Response
                avg_res = self.client.table("analyses").select("processing_time").limit(100).execute()
                times = [r["processing_time"] for r in (avg_res.data or []) if r.get("processing_time") is not None]
                avg_time = round(sum(times) / len(times), 2) if len(times) > 0 else 0.0

                return {
                    "total_analyses": total,
                    "oil_spills_detected": spills,
                    "no_oil_spill": clean,
                    "todays_analyses": today_count,
                    "average_response_seconds": avg_time,
                    "data_source": "Supabase PostgreSQL (Live)"
                }
            except Exception as e:
                print(f"[SupabaseService] get_kpi_statistics error: {e}")

        # Local memory fallback calculation
        total = len(self.local_memory_analyses)
        spills = sum(1 for a in self.local_memory_analyses if a.get("classification") == "OIL SPILL DETECTED")
        clean = sum(1 for a in self.local_memory_analyses if a.get("classification") == "NO OIL SPILL DETECTED")
        today_count = sum(1 for a in self.local_memory_analyses if a.get("created_at", "") >= today_start)
        times = [a.get("processing_time", 0.0) for a in self.local_memory_analyses]
        avg_time = round(sum(times) / len(times), 2) if len(times) > 0 else 0.0

        return {
            "total_analyses": total,
            "oil_spills_detected": spills,
            "no_oil_spill": clean,
            "todays_analyses": today_count,
            "average_response_seconds": avg_time,
            "data_source": "Local Store (Configure Supabase credentials in .env)"
        }

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves stored analysis records for the History tab."""
        if self.is_connected and self.client:
            try:
                res = self.client.table("analyses").select(
                    "*, segmentation_results(*), vessel_records(*)"
                ).order("created_at", desc=True).limit(limit).execute()
                if res.data:
                    return res.data
            except Exception as e:
                print(f"[SupabaseService] get_history error: {e}")

        return self.local_memory_analyses[:limit]

    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single historical analysis record with mask/overlay URLs."""
        if self.is_connected and self.client:
            try:
                res = self.client.table("analyses").select(
                    "*, segmentation_results(*), vessel_records(*), drift_predictions(*)"
                ).eq("analysis_id", analysis_id).execute()
                if res.data and len(res.data) > 0:
                    return res.data[0]
            except Exception as e:
                print(f"[SupabaseService] get_analysis_by_id error: {e}")

        for a in self.local_memory_analyses:
            if a.get("analysis_id") == analysis_id:
                return a
        return None


supabase_service = SupabaseService()
