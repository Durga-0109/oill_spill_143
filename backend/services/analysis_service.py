"""
End-to-end Pipeline Analysis Service for SpillWatch.

Coordinates:
  1. Image validation & SHA-256 cryptographic hashing
  2. Original file preservation in Supabase Storage (sar-originals)
  3. Attention U-Net inference via backend/models/oil_spill_model.py
  4. Binary segmentation mask & overlay generation
  5. Upload of mask & overlay to Supabase Storage (sar-masks, sar-overlays)
  6. MetOcean weather, ocean current & 2h drift prediction (only when oil detected)
  7. AIS vessel retrieval & kinematic attribution correlation (only when oil detected)
  8. Forensic report generation
  9. Supabase database persistence & audit logging
"""

import io
import time
import uuid
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from backend.models.oil_spill_model import oil_spill_model
from backend.services.storage_service import storage_service
from backend.services.supabase_service import supabase_service
from backend.services.weather_service import weather_service
from backend.services.ocean_current_service import ocean_current_service
from backend.services.drift_service import drift_service
from backend.services.ais_service import ais_service
from backend.services.correlation_service import correlation_service
from backend.services.report_service import report_service


class AnalysisService:

    def analyze_scene(
        self,
        image_bytes: bytes,
        filename: str,
        operator_name: str = "Durga C",
        operator_email: str = "durga@spillwatch.maritime.gov",
        base_lat: Optional[float] = None,
        base_lon: Optional[float] = None
    ) -> Dict[str, Any]:
        """Executes complete verified end-to-end SAR oil spill analysis."""
        t_start = time.perf_counter()
        analysis_id = f"SW-ANL-{datetime.datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

        # 1. Validation & SHA-256
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Uploaded image file is empty.")

        sha256_hash = hashlib.sha256(image_bytes).hexdigest()

        # Retrieve user profile
        user = supabase_service.get_or_create_user(name=operator_name, email=operator_email)
        user_id = user.get("id")

        # Log upload audit
        supabase_service.log_audit(user_id=user_id, analysis_id=analysis_id, action="IMAGE_UPLOADED", status="SUCCESS")

        # 2. Preserve original image in Supabase Storage
        ext = Path(filename).suffix.lower() or ".png"
        original_dest = f"{analysis_id}_orig{ext}"
        original_url = storage_service.upload_file(
            bucket_name="sar-originals",
            destination_path=original_dest,
            file_bytes=image_bytes,
            content_type=f"image/{ext.lstrip('.')}"
        )

        # 3. Model Inference (Attention U-Net)
        supabase_service.log_audit(user_id=user_id, analysis_id=analysis_id, action="MODEL_STARTED", status="SUCCESS")
        prediction = oil_spill_model.predict(image_bytes)
        supabase_service.log_audit(user_id=user_id, analysis_id=analysis_id, action="MODEL_COMPLETED", status="SUCCESS")

        oil_detected = prediction["oil_detected"]
        binary_mask = prediction["binary_mask"]

        # 4. Generate & Save Mask and Overlay Images
        raw_image, _, _ = oil_spill_model.preprocess_image(image_bytes)
        mask_pil = oil_spill_model.generate_mask_image(binary_mask)
        overlay_pil = oil_spill_model.create_overlay_image(raw_image, binary_mask)

        mask_buf = io.BytesIO()
        mask_pil.save(mask_buf, format="PNG")
        mask_bytes = mask_buf.getvalue()

        overlay_buf = io.BytesIO()
        overlay_pil.save(overlay_buf, format="PNG")
        overlay_bytes = overlay_buf.getvalue()

        mask_url = storage_service.upload_file(
            bucket_name="sar-masks",
            destination_path=f"{analysis_id}_mask.png",
            file_bytes=mask_bytes,
            content_type="image/png"
        )

        overlay_url = storage_service.upload_file(
            bucket_name="sar-overlays",
            destination_path=f"{analysis_id}_overlay.png",
            file_bytes=overlay_bytes,
            content_type="image/png"
        )

        supabase_service.log_audit(user_id=user_id, analysis_id=analysis_id, action="SEGMENTATION_CREATED", status="SUCCESS")

        # 5. Geolocation determination
        geo = prediction.get("geospatial", {})
        lat = geo.get("latitude") or base_lat
        lon = geo.get("longitude") or base_lon

        # 6. Environmental MetOcean, Drift & Vessel Attribution (ONLY IF OIL DETECTED)
        weather_data = None
        current_data = None
        drift_data = None
        vessels = []
        report_meta = None

        if oil_detected and lat is not None and lon is not None:
            # Live wind & current
            weather_data = weather_service.get_weather(lat, lon)
            current_data = ocean_current_service.get_current(lat, lon)

            # 2-hour Lagrangian drift prediction
            area_km2 = prediction.get("spill_area_km2") or 1.0
            drift_data = drift_service.calculate_drift(
                lat=lat,
                lon=lon,
                current_speed_kn=current_data.get("current_velocity_kn", 0.8),
                current_dir_deg=current_data.get("current_direction_deg", 65.0),
                wind_speed_kn=weather_data.get("wind_speed_kn", 12.0),
                wind_dir_deg=weather_data.get("wind_direction_deg", 45.0),
                initial_area_km2=area_km2,
                hours=2
            )

            # AIS Vessel correlation
            ais_raw = ais_service.get_vessels(lat, lon, radius_km=50.0)
            vessels = correlation_service.correlate_vessels(
                spill_lat=lat,
                spill_lon=lon,
                vessels=ais_raw.get("vessels", []),
                spill_area_km2=area_km2
            )

            # Generate Report
            report_payload = {
                "analysis_id": analysis_id,
                "operator_name": operator_name,
                "evidence": {"sha256_hash": sha256_hash},
                "classification": {
                    "label": prediction["classification"],
                    "confidence": prediction["confidence"],
                    "oil_detected": True
                },
                "segmentation": {
                    "spill_area": prediction["spill_area"],
                    "spill_area_km2": prediction["spill_area_km2"],
                    "mask_url": mask_url,
                    "overlay_url": overlay_url,
                    "severity": "HIGH"
                },
                "geospatial": {"latitude": lat, "longitude": lon},
                "weather": weather_data,
                "ocean_current": current_data,
                "drift": drift_data,
                "vessels": vessels,
                "model_metadata": {
                    "model_name": prediction["model_name"],
                    "model_version": prediction["model_version"],
                    "processing_time_seconds": prediction["processing_time"]
                }
            }
            report_meta = report_service.generate_report(report_payload)
            supabase_service.log_audit(user_id=user_id, analysis_id=analysis_id, action="REPORT_GENERATED", status="SUCCESS")

        elif not oil_detected:
            # No oil detected: Do NOT invent fake vessel or fake drift data
            weather_data = None
            current_data = None
            drift_data = None
            vessels = []
            report_meta = None

        # 7. Persist to Supabase Database
        total_time = round((time.perf_counter() - t_start), 3)

        analysis_row = supabase_service.save_analysis(
            analysis_id=analysis_id,
            user_id=user_id,
            image_name=filename,
            original_image_url=original_url,
            classification=prediction["classification"],
            confidence=prediction["confidence"],
            raw_probability=prediction["raw_probability"],
            oil_pixel_count=prediction["oil_pixel_count"],
            oil_ratio=prediction["oil_ratio"],
            spill_area=prediction["spill_area"],
            spill_area_unit="km²",
            latitude=lat,
            longitude=lon,
            processing_time=total_time,
            model_name=prediction["model_name"],
            model_version=prediction["model_version"],
            inference_device=prediction["inference_device"],
            image_hash=sha256_hash,
            status="completed"
        )

        supabase_service.save_segmentation_result(
            analysis_id=analysis_id,
            mask_url=mask_url,
            overlay_url=overlay_url,
            bounding_box=prediction["primary_bounding_box"],
            oil_pixel_count=prediction["oil_pixel_count"],
            oil_ratio=prediction["oil_ratio"]
        )

        if oil_detected and vessels:
            supabase_service.save_vessel_records(analysis_id, vessels)

        if oil_detected and drift_data:
            supabase_service.save_drift_prediction(analysis_id, drift_data)

        supabase_service.log_audit(user_id=user_id, analysis_id=analysis_id, action="DATABASE_SAVED", status="SUCCESS")

        return {
            "success": True,
            "status": "completed",
            "analysis_id": analysis_id,
            "operator": {
                "name": operator_name,
                "email": operator_email,
                "user_id": user_id
            },
            "classification": {
                "label": prediction["classification"],
                "decision": prediction["classification"],
                "oil_detected": oil_detected,
                "confidence": prediction["confidence"],
                "raw_probability": prediction["raw_probability"],
                "model_name": prediction["model_name"],
                "model_version": prediction["model_version"],
                "inference_device": prediction["inference_device"],
            },
            "segmentation": {
                "mask_url": mask_url,
                "overlay_url": overlay_url,
                "spill_area": prediction["spill_area"],
                "spill_area_km2": prediction["spill_area_km2"],
                "oil_pixel_count": prediction["oil_pixel_count"],
                "oil_ratio": prediction["oil_ratio"],
                "bounding_box": prediction["primary_bounding_box"],
                "bounding_boxes": prediction["bounding_boxes"],
            },
            "geospatial": {
                "has_geospatial": geo.get("has_geospatial", False),
                "latitude": lat,
                "longitude": lon,
                "message": geo.get("message", "Geolocation metadata unavailable.")
            },
            "images": {
                "original_url": original_url,
                "mask_url": mask_url,
                "overlay_url": overlay_url
            },
            "weather": weather_data,
            "ocean_current": current_data,
            "drift": drift_data,
            "vessels": vessels,
            "report": report_meta,
            "evidence": {
                "sha256_hash": sha256_hash,
                "original_stored": True,
                "storage": "Supabase / Verified Store"
            },
            "processing_time": total_time,
            "model_status": oil_spill_model.get_status()
        }


analysis_service = AnalysisService()
