import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def run_tests():
    print("=== STARTING END-TO-END PIPELINE TESTS ===")

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.status_code}"
    print(f"[PASS] /health: {res.json().get('status')}")

    # 2. Dashboard stats
    res = client.get("/api/dashboard/statistics")
    assert res.status_code == 200, f"Dashboard stats failed: {res.status_code}"
    print(f"[PASS] /api/dashboard/statistics: OK (total spills: {res.json().get('total_oil_spills')})")

    # 3. Model samples mount
    res = client.get("/model-samples/oil_sample_1.jpg")
    assert res.status_code == 200, f"Model samples static mount failed: {res.status_code}"
    print(f"[PASS] /model-samples/oil_sample_1.jpg: OK ({len(res.content)} bytes)")

    # 4. Manual analysis with sample image
    sample_path = Path("binary classification-20260916T134200Z-1-001/binary classification/model_service_handoff/sample_images/oil_sample_1.jpg")
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    res = client.post(
        "/api/analyze/manual",
        files={"file": ("oil_sample_1.jpg", file_bytes, "image/jpeg")},
        data={"operator_id": "TEST_OP", "base_lat": "11.23", "base_lon": "72.45"}
    )
    assert res.status_code == 200, f"Manual analysis failed: {res.status_code} - {res.text}"
    data = res.json()
    analysis_id = data.get("analysis_id")
    assert analysis_id, "Missing analysis_id"
    assert "evidence" in data and "sha256_hash" in data["evidence"], "Missing evidence sha256_hash"
    assert "classification" in data, "Missing classification"
    assert "segmentation" in data, "Missing segmentation"
    assert "drift" in data, "Missing drift"
    assert "weather" in data, "Missing weather"
    assert "vessels" in data, "Missing vessels"
    print(f"[PASS] /api/analyze/manual: ID={analysis_id}, Detected={data['classification']['oil_detected']}, Conf={data['classification']['confidence']}, Area={data['segmentation']['spill_area_km2']} km2")
    print(f"       Evidence SHA-256: {data['evidence']['sha256_hash'][:16]}...")
    print(f"       MetOcean Wind: {data['weather']['wind_speed_kn']} kn @ {data['weather']['wind_direction_deg']} deg")
    print(f"       Drift Speed: {data['drift']['drift_velocity_kn']} kn {data['drift']['drift_cardinal']}, Trajectories: {len(data['drift']['trajectories'])}")
    print(f"       Correlated Vessels: {len(data['vessels'])}")

    # 5. Automatic analysis
    res_auto = client.post("/api/analyze/automatic")
    assert res_auto.status_code == 200, f"Automatic analysis failed: {res_auto.status_code} - {res_auto.text}"
    data_auto = res_auto.json()
    print(f"[PASS] /api/analyze/automatic: ID={data_auto.get('analysis_id')}, Status={data_auto.get('status')}")

    # 6. Report endpoints
    res_html = client.get(f"/api/report/{analysis_id}")
    assert res_html.status_code == 200, f"HTML report failed: {res_html.status_code}"
    assert "SpillWatch" in res_html.text, "Report HTML missing title"
    print(f"[PASS] /api/report/{analysis_id} (HTML Forensic Report): OK")

    res_pdf = client.get(f"/api/report/{analysis_id}/pdf")
    assert res_pdf.status_code == 200, f"PDF report failed: {res_pdf.status_code}"
    print(f"[PASS] /api/report/{analysis_id}/pdf: OK ({len(res_pdf.content)} bytes)")

    # 7. Audit logs
    res_audit = client.get("/api/audit-logs")
    assert res_audit.status_code == 200, f"Audit logs failed: {res_audit.status_code}"
    print(f"[PASS] /api/audit-logs: OK ({len(res_audit.json().get('logs', []))} entries)")

    print("=== ALL END-TO-END PIPELINE TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
