import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class ReportService:
    """
    Generates standardized forensic maritime incident reports with SHA-256 evidence hashes,
    model versioning traceability, vessel attribution, MetOcean vectors, and regulatory disclaimers.
    """

    def __init__(self, reports_dir: Optional[Path] = None):
        if reports_dir is None:
            self.reports_dir = Path(__file__).resolve().parent.parent / "results" / "reports"
        else:
            self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds structured report JSON and printable HTML report file.
        """
        analysis_id = analysis_data.get("analysis_id", f"SW-REP-{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
        classification = analysis_data.get("classification", {})
        segmentation = analysis_data.get("segmentation", {})
        geospatial = analysis_data.get("geospatial", {})
        vessels = analysis_data.get("vessels", [])
        weather = analysis_data.get("weather", {})
        drift = analysis_data.get("drift", {})
        evidence = analysis_data.get("evidence", {})
        model_meta = analysis_data.get("model_metadata", {})

        top_vessel = vessels[0] if vessels else {}
        sha256 = evidence.get("sha256_hash", "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855")
        now_str = datetime.datetime.utcnow().strftime("%d %b %Y, %H:%M:%S UTC")

        disclaimer = (
            "NOTICE & LEGAL DISCLAIMER: AI output is a presumptive field-analysis result and should not be treated "
            "as final laboratory confirmation. Attributed vessel rankings represent kinematic and spatio-temporal correlation "
            "likelihood estimates based on AIS telemetry and Lagrangian hydrodynamic drift models."
        )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SpillWatch Intelligence Report - {analysis_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 32px; color: #0f172a; background: #ffffff; line-height: 1.5; }}
        .header {{ border-bottom: 2px solid #2563eb; padding-bottom: 16px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-start; }}
        .header h1 {{ margin: 0; font-size: 24px; color: #0f172a; letter-spacing: -0.5px; }}
        .header .sub {{ color: #64748b; font-size: 12px; margin-top: 4px; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 11px; text-transform: uppercase; }}
        .badge-detected {{ background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }}
        .badge-clean {{ background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }}
        .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 24px; }}
        .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; }}
        .card .label {{ font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
        .card .value {{ font-size: 16px; font-weight: 800; color: #0f172a; }}
        .section-title {{ font-size: 14px; font-weight: 800; text-transform: uppercase; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin: 24px 0 12px; letter-spacing: 0.5px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
        th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
        th {{ background: #f1f5f9; font-weight: 700; color: #475569; }}
        .hash-box {{ background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 6px; padding: 10px 14px; font-family: monospace; font-size: 11px; word-break: break-all; margin: 16px 0; }}
        .disclaimer {{ background: #fffbeb; border: 1px solid #fef3c7; border-radius: 6px; padding: 12px 14px; font-size: 11px; color: #92400e; margin-top: 24px; }}
        @media print {{ body {{ padding: 0; }} .no-print {{ display: none; }} }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>SPILLWATCH / MARITIME INTELLIGENCE</h1>
            <div class="sub">Automated Satellite SAR Oil Spill Detection & AIS Forensic Attribution Report</div>
        </div>
        <div style="text-align:right;">
            <div class="badge {'badge-detected' if classification.get('oil_detected') else 'badge-clean'}">
                {'OIL SPILL DETECTED' if classification.get('oil_detected') else 'CLEAN OCEAN'}
            </div>
            <div style="font-size:11px;color:#64748b;margin-top:4px;">Report ID: <strong>{analysis_id}</strong></div>
        </div>
    </div>

    <div class="grid">
        <div class="card"><div class="label">Analysis Timestamp</div><div class="value" style="font-size:13px;">{now_str}</div></div>
        <div class="card"><div class="label">AI Confidence</div><div class="value" style="color:#0284c7;">{classification.get('confidence', 0.0) * 100:.1f}%</div></div>
        <div class="card"><div class="label">Estimated Slick Area</div><div class="value" style="color:#e11d48;">{segmentation.get('spill_area_km2', 0.0)} km²</div></div>
        <div class="card"><div class="label">Spill Centroid Coordinates</div><div class="value" style="font-size:13px;">{geospatial.get('latitude', 11.23):.3f}° N, {geospatial.get('longitude', 72.45):.3f}° E</div></div>
        <div class="card"><div class="label">Top Correlated Vessel</div><div class="value" style="font-size:13px;color:#b91c1c;">{top_vessel.get('vessel_name', 'N/A')} ({top_vessel.get('attribution_score', 0)}%)</div></div>
        <div class="card"><div class="label">2-Hour Drift Vector</div><div class="value" style="font-size:13px;color:#d97706;">{drift.get('drift_cardinal', 'NE')} @ {drift.get('drift_velocity_kn', 1.8)} kn</div></div>
    </div>

    <div class="section-title">1. Forensic Evidence & Cryptographic Hash</div>
    <div class="hash-box">
        <strong>SHA-256 Evidence Hash:</strong> {sha256}<br>
        <strong>Original File:</strong> {evidence.get('original_filename', 'satellite_scene.jpg')} | <strong>File Size:</strong> {evidence.get('file_size_bytes', 0) / 1024:.1f} KB<br>
        <strong>AI Model:</strong> {model_meta.get('model_name', 'ResNet-18 + Attention Segmentor')} (v{model_meta.get('model_version', '1.0.0')}) | <strong>Hash:</strong> {model_meta.get('model_hash', 'N/A')}
    </div>

    <div class="section-title">2. MetOcean Environmental Conditions & 2-Hour Drift</div>
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Source / Model</th></tr>
        <tr><td>Surface Wind Vector</td><td>{weather.get('wind_speed_kn', 16.0)} kn @ {weather.get('wind_direction_deg', 240.0)}° ({weather.get('wind_cardinal', 'WSW')})</td><td>{weather.get('source', 'Open-Meteo Global Marine')}</td></tr>
        <tr><td>Surface Ocean Current</td><td>{drift.get('weather_input', {}).get('current_speed_kn', 1.8)} kn @ {drift.get('weather_input', {}).get('current_direction_deg', 45.0)}°</td><td>Ocean Hydrodynamic Model</td></tr>
        <tr><td>Lagrangian Drift Velocity</td><td>{drift.get('drift_velocity_kn', 1.8)} knots ({drift.get('drift_direction_deg', 45.0)}° {drift.get('drift_cardinal', 'NE')})</td><td>Physics-based Advection (alpha=0.030)</td></tr>
        <tr><td>2-Hour Drifted Distance</td><td>{drift.get('total_2h_distance_km', 6.7)} km (Predicted Area: {drift.get('predicted_2h_area_km2', 18.2)} km²)</td><td>Slick Expansion Model (+12%/h)</td></tr>
    </table>

    <div class="section-title">3. AIS Vessel Correlation & Kinematics Attribution</div>
    <table>
        <tr><th>Rank</th><th>Vessel Name</th><th>MMSI</th><th>Type</th><th>Distance</th><th>Speed / Course</th><th>Travel Time</th><th>Attribution Score</th></tr>
        {''.join([f"<tr><td><strong>{v.get('rank', '#')}</strong></td><td><strong>{v.get('vessel_name', '')}</strong></td><td>{v.get('mmsi', '')}</td><td>{v.get('vessel_type', '')}</td><td>{v.get('distance_km', '')} km</td><td>{v.get('speed_kn', '')} kn / {v.get('direction', '')}</td><td>{v.get('travel_time', '')}</td><td><strong style='color:{'#b91c1c' if v.get('attribution_score',0)>=80 else '#d97706'}'>{v.get('attribution_score', '')}%</strong></td></tr>" for v in vessels[:6]])}
    </table>

    <div class="disclaimer">
        {disclaimer}
    </div>
</body>
</html>"""

        report_file_path = self.reports_dir / f"{analysis_id}.html"
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return {
            "analysis_id": analysis_id,
            "report_url": f"/results/reports/{analysis_id}.html",
            "report_file_path": str(report_file_path),
            "generated_at": now_str,
            "html_content": html_content,
            "sha256_hash": sha256
        }

report_service = ReportService()
