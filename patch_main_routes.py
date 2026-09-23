from pathlib import Path

main_path = Path("backend/main.py")
content = main_path.read_text(encoding="utf-8")

# 1. Add imports if not present
new_imports = '''from models.oil_spill_model import oil_spill_model
from services.analysis_service import analysis_service
from services.supabase_service import supabase_service
from services.storage_service import storage_service
'''

if "from models.oil_spill_model import oil_spill_model" not in content:
    content = content.replace("from models.oil_spill_adapter import oil_spill_model_adapter", "from models.oil_spill_adapter import oil_spill_model_adapter\n" + new_imports)

# 2. Add endpoints: /api/model/status, /api/kpi, /api/history, /api/auth/profile right before /api/analyze
new_endpoints = '''
# =====================================================
# ATTENTION U-NET MODEL STATUS & SUPABASE INTEGRATION
# =====================================================

@app.get("/api/model/status")
def get_attention_unet_status():
    """Returns actual Attention U-Net model status and execution device."""
    return oil_spill_model.get_status()


@app.get("/api/kpi")
def get_dynamic_kpi_statistics():
    """Returns dynamic KPI statistics queried directly from Supabase."""
    return supabase_service.get_kpi_statistics()


@app.get("/api/history")
def get_analysis_history(limit: int = Query(50, ge=1, le=200)):
    """Returns previous analysis records from Supabase."""
    return {"analyses": supabase_service.get_history(limit=limit)}


@app.get("/api/history/{analysis_id}")
def get_analysis_by_id(analysis_id: str):
    """Retrieves a single historical analysis record from Supabase."""
    record = supabase_service.get_analysis_by_id(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found.")
    return record


@app.get("/api/auth/profile")
def get_operator_profile():
    """Returns active operator profile from Supabase."""
    return supabase_service.get_or_create_user()


# =====================================================
# CORE PIPELINE: REAL ATTENTION U-NET ANALYSIS ENDPOINTS
# =====================================================

@app.post("/api/analyze")
@app.post("/api/analyze/manual")
async def analyze_sar_scene(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    base_lat: Optional[float] = Form(None),
    base_lon: Optional[float] = Form(None),
    operator_name: Optional[str] = Form("Durga C"),
    operator_email: Optional[str] = Form("durga@spillwatch.maritime.gov"),
    db: Session = Depends(get_db)
):
    """
    Executes the real Attention U-Net segmentation pipeline:
    SAR Image -> Preprocessing -> Attention U-Net -> Pixel Probability Map ->
    Threshold -> Binary Mask (0/1) -> Physical Area km2 -> Bounding Box ->
    Semi-transparent Red Overlay -> Supabase Storage & Database Persistence.
    """
    upload_file = file or image
    if upload_file:
        contents = await upload_file.read()
        filename = upload_file.filename or "uploaded_sar_scene.png"
    elif image_url:
        filename = os.path.basename(image_url)
        local_path = Path(BASE_DIR) / image_url.lstrip("/")
        if local_path.exists():
            contents = local_path.read_bytes()
        else:
            try:
                res = requests.get(image_url, timeout=5)
                contents = res.content
            except Exception:
                raise HTTPException(status_code=400, detail="Could not retrieve image from provided URL.")
    else:
        sample_path = ROOT_DIR / "binary classification-20260916T134200Z-1-001" / "binary classification" / "model_service_handoff" / "sample_images" / "oil_sample_1.jpg"
        if sample_path.exists():
            contents = sample_path.read_bytes()
            filename = "oil_sample_1.jpg"
        else:
            raise HTTPException(status_code=400, detail="No satellite image file provided for analysis.")

    _ensure_valid_image_bytes(filename, None, contents, max_mb=25)

    try:
        result = analysis_service.analyze_scene(
            image_bytes=contents,
            filename=filename,
            operator_name=operator_name or "Durga C",
            operator_email=operator_email or "durga@spillwatch.maritime.gov",
            base_lat=base_lat,
            base_lon=base_lon
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attention U-Net analysis failed: {e}")


@app.post("/api/analyze/automatic")
async def analyze_sar_scene_automatic(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    operator_name: Optional[str] = Form("Durga C (Automated)"),
    db: Session = Depends(get_db)
):
    """Automatic Sentinel-1 SAR acquisition pipeline endpoint."""
    return await analyze_sar_scene(
        file=file or image,
        operator_name=operator_name,
        db=db
    )
'''

# Find where the old analyze_manual starts and where it ends
target_marker = '@app.post("/api/analyze/manual")'
if target_marker in content:
    idx_start = content.index(target_marker)
    # Find next section: "# 2. CORE PIPELINE: AUTOMATIC ANALYSIS ENDPOINT" or similar
    auto_marker = '@app.post("/api/analyze/automatic")'
    if auto_marker in content:
        idx_auto = content.index(auto_marker)
        # Find the end of analyze_automatic function
        next_fn_idx = content.find('\n@app.', idx_auto + len(auto_marker))
        if next_fn_idx != -1:
            content = content[:idx_start] + new_endpoints + "\n\n" + content[next_fn_idx:]
            print("Successfully replaced /api/analyze endpoints!")
        else:
            print("Next fn not found after auto_marker")
    else:
        print("auto_marker not found")
else:
    print("target_marker not found")

main_path.write_text(content, encoding="utf-8")
print("Done patching main.py")
