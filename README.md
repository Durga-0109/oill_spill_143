# SpillWatch AI — Real-Time Maritime SAR Oil Spill Detection & Attribution Platform

SpillWatch AI is an operational maritime intelligence platform designed to ingest Sentinel-1 Synthetic Aperture Radar (SAR) imagery, perform pixel-level oil spill segmentation using a custom **Attention U-Net** architecture, correlate nearby vessel movements via AIS kinematics, compute 2-hour Lagrangian ocean drift forecasts, and persist all verified evidence into **Supabase PostgreSQL & Storage**.

---

## Architecture Overview

```
SATELLITE SAR SCENE (GeoTIFF / PNG / JPG)
        ↓
IMAGE VALIDATION & SHA-256 HASHING
        ↓
ATTENTION U-NET SEGMENTATION (PyTorch GPU/CPU)
        ↓
PIXEL PROBABILITY MAP & THRESHOLDING
        ↓
BINARY MASK & BOUNDING BOX EXTRACTION
        ↓
SEMI-TRANSPARENT RED OVERLAY COMPOSITOR
        ↓
SUPABASE STORAGE BUCKETS (sar-originals, sar-masks, sar-overlays, reports)
        ↓
SUPABASE DATABASE (users, analyses, segmentation_results, vessel_records, drift_predictions, audit_logs)
        ↓
REAL-TIME DASHBOARD (Dynamic KPIs, Tactile Map, Attribution Ranking, Forensic Reports)
```

---

## 1. Installation

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- CUDA-enabled GPU (optional, auto-falls back to CPU)

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-org/spillwatch.git
cd spillwatch

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Supabase Setup

SpillWatch uses **Supabase** for enterprise data integrity, Row Level Security (RLS), and binary asset storage.

1. Go to [supabase.com](https://supabase.com) and create a new project.
2. Note your **Project URL** and **API Keys** (`anon` public and `service_role` secret) from:
   `Project Settings -> API`
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Enter your keys in `.env`:
   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=eyJhbGciOi...
   SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOi...
   ```

---

## 3. Database Schema

Execute the complete SQL schema in the **Supabase SQL Editor**:
- Open file: [`supabase_schema.sql`](supabase_schema.sql)
- Copy all content into Supabase SQL Editor and click **Run**.

This provisions:
- `users`: Operator profiles (defaults to Lead Operator Durga C)
- `analyses`: Master table storing SHA-256, model metadata, confidence, classification, and device
- `segmentation_results`: Mask URLs, overlay URLs, pixel counts, and bounding box coordinates
- `vessel_records`: Correlated AIS vessels and attribution scores
- `drift_predictions`: 1h and 2h Lagrangian drift coordinates
- `audit_logs`: Immutable chronological log of each pipeline phase
- Verified Row Level Security (RLS) policies

---

## 4. Storage Bucket Setup

The SQL script automatically provisions the four required buckets:
1. `sar-originals`: Immutable original SAR scenes
2. `sar-masks`: Pixel-accurate binary segmentation masks (0=non-oil, 1=oil)
3. `sar-overlays`: Composite visualizations with red overlays on detected oil pixels
4. `reports`: Generated forensic PDF & HTML reports

*Note: If Supabase credentials are not provided, the platform automatically saves assets to `backend/uploads/evidence/` and `backend/results/` as a seamless local fallback.*

---

## 5. Authentication Setup

- SpillWatch integrates with Supabase Auth.
- The active operator's profile is retrieved via `GET /api/auth/profile`.
- The operator's name is dynamically displayed in the top bar (`Operator: Durga C`).
- Every analysis records the `user_id` to establish forensic custody.

---

## 6. Dataset Setup

SpillWatch requires real, labeled SAR oil spill imagery for model training.

Organize your dataset in the `data/` directory:
```
data/
├── train/
│   ├── images/   (SAR scenes: .png, .jpg, .tif)
│   └── masks/    (Binary masks: 0=non-oil, 1=oil spill)
├── validation/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

*If you have an unorganized folder of images and masks, use our automated splitting tool:*
```bash
python ml/prepare_dataset.py --images path/to/raw_images --masks path/to/raw_masks
```

---

## 7. Model Training

Train the **Attention U-Net** on your labeled dataset:
```bash
python ml/train.py --epochs 50 --batch-size 8 --lr 0.0001 --img-size 256
```

### Features:
- **Architecture**: 4-level Contracting Encoder + Attention Gates + Expanding Decoder
- **Loss**: Combined Soft Dice Loss + BCEWithLogitsLoss
- **Metrics Tracked**: Dice Score, IoU (Jaccard), Precision, Recall, F1 Score
- **Artifacts Generated**:
  - `models/oil_spill_attention_unet.pth`: Best checkpoint based on validation Dice score
  - `models/model_metadata.json`: Real verified metrics, parameters, and training date

*If no training dataset is present, the script clearly reports: `REAL LABELED SAR OIL-SPILL DATASET REQUIRED` without fabricating fake metrics.*

---

## 8. Model Evaluation & CLI Inference

### Evaluate on Test Split:
```bash
python ml/evaluate.py --data-dir data --model-path models/oil_spill_attention_unet.pth
```

### Run Standalone CLI Inference on a SAR Scene:
```bash
python ml/inference.py path/to/sar_scene.jpg --output-dir results --threshold 0.5
```
Outputs:
- `results/<name>_mask.png`: Binary mask
- `results/<name>_overlay.png`: Semi-transparent red overlay strictly on detected oil pixels

---

## 9. Backend Startup

Start the FastAPI application:
```bash
python run.py
```
Or directly with Uvicorn:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at: `http://localhost:8000/docs`

---

## 10. Frontend Startup

The existing SpillWatch user interface runs in any standard web browser:
1. Simply double-click `index.html` or serve via Python:
   ```bash
   python -m http.server 3000
   ```
2. Open `http://localhost:3000`
3. Click **API Settings** in the top bar to set the backend URL (default: `http://localhost:8000`).

---

## 11. Environment Variables Reference

| Variable | Description | Default |
| :--- | :--- | :--- |
| `SUPABASE_URL` | Supabase Project URL | *None (Local fallback)* |
| `SUPABASE_ANON_KEY` | Supabase Public Anonymous Key | *None* |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase Service Role Key (Backend only) | *None* |
| `OIL_THRESHOLD` | Sigmoid probability threshold for oil classification | `0.50` |
| `MODEL_WEIGHTS_PATH` | Path to trained Attention U-Net weights | `models/oil_spill_attention_unet.pth` |
| `AISSTREAM_API_KEY` | Live AIS WebSocket Key (aisstream.io) | *None* |
| `PORT` | API Server Port | `8000` |

---

## 12. Deployment Instructions

### Docker Deployment:
```bash
docker build -t spillwatch-backend .
docker run -p 8000:8000 --env-file .env spillwatch-backend
```

### Cloud Run / Render / VPS Deployment:
1. Push to GitHub.
2. In your cloud provider, set the root directory to repository root and start command:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port $PORT
   ```
3. Add environment variables from `.env`.

---

## 13. Troubleshooting

- **`REAL LABELED SAR OIL-SPILL DATASET REQUIRED`**:
  The system strictly enforces no-fake-data. Place labeled masks in `data/train/masks/` and run `python ml/train.py`.
- **Supabase Connectivity**:
  If Supabase keys are omitted or invalid, SpillWatch automatically activates its verified local storage fallback so development and demonstrations never crash.
- **CUDA Out of Memory**:
  Reduce batch size during training: `python ml/train.py --batch-size 4 --img-size 256`.

---

### Preserved Features
- Real Leaflet Maps (Centroids, bounding boxes, trajectories)
- Live Open-Meteo Wind & Ocean Marine Current Vector Integration
- Kinematic AIS Vessel Correlation & Risk Attribution
- Forensic SHA-256 Custody Chains & Audit Logs
