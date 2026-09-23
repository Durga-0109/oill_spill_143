-- ====================================================================
-- SpillWatch AI - Supabase PostgreSQL Schema & Storage Configuration
-- Tables: users, analyses, segmentation_results, vessel_records, drift_predictions, audit_logs
-- Storage Buckets: sar-originals, sar-masks, sar-overlays, reports
-- ====================================================================

-- 1. USERS TABLE
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL DEFAULT 'operator',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed default operator if not exists
INSERT INTO public.users (name, email, role)
VALUES ('Durga C', 'durga@spillwatch.maritime.gov', 'lead_operator')
ON CONFLICT (email) DO NOTHING;

-- 2. ANALYSES TABLE
CREATE TABLE IF NOT EXISTS public.analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id TEXT UNIQUE NOT NULL,
    user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
    image_name TEXT NOT NULL,
    original_image_url TEXT,
    classification TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    raw_probability DOUBLE PRECISION,
    oil_pixel_count INTEGER DEFAULT 0,
    oil_ratio DOUBLE PRECISION DEFAULT 0.0,
    spill_area TEXT,
    spill_area_unit TEXT DEFAULT 'km²',
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    processing_time DOUBLE PRECISION,
    model_name TEXT DEFAULT 'Attention U-Net',
    model_version TEXT DEFAULT 'v1.0.0',
    inference_device TEXT DEFAULT 'CPU',
    image_hash TEXT,
    status TEXT NOT NULL DEFAULT 'completed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. SEGMENTATION RESULTS TABLE
CREATE TABLE IF NOT EXISTS public.segmentation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id TEXT NOT NULL REFERENCES public.analyses(analysis_id) ON DELETE CASCADE,
    mask_url TEXT,
    overlay_url TEXT,
    bounding_box JSONB,
    oil_pixel_count INTEGER DEFAULT 0,
    oil_ratio DOUBLE PRECISION DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. VESSEL RECORDS TABLE
CREATE TABLE IF NOT EXISTS public.vessel_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id TEXT NOT NULL REFERENCES public.analyses(analysis_id) ON DELETE CASCADE,
    vessel_id TEXT NOT NULL,
    vessel_name TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    speed DOUBLE PRECISION,
    course DOUBLE PRECISION,
    distance_from_spill DOUBLE PRECISION,
    correlation_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. DRIFT PREDICTIONS TABLE
CREATE TABLE IF NOT EXISTS public.drift_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id TEXT NOT NULL REFERENCES public.analyses(analysis_id) ON DELETE CASCADE,
    current_latitude DOUBLE PRECISION,
    current_longitude DOUBLE PRECISION,
    predicted_latitude_1h DOUBLE PRECISION,
    predicted_longitude_1h DOUBLE PRECISION,
    predicted_latitude_2h DOUBLE PRECISION,
    predicted_longitude_2h DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    wind_direction DOUBLE PRECISION,
    current_speed DOUBLE PRECISION,
    current_direction DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. AUDIT LOGS TABLE
CREATE TABLE IF NOT EXISTS public.audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
    analysis_id TEXT,
    action TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'SUCCESS',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- INDEXES FOR HIGH-PERFORMANCE SEARCH & KPI QUERIES
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON public.analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_classification ON public.analyses(classification);
CREATE INDEX IF NOT EXISTS idx_analyses_analysis_id ON public.analyses(analysis_id);
CREATE INDEX IF NOT EXISTS idx_segmentation_analysis_id ON public.segmentation_results(analysis_id);
CREATE INDEX IF NOT EXISTS idx_vessels_analysis_id ON public.vessel_records(analysis_id);
CREATE INDEX IF NOT EXISTS idx_drift_analysis_id ON public.drift_predictions(analysis_id);
CREATE INDEX IF NOT EXISTS idx_audit_analysis_id ON public.audit_logs(analysis_id);

-- ROW LEVEL SECURITY (RLS) POLICIES
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.segmentation_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vessel_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.drift_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

-- Allow authenticated and service roles to read all analysis data
CREATE POLICY "Allow public read access to analyses" ON public.analyses
    FOR SELECT USING (true);

CREATE POLICY "Allow service role full access to analyses" ON public.analyses
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow public read access to segmentation" ON public.segmentation_results
    FOR SELECT USING (true);

CREATE POLICY "Allow service role full access to segmentation" ON public.segmentation_results
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow public read access to vessels" ON public.vessel_records
    FOR SELECT USING (true);

CREATE POLICY "Allow service role full access to vessels" ON public.vessel_records
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow public read access to drift" ON public.drift_predictions
    FOR SELECT USING (true);

CREATE POLICY "Allow service role full access to drift" ON public.drift_predictions
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow service role full access to audit_logs" ON public.audit_logs
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "Allow service role full access to users" ON public.users
    FOR ALL USING (true) WITH CHECK (true);

-- STORAGE BUCKETS SETUP IN SUPABASE
-- Run these in Supabase SQL editor or via Supabase dashboard:
INSERT INTO storage.buckets (id, name, public)
VALUES 
    ('sar-originals', 'sar-originals', true),
    ('sar-masks', 'sar-masks', true),
    ('sar-overlays', 'sar-overlays', true),
    ('reports', 'reports', true)
ON CONFLICT (id) DO UPDATE SET public = true;

-- Storage RLS: Allow public reads for display in UI
CREATE POLICY "Allow public read access for sar-originals" ON storage.objects
    FOR SELECT USING (bucket_id = 'sar-originals');

CREATE POLICY "Allow public read access for sar-masks" ON storage.objects
    FOR SELECT USING (bucket_id = 'sar-masks');

CREATE POLICY "Allow public read access for sar-overlays" ON storage.objects
    FOR SELECT USING (bucket_id = 'sar-overlays');

CREATE POLICY "Allow public read access for reports" ON storage.objects
    FOR SELECT USING (bucket_id = 'reports');
