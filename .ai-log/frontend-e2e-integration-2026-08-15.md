# AI Log — AdverTest Frontend End-to-End Integration

**Date:** 2026-08-15
**Branch:** `feat/advertest-core-wave0`
**Task:** Hoàn thiện end-to-end FE ↔ BE integration

---

## Changes Made

### 1. API Base URL Fix (`frontend/src/lib/api.js`)
- Changed `API_BASE` from `http://localhost:8000` → `http://127.0.0.1:8000`
- **Reason:** Windows IPv6 resolution can cause `localhost` to resolve to `::1` instead of `127.0.0.1`, causing browser fetch to fail when Uvicorn only binds to IPv4

### 2. Fallback Data Cleanup (`frontend/src/hooks/useAdverTest.js`)
- Removed fake `FALLBACK_REPORT`, `FALLBACK_SAMPLES`, and `setActiveTab("comparison")` from fallback handler
- **Reason:** Fallback should only populate catalog data (modes, families, checkpoints, datasets, attacks) for UI rendering — not pretend a run has completed
- When backend is available, real data now flows through cleanly without interference

### 3. Duplicate Page Fix
- Deleted `frontend/src/app/reviews/page.js` (was a re-export of `page.jsx`)
- **Reason:** Next.js 16 treats both files as separate page routes, causing "Duplicate page detected" warning

### 4. Key Prop Fix (`frontend/src/components/UploadModal.jsx`)
- Changed `key={idx}` to `key={item.url || item.name || idx}` in file preview list
- **Reason:** React "Each child should have a unique key" warning

## Verification

### Backend (FastAPI on :8000)
- ✅ `/health` → 200 OK
- ✅ `/api/v1/catalog/attacks` → 41 attacks returned
- ✅ `/api/v1/catalog/datasets` → 5 datasets returned
- ✅ `/api/v1/perception-modes` → 3 modes returned
- ✅ `/api/v1/model-families` → OK
- ✅ `/api/v1/base-checkpoints` → OK
- ✅ `/api/v1/reviews` → OK
- ✅ `/api/v1/failure-clusters` → OK

### Frontend (Next.js on :3000)
- ✅ No compilation errors
- ✅ No "duplicate page" warnings
- ✅ No "key prop" warnings
- ✅ No "Backend unavailable" fallback messages
- ✅ All 8 catalog API calls return 200 OK from backend logs
- ✅ Reviews page loads and fetches data correctly
