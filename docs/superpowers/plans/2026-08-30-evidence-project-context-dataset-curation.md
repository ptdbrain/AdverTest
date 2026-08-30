# Evidence-first project flow and curated datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Remove fabricated scientific metrics, gate reports and promotion on verified evidence, scope every run/artifact to a project, and curate deterministic 100-sample 2D, segmentation, and nuScenes 3D datasets.

**Architecture:** A shared evidence evaluator is the only source of benchmark eligibility for API, export and dashboard. A dry-run-first curator creates immutable manifests and a recoverable quarantine journal. Project context serializes selected project/run into canonical links; backend authorization treats those values only as scopes to verify.

**Tech Stack:** Python, FastAPI, Pydantic, SQLAlchemy, pytest, React/Next.js, Vitest, Docker Compose, JSON, nuScenes v1.0-mini.

## Global Constraints

- Never derive AP, mAP, mIoU, NDS, robustness drop, or promotion from confidence.
- Verified evidence requires dataset-version id, split manifest hash, ground truth hash, checkpoint SHA-256, config SHA-256, artifact hashes, metric protocol, and non-simulation provenance.
- Missing evidence remains NOT_ELIGIBLE, INVALID, or WAITING_FOR_GPU_VALIDATION across API/UI/export.
- Runtime artifacts use only projects/{project_id}/runs/{run_id}/; no runtime write may target frontend/public.
- Curate exactly 100 KITTI 2D pairs, 100 KITTI semantic pairs, and 100 single-sweep nuScenes keyframes. Keep five KITTI 3D fixtures marked insufficient.
- Curator never touches artifacts, checkpoints, databases, reports, or source code data.
- Compose mounts DATA_ROOT_HOST at /data:ro; CUDA is excluded from Docker E2E.

## File map

- Create src/evaluation/evidence.py: immutable evidence status and missing-requirement evaluator.
- Modify src/evaluation/report.py, src/evaluation/pdf_export.py, src/api/routers/runs.py: shared report/export/promotion contract.
- Modify src/api/routers/live_inference.py, src/api/artifacts.py, src/api/routers/artifacts.py: visual-only project/run assets.
- Create src/datasets/registry.py, scripts/curate_local_datasets.py: manifest validation and safe curation.
- Modify src/datasets/kitti.py, src/datasets/kitti3d.py, src/datasets/nuscenes.py: manifest-aware loader gates.
- Create frontend/src/context/ProjectContext.jsx; modify frontend/src/lib/api.js, app layout, TopNavigation, DashboardView and experiment pages.
- Modify docker-compose.yml and .env.example; create scripts/docker_e2e.ps1.
- Add focused pytest and Vitest tests beside each module.

### Task 1: Evidence contract and promotion eligibility

**Files:**
- Create: src/evaluation/evidence.py
- Modify: src/evaluation/report.py
- Create: tests/test_evaluation/test_evidence_gate.py

**Interfaces:**
- evaluate_evidence(provenance: Mapping[str, Any], simulation_only: bool) -> EvidenceSnapshot
- EvidenceSnapshot.status, missing, as_dict()
- RunReport.evidence and RunReport.is_promotion_eligible

- [ ] Step 1: Write the failing test:

    def test_confidence_only_report_is_not_eligible_for_promotion():
        report = RunReport(run_id="r1", model="yolo", model_version="v1", dataset="kitti", n_samples=1, ap_clean=0.0)
        assert report.evidence.status == "NOT_ELIGIBLE"
        assert report.is_promotion_eligible is False

    def test_complete_non_simulation_provenance_is_verified():
        assert make_verified_report().evidence.status == "VERIFIED"

- [ ] Step 2: Run uv run pytest tests/test_evaluation/test_evidence_gate.py -q. Expected: RED because the evidence interface does not exist.
- [ ] Step 3: Add required fields dataset_version_id, split_manifest_hash, ground_truth_hash, checkpoint_sha256, config_sha256, artifact_hashes, metric_protocol. Simulation always returns NOT_ELIGIBLE; only complete non-simulation provenance returns VERIFIED. Serialize snapshot and eligibility in RunReport.as_dict().
- [ ] Step 4: Re-run the focused test. Expected: GREEN.
- [ ] Step 5: Commit only Task 1 files with message feat: gate reports on verified evidence.

### Task 2: Visual-only inference and scoped artifact storage

**Files:**
- Modify: src/api/routers/live_inference.py, src/api/artifacts.py, src/api/routers/artifacts.py
- Create: tests/test_api/test_live_inference_evidence.py

**Interfaces:**
- Request requires project_id and accepts optional run_id.
- Response exposes visual_only=True, evidence_status=NOT_ELIGIBLE, detections and image-quality values only.
- ArtifactRepository.write_bytes(project_id, run_id, name, content, media_type) validates segments.

- [ ] Step 1: Write RED tests:

    def test_visual_inference_never_returns_benchmark_metrics(client, member_headers):
        result = client.post("/api/v1/runs/live-inference", json=live_payload, headers=member_headers)
        assert result.status_code == 200
        assert result.json()["visual_only"] is True
        assert "map50" not in result.text and "miou" not in result.text

    def test_visual_assets_are_under_project_run_namespace(...):
        result = client.post(...).json()
        assert result["attacked_image_url"].startswith("/api/v1/artifacts/projects/project-a/runs/run-a/")

- [ ] Step 2: Run uv run pytest tests/test_api/test_live_inference_evidence.py -q. Expected: RED because dynamic images are written to frontend/public and mAP/mIoU are fabricated.
- [ ] Step 3: Remove map50, miou, confidence fallback values and benchmark claims. Retain finite detection mean confidence only when available plus L2, L-infinity, PSNR and SSIM. Run attacked inference from a temporary file, write clean/attacked/diff/perturbation/prediction JSON through repository, and delete temp files in finally.
- [ ] Step 4: Re-run focused tests. Expected: GREEN and no frontend/public write.
- [ ] Step 5: Commit only Task 2 files with message fix: scope visual inference artifacts and remove fake metrics.

### Task 3: Project/run deep links, exports and promotion

**Files:**
- Modify: src/api/routers/runs.py, src/api/platform_dependencies.py, src/evaluation/pdf_export.py
- Create: tests/test_api/test_project_run_deep_links.py
- Modify: tests/test_api/test_run_artifact_authorization.py

**Interfaces:**
- resolve_project_scope(project_id, current_user) -> ProjectScope verifies active membership.
- Every run/report/ZIP/PDF/CSV/promotion route takes canonical project_id.
- POST /runs/{run_id}/promote returns 409 with evidence unless VERIFIED.

- [ ] Step 1: Write RED tests:

    def test_foreign_project_deep_link_returns_403(client, owner_b_headers, run_a):
        response = client.get(f"/api/v1/runs/{run_a}?project_id=project-a", headers=owner_b_headers)
        assert response.status_code == 403

    def test_ineligible_run_cannot_be_promoted(client, member_headers, incomplete_run):
        response = client.post(f"/api/v1/runs/{incomplete_run}/promote?project_id=project-a", headers=member_headers)
        assert response.status_code == 409
        assert response.json()["detail"]["evidence"]["status"] == "NOT_ELIGIBLE"

- [ ] Step 2: Run uv run pytest tests/test_api/test_project_run_deep_links.py tests/test_api/test_run_artifact_authorization.py -q. Expected: RED.
- [ ] Step 3: Retire X-Project-Id fallback. Validate membership before every store/repository read; foreign scope is 403 and missing run is 404. Build JSON/CSV/ZIP/PDF from same evidence snapshot. Diagnostic exports begin NOT ELIGIBLE - NO BENCHMARK CONCLUSION.
- [ ] Step 4: Re-run focused tests. Expected: GREEN.
- [ ] Step 5: Commit only Task 3 files with message feat: enforce scoped evidence exports and promotion.

### Task 4: Deterministic dry-run-first data curator

**Files:**
- Create: scripts/curate_local_datasets.py
- Create: tests/test_scripts/test_curate_local_datasets.py
- Runtime output: data/curated/registry/*.json and data/curated/.quarantine/<operation-id>/journal.json

**Interfaces:**
- build_plan(data_root: Path, count: int = 100) -> CurationPlan performs no writes.
- validate_plan(plan) -> ValidationResult reports missing pairs, bad hashes and dangling tokens.
- CLI defaults dry-run. Only --apply --operation-id UUID moves data; only --finalize UUID deletes validated quarantine paths.

- [ ] Step 1: Write RED tests:

    def test_kitti_plan_retains_exactly_100_paired_ids(tmp_path):
        plan = build_plan(make_kitti_tree(tmp_path, count=120))
        assert len(plan.kitti2d.retained_ids) == 100
        assert plan.kitti2d.missing_pairs == ()

    def test_nuscenes_plan_has_no_dangling_reference(tmp_path):
        plan = build_plan(make_nuscenes_tree(tmp_path, scenes=10, samples_per_scene=12))
        assert len(plan.nuscenes.retained_sample_tokens) == 100
        assert validate_plan(plan).dangling_tokens == ()

- [ ] Step 2: Run uv run pytest tests/test_scripts/test_curate_local_datasets.py -q. Expected: RED.
- [ ] Step 3: Implement fixed-seed class-aware KITTI selection; select exactly 10 nuScenes keyframes per scene. Retain six camera keyframes, LiDAR, five radar keyframes, calibration, ego pose, annotations and closed rewritten metadata. Declare num_sweeps=1. Retain five data/training KITTI 3D fixtures with INSUFFICIENT_FOR_REQUESTED_SUBSET.
- [ ] Step 4: Include hashes, source paths, selected ids/tokens, split, limitations and exact removal targets in plan JSON. Re-run focused tests. Expected: GREEN.
- [ ] Step 5: Commit only Task 4 files with message feat: add deterministic local dataset curator.

### Task 5: Apply user-authorized curation only after verification

**Files:** Runtime source data bounded solely by generated journal.

- [ ] Step 1: Run dry-run:
  uv run python scripts/curate_local_datasets.py --data-root D:\Project\AIthucchien\P-195\data --count 100 --dry-run --output D:\Project\AIthucchien\P-195\data\curated
  Expected: exact 100/100/100, five not-ready KITTI fixtures, exact file/byte removal totals and zero unrelated targets.
- [ ] Step 2: Validate:
  uv run python scripts/curate_local_datasets.py --data-root D:\Project\AIthucchien\P-195\data --validate-plan D:\Project\AIthucchien\P-195\data\curated\registry\curation-plan.json
  Expected: zero missing pairs and zero dangling nuScenes references.
- [ ] Step 3: Apply:
  uv run python scripts/curate_local_datasets.py --data-root D:\Project\AIthucchien\P-195\data --apply --operation-id <operation-id-from-dry-run>
  Expected: source data moves to same-volume quarantine and immutable journal exists.
- [ ] Step 4: Verify:
  uv run python scripts/curate_local_datasets.py --verify D:\Project\AIthucchien\P-195\data\curated
  Expected: exact counts and hashes.
- [ ] Step 5: Finalize only after Step 4 passes:
  uv run python scripts/curate_local_datasets.py --finalize <operation-id>
  Expected: only journal-listed sources are deleted; journal and manifests remain.

### Task 6: Registry-backed loader readiness

**Files:**
- Create: src/datasets/registry.py, tests/test_datasets/test_registry.py
- Modify: src/datasets/kitti.py, src/datasets/kitti3d.py, src/datasets/nuscenes.py, src/datasets/__init__.py, tests/test_datasets/test_kitti3d.py

**Interfaces:**
- DatasetRegistry.load(path) -> DatasetManifest
- DatasetManifest.readiness(task_id) -> DatasetReadiness(status, missing, limitations)
- Loaders accept manifest_path and reject unavailable modalities before reads.

- [ ] Step 1: Write RED tests:

    def test_kitti3d_without_calib_and_velodyne_is_not_ready(tmp_path):
        readiness = DatasetRegistry.load(write_kitti2d_only_manifest(tmp_path)).readiness("detection3d")
        assert readiness.status == "INSUFFICIENT_FOR_REQUESTED_SUBSET"
        assert set(readiness.missing) == {"calib", "velodyne"}

    def test_curated_nuscenes_returns_only_manifest_samples(curated_nuscenes):
        rows = NuScenesDataset(dataroot=str(curated_nuscenes), manifest_path=str(curated_nuscenes / "manifest.json")).load()
        assert len(rows) == 100

- [ ] Step 2: Run uv run pytest tests/test_datasets/test_registry.py tests/test_datasets/test_kitti3d.py -q. Expected: RED.
- [ ] Step 3: Validate declared pairings before reads; do not synthesize empty input. Preserve finite Nx5 LiDAR, sensor transforms, ego pose and quaternion-to-yaw conversion. CPU checks can never clear PointPillars WAITING_FOR_GPU_VALIDATION.
- [ ] Step 4: Re-run focused tests. Expected: GREEN.
- [ ] Step 5: Commit only Task 6 files with message feat: validate curated dataset manifests before loading.

### Task 7: Global project context and evidence UX

**Files:**
- Create: frontend/src/context/ProjectContext.jsx and frontend/src/context/__tests__/ProjectContext.test.jsx
- Modify: frontend/src/app/layout.jsx, frontend/src/lib/api.js, frontend/src/components/layout/TopNavigation.jsx, frontend/src/components/DashboardView.jsx, frontend/src/app/experiments/[id]/attack/page.jsx
- Create: frontend/src/components/__tests__/evidence-state.test.jsx

**Interfaces:**
- useProjectContext() -> { projectId, projects, selectProject, scopedHref }
- scopedHref(path, runId?) outputs ?project_id=...&run_id=...
- Ineligible reports show dash, No verified data, limitations and next valid action.

- [ ] Step 1: Write RED tests:

    it("synchronizes selected project to URL and clears stale run", async () => {
      render(<ProjectProvider initialProjectId="p1"><Harness /></ProjectProvider>);
      await user.click(screen.getByRole("option", { name: /project two/i }));
      expect(window.location.search).toContain("project_id=p2");
      expect(screen.getByText("No run selected")).toBeInTheDocument();
    });

    it("does not render KPI for ineligible evidence", () => {
      render(<EvidenceState evidence={{ status: "NOT_ELIGIBLE", missing: ["ground_truth_hash"] }} />);
      expect(screen.getByText("No verified data")).toBeInTheDocument();
      expect(screen.queryByText(/mAP@0.5/i)).not.toBeInTheDocument();
    });

- [ ] Step 2: Run npm test -- --run src/context/__tests__/ProjectContext.test.jsx src/components/__tests__/evidence-state.test.jsx. Expected: RED.
- [ ] Step 3: Fetch authorized projects once, validate URL project against them, replace static URLs with scopedHref, and pass project_id through canonical query scope. Project changes clear stale run state. Add status badge/limitation/next action while retaining drawer/header and 44px controls.
- [ ] Step 4: Re-run focused Vitest. Expected: GREEN.
- [ ] Step 5: Commit only Task 7 files with message feat: synchronize project deep links and evidence states.

### Task 8: Docker E2E and verification

**Files:**
- Modify: docker-compose.yml, .env.example
- Create: scripts/docker_e2e.ps1, tests/e2e/test_docker_evidence_flow.py, docs/guide/evidence-and-curated-datasets.md

- [ ] Step 1: Write RED parser/E2E tests requiring a read-only data mount and project -> run -> artifact -> deep-link -> JSON/CSV/PDF -> 409 promotion flow.
- [ ] Step 2: Run uv run pytest tests/e2e/test_docker_evidence_flow.py -q. Expected: RED.
- [ ] Step 3: Make absent DATA_ROOT_HOST a Compose configuration failure, add backend/worker mounts and health waiting, and teardown without deleting named volumes.
- [ ] Step 4: Run:
  $env:DATA_ROOT_HOST='D:\Project\AIthucchien\P-195\data\curated'; powershell -ExecutionPolicy Bypass -File scripts/docker_e2e.ps1
  Expected: health/E2E pass and CUDA reported not-run.
- [ ] Step 5: Run uv run ruff check src tests scripts; uv run pytest -q --basetemp .pytest-evidence-final; npm test -- --run; npm run build; git diff --check.
- [ ] Step 6: Document manifests, evidence statuses, quarantine restoration/finalization, Docker root and CUDA external gate. State subsets are development evidence, not full-split benchmark results.
- [ ] Step 7: Commit only Task 8 files with message test: verify docker evidence flow with readonly data.

## Self-review

- [ ] Tasks 1-3 cover scientific truth, reports, promotion, artifacts and deep links.
- [ ] Tasks 4-6 cover exact curation counts, safe deletion, registry and CPU/CUDA boundary.
- [ ] Tasks 7-8 cover global context, UX, Docker and E2E.
- [ ] Every production behavior has a preceding test expected to fail for the right reason.
- [ ] No task claims full NDS, CUDA inference, or official benchmark completion.

