# Person D Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete every deliverable owned by Person D: versioned datasets and locked splits, leakage protection, attack catalog/recipes/composition, generated and training datasets, benchmark protocol/runner, robustness/recovery comparison, hard-example storage, and generic compute/training orchestration for YOLO11 and SAM2 consumers.

**Architecture:** Extend the working single-attack pipeline instead of replacing it. Introduce immutable versioned contracts first, then place dataset, recipe, generation, benchmark, metrics, and training orchestration behind small service interfaces. Existing single-attack CLI/configs become compatibility adapters that create one-step recipes, while model-specific objectives/evaluators/trainers remain owned by Persons B and C and the Web/API persistence remains owned by Person A.

**Tech Stack:** Python 3.11-3.13, Pydantic v2, NumPy, Pillow, SQLite, pytest, Ruff, uv; optional PyTorch/Ultralytics/SAM2 dependencies remain lazy-loaded.

## Global Constraints

- Work only under `D:\Project\AIthucchien\P-195`; do not modify `DAY01_2A202601138_PhanTrongDat`.
- Start execution in an isolated worktree/branch from the intended integration base; preserve `PRDs.md` and the current line-ending-only working-tree changes.
- Do not build frontend components or Web API routes in this plan.
- Do not implement `YoloTrainer`, `Sam2Trainer`, detection-domain metrics, segmentation-domain metrics, or choose model hyperparameters; expose contracts for B/C implementations.
- P0/P1 covers 2D detection and segmentation. PointPillars, BEVFusion, 3D defense, automated red-team search, and automated clustering remain P2 and do not block Person D Definition of Done.
- All public configs use Pydantic `extra="forbid"` and immutable/versioned identifiers.
- Every stochastic operation receives an explicit seed; never use process-global NumPy/Python RNG state.
- Every artifact records source hash, contract/version hashes, implementation versions, seed, model/checkpoint provenance, intended use, and validation status.
- No dataset, model checkpoint, generated data, cache database, or patch artifact is committed to Git.
- Detection headline metric is mAP@[.50:.95]; AP50/AP75 remain supplemental. Segmentation headline metric is mIoU with Boundary IoU supplemental.
- Metric payloads expose explicit units (`ratio`, `percent`, `point`, `percentage_point`) and never overload one field with multiple units.
- The generator never downloads model weights. Heavy model dependencies are optional and lazy-loaded.
- Existing `generate-attack`, `benchmark-attack-dataset`, `TestRunner`, and generated dataset format remain readable during migration.
- Use `uv run --no-sync` for routine validation after the environment has been synced.

## Source Specification and Scope

Authoritative requirements:

- `Ke-hoach-hop-nhat-cuc-ky-chi-tiet-AdverTest.md` sections 5-9, 13-14, 18-21.10, and 24.
- Person D owns compute services and shared contracts; Person A owns Web/API/database; Person B owns YOLO/detection logic; Person C owns SAM2/segmentation logic.
- Person D is complete only when both model families can use the same locked/versioned benchmark pipeline through their supplied adapters/evaluators.

## Current-State Audit

Keep and extend:

- `AttackDatasetGenerator`, generated dataset reload/tamper checks, resume, atomic writes, canonical `.npy`, previews, manifests.
- Attack registry, attack capability/annotation checks, Group A-F attack implementations, patch artifact flow.
- KITTI/folder/generated loaders and anonymization gate.
- Detection benchmark, prediction cache, async local test-run worker, evidence artifacts.
- Existing detection suite, RobustScore helpers, bootstrap AP helper, and 579 passing scoped tests.

Missing or incomplete:

- Keyword-only detection/segmentation prediction contracts and segmentation benchmark support.
- DatasetVersion, deterministic split manifests, locked-test enforcement, duplicate/leakage reports.
- Full storytelling AttackMetadata, recipe schema, presets, deterministic random/quota recipe generation.
- Ordered composition, annotation transformation, intermediate-step provenance, generation cache distinct from inference cache.
- TrainingDatasetBuilder, HardExampleBank, FailureCase/FailureCluster contracts.
- Immutable BenchmarkProtocol and protocol compatibility validation.
- Task-agnostic evaluator/runner contract, multi-model paired runs, recovery/model comparison.
- Generic ModelTrainer state machine, estimate, compute worker, checkpoint/report contracts.
- Root pytest discovery currently enters inaccessible `data/` directories; `pytest tests` passes but `pytest` does not.

## Delivery Sequence

```text
Foundation gates
  -> shared contracts
  -> dataset version/split/leakage
  -> attack catalog/recipe/presets
  -> annotation + composition
  -> recipe dataset generation/cache
  -> training dataset + hard examples
  -> benchmark protocol + generic runner
  -> robustness/recovery/comparison
  -> generic training worker
  -> B/C/A handoff fixtures
  -> full reproducibility and acceptance audit
```

---

### Task 1: Isolate Execution and Repair the Test Harness

**Files:**
- Modify: `pytest.ini`
- Create: `tests/test_repo_boundaries.py`

**Interfaces:**
- Consumes: current branch containing the unified plan and existing attack/runtime implementation.
- Produces: an isolated Person D branch and a root-level test command that never scans `data/`.

- [ ] **Step 1: Create an isolated worktree**

Run read-only detection first:

```powershell
git rev-parse --git-dir
git rev-parse --git-common-dir
git branch --show-current
git status --short
```

Then use `superpowers:using-git-worktrees` to create `feat/person-d-platform` from the intended integration base. Do not carry the primary worktree's CRLF-only modifications into the feature branch.

- [ ] **Step 2: Add a failing repository-boundary test**

```python
from pathlib import Path


def test_day01_is_outside_p195_repository() -> None:
    repo = Path(__file__).resolve().parents[1]
    assert repo.name == "P-195"
    assert not any(path.name.startswith("DAY01") for path in repo.iterdir())
```

- [ ] **Step 3: Restrict pytest discovery**

Add to `pytest.ini`:

```ini
[pytest]
testpaths = tests
norecursedirs = .git .venv data outputs artifacts frontend node_modules
markers =
    gpu: optional integration test requiring a local checkpoint and CUDA
```

- [ ] **Step 4: Verify baseline**

Run:

```powershell
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests
```

Expected: 579 existing tests plus the new boundary test pass; no `PermissionError` from `data/`.

- [ ] **Step 5: Commit only harness files**

```powershell
git add pytest.ini tests/test_repo_boundaries.py
git commit -m "test: isolate Person D test discovery"
```

---

### Task 2: Correct Prediction Types and Objective Contracts

**Files:**
- Modify: `src/core/types.py`
- Modify: `src/core/objectives.py`
- Modify: `src/adapters/base.py`
- Modify: every detection adapter constructing `Prediction`
- Modify: `src/pipeline/cache.py`
- Modify: `src/pipeline/evidence.py`
- Create: `tests/test_core/test_prediction_contracts.py`

**Interfaces:**
- Produces: `DetectionPrediction`, `MaskPrediction`, `SegmentationPrediction`, `ModelPrediction`, and versioned attack/training objectives.
- Compatibility: `Prediction` remains a temporary alias of `DetectionPrediction` until all external consumers migrate.

- [ ] **Step 1: Write tests that reject positional construction**

```python
def test_detection_prediction_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        DetectionPrediction("sample-1", ())


def test_segmentation_prediction_validates_mask_contract() -> None:
    mask = np.ones((8, 8), dtype=np.bool_)
    item = MaskPrediction(instance_id="car-1", mask=mask, label="Car", score=0.9)
    result = SegmentationPrediction(sample_id="sample-1", instances=(item,))
    assert result.instances[0].mask.dtype == np.bool_
```

- [ ] **Step 2: Implement keyword-only prediction contracts**

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class DetectionPrediction:
    sample_id: str
    boxes: tuple[Box, ...] = ()
    boxes3d: tuple[Box3D, ...] = ()
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MaskPrediction:
    instance_id: str
    mask: np.ndarray
    label: str | None = None
    score: float = 1.0


@dataclass(frozen=True, slots=True, kw_only=True)
class SegmentationPrediction:
    sample_id: str
    instances: tuple[MaskPrediction, ...] = ()
    prompt_id: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


ModelPrediction = DetectionPrediction | SegmentationPrediction
Prediction = DetectionPrediction
```

Validate mask shape, dtype, finite score, and score range in `MaskPrediction.__post_init__`.

- [ ] **Step 3: Version objectives**

Add `objective_version`, separate `AttackObjective` from `TrainingObjective`, and validate target requirements:

```python
@dataclass(frozen=True, slots=True)
class TrainingObjective:
    kind: Literal["clean", "robust_mix", "targeted_repair"]
    version: str = "1.0.0"
    weights: dict[str, float] = field(default_factory=dict)
```

- [ ] **Step 4: Migrate every constructor to keywords**

Use `rg -n "Prediction\(" src tests` and convert each call to `DetectionPrediction(sample_id=..., boxes=..., latency_ms=...)`. Update SQLite cache serialization with a `prediction_type` discriminator and reject unknown types.

- [ ] **Step 5: Run focused and full tests**

```powershell
uv run --no-sync pytest tests/test_core/test_prediction_contracts.py tests/test_adapters tests/test_pipeline/test_runner.py -q
uv run --no-sync pytest -q
```

- [ ] **Step 6: Commit**

```powershell
git add src/core/types.py src/core/objectives.py src/adapters src/pipeline/cache.py src/pipeline/evidence.py tests
git commit -m "refactor: define typed model predictions"
```

---

### Task 3: Define Shared Versioned Contracts and Handoff Fixtures

**Files:**
- Create: `src/core/contracts.py`
- Create: `src/core/events.py`
- Create: `src/evaluation/contracts.py`
- Create: `src/training/contracts.py`
- Modify: `src/core/__init__.py`
- Create: `tests/test_core/test_shared_contracts.py`
- Create: `tests/fixtures/contracts/*.json`

**Interfaces:**
- Produces: `MetricEnvelope`, `FailureCase`, `FailureCluster`, `JobRequest`, `ProgressEvent`, `DefenseProfile`, `ModelVersionMetadata`, `TrainingRunConfig`.
- Consumers: A serializes these in API/database; B/C produce model-specific prediction/metric payloads.

- [ ] **Step 1: Write schema round-trip tests**

```python
def test_metric_envelope_never_confuses_ratio_and_percent() -> None:
    metric = MetricEnvelope(
        name="degradation",
        value=0.42,
        unit="ratio",
        version="1.0.0",
        percent_value=42.0,
    )
    assert MetricEnvelope.model_validate_json(metric.model_dump_json()) == metric
```

- [ ] **Step 2: Implement immutable common contracts**

Every model uses `ConfigDict(extra="forbid", frozen=True)`. `MetricEnvelope` includes `name`, `value`, `unit`, `percent_value`, `version`, `higher_is_better`, `ci95`, and `metadata`. `FailureCase` includes sample/model/protocol IDs, clean/attacked metrics, reason, affected object/mask, and artifact links. `FailureCluster` contains deterministic member IDs and selection permission.

- [ ] **Step 3: Define compute event contract**

```python
class ProgressEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    job_id: str
    job_type: Literal["generation", "benchmark", "training"]
    state: str
    progress_ratio: float = Field(ge=0.0, le=1.0)
    sequence: int = Field(ge=0)
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
```

- [ ] **Step 4: Generate committed example fixtures**

Create valid JSON examples for detection prediction, segmentation prediction, metric envelope, failure payload, job request/event, and model metadata. Tests load every fixture through its Pydantic/dataclass contract.

- [ ] **Step 5: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_core/test_shared_contracts.py -q
git add src/core src/evaluation/contracts.py src/training/contracts.py tests/test_core tests/fixtures/contracts
git commit -m "feat: add shared platform contracts"
```

---

### Task 4: Implement DatasetVersion and Deterministic Split Manifests

**Files:**
- Create: `src/datasets/contracts.py`
- Create: `src/datasets/versioning.py`
- Create: `src/datasets/splits.py`
- Modify: `src/datasets/base.py`
- Modify: `src/datasets/kitti.py`
- Modify: `src/datasets/folder.py`
- Create: `tests/test_datasets/test_versioning.py`
- Create: `tests/test_datasets/test_splits.py`

**Interfaces:**
- Produces: `DatasetVersion`, `SampleRecord`, `SplitManifest`, `DatasetIngestor.ingest(source, config)`, and `SplitBuilder.build(version, policy)`.
- Invariants: stable sample IDs, content hashes, GT hashes, source provenance, anonymization state, annotation type, classes, and immutable locked-test designation.

- [ ] **Step 1: Test deterministic ingestion**

```python
def test_same_source_produces_same_dataset_version(tmp_path: Path) -> None:
    first = DatasetIngestor(tmp_path / "versions").ingest(source, IngestConfig(name="fixture"))
    second = DatasetIngestor(tmp_path / "versions").ingest(source, IngestConfig(name="fixture"))
    assert first.version_id == second.version_id
    assert first.manifest_hash == second.manifest_hash
```

- [ ] **Step 2: Implement manifest-first ingestion**

`SampleRecord` includes `sample_id`, `source_uri`, `source_hash`, `ground_truth_hash`, `annotation_type`, `class_labels`, `anonymized`, and provenance metadata. `DatasetVersion.version_id` is derived from ordered sample records plus schema/loader versions, never from creation time.

- [ ] **Step 3: Test official and generated split policies**

Cover official split preservation, seeded 70/15/15 fallback, class-stratified assignment, no sample overlap, and immutable locked-test membership.

- [ ] **Step 4: Implement `SplitBuilder`**

```python
class SplitBuilder:
    def build(self, version: DatasetVersion, policy: SplitPolicy) -> SplitManifest: ...

    def validate(self, version: DatasetVersion, manifest: SplitManifest) -> SplitValidationReport: ...
```

Hash sample IDs before seeded ordering so source filesystem order cannot affect a split.

- [ ] **Step 5: Wire loaders to provenance**

Ensure KITTI/folder loaders place source URI, native label, loader version, split, and anonymization manifest hash in `Sample.meta` without changing canonical pixels/labels.

- [ ] **Step 6: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_datasets/test_versioning.py tests/test_datasets/test_splits.py tests/test_datasets/test_kitti.py tests/test_datasets/test_folder_generated.py -q
git add src/datasets tests/test_datasets
git commit -m "feat: version datasets and lock deterministic splits"
```

---

### Task 5: Implement Leakage, Duplicate, and Provenance Validation

**Files:**
- Create: `src/datasets/leakage.py`
- Create: `tests/test_datasets/test_leakage.py`
- Modify: `src/datasets/__init__.py`

**Interfaces:**
- Consumes: `DatasetVersion`, `SplitManifest`, generated/training manifests.
- Produces: `LeakageReport` with machine-readable errors, warnings, duplicate groups, and `passed`.

- [ ] **Step 1: Write failing leakage cases**

Test exact sample overlap, duplicate source hash under different IDs, duplicate GT/image pairs across versions, benchmark artifact reuse, locked seed reuse, missing source, invalid class mapping, and missing provenance.

- [ ] **Step 2: Implement validation**

```python
class LeakageValidator:
    def validate_splits(self, version: DatasetVersion, splits: SplitManifest) -> LeakageReport: ...
    def validate_training(self, config: TrainingLeakageInput) -> LeakageReport: ...
    def validate_generated(self, generated: GeneratedDatasetVersion) -> LeakageReport: ...
```

Errors block generation/training; warnings include intentional duplicate lineage only when an explicit allowlist explains it.

- [ ] **Step 3: Add reproducible report hashing**

Exclude timestamps from `report_hash`; include validator version and ordered findings.

- [ ] **Step 4: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_datasets/test_leakage.py -q
git add src/datasets/leakage.py src/datasets/__init__.py tests/test_datasets/test_leakage.py
git commit -m "feat: block dataset and benchmark leakage"
```

---

### Task 6: Complete the Attack Catalog Metadata

**Files:**
- Modify: `src/attacks/base.py`
- Create: `src/attacks/catalog.py`
- Modify: `src/attacks/__init__.py`
- Modify: `src/api/schemas.py` only to consume the new catalog payload without adding routes
- Modify: `tests/test_attacks/test_contract.py`
- Create: `tests/test_attacks/test_catalog.py`

**Interfaces:**
- Produces: `AttackMetadata` and `AttackCatalog.list(task, model_capabilities, annotation_types)`.
- Every registered attack must expose the complete metadata required by section 7.4.

- [ ] **Step 1: Extend metadata completeness tests**

Require non-empty display name, plain/technical summaries, scenario, rationale, failure symptoms, severity labels/map, compatibility, cost/runtime class, defense hint, reference, implementation version, deterministic flag, and online/offline support.

- [ ] **Step 2: Implement a centralized versioned catalog**

Keep algorithm classes focused. Store narrative/compatibility metadata in `catalog.py`, keyed by registry name, and have `BaseAttack.describe()` merge executable metadata with the catalog entry. Fail discovery when an attack has no catalog entry or version mismatch.

- [ ] **Step 3: Add compatibility filtering**

Filtering evaluates task, modality, annotations, model capabilities, online/offline policy, production status, and artifact requirements; it returns exclusions with reasons rather than silently dropping attacks.

- [ ] **Step 4: Verify all A-F attacks**

```powershell
uv run --no-sync pytest tests/test_attacks/test_contract.py tests/test_attacks/test_catalog.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add src/attacks/base.py src/attacks/catalog.py src/attacks/__init__.py src/api/schemas.py tests/test_attacks
git commit -m "feat: publish a validated attack catalog"
```

---

### Task 7: Implement AttackRecipe, Presets, Deterministic Randomization, and Cost Guards

**Files:**
- Create: `src/attacks/recipes.py`
- Create: `src/attacks/presets.py`
- Create: `tests/test_attacks/test_recipes.py`
- Create: `tests/test_attacks/test_presets.py`

**Interfaces:**
- Produces: `AttackRecipe`, `AttackRecipeStep`, `RecipeConstraints`, `RecipeValidation`, `RecipeEstimate`, `RecipeBuilder`, and six named presets.

- [ ] **Step 1: Test canonical recipe hashing**

Two semantically equal recipes with differently ordered JSON parameter keys must have the same hash; changing order, seed, catalog version, implementation version, or severity must change it.

- [ ] **Step 2: Implement immutable recipe models**

```python
class AttackRecipeStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    position: int = Field(ge=0)
    attack_name: str
    implementation_version: str
    severity: int = Field(ge=0, le=5)
    parameters: dict[str, Any] = Field(default_factory=dict)
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int
    objective: AttackObjective | None = None
    expected_cost: float = Field(ge=0.0)
```

`AttackRecipe` validates contiguous unique positions and derives `recipe_id`/`recipe_hash` from canonical content.

- [ ] **Step 3: Implement recipe validation**

Block incompatible tasks/annotations/capabilities, duplicate attack names, multiple white-box or expensive steps, FGSM+PGD, PGD+C&W, unsupported spatial transforms, excessive occlusion, recipe length, variant/storage/GPU caps, and non-production implementations. Return warnings for realism and expensive-but-allowed cases.

- [ ] **Step 4: Implement deterministic random builders**

```python
class RecipeBuilder:
    def random_n(self, request: RandomNRequest, catalog: AttackCatalog) -> list[AttackRecipe]: ...
    def random_by_group(self, request: StratifiedRandomRequest, catalog: AttackCatalog) -> list[AttackRecipe]: ...
    def sweep(self, request: SweepRequest, catalog: AttackCatalog) -> list[AttackRecipe]: ...
```

Tests cover allowlist/blocklist/required attacks, no replacement, quotas, per-step seeds, intensity constraints, hard caps, and same-seed reproducibility.

- [ ] **Step 5: Implement six presets**

Low Visibility, Wet Camera, Poor Camera Pipeline, Partial Obstruction, Adversarial Stress YOLO, and Segmentation Boundary Stress SAM. Presets resolve to ordinary versioned recipes; they contain no special execution path.

- [ ] **Step 6: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_attacks/test_recipes.py tests/test_attacks/test_presets.py -q
git add src/attacks/recipes.py src/attacks/presets.py tests/test_attacks
git commit -m "feat: add deterministic attack recipes"
```

---

### Task 8: Implement Detection and Segmentation Annotation Transformation

**Files:**
- Create: `src/pipeline/annotations.py`
- Create: `tests/test_pipeline/test_annotations.py`

**Interfaces:**
- Produces: `AnnotationPolicy`, `SpatialTransform`, `AnnotationTransformLog`, and `AnnotationTransformer.apply(sample, transform, policy)`.
- Consumers: composition and recipe generation.

- [ ] **Step 1: Write property-focused tests**

Cover identity, crop, translate, scale, horizontal flip, 90-degree rotation, arbitrary affine rotation, resize/pad, clipping, visible ratio, box drop policy, mask interpolation, empty masks, instance ID preservation, and source immutability.

- [ ] **Step 2: Implement one transform representation**

Use a 3x3 homogeneous source-to-output matrix plus output shape. Image, boxes, masks, and valid regions all consume the same transform object; masks use nearest-neighbor interpolation.

- [ ] **Step 3: Implement occlusion policy**

Occlusion does not geometrically move GT. Record visible ratio and only drop an object when the versioned policy explicitly permits it. Never convert model predictions into GT.

- [ ] **Step 4: Emit complete logs**

Each object/mask log contains original geometry/hash, transformed geometry/hash, visible ratio, kept/dropped decision, reason, and policy version.

- [ ] **Step 5: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_pipeline/test_annotations.py -q
git add src/pipeline/annotations.py tests/test_pipeline/test_annotations.py
git commit -m "feat: transform attack annotations safely"
```

---

### Task 9: Implement the Ordered Composition Engine

**Files:**
- Create: `src/pipeline/composition.py`
- Modify: `src/attacks/base.py`
- Create: `tests/test_pipeline/test_composition.py`

**Interfaces:**
- Produces: `CompositionEngine.execute(sample, recipe, context) -> CompositionResult`.
- `CompositionResult` contains final sample, ordered step records, intermediate arrays/hashes, transform log, resolved parameters, cost, and errors.

- [ ] **Step 1: Test order and deterministic step seeds**

Verify `Fog -> JPEG` differs from `JPEG -> Fog`, while repeated execution with the same source/recipe/catalog versions produces byte-identical canonical arrays and identical step records.

- [ ] **Step 2: Add attack execution metadata without breaking `BaseAttack.run`**

Expose `BaseAttack.resolve_parameters(severity)` and optional `spatial_transform(...)`; retain `run(sample, severity, ctx) -> Sample` for existing callers.

- [ ] **Step 3: Implement execution**

For every step: resolve exact class/version, derive seed, validate probability and compatibility, apply attack, apply annotation transform, validate image/sensors/GT, record intermediate hash, enforce cumulative cost/occlusion/budget, then continue.

- [ ] **Step 4: Define partial-failure behavior**

`fail_fast=True` aborts the variant. `fail_fast=False` records a failed variant and does not emit a loadable final sample. Never silently skip a requested step.

- [ ] **Step 5: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_pipeline/test_composition.py tests/test_attacks/test_contract.py -q
git add src/pipeline/composition.py src/attacks/base.py tests/test_pipeline/test_composition.py
git commit -m "feat: execute ordered attack compositions"
```

---

### Task 10: Upgrade Dataset Generation to Versioned Recipes and Separate Generation Cache

**Files:**
- Modify: `src/pipeline/generator.py`
- Modify: `src/pipeline/cache.py`
- Modify: `src/datasets/contracts.py`
- Modify: `src/datasets/generated.py`
- Modify: `src/datasets/io.py`
- Modify: `src/cli.py`
- Create: `tests/test_pipeline/test_recipe_generator.py`
- Modify: `tests/test_pipeline/test_generator.py`

**Interfaces:**
- Produces: `RecipeGenerationConfig`, `GeneratedDatasetVersion`, `GenerationCache`, and recipe-capable `AttackDatasetGenerator.generate`.
- Compatibility: old `AttackGenerationConfig` is converted to a one-step `AttackRecipe` before execution.

- [ ] **Step 1: Test a composed generated dataset**

Generate `fog -> object_occlusion -> jpeg_compression`, reload through `GeneratedDatasetSource`, and assert ordered steps, intermediate hashes, transformed annotations, source/GT hashes, recipe hash, implementation versions, seeds, intended use, and validation status.

- [ ] **Step 2: Implement versioned manifest models**

Replace untyped manifest dict construction with Pydantic `GeneratedVariantRecord` and `GeneratedDatasetVersion`. Preserve the existing JSONL/`dataset.json` reader through a schema-version migration function.

- [ ] **Step 3: Separate caches**

`GenerationCache` key contains dataset version, source hash, recipe hash, ordered implementation versions, seed, and surrogate version. `PredictionCache` key contains generated output hash, model/checkpoint/preprocessing versions, and thresholds.

- [ ] **Step 4: Persist intermediate evidence and resume state**

Write each step output atomically under `intermediates/<variant_id>/<position>.npy`; mark run/variant `incomplete`, `failed`, or `complete`; resume only records whose file hashes and schema/version hashes validate.

- [ ] **Step 5: Add CLI compatibility**

Add `generate-recipe --config ...`, `recipe validate`, and `recipe sample`; retain `generate-attack` by converting old config to a single-step recipe.

- [ ] **Step 6: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_pipeline/test_generator.py tests/test_pipeline/test_recipe_generator.py tests/test_datasets/test_folder_generated.py -q
git add src/pipeline src/datasets/generated.py src/datasets/io.py src/cli.py tests/test_pipeline tests/test_datasets
git commit -m "feat: generate versioned recipe datasets"
```

---

### Task 11: Implement TrainingDatasetBuilder and Defense Profiles

**Files:**
- Create: `src/training/dataset_builder.py`
- Modify: `src/training/contracts.py`
- Modify: `src/training/__init__.py`
- Create: `tests/test_training_dataset_builder.py`

**Interfaces:**
- Produces: `TrainingDatasetBuilder.estimate(config)` and `.build(config) -> TrainingDatasetManifest`.
- Consumes: base DatasetVersion, training split, DefenseProfile, recipe set, generated versions, failure clusters, leakage validator.

- [ ] **Step 1: Test mandatory rejection cases**

Reject locked-test samples, benchmark artifact reuse, missing transform logs, duplicate source leakage, invalid ratios, incompatible attacks, missing files, invalid GT, unreproducible manifest hash, and storage budget overflow.

- [ ] **Step 2: Implement deterministic sampling strategies**

Support random, class-balanced, object-size-balanced, failure-cluster-targeted, severity-distribution sampling, clean replay floor, hard-example replay, max variants per source, and max variants per recipe.

- [ ] **Step 3: Implement dry-run estimate**

Return clean/generated counts, class/object-size distributions, estimated bytes, online/offline counts, recipe/severity distribution, warnings, and hard-cap violations without writing artifacts.

- [ ] **Step 4: Build a canonical training manifest**

The manifest records every source/generated ID mapping, split, recipe, severity, seed, GT hash, transform log, failure link, online/offline mode, distribution report, leakage report hash, and manifest hash.

- [ ] **Step 5: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_training_dataset_builder.py -q
git add src/training/dataset_builder.py src/training/contracts.py src/training/__init__.py tests/test_training_dataset_builder.py
git commit -m "feat: build leakage-safe defense datasets"
```

---

### Task 12: Implement Hard Example Bank and Deterministic Failure Grouping

**Files:**
- Create: `src/training/hard_example_bank.py`
- Create: `src/evaluation/failures.py`
- Create: `tests/test_training_hard_example_bank.py`
- Create: `tests/test_evaluation/test_failures.py`

**Interfaces:**
- Produces: `HardExampleBank.put/get/query`, `FailureGrouper.group(cases)`, and immutable artifact records.

- [ ] **Step 1: Test artifact integrity and permissions**

Store strong PGD, C&W, Square, patch, SAM-PGD, critical scenario, and targeted-repair records. Reject hash mismatch, missing provenance, locked-test items marked for training, and benchmark-only artifacts used by the training builder.

- [ ] **Step 2: Implement content-addressed storage**

Index records in SQLite and store arrays/files by digest. Each record includes source/model/attack/protocol versions, objective, parameters, seeds, before/after metrics, failure reason, affected instances, and `allowed_uses`.

- [ ] **Step 3: Implement deterministic basic grouping**

Group by task, failure type, class, object-size bucket, attack family, and severity band. Automated semantic clustering remains P2; deterministic grouping is sufficient for targeted sampling and reproducibility.

- [ ] **Step 4: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_training_hard_example_bank.py tests/test_evaluation/test_failures.py -q
git add src/training/hard_example_bank.py src/evaluation/failures.py tests
git commit -m "feat: store and group hard examples"
```

---

### Task 13: Implement Immutable BenchmarkProtocol and Evaluator Contracts

**Files:**
- Create: `src/pipeline/protocol.py`
- Create: `src/evaluation/base.py`
- Create: `tests/test_pipeline/test_protocol.py`
- Create: `tests/test_evaluation/test_evaluator_contract.py`

**Interfaces:**
- Produces: `BenchmarkProtocol`, `ProtocolValidation`, `TaskEvaluator`, `EvaluationResult`.
- B provides `DetectionEvaluator`; C provides `SegmentationEvaluator`; D's runner consumes either through `TaskEvaluator`.

- [ ] **Step 1: Test protocol identity and locking**

Protocol ID changes when dataset/sample/GT hashes, recipe/version/seed, preprocessing, thresholds, prompt protocol, metric version, bootstrap config, environment, or framework version changes. Creation timestamp must not change identity.

- [ ] **Step 2: Implement immutable protocol**

Include every field from section 5.3, plus explicit `class_mapping_version` and `schema_version`. Status transitions are `DRAFT -> VALIDATED -> LOCKED -> RETIRED`; locked protocols cannot mutate.

- [ ] **Step 3: Define evaluator boundary**

```python
class TaskEvaluator(Protocol):
    task: str
    metric_versions: dict[str, str]

    def evaluate(
        self,
        predictions: Sequence[ModelPrediction],
        samples: Sequence[Sample],
        protocol: BenchmarkProtocol,
    ) -> EvaluationResult: ...
```

`EvaluationResult` contains headline metric envelope, supplemental metrics, per-sample metrics, failures, and validation warnings.

- [ ] **Step 4: Add mock detection and segmentation evaluators in tests**

Mocks must prove the runner can process both prediction types without importing YOLO or SAM2.

- [ ] **Step 5: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_pipeline/test_protocol.py tests/test_evaluation/test_evaluator_contract.py -q
git add src/pipeline/protocol.py src/evaluation/base.py tests/test_pipeline/test_protocol.py tests/test_evaluation/test_evaluator_contract.py
git commit -m "feat: lock benchmark and evaluator contracts"
```

---

### Task 14: Upgrade the Benchmark Runner for Recipes, Multiple Models, YOLO, and SAM

**Files:**
- Modify: `src/pipeline/runner.py`
- Modify: `src/pipeline/benchmark.py`
- Modify: `src/evaluation/report.py`
- Modify: `src/pipeline/cache.py`
- Modify: `src/pipeline/evidence.py`
- Create: `tests/test_pipeline/test_generic_benchmark.py`
- Modify: `tests/test_pipeline/test_runner.py`
- Modify: `tests/test_pipeline/test_benchmark.py`

**Interfaces:**
- Produces: generic `BenchmarkRunner.run(protocol, models, evaluator_registry, callbacks)`.
- Compatibility: `TestRunner.run(RunConfig)` remains as a detection-oriented adapter during migration.

- [ ] **Step 1: Test clean, single, composite, sweep, and paired runs**

Use mock detection and segmentation adapters/evaluators. Assert same protocol/sample/recipe/seeds, per-sample pairing, progress sequence, resume checkpoints, cache isolation, and explicit skip/failure reasons.

- [ ] **Step 2: Implement task-generic inference and evaluation**

Select evaluator by model task; validate prediction type; compute clean predictions once per model/protocol; evaluate every generated recipe cell; preserve sample-level results required for bootstrap.

- [ ] **Step 3: Implement multi-model paired execution**

Run baseline and defended versions against identical generated artifacts or deterministic recipe outputs. Reject a direct comparison when protocol, sample, preprocessing, threshold, prompt, class map, or metric version differs.

- [ ] **Step 4: Extend evidence serialization**

Detection evidence writes boxes; segmentation evidence writes GT/predicted masks and boundary overlays. The runner delegates overlay conventions to B/C-provided evidence serializers instead of embedding domain rendering.

- [ ] **Step 5: Preserve resume and cancellation**

Checkpoint after each complete model/recipe/severity cell. Resume validates protocol/report/cache hashes before reuse; cancellation leaves a valid partial report marked incomplete.

- [ ] **Step 6: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_pipeline/test_runner.py tests/test_pipeline/test_benchmark.py tests/test_pipeline/test_generic_benchmark.py -q
git add src/pipeline src/evaluation/report.py tests/test_pipeline
git commit -m "feat: run generic paired robustness benchmarks"
```

---

### Task 15: Implement Robustness, Recovery, Bootstrap, and Model Comparison

**Files:**
- Modify: `src/evaluation/robustness_metrics.py`
- Create: `src/evaluation/recovery_metrics.py`
- Create: `src/evaluation/bootstrap.py`
- Create: `src/evaluation/model_comparison.py`
- Create: `tests/test_evaluation/test_recovery_metrics.py`
- Create: `tests/test_evaluation/test_bootstrap.py`
- Create: `tests/test_evaluation/test_model_comparison.py`
- Modify: `tests/test_evaluation/test_robustness_metrics.py`

**Interfaces:**
- Produces: task-agnostic degradation/RA/RobustScore, paired bootstrap CI, recovery, comparison warnings, and checkpoint gate result.

- [ ] **Step 1: Replace ambiguous scalar fields with metric envelopes**

For every headline value emit `degradation_ratio`, `degradation_pct`, absolute point delta, relative change, direction, version, and CI where available.

- [ ] **Step 2: Implement recovery edge cases**

```python
def recovery_rate(
    baseline_clean: float,
    baseline_attacked: float,
    defended_attacked: float,
    *,
    higher_is_better: bool = True,
) -> MetricEnvelope | UndefinedMetric: ...
```

Test undefined denominator, >100%, negative recovery, and lower-is-better metrics.

- [ ] **Step 3: Implement paired sample bootstrap**

Resample sample IDs with replacement and recompute the evaluator result for each draw. Never bootstrap aggregate cells. Seed and iteration count come only from the locked protocol.

- [ ] **Step 4: Implement protocol-aware comparison**

Produce clean/attacked/degradation/failure/RobustScore deltas, recovery, clean tradeoff, critical regression, seen/unseen and external comparisons. Mark `paired=False` with explicit incompatibilities instead of presenting an invalid direct comparison.

- [ ] **Step 5: Implement the versioned checkpoint gate**

Default gate: clean decrease <=2 points; RobustScore +8 points or mean degradation 15% relatively better; no critical scenario worse by >3 points; ASR not worse by >5 percentage points; external metric not worse by >3 points; paired CI reported.

- [ ] **Step 6: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_evaluation -q
git add src/evaluation tests/test_evaluation
git commit -m "feat: compare robustness and measured recovery"
```

---

### Task 16: Implement Generic ModelTrainer and Compute Worker Orchestration

**Files:**
- Create: `src/training/base.py`
- Create: `src/training/registry.py`
- Create: `src/training/report.py`
- Create: `src/training/worker.py`
- Modify: `src/training/contracts.py`
- Modify: `src/training/__init__.py`
- Create: `tests/test_training_orchestration.py`

**Interfaces:**
- Produces: `ModelTrainer`, `TrainerRegistry`, `TrainingEstimate`, `TrainingReport`, `TrainingStateMachine`, `ComputeWorker`.
- B implements `YoloTrainer`; C implements `Sam2Trainer`; A submits requests and persists emitted events.

- [ ] **Step 1: Test every legal and illegal state transition**

Cover `DRAFT -> VALIDATING -> ESTIMATING -> QUEUED -> PREPARING_DATA -> TRAINING -> VALIDATING_CHECKPOINT -> EXPORTING -> REGISTERING_MODEL -> COMPLETED`, plus `FAILED`, `CANCELLED`, and `BUDGET_EXCEEDED` from valid active states.

- [ ] **Step 2: Define the trainer interface**

```python
class ModelTrainer(ABC):
    @abstractmethod
    def validate_config(self, config: TrainingRunConfig) -> ValidationReport: ...
    @abstractmethod
    def estimate(self, config: TrainingRunConfig) -> TrainingEstimate: ...
    @abstractmethod
    def prepare_data(self, config: TrainingRunConfig) -> PreparedTrainingData: ...
    @abstractmethod
    def train(self, config: TrainingRunConfig, callbacks: TrainerCallbacks) -> TrainingReport: ...
    @abstractmethod
    def evaluate_checkpoint(self, checkpoint: CheckpointMetadata) -> MetricSnapshot: ...
    @abstractmethod
    def export_checkpoint(self, checkpoint: CheckpointMetadata) -> ExportedCheckpoint: ...
    @abstractmethod
    def metadata(self) -> TrainerMetadata: ...
```

- [ ] **Step 3: Implement guarded execution**

The worker validates leakage and manifests before queueing, checks estimate against GPU/storage/time caps, emits ordered events, supports cancellation, records per-epoch metrics, hashes checkpoints, validates export load, and only emits registration metadata after a completed run and passing lineage checks.

- [ ] **Step 4: Test with fake YOLO and SAM trainers**

Use deterministic CPU fakes to prove registry dispatch, event ordering, cancellation, budget guard, failed checkpoint hash, successful export, resume metadata, and model parent-child lineage.

- [ ] **Step 5: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_training_orchestration.py -q
git add src/training tests/test_training_orchestration.py
git commit -m "feat: orchestrate versioned model training jobs"
```

---

### Task 17: Expose Person D Services Through CLI and Stable Handoff Payloads

**Files:**
- Modify: `src/cli.py`
- Create: `src/services/__init__.py`
- Create: `src/services/person_d.py`
- Create: `tests/test_cli_person_d.py`
- Create: `tests/test_services_person_d.py`
- Create: `docs/contracts/person-d-handoff.md`

**Interfaces:**
- Produces: one service facade for A and CLI commands for local/GPU workers.

- [ ] **Step 1: Implement a service facade**

```python
class PersonDServices:
    datasets: DatasetService
    recipes: RecipeService
    generation: GenerationService
    training_data: TrainingDatasetService
    benchmarks: BenchmarkService
    comparisons: ComparisonService
    training: TrainingComputeService
```

The facade returns domain contracts and progress events; it imports no FastAPI or frontend code.

- [ ] **Step 2: Add CLI commands**

```text
dataset-ingest
dataset-split
dataset-validate
recipe-validate
recipe-sample
generate-recipe
inspect-attack-dataset
build-training-dataset
benchmark-run
compare-models
training-estimate
training-run
```

All commands accept JSON config paths, print JSON reports, return non-zero on validation failure, and never download weights.

- [ ] **Step 3: Test JSON payload compatibility**

Load all committed A/B/C handoff fixtures; assert service outputs validate against shared contracts and that job event sequence numbers are monotonic.

- [ ] **Step 4: Write the handoff document**

Document exact schemas, units, state transitions, error codes, sample payloads, ownership boundaries, and required B/C/A integration callbacks. No UI instructions are included.

- [ ] **Step 5: Verify and commit**

```powershell
uv run --no-sync pytest tests/test_cli_person_d.py tests/test_services_person_d.py -q
git add src/cli.py src/services tests/test_cli_person_d.py tests/test_services_person_d.py docs/contracts/person-d-handoff.md
git commit -m "feat: expose Person D compute services"
```

---

### Task 18: Run Scientific, Reproducibility, and End-to-End Acceptance Gates

**Files:**
- Create: `tests/integration/test_person_d_detection_e2e.py`
- Create: `tests/integration/test_person_d_segmentation_e2e.py`
- Create: `tests/integration/test_person_d_recovery_e2e.py`
- Create: `tests/integration/test_person_d_reproducibility.py`
- Create: `tests/gpu/test_person_d_yolo11.py`
- Create: `tests/gpu/test_person_d_sam2.py`
- Modify: `docs/ATTACK_DATASET_GENERATOR.md`
- Create: `docs/PERSON_D_COMPLETION_MATRIX.md`

**Interfaces:**
- Consumes: all Person D services plus B/C-provided adapters/evaluators when available.
- Produces: auditable evidence for every Person D Definition of Done item.

- [ ] **Step 1: Add CPU detection vertical slice**

```text
DatasetVersion -> locked split -> recipe -> generated data
-> clean/attacked mock detection inference -> evaluator
-> robustness report -> hard-example selection
-> TrainingDatasetManifest -> fake robust trainer
-> same BenchmarkProtocol -> RecoveryReport
```

- [ ] **Step 2: Add CPU segmentation vertical slice**

Use GT masks, fixed prompt IDs, transformed masks, mock segmentation predictions/evaluator, identical protocol, and Boundary IoU/mIoU envelopes supplied by the test evaluator.

- [ ] **Step 3: Add reproducibility and leakage gates**

Run the same seeded mini pipeline twice in separate directories and compare canonical array hashes, manifests, reports, event sequences, and model comparison results. Inject locked-test and duplicate leakage and assert pre-training failure.

- [ ] **Step 4: Add scientific sanity tests**

Verify severity 0 identity, source immutability, metric units, monotonicity reporting without assuming monotonicity always holds, strong PGD versus same-norm random noise on the reference surrogate, per-class/per-size output, paired bootstrap, clean tradeoff, seen/unseen labels, external split separation, and same-prompt SAM comparison.

- [ ] **Step 5: Add optional real-model gates**

Mark YOLO11 and SAM2 tests `gpu`. They require explicit local checkpoint paths and skip otherwise. Acceptance requires real B/C adapters/evaluators before final team integration; Person D's CPU contract tests must remain independent of those checkpoints.

- [ ] **Step 6: Run the full quality gate**

```powershell
uv run --no-sync ruff check src tests
uv run --no-sync pytest -q
uv run --no-sync pytest tests/integration -q
uv run --no-sync pytest -m gpu tests/gpu -q
```

Record skipped GPU reasons. Do not claim YOLO/SAM real integration if checkpoints or B/C implementations are absent.

- [ ] **Step 7: Complete the requirement matrix**

For every requirement in sections 5-9, 13-14, 20, 21.5, and Person D DoD, record implementation file, test name, command, result, and remaining external dependency. No row may be marked complete from indirect evidence.

- [ ] **Step 8: Verify DAY01 and commit**

```powershell
git -C D:\Project\AIthucchien\DAY01_2A202601138_PhanTrongDat status --short
git add tests/integration tests/gpu docs/ATTACK_DATASET_GENERATOR.md docs/PERSON_D_COMPLETION_MATRIX.md
git commit -m "test: verify Person D platform end to end"
```

---

## Mandatory Handoff Gates

Person D can implement and test all platform contracts with CPU fakes, but these team integration gates require explicit artifacts from the named owner:

| Gate | Required from owner | Person D acceptance |
|---|---|---|
| Detection | B: YOLO adapter, `DetectionPrediction`, detection evaluator, metric versions, checkpoint metadata | Same protocol runs clean/attacked and returns mAP@[.50:.95], AP50/AP75, per-class/per-size and failures |
| Segmentation | C: runnable SAM2 adapter, prompt protocol, `SegmentationPrediction`, evaluator, checkpoint metadata | Same prompt/protocol runs clean/attacked and returns mIoU, Boundary IoU, failure cases |
| Web integration | A: request, persistence, event and error contracts | A can enqueue Person D services, persist events/results, cancel/resume without importing compute internals |
| Training | B/C: concrete `YoloTrainer`/`Sam2Trainer` | Generic worker validates, estimates, dispatches, emits events, validates export and returns model lineage |

If a handoff is absent, mark that row `WAITING_FOR_OWNER`; do not weaken the contract or replace a real-model acceptance claim with a mock claim.

## Person D Definition of Done Audit

The objective is complete only when current evidence proves all rows:

| Requirement | Required evidence |
|---|---|
| Dataset version and split | deterministic version/split tests and locked manifest artifact |
| Leakage validator | injected overlap/duplicate/benchmark-reuse tests fail before training |
| Attack Recipe | single/manual/random N/random-by-group/preset tests pass |
| Deterministic composition | same-seed hashes match; ordered recipe changes output/hash |
| Generated manifest | reload, tamper, resume, transform and provenance tests pass |
| TrainingDatasetBuilder | all sampling strategies, estimate, leakage, balance and budget tests pass |
| BenchmarkProtocol locked | identity/immutability/compatibility tests pass |
| Runner works with YOLO and SAM | CPU contract tests plus real B/C integration artifacts |
| RobustScore/Recovery/bootstrap | formula, unit, edge-case and paired-sample tests pass |
| Compute worker integrates with A | request/event/cancel/resume/error fixture contract accepted |
| Full provenance | end-to-end lineage from source to model comparison is reconstructable by hashes |
| Quality | Ruff, full pytest, integration suite, scientific sanity and documented GPU result |

## Recommended Execution Strategy

Use subagent-driven development, one task at a time, with two review gates after each task:

1. Specification review against the exact task interfaces and invariants.
2. Code-quality review after focused tests pass.

Tasks 1-3 are strict foundation gates. Tasks 4-6 may then proceed independently. Tasks 7-10 are sequential. Tasks 11-12 can proceed after Task 10. Tasks 13-15 are sequential. Task 16 can begin after Tasks 3 and 11. Tasks 17-18 integrate everything.

Do not merge partial work to main until the completion matrix states exactly what is complete, what is waiting for B/C/A, and which commands produced the recorded evidence.

## Estimated Milestones for One Person

These are engineering estimates, not deadlines; GPU availability and B/C/A handoffs are excluded.

| Milestone | Tasks | Expected effort | Exit gate |
|---|---:|---:|---|
| M0 Correctness and contracts | 1-3 | 2-3 days | Clean root test command, typed predictions, shared fixtures accepted |
| M1 Dataset foundation | 4-5 | 3-4 days | Versioned split and leakage reports are deterministic |
| M2 Attack catalog and recipes | 6-7 | 3-5 days | All attacks described; all required recipe modes validate deterministically |
| M3 Composition and generation | 8-10 | 5-7 days | Ordered recipe dataset reloads with correct GT/provenance/cache/resume |
| M4 Defense data and hard examples | 11-12 | 3-4 days | Training manifest is balanced, budgeted, leakage-safe, and reproducible |
| M5 Benchmark and comparison | 13-15 | 5-7 days | Detection/segmentation mock paths and paired recovery reports pass |
| M6 Training compute and handoff | 16-17 | 3-5 days | Fake B/C trainers and A-facing events complete the state machine |
| M7 Acceptance and documentation | 18 | 3-5 days | Completion matrix and all available CPU/GPU evidence are recorded |

Expected Person D effort is approximately 27-40 focused engineering days. Tasks 4-6 can be parallelized after contracts stabilize; all later estimates assume review and regression testing after every task.
