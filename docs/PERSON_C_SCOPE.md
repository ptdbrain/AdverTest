# Người C — Phạm vi công việc (Defence / Training / Scientific Evaluation)

> Tài liệu làm rõ công việc cho **Người C** theo `AdverTest_Implementation_Plan_4_People.md`.
> Ngày tổng hợp: 2026-08-19.

---

## 1. Ownership của Người C

Theo plan §3/§4:

| | Nội dung |
|---|---|
| Domain | **Defence / Scientific Evaluation** |
| Trách nhiệm chính | split data, adversarial training, fine-tune, model lineage, recovery, analytics backend |
| Phạm vi code | `src/training/`, `src/defence/`, `src/evaluation/comparison*`, `src/analytics/`, `src/api/routers/defence.py`, `src/api/routers/analytics.py` |

---

## 2. Phát hiện quan trọng về hiện trạng

Sau khi khảo sát codebase: **phần lớn nền tảng defence/training/comparison mà plan giao cho Người C đã được xây sẵn và có test** (trong repo được ghi là "Person D" — người đi trước).

Đã tồn tại sẵn:

- `src/training/` (12 file): `YoloTrainer`, `Sam2Trainer`, `PatchTrainer`, `ComputeWorker`, `dataset_builder`, `closed_loop`, `hard_example_bank`, `registry`, `contracts`, `report`, `worker`, `yolo_dataset_formatter`.
- `src/evaluation/model_comparison.py` (paired comparison + checkpoint gate), `src/evaluation/recovery_metrics.py`.
- `src/datasets/splits.py` + `src/datasets/leakage.py` (immutable split + leakage gate).
- API endpoint defence/training/comparison/closed-loop (nằm rải rác trong `src/api/routes.py` — file monolith ~1780 dòng).

Do đó công việc của Người C **không phải xây lại**, mà là **kế thừa nền tảng có sẵn + lấp các khoảng trống còn thiếu**.

---

## 3. Bản đồ hiện trạng: plan C1–C17 → code

| Mục plan | Nội dung | Trạng thái | Vị trí code |
|---|---|---|---|
| C1 | Immutable split (TRAIN/VAL/TEST) | ✅ Có (2D) | `src/datasets/splits.py` (`SplitPolicy`/`SplitBuilder`/`SplitManifest`) + `src/datasets/leakage.py` |
| C2 | Split 3D theo scene/sequence | ⏸ Thuộc **A** | chưa có, chờ A bàn giao (A4/A5) |
| C3 | DefenceProfile | ✅ Có (contract) | `src/training/contracts.py::DefenseProfile` |
| C4/C5 | Train attack mix (30/15/15/15/10/5) | 🟡 Một phần | ratio đã có; thiếu "named mix preset" |
| C6 | 3 nhóm TRAIN/VAL/HELDOUT | 🟡 Chưa tách rõ | `DefenseProfile.recipe_ids` đơn nhất |
| C7 | Defence dataset generator | ✅ Có | `src/training/dataset_builder.py` + `src/api/generated_dataset_service.py` |
| C8 | Fine-tune / TrainingRun | ✅ Có | `TrainingRunConfig` + `YoloTrainer` + `ComputeWorker` + `TrainingJobService` |
| C9 | Training config | ✅ Có | `TrainingRunConfig` |
| C10 | Model lineage (B0→R1→R2) | ✅ Có | `src/models/versions.py` + `training/closed_loop.py` |
| C11 | Defence benchmark (strict paired) | ✅ Có | `evaluation/benchmark_protocol.py` + `POST /defence-runs` + `compare_models` |
| C12 | Benchmark matrix (B0/R1) | ✅ Có | `model_comparison.py::compare_models` |
| C13 | Degradation / Recovery | ✅ Có | `robustness_metrics.py` + `recovery_metrics.py::recovery_rate` |
| C14 | Failure transitions | ✅ Có | `detection_metrics.py::per_object_detection_comparison` + `GET /failure-cases` |
| **C15** | **Analytics backend** | ❌ **THIẾU** | không có `src/analytics/`, `routers/analytics.py`, endpoint `/analytics/*` |
| C16 | 3D analytics (distance/BEV/3D) | ❌⏸ Chờ A | phụ thuộc evaluator/attack 3D |

**Kết luận:** phần "training/defence/comparison" về cơ bản đã hoàn thiện. Công việc mới thật sự = **C15 (Analytics backend) + tách router + C16/3D (chờ A).**

---

## 4. Công việc còn lại của Người C

### Khối A — Analytics backend (C15) [chính]

**File mới:**
- `src/analytics/` — các hàm aggregate **thuần** (`run_analytics.py`, `comparison_analytics.py`), đọc `dict` report/record, không query filesystem.
- `src/api/routers/analytics.py` — router mới, đăng ký trong `src/main.py` (cùng cách `catalog`/`runs`/`datasets`).

**Endpoint (plan §53):**
```
GET /analytics/runs/{run_id}/summary
GET /analytics/runs/{run_id}/attacks
GET /analytics/runs/{run_id}/severity
GET /analytics/runs/{run_id}/classes
GET /analytics/runs/{run_id}/samples

GET /analytics/comparisons/{comparison_id}/summary
GET /analytics/comparisons/{comparison_id}/recovery
GET /analytics/comparisons/{comparison_id}/classes
GET /analytics/comparisons/{comparison_id}/failures
```

**Nguồn dữ liệu (tái dùng, không tính metric mới):**
- Report run: `SqliteRunStore.get(run_id)["report"]` — cấu trúc `RunReport.as_dict()` (`src/evaluation/report.py`): `ap_clean`, `cells` (attack/group/severity/`degradation_percent`/`metrics`), `heatmap`, `worst_cases`, `sample_results`, `metrics` (ap50/ap75/map50_95 + robust summary).
- Comparison: `SqliteRunStore.get_record("model_comparison", id)` — đã chứa `metric_deltas`, `recovery_report`, `paired`, `incompatibilities`.
- Tái dùng: `robustness_metrics.summary()`, `detection_metrics`, `model_comparison`.

**Lưu ý:** đã có `GET /benchmark-runs/{id}/metrics`, `GET /runs/{id}/samples`, `GET /failure-cases`. `analytics.py` đóng vai trò **namespace chuẩn `/analytics/*` + aggregate phong phú hơn** (per-attack/per-severity/per-class/recovery).

### Khối B — Tách `routers/defence.py` (§4, §85)

Move các endpoint defence/training/comparison từ `src/api/routes.py` sang `src/api/routers/defence.py`:

- **Defense profile:** `POST/GET /defense-profiles` (`routes.py:254–264`)
- **Training:** `POST /training-runs/estimate`, `POST /training-runs`, `GET /training-runs`, `GET /training-runs/{id}`, `POST /training-runs/{id}/cancel`, `GET /training-runs/{id}/checkpoints|events|events/ws` (`:267–358`), `GET /training-dataset-manifests/{id}` (`:329`)
- **Retraining backlogs:** `POST/GET /retraining-backlogs...`, `items`, `approve` (`:362–392`)
- **Comparison:** `POST /model-comparisons`, `GET /model-comparisons/{id}` + `/export` + `/metric-deltas` + `/recovery-report` + `/failures` (`:395–516`), `POST /comparisons` (`:760`)
- **Defence run:** `POST /defence-runs` (`:930`), `GET /defence-checkpoints` (`:801`), `GET /runs/{id}/defence-candidates` (`:1004`)
- **Failure analysis:** `GET /failure-cases` (`:1108`), `GET/POST /failure-clusters...` (`:1122–1149`)
- **Model lineage:** `GET /model-versions/{id}/lineage` + `/benchmark-history` + `/gate-evidence` (`:817–861`)
- **Closed-loop:** `POST /closed-loop/start`, `GET /closed-loop/{id}`, `POST /closed-loop/{id}/advance` (`:1155–1396`)

**Rủi ro khi tách:** các helper module-level của `routes.py` (`_store`, `_workflow_store`, `_training_jobs`, `_require_run`, `_require_completed_report`, `_comparison_signature`, `_registered_model_versions`). Nên chuyển sang module dùng chung (ví dụ `src/api/deps.py`; `src/api/dependencies.py` đã có) rồi cả `routes.py` lẫn `defence.py` import chung — làm qua PR contract nhỏ (plan §5).

### Khối C — 3D defence integration (C16) [bị chặn]

Chờ **Người A** bàn giao (plan §97): `TaskDefinition`, `DatasetVersion`, `split IDs`, `ModelAdapter`, `Evaluator`, `AttackMethods`, `AttackRecipe`, `Metric schema`, `failure schema`. Sau đó mới: `GET /analytics/runs/{id}/distance|bev|3d` + 3D defence dataset/fine-tune/recovery (Sprint 3). Không block công việc hiện tại.

---

## 5. Ranh giới ownership & handoff

**Shared files — không tự sửa trực tiếp (plan §5):** `src/core/contracts.py`, `src/core/types.py`, `src/api/schemas/`, `src/api/routes.py` (file đang refactor), `src/main.py`, database migration registry.

| Handoff | Nội dung |
|---|---|
| **A → C** (§97) | A bàn giao toàn bộ module 3D; C **không duplicate** evaluator/attack 3D. |
| **C → D** (§98) | C bàn giao API schema: analytics / comparison / failure / recovery. D chỉ render, **không tự đọc raw prediction files** (plan §91). |
| **B → C** (§99) | Dùng `ArtifactStorage`/`Job Service`/`Worker`/DB của B, không viết storage riêng. |

---

## 6. Gợi ý thứ tự (khi bắt tay vào code)

1. Tách helper `routes.py` → module dùng chung (PR contract nhỏ, không đổi hành vi).
2. Tách `routers/defence.py` (move cơ học), đăng ký `main.py`, chạy test giữ route không đổi.
3. Xây `src/analytics/` + `routers/analytics.py`, đăng ký `main.py`.
4. Thêm test theo pattern `tests/test_api/` (fixture `client` trong `tests/conftest.py`).
5. (Tuỳ chọn) C4/C5/C6: named mix preset + tách nhóm TRAIN/VAL/HELDOUT.

---

## 7. Tóm tắt

> Người C **không cần xây lại** defence/training — đã gần hoàn chỉnh. Việc chính:
> 1. **Analytics backend** — khoảng trống lớn nhất (tái dùng `RunReport`/`model_comparison` có sẵn).
> 2. **Tách `routers/defence.py`** ra khỏi `routes.py`.
> 3. **3D defence** — ghi nhận nhưng chờ A bàn giao.