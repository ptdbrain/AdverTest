# KẾ HOẠCH TRIỂN KHAI ADVERTEST GIAI ĐOẠN TIẾP THEO

> Mục tiêu: mở rộng AdverTest từ nền tảng robustness testing 2D hiện tại thành một hệ thống có thể:
>
> - chạy và đánh giá mô hình perception 3D;
> - tạo và benchmark các dạng tấn công trên dữ liệu 3D;
> - xây dựng closed-loop phòng thủ bằng cách sinh dữ liệu adversarial/corrupted rồi fine-tune lại;
> - triển khai production trên Render theo hướng multi-user;
> - hỗ trợ upload checkpoint riêng và import checkpoint pretrained từ Ultralytics;
> - có database, phân quyền `USER` / `ADMIN`;
> - có dashboard, bảng, biểu đồ, failure analysis và AI gợi ý bước tiếp theo;
> - chia việc rõ ràng cho 4 người, giảm tối đa việc cùng sửa một file hoặc cùng sở hữu một module.

---

# 1. Mục tiêu tổng thể

Giai đoạn tiếp theo không nên phát triển từng tính năng rời rạc.

Toàn bộ hệ thống cần hướng tới closed-loop sau:

```text
User
 ↓
Select / Upload / Import Checkpoint
 ↓
Select / Upload Dataset
 ↓
Validate Task + Model + Checkpoint + Dataset
 ↓
Run Clean Benchmark
 ↓
Generate Attacks
 ↓
Run Attacked Benchmark
 ↓
Analyze Failures
 ↓
Generate Defence Dataset
 ↓
Fine-tune
 ↓
Register New Checkpoint
 ↓
Run SAME Locked Benchmark
 ↓
Compare Baseline vs Defended
 ↓
Recovery Analysis
 ↓
AI recommends next action
```

Đối với 3D:

```text
3D Dataset
 ↓
3D Model Adapter
 ↓
Clean 3D Benchmark
 ↓
3D Attacks
 ↓
Attacked 3D Benchmark
 ↓
Failure Analysis
 ↓
3D Defence Dataset
 ↓
Fine-tune 3D Model
 ↓
3D Defence Evaluation
```

---

# 2. Nguyên tắc kiến trúc bắt buộc

## 2.1. Không xây thêm logic lớn trong runner 2D cũ

Các task mới phải đi theo kiến trúc generic:

```text
Task
 ↓
ModelAdapter
 ↓
Prediction Contract
 ↓
TaskEvaluator
 ↓
BenchmarkRunner
```

Không tiếp tục nhồi:

```text
if task == detection2d
elif task == segmentation
elif task == detection3d
```

vào một runner duy nhất.

Mục tiêu cuối:

```text
BenchmarkRunner
├── Detection2DEvaluator
├── SegmentationEvaluator
└── Detection3DEvaluator
```

---

## 2.2. Phải tách rõ 3 khái niệm

```text
Task
Model Family
Checkpoint
```

Ví dụ:

```text
Task:
    detection3d

Model Family:
    CenterPoint

Checkpoint:
    centerpoint-nuscenes-v1.ckpt
```

Không dùng model name để đại diện cho task.

---

## 2.3. Tất cả artifact phải là first-class entity

Các object sau phải có ID riêng:

```text
DatasetVersion
CheckpointArtifact
AttackRecipe
BenchmarkRun
GeneratedDataset
TrainingRun
DefenceRun
ModelComparison
AnalysisReport
```

Không truyền file path tạm thời xuyên toàn bộ hệ thống.

---

## 2.4. Mọi job dài phải asynchronous

Không chạy trực tiếp:

```text
HTTP request
 → model inference
 → attack generation
 → training
```

Mà phải:

```text
HTTP request
 ↓
Create Job
 ↓
Queue
 ↓
Worker
 ↓
Progress Events
 ↓
Completed / Failed / Cancelled
```

Áp dụng cho:

- upload / indexing;
- checkpoint validation;
- clean benchmark;
- attack generation;
- attacked inference;
- export;
- fine-tuning;
- defence benchmark;
- report generation.

---

# 3. Chia ownership cho 4 người

Không chia theo từng feature nhỏ.

Chia theo domain ownership để tránh conflict.

| Người | Ownership | Trách nhiệm chính |
|---|---|---|
| A | 3D Perception | 3D model, 3D dataset, 3D evaluator, 3D attack, 3D evidence |
| B | Platform / Deploy | Render, DB infra, object storage, worker, checkpoint platform, Ultralytics importer |
| C | Defence / Scientific Evaluation | split data, adversarial training, fine-tune, model lineage, recovery, analytics backend |
| D | User Product | auth, RBAC, admin, AI advisor, analytics frontend |

---

# 4. Phạm vi code của từng người

## Người A

```text
src/adapters/*3d*
src/datasets/*3d*
src/attacks/3d/
src/evaluation/*3d*
src/pipeline/3d/
tests/3d/
```

## Người B

```text
src/storage/
src/compute/
src/jobs/
src/api/routers/artifacts.py
src/api/routers/checkpoints.py
src/api/checkpoint_service.py
deploy/
render.yaml
Dockerfile*
```

## Người C

```text
src/training/
src/defence/
src/evaluation/comparison*
src/analytics/
src/api/routers/defence.py
src/api/routers/analytics.py
```

## Người D

```text
src/auth/
src/users/
src/api/routers/auth.py
src/api/routers/admin.py
src/api/routers/advisor.py

frontend/src/app/admin/
frontend/src/app/analysis/
frontend/src/components/analytics/
frontend/src/components/advisor/
```

---

# 5. Shared files — hạn chế sửa trực tiếp

Các file sau chỉ nên thay đổi qua PR contract nhỏ:

```text
src/core/contracts.py
src/core/types.py
src/api/schemas/
src/api/app.py
database migration registry
frontend global state
```

Quy trình:

```text
Need shared change
 ↓
Create small contract PR
 ↓
Review
 ↓
Merge
 ↓
Other branches rebase
```

---

# 6. NGƯỜI A — 3D PERCEPTION

# 6.1. Mục tiêu

Hoàn thành pipeline:

```text
3D Dataset
 ↓
3D Adapter
 ↓
Clean Inference
 ↓
3D Evaluation
 ↓
3D Attack
 ↓
Attacked Inference
 ↓
Clean vs Attacked Metrics
 ↓
Evidence
```

Chưa fine-tune ở bước này.

Fine-tune thuộc Người C.

---

# 7. A1 — Chọn một model 3D MVP

Không làm nhiều model cùng lúc.

Chỉ chọn một model để hoàn thiện toàn bộ pipeline.

Tiêu chí:

- có pretrained checkpoint;
- inference runnable;
- framework ổn định;
- dataset benchmark có metric rõ;
- có thể fine-tune sau;
- output convert được về canonical box format.

Ví dụ family có thể cân nhắc:

```text
CenterPoint
PointPillars
SECOND
```

Không cần làm cả ba.

---

# 8. A2 — Xây ModelAdapter cho 3D

Contract:

```python
class Detection3DAdapter:
    def load(...)
    def predict(...)
    def metadata(...)
    def capabilities(...)
```

Output canonical:

```text
Detection3DPrediction
├── sample_id
├── boxes_3d
├── scores
├── labels
└── latency_ms
```

Mỗi box:

```text
Box3D
├── center_x
├── center_y
├── center_z
├── length
├── width
├── height
├── yaw
├── class_id
└── score
```

---

# 9. A3 — Dataset 3D contract

Mỗi sample cần:

```text
sample_id
scene_id
frame_id

lidar
camera metadata        # nếu model cần
calibration
ground_truth_boxes_3d
class_labels
```

Nếu dataset nhiều sensor:

```text
sensors/
├── lidar
├── camera_front
├── camera_left
├── camera_right
└── calibration
```

---

# 10. A4 — Split dữ liệu đúng cách

Không random frame.

Phải split theo:

```text
scene / sequence
```

Ví dụ:

```text
TRAIN
scene_001
scene_002
scene_003

VALIDATION
scene_010

TEST
scene_020
scene_021
```

Không để:

```text
scene_001 frame 1 → train
scene_001 frame 2 → test
```

vì sẽ leakage.

---

# 11. A5 — Validation dataset 3D

Validation tối thiểu:

```text
LiDAR exists
calibration exists
finite point coordinates
box3d valid
class known
scene id present
coordinate frame known
```

Box validation:

```text
length > 0
width > 0
height > 0
yaw finite
center finite
```

---

# 12. A6 — Clean inference

Trước khi attack:

```text
Dataset
 ↓
Adapter
 ↓
Prediction
 ↓
Evaluator
```

Phải chạy được end-to-end trên:

```text
10 samples
↓
100 samples
↓
full validation subset
```

---

# 13. A7 — Metrics 3D

Evaluator tối thiểu:

```text
overall score
per-class score
BEV IoU metric
3D IoU metric
precision
recall
```

Thêm phân tích theo distance:

```text
near
medium
far
```

Ví dụ:

```text
0–20 m
20–40 m
40+ m
```

Khoảng cách phải configurable theo dataset.

---

# 14. A8 — 3D evidence

MVP dùng BEV trước.

Layout:

```text
┌────────────────────┬────────────────────┐
│ Original + GT      │ Attacked Input     │
├────────────────────┼────────────────────┤
│ Clean Prediction   │ Attacked Prediction│
└────────────────────┴────────────────────┘
```

Giữ đúng semantics giống 2D:

```text
TL = clean input + GT
TR = attacked input
BL = baseline prediction on clean
BR = same baseline prediction on attacked
```

Không đưa defended model vào view này.

Defence phải có màn riêng.

---

# 15. A9 — 3D attack MVP

Bắt đầu bằng sensor corruption.

Ưu tiên:

```text
1. Point Dropout
2. XYZ / Range Noise
3. Intensity Noise
4. Ring / Beam Dropout
5. Local Occlusion
6. Point Injection
```

Sau đó mới làm:

```text
Calibration Perturbation
Camera-LiDAR mismatch
White-box gradient attack
```

---

# 16. A10 — Contract của AttackMethod 3D

```text
AttackMethod
├── id
├── name
├── category
├── supported_tasks
├── required_modalities
├── severity_levels
├── required_capabilities
├── changes_geometry
├── stochastic
└── implementation_version
```

Ví dụ:

```text
PointDropout
task = detection3d
modality = lidar
severity = 1..5
changes_geometry = false
stochastic = true
```

---

# 17. A11 — Determinism

Mọi attack stochastic phải lưu:

```text
global_seed
sample_seed
recipe_step_seed
```

Seed final có thể derive từ:

```text
hash(
  global_seed,
  sample_id,
  attack_id,
  severity,
  recipe_position
)
```

Chạy lại phải tạo đúng artifact giống nhau.

---

# 18. A12 — Quy tắc Ground Truth

Nếu chỉ làm sensor corruption:

```text
GT giữ nguyên.
```

Ví dụ:

```text
point dropout
range noise
intensity noise
fog simulation
```

Nếu transform geometry thực:

```text
GT phải transform tương ứng.
```

Không để coordinate system của input và GT khác nhau.

---

# 19. A13 — Deliverables của Người A

```text
Detection3D TaskDefinition
Detection3D Prediction Contract
3D Dataset Loader
3D Dataset Validator
3D Model Adapter
3D Evaluator
3D Clean Benchmark
3D Attack Registry
>= 3 attacks
3D Evidence Renderer
tests
documentation
```

Definition of Done:

```text
real 3D model
+
real 3D data
+
clean metric
+
attacked metric
+
sample evidence
+
deterministic attack
```

---

# 20. NGƯỜI B — DEPLOYMENT / STORAGE / CHECKPOINT PLATFORM

# 20.1. Mục tiêu

Biến app hiện tại thành platform:

```text
Frontend
 ↓
API
 ↓
Database
 ↓
Object Storage
 ↓
Queue
 ↓
Workers
```

---

# 21. B1 — Kiến trúc production

```text
Browser
   │
   ▼
Frontend
   │
   ▼
FastAPI
   │
   ├── PostgreSQL
   ├── Redis / Queue
   ├── Object Storage
   │
   └── Worker Dispatcher
          │
          ▼
      Compute Worker
```

---

# 22. B2 — Render deployment

Render có thể host:

```text
Frontend
FastAPI Web Service
PostgreSQL
Redis-compatible Key Value
Background Worker
```

Không gắn compute architecture chặt với Render.

Phải có abstraction:

```text
ComputeBackend
├── LocalWorker
├── RenderWorker
└── ExternalGPUWorker
```

---

# 23. B3 — Không dùng local filesystem làm nguồn dữ liệu chính

Production không được lưu lâu dài:

```text
/uploads/model.pt
/uploads/dataset.zip
/results/run123/
```

trong filesystem local.

Phải dùng Object Storage.

---

# 24. B4 — Storage abstraction

Tạo interface:

```python
class ArtifactStorage:
    def put(...)
    def get(...)
    def delete(...)
    def exists(...)
    def signed_download_url(...)
    def signed_upload_url(...)
```

Implement:

```text
LocalArtifactStorage
S3CompatibleStorage
```

Local dùng development.

S3-compatible dùng production.

---

# 25. B5 — Phân loại dữ liệu

## PostgreSQL lưu metadata

```text
users
projects
datasets
dataset_versions
checkpoints
attack_recipes
runs
jobs
training_runs
defence_runs
comparisons
artifacts
audit_logs
```

## Object Storage lưu bytes

```text
datasets/
checkpoints/
generated-datasets/
attacked-datasets/
predictions/
evidence/
training/
reports/
exports/
```

---

# 26. B6 — Storage path convention

```text
users/{user_id}/
  projects/{project_id}/
    datasets/
      {dataset_version_id}/
    checkpoints/
      {checkpoint_id}/
    runs/
      {run_id}/
    training/
      {training_run_id}/
    reports/
```

DB lưu:

```text
artifact_id
owner_id
project_id
storage_key
sha256
size_bytes
mime_type
created_at
```

---

# 27. B7 — Dataset upload flow

Production flow:

```text
Browser
 ↓
POST create upload session
 ↓
API returns signed upload URL
 ↓
Browser uploads directly
 ↓
Object Storage
 ↓
POST finalize upload
 ↓
Validation job
 ↓
READY / FAILED
```

Không stream dataset lớn qua API nếu không bắt buộc.

---

# 28. B8 — Checkpoint state machine

Không dùng:

```text
upload → READY
```

Phải:

```text
UPLOADING
 ↓
UPLOADED
 ↓
QUARANTINED
 ↓
INTEGRITY_VALIDATING
 ↓
SANDBOX_LOADING
 ↓
METADATA_EXTRACTING
 ↓
SMOKE_TESTING
 ↓
READY
```

Failure:

```text
REJECTED
FAILED_VALIDATION
UNSUPPORTED
```

---

# 29. B9 — Secure checkpoint validation

Không load checkpoint không tin cậy trong FastAPI process.

Validation worker phải:

```text
isolated process/container
no unrestricted network
limited filesystem
timeout
memory limit
CPU/GPU limit
```

Validation:

```text
extension
size
hash
format
model family
task compatibility
deserialize/load
architecture
class names
dummy inference
output contract
capabilities
```

---

# 30. B10 — Native class metadata

Checkpoint phải lưu:

```text
native_class_names
num_classes
class_schema_hash
```

Ví dụ:

```text
["helmet", "worker", "forklift"]
```

Không hard-code COCO mapping cho mọi model.

---

# 31. B11 — ClassMapping

Tạo entity:

```text
ClassMapping
├── checkpoint_id
├── dataset_version_id
├── native_class_id
├── canonical_class_id
└── status
```

Nếu chưa map đủ:

```text
qualitative inference → allowed
scientific benchmark → blocked
```

Lỗi:

```text
CLASS_MAPPING_REQUIRED
```

---

# 32. B12 — Import checkpoint pretrained từ Ultralytics

UI:

```text
Add Checkpoint

[ Upload From Computer ]
[ Import Pretrained ]
```

Pretrained modal:

```text
Family: YOLO11

○ YOLO11n
○ YOLO11s
○ YOLO11m
○ YOLO11l
○ YOLO11x

[ Import ]
```

---

# 33. B13 — Ultralytics importer backend

Endpoint:

```text
POST /checkpoints/import/ultralytics
```

Body:

```json
{
  "model_id": "yolo11s"
}
```

Không nhận URL arbitrary.

Server whitelist:

```text
yolo11n
yolo11s
yolo11m
yolo11l
yolo11x
```

Flow:

```text
Validate Model ID
 ↓
Create Job
 ↓
Official download
 ↓
Hash
 ↓
Store Artifact
 ↓
Metadata Extraction
 ↓
Smoke Test
 ↓
Register Checkpoint
 ↓
READY
```

---

# 34. B14 — Job system

Entity:

```text
Job
├── id
├── type
├── owner_id
├── project_id
├── status
├── stage
├── completed_units
├── total_units
├── progress_percent
├── error_code
├── error_message
├── created_at
├── started_at
└── completed_at
```

Status:

```text
QUEUED
RUNNING
COMPLETED
FAILED
CANCELLED
```

---

# 35. B15 — Progress events

Event:

```text
JobProgressEvent
├── job_id
├── stage
├── completed
├── total
├── percent
├── message
└── timestamp
```

Không chỉ emit:

```text
20%
100%
```

Mà phải theo unit thật.

Ví dụ attack benchmark:

```text
prepare       5%
generate     25%
clean infer  20%
attack infer 25%
evaluate     15%
persist       5%
report        5%
```

---

# 36. B16 — Export attacked dataset

Tạo async job:

```text
POST /exports/attacked-dataset
```

Artifact ZIP:

```text
attacked_dataset.zip
├── media/
├── labels/
├── manifest.json
├── recipe.json
├── provenance.json
└── hashes.json
```

Manifest phải có:

```text
source dataset version
task
recipe
attack method
severity
seed
implementation version
sample mapping
created_at
hash
```

---

# 37. B17 — Deliverables của Người B

```text
Render deployment
PostgreSQL
Queue
Worker abstraction
Object Storage abstraction
Artifact model
Signed upload
Signed download
Secure checkpoint validation
Checkpoint state machine
Ultralytics importer
Job progress infrastructure
Attacked dataset export
```

---

# 38. NGƯỜI C — DEFENCE / TRAINING / SCIENTIFIC ANALYSIS

# 38.1. Mục tiêu

Hoàn thiện closed-loop:

```text
B0
 ↓
Locked Benchmark
 ↓
Failure Analysis
 ↓
Defence Dataset
 ↓
Fine-tune
 ↓
R1
 ↓
Same Benchmark
 ↓
Recovery
```

---

# 39. C1 — Tạo immutable split

```text
DatasetVersion
├── TRAIN
├── VALIDATION
└── TEST
```

Rules:

```text
TRAIN
→ defence data
→ fine-tune

VALIDATION
→ training decisions

TEST
→ benchmark only
```

Không sinh adversarial training data từ TEST.

---

# 40. C2 — Split 3D

3D phải:

```text
split by scene / sequence
```

Không split frame độc lập.

---

# 41. C3 — DefenceProfile

Entity:

```text
DefenceProfile
├── id
├── name
├── source_checkpoint_id
├── source_dataset_version_id
├── train_recipe_ids
├── validation_recipe_ids
├── heldout_recipe_ids
├── clean_ratio
├── seed
└── config
```

---

# 42. C4 — Train attack mix

Không train từng model riêng cho mỗi attack.

Bad:

```text
model_fog
model_noise
model_blur
model_pgd
```

Recommended:

```text
B0
 ↓
R1 Robust Mix
 ↓
R2 Targeted Repair
```

---

# 43. C5 — Attack mix design

Một starting configuration:

```text
30% clean
15% weather
15% noise/corruption
15% occlusion
10% sensor fault
10% adversarial
5% residual failures
```

Tỷ lệ phải configurable.

---

# 44. C6 — Tách 3 nhóm attack

```text
TRAIN_ATTACKS
VALIDATION_ATTACKS
HELDOUT_ATTACKS
```

Mục tiêu đo:

```text
robustness seen attacks
vs
generalization to unseen attacks
```

---

# 45. C7 — Defence dataset generator

Flow:

```text
TRAIN split
 ↓
Attack Recipes
 ↓
Generated Samples
 ↓
Labels / GT
 ↓
Manifest
 ↓
Training Dataset Version
```

Manifest:

```text
source_sample_id
source_dataset_version
attack_method
severity
params
seed
implementation_version
source_hash
output_hash
split=train
```

---

# 46. C8 — Fine-tune pipeline

Entity:

```text
TrainingRun
├── id
├── parent_checkpoint_id
├── training_dataset_id
├── defence_profile_id
├── config
├── seed
├── status
├── best_metric
├── output_checkpoint_id
└── logs_artifact_id
```

---

# 47. C9 — Training config

Lưu đầy đủ:

```text
epochs
batch_size
learning_rate
optimizer
scheduler
augmentation
early_stopping
seed
framework version
GPU
base checkpoint
```

---

# 48. C10 — Model lineage

```text
B0
└── R1
    └── R2
```

Checkpoint metadata:

```text
role = base | fine_tuned | repaired
parent_checkpoint_id
training_run_id
training_dataset_version
defence_profile
```

---

# 49. C11 — Defence benchmark

Phải chạy:

```text
same dataset
same test split
same sample IDs
same attacks
same recipe order
same severity
same seed
same thresholds
same preprocessing
same evaluator
same metric version
```

Nếu khác:

```text
paired = false
```

---

# 50. C12 — Benchmark matrix

```text
             CLEAN     ATTACKED
B0            A           B
R1            C           D
```

Tính:

```text
clean delta
attacked delta
baseline degradation
defended degradation
recovery rate
ASR
failure transitions
```

---

# 51. C13 — Metrics

## Degradation

```text
(CleanScore - AttackedScore)
---------------------------- × 100
         CleanScore
```

## Recovery

```text
DefendedAttacked - BaselineAttacked
------------------------------------ × 100
BaselineClean - BaselineAttacked
```

Recovery có thể:

```text
< 0%
> 100%
```

Không clamp nếu mục tiêu là scientific analysis.

---

# 52. C14 — Failure transitions

Tối thiểu:

```text
CORRECT → MISSED
CORRECT → MISCLASSIFIED
CORRECT → LOW_CONFIDENCE
CORRECT → LOCALIZATION_FAILURE
FAILED  → RECOVERED
FAILED  → STILL_FAILED
```

---

# 53. C15 — Analytics backend

Tạo API:

```text
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

---

# 54. C16 — 3D analytics

Khi A bàn giao evaluator:

```text
GET /analytics/runs/{id}/distance
GET /analytics/runs/{id}/bev
GET /analytics/runs/{id}/3d
```

Phân tích:

```text
class × distance
attack × distance
severity × distance
```

---

# 55. C17 — Deliverables của Người C

```text
Dataset split contracts
DefenceProfile
DefenceDataset generator
TrainingRun
Fine-tune runner
Model lineage
Strict paired comparison
Recovery metrics
Failure transitions
Analytics backend
YOLO R1 closed loop
3D defence integration after A completes
```

---

# 56. NGƯỜI D — USER / ADMIN / AI ADVISOR / ANALYTICS FRONTEND

# 56.1. Mục tiêu

Biến AdverTest thành multi-user platform có:

```text
Login
Project isolation
USER / ADMIN
Admin dashboard
Advanced analytics
AI next-action advisor
```

---

# 57. D1 — Role model

Chỉ dùng 2 role:

```text
USER
ADMIN
```

Không mở rộng role nếu chưa cần.

---

# 58. D2 — User schema

```text
users
├── id
├── email
├── auth_subject
├── display_name
├── role
├── status
├── storage_quota
├── compute_quota
├── created_at
└── last_login_at
```

Status:

```text
ACTIVE
SUSPENDED
DISABLED
```

---

# 59. D3 — Project ownership

Mọi resource:

```text
owner_user_id
project_id
```

Áp dụng cho:

```text
dataset
checkpoint
recipe
run
training
comparison
report
artifact
```

---

# 60. D4 — Authorization

Backend phải kiểm tra:

```text
resource.owner_id == current_user.id
OR
current_user.role == ADMIN
```

Không chỉ hide UI.

---

# 61. D5 — Permission matrix

| Action | USER | ADMIN |
|---|---:|---:|
| Upload dataset | Yes | Yes |
| Upload checkpoint | Yes | Yes |
| Import pretrained model | Yes | Yes |
| Run benchmark | Yes | Yes |
| Generate attacked data | Yes | Yes |
| Run defence | Yes | Yes |
| View own project | Yes | Yes |
| View other users | No | Yes |
| Suspend user | No | Yes |
| Change quotas | No | Yes |
| Cancel own jobs | Yes | Yes |
| Cancel any job | No | Yes |
| Inspect system logs | No | Yes |
| Manage flagged artifact | No | Yes |

---

# 62. D6 — Admin dashboard

Sections:

```text
Users
Jobs
Storage
Compute
Checkpoints
Datasets
Security
Audit Logs
```

Metrics:

```text
active users
total users
running jobs
queued jobs
failed jobs
storage used
GPU hours
checkpoint validation failures
dataset validation failures
```

---

# 63. D7 — Admin actions

```text
Suspend User
Reactivate User
Change Storage Quota
Change Compute Quota
Cancel Job
Inspect Artifact
Quarantine Checkpoint
Delete Invalid Artifact
```

Mọi admin action phải tạo:

```text
AuditLog
```

---

# 64. D8 — AI Advisor

Không để LLM trực tiếp điều khiển training.

Architecture:

```text
Current Project State
 ↓
Rule Engine
 ↓
Candidate Actions
 ↓
LLM Explanation / Ranking
 ↓
Recommendation
 ↓
User Confirms
```

---

# 65. D9 — Recommendation schema

```text
Recommendation
├── priority
├── action_type
├── title
├── reason
├── evidence
├── risks
└── suggested_parameters
```

Ví dụ:

```json
{
  "priority": "HIGH",
  "action_type": "GENERATE_DEFENCE_DATASET",
  "title": "Prioritize weather robustness",
  "reason": "Fog severity 4 causes the largest degradation.",
  "evidence": [
    "run_123",
    "attack:fog",
    "degradation:41.8%"
  ]
}
```

---

# 66. D10 — Rule Engine trước LLM

Các rule deterministic:

```text
No GT
→ Recommend labeling

Checkpoint not READY
→ Recommend validation

No clean baseline
→ Recommend clean benchmark

Worst attack > threshold
→ Recommend defence dataset

R1 clean regression too high
→ Recommend training adjustment

Only seen attacks tested
→ Recommend held-out evaluation

3D failures concentrated far range
→ Recommend far-range robustness analysis
```

LLM chỉ:

```text
explain
rank
summarize
```

Không tự tạo metric.

---

# 67. D11 — Benchmark Overview UI

Hiển thị:

```text
Checkpoint
Dataset
Task
Clean Score
Attacked Score
Worst Attack
Worst Severity
Average Degradation
ASR
RobustScore
```

---

# 68. D12 — Attack Matrix

Heatmap:

```text
            S1     S2     S3     S4     S5

Fog
Rain
Noise
Blur
PGD
Occlusion
```

Click cell:

```text
→ evidence samples
→ metric details
→ failures
```

---

# 69. D13 — Per-Class Analysis

Table:

```text
Class
Clean Score
Attacked Score
Delta
ASR
Failure Count
Recovery
```

---

# 70. D14 — Failure Explorer

Filters:

```text
attack
severity
class
failure type
confidence
distance
```

Rows:

```text
sample
clean prediction
attacked prediction
GT
failure transition
```

---

# 71. D15 — Defence Recovery Dashboard

```text
             Baseline   Defended    Delta

Clean
Attacked
Degradation
ASR
Recovery
```

Charts:

```text
per attack
per severity
per class
```

---

# 72. D16 — 3D analytics UI

Khi backend A/C hoàn thành:

```text
3D score by distance
BEV score
class × distance
attack × distance
severity × distance
```

MVP visualization có thể dùng BEV images trước.

3D interactive viewer làm sau.

---

# 73. D17 — Deliverables của Người D

```text
Authentication
USER / ADMIN RBAC
Project ownership
Admin Dashboard
Quotas
Audit Logs
Analytics UI
Failure Explorer
Defence Dashboard
AI Advisor
Recommendation UI
```

---

# 74. SPRINT 0 — KHÓA CONTRACT

Thời gian:

```text
1–2 ngày
```

Cả team thống nhất:

```text
TaskDefinition
ModelFamily
CheckpointArtifact
DatasetVersion
AttackRecipe
BenchmarkRun
Job
TrainingRun
DefenceRun
ModelComparison
User
Project
Artifact
```

Không code feature lớn trước khi contract này ổn định.

---

# 75. Sprint 0 Checklist

- [ ] Tách Task / Model Family / Checkpoint.
- [ ] Chốt Prediction contract cho detection3d.
- [ ] Chốt Artifact interface.
- [ ] Chốt Job interface.
- [ ] Chốt ownership ID.
- [ ] Chốt Project ID.
- [ ] Chốt DatasetVersion ID.
- [ ] Chốt Checkpoint lifecycle.
- [ ] Tách API routers.
- [ ] Ghi CODEOWNERS hoặc ownership document.
- [ ] Đặt branch convention.
- [ ] Đặt migration convention.

---

# 76. SPRINT 1 — FOUNDATION

## Người A

- [ ] chọn model 3D MVP;
- [ ] loader dataset 3D;
- [ ] 3D prediction contract;
- [ ] 3D adapter;
- [ ] inference 10 samples;
- [ ] inference 100 samples.

## Người B

- [ ] Render skeleton;
- [ ] PostgreSQL;
- [ ] storage abstraction;
- [ ] object storage dev/prod;
- [ ] job abstraction;
- [ ] worker abstraction.

## Người C

- [ ] immutable dataset split;
- [ ] DefenceProfile;
- [ ] TrainingRun contract;
- [ ] model lineage;
- [ ] strict comparison contract.

## Người D

- [ ] User schema;
- [ ] Project schema;
- [ ] auth;
- [ ] RBAC;
- [ ] ownership middleware;
- [ ] basic login UI.

---

# 77. Sprint 1 Acceptance

A:

```text
real 3D inference works
```

B:

```text
upload artifact → object storage → DB record works
```

C:

```text
defence data/training entities are persisted
```

D:

```text
User A cannot access User B project
```

---

# 78. SPRINT 2 — CORE FEATURES

## Người A

- [ ] 3D evaluator;
- [ ] clean benchmark;
- [ ] BEV evidence;
- [ ] Point Dropout;
- [ ] Range Noise;
- [ ] Ring Dropout;
- [ ] deterministic seed.

## Người B

- [ ] signed upload;
- [ ] checkpoint sandbox;
- [ ] runtime validation;
- [ ] class metadata;
- [ ] Ultralytics importer;
- [ ] progress events.

## Người C

- [ ] defence dataset generator;
- [ ] train/val/heldout attacks;
- [ ] YOLO R1 training;
- [ ] output checkpoint registration;
- [ ] recovery metric.

## Người D

- [ ] Admin UI;
- [ ] user management;
- [ ] quota management;
- [ ] audit logs;
- [ ] initial analytics dashboard.

---

# 79. Sprint 2 Acceptance

A:

```text
clean 3D vs attacked 3D metric works
```

B:

```text
user can upload or import checkpoint and reach READY safely
```

C:

```text
B0 → R1 closed loop works for YOLO
```

D:

```text
admin can inspect users/jobs without direct DB access
```

---

# 80. SPRINT 3 — INTEGRATION

## Người A

- [ ] add more 3D attacks;
- [ ] 3D attack recipe;
- [ ] sample-level failures;
- [ ] distance analysis.

## Người B

- [ ] production deploy;
- [ ] queue/worker monitoring;
- [ ] attacked dataset export;
- [ ] signed download;
- [ ] job retry/cancel.

## Người C

- [ ] consume A's 3D split;
- [ ] 3D defence dataset;
- [ ] initial 3D fine-tune;
- [ ] 3D recovery comparison.

## Người D

- [ ] attack matrix;
- [ ] failure explorer;
- [ ] defence recovery;
- [ ] AI recommendation engine;
- [ ] recommendation cards.

---

# 81. SPRINT 4 — CLOSED LOOP

Target:

```text
Upload / Import model
 ↓
Upload / select dataset
 ↓
Run clean
 ↓
Run attacks
 ↓
Analyze
 ↓
Generate defence dataset
 ↓
Fine-tune
 ↓
Register R1
 ↓
Re-evaluate
 ↓
Recovery
 ↓
AI recommendation
```

Cho cả:

```text
2D
3D
```

---

# 82. Branch strategy

Đề xuất:

```text
feature/3d-core
feature/platform-deploy
feature/defence
feature/user-admin-analytics
```

Không tạo branch theo từng task nhỏ nếu code cùng domain.

---

# 83. Merge strategy

Mỗi PR nên:

```text
small
independent
tested
contract-compatible
```

Thứ tự merge:

```text
1. shared contracts
2. infrastructure foundations
3. independent domain features
4. integration
5. UI wiring
```

---

# 84. Shared integration owner

Nên chọn Người B làm integration owner cho:

```text
API app registration
router registration
deployment config
environment variables
migration coordination
```

Không có nghĩa B được sửa logic của A/C/D.

---

# 85. API organization đề xuất

```text
src/api/
├── routers/
│   ├── auth.py
│   ├── users.py
│   ├── admin.py
│   ├── datasets.py
│   ├── checkpoints.py
│   ├── artifacts.py
│   ├── runs.py
│   ├── perception3d.py
│   ├── defence.py
│   ├── analytics.py
│   └── advisor.py
│
├── schemas/
└── app.py
```

---

# 86. Database schema mức cao

```text
users
projects
project_memberships

datasets
dataset_versions
dataset_samples

checkpoints
checkpoint_validations
class_mappings

attack_methods
attack_recipes
attack_recipe_steps

benchmark_runs
benchmark_samples
benchmark_metrics

jobs
job_events

generated_datasets

defence_profiles
training_runs

model_comparisons

artifacts
analysis_reports

recommendations
audit_logs
usage_records
```

---

# 87. Project isolation

Query không bao giờ chỉ:

```sql
SELECT * FROM checkpoints WHERE id = ?
```

Mà phải enforce:

```sql
WHERE id = ?
AND project_id IN user_accessible_projects
```

Hoặc domain service thực hiện tương đương.

---

# 88. Quota

USER có thể có:

```text
storage quota
monthly compute quota
max concurrent jobs
max checkpoint size
max dataset size
```

Admin chỉnh được.

---

# 89. Usage accounting

Ghi:

```text
job runtime
CPU time
GPU time
storage bytes
artifact count
training duration
```

Dùng cho:

```text
admin dashboard
quota
future billing
```

---

# 90. Error codes nên chuẩn hóa

Ví dụ:

```text
CHECKPOINT_INVALID
CHECKPOINT_UNSUPPORTED
CHECKPOINT_RUNTIME_FAILED

DATASET_INVALID
DATASET_NOT_LABELED

CLASS_MAPPING_REQUIRED

TASK_MODEL_INCOMPATIBLE
ATTACK_NOT_SUPPORTED

JOB_CANCELLED
JOB_TIMEOUT

BENCHMARK_NOT_PAIRED

INSUFFICIENT_PERMISSION
QUOTA_EXCEEDED
```

---

# 91. Không để frontend tự tính scientific metrics

Frontend chỉ render:

```text
backend metric
```

Không tính:

```text
mAP
ASR
Recovery
RobustScore
```

ở JavaScript.

---

# 92. Analytics table design

Mọi table nên hỗ trợ:

```text
sort
filter
search
pagination
export
```

Các bảng:

```text
Runs
Checkpoints
Datasets
Attacks
Classes
Failures
Defence Comparisons
Users
Jobs
```

---

# 93. Report export

Export:

```text
JSON
CSV
HTML
```

Sau này:

```text
PDF
```

Report phải ghi:

```text
dataset version
checkpoint hash
recipe hash
seed
metric version
timestamp
software version
```

---

# 94. AI Advisor — không hallucinate evidence

LLM input chỉ gồm structured facts:

```text
run metrics
failure clusters
class metrics
distance metrics
training history
validation status
```

Không cho AI tự query filesystem hoặc suy đoán model state.

---

# 95. AI Advisor action types

```text
LABEL_DATASET
RUN_CLEAN_BASELINE
RUN_ATTACK
RUN_HELDOUT_ATTACK
GENERATE_DEFENCE_DATASET
FINE_TUNE
RE_EVALUATE
REVIEW_CLASS_MAPPING
UPLOAD_CHECKPOINT
IMPORT_PRETRAINED
ADJUST_TRAINING
INVESTIGATE_FAILURE_CLUSTER
```

---

# 96. AI action safety

AI không được tự:

```text
delete artifact
start expensive training
suspend user
overwrite checkpoint
modify permissions
```

Phải user/admin confirm.

---

# 97. 3D → Defence handoff contract

A bàn giao cho C:

```text
TaskDefinition
DatasetVersion
split IDs
ModelAdapter
Evaluator
AttackMethods
AttackRecipe
Metric schema
failure schema
```

C không duplicate các module này.

---

# 98. C → D handoff contract

C bàn giao:

```text
analytics API schemas
comparison API
failure API
recovery API
```

D không tự đọc raw prediction files.

---

# 99. B → tất cả handoff

B cung cấp:

```text
ArtifactStorage
Job Service
Checkpoint Service
Object URL
Worker abstraction
DB infrastructure
```

A/C/D không tự viết storage riêng.

---

# 100. D → B handoff

D định nghĩa:

```text
CurrentUser
Permission check
Project context
```

B dùng cho:

```text
storage authorization
job ownership
checkpoint access
dataset access
```

---

# 101. Các việc không nên làm ngay

Không ưu tiên:

```text
multiple 3D architectures
interactive full WebGL point-cloud viewer
many user roles
billing
Kubernetes
microservices per feature
one model per attack
full autonomous AI agent
```

---

# 102. Thứ tự ưu tiên P0

P0:

```text
3D model clean inference
3D evaluator
Object storage
PostgreSQL
User isolation
Checkpoint sandbox
YOLO defence R1
```

---

# 103. P1

```text
3D attacks
Ultralytics importer
Defence dataset generator
strict recovery comparison
Admin dashboard
advanced analytics
```

---

# 104. P2

```text
3D fine-tune
3D recovery
AI advisor
attacked dataset export
failure explorer
```

---

# 105. P3

```text
more 3D models
more 2D families
interactive 3D viewer
GPU worker scaling
advanced automated recommendations
```

---

# 106. Definition of Done toàn hệ thống

Hệ thống được xem là hoàn thành giai đoạn này khi một user thực hiện được:

```text
Create account
 ↓
Create project
 ↓
Import YOLO11s
 ↓
Upload dataset
 ↓
Validate
 ↓
Run benchmark
 ↓
View 2×2 evidence
 ↓
View attack analysis
 ↓
Generate defence dataset
 ↓
Fine-tune R1
 ↓
Compare B0 vs R1
 ↓
View recovery
```

và:

```text
Upload/select 3D dataset
 ↓
Load real 3D model
 ↓
Run clean benchmark
 ↓
Run >= 3 3D attacks
 ↓
View attacked results
 ↓
View 3D failure analysis
 ↓
Generate defence data
 ↓
Fine-tune
 ↓
Compare defended model
```

---

# 107. Definition of Done production

Production phải đảm bảo:

```text
No persistent user data depends on ephemeral filesystem
No untrusted checkpoint loaded inside API process
Every artifact has owner/project
Every long job has progress
Every job can fail safely
Scientific comparisons require matching protocol
Admin actions audited
User A cannot access User B data
```

---

# 108. Checklist cuối trước release

## 3D

- [ ] real model;
- [ ] real dataset;
- [ ] clean benchmark;
- [ ] 3+ attacks;
- [ ] deterministic seeds;
- [ ] 3D metrics;
- [ ] sample evidence;
- [ ] distance analysis.

## Deployment

- [ ] Render frontend;
- [ ] Render API;
- [ ] Postgres;
- [ ] queue;
- [ ] workers;
- [ ] object storage;
- [ ] environment secrets;
- [ ] monitoring.

## Checkpoint

- [ ] local upload;
- [ ] Ultralytics import;
- [ ] sandbox validation;
- [ ] class extraction;
- [ ] class mapping;
- [ ] SHA256;
- [ ] lineage.

## Defence

- [ ] immutable split;
- [ ] no test leakage;
- [ ] attack mix;
- [ ] fine-tune;
- [ ] register R1;
- [ ] paired evaluation;
- [ ] recovery;
- [ ] heldout attacks.

## User/Admin

- [ ] login;
- [ ] USER;
- [ ] ADMIN;
- [ ] project isolation;
- [ ] quota;
- [ ] admin dashboard;
- [ ] audit log.

## Analytics

- [ ] summary;
- [ ] attack matrix;
- [ ] severity;
- [ ] class;
- [ ] failure explorer;
- [ ] defence recovery;
- [ ] 3D distance analysis;
- [ ] exports.

## AI Advisor

- [ ] rule engine;
- [ ] structured evidence;
- [ ] priority;
- [ ] recommended action;
- [ ] user confirmation;
- [ ] no autonomous destructive actions.

---

# 109. Kết luận phân công

Cách chia công việc cuối cùng:

```text
PERSON A
3D Perception
    ↓
3D Benchmark
    ↓
3D Attack
    ↓
handoff to C

PERSON B
Platform
    ↓
Storage
    ↓
Checkpoint
    ↓
Deploy
    ↓
supports everyone

PERSON C
Defence
    ↓
Training
    ↓
Comparison
    ↓
Analytics Backend
    ↓
handoff to D

PERSON D
Users
    ↓
Admin
    ↓
Analytics UI
    ↓
AI Advisor
```

Dependency chính:

```text
          B Platform
        /     |      \
       /      |       \
      A       C        D
      │       ▲
      └───────┘
      3D → Defence
```

C có thể phát triển defence trên 2D trong lúc A làm 3D.

D có thể làm auth/admin trong lúc C làm analytics.

B có thể triển khai storage/checkpoint/deploy độc lập.

Nhờ vậy cả 4 người đều có thể làm song song mà chỉ cần đồng bộ tại các contract đã thống nhất trong Sprint 0.
