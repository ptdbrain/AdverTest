# AdverTest — Hướng Dẫn Kỹ Thuật Triển Khai Database & Storage

> **Dành cho:** Kỹ sư DevOps / Backend phụ trách triển khai hạ tầng & cơ sở dữ liệu.  
> **Cập nhật:** 2026-08-29  
> **Hạ tầng mục tiêu:** Render (Control Plane: FastAPI + PostgreSQL) + GCP (Compute & Storage: Cloud Run GPU L4 + GCS Bucket).

---

## 1. Tổng quan kiến trúc lưu trữ

Hệ thống tuân thủ nguyên tắc phân tách trách nhiệm (**Separation of Concerns**) giữa Database và Object Storage:

1. **PostgreSQL (Render Database)**: Lưu trữ dữ liệu có cấu trúc, quan hệ, metadata, thông tin tài khoản người dùng, phiên làm việc (sessions), tiến trình job và toàn bộ báo cáo JSON metrics (mAP, PSNR, SSIM, mIoU, độ suy giảm...). **Tuyệt đối không lưu file nhị phân (ảnh, dataset, weights) trực tiếp trong PostgreSQL**.
2. **Google Cloud Storage (GCS Bucket: `advertest-prod-artifacts`)**: Lưu trữ các file nhị phân lớn:
   - Tập dataset kiểm thử (~200 ảnh và nhãn ground-truth cho mỗi bài toán).
   - Checkpoint trọng số mô hình (~3 model cho mỗi bài toán).
   - Ảnh bằng chứng (Evidence images) sinh ra sau mỗi lượt tấn công (Clean vs Attacked vs Overlay).

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                    NGƯỜI DÙNG / BROWSER                │
                               └───────────┬────────────────────────────────┬───────────┘
                                           │ HTTPS                          │ Presigned URL (Xem ảnh)
                                           ▼                                ▼
┌───────────────────────────────────────────────────────┐       ┌──────────────────────────────┐
│                  RENDER CONTROL PLANE                 │       │    GCP DATA & STORAGE PLANE  │
│                                                       │       │                              │
│  ┌──────────────────┐         ┌────────────────────┐  │       │  ┌────────────────────────┐  │
│  │   Frontend Web   │────────►│    FastAPI Backend │  │       │  │ GCS Bucket:            │  │
│  │    (Next.js)     │         │       (Render)     │  │       │  │ advertest-prod-artifacts│  │
│  └──────────────────┘         └─────────┬──────────┘  │       │  │                        │  │
│                                         │             │       │  │ • /catalog/datasets/   │  │
│                                         ▼             │       │  │   (200 ảnh + labels)   │  │
│                               ┌────────────────────┐  │       │  │ • /catalog/models/     │  │
│                               │ Render PostgreSQL  │  │       │  │   (Weights .pt, .pth)  │  │
│                               │                    │  │       │  │ • /runs/<job_id>/evid/ │  │
│                               │ • Users & Projects │  │       │  │   (Clean/Perturbed PNG)│  │
│                               │ • Sessions & Runs  │  │       │  └───────────▲────────────┘  │
│                               │ • Checkpoints Meta │  │       │              │               │
│                               │ • Datasets Meta    │  │       │              │ Read inputs   │
│                               │ • Results & Metrics│◄─┼───────┼──────────────┼ Write evidence│
│                               └────────────────────┘  │       │              │               │
│                                                       │       │  ┌───────────┴────────────┐  │
│                                                       │       │  │ Cloud Run GPU Worker   │  │
│                                                       │       │  │ (NVIDIA L4 - Min=0)    │  │
│                                                       │       │  └────────────────────────┘  │
└───────────────────────────────────────────────────────┘       └──────────────────────────────┘
```

---

## 2. Thiết kế Cơ sở dữ liệu (PostgreSQL Schema)

### 2.1 Sơ đồ thực thể quan hệ (ERD)

```mermaid
erDiagram
    USERS ||--o{ PROJECTS : "owns"
    USERS ||--o{ SESSIONS : "creates"
    USERS ||--o{ AUDIT_LOGS : "acts"
    PROJECTS ||--o{ ARTIFACTS : "contains"
    PROJECTS ||--o{ DATASET_VERSIONS : "registers"
    PROJECTS ||--o{ CHECKPOINTS : "manages"
    ARTIFACTS ||--o| DATASET_VERSIONS : "files backing"
    ARTIFACTS ||--o| CHECKPOINTS : "weights backing"
    SESSIONS ||--o{ JOBS : "contains sequential runs"
    JOBS ||--o{ JOB_EVENTS : "logs progress events"

    USERS {
        string id PK "usr-uuid"
        string email UK
        string password_hash
        string display_name
        string role "ADMIN | ENGINEER | VIEWER"
        string status "ACTIVE | SUSPENDED"
        bigint storage_quota_bytes
        float compute_quota_hours
        timestamp created_at
        timestamp last_login_at
    }

    SESSIONS {
        string id PK "EXP-YYYYMMDD-XXX"
        string user_id FK
        string name
        string description
        string task_id "detection2d | segmentation | detection3d"
        string model_id
        string dataset_id
        string status "active | completed"
        timestamp created_at
        timestamp updated_at
    }

    JOBS {
        string id PK "job-uuid / RUN-XXX"
        string session_id FK
        string project_id FK
        string owner_user_id FK
        string type "benchmark | attack | validation"
        string status "QUEUED | GPU_STARTING | RUNNING | COMPLETED | FAILED"
        string stage
        int completed_units
        int total_units
        text request_json "Tham số tấn công & cấu hình"
        text result_json "Toàn bộ metrics & URL evidence"
        text error_message
        timestamp created_at
        timestamp started_at
        timestamp completed_at
    }

    CHECKPOINTS {
        string id PK "ckpt-uuid"
        string artifact_id FK
        string task_id "detection2d | segmentation | detection3d"
        string model_family_id "yolo11s | faster_rcnn | sam2..."
        string source "catalog | user_upload"
        string status "READY | QUARANTINE | FAILED"
        text native_class_names_json
        int num_classes
    }

    DATASET_VERSIONS {
        string id PK "dsv-uuid"
        string artifact_id FK
        string dataset_id "kitti-200 | bdd100k-200"
        string display_name
        string task_id "detection2d | segmentation"
        string status "READY | PROCESSING"
        int sample_count "200"
        text manifest_json "Tóm tắt schema & phân bố class"
    }
```

### 2.2 Chi tiết các bảng nghiệp vụ

| Bảng | Mục đích | Dữ liệu chính |
|---|---|---|
| `users` | Quản lý tài khoản & phân quyền | Email, hashed password, role (ADMIN/ENGINEER), quota lưu trữ/GPU |
| `projects` | Phân chia không gian làm việc | Tên dự án, owner user ID, mô tả |
| `sessions` | Phiên làm việc của kỹ sư | Mã phiên (`EXP-2026-...`), bài toán, model chọn, dataset chọn, trạng thái |
| `jobs` (Runs) | Từng lần chạy tấn công / benchmark | Mã lượt chạy (`RUN-001`), liên kết session, loại tấn công, tham số `request_json`, kết quả `result_json` |
| `job_events` | Nhật ký tiến trình thời gian thực | Sequence, stage (`GPU_STARTING`, `INFERENCE_CLEAN`, `ATTACKING`), % tiến độ |
| `artifacts` | Quản lý object nhị phân trên GCS | `storage_key`, mã băm SHA-256, kích thước bytes, MIME type |
| `checkpoints` | Quản lý các mô hình AI | Model family (YOLO11, SAM2, Faster R-CNN...), task ID, trạng thái validation |
| `dataset_versions` | Quản lý các bộ dữ liệu test | Dataset ID, số lượng ảnh (200), task ID, hash schema |
| `audit_logs` | Nhật ký bảo mật | Người thực hiện, hành động (`USER_LOGIN`, `RUN_ATTACK`, `EXPORT_REPORT`), timestamp |

---

## 3. Cấu trúc Object Storage (GCS Bucket)

Tên bucket: `advertest-prod-artifacts`

```text
gs://advertest-prod-artifacts/
│
├── catalog/                                    # Dữ liệu mẫu chuẩn của hệ thống
│   ├── datasets/                               # 200 ảnh cho từng bài toán
│   │   ├── kitti-200/v1/                       # Bài toán 2D Detection
│   │   │   ├── dataset.json                    # Metadata & anonymization status
│   │   │   ├── manifest.jsonl                  # 200 dòng map ảnh và ground truth bbox
│   │   │   └── images/                         # 200 file ảnh: 000001.png -> 000200.png
│   │   ├── cityscapes-200/v1/                  # Bài toán Instance Segmentation
│   │   └── kitti3d-200/v1/                     # Bài toán 3D LiDAR Detection
│   │
│   └── models/                                 # 3 Model cho mỗi bài toán
│       ├── detection2d/
│       │   ├── yolo11s/v1/yolo11s.pt           # Model 1 (Ultralytics Vision)
│       │   ├── faster_rcnn/v1/faster_rcnn.pth  # Model 2 (TorchVision ResNet50)
│       │   └── rtdetr/v1/rtdetr-l.pt           # Model 3 (Real-time DETR)
│       ├── segmentation/
│       │   ├── sam2_tiny/v1/sam2_hiera_t.pt    # Model 1 (Segment Anything 2 Tiny)
│       │   ├── sam2_base/v1/sam2_hiera_b.pt    # Model 2 (Segment Anything 2 Base)
│       │   └── mask_rcnn/v1/mask_rcnn.pth      # Model 3 (Mask R-CNN)
│       └── detection3d/
│           ├── pointpillars/v1/pointpillars.pth# Model 1 (PointPillars LiDAR)
│           ├── bevfusion/v1/bevfusion.pth      # Model 2 (BEVFusion Multi-modal)
│           └── blob_detector/v1/baseline.pth   # Model 3 (Baseline Geometry)
│
└── runs/                                       # Bằng chứng sinh ra sau mỗi lượt tấn công
    └── <job_id>/
        └── evidence/
            ├── 000001_clean.png                # Ảnh sạch ban đầu
            ├── 000001_attacked.png             # Ảnh sau khi bị chèn nhiễu
            ├── 000001_overlay_diff.png         # Bounding box so sánh sai lệch
            └── ...
```

---

## 4. Đặc tả dữ liệu chi tiết của từng lượt tấn công (Metrics & Reports)

Mỗi lần tấn công được thực thi trên Cloud Run GPU L4, kết quả sẽ được tổng hợp thành JSON và lưu vào trường `result_json` của bảng `jobs` trong PostgreSQL:

### Cấu trúc JSON Metrics lưu trong PostgreSQL:

```json
{
  "run_id": "RUN-20260829-001",
  "session_id": "EXP-2026-0512-001",
  "task_id": "detection2d",
  "model": {
    "model_id": "yolo11s",
    "model_name": "YOLO11s (Ultralytics Vision)",
    "family": "yolo"
  },
  "dataset": {
    "dataset_id": "kitti-200",
    "total_samples": 200,
    "anonymized": true
  },
  "attack": {
    "type": "pgd_linf",
    "name": "Projected Gradient Descent (L-infinity)",
    "threat_model": "whitebox",
    "severity": 3,
    "parameters": {
      "epsilon": 0.03137,
      "alpha": 0.00784,
      "steps": 10,
      "random_start": true
    }
  },
  "metrics": {
    "detection_performance": {
      "clean_map50": 0.8425,
      "attacked_map50": 0.3120,
      "map_drop_pct": 62.97,
      "clean_map50_95": 0.5840,
      "attacked_map50_95": 0.1830,
      "clean_avg_confidence": 0.892,
      "attacked_avg_confidence": 0.415,
      "clean_bbox_count": 842,
      "attacked_bbox_count": 310
    },
    "segmentation_performance": {
      "clean_miou": null,
      "attacked_miou": null,
      "miou_drop_pct": null
    },
    "perturbation_quality": {
      "psnr_db": 28.64,
      "ssim": 0.8872,
      "lpips": 0.1145,
      "l2_norm": 1.42,
      "linf_norm": 0.0313
    },
    "computational_cost": {
      "clean_latency_ms": 12.4,
      "attack_generation_latency_ms": 48.6,
      "total_gpu_time_seconds": 18.2
    },
    "composite_scores": {
      "robustness_score": 37.03,
      "safety_rating": "HIGH_VULNERABILITY"
    }
  },
  "evidence_samples": [
    {
      "sample_id": "000001",
      "clean_image_url": "https://storage.googleapis.com/advertest-prod-artifacts/runs/job-123/evidence/000001_clean.png?Expires=...",
      "attacked_image_url": "https://storage.googleapis.com/advertest-prod-artifacts/runs/job-123/evidence/000001_attacked.png?Expires=...",
      "diff_overlay_url": "https://storage.googleapis.com/advertest-prod-artifacts/runs/job-123/evidence/000001_diff.png?Expires=...",
      "clean_detections": [
        {"class": "car", "confidence": 0.94, "bbox": [120, 340, 260, 480]}
      ],
      "attacked_detections": []
    }
  ],
  "expert_notes": "Mô hình sụt giảm nghiêm trọng khi tấn công PGD ở cấp độ 3, mất hoàn toàn nhận diện phương tiện ở khoảng cách xa."
}
```

---

## 5. Hướng dẫn nạp dữ liệu mẫu (Ingestion Guide)

### 5.1 Upload 200 ảnh lên GCS & đăng ký Dataset

1. **Chuẩn bị cấu trúc thư mục local:**
   ```text
   data/datasets/kitti-200/
   ├── dataset.json
   ├── manifest.jsonl
   └── images/ (chứa 200 ảnh)
   ```
2. **Nội dung `dataset.json`:**
   ```json
   {
     "name": "kitti-200",
     "display_name": "KITTI Test Set (200 Samples)",
     "task_id": "detection2d",
     "sample_count": 200,
     "anonymized": true,
     "license": "CC BY-NC-SA 3.0"
   }
   ```
3. **Upload lên GCS:**
   ```bash
   gcloud storage cp -r data/datasets/kitti-200 gs://advertest-prod-artifacts/catalog/datasets/
   ```

### 5.2 Upload Checkpoint Model lên GCS & đăng ký

1. **Upload các file trọng số:**
   ```bash
   gcloud storage cp weights/yolo11s.pt gs://advertest-prod-artifacts/catalog/models/detection2d/yolo11s/v1/yolo11s.pt
   gcloud storage cp weights/faster_rcnn.pth gs://advertest-prod-artifacts/catalog/models/detection2d/faster_rcnn/v1/faster_rcnn.pth
   gcloud storage cp weights/rtdetr.pt gs://advertest-prod-artifacts/catalog/models/detection2d/rtdetr/v1/rtdetr.pt
   ```

2. **Kích hoạt tự động đồng bộ khi khởi động (Bootstrap):**
   Cấu hình trong biến môi trường của Worker GPU:
   ```text
   BOOTSTRAP_DEMO_MODEL=true
   BOOTSTRAP_DEMO_KITTI=true
   BOOTSTRAP_DEMO_CATALOG=true
   ```

---

## 6. Danh mục Biến môi trường Cần thiết (Environment Variables)

### 6.1 Backend API (Render)
```env
APP_ENV=production
PLATFORM_DATABASE_URL=postgresql://user:password@dpg-xxxx.render.com/advertest_db
RUN_EXECUTION_BACKEND=platform
QUEUE_BACKEND=http_dispatcher
EXTERNAL_QUEUE_DISPATCH_URL=https://dispatcher-service-xxxx.run.app/dispatch
EXTERNAL_QUEUE_DISPATCH_TOKEN=super_secret_dispatcher_token_here

OBJECT_STORAGE_BACKEND=s3
OBJECT_STORAGE_BUCKET=advertest-prod-artifacts
OBJECT_STORAGE_ENDPOINT_URL=https://storage.googleapis.com
OBJECT_STORAGE_ACCESS_KEY_ID=<GCS_HMAC_ACCESS_KEY>
OBJECT_STORAGE_SECRET_ACCESS_KEY=<GCS_HMAC_SECRET_KEY>
OBJECT_STORAGE_SIGNED_URL_TTL_SECONDS=900

JWT_SECRET=<strong_jwt_random_secret>
ADMIN_DEFAULT_PASSWORD=<secure_initial_admin_password>
CORS_ORIGINS=https://advertest.ai,https://advertest-frontend.onrender.com
```

### 6.2 GPU Worker (GCP Cloud Run L4)
```env
GOOGLE_CLOUD_PROJECT=ai20k-build
MODEL_DEVICE=cuda:0
MODEL_HALF_PRECISION=true
MODEL_BATCH_SIZE=1

# Lấy từ Secret Manager
PLATFORM_DATABASE_URL=<Secret: PLATFORM_DATABASE_URL>
OBJECT_STORAGE_BACKEND=s3
OBJECT_STORAGE_BUCKET=advertest-prod-artifacts
OBJECT_STORAGE_ENDPOINT_URL=https://storage.googleapis.com
OBJECT_STORAGE_ACCESS_KEY_ID=<Secret: GCS_HMAC_KEY>
OBJECT_STORAGE_SECRET_ACCESS_KEY=<Secret: GCS_HMAC_SECRET>

BOOTSTRAP_DEMO_MODEL=true
BOOTSTRAP_DEMO_KITTI=true
BOOTSTRAP_DEMO_CATALOG=true
```

---

## 7. Quy trình triển khai từng bước (DevOps Deployment Checklist)

- [ ] **1. Cơ sở dữ liệu (PostgreSQL)**:
  - Tạo PostgreSQL instance trên Render hoặc Cloud SQL.
  - Cấu hình Pre-deploy Command trên Render: `alembic upgrade head`.
  - Cấu hình **Inbound IP Rules / Allowlist** trên PostgreSQL để Cloud Run Worker (qua Static NAT IP) có thể kết nối.

- [ ] **2. Lưu trữ (GCS Bucket)**:
  - Tạo private bucket `advertest-prod-artifacts` trên GCP.
  - Tạo GCS HMAC Keys cho Render API (quyền Read/Write Presigned URL).
  - Cấp quyền `roles/storage.objectAdmin` cho Service Account của Cloud Run GPU Worker.
  - Đẩy 200 ảnh test và các file trọng số mô hình vào các prefix `catalog/datasets/` và `catalog/models/`.

- [ ] **3. Control Plane (Render API & Frontend)**:
  - Deploy Render Web Service cho API từ `Dockerfile`.
  - Cấu hình đầy đủ biến môi trường và kiểm tra endpoint `GET /health` trả về `200 OK`.
  - Deploy Next.js Frontend từ `frontend/Dockerfile` với `NEXT_PUBLIC_API_URL`.

- [ ] **4. Compute Plane (Cloud Run Dispatcher & GPU Worker)**:
  - Build và push image lên Artifact Registry:
    - Dispatcher: `deploy/gcp/dispatcher/Dockerfile`
    - GPU Worker: `deploy/gcp/gce-worker/Dockerfile`
  - Deploy Cloud Run GPU L4 với `min-instances=0`, cấu hình Pub/Sub Push Subscription `advertest-jobs` trỏ vào endpoint `POST /pubsub` của Worker.

- [ ] **5. Kiểm thử xác thực End-to-End**:
  - Đăng ký tài khoản Admin đầu tiên qua UI / API `/auth/register`.
  - Tạo một Phiên làm việc mới (Session `EXP-001`).
  - Chọn model (YOLO11s), chọn bộ dữ liệu 200 ảnh và kích hoạt một lượt tấn công (PGD).
  - Kiểm tra trạng thái chuyển từ `GPU_STARTING` → `RUNNING` → `COMPLETED`.
  - Xác nhận báo cáo metrics hiển thị đầy đủ trên giao diện và các ảnh so sánh (Evidence) xem được qua Presigned URL.
