# AdverTest P-195 — Runbook vận hành deployment

> Cập nhật: 2026-08-31. Đọc file này trước khi deploy, sửa hạ tầng, hoặc chẩn đoán lỗi production.
>
> Đây là **trạng thái đang vận hành**, không phải blueprint cũ trong `render-production.yaml`. Blueprint đó mô tả kiến trúc Redis worker trước đây; production hiện dùng Render làm control plane và GCP Cloud Run GPU làm execution plane.

## 1. Mục tiêu và nguyên tắc

Hệ thống tách phần web/API/DB rẻ tiền khỏi phần GPU đắt tiền:

```text
Browser
  ├─ Frontend Render: https://c4-app-195-advertest.is-pro.dev
  └─ API Render:      https://advertest-api-free-arqt.onrender.com
                         │
                         ├─ PostgreSQL Render: job state, metadata, report JSON
                         │
                         └─ Dispatcher Cloud Run ──> Pub/Sub `advertest-jobs`
                                                       │
                                                       ▼
                                              Cloud Run GPU worker (NVIDIA L4)
                                                       │
                                                       ▼
                                             GCS `advertest-prod-artifacts`
                                             ├─ catalog/datasets
                                             ├─ catalog/models
                                             └─ runs/<job_id>/evidence
```

- **Render API là control plane duy nhất ghi PostgreSQL.** GPU worker không có database URL/password.
- **GPU worker scale to zero**: job đầu tiên có thể mất vài phút để khởi động và nạp checkpoint/dataset từ GCS. Đây là hành vi bình thường, không phải lỗi.
- Ảnh, weights và evidence **không nằm trong PostgreSQL**. DB chỉ giữ metadata và report JSON.
- Không đưa secret, presigned URL, HMAC key hay token vào Git/tài liệu/log chat.

## 2. Tài nguyên production đã biết

### Render

| Thành phần | Service ID | URL / ghi chú |
|---|---|---|
| API | `srv-da2rsnmgekts73bih3l0` | `https://advertest-api-free-arqt.onrender.com` |
| Frontend | `srv-da2rm1vqj5pc738dm25g` | `https://c4-app-195-advertest.is-pro.dev` |
| PostgreSQL đang dùng | `dpg-da2rsd6gekts73bigbs0-a` | `advertest-postgres-free-arqt`, Singapore |
| Render workspace | `tea-d5rjgep4tr6s73e492ng` | Việt Phong's workspace |

`dpg-da2o0r7qj5pc7386vh8g-a` là PostgreSQL cũ/suspended; **không dùng để vận hành production**.

### GCP

| Thành phần | Tên / giá trị |
|---|---|
| Project | `ai20k-build` |
| Region | `asia-southeast1` |
| GCS bucket | `gs://advertest-prod-artifacts` |
| Dispatcher | `advertest-dispatcher` |
| GPU worker | `advertest-gpu-worker` |
| Checkpoint sandbox | `advertest-checkpoint-sandbox` |
| Pub/Sub topic | `advertest-jobs` |
| Push subscription | `advertest-gce-worker` |
| GPU policy | 1 NVIDIA L4, 4 vCPU, 16 GiB, concurrency 1, min instances 0, max instances 1, timeout 3600s |

Lấy URL hoặc revision hiện tại bằng lệnh `gcloud run services describe`; không hard-code revision URL vào code.

## 3. Repository và nhánh deploy

Có hai remote cần phân biệt:

```text
origin = git@github.com:AI20K-Build-Phase-Cohort-3/P-195.git
deploy = git@github.com:phongviet/advertest-deploy.git
```

- `origin`: repo nhóm; nhánh tích hợp hiện dùng là `feat/control-data-plane-callback`.
- `deploy`: repo mà Render theo dõi; Render deploy từ nhánh `main`.
- Sau khi test trên nhánh tích hợp, push cùng commit sang cả hai:

```bash
git push origin HEAD:feat/control-data-plane-callback
git push deploy HEAD:main
```

Không push file phát sinh như `frontend/public/samples/kitti/dynamic_*.png`, output trong `runs/`, artifact benchmark hay secret.

## 4. Biến môi trường bắt buộc

### Render API

Các giá trị thực phải đặt trong Render Environment; chỉ ghi **tên** biến ở đây:

```text
APP_ENV=production
PLATFORM_DATABASE_URL=<Render PostgreSQL URL>
RUN_EXECUTION_BACKEND=platform
QUEUE_BACKEND=http_dispatcher
EXTERNAL_QUEUE_DISPATCH_URL=https://<dispatcher>/dispatch
EXTERNAL_QUEUE_DISPATCH_TOKEN=<shared dispatcher token>
WORKER_CALLBACK_TOKEN=<callback token>
CHECKPOINT_SANDBOX_URL=https://<sandbox>/inspect
CHECKPOINT_SANDBOX_TOKEN=<sandbox token>
OBJECT_STORAGE_BACKEND=s3
OBJECT_STORAGE_BUCKET=advertest-prod-artifacts
OBJECT_STORAGE_ENDPOINT_URL=https://storage.googleapis.com
OBJECT_STORAGE_REGION=auto
OBJECT_STORAGE_ACCESS_KEY_ID=<GCS HMAC key>
OBJECT_STORAGE_SECRET_ACCESS_KEY=<GCS HMAC secret>
CORS_ORIGINS=https://c4-app-195-advertest.is-pro.dev,...
```

`RUN_EXECUTION_BACKEND=platform` và `QUEUE_BACKEND=http_dispatcher` là hai biến quyết định job có đi tới Cloud Run L4 hay không. Nếu thiếu/chuyển thành `local`, Render API sẽ chạy CPU local hoặc UI sẽ không phản ánh execution plane đúng.

### Render frontend

Frontend phải có runtime config trỏ tới API:

```text
NEXT_PUBLIC_API_URL=https://advertest-api-free-arqt.onrender.com
```

Script `deploy/start-frontend.sh` tạo `/runtime-config.js`; `frontend/src/app/layout.js` phải load file này trước hydration. Nếu quên, frontend có thể gọi nhầm `127.0.0.1:8000` và báo `Failed to fetch`.

### Cloud Run GPU worker

```text
MODEL_DEVICE=cuda:0
MODEL_HALF_PRECISION=true
WORKER_CALLBACK_API_URL=https://advertest-api-free-arqt.onrender.com
WORKER_CALLBACK_TOKEN=<cùng callback token ở Render API>
OBJECT_STORAGE_BACKEND=s3
OBJECT_STORAGE_BUCKET=advertest-prod-artifacts
OBJECT_STORAGE_ENDPOINT_URL=https://storage.googleapis.com
BOOTSTRAP_DEMO_MODEL=true
BOOTSTRAP_KITTI_CATALOG=true
BOOTSTRAP_CITYSCAPES_CATALOG=true
BOOTSTRAP_KITTI3D_CATALOG=true
```

Worker lấy object từ GCS sau khi claim job, chạy inference `cuda:0`, rồi callback trạng thái/kết quả về Render API.

## 5. Deploy code an toàn

1. Kiểm tra worktree; không ghi đè thay đổi không liên quan.

```bash
git status --short
```

2. Chạy kiểm thử theo vùng sửa đổi:

```bash
.venv/bin/pytest -q --capture=no tests/test_api/test_product_catalog.py tests/test_models/test_catalog.py
cd frontend && npm test -- --run src/app/experiments/[id]/attack/__tests__/attack-run-flow.test.jsx
cd frontend && npm run build
```

3. Commit chỉ file nguồn, push hai remote như mục 3. Render có auto-deploy; **không trigger deploy thủ công ngay sau push**.

4. Xem deploy qua Render MCP:

```text
mcp__render__list_deploys(
  workspaceId="tea-d5rjgep4tr6s73e492ng",
  serviceId="srv-da2rsnmgekts73bih3l0" | "srv-da2rm1vqj5pc738dm25g",
  limit=3
)
```

Trạng thái đúng cuối cùng là `live`. Nếu `build_in_progress`, chờ build thay vì redeploy chồng lên.

## 6. Render MCP: cách dùng trong phiên Codex mới

Không cần Render CLI. Dùng các tool MCP có tên dưới đây (một số session phải tìm tool trước):

| Nhu cầu | Tool |
|---|---|
| Xem services | `mcp__render__list_services` |
| Xem một service | `mcp__render__get_service` |
| Xem deploy | `mcp__render__list_deploys`, `mcp__render__get_deploy` |
| Xem log | `mcp__render__list_logs` |
| Xem metric CPU/memory/HTTP | `mcp__render__get_metrics` |
| Sửa biến env | `mcp__render__update_environment_variables` |
| Truy vấn DB read-only | `mcp__render__query_render_postgres` |

Quy tắc:

- Luôn truyền `workspaceId="tea-d5rjgep4tr6s73e492ng"` và service/database ID rõ ràng.
- Không dùng `update_environment_variables` để replace toàn bộ env trừ khi đã lấy đầy đủ biến; mặc định merge.
- Không paste secret vào output/tool commentary.
- `query_render_postgres` là read-only. Nếu tool báo TLS/EOF, coi đó là lỗi connector; dùng API/log hoặc Render dashboard, không tự thay đổi database.

Ví dụ đọc log API lỗi 5xx gần đây:

```text
mcp__render__list_logs({
  workspaceId: "tea-d5rjgep4tr6s73e492ng",
  resource: ["srv-da2rsnmgekts73bih3l0"],
  type: ["app", "request"],
  statusCode: ["5*"],
  limit: 100
})
```

## 7. GCP CLI: kiểm tra execution plane

Máy WSL cần có account `gcloud` đã đăng nhập và project `ai20k-build`.

```bash
gcloud auth list
gcloud run services list --project ai20k-build --region asia-southeast1 \
  --format='table(metadata.name,status.url,status.conditions[0].status)'
gcloud run services describe advertest-gpu-worker --project ai20k-build \
  --region asia-southeast1 \
  --format='yaml(status.url,status.traffic,spec.template.spec.containers[0].resources,spec.template.metadata.annotations)'
gcloud pubsub subscriptions describe advertest-gce-worker --project ai20k-build \
  --format='yaml(topic,pushConfig,ackDeadlineSeconds)'
```

Log worker/dispatcher:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="advertest-gpu-worker"' \
  --project ai20k-build --freshness=2h --limit=100 \
  --format='value(timestamp,severity,textPayload,httpRequest.status,httpRequest.latency)'

gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="advertest-dispatcher"' \
  --project ai20k-build --freshness=2h --limit=100 \
  --format='value(timestamp,severity,textPayload,httpRequest.status)'
```

`advertest-gpu-worker` có thể không có instance khi rảnh — đó là expected (`minScale=0`). Khi có job, log sẽ thấy `Starting new instance`, sau đó `POST /pubsub 200` khi job hoàn tất.

## 8. Kiểm tra health và một run

```bash
curl -fsS https://advertest-api-free-arqt.onrender.com/health
curl -fsS https://advertest-api-free-arqt.onrender.com/api/v1/system/runtime-specs
curl -fsS https://advertest-api-free-arqt.onrender.com/api/v1/model-versions
curl -fsS 'https://advertest-api-free-arqt.onrender.com/api/v1/catalog/datasets?task_id=detection2d'
curl -fsS https://advertest-api-free-arqt.onrender.com/api/v1/runs/<run_id>
curl -fsS https://advertest-api-free-arqt.onrender.com/api/v1/runs/<run_id>/samples
```

Diễn giải đúng runtime-specs:

- `control_plane`/root có thể là `CPU`: đó là Render API và hoàn toàn đúng.
- `execution_plane` phải là `cloud_run_gpu`, `cuda:0`, NVIDIA L4, `status=on_demand` khi platform backend đã cấu hình đúng.

Trạng thái run chuẩn:

```text
GPU_STARTING -> PREPARING -> GENERATING / INFERENCING -> EVALUATING -> COMPLETED
```

UI cần hiện **“GPU đang khởi động”** khi `GPU_STARTING`. Không báo lỗi chỉ vì bước này kéo dài vài phút.

## 9. Dataset, checkpoint và evidence

### Dataset source (GCS catalog)

| Dataset | Số mẫu đã xác minh | Ghi chú |
|---|---:|---|
| KITTI 2D | 200 | `catalog/datasets/kitti-200/v1` |
| KITTI 3D | 200 | `catalog/datasets/kitti3d-200/v1` |
| Cityscapes segmentation | 200 | có nhiều file PNG/mẫu vì mask/label; `dataset.json.sample_count=200` |
| BDD100K detection | 1 | smoke bundle, chưa phải BDD 200 |
| BDD100K semantic | 1 | smoke bundle |
| Folder dataset | 6 | smoke bundle |
| Generated dataset | 1 | smoke bundle |
| Synthetic Shapes | 24 | sinh động, không phải object GCS catalog |

Lệnh kiểm tra GCS (đếm object, không tự động suy ra sample từ số PNG Cityscapes):

```bash
gcloud storage cat gs://advertest-prod-artifacts/catalog/datasets/kitti-200/v1/dataset.json
gcloud storage cat gs://advertest-prod-artifacts/catalog/datasets/cityscapes-200/v1/dataset.json
gcloud storage ls --recursive gs://advertest-prod-artifacts/catalog/datasets/kitti-200/v1/image_2/ | wc -l
```

### Evidence khác dataset

`runs/<run_id>/evidence/...` là ảnh clean/attacked/prediction sinh ra **sau một run**. Một benchmark 200 mẫu có thể có khoảng 800 evidence object (4 artifact/mẫu), nhưng điều đó **không có nghĩa** BDD hoặc dataset catalog có 800 ảnh nguồn.

Evidence chỉ hiện trên Results của đúng `run_id`. Kiểm tra API `/samples`; mỗi sample phải có `artifacts.clean_input_url`, `attacked_input_url`, `clean_prediction_url`, `attacked_prediction_url`. Các URL này là presigned URL ngắn hạn và phải trả HTTP 200 khi còn hạn.

## 10. Lỗi thường gặp và cách chẩn đoán

| Triệu chứng | Nguyên nhân thường gặp | Kiểm tra / xử lý |
|---|---|---|
| `Failed to fetch` trên web | frontend dùng sai API URL hoặc CORS | kiểm tra `/runtime-config.js`, `NEXT_PUBLIC_API_URL`, API CORS và Network; không mặc định đổ lỗi GPU |
| UI báo CPU dù có L4 | UI đang đọc Render control plane | `/runtime-specs` phải phân biệt `control_plane` CPU với `execution_plane` Cloud Run GPU |
| `GPU_STARTING` lâu | Cloud Run scale từ 0 + tải GCS assets | xem worker logs; giữ thông báo “GPU đang khởi động”; không cancel sớm |
| INFERENCING/GENERATING 0% lâu | callback progress chưa tới sau mỗi cell hoặc recipe | kiểm tra `src/pipeline/runner.py`, `src/cloud_run_gpu_worker.py`, endpoint callback và `job_events` |
| BDD `extra_forbidden` với `difficulty`, `merge_van_truck` | UI gửi param KITTI sang BDD | chỉ thêm `difficulty`/`merge_van_truck` khi dataset là `kitti` |
| attack không compatible / `CHECKPOINT_MISSING` | catalog quảng cáo checkpoint chưa materialize | kiểm tra `/api/v1/model-versions`, GCS storage key, `catalog_model_versions()`; không để model thiếu bytes có `runnable=true` |
| Evidence không hiện | xem sai run hoặc sample không có artifact URL | gọi `/api/v1/runs/<id>/samples`, kiểm tra URL HTTP 200 và hạn presigned URL |
| AP/mAP bằng 0 | run không có GT phù hợp hoặc quick inference | chỉ tin AP/mAP từ benchmark có GT; không suy từ confidence ảnh đơn |

## 11. Files/code quan trọng

| File | Vai trò |
|---|---|
| `src/api/routers/runs.py` | tạo/poll run platform, hiển thị status GPU_STARTING |
| `src/api/routers/worker_callbacks.py` | endpoint callback authenticated của GPU worker |
| `src/cloud_run_gpu_worker.py` | claim job, hydrate GCS, inference GPU, publish evidence |
| `src/jobs/queue.py` | HTTP dispatcher queue |
| `deploy/gcp/dispatcher/main.py` | dispatcher Render -> Pub/Sub |
| `src/api/routers/catalog.py` | catalog dataset/model/attack và sample_count |
| `frontend/src/app/experiments/new/page.jsx` | chọn model/dataset và runtime banner |
| `frontend/src/app/experiments/[id]/attack/page.jsx` | progress modal, báo GPU đang khởi động |
| `frontend/src/app/experiments/[id]/results/page.jsx` | ảnh evidence từ `artifacts.*_url` |

## 12. Handoff prompt cho phiên Codex mới

Có thể copy nguyên văn đoạn sau:

```text
Đọc docs/DEPLOYMENT_OPERATIONS_RUNBOOK.md trước khi thao tác. Production dùng
Render control plane + GCP Cloud Run NVIDIA L4 execution plane, không dùng
Render Redis worker blueprint cũ. Repo deploy là remote `deploy` và branch
`main`; source nhóm là origin branch `feat/control-data-plane-callback`.
Không in secret. Trước khi deploy: test, push cả origin và deploy, sau đó theo
dõi Render MCP auto-deploy. Khi chẩn đoán job: kiểm tra Render API, dispatcher,
Pub/Sub, GPU worker logs và GCS evidence; phân biệt source dataset với evidence.
```

