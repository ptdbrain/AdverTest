# 🛡️ AdverTest — Sinh & kiểm thử adversarial cho model perception

> **SIMULATION ONLY.** Mọi con số ở đây là kết quả mô phỏng. Hệ thống không có
> đường nối tới pipeline triển khai, và không tự kết luận model "đủ an toàn" —
> quyết định cuối luôn thuộc về con người.

Bản kế hoạch kỹ thuật đầy đủ: [`docs/advertest-plan.md`](docs/advertest-plan.md).

## Bản starter này là gì

Khung code tổng quát để **cả nhóm làm song song**: mỗi người thêm một phép tấn
công (hoặc một model, một dataset, một chỉ số) trong **file riêng của mình**,
không sửa file chung, nên gần như không có merge conflict.

Đã chạy được end-to-end với hai ví dụ mẫu (`gaussian_noise`, `fgsm`) trên một
model tham chiếu thuần numpy — không cần GPU, weight, hay tải dataset:

```
dataset (đã ẩn danh) → attack plugin → model adapter → AP/Degradation → RunReport
```

Nhóm D/E đã có pipeline sinh attack dataset độc lập, gồm FGSM, PGD, MI-FGSM,
C&W, TOG, DAG, SAM2-PGD, DPatch và Thys patch. Các phần benchmark
(mPC/rPC/ASR/RobustScore), nhóm attack còn lại, review queue và Optuna red-team
vẫn nằm ngoài pipeline này.

## ⚡ Quick Start

```bash
uv sync                                  # core + dev, Python 3.11 từ .python-version
uv sync --extra models-cpu               # Torch/Torchvision/Ultralytics cho máy CPU
# uv sync --extra models-gpu             # dùng trên máy NVIDIA/CUDA
cp .env.example .env                     # điền AI_LOG_API_KEY của nhóm

make catalog                             # attack/model/dataset nào đang có, ai giữ
make demo                                # chạy thử một test run nhỏ
make test                                # pytest
make run                                 # API: http://localhost:8000/docs
```

### Environment variables

Copy `.env.example` to `.env` for local development. Keep secrets and private
dataset paths out of tracked files.

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Runtime profile. |
| `APP_HOST` | `0.0.0.0` | Uvicorn bind address. |
| `APP_PORT` | `8000` | API port. |
| `DATABASE_URL` | `sqlite:///./data/app.db` | Run and job storage. |
| `DATA_ROOT` | `./data` | Uploads, datasets, and artifacts. |
| `CORS_ORIGINS` | local origins | Browser origins allowed by the API. |
| `AI_LOG_API_KEY` | unset | Optional AI-usage log submission key. |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend API base URL. |

Use the deployment secret manager for production credentials; never commit a
real token or checkpoint path.

Output thật của `make demo`:

```
run 713527bdb975  model=blob-1.0.0:thr0.45  dataset=synthetic_shapes
samples=4  AP_clean=1.000  seconds=0.10
group  attack          severity  ap      D%     cache_hits
-----  --------------  --------  ------  -----  ----------
A      gaussian_noise  1         1.0     0.0    0
A      gaussian_noise  3         1.0     0.0    0
A      gaussian_noise  5         0.75    25.0   0
D      fgsm            1         1.0     0.0    0
D      fgsm            3         0.9583  4.2    0
D      fgsm            5         0.0     100.0  0
```

> Lưu ý về model tham chiếu: `blob_detector` là detector ngưỡng thuần numpy, biên
> quyết định của nó rộng hơn CNN thật rất nhiều. Vì vậy `make demo` truyền `ε` lớn
> hơn mặc định của plan (`{1..16}/255`) cho `fgsm` — dùng nó để kiểm tra *cơ chế*
> attack, còn kiểm tra *độ mạnh* thì phải chạy trên adapter model thật.

## 🧩 Thêm một attack (việc chính của mỗi thành viên)

```bash
git checkout -b feat/attack-motion-blur
cp src/attacks/_template.py src/attacks/corruption/motion_blur.py
# sửa name/group/owner + hàm apply()
uv run pytest tests/test_attacks -q      # contract test tự bao phủ file mới
uv run python -m src.cli run --attacks motion_blur --severities 1,3,5 --limit 4
```

Không cần đăng ký ở đâu khác: registry tự phát hiện file mới. Hướng dẫn đầy đủ
(hợp đồng bắt buộc, attack cần gradient, cách chọn `cost_class`, checklist PR,
bảng slot còn trống): **[docs/CONTRIBUTING_ATTACKS.md](docs/CONTRIBUTING_ATTACKS.md)**.

## 📁 Cấu trúc

```
├── src/
│   ├── core/               # 🧱 types, registry + auto-discovery, hashing, image ops
│   ├── attacks/            # 💥 plugin tấn công — MỘT FILE / MỘT ATTACK
│   │   ├── base.py         #    hợp đồng BaseAttack (đừng sửa)
│   │   ├── _template.py    #    copy file này để bắt đầu
│   │   ├── corruption/     #    nhóm A — có gaussian_noise
│   │   ├── weather/        #    nhóm B — slot (depth-aware)
│   │   ├── occlusion/      #    nhóm C — slot
│   │   ├── adversarial/    #    nhóm D — gradient attack
│   │   ├── patch/          #    nhóm E — artifact patch
│   │   └── blackbox/       #    nhóm F — slot
│   ├── adapters/           # 🤖 model under test (M1–M6); blob_detector = model tham chiếu
│   ├── datasets/           # 🗂️ nguồn dữ liệu + cổng ẩn danh bắt buộc
│   ├── evaluation/         # 📏 IoU, AP, degradation, RunReport
│   ├── pipeline/           # ⚙️ TestRunner + AttackDatasetGenerator
│   ├── training/           # 🧪 PatchTrainer
│   ├── anonymization/      # 🔒 ẩn danh khuôn mặt/biển số
│   ├── api/                # 🌐 FastAPI: catalog + runs
│   ├── cli.py              # 🖥️ python -m src.cli attacks|run|estimate
│   └── config.py           # 🔧 Pydantic Settings
├── tests/
│   ├── test_attacks/       #    contract test (tự chạy cho mọi attack mới)
│   ├── test_adapters/ test_datasets/ test_evaluation/ test_pipeline/ test_api/
├── docs/
│   ├── advertest-plan.md   # 📋 kế hoạch kỹ thuật (nguồn sự thật)
│   ├── CONTRIBUTING_ATTACKS.md
│   └── guide/              # 📖 Technical Guidebook của BTC
├── scripts/                # 🔌 AI usage logging hooks
├── .ai-log/                # 📊 log AI (tự sinh, submit khi git push)
├── ARCHITECTURE.md         # 🏗️ kiến trúc + design decisions
└── eval/ presentation/     # 📊 evidence + slides cho Demo Day
```

## 🏗️ Architecture and data flow

The concise component and sequence diagrams live in
[`docs/architecture-diagram.md`](docs/architecture-diagram.md). The longer
design record remains in [`ARCHITECTURE.md`](ARCHITECTURE.md). Start with the
diagram when tracing a request from ingestion through validation, attack
composition, inference, metrics, and report export.

## Attack Dataset Generator

### Dataset and checkpoint request shapes

A minimal reproducible run keeps the task, family, checkpoint, dataset, recipe,
seed, and limit explicit:

```json
{
  "task_id": "detection2d",
  "model_family_id": "yolo11",
  "checkpoint_id": "yolo11s-base",
  "dataset": "synthetic_shapes",
  "recipe": {"steps": [{"attack_name": "gaussian_noise", "severity": 3}]},
  "seed": 42,
  "limit": 4
}
```

An annotated upload also needs a canonical label document per sample:

```json
{
  "task_id": "detection2d",
  "annotations": [{"class_id": "car", "bbox_xyxy": [120, 80, 420, 300]}]
}
```

The server validates coordinates and class-map membership before a dataset is
marked benchmark-ready.

Nhóm D/E có pipeline riêng để sinh dataset bị tấn công mà không gọi evaluator
hoặc tính AP. Config synthetic chạy không cần checkpoint; config KITTI dùng
checkpoint local và không tự tải weight:

```bash
uv run python -m src.cli generate-attack --config configs/pgd.json
uv run python -m src.cli anonymize-dataset --config configs/kitti-anonymize-smoke.json
uv run python -m src.cli generate-attack --config configs/kitti-fgsm.json
uv run python -m src.cli generate-attack --config configs/kitti-pgd.json
uv run python -m src.cli train-patch --config configs/patch.json
uv run python -m src.cli anonymize-dataset --config configs/kitti-anonymize-de.json
uv run python -m src.cli generate-attack --config configs/kitti-mi-fgsm.json
uv run python -m src.cli generate-attack --config configs/kitti-tog-vanishing.json
uv run python -m src.cli generate-attack --config configs/kitti-cw-l2.json
uv run python -m src.cli train-patch --config configs/kitti-dpatch-train.json
uv run python -m src.cli generate-attack --config configs/kitti-dpatch-apply.json
uv run python -m src.cli benchmark-attack-datasets --config configs/kitti-yolo11-benchmark.json
```

Hướng dẫn input, checkpoint, attack params, output manifest và resume:
**[docs/ATTACK_DATASET_GENERATOR.md](docs/ATTACK_DATASET_GENERATOR.md)**.

## 🔌 API

### Sample queries

```bash
curl -s http://localhost:8000/health | jq
curl -s http://localhost:8000/api/v1/catalog/attacks | jq '.[] | {name, group, required_annotations}'
curl -s http://localhost:8000/api/v1/catalog/models | jq '.[] | {name, task}'
curl -s http://localhost:8000/api/v1/catalog/datasets | jq '.[] | {name, task, anonymized}'
```

Estimate and queue a small reproducible run:

```bash
curl -s http://localhost:8000/api/v1/runs/estimate \
  -H 'content-type: application/json' \
  -d '{"attacks":["gaussian_noise"],"severities":[1,3],"limit":4,"seed":42}' | jq

curl -s http://localhost:8000/api/v1/runs \
  -H 'content-type: application/json' \
  -d '{"attacks":["gaussian_noise","fgsm"],"severities":[1,3,5],"limit":4,"seed":42}' | jq
```

Replace identifiers with values returned by the catalog endpoints and preserve
the returned run ID when sharing simulation evidence.

| Endpoint | Mô tả |
|---|---|
| `GET /health` | trạng thái + banner simulation |
| `GET /api/v1/catalog/attacks` | catalog attack kèm `owner`, `params_schema` |
| `GET /api/v1/catalog/models` · `/datasets` | adapter và dataset đã đăng ký |
| `POST /api/v1/runs/estimate` | ước tính chi phí **trước** khi chạy |
| `POST /api/v1/runs` | chạy test run, trả `RunReport` |
| `GET /api/v1/runs` · `/runs/{id}` | danh sách / báo cáo chi tiết |

```bash
curl -s localhost:8000/api/v1/runs -H 'content-type: application/json' \
  -d '{"attacks":["gaussian_noise","fgsm"],"severities":[1,3,5],"limit":4}' | jq .heatmap
```

## 🛠 Tech Stack

## 🧭 Local development workflow

```bash
uv sync
make catalog
uv run pytest tests/test_api tests/test_pipeline -q
uv run ruff check src tests
make demo
```

Run `make run` in one terminal and query `http://localhost:8000/docs` from a
second terminal. Record the seed, dataset version, checkpoint identifier,
recipe, and run ID with any shared evidence.

| Layer | Hiện tại | Khi lên model thật (plan §4) |
|---|---|---|
| Attack/metric | numpy | + kornia (GPU), imagecorruptions, torchattacks |
| Model | adapter thuần numpy (`blob_detector`) | + torch, ultralytics, MMDetection(3D) |
| Metric | AP50 tự cài | + pycocotools, bootstrap CI |
| Backend | FastAPI + Uvicorn | + Celery/Redis, PostgreSQL, MinIO, W&B |
| Test/CI | pytest + ruff + GitHub Actions | + sanity-check gate (plan §3) |

## 📋 Deliverables

| # | Deliverable | Vị trí |
|---|---|---|
| 1 | Source Code | `src/` |
| 2 | README | file này |
| 3 | Architecture Diagram | `ARCHITECTURE.md`, `docs/architecture-diagram.md` |
| 4 | AI Logs | `.ai-log/` (hook tự động, submit khi `git push`) |
| 5 | Live URL | Dockerfile + CI đã sẵn |
| 6–7 | Video + Pitch Deck | `presentation/` |
| 8–9 | Journal + Worklog | `JOURNAL.md`, `WORKLOG.md` |
| 10 | Evaluation Evidence | `eval/` |

## 🩺 Troubleshooting

- **Port 8000 is already in use:** stop the existing Uvicorn process or set a
  different `APP_PORT`, then update `NEXT_PUBLIC_API_URL`.
- **Missing anonymisation manifest:** use an anonymised dataset import and
  verify the manifest is inside the configured dataset root.
- **Missing model capability or annotation:** inspect catalog fields such as
  `required_capabilities` and `required_annotations` before selecting an attack.
- **Empty metrics:** raw images support quick inference only; AP/mAP requires
  reviewed ground truth and a finalized dataset version.
- **The UI cannot reach the API:** check CORS origins, API port, and `/health`.

## 📊 AI Usage Logging

Hook đã cấu hình sẵn cho Claude Code, Cursor, Codex, Gemini CLI, Copilot,
Antigravity. Mọi prompt/tool call ghi vào `.ai-log/session.jsonl` và tự submit lên
grading server mỗi lần `git push`.

```bash
bash scripts/setup_hooks.sh   # chạy một lần sau khi clone
```

Log thủ công cho ChatGPT/web tool:

```bash
uv run python scripts/log_manual.py --tool chatgpt --prompt "What you asked"
```

> ⚠️ Đừng sửa/xoá file trong `.ai-log/`, đừng `git push --no-verify`.

## 📄 License

MIT — dùng cho mục đích giáo dục.

## 📚 Documentation map

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — full component boundaries and design decisions.
- [`docs/architecture-diagram.md`](docs/architecture-diagram.md) — concise Mermaid diagrams.
- [`docs/advertest-plan.md`](docs/advertest-plan.md) — technical source-of-truth plan.
- [`docs/CONTRIBUTING_ATTACKS.md`](docs/CONTRIBUTING_ATTACKS.md) — attack plugin contract and PR checklist.
- [`docs/ATTACK_DATASET_GENERATOR.md`](docs/ATTACK_DATASET_GENERATOR.md) — generated dataset inputs and manifests.

When documentation and implementation appear to disagree, record the observed
commit/run ID and update the relevant contract documentation before making a
scientific claim from a simulation result.
