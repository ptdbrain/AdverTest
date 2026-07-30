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

Phần còn lại của plan (19 corruption, thời tiết, patch, PGD/C&W/TOG, mPC/rPC/ASR/
RobustScore, review queue, Optuna red-team…) là **slot** đã có chỗ và có hợp đồng
rõ ràng — xem bảng ở cuối `docs/CONTRIBUTING_ATTACKS.md`.

## ⚡ Quick Start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # chỉ numpy + fastapi + dev tools
cp .env.example .env                     # điền AI_LOG_API_KEY của nhóm

make catalog                             # attack/model/dataset nào đang có, ai giữ
make demo                                # chạy thử một test run nhỏ
make test                                # pytest
make run                                 # API: http://localhost:8000/docs
```

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
pytest tests/test_attacks -q             # contract test tự bao phủ file mới
python -m src.cli run --attacks motion_blur --severities 1,3,5 --limit 4
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
│   │   ├── adversarial/    #    nhóm D — có fgsm
│   │   ├── patch/          #    nhóm E — slot
│   │   └── blackbox/       #    nhóm F — slot
│   ├── adapters/           # 🤖 model under test (M1–M6); blob_detector = model tham chiếu
│   ├── datasets/           # 🗂️ nguồn dữ liệu + cổng ẩn danh bắt buộc
│   ├── evaluation/         # 📏 IoU, AP, degradation, RunReport
│   ├── pipeline/           # ⚙️ TestRunner + cache + ước tính chi phí
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

## 🔌 API

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
| 3 | Architecture Diagram | `ARCHITECTURE.md`, `docs/architecture_diagram.md` |
| 4 | AI Logs | `.ai-log/` (hook tự động, submit khi `git push`) |
| 5 | Live URL | Dockerfile + CI đã sẵn |
| 6–7 | Video + Pitch Deck | `presentation/` |
| 8–9 | Journal + Worklog | `JOURNAL.md`, `WORKLOG.md` |
| 10 | Evaluation Evidence | `eval/` |

## 📊 AI Usage Logging

Hook đã cấu hình sẵn cho Claude Code, Cursor, Codex, Gemini CLI, Copilot,
Antigravity. Mọi prompt/tool call ghi vào `.ai-log/session.jsonl` và tự submit lên
grading server mỗi lần `git push`.

```bash
bash scripts/setup_hooks.sh   # chạy một lần sau khi clone
```

Log thủ công cho ChatGPT/web tool:

```bash
bash scripts/_pyrun.sh scripts/log_manual.py --tool chatgpt --prompt "What you asked"
```

> ⚠️ Đừng sửa/xoá file trong `.ai-log/`, đừng `git push --no-verify`.

## 📄 License

MIT — dùng cho mục đích giáo dục.
