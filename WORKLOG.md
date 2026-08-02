# Worklog — Team [Tên Team]

> Ghi lại tất cả công việc đã làm theo ngày. Ai làm gì, kết quả gì.

---

## [YYYY-MM-DD]

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| [Tên] | [mô tả task] | ✅ Done | [link/kết quả] | 2h |
| [Tên] | [mô tả task] | 🔄 WIP | [mô tả tiến độ] | 1.5h |
| [Tên] | [mô tả task] | ❌ Blocked | [lý do block] | - |

**Tổng kết ngày:** [1-2 câu về tiến độ chung]

---

## [YYYY-MM-DD]

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| | | | | |

**Tổng kết ngày:**

---

<!-- Format: copy block trên cho mỗi ngày làm việc -->
## [2026-07-26]

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| Phan Trong Dat | Verify team repo setup and AI Usage Logging | Done | AI Log smoke test accepted with HTTP 202; local tests and lint passed | - |

Summary: Restored the checkout to the team repository and verified the local AI logging submission flow.
## [2026-07-27]

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| Phan Trong Dat | Implement Chapter 4 LangGraph foundation | Done | Added state routing, calculation/search tools, graceful tool errors, and agent tests | - |

Summary: Replaced the placeholder graph with a deterministic, testable multi-route LangGraph flow.

## [2026-07-31]

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| phong | Group C attacks (occlusion & sensor faults) | Done | `random_erasing`, `object_occlusion`, `sensor_fault`, `frame_freeze` + tests; LiDAR/multi-cam rows documented as core-blocked in `docs/groupC-sensor-slots.md` | - |
| phong | KITTI 2D loader + placeholder anonymiser | Done | `src/datasets/kitti.py`, `src/datasets/kitti_anonymize.py`, `scripts/fetch_kitti.sh` | - |
| phong | YOLO11 adapter (M1) | Done | `src/adapters/yolo11.py`, lazy ultralytics import, COCO->KITTI label map | - |
| phong | Robustness metrics + benchmark | Done | `src/evaluation/robustness_metrics.py` (mPC/rPC/RR/RA/RobustScore), `scripts/benchmark_kitti_yolo11.py`, `scripts/train_yolo11_kitti.py` | - |

Summary: Group C is complete for the image modality and the KITTI/YOLO11 benchmark
runs end to end. Pending: download KITTI (`make kitti-data`) and produce the real
`eval/results/kitti_yolo11_groupC.*` artifacts; gradients on the YOLO11 adapter are
the next slot, which is what unblocks groups D and E on a real model.

## [2026-08-01]

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| phong | Full group C benchmark on real KITTI | Done | `eval/results/kitti_yolo11_groupC.{json,md}` + `groupC_examples.png`; 500 val frames, 25 cells, 13k forward passes in 381s | - |

Results (YOLO11s COCO-pretrained, KITTI val 500, moderate difficulty, AP50):
`AP_clean = 0.2451`, `mPC = 0.187`, `rPC = 76.3 %`, RobustScore 83.5/100 normalised.
Severity-5 degradation: `object_occlusion` 100 % > `frame_freeze` 38 % >
`random_erasing` 31 % > `sensor_fault` 25 % > `gaussian_noise` (control) 16 %.
All four group C attacks pass sanity check #2; `gaussian_noise` shows a
non-monotonic blip at s1→s2 that sits inside its own bootstrap CI.

⚠️ **The placeholder anonymiser is now the single largest source of AP loss.**
It costs ΔAP 0.147 (0.392 raw → 0.245 anonymised, i.e. 37 % of clean AP), which is
worse than three of the four attacks at severity 5. Cause: it mosaics the bottom
40 % of every `Car` box as a "plate band", and KITTI cars are small enough that
this erases the object. Fix is either a smaller `plate_fraction` or the real
detector-based anonymiser of plan §6 — until then, remember the benchmark baseline
is the anonymised one.

⚠️ Finding for anyone else using the GPU box: **do not enable FP16 for YOLO11
inference** (plan §5 suggests it). On the GTX 1660 Ti with torch 2.13+cu130 and
ultralytics 8.4.113, half precision returns wrong detections instead of failing —
the same image repeated N times gives 5 boxes in FP32 but 0 boxes at batch 2/4/8
and 9 boxes at batch 5. It would silently corrupt every robustness number.
`src/adapters/yolo11.py` therefore defaults to FP32 and makes `half=True` opt-in.
