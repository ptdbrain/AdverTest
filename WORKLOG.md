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
| phong | KITTI 2D loader + anonymization gate | Done | `src/datasets/kitti.py` now requires a real anonymized export; placeholder loader and destructive fetch script were not merged | - |
| phong | YOLO11 adapter (M1) | Done | `src/adapters/yolo11.py`, lazy ultralytics import, COCO->KITTI label map | - |
| phong | Robustness metrics + benchmark | Done | `src/evaluation/robustness_metrics.py` (mPC/rPC/RR/RA/RobustScore), `scripts/benchmark_kitti_yolo11.py` | - |

Summary: Group C is complete for the image modality. The final integration reuses
the real anonymization pipeline and the differentiable YOLO11 adapter from DE.

## [2026-08-01]

| Member | Task | Status | Output | Time |
|--------|------|--------|--------|------|
| phong | Full group C benchmark on real KITTI | Reference | 500 val frames were benchmarked on the source branch; rerun is required after replacing the placeholder anonymizer and adapter | - |

Results (YOLO11s COCO-pretrained, KITTI val 500, moderate difficulty, AP50):
`AP_clean = 0.2451`, `mPC = 0.187`, `rPC = 76.3 %`, RobustScore 83.5/100 normalised.
Severity-5 degradation: `object_occlusion` 100 % > `frame_freeze` 38 % >
`random_erasing` 31 % > `sensor_fault` 25 % > `gaussian_noise` (control) 16 %.
All four group C attacks pass sanity check #2; `gaussian_noise` shows a
non-monotonic blip at s1→s2 that sits inside its own bootstrap CI.

The source-branch result used a placeholder anonymizer and is retained only as
historical evidence. The integrated benchmark must use the detector-based
anonymizer from `src/anonymization/` and report its checkpoint hashes.

⚠️ Finding for anyone else using the GPU box: **do not enable FP16 for YOLO11
inference** (plan §5 suggests it). On the GTX 1660 Ti with torch 2.13+cu130 and
ultralytics 8.4.113, half precision returns wrong detections instead of failing —
the same image repeated N times gives 5 boxes in FP32 but 0 boxes at batch 2/4/8
and 9 boxes at batch 5. It would silently corrupt every robustness number.
`src/adapters/yolo11.py` therefore defaults to FP32 and makes `half=True` opt-in.
