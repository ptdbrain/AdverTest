# KẾ HOẠCH HỢP NHẤT CỰC KỲ CHI TIẾT CHO ADVERTEST
## Defense, Retraining, Attack Composition và WebApp cho YOLO11 + SAM2

> **Trạng thái tài liệu:** Kế hoạch triển khai hợp nhất, dùng làm nguồn tham chiếu chính cho phát triển, phân công, kiểm thử và demo.  
> **Phạm vi giai đoạn hiện tại:** hoàn thiện đầy đủ vòng lặp cho **YOLO11s 2D Detection** và **SAM2.1 Hiera Small Segmentation**.  
> **Nguyên tắc hợp nhất:** khi các bản cũ chồng chéo hoặc mâu thuẫn, ưu tiên phương án ngắn gọn và mới hơn; chỉ giữ phần chi tiết bổ sung không làm thay đổi quyết định đã chốt.

---

## Mục lục

1. [Quy tắc hợp nhất và quyết định cuối cùng](#1-quy-tắc-hợp-nhất-và-quyết-định-cuối-cùng)
2. [Mục tiêu sản phẩm và câu chuyện xuyên suốt](#2-mục-tiêu-sản-phẩm-và-câu-chuyện-xuyên-suốt)
3. [Phạm vi, ngoài phạm vi và giả định nguồn lực](#3-phạm-vi-ngoài-phạm-vi-và-giả-định-nguồn-lực)
4. [Kiến trúc vòng lặp khép kín](#4-kiến-trúc-vòng-lặp-khép-kín)
5. [Quy tắc dữ liệu và Benchmark Protocol](#5-quy-tắc-dữ-liệu-và-benchmark-protocol)
6. [Hệ thống metric và nguyên tắc so sánh](#6-hệ-thống-metric-và-nguyên-tắc-so-sánh)
7. [Attack Catalog và phân loại attack](#7-attack-catalog-và-phân-loại-attack)
8. [Attack Recipe, composition và randomization](#8-attack-recipe-composition-và-randomization)
9. [Pipeline sinh dữ liệu attack và provenance](#9-pipeline-sinh-dữ-liệu-attack-và-provenance)
10. [Chiến lược defense và retraining tổng thể](#10-chiến-lược-defense-và-retraining-tổng-thể)
11. [Kế hoạch dữ liệu và huấn luyện YOLO11](#11-kế-hoạch-dữ-liệu-và-huấn-luyện-yolo11)
12. [Kế hoạch dữ liệu và huấn luyện SAM2](#12-kế-hoạch-dữ-liệu-và-huấn-luyện-sam2)
13. [Training Dataset Builder dùng chung](#13-training-dataset-builder-dùng-chung)
14. [Trainer architecture và orchestration](#14-trainer-architecture-và-orchestration)
15. [Thiết kế WebApp cuối cùng](#15-thiết-kế-webapp-cuối-cùng)
16. [Data model và quan hệ lineage](#16-data-model-và-quan-hệ-lineage)
17. [API cần bổ sung](#17-api-cần-bổ-sung)
18. [Thay đổi code theo module](#18-thay-đổi-code-theo-module)
19. [Các lỗi nền tảng phải sửa trước](#19-các-lỗi-nền-tảng-phải-sửa-trước)
20. [Chiến lược kiểm thử](#20-chiến-lược-kiểm-thử)
21. [Phân công nhóm bốn người](#21-phân-công-nhóm-bốn-người)
22. [Lộ trình triển khai tám tuần](#22-lộ-trình-triển-khai-tám-tuần)
23. [Kịch bản demo](#23-kịch-bản-demo)
24. [Ưu tiên, rủi ro và Definition of Done](#24-ưu-tiên-rủi-ro-và-definition-of-done)
25. [Phụ lục: kiến trúc mở rộng 3D sau giai đoạn hiện tại](#25-phụ-lục-kiến-trúc-mở-rộng-3d-sau-giai-đoạn-hiện-tại)

---

# 1. Quy tắc hợp nhất và quyết định cuối cùng

## 1.1 Quy tắc ưu tiên khi nội dung chồng chéo

1. **Phạm vi mới nhất thắng phạm vi cũ:** giai đoạn hiện tại chỉ triển khai YOLO11 và SAM2 end-to-end; PointPillars, BEVFusion và toàn bộ 3D được chuyển sang backlog sau.
2. **Bản rút gọn làm xương sống:** các nguyên tắc, metric, workflow và cách kể chuyện được trình bày một lần theo phiên bản cô đọng nhất.
3. **Bản chi tiết chỉ bổ sung phần còn thiếu:** data model, API, module code, job states, kiểm thử, demo, phân công và các ràng buộc khoa học được đưa vào đúng chương tương ứng.
4. **Không giữ hai phương án UI song song:** chọn bố cục **hai vùng** gồm panel cấu hình bên trái và workspace trung tâm; loại bỏ panel metric hẹp bên phải.
5. **Không duy trì hai roadmap song song:** dùng roadmap tám tuần dành cho YOLO + SAM; công việc 3D không nằm trong sprint hiện tại.
6. **Không lặp lại cùng nội dung ở nhiều nơi:** một khái niệm chỉ có một định nghĩa chính; các chương khác dẫn chiếu thay vì mô tả lại.
7. **Không biến đề xuất thành cam kết khoa học nếu chưa có dữ liệu:** mọi ngưỡng, tỷ lệ và cấu hình huấn luyện là giá trị khởi điểm, phải được version hóa và kiểm chứng bằng ablation/benchmark.

## 1.2 Quyết định kỹ thuật cuối cùng

| Hạng mục | Quyết định |
|---|---|
| 2D detection | YOLO11s |
| Segmentation | SAM2.1 Hiera Small |
| Ground truth YOLO | Bounding box thật |
| Ground truth SAM | Mask thật đã được kiểm tra |
| Dataset YOLO chính | KITTI Object hoặc dataset detection nội bộ |
| Dataset YOLO external | BDD100K detection, chỉ trên class intersection |
| Dataset SAM chính | Cityscapes hoặc dataset nội bộ có mask thật |
| Dataset SAM external | BDD100K segmentation hoặc chiều ngược lại |
| Version model | Baseline clean → Robust mix → Targeted repair khi cần |
| Benchmark | Locked, versioned, deterministic, dùng lại nguyên vẹn cho model trước/sau defense |
| UI | Hai vùng: Left Config Panel + Center Workspace |
| Report | Năm metric chính trên màn hình; metric nâng cao trong drawer/modal |
| 3D | Chỉ giữ data contract và extension point, chưa triển khai |
| Demo | Dùng benchmark/model đã chuẩn bị trước; chỉ chạy live một mini sample có seed |

---

# 2. Mục tiêu sản phẩm và câu chuyện xuyên suốt

## 2.1 Khoảng trống cần giải quyết

AdverTest hiện mô tả tốt cách tạo attack, làm model thất bại, đánh giá và review. Tuy nhiên, sản phẩm chưa chứng minh đầy đủ rằng kết quả robustness test giúp đội phát triển **cải thiện model** như thế nào.

Sản phẩm phải chuyển từ một công cụ “tấn công và báo lỗi” thành một hệ thống “crash test → sửa chữa → kiểm định lại”.

## 2.2 Tuyên bố sản phẩm

> AdverTest tạo các bài crash test có kiểm soát cho perception model, phát hiện model yếu ở đâu, xây dựng defense data từ training split, robust fine-tune model, rồi chạy lại đúng benchmark đã khóa để chứng minh mức cải thiện và đánh đổi.

## 2.3 Câu chuyện dành cho người không chuyên

```text
Model hoạt động tốt trong điều kiện bình thường
→ AdverTest mô phỏng tình huống xấu có kiểm soát
→ Model thất bại ở một nhóm tình huống cụ thể
→ Hệ thống phân tích mẫu lỗi
→ Nhóm tạo dữ liệu phòng thủ phù hợp
→ Model được fine-tune
→ Phiên bản mới chạy lại đúng bài kiểm tra cũ
→ Website cho biết model phục hồi bao nhiêu và phải đánh đổi gì
```

## 2.4 Nguyên tắc truyền thông

- Không nói “tấn công model để phá model”.
- Nói “kiểm thử giới hạn an toàn và khả năng chịu lỗi”.
- Không chỉ hiển thị một con số tổng hợp.
- Luôn trả lời năm câu hỏi:
  1. Model ban đầu tốt đến đâu?
  2. Tình huống nào làm model yếu đi?
  3. Object/class nào bị ảnh hưởng nhiều nhất?
  4. Defense đã thay đổi dữ liệu và model như thế nào?
  5. Model mới phục hồi bao nhiêu, còn rủi ro gì và clean performance có giảm không?

---

# 3. Phạm vi, ngoài phạm vi và giả định nguồn lực

## 3.1 Phạm vi bắt buộc

### Chế độ 1 — 2D Object Detection

| Thuộc tính | Giá trị |
|---|---|
| Model | YOLO11s |
| Task | 2D object detection |
| Ground truth | Bounding box |
| Prediction | Bounding box, class, confidence |
| Metric chính | mAP@[.50:.95], ASR, degradation, RobustScore |
| Viewer | Image + GT box + predicted box + object status |

### Chế độ 2 — Object Segmentation

| Thuộc tính | Giá trị |
|---|---|
| Model | SAM2.1 Hiera Small |
| Task | Prompt-based instance segmentation |
| Ground truth | Mask thật |
| Prompt chính | Ground-truth box prompt |
| Prediction | Mask, mask score, boundary |
| Metric chính | mIoU, Boundary IoU, mask failure rate |
| Viewer | Image + GT mask + predicted mask + boundary overlay |

## 3.2 Vòng lặp bắt buộc cho mỗi chế độ

```text
Dữ liệu sạch
→ Baseline model
→ Chọn attack/recipe
→ Sinh attacked data
→ Chạy model trên clean input
→ Chạy model trên attacked input
→ Tính metric và failure cases
→ Phân tích failure cluster
→ Tạo defense profile
→ Tạo defense training dataset từ training split
→ Robust fine-tune
→ Đăng ký ModelVersion mới
→ Chạy lại cùng locked benchmark
→ So sánh baseline và defended model
→ Xuất Recovery Report
```

## 3.3 Ngoài phạm vi giai đoạn hiện tại

- PointPillars.
- BEVFusion.
- LiDAR corruption.
- 3D mAP, NDS và error components.
- BEV point-cloud viewer.
- Multi-sensor camera–LiDAR composition.
- Web-triggered full training trên hạ tầng nhiều GPU.
- Auto Red-Team Search đầy đủ.
- Failure clustering hoàn toàn tự động.
- CI robustness gate cho production.
- Active learning/active sampling quy mô lớn.

## 3.4 Cách giữ khả năng mở rộng 3D mà không làm tăng phạm vi

- `modality` và `task` phải là trường bắt buộc trong dataset/model/attack/metric.
- `Prediction` không được khóa chỉ cho bounding box 2D.
- UI selector có thể giữ enum 3D ở trạng thái `coming_later`, nhưng không cho chạy.
- Attack compatibility hỗ trợ `image`, `segmentation`, `lidar`, `fusion`.
- API và database không hard-code chỉ YOLO/SAM.
- Không xây loader, trainer, evaluator hoặc viewer 3D trong sprint hiện tại.

## 3.5 Giả định nguồn lực

- Nhóm: bốn người.
- Một người phụ trách frontend toàn thời gian.
- Ba người còn lại chia attack/data, model/training, backend/evaluation.
- GPU tham chiếu: khoảng 24 GB VRAM.
- Thời gian: khoảng tám tuần.
- YOLO hoàn thiện trước và là demo chính.
- SAM triển khai sau khi có dataset mask và adapter chạy thật.
- Full benchmark được chạy trước ngày demo.


# 4. Kiến trúc vòng lặp khép kín

## 4.1 Luồng nghiệp vụ chính

```text
┌─────────────────────────┐
│ Clean ModelVersion v1   │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ Locked BenchmarkProtocol│
│ Clean + Attack Recipes  │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ BenchmarkRun            │
│ Metrics + Failure Cases │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ Failure Cluster + Review│
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ DefenseProfile          │
│ TrainingDatasetManifest │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ TrainingRun             │
│ Robust ModelVersion v2  │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ Re-run same Benchmark   │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│ ModelComparison         │
│ RecoveryReport          │
└─────────────────────────┘
```

## 4.2 Ba giai đoạn kiểm thử và cải thiện

### Giai đoạn A — Single-method stress testing

Mục tiêu:

- Chạy từng attack độc lập.
- Quét severity 1–5.
- Tạo vulnerability profile.
- Xác định failure threshold.
- Tìm class/object size bị ảnh hưởng mạnh.
- Xác định attack nào cần ưu tiên defense.

Đầu ra:

- Per-attack metric table.
- RA(s) curve.
- Failure cases.
- Failure cluster sơ bộ.
- Danh sách critical scenarios.

### Giai đoạn B — Defense và robust fine-tuning

Mục tiêu:

- Tạo defense data từ training split.
- Giữ clean replay.
- Fine-tune model baseline thành robust model.
- Chọn checkpoint theo robustness có ràng buộc clean performance.
- Đăng ký model lineage.

Đầu ra:

- DefenseProfile.
- TrainingDatasetManifest.
- TrainingRun.
- ModelVersion robust.
- Training report.
- Checkpoint selection report.

### Giai đoạn C — Composite scenario và retest

Mục tiêu:

- Chạy ordered recipes phản ánh tình huống thực tế.
- Dùng cùng benchmark cho baseline và robust model.
- Đo recovery, clean tradeoff và external generalization.
- Xác nhận defense có cải thiện tổng quát hay chỉ overfit một attack.

Đầu ra:

- Paired ModelComparison.
- RecoveryReport.
- Worst-case scenario list.
- Residual risk list.
- Quyết định accept/retrain/reject.

## 4.3 Nguyên tắc không train một model riêng cho từng attack

Không tạo hàng loạt model như:

```text
YOLO-fog
YOLO-rain
YOLO-blur
YOLO-PGD
YOLO-patch
```

Mỗi kiến trúc chỉ có tối đa ba lớp version:

1. **B0 — Clean baseline:** dữ liệu sạch + augmentation tiêu chuẩn.
2. **R1 — Robust mix:** hỗn hợp clean, corruption, occlusion, fast adversarial và patch.
3. **R2 — Targeted repair:** chỉ tạo khi R1 vẫn còn một failure cluster ổn định và quan trọng.

Lợi ích:

- Giảm số checkpoint.
- Dễ quản lý lineage.
- Tránh tối ưu một attack làm hỏng attack khác.
- Giảm chi phí train.
- Dễ trình bày câu chuyện baseline → robust → targeted.
- Cho phép đánh giá clean-to-robust tradeoff nhất quán.

---

# 5. Quy tắc dữ liệu và Benchmark Protocol

## 5.1 Quy tắc chống leakage

Tuyệt đối không đưa ảnh từ locked test benchmark vào training.

Khi benchmark phát hiện pattern:

```text
Pedestrian nhỏ/xa + Fog severity 4 → thường bị miss
```

Defense data phải được tạo bằng cách:

```text
Lọc pedestrian nhỏ/xa từ training split
→ sinh fog severity 2–4 bằng seed mới
→ trộn với clean replay
→ fine-tune
```

Không được:

- Sao chép ảnh failure từ locked test vào training.
- Dùng attacked artifact giống hệt cho cả training và benchmark.
- Tuning checkpoint bằng external test.
- Thay đổi locked benchmark sau khi đã xem kết quả model robust, trừ khi tạo BenchmarkProtocol version mới.
- Dùng pseudo-label chưa review làm GT validation/test.

## 5.2 Dataset split policy

Mặc định, nếu dự án chưa có split chuẩn:

| Split | Tỷ lệ tham chiếu | Mục đích |
|---|---:|---|
| Train | 70% | Baseline training và defense data generation |
| Validation | 15% | Chọn checkpoint và early stopping |
| Locked test | 15% | Benchmark cuối, không dùng train |
| External | Dataset khác | Generalization, không dùng chọn checkpoint |

Nếu dataset đã có official split, giữ official split nhưng vẫn phải tạo:

- Training manifest.
- Validation manifest.
- Locked robustness test manifest.
- External mapping manifest.

## 5.3 BenchmarkProtocol

Mỗi benchmark phải là một thực thể bất biến hoặc versioned, gồm:

```text
benchmark_protocol_id
name
task
modality
dataset_id
dataset_version
sample_ids
sample_hashes
ground_truth_hashes
attack_recipe_ids
attack_implementation_versions
severity_values
parameter_ranges
global_seed
per-recipe seeds
model_preprocessing_version
input_resolution
confidence_threshold
iou_threshold
mask_threshold
metric_implementation_versions
bootstrap_iterations
bootstrap_seed
environment_image
framework_versions
created_by
created_at
status
```

## 5.4 Điều kiện so sánh hợp lệ

Hai model chỉ được so sánh trực tiếp khi:

- Cùng `BenchmarkProtocol`.
- Cùng dataset version và sample IDs.
- Cùng attacked artifacts hoặc cùng deterministic recipe.
- Cùng preprocessing.
- Cùng thresholds.
- Cùng metric implementation version.
- Cùng prompt protocol đối với SAM.
- Cùng class mapping.
- Có paired sample-level results.
- Có thông tin CI/uncertainty tương thích.

Nếu một điều kiện khác nhau, UI phải hiển thị cảnh báo:

> Kết quả không phải paired comparison hoàn toàn; chỉ dùng để tham khảo.

## 5.5 Determinism và reproducibility

Mỗi run phải lưu:

- Seed toàn run.
- Seed từng recipe.
- Seed từng attack step.
- Dataset version.
- Source sample hash.
- Attack implementation version.
- Model version/checkpoint hash.
- Git commit.
- Container/environment version.
- Framework/library versions.
- Hardware.
- Metric version.

Cùng:

```text
dataset version
+ sample IDs
+ recipe
+ implementation versions
+ seed
```

phải tạo lại cùng attacked variant trong phạm vi sai số số học cho phép.

---

# 6. Hệ thống metric và nguyên tắc so sánh

## 6.1 Nguyên tắc chung

- Năm metric chính được hiển thị trực tiếp.
- Metric nâng cao nằm trong `Xem thêm chỉ số`.
- Mọi metric phải ghi rõ đơn vị: ratio, percent, point hoặc percentage point.
- Backend trả cả ratio và percent nếu dễ nhầm.
- Mọi so sánh model phải có absolute delta.
- Khi phù hợp, thêm relative change và bootstrap 95% CI.
- Không kết luận model tốt hơn chỉ dựa trên một metric tổng hợp.
- Luôn kiểm tra per-class, per-object-size và critical scenarios.

## 6.2 Metric cốt lõi cho YOLO

### 6.2.1 Clean Detection Score

Giá trị kỹ thuật:

```text
Clean mAP@[.50:.95]
```

Ý nghĩa: khả năng phát hiện trên dữ liệu sạch.

### 6.2.2 Score After Attack

Giá trị kỹ thuật:

```text
Attacked mAP@[.50:.95]
```

Ý nghĩa: hiệu năng sau attack/recipe.

### 6.2.3 Performance Lost

```math
Degradation(c,s) =
(CleanAP - AttackedAP(c,s)) / CleanAP × 100%
```

Quy ước:

- `degradation_ratio`: 0–1.
- `degradation_pct`: 0–100.
- Giảm 42% nghĩa là mất 42% hiệu năng tương đối, không nhất thiết giảm 42 điểm AP.

### 6.2.4 Objects Broken by Attack

Object-level ASR:

```text
Số object đúng trên clean nhưng miss/misclassified trên attacked
/
Số object đúng trên clean
```

Cần định nghĩa rõ match policy:

- Ground-truth object.
- Clean prediction matched với GT tại IoU threshold.
- Attacked prediction không còn match hoặc sai class.
- Có thể mở rộng trạng thái `confidence_collapsed` nếu confidence giảm dưới threshold.

### 6.2.5 Overall Robustness

`RobustScore` là điểm 0–100 tổng hợp theo nhóm attack.

Yêu cầu triển khai:

- Weight config phải versioned.
- Không hard-code weight trong frontend.
- Report phải hiển thị nhóm yếu nhất.
- Không dùng RobustScore thay thế AP/ASR.
- Khi thay đổi công thức phải tăng metric version.

## 6.3 Metric nâng cao cho YOLO

- AP50.
- AP75.
- Precision.
- Recall.
- F1.
- AP theo class: Car, Pedestrian, Cyclist.
- AP theo object size: small, medium, large.
- False-positive rate.
- Miss rate.
- Confidence drop.
- IoU distribution.
- Per-object transition matrix.
- Robustness Accuracy theo severity.
- Latency.
- Throughput.
- GPU usage.
- Bootstrap 95% CI.
- External dataset AP.
- Clean-to-robust tradeoff.

## 6.4 Robustness Accuracy theo severity

```math
RA(s) = mean_c Metric(c,s) / Metric_clean
```

Trong đó:

- `c` là attack hoặc scenario thuộc scope benchmark.
- `s` là severity.
- Với YOLO dùng AP.
- Với SAM dùng mIoU hoặc metric chính đã chốt.

RA(s) được hiển thị bằng đường cong severity 1–5 để:

- Quan sát monotonicity.
- Xác định failure threshold.
- So sánh baseline và robust model.
- Phát hiện defense chỉ hiệu quả ở severity thấp.

## 6.5 Metric cốt lõi cho SAM2

### 6.5.1 Clean Mask Accuracy

```text
Clean mIoU
```

### 6.5.2 Mask Accuracy After Attack

```text
Attacked mIoU
```

### 6.5.3 Mask Performance Lost

```math
mIoU Degradation =
(Clean mIoU - Attacked mIoU) / Clean mIoU × 100%
```

### 6.5.4 Boundary Accuracy

```text
Boundary IoU
```

Boundary metric bắt buộc vì mask có thể giữ IoU tương đối cao nhưng biên bị sai đáng kể.

### 6.5.5 Masks Broken by Attack

Một mask được tính là thất bại nếu thỏa ít nhất một điều kiện versioned:

- IoU dưới threshold.
- Mask mất hoàn toàn.
- Mask nhầm object.
- Mask bị split/merge nghiêm trọng.
- Boundary error vượt ngưỡng.
- Prediction không hợp lệ hoặc rỗng khi GT không rỗng.

## 6.6 Metric nâng cao cho SAM2

- Dice score.
- Pixel precision.
- Pixel recall.
- Boundary F-score.
- Per-class mIoU.
- Per-object IoU.
- IoU theo object size.
- Prompt consistency.
- Mask confidence drop.
- Mask area ratio.
- Split/merge error.
- RA(s).
- Latency.
- Throughput.
- GPU usage.
- Bootstrap 95% CI.
- External dataset metric.

## 6.7 Recovery Rate

```math
Recovery =
(DefendedAttacked - BaselineAttacked)
/
(BaselineClean - BaselineAttacked)
× 100%
```

Giải thích:

> Model mới lấy lại bao nhiêu phần hiệu năng mà attack đã làm mất ở model baseline.

Cần xử lý edge case:

- Nếu `BaselineClean == BaselineAttacked`, Recovery không xác định.
- Nếu defended metric vượt baseline clean, có thể >100%; report cần ghi rõ.
- Nếu defended attacked thấp hơn baseline attacked, Recovery âm.
- Với metric “càng thấp càng tốt” như failure rate, dùng công thức hoặc sign convention riêng đã version hóa.

## 6.8 Bảng so sánh chuẩn

| Metric | Baseline | Defended | Absolute Δ | Relative change | CI |
|---|---:|---:|---:|---:|---|
| Clean metric | 68.5 | 67.8 | -0.7 | -1.0% | 95% |
| Attacked metric | 39.2 | 55.8 | +16.6 | +42.3% | 95% |
| Degradation | 42.8% | 17.7% | -25.1 pp | -58.6% | 95% |
| Failure rate | 61% | 34% | -27 pp | -44.3% | 95% |
| RobustScore | 62 | 76 | +14 | +22.6% | versioned |

## 6.9 Gate mặc định chọn checkpoint

Checkpoint robust được chấp nhận khi đồng thời:

- Clean metric giảm không quá 2 điểm.
- RobustScore tăng ít nhất 8 điểm **hoặc** mean degradation giảm ít nhất 15% tương đối.
- Không có critical scenario giảm quá 3 điểm.
- Không tăng ASR ở critical scenario quá 5 percentage points.
- External metric không giảm quá 3 điểm.
- Paired bootstrap difference được báo cáo.
- Không tuyên bố “tốt hơn” nếu uncertainty chưa đủ rõ.


# 7. Attack Catalog và phân loại attack

## 7.1 Cách tổ chức dành cho người dùng

Catalog hiển thị theo tình huống thực tế trước, thuật ngữ kỹ thuật sau.

### Nhóm A — Môi trường và tầm nhìn

- Fog.
- Rain.
- Snow.
- Frost.
- Brightness shift.
- Contrast reduction.

### Nhóm B — Chất lượng camera và hình ảnh

- Gaussian noise.
- Shot noise.
- Impulse noise.
- Motion blur.
- Defocus blur.
- JPEG compression.
- Pixelation.
- Spatter khi implementation có sẵn.

### Nhóm C — Che khuất và vật thể lạ

- Random erasing.
- CutOut.
- Object-level occlusion.
- Adversarial patch.

### Nhóm D — Tấn công có chủ đích

- FGSM.
- PGD.
- C&W.
- Square Attack.
- Transfer attack.
- SAM-PGD cho SAM2.
- TOG/DAG chỉ đưa vào expert/backlog nếu implementation chưa ổn định.

## 7.2 Phân loại theo mức độ truy cập model

| Loại | Quyền truy cập | Ví dụ | Vai trò |
|---|---|---|---|
| Black-box | Không biết gradient/trọng số | Corruption, weather, occlusion, Square Attack | Trọng tâm tình huống thực tế |
| Gray-box | Biết một phần hoặc dùng surrogate | Transfer attack, physical patch EOT | Mô phỏng kẻ tấn công biết kiến trúc/phân phối |
| White-box | Có loss/gradient model mục tiêu | FGSM, PGD, C&W, SAM-PGD | Worst-case bound và hard-example generation |

## 7.3 Phân loại theo chi phí

### Cheap/online

- Noise.
- Blur.
- Brightness/contrast.
- JPEG/pixelation.
- Fog/rain đơn giản.
- Random erasing.
- Partial occlusion.

Dùng trực tiếp trong training với seed thay đổi.

### Expensive/offline

- Multi-step PGD.
- C&W.
- Square Attack với query budget lớn.
- Optimized patch.
- SAM-PGD mạnh.
- Attack cần surrogate phức tạp.

Sinh trước, cache vào Hard Example Bank và replay.

## 7.4 Attack metadata bắt buộc

```text
attack_name
display_name
plain_language_summary
technical_summary
real_world_scenario
why_test_this
model_failure_symptoms
severity_labels
severity_parameter_map
input_modality
required_annotations
required_model_capabilities
compatible_tasks
compatible_models
compatible_with
incompatible_with
recommended_presets
cost_class
estimated_runtime_class
defense_hint
reference
implementation_version
deterministic_capability
supports_online_generation
supports_offline_cache
```

## 7.5 Nội dung popover mẫu

```text
Fog — Sương mù

Điều gì xảy ra?
Độ tương phản giảm; vật thể xa hòa dần vào nền.

Tình huống thực tế:
Camera hoạt động trong sương mù, khói hoặc tầm nhìn kém.

Model thường thất bại thế nào?
YOLO có thể bỏ sót pedestrian/xe ở xa.
SAM có thể mất biên mask hoặc thiếu một phần object.

Severity 3 nghĩa là gì?
Mức sương trung bình; object xa suy giảm rõ.

Hỗ trợ:
YOLO11 và SAM2.

Chi phí:
Thấp.

Defense gợi ý:
Fine-tune bằng fog nhiều mức độ, giữ clean replay và ưu tiên object nhỏ/xa.
```

Nút `Chi tiết kỹ thuật` mở:

- Công thức.
- Parameter.
- Norm/epsilon nếu có.
- Implementation version.
- Paper/reference.
- Compatibility.
- Runtime estimate.
- Known limitations.

## 7.6 Compatibility matrix tối thiểu

| Attack | YOLO | SAM | Yêu cầu đặc biệt |
|---|---:|---:|---|
| Fog/Rain/Noise/Blur/JPEG | Có | Có | Image input |
| Random erasing/CutOut | Có | Có | Cập nhật box/mask nếu spatial |
| Object occlusion | Có | Có | GT object geometry |
| Adversarial patch | Có | Có điều kiện | Objective task-specific |
| FGSM/PGD YOLO | Có | Không | Detection loss + gradient |
| SAM-PGD | Không | Có | Segmentation loss + mask GT |
| C&W | Có điều kiện | Có điều kiện | Expensive, objective cụ thể |
| Square Attack | Có điều kiện | Có điều kiện | Query interface và budget |

## 7.7 Severity semantics

Severity số không đủ; mỗi attack phải có nhãn vật lý/dễ hiểu:

| Mức | Nhãn chung | Ý nghĩa |
|---:|---|---|
| 0 | No-op | Không thay đổi, dùng sanity check |
| 1 | Very mild | Gần điều kiện sạch |
| 2 | Mild | Suy giảm nhẹ |
| 3 | Moderate | Suy giảm rõ nhưng còn thực tế |
| 4 | Strong | Điều kiện khó, có thể gây failure |
| 5 | Extreme | Stress limit, cần cảnh báo realism |

Mapping parameter phải riêng cho từng attack. Severity 3 của fog không tương đương severity 3 của PGD.

---

# 8. Attack Recipe, composition và randomization

## 8.1 Data structure

```text
AttackRecipe
├── recipe_id
├── name
├── description
├── task
├── modality
├── mode
├── seed
├── scenario_intensity
├── sample_policy
├── repeat_count
├── constraints
├── catalog_version
└── ordered_steps[]
```

```text
AttackRecipeStep
├── position
├── attack_name
├── implementation_version
├── severity
├── parameters
├── probability
├── seed
├── objective
└── expected_cost
```

## 8.2 Các mode bắt buộc

### Single Attack

- Chọn một attack.
- Chọn một hoặc nhiều severity.
- Tùy chỉnh parameter.
- Có thể chạy auto sweep severity.

### Manual Composition

- Chọn nhiều attack.
- Kéo thả thứ tự.
- Chỉnh severity từng step.
- Xem preview.
- Validate trước khi enqueue.

Ví dụ:

```text
Fog severity 4
→ Object Occlusion 25%
→ JPEG quality 35
```

### Random N Attacks

- Chọn N attack tương thích.
- Sampling không hoàn lại mặc định.
- Chọn số recipe cần tạo.
- Lưu seed.
- Cho phép exclude/require attack.
- Có max recipe length.

### Random by Group

Ví dụ:

```text
2 weather
+ 1 camera quality
+ 1 occlusion
```

Hoặc:

```text
3 attack nhóm A
+ 3 attack nhóm B
```

Tên kỹ thuật `Stratified Random` dùng trong API; UI dùng `Random by Group`.

### Scenario Preset

Preset chuẩn giai đoạn hiện tại:

1. **Low Visibility**
   ```text
   Fog → Contrast reduction → Mild blur
   ```

2. **Wet Camera**
   ```text
   Rain → Spatter → Motion blur nhẹ
   ```

3. **Poor Camera Pipeline**
   ```text
   Gaussian/shot noise → JPEG compression → Mild blur
   ```

4. **Partial Obstruction**
   ```text
   Object occlusion → Brightness/contrast shift
   ```

5. **Adversarial Stress — YOLO**
   ```text
   FGSM hoặc PGD → Optional JPEG survival test
   ```

6. **Segmentation Boundary Stress — SAM**
   ```text
   Fog hoặc blur → Partial occlusion → Optional SAM-PGD
   ```

### Auto Sweep

- Chạy từng attack độc lập.
- Quét severity đã chọn.
- Không composition.
- Dùng lập vulnerability profile.

### Red-Team Search

- Để ở expert/backlog.
- Tối ưu type/parameter/order theo objective.
- Có GPU/query/storage budget.
- Không phải P0 của giai đoạn hiện tại.

## 8.3 Thứ tự composition

Composition không giao hoán. Mặc định:

```text
1. Environment/weather
2. Spatial occlusion/patch
3. Camera post-processing corruption
4. Model-dependent adversarial optimization
5. Optional survival transform
```

Ví dụ:

```text
Fog → JPEG
```

không giống:

```text
JPEG → Fog
```

Mọi manifest/report phải hiển thị ordered steps.

## 8.4 Ràng buộc bắt buộc

Hệ thống chặn hoặc cảnh báo:

- FGSM và PGD trong cùng recipe.
- PGD và C&W trong cùng recipe.
- Nhiều blur tương tự liên tiếp.
- Nhiều noise tương tự liên tiếp.
- Hơn một expensive attack.
- SAM-PGD với model không phải SAM.
- YOLO white-box attack khi adapter không có gradient.
- Attack yêu cầu mask khi dataset không có mask.
- Spatial attack mà annotation transform chưa hỗ trợ.
- Patch và occlusion che quá nhiều cùng object.
- Recipe vượt `max_recipe_length`.
- Biến thể vượt GPU/storage cap.
- Full Cartesian product ngoài Expert Mode.
- Attack implementation chưa đạt trạng thái production-ready.

## 8.5 Quy tắc random

- Lọc compatibility trước khi sampling.
- Không chọn trùng attack mặc định.
- Có quota theo group/family.
- Có allowlist/blocklist.
- Có attack bắt buộc.
- Tối đa một white-box attack.
- Tối đa một expensive attack.
- Lưu global seed và per-step seed.
- Cùng seed + catalog version phải tạo cùng recipe.
- Nếu catalog thay đổi, recipe hash phải thay đổi.
- Randomization phải trả preview trước khi run.

## 8.6 Scenario intensity

Không dùng intensity thay severity từng attack.

### Mild

- 1–2 attack.
- Severity 1–2.
- Không expensive attack.

### Moderate

- 2–3 attack.
- Severity trung bình 2–3.
- Tối đa một expensive attack.

### Severe

- 2–4 attack.
- Có tối đa một step severity 4–5.
- Các step khác cap 2–3.
- Hiển thị cảnh báo realism và chi phí.

## 8.7 Chống combinatorial explosion

Với 6 attack và 5 severity:

```text
5^6 = 15,625 tổ hợp severity
```

UI phải phân biệt:

- `Sample K random recipes`: mặc định.
- `Full combinatorial sweep`: Expert Mode, có estimate và hard cap.

Estimate phải hiển thị:

- Số variant.
- Số inference.
- Storage.
- GPU-hours.
- Thời gian tương đối.
- Số expensive calls.
- Query budget nếu black-box.


# 9. Pipeline sinh dữ liệu attack và provenance

## 9.1 Luồng xử lý

```text
Source DatasetVersion
→ Select sample manifest
→ AttackRecipe
→ Compatibility validation
→ Cost estimate
→ Generation job
→ Ordered attack execution
→ Annotation transformation
→ Output validation
→ GeneratedDatasetVersion
→ Multi-model inference
→ Evaluation
→ Provenance storage
→ Report
```

## 9.2 Từng bước xử lý một sample

1. Load source image và GT.
2. Kiểm tra source hash.
3. Khởi tạo recipe seed.
4. Với từng step:
   - Resolve implementation version.
   - Resolve severity → physical parameter.
   - Sinh per-step seed.
   - Apply attack.
   - Transform annotation nếu cần.
   - Validate output type/range.
   - Lưu intermediate hash.
5. Validate final GT.
6. Lưu attacked artifact.
7. Ghi manifest.
8. Chạy model version đã chọn.
9. Tính metric và object/mask status.

## 9.3 Manifest mỗi variant

```text
variant_id
source_dataset_id
source_dataset_version
source_sample_id
source_hash
ground_truth_hash_before
ground_truth_hash_after
recipe_id
recipe_hash
ordered_steps
global_seed
per_step_seeds
attack_implementation_versions
severity_values
resolved_parameters
surrogate_model_version
intermediate_hashes
output_hash
norm_linf
norm_l2
annotation_transform_log
generation_started_at
generation_finished_at
storage_uri
status
validation_errors
```

## 9.4 Annotation transformation

### YOLO

Phải cập nhật box khi:

- Crop.
- Translate.
- Scale.
- Rotate nếu implementation cho phép.
- Spatial occlusion làm object không còn hợp lệ theo policy.

Phải lưu:

- Original box.
- Transformed box.
- Visible ratio.
- Dropped/kept decision.
- Policy version.

### SAM

Phải cập nhật mask khi:

- Crop/translate/scale.
- Spatial warp.
- Object occlusion theo policy.
- Resize/pad.

Không được:

- Giữ mask cũ khi geometry đã thay đổi.
- Xóa toàn bộ object nhưng vẫn coi GT mask nguyên vẹn.
- Dùng attacked prediction làm GT.

## 9.5 Cache

Cache key tối thiểu:

```text
dataset_version
sample_hash
recipe_hash
ordered_steps
attack_implementation_versions
seed
surrogate_model_version nếu có
```

Không sinh lại attacked data nếu key giống nhau.

Model inference cache phải tách khỏi generation cache:

```text
generated_variant_hash
model_version
preprocessing_version
thresholds
```

## 9.6 GeneratedDatasetVersion

Mỗi generated dataset phải có:

- Parent DatasetVersion.
- Manifest hash.
- Recipe set.
- Sample count.
- Split designation.
- Total storage.
- Generated by job.
- Created at.
- Validation status.
- Anonymization status.
- Intended use: benchmark/training/hard-example.
- Leakage check result.

## 9.7 Error handling

Job phải dừng hoặc đánh dấu sample lỗi khi:

- Input/GT thiếu.
- Attack không tương thích.
- NaN/Inf.
- Output sai shape/range.
- Annotation rỗng bất hợp lý.
- Storage write lỗi.
- Seed/provenance thiếu.
- Budget vượt hard cap.
- Attack process timeout.
- Surrogate/model gradient không khả dụng.

---

# 10. Chiến lược defense và retraining tổng thể

## 10.1 DefenseProfile

`DefenseProfile` mô tả cách chuyển failure pattern thành dữ liệu huấn luyện.

```text
defense_profile_id
name
target_task
parent_model_version
source_failure_cluster_ids
target_classes
target_object_sizes
target_conditions
attack_families
severity_distribution
clean_replay_ratio
online_generation_policy
offline_hard_example_policy
data_caps
annotation_policy
training_hyperparameter_overrides
acceptance_gates
created_by
review_status
```

## 10.2 Robust mix mặc định

### YOLO

| Thành phần | Tỷ lệ batch khởi điểm |
|---|---:|
| Clean replay | 40% |
| Weather/common corruption | 30% |
| Occlusion | 15% |
| Fast adversarial | 10% |
| Patch/hard examples | 5% |

### SAM

| Thành phần | Tỷ lệ batch khởi điểm |
|---|---:|
| Clean image + GT mask | 45–50% |
| Weather/noise | 25% |
| Blur/compression | 10% |
| Occlusion | 15% |
| Fast SAM adversarial/cached hard masks | 5% |

Tỷ lệ là config versioned, không phải định luật cố định.

## 10.3 Curriculum learning

### Pha 1 — Ổn định clean representation

- Freeze phần lớn backbone/encoder.
- 60% clean.
- Corruption nhẹ.
- Occlusion nhẹ.
- Chưa dùng attack đắt.

### Pha 2 — Robust mix đầy đủ

- Unfreeze có kiểm soát.
- Dùng tỷ lệ robust mix chuẩn.
- Thêm fast adversarial.
- Theo dõi clean metric.

### Pha 3 — Hard-example replay

- Replay failure clusters.
- Tăng object nhỏ/biên mỏng/partial occlusion.
- Giữ ít nhất 35–40% clean.
- Không để severe samples chiếm đa số.

## 10.4 Online và offline attack

### Online

- Noise.
- Blur.
- Brightness/contrast.
- JPEG.
- Fog/rain đơn giản.
- Erasing.
- Partial occlusion.

### Offline cache

- Strong PGD.
- C&W.
- Square Attack.
- Optimized patch.
- SAM-PGD mạnh.
- Surrogate transfer examples.

## 10.5 Fast adversarial training

- Chỉ 5–10% batch.
- FGSM hoặc PGD 2–3 bước.
- Epsilon nhẹ/trung bình.
- Không chạy PGD 20–40 bước mọi batch.
- Mỗi vài epoch replay strong PGD cached.
- Full-strength attacks dùng benchmark và hard-example generation.

## 10.6 Targeted repair

Chỉ tạo R2 khi:

- Failure cluster lặp lại trên đủ mẫu.
- Có ý nghĩa an toàn/sản phẩm.
- R1 chưa giải quyết.
- Có training samples tương tự từ train split.
- Có acceptance gate riêng.
- Không gây regression lớn cho các scenario khác.

Ví dụ:

```text
Cluster:
Pedestrian nhỏ, xa, partial occlusion, fog severity 4.

R2 mix:
60% general robust replay
25% target cluster analogues từ training split
15% clean class-balanced replay
```

## 10.7 Review workflow

```text
FailureCase
→ Reviewer triage
→ Add to FailureCluster
→ Request retrain
→ Create RetrainingBacklogItem
→ Approve DefenseProfile
→ Estimate TrainingRun
→ Start TrainingRun
→ Register ModelVersion
→ Re-benchmark
→ Review RecoveryReport
→ Accept / request targeted repair / reject
```

## 10.8 Model lineage

Ví dụ:

```text
yolo11s-kitti-clean-v1
└── yolo11s-kitti-robust-mix-v2
    └── yolo11s-kitti-small-ped-fog-v3
```

Mỗi child version lưu:

- Parent model version.
- Training run.
- Training dataset manifest.
- Defense profile.
- Code commit.
- Hyperparameters.
- Checkpoint hash.
- Acceptance gate result.
- Benchmark comparisons.


# 11. Kế hoạch dữ liệu và huấn luyện YOLO11

## 11.1 Dataset protocol

### In-domain

Ưu tiên:

- KITTI Object.
- Hoặc dataset detection nội bộ đã anonymize.

Class chuẩn hóa:

- Car.
- Pedestrian.
- Cyclist.

### External

BDD100K detection với class mapping công bố rõ:

| BDD100K | AdverTest |
|---|---|
| car | Car |
| person/pedestrian | Pedestrian |
| bike/rider | Cyclist nếu rule được chốt |

Không dùng external test chọn checkpoint.

## 11.2 YOLO-B0 — Clean baseline

### Khởi tạo

```text
Model: YOLO11s
Checkpoint: pretrained YOLO11s
Task: detection
Input size: 640 tham chiếu
```

### Dữ liệu

- 100% từ training split.
- Bounding-box GT.
- Không strong attack.
- Standard augmentation:
  - Resize.
  - Horizontal flip hợp lý.
  - Scale.
  - Translate nhẹ.
  - Color augmentation nhẹ.
  - Mosaic/mixup có kiểm soát.

### Training config khởi điểm

```text
epochs: 30
optimizer: AdamW hoặc SGD theo framework baseline
AMP: enabled
early_stopping_patience: 5–10
checkpoint_metric: validation mAP@[.50:.95]
```

### Artifact bắt buộc

- Best checkpoint.
- Last checkpoint.
- Training config.
- Dataset version.
- Split manifest.
- Random seed.
- Framework version.
- Git commit.
- GPU-hours.
- Training curves.
- Checkpoint hash.

### Đánh giá

- Clean validation.
- Clean locked test.
- Locked attack benchmark.
- External BDD100K subset.

## 11.3 YOLO-R1 — Robust mix

### Khởi tạo

- Parent: YOLO-B0.
- Không train từ đầu.
- 15–25 epoch bổ sung tham chiếu.
- Freeze backbone 3–5 epoch đầu nếu GPU hạn chế.
- Sau đó unfreeze.

### Mix

| Dữ liệu | Tỷ lệ |
|---|---:|
| Clean | 40% |
| Weather/common corruption | 30% |
| Occlusion | 15% |
| Fast adversarial | 10% |
| Patch | 5% |

### Weather/common corruption

Sinh online, chủ yếu severity 1–3:

- Fog.
- Rain.
- Brightness.
- Contrast.
- Gaussian noise.
- Motion blur.
- JPEG.

Không để severity 5 chiếm phần lớn.

### Occlusion

- Random erasing.
- CutOut.
- Object-level occlusion.
- Che một phần object.
- Không xóa object vô lý nhưng vẫn giữ label.
- Cập nhật box/visible ratio.

### Fast adversarial

- FGSM hoặc PGD 2–3 bước.
- 5–10% batch.
- Epsilon nhẹ/trung bình.
- Theo dõi clean AP.

### Patch

- Patch đã tối ưu và cache.
- Random position hợp lệ.
- Scale/brightness augmentation.
- Không che toàn bộ object ở mọi sample.

### Checkpoint selection

- Early stop theo robust validation score.
- Clean AP regression ≤ 2 điểm.
- RobustScore tăng ≥ 8 hoặc attacked AP tăng rõ.
- Không làm critical scenario xấu đi.
- External AP không regression quá ngưỡng.

## 11.4 YOLO-R2 — Targeted repair

Chỉ chạy nếu YOLO-R1 còn failure cluster quan trọng.

Ví dụ mix:

| Thành phần | Tỷ lệ |
|---|---:|
| General robust replay | 60% |
| Target failure analogues | 25% |
| Clean class-balanced replay | 15% |

Target samples:

- Lấy từ training split.
- Match class/object size/condition.
- Sinh attack bằng seed mới.
- Không reuse locked-test artifact.

## 11.5 Detection prediction contract

```text
DetectionPrediction
├── sample_id
├── model_version_id
├── boxes_xyxy
├── labels
├── confidences
├── latency_ms
├── preprocessing_version
└── metadata
```

Per-object evaluation:

```text
object_id
gt_box
clean_match
attacked_match
clean_confidence
attacked_confidence
status_clean
status_attacked
iou_clean
iou_attacked
failure_reason
```

Status:

- Correct.
- Missed.
- False positive.
- Misclassified.
- Confidence collapsed.
- Localization degraded.

## 11.6 Definition of Done cho YOLO

- Có YOLO-B0 và YOLO-R1.
- Model lineage đầy đủ.
- Cả hai chạy cùng locked benchmark.
- Năm metric chính hiển thị đúng.
- Có per-class và object-size analysis.
- Có BDD100K external result.
- Có RecoveryReport.
- Gate được đánh giá tự động.
- Mỗi metric truy ngược được về model, dataset, recipe, sample và metric version.

---

# 12. Kế hoạch dữ liệu và huấn luyện SAM2

## 12.1 Điều kiện tiên quyết

Không dùng KITTI Object làm dataset segmentation chính vì không có GT mask phù hợp.

Phải có:

- Dataset mask thật.
- Runnable SAM2 adapter.
- Segmentation prediction contract.
- Prompt protocol cố định.
- Segmentation evaluator.
- Mask viewer.

## 12.2 Dataset

### Hướng ưu tiên

- Cityscapes.
- BDD100K segmentation.
- Dataset nội bộ có mask thật.

Class có thể giữ chi tiết:

- Person.
- Rider.
- Car.
- Truck.
- Bus.
- Bicycle.
- Motorcycle.

Hoặc map về nhóm kể chuyện chung:

- Vehicle.
- Pedestrian.
- Cyclist/rider.

### Dataset nội bộ

- SAM có thể tạo mask gợi ý.
- Người gán nhãn phải review/chỉnh.
- Chỉ mask đã review được dùng validation/test.
- Pseudo-mask chưa review chỉ được dùng training phụ và gắn nguồn.

## 12.3 Split SAM

- Training: official train.
- Validation: checkpoint selection.
- Locked robustness test: subset riêng hoặc phần val khóa.
- External test: dataset segmentation khác.
- Không sinh defense data từ locked test.

## 12.4 Runnable Sam2Adapter

Tách khỏi surrogate-only adapter.

Interface:

```python
predict_masks(samples, prompts)
loss_for_attack(sample, target)
input_gradient(sample, target)
metadata()
```

Output:

```text
SegmentationPrediction
├── sample_id
├── model_version_id
├── masks
├── mask_scores
├── prompt_type
├── prompt_coordinates
├── latency_ms
├── preprocessing_version
└── metadata
```

## 12.5 Prompt protocol

Benchmark chính:

- Dùng cùng ground-truth box prompt cho clean và attacked image.
- Prompt lưu trong benchmark manifest.
- Không dùng box do YOLO dự đoán.
- Không thay prompt giữa model baseline và robust.
- Đo đúng khả năng segmentation, không trộn lỗi detector.

Benchmark end-to-end tùy chọn sau:

```text
YOLO predicted box → SAM segmentation
```

Không dùng thay benchmark độc lập.

## 12.6 SAM-B0 — Clean baseline

### Khởi tạo

```text
Model: SAM2.1 Hiera Small
Checkpoint: official pretrained
```

### Fine-tuning

Pha đầu:

- Freeze image encoder.
- Fine-tune mask decoder.
- Fine-tune prompt-related modules.
- AMP.
- Gradient accumulation nếu thiếu VRAM.

Nếu chưa đủ:

- Mở một số block cuối encoder.
- Fine-tune thêm 5–10 epoch.
- Learning rate thấp hơn.

### Sample

- Clean image.
- GT instance mask.
- GT box prompt.
- Class.
- Object size.
- Occlusion metadata nếu có.

### Epoch tham chiếu

- Decoder-only: 10–20 epoch.
- Early stopping theo validation mIoU.
- Theo dõi Boundary IoU.

## 12.7 SAM-R1 — Robust mix

### Mix tham chiếu

| Dữ liệu | Tỷ lệ |
|---|---:|
| Clean image + mask | 45% |
| Weather/noise | 25% |
| Blur/compression | 10% |
| Occlusion | 15% |
| Fast adversarial/cached hard masks | 5% |

### Weather/noise

- Fog.
- Rain.
- Brightness.
- Contrast.
- Gaussian noise.

### Blur/compression

- Motion blur.
- Defocus blur.
- JPEG.
- Pixelation.

### Occlusion

- Partial object occlusion.
- Random erasing.
- Cập nhật mask/valid region.
- Không phá toàn bộ mask nhưng giữ annotation cũ.

### Adversarial

- Gradient attack nhẹ trên tỷ lệ nhỏ.
- Hoặc replay cached attacked samples.
- Full SAM-PGD chỉ dùng benchmark/hard-example.

### Curriculum

- Đầu: 60% clean, corruption nhẹ, occlusion nhẹ.
- Giữa: robust mix đầy đủ.
- Cuối: hard mask cases, small objects, thin boundaries, partial occlusion, một phần adversarial.

## 12.8 SAM-R2 — Targeted repair

Ví dụ:

```text
Cluster:
Object nhỏ trong fog severity 4
→ mask mất hoàn toàn hoặc thiếu biên.

Mix:
60% general robust replay
25% targeted small-object/fog samples
15% clean replay
```

## 12.9 Gate cho SAM

- Clean mIoU giảm ≤ 2 điểm.
- Attacked mIoU tăng rõ.
- Clean Boundary IoU không regression đáng kể.
- Mask failure rate giảm ≥ 15% tương đối.
- External metric không giảm quá ngưỡng.
- Same prompt + same benchmark.
- Không dùng pseudo-mask chưa review làm GT test.

## 12.10 Definition of Done cho SAM

- Có GT mask hợp lệ.
- Sam2Adapter runnable.
- Có SAM-B0 và SAM-R1.
- Viewer render clean/attacked/defended mask.
- Có mIoU, degradation, Boundary IoU, mask failure rate.
- Có external-domain evaluation.
- Có RecoveryReport.
- Có lineage và reproducible checkpoint.


# 13. Training Dataset Builder dùng chung

## 13.1 Mục tiêu

Một `TrainingDatasetBuilder` dùng chung cho detection và segmentation, chịu trách nhiệm tạo manifest defense data mà không gây leakage.

## 13.2 Input

```text
base_dataset_version
task
modality
training_split_manifest
defense_profile
attack_recipe_set
data_ratios
severity_distribution
global_seed
max_generated_samples
storage_budget
online_offline_policy
failure_cluster_ids
annotation_policy_version
```

## 13.3 Output

`TrainingDatasetManifest`:

```text
manifest_id
base_dataset_version
task
clean_sample_ids
generated_sample_ids
source_sample_ids
source_split
attack_recipe_ids
severity_values
seeds
ground_truth_hashes
annotation_transform_logs
online_or_offline
failure_cluster_links
train_validation_designation
class_distribution
object_size_distribution
storage_estimate
manifest_hash
leakage_check
validation_status
```

## 13.4 Quy tắc bắt buộc

- Không chứa sample locked test.
- Không reuse attacked artifact benchmark cho training.
- Không thay annotation mà thiếu transform log.
- Mask được transform đúng.
- Box được update đúng.
- Mọi dataset version có hash.
- Có class/object-size balance report.
- Có duplicate detection.
- Có source provenance.
- Có hard cap storage.
- Có dry-run estimate.
- Có deterministic build mode.

## 13.5 Sampling strategy

Builder hỗ trợ:

- Random sampling.
- Class-balanced sampling.
- Object-size-balanced sampling.
- Failure-cluster-targeted sampling.
- Severity distribution sampling.
- Clean replay floor.
- Maximum per-source-sample variants.
- Maximum per-recipe variants.
- Hard-example replay.

## 13.6 Validation

Trước training:

- No test leakage.
- Source files tồn tại.
- GT hợp lệ.
- Class mapping hợp lệ.
- Box/mask không rỗng bất thường.
- Ratio tổng bằng 100% trong sai số cho phép.
- Attack compatibility hợp lệ.
- Manifest hash reproducible.
- Storage trong budget.

---

# 14. Trainer architecture và orchestration

## 14.1 Interface

```python
class ModelTrainer:
    def validate_config(self, config): ...
    def estimate(self, config): ...
    def prepare_data(self, config): ...
    def train(self, config, callbacks): ...
    def evaluate_checkpoint(self, checkpoint): ...
    def export_checkpoint(self, checkpoint): ...
    def metadata(self): ...
```

Implement hiện tại:

```text
YoloTrainer
Sam2Trainer
```

Extension point sau:

```text
PointPillarsTrainer
```

## 14.2 Nguyên tắc

- Training logic không nằm trong API route.
- Trainer chỉ nhận config/versioned manifest.
- Mỗi phase phát event.
- Checkpoint được hash.
- Validation metric được lưu theo epoch.
- Có cancel.
- Có budget guard.
- Có resume nếu framework hỗ trợ.
- Registration model chỉ xảy ra sau export validation.

## 14.3 TrainingRun states

```text
DRAFT
VALIDATING
ESTIMATING
QUEUED
PREPARING_DATA
TRAINING
VALIDATING_CHECKPOINT
EXPORTING
REGISTERING_MODEL
COMPLETED
FAILED
CANCELLED
BUDGET_EXCEEDED
```

State transition phải được kiểm thử và audit.

## 14.4 Estimate trước khi train

UI/API trả:

- Clean sample count.
- Online attacked sample estimate.
- Offline hard-example count.
- Epoch.
- Batch size.
- Gradient accumulation.
- Trainable parameter count.
- Estimated GPU-hours.
- Estimated storage.
- Expected checkpoint count.
- Hard cap.
- Assumption/warning.

## 14.5 Checkpoint management

Mỗi checkpoint lưu:

```text
checkpoint_id
training_run_id
epoch
step
metric_snapshot
clean_metric
robust_metric
critical_scenario_metrics
file_hash
storage_uri
framework_version
is_best_clean
is_best_robust
gate_status
```

## 14.6 Model registration

Chỉ đăng ký ModelVersion khi:

- Export load test thành công.
- Metadata đầy đủ.
- Checkpoint hash khớp.
- Parent lineage hợp lệ.
- TrainingRun completed.
- Gate result được lưu.
- Model catalog compatibility được cập nhật.

---

# 15. Thiết kế WebApp cuối cùng

## 15.1 Bố cục tổng thể

Chọn một bố cục duy nhất:

```text
┌──────────────────────────┬─────────────────────────────────────────────┐
│ LEFT CONFIG PANEL        │ CENTER WORKSPACE                            │
│                          │                                             │
│ - Perception mode        │ 1. Input + Ground Truth                     │
│ - Dataset                │ 2. Attacked Input                           │
│ - Model/version          │ 3. Clean Model Prediction                   │
│ - Attack/recipe          │ 4. Attacked Model Prediction                │
│ - Severity/intensity     │ 5. Five Key Metrics / Comparison            │
│ - Estimate               │                                             │
│ - Run                    │ Advanced drawer + evidence + lineage         │
└──────────────────────────┴─────────────────────────────────────────────┘
```

Không dùng panel metric hẹp bên phải.

## 15.2 Left Config Panel

### A. Perception Mode

```text
[2D Object Detection — YOLO11]
[Object Segmentation — SAM2]
```

Khi đổi mode, tự lọc:

- Dataset.
- Model version.
- Attack.
- Metric.
- Ground truth type.
- Prediction overlay.
- Prompt controls.

### B. Dataset

Hiển thị:

- Tên.
- Version.
- Số ảnh.
- Annotation type.
- Classes.
- Split.
- Anonymization.
- Internal/external.
- Locked status.

Chặn:

- Dataset chưa anonymize khi bắt buộc.
- SAM + dataset không mask.
- YOLO + dataset không box.
- Benchmark dataset dùng nhầm làm training source.

### C. Model và version

Ví dụ:

```text
YOLO11s
○ yolo11s-kitti-clean-v1
○ yolo11s-kitti-robust-v2
○ yolo11s-kitti-targeted-v3
```

```text
SAM2.1 Hiera Small
○ sam2-cityscapes-clean-v1
○ sam2-cityscapes-robust-v2
```

Loại bỏ hard-code `blob_detector`.

### D. Attack Catalog

- Group theo tình huống.
- Compatibility badge.
- Cost badge.
- White/gray/black-box badge.
- Info popover.
- `Xem chi tiết kỹ thuật`.

### E. Attack Mode

```text
Single Attack
Manual Composition
Random N Attacks
Random by Group
Scenario Preset
Auto Sweep
```

### F. Estimate

- Variants.
- Storage.
- Inference count.
- GPU estimate.
- Expensive attacks.
- Warning.
- Hard cap.

## 15.3 Center Workspace: năm phần

### Phần 1 — Input + Ground Truth

YOLO:

- Original image.
- GT boxes.
- Class.
- Object ID.
- difficult/occluded nếu có.
- Toggle GT.

SAM:

- Original image.
- GT mask.
- Box/point prompt.
- Object ID.
- Boundary.
- Opacity.
- Nhãn rõ `Ground Truth`.

### Phần 2 — Attacked Input

- Final attacked image.
- Recipe name.
- Ordered steps.
- Severity từng step.
- Seed.
- Parameters.
- Intermediate step viewer.

### Phần 3 — Clean Model Prediction

YOLO:

- Predicted boxes.
- Class/confidence.
- GT match.
- Correct/missed/FP/misclassified.

SAM:

- Predicted mask clean.
- Mask score.
- Boundary.
- Prompt.
- GT overlap.

### Phần 4 — Attacked Model Prediction

YOLO:

- Boxes sau attack.
- Confidence change.
- Lost object.
- Misclassification.
- New false positive.

Status encoding:

- Xanh: đúng.
- Vàng: confidence/localization giảm.
- Đỏ: miss/misclassified.
- Xám: false positive.
- Luôn có icon/text, không chỉ màu.

SAM:

- Mask attacked.
- Missing region.
- Extra region.
- Boundary error.
- Complete failure.

Overlay:

- Xanh: overlap.
- Đỏ: GT bị bỏ sót.
- Vàng: prediction thừa.

### Phần 5 — Five Key Metrics

YOLO:

1. Clean Detection Score.
2. Score After Attack.
3. Performance Lost.
4. Objects Broken by Attack.
5. Overall Robustness.

SAM:

1. Clean Mask Accuracy.
2. Mask Accuracy After Attack.
3. Mask Performance Lost.
4. Boundary Accuracy.
5. Masks Broken by Attack.

## 15.4 Advanced Metrics Drawer

Bao gồm:

- Full metric table.
- Per-class.
- Per-size.
- RA(s).
- Confidence/IoU distribution.
- Failure clusters.
- Latency/throughput/GPU.
- CI.
- External test.
- Provenance.
- Export.

## 15.5 Comparison Mode

Input:

```text
Baseline ModelVersion
Defended ModelVersion
BenchmarkProtocol
```

Hiển thị:

- Narrative headline.
- Five-metric comparison.
- Absolute/relative delta.
- Recovery Rate.
- Clean tradeoff.
- Per-class/per-size delta.
- Failure examples.
- Model lineage.
- Training composition.
- Reviewer status.

Narrative mẫu:

> Robust-v2 phục hồi 16.6 điểm AP trong Low Visibility, giảm object failure rate từ 61% xuống 34%, trong khi clean AP chỉ giảm 0.7 điểm.

SAM mẫu:

> SAM robust-v2 tăng attacked mIoU từ 0.43 lên 0.61, giảm mask failure rate 22 percentage points và cải thiện rõ nhất ở object nhỏ bị che khuất.

## 15.6 Accessibility và UX rules

- Không dùng màu là tín hiệu duy nhất.
- Có text/icon.
- Keyboard navigation.
- Tooltip có thể mở bằng click.
- Số có đơn vị.
- Ratio/percent không trộn.
- Severity có nhãn thực tế.
- Cảnh báo khi comparison không paired.
- Không hiển thị pseudo-mask như GT.
- Không cho run incompatible recipe.
- Report mặc định ưu tiên kết luận và metric; evidence ảnh vẫn mở rộng được.


# 16. Data model và quan hệ lineage

## 16.1 Attack và generated data

```text
Attack
AttackImplementationVersion
AttackRecipe
AttackRecipeStep
ScenarioPreset
GeneratedDataset
GeneratedDatasetVersion
GeneratedVariant
```

## 16.2 Defense và training

```text
DefenseProfile
RetrainingBacklog
RetrainingBacklogItem
TrainingDatasetManifest
TrainingRun
TrainingJob
TrainingMetric
TrainingCheckpoint
ModelVersionLineage
HardExampleBank
```

## 16.3 Evaluation

```text
BenchmarkProtocol
BenchmarkRun
MetricResult
MetricDelta
FailureCase
FailureCluster
ModelComparison
RecoveryReport
ReviewDecision
```

## 16.4 Quan hệ chính

```text
DatasetVersion
  ├── BenchmarkProtocol
  └── TrainingDatasetManifest

AttackRecipe
  └── GeneratedDatasetVersion
       └── BenchmarkRun

ModelVersion baseline
  └── BenchmarkRun
       └── FailureCase
            └── FailureCluster
                 └── RetrainingBacklogItem
                      └── DefenseProfile
                           └── TrainingRun
                                └── ModelVersion defended
                                     └── BenchmarkRun
                                          └── ModelComparison
                                               └── RecoveryReport
```

## 16.5 Audit fields chung

Mọi entity quan trọng nên có:

```text
id
version
status
created_by
created_at
updated_at
project_id
git_commit
environment_version
notes
audit_log
```

---

# 17. API cần bổ sung

## 17.1 Attack Catalog và Recipe

```text
GET  /api/v1/attacks
GET  /api/v1/attacks/{attack_name}
GET  /api/v1/scenario-presets

POST /api/v1/attack-recipes/validate
POST /api/v1/attack-recipes/randomize
POST /api/v1/attack-recipes/estimate
POST /api/v1/attack-recipes
GET  /api/v1/attack-recipes/{recipe_id}
```

## 17.2 Generated Dataset

```text
POST /api/v1/generated-datasets
GET  /api/v1/generated-datasets/{id}
GET  /api/v1/generated-datasets/{id}/manifest
GET  /api/v1/generated-datasets/{id}/variants
POST /api/v1/generated-datasets/{id}/validate
```

## 17.3 Benchmark

```text
POST /api/v1/benchmark-protocols
GET  /api/v1/benchmark-protocols/{id}
POST /api/v1/benchmark-runs
GET  /api/v1/benchmark-runs/{id}
GET  /api/v1/benchmark-runs/{id}/events/ws
GET  /api/v1/benchmark-runs/{id}/metrics
GET  /api/v1/benchmark-runs/{id}/failures
```

## 17.4 Retraining

```text
POST /api/v1/retraining-backlogs
GET  /api/v1/retraining-backlogs/{id}
POST /api/v1/retraining-backlogs/{id}/items
POST /api/v1/retraining-backlogs/{id}/approve
POST /api/v1/defense-profiles
GET  /api/v1/defense-profiles/{id}
```

## 17.5 Training

```text
POST /api/v1/training-runs/estimate
POST /api/v1/training-runs
GET  /api/v1/training-runs
GET  /api/v1/training-runs/{id}
POST /api/v1/training-runs/{id}/cancel
GET  /api/v1/training-runs/{id}/events/ws
GET  /api/v1/training-runs/{id}/checkpoints
```

## 17.6 ModelVersion

```text
GET  /api/v1/models/{model_id}/versions
GET  /api/v1/model-versions/{version_id}
GET  /api/v1/model-versions/{version_id}/lineage
POST /api/v1/model-versions/{version_id}/benchmark
```

## 17.7 Comparison và report

```text
POST /api/v1/model-comparisons
GET  /api/v1/model-comparisons/{comparison_id}
GET  /api/v1/model-comparisons/{comparison_id}/metric-deltas
GET  /api/v1/model-comparisons/{comparison_id}/failures
GET  /api/v1/model-comparisons/{comparison_id}/recovery-report
GET  /api/v1/model-comparisons/{comparison_id}/export
```

## 17.8 API response rules

- Metric có unit rõ.
- Ratio và percent tách field.
- Version IDs luôn trả.
- Recipe trả ordered steps.
- Model trả lineage.
- Long-running job trả status/event.
- Error trả machine code + user message.
- Estimate trả assumption.
- Comparison trả paired/unpaired flag.
- Frontend không tự suy luận metric unit.

---

# 18. Thay đổi code theo module

## 18.1 Core

### `src/core/types.py`

- Sửa `Prediction` constructor.
- Chuyển sang keyword-only.
- Thêm `DetectionPrediction`.
- Thêm `SegmentationPrediction`.
- Chuẩn hóa task/modality.
- Thêm schema validation.

### `src/core/objectives.py`

- Tách attack objective và training objective.
- Chuẩn hóa detection/segmentation loss target.
- Thêm metadata objective version.

## 18.2 Attack

### `src/attacks/base.py`

- Metadata kể chuyện.
- Compatibility.
- Cost.
- Severity mapping.
- Implementation version.

### `src/attacks/recipes.py`

- Recipe schema.
- Ordering.
- Validation.
- Deterministic random.
- Quota sampling.
- Hashing.

### `src/attacks/presets.py`

- Low Visibility.
- Wet Camera.
- Poor Camera Pipeline.
- Partial Obstruction.
- Adversarial Stress.
- Segmentation Boundary Stress.

## 18.3 Pipeline

### `src/pipeline/generator.py`

- Từ single attack sang ordered recipe.
- Intermediate outputs.
- Annotation transformation.
- Manifest.

### `src/pipeline/composition.py`

- Apply ordered steps.
- Cost accounting.
- Error handling.
- Determinism.

### `src/pipeline/runner.py`

- Independent benchmark.
- Composed benchmark.
- Multi-model inference.
- Paired comparison.
- Backward compatibility.

### `src/pipeline/cache.py`

- Generation cache.
- Inference cache.
- Recipe hash.
- Version-aware invalidation.

## 18.4 Evaluation

Thêm:

```text
src/evaluation/detection_metrics.py
src/evaluation/segmentation_metrics.py
src/evaluation/robustness_metrics.py
src/evaluation/model_comparison.py
src/evaluation/recovery_metrics.py
src/evaluation/bootstrap.py
```

## 18.5 Training

Thêm:

```text
src/training/base.py
src/training/dataset_builder.py
src/training/yolo_trainer.py
src/training/sam2_trainer.py
src/training/registry.py
src/training/report.py
src/training/hard_example_bank.py
```

## 18.6 Adapters

- Hoàn thiện YOLO adapter metadata/gradient.
- Tạo runnable `src/adapters/sam2.py`.
- Giữ surrogate adapter riêng nếu cần.
- Không để surrogate-only adapter xuất hiện như runnable model.

## 18.7 API

Tách route:

```text
src/api/run_routes.py
src/api/training_routes.py
src/api/model_routes.py
src/api/recipe_routes.py
src/api/review_routes.py
src/api/dataset_routes.py
```

## 18.8 Frontend components

```text
PerceptionModeSelector
DatasetModelSelector
ModelVersionSelector
AttackCatalog
AttackInfoPopover
AttackRecipeBuilder
RandomRecipeDialog
RecipeStepList
RecipePreview
CompatibilityWarning
ResourceEstimate
YoloComparisonWorkspace
SamComparisonWorkspace
FiveMetricSummary
AdvancedMetricsDrawer
ModelVersionComparison
MetricDeltaTable
RecoveryChart
FailureClusterPanel
EvidenceDrawer
TrainingLineagePanel
TrainingRunHistory
DefenseProfileViewer
```

---

# 19. Các lỗi nền tảng phải sửa trước

## 19.1 Prediction positional bug

Hiện có nguy cơ latency positional bị gán vào `boxes3d`.

Cách sửa:

- Keyword-only constructor.
- Migrate toàn bộ adapter.
- Contract test.
- Runtime type validation.
- Không chấp nhận positional với field mở rộng.

## 19.2 Chuẩn hóa degradation

API:

```json
{
  "degradation_ratio": 0.42,
  "degradation_pct": 42.0
}
```

Frontend hiển thị `degradation_pct`.

## 19.3 Auto-review threshold

- Threshold 30 nghĩa là 30%.
- So với `degradation_pct`.
- Hoặc threshold 0.30 nếu field là ratio, nhưng không trộn.
- Config phải ghi unit.

## 19.4 Model selector

Loại bỏ hard-code:

```text
model = blob_detector
limit = 8
một severity duy nhất
```

UI phải gửi:

- Model ID.
- ModelVersion ID.
- Task.
- Dataset.
- Recipe.
- BenchmarkProtocol.

## 19.5 Frontend CI

Bắt buộc:

```text
npm ci
npm run lint
npm run build
```

Khuyến nghị thêm:

```text
npm test
```

## 19.6 Correctness trước feature

Không triển khai training/composition lớn trước khi:

- Prediction contract ổn.
- Metric unit ổn.
- Dataset split policy khóa.
- Benchmark deterministic.
- Model selector thật.
- CI chạy.

---

# 20. Chiến lược kiểm thử

## 20.1 Unit tests

- Attack metadata schema.
- Severity mapping.
- Compatibility.
- Recipe ordering.
- Recipe hash.
- Deterministic random.
- Quota sampling.
- Cost estimate.
- Prediction types.
- Metric units.
- Recovery formula.
- Training config validation.
- Dataset leakage validator.
- Annotation transformation.

## 20.2 Integration tests

- Manual recipe.
- Random N.
- Random by group.
- Scenario preset.
- Generated dataset reload.
- Cache hit/miss.
- Multi-model benchmark.
- YOLO end-to-end.
- SAM end-to-end.
- TrainingRun state transitions.
- ModelVersion registration.
- Paired comparison.
- Recovery report.
- Export.

## 20.3 Scientific validation

- Severity 0 là no-op.
- Severity monotonicity được kiểm tra nhưng không giả định tuyệt đối.
- Strong PGD phải mạnh hơn random noise cùng norm trong sanity test phù hợp.
- Không train/test leakage.
- Clean tradeoff.
- Seen vs unseen attack.
- External generalization.
- Per-class.
- Per-object-size.
- Paired bootstrap CI.
- Same-prompt SAM comparison.
- Annotation transform correctness.
- Reproducibility với seed.

## 20.4 UI tests

- Mode filtering.
- Attack hover/click.
- Compatibility block.
- Recipe drag/drop.
- Random preview.
- Budget warning.
- Năm metric đúng theo mode.
- Advanced drawer.
- Baseline/defended comparison.
- Evidence viewer.
- Keyboard/accessibility.
- Không chỉ dùng màu.
- Ratio/percent render đúng.
- Error message dễ hiểu.

## 20.5 End-to-end acceptance tests

### YOLO

```text
Select YOLO
→ select dataset/model
→ choose Low Visibility
→ generate variants
→ benchmark baseline
→ inspect failure
→ create retraining request
→ open completed robust TrainingRun
→ benchmark robust model
→ compare and export
```

### SAM

```text
Select SAM
→ select mask dataset
→ verify prompt protocol
→ run boundary stress
→ inspect clean/attacked masks
→ compare SAM-B0 vs SAM-R1
→ view mIoU/Boundary IoU/failure rate
```

---

# 21. PHÂN CÔNG NHÓM BỐN NGƯỜI

## 21.1 Nguyên tắc phân công

Nhóm được tổ chức theo mô hình:

```text
Người A: Toàn bộ WebApp — Frontend + Backend
Người B: Toàn bộ luồng YOLO11 — Detection
Người C: Toàn bộ luồng SAM2 — Segmentation
Người D: Data, Attack Engine, Benchmark và nền tảng dùng chung
```

Mỗi hạng mục chỉ có **một người chịu trách nhiệm chính**. Các thành viên khác có thể hỗ trợ kiểm thử hoặc review nhưng không cùng sở hữu một phần việc, nhằm tránh:

* Hai người cùng sửa một module.
* Không rõ ai chịu trách nhiệm khi chức năng lỗi.
* Chồng chéo giữa backend Web và pipeline AI.
* Chồng chéo giữa xử lý dữ liệu YOLO và SAM.
* Chồng chéo giữa metric riêng của mô hình và metric robustness dùng chung.

Ranh giới tổng quát:

* Người A chịu trách nhiệm biến toàn bộ chức năng thành một WebApp sử dụng được.
* Người B chịu trách nhiệm làm cho YOLO11 chạy, huấn luyện, đánh giá và tạo ra model robust.
* Người C chịu trách nhiệm làm cho SAM2 chạy, huấn luyện, đánh giá và tạo ra model robust.
* Người D chịu trách nhiệm chuẩn bị dữ liệu, sinh attack, xây benchmark và cung cấp pipeline dùng chung cho Người B và Người C.

---

## 21.2 Người A — Full-stack Web: Frontend, Backend, Database và tích hợp sản phẩm

### Vai trò chính

Người A chịu trách nhiệm **toàn bộ phần WebApp**, bao gồm:

```text
Frontend
+
Backend API
+
Database
+
WebSocket
+
Job dispatch
+
Review workflow
+
Report và export
+
Tích hợp các pipeline AI vào WebApp
```

Người A là người duy nhất chịu trách nhiệm chính đối với mã nguồn giao diện và backend Web.

---

### 21.2.1 Frontend

Người A xây dựng toàn bộ giao diện theo bố cục hai vùng:

```text
LEFT CONFIG PANEL
+
CENTER WORKSPACE
```

#### A. Khung giao diện và điều hướng

* Xây dựng layout hai vùng.
* Xây dựng routing giữa các màn hình.
* Quản lý trạng thái chung của ứng dụng.
* Quản lý trạng thái các job đang chạy.
* Xử lý loading, empty state, error state và completed state.
* Bảo đảm giao diện hoạt động cho cả YOLO11 và SAM2.
* Không tạo hai giao diện hoàn toàn tách biệt nếu có thể tái sử dụng component.

#### B. Panel cấu hình bên trái

Xây dựng các component:

* `PerceptionModeSelector`.
* `DatasetModelSelector`.
* `ModelVersionSelector`.
* `AttackCatalog`.
* `AttackInfoPopover`.
* `AttackRecipeBuilder`.
* `RecipeStepList`.
* `RandomRecipeDialog`.
* `RecipePreview`.
* `CompatibilityWarning`.
* `ResourceEstimate`.

Các chức năng cụ thể:

* Chọn chế độ YOLO11 hoặc SAM2.
* Tự động lọc dataset tương thích.
* Tự động lọc model và version tương thích.
* Tự động lọc attack tương thích.
* Hiển thị dataset version, số sample, annotation type và split.
* Chọn baseline model hoặc defended model.
* Hiển thị lineage của model version.
* Chọn Single Attack.
* Chọn Manual Composition.
* Chọn Random N Attacks.
* Chọn Random by Group.
* Chọn Scenario Preset.
* Chọn Auto Sweep.
* Kéo thả thay đổi thứ tự attack.
* Cấu hình severity và tham số attack.
* Hiển thị cảnh báo tổ hợp không hợp lệ.
* Hiển thị ước tính sample, GPU-hours, storage và runtime.
* Chặn người dùng chạy cấu hình không tương thích.

#### C. Khu vực hiển thị kết quả

Xây dựng đầy đủ năm phần:

1. Input và Ground Truth.
2. Attacked Input.
3. Clean Model Prediction.
4. Attacked Model Prediction.
5. Five Key Metrics.

Component chính:

* `YoloComparisonWorkspace`.
* `SamComparisonWorkspace`.
* `FiveMetricSummary`.
* `AdvancedMetricsDrawer`.
* `EvidenceDrawer`.

#### D. Hiển thị YOLO

* Render ảnh gốc.
* Render ground-truth bounding box.
* Render clean prediction.
* Render attacked prediction.
* Render defended prediction.
* Hiển thị class và confidence.
* Hiển thị trạng thái:

  * Correct.
  * Missed.
  * False positive.
  * Misclassified.
  * Confidence collapsed.
  * Localization degraded.
* Hiển thị các object bị mất sau attack.
* Hiển thị IoU và confidence delta.
* Cho phép bật hoặc tắt từng loại overlay.
* Đồng bộ zoom và pan giữa các khung ảnh.

#### E. Hiển thị SAM2

* Render ground-truth mask.
* Render clean predicted mask.
* Render attacked predicted mask.
* Render defended predicted mask.
* Hiển thị mask opacity.
* Hiển thị mask boundary.
* Hiển thị vùng dự đoán thiếu.
* Hiển thị vùng dự đoán thừa.
* Hiển thị prompt type và prompt coordinates.
* Hiển thị mask score.
* Không chỉ sử dụng màu để biểu thị trạng thái lỗi.

#### F. Metric và comparison

Xây dựng:

* `ModelVersionComparison`.
* `MetricDeltaTable`.
* `RecoveryChart`.
* `FailureClusterPanel`.
* `TrainingLineagePanel`.
* `TrainingRunHistory`.
* `DefenseProfileViewer`.

Giao diện phải hiển thị:

* Clean metric.
* Attacked metric.
* Degradation.
* Attack Success Rate hoặc Failure Rate.
* RobustScore.
* Recovery Rate.
* Baseline và defended model.
* Mức tăng hoặc giảm của từng metric.
* Clean-to-robust trade-off.
* Per-class result.
* Object-size result.
* Severity curve.
* Confidence interval.
* External-dataset result.
* Narrative headline tự động.

#### G. Review và retraining trên giao diện

* Hiển thị failure case.
* Hiển thị failure cluster.
* Cho reviewer chọn các failure quan trọng.
* Tạo retraining backlog.
* Tạo hoặc xem DefenseProfile.
* Gửi yêu cầu TrainingRun.
* Theo dõi trạng thái TrainingRun.
* Xem checkpoint.
* Xem model version mới.
* Chạy benchmark lại.
* So sánh baseline và defended model.
* Xuất Recovery Report.

---

### 21.2.2 Backend Web và API

Người A xây dựng toàn bộ API Web.

#### Attack Catalog và Recipe API

```text
GET  /api/v1/attacks
GET  /api/v1/attacks/{attack_name}
GET  /api/v1/scenario-presets

POST /api/v1/attack-recipes/validate
POST /api/v1/attack-recipes/randomize
POST /api/v1/attack-recipes/estimate
POST /api/v1/attack-recipes
GET  /api/v1/attack-recipes/{recipe_id}
```

Backend Web không tự triển khai thuật toán attack. Backend gọi các service hoặc library do Người D cung cấp.

#### Generated Dataset API

```text
POST /api/v1/generated-datasets
GET  /api/v1/generated-datasets/{id}
GET  /api/v1/generated-datasets/{id}/manifest
GET  /api/v1/generated-datasets/{id}/variants
POST /api/v1/generated-datasets/{id}/validate
```

#### Benchmark API

```text
POST /api/v1/benchmark-protocols
GET  /api/v1/benchmark-protocols/{id}

POST /api/v1/benchmark-runs
GET  /api/v1/benchmark-runs/{id}
GET  /api/v1/benchmark-runs/{id}/events/ws
GET  /api/v1/benchmark-runs/{id}/metrics
GET  /api/v1/benchmark-runs/{id}/failures
```

#### Retraining và Defense API

```text
POST /api/v1/retraining-backlogs
GET  /api/v1/retraining-backlogs/{id}
POST /api/v1/retraining-backlogs/{id}/items
POST /api/v1/retraining-backlogs/{id}/approve

POST /api/v1/defense-profiles
GET  /api/v1/defense-profiles/{id}
```

#### Training API

```text
POST /api/v1/training-runs/estimate
POST /api/v1/training-runs
GET  /api/v1/training-runs
GET  /api/v1/training-runs/{id}
POST /api/v1/training-runs/{id}/cancel
GET  /api/v1/training-runs/{id}/events/ws
GET  /api/v1/training-runs/{id}/checkpoints
```

#### Model Registry API

```text
GET  /api/v1/models/{model_id}/versions
GET  /api/v1/model-versions/{version_id}
GET  /api/v1/model-versions/{version_id}/lineage
POST /api/v1/model-versions/{version_id}/benchmark
```

#### Comparison và Report API

```text
POST /api/v1/model-comparisons
GET  /api/v1/model-comparisons/{comparison_id}
GET  /api/v1/model-comparisons/{comparison_id}/metric-deltas
GET  /api/v1/model-comparisons/{comparison_id}/failures
GET  /api/v1/model-comparisons/{comparison_id}/recovery-report
GET  /api/v1/model-comparisons/{comparison_id}/export
```

---

### 21.2.3 Database và persistence

Người A tạo database schema, migration, repository và query cho các thực thể:

* Dataset.
* DatasetVersion.
* GeneratedDatasetVersion.
* AttackDefinition.
* AttackRecipe.
* BenchmarkProtocol.
* BenchmarkRun.
* MetricResult.
* FailureCase.
* FailureCluster.
* RetrainingBacklog.
* RetrainingBacklogItem.
* DefenseProfile.
* TrainingDatasetManifest.
* TrainingRun.
* TrainingCheckpoint.
* Model.
* ModelVersion.
* ModelVersionLineage.
* ModelComparison.
* RecoveryReport.
* AuditLog.

Người A chịu trách nhiệm:

* ID và quan hệ khóa ngoại.
* Trạng thái của từng job.
* Lưu model lineage.
* Lưu dataset lineage.
* Lưu attack recipe và recipe hash.
* Lưu metric version và metric unit.
* Lưu checkpoint metadata.
* Lưu reviewer decision.
* Lưu audit fields.
* Truy vấn lịch sử training.
* Truy vấn lịch sử benchmark.
* Truy vấn comparison và report.

---

### 21.2.4 Job dispatch và event streaming

Người A chịu trách nhiệm phần Web của job orchestration:

* Nhận yêu cầu từ frontend.
* Validate request schema.
* Tạo job record.
* Đưa job vào hàng đợi hoặc gọi worker.
* Gửi đúng config cho pipeline của Người D.
* Gửi đúng trainer config cho Người B hoặc Người C.
* Nhận progress event.
* Cập nhật trạng thái database.
* Gửi trạng thái cho frontend qua WebSocket.
* Xử lý cancel request.
* Hiển thị lỗi bằng machine code và thông báo dễ hiểu.
* Không đặt training logic trực tiếp trong API route.

Phân biệt rõ:

```text
Người A:
API, database, enqueue, cancel, status, WebSocket và lưu kết quả.

Người D:
Compute pipeline, worker execution, attack generation và benchmark runner.

Người B/C:
Trainer và model-specific inference.
```

---

### 21.2.5 Kiểm thử do Người A phụ trách

* Frontend unit test.
* Backend API test.
* Database integration test.
* WebSocket test.
* UI interaction test.
* Mode filtering test.
* Model selector test.
* Attack compatibility UI test.
* Recipe builder test.
* Metric unit rendering test.
* Baseline/defended comparison test.
* Accessibility test.
* Error-state test.
* Export test.
* End-to-end happy path trên WebApp.

---

### 21.2.6 Đầu ra bắt buộc của Người A

* WebApp chạy được từ đầu đến cuối.
* Frontend không còn hard-code model.
* Backend có đầy đủ API đã định nghĩa.
* Database có migration và lineage đầy đủ.
* WebSocket hiển thị được tiến độ benchmark và training.
* UI chạy được với dữ liệu mock trước khi AI pipeline hoàn thành.
* UI nối được với pipeline thật sau khi Người B, C và D bàn giao.
* Có màn hình comparison và Recovery Report.
* Có review workflow và retraining workflow.
* Có export report.
* Frontend CI hoạt động.
* Có E2E test cho YOLO và SAM2.

---

### 21.2.7 Người A không phụ trách

* Viết thuật toán attack.
* Viết logic biến đổi ảnh.
* Huấn luyện YOLO11.
* Huấn luyện SAM2.
* Tính mAP hoặc mIoU ở tầng thuật toán.
* Xây dựng detection evaluator.
* Xây dựng segmentation evaluator.
* Sinh defense dataset.
* Quyết định hyperparameter chuyên môn của mô hình.
* Chọn checkpoint tốt nhất về mặt khoa học.

---

## 21.3 Người B — YOLO11, Detection và huấn luyện mô hình phát hiện vật thể

### Vai trò chính

Người B sở hữu toàn bộ luồng:

```text
Detection dataset
→ YOLO11 adapter
→ YOLO-B0
→ YOLO attack benchmark
→ YOLO-R1
→ YOLO-R2 nếu cần
→ Detection evaluation
→ External evaluation
→ Checkpoint và training report
```

Người B chịu trách nhiệm cuối cùng về tính đúng đắn của kết quả YOLO11.

---

### 21.3.1 Chuẩn bị dữ liệu detection

Phối hợp với Người D để hoàn thiện dữ liệu cho YOLO11.

Người B chịu trách nhiệm phần chuyên môn detection:

* Xác định dataset chính:

  * KITTI Object.
  * Hoặc dataset detection nội bộ đã anonymize.
* Chốt ba class chuẩn hóa:

  * Car.
  * Pedestrian.
  * Cyclist.
* Chốt class mapping từ BDD100K.
* Xác định quy tắc map `bike/rider` sang `Cyclist`.
* Kiểm tra bounding box.
* Kiểm tra class distribution.
* Kiểm tra object-size distribution.
* Kiểm tra ảnh hoặc annotation bất thường.
* Chốt preprocessing của YOLO.
* Chốt image size.
* Chốt augmentation tiêu chuẩn.
* Xác nhận split manifest do Người D tạo là phù hợp cho detection.

Người D thực hiện pipeline dữ liệu và manifest; Người B duyệt tính đúng đắn về mặt detection.

---

### 21.3.2 YOLO11 adapter

Người B hoàn thiện YOLO adapter:

* Load YOLO11s.
* Load đúng model version.
* Chạy inference.
* Trả prediction theo keyword fields.
* Trả bounding box theo `xyxy`.
* Trả class label.
* Trả confidence.
* Trả latency.
* Trả preprocessing version.
* Trả metadata.
* Hỗ trợ loss cho attack.
* Hỗ trợ input gradient cho FGSM và PGD.
* Hỗ trợ batch inference.
* Hỗ trợ export và reload checkpoint.

Output bắt buộc:

```text
DetectionPrediction
├── sample_id
├── model_version_id
├── boxes_xyxy
├── labels
├── confidences
├── latency_ms
├── preprocessing_version
└── metadata
```

---

### 21.3.3 Huấn luyện YOLO-B0

Người B huấn luyện clean baseline:

```text
Tên model: YOLO11s
Version: YOLO-B0
Mục tiêu: Clean baseline
```

Công việc:

* Khởi tạo từ pretrained YOLO11s.
* Huấn luyện trên training split.
* Không sử dụng locked test.
* Không đưa strong attack vào baseline.
* Sử dụng standard augmentation.
* Huấn luyện khoảng 30 epoch làm cấu hình tham chiếu.
* Bật AMP.
* Thiết lập early stopping.
* Chọn checkpoint theo validation mAP@[.50:.95].
* Lưu best checkpoint.
* Lưu last checkpoint.
* Lưu training curves.
* Lưu random seed.
* Lưu framework version.
* Lưu Git commit.
* Lưu GPU-hours.
* Tạo model registration metadata.

---

### 21.3.4 Huấn luyện YOLO-R1

Người B huấn luyện robust model:

```text
Parent: YOLO-B0
Version: YOLO-R1
Mục tiêu: Robust mix
```

Mix mặc định:

| Thành phần                | Tỷ lệ |
| ------------------------- | ----: |
| Clean                     |   40% |
| Weather/common corruption |   30% |
| Occlusion                 |   15% |
| Fast adversarial          |   10% |
| Patch                     |    5% |

Công việc:

* Không train lại từ đầu.
* Fine-tune từ YOLO-B0.
* Huấn luyện bổ sung khoảng 15–25 epoch.
* Freeze backbone 3–5 epoch đầu khi cần.
* Sau đó unfreeze.
* Giữ clean replay trong mọi giai đoạn.
* Dùng corruption severity chủ yếu từ 1–3.
* Dùng FGSM hoặc PGD 2–3 bước cho tỷ lệ batch nhỏ.
* Dùng patch đã cache.
* Theo dõi clean AP trong suốt quá trình.
* Early stop theo robust validation score.
* Chọn checkpoint thỏa gate.

Gate YOLO:

* Clean AP giảm không quá 2 điểm.
* RobustScore tăng ít nhất 8 điểm hoặc attacked AP tăng rõ.
* Không làm critical scenario xấu hơn đáng kể.
* External AP không giảm quá ngưỡng.
* Kết quả phải chạy trên cùng locked benchmark.

---

### 21.3.5 Huấn luyện YOLO-R2

Chỉ thực hiện khi YOLO-R1 còn failure cluster rõ ràng.

Ví dụ:

```text
Failure cluster:
Pedestrian nhỏ hoặc ở xa
+
Fog severity 4
+
Partial occlusion
```

Mix tham chiếu:

| Thành phần                  | Tỷ lệ |
| --------------------------- | ----: |
| General robust replay       |   60% |
| Target failure analogues    |   25% |
| Clean class-balanced replay |   15% |

Người B xác định:

* Cluster nào đủ nghiêm trọng để repair.
* Điều kiện chọn target sample.
* Tỷ lệ targeted replay.
* Tiêu chí dừng.
* Có giữ được general robustness hay không.

---

### 21.3.6 Attack chuyên biệt cho YOLO

Người D xây attack framework; Người B phụ trách phần phụ thuộc trực tiếp vào YOLO:

* Detection attack objective.
* Objectness loss target.
* Classification loss target.
* Localization loss target.
* FGSM cho YOLO.
* PGD cho YOLO.
* C&W hoặc attack mạnh nếu nằm trong phạm vi thực hiện.
* Kiểm tra gradient.
* Kiểm tra attack có thực sự làm giảm detection performance.
* Xác định epsilon hợp lý.
* Xác định số bước benchmark.
* Xác định số bước fast adversarial training.
* Sinh hoặc duyệt YOLO hard examples.
* Kiểm tra patch placement đối với bounding box.
* Kiểm tra object occlusion không làm annotation vô nghĩa.

---

### 21.3.7 Detection evaluator

Người B xây dựng:

```text
src/evaluation/detection_metrics.py
```

Metric bắt buộc:

* AP50.
* AP75.
* mAP@[.50:.95].
* Precision.
* Recall.
* F1.
* AP theo class.
* AP theo object size.
* False positive rate.
* Miss rate.
* Confidence drop.
* IoU distribution.
* Object-level Attack Success Rate.
* Image-level Attack Success Rate.
* Detection failure reason.

Per-object evaluation phải trả:

```text
object_id
gt_box
clean_match
attacked_match
clean_confidence
attacked_confidence
status_clean
status_attacked
iou_clean
iou_attacked
failure_reason
```

Người D sử dụng kết quả detection này để tính RobustScore, Recovery Rate và comparison dùng chung.

---

### 21.3.8 Benchmark và external evaluation

Người B chịu trách nhiệm chạy và xác nhận:

* Clean validation.
* Clean locked test.
* Single-attack benchmark.
* Composite-attack benchmark.
* Severity sweep.
* Baseline versus robust comparison.
* BDD100K external evaluation.
* Per-class analysis.
* Object-size analysis.
* Failure-cluster analysis.
* Inference latency.
* Throughput.

Benchmark runner do Người D cung cấp; Người B chịu trách nhiệm model, evaluator và tính hợp lệ của kết quả detection.

---

### 21.3.9 Kiểm thử do Người B phụ trách

* YOLO adapter unit test.
* Prediction schema test.
* Checkpoint load test.
* Gradient sanity test.
* Detection metric test.
* Bounding-box matching test.
* Per-class metric test.
* Object-size metric test.
* FGSM/PGD sanity test.
* Clean inference regression test.
* Reproducible training test.
* YOLO end-to-end compute test.
* Scientific validation cho YOLO.

---

### 21.3.10 Đầu ra bắt buộc của Người B

* YOLO adapter chạy được.
* YOLO-B0 checkpoint.
* YOLO-R1 checkpoint.
* YOLO-R2 khi thực sự cần.
* Training config cho từng run.
* Training curves.
* Checkpoint hash.
* Detection evaluator.
* YOLO hard-example set.
* BDD100K external result.
* Baseline-versus-robust result.
* YOLO Recovery Report input.
* Model registration metadata.
* Payload prediction và metric để Người A hiển thị.
* Tài liệu hướng dẫn tái chạy training và benchmark.

---

### 21.3.11 Người B không phụ trách

* Xây dựng WebApp.
* Viết API route.
* Thiết kế database.
* Xây dựng attack recipe framework dùng chung.
* Xây dựng cache dùng chung.
* Xây dựng segmentation pipeline.
* Huấn luyện SAM2.
* Quản lý WebSocket.
* Xây dựng review UI.

---

## 21.4 Người C — SAM2, Segmentation và huấn luyện mô hình phân đoạn

### Vai trò chính

Người C sở hữu toàn bộ luồng:

```text
Segmentation dataset
→ Mask validation
→ Prompt protocol
→ Runnable Sam2Adapter
→ SAM-B0
→ SAM attack benchmark
→ SAM-R1
→ SAM-R2 nếu cần
→ Segmentation evaluation
→ External evaluation
```

Người C chịu trách nhiệm cuối cùng về tính đúng đắn của kết quả SAM2.

---

### 21.4.1 Chuẩn bị dữ liệu segmentation

Người C lựa chọn và chuẩn hóa dataset:

* Cityscapes.
* BDD100K segmentation.
* Hoặc dataset nội bộ có ground-truth mask thật.

Người C không sử dụng KITTI Object làm benchmark segmentation chính nếu dataset không có ground-truth mask.

Công việc cụ thể:

* Xác định task là instance segmentation hoặc protocol segmentation cụ thể.
* Xác định class sử dụng.
* Xác định class intersection giữa in-domain và external dataset.
* Kiểm tra mask có hợp lệ.
* Kiểm tra mask rỗng.
* Kiểm tra mask bị lỗi biên.
* Kiểm tra instance ID.
* Kiểm tra object size.
* Kiểm tra occlusion metadata nếu có.
* Chốt quy trình review mask.
* Chỉ cho phép mask đã review trong validation và locked test.
* Gắn nguồn cho pseudo-mask.
* Chỉ dùng pseudo-mask chưa review cho training phụ khi được cho phép.

Người D xây manifest và pipeline dữ liệu; Người C duyệt tính đúng đắn của mask.

---

### 21.4.2 Prompt protocol

Người C chịu trách nhiệm định nghĩa và kiểm thử prompt protocol.

Benchmark chính:

* Dùng ground-truth box prompt.
* Dùng cùng prompt cho clean image và attacked image.
* Dùng cùng prompt cho baseline và robust model.
* Lưu prompt trong benchmark manifest.
* Không lấy box do YOLO dự đoán làm prompt cho benchmark SAM độc lập.
* Không thay đổi prompt giữa hai model đang so sánh.

Mục tiêu là đo đúng khả năng segmentation, không trộn lỗi detection vào kết quả SAM.

---

### 21.4.3 Runnable Sam2Adapter

Người C tạo adapter chạy được, tách khỏi surrogate-only adapter.

File chính:

```text
src/adapters/sam2.py
```

Interface:

```python
predict_masks(samples, prompts)
loss_for_attack(sample, target)
input_gradient(sample, target)
metadata()
```

Output:

```text
SegmentationPrediction
├── sample_id
├── model_version_id
├── masks
├── mask_scores
├── prompt_type
├── prompt_coordinates
├── latency_ms
├── preprocessing_version
└── metadata
```

Người C chịu trách nhiệm:

* Load checkpoint.
* Encode image.
* Xử lý box prompt.
* Chạy mask decoder.
* Trả mask score.
* Trả mask đúng kích thước.
* Trả prompt metadata.
* Hỗ trợ input gradient.
* Hỗ trợ batch hoặc batch strategy phù hợp.
* Export checkpoint.
* Reload checkpoint.
* Không để surrogate-only adapter xuất hiện như runnable production model.

---

### 21.4.4 Huấn luyện SAM-B0

```text
Model: SAM2.1 Hiera Small
Version: SAM-B0
Mục tiêu: Clean baseline
```

Pha đầu:

* Freeze image encoder.
* Fine-tune mask decoder.
* Fine-tune prompt-related module.
* Bật AMP.
* Dùng gradient accumulation khi thiếu VRAM.
* Huấn luyện khoảng 10–20 epoch làm tham chiếu.
* Early stop theo validation mIoU.
* Theo dõi Boundary IoU.

Nếu kết quả chưa đủ:

* Mở một số block cuối của encoder.
* Fine-tune thêm khoảng 5–10 epoch.
* Giảm learning rate.
* So sánh rõ với decoder-only checkpoint.

---

### 21.4.5 Huấn luyện SAM-R1

```text
Parent: SAM-B0
Version: SAM-R1
Mục tiêu: Robust mix
```

Mix tham chiếu:

| Thành phần                             | Tỷ lệ |
| -------------------------------------- | ----: |
| Clean image và mask                    |   45% |
| Weather/noise                          |   25% |
| Blur/compression                       |   10% |
| Occlusion                              |   15% |
| Fast adversarial hoặc cached hard mask |    5% |

Người C thực hiện:

* Fine-tune từ SAM-B0.
* Giữ clean replay.
* Dùng weather và noise.
* Dùng blur và compression.
* Dùng partial object occlusion.
* Dùng random erasing có cập nhật valid mask.
* Dùng gradient attack nhẹ ở tỷ lệ nhỏ.
* Replay cached attacked mask.
* Không dùng full SAM-PGD cho mọi batch.
* Theo dõi clean mIoU.
* Theo dõi Boundary IoU.
* Theo dõi mask failure rate.
* Chọn checkpoint theo robust validation result.

Gate SAM:

* Clean mIoU giảm không quá 2 điểm.
* Attacked mIoU tăng rõ.
* Clean Boundary IoU không giảm đáng kể.
* Mask failure rate giảm ít nhất 15% tương đối.
* External result không giảm quá ngưỡng.
* Dùng cùng prompt và cùng benchmark.
* Không dùng pseudo-mask chưa review làm ground truth test.

---

### 21.4.6 Huấn luyện SAM-R2

Chỉ thực hiện khi SAM-R1 còn failure cluster ổn định.

Ví dụ:

```text
Object nhỏ
+
Fog severity 4
+
Partial occlusion
→ mask mất hoàn toàn hoặc mất biên
```

Mix tham chiếu:

| Thành phần                        | Tỷ lệ |
| --------------------------------- | ----: |
| General robust replay             |   60% |
| Targeted small-object/fog samples |   25% |
| Clean replay                      |   15% |

Người C quyết định:

* Failure cluster cần repair.
* Loại mask cần ưu tiên.
* Object-size bucket.
* Boundary cases.
* Thin structures.
* Prompt sensitivity.
* Tiêu chí nghiệm thu targeted repair.

---

### 21.4.7 Attack chuyên biệt cho SAM2

Người D xây attack framework; Người C phụ trách phần phụ thuộc SAM2:

* Segmentation attack objective.
* Mask loss target.
* Boundary loss target.
* SAM-PGD.
* Gradient sanity.
* Prompt preservation.
* Partial mask occlusion.
* Hard-mask example.
* Kiểm tra attack làm sai mask nhưng không làm prompt sai.
* Kiểm tra mask transform sau geometric attack.
* Kiểm tra valid region.
* Xác định threshold mask failure.
* Xác định severity phù hợp cho segmentation.

---

### 21.4.8 Segmentation evaluator

Người C xây dựng:

```text
src/evaluation/segmentation_metrics.py
```

Metric bắt buộc:

* IoU.
* Mean IoU.
* Dice score.
* Pixel precision.
* Pixel recall.
* Boundary IoU.
* Boundary F-score.
* Per-class mIoU.
* Per-object IoU.
* Small/medium/large object IoU.
* Prompt consistency.
* Mask confidence drop.
* Mask failure rate.

Một mask được coi là thất bại khi:

* IoU thấp hơn threshold.
* Mask mất hoàn toàn.
* Mask nhầm sang object khác.
* Mask bị tách nghiêm trọng.
* Mask hợp nhất sai nghiêm trọng.
* Boundary sai vượt ngưỡng cho phép.

Người D sử dụng metric SAM để tính degradation, RobustScore, Recovery Rate và comparison dùng chung.

---

### 21.4.9 Benchmark và external evaluation

Người C chịu trách nhiệm chạy và xác nhận:

* Clean SAM validation.
* Clean locked test.
* Boundary stress benchmark.
* Weather/noise benchmark.
* Blur/compression benchmark.
* Occlusion benchmark.
* SAM-PGD benchmark.
* Severity sweep.
* Baseline-versus-robust comparison.
* External segmentation evaluation.
* Same-prompt paired comparison.
* Object-size analysis.
* Boundary failure analysis.
* Mask failure clustering.
* Inference latency.

---

### 21.4.10 Kiểm thử do Người C phụ trách

* Sam2Adapter unit test.
* Mask shape test.
* Prompt consistency test.
* Same-prompt benchmark test.
* Segmentation prediction schema test.
* Mask metric test.
* Boundary IoU test.
* Mask failure-rate test.
* SAM-PGD sanity test.
* Checkpoint load test.
* Clean inference regression test.
* Reproducible training test.
* SAM end-to-end compute test.
* Scientific validation cho SAM.

---

### 21.4.11 Đầu ra bắt buộc của Người C

* Dataset segmentation hợp lệ.
* Quy trình mask review.
* Prompt protocol.
* Runnable Sam2Adapter.
* SAM-B0 checkpoint.
* SAM-R1 checkpoint.
* SAM-R2 khi thực sự cần.
* Segmentation evaluator.
* Hard-mask example set.
* External segmentation result.
* Baseline-versus-robust result.
* SAM Recovery Report input.
* Training curves.
* Checkpoint hash.
* Model registration metadata.
* Payload mask và metric để Người A hiển thị.
* Tài liệu hướng dẫn tái chạy training và benchmark.

---

### 21.4.12 Người C không phụ trách

* Xây dựng WebApp.
* Viết API route.
* Thiết kế database.
* Xây dựng attack recipe framework dùng chung.
* Huấn luyện YOLO11.
* Xây dựng detection evaluator.
* Quản lý WebSocket.
* Xây dựng review UI.
* Dùng pseudo-mask chưa review làm benchmark ground truth.

---

## 21.5 Người D — Data Platform, Attack Engine, Benchmark và pipeline dùng chung

### Vai trò chính

Người D sở hữu toàn bộ nền tảng kỹ thuật dùng chung:

```text
Dataset ingestion và versioning
→ Split và leakage protection
→ Attack Catalog
→ Attack Recipe
→ Attack Composition
→ Generated Dataset
→ Training Dataset Builder
→ Benchmark Protocol
→ Benchmark Runner
→ Robustness/Recovery Metrics
→ Compute Worker
→ Hard Example Bank
```

Người D không huấn luyện thay Người B hoặc Người C, nhưng cung cấp toàn bộ dữ liệu và pipeline để hai người đó huấn luyện.

---

### 21.5.1 Core contracts

Người D chủ trì định nghĩa các contract dùng chung:

* `DetectionPrediction`.
* `SegmentationPrediction`.
* `AttackRecipe`.
* `AttackStep`.
* `AttackMetadata`.
* `GeneratedDatasetVersion`.
* `TrainingDatasetManifest`.
* `BenchmarkProtocol`.
* `BenchmarkRun`.
* `TrainingRunConfig`.
* `MetricEnvelope`.
* `FailureCase`.
* `FailureCluster`.
* `DefenseProfile`.
* `ModelVersionMetadata`.

Người B xác nhận phần detection.

Người C xác nhận phần segmentation.

Người A sử dụng các contract để tạo API và database.

---

### 21.5.2 Dataset ingestion, versioning và split

Người D chịu trách nhiệm:

* Import dataset.
* Tạo DatasetVersion.
* Tạo sample ID ổn định.
* Tạo dataset hash.
* Tạo split manifest.
* Khóa locked test.
* Kiểm tra duplicate.
* Kiểm tra source provenance.
* Ghi anonymization status.
* Ghi annotation type.
* Ghi class list.
* Ghi train/validation/test split.
* Hỗ trợ detection dataset.
* Hỗ trợ segmentation dataset.
* Kiểm tra không có sample overlap giữa các split.
* Kiểm tra không có source image trùng qua nhiều version ngoài ý muốn.

Người B cung cấp quy tắc detection.

Người C cung cấp quy tắc segmentation và mask.

---

### 21.5.3 Attack Catalog

Người D xây attack metadata cho tất cả attack:

* Tên.
* Nhóm.
* Plain-language description.
* Tình huống thực tế.
* Cách ảnh bị thay đổi.
* Cách model có thể thất bại.
* Severity semantics.
* Cost class.
* Compatibility.
* Required modality.
* Required gradient.
* Required annotation.
* Implementation version.
* Default parameters.
* Parameter ranges.
* Estimated runtime.
* Defense suggestion.

Nhóm attack:

* Môi trường và tầm nhìn.
* Chất lượng camera và hình ảnh.
* Che khuất và vật thể lạ.
* Tấn công có chủ đích.

---

### 21.5.4 Attack Recipe và composition

Người D xây dựng:

```text
src/attacks/recipes.py
src/attacks/presets.py
src/pipeline/composition.py
```

Hỗ trợ:

* Single Attack.
* Manual Composition.
* Random N Attacks.
* Random by Group.
* Scenario Preset.
* Auto Sweep.
* Red-Team Search nếu đủ nguồn lực.

Scenario preset:

* Low Visibility.
* Wet Camera.
* Poor Camera Pipeline.
* Partial Obstruction.
* Adversarial Stress.
* Segmentation Boundary Stress.

Công việc cụ thể:

* Recipe schema.
* Ordered steps.
* Recipe validation.
* Compatibility validation.
* Deterministic random.
* Random seed.
* Quota sampling.
* Recipe hashing.
* Cost accounting.
* Severity aggregation.
* Hard cap.
* Storage cap.
* GPU budget cap.
* Cảnh báo expensive composition.
* Chặn attack trùng hoặc xung đột không hợp lý.

---

### 21.5.5 Pipeline sinh dữ liệu attack

Người D xây dựng:

```text
src/pipeline/generator.py
src/pipeline/composition.py
src/pipeline/cache.py
```

Pipeline:

```text
Original sample
→ validate recipe
→ apply ordered attack steps
→ transform annotation
→ save intermediate outputs
→ save final output
→ create manifest
→ calculate hashes
→ cache
```

Mỗi variant phải lưu:

* Source sample ID.
* Dataset version.
* Source split.
* Recipe ID.
* Ordered attack steps.
* Severity.
* Parameter.
* Seed.
* Intermediate artifact.
* Final artifact.
* Annotation transform log.
* Ground-truth hash.
* Implementation version.
* Creation timestamp.
* Cache key.

---

### 21.5.6 Annotation transformation

Người D xây framework transform dùng chung.

#### Detection

* Update bounding box sau geometric transform.
* Clip box vào image boundary.
* Tính visible ratio.
* Loại box không còn hợp lệ theo rule đã chốt.
* Ghi transform log.
* Gọi validation do Người B cung cấp.

#### Segmentation

* Transform mask cùng phép biến đổi với ảnh.
* Giữ instance ID.
* Cập nhật valid region.
* Kiểm tra mask rỗng.
* Ghi transform log.
* Gọi validation do Người C cung cấp.

Người D triển khai pipeline.

Người B và Người C xác nhận tính đúng đắn theo task.

---

### 21.5.7 Training Dataset Builder

Người D xây dựng:

```text
src/training/dataset_builder.py
```

Builder nhận:

* Base dataset version.
* Training split manifest.
* DefenseProfile.
* Attack recipe set.
* Data ratios.
* Severity distribution.
* Failure cluster IDs.
* Global seed.
* Storage budget.
* Online/offline policy.

Builder tạo:

* Clean sample list.
* Generated sample list.
* Source sample mapping.
* Recipe mapping.
* Severity mapping.
* Seed mapping.
* Annotation transform log.
* Class distribution.
* Object-size distribution.
* Leakage report.
* Storage estimate.
* Manifest hash.

Sampling hỗ trợ:

* Random sampling.
* Class-balanced sampling.
* Object-size-balanced sampling.
* Failure-cluster-targeted sampling.
* Severity-distribution sampling.
* Clean replay floor.
* Hard-example replay.
* Maximum variant per source sample.
* Maximum variant per recipe.

---

### 21.5.8 Leakage và dữ liệu hợp lệ

Người D chịu trách nhiệm tự động kiểm tra:

* Không có locked-test sample trong training.
* Không lấy trực tiếp benchmark artifact để retrain.
* Không reuse attack seed của locked benchmark một cách không kiểm soát.
* Không có duplicate giữa training và test.
* Source file tồn tại.
* Ground truth hợp lệ.
* Class mapping hợp lệ.
* Box không rỗng bất thường.
* Mask không rỗng bất thường.
* Data ratio hợp lệ.
* Attack tương thích.
* Manifest hash tái tạo được.
* Storage không vượt budget.
* Mọi artifact có provenance.

Đây là điều kiện bắt buộc trước khi Người B hoặc Người C bắt đầu training.

---

### 21.5.9 Hard Example Bank

Người D xây:

```text
src/training/hard_example_bank.py
```

Lưu các attack đắt hoặc hard example:

* Strong PGD.
* C&W.
* Square Attack.
* Adversarial patch.
* SAM-PGD.
* Failure cluster nghiêm trọng.
* Targeted repair samples.

Hard Example Bank phải lưu:

* Source sample.
* Model version tạo attack.
* Attack objective.
* Attack implementation version.
* Parameter.
* Seed.
* Metric trước và sau attack.
* Failure reason.
* Đối tượng hoặc mask bị ảnh hưởng.
* Quyền sử dụng cho training hay chỉ benchmark.

---

### 21.5.10 BenchmarkProtocol

Người D xây thực thể và validation cho `BenchmarkProtocol`.

Protocol phải khóa:

* Dataset version.
* Sample ID.
* Attack recipe.
* Attack implementation version.
* Severity.
* Parameter range.
* Seed.
* Model preprocessing.
* Metric implementation version.
* Confidence threshold.
* IoU threshold.
* Prompt protocol với SAM.
* Bootstrap configuration.

Chỉ so sánh hai model khi:

* Cùng BenchmarkProtocol.
* Cùng sample.
* Cùng attack recipe.
* Cùng severity.
* Cùng seed.
* Cùng preprocessing rule.
* Cùng metric version.
* Cùng prompt đối với SAM.

---

### 21.5.11 Benchmark runner

Người D xây dựng:

```text
src/pipeline/runner.py
```

Runner hỗ trợ:

* Clean benchmark.
* Single-attack benchmark.
* Composite benchmark.
* Severity sweep.
* Multi-model inference.
* Baseline-versus-defended paired comparison.
* Cache inference.
* Resume job.
* Progress event.
* Error handling.
* Backward compatibility.
* Deterministic mini run.
* External-dataset benchmark.

Runner gọi:

* YOLO adapter và evaluator của Người B.
* SAM2 adapter và evaluator của Người C.

---

### 21.5.12 Robustness và recovery evaluator dùng chung

Người D xây dựng:

```text
src/evaluation/robustness_metrics.py
src/evaluation/model_comparison.py
src/evaluation/recovery_metrics.py
src/evaluation/bootstrap.py
```

Metric dùng chung:

* Degradation.
* Robustness Accuracy theo severity.
* RobustScore.
* Recovery Rate.
* Mean attacked performance.
* Clean-to-robust trade-off.
* Critical-scenario regression.
* Paired metric delta.
* Bootstrap 95% confidence interval.
* Seen-versus-unseen attack comparison.
* External generalization comparison.

Phân chia metric:

```text
Người B:
mAP, AP, detection ASR và detection failure.

Người C:
mIoU, Boundary IoU, Dice và mask failure.

Người D:
Degradation, RobustScore, Recovery, bootstrap và model comparison.
```

---

### 21.5.13 Generic training orchestration

Người D xây phần compute dùng chung:

```text
src/training/base.py
src/training/registry.py
src/training/report.py
```

Interface:

```python
class ModelTrainer:
    def validate_config(self, config): ...
    def estimate(self, config): ...
    def prepare_data(self, config): ...
    def train(self, config, callbacks): ...
    def evaluate_checkpoint(self, checkpoint): ...
    def export_checkpoint(self, checkpoint): ...
    def metadata(self): ...
```

Phân công:

* Người D xây `ModelTrainer` interface và worker execution.
* Người B cài đặt `YoloTrainer`.
* Người C cài đặt `Sam2Trainer`.
* Người A xây Training API, database state và WebSocket.

TrainingRun states:

```text
DRAFT
VALIDATING
ESTIMATING
QUEUED
PREPARING_DATA
TRAINING
VALIDATING_CHECKPOINT
EXPORTING
REGISTERING_MODEL
COMPLETED
FAILED
CANCELLED
BUDGET_EXCEEDED
```

---

### 21.5.14 Kiểm thử do Người D phụ trách

* Attack metadata test.
* Compatibility test.
* Severity mapping test.
* Recipe ordering test.
* Deterministic random test.
* Quota sampling test.
* Recipe hash test.
* Cost estimate test.
* Data leakage test.
* Duplicate detection test.
* Annotation transformation test.
* Cache hit/miss test.
* Benchmark determinism test.
* Paired comparison test.
* Recovery formula test.
* Bootstrap test.
* TrainingRun state-transition test.
* Integration test giữa attack, data và benchmark.
* Scientific reproducibility test.

---

### 21.5.15 Đầu ra bắt buộc của Người D

* Dataset ingestion pipeline.
* DatasetVersion và split manifest.
* Leakage validator.
* Attack Catalog.
* Attack Recipe schema.
* Composition engine.
* Scenario presets.
* Random recipe generator.
* Attack generation pipeline.
* GeneratedDatasetVersion.
* Annotation transformation framework.
* Cache.
* TrainingDatasetBuilder.
* Hard Example Bank.
* BenchmarkProtocol.
* Benchmark runner.
* Robustness evaluator.
* Recovery evaluator.
* Bootstrap evaluator.
* Model comparison engine.
* Generic ModelTrainer interface.
* Compute worker.
* Payload và event contract để Người A tích hợp.

---

### 21.5.16 Người D không phụ trách

* Xây dựng frontend.
* Viết Web API route.
* Thiết kế giao diện.
* Huấn luyện và chốt checkpoint YOLO.
* Huấn luyện và chốt checkpoint SAM2.
* Tự quyết định detection hyperparameter.
* Tự quyết định segmentation prompt protocol.
* Render bounding box hoặc mask trên WebApp.

---

## 21.6 Ma trận sở hữu công việc

| Hạng mục                             | Người chịu trách nhiệm chính | Người phối hợp                      |
| ------------------------------------ | ---------------------------- | ----------------------------------- |
| Frontend WebApp                      | A                            | B, C và D cung cấp payload          |
| Backend API                          | A                            | D cung cấp service contract         |
| Database và migration                | A                            | D góp ý data lineage                |
| WebSocket và job status              | A                            | D phát compute event                |
| Review và retraining workflow        | A                            | B, C và D cung cấp dữ liệu          |
| Report và export                     | A                            | D cung cấp comparison result        |
| Dataset ingestion dùng chung         | D                            | B và C xác nhận chuyên môn          |
| Dataset split và leakage             | D                            | B và C review                       |
| KITTI/BDD detection protocol         | B                            | D triển khai manifest               |
| Cityscapes/BDD segmentation protocol | C                            | D triển khai manifest               |
| Attack Catalog                       | D                            | B và C kiểm tra compatibility       |
| Attack Recipe và composition         | D                            | B và C kiểm tra model-specific rule |
| Attack generation                    | D                            | B/C cung cấp model objective        |
| Detection white-box attack           | B                            | D tích hợp vào engine               |
| SAM-PGD                              | C                            | D tích hợp vào engine               |
| Bounding-box transform               | D                            | B nghiệm thu                        |
| Mask transform                       | D                            | C nghiệm thu                        |
| TrainingDatasetBuilder               | D                            | B và C cung cấp data ratio          |
| Generic trainer interface            | D                            | B và C cài trainer                  |
| YoloTrainer                          | B                            | D hỗ trợ orchestration              |
| Sam2Trainer                          | C                            | D hỗ trợ orchestration              |
| YOLO-B0/R1/R2                        | B                            | D cung cấp data/benchmark           |
| SAM-B0/R1/R2                         | C                            | D cung cấp data/benchmark           |
| Detection metrics                    | B                            | D tổng hợp robustness               |
| Segmentation metrics                 | C                            | D tổng hợp robustness               |
| RobustScore và Recovery Rate         | D                            | B và C xác nhận input               |
| Bootstrap và paired comparison       | D                            | B và C phân tích kết quả            |
| Benchmark runner                     | D                            | B/C cung cấp adapter                |
| Model registry API                   | A                            | B/C cung cấp metadata, D validate   |
| Model lineage database               | A                            | D cung cấp lineage contract         |
| YOLO overlays                        | A                            | B cung cấp prediction               |
| SAM overlays                         | A                            | C cung cấp mask                     |
| UI integration test                  | A                            | B/C/D cung cấp test fixtures        |
| Scientific validation YOLO           | B                            | D                                   |
| Scientific validation SAM            | C                            | D                                   |
| Reproducibility toàn pipeline        | D                            | A, B và C                           |

---

## 21.7 Ranh giới giữa bốn người

### A và D

```text
A sở hữu Web backend.
D sở hữu compute backend.
```

A làm:

* API.
* Database.
* Queue request.
* Job status.
* WebSocket.
* Persistence.
* Export.

D làm:

* Attack computation.
* Dataset generation.
* Benchmark execution.
* Robustness computation.
* Training worker execution.

### B và D

```text
B sở hữu mô hình YOLO và detection logic.
D sở hữu pipeline đưa dữ liệu vào YOLO.
```

B làm:

* Adapter.
* Trainer.
* Detection objective.
* Detection evaluator.
* Checkpoint.

D làm:

* Recipe.
* Generator.
* Manifest.
* Benchmark runner.
* Data builder.

### C và D

```text
C sở hữu mô hình SAM2 và segmentation logic.
D sở hữu pipeline đưa dữ liệu vào SAM2.
```

C làm:

* Adapter.
* Prompt protocol.
* Trainer.
* Segmentation objective.
* Segmentation evaluator.
* Checkpoint.

D làm:

* Recipe.
* Generator.
* Mask transform framework.
* Manifest.
* Benchmark runner.
* Data builder.

### A với B và C

A không tự suy luận cách render output.

B và C phải cung cấp:

* Schema.
* Example payload.
* Fixture.
* Ý nghĩa từng field.
* Unit.
* Failure status.
* Overlay convention.

A chịu trách nhiệm hiển thị đúng các dữ liệu đó.

---

## 21.8 Hợp đồng bàn giao bắt buộc

### Người D bàn giao đầu tiên

* `AttackRecipe` schema.
* `GeneratedDatasetManifest` schema.
* `TrainingDatasetManifest` schema.
* `BenchmarkProtocol` schema.
* Job request và event schema.
* Common metric envelope.
* Recipe validation service.
* Mock benchmark result.

### Người B bàn giao cho A và D

* YOLO adapter interface.
* DetectionPrediction schema.
* Detection metric schema.
* Example prediction payload.
* Example failure payload.
* YoloTrainer interface.
* Checkpoint metadata.
* Detection overlay convention.

### Người C bàn giao cho A và D

* Sam2Adapter interface.
* SegmentationPrediction schema.
* Prompt schema.
* Segmentation metric schema.
* Example mask payload.
* Example failure payload.
* Sam2Trainer interface.
* Checkpoint metadata.
* Mask overlay convention.

### Người A bàn giao cho cả nhóm

* API contract.
* Database entity diagram.
* Job status contract.
* WebSocket event contract.
* Error-response contract.
* Mock UI.
* Integration environment.
* E2E test flow.

---

## 21.9 Thứ tự triển khai và tích hợp

### Bước 1 — Chốt contract

Người D chủ trì, cả nhóm tham gia:

* Prediction schema.
* Recipe schema.
* Manifest schema.
* BenchmarkProtocol.
* Metric unit.
* Job event.
* Model metadata.
* Definition of Done.

### Bước 2 — Phát triển song song

Người A:

* Dựng frontend bằng mock.
* Dựng backend API skeleton.
* Dựng database schema.

Người B:

* Hoàn thiện YOLO adapter.
* Chuẩn bị YOLO-B0.
* Xây detection evaluator.

Người C:

* Chuẩn bị mask dataset.
* Hoàn thiện Sam2Adapter.
* Xây segmentation evaluator.

Người D:

* Xây Attack Catalog.
* Xây recipe engine.
* Xây dataset generator.
* Xây BenchmarkProtocol.

### Bước 3 — Hoàn thiện YOLO end-to-end

* D tạo split và defense data.
* B train YOLO-B0.
* D chạy locked benchmark.
* B train YOLO-R1.
* D chạy lại cùng benchmark.
* A tích hợp kết quả vào WebApp.
* C tiếp tục SAM song song.

### Bước 4 — Hoàn thiện SAM end-to-end

* C chốt mask dataset và prompt.
* D tạo split và defense data.
* C train SAM-B0.
* D chạy locked benchmark.
* C train SAM-R1.
* D chạy lại cùng benchmark.
* A tích hợp SAM workspace.

### Bước 5 — Hoàn thiện review và retraining

* D sinh failure cluster.
* A hiển thị failure cluster.
* A tạo retraining backlog.
* B hoặc C xác nhận DefenseProfile theo mô hình.
* D tạo TrainingDatasetManifest.
* A tạo TrainingRun qua API.
* B hoặc C chạy trainer.
* D chạy paired benchmark.
* A hiển thị Recovery Report.

### Bước 6 — Kiểm thử và ổn định

* A chạy Web E2E.
* B chạy YOLO scientific validation.
* C chạy SAM scientific validation.
* D chạy reproducibility, leakage và benchmark tests.
* Cả nhóm chạy demo rehearsal.

---

## 21.10 Definition of Done theo từng người

### Người A hoàn thành khi

* Frontend và backend Web chạy được.
* Không còn hard-code model.
* Database và migration đầy đủ.
* API hoạt động.
* WebSocket hoạt động.
* YOLO và SAM đều hiển thị được.
* Review và retraining workflow hoạt động.
* Baseline/defended comparison hoạt động.
* Export report hoạt động.
* E2E test vượt qua.

### Người B hoàn thành khi

* YOLO adapter chạy được.
* Có YOLO-B0 và YOLO-R1.
* Detection evaluator hoàn chỉnh.
* Có locked benchmark.
* Có external BDD100K result.
* Có reproducible checkpoint.
* Gate YOLO được đánh giá.
* Payload đủ để WebApp hiển thị.

### Người C hoàn thành khi

* Có ground-truth mask hợp lệ.
* Sam2Adapter chạy được.
* Prompt protocol được khóa.
* Có SAM-B0 và SAM-R1.
* Segmentation evaluator hoàn chỉnh.
* Có external segmentation result.
* Có reproducible checkpoint.
* Gate SAM được đánh giá.
* Payload đủ để WebApp hiển thị.

### Người D hoàn thành khi

* Dataset version và split hoạt động.
* Leakage validator hoạt động.
* Attack Recipe hoạt động.
* Composition deterministic.
* Generated dataset có manifest.
* TrainingDatasetBuilder hoạt động.
* BenchmarkProtocol được khóa.
* Benchmark runner chạy được với cả YOLO và SAM.
* RobustScore, Recovery Rate và bootstrap hoạt động.
* Compute worker tích hợp được với backend Web.
* Toàn bộ artifact truy ngược được nguồn gốc.

---

## 21.11 Kết quả phân công cuối cùng

```text
NGƯỜI A — FULL-STACK WEB
Frontend + Backend + Database + API + WebSocket
+ Review workflow + Report + Tích hợp sản phẩm.

NGƯỜI B — YOLO11
Detection dataset protocol + YOLO adapter
+ YOLO-B0/R1/R2 + Detection metrics
+ YOLO attacks + External evaluation.

NGƯỜI C — SAM2
Segmentation dataset + Mask validation + Prompt protocol
+ Sam2Adapter + SAM-B0/R1/R2
+ Segmentation metrics + SAM-PGD + External evaluation.

NGƯỜI D — DATA, ATTACK VÀ BENCHMARK
Dataset version + Split + Leakage
+ Attack Catalog + Recipe + Composition
+ Generated Dataset + TrainingDatasetBuilder
+ BenchmarkProtocol + Runner
+ Robustness/Recovery + Compute worker.
```

Cách chia này bảo đảm:

* Một người sở hữu toàn bộ Web.
* Một người sở hữu toàn bộ YOLO.
* Một người sở hữu toàn bộ SAM.
* Một người sở hữu toàn bộ dữ liệu, attack và benchmark dùng chung.
* Không có phần việc quan trọng nào không có người chịu trách nhiệm.
* Không có hai người cùng chịu trách nhiệm chính cho một module.
* Các thành viên có thể phát triển song song sau khi chốt contract.
n storytelling.

---

# 22. Lộ trình triển khai tám tuần

## Sprint 0 — Correctness và contracts

**Thời lượng:** 3–4 ngày.

### Công việc

- Sửa Prediction bug.
- Chuẩn hóa metric units.
- Sửa review threshold.
- Chốt prediction schemas.
- Chốt BenchmarkProtocol.
- Chốt split policy.
- Model selector thật.
- Frontend CI.
- Leakage tests.

### Nghiệm thu

- Unit test bắt bug.
- UI hiển thị 42%, không phải 0.42%.
- Threshold 30% đúng.
- Benchmark deterministic.
- Model/version chọn thật.

## Sprint 1 — UI shell và Attack Catalog

**Thời lượng:** Tuần 1.

### Công việc

- Layout hai vùng.
- YOLO/SAM selector.
- Attack grouping.
- Plain-language explanation.
- Compatibility.
- Presets.
- Workspace năm phần bằng mock.
- Five metrics mock.

### Nghiệm thu

- Mỗi attack có đủ giải thích.
- Incompatible attack bị lọc.
- Severity có nhãn.
- Không còn panel metric hẹp.

## Sprint 2 — Composition và dataset generation

**Thời lượng:** Tuần 2.

### Công việc

- Manual.
- Random N.
- Random by group.
- Ordered recipe.
- Validation.
- Estimate.
- Manifest.
- YOLO box transform.
- SAM mask transform.
- Cache.
- Preview.

### Nghiệm thu

Chạy được:

```text
Random 6 compatible attacks
Random theo quota nhóm
Low Visibility
Fog → Occlusion → JPEG
```

- Cùng seed tạo cùng recipe.
- Hard cap hoạt động.
- Manifest đầy đủ.

## Sprint 3 — YOLO end-to-end

**Thời lượng:** Tuần 3–4.

### Công việc

- KITTI loader/split.
- YOLO-B0.
- YOLO-R1.
- Locked benchmark.
- BDD100K.
- Five metrics.
- Baseline vs robust.
- Recovery report.
- Retraining backlog/DefenseProfile tối thiểu.

### Nghiệm thu

```text
Baseline
→ attack failure
→ retraining request
→ robust version
→ same benchmark
→ recovery
```

## Sprint 4 — SAM inference và evaluation

**Thời lượng:** Tuần 5.

### Công việc

- Dataset mask.
- Loader.
- Sam2Adapter.
- SegmentationPrediction.
- GT mask viewer.
- mIoU.
- Boundary IoU.
- Mask failure rate.
- Five metrics SAM.
- Prompt protocol.

### Nghiệm thu

- GT mask thật.
- Clean/attacked masks.
- Metric chạy.
- Không còn surrogate-only trên benchmark path.

## Sprint 5 — SAM defense loop

**Thời lượng:** Tuần 6.

### Công việc

- SAM-B0.
- SAM-R1.
- Attack benchmark.
- External segmentation test.
- Baseline vs robust.
- Recovery report.
- Training lineage.

### Nghiệm thu

- Same-prompt paired comparison.
- Gate SAM.
- External result.
- Viewer hoàn chỉnh.

## Sprint 6 — Review workflow và demo polishing

**Thời lượng:** Tuần 7.

### Công việc

- Failure cluster.
- Retraining backlog.
- DefenseProfile.
- TrainingRun history.
- Reviewer decisions.
- Export.
- Narrative.
- Precomputed benchmarks.
- Demo presets.

### Nghiệm thu

Người không chuyên trả lời được:

1. Model tốt đến đâu?
2. Tình huống nào làm hỏng?
3. Object nào thất bại?
4. Defense là gì?
5. Model mới cải thiện bao nhiêu?
6. Clean tradeoff?
7. External generalization?
8. Rủi ro còn lại?

## Sprint 7 — Stabilization

**Thời lượng:** Tuần 8.

### Công việc

- E2E tests.
- Scientific sanity.
- UI tests.
- Performance.
- Cache.
- Error handling.
- Demo rehearsal.
- Documentation.
- Backup artifacts.

### Nghiệm thu

- Full happy path ổn định.
- Seeded mini run ổn định.
- Precomputed report load được.
- Không có blocker P0.
- DoD được ký xác nhận.


# 23. Kịch bản demo

## 23.1 Bước 1 — Nêu vấn đề

```text
Model nhận diện tốt trên ảnh sạch,
nhưng điều kiện thực tế có sương, rung, che khuất,
nén ảnh hoặc can thiệp có chủ đích.
```

## 23.2 Bước 2 — Baseline

- Chọn YOLO11 clean-v1.
- Hiển thị clean AP.
- Hiển thị prediction đúng.
- Giải thích dataset và locked benchmark.

## 23.3 Bước 3 — Crash-test scenario

Chọn:

```text
Low Visibility
Fog severity 4
Contrast reduction severity 2
Motion blur severity 1
```

Hiển thị:

- Ý nghĩa từng step.
- Ordered recipe.
- Seed.
- Estimate.

## 23.4 Bước 4 — Failure evidence

Chạy live 3–5 ảnh hoặc dùng artifact precomputed.

Mở một pedestrian bị miss:

- GT.
- Clean prediction.
- Attacked prediction.
- Confidence.
- IoU.
- Degradation.
- ASR.
- Lý do dễ hiểu.

## 23.5 Bước 5 — Review

Reviewer:

```text
Yêu cầu retrain

Lý do:
Pedestrian nhỏ ở xa thường bị bỏ sót
trong điều kiện sương mù mạnh.
```

## 23.6 Bước 6 — Defense

Không train live.

Mở TrainingRun hoàn tất:

- Parent model.
- Defense profile.
- Data composition.
- Clean ratio.
- Weather ratio.
- Occlusion ratio.
- Adversarial ratio.
- GPU-hours.
- Checkpoint.
- Gate.

## 23.7 Bước 7 — Re-test

Chọn:

```text
YOLO clean-v1
YOLO robust-v2
```

Cùng BenchmarkProtocol.

## 23.8 Bước 8 — Recovery Report

Ví dụ:

```text
Attacked AP: 39.2 → 55.8
ASR: 61% → 34%
RobustScore: 62 → 76
Clean AP: 68.5 → 67.8
External AP: ...
```

## 23.9 Bước 9 — Kết luận an toàn

```text
Đây chưa phải chứng nhận triển khai.

AdverTest cung cấp bằng chứng:
model yếu ở đâu,
đã được cải thiện thế nào,
và rủi ro nào vẫn cần con người đánh giá.
```

## 23.10 Demo SAM tùy chọn

Sau demo YOLO:

- Chuyển sang SAM.
- Chọn Segmentation Boundary Stress.
- Hiển thị GT mask.
- Clean mask.
- Attacked mask.
- Defended mask.
- mIoU, Boundary IoU, failure rate.
- Nhấn mạnh cùng prompt và cùng benchmark.

---

# 24. Ưu tiên, rủi ro và Definition of Done

## 24.1 P0 — Bắt buộc

1. Correctness bugs.
2. Benchmark/split contracts.
3. Model selector.
4. Attack explanation.
5. Recipe/preset.
6. YOLO baseline vs robust.
7. Locked re-benchmark.
8. Before/after metrics.
9. UI hai vùng.
10. Demo crash-test → defense → retest.
11. SAM runnable với mask thật.
12. SAM baseline vs robust.

## 24.2 P1 — Hoàn thiện kỹ thuật

1. Random by group.
2. External tests.
3. Training lineage.
4. Advanced metrics.
5. Failure cluster.
6. Retraining workflow.
7. Export.
8. CI/E2E/scientific validation.

## 24.3 P2 — Sau luồng chính

1. Red-Team Search.
2. Automated failure clustering.
3. Active sampling.
4. CI robustness gate.
5. Grad-CAM/advanced explainability.
6. 3D PointPillars.
7. BEVFusion.
8. Multi-sensor defense.

## 24.4 Rủi ro chính và giảm thiểu

| Rủi ro | Hậu quả | Giảm thiểu |
|---|---|---|
| Leakage | Kết quả defense giả tạo | Locked manifest + automated leakage test |
| Metric unit sai | UI/review sai | Ratio/pct fields tách biệt |
| SAM không có GT mask | mIoU không hợp lệ | Dùng Cityscapes/BDD/internal reviewed masks |
| Train quá nặng | Trễ sprint | Decoder-only SAM, freeze, AMP, precompute |
| Overfit attack | Robust một chiều | Robust mix + unseen/external test |
| Clean regression | Model không thực dụng | Clean replay + gate ≤2 điểm |
| Combinatorial explosion | Vượt GPU/storage | Sample K + estimate + hard cap |
| Incompatible attack | Run lỗi/kết quả vô nghĩa | Compatibility validation trước enqueue |
| Demo live không ổn định | Demo thất bại | Seeded mini run + precomputed full results |
| UI quá tải | Người dùng không hiểu | Five metrics + advanced drawer |
| Pseudo-mask bị coi là GT | Kết luận sai | Review status + benchmark validator |
| Recipe không reproducible | Không audit được | Hash + seed + implementation version |

## 24.5 Definition of Done toàn giai đoạn

### Product

- Người dùng hiểu attack gắn với tình huống thực tế.
- Chọn YOLO hoặc SAM.
- Dataset/model/attack lọc đúng.
- Chạy single, manual, random N, random by group và preset.
- Có estimate.
- Có recipe preview.
- Có five-part workspace.
- Có five metrics theo mode.
- Có advanced metrics drawer.
- Có comparison baseline/defended.
- Có narrative và recovery report.

### Data

- Split policy khóa.
- Locked benchmark versioned.
- Defense data chỉ từ training split.
- Generated data có manifest.
- Annotation transform đúng.
- Hash/provenance đầy đủ.
- External test riêng.

### YOLO

- GT box thật.
- YOLO-B0.
- YOLO-R1.
- Same benchmark.
- Gate.
- BDD100K result.
- RecoveryReport.

### SAM

- GT mask thật.
- Runnable adapter.
- Fixed prompt protocol.
- SAM-B0.
- SAM-R1.
- Same benchmark.
- mIoU/Boundary IoU/failure rate.
- External result.
- RecoveryReport.

### Training và lineage

- DefenseProfile.
- TrainingDatasetManifest.
- TrainingRun.
- Checkpoints.
- Parent-child ModelVersion.
- Gate result.
- Audit.

### Quality

- Unit/integration/E2E.
- Scientific sanity.
- Frontend CI.
- No P0 blocker.
- Deterministic seeded run.
- Demo rehearsal.
- Documentation.

### Demo kể được trọn câu chuyện

```text
Ảnh + GT
→ attacked input
→ clean prediction
→ attacked failure
→ metric degradation
→ defense data
→ robust fine-tune
→ new ModelVersion
→ same benchmark
→ measured recovery
→ residual risks
```

---

# 25. Phụ lục: kiến trúc mở rộng 3D sau giai đoạn hiện tại

## 25.1 Trạng thái

3D không nằm trong sprint hiện tại. Phần này chỉ định nghĩa hướng mở rộng để kiến trúc hiện tại không chặn tương lai.

## 25.2 Thứ tự đề xuất

1. Hoàn thiện nuScenes loader.
2. Native 3D GT boxes.
3. 3D prediction contract.
4. 3D evaluator.
5. PointPillars adapter/inference.
6. PointPillars baseline.
7. LiDAR corruption.
8. PointPillars robust mix.
9. BEV viewer.
10. Sau cùng mới BEVFusion.

## 25.3 Data contract 3D dự kiến

```text
boxes3d
center
dimensions
yaw
velocity
label
confidence
coordinate_frame
calibration_version
```

## 25.4 Metric dự kiến

- nuScenes mAP.
- NDS.
- mATE.
- mASE.
- mAOE.
- mAVE.
- mAAE.
- Per-class AP.
- Point-density-conditioned AP.
- Corruption-wise retention.

## 25.5 Attack 3D dự kiến

- Point/beam drop.
- Sector drop.
- LiDAR fog.
- LiDAR snow.
- Intensity/range corruption.
- Camera/LiDAR dropout cho fusion.

## 25.6 Nguyên tắc không ảnh hưởng giai đoạn hiện tại

- Không phân công sprint.
- Không build UI 3D.
- Không đưa 3D vào DoD hiện tại.
- Không dùng 3D làm dependency cho YOLO/SAM.
- Chỉ giữ enum, schema extension và compatibility design.

---

## Kết luận thực thi

Thứ tự tối ưu để giảm rủi ro:

```text
Correctness + contracts
→ Attack Catalog + Recipe
→ YOLO end-to-end defense loop
→ SAM runnable + segmentation metrics
→ SAM defense loop
→ Review/lineage/report
→ Stabilization + demo
```

Kế hoạch này là tài liệu nguồn duy nhất cho giai đoạn YOLO11 + SAM2. Mọi thay đổi phạm vi, metric, split, recipe, training ratio hoặc acceptance gate phải được cập nhật bằng version và ghi rõ lý do.
