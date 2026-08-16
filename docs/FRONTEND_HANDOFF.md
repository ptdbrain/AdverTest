# 📋 Tài liệu Bàn giao Frontend (Frontend Handoff Spec) — AdverTest

> **Mục tiêu:** Xây dựng giao diện web tối giản, trực quan, tập trung vào dữ liệu và hình ảnh đối chiếu thực tế.  
> **Nguyên tắc thiết kế cốt lõi:** **Tối giản — Ít chữ — Không dùng theme màu mè/neon — Trực quan tuyệt đối.**

---

## 1. Bố cục Tổng thể (2-Column Layout)

Màn hình chia làm **2 phần độc lập** (vừa vặn 1 khung nhìn desktop, không cần cuộn trang lớn):

- **Cột Trái (Config Panel — ~30% - 35% chiều rộng):** Khu vực điều khiển, cấu hình bài toán, chọn model, dataset, chọn attack và thanh progress bar khi tải.
- **Cột Phải (Image & Evidence Grid — ~65% - 70% chiều rộng):** Khu vực hiển thị trung tâm gồm **4 ô ảnh đối chiếu** và các chỉ số đo lường nhanh ($AP_{50}, mAP_{50-95}$).

```
+-----------------------------------------------------------------------------------------------+
| Header: [AdverTest] | Task: 2D Detection | Status: Ready | [SIMULATION ONLY]                  |
+-----------------------------------+-----------------------------------------------------------+
| CỘT TRÁI: CONFIG & CONTROLS (~30%)| CỘT PHẢI: IMAGE & EVIDENCE GRID (~70%)                    |
|                                   |                                                           |
| 1. Chọn Bài toán (Task)           |  +-----------------------------+-----------------------+  |
|    [2D Det] [Segm] [3D Det]       |  | 1. Ảnh gốc + Ground Truth   | 2. Ảnh bị tấn công    |  |
|                                   |  |    (BBox Ground Truth)      |    (Perturbed Input)  |  |
| 2. Chọn Model / Checkpoint        |  +-----------------------------+-----------------------+  |
|    [YOLO11n] [YOLO11s] ...        |  | 3. Ảnh gốc + YOLO Predict   | 4. Ảnh tấn công+YOLO  |  |
|    (Tự tải Ultralytics nếu thiếu) |  |    (Clean Prediction)       |    (Attacked Pred)    |  |
|                                   |  +-----------------------------+-----------------------+  |
| 3. Chọn Bộ ảnh (Dataset)          |  | METRICS DƯỚI ẢNH 3 (NHỎ):   | METRICS DƯỚI ẢNH 4:   |  |
|    [KITTI] [COCO] [Synthetic]     |  | AP50: 0.89 | mAP50-95: 0.64 | AP50: 0.24 | D: -73%  |  |
|                                   |  | IoU: 0.88  | 5 boxes        | mAP50-95: 0.12 | Lost 4 |  |
| 4. Phương thức Tấn công           |  +-----------------------------+-----------------------+  |
|    Nhóm: [White] [Gray] [Black]   |                                                           |
|    Card Attack (Bấm mở rộng 1-5,  |                                                           |
|    thu gọn kèm thanh 5 ô nhỏ)     |                                                           |
|                                   |                                                           |
| [================ 65% ==========] |  <-- Progress bar khi tải weight / dữ liệu lâu             |
| [ ▶ Chạy Kiểm Thử (Run Test) ]    |                                                           |
+-----------------------------------+-----------------------------------------------------------+
```

---

## 2. Chi tiết Cột Phải: Lưới 4 Hình Ảnh & Metrics Nhỏ Gọn

Khu vực này hiển thị lưới $2 \times 2$ lớn, chiếm trọn tầm nhìn:

### 2.1. Bốn ô hình ảnh (2x2 Grid)
1. **Ảnh 1 (Góc trên - Trái):** `Ảnh gốc + Bounding Box từ Ground Truth`
   - Hiển thị ảnh sạch nguyên bản kèm nhãn hộp bao chuẩn (Ground Truth BBox, nét viền xanh lá mỏng).
2. **Ảnh 2 (Góc trên - Phải):** `Ảnh bị tấn công`
   - Hiển thị bức ảnh sau khi bị áp nhiễu / tấn công (không vẽ BBox) để quan sát mức độ mắt thường nhìn thấy nhiễu.
3. **Ảnh 3 (Góc dưới - Trái):** `Ảnh gốc + Bounding Box từ YOLO`
   - Hiển thị ảnh sạch kèm kết quả dự đoán của model (YOLO Predicted BBox, nét viền xanh dương/trắng).
4. **Ảnh 4 (Góc dưới - Phải):** `Ảnh bị tấn công + Bounding Box từ YOLO`
   - Hiển thị ảnh bị tấn công kèm kết quả dự đoán của model (thấy rõ box bị mất, box bị lệch, hoặc box sinh sai).

### 2.2. Khu vực Metrics Nhỏ Gọn (Ưu tiên AP50 & mAP50-95)
Thiết kế dạng hàng ngang các chip/badge siêu nhỏ, nằm ngay sát dưới chân Ảnh 3 và Ảnh 4:

- **Dưới Ảnh 3 (Clean Metrics):**
  - `AP50: 0.89`
  - `mAP50-95: 0.64`
  - `IoU: 0.88`
  - `5 boxes`
- **Dưới Ảnh 4 (Attacked Metrics):**
  - `AP50: 0.24`
  - `mAP50-95: 0.12`
  - `D: -73.0%` *(Mức suy giảm degradation, viền hoặc màu đỏ nhạt)*
  - `Lost: 4 boxes`

---

## 3. Chi tiết Cột Trái: Khu vực Cấu hình & Thanh Tiến Độ

### 3.1. Chọn Bài toán (Task)
- Dạng nút chọn nhanh (Segmented Control / Radio Pill):
  - **`2D Object Detection`** *(Mặc định — Khả dụng)*
  - **`Instance Segmentation`**
  - **`3D Object Detection`**

### 3.2. Chọn Model / Checkpoint (On-demand Download)
- Danh sách các phiên bản checkpoint:
  - Họ YOLO: `YOLO11n` (Nano), `YOLO11s` (Small), `YOLO11m` (Medium), `YOLO11l` (Large), `YOLO11x` (Extra Large).
- **Cơ chế tải tự động (Lazy Download):**
  - Checkpoint không cần lưu sẵn trong repo.
  - Khi user chọn bản chưa có sẵn $\rightarrow$ hiển thị icon tải nhỏ `⬇`.
  - Khi bấm chạy, backend sẽ tự động tải file trọng số từ `ultralytics`.

### 3.3. Chọn Bộ Dữ liệu & Duyệt Ảnh (Dataset & Sample)
- Chọn Dataset: `KITTI` (nạp/tải tự động qua `torchvision`), `COCO`, hoặc `Synthetic Shapes`.
- Chọn nhanh Sample ID hoặc nút `Ảnh tiếp theo ➡` để duyệt nhanh qua các frame.

### 3.4. Chọn Phương thức Tấn công & Tương tác Severity (1-5)
- **Phân nhóm hiển thị (Tabs / Filters):**
  - Chia làm 3 tab/nhóm: **`White-box`**, **`Gray-box`**, **`Black-box`** (Dễ duyệt, không ràng buộc cứng).
- **Danh sách Attack Cards:**
  - *White-box:* `FGSM`, `PGD`, `MI-FGSM`, `CW-L2`, `TOG`, `DPatch`.
  - *Gray-box / Physical / Weather:* `Depth Fog`, `Depth Rain`, `Object Occlusion`, `Sensor Fault`.
  - *Black-box / Corruptions:* `Gaussian Noise`, `Motion Blur`, `Defocus Blur`, `Square Attack`.
- **Tương tác Phóng to / Thu nhỏ chọn mức độ (Accordion/Expand):**
  - **Trạng thái bình thường (Collapsed):** Thẻ chỉ hiện tên và **thanh 5 ô nhỏ** thể hiện mức độ đang chọn (ví dụ: `[■][■][■][ ][ ]` = mức 3).
  - **Khi bấm vào thẻ (Expanded):** Thẻ phóng to ra tại chỗ, hiển thị 5 nút chọn mức độ từ `1` đến `5` (kèm thông số như $\epsilon$, blur radius,...).
  - **Khi chọn xong mức độ (hoặc bấm ra ngoài):** Thẻ tự động thu nhỏ lại, cập nhật thanh 5 ô nhỏ hiển thị mức severity vừa chọn.

### 3.5. Thanh Tiến Độ (Progress Bar) cho các Tác vụ Lâu
> **Bắt buộc:** Mọi tác vụ kéo dài (tải weight, tải dataset, chạy inference nhiều ảnh) đều phải hiển thị Progress Bar rõ ràng, tránh để người dùng tưởng treo ứng dụng.

- **Vị trí hiển thị:** Đặt ngay phía trên nút bấm "Chạy Kiểm Thử" (hoặc inline ngay tại thẻ đang thao tác).
- **Quy cách hiển thị:**
  - Thanh tiến độ mảnh (chiều cao 4-6px), nền xám trung tính (`bg-slate-200`), thanh chạy màu xanh thép/xanh dương (`bg-blue-600`).
  - Đi kèm dòng text trạng thái ngắn và phần trăm cụ thể:
    - *Khi tải weights:* `"Đang tải YOLO11x weights: 45MB / 130MB (35%)"`
    - *Khi tải dataset:* `"Đang tải KITTI validation: 60%"`
    - *Khi chạy inference:* `"Đang kiểm thử: 4/10 ảnh (40%)"`

### 3.6. Nút Thao tác (Action Button)
- Nút bấm to, rõ ràng: **`[ ▶ Chạy Kiểm Thử / Run ]`**.
- Vô hiệu hóa (disable) nhẹ kèm icon quay khi tác vụ đang chạy.

---

## 4. Quy chuẩn Giao diện & Thẩm mỹ (Strict UI/UX Rules)

- 🚫 **TUYỆT ĐỐI KHÔNG DÙNG THEME MÀU MÈ:**
  - Không dùng màu nền tối tím (purple/violet dark theme), không dùng gradient lòe loẹt, không dùng viền neon phát sáng.
- ✅ **Bảng màu trung tính & Tinh tế:**
  - Nền: Trắng sáng (`#FFFFFF`) hoặc Slate xám sáng (`#F8FAFC`), hoặc Dark theme trung tính Slate/Zinc (`#0F172A`, `#1E293B`).
  - Viền: Mỏng, sắc nét (`border-slate-200` hoặc `border-slate-700`).
  - Điểm nhấn (Accents): Đơn sắc hoặc màu chức năng nhã nhặn (Xanh thép, Xanh lá cho GT, Xanh dương cho Pred, Đỏ nhẹ cho Degradation).
- ✍️ **Thật ít chữ, bớt mô tả:**
  - Bỏ toàn bộ các đoạn văn giải thích dài dòng, bỏ các tooltip rườm rà.
  - Sử dụng nhãn ngắn (1-2 từ): `Model`, `Task`, `Dataset`, `Attack`, `Severity`, `AP50`, `mAP50-95`, `D%`.
- 🖼️ **Ưu tiên kích thước ảnh:** Khung 4 ảnh ở cột phải phải lớn, chiếm trọn không gian để quan sát rõ bounding box.

---

## 5. Cấu trúc Dữ liệu Tích hợp API (Frontend Integration)

### 5.1. Request gửi đi khi bấm Run
```json
{
  "task": "2d_detection",
  "model": "yolo11s",
  "dataset": "kitti",
  "sample_id": "000008",
  "attacks": [
    {
      "name": "fgsm",
      "severity": 3
    }
  ]
}
```

### 5.2. Response trả về để vẽ 4 ô ảnh và metrics nhỏ
```json
{
  "sample_id": "000008",
  "images": {
    "clean_gt_url": "/api/v1/data/runs/123/clean_gt.jpg",
    "attacked_raw_url": "/api/v1/data/runs/123/attacked_raw.jpg",
    "clean_pred_url": "/api/v1/data/runs/123/clean_pred.jpg",
    "attacked_pred_url": "/api/v1/data/runs/123/attacked_pred.jpg"
  },
  "metrics": {
    "clean": {
      "ap50": 0.89,
      "map50_95": 0.64,
      "iou": 0.88,
      "detected_boxes": 5
    },
    "attacked": {
      "ap50": 0.24,
      "map50_95": 0.12,
      "iou": 0.21,
      "detected_boxes": 1,
      "degradation_percent": 73.0,
      "lost_boxes": 4
    }
  }
}
```

### 5.3. WebSocket Events cho Progress Bar
```json
// Event gửi từ backend khi tải weight / chạy bài test
{
  "type": "DOWNLOAD_PROGRESS",
  "target": "weights",
  "name": "yolo11x.pt",
  "downloaded_bytes": 47185920,
  "total_bytes": 136314880,
  "percentage": 34.6
}
```

---

## 6. Checklist Kiểm thử Giao diện trước khi Merge

1. [ ] Màn hình chia đúng 2 cột: Cột trái (Config nhỏ gọn) — Cột phải (4 ảnh lớn chiếm đa số diện tích).
2. [ ] Lưới 4 ảnh hiển thị đúng thứ tự: (1) Clean GT, (2) Attacked raw, (3) Clean Pred, (4) Attacked Pred.
3. [ ] Dưới ảnh 3 và 4 có các metrics nhỏ gọn: Ưu tiên `AP50`, `mAP50-95`, `IoU`, `D%`, số box.
4. [ ] Thanh Progress Bar hiển thị mượt mà khi có tác vụ lâu (tải weights Ultralytics, nạp dataset KITTI/COCO, chạy inference).
5. [ ] Chọn được Task (2D, Seg, 3D), Checkpoint YOLO (hỗ trợ tự động download), Dataset.
6. [ ] Danh sách Attack phân 3 tab (White, Gray, Black); click vào thẻ mở rộng chọn mức 1-5, chọn xong thu nhỏ lại hiển thị thanh 5 ô nhỏ.
7. [ ] Giao diện sạch sẽ, ít chữ, tuyệt đối không dùng màu sắc lòe loẹt hay neon theme.
