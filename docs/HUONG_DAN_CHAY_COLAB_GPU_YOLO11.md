# HƯỚNG DẪN CHẠY HUẤN LUYỆN YOLO11 TRÊN GOOGLE COLAB
## Hỗ trợ Repo Private (`AI20K-Build-Phase-Cohort-3/P-195`) & Cài đặt siêu tốc bằng `uv`

> **Mục đích:** Hướng dẫn kết nối Google Colab với Private GitHub Repository một cách an toàn và sử dụng trình quản lý gói siêu tốc **`uv`** để thiết lập môi trường GPU, huấn luyện YOLO11 (B0 Baseline, R1 Robust Mix, R2 Targeted Repair) và tải kết quả về.

---

## 🚀 TỔNG QUAN LUỒNG THỰC THI TRÊN COLAB

```text
[1. Chọn T4 / A100 GPU Runtime]
            │
            ▼
[2. Clone Private Repo qua Colab Secrets / GitHub Token]
            │
            ▼
[3. Cài đặt Dependencies siêu tốc qua `uv` (~10 giây)]
            │
            ▼
[4. Chạy Huấn luyện YOLO-B0 / YOLO-R1 / YOLO-R2]
            │
            ▼
[5. Kiểm định Checkpoint Gate & Chạy Benchmark KITTI]
            │
            ▼
[6. Tải Artifacts (.pt, reports .json) về máy Local]
```

---

## 🔑 BƯỚC 1: Chuẩn bị GitHub Token cho Private Repo

Do repository `https://github.com/AI20K-Build-Phase-Cohort-3/P-195/` là **Private**, bạn cần có GitHub Personal Access Token (PAT):

1. Trên GitHub: Vào **Settings** $\rightarrow$ **Developer Settings** $\rightarrow$ **Personal access tokens** $\rightarrow$ **Tokens (classic)**.
2. Bấm **Generate new token (classic)** $\rightarrow$ Đặt tên (vd: `colab-p195`), tích chọn quyền **`repo`** $\rightarrow$ Bấm **Generate token** và sao chép mã token (dạng `ghp_...`).
3. **Trên Google Colab:**
   - Nhìn sang thanh công cụ bên trái, bấm vào biểu tượng chiếc chìa khóa 🔑 (**Secrets**).
   - Bấm **Add new secret** $\rightarrow$ Name: `GH_TOKEN` $\rightarrow$ Value: *Dán mã token vừa sao chép*.
   - Gạt công tắc **Notebook access** sang **ON**.

---

## 📋 BƯỚC 2: Các Code Cell Chạy Tuần Tự Trên Colab

### [CELL 1] Kiểm tra phần cứng GPU
```python
!nvidia-smi

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
```

---

### [CELL 2] Clone đúng nhánh làm việc (`feat/person-d-platform` hoặc nhánh của bạn)
```python
import os
from google.colab import userdata

# 1. Lấy GitHub Token từ Colab Secrets
try:
    gh_token = userdata.get('GH_TOKEN')
except Exception:
    from getpass import getpass
    gh_token = getpass("Nhập GitHub Personal Access Token của bạn: ")

# 2. Đặt tên nhánh bạn đang làm việc (ví dụ: 'feat/person-d-platform')
BRANCH_NAME = "feat/person-d-platform"

# 3. Clone chính xác nhánh đang làm việc
!git clone --branch {BRANCH_NAME} https://{gh_token}@github.com/AI20K-Build-Phase-Cohort-3/P-195.git
%cd P-195

# 4. Xác nhận nhánh và cập nhật code mới nhất
!git branch --show-current
!git pull origin {BRANCH_NAME}
```


> **Mẹo:** Nếu bạn muốn checkout vào branch cụ thể (ví dụ: `feat/person-d-platform`):
> ```python
> !git checkout feat/person-d-platform
> !git pull
> ```

---

### [CELL 3] Cài đặt `uv` và Đồng bộ Môi trường GPU (~10-15s)
```python
# Cài đặt trình quản lý gói siêu tốc uv
!pip install -q uv

# Dùng uv cài đặt toàn bộ gói từ pyproject.toml với cấu hình GPU & dev tools
!uv pip install --system -e ".[models-gpu,dev]"
```

---

### [CELL 4] Huấn luyện YOLO-B0 (Tối ưu 100% Công suất GPU: AMP FP16 + RAM Cache + Multi-workers)
```python
# Huấn luyện mô hình Baseline sạch với toàn bộ sức mạnh GPU T4 / A100
!python scripts/train_yolo.py \
    --mode b0 \
    --epochs 30 \
    --batch-size 32 \
    --amp \
    --cache ram \
    --workers 8 \
    --device 0 \
    --lr 0.001 \
    --output-dir runs/train/yolo_b0
```

---

### [CELL 5] Huấn luyện YOLO-R1 (Robust Mix - Max GPU Performance)
```python
# Huấn luyện mô hình Robust Mix (Fine-tune từ B0 với FP16 Tensor Cores)
!python scripts/train_yolo.py \
    --mode r1 \
    --base-checkpoint runs/train/yolo_b0/yolo11s-clean-b0_best.pt \
    --epochs 20 \
    --batch-size 32 \
    --amp \
    --cache ram \
    --workers 8 \
    --device 0 \
    --lr 0.0005 \
    --output-dir runs/train/yolo_r1
```


> **Tự động kiểm tra Acceptance Gate:**  
> Kết thúc huấn luyện R1, hệ thống sẽ tự động đối chiếu các tiêu chuẩn nghiệm thu:  
> - $\Delta\text{Clean mAP} \ge -0.0200$ (không suy giảm quá 2.0% trên ảnh sạch)  
> - $\Delta\text{RobustScore} \ge +8.0$ điểm  
> Terminal sẽ hiển thị `[*] Gate Passed: PASSED (ACCEPT)`.

---

### [CELL 6] Huấn luyện YOLO-R2 (Targeted Repair - nếu phát hiện cụm lỗi)
```python
# Huấn luyện mô hình sửa lỗi đích danh (ví dụ: người đi bộ trong sương mù)
!python scripts/train_yolo.py \
    --mode r2 \
    --target-cluster "pedestrian_fog_occlusion" \
    --epochs 15 \
    --batch-size 16 \
    --output-dir runs/train/yolo_r2
```

---

### [CELL 7] Chạy Benchmark Đánh giá Kịch bản Tấn công trên KITTI
```python
# Chạy benchmark kiểm tra robustness thực tế trên tập KITTI
!python scripts/benchmark_kitti_yolo11.py \
    --root data/anonymized/kitti \
    --limit 500
```

---

### [CELL 8] Đóng gói & Tải Checkpoints (.pt) + Báo cáo về Máy Local
```python
# Đóng gói toàn bộ checkpoints và kết quả báo cáo ra file zip
!zip -r advertest_yolo_artifacts.zip runs/train/ eval/results/

# Tải file zip về máy tính cá nhân
from google.colab import files
files.download('advertest_yolo_artifacts.zip')
```

---

## 🔄 BƯỚC 3: Đồng bộ về Máy Local

Sau khi tải file `advertest_yolo_artifacts.zip` về máy:
1. Giải nén vào thư mục dự án `P-195/`:
   ```bash
   unzip -o advertest_yolo_artifacts.zip -d .
   ```
2. Chạy test xác thực checkpoint trên môi trường local:
   ```bash
   pytest tests/test_adapters/test_yolo11.py tests/test_training_yolo_trainer.py -v
   ```
3. Bàn giao checkpoint `.pt` (kèm SHA256) và báo cáo JSON cho Người A (hiển thị giao diện Web) và Người D (chạy Benchmark tổng thể).
