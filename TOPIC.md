## 1. Tên đề tài / Dự án

* **AdverTest** - Công cụ sinh & kiểm thử adversarial cho perception (robustness testing)

---

## 2. Mô tả bài toán & Yêu cầu

* 📍 **Thực trạng:** Mô hình perception có thể bị đánh lừa bởi nhiễu nhỏ, thời tiết xấu, che khuất hay patch adversarial, gây rủi ro an toàn, nhưng đội kỹ sư thiếu công cụ hệ thống để kiểm thử độ bền vững trước khi triển khai.
* 🎯 **Vấn đề:** Xây dựng công cụ sinh các phép biến đổi và tấn công adversarial (nhiễu, corruption thời tiết, che khuất, patch) áp lên dữ liệu perception và đo mức suy giảm hiệu năng của mô hình, giúp kỹ sư tìm điểm yếu và đánh giá robustness.
* 🔒 **Ràng buộc:**
* Kỹ sư review các trường hợp lỗi và quyết định biện pháp (human-in-the-loop).
* Chỉ kiểm thử trên dữ liệu/mô phỏng, kết quả không dùng để triển khai thật khi chưa validate.
* Chỉ số đo được: $\text{mAP}$/$\text{IoU}$ trước và sau tấn công, robustness accuracy theo mức độ nhiễu, tỷ lệ tấn công thành công.
* Tối ưu chi phí GPU khi chạy nhiều biến thể.
* Ẩn danh khuôn mặt/biển số trong dữ liệu.



---

## 3. Công nghệ / Kỹ thuật sử dụng (Tech Stack)

* **Python, PyTorch**
* **Thư viện adversarial/corruption:** `torchattacks`, `imagecorruptions`
* **Mô hình perception:** YOLO, SAM2, MMDetection3D
* **Dữ liệu:** nuScenes/KITTI
* **Giao diện & Backend:** FastAPI + React/Next.js dashboard (so sánh trước/sau tấn công)
* **Theo dõi & Triển khai:** Weights & Biases, Docker + GPU

---

## 4. Mục tiêu & Tính năng triển khai

### Cơ bản:

* Tool áp một số phép biến đổi/tấn công lên ảnh mẫu, chạy mô hình và hiển thị suy giảm hiệu năng trực quan, $\ge 2$ vai trò, báo cáo $\text{mAP}$/$\text{IoU}$ trước-sau.

### Nâng cao:

* Sinh và quét nhiều loại tấn công theo mức độ, tự động lập bảng robustness và tìm biến thể làm mô hình thất bại nhiều nhất, benchmark định lượng và tối ưu chi phí GPU.
