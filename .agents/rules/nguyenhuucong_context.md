# Rule: NguyenHuuCong Context
# Khi giao tiếp với người dùng, hãy áp dụng ngữ cảnh sau:

1. **User Identity**: Người dùng là `nguyenhuucong`.
2. **Current Project Goal**: Đánh giá độ bền (Robustness Evaluation) của các mô hình YOLO11 đối với các loại nhiễu/biến dạng ảnh (Group A - `imagecorruptions` library).
3. **Key Tools & Scripts**:
   - `scripts/evaluate_robustness.py`: Pipeline CLI chính để đo đạc và vẽ biểu đồ.
   - `src/pipeline/runner.py`: Logic chạy test nội bộ (TestRunner) có sử dụng `tqdm`.
4. **Behavior**: 
   - Chủ động phớt lờ các nhiệm vụ của những thành viên khác (VD: Phan Trong Dat) trong dự án này trừ khi được hỏi rõ ràng.
   - Khi giúp đỡ code, tự động hiểu các thao tác đang phục vụ luồng evaluate robustness (YOLO11 vs Corruptions).
