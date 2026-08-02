import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.datasets.kitti import KittiDataset
from src.adapters.yolo import YoloAdapter
from src.attacks import ATTACKS
from src.attacks.base import AttackContext

# Import common để load các plugin nhiễu nhóm A
import src.attacks.corruption.common

def draw_boxes(image_float, boxes, ground_truth=None):
    img = np.clip(image_float * 255.0, 0, 255).astype(np.uint8)
    img_draw = img.copy()
    
    # Vẽ hộp ground truth (màu xanh dương)
    if ground_truth:
        for box in ground_truth:
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 100, 255), 2)
            cv2.putText(img_draw, f"[GT] {box.label}", (x1, max(15, int(y2) + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

    # Vẽ hộp YOLO dự đoán (màu xanh lá)
    for box in boxes:
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{box.label} {box.score:.2f}"
        cv2.putText(img_draw, label_text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return img_draw

def main():
    # 1. Lấy ảnh đầu tiên của KITTI
    ds = KittiDataset()
    samples = ds.load(limit=1)
    sample = samples[0]
    
    # 2. Thiết lập Attack Snow cấp độ 5
    rng = np.random.default_rng(42)
    ctx = AttackContext(rng=rng)
    attack = ATTACKS.get("snow")()
    corrupted_sample = attack.run(sample, severity=5, ctx=ctx)
    
    # 3. Chạy model YOLO11s
    yolo = YoloAdapter()
    
    pred_clean = yolo.predict([sample])[0]
    pred_corr = yolo.predict([corrupted_sample])[0]
    
    # 4. Trực quan hoá
    img_clean_viz = draw_boxes(sample.image, pred_clean.boxes, sample.boxes)
    img_corr_viz = draw_boxes(corrupted_sample.image, pred_corr.boxes, corrupted_sample.boxes)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].imshow(img_clean_viz)
    axes[0].set_title(f"Ảnh Gốc (Clean) - Phát hiện: {len(pred_clean.boxes)} objects (Xanh lá) vs Thực tế: {len(sample.boxes)} objects (Xanh biển)")
    axes[0].axis("off")
    
    axes[1].imshow(img_corr_viz)
    axes[1].set_title(f"Ảnh bị tấn công Tuyết Rơi (Snow lvl 5) - Phát hiện: {len(pred_corr.boxes)} objects")
    axes[1].axis("off")
    
    out_file = "results/demo_yolo_kitti.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Đã lưu ảnh trực quan hoá tại {out_file}")

if __name__ == "__main__":
    main()
