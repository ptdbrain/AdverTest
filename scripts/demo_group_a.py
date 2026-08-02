import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import os

from src.core.types import Sample
from src.attacks.base import AttackContext
from src.attacks import ATTACKS

# We need to import common to trigger the registration
import src.attacks.corruption.common

def main():
    # Sử dụng một ảnh mẫu (thay cho toàn bộ dataset KITTI vì bộ này rất nặng để tải tự động)
    url = "https://raw.githubusercontent.com/yhenon/pytorch-retinanet/master/kitti_sample.png"
    sample_path = "data/kitti_sample.png"
    
    if not os.path.exists(sample_path):
        print("Đang tải ảnh mẫu KITTI...")
        try:
            urllib.request.urlretrieve(url, sample_path)
        except Exception:
            # Fallback in case URL fails, create a dummy image
            print("Lỗi tải ảnh. Đang tạo ảnh giả định...")
            dummy = np.ones((375, 1242, 3), dtype=np.uint8) * 127
            Image.fromarray(dummy).save(sample_path)
            
    img = Image.open(sample_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Tạo sample giả định để test (chưa cần ground truth box lúc này)
    sample = Sample(sample_id="demo_0", image=img_np)
    
    # Khởi tạo context với seed cố định
    rng = np.random.default_rng(42)
    ctx = AttackContext(rng=rng)
    
    # Các loại tấn công muốn test
    attacks_to_test = ["snow", "fog", "gaussian_noise", "pixelate"]
    
    fig, axes = plt.subplots(len(attacks_to_test) + 1, 1, figsize=(10, 15))
    axes[0].imshow(sample.image)
    axes[0].set_title("Ảnh gốc (Clean)")
    axes[0].axis('off')
    
    for i, attack_name in enumerate(attacks_to_test):
        AttackClass = ATTACKS.get(attack_name)
        if AttackClass is None:
            print(f"Không tìm thấy attack {attack_name}")
            continue
            
        attack = AttackClass()
        # Áp dụng nhiễu với mức độ 4
        corrupted_sample = attack.run(sample, severity=4, ctx=ctx)
        
        axes[i+1].imshow(corrupted_sample.image)
        axes[i+1].set_title(f"Attack: {attack_name} (Mức độ 4)")
        axes[i+1].axis('off')
        print(f"Đã tạo thành công nhiễu: {attack_name}")
        
    plt.tight_layout()
    out_file = "results/demo_corruptions.png"
    plt.savefig(out_file, dpi=150)
    print(f"\nĐã lưu kết quả vào file: {out_file}")

if __name__ == "__main__":
    main()
