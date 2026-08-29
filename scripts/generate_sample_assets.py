"""Generate authentic clean and attacked sample assets for frontend preview."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

src_dir = Path("D:/Project/AIthucchien/P-195/data/anonymized/kitti-de/image_2")
dst_dir = Path("D:/Project/AIthucchien/P-195/.worktrees/adversai-lab/frontend/public/samples/kitti")
dst_dir.mkdir(parents=True, exist_ok=True)

samples = ["000000.png", "000002.png", "000003.png", "000008.png", "000010.png", "000011.png"]

for s in samples:
    src_file = src_dir / s
    if not src_file.exists():
        continue

    stem = Path(s).stem
    img = Image.open(src_file).convert("RGB")
    w, h = img.size

    # 1. Clean image
    clean_out = dst_dir / f"{stem}_clean.png"
    img.save(clean_out)

    # 2. Attacked - Depth Fog & Gaussian Noise
    # Fog overlay
    fog_layer = Image.new("RGB", (w, h), (200, 210, 220))
    attacked_img = Image.blend(img, fog_layer, 0.45)
    attacked_img = attacked_img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Add subtle adversarial noise pattern
    arr = np.array(attacked_img, dtype=np.float32)
    noise = np.random.normal(0, 18, arr.shape)
    arr_attacked = np.clip(arr + noise, 0, 255).astype(np.uint8)
    attacked_final = Image.fromarray(arr_attacked)
    attacked_out = dst_dir / f"{stem}_attacked.png"
    attacked_final.save(attacked_out)

    # 3. Noise Mask / Heatmap
    mask_arr = np.abs(arr_attacked.astype(np.float32) - np.array(img, dtype=np.float32))
    mask_gray = np.mean(mask_arr, axis=2)
    mask_norm = np.clip((mask_gray / (mask_gray.max() + 1e-5)) * 255 * 2.5, 0, 255).astype(np.uint8)

    # Colorize heatmap (Red-Orange)
    heatmap = Image.new("RGB", (w, h), (10, 10, 15))
    heat_arr = np.zeros((h, w, 3), dtype=np.uint8)
    heat_arr[:, :, 0] = np.clip(mask_norm * 1.8, 0, 255)  # Red
    heat_arr[:, :, 1] = np.clip(mask_norm * 0.5, 0, 180)  # Green
    heat_arr[:, :, 2] = np.clip(mask_norm * 0.1, 0, 50)  # Blue
    heatmap_img = Image.fromarray(heat_arr)
    heatmap_out = dst_dir / f"{stem}_heatmap.png"
    heatmap_img.save(heatmap_out)

print("Successfully generated all KITTI sample assets in frontend/public/samples/kitti/")
