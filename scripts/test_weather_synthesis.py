"""Test and verify rain, snow, fog, blur, and PGD visual generators."""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from pathlib import Path
import imagecorruptions

def apply_realistic_rain(img_arr: np.ndarray, severity: int = 3) -> np.ndarray:
    """Generate realistic depth rain streaks and atmospheric contrast attenuation."""
    h, w, c = img_arr.shape
    # 1. Darken and reduce contrast slightly due to stormy overcast sky
    enhancer = ImageEnhance.Brightness(Image.fromarray(img_arr))
    dimmed = np.array(enhancer.enhance(0.85 - severity * 0.05))
    
    # 2. Generate rain streaks layer
    rain_layer = np.zeros((h, w), dtype=np.float32)
    # Number of rain drops scaled by severity
    num_drops = int(w * h * 0.0008 * severity)
    
    # Random drop locations
    xs = np.random.randint(0, w, num_drops)
    ys = np.random.randint(0, h, num_drops)
    
    # Slanted rain angles (-15 to -25 degrees)
    length = 15 + severity * 6
    slant = int(length * 0.35)
    
    for x, y in zip(xs, ys):
        cv2.line(rain_layer, (x, y), (x - slant, y + length), 255, thickness=1 + (1 if severity >= 4 else 0))
        
    # Blur rain streaks slightly for motion blur look
    rain_blur = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    rain_rgb = np.stack([rain_blur, rain_blur, rain_blur], axis=2) / 255.0
    
    # 3. Blend rain with dimmed image
    rainy = (dimmed.astype(np.float32) * (1.0 - rain_rgb * 0.4) + rain_rgb * 235 * 0.6).clip(0, 255).astype(np.uint8)
    return rainy

def apply_realistic_snow(img_arr: np.ndarray, severity: int = 3) -> np.ndarray:
    """Generate realistic snowfall flakes and cold atmospheric haze."""
    return imagecorruptions.corrupt(img_arr, corruption_name="snow", severity=severity)

# Test on 000000.png
clean_path = Path("frontend/public/samples/kitti/000000_clean.png")
clean_img = Image.open(clean_path).convert("RGB")
clean_arr = np.array(clean_img)

rain_arr = apply_realistic_rain(clean_arr, severity=4)
Image.fromarray(rain_arr).save("frontend/public/samples/kitti/test_rain_sample.png")

snow_arr = apply_realistic_snow(clean_arr, severity=4)
Image.fromarray(snow_arr).save("frontend/public/samples/kitti/test_snow_sample.png")

print("Generated test_rain_sample.png and test_snow_sample.png successfully!")
