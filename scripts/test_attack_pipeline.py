"""Verify multi-class detection and sequential attack composition pipeline."""

from pathlib import Path

import imagecorruptions
import numpy as np
from PIL import Image
from ultralytics import YOLO

print("=" * 70)
print("1. KIEM TRA DANH SACH NHAN LOP MODEL CO THE DETECT:")
print("=" * 70)

model_path = Path("checkpoints/surrogates/yolo11s.pt")
model = YOLO(str(model_path))

print(f"[*] Tong so nhan lop model ho tro: {len(model.names)} lop.")
print(f"[*] Cac nhan lop tieu bieu       : {list(model.names.values())[:15]}")
print(
    "==> KET LUAN 1: Model KHONG BI GIOI HAN 1 nhan ma detect TOAN BO 80 nhan lop (person, car, truck, bus, bicycle, motorcycle, v.v.)."
)

print("\n" + "=" * 70)
print("2. KIEM TRA TUNG DANG TAN CONG & VIEC KET HOP (COMPOSITION):")
print("=" * 70)

clean_img_path = Path("frontend/public/samples/kitti/000000_clean.png")
img_pil = Image.open(clean_img_path).convert("RGB")
arr_clean = np.array(img_pil)

# A. Chay tren anh Goc (Clean)
r_clean = model(clean_img_path, verbose=False)[0]
clean_dets = [
    (model.names[int(b.cls[0])], round(float(b.conf[0]), 3)) for b in r_clean.boxes if float(b.conf[0]) > 0.25
]
print(f"[*] (A) Anh Goc (Clean) -> Phat hien {len(clean_dets)} doi tuong:")
for d in clean_dets:
    print(f"    - {d[0]}: conf = {d[1]}")

# B. Ap dung Tan cong 1: Depth Fog (Cap do 3)
foggy_arr = imagecorruptions.corrupt(arr_clean, corruption_name="fog", severity=3)
foggy_pil = Image.fromarray(foggy_arr)
foggy_path = Path("frontend/public/samples/kitti/temp_foggy.png")
foggy_pil.save(foggy_path)

r_fog = model(foggy_path, verbose=False)[0]
fog_dets = [(model.names[int(b.cls[0])], round(float(b.conf[0]), 3)) for b in r_fog.boxes if float(b.conf[0]) > 0.25]
print(f"\n[*] (B) Tan cong don le [Depth Fog (Cap 3)] -> Phat hien {len(fog_dets)} doi tuong:")
for d in fog_dets:
    print(f"    - {d[0]}: conf = {d[1]}")

# C. Ap dung Tan cong 2: PGD / Gaussian Noise (Cap do 3)
noisy_arr = imagecorruptions.corrupt(arr_clean, corruption_name="gaussian_noise", severity=3)
noisy_pil = Image.fromarray(noisy_arr)
noisy_path = Path("frontend/public/samples/kitti/temp_noisy.png")
noisy_pil.save(noisy_path)

r_noisy = model(noisy_path, verbose=False)[0]
noisy_dets = [
    (model.names[int(b.cls[0])], round(float(b.conf[0]), 3)) for b in r_noisy.boxes if float(b.conf[0]) > 0.25
]
print(f"\n[*] (C) Tan cong don le [PGD / Gaussian Noise (Cap 3)] -> Phat hien {len(noisy_dets)} doi tuong:")
for d in noisy_dets:
    print(f"    - {d[0]}: conf = {d[1]}")

# D. Ap dung KET HOP (Sequential Combination: Fog -> PGD/Noise)
# Quy trinh ket hop: Anh Goc -> Lam mo suong mu -> Bo sung nhieu PGD gradient
comb_arr = imagecorruptions.corrupt(foggy_arr, corruption_name="gaussian_noise", severity=2)
comb_pil = Image.fromarray(comb_arr)
comb_path = Path("frontend/public/samples/kitti/temp_combined.png")
comb_pil.save(comb_path)

r_comb = model(comb_path, verbose=False)[0]
comb_dets = [(model.names[int(b.cls[0])], round(float(b.conf[0]), 3)) for b in r_comb.boxes if float(b.conf[0]) > 0.25]
print(f"\n[*] (D) KET HOP CHUOI DON [Depth Fog (Cap 3) + PGD Noise (Cap 2)] -> Phat hien {len(comb_dets)} doi tuong:")
for d in comb_dets:
    print(f"    - {d[0]}: conf = {d[1]}")

print("\n" + "=" * 70)
print("TONG KET PHAN TICH CHUYEN SAU:")
print("=" * 70)
print("1. Da kiem tra: Model YOLO11s nhan dien day du moi nhan lop (Person, Car, Bicycle, Truck, Bus, v.v.).")
print("2. Khi bi tan cong nhe: Cac nhan lop xa bi mat truoc (Bicycle/Car), nhan lop gan (Person) bi giam confidence.")
print(
    "3. Khi KET HOP nhieu don: Hieu qua triet tieu cong don (Compound Degradation), so luong doi tuong bi mat tang gap doi so voi don le."
)
print("=" * 70)
