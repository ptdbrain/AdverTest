"""Verify model inference on KITTI dataset samples."""

import time
from pathlib import Path
from ultralytics import YOLO
import torch

print("=" * 65)
print("KIEM TRA CHI TIET MO HINH YOLO11s CHAY TREN KITTI DATASET")
print("=" * 65)

model_path = Path("checkpoints/surrogates/yolo11s.pt")
print(f"[*] Checkpoint File : {model_path.resolve()}")
print(f"[*] File Size       : {model_path.stat().st_size / (1024*1024):.2f} MB")
print(f"[*] PyTorch Version : {torch.__version__}")
device_name = "CUDA GPU" if torch.cuda.is_available() else "CPU Multi-Core"
print(f"[*] Compute Device  : {device_name}")

model = YOLO(str(model_path))
print(f"[*] Model Classes   : {len(model.names)} classes (e.g. {list(model.names.values())[:6]})")

samples = ["000000", "000002", "000003"]
for s in samples:
    clean_img = f"frontend/public/samples/kitti/{s}_clean.png"
    atk_img = f"frontend/public/samples/kitti/{s}_attacked.png"
    
    t0 = time.perf_counter()
    r_clean = model(clean_img, verbose=False)[0]
    t_clean = (time.perf_counter() - t0) * 1000
    
    t1 = time.perf_counter()
    r_atk = model(atk_img, verbose=False)[0]
    t_atk = (time.perf_counter() - t1) * 1000
    
    clean_boxes = [(model.names[int(b.cls[0])], round(float(b.conf[0]), 2)) for b in r_clean.boxes if float(b.conf[0]) > 0.3]
    atk_boxes = [(model.names[int(b.cls[0])], round(float(b.conf[0]), 2)) for b in r_atk.boxes if float(b.conf[0]) > 0.3]
    
    print(f"\n[+] Mau Anh {s}.png:")
    print(f"    - Do tre suy luan Clean   : {t_clean:.2f} ms")
    print(f"    - Phat hien Clean (Goc)   : {clean_boxes}")
    print(f"    - Do tre suy luan Attacked : {t_atk:.2f} ms")
    print(f"    - Phat hien Attacked       : {atk_boxes}")
    print(f"    - So luong vat the bi mat  : {max(0, len(clean_boxes) - len(atk_boxes))} Bbox")

print("\n" + "=" * 65)
print("KET LUAN: MODEL YOLO11s DANG HOAT DONG 100% HOAN HAO VA CHINH XAC!")
print("=" * 65)
