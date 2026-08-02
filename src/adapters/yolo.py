"""YOLOv8 / YOLO11 adapter using ultralytics."""

from __future__ import annotations

import time
from typing import ClassVar, Sequence

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.adapters import MODELS
from src.adapters.base import ModelAdapter
from src.core.types import Box, ModelInfo, Prediction, Sample


# Map COCO classes to our KITTI/nuScenes standard 3 classes
COCO_TO_STANDARD = {
    0: "Pedestrian",
    1: "Cyclist",
    2: "Car",
    3: "Cyclist",
    5: "Car",
    7: "Car",
}

# Hỗ trợ tự động tạo nhiều phiên bản YOLO khác nhau
YOLO_MODELS = [
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"
]

def _create_yolo_adapter(model_id: str) -> type[ModelAdapter]:
    class YoloAdapter(ModelAdapter):
        name: ClassVar[str] = model_id
        version: ClassVar[str] = "ultralytics"
        owner: ClassVar[str] = "core"
        
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            if YOLO is None:
                raise RuntimeError("Please install ultralytics to use YOLO adapter.")
            
            self.model_name = f"models/{model_id}.pt"
            self.model = YOLO(self.model_name)
            
        def metadata(self) -> ModelInfo:
            return ModelInfo(
                name=self.name,
                task=self.task,
                version=self.version,
            )

        def predict(self, samples: Sequence[Sample]) -> list[Prediction]:
            predictions = []
            for sample in samples:
                start_time = time.perf_counter()
                
                # Convert float32 [0,1] -> uint8 [0,255]
                img_uint8 = np.clip(sample.image * 255.0, 0, 255).astype(np.uint8)
                
                # Run inference
                results = self.model(img_uint8, verbose=False)
                
                raw_boxes = []
                if len(results) > 0 and results[0].boxes is not None:
                    xyxy = results[0].boxes.xyxy.cpu().numpy()
                    conf = results[0].boxes.conf.cpu().numpy()
                    cls = results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(xyxy)):
                        coco_cls_id = cls[i]
                        if coco_cls_id in COCO_TO_STANDARD:
                            label = COCO_TO_STANDARD[coco_cls_id]
                            score = float(conf[i])
                            x1, y1, x2, y2 = map(float, xyxy[i])
                            
                            raw_boxes.append(
                                Box(
                                    x1=x1,
                                    y1=y1,
                                    x2=x2,
                                    y2=y2,
                                    label=label,
                                    score=score,
                                )
                            )
                
                final_boxes = self.postprocess(raw_boxes)
                
                latency = (time.perf_counter() - start_time) * 1000.0
                predictions.append(
                    Prediction(
                        sample_id=sample.sample_id,
                        boxes=final_boxes,
                        latency_ms=latency,
                    )
                )
                
            return predictions
            
    YoloAdapter.__name__ = f"YoloAdapter_{model_id.upper()}"
    return YoloAdapter

# Đăng ký tự động toàn bộ danh sách model
for m in YOLO_MODELS:
    MODELS.register(_create_yolo_adapter(m))
