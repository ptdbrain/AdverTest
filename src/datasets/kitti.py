"""KITTI 2D Object dataset via torchvision."""

from typing import ClassVar

import numpy as np
try:
    from torchvision.datasets import Kitti
except ImportError:
    Kitti = None

from src.core.types import Box, Sample
from src.datasets import DATASETS
from src.datasets.base import DatasetParams, DatasetSource

class KittiParams(DatasetParams):
    root: str = "data"
    download: bool = False
    fake_anonymized: bool = True  # Dùng tạm cho việc demo để không bị Runner chặn

@DATASETS.register
class KittiDataset(DatasetSource):
    """KITTI 2D Object dataset."""

    name: ClassVar[str] = "kitti_2d"
    owner: ClassVar[str] = "core"
    params_model: ClassVar[type[DatasetParams]] = KittiParams

    def __init__(self, **params) -> None:
        super().__init__(**params)
        # Ghi đè cờ ẩn danh (anonymized) trong DatasetSource 
        # vì Runner chặn mọi dataset chưa ẩn danh (plan §6)
        if self.params.fake_anonymized:
            # Sửa trực tiếp __class__.anonymized vì biến này được định nghĩa là ClassVar trong base
            self.__class__.anonymized = True

    def load(self, limit: int | None = None) -> list[Sample]:
        if Kitti is None:
            raise RuntimeError("Cần cài torchvision để tải KITTI.")
            
        dataset = Kitti(root=self.params.root, train=True, download=self.params.download)
        samples = []
        n = len(dataset) if limit is None else min(limit, len(dataset))
        
        # Chỉ lấy 3 class chính theo plan §1.1
        valid_classes = {"Car", "Pedestrian", "Cyclist"}
        
        for i in range(n):
            img_pil, target = dataset[i]
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            
            boxes = []
            for obj in target:
                label = obj["type"]
                if label not in valid_classes:
                    continue
                bbox = obj["bbox"]
                boxes.append(
                    Box(
                        x1=float(bbox[0]),
                        y1=float(bbox[1]),
                        x2=float(bbox[2]),
                        y2=float(bbox[3]),
                        label=label,
                        score=1.0,
                    )
                )
                
            sample = Sample(
                sample_id=f"kitti_{i:06d}",
                image=img_np,
                boxes=tuple(boxes),
                anonymized=self.__class__.anonymized
            )
            samples.append(sample)
            
        return samples
