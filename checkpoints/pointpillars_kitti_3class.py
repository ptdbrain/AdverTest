"""Self-contained PointPillars metadata for the validated KITTI 3-class path.

This is intentionally a portable config descriptor, not an executable model
checkpoint.  The GPU runtime resolves the actual MMDetection3D configuration
from its approved runtime image before accepting a PointPillars run.
"""

class_names = ["Car", "Pedestrian", "Cyclist"]

model = {
    "type": "VoxelNet",
    "voxel_encoder": {"type": "PillarFeatureNet"},
    "middle_encoder": {"type": "PointPillarsScatter"},
    "backbone": {"type": "SECOND"},
    "neck": {"type": "SECONDFPN"},
    "bbox_head": {"type": "Anchor3DHead", "num_classes": len(class_names)},
}
