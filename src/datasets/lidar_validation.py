"""LiDAR Point Cloud Validation & Ingestion Guards (P2.1).

Validates:
- Non-empty point cloud (N > 0).
- Expected point stride and dimensional consistency.
- Finite values (Zero NaN / Inf).
- Minimum point count thresholds.
- Spatial coordinate bounding ranges.
- Valid intensity / reflectance bounds.
- Protection against truncated / corrupted files.
- Support for both raw binary (.bin) and Point Cloud Data (.pcd) formats.
"""

from __future__ import annotations

import numpy as np


class LidarValidationError(ValueError):
    """Specific error with a machine-readable error code for HTTP 400/422 responses."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def validate_lidar_bin_bytes(
    content: bytes,
    *,
    min_points: int = 10,
    point_dim: int = 4,
    coord_range: tuple[float, float] = (-500.0, 500.0),
    intensity_range: tuple[float, float] = (0.0, 255.0),
) -> np.ndarray:
    """Validate and parse raw float32 LiDAR .bin bytes (e.g. KITTI/nuScenes)."""
    if not content:
        raise LidarValidationError("LIDAR_EMPTY", "LiDAR byte buffer is empty (0 bytes).")

    stride = point_dim * 4  # 4 bytes per float32
    if len(content) % stride != 0:
        raise LidarValidationError(
            "LIDAR_TRUNCATED",
            f"LiDAR file length ({len(content)} bytes) is not divisible by point stride ({stride} bytes).",
        )

    points = np.frombuffer(content, dtype=np.float32).reshape(-1, point_dim)
    n_points = points.shape[0]

    if n_points < min_points:
        raise LidarValidationError(
            "LIDAR_POINT_COUNT_BELOW_MINIMUM",
            f"Point count ({n_points}) is below required minimum ({min_points}).",
        )

    if not np.all(np.isfinite(points)):
        raise LidarValidationError("LIDAR_NAN_INF", "LiDAR points contain NaN or infinite values.")

    # Spatial coordinate checks (x, y, z)
    xyz = points[:, :3]
    if np.any(xyz < coord_range[0]) or np.any(xyz > coord_range[1]):
        raise LidarValidationError(
            "LIDAR_COORDINATES_OUT_OF_RANGE",
            f"Coordinates exceed allowed spatial range [{coord_range[0]}, {coord_range[1]}].",
        )

    # Intensity check if 4th column exists
    if point_dim >= 4:
        intensity = points[:, 3]
        if np.any(intensity < intensity_range[0]) or np.any(intensity > intensity_range[1]):
            raise LidarValidationError(
                "LIDAR_INTENSITY_INVALID",
                f"Intensity values exceed valid range [{intensity_range[0]}, {intensity_range[1]}].",
            )

    return points


def validate_pcd_content(
    content: bytes | str,
    *,
    min_points: int = 10,
    coord_range: tuple[float, float] = (-500.0, 500.0),
) -> np.ndarray:
    """Validate and parse ASCII PCD (Point Cloud Data) format."""
    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        raise LidarValidationError("LIDAR_EMPTY", "PCD file is empty.")

    header: dict[str, str] = {}
    data_start = 0
    for idx, line in enumerate(lines):
        if line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            key, val = parts[0].upper(), parts[1]
            header[key] = val
            if key == "DATA":
                data_start = idx + 1
                break

    if "DATA" not in header:
        raise LidarValidationError("LIDAR_FORMAT_INVALID", "Missing DATA header in PCD file.")

    if header.get("DATA").lower() != "ascii":
        raise LidarValidationError("LIDAR_FORMAT_UNSUPPORTED", "Only ASCII PCD parsing supported in standard validator.")

    data_lines = lines[data_start:]
    if not data_lines:
        raise LidarValidationError("LIDAR_EMPTY", "PCD file contains no point records.")

    parsed_points = []
    for line in data_lines:
        values = [float(v) for v in line.split()]
        if len(values) < 3:
            raise LidarValidationError("LIDAR_DIMENSION_MISMATCH", f"PCD row has fewer than 3 dimensions: {line}")
        parsed_points.append(values)

    points = np.array(parsed_points, dtype=np.float32)
    n_points = len(points)

    if n_points < min_points:
        raise LidarValidationError(
            "LIDAR_POINT_COUNT_BELOW_MINIMUM",
            f"PCD point count ({n_points}) is below minimum ({min_points}).",
        )

    if not np.all(np.isfinite(points)):
        raise LidarValidationError("LIDAR_NAN_INF", "PCD contains NaN or infinite values.")

    xyz = points[:, :3]
    if np.any(xyz < coord_range[0]) or np.any(xyz > coord_range[1]):
        raise LidarValidationError(
            "LIDAR_COORDINATES_OUT_OF_RANGE",
            f"Coordinates exceed allowed spatial range [{coord_range[0]}, {coord_range[1]}].",
        )

    return points
