"""Group C — occlusion and sensor faults (plan §2).

Open slots: camera dropout (drop 1–3 of the 6 nuScenes cameras), LiDAR beam
drop (25/50/75 % of beams), LiDAR sector drop (30–180°), frame freeze.
Those need multi-sensor samples, so they declare ``modality = "multi"`` or
``"lidar"`` and the runner skips them for image-only datasets.
"""
