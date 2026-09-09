"""Group A — common corruptions (plan §2, ImageNet-C style).

Open slots for the 19 ImageNet-C corruptions × 5 severities. Implemented here
with numpy so the tier-1 scan needs no extra dependency; swap in
``imagecorruptions`` or a kornia/CUDA implementation per plan §5 when the
CPU cost becomes the bottleneck.

Naming: keep the ImageNet-C names (``gaussian_noise``, ``motion_blur``, …) so
the report is comparable with published mPC / rPC numbers.
"""
