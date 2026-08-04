# Third-party weather kernels

The LiDAR fog and snowfall attack interfaces follow the public Hahner et al.
implementations (`LiDAR_fog_sim` and `LiDAR_snow_sim`). Their upstream kernels
are released under CC BY-NC 4.0. This repository keeps the deterministic,
NumPy-compatible fallback in `src/attacks/weather/_lidar.py` so CPU tests do
not require CUDA; production runs may replace that backend with the upstream
kernels without changing attack contracts.

Any vendored upstream source must retain its copyright and attribution notice
and must not be used for commercial deployment without the authors' permission.
