"""Group B — physical weather, depth aware (plan §2).

Difference from group A: these use the real scene depth, so fog is thicker far
away than nearby. Depth comes from ``sample.depth`` (LiDAR projected onto the
image); when a dataset has none, plugins fall back to
:func:`src.core.image_ops.linear_depth_prior` and must say so in their docstring.

Open slots: rain (10–100 mm/h), snow overlay, LiDAR fog / snowfall
(Hahner et al. 2021, 2022).
"""
