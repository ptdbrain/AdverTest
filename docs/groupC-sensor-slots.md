# Group C: what is implemented, and what needs a core change first

The current Group C scope is image-only. Four attacks are implemented as
image-modality plugins; LiDAR and multi-camera variants remain blocked by the
shared data contract. This note keeps that boundary explicit.

## Implemented (`src/attacks/occlusion/`)

| Plugin | Plan row | Severity ladder |
|---|---|---|
| `random_erasing` | Random Erasing / CutOut | 2 / 5 / 10 / 15 / 20 % of frame area, 1-3 regions |
| `object_occlusion` | Occlusion per object | 10 / 25 / 50 / 75 / **90** % of each GT box |
| `sensor_fault` | Camera dropout (single-sensor analogue) | 0.1-5 % dead pixels + 0-4 lost readout bands |
| `frame_freeze` | Frame freeze | 1-5 stale keyframes, as an ego-motion warp |

Two deviations from the plan text, both deliberate and documented in the plugin
docstrings:

* **`object_occlusion` has five levels, not four.** The plan lists four ratios.
  `RunConfig.severities` defaults to `[1, 3, 5]`, and `BaseAttack.run` raises on
  `severity > severity_levels` — the runner does not catch it, so a four-level
  attack would abort every default run. Level 5 is 90 % coverage.
* **`object_occlusion` requires boxes.** An object-targeted attack without
  ground-truth boxes is rejected before generation instead of using a frame
  fallback.
* **`frame_freeze` is a surrogate.** `apply()` is a pure function of one sample
  and has no access to the previous frame, so staleness is approximated by the
  ego motion that would have accumulated during the freeze (forward zoom plus
  lateral drift). It is not literally the previous frame.

## Blocked on a core change

`lidar_beam_drop`, `lidar_sector_drop` and `camera_dropout` (the real 6-camera
nuScenes version) cannot be plugins today:

1. `Sample` exposes only `with_image` (`src/core/types.py`). There is no
   `with_lidar`, so a LiDAR attack has no way to return modified points.
2. `BaseAttack.run` validates and returns an **image**, and the shared contract
   test asserts the image actually changed at severity 1
   (`tests/test_attacks/test_contract.py`). A LiDAR-only plugin fails it.
3. Multi-camera dropout needs a `Sample` that carries six views. There is one
   image per sample.
4. None of the three is measurable without a 3D or multi-modal adapter
   (M5 PointPillars / M6 BEVFusion), which nobody owns yet.

`docs/CONTRIBUTING_ATTACKS.md` §1 is explicit that needing a core edit means the
plugin contract is missing something, and that it belongs in its own PR. The
proposed core change, for whoever picks it up:

* add `Sample.with_lidar(points)` alongside `with_image`;
* branch `BaseAttack.run` on `modality`, validating points for `lidar` and both
  for `multi`, instead of assuming an image;
* split the contract test by modality so "the image changed" is only asserted for
  image-modality attacks;
* only then add the three plugins, together with an adapter that can score them.

Until that lands, the runner already reports these rows honestly: an attack whose
`modality` does not match the dataset is recorded in `report.skipped` with a
reason, never dropped silently (`src/pipeline/runner.py`).
