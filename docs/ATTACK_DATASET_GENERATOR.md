# Attack Dataset Generator

This pipeline creates adversarial datasets. It does not run clean/attacked
inference, AP, ASR, RobustScore, or any other model evaluation.

## Commands

The checked-in CPU configs are small and run without downloading a dataset or
checkpoint:

```powershell
uv run python -m src.cli generate-attack --config configs/pgd.json
uv run python -m src.cli train-patch --config configs/patch.json
uv run python -m src.cli inspect-attack-dataset --path data/attacked/<source>/<attack>/<generation_id>
```

Use JSON config files as the stable interface. A generation config contains
exactly one input source and exactly one attack. Run separate configs for
separate attacks so provenance is never mixed.

## Input Formats

`dataset_name` loads a registered `DatasetSource`. `input_dir` loads
`FolderDataset` and requires `input_format`:

```text
advertest/
|-- images/
|-- labels/<sample_id>.json
|-- masks/<sample_id>.npy|png
`-- dataset.json
```

`dataset.json` must contain `"anonymized": true`. A separate manifest can be
selected with `anonymization_manifest`.

KITTI V1 uses `image_2/` and `label_2/`. The loader maps KITTI 2D labels to
`Car`, `Pedestrian`, and `Cyclist`.

Attacks declare their annotation requirements. DAG and patch attacks require
boxes. SAM2-PGD requires a box and a ground-truth mask. The generator checks all
selected samples before writing the first variant.

## KITTI Anonymization

Raw KITTI cannot pass the generation gate. Run the face and license-plate
anonymizer first:

```powershell
uv run python -m src.cli anonymize-dataset --config configs/kitti-anonymize-smoke.json
uv run python -m src.cli inspect-anonymized-dataset --path data/anonymized/kitti-smoke
```

The anonymizer requires local ONNX checkpoints and never downloads weights.
It runs full-frame and overlapping tiled inference, expands each detection,
applies mosaic plus Gaussian blur, preserves KITTI labels, and records image,
label, and checkpoint hashes in `manifest.jsonl`. `dataset.json` remains
`anonymized: false` until every selected image is written successfully.
Re-running an identical config resumes valid outputs. A changed source image,
checkpoint hash, config, or output hash forces regeneration or inspection
failure.

The checked-in smoke config selects real KITTI samples with detected privacy
regions. To process a larger deterministic prefix, replace `sample_ids` with
`limit`. To process the full training set, remove both fields. Automated
detection can miss small or occluded regions, so the descriptor records
`review_status: pending_spot_check`; review representative outputs before
using the full export.

The local smoke run used these untracked checkpoints:

| Purpose | Source | File | SHA-256 |
|---|---|---|---|
| Face detection | [AdamCodd YOLOv11n face model](https://huggingface.co/AdamCodd/YOLOv11n-face-detection) | `checkpoints/anonymization/yolo11n-face.onnx` | `2dfe14171f5b76a05f9bcf0dac7f94b7bff4416b1f29eff7c9ef5830f51c5719` |
| Plate detection | [ml-debi YOLOv8 plate model](https://huggingface.co/ml-debi/yolov8-license-plate-detection) | `checkpoints/anonymization/yolov8n-license-plate.onnx` | `85d236280a1301ad98907947d284951dd2b20c23a6786ff50f7e6a8ec515bd50` |
| Attack surrogate | [Ultralytics YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt) | `checkpoints/surrogates/yolo11s.pt` | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |

The face model card identifies Apache-2.0, the plate repository identifies
MIT, and Ultralytics YOLO uses AGPL-3.0 or an enterprise license. Confirm that
the applicable licenses fit the deployment target. Detector inference uses
ONNX files; the official YOLO11s PyTorch checkpoint is loaded only by the
differentiable attack adapter.

After anonymization:

```powershell
uv run python -m src.cli generate-attack --config configs/kitti-fgsm.json
uv run python -m src.cli generate-attack --config configs/kitti-pgd.json
uv run python -m src.cli generate-attack --config configs/kitti-mi-fgsm.json
uv run python -m src.cli generate-attack --config configs/kitti-tog-vanishing.json
uv run python -m src.cli generate-attack --config configs/kitti-tog-fabrication.json
uv run python -m src.cli generate-attack --config configs/kitti-tog-mislabeling.json
uv run python -m src.cli generate-attack --config configs/kitti-cw-l2.json
```

## Surrogates

The surrogate is only used to produce gradients. The generator never evaluates
its predictions.

| Attack | Adapter | Required checkpoint |
|---|---|---|
| FGSM, PGD, MI-FGSM, C&W, TOG | `yolo11` | local YOLO11s file |
| DAG | `faster_rcnn` | local Faster R-CNN R50-FPN state dict |
| SAM2-PGD | `sam2_surrogate` | local SAM2.1 Hiera-small file |
| CPU contract tests | `blob_detector` | none |

Real adapters require `surrogate.checkpoint`; automatic weight download is
disabled. Install model packages with `uv sync --extra models-cpu` locally or
`uv sync --extra models-gpu` on an NVIDIA/CUDA worker. The SAM2
adapter also requires a differentiable
`forward_image_with_box(image, box)` bridge in the installed official SAM2
package.

## Person D recipe pipeline

The versioned path is `recipe-validate -> recipe-sample -> generate-recipe ->
inspect-attack-dataset`. Recipe identity includes ordered steps, implementation
versions, per-step seeds/objectives, constraints, and metadata. Resume validates
manifest and artifact hashes; prediction cache keys are a separate namespace.
All source samples remain immutable, spatial transforms update images and
annotations through one transform, and generated lineage must pass before a
training manifest can consume it.

Example real surrogate section:

```json
{
  "surrogate": {
    "name": "yolo11",
    "checkpoint": "checkpoints/yolo11s.pt",
    "device": "cuda",
    "objective": "untargeted"
  }
}
```

## Attacks

Group D plugins:

- `fgsm`: one L-infinity gradient step.
- `pgd`: random start, projection, configurable steps and restarts.
- `mi_fgsm`: normalized gradient and momentum, default momentum 1.
- `cw_l2`: confidence-margin optimization in tanh space with Adam and binary
  search over the loss constant. Defaults to 100 iterations and five
  constant-search steps. With no explicit `limit`, at most 100 source images
  are loaded.
- `tog`: `vanishing`, `fabrication`, or `mislabeling`.
- `dag`: dense proposal objective, default 150 iterations.
- `sam2_pgd`: box-prompted mask BCE, default epsilon 4/255 at severity 4 and
  20 steps.

Group E separates training from application:

1. `train-patch` writes `patch.npy`, `patch.png`, loss history, training sample
   IDs, source fingerprint, and `patch-manifest.json`.
2. `dpatch` or `thys_patch` loads and verifies the artifact hash.
3. `generate-attack` pastes the artifact. It never retrains the patch.

DPatch supports `objective` values `untargeted` and `targeted`. `source_label`
selects the input object box; targeted mode separately requires `target_label`
as the desired detector class. Its severity ladder covers 5, 10, 15, and 20
percent of the selected box. Thys patch uses a person-vanishing objective with
TV and NPS regularization. EOT uses seeded physical scale, transparent
rotation, brightness, optional blur, and placement offset. Application
requires a trained patch plus its manifest; built-in patches exist only for
unit tests.

Example patch application parameters:

```json
{
  "attack_name": "dpatch",
  "attack_params": {
    "patch_path": "data/patches/dpatch/<artifact_id>/patch.npy",
    "artifact_hash": "<hash from patch-manifest.json>",
    "objective": "untargeted",
    "source_label": "Car",
    "eot": true
  }
}
```

The training sample IDs remain in the artifact manifest. A later benchmark
must exclude those IDs from its held-out evaluation split.

## YOLO11 Benchmark

Benchmark one or more completed generations without regenerating attacks:

```powershell
uv run python -m src.cli benchmark-attack-datasets --config configs/kitti-yolo11-benchmark.json
```

The benchmark reloads each generation, infers its clean source from
`config.json`, verifies every clean image and label hash, and runs the model on
paired clean/attacked pixels. Output is written to
`data/benchmarks/<model>/<benchmark_id>/report.json` and `summary.csv`.

Each cell reports macro AP at the configured IoU threshold, per-class AP,
relative degradation, class-aware TP/FP/FN, precision, recall, detection-loss
attack success rate, and clean/attacked latency. Attack success rate is the
fraction of ground-truth objects detected on clean pixels that are no longer
matched after the attack. Fabrication attacks should also be interpreted using
`false_positive_delta` because they need not remove a clean detection.

The checked-in YOLO11 config uses confidence `0.001` so low-score detections
remain available for the AP precision-recall ranking. Precision/recall counts
in the report are therefore also measured at `0.001`; use a separate config
with the deployment confidence threshold for operational precision/recall.

`white_box_same_checkpoint` is true when the benchmark checkpoint hash matches
the attack surrogate (or patch-training surrogate). Such a result measures a
white-box attack, not transfer robustness. This implementation reports a
single-IoU-threshold AdverTest AP50 when IoU is 0.5; it is not COCO
AP@[0.50:0.95].

## Output Contract

```text
data/attacked/<source>/<attack>/<generation_id>/
|-- images/<variant_id>.npy
|-- previews/<variant_id>.png
|-- labels/<variant_id>.json
|-- masks/<variant_id>.npy
|-- artifacts/
|-- manifest.jsonl
|-- dataset.json
`-- config.json
```

The `.npy` image is canonical: float32 HWC RGB in `[0, 1]`. PNG is only a
preview and must not be used as model input. `GeneratedDatasetSource` reloads
completed output into the shared `Sample` contract.

The manifest records source, output, label, mask, checkpoint, and patch hashes;
derived seed; attack version and parameters; severity; actual L-infinity/L2
norms; and annotation paths. `dataset.json` records source fingerprint,
estimated variant count, canonical bytes, gradient steps, and completion
status.

Interrupted runs are marked `incomplete`. Re-running the same config resumes
only records whose image, label, mask, and preview still pass validation.
Changing source content, generation config, resolved attack defaults/version,
surrogate version, or checkpoint hash creates a different `generation_id`.

## Verification

CPU tests use deterministic mock gradients:

```powershell
uv run pytest -m "not gpu" -q
uv run ruff check src tests
```

Optional checkpoint-backed tests are marked `gpu`. Set the checkpoint
environment variables listed in `tests/test_adapters/test_surrogate_gpu.py`,
then run:

```powershell
uv run pytest -m gpu tests/test_adapters/test_surrogate_gpu.py -q
```
