# SAM2 segmentation protocol (Người C)

## Dataset boundary

- `Cityscapes` is the in-domain dataset. Only reviewed instance masks may enter validation or the locked benchmark.
- `BDD100K Segmentation` is external-only. It never selects a checkpoint or tunes a threshold.
- Canonical classes are Person, Rider, Car, Truck, Bus, Bicycle, and Motorcycle. UI story groups are Vehicle, Pedestrian, and Cyclist-Rider.
- Instance masks use zero for background and positive integer instance IDs. `meta.instance_labels` maps IDs to canonical classes; `meta.mask_reviewed` and `meta.mask_source` preserve review provenance.

## Fixed prompt benchmark

Each evaluated object receives a ground-truth box `SegmentationPrompt`. The identical prompt coordinates and object ID must be stored in the benchmark manifest and reused for clean/attacked plus baseline/robust inference. YOLO-predicted boxes are not valid for this independent SAM benchmark.

## Failure and overlay contract

`sam-mask-failure-v1` declares a failure for empty predictions, wrong-object masks, severe split/merge, IoU below 0.50, or Boundary IoU below 0.50. UI payloads provide GT, prediction, missing, extra, boundary, score, prompt, and the explicit failure reason; green/amber/red colouring must be supplemented by text/icon.

## Attack handoff

SAM-PGD maximises segmentation BCE through a fixed GT-box prompt. The prompt never changes during attack generation. Spatial attacks must transform image, instance mask, and valid region together; partial occlusion cannot retain an unmodified mask as GT. Full SAM-PGD belongs in benchmark/hard-example generation; R1 uses only lightweight adversarial replay (5%).
