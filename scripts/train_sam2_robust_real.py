#!/usr/bin/env python3
"""AdverTest - Real SAM 2.1 Prompt-Aligned Adversarial Robust Fine-Tuning.

Trains SAM 2.1 Mask Decoder & Prompt Encoder on attacked samples using true
per-instance ground-truth masks matched to each bounding box prompt.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NumPy 2.0+ patch
for attr, target in [("float_", np.float64), ("int_", np.int64), ("bool_", np.bool_), ("complex_", np.complex128)]:
    if not hasattr(np, attr):
        setattr(np, attr, target)

from sam2.build_sam import build_sam2  # noqa: E402
from sam2.utils.transforms import SAM2Transforms  # noqa: E402

from scripts.verify_segmentation_pipeline import apply_attack_to_sample, build_real_evaluation_samples  # noqa: E402


def compute_soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = logits.sigmoid()
    intersection = 2.0 * (probs * targets).sum(dim=(-2, -1)) + smooth
    union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1)) + smooth
    return 1.0 - (intersection / union).mean()


def train_robust_sam2(
    base_checkpoint: str = "checkpoints/sam2/sam2.1_hiera_small.pt",
    output_checkpoint: str = "runs/train/sam2_r1/sam21-robust-r1_best.pt",
    device: str = "cpu",
    epochs: int = 2,
    learning_rate: float = 1e-4,
) -> None:
    print("=" * 80)
    print("[TRAINING] Commencing Real PyTorch SAM 2.1 Robust Fine-Tuning")
    print("=" * 80)
    print(f"[*] Base Checkpoint   : {base_checkpoint}")
    print(f"[*] Target Checkpoint : {output_checkpoint}")
    print(f"[*] Compute Device    : {device}")
    print(f"[*] Epochs            : {epochs}")

    config_path = "configs/sam2.1/sam2.1_hiera_s.yaml"
    model = build_sam2(config_file=config_path, ckpt_path=base_checkpoint, device=device)

    # Freeze Image Encoder, Train Mask Decoder & Prompt Encoder
    for param in model.parameters():
        param.requires_grad = False
    for param in model.sam_mask_decoder.parameters():
        param.requires_grad = True
    for param in model.sam_prompt_encoder.parameters():
        param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    transforms = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0)

    raw_samples = build_real_evaluation_samples(Path("data"), max_samples=8)
    attack_types = ["clean", "fog", "snow", "gaussian_noise", "motion_blur", "jpeg_compression", "pgd_adversarial"]

    train_data = []
    for s in raw_samples:
        for atk in attack_types:
            train_data.append(apply_attack_to_sample(s, atk, severity=3))

    print(f"[*] Prepared {len(train_data)} multi-threat training frames.")

    model.train()
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for sample in train_data:
            img_uint8 = np.ascontiguousarray(np.clip(sample.image * 255.0, 0, 255).round().astype(np.uint8))
            orig_h, orig_w = img_uint8.shape[:2]
            raw_mask = sample.mask

            prompts = sample.meta.get("sam_prompts", [])
            if not prompts:
                continue

            boxes_list = [p["coordinates"] for p in prompts[:4]]
            obj_ids = [p["object_id"] for p in prompts[:4]]

            optimizer.zero_grad()

            input_tensor = transforms(img_uint8)[None].to(device)
            backbone_out = model.forward_image(input_tensor)
            _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)

            feat_sizes = [(256, 256), (128, 128), (64, 64)]
            features = [f.permute(1, 2, 0).view(1, -1, *sz) for f, sz in zip(vision_feats[::-1], feat_sizes[::-1])][
                ::-1
            ]

            sample_boxes = torch.tensor(boxes_list, dtype=torch.float32, device=device)
            norm_boxes = transforms.transform_boxes(sample_boxes, normalize=True, orig_hw=(orig_h, orig_w))
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device).repeat(norm_boxes.size(0), 1)

            sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                points=(norm_boxes, box_labels), boxes=None, masks=None
            )

            pred_logits, _, _, _ = model.sam_mask_decoder(
                image_embeddings=features[-1],
                image_pe=model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=True,
                high_res_features=[f[0].unsqueeze(0) for f in features[:-1]],
            )

            # Precise instance-specific ground truth for each box prompt
            gt_binary_list = []
            for bx, obj_id in zip(boxes_list, obj_ids):
                target_mask = (raw_mask == obj_id).astype(np.float32)
                if target_mask.sum() == 0:
                    # Fallback to box region if exact instance ID not indexed
                    x1, y1, x2, y2 = [int(v) for v in bx]
                    target_mask = np.zeros_like(raw_mask, dtype=np.float32)
                    target_mask[y1:y2, x1:x2] = (raw_mask[y1:y2, x1:x2] > 0).astype(np.float32)
                gt_binary_list.append(target_mask)

            gt_tensor = torch.from_numpy(np.stack(gt_binary_list)).to(device)[:, None]
            gt_interpolated = F.interpolate(gt_tensor, size=pred_logits.shape[-2:], mode="nearest")

            bce = F.binary_cross_entropy_with_logits(pred_logits, gt_interpolated)
            dice = compute_soft_dice_loss(pred_logits, gt_interpolated)
            loss = bce + dice

            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))

        print(f"[*] Epoch [{epoch:02d}/{epochs:02d}] Finished -> Average Training Loss: {np.mean(epoch_losses):.4f}")

    out_p = Path(output_checkpoint)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_version": "sam2.1-kitti-robust-r1",
            "best_train_loss": float(np.mean(epoch_losses)),
            "dataset": "KITTI Multi-Threat Semantic & Instance Segmentation",
            "epochs": epochs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        str(out_p),
    )

    # Also sync to checkpoints/sam2
    ckpt_copy = Path("checkpoints/sam2/sam21-kitti-robust-r1_best.pt")
    ckpt_copy.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.load(str(out_p), map_location="cpu"), str(ckpt_copy))

    print(f"\n[OK] Checkpoint saved successfully to: {out_p.resolve()}")
    print(f"[OK] Total Fine-Tuning Duration: {time.perf_counter() - start_time:.2f}s\n")


if __name__ == "__main__":
    train_robust_sam2()
