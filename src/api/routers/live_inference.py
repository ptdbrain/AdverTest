"""Live Model Inference & Dynamic Adversarial Evaluation Pipeline.

Executes visual inference on one image and preserves project-scoped evidence.

This endpoint is deliberately not a dataset benchmark: it reports detections
and image perturbation measurements, never AP/mAP/mIoU/NDS or promotion data.
"""

from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image, ImageEnhance
from pydantic import BaseModel, ConfigDict, Field
from ultralytics import YOLO

from src.api.platform_dependencies import get_platform_artifacts, require_project_member
from src.attacks.corruption._base import corrupt as imagecorruptions_corrupt
from src.core.platform_contracts import ArtifactKind, ArtifactState
from src.storage.service import ArtifactService

router = APIRouter(prefix="/runs/live-inference", tags=["Live Model Inference"])


class LiveInferenceRequest(BaseModel):
    model_id: str = Field(default="local_yolo11s_clean", description="Model checkpoint identifier")
    sample_id: str = Field(default="000000", description="Sample image ID (e.g. 000000, 000002, 000003)")
    attack_type: str = Field(
        default="depth_fog",
        description="Attack type or comma-separated recipe (depth_rain, snow, depth_fog, pgd, motion_blur, gaussian_noise)",
    )
    severity: int = Field(default=3, ge=1, le=5, description="Severity ladder 1..5")
    run_id: str = Field(min_length=1, max_length=128, description="Existing visual or benchmark run namespace")
    source_artifact_id: str = Field(min_length=1, description="Project-scoped uploaded image artifact")


class DetectionItem(BaseModel):
    label: str
    conf: float
    box: list[float]  # [x1, y1, x2, y2]
    style: dict[str, str]  # { left, top, width, height }
    color: str
    status: str = "normal"


class PredictionStat(BaseModel):
    name: str
    conf: float


class DetectionSummary(BaseModel):
    bbox_count: int = Field(alias="bboxCount")
    conf_avg: float | None = Field(alias="confAvg")
    top: list[PredictionStat]

    model_config = ConfigDict(populate_by_name=True)


class LiveInferenceResponse(BaseModel):
    project_id: str
    run_id: str
    visual_only: bool = True
    evidence_status: str = "NOT_ELIGIBLE"
    sample_id: str
    filename: str
    resolution: str
    model_name: str
    total_model_classes: int
    inference_time_clean_ms: float
    inference_time_attacked_ms: float
    clean_image_url: str
    attacked_image_url: str
    diff_image_url: str
    perturbation_image_url: str
    clean_detections: list[DetectionItem]
    attacked_detections: list[DetectionItem]
    clean_stats: DetectionSummary
    atk_stats: DetectionSummary
    l2_norm: float
    linf: str
    psnr: str
    ssim: str
    observations: list[str]
    metrics: dict[str, Any]


def _format_box_to_style(xyxy: list[float], img_w: int, img_h: int) -> dict[str, str]:
    x1, y1, x2, y2 = xyxy
    left_pct = (x1 / img_w) * 100
    top_pct = (y1 / img_h) * 100
    width_pct = ((x2 - x1) / img_w) * 100
    height_pct = ((y2 - y1) / img_h) * 100
    return {
        "left": f"{left_pct:.2f}%",
        "top": f"{top_pct:.2f}%",
        "width": f"{width_pct:.2f}%",
        "height": f"{height_pct:.2f}%",
    }


def _get_color_for_label(label: str) -> str:
    lbl = label.lower()
    if lbl in ("person", "pedestrian"):
        return "teal"
    if lbl in ("car", "van"):
        return "emerald"
    if lbl in ("truck", "bus", "train"):
        return "blue"
    if lbl in ("bicycle", "motorcycle"):
        return "amber"
    return "purple"


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity index on luminance channel."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(gray1 * gray1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 * gray2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def _image_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def _store_visual_artifact(
    artifacts: ArtifactService,
    *,
    project_id: str,
    run_id: str,
    actor_id: str,
    filename: str,
    content: bytes,
    mime_type: str,
    kind: ArtifactKind,
) -> str:
    artifact = artifacts.create_internal(
        project_id=project_id,
        run_id=run_id,
        actor_id=actor_id,
        kind=kind,
        original_filename=filename,
        mime_type=mime_type,
        content=content,
        state=ArtifactState.READY,
        metadata={"visual_only": True},
    )
    return f"/api/v1/projects/{project_id}/runs/{run_id}/artifacts/{artifact['id']}/content"


def _apply_realistic_rain(img_arr: np.ndarray, severity: int = 3) -> np.ndarray:
    """Generate authentic depth rain streaks and atmospheric contrast attenuation."""
    h, w, c = img_arr.shape
    enhancer = ImageEnhance.Brightness(Image.fromarray(img_arr))
    dimmed = np.array(enhancer.enhance(max(0.65, 0.90 - severity * 0.06)))

    rain_layer = np.zeros((h, w), dtype=np.float32)
    num_drops = int(w * h * 0.0009 * severity)

    xs = np.random.randint(0, w, num_drops)
    ys = np.random.randint(0, h, num_drops)

    length = 14 + severity * 7
    slant = int(length * 0.35)

    for x, y in zip(xs, ys):
        cv2.line(rain_layer, (x, y), (x - slant, y + length), 255, thickness=1 + (1 if severity >= 4 else 0))

    rain_blur = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    rain_rgb = np.stack([rain_blur, rain_blur, rain_blur], axis=2) / 255.0
    return (dimmed.astype(np.float32) * (1.0 - rain_rgb * 0.35) + rain_rgb * 230 * 0.65).clip(0, 255).astype(np.uint8)


@router.post("", response_model=LiveInferenceResponse)
async def run_live_inference(
    payload: LiveInferenceRequest,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    artifacts: ArtifactService = Depends(get_platform_artifacts),
) -> LiveInferenceResponse:
    """Run image-level visual inference without making a benchmark claim."""
    if imagecorruptions_corrupt is None:
        raise HTTPException(
            status_code=503,
            detail="Image corruption runtime is unavailable; live visual corruption cannot be executed.",
        )
    sample_stem = payload.sample_id.replace(".png", "")
    try:
        source_bytes = artifacts.read_bytes(project_id, payload.source_artifact_id, actor_id=actor_id)
    except (KeyError, FileNotFoundError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="SOURCE_ARTIFACT_NOT_FOUND") from exc
    try:
        img_pil = Image.open(io.BytesIO(source_bytes)).convert("RGB")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="SOURCE_ARTIFACT_IS_NOT_A_DECODABLE_IMAGE") from exc
    w, h = img_pil.size
    arr_clean = np.array(img_pil)

    # 1. Load model
    checkpoint_path = Path("checkpoints/surrogates/yolo11s.pt")
    model = YOLO(str(checkpoint_path))

    # 2. Real Clean Inference
    with tempfile.TemporaryDirectory(prefix="advertest-live-") as temporary_directory:
        clean_path = Path(temporary_directory) / "clean.png"
        img_pil.save(clean_path)
        t0 = time.perf_counter()
        res_clean = model(clean_path, verbose=False)[0]
        clean_duration_ms = (time.perf_counter() - t0) * 1000

    clean_detections: list[DetectionItem] = []
    for b in res_clean.boxes:
        conf = float(b.conf[0])
        if conf >= 0.25:
            cls_id = int(b.cls[0])
            cls_name = model.names[cls_id]
            box_xyxy = [float(x) for x in b.xyxy[0].tolist()]
            clean_detections.append(
                DetectionItem(
                    label=cls_name.capitalize(),
                    conf=round(conf, 2),
                    box=box_xyxy,
                    style=_format_box_to_style(box_xyxy, w, h),
                    color=_get_color_for_label(cls_name),
                    status="normal",
                )
            )

    # 3. Dynamic Sequential Attack Pipeline Execution
    raw_attacks = [a.strip().lower() for a in payload.attack_type.split(",") if a.strip()]
    if not raw_attacks:
        raw_attacks = ["depth_fog"]

    arr_current = arr_clean.copy()
    sev = max(1, min(5, payload.severity))

    for atk in raw_attacks:
        if "rain" in atk:
            arr_current = _apply_realistic_rain(arr_current, severity=sev)
        elif "snow" in atk:
            arr_current = imagecorruptions_corrupt(arr_current, corruption_name="snow", severity=sev)
        elif "frost" in atk:
            arr_current = imagecorruptions_corrupt(arr_current, corruption_name="frost", severity=sev)
        elif "fog" in atk or "depth_fog" in atk:
            arr_current = imagecorruptions_corrupt(arr_current, corruption_name="fog", severity=sev)
        elif "blur" in atk or "motion" in atk:
            arr_current = imagecorruptions_corrupt(arr_current, corruption_name="motion_blur", severity=sev)
        elif "noise" in atk or "pgd" in atk or "fgsm" in atk:
            arr_current = imagecorruptions_corrupt(arr_current, corruption_name="gaussian_noise", severity=sev)
        else:
            arr_current = imagecorruptions_corrupt(arr_current, corruption_name="fog", severity=sev)

    # 4. Compute visual differences. The attacked image is temporary only while
    # inference runs; persistent results go through project/run artifact storage.
    attacked_image = Image.fromarray(arr_current)

    # Compute difference and metrics
    diff = np.abs(arr_current.astype(np.float32) - arr_clean.astype(np.float32))
    diff_gray = np.mean(diff, axis=2)
    diff_norm = np.clip((diff_gray / (diff_gray.max() + 1e-5)) * 255 * 2.2, 0, 255).astype(np.uint8)
    diff_colored = cv2.applyColorMap(diff_norm, cv2.COLORMAP_INFERNO)
    diff_img = Image.fromarray(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB))

    perturbation = arr_current.astype(np.float32) - arr_clean.astype(np.float32)
    pert_vis = ((perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-5) * 255).astype(
        np.uint8
    )
    perturbation_image = Image.fromarray(pert_vis)

    # 5. Real Attacked Inference on All 80 Classes
    t1 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="advertest-live-") as temporary_directory:
        dynamic_atk_path = Path(temporary_directory) / "attacked.png"
        attacked_image.save(dynamic_atk_path)
        res_atk = model(dynamic_atk_path, verbose=False)[0]
    attacked_duration_ms = (time.perf_counter() - t1) * 1000

    attacked_detections: list[DetectionItem] = []
    for b in res_atk.boxes:
        conf = float(b.conf[0])
        if conf >= 0.25:
            cls_id = int(b.cls[0])
            cls_name = model.names[cls_id]
            box_xyxy = [float(x) for x in b.xyxy[0].tolist()]
            status = "low_conf" if conf < 0.6 else "degraded"
            attacked_detections.append(
                DetectionItem(
                    label=cls_name.capitalize(),
                    conf=round(conf, 2),
                    box=box_xyxy,
                    style=_format_box_to_style(box_xyxy, w, h),
                    color=_get_color_for_label(cls_name),
                    status=status,
                )
            )

    # 6. Compute image-level values and prediction deltas, not benchmark metrics.
    clean_count = len(clean_detections)
    attacked_count = len(attacked_detections)
    avg_clean_conf = round(float(np.mean([d.conf for d in clean_detections])), 2) if clean_detections else None
    avg_atk_conf = round(float(np.mean([d.conf for d in attacked_detections])), 2) if attacked_detections else None

    top_clean = [PredictionStat(name=d.label.lower(), conf=d.conf) for d in clean_detections[:3]]
    top_atk = [PredictionStat(name=d.label.lower(), conf=d.conf) for d in attacked_detections[:3]]

    l2_norm_val = round(float(np.linalg.norm(diff) / np.sqrt(diff.size)), 2)
    linf_val = f"{float(np.max(diff) / 255.0):.4f}"
    psnr_val = f"{float(cv2.PSNR(arr_clean, arr_current)):.2f} dB"
    ssim_val = f"{_compute_ssim(arr_clean, arr_current):.3f}"

    attack_vector_str = " + ".join(a.replace("_", " ").title() for a in raw_attacks) + f" (Cấp {payload.severity})"

    observations = [
        "Visual inference only: this single image does not establish benchmark accuracy or robustness.",
        f"[{attack_vector_str}] changed detections from {clean_count} to {attacked_count}.",
        f"PSNR {psnr_val} and SSIM {ssim_val} describe image perturbation, not model accuracy.",
    ]

    clean_url = _store_visual_artifact(
        artifacts, project_id=project_id, run_id=payload.run_id, actor_id=actor_id,
        filename=f"{sample_stem}_clean.png", content=_image_bytes(img_pil), mime_type="image/png", kind=ArtifactKind.EVIDENCE,
    )
    attacked_url = _store_visual_artifact(
        artifacts, project_id=project_id, run_id=payload.run_id, actor_id=actor_id,
        filename=f"{sample_stem}_attacked.png", content=_image_bytes(attacked_image), mime_type="image/png", kind=ArtifactKind.EVIDENCE,
    )
    diff_url = _store_visual_artifact(
        artifacts, project_id=project_id, run_id=payload.run_id, actor_id=actor_id,
        filename=f"{sample_stem}_diff.png", content=_image_bytes(diff_img), mime_type="image/png", kind=ArtifactKind.EVIDENCE,
    )
    perturbation_url = _store_visual_artifact(
        artifacts, project_id=project_id, run_id=payload.run_id, actor_id=actor_id,
        filename=f"{sample_stem}_perturbation.png", content=_image_bytes(perturbation_image), mime_type="image/png", kind=ArtifactKind.EVIDENCE,
    )
    _store_visual_artifact(
        artifacts, project_id=project_id, run_id=payload.run_id, actor_id=actor_id,
        filename=f"{sample_stem}_predictions.json",
        content=json.dumps({"clean": [item.model_dump() for item in clean_detections], "attacked": [item.model_dump() for item in attacked_detections]}).encode(),
        mime_type="application/json", kind=ArtifactKind.PREDICTION,
    )

    return LiveInferenceResponse(
        project_id=project_id,
        run_id=payload.run_id,
        sample_id=sample_stem,
        filename=f"{sample_stem}.png (KITTI Dataset)",
        resolution=f"{w} × {h} px",
        model_name="YOLO11s (Ultralytics Vision)",
        total_model_classes=len(model.names),
        inference_time_clean_ms=round(clean_duration_ms, 2),
        inference_time_attacked_ms=round(attacked_duration_ms, 2),
        clean_image_url=clean_url,
        attacked_image_url=attacked_url,
        diff_image_url=diff_url,
        perturbation_image_url=perturbation_url,
        clean_detections=clean_detections,
        attacked_detections=attacked_detections,
        clean_stats=DetectionSummary(bboxCount=clean_count, confAvg=avg_clean_conf, top=top_clean),
        atk_stats=DetectionSummary(bboxCount=attacked_count, confAvg=avg_atk_conf, top=top_atk),
        l2_norm=l2_norm_val,
        linf=linf_val,
        psnr=psnr_val,
        ssim=ssim_val,
        observations=observations,
        metrics={
            "cleanCount": clean_count,
            "attackedCount": attacked_count,
            "detectionCountDelta": attacked_count - clean_count,
            "meanConfidenceDelta": (
                round(avg_atk_conf - avg_clean_conf, 4)
                if avg_clean_conf is not None and avg_atk_conf is not None
                else None
            ),
            "attackVector": attack_vector_str,
        },
    )
