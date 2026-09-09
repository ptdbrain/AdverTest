"""Build the small, deterministic image/report bundle used by the fake demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1] / "data" / "demo" / "fake-sessions"
ASSET_ROOT = ROOT / "assets"
WIDTH, HEIGHT = 960, 540


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    fixtures = [
        build_detection_fixture(
            fixture_id="demo-robust-001",
            session_id="DEMO-ROBUST-001",
            project_id="demo-project-robust-001",
            baseline_run_id="demo-run-robust-001",
            defence_run_id="demo-defence-robust-001",
            review_id="demo-review-robust-001",
            title="Vehicle Safety Robustness",
            attack="occlusion",
            severity=4,
            clean_ap=0.91,
            attacked_ap=0.24,
            defence_ap=0.84,
            prefix="robust",
            defence_checkpoint_id="yolo11s-kitti-robust-r1",
        ),
        build_segmentation_fixture(
            fixture_id="demo-seg-002",
            session_id="DEMO-SEG-002",
            project_id="demo-project-seg-002",
            baseline_run_id="demo-run-seg-002",
            defence_run_id="demo-defence-seg-002",
            review_id="demo-review-seg-002",
            title="Urban Scene Segmentation",
            attack="motion_blur",
            severity=3,
            clean_miou=0.89,
            attacked_miou=0.47,
            defence_miou=0.82,
            prefix="seg",
            defence_checkpoint_id="sam21-robust-r1",
        ),
        build_detection_fixture(
            fixture_id="demo-review-003",
            session_id="DEMO-REVIEW-003",
            project_id="demo-project-review-003",
            baseline_run_id="demo-run-review-003",
            defence_run_id="demo-defence-review-003",
            review_id="demo-review-review-003",
            title="Production Review Drill",
            attack="brightness",
            severity=3,
            clean_ap=0.88,
            attacked_ap=0.39,
            defence_ap=0.81,
            prefix="review",
            defence_checkpoint_id="yolo11s-kitti-robust-r1",
        ),
    ]
    (ROOT / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "fixtures": fixtures}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def base_entry(**kwargs: Any) -> dict[str, Any]:
    return {
        "fixture_id": kwargs["fixture_id"],
        "session_id": kwargs["session_id"],
        "project_id": kwargs["project_id"],
        "baseline_run_id": kwargs["baseline_run_id"],
        "defence_run_id": kwargs["defence_run_id"],
        "defence_checkpoint_id": kwargs["defence_checkpoint_id"],
        "task_id": kwargs["task_id"],
        "task_name": kwargs["task_name"],
        "model_id": kwargs["model_id"],
        "model_family_id": kwargs["model_family_id"],
        "model_name": kwargs["model_name"],
        "dataset_id": kwargs["dataset_id"],
        "dataset_name": kwargs["dataset_name"],
        "report": f"{kwargs['fixture_id']}/report.json",
        "defence_report": f"{kwargs['fixture_id']}/defence_report.json",
        "assets": kwargs["assets"],
        "review": {
            "review_id": kwargs["review_id"],
            "run_id": kwargs["baseline_run_id"],
            "attack": kwargs["attack"],
            "severity": kwargs["severity"],
            "degradation": kwargs["degradation"],
            "dataset": kwargs["dataset_id"],
            "model": kwargs["model_name"],
            "flagged_by": "demo_fixture",
            "notes": kwargs["review_notes"],
            "status": kwargs.get("review_status", "PENDING"),
            "decision": kwargs.get("review_decision"),
            "decision_note": kwargs.get("review_decision_note"),
            "resolved_by": kwargs.get("review_resolved_by"),
            "risk_level": kwargs["risk_level"],
            "risk_category": kwargs["risk_category"],
            "affected_class": kwargs["affected_class"],
        },
        "sample_reviews": kwargs["sample_reviews"],
    }


def build_detection_fixture(**kwargs: Any) -> dict[str, Any]:
    prefix = kwargs["prefix"]
    asset_names = [
        f"{prefix}-clean.png",
        f"{prefix}-attacked.png",
        f"{prefix}-clean-prediction.png",
        f"{prefix}-attacked-prediction.png",
        f"{prefix}-defence-attacked.png",
        f"{prefix}-defence-prediction.png",
        f"{prefix}-diff.png",
        f"{prefix}-perturbation.png",
    ]
    for name in asset_names:
        draw_detection_asset(ASSET_ROOT / name, prefix, name)
    samples = [detection_sample(prefix, index, kwargs["attack"], kwargs["severity"]) for index in range(1, 4)]
    defence_samples = [
        detection_sample(prefix, index, kwargs["attack"], kwargs["severity"], defence=True)
        for index in range(1, 4)
    ]
    baseline = detection_report(
        kwargs["baseline_run_id"], "yolo11", "yolo11s-base", kwargs["dataset_id"] if "dataset_id" in kwargs else "kitti",
        kwargs["clean_ap"], kwargs["attacked_ap"], samples, kwargs["attack"], kwargs["severity"], prefix,
    )
    defence = detection_report(
        kwargs["defence_run_id"], "yolo11", "yolo11s-kitti-robust-r1", kwargs["dataset_id"] if "dataset_id" in kwargs else "kitti",
        kwargs["clean_ap"] * 0.98, kwargs["defence_ap"], defence_samples, kwargs["attack"], kwargs["severity"], prefix,
    )
    return write_fixture(
        kwargs,
        task_id="detection2d",
        task_name="Object Detection (2D)",
        model_id="yolo11s-base",
        model_family_id="yolo11",
        model_name="YOLO11s (Ultralytics)",
        dataset_id="kitti",
        dataset_name="KITTI 2D — Demo Scene Set",
        baseline=baseline,
        defence=defence,
        asset_names=asset_names,
        degradation=round((kwargs["clean_ap"] - kwargs["attacked_ap"]) / kwargs["clean_ap"], 4),
        risk_level="HIGH" if prefix == "robust" else "CRITICAL",
        risk_category=f"adversarial_{prefix}",
        affected_class="Car",
        review_notes="Synthetic review fixture prepared for the presentation journey.",
        review_status="RESOLVED" if prefix == "review" else "PENDING",
        review_decision="BLOCK_DEPLOY" if prefix == "review" else None,
        review_decision_note=(
            "Synthetic production review: hold deployment until robust checkpoint is selected."
            if prefix == "review"
            else None
        ),
        review_resolved_by="Demo Engineer" if prefix == "review" else None,
    )


def build_segmentation_fixture(**kwargs: Any) -> dict[str, Any]:
    prefix = kwargs["prefix"]
    asset_names = [
        f"{prefix}-clean.png",
        f"{prefix}-attacked.png",
        f"{prefix}-clean-prediction.png",
        f"{prefix}-attacked-prediction.png",
        f"{prefix}-defence-attacked.png",
        f"{prefix}-defence-prediction.png",
        f"{prefix}-diff.png",
        f"{prefix}-perturbation.png",
    ]
    for name in asset_names:
        draw_segmentation_asset(ASSET_ROOT / name, prefix, name)
    samples = [segmentation_sample(prefix, index, kwargs["attack"], kwargs["severity"]) for index in range(1, 4)]
    defence_samples = [
        segmentation_sample(prefix, index, kwargs["attack"], kwargs["severity"], defence=True)
        for index in range(1, 4)
    ]
    baseline = segmentation_report(
        kwargs["baseline_run_id"], "sam2", "sam2.1-hiera-small-v1", kwargs["clean_miou"], kwargs["attacked_miou"],
        samples, kwargs["attack"], kwargs["severity"], prefix,
    )
    defence = segmentation_report(
        kwargs["defence_run_id"], "sam2", "sam21-robust-r1", kwargs["clean_miou"] * 0.99, kwargs["defence_miou"],
        defence_samples, kwargs["attack"], kwargs["severity"], prefix,
    )
    return write_fixture(
        kwargs,
        task_id="segmentation",
        task_name="Instance Segmentation (2D)",
        model_id="sam2-hiera-small-base",
        model_family_id="sam2",
        model_name="SAM2.1 Hiera Small",
        dataset_id="cityscapes_segmentation",
        dataset_name="Cityscapes Instance — Demo Scene Set",
        baseline=baseline,
        defence=defence,
        asset_names=asset_names,
        degradation=round((kwargs["clean_miou"] - kwargs["attacked_miou"]) / kwargs["clean_miou"], 4),
        risk_level="HIGH",
        risk_category="weather_segmentation",
        affected_class="Car",
        review_notes="Synthetic mask review fixture prepared for the presentation journey.",
    )


def write_fixture(kwargs: dict[str, Any], **details: Any) -> dict[str, Any]:
    fixture_dir = ROOT / kwargs["fixture_id"]
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "report.json").write_text(
        json.dumps(details["baseline"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (fixture_dir / "defence_report.json").write_text(
        json.dumps(details["defence"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assets = [f"assets/{name}" for name in details["asset_names"]]
    return base_entry(
        fixture_id=kwargs["fixture_id"],
        session_id=kwargs["session_id"],
        project_id=kwargs["project_id"],
        baseline_run_id=kwargs["baseline_run_id"],
        defence_run_id=kwargs["defence_run_id"],
        defence_checkpoint_id=kwargs["defence_checkpoint_id"],
        review_id=kwargs["review_id"],
        assets=assets,
        task_id=details["task_id"],
        task_name=details["task_name"],
        model_id=details["model_id"],
        model_family_id=details["model_family_id"],
        model_name=details["model_name"],
        dataset_id=details["dataset_id"],
        dataset_name=details["dataset_name"],
        attack=kwargs["attack"],
        severity=kwargs["severity"],
        degradation=details["degradation"],
        risk_level=details["risk_level"],
        risk_category=details["risk_category"],
        affected_class=details["affected_class"],
        review_notes=details["review_notes"],
        review_status=details.get("review_status", "PENDING"),
        review_decision=details.get("review_decision"),
        review_decision_note=details.get("review_decision_note"),
        review_resolved_by=details.get("review_resolved_by"),
        sample_reviews=[
            {
                "sample_id": f"{kwargs['prefix']}/scene-{index:02d}",
                "attack": kwargs["attack"],
                "severity": kwargs["severity"],
                "decision": "PENDING" if index != 1 else "APPROVED",
                "note": "Synthetic evidence reviewed in demo flow.",
                "reviewed_by": "Demo Engineer",
            }
            for index in range(1, 4)
        ],
    )


def detection_report(run_id: str, model: str, model_version: str, dataset: str, clean_ap: float, attacked_ap: float,
                     samples: list[dict[str, Any]], attack: str, severity: int, prefix: str) -> dict[str, Any]:
    degradation = round((clean_ap - attacked_ap) / clean_ap, 4)
    return common_report(
        run_id, model, model_version, dataset, clean_ap, attacked_ap, samples, attack, severity, prefix,
        clean_metrics={"ap50": clean_ap, "map50_95": round(clean_ap * 0.95, 4), "mean_confidence": 0.87},
        attacked_metrics={"ap50": attacked_ap, "map50_95": round(attacked_ap * 0.92, 4), "mean_confidence": 0.31},
        metric_name="ap50", degradation=degradation,
    )


def segmentation_report(run_id: str, model: str, model_version: str, clean_miou: float, attacked_miou: float,
                         samples: list[dict[str, Any]], attack: str, severity: int, prefix: str) -> dict[str, Any]:
    degradation = round((clean_miou - attacked_miou) / clean_miou, 4)
    return common_report(
        run_id, model, model_version, "cityscapes_segmentation", clean_miou, attacked_miou, samples, attack, severity, prefix,
        clean_metrics={"miou": clean_miou, "mIoU": clean_miou, "mean_confidence": 0.91},
        attacked_metrics={"miou": attacked_miou, "mIoU": attacked_miou, "mean_confidence": 0.46},
        metric_name="miou", degradation=degradation,
    )


def common_report(run_id: str, model: str, model_version: str, dataset: str, clean_value: float, attacked_value: float,
                  samples: list[dict[str, Any]], attack: str, severity: int, prefix: str, clean_metrics: dict[str, Any],
                  attacked_metrics: dict[str, Any], metric_name: str, degradation: float) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model": model,
        "model_version": model_version,
        "dataset": dataset,
        "n_samples": len(samples),
        "ap_clean": clean_value,
        "cells": [{
            "attack": attack,
            "group": "C" if attack == "occlusion" else "A",
            "severity": severity,
            "ap": attacked_value,
            "degradation": degradation,
            "degradation_ratio": degradation,
            "degradation_percent": round(degradation * 100, 1),
            "unit": "ratio",
            "n_samples": len(samples),
            "seconds": 0.18,
            "cache_hits": 0,
            "category": "demo_fixture",
            "metrics": attacked_metrics,
            "provenance": {"source": "demo_fixture"},
        }],
        "heatmap": {attack: {str(severity): attacked_value}},
        "worst_cases": samples[:1],
        "skipped": [],
        "sample_results": samples,
        "metrics": {"clean": clean_metrics, "robustness": {"attack_success_rate": degradation}},
        "provenance": {"demo_fixture": True, "run_config": {"task_id": "segmentation" if model == "sam2" else "detection2d"}},
        "seconds": 0.18,
        "simulation_only": True,
        "benchmark_metrics_available": True,
        "needs_review": True,
    }


def detection_sample(prefix: str, index: int, attack: str, severity: int, defence: bool = False) -> dict[str, Any]:
    stem = f"{prefix}/scene-{index:02d}"
    if defence:
        attacked_image = f"assets/{prefix}-defence-attacked.png"
        attacked_prediction = f"assets/{prefix}-defence-prediction.png"
        attacked_boxes = [{"xyxy": [124, 184, 304, 414], "label": "Car", "score": 0.79}, {"xyxy": [438, 195, 540, 301], "label": "Car", "score": 0.62}]
    else:
        attacked_image = f"assets/{prefix}-attacked.png"
        attacked_prediction = f"assets/{prefix}-attacked-prediction.png"
        attacked_boxes = [{"xyxy": [145, 205, 270, 370], "label": "Car", "score": 0.31}]
    clean_boxes = [{"xyxy": [112, 180, 306, 420], "label": "Car", "score": 0.94}, {"xyxy": [432, 190, 542, 304], "label": "Car", "score": 0.82}]
    artifacts = {
        "clean_input_url": f"/data/demo/fake-sessions/assets/{prefix}-clean.png",
        "attacked_input_url": f"/data/demo/fake-sessions/{attacked_image}",
        "clean_prediction_url": f"/data/demo/fake-sessions/assets/{prefix}-clean-prediction.png",
        "attacked_prediction_url": f"/data/demo/fake-sessions/{attacked_prediction}",
        "diff_url": f"/data/demo/fake-sessions/assets/{prefix}-diff.png",
        "perturbation_url": f"/data/demo/fake-sessions/assets/{prefix}-perturbation.png",
        "zoom_clean_url": f"/data/demo/fake-sessions/assets/{prefix}-clean-prediction.png",
        "zoom_attacked_url": f"/data/demo/fake-sessions/{attacked_prediction}",
    }
    return {
        "sample_id": stem,
        "attack": attack,
        "severity": severity,
        "clean_prediction": {"prediction_type": "detection", "sample_id": stem, "latency_ms": 48.2, "boxes": clean_boxes},
        "attacked_prediction": {"prediction_type": "detection", "sample_id": stem, "latency_ms": 49.7, "boxes": attacked_boxes},
        "object_evidence": [{"object_id": f"{stem}:car-1", "status_attacked": "correct" if defence else "confidence_collapsed", "clean_confidence": 0.94, "attacked_confidence": attacked_boxes[0]["score"]}],
        "degradation_hint": 0.09 if defence else 0.76,
        "attack_version": "demo-fixture-1.0",
        "attack_params": {"severity": severity},
        "ground_truth": {"type": "boxes", "image_width": WIDTH, "image_height": HEIGHT, "objects": [{"object_id": f"{stem}:0", "label": "Car", "xyxy": [108, 178, 310, 424]}]},
        "clean_image_path": f"assets/{prefix}-clean.png",
        "attacked_image_path": attacked_image,
        "clean_prediction_path": f"assets/{prefix}-clean-prediction.png",
        "attacked_prediction_path": attacked_prediction,
        "artifacts": artifacts,
    }


def segmentation_sample(prefix: str, index: int, attack: str, severity: int, defence: bool = False) -> dict[str, Any]:
    stem = f"{prefix}/scene-{index:02d}"
    attacked_image = f"assets/{prefix}-defence-attacked.png" if defence else f"assets/{prefix}-attacked.png"
    attacked_prediction = f"assets/{prefix}-defence-prediction.png" if defence else f"assets/{prefix}-attacked-prediction.png"
    clean_instances = [
        {"instance_id": 1, "label": "Road", "score": 0.95},
        {"instance_id": 2, "label": "Car", "score": 0.91},
        {"instance_id": 3, "label": "Building", "score": 0.88},
    ]
    attacked_instances = clean_instances if defence else [
        {"instance_id": 1, "label": "Road", "score": 0.62},
        {"instance_id": 2, "label": "Car", "score": 0.43},
    ]
    artifacts = {
        "clean_input_url": f"/data/demo/fake-sessions/assets/{prefix}-clean.png",
        "attacked_input_url": f"/data/demo/fake-sessions/{attacked_image}",
        "clean_prediction_url": f"/data/demo/fake-sessions/assets/{prefix}-clean-prediction.png",
        "attacked_prediction_url": f"/data/demo/fake-sessions/{attacked_prediction}",
        "segmentation_clean_url": f"/data/demo/fake-sessions/assets/{prefix}-clean-prediction.png",
        "segmentation_attacked_url": f"/data/demo/fake-sessions/{attacked_prediction}",
        "diff_url": f"/data/demo/fake-sessions/assets/{prefix}-diff.png",
        "perturbation_url": f"/data/demo/fake-sessions/assets/{prefix}-perturbation.png",
    }
    return {
        "sample_id": stem,
        "attack": attack,
        "severity": severity,
        "clean_prediction": {"prediction_type": "segmentation", "sample_id": stem, "latency_ms": 82.4, "instances": clean_instances},
        "attacked_prediction": {"prediction_type": "segmentation", "sample_id": stem, "latency_ms": 83.1, "instances": attacked_instances},
        "degradation_hint": 0.08 if defence else 0.47,
        "attack_version": "demo-fixture-1.0",
        "attack_params": {"severity": severity},
        "ground_truth": {"type": "masks", "image_width": WIDTH, "image_height": HEIGHT, "objects": [{"object_id": 1, "label": "Road", "polygon": [[0, 540], [960, 540], [960, 350], [0, 390]]}]},
        "clean_image_path": f"assets/{prefix}-clean.png",
        "attacked_image_path": attacked_image,
        "clean_prediction_path": f"assets/{prefix}-clean-prediction.png",
        "attacked_prediction_path": attacked_prediction,
        "artifacts": artifacts,
    }


def draw_detection_asset(path: Path, prefix: str, name: str) -> None:
    attacked = "attacked" in name and "defence" not in name
    defence = "defence" in name
    image = scene(prefix, attacked=attacked, defence=defence)
    draw = ImageDraw.Draw(image, "RGBA")
    if "prediction" in name:
        boxes = [(112, 180, 306, 420, (34, 197, 94, 230)), (432, 190, 542, 304, (59, 130, 246, 230))]
        if attacked and not defence:
            boxes = [(145, 205, 270, 370, (244, 63, 94, 230))]
        for x1, y1, x2, y2, color in boxes:
            draw.rectangle((x1, y1, x2, y2), outline=color, width=6)
            draw.rectangle((x1, y1 - 24, x1 + 90, y1), fill=color)
            draw.text((x1 + 6, y1 - 20), "Car", fill="white")
    if name.endswith("diff.png"):
        draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(220, 38, 127, 120))
        for x in range(0, WIDTH, 36):
            draw.line((x, 0, x + 240, HEIGHT), fill=(253, 224, 71, 180), width=5)
    if name.endswith("perturbation.png"):
        for x in range(0, WIDTH, 24):
            draw.line((x, 0, x, HEIGHT), fill=((59, 130, 246, 130) if x % 48 else (239, 68, 68, 160)), width=3)
    image.save(path, format="PNG", optimize=True)


def draw_segmentation_asset(path: Path, prefix: str, name: str) -> None:
    attacked = "attacked" in name and "defence" not in name
    defence = "defence" in name
    image = scene(prefix, attacked=attacked, defence=defence)
    draw = ImageDraw.Draw(image, "RGBA")
    if "prediction" in name:
        draw.polygon([(0, HEIGHT), (WIDTH, HEIGHT), (WIDTH, 350), (0, 390)], fill=(34, 197, 94, 120))
        draw.polygon([(390, 310), (530, 310), (560, 420), (360, 420)], fill=(168, 85, 247, 150))
        draw.rectangle((650, 110, 920, 300), fill=(6, 182, 212, 90))
    if name.endswith("diff.png"):
        draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(245, 158, 11, 110))
        for y in range(0, HEIGHT, 28):
            draw.line((0, y, WIDTH, y + 100), fill=(239, 68, 68, 180), width=4)
    if name.endswith("perturbation.png"):
        for y in range(0, HEIGHT, 24):
            draw.line((0, y, WIDTH, y), fill=(59, 130, 246, 150), width=5)
    image.save(path, format="PNG", optimize=True)


def scene(prefix: str, attacked: bool, defence: bool) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (185, 220, 242) if not attacked else (135, 164, 185))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 350, WIDTH, HEIGHT), fill=(65, 82, 92, 255))
    draw.polygon([(0, 350), (960, 350), (960, 540), (0, 540)], fill=(76, 92, 102, 255))
    for x in range(-100, WIDTH, 140):
        draw.rectangle((x, 190, x + 110, 350), fill=(95, 112, 130, 255))
        draw.rectangle((x + 14, 210, x + 36, 240), fill=(245, 201, 83, 210))
        draw.rectangle((x + 58, 210, x + 80, 240), fill=(245, 201, 83, 210))
    draw.rectangle((112, 180, 306, 420), fill=(30, 41, 59, 230))
    draw.rectangle((136, 208, 286, 300), fill=(78, 115, 145, 255))
    draw.rectangle((432, 190, 542, 304), fill=(39, 49, 61, 230))
    draw.rectangle((452, 208, 522, 258), fill=(92, 128, 154, 255))
    if attacked and not defence:
        haze = Image.new("RGBA", (WIDTH, HEIGHT), (228, 63, 94, 62))
        image = Image.alpha_composite(image.convert("RGBA"), haze).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        for y in range(40, HEIGHT, 34):
            draw.line((0, y, WIDTH, y + 30), fill=(255, 255, 255, 45), width=9)
    if defence:
        draw.rectangle((8, 8, WIDTH - 8, HEIGHT - 8), outline=(16, 185, 129, 210), width=10)
    draw.rectangle((20, 20, 290, 58), fill=(15, 23, 42, 210))
    draw.text((34, 31), f"DEMO • {prefix.upper()} • {'DEFENDED' if defence else 'ATTACKED' if attacked else 'CLEAN'}", fill="white")
    return image.filter(ImageFilter.GaussianBlur(0.15))


if __name__ == "__main__":
    main()
