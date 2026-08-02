"""Benchmark YOLO11 on KITTI under the group C (occlusion & sensor fault) attacks.

    python scripts/benchmark_kitti_yolo11.py --limit 500
    python scripts/benchmark_kitti_yolo11.py --weights runs/detect/train/weights/best.pt

Produces ``eval/results/kitti_yolo11_groupC.{json,md}``: the per-cell AP and
degradation grid, the plan §3 aggregates (mPC, rPC, RR, RA(s), RobustScore),
bootstrap confidence intervals, the ΔAP caused by the placeholder anonymiser, and
the sanity-check table.

Three details make the numbers trustworthy rather than merely plausible:

* the cost estimate is printed **before** the run, never after (plan §5);
* the bootstrap replicates are computed from the prediction cache, so a 1000-fold
  resample costs no extra forward passes (plan §5 content-addressed cache);
* the AP lost to anonymisation is measured and reported separately, so blur is
  never mistaken for attack damage (plan §6).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

# Runnable as `python scripts/benchmark_kitti_yolo11.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters import get_adapter  # noqa: E402
from src.attacks import get_attack
from src.core.hashing import clean_key, variant_key
from src.core.types import Prediction, Sample
from src.datasets import get_dataset
from src.evaluation.detection_metrics import average_precision
from src.evaluation.report import RunReport
from src.evaluation.robustness_metrics import (
    ap_grid,
    bootstrap_ci,
    severities,
    summary,
)
from src.pipeline import MemoryCache, RunConfig, TestRunner

#: Group C is the subject; one group A corruption rides along as a control row.
DEFAULT_ATTACKS = (
    "random_erasing",
    "object_occlusion",
    "sensor_fault",
    "frame_freeze",
    "gaussian_noise",
)
DEFAULT_OUT_DIR = Path("eval/results")
STEM = "kitti_yolo11_groupC"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runner = TestRunner(cache=MemoryCache())
    config = _config(args)

    estimate = runner.estimate(config)
    print("cost estimate (before running, plan §5):")
    print(json.dumps(estimate.as_dict(), indent=2))
    if args.estimate_only:
        return 0

    started = perf_counter()
    report = runner.run(config)
    print(f"\nrun {report.run_id}: AP_clean={report.ap_clean:.4f} in {report.seconds:.1f}s")

    samples = get_dataset(config.dataset, **config.dataset_params).load(config.limit)
    intervals = _bootstrap(runner, config, report, samples, args.bootstrap, args.seed)
    anonymisation = _anonymisation_delta(args, config, report) if args.anonymization_delta else None

    payload: dict[str, Any] = {
        "report": report.as_dict(),
        "metrics": summary(report),
        "estimate": estimate.as_dict(),
        "bootstrap": {"replicates": args.bootstrap, "intervals": intervals},
        "anonymization": anonymisation,
        "config": config.model_dump(),
        "wall_seconds": round(perf_counter() - started, 2),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{STEM}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / f"{STEM}.md").write_text(_markdown(report, payload), encoding="utf-8")
    print(f"\nwrote {out_dir / STEM}.json and {out_dir / STEM}.md")
    _print_sanity(payload["metrics"])
    return 0


# ------------------------------------------------------------------ configuration


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="KITTI root (default: $ADVERTEST_KITTI_ROOT)")
    parser.add_argument(
        "--model",
        default="yolo11",
        help="adapter name; 'blob_detector' exercises the whole pipeline with no GPU",
    )
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu; default is auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--difficulty", default="moderate", choices=["all", "easy", "moderate", "hard"])
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--severities", default="1,2,3,4,5")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--bootstrap", type=int, default=1000, help="0 disables the CI")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--no-anonymization-delta",
        dest="anonymization_delta",
        action="store_false",
        help="skip the plan §6 ΔAP measurement (it needs one extra clean pass)",
    )
    return parser.parse_args(argv)


def _config(args: argparse.Namespace) -> RunConfig:
    dataset_params: dict[str, Any] = {"split": args.split, "difficulty": args.difficulty}
    if args.root:
        dataset_params["root"] = args.root
    adapter_params: dict[str, Any] = {}
    if args.model == "yolo11":
        adapter_params = {
            "weights": args.weights,
            "imgsz": args.imgsz,
            "batch_size": args.batch_size,
            "device": args.device,
        }
    return RunConfig(
        model=args.model,
        adapter_params=adapter_params,
        dataset="kitti",
        dataset_params=dataset_params,
        attacks=[name for name in args.attacks.split(",") if name],
        severities=[int(value) for value in args.severities.split(",") if value.strip()],
        limit=args.limit,
        seed=args.seed,
    )


# -------------------------------------------------------------------- bootstrap


def _bootstrap(
    runner: TestRunner,
    config: RunConfig,
    report: RunReport,
    samples: Sequence[Sample],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """95 % intervals for the clean AP and every cell, from cached predictions.

    Resampling frames and re-running the pipeline would repeat the attack
    generation for every replicate; instead the predictions are pulled back out
    of the content-addressed cache and the AP is recomputed in numpy, which is
    free. Duplicated ids in a resample are handled correctly by
    :func:`average_precision`, which looks predictions up per sample.
    """
    if replicates <= 0:
        return {}
    version = get_adapter(config.model, **config.adapter_params).metadata().version
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(samples), len(samples)) for _ in range(replicates)]

    intervals: dict[str, Any] = {}
    clean = _cached(runner, [clean_key(sample_id=s.sample_id, model_version=version) for s in samples])
    if clean is not None:
        intervals["clean"] = _interval(clean, samples, draws, config.iou_threshold)
    for cell in report.cells:
        attack = get_attack(cell.attack, **config.attack_params.get(cell.attack, {}))
        keys = [
            variant_key(
                sample_id=sample.sample_id,
                attack=attack.name,
                params=attack.param_dict(),
                severity=cell.severity,
                model_version=version,
            )
            for sample in samples
        ]
        predictions = _cached(runner, keys)
        if predictions is not None:
            key = f"{cell.attack}@{cell.severity}"
            intervals[key] = _interval(predictions, samples, draws, config.iou_threshold)
    return intervals


def _cached(runner: TestRunner, keys: Sequence[str]) -> list[Prediction] | None:
    """All predictions for these keys, or ``None`` if any is missing."""
    found = [runner.cache.get(key) for key in keys]
    return None if any(item is None for item in found) else found  # type: ignore[return-value]


def _interval(
    predictions: Sequence[Prediction],
    samples: Sequence[Sample],
    draws: Sequence[np.ndarray],
    iou_threshold: float,
) -> dict[str, float]:
    replicate_aps = [
        average_precision(
            [predictions[index] for index in draw],
            [samples[index] for index in draw],
            iou_threshold,
        )
        for draw in draws
    ]
    low, high = bootstrap_ci(replicate_aps)
    return {"ap_low": round(low, 4), "ap_high": round(high, 4), "width": round(high - low, 4)}


# ------------------------------------------------------------ anonymisation ΔAP


def _anonymisation_delta(
    args: argparse.Namespace,
    config: RunConfig,
    report: RunReport,
) -> dict[str, Any]:
    """AP lost to the placeholder anonymiser itself (plan §6).

    This calls the adapter directly instead of going through :class:`TestRunner`.
    That is deliberate: the runner's gate must keep refusing un-anonymised *test
    runs*, while this measurement needs one clean pass over the raw frames to
    tell blur damage apart from attack damage. Nothing here is persisted as a run.
    """
    raw_params = {**config.dataset_params, "anonymize": "off"}
    raw = get_dataset(config.dataset, **raw_params)
    samples = raw.load(config.limit)
    adapter = get_adapter(config.model, **config.adapter_params)
    ap_raw = average_precision(adapter.predict(samples), samples, config.iou_threshold)
    return {
        "ap_raw": round(ap_raw, 4),
        "ap_anonymized": round(report.ap_clean, 4),
        "delta_ap": round(ap_raw - report.ap_clean, 4),
        "note": "placeholder anonymiser (GT-box heuristic), not a privacy guarantee",
    }


# ----------------------------------------------------------------- presentation


def _markdown(report: RunReport, payload: dict[str, Any]) -> str:
    metrics: dict[str, Any] = payload["metrics"]
    levels = severities(report)
    lines = [
        f"# KITTI x YOLO11 — group C robustness ({report.run_id})",
        "",
        "> SIMULATION ONLY. Generated by `scripts/benchmark_kitti_yolo11.py`; the numbers",
        "> describe a simulated test run and are not a deployment decision (plan §7).",
        "",
        f"- model: `{report.model_version}`",
        f"- dataset: `{report.dataset}`, {report.n_samples} frames",
        f"- clean AP50: **{report.ap_clean:.4f}**",
        f"- wall time: {payload['wall_seconds']}s (run {report.seconds:.1f}s)",
        "",
        "## Degradation D(c, s) — % of clean AP lost",
        "",
        "| attack | " + " | ".join(f"s{level}" for level in levels) + " |",
        "|---" * (len(levels) + 1) + "|",
    ]
    heatmap = report.heatmap()
    for attack, row in heatmap.items():
        cells = " | ".join(f"{row.get(level, 0.0) * 100:.1f}" for level in levels)
        lines.append(f"| `{attack}` | {cells} |")

    grid = ap_grid(report)
    lines += [
        "",
        "## AP(c, s)",
        "",
        "| attack | " + " | ".join(f"s{level}" for level in levels) + " |",
        "|---" * (len(levels) + 1) + "|",
    ]
    for attack, row in grid.items():
        cells = " | ".join(f"{row.get(level, float('nan')):.4f}" for level in levels)
        lines.append(f"| `{attack}` | {cells} |")

    lines += [
        "",
        "## Aggregates (plan §3)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| mPC (3) | {metrics['mpc']:.4f} |",
        f"| rPC (4) | {metrics['rpc'] * 100:.1f}% |",
        f"| RobustScore, plan formula (13) | {metrics['robust_score_plan']:.1f} / 100 |",
        f"| RobustScore, normalised to measured categories | {metrics['robust_score_normalized']:.1f} / 100 |",
        f"| categories measured | {', '.join(metrics['covered_categories'])} |",
        "",
        "The plan's RobustScore weights four categories at 0.25 each, so a run that only",
        "measured some of them cannot reach 100 — the normalised row rescales to what was",
        "actually covered.",
        "",
        "### RA(s) — robustness accuracy per severity (7)",
        "",
        "| severity | " + " | ".join(str(level) for level in levels) + " |",
        "|---" * (len(levels) + 1) + "|",
        "| RA(s) | "
        + " | ".join(f"{metrics['robustness_accuracy'].get(level, 0.0):.3f}" for level in levels)
        + " |",
        "",
        "### RR(c) — resilience rate per attack (5)",
        "",
        "| attack | RR |",
        "|---|---|",
    ]
    for attack, value in metrics["resilience_rate"].items():
        lines.append(f"| `{attack}` | {value:.3f} |")

    if payload["bootstrap"]["intervals"]:
        lines += [
            "",
            f"## Bootstrap 95 % intervals ({payload['bootstrap']['replicates']} replicates)",
            "",
            "Plan §3: do not compare two models whose intervals overlap.",
            "",
            "| cell | AP low | AP high | width |",
            "|---|---|---|---|",
        ]
        for key, interval in payload["bootstrap"]["intervals"].items():
            lines.append(
                f"| `{key}` | {interval['ap_low']:.4f} | {interval['ap_high']:.4f} | {interval['width']:.4f} |"
            )

    if payload["anonymization"]:
        anon = payload["anonymization"]
        lines += [
            "",
            "## Anonymisation cost (plan §6)",
            "",
            f"- AP on raw frames: {anon['ap_raw']:.4f}",
            f"- AP after the placeholder anonymiser: {anon['ap_anonymized']:.4f}",
            f"- **ΔAP from anonymisation alone: {anon['delta_ap']:.4f}**",
            "",
            f"_{anon['note']}._ Attack degradation above is measured against the",
            "anonymised baseline, so this loss is not double-counted.",
        ]

    lines += ["", "## Sanity checks (plan §3)", "", "| attack | AP non-increasing in severity |", "|---|---|"]
    for attack, passed in metrics["severity_monotonicity"].items():
        lines.append(f"| `{attack}` | {'PASS' if passed else 'FAIL'} |")

    lines += ["", "## Worst cases", "", "| attack | severity | AP | D |", "|---|---|---|---|"]
    for case in report.worst_cases():
        lines.append(
            f"| `{case['attack']}` | {case['severity']} | {case['ap']:.4f} | {case['degradation'] * 100:.1f}% |"
        )
    if report.skipped:
        lines += ["", "## Skipped attacks", ""]
        lines += [f"- `{item.attack}`: {item.reason}" for item in report.skipped]
    return "\n".join(lines) + "\n"


def _print_sanity(metrics: dict[str, Any]) -> None:
    print("\nsanity check #2 (AP non-increasing in severity):")
    for attack, passed in metrics["severity_monotonicity"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {attack}")
    print(
        f"RobustScore: {metrics['robust_score_plan']:.1f}/100 plan formula, "
        f"{metrics['robust_score_normalized']:.1f}/100 normalised over "
        f"{', '.join(metrics['covered_categories'])}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
