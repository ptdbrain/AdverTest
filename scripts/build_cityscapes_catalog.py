#!/usr/bin/env python3
"""Create a deterministic, anonymised Cityscapes evaluation bundle.

The command copies a selected subset of an existing, already-anonymised
Cityscapes export.  It never reads from, writes to, or republishes the raw
source directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _records(manifest: Path, split: str) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if str(record.get("sample_id", "")).startswith(f"{split}/"):
            selected.append(record)
    return selected


def _copy_relative(source_root: Path, output_root: Path, relative: str) -> None:
    source = source_root / relative
    if not source.is_file():
        raise FileNotFoundError(f"manifest refers to missing file: {source}")
    destination = output_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build(source: Path, output: Path, *, split: str, count: int) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    descriptor = source / "dataset.json"
    manifest = source / "manifest.jsonl"
    if not descriptor.is_file() or not manifest.is_file():
        raise FileNotFoundError("source must contain dataset.json and manifest.jsonl")

    source_descriptor = json.loads(descriptor.read_text(encoding="utf-8"))
    if not source_descriptor.get("anonymized") or source_descriptor.get("status") != "complete":
        raise ValueError("source export is not a completed anonymised dataset")

    records = _records(manifest, split)
    if len(records) < count:
        raise ValueError(f"source has only {len(records)} records for split {split!r}; need {count}")
    records = records[:count]
    output.mkdir(parents=True)
    for record in records:
        _copy_relative(source, output, str(record["output_path"]))
        for annotation in record.get("annotations", []):
            _copy_relative(source, output, str(annotation["output_path"]))

    descriptor_out = {
        "name": "cityscapes-200",
        "display_name": "Cityscapes anonymized validation set (200 samples)",
        "task_id": "segmentation",
        "format": "cityscapes",
        "split": split,
        "sample_count": count,
        "anonymized": True,
        "status": "complete",
        "source_manifest_hash": source_descriptor.get("manifest_hash"),
        "source_fingerprint": source_descriptor.get("source_fingerprint"),
        "detectors": source_descriptor.get("detectors"),
        "review_status": source_descriptor.get("review_status"),
    }
    (output / "dataset.json").write_text(json.dumps(descriptor_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve(), split=args.split, count=args.count)


if __name__ == "__main__":
    main()
