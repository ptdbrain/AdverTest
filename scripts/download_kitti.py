#!/usr/bin/env python3
"""Script to download and verify the official KITTI dataset via torchvision."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torchvision.datasets as tv_datasets


def download_kitti(
    root: str = "data",
    download_test: bool = False,
) -> None:
    """Download KITTI dataset using torchvision.datasets.Kitti.

    Args:
        root: Root directory to store KITTI dataset.
        download_test: Whether to also download the test split.
    """
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Target root directory: {root_path}")
    print("[*] Starting KITTI dataset download via torchvision (train split)...")
    print("    Note: Training images (~12 GB) and labels (~5 MB) will be downloaded and extracted.")

    start_time = time.time()
    train_dataset = tv_datasets.Kitti(
        root=str(root_path),
        train=True,
        download=True,
    )
    elapsed = time.time() - start_time
    print(f"[✓] KITTI train split ready! Total samples: {len(train_dataset)} (took {elapsed:.1f}s)")

    if download_test:
        print("[*] Downloading test split...")
        test_dataset = tv_datasets.Kitti(
            root=str(root_path),
            train=False,
            download=True,
        )
        print(f"[✓] KITTI test split ready! Total samples: {len(test_dataset)}")

    raw_dir = root_path / "Kitti" / "raw"
    print("\nDataset structure verification:")
    if raw_dir.exists():
        for item in raw_dir.iterdir():
            if item.is_dir():
                sub_count = len(list(item.glob("*/*")))
                print(f"  - {item.name}/ ({sub_count} files)")
            else:
                print(f"  - {item.name} ({item.stat().st_size / (1024 * 1024):.1f} MB)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download KITTI dataset using torchvision.")
    parser.add_argument(
        "--root",
        type=str,
        default="data",
        help="Root directory for dataset storage (default: data).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Also download test split.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    try:
        download_kitti(root=args.root, download_test=args.test)
    except Exception as exc:
        print(f"[!] Error during KITTI download: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
