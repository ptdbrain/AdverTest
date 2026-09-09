"""Build the self-contained Kaggle dataset archive for Person C's SAM2 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "kaggle_upload"
CODE_PATHS = ("src", "configs", "scripts", "pyproject.toml", "uv.lock", "README.md")
DATA_PATHS = ("data/cityscapes", "data/bdd100k")
MODEL_PATHS = (
    "checkpoints/anonymization/yolo11n-face.onnx",
    "checkpoints/anonymization/yolov8n-license-plate.onnx",
    "checkpoints/sam2/sam2.1_hiera_small.pt",
)


def iter_files(relative_path: str) -> list[Path]:
    source = ROOT / relative_path
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"required Kaggle asset is missing: {source}")
    return [path for path in sorted(source.rglob("*")) if path.is_file()]


def archive_name(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if relative.parts[:2] == ("checkpoints", "anonymization"):
        return f"{ARCHIVE_ROOT}/models/anonymization/{path.name}"
    if relative.parts[:2] == ("checkpoints", "sam2"):
        return f"{ARCHIVE_ROOT}/models/sam2/{path.name}"
    if relative.parts[0] in {"src", "configs", "scripts"} or relative.name in {
        "pyproject.toml",
        "uv.lock",
        "README.md",
    }:
        return f"{ARCHIVE_ROOT}/code/{relative.as_posix()}"
    return f"{ARCHIVE_ROOT}/{relative.as_posix()}"


def compression_for(path: Path) -> int:
    return ZIP_STORED if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".onnx", ".pt", ".zip"} else ZIP_DEFLATED


def build_archive(output: Path) -> None:
    files = [path for relative_path in (*CODE_PATHS, *DATA_PATHS, *MODEL_PATHS) for path in iter_files(relative_path)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", allowZip64=True) as archive:
        for index, path in enumerate(files, start=1):
            archive.write(path, archive_name(path), compress_type=compression_for(path))
            if index % 1000 == 0 or index == len(files):
                print(f"packed {index}/{len(files)} files", flush=True)
    print(f"wrote {output} ({output.stat().st_size / 2**30:.2f} GiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="kaggle/advertest-person-c-sam2.zip")
    args = parser.parse_args()
    build_archive(Path(args.output).expanduser().resolve())


if __name__ == "__main__":
    main()
