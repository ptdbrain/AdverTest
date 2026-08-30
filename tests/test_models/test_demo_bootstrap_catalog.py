from pathlib import Path

from src.demo_bootstrap import ensure_anonymized_catalog_bundle, ensure_demo_checkpoint, ensure_demo_kitti
from src.models.versions import scan_base_checkpoints
from src.storage.local import LocalArtifactStorage


def test_scan_base_checkpoints_accepts_yolo11n(tmp_path: Path) -> None:
    checkpoint = tmp_path / "surrogates" / "yolo11n.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"trusted-demo-weight")

    versions = scan_base_checkpoints(tmp_path)

    assert [(item.id, item.model_name, item.model_family_id) for item in versions] == [
        ("yolo11n-base", "yolo11n", "yolo11")
    ]


def test_bootstrap_demo_checkpoint_fetches_fixed_storage_object(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(str(tmp_path / "objects"))
    storage.put_bytes("catalog/models/yolo11n/v1/yolo11n.pt", b"trusted-weight", mime_type="application/octet-stream")

    checkpoint = ensure_demo_checkpoint(
        enabled=True,
        checkpoint_root=str(tmp_path / "data" / "checkpoints"),
        model_id="yolo11n",
        storage=storage,
        storage_key="catalog/models/yolo11n/v1/yolo11n.pt",
    )

    assert checkpoint is not None
    assert checkpoint.read_bytes() == b"trusted-weight"


def test_bootstrap_demo_kitti_requires_and_materializes_anonymized_export(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(str(tmp_path / "objects"))
    prefix = "catalog/datasets/kitti/v1/anonymized/kitti-de/"
    storage.put_bytes(f"{prefix}dataset.json", b'{"anonymized": true}', mime_type="application/json")
    storage.put_bytes(f"{prefix}manifest.jsonl", b'{"sample_id": "000000"}\n', mime_type="application/json")
    storage.put_bytes(f"{prefix}image_2/000000.png", b"demo-image", mime_type="image/png")
    storage.put_bytes(f"{prefix}label_2/000000.txt", b"", mime_type="text/plain")

    root = ensure_demo_kitti(enabled=True, storage=storage, storage_prefix=prefix, data_root=str(tmp_path / "data"))

    assert root == tmp_path / "data" / "anonymized" / "kitti-de"
    assert (root / "manifest.jsonl").is_file()
    assert (root / "image_2" / "000000.png").read_bytes() == b"demo-image"


def test_bootstrap_catalog_bundle_requires_completed_anonymization(tmp_path: Path) -> None:
    storage = LocalArtifactStorage(str(tmp_path / "objects"))
    prefix = "catalog/datasets/cityscapes-200/v1/"
    storage.put_bytes(
        f"{prefix}dataset.json", b'{"anonymized": true, "status": "complete"}', mime_type="application/json"
    )
    storage.put_bytes(f"{prefix}manifest.jsonl", b'{"sample_id": "val/aachen/frame"}\n', mime_type="application/json")
    storage.put_bytes(f"{prefix}leftImg8bit/val/aachen/frame_leftImg8bit.png", b"image", mime_type="image/png")

    root = ensure_anonymized_catalog_bundle(
        enabled=True, storage=storage, storage_prefix=prefix, data_root=str(tmp_path / "data"), bundle_name="cityscapes-200"
    )

    assert root == tmp_path / "data" / "catalog" / "cityscapes-200"
    assert (root / "manifest.jsonl").is_file()
