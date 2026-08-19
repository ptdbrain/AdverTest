from pathlib import Path

from src.models.versions import scan_base_checkpoints


def test_scan_base_checkpoints_accepts_yolo11n(tmp_path: Path) -> None:
    checkpoint = tmp_path / "surrogates" / "yolo11n.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"trusted-demo-weight")

    versions = scan_base_checkpoints(tmp_path)

    assert [(item.id, item.model_name, item.model_family_id) for item in versions] == [
        ("yolo11n-base", "yolo11n", "yolo11")
    ]
