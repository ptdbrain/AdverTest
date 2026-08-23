from pathlib import Path

from src.demo_bootstrap import ensure_demo_catalog


class _Storage:
    def __init__(self, root: Path) -> None:
        self.root = root

    def list_keys(self, prefix: str) -> list[str]:
        return [prefix + path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()]

    def get_bytes(self, key: str) -> bytes:
        return (self.root / key.rsplit("/", 1)[-1]).read_bytes()


def test_ensure_demo_catalog_materializes_ready_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / ".ready").write_text("ok", encoding="utf-8")
    (source / "sample.txt").write_text("data", encoding="utf-8")
    target = ensure_demo_catalog(
        enabled=True, storage=_Storage(source), storage_prefix="catalog", data_root=str(tmp_path / "data")
    )
    assert target == (tmp_path / "data" / "demo-catalog")
    assert (target / "sample.txt").read_text(encoding="utf-8") == "data"
