"""Filesystem object-storage implementation for development and tests only."""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from src.storage.base import StoredObject


class LocalArtifactStorage:
    def __init__(self, root: str) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, content: bytes, *, mime_type: str, sha256: str | None = None) -> StoredObject:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        if sha256 and sha256 != digest:
            path.unlink(missing_ok=True)
            raise ValueError("ARTIFACT_HASH_MISMATCH")
        return StoredObject(key=key, size_bytes=len(content), mime_type=mime_type, sha256=digest)

    def get_bytes(self, key: str) -> bytes:
        return self._path_for(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path_for(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._path_for(key).is_file()

    def head(self, key: str) -> StoredObject:
        path = self._path_for(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return StoredObject(
            key=key,
            size_bytes=path.stat().st_size,
            mime_type=mimetypes.guess_type(path.name)[0],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def signed_download_url(self, key: str, expires_seconds: int) -> str:
        del expires_seconds
        return f"local://{key}"

    def signed_upload_url(self, key: str, *, mime_type: str, expires_seconds: int, sha256: str | None) -> str:
        del mime_type, expires_seconds, sha256
        return f"local://{key}"

    def _path_for(self, key: str) -> Path:
        if not key or key.startswith("/"):
            raise ValueError("storage key must be a relative path")
        resolved = (self._root / key).resolve()
        if self._root not in resolved.parents and resolved != self._root:
            raise ValueError("storage key escapes local artifact root")
        return resolved
