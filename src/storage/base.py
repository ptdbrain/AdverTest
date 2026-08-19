"""Storage interface implemented by local and S3-compatible backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StoredObject:
    key: str
    size_bytes: int
    mime_type: str | None
    sha256: str | None = None


class ArtifactStorage(Protocol):
    def put_bytes(self, key: str, content: bytes, *, mime_type: str, sha256: str | None = None) -> StoredObject: ...
    def get_bytes(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    def head(self, key: str) -> StoredObject: ...
    def signed_download_url(self, key: str, expires_seconds: int) -> str: ...
    def signed_upload_url(self, key: str, *, mime_type: str, expires_seconds: int, sha256: str | None) -> str: ...
