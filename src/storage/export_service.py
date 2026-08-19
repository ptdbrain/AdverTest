"""Asynchronous attacked-dataset ZIP exports with immutable provenance."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from typing import Any

from src.core.platform_contracts import ArtifactKind, ArtifactState
from src.storage.service import ArtifactService


class AttackedDatasetExportService:
    """Builds the portable export format specified by the platform plan."""

    def __init__(self, artifacts: ArtifactService) -> None:
        self._artifacts = artifacts

    def run(self, *, project_id: str, actor_id: str, request: dict[str, Any]) -> dict[str, Any]:
        media_artifact_ids = tuple(str(item) for item in request.get("media_artifact_ids", ()))
        label_artifact_ids = tuple(str(item) for item in request.get("label_artifact_ids", ()))
        manifest = dict(request["manifest"])
        recipe = dict(request["recipe"])
        if len(media_artifact_ids) != len(label_artifact_ids):
            raise ValueError("EXPORT_MEDIA_LABEL_COUNT_MISMATCH")
        provenance = {
            "source_dataset_version_id": request["source_dataset_version_id"],
            "task": request["task"],
            "attack_method": request["attack_method"],
            "severity": request["severity"],
            "seed": request["seed"],
            "implementation_version": request["implementation_version"],
            "created_at": datetime.now(UTC).isoformat(),
        }
        archive, hashes = self._build_archive(
            project_id, media_artifact_ids, label_artifact_ids, manifest, recipe, provenance
        )
        artifact = self._artifacts.create_internal(
            project_id=project_id,
            actor_id=actor_id,
            kind=ArtifactKind.EXPORT,
            original_filename="attacked_dataset.zip",
            mime_type="application/zip",
            content=archive,
            state=ArtifactState.READY,
            metadata={"provenance": provenance, "sha256": hashlib.sha256(archive).hexdigest()},
        )
        return {"artifact_id": artifact["id"], "sha256": artifact["sha256"], "hashes": hashes}

    def _build_archive(
        self,
        project_id: str,
        media_artifact_ids: tuple[str, ...],
        label_artifact_ids: tuple[str, ...],
        manifest: dict[str, Any],
        recipe: dict[str, Any],
        provenance: dict[str, Any],
    ) -> tuple[bytes, dict[str, str]]:
        hashes: dict[str, str] = {}
        output = io.BytesIO()
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, artifact_id in enumerate(media_artifact_ids):
                content = self._artifacts.read_bytes(project_id, artifact_id)
                filename = f"media/{index:06d}_{artifact_id}"
                archive.writestr(filename, content)
                hashes[filename] = hashlib.sha256(content).hexdigest()
            for index, artifact_id in enumerate(label_artifact_ids):
                content = self._artifacts.read_bytes(project_id, artifact_id)
                filename = f"labels/{index:06d}_{artifact_id}"
                archive.writestr(filename, content)
                hashes[filename] = hashlib.sha256(content).hexdigest()
            archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, indent=2))
            archive.writestr("recipe.json", json.dumps(recipe, sort_keys=True, indent=2))
            archive.writestr("provenance.json", json.dumps(provenance, sort_keys=True, indent=2))
            archive.writestr("hashes.json", json.dumps(hashes, sort_keys=True, indent=2))
        return output.getvalue(), hashes
