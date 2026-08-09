"""Background orchestration for generated-dataset jobs."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.adapters import get_adapter
from src.api.jobs import SqliteRunStore
from src.api.schemas import GeneratedDatasetCreateIn
from src.api.workflow_store import WorkflowJobStore
from src.attacks import ATTACK_CATALOG, load_attacks
from src.attacks.recipes import AttackRecipe, AttackRecipeStep, RecipeBuilder
from src.core.objectives import RequiredAnnotation, SurrogateCapability
from src.datasets.contracts import DatasetVersion
from src.pipeline.generator import (
    AttackDatasetGenerator,
    RecipeGenerationConfig,
    inspect_generated_dataset,
)


class GeneratedDatasetService:
    """Queues generation; FastAPI routes only translate HTTP to this service."""

    def __init__(
        self,
        workflow_store: WorkflowJobStore,
        record_store: SqliteRunStore,
        artifact_root: str | Path,
        *,
        max_workers: int = 1,
        generator: AttackDatasetGenerator | None = None,
    ) -> None:
        self.workflow_store = workflow_store
        self.record_store = record_store
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.generator = generator or AttackDatasetGenerator()
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="advertest-generated")

    def enqueue(self, request: GeneratedDatasetCreateIn) -> str:
        job_id = self.workflow_store.create_job("generated_dataset", request.model_dump(mode="json"))
        self.pool.submit(self._execute, job_id, request)
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self.workflow_store.get_job(job_id)

    def manifest(self, job_id: str) -> dict[str, Any] | None:
        return self._result_part(job_id, "manifest")

    def variants(self, job_id: str) -> dict[str, Any] | None:
        return self._result_part(job_id, "variants")

    def validation(self, job_id: str) -> dict[str, Any] | None:
        return self._result_part(job_id, "validation")

    def _result_part(self, job_id: str, key: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None or job["result"] is None:
            return None
        return {"id": job_id, key: job["result"].get(key)}

    def _execute(self, job_id: str, request: GeneratedDatasetCreateIn) -> None:
        try:
            self._raise_if_cancelled(job_id)
            self.workflow_store.append_event(job_id, "VALIDATING", {"progress_ratio": 0.05})
            version, source = self._resolve_dataset(request.dataset_version_id)
            recipe = self._resolve_recipe(request.recipe_id)
            self._validate_recipe(recipe, version, request)
            self._raise_if_cancelled(job_id)
            job_root = self._job_root(job_id)
            self.workflow_store.checkpoint(job_id, {"artifact_root": self._relative(job_root)})
            self.workflow_store.append_event(job_id, "GENERATING", {"progress_ratio": 0.15})
            config = RecipeGenerationConfig(
                logical_source_id=version.logical_source_id,
                recipe=recipe,
                seed=request.seed,
                surrogate=request.surrogate,
                output_dir=str(job_root),
                intended_use=request.intended_use,
                preview=request.preview,
                limit=request.limit,
                **source,
            )
            # AttackDatasetGenerator owns per-variant composition. The worker boundary
            # keeps every potentially expensive generation call off the request thread.
            self._raise_if_cancelled(job_id)
            report = self.generator.generate(config)
            self._raise_if_cancelled(job_id)
            descriptor = _read_json(report.root / "dataset.json")
            manifest = _read_manifest(report.root / "manifest.jsonl")
            validation = inspect_generated_dataset(report.root)
            if not validation.get("valid"):
                raise ValueError("generated dataset validation failed")
            result = {
                "descriptor": descriptor,
                "manifest": manifest,
                "variants": manifest,
                "validation": validation,
                "lineage": {"report_hash": descriptor.get("lineage_report_hash")},
                "artifact_root": self._relative(report.root),
            }
            self.workflow_store.complete_job(job_id, result)
        except _CancelledError as exc:
            self.workflow_store.fail_job(job_id, str(exc), cancelled=True)
        except Exception as exc:
            self.workflow_store.fail_job(job_id, f"{type(exc).__name__}: {exc}")

    def _resolve_dataset(self, dataset_version_id: str) -> tuple[DatasetVersion, dict[str, Any]]:
        record = self.record_store.get_record("dataset_version", dataset_version_id)
        if record is None:
            raise ValueError(f"unknown dataset version {dataset_version_id!r}")
        version_payload = {key: value for key, value in record.items() if key not in {"id", "record_type", "created_at", "updated_at", "generation_source"}}
        version = DatasetVersion.model_validate(version_payload)
        if not version.records or not all(record.anonymized for record in version.records):
            raise ValueError("source dataset version is not ready for generation")
        source = record.get("generation_source")
        if not isinstance(source, dict):
            raise ValueError("dataset version has no durable generation source")
        if ("dataset_name" in source) == ("input_dir" in source):
            raise ValueError("dataset generation source must identify exactly one source")
        if "input_dir" in source:
            root = Path(str(source["input_dir"])).expanduser().resolve()
            if not root.is_dir():
                raise ValueError("persisted dataset source directory is unavailable")
            source = {**source, "input_dir": str(root)}
        return version, source

    def _resolve_recipe(self, recipe_id: str) -> AttackRecipe:
        record = self.record_store.get_record("attack_recipe", recipe_id)
        if record is None:
            raise ValueError(f"unknown attack recipe {recipe_id!r}")
        if isinstance(record.get("recipe"), dict):
            return AttackRecipe.model_validate(record["recipe"])
        return AttackRecipe(
            name=str(record["name"]),
            steps=tuple(AttackRecipeStep.model_validate(step) for step in record.get("steps", [])),
        )

    def _validate_recipe(
        self, recipe: AttackRecipe, version: DatasetVersion, request: GeneratedDatasetCreateIn
    ) -> None:
        load_attacks()
        capabilities: frozenset[SurrogateCapability] = frozenset()
        if request.surrogate is not None:
            adapter = get_adapter(request.surrogate.name, **request.surrogate.params)
            capabilities = frozenset(adapter.metadata().capabilities)
        annotations: set[RequiredAnnotation] = set()
        for record in version.records:
            if record.annotation_type in {"boxes", "boxes_and_mask"}:
                annotations.add("boxes")
            if record.annotation_type in {"mask", "boxes_and_mask"}:
                annotations.add("mask")
            if record.annotation_type == "boxes3d":
                annotations.add("boxes3d")
        validation = RecipeBuilder().validate(
            recipe,
            ATTACK_CATALOG,
            task="detection2d",
            model_capabilities=capabilities,
            annotation_types=frozenset(annotations),
            modality="image",
            online=False,
            requested_variants=len(version.records),
            available_artifacts=frozenset({"patch_artifact"}),
        )
        if not validation.valid:
            raise ValueError(f"incompatible attack recipe: {list(validation.errors)}")

    def _raise_if_cancelled(self, job_id: str) -> None:
        if self.workflow_store.cancel_requested(job_id):
            raise _CancelledError("generated dataset job cancelled before generation cell")

    def _job_root(self, job_id: str) -> Path:
        candidate = (self.artifact_root / job_id).resolve()
        if self.artifact_root != candidate and self.artifact_root not in candidate.parents:
            raise ValueError("generated artifact path escapes configured artifact root")
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _relative(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.artifact_root).as_posix()
        except ValueError as exc:
            raise ValueError("generated artifact path escapes configured artifact root") from exc


class _CancelledError(Exception):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
