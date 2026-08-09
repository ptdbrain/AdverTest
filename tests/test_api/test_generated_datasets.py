from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio
from PIL import Image

from src.api.generated_dataset_service import GeneratedDatasetService
from src.api.jobs import SqliteRunStore
from src.api.schemas import GeneratedDatasetCreateIn
from src.api.workflow_store import WorkflowJobStore


@pytest_asyncio.fixture
async def generated_job_fixture(client, tmp_path):
    root = tmp_path / "source"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (12, 12), color="white").save(root / "images" / "frame.png")
    (root / "labels" / "frame.json").write_text(
        '[{"x1": 1, "y1": 1, "x2": 10, "y2": 10, "label": "Car"}]',
        encoding="utf-8",
    )
    (root / "dataset.json").write_text('{"anonymized": true, "split": "test"}', encoding="utf-8")
    suffix = uuid.uuid4().hex[:12]
    imported = await client.post(
        "/api/v1/datasets/import",
        json={
            "root": str(root),
            "name": f"generated-source-{suffix}",
            "logical_source_id": f"generated-source-{suffix}",
            "input_format": "advertest",
        },
    )
    assert imported.status_code == 201, imported.text
    recipe = await client.post(
        "/api/v1/attack-recipes",
        json={
            "id": f"recipe-noise-{suffix}",
            "name": "Single Gaussian noise",
            "seed": 17,
            "steps": [
                {
                    "position": 0,
                    "attack_name": "gaussian_noise",
                    "implementation_version": "1.0.0",
                    "severity": 1,
                    "parameters": {},
                    "seed": 17,
                    "expected_cost": 1.0,
                }
            ],
        },
    )
    assert recipe.status_code == 201, recipe.text

    class Fixture:
        request = {
            "dataset_version_id": imported.json()["version_id"],
            "recipe_id": recipe.json()["id"],
            "seed": 17,
            "preview": False,
            "intended_use": "training",
        }

        async def wait_terminal(self, job_id: str) -> dict:
            for _ in range(100):
                item = (await client.get(f"/api/v1/generated-datasets/{job_id}")).json()
                if item["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                    return item
                await asyncio.sleep(0.03)
            raise AssertionError(f"generated dataset job {job_id} did not finish")

    return Fixture()


@pytest.mark.asyncio
async def test_generated_dataset_creation_is_queued_and_exposes_manifest(client, generated_job_fixture) -> None:
    created = await client.post("/api/v1/generated-datasets", json=generated_job_fixture.request)

    assert created.status_code == 202
    item = await generated_job_fixture.wait_terminal(created.json()["id"])
    assert item["status"] == "COMPLETED", item.get("error")
    assert (await client.get(f"/api/v1/generated-datasets/{item['id']}/manifest")).status_code == 200
    assert (await client.get(f"/api/v1/generated-datasets/{item['id']}/variants")).status_code == 200
    assert (await client.get(f"/api/v1/generated-datasets/{item['id']}/events")).status_code == 200
    assert (await client.post(f"/api/v1/generated-datasets/{item['id']}/validate")).status_code == 200


@pytest.mark.asyncio
async def test_generated_dataset_request_rejects_unknown_public_fields(client) -> None:
    response = await client.post(
        "/api/v1/generated-datasets",
        json={"dataset_version_id": "missing", "recipe_id": "missing", "unexpected": True},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_generated_dataset_cancel_endpoint_marks_a_queued_job(client) -> None:
    from src.api import routes

    job_id = routes._workflow_store.create_job("generated_dataset", {"seed": 17})

    response = await client.post(f"/api/v1/generated-datasets/{job_id}/cancel")

    assert response.status_code == 200
    assert response.json()["id"] == job_id
    assert response.json()["cancel_requested"] is True
    events = await client.get(f"/api/v1/generated-datasets/{job_id}/events")
    assert events.json()["events"][-1]["state"] == "CANCEL_REQUESTED"


def test_generated_dataset_service_recovers_persisted_jobs_after_restart(tmp_path) -> None:
    workflow_store = WorkflowJobStore(f"sqlite:///{tmp_path / 'jobs.db'}")
    record_store = SqliteRunStore(f"sqlite:///{tmp_path / 'records.db'}")
    request = GeneratedDatasetCreateIn(dataset_version_id="dataset-a", recipe_id="recipe-a")
    job_id = workflow_store.create_job("generated_dataset", request.model_dump(mode="json"))
    workflow_store.append_event(job_id, "GENERATING", {"progress_ratio": 0.5})
    restarted = GeneratedDatasetService(
        workflow_store,
        record_store,
        tmp_path / "artifacts",
    )
    submitted: list[tuple[object, tuple[object, ...]]] = []

    class RecordingPool:
        def submit(self, fn, *args):
            submitted.append((fn, args))

    restarted.pool = RecordingPool()

    assert restarted.recover() == [job_id]
    assert submitted == [(restarted._execute, (job_id, request))]
    assert workflow_store.events(job_id)[-1]["payload"] == {"recovered": True}
