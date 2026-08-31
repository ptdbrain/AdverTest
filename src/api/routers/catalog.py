from fastapi import APIRouter, Query

from src.adapters import load_adapters
from src.api.routes import _attack_catalog_availability
from src.api.schemas.catalog import AttackCatalogItem, DatasetCatalogItem, ModelCatalogItem
from src.attacks import load_attacks
from src.datasets import load_datasets

router = APIRouter(prefix="/catalog", tags=["Catalog"])

# Counts are part of the reviewed, versioned catalog bundles in GCS.  Keep
# them in the control-plane response so the UI never has to label a dataset as
# a vague "demo bundle" or probe object storage from the browser.
_CATALOG_SAMPLE_COUNTS: dict[str, int] = {
    "cityscapes_segmentation": 100,
    "kitti": 100,
    "nuscenes": 100,
}


def _demo_dataset_params(name: str) -> dict[str, object]:
    """Runtime paths for the reviewed Drive-export bundles on the GPU worker."""
    root = "/app/data/catalog"
    params: dict[str, dict[str, object]] = {
        "cityscapes_segmentation": {
            "root": f"{root}/cityscapes-instance-100", "split": "val", "anonymization_manifest": "manifest.jsonl",
        },
        "kitti": {
            "root": f"{root}/kitti2d-100", "split": "val", "manifest_path": "manifest.jsonl",
        },
        "nuscenes": {
            "dataroot": f"{root}/nuscenes-mini-100", "version": "v1.0-mini", "split": "mini_val",
            "anonymization_manifest": f"{root}/nuscenes-mini-100/manifest.jsonl",
        },
    }
    return params.get(name, {})

@router.get("/attacks", response_model=list[AttackCatalogItem])
async def list_attacks(
    group: str | None = Query(default=None, min_length=1, max_length=1),
    cost_class: str | None = None,
    modality: str | None = None,
    task_id: str | None = None,
    threat_model: str | None = None,
    attack_type: str | None = None,
    scenario_kind: str | None = None,
    model_family_id: str | None = None,
    checkpoint_id: str | None = None,
    dataset: str | None = None,
) -> list[AttackCatalogItem]:
    items = [attack.describe() for attack in load_attacks().values()]
    for field, wanted in (("group", group), ("cost_class", cost_class), ("modality", modality)):
        if wanted is not None:
            items = [item for item in items if item.get(field) == wanted]
    if task_id is not None:
        items = [item for item in items if task_id in item.get("task_ids", [])]
    for field, wanted in (("threat_model", threat_model), ("attack_type", attack_type), ("scenario_kind", scenario_kind)):
        if wanted is not None:
            items = [item for item in items if item.get(field) == wanted]

    availability = _attack_catalog_availability(
        task_id=task_id,
        model_family_id=model_family_id,
        checkpoint_id=checkpoint_id,
        dataset_name=dataset,
    )
    if availability is not None:
        unavailable_reason, exclusions = availability
        for item in items:
            reasons = (unavailable_reason,) if unavailable_reason else exclusions.get(item["name"], ())
            item["available"] = not reasons
            item["reason"] = ", ".join(reasons) if reasons else None

    return [AttackCatalogItem(**item) for item in items]

@router.get("/models", response_model=list[ModelCatalogItem])
async def list_models() -> list[ModelCatalogItem]:
    return [ModelCatalogItem(**adapter.describe()) for adapter in load_adapters().values()]

@router.get("/datasets", response_model=list[DatasetCatalogItem])
async def list_datasets(task_id: str | None = None) -> list[DatasetCatalogItem]:
    items = []
    for dataset in load_datasets().values():
        # The public product catalog is deliberately limited to the three
        # reviewed Drive exports.  Other registered loaders are implementation
        # details or developer fixtures and must not be selectable from UI.
        if dataset.name not in _CATALOG_SAMPLE_COUNTS:
            continue
        item = dataset.describe()
        params = _demo_dataset_params(dataset.name)
        item["dataset_params"] = params
        item["demo"] = False
        item["sample_count"] = _CATALOG_SAMPLE_COUNTS.get(dataset.name)
        if params:
            item["anonymized"] = True
        items.append(DatasetCatalogItem(**item))
    if task_id:
        items = [item for item in items if item.task_id == task_id]
    return items
