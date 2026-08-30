from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.core.platform_contracts import ArtifactKind, ArtifactState


class _FakeBox:
    def __init__(self) -> None:
        self.conf = np.array([0.9], dtype=np.float32)
        self.cls = np.array([0], dtype=np.float32)
        self.xyxy = np.array([[1.0, 1.0, 8.0, 8.0]], dtype=np.float32)


class _FakeYOLO:
    names = {0: "car"}

    def __init__(self, _checkpoint: str) -> None:
        pass

    def __call__(self, _image, **_kwargs):
        return [SimpleNamespace(boxes=[_FakeBox()])]


@pytest.mark.asyncio
async def test_visual_inference_returns_only_visual_measurements_and_scoped_artifacts(client, tmp_path, monkeypatch) -> None:
    import src.api.routers.live_inference as live_inference

    image = Image.fromarray(np.full((12, 12, 3), 128, dtype=np.uint8))
    image_bytes = __import__("io").BytesIO()
    image.save(image_bytes, format="PNG")
    monkeypatch.setattr(live_inference, "YOLO", _FakeYOLO)
    monkeypatch.setattr(live_inference, "imagecorruptions_corrupt", lambda image, **_kwargs: image)

    project_id = client.default_project_id
    import src.api.platform_dependencies as platform_dependencies

    source = platform_dependencies.get_platform_artifacts().create_internal(
        project_id=project_id,
        actor_id="system",
        kind=ArtifactKind.EVIDENCE,
        original_filename="input.png",
        mime_type="image/png",
        content=image_bytes.getvalue(),
        state=ArtifactState.READY,
    )
    response = await client.post(
        f"/api/v1/runs/live-inference?project_id={project_id}",
        json={
            "sample_id": "000000",
            "attack_type": "depth_fog",
            "severity": 3,
            "run_id": "visual-run-1",
            "source_artifact_id": source["id"],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["visual_only"] is True
    assert body["evidence_status"] == "NOT_ELIGIBLE"
    assert "map50" not in response.text
    assert "miou" not in response.text
    assert body["attacked_image_url"].startswith(
        f"/api/v1/projects/{project_id}/runs/visual-run-1/artifacts/"
    )
    assert not (tmp_path / "api-state" / "frontend" / "public").exists()
