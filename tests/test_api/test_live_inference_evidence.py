from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


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

    image_path = tmp_path / "api-state" / "data" / "anonymized" / "kitti-de" / "image_2" / "000000.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 12, 3), 128, dtype=np.uint8)).save(image_path)
    monkeypatch.setattr(live_inference, "YOLO", _FakeYOLO)
    monkeypatch.setattr(live_inference, "imagecorruptions_corrupt", lambda image, **_kwargs: image)

    project_id = client.headers["X-Project-Id"]
    response = await client.post(
        f"/api/v1/runs/live-inference?project_id={project_id}",
        json={"sample_id": "000000", "attack_type": "depth_fog", "severity": 3, "run_id": "visual-run-1"},
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
