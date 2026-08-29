"""Experiment Sessions & Multi-Run History Management Router.

Stores, retrieves, and compares experiment sessions and their sequential attack runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/sessions", tags=["Experiment Sessions"])

SESSIONS_STORAGE_FILE = Path("data/storage/sessions.json")
SESSIONS_STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)


class RunRecord(BaseModel):
    id: str
    name: str
    timestamp: str
    attack_type: str
    attack_name: str
    severity: int
    clean_map: float
    attacked_map: float
    map_drop_pct: float
    clean_conf: float
    attacked_conf: float
    psnr: str
    ssim: str
    inference_ms: float
    robustness_score: float
    clean_bbox_count: int
    attacked_bbox_count: int
    sample_id: str = "000000"

    # Segmentation-specific (optional, used when task is segmentation)
    clean_miou: float | None = None
    attacked_miou: float | None = None
    miou_drop_pct: float | None = None

    # Combined attack indicator
    is_combined: bool = False
    attack_components: list[str] = Field(default_factory=list)

    # Researcher note (auto-saved from the visual results page)
    note: str = ""

    # Reproducibility metadata
    seed: int | None = None
    run_config_hash: str = ""
    backend_run_id: str = ""


class SessionRecord(BaseModel):
    id: str
    name: str
    description: str = ""
    task_id: str = "detection2d"
    task_name: str = "Object Detection"
    model_id: str = "local_yolo11s_clean"
    model_name: str = "YOLO11s (Ultralytics Vision)"
    dataset_id: str = "kitti_anonymized_de"
    dataset_name: str = "KITTI De-anonymized Set"
    created_at: str
    updated_at: str
    runs: list[RunRecord] = Field(default_factory=list)

    # Session lifecycle
    status: str = "active"
    ended_at: str | None = None
    total_duration_seconds: float | None = None


def _load_sessions() -> dict[str, SessionRecord]:
    if not SESSIONS_STORAGE_FILE.exists():
        # Seed with initial default active session
        default_session = SessionRecord(
            id="EXP-2025-0512-001",
            name="Đánh giá Robustness YOLO11s trên KITTI (Phiên tiêu chuẩn)",
            description="Kiểm thử độ suy giảm của mô hình YOLO11s trước các điều kiện thời tiết khắc nghiệt và nhiễu đối kháng.",
            task_id="detection2d",
            task_name="Object Detection (2D)",
            model_id="local_yolo11s_clean",
            model_name="YOLO11s (Ultralytics Vision)",
            dataset_id="kitti_anonymized_de",
            dataset_name="KITTI De-anonymized Set",
            created_at="12/05/2025 10:24:31",
            updated_at="12/05/2025 10:24:31",
            runs=[
                RunRecord(
                    id="RUN-001",
                    name="Lần 1: Depth Fog (Sương mù Cấp 3)",
                    timestamp="12/05/2025 10:25:12",
                    attack_type="depth_fog",
                    attack_name="Depth Fog (Sương mù)",
                    severity=3,
                    clean_map=0.82,
                    attacked_map=0.45,
                    map_drop_pct=45.1,
                    clean_conf=0.92,
                    attacked_conf=0.68,
                    psnr="24.21 dB",
                    ssim="0.781",
                    inference_ms=56.9,
                    robustness_score=54.9,
                    clean_bbox_count=10,
                    attacked_bbox_count=5,
                    sample_id="000000",
                ),
                RunRecord(
                    id="RUN-002",
                    name="Lần 2: Depth Rain (Mưa giông Cấp 4)",
                    timestamp="12/05/2025 10:28:44",
                    attack_type="depth_rain",
                    attack_name="Depth Rain (Mưa giông)",
                    severity=4,
                    clean_map=0.82,
                    attacked_map=0.22,
                    map_drop_pct=73.2,
                    clean_conf=0.92,
                    attacked_conf=0.35,
                    psnr="22.15 dB",
                    ssim="0.710",
                    inference_ms=52.8,
                    robustness_score=26.8,
                    clean_bbox_count=10,
                    attacked_bbox_count=2,
                    sample_id="000000",
                ),
                RunRecord(
                    id="RUN-003",
                    name="Lần 3: PGD Linf (Nhiễu đối kháng 8/255)",
                    timestamp="12/05/2025 10:32:05",
                    attack_type="pgd",
                    attack_name="PGD (L∞ 8/255)",
                    severity=3,
                    clean_map=0.82,
                    attacked_map=0.11,
                    map_drop_pct=86.6,
                    clean_conf=0.92,
                    attacked_conf=0.16,
                    psnr="24.21 dB",
                    ssim="0.781",
                    inference_ms=63.0,
                    robustness_score=13.4,
                    clean_bbox_count=10,
                    attacked_bbox_count=1,
                    sample_id="000000",
                ),
            ],
        )
        data = {default_session.id: default_session.model_dump()}
        SESSIONS_STORAGE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return {default_session.id: default_session}

    try:
        raw = json.loads(SESSIONS_STORAGE_FILE.read_text(encoding="utf-8"))
        return {k: SessionRecord(**v) for k, v in raw.items()}
    except Exception:
        return {}


def _save_sessions(sessions: dict[str, SessionRecord]) -> None:
    data = {k: v.model_dump() for k, v in sessions.items()}
    SESSIONS_STORAGE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@router.get("", response_model=list[SessionRecord])
async def list_sessions() -> list[SessionRecord]:
    """Get all experiment sessions with full run histories."""
    sessions = _load_sessions()
    return list(sessions.values())


@router.post("", response_model=SessionRecord)
async def create_or_update_session(session: SessionRecord) -> SessionRecord:
    """Create a new experiment session or update existing."""
    sessions = _load_sessions()
    session.updated_at = time.strftime("%d/%m/%Y %H:%M:%S")
    sessions[session.id] = session
    _save_sessions(sessions)
    return session


@router.get("/{session_id}", response_model=SessionRecord)
async def get_session(session_id: str) -> SessionRecord:
    """Get a specific session by ID."""
    sessions = _load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return sessions[session_id]


@router.post("/{session_id}/runs", response_model=SessionRecord)
async def add_run_to_session(session_id: str, run: RunRecord) -> SessionRecord:
    """Append a new attack execution run record to the session."""
    sessions = _load_sessions()
    if session_id not in sessions:
        # Create session if not found
        sessions[session_id] = SessionRecord(
            id=session_id,
            name=f"Phiên thử nghiệm {session_id}",
            created_at=time.strftime("%d/%m/%Y %H:%M:%S"),
            updated_at=time.strftime("%d/%m/%Y %H:%M:%S"),
            runs=[],
        )

    sess = sessions[session_id]
    # Check if run with same ID exists, update or append
    existing_idx = next((i for i, r in enumerate(sess.runs) if r.id == run.id), None)
    if existing_idx is not None:
        sess.runs[existing_idx] = run
    else:
        sess.runs.append(run)

    sess.updated_at = time.strftime("%d/%m/%Y %H:%M:%S")
    _save_sessions(sessions)
    return sess


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete an entire experiment session."""
    sessions = _load_sessions()
    if session_id in sessions:
        del sessions[session_id]
        _save_sessions(sessions)
    return {"status": "deleted", "session_id": session_id}


@router.delete("/{session_id}/runs/{run_id}")
async def delete_run(session_id: str, run_id: str) -> SessionRecord:
    """Delete a specific run from an experiment session."""
    sessions = _load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    sess = sessions[session_id]
    sess.runs = [r for r in sess.runs if r.id != run_id]
    sess.updated_at = time.strftime("%d/%m/%Y %H:%M:%S")
    _save_sessions(sessions)
    return sess


# ── Researcher Notes ─────────────────────────────────────────────


class UpdateRunNoteIn(BaseModel):
    """Payload for updating a researcher note on a specific run."""

    note: str


@router.patch("/{session_id}/runs/{run_id}/note", response_model=SessionRecord)
async def update_run_note(session_id: str, run_id: str, body: UpdateRunNoteIn) -> SessionRecord:
    """Update or set researcher note for a specific run."""
    sessions = _load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    sess = sessions[session_id]
    run_found = next((r for r in sess.runs if r.id == run_id), None)
    if not run_found:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in session.")
    run_found.note = body.note
    sess.updated_at = time.strftime("%d/%m/%Y %H:%M:%S")
    _save_sessions(sessions)
    return sess


# ── Session Lifecycle ────────────────────────────────────────────


@router.post("/{session_id}/end", response_model=SessionRecord)
async def end_session(session_id: str) -> SessionRecord:
    """Mark session as completed and lock it from further modifications."""
    sessions = _load_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    sess = sessions[session_id]
    if sess.status == "completed":
        raise HTTPException(status_code=409, detail="Session has already been completed.")
    sess.status = "completed"
    sess.ended_at = time.strftime("%d/%m/%Y %H:%M:%S")
    sess.updated_at = time.strftime("%d/%m/%Y %H:%M:%S")
    _save_sessions(sessions)
    return sess
