import importlib.util
from pathlib import Path


def _load_scanner():
    path = Path(__file__).parents[1] / "scripts" / "log_codex_desktop.py"
    spec = importlib.util.spec_from_file_location("log_codex_desktop", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _session_records(*, cwd: str, session_id: str, message: str) -> list[dict]:
    return [
        {
            "timestamp": "2026-08-10T14:00:00.000Z",
            "type": "session_meta",
            "payload": {"session_id": session_id, "cwd": cwd, "originator": "Codex Desktop"},
        },
        {
            "timestamp": "2026-08-10T14:01:00.000Z",
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "client_id": "client-1",
                "message": message,
            },
        },
    ]


def test_iter_user_messages_filters_to_repo_and_extracts_desktop_prompt(monkeypatch):
    scanner = _load_scanner()
    session = Path("rollout-one.jsonl")
    monkeypatch.setattr(
        scanner,
        "_iter_records",
        lambda _path: iter(
            _session_records(
                cwd=r"D:\Project\AIthucchien\P-195",
                session_id="session-1",
                message="auto logging works",
            )
        ),
    )

    messages = list(
        scanner.iter_user_messages(
            [session],
            cutoff=None,
            repo_root_n=scanner._normalize(r"D:\Project\AIthucchien\P-195"),
        )
    )

    assert len(messages) == 1
    assert messages[0]["text"] == "auto logging works"
    assert messages[0]["session_id"] == "session-1"
    assert messages[0]["client_id"] == "client-1"


def test_build_entry_id_is_stable_for_deduplication():
    scanner = _load_scanner()
    message = {
        "session_id": "session-1",
        "client_id": "client-1",
        "timestamp": "2026-08-10T14:01:00.000Z",
        "text": "prompt",
    }

    first = scanner.build_entry(message, "advertest", "main", "abc123", "student@example.com")
    second = scanner.build_entry(message, "advertest", "main", "abc123", "student@example.com")

    assert first == second
    assert first["entry_id"].startswith("codex-desktop-session-1-")
    assert first["tool"] == "codex-desktop"
    assert first["event"] == "UserPrompt"
