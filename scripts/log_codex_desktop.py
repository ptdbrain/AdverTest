#!/usr/bin/env python3
"""Collect user prompts from Codex Desktop's local session transcripts.

Codex Desktop does not currently emit the repository hook events supported by
Codex CLI.  Desktop does persist JSONL transcripts under ``~/.codex/sessions``;
this scanner reads only ``event_msg`` records whose payload type is
``user_message`` and appends deduplicated entries to ``.ai-log/session.jsonl``.
It is intended to run from the repository pre-push hook.

Usage::

    python scripts/log_codex_desktop.py --auto
    python scripts/log_codex_desktop.py --auto --dry-run

Environment overrides:
    CODEX_SESSIONS_DIR  alternate sessions directory (useful for tests)
    AI_LOG_DIR          destination directory (default: .ai-log)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


VN_TZ = timezone(timedelta(hours=7))
DEFAULT_SESSIONS_DIR = Path.home() / ".codex" / "sessions"


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _normalize(path: str) -> str:
    return path.strip().lower().replace("/", "\\").rstrip("\\") if path else ""


def _matches_repo(cwds: Iterable[str], repo_root_n: str) -> bool:
    if not repo_root_n:
        return True
    for cwd in cwds:
        normalized = _normalize(cwd)
        if (
            normalized == repo_root_n
            or normalized.startswith(repo_root_n + "\\")
            or repo_root_n.startswith(normalized + "\\")
        ):
            return True
    return False


def _parse_timestamp(value: str) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _iter_records(path: Path) -> Iterable[dict]:
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
    except OSError:
        return


def get_session_files(sessions_dir: Path) -> list[Path]:
    if not sessions_dir.exists():
        return []
    return sorted(
        (path for path in sessions_dir.rglob("rollout-*.jsonl") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def iter_user_messages(
    session_files: Iterable[Path],
    cutoff: datetime | None,
    repo_root_n: str,
    only_session: str | None = None,
) -> Iterable[dict]:
    """Yield recent Desktop user messages belonging to the current repo."""
    for path in session_files:
        session_id = ""
        originator = ""
        cwds: set[str] = set()
        messages: list[dict] = []

        for record in _iter_records(path):
            payload = record.get("payload") or {}
            record_type = record.get("type")
            if record_type == "session_meta":
                session_id = str(payload.get("session_id") or payload.get("id") or session_id)
                originator = str(payload.get("originator") or originator)
                if payload.get("cwd"):
                    cwds.add(str(payload["cwd"]))
            elif record_type == "turn_context":
                if payload.get("cwd"):
                    cwds.add(str(payload["cwd"]))
                session_id = str(payload.get("session_id") or session_id)
            elif record_type == "event_msg" and payload.get("type") == "user_message":
                messages.append(
                    {
                        "timestamp": record.get("timestamp") or payload.get("timestamp") or "",
                        "client_id": str(payload.get("client_id") or payload.get("id") or ""),
                        "text": str(payload.get("message") or "").strip(),
                    }
                )

        if originator and originator.lower() != "codex desktop":
            continue
        if only_session and only_session not in {session_id, path.stem}:
            continue
        if not _matches_repo(cwds, repo_root_n):
            continue

        session_id = session_id or path.stem.removeprefix("rollout-")
        for index, message in enumerate(messages):
            text = message["text"]
            if len(text) < 2:
                continue
            timestamp = _parse_timestamp(message["timestamp"])
            if cutoff and timestamp and timestamp < cutoff:
                continue
            client_id = message["client_id"] or str(index)
            yield {
                "session_id": session_id,
                "client_id": client_id,
                "timestamp": message["timestamp"],
                "text": text,
            }


def get_logged_entry_ids(log_file: Path) -> set[str]:
    logged: set[str] = set()
    if not log_file.exists():
        return logged
    try:
        with log_file.open(encoding="utf-8-sig") as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if entry.get("entry_id"):
                    logged.add(entry["entry_id"])
    except OSError:
        pass
    return logged


def build_entry(message: dict, repo: str, branch: str, commit: str, student: str) -> dict:
    timestamp = _parse_timestamp(message.get("timestamp", ""))
    ts = timestamp.astimezone(VN_TZ).isoformat() if timestamp else datetime.now(VN_TZ).isoformat()
    identity = f"{message['session_id']}\0{message['client_id']}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return {
        "ts": ts,
        "tool": "codex-desktop",
        "event": "UserPrompt",
        "entry_id": f"codex-desktop-{message['session_id']}-{digest}",
        "session_id": message["session_id"],
        "model": "codex-desktop",
        "repo": repo,
        "branch": branch,
        "commit": commit,
        "student": student,
        "prompt": message["text"][:1000],
        "response_summary": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", action="store_true", help="Scan recent Desktop sessions.")
    parser.add_argument("--hours", type=int, default=24, help="Recent window in hours (default: 24).")
    parser.add_argument("--all", action="store_true", help="Scan all sessions, ignoring the time window.")
    parser.add_argument("--session-id", help="Limit scanning to one session id.")
    parser.add_argument("--no-repo-filter", action="store_true", help="Include sessions from other repositories.")
    parser.add_argument("--dry-run", action="store_true", help="Preview entries without writing them.")
    args = parser.parse_args()

    sessions_dir = Path(os.environ.get("CODEX_SESSIONS_DIR", str(DEFAULT_SESSIONS_DIR)))
    session_files = get_session_files(sessions_dir)
    if not session_files:
        print(f"[codex-desktop-log] No session transcripts found in {sessions_dir}.", file=sys.stderr)
        return

    cutoff = None if args.all else datetime.now(UTC) - timedelta(hours=args.hours)
    repo_root_n = "" if args.no_repo_filter else _normalize(str(Path.cwd()))
    origin = git("remote", "get-url", "origin")
    repo = origin.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git") or Path.cwd().name
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "--short", "HEAD")
    student = git("config", "user.email") or os.environ.get("USERNAME", "unknown")

    log_dir = Path(os.environ.get("AI_LOG_DIR", ".ai-log"))
    log_file = log_dir / "session.jsonl"
    logged_ids = get_logged_entry_ids(log_file)
    new_entries: list[dict] = []
    for message in iter_user_messages(session_files, cutoff, repo_root_n, args.session_id):
        entry = build_entry(message, repo, branch, commit, student)
        if entry["entry_id"] in logged_ids:
            continue
        logged_ids.add(entry["entry_id"])
        new_entries.append(entry)

    if not new_entries:
        print("[codex-desktop-log] No new prompts to log.", file=sys.stderr)
        return
    if args.dry_run:
        print(f"[codex-desktop-log] DRY RUN — would log {len(new_entries)} prompt(s):")
        for entry in new_entries:
            print(f"  [{entry['ts'][:19]}] {entry['prompt'].replace(chr(10), ' ')[:120]}")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as stream:
        for entry in new_entries:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[codex-desktop-log] Logged {len(new_entries)} prompt(s) from Codex Desktop.", file=sys.stderr)


if __name__ == "__main__":
    main()
