from __future__ import annotations

import subprocess
from pathlib import Path


def test_windows_hook_installer_writes_a_bomless_shebang(tmp_path: Path) -> None:
    """Git for Windows must be able to execute the generated pre-push hook."""
    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    installer = Path(__file__).resolve().parents[2] / "scripts" / "setup_hooks.ps1"

    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(installer),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (tmp_path / ".git" / "hooks" / "pre-push").read_bytes().startswith(b"#!/usr/bin/env bash")
