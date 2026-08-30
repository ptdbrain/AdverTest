from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_windows_hook_installer_writes_a_bomless_shebang(tmp_path: Path) -> None:
    """Git for Windows must be able to execute the generated pre-push hook."""
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is not installed on this runner")

    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    installer = Path(__file__).resolve().parents[2] / "scripts" / "setup_hooks.ps1"

    subprocess.run(
        [
            powershell,
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
