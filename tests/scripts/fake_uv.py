"""An executable stand-in for uv that records every argv it is run with (task 8270).

The crash-loop page runs ``uv run ... escalation submit``; tests point
ORCH_WATCHDOG_UV_BIN at this stand-in and read back what was filed. Shared by
tests/scripts/test_watchdog_crash_loop.py and
tests/scripts/test_orchestrator_watchdog.py so the record format has one home.
"""

from __future__ import annotations

import json
import pathlib
import sys


def write_fake_uv(
    tmp_path: pathlib.Path, *, exit_code: int = 0
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write the stand-in under *tmp_path*; return (uv, record)."""
    record = tmp_path / "fake-uv-calls.jsonl"
    uv = tmp_path / "fake-uv"
    uv.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"with open({str(record)!r}, 'a') as f:\n"
        "    f.write(json.dumps(sys.argv) + '\\n')\n"
        f"sys.exit({exit_code})\n"
    )
    uv.chmod(0o755)
    return uv, record


def recorded_argvs(record: pathlib.Path) -> list[list[str]]:
    """Every argv the stand-in was run with, oldest first; [] before its first run."""
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text().splitlines()]


def argv_flag(argv: list[str], name: str) -> str:
    """The value following flag *name* in *argv*."""
    return argv[argv.index(name) + 1]
