"""How a test waits for the detached reap that ``session_registry``'s ``launching`` verb starts."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from orchestrator import session_registry as sr


def await_reap_report(root: Path, *, deadline_secs: float = 120.0) -> None:
    """Poll for ``run_reap``'s report, the detached reap's completion signal."""
    report_path = sr.reap_report_path(root)
    deadline = time.monotonic() + deadline_secs
    while not report_path.is_file():
        if time.monotonic() > deadline:
            pytest.fail(f'no reap report at {report_path} within {deadline_secs}s')
        time.sleep(0.05)
