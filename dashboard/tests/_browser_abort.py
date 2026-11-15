"""The browser's fetch abort for one endpoint, as data.js's production poll arms it."""

from __future__ import annotations

import subprocess
from pathlib import Path

from _dashboard_helpers import node_path

_PROBE = Path(__file__).parent / 'js' / '_abort_delay_probe.mjs'


def browser_abort_ms(endpoint_path: str) -> int:
    """The delay, in ms, after which data.js aborts its fetch of *endpoint_path*."""
    result = subprocess.run(
        [node_path(), str(_PROBE), endpoint_path],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f'{_PROBE.name} {endpoint_path} exited {result.returncode}\n'
        f'stdout: {result.stdout}\nstderr: {result.stderr}'
    )
    return int(result.stdout)
