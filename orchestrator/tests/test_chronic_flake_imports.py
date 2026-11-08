"""Importing chronic_flake must not load orchestrator.mcp_lifecycle (task 6216).

The probe runs in a FRESH interpreter: this pytest process has already
imported mcp_lifecycle through sibling test modules, so an in-process
``sys.modules`` check would pass vacuously.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_LOCAL_SRC_DIRS = (
    _THIS_FILE.parents[1] / 'src',
    _THIS_FILE.parents[2] / 'shared' / 'src',
)

_PROBE = (
    'import json, sys\n'
    'import orchestrator.chronic_flake\n'
    "print(json.dumps(sorted(m for m in sys.modules if m.startswith('orchestrator'))))\n"
)


def _local_tree_pythonpath() -> str:
    inherited = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    entries = [str(d) for d in _LOCAL_SRC_DIRS] + inherited
    return os.pathsep.join(e for e in entries if e)


def test_importing_chronic_flake_does_not_load_mcp_lifecycle():
    result = subprocess.run(
        [sys.executable, '-c', _PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, 'PYTHONPATH': _local_tree_pythonpath()},
    )
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    assert 'orchestrator.mcp_lifecycle' not in loaded, 'loaded: ' + ', '.join(loaded)
