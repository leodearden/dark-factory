"""Each content-key consumer script resolves ``fused_memory`` from ITS OWN checkout.

sys.path ORDER, not the editable install, decides which checkout's package a
script imports (commit 950de44d8a), so a decoy package earlier on PYTHONPATH
must lose to the script's own ``fused-memory/src``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

FM_ROOT = Path(__file__).resolve().parents[1]

#: The sweep imports mem0 and qdrant_client at module scope.
SUBPROCESS_TIMEOUT_SECS = 120


@pytest.mark.parametrize(
    'script_name',
    [
        'audit_duplicate_memories.py',
        'clear_malformed_empty_memory.py',
        'sweep_toolcall_xml_leak.py',
        'memory_eval_staleness_sweep.py',
    ],
)
def test_the_script_imports_the_fused_memory_tree_beside_it(
    tmp_path: Path, script_name: str
) -> None:
    decoy = tmp_path / 'decoy'
    (decoy / 'fused_memory').mkdir(parents=True)
    (decoy / 'fused_memory' / '__init__.py').write_text('DECOY = True\n')

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import runpy, sys; runpy.run_path(sys.argv[1], run_name="script_provenance");'
            ' import fused_memory; print(fused_memory.__file__)',
            str(FM_ROOT / 'scripts' / script_name),
        ],
        cwd=FM_ROOT.parent,
        env={**os.environ, 'PYTHONPATH': str(decoy)},
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECS,
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve().is_relative_to((FM_ROOT / 'src').resolve())
