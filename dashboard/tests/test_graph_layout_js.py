"""pytest wrapper for the graph_layout.js `node --test` suite.

graph_layout.js (dashboard/src/dashboard/static/redux/graph_layout.js) is a
plain-JS module (no JSX/Babel) with its own node:test-based suite under
dashboard/tests/js/. This wrapper subprocess-runs `node --test` over that
directory so the JS suite is surfaced as part of the normal pytest run (and
therefore CI), instead of requiring a separate invocation.

Hard-fails (does not skip) when node is missing: node v22.22.3 is a verified
part of the host/CI toolchain, so an absent node indicates an environment
regression rather than an optional dependency.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_JS_TESTS_DIR = Path(__file__).parent / 'js'


def test_graph_layout_js_suite_passes() -> None:
    """Run `node --test` over dashboard/tests/js/ and assert a clean exit.

    Surfaces stdout/stderr in the assertion message so a failing JS test's
    actual failure shows up inline in the pytest output.
    """
    node = shutil.which('node')
    assert node is not None, (
        'node executable not found on PATH — node v22.22.3 is required to '
        'run the graph_layout.js test suite (dashboard/tests/js/). This is a '
        'hard failure, not a skip: node is a verified part of the host/CI '
        'toolchain, so its absence is a regression that must not be hidden.'
    )

    result = subprocess.run(
        [node, '--test', str(_JS_TESTS_DIR)],
        capture_output=True,
        text=True,
        cwd=str(_JS_TESTS_DIR.parent),
    )

    assert result.returncode == 0, (
        f'node --test {_JS_TESTS_DIR} exited {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n'
        f'--- stderr ---\n{result.stderr}'
    )
