"""The two-way boundary suite: real served bodies through the real client (sketch #1-#13).

``plans/dashboard-one-datum-one-path-prd.md`` sketches each datum's server row
and the client decision that renders it. Component tests prove each half
alone; this suite proves the pair agrees. ``_boundary_payloads`` drives the
real routes over fixture substrates, and ``js/boundary_*.test.mjs`` apply
those exact bodies through data.js's real refreshOne. This file is the node
half's one owner: ``test_graph_layout_js.py`` does not run it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import _boundary_payloads
import pytest
from _lock_chip_matrix import node_path

_JS_DIR = Path(__file__).parent / 'js'

_SKETCH_ROWS = frozenset(range(1, 14))

# A top-level TAP result for a passing test named 'sketch #N: ...'; TAP
# escapes the '#' because a bare one opens a directive. A `not ok` line never
# matches, so a failing row counts as uncovered.
_PASSING_ROW_RE = re.compile(r'^ok \d+ - sketch \\#(\d+):', re.MULTILINE)


@pytest.fixture(scope='module')
def served_bodies(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _boundary_payloads.build_all(tmp_path_factory.mktemp('substrates'))


def test_the_boundary_suite_covers_every_sketch_row(served_bodies, tmp_path):
    for scenario, body in served_bodies.items():
        (tmp_path / f'{scenario}.json').write_text(json.dumps(body))
    files = sorted(_JS_DIR.glob('boundary_*.test.mjs'))
    assert files, f'no boundary_*.test.mjs under {_JS_DIR}, so no sketch row crosses the wire'

    result = subprocess.run(
        [node_path(), '--test', '--test-reporter=tap', *map(str, files)],
        capture_output=True,
        text=True,
        cwd=_JS_DIR,
        env={**os.environ, 'DF_BOUNDARY_PAYLOADS': str(tmp_path)},
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        f'the boundary suite exited {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
    covered = {int(row) for row in _PASSING_ROW_RE.findall(result.stdout)}
    missing = sorted(_SKETCH_ROWS - covered)
    assert not missing, (
        f'sketch row(s) {missing} have no passing top-level "sketch #N:" test in '
        f'{[file.name for file in files]}\n--- stdout ---\n{result.stdout}'
    )
