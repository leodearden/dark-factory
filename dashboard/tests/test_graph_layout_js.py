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

import re
import shutil
import subprocess
from pathlib import Path

# node's TAP reporter emits a `# tests N` summary line. Matched against
# stdout to confirm the suite actually executed tests — see the docstring
# below for why returncode alone can't be trusted for that.
_TESTS_SUMMARY_RE = re.compile(r'^# tests (\d+)$', re.MULTILINE)

_JS_TESTS_DIR = Path(__file__).parent / 'js'

# node v22.22.3 does NOT recursively discover test files when given a bare
# directory path as a CLI argument — it tries to `require()` the directory
# itself and fails with MODULE_NOT_FOUND regardless of what's inside (verified
# empirically: only an argument-less cwd walk or an explicit glob pattern
# triggers node's test-file discovery). Passing an explicit "**/*.test.mjs"
# glob makes node's own glob engine find and run the suite's files.
_JS_TESTS_GLOB = str(_JS_TESTS_DIR / '**' / '*.test.mjs')


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
        [node, '--test', _JS_TESTS_GLOB],
        capture_output=True,
        text=True,
        cwd=str(_JS_TESTS_DIR.parent),
    )

    assert result.returncode == 0, (
        f'node --test {_JS_TESTS_GLOB} exited {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n'
        f'--- stderr ---\n{result.stderr}'
    )

    # A zero-match glob (e.g. from a rename to .test.js, a moved js/ dir, or a
    # path-resolution change) makes `node --test` run zero tests and still
    # exit 0 — verified empirically: `# tests 0` / returncode 0. Without this
    # check, that failure mode would pass silently forever. Require at least
    # one test to actually have run.
    tests_summary = _TESTS_SUMMARY_RE.search(result.stdout)
    assert tests_summary is not None, (
        f"could not find a '# tests N' summary line in node --test output — "
        f'unable to confirm the suite actually ran\n'
        f'--- stdout ---\n{result.stdout}'
    )
    tests_run = int(tests_summary.group(1))
    assert tests_run > 0, (
        f'node --test {_JS_TESTS_GLOB} exited 0 but reported 0 tests — the '
        f'glob likely matched no test files, which would silently drop all '
        f'JS layout coverage from CI\n'
        f'--- stdout ---\n{result.stdout}'
    )
