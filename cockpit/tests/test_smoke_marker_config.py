"""Behavioral guard for cockpit's smoke-marker deselection safety net (task 2446).

cockpit/orchestrator.yaml's test_command runs the unscoped `pytest tests/`
tree so that a task touching only cockpit files never selects zero tests
(see task 2446 / precedent task 2300, esc-2300-2/3/4). Any test marked
`@pytest.mark.smoke` exercises real X11/tmux against this worktree's live
DISPLAY=:0 host, so smoke tests must be deselected by default -- opt-in only
via `-m smoke` -- regardless of which directory they live in. The guarantee
is registered as a marker in cockpit/pyproject.toml's
[tool.pytest.ini_options] `addopts` (task 2446 step-3), not a tests/smoke/
path convention.

This test proves the guarantee behaviorally rather than by string-matching
config: it writes two throwaway probe tests to a scratch directory and runs
a real `pytest --collect-only` subprocess bound to cockpit's ACTUAL
pyproject.toml (via -c), so cockpit's real addopts governs collection.

RED before task 2446 step-3 lands: cockpit/pyproject.toml has no addopts
yet, so the smoke-marked probe is collected by default and the first
assertion fails.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
COCKPIT_DIR = TESTS_DIR.parent
COCKPIT_PYPROJECT = COCKPIT_DIR / 'pyproject.toml'

_PROBE_SRC = textwrap.dedent(
    """\
    import pytest

    @pytest.mark.smoke
    def test_marked_smoke():
        assert True

    def test_plain():
        assert True
    """
)


@pytest.fixture
def probe_dir() -> Iterator[Path]:
    """A scratch dir INSIDE cockpit/tests/, not the tmp_path fixture's system tmp dir.

    pytest's collection, when the probed file shares no close common ancestor
    with the -c config file's directory (e.g. a system /tmp path vs. this
    repo), falls back to walking from the filesystem root looking for a route
    between the two -- tens of seconds of wasted work on this monorepo.
    Keeping the probe under cockpit/tests/ keeps the common ancestor close so
    collection stays near-instant.
    """
    d = Path(tempfile.mkdtemp(dir=str(TESTS_DIR), prefix='.smoke_marker_probe_'))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _collect(probe_dir: Path, *extra_args: str) -> str:
    """Run `pytest --collect-only` bound to cockpit's real pyproject.toml; return stdout."""
    test_file = probe_dir / 'test_probe.py'
    test_file.write_text(_PROBE_SRC)
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '--collect-only',
            '-q',
            '-p',
            'no:cacheprovider',
            '-c',
            str(COCKPIT_PYPROJECT),
            *extra_args,
            str(test_file),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(COCKPIT_DIR),
    )
    return result.stdout


class TestSmokeMarkerDeselection:
    def test_smoke_marked_tests_deselected_by_default(self, probe_dir: Path) -> None:
        """Without a -m override, @pytest.mark.smoke tests are NOT collected."""
        output = _collect(probe_dir)
        assert 'test_marked_smoke' not in output, (
            'A @pytest.mark.smoke test was collected without a -m override -- '
            'cockpit/pyproject.toml must set addopts = "-m \'not smoke\'" (task 2446 '
            f'step-3) to keep routine unscoped runs from spawning real X11/tmux. '
            f'Output:\n{output}'
        )
        assert 'test_plain' in output, f'Expected the plain test to be collected. Output:\n{output}'

    def test_smoke_marked_tests_selected_with_marker_override(self, probe_dir: Path) -> None:
        """`-m smoke` selects ONLY the smoke-marked test."""
        output = _collect(probe_dir, '-m', 'smoke')
        assert 'test_marked_smoke' in output, (
            f'Expected the smoke test to be collected under -m smoke. Output:\n{output}'
        )
        assert 'test_plain' not in output, (
            f'Expected the plain test to be deselected under -m smoke. Output:\n{output}'
        )
