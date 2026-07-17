"""Behavioral guard for fused-memory's `integration` marker deselection wiring (task 2736).

fused-memory/tests/test_recon_dedup_premise.py::test_identical_writes_land_with_real_openai_embeddings
is currently gated only by a skipif on OPENAI_API_KEY. With no key it silently
skips (skip-by-accident); with a key present but no network egress (a common
CI/sandbox shape) it runs and errors on the real OpenAI embeddings call. Task
2736 gates that test behind the repo-wide `@pytest.mark.integration` convention
(already used by shared/, graphiti/) instead: registered + deselected by
default in fused-memory/pyproject.toml's [tool.pytest.ini_options], selectable
via `-m integration`.

This test proves the marker-deselection wiring behaviorally rather than by
string-matching config: it writes two throwaway probe tests to a scratch
directory and runs a real `pytest --collect-only` subprocess bound to
fused-memory's ACTUAL pyproject.toml (via -c), so fused-memory's real addopts
governs collection.

RED before task 2736 step-2 lands: fused-memory/pyproject.toml registers no
`integration` marker and has no `-m 'not integration'` deselection, so the
integration-marked probe is collected by default (assertion (a) fails) and an
unknown-mark warning appears (assertion (c) fails).
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
FUSED_MEMORY_DIR = TESTS_DIR.parent
FUSED_MEMORY_PYPROJECT = FUSED_MEMORY_DIR / 'pyproject.toml'

_PROBE_SRC = textwrap.dedent(
    """\
    import pytest

    @pytest.mark.integration
    def test_marked_integration():
        assert True

    def test_plain():
        assert True
    """
)


@pytest.fixture
def probe_dir() -> Iterator[Path]:
    """A scratch dir INSIDE fused-memory/tests/, not the tmp_path fixture's system tmp dir.

    pytest's collection, when the probed file shares no close common ancestor
    with the -c config file's directory (e.g. a system /tmp path vs. this
    repo), falls back to walking from the filesystem root looking for a route
    between the two -- tens of seconds of wasted work on this monorepo.
    Keeping the probe under fused-memory/tests/ keeps the common ancestor
    close so collection stays near-instant.
    """
    d = Path(tempfile.mkdtemp(dir=str(TESTS_DIR), prefix='.integration_marker_probe_'))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _collect(probe_dir: Path, *extra_args: str) -> str:
    """Run `pytest --collect-only` bound to fused-memory's real pyproject.toml.

    Returns combined stdout+stderr. `-n0` overrides fused-memory's `-n auto`
    addopts to keep collection serial without disabling the xdist plugin
    outright (`-p no:xdist` would conflict with the surviving `-n auto` and
    make pytest exit with "unrecognized arguments: -n").
    """
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
            '-n0',
            '-c',
            str(FUSED_MEMORY_PYPROJECT),
            *extra_args,
            str(test_file),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(FUSED_MEMORY_DIR),
    )
    return result.stdout + result.stderr


class TestIntegrationMarkerDeselection:
    def test_integration_marked_tests_deselected_by_default(self, probe_dir: Path) -> None:
        """Without a -m override: not collected by default, and marker is registered."""
        output = _collect(probe_dir)
        assert 'test_marked_integration' not in output, (
            'A @pytest.mark.integration test was collected without a -m override -- '
            'fused-memory/pyproject.toml must append "-m \'not integration\'" to '
            f'addopts (task 2736 step-2). Output:\n{output}'
        )
        assert 'test_plain' in output, f'Expected the plain test to be collected. Output:\n{output}'

        # MARKER REGISTERED -- no unknown-mark warning for @pytest.mark.integration.
        assert 'PytestUnknownMarkWarning' not in output, (
            f'the `integration` marker is not registered under fused-memory pytest config:\n{output}'
        )
        assert 'Unknown pytest.mark.integration' not in output, (
            f'the `integration` marker is not registered under fused-memory pytest config:\n{output}'
        )

    def test_integration_marked_tests_selected_with_marker_override(self, probe_dir: Path) -> None:
        """`-m integration` selects ONLY the integration-marked test."""
        output = _collect(probe_dir, '-m', 'integration')
        assert 'test_marked_integration' in output, (
            f'Expected the integration test to be collected under -m integration. Output:\n{output}'
        )
        assert 'test_plain' not in output, (
            f'Expected the plain test to be deselected under -m integration. Output:\n{output}'
        )
