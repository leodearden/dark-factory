"""Behavioral guard for fused-memory's `integration` marker deselection wiring (task 2736/3019).

Both tests in fused-memory/tests/test_recon_dedup_premise.py are real-Qdrant
integration probes: test_identical_writes_land_with_real_openai_embeddings
additionally makes a real OpenAI network call, but
test_identical_infer_false_writes_all_land_distinct is not "hermetic" either --
it drives a real Mem0Backend against a real, isolated Qdrant collection
(module-level qdrant_skipif()). Both are gated behind the repo-wide
`@pytest.mark.integration` convention (already used by shared/, graphiti/):
registered + deselected by default in fused-memory/pyproject.toml's
[tool.pytest.ini_options], selectable via `-m integration`.

This test proves the marker-deselection wiring behaviorally rather than by
string-matching config: it writes two throwaway probe tests to a scratch
directory and runs a real `pytest --collect-only` subprocess bound to
fused-memory's ACTUAL pyproject.toml (via -c), so fused-memory's real addopts
governs collection.

RED before task 2736 step-2 lands: fused-memory/pyproject.toml registers no
`integration` marker and has no `-m 'not integration'` deselection, so the
integration-marked probe is collected by default (assertion (a) fails) and an
unknown-mark warning appears (assertion (c) fails).

A second guard below, modeled on
cockpit/tests/test_root_config_smoke_deselection.py, collects the REAL
test_recon_dedup_premise.py module (not a throwaway probe) and asserts on the
specific test function names: BOTH real-Qdrant tests must be deselected by
default, and BOTH must become selectable again under `-m integration`.

RED before task 3019 step-2 lands: test_identical_infer_false_writes_all_land_distinct
carries no @pytest.mark.integration (only the module-level qdrant_skipif() +
timeout), so it is still collected by default and that guard's default-run
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
FUSED_MEMORY_DIR = TESTS_DIR.parent
FUSED_MEMORY_PYPROJECT = FUSED_MEMORY_DIR / 'pyproject.toml'

REAL_TEST_MODULE = TESTS_DIR / 'test_recon_dedup_premise.py'
# Must stay in sync with the function names in test_recon_dedup_premise.py --
# a rename there without a matching rename here turns this guard into a
# silent no-op (the collect-only output just won't contain the old name
# either way) rather than an obvious reference error.
REAL_EMBEDDER_TEST_NAME = 'test_identical_writes_land_with_real_openai_embeddings'
DEDUP_QDRANT_TEST_NAME = 'test_identical_infer_false_writes_all_land_distinct'

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

    The probe file lives under fused-memory/tests/, so pytest also loads
    fused-memory/tests/conftest.py to collect it -- the same GraphitiBackend
    / config-schema imports that earned the real-module test below a 90s
    subprocess timeout over the file's default 60s test budget.
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
        timeout=90,
        cwd=str(FUSED_MEMORY_DIR),
    )
    return result.stdout + result.stderr


@pytest.mark.timeout(120)
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


@pytest.mark.timeout(120)
def test_real_embedder_test_gated_by_default_and_selectable_via_marker() -> None:
    """Both real-Qdrant tests in the module must be gated by default and re-selectable.

    Modeled on cockpit/tests/test_root_config_smoke_deselection.py: collects
    the REAL fused-memory/tests/test_recon_dedup_premise.py module (not a
    throwaway probe) under fused-memory's actual pyproject.toml, and asserts
    on the specific test FUNCTION names collected -- ground truth of what
    pytest would actually run, regardless of how the marker is applied.
    Neither test in this module is hermetic: both drive a real, isolated
    Qdrant collection (module-level qdrant_skipif()), and the embeddings test
    additionally makes a real OpenAI network call. Both are therefore
    @pytest.mark.integration and must be deselected by default.

    Collecting the real module imports mem0/qdrant_client, so this gets
    generous timeout headroom over the file's own default 60s cap.
    """

    def _collect_real(*extra_args: str) -> str:
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
                str(REAL_TEST_MODULE),
            ],
            capture_output=True,
            text=True,
            timeout=90,
            cwd=str(FUSED_MEMORY_DIR),
        )
        return result.stdout + result.stderr

    # (a) DEFAULT run: both real-Qdrant tests deselected.
    default_output = _collect_real()
    assert REAL_EMBEDDER_TEST_NAME not in default_output, (
        f'{REAL_EMBEDDER_TEST_NAME} was collected by default -- it must be marked '
        f"@pytest.mark.integration (task 2736 step-4) so the repo's "
        f"-m 'not integration' addopts deselects it. Output:\n{default_output}"
    )
    assert DEDUP_QDRANT_TEST_NAME not in default_output, (
        f'{DEDUP_QDRANT_TEST_NAME} was collected by default -- it drives a real, '
        f'isolated Qdrant collection and must be marked @pytest.mark.integration '
        f"(task 3019) so the repo's -m 'not integration' addopts deselects it "
        f'from the merge lane post-merge sweep. Output:\n{default_output}'
    )

    # (b) `-m integration` override: both real-Qdrant tests become selectable (opt-in, not deleted).
    override_output = _collect_real('-m', 'integration')
    assert REAL_EMBEDDER_TEST_NAME in override_output, (
        f'expected {REAL_EMBEDDER_TEST_NAME} to be collected under -m integration '
        f'(it must remain selectable as an explicit opt-in). Output:\n{override_output}'
    )
    assert DEDUP_QDRANT_TEST_NAME in override_output, (
        f'expected {DEDUP_QDRANT_TEST_NAME} to be collected under -m integration '
        f'(it must remain selectable as an explicit opt-in). Output:\n{override_output}'
    )
