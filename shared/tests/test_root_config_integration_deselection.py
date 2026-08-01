"""Repo-root pytest config must deselect `integration` and register its marker (task 3444).

WHY THIS EXISTS: pytest reads only ONE [tool.pytest.ini_options] -- the
rootdir's single inifile -- and NEVER merges addopts/markers across
pyproject.toml files. So `shared/pyproject.toml`'s `-m 'not integration'` +
`integration` marker are silently ignored whenever rootdir resolves to the
REPO ROOT, which happens on:

  * an explicit `pytest -c pyproject.toml ...`,
  * a bare `pytest` / `pytest .` from the repo root, and
  * any arg set whose common ancestor is the repo root, e.g.
    `pytest shared/tests/ orchestrator/tests/test_deploy_state.py`
    -- the exact shape of task 3352's observation 1, with no `-c` at all.

Under any of those, the 5 live-CLI tests in
shared/tests/test_cli_invoke_integration.py (plus 4 more elsewhere in
shared/tests/) are COLLECTED AND RUN against real OAuth accounts: ~6 min of
wall clock and real spend per full-class run, and a red suite whenever a live
account is capped. Their `skipif` token guards do NOT fire when tokens are
present -- being marked `integration` is the whole gate, and root ignored it.

This is a FAST, non-live meta-test (deliberately NOT @pytest.mark.integration
itself) that drives a `--collect-only` subprocess bound to the ROOT config via
`-c`, so the ROOT pyproject actually governs collection. `--collect-only` runs
no fixtures, so this guard makes ZERO live CLI calls and costs nothing.

Sibling guards for the same one bug class:
  * cockpit/tests/test_root_config_smoke_deselection.py     (root/`smoke`)
  * fused-memory/tests/test_integration_marker_config.py    (fused-memory/`integration`)
  * tests/scripts/test_pytest_workspace_collection.py
    ::test_root_pyproject_mirrors_member_marker_deselections (generalised, static)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_PYPROJECT = REPO_ROOT / 'pyproject.toml'
TARGET_MODULE = REPO_ROOT / 'shared' / 'tests' / 'test_cli_invoke_integration.py'

# Must stay in sync with the function names in test_cli_invoke_integration.py --
# a rename there without a matching rename here turns this guard into a silent
# no-op (the collect-only output just won't contain the old name either way)
# rather than an obvious reference error.
#
# The three live-CLI tests of TestCrossAccountResume: each invokes the real
# Claude CLI against a real OAuth account.
LIVE_TEST_NAMES = (
    'test_invoke_returns_session_id',
    'test_session_resume_same_account_baseline',
    'test_session_resume_preserves_context_across_accounts',
)
# A TestLooksLikeCapacityFailure unit test: NOT integration-marked, pure
# in-process assertions, and it must KEEP running in ordinary CI.
UNIT_TEST_NAME = 'test_capacity_output_returns_true'


def _collect(*extra_args: str) -> str:
    """Run `pytest --collect-only` on the target module bound to the ROOT pyproject.

    Returns combined stdout+stderr. No `-n` is passed: the root config declares
    no xdist addopts, so adding one would be a behaviour the real root-bound
    runs this guards do not have.
    """
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '-c',
            str(ROOT_PYPROJECT),
            str(TARGET_MODULE),
            '--collect-only',
            '-q',
            '-p',
            'no:cacheprovider',
            *extra_args,
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout + result.stderr


# shared/pyproject.toml caps tests at 60s (signal method). A warm collect is
# ~2s, but this pays a cold root-conftest import that reaches across all seven
# subprojects, so take generous headroom rather than flake on a cold cache.
@pytest.mark.timeout(180)
class TestRootConfigIntegrationDeselection:
    """The root config must gate `integration` -- without over-gating.

    Assertions are keyed on test FUNCTION names, never file paths: a function
    name appears in `--collect-only` output only when the item is actually
    SELECTED, which rules out a collection-ERROR line printing the path and
    producing a false pass. (Same reasoning as
    cockpit/tests/test_root_config_smoke_deselection.py:50-53.)
    """

    def test_live_cli_tests_deselected_under_root_config(self) -> None:
        """(a) The live-CLI tests must NOT be collected by a default root-bound run."""
        combined = _collect()

        for name in LIVE_TEST_NAMES:
            assert name not in combined, (
                f'{name} was COLLECTED under the root pytest config -- a live Claude '
                f'CLI test is not deselected by default, so any root-bound run '
                f'(`-c pyproject.toml`, a bare `pytest` from the repo root, or any '
                f'arg set spanning two subprojects) spends real OAuth budget and goes '
                f'red on a capped account.\n'
                f'FIX: in the ROOT {ROOT_PYPROJECT.name}, extend '
                f'[tool.pytest.ini_options].addopts to use the single combined '
                f'expression -m \'not smoke and not integration\' '
                f'(NOT two -m flags -- argparse is last-wins and the second would '
                f'silently drop the smoke deselect).\n'
                f'Output:\n{combined}'
            )

    def test_non_integration_unit_tests_still_collected_under_root_config(self) -> None:
        """(b) Guard the other side: the deselect must not be over-broad.

        TestLooksLikeCapacityFailure carries no `integration` marker and is
        pure in-process assertion. A deselect that swallowed it would make
        every "not collected" assertion above pass while silently deleting
        real coverage from CI.
        """
        combined = _collect()

        assert UNIT_TEST_NAME in combined, (
            f'{UNIT_TEST_NAME} was NOT collected under the root pytest config. It '
            f'carries no @pytest.mark.integration and must keep running in ordinary '
            f'CI -- the root deselect is over-broad and is silently dropping real '
            f'coverage.\nOutput:\n{combined}'
        )

    def test_integration_marker_registered_under_root_config(self) -> None:
        """(c) The marker must be REGISTERED at root, not merely filtered."""
        combined = _collect()

        assert 'PytestUnknownMarkWarning' not in combined, (
            f'the `integration` marker is not registered under the root pytest '
            f'config -- add it to [tool.pytest.ini_options].markers in the ROOT '
            f'{ROOT_PYPROJECT.name}.\nOutput:\n{combined}'
        )
        assert 'Unknown pytest.mark.integration' not in combined, (
            f'the `integration` marker is not registered under the root pytest '
            f'config -- add it to [tool.pytest.ini_options].markers in the ROOT '
            f'{ROOT_PYPROJECT.name}.\nOutput:\n{combined}'
        )

    def test_integration_tests_still_selectable_via_marker_override(self) -> None:
        """(d) `-m integration` must re-select the live tests: gated, not deleted.

        CLI `-m` overrides the ini addopts `-m` (argparse last-wins), so the
        deliberate opt-in lane stays intact. This asserts the fix gates the
        cross-account invariant behind an explicit flag rather than weakening,
        skipping, or deleting the assertion.
        """
        combined = _collect('-m', 'integration')

        assert 'test_session_resume_preserves_context_across_accounts' in combined, (
            f'expected the cross-account resume test to be collected under '
            f'-m integration -- it must remain selectable as an explicit opt-in '
            f'(it guards a real production path: shared/src/shared/cli_invoke.py '
            f'resumes a capped session on the NEXT account).\nOutput:\n{combined}'
        )
