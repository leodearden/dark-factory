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

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_PYPROJECT = REPO_ROOT / 'pyproject.toml'
TARGET_MODULE = REPO_ROOT / 'shared' / 'tests' / 'test_cli_invoke_integration.py'

# ALL FIVE integration-marked live-CLI tests in the target module: the three of
# TestCrossAccountResume and the two of TestConfigDirCredentials. Each invokes
# the real Claude CLI against a real OAuth account.
#
# Must stay in sync with the function names in test_cli_invoke_integration.py.
# A rename there would otherwise turn assertion (a) into a silent no-op -- the
# collect-only output just won't contain the old name either way -- rather than
# an obvious reference error. Assertion (d) is what makes that drift LOUD: it
# re-selects with `-m integration` and requires EVERY name here to be collected,
# so a stale entry fails there instead of passing vacuously here.
LIVE_TEST_NAMES = (
    'test_invoke_returns_session_id',
    'test_session_resume_same_account_baseline',
    'test_session_resume_preserves_context_across_accounts',
    'test_env_var_auth_succeeds',
    'test_config_dir_plus_env_var_auth_succeeds',
)
# The non-integration canary module for assertion (b), collected in the SAME
# root-bound subprocess as TARGET_MODULE (one `-c` run, two path args) rather
# than in a second one: (a)/(b)/(c) then still share a single cached collect,
# and an arg set spanning two modules is itself one of the root-bound shapes
# this guard exists for.
#
# It HAS to be a separate module: TARGET_MODULE is now 100% integration-marked.
# Task 3483 deduplicated the capacity-skip helper and moved that module's
# TestLooksLikeCapacityFailure unit class out to tests/test_capacity_skip.py,
# leaving nothing non-integration inside TARGET_MODULE to anchor (b) on.
UNIT_TEST_MODULE = REPO_ROOT / 'shared' / 'tests' / 'test_capacity_skip.py'

# A test_capacity_skip.py unit test: NOT integration-marked, pure in-process
# assertions, and it must KEEP running in ordinary CI.
#
# Unlike LIVE_TEST_NAMES this pointer cannot go stale silently: a rename or a
# move makes the name vanish from the collect output, which is the very thing
# (b) asserts against, so it fails LOUDLY. That is exactly how task 3483's move
# of this test out of TARGET_MODULE was caught.
UNIT_TEST_NAME = 'test_capacity_output_returns_true'

# The only exit codes a `--collect-only` run legitimately produces: 0 (OK) and
# 5 (NO_TESTS_COLLECTED -- e.g. everything deselected). Anything else means the
# subprocess never collected at all: 4 is a USAGE_ERROR (a bad `-c` binding),
# 3 an INTERNAL_ERROR (a root-conftest ImportError). Without this check every
# "was NOT collected" assertion below would pass VACUOUSLY on a broken config.
_OK_COLLECT_EXIT_CODES = frozenset({0, 5})


def _collect(
    *extra_args: str,
    targets: tuple[Path, ...] = (TARGET_MODULE, UNIT_TEST_MODULE),
) -> str:
    """Run `pytest --collect-only` on `targets` bound to the ROOT pyproject.

    Returns combined stdout+stderr. No `-n` is passed: the root config declares
    no xdist addopts, so adding one would be a behaviour the real root-bound
    runs this guards do not have.
    """
    argv = [
        sys.executable,
        '-m',
        'pytest',
        '-c',
        str(ROOT_PYPROJECT),
        *(str(target) for target in targets),
        '--collect-only',
        '-q',
        '-p',
        'no:cacheprovider',
        *extra_args,
    ]
    result = subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = result.stdout + result.stderr

    assert result.returncode in _OK_COLLECT_EXIT_CODES, (
        f'the root-bound `pytest --collect-only` subprocess exited '
        f'{result.returncode}; expected 0 (OK) or 5 (no tests collected). It never '
        f'collected anything, so the assertions in this file would pass vacuously '
        f'on a config that is in fact broken -- most likely a usage error in the '
        f'`-c {ROOT_PYPROJECT.name}` binding or an import error in the root '
        f'conftest, NOT the marker gating this guard is about.\n'
        f'argv: {shlex.join(argv)}\nOutput:\n{combined}'
    )
    return combined


@pytest.fixture(scope='module')
def default_collect() -> str:
    """One cached DEFAULT (no extra args) root-bound collect, shared by (a)/(b)/(c).

    Those three assert against a byte-identical run, and each `_collect()` spawns
    a full pytest whose root-conftest import reaches across all seven
    subprojects. Module scope pays that once instead of three times.

    Covers BOTH modules -- the all-integration TARGET_MODULE that (a) needs and
    the non-integration UNIT_TEST_MODULE that (b) needs -- so the two-sided
    check still costs exactly one subprocess.
    """
    return _collect()


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

    def test_live_cli_tests_deselected_under_root_config(self, default_collect: str) -> None:
        """(a) The live-CLI tests must NOT be collected by a default root-bound run."""
        combined = default_collect

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

    def test_non_integration_unit_tests_still_collected_under_root_config(
        self, default_collect: str
    ) -> None:
        """(b) Guard the other side: the deselect must not be over-broad.

        UNIT_TEST_MODULE carries no `integration` marker and is pure
        in-process assertion. A deselect that swallowed it would make every
        "not collected" assertion above pass while silently deleting real
        coverage from CI.
        """
        combined = default_collect

        assert UNIT_TEST_NAME in combined, (
            f'{UNIT_TEST_NAME} was NOT collected under the root pytest config. It '
            f'carries no @pytest.mark.integration and must keep running in ordinary '
            f'CI -- the root deselect is over-broad and is silently dropping real '
            f'coverage.\nOutput:\n{combined}'
        )

    def test_integration_marker_registered_under_root_config(self, default_collect: str) -> None:
        """(c) The marker must be REGISTERED at root, not merely filtered."""
        combined = default_collect

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

        Checks EVERY name in LIVE_TEST_NAMES, not just one: this is also the
        anti-drift half of assertion (a). A rename (or deletion) in
        test_cli_invoke_integration.py makes (a) pass vacuously -- the old name
        is absent from the collect output either way -- and fails LOUDLY here.
        """
        # TARGET_MODULE only: this assertion is about re-selecting the live
        # tests, and its failure message names that module specifically.
        combined = _collect('-m', 'integration', targets=(TARGET_MODULE,))

        missing = [name for name in LIVE_TEST_NAMES if name not in combined]
        assert not missing, (
            f'these integration-marked tests were NOT collected under '
            f'-m integration: {", ".join(missing)}.\n'
            f'Either (1) they were renamed/removed in {TARGET_MODULE.name} and '
            f'LIVE_TEST_NAMES in this file is now stale -- which would ALSO make the '
            f'"deselected by default" assertion above pass vacuously, so fix the list '
            f'rather than this test -- or (2) they lost their @pytest.mark.integration '
            f'and are now running live on every ordinary run.\n'
            f'They must remain selectable as an explicit opt-in: '
            f'test_session_resume_preserves_context_across_accounts guards a real '
            f'production path (shared/src/shared/cli_invoke.py resumes a capped '
            f'session on the NEXT account).\nOutput:\n{combined}'
        )
