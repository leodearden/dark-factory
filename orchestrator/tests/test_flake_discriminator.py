"""Tests for the ONE isolated-rerun discriminator (flake-ledger PRD task β).

`plans/flake-ledger-prd.md` §8.1: extract a single
``confirm_isolated_rerun_verdict`` that both existing gates
(``confirm_merge_verify_flake_suppressible`` and
``_main_probe_failure_is_isolated_flake``) become thin wrappers over, so the
two can no longer drift into different notions of "passes in isolation"
(INV-5, no-lockstep-duplication), and so the facts each gate currently drops
on the floor as a bare ``return None`` become a typed ``FlakeSuppression``
verdict (INV-2, structured-facts-at-failure).

The discriminator lives in ``orchestrator/src/orchestrator/verify.py``, NOT in
``flake_ledger.py``: both existing suites patch verify's MODULE GLOBALS by
attribute (``patch.object(verify_module, 'run_verification', ...)`` etc.), and
a discriminator holding its own ``from orchestrator.verify import ...``
binding would silently miss every one of those patches. It imports only the
three vocabulary types (``FlakeVerdict``, ``FlakeCallSite``,
``FlakeSuppression``) from ``flake_ledger`` and never re-declares them.

Test structure mirrors test_verify_merge_flake_suppression.py and
test_verify_preexisting_main_break.py — the house style for this area: a fake
on-disk project layout so the node-id -> subproject existence mapping runs
against real files, ``run_verification`` patched so no real subprocess runs,
a REAL ``OrchestratorConfig`` (never a bare MagicMock — the
``check_bare_magicmock_config`` lint gate), and ``unittest.mock.patch``
exclusively (neither existing file uses pytest ``monkeypatch``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    """A real minimal OrchestratorConfig (never a bare MagicMock — the
    check_bare_magicmock_config lint gate). run_verification is fully patched
    in these tests, so only project_root/git are load-bearing.
    """
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _materialize(worktree: Path, *relpaths: str) -> None:
    """Create real files at *relpaths* under *worktree* so the node-id ->
    subproject existence mapping (`(worktree / prefix / file).exists()` etc.)
    resolves against disk without a real git checkout.
    """
    for rel in relpaths:
        p = worktree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('def test_x():\n    pass\n')


def _module_config(prefix: str) -> ModuleConfig:
    """A subproject whose commands mirror dark_factory's real per-subproject
    shape (``uv run --project X --directory X pytest ...``).
    """
    return ModuleConfig(
        prefix=prefix,
        test_command=(
            f'uv run --project {prefix} --directory {prefix} '
            'pytest tests/ --tb=short -q'
        ),
        lint_command=f'uv run --project {prefix} ruff check src/',
        type_check_command=f'uv run --project {prefix} pyright src/',
    )


def _result(passed: bool, *, category: str = '') -> VerifyResult:
    return VerifyResult(
        passed=passed,
        test_output='',
        lint_output='',
        type_output='',
        summary='ok' if passed else 'fail',
        category=category or ('' if passed else 'test_failure'),
    )


def _fmt_log(call) -> str:
    """Render a mocked ``logger.<level>(fmt, *args)`` call to its final text,
    so a substring assertion sees what an operator would actually read."""
    args = call.args
    if not args:
        return ''
    return (args[0] % args[1:]) if len(args) > 1 else str(args[0])


# ---------------------------------------------------------------------------
# Shared fixtures: two node-ids owned by one subproject file, so they group
# into ONE isolated re-run (mirrors test_verify_merge_flake_suppression's B1).
# ---------------------------------------------------------------------------

FAILED_ID = 'orchestrator/tests/test_x.py::test_y'
CRASH_ID = 'orchestrator/tests/test_x.py::test_z'
TEST_OUTPUT = (
    f'FAILED {FAILED_ID}\n'
    f'{CRASH_ID}\n'
    '[gw3] node down: Not properly terminated\n'
)

#: A test_output with no recoverable pytest node-id (a lint-shaped failure).
NO_NODEID_TEST_OUTPUT = 'ruff: 3 errors found in src/\n'


def _failing_result(
    test_output: str = TEST_OUTPUT,
    *,
    lint_output: str = '',
    type_output: str = '',
) -> VerifyResult:
    """The failing VerifyResult handed to the discriminator."""
    return VerifyResult(
        passed=False,
        test_output=test_output,
        lint_output=lint_output,
        type_output=type_output,
        summary='fail',
        category='test_failure',
        cause_hint=f'FAILED {FAILED_ID}',
    )
