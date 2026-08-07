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


# ---------------------------------------------------------------------------
# S1: the 3-valued isolated-rerun outcome that `_run_isolated_confirm_group`
# currently destroys by collapsing "genuinely failed" and "could not be
# re-run" into one False.
# ---------------------------------------------------------------------------

#: A category in INFRA_TRANSIENT_CATEGORIES — never trusted as confirmation
#: EITHER WAY (its own log line already says "unconfirmable, not counted as a
#: pass"; today that fact dies in the log while the bool says False).
_INFRA_CATEGORY = 'pytest_internalerror'


class TestRunIsolatedConfirmGroupOutcome:
    """`_run_isolated_confirm_group_outcome` -> `_RerunOutcome` (3-valued).

    RED today: neither `_RerunOutcome` nor
    `_run_isolated_confirm_group_outcome` exists.
    """

    def _run(self, verify_module, config, tmp_path):
        return asyncio.run(
            verify_module._run_isolated_confirm_group_outcome(
                tmp_path, config, _module_config('orchestrator'),
            )
        )

    def test_any_attempt_passing_yields_passed_and_short_circuits(
        self, tmp_path: Path,
    ) -> None:
        """A pass on attempt 1 short-circuits — no second attempt is spent."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))
        with patch.object(verify_module, 'run_verification', rv):
            outcome = self._run(verify_module, config, tmp_path)

        assert outcome is verify_module._RerunOutcome.passed, outcome
        assert rv.await_count == 1, (
            f'A pass must short-circuit the attempt loop, got '
            f'{rv.await_count} awaits'
        )

    def test_all_attempts_ordinary_failure_yields_failed(
        self, tmp_path: Path,
    ) -> None:
        """Every attempt returns an ordinary red -> `failed`, and the full
        `_SWEEP_CONFIRM_MAX_ATTEMPTS` budget is spent."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(False))
        with patch.object(verify_module, 'run_verification', rv):
            outcome = self._run(verify_module, config, tmp_path)

        assert outcome is verify_module._RerunOutcome.failed, outcome
        assert rv.await_count == verify_module._SWEEP_CONFIRM_MAX_ATTEMPTS, (
            rv.await_count
        )

    def test_all_attempts_infra_sentinel_yields_unconfirmable(
        self, tmp_path: Path,
    ) -> None:
        """An infra-sentinel category is "we could not re-run", NEVER "the test
        is really red" — the distinction the bool return destroys."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(False, category=_INFRA_CATEGORY))
        with patch.object(verify_module, 'run_verification', rv):
            outcome = self._run(verify_module, config, tmp_path)

        assert outcome is verify_module._RerunOutcome.unconfirmable, outcome

    def test_infra_sentinel_paired_with_passed_true_is_still_unconfirmable(
        self, tmp_path: Path,
    ) -> None:
        """Category-first, deliberately independent of the passed flag —
        mirrors run_main_tip_sweep's own INFRA_TRANSIENT_CATEGORIES branch."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True, category=_INFRA_CATEGORY))
        with patch.object(verify_module, 'run_verification', rv):
            outcome = self._run(verify_module, config, tmp_path)

        assert outcome is verify_module._RerunOutcome.unconfirmable, outcome

    def test_every_attempt_raising_yields_unconfirmable_and_never_propagates(
        self, tmp_path: Path,
    ) -> None:
        """A raise is "could not re-run", not evidence of a real red."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        rv = AsyncMock(side_effect=RuntimeError('boom'))
        with patch.object(verify_module, 'run_verification', rv):
            outcome = self._run(verify_module, config, tmp_path)

        assert outcome is verify_module._RerunOutcome.unconfirmable, outcome
        assert rv.await_count == verify_module._SWEEP_CONFIRM_MAX_ATTEMPTS, (
            rv.await_count
        )

    def test_mixed_infra_then_ordinary_failure_yields_failed(
        self, tmp_path: Path,
    ) -> None:
        """A genuine failure observed ANYWHERE outranks "could not re-run"
        (and a pass anywhere outranks both) — the precedence that keeps an
        unconfirmable verdict honest rather than merely absorbing reds."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        rv = AsyncMock(side_effect=[
            _result(False, category=_INFRA_CATEGORY),
            _result(False),
        ])
        with patch.object(verify_module, 'run_verification', rv):
            outcome = self._run(verify_module, config, tmp_path)

        assert outcome is verify_module._RerunOutcome.failed, outcome

    def test_mixed_ordinary_failure_then_pass_yields_passed(
        self, tmp_path: Path,
    ) -> None:
        """A pass outranks a prior genuine failure — today's semantics
        (`return True` as soon as any attempt passes) preserved exactly."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        rv = AsyncMock(side_effect=[_result(False), _result(True)])
        with patch.object(verify_module, 'run_verification', rv):
            outcome = self._run(verify_module, config, tmp_path)

        assert outcome is verify_module._RerunOutcome.passed, outcome


class TestRunIsolatedConfirmGroupBackCompat:
    """`_run_isolated_confirm_group` still returns a plain `bool`.

    `confirm_main_tip_failure_is_real` and
    `_sweep_failure_reproduces_in_isolation` still consume it and must not
    observe the S2 refactor at all: True iff `passed`, False for BOTH `failed`
    and `unconfirmable`.
    """

    def _run(self, verify_module, config, tmp_path):
        return asyncio.run(
            verify_module._run_isolated_confirm_group(
                tmp_path, config, _module_config('orchestrator'),
            )
        )

    def test_pass_returns_true(self, tmp_path: Path) -> None:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        with patch.object(
            verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
        ):
            got = self._run(verify_module, config, tmp_path)

        assert got is True, got

    def test_ordinary_failure_returns_false(self, tmp_path: Path) -> None:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        with patch.object(
            verify_module, 'run_verification', AsyncMock(return_value=_result(False)),
        ):
            got = self._run(verify_module, config, tmp_path)

        assert got is False, got

    def test_infra_sentinel_returns_false_not_true(self, tmp_path: Path) -> None:
        """`unconfirmable` must NOT leak out of the bool shim as a pass — that
        would newly suppress reds at the three legacy call sites."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        with patch.object(
            verify_module,
            'run_verification',
            AsyncMock(return_value=_result(False, category=_INFRA_CATEGORY)),
        ):
            got = self._run(verify_module, config, tmp_path)

        assert got is False, got

    def test_raising_attempts_return_false_and_never_propagate(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        with patch.object(
            verify_module, 'run_verification', AsyncMock(side_effect=RuntimeError('boom')),
        ):
            got = self._run(verify_module, config, tmp_path)

        assert got is False, got
