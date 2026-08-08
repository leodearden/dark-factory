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


# ---------------------------------------------------------------------------
# S3: the ONE discriminator — `confirm_isolated_rerun_verdict` — and its
# fully-populated FlakeSuppression on the happy path.
# ---------------------------------------------------------------------------


def _psi(cpu_some10: float, *, read_ok: bool = True):
    """A shared.psi.PsiSample with only cpu_some10 / read_ok load-bearing."""
    from shared.psi import PsiSample

    return PsiSample(
        cpu_some10=cpu_some10,
        mem_some10=0.0,
        mem_full10=0.0,
        io_some10=0.0,
        read_ok=read_ok,
    )


class TestConfirmIsolatedRerunVerdictMergeGate:
    """`confirm_isolated_rerun_verdict(worktree, config, module_configs,
    failing_result, *, call_site, ...) -> FlakeSuppression` at the merge gate.

    RED today: confirm_isolated_rerun_verdict does not exist.
    """

    def _run(self, verify_module, config, module_configs, failing, worktree, **kw):
        return asyncio.run(
            verify_module.confirm_isolated_rerun_verdict(
                worktree, config, module_configs, failing,
                call_site='merge_gate', **kw,
            )
        )

    def test_happy_path_returns_populated_flake_suppression(
        self, tmp_path: Path,
    ) -> None:
        """A clean isolated re-run yields a FULLY-populated FlakeSuppression —
        the whole point of β: the verdict is an object with provenance, not a
        bare list."""
        from datetime import datetime

        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import (
            FlakeCallSite,
            FlakeSuppression,
            FlakeVerdict,
        )

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))

        with (
            patch.object(verify_module, 'run_verification', rv),
            patch.object(
                verify_module, 'read_psi_sample', MagicMock(return_value=_psi(12.5)),
            ),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert isinstance(s, FlakeSuppression), s
        assert s.verdict is FlakeVerdict.passes_in_isolation, s.verdict
        # RAW extracted node-ids in extraction order — never the
        # prefix-qualified group ids the re-run command is built from.
        assert s.test_ids == (FAILED_ID, CRASH_ID), s.test_ids
        assert s.call_site == FlakeCallSite.merge_gate, s.call_site
        assert s.unconfirmable_reason is None, s.unconfirmable_reason
        assert s.runner == 'local', s.runner
        # ISO-8601 and TIMEZONE-AWARE — asserted by parsing, never by matching
        # a literal string.
        stamped = datetime.fromisoformat(s.observed_at)
        assert stamped.tzinfo is not None, s.observed_at
        assert s.psi_cpu_some10 == 12.5, s.psi_cpu_some10

    def test_psi_read_not_ok_maps_to_none(self, tmp_path: Path) -> None:
        """`read_ok=False` is shared.psi's documented fail-open sentinel; the
        contract says psi_cpu_some10 is None in that case, NOT 0.0."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)

        with (
            patch.object(
                verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
            ),
            patch.object(
                verify_module,
                'read_psi_sample',
                MagicMock(return_value=_psi(0.0, read_ok=False)),
            ),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s.verdict is FlakeVerdict.passes_in_isolation, s.verdict
        assert s.psi_cpu_some10 is None, s.psi_cpu_some10

    def test_psi_is_optional_float_without_patching(self, tmp_path: Path) -> None:
        """Unpatched, against the real host: still None-or-float, never a
        crash and never a str."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)

        with patch.object(
            verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s.psi_cpu_some10 is None or isinstance(s.psi_cpu_some10, float), (
            s.psi_cpu_some10
        )

    def test_isolated_command_shape_and_same_tree(self, tmp_path: Path) -> None:
        """INV-3 (SAME-TREE) + INV-4 (serial, isolated, generous timeout), and
        the merge gate's single-shot merge-role call shape."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))

        with patch.object(verify_module, 'run_verification', rv):
            self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        rv.assert_awaited_once()
        # SAME-TREE: the discriminator judges the tree it was HANDED.
        assert rv.call_args.args[0] is tmp_path, rv.call_args.args[0]
        called_mc = rv.call_args.args[2]
        assert '-p no:xdist' in called_mc.test_command, called_mc.test_command
        assert '-o addopts=' in called_mc.test_command, called_mc.test_command
        assert (
            f'--timeout {verify_module._MERGE_FLAKE_CONFIRM_TIMEOUT_SECS}'
            in called_mc.test_command
        ), called_mc.test_command
        assert FAILED_ID in called_mc.test_command, called_mc.test_command
        assert called_mc.lint_command is None, called_mc.lint_command
        assert called_mc.type_check_command is None, called_mc.type_check_command
        # Merge-role, single-shot, cold-timeout semantics.
        assert rv.call_args.kwargs['max_retries'] == 0, rv.call_args.kwargs
        assert rv.call_args.kwargs['is_merge_verify'] is True, rv.call_args.kwargs
        assert rv.call_args.kwargs['role'] == 'merge', rv.call_args.kwargs

    def test_injected_now_is_used_verbatim(self, tmp_path: Path) -> None:
        """`now=` is the determinism seam (α's own open_debt(..., now=None)
        precedent), so a test can pin observed_at without a wall-clock race."""
        from datetime import UTC, datetime

        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        fixed = datetime(2026, 8, 7, 12, 34, 56, tzinfo=UTC)

        with patch.object(
            verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path, now=fixed,
            )

        assert s.observed_at == fixed.isoformat(), s.observed_at

    def test_runner_defaults_to_local_and_is_overridable(
        self, tmp_path: Path,
    ) -> None:
        """`runner` exists so task ε can supply the true origin at the remote
        boundary; 'local' is correct for both of β's in-process call sites."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)

        with patch.object(
            verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path, runner='builder-07',
            )

        assert s.runner == 'builder-07', s.runner


# ---------------------------------------------------------------------------
# S5: TOTALITY (INV-2) — every non-pass path returns a FlakeSuppression, never
# None, and `unconfirmable` is DISTINGUISHABLE from `fails_in_isolation`.
# ---------------------------------------------------------------------------


class TestConfirmIsolatedRerunVerdictTotality:
    """The whole point of β: the facts both gates drop today as a bare
    `return None` become typed verdicts with named reasons."""

    def _run(self, verify_module, config, module_configs, failing, worktree):
        return asyncio.run(
            verify_module.confirm_isolated_rerun_verdict(
                worktree, config, module_configs, failing, call_site='merge_gate',
            )
        )

    def test_no_recoverable_node_ids_is_unconfirmable(self, tmp_path: Path) -> None:
        """A lint-shaped failure names no test — we examined NOTHING, so
        test_ids is empty and no re-run is attempted."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))
        with patch.object(verify_module, 'run_verification', rv):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(NO_NODEID_TEST_OUTPUT), tmp_path,
            )

        assert s is not None, s
        assert s.verdict is FlakeVerdict.unconfirmable, s.verdict
        assert s.test_ids == (), s.test_ids
        assert s.unconfirmable_reason == 'no_recoverable_node_ids', (
            s.unconfirmable_reason
        )
        rv.assert_not_awaited()

    def test_unmapped_node_id_is_unconfirmable_and_keeps_the_ids(
        self, tmp_path: Path,
    ) -> None:
        """Nothing materialized on disk -> the real helper returns its None
        sentinel. We DID examine specific tests, so they are named (PRD B6)."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))
        with patch.object(verify_module, 'run_verification', rv):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s is not None, s
        assert s.verdict is FlakeVerdict.unconfirmable, s.verdict
        assert s.test_ids == (FAILED_ID, CRASH_ID), s.test_ids
        assert s.unconfirmable_reason == 'node_ids_unmapped_to_subproject', (
            s.unconfirmable_reason
        )
        rv.assert_not_awaited()

    def test_grouping_none_sentinel_is_unconfirmable(self, tmp_path: Path) -> None:
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))
        with (
            patch.object(verify_module, 'run_verification', rv),
            patch.object(
                verify_module,
                '_group_node_ids_by_subproject',
                MagicMock(return_value=None),
            ),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s.verdict is FlakeVerdict.unconfirmable, s.verdict
        assert s.unconfirmable_reason == 'node_ids_unmapped_to_subproject', (
            s.unconfirmable_reason
        )
        rv.assert_not_awaited()

    def test_grouping_empty_dict_sentinel_is_unconfirmable(
        self, tmp_path: Path,
    ) -> None:
        """An empty dict is ZERO evidence and must never reach the "all groups
        confirmed" return — pins the `not groups` guard, not an `is None`
        check. Falling through would suppress a genuinely red merge having run
        zero isolated re-runs."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))
        with (
            patch.object(verify_module, 'run_verification', rv),
            patch.object(
                verify_module,
                '_group_node_ids_by_subproject',
                MagicMock(return_value={}),
            ),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s.verdict is FlakeVerdict.unconfirmable, s.verdict
        assert s.unconfirmable_reason == 'node_ids_unmapped_to_subproject', (
            s.unconfirmable_reason
        )
        rv.assert_not_awaited()

    def test_rerun_still_failing_is_fails_in_isolation(self, tmp_path: Path) -> None:
        """A REAL red — the one verdict that means "this is not a flake"."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        with patch.object(
            verify_module, 'run_verification', AsyncMock(return_value=_result(False)),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s.verdict is FlakeVerdict.fails_in_isolation, s.verdict
        assert s.test_ids == (FAILED_ID, CRASH_ID), s.test_ids
        assert s.unconfirmable_reason is None, s.unconfirmable_reason

    def test_infra_sentinel_rerun_is_unconfirmable_naming_the_category(
        self, tmp_path: Path,
    ) -> None:
        """An infra sentinel is never trusted as confirmation EITHER WAY — so
        it must NOT be reported as a real red. Category-first, deliberately
        independent of the passed flag."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        with patch.object(
            verify_module,
            'run_verification',
            AsyncMock(return_value=_result(True, category=_INFRA_CATEGORY)),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s.verdict is FlakeVerdict.unconfirmable, s.verdict
        assert s.unconfirmable_reason is not None
        assert s.unconfirmable_reason.startswith('infra_transient_rerun:'), (
            s.unconfirmable_reason
        )
        assert _INFRA_CATEGORY in s.unconfirmable_reason, s.unconfirmable_reason
        assert s.test_ids == (FAILED_ID, CRASH_ID), s.test_ids

    def test_unconfirmable_is_never_conflated_with_not_a_flake(
        self, tmp_path: Path,
    ) -> None:
        """THE claim this task is built on, spelled out: for BOTH mapping
        failures the answer is a verdict that is neither None nor
        `fails_in_isolation`. "We could not tell" is not "it is really red"."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        config = _make_config(tmp_path)
        module_configs = [_module_config('orchestrator')]

        with patch.object(
            verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
        ):
            # (a) nothing to examine
            no_ids = self._run(
                verify_module, config, module_configs,
                _failing_result(NO_NODEID_TEST_OUTPUT), tmp_path,
            )
            # (b) examined, but unmappable
            unmapped = self._run(
                verify_module, config, module_configs, _failing_result(), tmp_path,
            )

        for s in (no_ids, unmapped):
            assert s is not None, 'the discriminator is TOTAL — never None'
            assert s.verdict != FlakeVerdict.fails_in_isolation, s.verdict
            assert s.verdict is FlakeVerdict.unconfirmable, s.verdict
            assert s.unconfirmable_reason, s.unconfirmable_reason

    def test_unmapped_path_still_logs_not_suppressing_with_the_node_id(
        self, tmp_path: Path,
    ) -> None:
        """The merge-lane operator line survives the extraction verbatim —
        `test_unmapped_node_id_logs_not_suppressing` in the existing suite
        matches on exactly this."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        with (
            patch.object(
                verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
            ),
            patch.object(verify_module, 'logger') as mock_logger,
        ):
            self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        rendered = [_fmt_log(call) for call in mock_logger.info.call_args_list]
        matching = [
            msg for msg in rendered
            if 'not suppressing' in msg and FAILED_ID in msg
        ]
        assert len(matching) == 1, (
            f'Expected exactly ONE merge-lane INFO naming both the offending '
            f'node-id and the "not suppressing" verdict; got {rendered!r}'
        )


# ---------------------------------------------------------------------------
# S7: NEVER RAISES + fail CLOSED (INV-1), and the unsupported-call-site path.
# ---------------------------------------------------------------------------


class TestConfirmIsolatedRerunVerdictNeverRaises:
    """The merge path (merge_queue.py) has NO VerifyInfraError handler, so an
    uncaught raise here stalls the merge queue. Every test asserts
    `result is not None` FIRST, so a regression to a bare None return is
    caught as a distinct failure rather than as a confusing attribute error.
    """

    def _run(self, verify_module, config, module_configs, failing, worktree, **kw):
        kw.setdefault('call_site', 'merge_gate')
        return asyncio.run(
            verify_module.confirm_isolated_rerun_verdict(
                worktree, config, module_configs, failing, **kw,
            )
        )

    def test_run_verification_raising_fails_closed_to_red(
        self, tmp_path: Path,
    ) -> None:
        """Merge stays RED. Fail-closed, not fail-open."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeSuppression, FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)

        with patch.object(
            verify_module, 'run_verification', AsyncMock(side_effect=RuntimeError('boom')),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s is not None, 'the discriminator is TOTAL — never None'
        assert isinstance(s, FlakeSuppression), s
        assert s.verdict is FlakeVerdict.fails_in_isolation, s.verdict

    def test_grouping_helper_raising_fails_closed_to_red(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)

        with (
            patch.object(
                verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
            ),
            patch.object(
                verify_module,
                '_group_node_ids_by_subproject',
                MagicMock(side_effect=RuntimeError('boom')),
            ),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s is not None
        assert s.verdict is FlakeVerdict.fails_in_isolation, s.verdict

    def test_node_id_extraction_raising_fails_closed_with_a_tuple(
        self, tmp_path: Path,
    ) -> None:
        """The handler must not itself blow up on an unbound local — so
        node_ids and call_site have to be bound BEFORE the try (α's
        record_flake_occurrence sets this precedent)."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)

        with (
            patch.object(
                verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
            ),
            patch.object(
                verify_module,
                '_extract_failing_test_ids',
                MagicMock(side_effect=RuntimeError('boom')),
            ),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s is not None
        assert s.verdict is FlakeVerdict.fails_in_isolation, s.verdict
        assert isinstance(s.test_ids, tuple), s.test_ids

    def test_psi_read_raising_does_not_change_the_verdict(
        self, tmp_path: Path,
    ) -> None:
        """A TELEMETRY read must never change a gate's answer: still
        passes_in_isolation on a clean re-run, with psi simply absent."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)

        with (
            patch.object(
                verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
            ),
            patch.object(
                verify_module,
                'read_psi_sample',
                MagicMock(side_effect=OSError('/proc/pressure/cpu unreadable')),
            ),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s is not None
        assert s.verdict is FlakeVerdict.passes_in_isolation, s.verdict
        assert s.psi_cpu_some10 is None, s.psi_cpu_some10

    def test_chronic_marker_call_site_is_unconfirmable_not_an_exception(
        self, tmp_path: Path,
    ) -> None:
        """A real FlakeCallSite member with no β policy (task κ owns it) is a
        KNOWN-UNKNOWN, not a defect: `unconfirmable` with a naming reason is
        the honest, countable answer. Routing it through the fail-closed
        exception path would both mislabel it and hide it from θ's class-1
        rate."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))

        with patch.object(verify_module, 'run_verification', rv):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path, call_site='chronic_marker',
            )

        assert s is not None
        assert s.verdict is FlakeVerdict.unconfirmable, s.verdict
        assert s.unconfirmable_reason == 'unsupported_call_site:chronic_marker', (
            s.unconfirmable_reason
        )
        assert s.test_ids == (), s.test_ids
        rv.assert_not_awaited()

    def test_uncoercible_call_site_fails_closed_without_raising(
        self, tmp_path: Path,
    ) -> None:
        """A garbage call_site is a PROGRAMMING error, so it fails closed —
        but it still may not raise into the merge path."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeSuppression, FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)

        with patch.object(
            verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path, call_site='not_a_real_site',
            )

        assert s is not None
        assert isinstance(s, FlakeSuppression), s
        assert s.verdict is FlakeVerdict.fails_in_isolation, s.verdict


# ---------------------------------------------------------------------------
# S9: the `main_probe` policy — the SAME discriminator, different knobs.
# ---------------------------------------------------------------------------

#: A node-id owned by a SECOND subproject, so a two-group run can be built.
OTHER_ID = 'fused-memory/tests/test_q.py::test_q'
TWO_GROUP_TEST_OUTPUT = f'FAILED {FAILED_ID}\nFAILED {OTHER_ID}\n'


class TestConfirmIsolatedRerunVerdictMainProbe:
    """Same body, main_probe calibration: the BOUNDED 2-attempt engine, its own
    timeout constant, its own log label, and the other-leg precondition."""

    def _run(self, verify_module, config, module_configs, failing, worktree, **kw):
        return asyncio.run(
            verify_module.confirm_isolated_rerun_verdict(
                worktree, config, module_configs, failing,
                call_site='main_probe', **kw,
            )
        )

    def test_clean_isolated_rerun_passes_in_isolation(self, tmp_path: Path) -> None:
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeCallSite, FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)

        with patch.object(
            verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
        ):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s.verdict is FlakeVerdict.passes_in_isolation, s.verdict
        assert s.test_ids == (FAILED_ID, CRASH_ID), s.test_ids
        assert s.call_site == FlakeCallSite.main_probe, s.call_site

    def test_uses_the_bounded_two_attempt_engine(self, tmp_path: Path) -> None:
        """The main probe retries; the merge gate does not. Asserted against
        the CONSTANT so retuning the bound never silently passes here."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(False))

        with patch.object(verify_module, 'run_verification', rv):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert s.verdict is FlakeVerdict.fails_in_isolation, s.verdict
        assert rv.await_count == verify_module._SWEEP_CONFIRM_MAX_ATTEMPTS, (
            f'The main probe uses the BOUNDED multi-attempt engine, not the '
            f'merge gate single shot; got {rv.await_count} awaits'
        )

    def test_isolated_command_uses_the_main_probe_timeout_constant(
        self, tmp_path: Path,
    ) -> None:
        """A constant SEPARATE from the merge gate's BY DESIGN — asserted
        against the constant, not the literal 300, so retuning one can never
        silently retune the other."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))

        with patch.object(verify_module, 'run_verification', rv):
            self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        called_mc = rv.call_args.args[2]
        assert '-p no:xdist' in called_mc.test_command, called_mc.test_command
        assert '-o addopts=' in called_mc.test_command, called_mc.test_command
        assert (
            f'--timeout {verify_module._MAIN_PROBE_CONFIRM_TIMEOUT_SECS}'
            in called_mc.test_command
        ), called_mc.test_command
        assert called_mc.lint_command is None, called_mc.lint_command
        assert called_mc.type_check_command is None, called_mc.type_check_command

    def test_run_verification_called_with_exactly_three_positionals(
        self, tmp_path: Path,
    ) -> None:
        """No role / is_merge_verify kwargs — the main probe runs at
        run_verification's own defaults. The sync side_effect would TypeError
        on a fourth positional; this mirrors the existing suite's fake and is
        the load-bearing shape pin."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        seen: list[dict] = []

        def _fake_run_verification(worktree, config, module_config, **kwargs):
            seen.append(kwargs)
            return _result(True)

        rv = AsyncMock(side_effect=_fake_run_verification)
        with patch.object(verify_module, 'run_verification', rv):
            self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert len(rv.call_args.args) == 3, rv.call_args.args
        assert rv.call_args.args[0] is tmp_path, rv.call_args.args[0]
        assert seen == [{'max_retries': 0}], seen

    def test_non_empty_lint_output_bails_as_other_leg_failed(
        self, tmp_path: Path,
    ) -> None:
        """_summarize_checks/_worst_category picks ONE category across up to
        three legs, so a matched signature does not prove the lint leg was
        clean. Bails BEFORE any work."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))

        with patch.object(verify_module, 'run_verification', rv):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(lint_output='E501 line too long\n'), tmp_path,
            )

        assert s.verdict is FlakeVerdict.unconfirmable, s.verdict
        assert s.unconfirmable_reason == 'other_leg_failed', s.unconfirmable_reason
        assert s.test_ids == (), s.test_ids
        rv.assert_not_awaited()

    def test_non_empty_type_output_bails_as_other_leg_failed(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))

        with patch.object(verify_module, 'run_verification', rv):
            s = self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(type_output='error: incompatible type\n'), tmp_path,
            )

        assert s.verdict is FlakeVerdict.unconfirmable, s.verdict
        assert s.unconfirmable_reason == 'other_leg_failed', s.unconfirmable_reason
        rv.assert_not_awaited()

    def test_the_precondition_is_policy_scoped_not_global(
        self, tmp_path: Path,
    ) -> None:
        """THE asymmetry test: the SAME failing result with non-empty
        lint_output bails under main_probe and does NOT under merge_gate. The
        merge gate has no such bail today, PRD §3 does not ask for one, and
        adding it would newly refuse to suppress merges carrying any lint
        output."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        failing = _failing_result(lint_output='E501 line too long\n')

        with patch.object(
            verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
        ):
            probe = self._run(
                verify_module, config, [_module_config('orchestrator')],
                failing, tmp_path,
            )
            merge = asyncio.run(
                verify_module.confirm_isolated_rerun_verdict(
                    tmp_path, config, [_module_config('orchestrator')], failing,
                    call_site='merge_gate',
                )
            )

        assert probe.verdict is FlakeVerdict.unconfirmable, probe.verdict
        assert probe.unconfirmable_reason == 'other_leg_failed', probe
        assert merge.verdict is FlakeVerdict.passes_in_isolation, merge.verdict

    def test_grouping_helper_receives_the_main_probe_log_label(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        spy = MagicMock(return_value={'orchestrator': [FAILED_ID]})

        with (
            patch.object(
                verify_module, 'run_verification', AsyncMock(return_value=_result(True)),
            ),
            patch.object(verify_module, '_group_node_ids_by_subproject', spy),
        ):
            self._run(
                verify_module, config, [_module_config('orchestrator')],
                _failing_result(), tmp_path,
            )

        assert spy.call_count == 1, spy.call_args_list
        assert spy.call_args.kwargs['log_label'] == (
            'verify_failure_is_preexisting_on_main confirm gate'
        ), spy.call_args.kwargs

    def test_second_group_still_failing_yields_fails_in_isolation(
        self, tmp_path: Path,
    ) -> None:
        """Two subprojects: the first confirms, the second does not — ALL
        groups must confirm green, and both node-ids are re-run."""
        from orchestrator import verify as verify_module
        from orchestrator.flake_ledger import FlakeVerdict

        _materialize(
            tmp_path,
            'orchestrator/tests/test_x.py',
            'fused-memory/tests/test_q.py',
        )
        config = _make_config(tmp_path)
        rv = AsyncMock(side_effect=[_result(True), _result(False), _result(False)])

        with patch.object(verify_module, 'run_verification', rv):
            s = self._run(
                verify_module, config,
                [_module_config('orchestrator'), _module_config('fused-memory')],
                _failing_result(TWO_GROUP_TEST_OUTPUT), tmp_path,
            )

        assert s.verdict is FlakeVerdict.fails_in_isolation, s.verdict
        commands = [c.args[2].test_command for c in rv.await_args_list]
        assert any(FAILED_ID in cmd for cmd in commands), commands
        assert any(OTHER_ID in cmd for cmd in commands), commands
