"""Tests for orchestrator.merge_shadow: extracted warm-vs-cold shadow-compare
subsystem (MQ-refactor task γ).

These tests encode the behavior-preserving contracts of the module split,
mirroring task β's test_merge_gates.py:

1. Module-existence — ``orchestrator.merge_shadow`` exists and exports the
   full closure of moved symbols (state/diff types, per-test parsers,
   sentinels, and the shadow-compare functions).
2. Logger-name — the module logs under the ``orchestrator.merge_queue``
   logger name (not ``orchestrator.merge_shadow``) so existing ``caplog``
   assertions filtered to the merge_queue logger keep capturing the moved
   shadow-compare's WARNING/INFO lines.
3. Reach-back / string-path monkeypatch routing — the existing test suite
   monkeypatches shadow-compare dependencies by STRING PATH
   ``orchestrator.merge_queue.<name>``.  A moved function must resolve a
   monkeypatched-or-staying sibling via a function-local deferred import so
   those patches stay effective even though the function body now lives in
   this module.  Each ``TestReachBackRouting`` test below patches BOTH
   namespaces (merge_shadow-local naive vs. merge_queue reach-back target)
   with CONTRASTING behaviour — or, for a dependency with no merge_shadow
   local copy at all (``_run_unscoped_typechecks``), patches only the
   merge_queue side — so the assertion is unambiguous about which one
   governed.  Includes ``run_scoped_verification``, which the step-3 plan
   prose didn't enumerate by name but which
   :class:`~orchestrator.verify_runner.LocalRunner`'s own docstring and the
   existing ``TestColdShadowVerifyLocalOnly`` tests in
   ``test_merge_queue_multihost_wiring.py`` require: those tests construct a
   real ``LocalRunner`` and patch ``orchestrator.merge_queue.run_scoped_verification``
   to control it, which only works if the injected callable is resolved via
   the merge_queue reach-back rather than a bare local import.
4. Shim re-export identity (added in a later step, once merge_queue.py's
   shim swap lands).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult


def test_merge_shadow_exports_moved_public_symbols() -> None:
    from orchestrator.merge_shadow import (
        _LIBTEST_TEST_LINE_RE,
        _NEXTEST_SUMMARY_LINE_RE,
        _NEXTEST_TEST_LINE_RE,
        _WARM_COLD_SHADOW_SENTINEL,
        _WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL,
        ShadowCompareDiff,
        ShadowCompareState,
        _alarm_warm_shadow_unparseable,
        _classify_test_status,
        _load_shadow_compare_state,
        _maybe_schedule_shadow_compare,
        _nextest_reported_test_count,
        _persistent_alarm_tests,
        _run_cold_shadow_verify,
        _run_shadow_compare,
        _save_shadow_compare_state,
        _shadow_compare_due,
        _submit_shadow_divergence_escalation,
        diff_per_test_results,
        parse_per_test_results,
    )

    for name, obj in {
        'ShadowCompareState': ShadowCompareState,
        'ShadowCompareDiff': ShadowCompareDiff,
        'parse_per_test_results': parse_per_test_results,
        '_classify_test_status': _classify_test_status,
        '_nextest_reported_test_count': _nextest_reported_test_count,
        '_NEXTEST_TEST_LINE_RE': _NEXTEST_TEST_LINE_RE,
        '_LIBTEST_TEST_LINE_RE': _LIBTEST_TEST_LINE_RE,
        '_NEXTEST_SUMMARY_LINE_RE': _NEXTEST_SUMMARY_LINE_RE,
        'diff_per_test_results': diff_per_test_results,
        '_persistent_alarm_tests': _persistent_alarm_tests,
        '_load_shadow_compare_state': _load_shadow_compare_state,
        '_save_shadow_compare_state': _save_shadow_compare_state,
        '_shadow_compare_due': _shadow_compare_due,
        '_WARM_COLD_SHADOW_SENTINEL': _WARM_COLD_SHADOW_SENTINEL,
        '_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL': _WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL,
        '_submit_shadow_divergence_escalation': _submit_shadow_divergence_escalation,
        '_alarm_warm_shadow_unparseable': _alarm_warm_shadow_unparseable,
        '_run_cold_shadow_verify': _run_cold_shadow_verify,
        '_run_shadow_compare': _run_shadow_compare,
        '_maybe_schedule_shadow_compare': _maybe_schedule_shadow_compare,
    }.items():
        assert obj is not None, f'{name} must not be None'


def test_merge_shadow_logger_name_is_merge_queue() -> None:
    """merge_shadow emits under the 'orchestrator.merge_queue' logger name.

    RED (pre-module): ``orchestrator.merge_shadow`` does not exist yet.

    Required so existing ``caplog.at_level(..., logger='orchestrator.merge_queue')``
    assertions in the warm/cold shadow-compare test files keep capturing the
    moved functions' WARNING/INFO-level messages after relocation.
    """
    import orchestrator.merge_shadow as merge_shadow

    assert merge_shadow.logger.name == 'orchestrator.merge_queue'


@pytest.mark.asyncio
class TestReachBackRouting:
    """Reach-back / string-path monkeypatch routing contract.

    Each test patches the SAME logical dependency in both namespaces with
    CONTRASTING behaviour: the merge_shadow-local (naive) patch — where one
    exists — is engineered to raise/diverge so an accidental direct
    reference is caught unambiguously; the orchestrator.merge_queue
    (reach-back target) patch supplies the value that must actually govern
    the outcome.  A dependency with no merge_shadow-local copy at all
    (``_run_unscoped_typechecks``) is patched only on the merge_queue side,
    mirroring task β's analogous ``_check_post_merge_pyright`` test.
    """

    async def test_run_shadow_compare_reachback_to_run_cold_shadow_verify(self) -> None:
        """(1) _run_shadow_compare must resolve _run_cold_shadow_verify via
        orchestrator.merge_queue, not the co-located merge_shadow copy."""
        from orchestrator.merge_shadow import _run_shadow_compare

        git_ops = MagicMock()
        req = MagicMock()
        req.task_id = 'task-shadow-compare-reachback'

        warm_results = {'pkg t': 'pass'}
        naive_cold = AsyncMock(side_effect=AssertionError(
            'naive merge_shadow._run_cold_shadow_verify must not be called'
        ))
        # Reach-back target: identical to warm → no divergence → parity-ok event.
        reachback_cold = AsyncMock(return_value=dict(warm_results))

        event_store = MagicMock()

        with (
            patch('orchestrator.merge_shadow._run_cold_shadow_verify', naive_cold),
            patch('orchestrator.merge_queue._run_cold_shadow_verify', reachback_cold),
        ):
            await _run_shadow_compare(
                git_ops, req, 'commit-sha', warm_results, None, event_store,
            )

        naive_cold.assert_not_called()
        reachback_cold.assert_awaited_once()
        event_store.emit.assert_called_once()
        from orchestrator.event_store import EventType
        emitted_type = event_store.emit.call_args.args[0]
        assert emitted_type == EventType.verdict_parity_ok, (
            f'expected the orchestrator.merge_queue-patched cold results (agreeing '
            f'with warm) to govern and emit verdict_parity_ok, got emit call '
            f'{event_store.emit.call_args!r}'
        )

    async def test_maybe_schedule_shadow_compare_reachback_to_run_shadow_compare(
        self, tmp_path: Path,
    ) -> None:
        """(2) _maybe_schedule_shadow_compare must resolve _run_shadow_compare via
        orchestrator.merge_queue, not the co-located merge_shadow copy."""
        from orchestrator.merge_shadow import _maybe_schedule_shadow_compare

        git_ops = MagicMock()
        config = OrchestratorConfig(
            project_root=tmp_path,
            git=GitConfig(
                warm_verify_shadow_compare=True,
                warm_verify_shadow_compare_every_n_merges=1,
            ),
        )
        req = MagicMock()
        req.config = config

        worker = MagicMock()
        worker._shadow_state_path = tmp_path / 'warm_verify_shadow.json'
        worker._shadow_compare_tasks = set()

        naive = AsyncMock(side_effect=AssertionError(
            'naive merge_shadow._run_shadow_compare must not be called'
        ))
        reachback = AsyncMock(return_value=None)

        with (
            patch('orchestrator.merge_shadow._run_shadow_compare', naive),
            patch('orchestrator.merge_queue._run_shadow_compare', reachback),
        ):
            await _maybe_schedule_shadow_compare(
                worker, git_ops, req, 'commit-sha', {'pkg t': 'pass'}, None, None,
            )
            assert len(worker._shadow_compare_tasks) == 1, (
                'expected exactly one shadow-compare task to be scheduled'
            )
            task = next(iter(worker._shadow_compare_tasks))
            await task

        naive.assert_not_called()
        reachback.assert_awaited_once()

    async def test_run_cold_shadow_verify_reachback_to_pool_construction_deps(
        self, tmp_path: Path,
    ) -> None:
        """(3) _run_cold_shadow_verify must resolve build_merge_verify_spec,
        VerifyRunnerPool, LocalRunner, and run_scoped_verification via
        orchestrator.merge_queue, not the co-located merge_shadow imports.

        orchestrator.merge_queue's own bindings of build_merge_verify_spec /
        VerifyRunnerPool / LocalRunner are left unpatched (genuine, working
        objects — merge_queue.py has not been touched yet) so this exercises
        the real construction/dispatch chain; only run_scoped_verification is
        overridden (on the merge_queue side) to a fast, distinctive result so
        the test never shells out to a real verify command.
        """
        from orchestrator.merge_shadow import _run_cold_shadow_verify

        git_ops = MagicMock()
        git_ops.create_throwaway_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-cold-reachback'
        req.task_files = None
        req.module_configs = []
        req.config = OrchestratorConfig(project_root=tmp_path)

        reachback_result = VerifyResult(
            passed=True,
            test_output='PASS [0.01s] pkg reachback::marker',
            lint_output='', type_output='', summary='reachback',
        )

        with (
            patch(
                'orchestrator.merge_shadow.build_merge_verify_spec',
                MagicMock(side_effect=AssertionError('naive build_merge_verify_spec used')),
            ),
            patch(
                'orchestrator.merge_shadow.VerifyRunnerPool',
                MagicMock(side_effect=AssertionError('naive VerifyRunnerPool used')),
            ),
            patch(
                'orchestrator.merge_shadow.LocalRunner',
                MagicMock(side_effect=AssertionError('naive LocalRunner used')),
            ),
            patch(
                'orchestrator.merge_shadow.run_scoped_verification',
                AsyncMock(side_effect=AssertionError('naive run_scoped_verification used')),
            ),
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                AsyncMock(return_value=reachback_result),
            ),
        ):
            result = await _run_cold_shadow_verify(git_ops, req, 'commit-sha', None)

        assert result == {'pkg reachback::marker': 'pass'}, (
            f'expected the orchestrator.merge_queue-patched dependency chain to '
            f'govern _run_cold_shadow_verify, got {result!r}'
        )

    async def test_run_cold_shadow_verify_reachback_to_run_unscoped_typechecks(
        self, tmp_path: Path,
    ) -> None:
        """(3 cont'd) _run_cold_shadow_verify must resolve _run_unscoped_typechecks
        via orchestrator.merge_queue.  It has no merge_shadow-local copy at all
        (mirrors task β's analogous _check_post_merge_pyright ↔
        _run_unscoped_typechecks test) so only the merge_queue side is patched;
        with module_configs=[] the REAL _run_unscoped_typechecks always returns
        broken=False, so a broken=True result can only appear in the outcome if
        the merge_queue-patched mock actually governed.
        """
        from orchestrator.merge_shadow import _run_cold_shadow_verify

        git_ops = MagicMock()
        git_ops.create_throwaway_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-cold-unscoped-reachback'
        req.task_files = None
        req.module_configs = []
        req.config = OrchestratorConfig(project_root=tmp_path)

        scoped_result = VerifyResult(
            passed=True, test_output='PASS [0.01s] pkg t',
            lint_output='', type_output='', summary='ok',
        )
        broken_gate = MagicMock(
            broken=True, failing_subprojects=['pkg'], timed_out_subprojects=[],
            detail='mocked-detail',
        )

        with (
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                AsyncMock(return_value=scoped_result),
            ),
            patch(
                'orchestrator.merge_queue._run_unscoped_typechecks',
                AsyncMock(return_value=broken_gate),
            ),
        ):
            result = await _run_cold_shadow_verify(git_ops, req, 'commit-sha', None)

        assert result == {}, (
            f'expected the orchestrator.merge_queue-patched (broken) '
            f'_run_unscoped_typechecks result to short-circuit the scoped '
            f'test_output to empty, got {result!r}'
        )
