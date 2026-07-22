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


# ---------------------------------------------------------------------------
# FIX 2 (task 2886, PRD leaf δ §3.4): off-lane sampling of MAP-LESS lands.
#
# _maybe_schedule_shadow_compare early-returned on empty warm_results, so the
# two populations that CAN actually diverge were NEVER sampled:
#   * trivial-pass lands (ran no suite → empty per-test map), and
#   * remote-verdict lands (verify ran off the warm local _merge-verify lane).
# The empty-warm_results early-return is removed: when cadence is due, a
# map-less land is routed to a COARSE suite-level compare instead of being
# silently skipped.  A land WITH a per-test map still uses the per-test
# _run_shadow_compare (unchanged).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScheduleShadowCompareMapLess:
    """FIX 2 scheduling: map-less (empty warm_results) lands are still sampled."""

    async def test_empty_warm_results_schedules_coarse_compare(
        self, tmp_path: Path,
    ) -> None:
        """Empty warm_results + due cadence ⇒ a COARSE compare task is scheduled.

        RED today: ``_maybe_schedule_shadow_compare`` early-returns on empty
        ``warm_results`` (merge_shadow.py ~1162-1163), so no task is ever
        scheduled for a map-less land and ``worker._shadow_compare_tasks`` stays
        empty.  The coarse coroutine must be dispatched via the merge_queue
        reach-back seam (``orchestrator.merge_queue._run_coarse_shadow_compare``),
        NOT the per-test ``_run_shadow_compare``.
        """
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

        coarse = AsyncMock(return_value=None)
        # The per-test compare MUST NOT be used for a map-less land.
        per_test = AsyncMock(side_effect=AssertionError(
            'per-test _run_shadow_compare must not be scheduled for a map-less land'
        ))

        with (
            # create=True: the reach-back target does not exist until the impl
            # step adds the coroutine + shim re-export; this keeps the RED
            # failure the scheduling assertion, not a patch-time AttributeError.
            patch('orchestrator.merge_queue._run_coarse_shadow_compare', coarse,
                  create=True),
            patch('orchestrator.merge_queue._run_shadow_compare', per_test),
        ):
            await _maybe_schedule_shadow_compare(
                worker, git_ops, req, 'commit-sha', {}, None, None,
            )
            assert len(worker._shadow_compare_tasks) == 1, (
                'expected a coarse shadow-compare task to be scheduled for a '
                'map-less (empty warm_results) land'
            )
            task = next(iter(worker._shadow_compare_tasks))
            await task

        per_test.assert_not_called()
        coarse.assert_awaited_once()

    async def test_empty_warm_results_persists_cadence_when_not_due(
        self, tmp_path: Path,
    ) -> None:
        """A map-less land that is NOT yet due still increments persisted cadence.

        With ``every_n_merges=3`` the first map-less land is under cadence: no
        task is scheduled, but the persisted counter must advance to 1 so the
        map-less population contributes to the cadence exactly like a
        map-bearing land (no silent cadence loss).
        """
        from orchestrator.merge_shadow import (
            _load_shadow_compare_state,
            _maybe_schedule_shadow_compare,
        )

        git_ops = MagicMock()
        config = OrchestratorConfig(
            project_root=tmp_path,
            git=GitConfig(
                warm_verify_shadow_compare=True,
                warm_verify_shadow_compare_every_n_merges=3,
                # Isolate the count leg: push the nightly-timer leg far into the
                # future so it never fires (last_shadow_run_at starts at 0.0, so
                # a default 86400 s interval would trivially be "due" against a
                # 2026 wall clock and mask the count-leg cadence under test).
                warm_verify_shadow_compare_nightly_interval_secs=1e12,
            ),
        )
        req = MagicMock()
        req.config = config

        worker = MagicMock()
        worker._shadow_state_path = tmp_path / 'warm_verify_shadow.json'
        worker._shadow_compare_tasks = set()

        coarse = AsyncMock(return_value=None)
        with patch('orchestrator.merge_queue._run_coarse_shadow_compare', coarse,
                   create=True):
            await _maybe_schedule_shadow_compare(
                worker, git_ops, req, 'commit-sha', {}, None, None,
            )

        assert len(worker._shadow_compare_tasks) == 0, (
            'under cadence (1 of 3), no compare task should be scheduled'
        )
        coarse.assert_not_awaited()
        state = _load_shadow_compare_state(worker._shadow_state_path)
        assert state.merges_since_last_shadow == 1, (
            'a map-less land must still advance the persisted cadence counter'
        )


@pytest.mark.asyncio
class TestCoarseShadowCompare:
    """FIX 2: the COARSE suite-level compare coroutine (task 2886, step-9/10).

    A map-less warm-passed land (trivial-pass / remote-verdict) is compared
    against a cold FULL-gate verify.  The warm land implicitly PASSED (it
    landed), so:

    * cold FULL-gate FAIL  ⇒ born-at-L2 suspected-red alarm,
    * cold FULL-gate PASS  ⇒ ``verdict_parity_ok`` (coarse=True),
    * empty/unparseable cold ⇒ inconclusive no-op (build/compile/infra hiccup —
      mirrors ``_run_shadow_compare``'s inconclusive guard).

    The cold FULL-gate verify is exercised through the merge_queue reach-back
    seam (``orchestrator.merge_queue._run_cold_shadow_verify_suite``) so a test
    patch controls the suite-level :class:`VerifyResult` without shelling out.
    """

    @staticmethod
    def _req() -> MagicMock:
        req = MagicMock()
        req.task_id = 'task-coarse-shadow'
        req.task_files = None
        req.module_configs = []
        return req

    @staticmethod
    def _esc_queue() -> MagicMock:
        eq = MagicMock()
        eq.has_open_l1.return_value = False
        eq.make_id.side_effect = lambda sentinel: f'id::{sentinel}'
        return eq

    async def test_cold_fail_vs_warm_passed_alarms_born_at_l2(self) -> None:
        """cold FULL-gate FAIL on a map-less warm-passed land ⇒ born-at-L2 alarm."""
        from orchestrator.event_store import EventType
        from orchestrator.merge_shadow import _run_coarse_shadow_compare

        git_ops = MagicMock()
        req = self._req()
        esc_queue = self._esc_queue()
        event_store = MagicMock()

        # Cold suite genuinely RAN (FAIL + PASS lines) and FAILED.
        cold_fail = VerifyResult(
            passed=False,
            test_output='FAIL [0.20s] pkg mod::test_x\nPASS [0.01s] pkg mod::test_y',
            lint_output='', type_output='', summary='cold full-gate suite failed',
        )
        with patch(
            'orchestrator.merge_queue._run_cold_shadow_verify_suite',
            AsyncMock(return_value=cold_fail), create=True,
        ):
            await _run_coarse_shadow_compare(
                git_ops, req, 'deadbeefcafef00d', esc_queue, event_store,
            )

        esc_queue.submit.assert_called_once()
        esc = esc_queue.submit.call_args.args[0]
        assert esc.level == 2, 'coarse divergence must be born at L2'
        assert esc.severity == 'critical'
        assert esc.agent_role.startswith('orchestrator-'), (
            'agent_role must carry the orchestrator- harness-sentinel prefix'
        )
        blob = f'{esc.summary}\n{esc.detail}'
        assert 'deadbeef' in blob, 'escalation must name the merge commit'
        assert 'landed' in blob.lower(), (
            'detail must state the warm merge ALREADY LANDED'
        )
        assert 'main' in blob.lower(), 'detail must state main may be red'
        # A divergence alarm must NOT also emit a parity-ok event.
        for call in event_store.emit.call_args_list:
            assert call.args[0] != EventType.verdict_parity_ok

    async def test_cold_pass_emits_coarse_parity_ok_no_alarm(self) -> None:
        """cold FULL-gate PASS ⇒ verdict_parity_ok(coarse=True), no escalation."""
        from orchestrator.event_store import EventType
        from orchestrator.merge_shadow import _run_coarse_shadow_compare

        git_ops = MagicMock()
        req = self._req()
        esc_queue = self._esc_queue()
        event_store = MagicMock()

        cold_pass = VerifyResult(
            passed=True,
            test_output='PASS [0.01s] pkg mod::test_x\nPASS [0.02s] pkg mod::test_y',
            lint_output='', type_output='', summary='cold full-gate suite passed',
        )
        with patch(
            'orchestrator.merge_queue._run_cold_shadow_verify_suite',
            AsyncMock(return_value=cold_pass), create=True,
        ):
            await _run_coarse_shadow_compare(
                git_ops, req, 'cafef00ddeadbeef', esc_queue, event_store,
            )

        esc_queue.submit.assert_not_called()
        event_store.emit.assert_called_once()
        assert event_store.emit.call_args.args[0] == EventType.verdict_parity_ok
        data = event_store.emit.call_args.kwargs['data']
        assert data.get('coarse') is True, (
            'the coarse parity-ok event must carry a coarse=True flag'
        )

    async def test_empty_cold_is_inconclusive_no_alarm_no_event(self) -> None:
        """Empty/unparseable cold (build/compile/infra hiccup) ⇒ inconclusive no-op."""
        from orchestrator.merge_shadow import _run_coarse_shadow_compare

        git_ops = MagicMock()
        req = self._req()
        esc_queue = self._esc_queue()
        event_store = MagicMock()

        # Full-gate FAILED to run any test (compile error) → no parseable map.
        cold_broken = VerifyResult(
            passed=False,
            test_output='error[E0433]: failed to resolve: use of undeclared crate\n',
            lint_output='', type_output='', summary='cold build failed',
        )
        with patch(
            'orchestrator.merge_queue._run_cold_shadow_verify_suite',
            AsyncMock(return_value=cold_broken), create=True,
        ):
            await _run_coarse_shadow_compare(
                git_ops, req, 'facefeed12345678', esc_queue, event_store,
            )

        esc_queue.submit.assert_not_called()
        event_store.emit.assert_not_called()

    async def test_cold_leg_exception_is_swallowed(self) -> None:
        """A cold-leg exception is swallowed (WARNING) — never crashes the worker."""
        from orchestrator.merge_shadow import _run_coarse_shadow_compare

        git_ops = MagicMock()
        req = self._req()
        esc_queue = self._esc_queue()
        event_store = MagicMock()

        with patch(
            'orchestrator.merge_queue._run_cold_shadow_verify_suite',
            AsyncMock(side_effect=RuntimeError('cold worktree exploded')),
            create=True,
        ):
            # Must NOT raise.
            await _run_coarse_shadow_compare(
                git_ops, req, 'c0ffee00c0ffee00', esc_queue, event_store,
            )

        esc_queue.submit.assert_not_called()
        event_store.emit.assert_not_called()


def test_merge_queue_reexports_identical_objects() -> None:
    """merge_queue re-exports the SAME objects from merge_shadow (shim identity).

    Covers every one of the 20 originally-moved names, plus the two later
    co-located-and-re-exported pure helpers (``build_fail_fast_map`` /
    ``did_not_pass_subset``, PRD verify-retry-failed-only D2) so the identity
    guard extends to them — a future refactor cannot silently break their
    re-export without this test going red.

    RED (pre-shim): merge_queue.py still defines its own independent copies
    of these names (the duplicate definitions left in place by the EXPAND
    step), so ``getattr(merge_queue, name) is getattr(merge_shadow, name)``
    fails for every name — two distinct objects that merely share a name.
    (The two D2 helpers were born re-exported, never duplicated, so the RED
    narrative applies only to the original 20; they are guarded here purely
    forward-looking.)
    """
    import orchestrator.merge_queue as merge_queue
    import orchestrator.merge_shadow as merge_shadow

    moved_names = [
        'ShadowCompareState',
        'ShadowCompareDiff',
        'parse_per_test_results',
        '_classify_test_status',
        '_nextest_reported_test_count',
        '_NEXTEST_TEST_LINE_RE',
        '_LIBTEST_TEST_LINE_RE',
        '_NEXTEST_SUMMARY_LINE_RE',
        'diff_per_test_results',
        '_persistent_alarm_tests',
        '_load_shadow_compare_state',
        '_save_shadow_compare_state',
        '_shadow_compare_due',
        '_WARM_COLD_SHADOW_SENTINEL',
        '_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL',
        '_submit_shadow_divergence_escalation',
        '_alarm_warm_shadow_unparseable',
        '_run_cold_shadow_verify',
        '_run_shadow_compare',
        '_maybe_schedule_shadow_compare',
        # PRD verify-retry-failed-only D2: born re-exported through the same
        # shim (not part of the original EXPAND-then-shim move), guarded here
        # so their re-export identity cannot silently regress.
        'build_fail_fast_map',
        'did_not_pass_subset',
    ]

    for name in moved_names:
        mq_obj = getattr(merge_queue, name)
        ms_obj = getattr(merge_shadow, name)
        assert mq_obj is ms_obj, (
            f'{name}: orchestrator.merge_queue.{name} and '
            f'orchestrator.merge_shadow.{name} must be the identical object'
        )


# ---------------------------------------------------------------------------
# {did-not-pass} retry-subset construction (PRD verify-retry-failed-only D2).
#
# The core user-observable signal of the failed-only merge-verify retry: under
# nextest fail-fast a failing attempt-0 CANCELS the not-yet-started tests, which
# are ABSENT from parse_per_test_results output.  The sound retry subset is
# therefore {did-not-pass} = failed ∪ not-started ∪ inconclusive, NOT {failed}.
# ---------------------------------------------------------------------------


def test_did_not_pass_subset_includes_non_pass_verdicts() -> None:
    """did_not_pass_subset selects every non-'pass' verdict, sorted.

    The subset MUST include not-started (fail-fast-cancelled) and inconclusive
    tests, not only the 'fail' entry — a failed-only filter would be unsound
    under nextest fail-fast.
    """
    from orchestrator.merge_shadow import did_not_pass_subset

    fail_fast_map = {
        'crate a::x': 'pass',
        'crate a::y': 'fail',
        'crate a::z': 'not-started',
        'crate a::w': 'inconclusive',
    }
    # sorted; failed ∪ not-started ∪ inconclusive — NOT just the 'fail' entry.
    assert did_not_pass_subset(fail_fast_map) == [
        'crate a::w',
        'crate a::y',
        'crate a::z',
    ]


def test_did_not_pass_subset_all_pass_is_empty() -> None:
    """An all-'pass' fail-fast map yields the empty subset (nothing to retry)."""
    from orchestrator.merge_shadow import did_not_pass_subset

    assert did_not_pass_subset({'a::x': 'pass', 'a::y': 'pass'}) == []


def test_build_fail_fast_map_marks_cancelled_tests_not_started() -> None:
    """build_fail_fast_map annotates the authoritative plan with attempt-0 verdicts.

    A test present in ``planned`` but absent from ``verdicts`` was cancelled by
    nextest fail-fast and is annotated ``'not-started'`` — the crux of the
    soundness core.  ``planned`` is authoritative: a verdict key not in
    ``planned`` is ignored.
    """
    from orchestrator.merge_shadow import build_fail_fast_map

    planned = ['crate a::x', 'crate a::y', 'crate a::z']
    # As produced by parse_per_test_results on attempt-0 output: 'crate a::z'
    # was cancelled by fail-fast and is ABSENT.  'crate a::stale' is a verdict
    # for a test NOT in the plan and must be ignored.
    verdicts = {
        'crate a::x': 'pass',
        'crate a::y': 'fail',
        'crate a::stale': 'pass',
    }
    assert build_fail_fast_map(planned, verdicts) == {
        'crate a::x': 'pass',
        'crate a::y': 'fail',
        'crate a::z': 'not-started',
    }


def test_build_fail_fast_map_then_subset_retains_not_started() -> None:
    """End-to-end: the {did-not-pass} subset retains the fail-fast-cancelled test.

    A raw failed-only view would drop 'crate a::z'; feeding the fail-fast map
    through did_not_pass_subset keeps it (soundness).
    """
    from orchestrator.merge_shadow import build_fail_fast_map, did_not_pass_subset

    planned = ['crate a::x', 'crate a::y', 'crate a::z']
    verdicts = {'crate a::x': 'pass', 'crate a::y': 'fail'}
    assert did_not_pass_subset(build_fail_fast_map(planned, verdicts)) == [
        'crate a::y',
        'crate a::z',
    ]


# ---------------------------------------------------------------------------
# shadow-baseline map merge (PRD verify-retry-failed-only D4, §5.4).
#
# A narrowed {did-not-pass} merge-verify retry re-runs ONLY the tests that did
# not pass in attempt-0, so its per-test map is PARTIAL — it omits every test
# that already passed.  Storing that partial map as the warm shadow baseline
# makes the next FULL cold shadow compare classify every attempt-0-passed test
# as only_cold → phantom born-at-L2 divergence alarm.  merge_retry_shadow_baseline
# unions attempt-0's passes with the fresh retry map so the baseline is whole.
# ---------------------------------------------------------------------------


def test_merge_retry_shadow_baseline_carries_forward_and_retry_overwrites() -> None:
    """attempt-0 passes carry forward; retry wins on overlap (did-not-pass→pass).

    A,B passed in attempt-0 and are absent from the narrowed retry map — they
    must reappear in the merged baseline (else the full cold suite flags them
    only_cold → phantom divergence).  C did-not-pass in attempt-0 and the fresh
    retry pass is the latest verdict, so it overwrites C's attempt-0 'fail'.
    """
    from orchestrator.merge_shadow import merge_retry_shadow_baseline

    attempt0 = {'A': 'pass', 'B': 'pass', 'C': 'fail'}
    retry = {'C': 'pass'}
    assert merge_retry_shadow_baseline(attempt0, retry) == {
        'A': 'pass',
        'B': 'pass',
        'C': 'pass',
    }


def test_merge_retry_shadow_baseline_retry_precedence_over_inconclusive() -> None:
    """retry verdict wins on overlap even over an attempt-0 'inconclusive'."""
    from orchestrator.merge_shadow import merge_retry_shadow_baseline

    attempt0 = {'A': 'pass', 'C': 'inconclusive'}
    retry = {'C': 'pass'}
    assert merge_retry_shadow_baseline(attempt0, retry) == {'A': 'pass', 'C': 'pass'}


def test_merge_retry_shadow_baseline_drops_attempt0_non_pass_absent_from_retry() -> None:
    """attempt-0 non-pass verdicts NOT re-run by the retry are dropped, not carried.

    A stray 'fail'/'not-started' surviving into the baseline would be flipped by
    the full cold suite into a genuine-looking divergence — the exact phantom
    this helper exists to remove.  Only attempt-0 PASSES are carried forward.
    """
    from orchestrator.merge_shadow import merge_retry_shadow_baseline

    assert merge_retry_shadow_baseline({'A': 'pass', 'C': 'fail'}, {}) == {'A': 'pass'}


def test_merge_retry_shadow_baseline_does_not_mutate_inputs() -> None:
    """The input maps are treated as read-only (no in-place mutation)."""
    from orchestrator.merge_shadow import merge_retry_shadow_baseline

    attempt0 = {'A': 'pass', 'C': 'fail'}
    retry = {'C': 'pass'}
    merge_retry_shadow_baseline(attempt0, retry)
    assert attempt0 == {'A': 'pass', 'C': 'fail'}
    assert retry == {'C': 'pass'}


def test_build_warm_shadow_results_narrowed_merges_attempt0_passes() -> None:
    """NARROWED warm retry: partial retry output ∪ attempt-0 passes → whole map.

    ``test_output`` is the PARTIAL narrowed-retry output containing ONLY the
    re-run test; ``attempt0_verdicts`` supplies the attempt-0 pass that the
    narrowed run omitted.  The result must be the merged full-suite baseline.
    """
    from orchestrator.merge_shadow import build_warm_shadow_results

    test_output = 'PASS [   0.05s] reify-spec test_retried\n'
    attempt0 = {'reify-spec test_passed': 'pass'}
    assert build_warm_shadow_results(test_output, attempt0) == {
        'reify-spec test_passed': 'pass',
        'reify-spec test_retried': 'pass',
    }


def test_build_warm_shadow_results_non_narrowed_is_parse_only() -> None:
    """NON-narrowed (no attempt-0 map): byte-identical to a plain parse.

    ``attempt0_verdicts=None`` (the default) and an explicit empty ``{}`` must
    both return exactly ``parse_per_test_results(test_output)`` — the
    non-narrowed warm path stays byte-identical.
    """
    from orchestrator.merge_shadow import (
        build_warm_shadow_results,
        parse_per_test_results,
    )

    test_output = (
        'PASS [   0.05s] reify-spec test_a\n'
        'PASS [   0.06s] reify-spec test_b\n'
    )
    expected = parse_per_test_results(test_output)
    assert expected == {'reify-spec test_a': 'pass', 'reify-spec test_b': 'pass'}
    # default None
    assert build_warm_shadow_results(test_output) == expected
    # explicit empty attempt0 behaves like None (parse-only)
    assert build_warm_shadow_results(test_output, {}) == expected


def test_build_warm_shadow_results_empty_parse_returned_unchanged() -> None:
    """EMPTY/unparseable retry: return ``{}`` verbatim, NEVER merged.

    An empty retry parse must NOT be unioned with attempt-0 (that would turn an
    unparseable narrowed retry into a non-empty PARTIAL baseline, masking the
    fail-closed ``_alarm_warm_shadow_unparseable`` and feeding the shadow
    compare the exact partial map D4 exists to eliminate).
    """
    from orchestrator.merge_shadow import build_warm_shadow_results

    assert build_warm_shadow_results('', {'reify-spec test_passed': 'pass'}) == {}
