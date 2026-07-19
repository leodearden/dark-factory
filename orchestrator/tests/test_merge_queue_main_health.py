"""Tests for merge-queue main-health probe wiring (task 1690).

Steps 3-12: Classification, config-flag guard, timeout/flaky-category guard,
signal emission, and dedupe-fold/cache-reuse contract, all driven through
_run_post_merge_verify (the single chokepoint).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    TRANSIENT_INFRA_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
    _build_main_health_outcome,
    _main_health_fingerprint,
    _MainHealthProbeHandles,
    _run_post_merge_verify,
)
from orchestrator.verify import _PROBE_CACHE, VerifyResult
from orchestrator.verify_runner import (
    FLOCK_CONTENTION_CATEGORY,
    UNSCOPED_TYPECHECK_FAILED_CATEGORY,
)

MAIN_SHA = 'cafecafe1234567890deadbeef'

COMPILE_ERROR_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='error TS2322: StatusBar.tsx:12',
    summary='tsc failed',
    cause_hint='error TS2322: StatusBar.tsx',
    category='compile_error',
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_probe_cache():
    """Clear the process-wide _PROBE_CACHE between tests."""
    _PROBE_CACHE.clear()
    yield
    _PROBE_CACHE.clear()


def _make_config(
    tmp_path: Path,
    *,
    escalate_preexisting: bool = True,
    merge_verify_breadth: Literal['scoped', 'full'] = 'scoped',
) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        escalate_preexisting_main_break=escalate_preexisting,
        merge_verify_breadth=merge_verify_breadth,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _make_git_ops(tmp_path: Path) -> GitOps:
    git_ops = MagicMock(spec=GitOps)
    git_ops.project_root = tmp_path
    git_ops.cleanup_merge_worktree = AsyncMock(return_value=None)
    git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)
    return git_ops


def _make_req(
    task_id: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    future = make_placeholder_future()
    return MergeRequest(
        task_id=task_id,
        branch=f'task/{task_id}',
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        lane='normal',
    )


async def _drive_verify(
    req: MergeRequest,
    merge_wt: Path,
    git_ops: GitOps,
    *,
    event_store: EventStore | None = None,
) -> MergeOutcome | None:
    """Call _run_post_merge_verify with standard test parameters."""
    return await _run_post_merge_verify(
        git_ops, req, merge_wt,
        timeouts={},
        enospc_retries={},
        max_timeouts=3,
        max_enospc=1,
        event_store=event_store,
    )


# ---------------------------------------------------------------------------
# Regression: _make_req must work even when there is no current event loop.
# (task 1711 step-1 RED)
# ---------------------------------------------------------------------------


def test_make_req_works_with_no_current_event_loop(tmp_path: Path) -> None:
    """_make_req must not raise when the thread's current event loop is None.

    asyncio.run() calls set_event_loop(None) in its finally block, so any
    test that calls _make_req() AFTER a prior asyncio.run() finds the thread
    loop explicitly None and gets RuntimeError: There is no current event
    loop.  This regression test pins that behaviour: set_event_loop(None)
    first, then assert _make_req() completes and returns a MergeRequest whose
    .result is an asyncio.Future.
    """
    asyncio.set_event_loop(None)
    try:
        req = _make_req('99', tmp_path / 'wt', _make_config(tmp_path))
        assert isinstance(req.result, asyncio.Future)
    finally:
        # Restore None (the realistic post-asyncio.run() state) so this test
        # does not mutate loop state for other tests on the same xdist worker.
        asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# Step-1/2: _build_main_health_outcome pure helper (task 2564)
# ---------------------------------------------------------------------------


class TestBuildMainHealthOutcomeHelper:
    """Step-1 (RED): _build_main_health_outcome(verify, probe_sha) must produce
    a MergeOutcome with parity to the inline construction previously in
    _classify_main_health_red (merge_queue.py:746-762) — task 2564 extracts it
    into a pure helper so the synchronous and deferred probe paths cannot
    diverge in their reason/fingerprint/failure_category/failure_cause_hint
    composition."""

    def test_build_main_health_outcome_parity(self) -> None:
        outcome = _build_main_health_outcome(COMPILE_ERROR_RESULT, MAIN_SHA)

        assert isinstance(outcome, MergeOutcome)
        assert outcome.status == 'blocked', f'Expected blocked; got {outcome.status}'
        assert outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Expected reason to start with MAIN_HEALTH_RED_REASON_PREFIX; '
            f'got: {outcome.reason!r}'
        )
        assert outcome.failure_category == COMPILE_ERROR_RESULT.category, (
            f'Expected failure_category={COMPILE_ERROR_RESULT.category!r}; '
            f'got {outcome.failure_category!r}'
        )
        assert outcome.failure_cause_hint == COMPILE_ERROR_RESULT.cause_hint, (
            f'Expected failure_cause_hint={COMPILE_ERROR_RESULT.cause_hint!r}; '
            f'got {outcome.failure_cause_hint!r}'
        )
        expected_fp = _main_health_fingerprint(
            COMPILE_ERROR_RESULT.category or '', COMPILE_ERROR_RESULT.cause_hint, MAIN_SHA,
        )
        assert outcome.dedupe_fingerprint == expected_fp, (
            f'Expected dedupe_fingerprint={expected_fp!r}; '
            f'got {outcome.dedupe_fingerprint!r}'
        )
        assert outcome.dedupe_fingerprint, 'dedupe_fingerprint must be non-empty'


# ---------------------------------------------------------------------------
# Step-3: Positive classification — probe returns (True, MAIN_SHA)
# ---------------------------------------------------------------------------


class TestMainHealthClassification:
    """Step-3 (RED): _run_post_merge_verify with a failing verify and probe
    returning (True, MAIN_SHA) must produce a main-health-red MergeOutcome."""

    def test_positive_classification(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(True, MAIN_SHA)),
                ),
            ):
                return await _drive_verify(req, merge_wt, git_ops)

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert outcome.status == 'blocked', f'Expected blocked; got {outcome.status}'
        assert outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Expected reason to start with MAIN_HEALTH_RED_REASON_PREFIX; '
            f'got: {outcome.reason!r}'
        )
        assert outcome.failure_category == 'compile_error', (
            f'Expected failure_category=compile_error; got {outcome.failure_category!r}'
        )
        assert outcome.failure_cause_hint == 'error TS2322: StatusBar.tsx', (
            f'Expected failure_cause_hint set; got {outcome.failure_cause_hint!r}'
        )
        from orchestrator.workflow import compute_preexisting_main_break_fingerprint
        expected_fp = compute_preexisting_main_break_fingerprint(
            'compile_error', 'error TS2322: StatusBar.tsx', MAIN_SHA,
        )
        assert outcome.dedupe_fingerprint == expected_fp, (
            f'Expected dedupe_fingerprint={expected_fp!r}; '
            f'got {outcome.dedupe_fingerprint!r}'
        )
        assert outcome.dedupe_fingerprint, 'dedupe_fingerprint must be non-empty'

    def test_negative_classification_falls_through(self, tmp_path: Path) -> None:
        """When probe returns (False, ''), outcome is the normal task-fault."""
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
            ):
                return await _drive_verify(req, merge_wt, git_ops)

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert not outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Task-fault outcome must NOT start with MAIN_HEALTH_RED_REASON_PREFIX; '
            f'got: {outcome.reason!r}'
        )
        assert outcome.reason.startswith('Post-merge verification failed'), (
            f'Expected normal failure reason; got {outcome.reason!r}'
        )
        assert outcome.dedupe_fingerprint == '', (
            f'Task-fault outcome must have empty dedupe_fingerprint; '
            f'got {outcome.dedupe_fingerprint!r}'
        )


# ---------------------------------------------------------------------------
# Step-5: Config-flag guard — escalate_preexisting_main_break=False
# ---------------------------------------------------------------------------


class TestConfigFlagGuard:
    """Step-5 (RED): when escalate_preexisting_main_break=False, the probe
    must NOT be called and the outcome is the normal task-fault outcome."""

    def test_flag_off_skips_probe(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, escalate_preexisting=False)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        probe_spy = AsyncMock(return_value=(True, MAIN_SHA))

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=probe_spy,
                ),
            ):
                return await _drive_verify(req, merge_wt, git_ops)

        outcome = asyncio.run(_run())

        assert probe_spy.call_count == 0, (
            f'Probe must NOT be called when flag is off; call_count={probe_spy.call_count}'
        )
        assert outcome is not None
        assert not outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Expected normal task-fault outcome; got: {outcome.reason!r}'
        )
        assert outcome.reason.startswith('Post-merge verification failed'), (
            f'Expected normal failure reason; got {outcome.reason!r}'
        )
        assert outcome.dedupe_fingerprint == '', (
            f'Expected empty dedupe_fingerprint; got {outcome.dedupe_fingerprint!r}'
        )


# ---------------------------------------------------------------------------
# Step-7: Timeout + flaky-category skip guards
# ---------------------------------------------------------------------------


INFRA_TIMEOUT_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='verify timed out',
    category='infra_timeout',
    timed_out=True,
)

FLOCK_ERROR_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='flock error',
    category='flock_error',
    timed_out=False,
)


class TestSkipGuards:
    """Step-7 (RED): probe must NOT be called for timed_out=True or categories
    in PREEXISTING_BREAK_SKIP_CATEGORIES."""

    @pytest.mark.parametrize('verify_result,label', [
        (INFRA_TIMEOUT_RESULT, 'timed_out'),
        pytest.param(
            VerifyResult(
                passed=False, test_output='', lint_output='', type_output='',
                summary='infra timeout category', category='infra_timeout',
            ),
            'infra_timeout_category',
        ),
        pytest.param(
            VerifyResult(
                passed=False, test_output='', lint_output='', type_output='',
                summary='flock error category', category='flock_error',
            ),
            'flock_error_category',
        ),
    ])
    def test_skip_guards(
        self, tmp_path: Path, verify_result: VerifyResult, label: str,
    ) -> None:
        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        probe_spy = AsyncMock(return_value=(True, MAIN_SHA))

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=verify_result),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=probe_spy,
                ),
            ):
                return await _drive_verify(req, merge_wt, git_ops)

        outcome = asyncio.run(_run())

        assert probe_spy.call_count == 0, (
            f'[{label}] Probe must NOT be called; call_count={probe_spy.call_count}'
        )
        assert outcome is not None
        assert not outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'[{label}] Expected normal task-fault outcome; got {outcome.reason!r}'
        )
        assert outcome.dedupe_fingerprint == '', (
            f'[{label}] Expected empty dedupe_fingerprint; got {outcome.dedupe_fingerprint!r}'
        )


# ---------------------------------------------------------------------------
# Step-9: Structured main_health_red signal via EventStore
# ---------------------------------------------------------------------------


class TestMainHealthSignalEmission:
    """Step-9 (RED): on main-health-red, _run_post_merge_verify must emit
    exactly one merge_attempt event with outcome='main_health_red'.
    On task-fault (probe returns False), no main_health_red event is emitted."""

    def test_signal_emitted_on_main_health_red(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        event_store = MagicMock(spec=EventStore)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(True, MAIN_SHA)),
                ),
            ):
                return await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                    event_store=event_store,
                )

        asyncio.run(_run())

        # Must have at least one call to emit() with data['outcome']='main_health_red'
        calls = event_store.emit.call_args_list
        main_health_calls = [
            c for c in calls
            if c.kwargs.get('data', {}).get('outcome') == 'main_health_red'
        ]
        assert len(main_health_calls) >= 1, (
            f'Expected at least one main_health_red event; '
            f'event_store.emit calls: {calls}'
        )

    def test_no_signal_on_task_fault(self, tmp_path: Path) -> None:
        """When probe returns (False, ''), no main_health_red event is emitted."""
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        event_store = MagicMock(spec=EventStore)

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
            ):
                return await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                    event_store=event_store,
                )

        asyncio.run(_run())

        calls = event_store.emit.call_args_list
        main_health_calls = [
            c for c in calls
            if c.kwargs.get('data', {}).get('outcome') == 'main_health_red'
        ]
        assert len(main_health_calls) == 0, (
            f'No main_health_red event must be emitted for task-fault; '
            f'calls: {main_health_calls}'
        )


# ---------------------------------------------------------------------------
# Step-11: Dedup fold + verdict-cache reuse across concurrent failing merges
# ---------------------------------------------------------------------------


class TestDedupeFingerprrintAndCacheReuse:
    """Step-11 (RED): two failing merges with the same signature against the
    same main SHA must produce equal dedupe_fingerprints (fold to one parent)
    and the probe is served from _PROBE_CACHE on the second call (no double-probe)."""

    def test_concurrent_merges_same_signature_fold_and_cache(
        self, tmp_path: Path,
    ) -> None:
        import time

        from orchestrator.verify import _PROBE_CACHE
        from orchestrator.workflow import (
            _normalize_cause_hint as _norm,
        )
        from orchestrator.workflow import (
            compute_preexisting_main_break_fingerprint,
        )

        _PROBE_CACHE.clear()

        # Pre-seed the cache so the real verify_failure_is_preexisting_on_main
        # hits the cache immediately (no git subprocess / worktree creation).
        # This exercises the production code path that concurrent failing merges
        # follow when the first probe already populated the cache.
        _norm_hint = _norm(COMPILE_ERROR_RESULT.cause_hint)
        _cache_key = (MAIN_SHA, COMPILE_ERROR_RESULT.category or '', _norm_hint)
        _PROBE_CACHE[_cache_key] = (time.monotonic(), True)

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        # Two separate task worktrees / merge_wts simulating concurrent tasks
        wt_a = tmp_path / 'task-a'
        wt_b = tmp_path / 'task-b'
        mwt_a = tmp_path / 'mwt-a'
        mwt_b = tmp_path / 'mwt-b'
        for d in (wt_a, wt_b, mwt_a, mwt_b):
            d.mkdir()

        req_a = _make_req('101', wt_a, config)
        req_b = _make_req('102', wt_b, config)

        async def _run() -> tuple[MergeOutcome | None, MergeOutcome | None]:
            # Use the REAL verify_failure_is_preexisting_on_main — it will
            # call git_ops.get_main_sha() then hit _PROBE_CACHE immediately,
            # so no subprocess or worktree is created.
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
            ):
                oc_a = await _run_post_merge_verify(
                    git_ops, req_a, mwt_a,
                    timeouts={}, enospc_retries={}, max_timeouts=3, max_enospc=1,
                )
                oc_b = await _run_post_merge_verify(
                    git_ops, req_b, mwt_b,
                    timeouts={}, enospc_retries={}, max_timeouts=3, max_enospc=1,
                )
                return oc_a, oc_b

        oc_a, oc_b = asyncio.run(_run())

        assert oc_a is not None, 'Task-A outcome must be non-None'
        assert oc_b is not None, 'Task-B outcome must be non-None'

        # (a) Both fingerprints equal and non-empty → two escalations fold to one parent
        assert oc_a.dedupe_fingerprint == oc_b.dedupe_fingerprint, (
            f'dedupe_fingerprints must be equal for same signature; '
            f'a={oc_a.dedupe_fingerprint!r}, b={oc_b.dedupe_fingerprint!r}'
        )
        assert oc_a.dedupe_fingerprint, 'dedupe_fingerprint must be non-empty'

        # (b) Both are main_health_red outcomes (not task-fault)
        assert oc_a.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Task-A must be main-health-red; got {oc_a.reason!r}'
        )
        assert oc_b.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Task-B must be main-health-red; got {oc_b.reason!r}'
        )

        # (c) Fingerprint equals compute_preexisting_main_break_fingerprint
        # using probe-returned main_sha (guarantees cross-path fold with task-verify)
        expected_fp = compute_preexisting_main_break_fingerprint(
            COMPILE_ERROR_RESULT.category or '',
            COMPILE_ERROR_RESULT.cause_hint,
            MAIN_SHA,
        )
        assert oc_a.dedupe_fingerprint == expected_fp, (
            f'Fingerprint must match compute_preexisting_main_break_fingerprint; '
            f'actual={oc_a.dedupe_fingerprint!r}, expected={expected_fp!r}'
        )

        # (d) Cache-hit contract: get_main_sha was called once per merge to
        # build the cache key, but no new probe entry was written (cache had
        # exactly one entry before and after, confirming no worktree was created).
        _get_main_sha_mock = cast(AsyncMock, git_ops.get_main_sha)
        assert _get_main_sha_mock.call_count == 2, (
            f'get_main_sha must be called once per merge (cache-key lookup); '
            f'call_count={_get_main_sha_mock.call_count}'
        )
        assert len(_PROBE_CACHE) == 1, (
            f'_PROBE_CACHE must still have exactly one entry — no second probe was run; '
            f'len={len(_PROBE_CACHE)}'
        )


# ---------------------------------------------------------------------------
# Fail-safe branches in _classify_main_health_red (suggestion 3)
# ---------------------------------------------------------------------------


class TestClassifyFailSafeBranches:
    """Cover the two silent fail-safe paths in _classify_main_health_red:
    (1) probe raises → falls through to normal task-fault outcome
    (2) probe confirms preexisting but fingerprint helper returns '' →
        outcome is still main_health_red with empty dedupe_fingerprint.
    """

    def test_probe_exception_falls_through_to_task_fault(
        self, tmp_path: Path,
    ) -> None:
        """When verify_failure_is_preexisting_on_main raises, the exception is
        swallowed and the outcome is the normal 'Post-merge verification failed'
        task-fault outcome (not a crash, not main_health_red)."""
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        probe_spy = AsyncMock(side_effect=RuntimeError('probe crashed'))

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=probe_spy,
                ),
            ):
                return await _drive_verify(req, merge_wt, git_ops)

        outcome = asyncio.run(_run())

        assert probe_spy.call_count == 1, 'Probe must have been called once'
        assert outcome is not None
        assert outcome.reason.startswith('Post-merge verification failed'), (
            f'Exception in probe must fall through to task-fault outcome; '
            f'got: {outcome.reason!r}'
        )
        assert not outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            'Task-fault reason must not start with MAIN_HEALTH_RED_REASON_PREFIX'
        )
        assert outcome.dedupe_fingerprint == '', (
            f'Task-fault outcome must have empty dedupe_fingerprint; '
            f'got {outcome.dedupe_fingerprint!r}'
        )

    def test_empty_fingerprint_still_routes_main_health_red(
        self, tmp_path: Path,
    ) -> None:
        """When the probe confirms preexisting (True, sha) but
        _main_health_fingerprint returns '' due to a composition error,
        the outcome is still main_health_red with dedupe_fingerprint=''
        (rather than silently downgrading to task-fault)."""
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(True, MAIN_SHA)),
                ),
                patch(
                    'orchestrator.merge_queue._main_health_fingerprint',
                    return_value='',
                ),
            ):
                return await _drive_verify(req, merge_wt, git_ops)

        outcome = asyncio.run(_run())

        assert outcome is not None
        assert outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Empty fingerprint must NOT downgrade to task-fault; '
            f'got: {outcome.reason!r}'
        )
        assert outcome.dedupe_fingerprint == '', (
            f'dedupe_fingerprint must be empty (fingerprint helper failed); '
            f'got {outcome.dedupe_fingerprint!r}'
        )


# ---------------------------------------------------------------------------
# Step-11: _main_health_fingerprint emits WARNING on composition failure
# (task 1809 step-11 RED)
# ---------------------------------------------------------------------------


class TestMainHealthFingerprintWarning:
    """Step-11 (RED): when compute_preexisting_main_break_fingerprint raises,
    _main_health_fingerprint must return '' AND emit a WARNING at
    'orchestrator.merge_queue'.

    Before step-12 impl: the except block is silent (no WARNING) → RED.
    After step-12 impl: WARNING is emitted.
    """

    def test_fingerprint_failure_emits_warning(self, caplog) -> None:
        import logging

        from orchestrator.merge_queue import _main_health_fingerprint

        with (
            patch(
                'orchestrator.workflow.compute_preexisting_main_break_fingerprint',
                side_effect=RuntimeError('fingerprint composition exploded'),
            ),
            caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'),
        ):
            result = _main_health_fingerprint('compile_error', 'some hint', 'abc123')

        assert result == '', (
            f'Expected empty string on composition failure; got {result!r}'
        )

        warning_texts = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert warning_texts, (
            'Expected a WARNING at orchestrator.merge_queue; got no warnings'
        )
        assert any(
            'fingerprint' in t.lower() for t in warning_texts
        ), f'Expected WARNING to mention fingerprint; got: {warning_texts}'


# ---------------------------------------------------------------------------
# Task 2564 step-3: main_health_probe_handles deferral core.
#
# _run_post_merge_verify must return PROMPTLY with the provisional
# task-fault outcome (not wait on the main-health probe) when
# main_health_probe_handles is supplied, and register a live detached probe
# task in handles.background_tasks — the fix for the reify 5067 merge-slot
# stall (a slow synchronous probe holding verify_task, hence the merge slot
# / host lease, for its full run).
# ---------------------------------------------------------------------------


class TestMainHealthDeferralCore:
    """Step-3 (RED): main_health_probe_handles gates deferred vs synchronous
    probing.  A never-resolving probe must not block _run_post_merge_verify's
    return when handles are supplied."""

    def test_deferred_returns_promptly_with_provisional_outcome(
        self, tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        blocker = asyncio.Event()  # never set — the probe must not be awaited

        async def _blocked_probe(*_args: object, **_kwargs: object) -> tuple[bool, str]:
            await blocker.wait()
            return (True, MAIN_SHA)  # pragma: no cover - never reached in this test

        handles = _MainHealthProbeHandles(background_tasks=set())

        async def _run() -> tuple[MergeOutcome | None, list[asyncio.Task]]:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(side_effect=_blocked_probe),
                ),
            ):
                outcome = await asyncio.wait_for(
                    _run_post_merge_verify(
                        git_ops, req, merge_wt,
                        timeouts={},
                        enospc_retries={},
                        max_timeouts=3,
                        max_enospc=1,
                        main_health_probe_handles=handles,
                    ),
                    timeout=5,
                )
            # Snapshot background_tasks state WHILE the loop is still alive —
            # asyncio.run()'s post-return cleanup cancels + discards any
            # still-pending task, so checking after it returns would always
            # observe an empty set regardless of whether the spawn happened.
            live_probe_tasks = [
                t for t in handles.background_tasks
                if t.get_name() == 'main-health-probe-99' and not t.done()
            ]
            # Release + cancel the detached probe task so it doesn't leak
            # across tests or emit an "exception never retrieved" warning.
            for t in handles.background_tasks:
                t.cancel()
            blocker.set()
            return outcome, live_probe_tasks

        outcome, live_probe_tasks = asyncio.run(_run())

        assert outcome is not None
        assert outcome.reason.startswith('Post-merge verification failed'), (
            f'Expected the prompt provisional task-fault outcome; '
            f'got: {outcome.reason!r}'
        )
        assert not outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Deferred mode must never return the main-health-red outcome '
            f'inline (the probe has not resolved yet); got: {outcome.reason!r}'
        )
        assert outcome.failure_category == 'compile_error', (
            f'Expected failure_category=compile_error; got {outcome.failure_category!r}'
        )
        assert outcome.failure_cause_hint == 'error TS2322: StatusBar.tsx', (
            f'Expected failure_cause_hint set; got {outcome.failure_cause_hint!r}'
        )

        assert len(live_probe_tasks) == 1, (
            f'Expected exactly one live main-health-probe task registered in '
            f'handles.background_tasks; got {live_probe_tasks}'
        )


# ---------------------------------------------------------------------------
# Step-7: sync-path byte-identity + sentinel-branch preservation with
# main_health_probe_handles supplied.
# ---------------------------------------------------------------------------


# Mirror test_merge_queue_dry_run_unblock.py's sentinel VerifyResult fixtures
# (cannot import them directly — that module imports FROM this one, so a
# reverse import would be circular).
FLOCK_CONTENTION_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='flock contention',
    category=FLOCK_CONTENTION_CATEGORY,
    contention={'host': 'laptop1', 'holder_pgid': 123, 'waiter_pgid': 456},
)

UNSCOPED_TYPECHECK_FAILED_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='error TS2322: some type error',
    summary='frontend',
    category=UNSCOPED_TYPECHECK_FAILED_CATEGORY,
)

PERSISTENT_ENOSPC_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='no space left on device',
    category='',
)


class TestSyncPathAndSentinelPreservation:
    """Step-7 (RED if branch ordering regresses): (a) the SYNCHRONOUS path
    (main_health_probe_handles=None) still returns the main-health-red
    outcome INLINE — proving train/solo/gate paths stay byte-identical; (b)
    the flock-contention, unscoped-gate, and persistent-ENOSPC sentinel
    branches return their existing outcome AND register NO main-health probe
    even when handles ARE provided, because they return before the
    main-health branch is ever reached."""

    def test_sync_path_returns_main_red_inline(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(True, MAIN_SHA)),
                ),
            ):
                return await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                    main_health_probe_handles=None,
                )

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Expected the main-health-red outcome INLINE when handles=None '
            f'(byte-identical sync path); got: {outcome.reason!r}'
        )

    @pytest.mark.parametrize('verify_result,expected_prefix,label', [
        (
            FLOCK_CONTENTION_RESULT,
            'Post-merge verification blocked:',
            'flock_contention',
        ),
        (
            UNSCOPED_TYPECHECK_FAILED_RESULT,
            'Post-merge verification failed: unscoped type-check failed',
            'unscoped_gate',
        ),
        (
            PERSISTENT_ENOSPC_RESULT,
            TRANSIENT_INFRA_REASON_PREFIX,
            'persistent_enospc',
        ),
    ])
    def test_sentinel_branches_preserved_with_handles_provided(
        self,
        tmp_path: Path,
        verify_result: VerifyResult,
        expected_prefix: str,
        label: str,
    ) -> None:
        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        handles = _MainHealthProbeHandles(background_tasks=set())
        probe_spy = AsyncMock(return_value=(True, MAIN_SHA))

        async def _run() -> tuple[MergeOutcome | None, set[asyncio.Task]]:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=verify_result),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=probe_spy,
                ),
            ):
                outcome = await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                    main_health_probe_handles=handles,
                )
            # Snapshot while the loop is alive (see TestMainHealthDeferralCore
            # above) — irrelevant here since nothing should ever be spawned,
            # but kept for parity/defensiveness.
            snapshot = set(handles.background_tasks)
            for t in snapshot:
                t.cancel()
            return outcome, snapshot

        outcome, spawned = asyncio.run(_run())

        assert outcome is not None
        assert outcome.reason.startswith(expected_prefix), (
            f'[{label}] Expected sentinel outcome reason to start with '
            f'{expected_prefix!r}; got {outcome.reason!r}'
        )
        assert probe_spy.call_count == 0, (
            f'[{label}] Main-health probe must NOT be called for a sentinel '
            f'outcome; call_count={probe_spy.call_count}'
        )
        assert spawned == set(), (
            f'[{label}] Expected no main-health probe task to be spawned '
            f'even though handles were provided; got {spawned}'
        )


# ---------------------------------------------------------------------------
# Step-17 (task μ, verify-scope-inversion-prd.md): _run_post_merge_verify
# seeds the per-main-SHA failing-test-id baseline for free on the PASS path
# (B2) — see verify.seed_main_baseline / verify._BASELINE_FAILING_IDS_CACHE.
# ---------------------------------------------------------------------------


class TestPostMergeVerifySeedsBaseline:
    """A successful merge+full gate run seeds
    verify._BASELINE_FAILING_IDS_CACHE[merge_sha] for free, so the very next
    gate's baseline lookup (main_baseline_failing_ids) is a cache hit and
    never pays for a probe. Seeding must NOT happen when there is no junit-
    derived id set (scoped/degraded) or no merge_sha to key on."""

    def test_passing_merge_full_verify_seeds_baseline(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, merge_verify_breadth='full')
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        passing_result = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='',
            summary='all checks passed', failing_test_ids=[],
        )

        async def _run() -> MergeOutcome | None:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                new=AsyncMock(return_value=passing_result),
            ):
                return await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                    merge_sha='abcsha',
                )

        outcome = asyncio.run(_run())
        assert outcome is None, f'Expected the verify-passed sentinel (None); got {outcome!r}'

        from orchestrator.verify import _BASELINE_FAILING_IDS_CACHE
        assert 'abcsha' in _BASELINE_FAILING_IDS_CACHE, (
            f'Expected the baseline cache to be seeded for merge_sha=abcsha; '
            f'keys={list(_BASELINE_FAILING_IDS_CACHE)!r}'
        )
        _, seeded_ids = _BASELINE_FAILING_IDS_CACHE['abcsha']
        assert seeded_ids == frozenset(), (
            f'Expected the seeded ids to be the empty frozenset (all pass); got {seeded_ids!r}'
        )

    def test_passing_verify_with_no_failing_test_ids_does_not_seed(self, tmp_path: Path) -> None:
        """A pass under 'scoped' breadth (or any other degrade path) never
        collects junit ids -- failing_test_ids stays None -- and must NOT
        seed the baseline: an empty seed there would be WRONG (scoped
        passing doesn't mean main-at-large is clean, just that the touched
        subset passed)."""
        config = _make_config(tmp_path, merge_verify_breadth='scoped')
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        passing_result_no_ids = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='',
            summary='all checks passed', failing_test_ids=None,
        )

        async def _run() -> MergeOutcome | None:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                new=AsyncMock(return_value=passing_result_no_ids),
            ):
                return await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                    merge_sha='scopedsha',
                )

        outcome = asyncio.run(_run())
        assert outcome is None

        from orchestrator.verify import _BASELINE_FAILING_IDS_CACHE
        assert 'scopedsha' not in _BASELINE_FAILING_IDS_CACHE, (
            f'Must NOT seed when failing_test_ids is None; keys='
            f'{list(_BASELINE_FAILING_IDS_CACHE)!r}'
        )

    def test_passing_verify_with_empty_merge_sha_does_not_seed(self, tmp_path: Path) -> None:
        """merge_sha='' (module-level/train callers that pass no merge_sha)
        must not attempt to seed -- there is no future main tip to key on."""
        config = _make_config(tmp_path, merge_verify_breadth='full')
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        passing_result = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='',
            summary='all checks passed', failing_test_ids=[],
        )

        from orchestrator.verify import _BASELINE_FAILING_IDS_CACHE

        async def _run() -> MergeOutcome | None:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                new=AsyncMock(return_value=passing_result),
            ):
                return await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                    merge_sha='',
                )

        before_keys = set(_BASELINE_FAILING_IDS_CACHE)
        outcome = asyncio.run(_run())
        assert outcome is None
        after_keys = set(_BASELINE_FAILING_IDS_CACHE)
        assert after_keys == before_keys, (
            f'Empty merge_sha must not seed anything; before={before_keys!r} '
            f'after={after_keys!r}'
        )


# ---------------------------------------------------------------------------
# Step-19 (task μ, verify-scope-inversion-prd.md): end-to-end baseline
# attribution over _run_post_merge_verify's BLOCK path (SYNC mode —
# main_health_probe_handles=None, the default — matches every other test in
# this file). Row numbering follows the PRD's failing-id-diff truth table.
# ---------------------------------------------------------------------------


class TestBaselineAttributionOverBlockPath:
    """row 5 (wholly preexisting -> MAIN_HEALTH_RED, no probe on repeat);
    row 4 (mixed -> branch block citing only the new id); B3 (failing_test_ids
    is None -> today's category-level reason, no crash)."""

    def test_row5_wholly_preexisting_routes_main_health_red_with_no_probe(
        self, tmp_path: Path,
    ) -> None:
        """baseline={X}, branch fails with only {X} -> MAIN_HEALTH_RED
        (branch not charged). A second failing merge against the same main
        SHA must reuse the cache: the probe machinery (ephemeral_worktree)
        must never be invoked for either merge."""
        from orchestrator.verify import seed_main_baseline

        config = _make_config(tmp_path, merge_verify_breadth='full')
        git_ops = _make_git_ops(tmp_path)

        probe_calls: list[tuple] = []

        def _track_and_explode(*args, **kwargs):
            probe_calls.append((args, kwargs))
            raise AssertionError('probe machinery must not run — baseline cache is warm')
        git_ops.ephemeral_worktree = MagicMock(side_effect=_track_and_explode)

        seed_main_baseline(MAIN_SHA, frozenset({'X'}))

        branch_result = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='Failures: tests failed', failing_test_ids=['X'],
        )

        async def _one_merge(task_id: str) -> MergeOutcome | None:
            merge_wt = tmp_path / f'merge-wt-{task_id}'
            merge_wt.mkdir()
            req = _make_req(task_id, tmp_path / f'task-wt-{task_id}', config)
            (tmp_path / f'task-wt-{task_id}').mkdir()
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                new=AsyncMock(return_value=branch_result),
            ):
                return await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={}, enospc_retries={}, max_timeouts=3, max_enospc=1,
                )

        outcome1 = asyncio.run(_one_merge('201'))
        assert outcome1 is not None
        assert outcome1.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'branch wholly covered by baseline must route MAIN_HEALTH_RED; '
            f'got {outcome1.reason!r}'
        )

        # Second failing merge against the SAME (unchanged) main SHA.
        outcome2 = asyncio.run(_one_merge('202'))
        assert outcome2 is not None
        assert outcome2.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'second merge against unchanged main must also route MAIN_HEALTH_RED; '
            f'got {outcome2.reason!r}'
        )

        assert probe_calls == [], (
            f'expected the baseline-probe machinery to never run across both '
            f'merges (cache pre-seeded); got {len(probe_calls)} call(s): {probe_calls!r}'
        )

    def test_row4_mixed_failure_cites_only_new_id(self, tmp_path: Path) -> None:
        """baseline={X}, branch fails with {X,Y} -> the outcome is a branch
        block whose reason cites only Y (the new id) and NOT X; the
        wholly-preexisting (MAIN_HEALTH_RED) route must NOT be taken."""
        from orchestrator.verify import seed_main_baseline

        config = _make_config(tmp_path, merge_verify_breadth='full')
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        seed_main_baseline(MAIN_SHA, frozenset({'X'}))

        branch_result = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='Failures: tests failed', failing_test_ids=['X', 'Y'],
        )

        async def _run() -> MergeOutcome | None:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                new=AsyncMock(return_value=branch_result),
            ):
                return await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={}, enospc_retries={}, max_timeouts=3, max_enospc=1,
                )

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert not outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'mixed failure (one new id) must NOT route MAIN_HEALTH_RED; got {outcome.reason!r}'
        )
        assert 'Y' in outcome.reason, f'expected the new id Y to be cited; got {outcome.reason!r}'
        assert 'X' not in outcome.reason, (
            f'the preexisting id X must NOT be cited in the branch-block reason; '
            f'got {outcome.reason!r}'
        )

    def test_b3_failing_test_ids_none_uses_legacy_reason(self, tmp_path: Path) -> None:
        """failing_test_ids=None (scoped/degraded/OPAQUE) -> today's
        category-level reason, unchanged, no crash -- even with a warm
        baseline cache sitting there for an unrelated sha comparison."""
        from orchestrator.verify import seed_main_baseline

        config = _make_config(tmp_path, merge_verify_breadth='scoped')
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        # Warm cache present regardless -- must be irrelevant to the B3 path.
        seed_main_baseline(MAIN_SHA, frozenset({'X'}))

        branch_result = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='generic task failure', category='test_failure',
            cause_hint='AssertionError somewhere', failing_test_ids=None,
        )

        async def _run() -> MergeOutcome | None:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=branch_result),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
            ):
                return await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={}, enospc_retries={}, max_timeouts=3, max_enospc=1,
                )

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert outcome.reason.startswith('Post-merge verification failed: generic task failure'), (
            f'expected the legacy category-level reason unchanged; got {outcome.reason!r}'
        )
        assert '[category: test_failure]' in outcome.reason, (
            f'expected the category suffix preserved; got {outcome.reason!r}'
        )


# ---------------------------------------------------------------------------
# Task 2823: config-only trivial-pass gate over _run_post_merge_verify's PASS
# path. A config-only merge (no .py/.rs) short-circuits to verify._trivial_pass
# WITHOUT running the suite; the pass path must refuse to advance main over a
# KNOWN-red baseline (else the red persists — the reify 2026-07-19 incident),
# while still letting a NON-trivial pass (full suite actually ran and passed)
# heal a red main. Cache-ONLY peek — never a probe on the critical path (G4,
# task 2564); fail-OPEN on a cold/unknown baseline (matches "known-red").
#
# SYNC mode (main_health_probe_handles=None, the file default) reaches the
# pass-path gate identically to production DEFERRED mode. RED today: case (a)
# both returns None (no gate) and cannot import the reason-prefix constant.
# ---------------------------------------------------------------------------


class TestTrivialPassMainRedGate:
    """A trivial pass is BLOCKED over a known-red main (MAIN_RED disposition,
    worktree cleaned up), and ONLY then: a green baseline, a cold/unknown
    baseline, and a non-trivial pass over red all advance (return None)."""

    _RED_ID = 'orchestrator/tests/test_x.py::test_foo'

    @pytest.fixture(autouse=True)
    def reset_baseline_cache(self):
        """Clear the per-main-SHA baseline cache around each test. The file's
        module-level autouse fixture only clears _PROBE_CACHE; the cold-cache
        case below needs a guaranteed-empty baseline for MAIN_SHA."""
        from orchestrator.verify import _BASELINE_FAILING_IDS_CACHE
        _BASELINE_FAILING_IDS_CACHE.clear()
        yield
        _BASELINE_FAILING_IDS_CACHE.clear()

    def _trivial_pass(self) -> VerifyResult:
        return VerifyResult(
            passed=True, test_output='', lint_output='', type_output='',
            summary='No source files changed — verify trivially passes',
            trivial=True,
        )

    def _drive(
        self,
        tmp_path: Path,
        git_ops: GitOps,
        config: OrchestratorConfig,
        result: VerifyResult,
        merge_wt: Path,
    ) -> MergeOutcome | None:
        req = _make_req('2823', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()
        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            new=AsyncMock(return_value=result),
        ):
            return asyncio.run(_drive_verify(req, merge_wt, git_ops))

    def test_a_known_red_blocks_trivial_pass(self, tmp_path: Path) -> None:
        """seed red baseline + trivial pass -> blocked/MAIN_RED, worktree cleaned."""
        from orchestrator.merge_queue import (
            TRIVIAL_PASS_MAIN_RED_REASON_PREFIX,
            MergeFailureDisposition,
        )
        from orchestrator.verify import seed_main_baseline

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()

        seed_main_baseline(MAIN_SHA, frozenset({self._RED_ID}))

        outcome = self._drive(tmp_path, git_ops, config, self._trivial_pass(), merge_wt)

        assert outcome is not None, (
            'a trivial pass over a known-red main must be blocked, not advanced'
        )
        assert outcome.status == 'blocked', f'expected blocked; got {outcome.status!r}'
        assert outcome.reason.startswith(TRIVIAL_PASS_MAIN_RED_REASON_PREFIX), (
            f'expected the trivial-pass main-red reason prefix; got {outcome.reason!r}'
        )
        assert outcome.disposition == MergeFailureDisposition.MAIN_RED, (
            f'a pre-existing main red must be attributed MAIN_RED; got {outcome.disposition!r}'
        )
        cast(AsyncMock, git_ops.cleanup_merge_worktree).assert_awaited_with(merge_wt)

    def test_b_green_baseline_advances_trivial_pass(self, tmp_path: Path) -> None:
        """seed EMPTY (green) baseline + trivial pass -> None (advance)."""
        from orchestrator.verify import seed_main_baseline

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()

        seed_main_baseline(MAIN_SHA, frozenset())

        outcome = self._drive(tmp_path, git_ops, config, self._trivial_pass(), merge_wt)

        assert outcome is None, (
            f'a trivial pass over a KNOWN-green main must advance; got {outcome!r}'
        )
        cast(AsyncMock, git_ops.cleanup_merge_worktree).assert_not_awaited()

    def test_c_cold_baseline_fails_open_advances(self, tmp_path: Path) -> None:
        """no seed (cold/unknown baseline) + trivial pass -> None (fail-open)."""
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()

        # No seed_main_baseline: the cache is empty for MAIN_SHA (fixture cleared),
        # so cached_main_baseline_failing_ids returns None -> fail open.
        outcome = self._drive(tmp_path, git_ops, config, self._trivial_pass(), merge_wt)

        assert outcome is None, (
            f'a trivial pass over a COLD/unknown baseline must fail open; got {outcome!r}'
        )

    def test_d_non_trivial_pass_over_red_advances(self, tmp_path: Path) -> None:
        """seed red baseline + NON-trivial pass -> None: a real suite pass
        heals the red main and must NOT be blocked."""
        from orchestrator.verify import seed_main_baseline

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()

        seed_main_baseline(MAIN_SHA, frozenset({self._RED_ID}))

        non_trivial_pass = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='',
            summary='full suite passed', trivial=False, failing_test_ids=[],
        )

        outcome = self._drive(tmp_path, git_ops, config, non_trivial_pass, merge_wt)

        assert outcome is None, (
            f'a NON-trivial pass (full suite ran and passed) must heal a red main, '
            f'not be blocked; got {outcome!r}'
        )
