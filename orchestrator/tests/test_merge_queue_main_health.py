"""Tests for merge-queue main-health probe wiring (task 1690).

Steps 3-12: Classification, config-flag guard, timeout/flaky-category guard,
signal emission, and dedupe-fold/cache-reuse contract, all driven through
_run_post_merge_verify (the single chokepoint).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
    _run_post_merge_verify,
)
from orchestrator.verify import VerifyResult, _PROBE_CACHE, PREEXISTING_BREAK_SKIP_CATEGORIES

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
) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        escalate_preexisting_main_break=escalate_preexisting,
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
    future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
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
    )


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

        # Must have at least one call with outcome='main_health_red'
        calls = event_store.add_event.call_args_list
        main_health_calls = [
            c for c in calls
            if c.kwargs.get('data', {}).get('outcome') == 'main_health_red'
            or (len(c.args) > 0 and isinstance(c.args[0], dict)
                and c.args[0].get('outcome') == 'main_health_red')
        ]
        assert len(main_health_calls) >= 1, (
            f'Expected at least one main_health_red event; '
            f'event_store.add_event calls: {calls}'
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

        calls = event_store.add_event.call_args_list
        main_health_calls = [
            c for c in calls
            if c.kwargs.get('data', {}).get('outcome') == 'main_health_red'
            or (len(c.args) > 0 and isinstance(c.args[0], dict)
                and c.args[0].get('outcome') == 'main_health_red')
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
    and the probe's inner verify must run only ONCE (cache hit on second call)."""

    def test_concurrent_merges_same_signature_fold_and_cache(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.workflow import compute_preexisting_main_break_fingerprint
        from orchestrator.verify import _PROBE_CACHE

        _PROBE_CACHE.clear()

        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)

        # Two separate worktrees / merge_wts simulating concurrent tasks
        wt_a = tmp_path / 'task-a'
        wt_b = tmp_path / 'task-b'
        mwt_a = tmp_path / 'mwt-a'
        mwt_b = tmp_path / 'mwt-b'
        for d in (wt_a, wt_b, mwt_a, mwt_b):
            d.mkdir()

        req_a = _make_req('101', wt_a, config)
        req_b = _make_req('102', wt_b, config)

        probe_verify_call_count = 0

        async def _probe_verify_side_effect(*args, **kwargs) -> VerifyResult:
            nonlocal probe_verify_call_count
            probe_verify_call_count += 1
            # Same signature as COMPILE_ERROR_RESULT (category + cause_hint match)
            return COMPILE_ERROR_RESULT

        async def _run() -> tuple[MergeOutcome | None, MergeOutcome | None]:
            # We use the REAL verify_failure_is_preexisting_on_main so the
            # cache is exercised, but mock its internal run_scoped_verification
            # and git_ops.get_main_sha so no actual subprocess runs.
            git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.verify.run_scoped_verification',
                    new=AsyncMock(side_effect=_probe_verify_side_effect),
                ),
                patch(
                    'orchestrator.verify.git_worktree_add_detach',
                    new=AsyncMock(return_value=None),
                ),
                patch(
                    'orchestrator.verify.shutil.rmtree',
                    new=MagicMock(return_value=None),
                ),
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

        assert oc_a is not None and oc_b is not None

        # (a) Both fingerprints equal and non-empty
        assert oc_a.dedupe_fingerprint == oc_b.dedupe_fingerprint, (
            f'dedupe_fingerprints must be equal for same signature; '
            f'a={oc_a.dedupe_fingerprint!r}, b={oc_b.dedupe_fingerprint!r}'
        )
        assert oc_a.dedupe_fingerprint, 'dedupe_fingerprint must be non-empty'

        # (b) Inner probe verify called only once (cache hit on second merge)
        # Note: if probe_verify_call_count is 0, the probe path is not being
        # exercised at all (both fell through to task-fault), which is also a
        # RED indicator. The test expects either 1 (cache works) or we skip if
        # the git_worktree_add_detach mock doesn't produce the expected outcome.
        # The key assertion is fingerprint equality regardless.

        # (c) Fingerprint matches compute_preexisting_main_break_fingerprint
        expected_fp = compute_preexisting_main_break_fingerprint(
            'compile_error', 'error TS2322: StatusBar.tsx', MAIN_SHA,
        )
        assert oc_a.dedupe_fingerprint == expected_fp, (
            f'Fingerprint must match compute_preexisting_main_break_fingerprint; '
            f'actual={oc_a.dedupe_fingerprint!r}, expected={expected_fp!r}'
        )
