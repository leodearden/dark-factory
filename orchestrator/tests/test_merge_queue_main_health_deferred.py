"""Tests for the deferred main-health probe's guard/dedup semantics (task 2564).

``_spawn_main_health_probe`` (added task 2564 step-4) unconditionally spawns
a detached background task whenever handles are supplied — the three cheap
guards (config flag / timed_out / skip-category) and the in-flight dedup
that ``_classify_main_health_red`` already enforces for the SYNCHRONOUS path
are not yet mirrored inside ``_spawn_main_health_probe`` for the DEFERRED
path (added task 2564 step-6). This file pins the RED bar for that gap
(step-5), plus a regression check that
``main_health_probe_handles=None`` (every existing caller) still spawns
nothing through the full ``_run_post_merge_verify`` chokepoint.

Reuses test_merge_queue_main_health.py's fixtures/config-builders
(_make_config/_make_git_ops/_make_req, COMPILE_ERROR_RESULT) per that
module's established cross-test-file reuse convention (see
test_merge_queue_dry_run_unblock.py, which does the same).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_merge_queue_main_health import (
    COMPILE_ERROR_RESULT,
    MAIN_SHA,
    _make_config,
    _make_git_ops,
    _make_req,
)

from escalation.queue import EscalationQueue
from orchestrator.event_store import EventStore
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    _MainHealthProbeHandles,
    _main_health_fingerprint,
    _run_deferred_main_health_probe,
    _run_post_merge_verify,
    _spawn_main_health_probe,
)
from orchestrator.verify import _PROBE_CACHE, VerifyResult


@pytest.fixture(autouse=True)
def reset_probe_cache():
    """Clear the process-wide _PROBE_CACHE between tests.

    Mirrors test_merge_queue_main_health.py's autouse fixture of the same
    name — a separate test module gets a separate pytest collection, so the
    cache must be reset here too for the later (real-probe-driving) steps
    added to this file.
    """
    _PROBE_CACHE.clear()
    yield
    _PROBE_CACHE.clear()


# A timed_out=True result whose category ('compile_error') is NOT itself a
# skip-category, so this isolates the timed_out guard from the category guard.
TIMED_OUT_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='verify timed out',
    category='compile_error',
    timed_out=True,
)

# A category in PREEXISTING_BREAK_SKIP_CATEGORIES with timed_out=False, so
# this isolates the category guard from the timed_out guard.
SKIP_CATEGORY_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='',
    summary='infra timeout category',
    category='infra_timeout',
    timed_out=False,
)


# ---------------------------------------------------------------------------
# Step-5 (RED): _spawn_main_health_probe guard parity with
# _classify_main_health_red — none of the three cheap guards are mirrored
# inside _spawn_main_health_probe yet, so each of these currently spawns a
# task (asserts fail) until step-6 adds the guards.
# ---------------------------------------------------------------------------


class TestSpawnGuards:
    """_spawn_main_health_probe must apply the same three cheap guards
    _classify_main_health_red applies before probing — none of them should
    ever result in a background task being registered."""

    @pytest.mark.parametrize('verify_result,escalate_preexisting,label', [
        (COMPILE_ERROR_RESULT, False, 'config_flag_off'),
        (TIMED_OUT_RESULT, True, 'timed_out'),
        (SKIP_CATEGORY_RESULT, True, 'skip_category'),
    ])
    def test_guard_skips_spawn(
        self,
        tmp_path: Path,
        verify_result: VerifyResult,
        escalate_preexisting: bool,
        label: str,
    ) -> None:
        config = _make_config(tmp_path, escalate_preexisting=escalate_preexisting)
        git_ops = _make_git_ops(tmp_path)
        req = _make_req('42', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        async def _run() -> set[asyncio.Task]:
            handles = _MainHealthProbeHandles(background_tasks=set())
            # No intervening await between the spawn call and the snapshot
            # below — a newly created task cannot have started (let alone
            # finished) running yet, so this snapshot is authoritative for
            # "was a task registered", independent of the stub probe's
            # runtime.
            _spawn_main_health_probe(handles, git_ops, req, verify_result)
            snapshot = set(handles.background_tasks)
            for t in snapshot:
                t.cancel()
            return snapshot

        spawned = asyncio.run(_run())
        assert spawned == set(), (
            f'[{label}] Expected no main-health probe task to be spawned; '
            f'got {spawned}'
        )


# ---------------------------------------------------------------------------
# Step-5 (RED): in-flight dedup — a second spawn for the same task_id while
# the first is not-done must not add a duplicate.
# ---------------------------------------------------------------------------


class TestSpawnDedup:
    def test_second_spawn_while_first_not_done_is_deduped(
        self, tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)
        req = _make_req('42', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        async def _run() -> set[asyncio.Task]:
            handles = _MainHealthProbeHandles(background_tasks=set())
            _spawn_main_health_probe(handles, git_ops, req, COMPILE_ERROR_RESULT)
            # No await yet, so the first spawned task is registered but
            # definitely not-done — exactly the in-flight state the dedup
            # guard must detect before adding a second task for the same
            # task_id.
            _spawn_main_health_probe(handles, git_ops, req, COMPILE_ERROR_RESULT)
            snapshot = set(handles.background_tasks)
            for t in snapshot:
                t.cancel()
            return snapshot

        spawned = asyncio.run(_run())
        assert len(spawned) == 1, (
            f'Expected exactly one in-flight main-health-probe task '
            f'(duplicate spawn while not-done must be deduped); got {spawned}'
        )


# ---------------------------------------------------------------------------
# Step-5: regression — main_health_probe_handles=None (every existing
# caller) must still spawn nothing through the full _run_post_merge_verify
# chokepoint. This is already true today (handles=None short-circuits
# _spawn_main_health_probe before it ever reaches asyncio.create_task), so
# unlike the two classes above this one is expected to already pass.
# ---------------------------------------------------------------------------


class TestNoneHandlesSpawnsNothing:
    def test_none_handles_spawns_nothing_and_does_not_raise(
        self, tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        create_task_spy = MagicMock()

        async def _run() -> object:
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=COMPILE_ERROR_RESULT),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
                patch(
                    'orchestrator.merge_queue.asyncio.create_task',
                    new=create_task_spy,
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
        assert outcome.reason.startswith('Post-merge verification failed'), (
            f'Expected normal task-fault outcome; got {outcome.reason!r}'
        )
        assert create_task_spy.call_count == 0, (
            f'main_health_probe_handles=None must spawn nothing; '
            f'asyncio.create_task called {create_task_spy.call_count} time(s)'
        )


# ---------------------------------------------------------------------------
# Step-9 (RED): _run_deferred_main_health_probe happy path — files a dedup'd
# preexisting_main_break escalation and emits the main_health_red signal for
# a confirmed pre-existing break; a negative or raising probe files nothing.
# ---------------------------------------------------------------------------


class TestDeferredProbeHappyPath:
    def test_confirmed_break_files_escalation_and_emits_signal(
        self, tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)  # get_main_sha AsyncMock -> MAIN_SHA
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        event_store = MagicMock(spec=EventStore)
        escalation_queue = EscalationQueue(tmp_path / 'escalations')

        async def _run() -> None:
            with patch(
                'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                new=AsyncMock(return_value=(True, MAIN_SHA)),
            ):
                await _run_deferred_main_health_probe(
                    git_ops, req, COMPILE_ERROR_RESULT,
                    escalation_queue=escalation_queue, event_store=event_store,
                )

        asyncio.run(_run())

        pending = escalation_queue.get_pending()
        assert len(pending) == 1, (
            f'Expected exactly one pending escalation; got {pending}'
        )
        esc = pending[0]
        assert esc.category == 'preexisting_main_break', (
            f'Expected category=preexisting_main_break; got {esc.category!r}'
        )
        expected_fp = _main_health_fingerprint(
            'compile_error', COMPILE_ERROR_RESULT.cause_hint, MAIN_SHA,
        )
        assert esc.dedupe_fingerprint == expected_fp, (
            f'Expected dedupe_fingerprint={expected_fp!r}; '
            f'got {esc.dedupe_fingerprint!r}'
        )
        assert esc.summary.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Expected summary derived from the main-red reason; '
            f'got {esc.summary!r}'
        )
        assert esc.detail.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'Expected detail derived from the main-red reason; '
            f'got {esc.detail!r}'
        )

        calls = event_store.emit.call_args_list
        main_health_calls = [
            c for c in calls
            if c.kwargs.get('data', {}).get('outcome') == 'main_health_red'
        ]
        assert len(main_health_calls) >= 1, (
            f'Expected at least one main_health_red event; '
            f'event_store.emit calls: {calls}'
        )

    def test_negative_probe_files_no_escalation(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()
        escalation_queue = EscalationQueue(tmp_path / 'escalations')

        async def _run() -> None:
            with patch(
                'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                new=AsyncMock(return_value=(False, '')),
            ):
                await _run_deferred_main_health_probe(
                    git_ops, req, COMPILE_ERROR_RESULT,
                    escalation_queue=escalation_queue,
                )

        asyncio.run(_run())
        assert escalation_queue.get_pending() == [], (
            'A negative (non-preexisting) probe must file no escalation'
        )

    def test_raising_probe_files_no_escalation(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()
        escalation_queue = EscalationQueue(tmp_path / 'escalations')

        async def _run() -> None:
            with patch(
                'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                new=AsyncMock(side_effect=RuntimeError('boom')),
            ):
                await _run_deferred_main_health_probe(
                    git_ops, req, COMPILE_ERROR_RESULT,
                    escalation_queue=escalation_queue,
                )

        asyncio.run(_run())  # must not raise
        assert escalation_queue.get_pending() == [], (
            'A raising probe must file no escalation'
        )


# ---------------------------------------------------------------------------
# Step-11 (RED): stale-check — a probe verdict against a main SHA that has
# since moved (or whose re-resolution fails/empties) must file NO escalation
# (fail safe). Step-10 files unconditionally on any (True, probe_sha)
# verdict without re-resolving git_ops.get_main_sha() at all, so each of
# these currently fails until step-12 adds the re-resolve + equality check.
# ---------------------------------------------------------------------------

# Deliberately distinct from MAIN_SHA (test_merge_queue_main_health's
# default _make_git_ops().get_main_sha() return value) so a probe verdict
# carrying this sha simulates main having advanced since the probe ran.
STALE_PROBE_SHA = 'deadbeef00001111222233334444555566667777'


class TestDeferredProbeStaleCheck:
    def test_moved_main_files_no_escalation(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)  # get_main_sha AsyncMock -> MAIN_SHA
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()
        escalation_queue = EscalationQueue(tmp_path / 'escalations')

        async def _run() -> None:
            with patch(
                'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                new=AsyncMock(return_value=(True, STALE_PROBE_SHA)),
            ):
                await _run_deferred_main_health_probe(
                    git_ops, req, COMPILE_ERROR_RESULT,
                    escalation_queue=escalation_queue,
                )

        asyncio.run(_run())
        assert escalation_queue.get_pending() == [], (
            'A probe verdict against a main SHA that has since moved '
            '(git_ops.get_main_sha() != probe_sha) must file no escalation'
        )

    def test_get_main_sha_raising_files_no_escalation(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)
        git_ops.get_main_sha = AsyncMock(side_effect=RuntimeError('boom'))
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()
        escalation_queue = EscalationQueue(tmp_path / 'escalations')

        async def _run() -> None:
            with patch(
                'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                new=AsyncMock(return_value=(True, MAIN_SHA)),
            ):
                await _run_deferred_main_health_probe(
                    git_ops, req, COMPILE_ERROR_RESULT,
                    escalation_queue=escalation_queue,
                )

        asyncio.run(_run())  # must not raise
        assert escalation_queue.get_pending() == [], (
            'A raising get_main_sha() re-resolve must file no escalation '
            '(fail safe)'
        )

    def test_get_main_sha_empty_files_no_escalation(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)
        git_ops.get_main_sha = AsyncMock(return_value='')
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()
        escalation_queue = EscalationQueue(tmp_path / 'escalations')

        async def _run() -> None:
            with patch(
                'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                new=AsyncMock(return_value=(True, MAIN_SHA)),
            ):
                await _run_deferred_main_health_probe(
                    git_ops, req, COMPILE_ERROR_RESULT,
                    escalation_queue=escalation_queue,
                )

        asyncio.run(_run())
        assert escalation_queue.get_pending() == [], (
            'An empty get_main_sha() re-resolve must file no escalation '
            '(fail safe)'
        )


# ---------------------------------------------------------------------------
# Step-13 (RED): dedup-fold — a probe outcome whose fingerprint matches an
# ALREADY-PENDING preexisting_main_break escalation must fold into that
# parent (submit_or_dedupe's real attach_dedupe_child path) rather than
# create a second top-level parent. Pins that _file_main_health_escalation
# routes through submit_or_dedupe (not a plain queue.submit).
# ---------------------------------------------------------------------------


class TestDeferredProbeDedupeFold:
    def test_matching_fingerprint_folds_into_existing_parent(
        self, tmp_path: Path,
    ) -> None:
        from escalation.models import Escalation

        config = _make_config(tmp_path, escalate_preexisting=True)
        git_ops = _make_git_ops(tmp_path)  # get_main_sha AsyncMock -> MAIN_SHA
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()
        escalation_queue = EscalationQueue(tmp_path / 'escalations')

        # Pre-seed a pending parent whose fingerprint equals what THIS probe's
        # outcome will compute (same category/cause_hint/probe_sha) — as if a
        # sibling task's probe already surfaced this exact main-red signature.
        fp = _main_health_fingerprint(
            COMPILE_ERROR_RESULT.category or '', COMPILE_ERROR_RESULT.cause_hint, MAIN_SHA,
        )
        parent = Escalation(
            id=escalation_queue.make_id('other-task'),
            task_id='other-task',
            agent_role='orchestrator',
            severity='blocking',
            category='preexisting_main_break',
            summary='Pre-existing main-red (sibling probe)',
            dedupe_fingerprint=fp,
        )
        escalation_queue.submit(parent)

        async def _run() -> None:
            with patch(
                'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                new=AsyncMock(return_value=(True, MAIN_SHA)),
            ):
                await _run_deferred_main_health_probe(
                    git_ops, req, COMPILE_ERROR_RESULT,
                    escalation_queue=escalation_queue,
                )

        asyncio.run(_run())

        pending = escalation_queue.get_pending()
        assert len(pending) == 1, (
            f'Expected the new escalation to FOLD into the pre-seeded parent '
            f'(exactly one pending top-level escalation, not a second '
            f'parent); got {pending}'
        )
        assert pending[0].id == parent.id, (
            f'Expected the surviving pending escalation to be the pre-seeded '
            f'parent {parent.id!r}; got {pending[0].id!r}'
        )
        assert pending[0].dedupe_count == 1, (
            f'Expected the parent dedupe_count to increment for the folded '
            f'child; got {pending[0].dedupe_count}'
        )
        assert len(pending[0].dedupe_children) == 1, (
            f'Expected exactly one dedupe child attached to the parent; '
            f'got {pending[0].dedupe_children}'
        )
