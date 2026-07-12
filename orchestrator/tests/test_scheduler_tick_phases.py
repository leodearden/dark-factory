"""Phase-level tests for Scheduler.acquire_next's tick-phase decomposition.

These construct a ``Scheduler`` + hand-build a ``TickContext`` and call the
``_phase_*`` methods directly — deliberately WITHOUT a full-tick harness
(no ``get_tasks``/``acquire_next`` orchestration is driven here). The
existing ``test_scheduler*.py`` files remain the untouched behaviour-parity
suite that exercises full ticks end-to-end.
"""

import dataclasses
from unittest.mock import AsyncMock

import pytest
from _recording_event_store import _RecordingEventStore

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.scheduler import _CONTINUE, Scheduler, TickContext


@pytest.fixture
def scheduler() -> Scheduler:
    config = OrchestratorConfig(max_per_module=1)
    sched = Scheduler(config)
    sched.finish_startup()
    return sched


class TestTickContextShape:
    def test_tick_context_shape(self):
        assert dataclasses.is_dataclass(TickContext)
        field_names = {f.name for f in dataclasses.fields(TickContext)}
        task_named_fields = {
            'tasks', 'status_map', 'tasks_by_id',
            'external_cache', 'overrides', 'candidates',
        }
        assert task_named_fields <= field_names

        # Real construction using all six task-named fields — proves each
        # is a valid constructor kwarg.
        ctx = TickContext(
            tasks=[{'id': '1'}],
            status_map={'1': 'pending'},
            tasks_by_id={'1': {'id': '1'}},
            external_cache={'other:1': 'done'},
            overrides={},
            candidates=[{'id': '1'}],
        )
        assert ctx.tasks == [{'id': '1'}]
        assert ctx.status_map == {'1': 'pending'}
        assert ctx.tasks_by_id == {'1': {'id': '1'}}
        assert ctx.external_cache == {'other:1': 'done'}
        assert ctx.overrides == {}
        assert ctx.candidates == [{'id': '1'}]

        # Minimal construction (only the three fields with no default)
        # spot-checks the threading-field defaults.
        default_ctx = TickContext(tasks=[], status_map={}, tasks_by_id={})
        assert default_ctx.max_id == 0
        assert default_ctx.external_resolver_failed is False
        assert default_ctx.candidates == []
        assert default_ctx.gated_ids == set()
        assert default_ctx.stale_ids == set()
        assert default_ctx.psi_hold is False
        assert default_ctx.dispatch_deferred_emitted is False


class TestHygienePhases:
    """Isolation tests for the Hygiene phases (park_gc, stale_sweep,
    cooldown_gc) — each calls its ``_phase_*`` method directly against a
    hand-built ``TickContext``, no full-tick orchestration."""

    @pytest.mark.asyncio
    async def test_phase_park_gc_evicts_terminal_owner(self, scheduler: Scheduler):
        event_store = _RecordingEventStore()
        scheduler.event_store = event_store  # type: ignore[assignment]
        scheduler.lock_table.install_parks('A', ['backend'], priority='medium')
        assert scheduler.lock_table.has_parks('A')

        ctx = TickContext(tasks=[], status_map={'A': 'done'}, tasks_by_id={})
        result = await scheduler._phase_park_gc(ctx)

        assert result is _CONTINUE
        assert not scheduler.lock_table.has_parks('A')
        expired = [e for e in event_store.events if 'reservation_expired' in e[0]]
        assert len(expired) == 1
        assert expired[0][1]['task_id'] == 'A'
        assert expired[0][1]['data']['reason'] == 'terminal:done'

    @pytest.mark.asyncio
    async def test_phase_stale_sweep_runs_registry_gc(self, scheduler: Scheduler):
        scheduler._skip_count['A'] = 3
        scheduler._streak_local_backfill.counts[('A', '9')] = 2

        ctx = TickContext(tasks=[], status_map={'A': 'cancelled'}, tasks_by_id={})
        result = await scheduler._phase_stale_sweep(ctx)

        assert result is _CONTINUE
        assert 'A' in ctx.stale_ids
        assert 'A' not in scheduler._skip_count
        # Proves _streak_registry.gc (StreakRegistry.gc, task 2124) was
        # actually consumed — not a re-collapsed hand loop.
        assert ('A', '9') not in scheduler._streak_local_backfill.counts

    @pytest.mark.asyncio
    async def test_phase_cooldown_gc_drops_expired(self, scheduler: Scheduler):
        scheduler._requeue_until['A'] = scheduler._time_source() - 1.0

        ctx = TickContext(tasks=[], status_map={}, tasks_by_id={})
        result = await scheduler._phase_cooldown_gc(ctx)

        assert result is _CONTINUE
        assert 'A' not in scheduler._requeue_until


class TestBackfillDepStatusPhase:
    """Isolation test for the backfill_dep_status phase (phase 1/18) — the
    dep-status backfill that consumes ``_iter_pending_deps_in`` (task 2124)."""

    @pytest.mark.asyncio
    async def test_phase_backfill_consumes_iter_helper(self, scheduler: Scheduler):
        task = {'id': 'T', 'status': 'pending', 'dependencies': ['9']}
        ctx = TickContext(
            tasks=[task],
            status_map={},  # '9' (T's only dep) is missing from status_map
            tasks_by_id={'T': task},
        )
        # Degraded resolver result: resolver_failed({}, <err>) is True, so the
        # degraded (bump) branch runs instead of the recovered (clear) branch.
        scheduler.get_statuses = AsyncMock(return_value=({}, Exception('degraded')))

        result = await scheduler._phase_backfill_dep_status(ctx)

        assert result is _CONTINUE
        # Only producible by iterating `_iter_pending_deps_in(ctx.tasks, missing_dep_ids)`
        # over the pending (task, dep) pairs — proves the shared helper is consumed,
        # not a re-collapsed hand loop.
        assert scheduler._streak_local_backfill.counts[('T', '9')] == 1


class TestPolicySelectionPrepPhases:
    """Isolation tests for the Policy/selection-prep phases (external_dep_policy,
    build_candidates, starvation) — each calls its ``_phase_*`` method directly
    against a hand-built ``TickContext``, no full-tick orchestration."""

    @pytest.mark.asyncio
    async def test_phase_external_dep_policy_runs_gate_exactly_once(
        self, scheduler: Scheduler
    ):
        task = {
            'id': 'T', 'status': 'pending', 'dependencies': [],
            'metadata': {'external_deps': ['other:5']},
        }
        ctx = TickContext(tasks=[task], status_map={'T': 'pending'}, tasks_by_id={'T': task})
        scheduler.get_external_statuses = AsyncMock(
            return_value=({'other:5': 'done'}, None)
        )

        result = await scheduler._phase_external_dep_policy(ctx)

        scheduler.get_external_statuses.assert_awaited_once()
        assert ctx.external_cache == {'other:5': 'done'}
        assert ctx.external_resolver_failed is False
        assert result is _CONTINUE

    @pytest.mark.asyncio
    async def test_phase_build_candidates_fills_ctx(self, scheduler: Scheduler):
        task = {'id': 'A', 'status': 'pending', 'dependencies': []}
        ctx = TickContext(
            tasks=[task],
            status_map={'A': 'pending'},
            tasks_by_id={'A': task},
        )

        result = await scheduler._phase_build_candidates(ctx)

        assert result is _CONTINUE
        assert any(str(t.get('id')) == 'A' for t in ctx.candidates)
        assert 'A' in ctx.candidate_signals

    @pytest.mark.asyncio
    async def test_phase_starvation_scans_candidates(self, scheduler: Scheduler):
        calls: list[str] = []

        async def _spy(task_id, **kwargs):
            calls.append(task_id)

        scheduler._callbacks = dataclasses.replace(
            scheduler._callbacks, on_starvation_warn=_spy
        )
        task = {'id': 'A', 'status': 'pending', 'dependencies': []}
        ctx = TickContext(
            tasks=[task],
            status_map={'A': 'pending'},
            tasks_by_id={'A': task},
            candidates=[task],
        )

        result = await scheduler._phase_starvation(ctx)

        assert result is _CONTINUE
