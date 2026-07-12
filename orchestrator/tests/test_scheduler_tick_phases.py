"""Phase-level tests for Scheduler.acquire_next's tick-phase decomposition.

These construct a ``Scheduler`` + hand-build a ``TickContext`` and call the
``_phase_*`` methods directly — deliberately WITHOUT a full-tick harness
(no ``get_tasks``/``acquire_next`` orchestration is driven here). The
existing ``test_scheduler*.py`` files remain the untouched behaviour-parity
suite that exercises full ticks end-to-end.
"""

import dataclasses
import inspect
from unittest.mock import AsyncMock

import pytest
from _recording_event_store import _RecordingEventStore
from shared.psi import PsiSample

from orchestrator.config import OrchestratorConfig
from orchestrator.overrides import OverrideRow, OverrideStore
from orchestrator.scheduler import _CONTINUE, Scheduler, TickContext, TickOutcome


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


class TestSelectionPhases:
    """Isolation tests for the Selection phases (empty_candidate_gate,
    select_pins, select_scored) — each calls its ``_phase_*`` method
    directly against a hand-built ``TickContext``, no full-tick
    orchestration."""

    @pytest.mark.asyncio
    async def test_phase_empty_candidate_gate_returns_none(self, scheduler: Scheduler):
        ctx = TickContext(tasks=[], status_map={}, tasks_by_id={}, candidates=[])

        result = await scheduler._phase_empty_candidate_gate(ctx)

        assert isinstance(result, TickOutcome)
        assert result.assignment is None

    @pytest.mark.asyncio
    async def test_phase_empty_candidate_gate_continues_when_nonempty(
        self, scheduler: Scheduler
    ):
        task = {'id': 'A', 'status': 'pending', 'dependencies': []}
        ctx = TickContext(
            tasks=[task], status_map={'A': 'pending'}, tasks_by_id={'A': task},
            candidates=[task],
        )

        result = await scheduler._phase_empty_candidate_gate(ctx)

        assert result is _CONTINUE

    @pytest.mark.asyncio
    async def test_phase_select_scored_dispatches_top(self, scheduler: Scheduler):
        task = {
            'id': 'A', 'status': 'pending', 'dependencies': [],
            'metadata': {'files': ['mod_a/x.py']}, 'priority': 'medium',
        }
        ctx = TickContext(
            tasks=[task],
            status_map={'A': 'pending'},
            tasks_by_id={'A': task},
            candidates=[task],
            candidate_signals={'A': None},
            effective_priorities={'A': 'medium'},
            transitive_counts={'A': 0},
        )

        result = await scheduler._phase_select_scored(ctx)

        assert isinstance(result, TickOutcome)
        assert result.assignment is not None
        assert result.assignment.task_id == 'A'
        assert scheduler.lock_table.is_held('A')

    @pytest.mark.asyncio
    async def test_phase_select_scored_lock_conflict_returns_none(
        self, scheduler: Scheduler
    ):
        task = {
            'id': 'A', 'status': 'pending', 'dependencies': [],
            'metadata': {'files': ['mod_a/x.py']}, 'priority': 'medium',
        }
        # Pre-acquire A's module for a different owner so A's try_acquire fails.
        modules = scheduler._get_modules(task)
        scheduler.lock_table._held['other'] = set(modules)
        ctx = TickContext(
            tasks=[task],
            status_map={'A': 'pending'},
            tasks_by_id={'A': task},
            candidates=[task],
            candidate_signals={'A': None},
            effective_priorities={'A': 'medium'},
            transitive_counts={'A': 0},
        )

        result = await scheduler._phase_select_scored(ctx)

        assert isinstance(result, TickOutcome)
        assert result.assignment is None
        assert scheduler._skip_count.get('A') == 1

    @pytest.mark.asyncio
    async def test_phase_select_pins_dispatches_pinned(self, tmp_path):
        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'o.db')
        scheduler = Scheduler(config, override_store=store)
        scheduler.finish_startup()
        scheduler._project_root = '/proj'

        task = {
            'id': 'A', 'status': 'pending', 'dependencies': [],
            'metadata': {'files': ['mod_a/x.py']}, 'priority': 'medium',
        }
        pinned_row = OverrideRow(
            boost_tier=None, pinned=True, pin_order=0, reserve_now=False,
            ttl_until=None,
        )
        ctx = TickContext(
            tasks=[task],
            status_map={'A': 'pending'},
            tasks_by_id={'A': task},
            candidates=[task],
            overrides={'A': pinned_row},
            effective_priorities={'A': 'medium'},
        )

        result = await scheduler._phase_select_pins(ctx)

        assert isinstance(result, TickOutcome)
        assert result.assignment is not None
        assert result.assignment.task_id == 'A'
        assert scheduler.lock_table.is_held('A')

    @pytest.mark.asyncio
    async def test_phase_select_pins_continues_when_not_pinned(self, tmp_path):
        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'o.db')
        scheduler = Scheduler(config, override_store=store)
        scheduler.finish_startup()
        scheduler._project_root = '/proj'

        task = {
            'id': 'A', 'status': 'pending', 'dependencies': [],
            'metadata': {'files': ['mod_a/x.py']}, 'priority': 'medium',
        }
        unpinned_row = OverrideRow(
            boost_tier=None, pinned=False, pin_order=None, reserve_now=False,
            ttl_until=None,
        )
        ctx = TickContext(
            tasks=[task],
            status_map={'A': 'pending'},
            tasks_by_id={'A': task},
            candidates=[task],
            overrides={'A': unpinned_row},
            effective_priorities={'A': 'medium'},
        )

        result = await scheduler._phase_select_pins(ctx)

        assert result is _CONTINUE
        assert not scheduler.lock_table.is_held('A')

    @pytest.mark.asyncio
    async def test_dispatch_deferred_emitted_shared_across_pins_and_scored(
        self, tmp_path
    ):
        """``ctx.dispatch_deferred_emitted`` (promoted from a nested-closure
        ``nonlocal`` to per-tick ``TickContext`` state, task 2236 step-10) must
        stay SHARED across the pins/scored split: a heavy PINNED candidate
        deferred by ``_phase_select_pins`` must suppress a second
        ``dispatch_deferred`` emission when ``_phase_select_scored`` later
        defers its own heavy candidate against the SAME ctx — exactly one
        event per tick, as when both loops lived in one method body."""
        event_store = _RecordingEventStore()
        config = OrchestratorConfig(max_per_module=1)
        store = OverrideStore(tmp_path / 'o.db')
        scheduler = Scheduler(config, event_store=event_store, override_store=store)  # type: ignore[arg-type]
        scheduler.finish_startup()
        scheduler._project_root = '/proj'

        pinned_task = {
            'id': 'P', 'status': 'pending', 'dependencies': [],
            'metadata': {'files': ['mod_p/x.py']}, 'priority': 'medium',
        }
        scored_task = {
            'id': 'S', 'status': 'pending', 'dependencies': [],
            'metadata': {'files': ['mod_s/x.py']}, 'priority': 'medium',
        }
        pinned_row = OverrideRow(
            boost_tier=None, pinned=True, pin_order=0, reserve_now=False,
            ttl_until=None,
        )
        ctx = TickContext(
            tasks=[pinned_task, scored_task],
            status_map={'P': 'pending', 'S': 'pending'},
            tasks_by_id={'P': pinned_task, 'S': scored_task},
            candidates=[scored_task],
            overrides={'P': pinned_row},
            effective_priorities={'P': 'medium', 'S': 'medium'},
            psi_sample=PsiSample(
                cpu_some10=90.0, mem_some10=0.0, mem_full10=0.0,
                io_some10=0.0, read_ok=True,
            ),
            psi_hold=True,
        )

        pins_result = await scheduler._phase_select_pins(ctx)
        scored_result = await scheduler._phase_select_scored(ctx)

        assert pins_result is _CONTINUE, 'pinned heavy candidate deferred, not dispatched'
        assert isinstance(scored_result, TickOutcome)
        assert scored_result.assignment is None, 'scored heavy candidate also deferred'
        assert 'P' not in scheduler._dispatched
        assert 'S' not in scheduler._dispatched

        deferred_events = [e for e in event_store.events if e[0] == 'dispatch_deferred']
        assert len(deferred_events) == 1, (
            'the scored phase must NOT re-emit dispatch_deferred once the pin '
            'phase already emitted it this tick — dispatch_deferred_emitted is '
            'shared per-tick ctx state, not per-phase'
        )
        assert deferred_events[0][1]['task_id'] == 'P', (
            'the pin loop runs first and hits the deferral first'
        )
        assert ctx.dispatch_deferred_emitted is True


class TestTickPhaseOrderLiteral:
    """Ordering-as-data: ``Scheduler._TICK_PHASE_ORDER`` is the SINGLE source
    of dispatch order — asserted here AND iterated by ``acquire_next`` (see
    ``test_acquire_next_delegates_to_phase_list`` below), so the assertions
    on the literal are a behavioural check, not a documentation meta-test.
    """

    def test_tick_phase_order_literal(self):
        order = Scheduler._TICK_PHASE_ORDER

        assert isinstance(order, tuple)
        assert all(isinstance(label, str) for label in order)

        # (1) drain_park_eviction before park_gc.
        assert order.index('drain_park_eviction') < order.index('park_gc')
        # (2) cooldown_gc before BOTH candidate-dispatch loops.
        assert order.index('cooldown_gc') < order.index('select_pins')
        assert order.index('cooldown_gc') < order.index('select_scored')
        # (3) external-dep policy runs exactly once per tick.
        assert order.count('external_dep_policy') == 1

        # Every label genuinely drives dispatch — it maps to a real
        # coroutine method on Scheduler, not just a decorative string.
        for label in order:
            method = getattr(Scheduler, f'_phase_{label}', None)
            assert method is not None, f'Scheduler._phase_{label} does not exist'
            assert inspect.iscoroutinefunction(method), (
                f'Scheduler._phase_{label} must be an async method'
            )

    @pytest.mark.asyncio
    async def test_acquire_next_delegates_to_phase_list(self, scheduler: Scheduler):
        """A light end-to-end tick proves acquire_next dispatches THROUGH
        ``_TICK_PHASE_ORDER`` — every phase named in the tuple is invoked,
        in the tuple's own order, on the way to the correct dispatch."""
        task = {
            'id': 'A', 'title': 'Task A', 'status': 'pending', 'dependencies': [],
            'metadata': {'files': ['backend']}, 'priority': 'medium',
        }
        scheduler.get_tasks = AsyncMock(return_value=[task])

        called: list[str] = []
        for label in Scheduler._TICK_PHASE_ORDER:
            attr_name = f'_phase_{label}'
            original = getattr(scheduler, attr_name)

            def _make_wrapper(attr_name=attr_name, original=original):
                async def _wrapper(ctx):
                    called.append(attr_name)
                    return await original(ctx)
                return _wrapper

            setattr(scheduler, attr_name, _make_wrapper())

        result = await scheduler.acquire_next()

        assert result is not None
        assert result.task_id == 'A'
        assert scheduler.lock_table.is_held('A')
        assert called == [f'_phase_{label}' for label in Scheduler._TICK_PHASE_ORDER]
