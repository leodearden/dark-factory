"""Unit tests for the shared out-of-band routing helper (task η,
plans/adaptive-model-routing-prd.md).

``orchestrator.routing_dispatch.resolve_and_record_route`` centralizes, for
the five out-of-band LLM dispatch sites (steward main + inner triage,
deep_reviewer, module_tagger, unblock_auto), the same resolve → emit → mirror
telemetry that ``TaskWorkflow._invoke``/``_record_routing_decision`` do inline
(workflow.py, task ε/γ — out of this task's locked scope). It:

1. builds a ``RouteInputs`` (with best-effort ceiling spend) and calls
   ``orchestrator.routing.resolve_route``, returning the ``RoutingDecision``;
2. emits exactly one ``EventType.routing_decision`` event (guarded on
   ``event_store``);
3. best-effort mirrors ``metadata.routing`` — persisted via
   ``scheduler.update_task(metadata_mode='merge')`` when a scheduler+task_id
   are given, and/or in-memory onto a supplied mutable task dict.

RED phase (step-1): ``orchestrator.routing_dispatch`` does not exist yet, so
every test below fails at import.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec, stamp_stock_routing_config
from _recording_event_store import _RecordingEventStore

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.routing import RoleDefaults, RoutingDecision
from orchestrator.routing_dispatch import resolve_and_record_route

_HELPER = 'orchestrator.routing_dispatch'

_STEWARD_DEFAULTS = RoleDefaults(model='opus', effort='high', budget_usd=5.0, max_turns=100)


def _mk_cfg(*, ceilings: dict | None = None):
    """Stock mock config with a deterministic per-role config layer for 'steward'."""
    cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    cfg.fused_memory.project_id = 'dark_factory'
    stamp_stock_routing_config(cfg)
    if ceilings is not None:
        cfg.routing.per_model_daily_ceiling_usd = ceilings
    # Deterministic config-layer values for role_name='steward' (real
    # resolve_route path tests read these).
    cfg.models.steward = 'sonnet'
    cfg.budgets.steward = 5.0
    cfg.max_turns.steward = 100
    cfg.effort.steward = 'high'
    cfg.backends.steward = 'claude'
    return cfg


def _fake_decision(*, model: str = 'haiku') -> RoutingDecision:
    return RoutingDecision(
        model=model,
        effort='low',
        budget_usd=1.23,
        max_turns=17,
        source_layer='config',
        rule_id=None,
        rejected=(),
    )


def _routing_entries(rec: _RecordingEventStore) -> list[dict]:
    return [entry for (etype, entry) in rec.events if etype == EventType.routing_decision]


@pytest.mark.asyncio
class TestReturnsDecisionAndBuildsInputs:
    """(a) resolve_and_record_route returns resolve_route's decision and builds
    the RouteInputs correctly."""

    async def test_returns_decision_and_builds_route_inputs(self) -> None:
        cfg = _mk_cfg()
        fake = _fake_decision()
        with patch(f'{_HELPER}.resolve_route', return_value=fake) as mock_resolve:
            decision = await resolve_and_record_route(
                role_name='steward',
                role_defaults=_STEWARD_DEFAULTS,
                config=cfg,
                task_id='2537',
                task_metadata={'routing': {'routing_tier': 3}, 'dispatch_count': 5},
            )

        assert decision is fake
        mock_resolve.assert_called_once()
        route_inputs = mock_resolve.call_args.args[0]
        assert route_inputs.role_name == 'steward'
        assert route_inputs.routing_tier == 3
        assert route_inputs.dispatch_count == 5
        assert route_inputs.role_defaults == _STEWARD_DEFAULTS
        assert route_inputs.spend_by_model == {}
        assert mock_resolve.call_args.args[1] is cfg


@pytest.mark.asyncio
class TestEmitsRoutingDecisionEvent:
    """(b)/(c) exactly one routing_decision event when an event_store is given;
    none (and no raise) when it is None."""

    async def test_emits_one_event_with_mirror_payload(self) -> None:
        cfg = _mk_cfg()
        fake = _fake_decision()
        rec = _RecordingEventStore()
        with patch(f'{_HELPER}.resolve_route', return_value=fake):
            await resolve_and_record_route(
                role_name='steward',
                role_defaults=_STEWARD_DEFAULTS,
                config=cfg,
                task_id='2537',
                task_metadata={},
                event_store=rec,
            )

        entries = _routing_entries(rec)
        assert len(entries) == 1, f'expected one routing_decision event; got {rec.events!r}'
        data = entries[0]['data']
        assert data['role'] == 'steward'
        assert data['model'] == 'haiku'
        assert data['effort'] == 'low'
        assert data['budget_usd'] == 1.23
        assert data['max_turns'] == 17
        assert data['source_layer'] == 'config'
        assert isinstance(data['inputs_digest'], str) and data['inputs_digest']

    async def test_no_event_store_does_not_raise(self) -> None:
        cfg = _mk_cfg()
        fake = _fake_decision()
        with patch(f'{_HELPER}.resolve_route', return_value=fake):
            decision = await resolve_and_record_route(
                role_name='steward',
                role_defaults=_STEWARD_DEFAULTS,
                config=cfg,
                task_id='2537',
                task_metadata={},
                event_store=None,
            )
        assert decision is fake


@pytest.mark.asyncio
class TestHonorsModelOverride:
    """(d) a model_overrides entry in the supplied task_metadata wins via the
    real resolve_route path."""

    async def test_override_resolves_to_that_model(self) -> None:
        cfg = _mk_cfg()
        rec = _RecordingEventStore()
        decision = await resolve_and_record_route(
            role_name='steward',
            role_defaults=_STEWARD_DEFAULTS,
            config=cfg,
            task_id='2537',
            task_metadata={'model_overrides': {'steward': 'haiku'}},
            event_store=rec,
        )
        assert decision.model == 'haiku'
        assert decision.source_layer == 'metadata_override'
        data = _routing_entries(rec)[0]['data']
        assert data['model'] == 'haiku'
        assert data['source_layer'] == 'metadata_override'


@pytest.mark.asyncio
class TestMirrorsMetadata:
    """(e)/(f) persisted scheduler mirror + in-memory task-dict mirror."""

    async def test_scheduler_mirror_written_merge_mode(self) -> None:
        cfg = _mk_cfg()
        fake = _fake_decision()
        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)
        with patch(f'{_HELPER}.resolve_route', return_value=fake):
            await resolve_and_record_route(
                role_name='steward',
                role_defaults=_STEWARD_DEFAULTS,
                config=cfg,
                task_id='2537',
                task_metadata={},
                scheduler=scheduler,
            )

        scheduler.update_task.assert_awaited_once()
        call = scheduler.update_task.call_args
        assert call.args[0] == '2537'
        assert call.args[1]['routing']['latest']['model'] == 'haiku'
        assert call.kwargs.get('metadata_mode') == 'merge'

    async def test_in_memory_task_dict_mutated(self) -> None:
        cfg = _mk_cfg()
        fake = _fake_decision()
        task: dict = {}
        with patch(f'{_HELPER}.resolve_route', return_value=fake):
            await resolve_and_record_route(
                role_name='steward',
                role_defaults=_STEWARD_DEFAULTS,
                config=cfg,
                task_id='2537',
                task_metadata={},
                in_memory_task=task,
            )
        assert task['metadata']['routing']['latest']['model'] == 'haiku'

    async def test_no_scheduler_mirror_without_task_id(self) -> None:
        """module_tagger's batch shape: scheduler present but task_id None → no mirror."""
        cfg = _mk_cfg()
        fake = _fake_decision()
        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)
        with patch(f'{_HELPER}.resolve_route', return_value=fake):
            await resolve_and_record_route(
                role_name='module_tagger',
                role_defaults=_STEWARD_DEFAULTS,
                config=cfg,
                task_id=None,
                task_metadata={},
                scheduler=scheduler,
            )
        scheduler.update_task.assert_not_awaited()


@pytest.mark.asyncio
class TestGracefulDegradation:
    """(g) telemetry never crashes a caller: a scheduler write failure still
    emits the event; a decision-record build failure swallows event + mirror."""

    async def test_scheduler_failure_still_emits_event(self) -> None:
        cfg = _mk_cfg()
        fake = _fake_decision()
        rec = _RecordingEventStore()
        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(side_effect=RuntimeError('boom'))
        with patch(f'{_HELPER}.resolve_route', return_value=fake):
            decision = await resolve_and_record_route(
                role_name='steward',
                role_defaults=_STEWARD_DEFAULTS,
                config=cfg,
                task_id='2537',
                task_metadata={},
                event_store=rec,
                scheduler=scheduler,
            )
        assert decision is fake
        assert len(_routing_entries(rec)) == 1

    async def test_mirror_build_failure_swallowed(self) -> None:
        cfg = _mk_cfg()
        fake = _fake_decision()
        rec = _RecordingEventStore()
        task: dict = {}
        with (
            patch(f'{_HELPER}.resolve_route', return_value=fake),
            patch(f'{_HELPER}.RoutingDecisionMirror', side_effect=RuntimeError('boom')),
        ):
            decision = await resolve_and_record_route(
                role_name='steward',
                role_defaults=_STEWARD_DEFAULTS,
                config=cfg,
                task_id='2537',
                task_metadata={},
                event_store=rec,
                in_memory_task=task,
            )
        # The build failure is caught before any emit/mirror is attempted.
        assert decision is fake
        assert _routing_entries(rec) == []
        assert 'metadata' not in task


@pytest.mark.asyncio
class TestCeilingSpendQuery:
    """(h) cost_store.model_cost_in_window is awaited only when
    per_model_daily_ceiling_usd is non-empty AND a cost_store is present."""

    async def test_stock_ceilings_fire_zero_queries(self) -> None:
        cfg = _mk_cfg()  # per_model_daily_ceiling_usd == {} (stock)
        fake = _fake_decision()
        cost_store = MagicMock()
        cost_store.model_cost_in_window = AsyncMock(return_value=0.0)
        with patch(f'{_HELPER}.resolve_route', return_value=fake):
            await resolve_and_record_route(
                role_name='steward',
                role_defaults=_STEWARD_DEFAULTS,
                config=cfg,
                task_id='2537',
                task_metadata={},
                cost_store=cost_store,
            )
        cost_store.model_cost_in_window.assert_not_awaited()

    async def test_configured_ceiling_awaits_query_for_that_model(self) -> None:
        cfg = _mk_cfg(ceilings={'opus': 50.0})
        fake = _fake_decision()
        cost_store = MagicMock()
        cost_store.model_cost_in_window = AsyncMock(return_value=0.0)
        with patch(f'{_HELPER}.resolve_route', return_value=fake) as mock_resolve:
            await resolve_and_record_route(
                role_name='steward',
                role_defaults=_STEWARD_DEFAULTS,
                config=cfg,
                task_id='2537',
                task_metadata={},
                cost_store=cost_store,
            )
        cost_store.model_cost_in_window.assert_awaited_once()
        assert cost_store.model_cost_in_window.call_args.args[0] == 'opus'
        # The gathered spend flows into RouteInputs.spend_by_model.
        assert mock_resolve.call_args.args[0].spend_by_model == {'opus': 0.0}

    async def test_no_cost_store_fires_zero_queries(self) -> None:
        cfg = _mk_cfg(ceilings={'opus': 50.0})
        fake = _fake_decision()
        with patch(f'{_HELPER}.resolve_route', return_value=fake) as mock_resolve:
            await resolve_and_record_route(
                role_name='steward',
                role_defaults=_STEWARD_DEFAULTS,
                config=cfg,
                task_id='2537',
                task_metadata={},
                cost_store=None,
            )
        assert mock_resolve.call_args.args[0].spend_by_model == {}
