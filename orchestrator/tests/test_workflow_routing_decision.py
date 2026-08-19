"""Tests for TaskWorkflow._invoke's routing_decision telemetry + metadata
mirror (plans/adaptive-model-routing-prd.md task γ, task 2533).

``_invoke`` (workflow.py:7853) is the shared per-invocation chokepoint.  Task
γ adds: (1) a ``routing_decision`` event emitted once per invocation with the
resolved model/effort/budget_usd/max_turns, source_layer, rule_id,
routing_tier, and an inputs digest; (2) a best-effort ``metadata.routing``
mirror (latest decision + bounded history) written via
``scheduler.update_task(metadata_mode='merge')`` and mirrored in-memory onto
``self.task['metadata']['routing']``.

RED phase (step-9): ``TaskWorkflow._record_routing_decision`` does not exist
yet and ``_invoke`` never calls it, so no ``routing_decision`` event is
emitted and ``scheduler.update_task`` is never awaited for a ``'routing'``
key — all four test bodies below fail today.

RED phase (step-13, task epsilon): ``_invoke`` does not yet call
``orchestrator.routing.resolve_route`` — it still resolves model/effort/
budget/max_turns via the pre-epsilon inline block + ``_select_model_for_role``
(workflow.py:7801). The ``TestInvokeAdoptsResolveRoute``,
``TestInvokeRoutingDecisionRejectedField``, ``TestInvokeCeilingSpendQuery``,
and ``TestSelectModelForRoleRetired`` classes below fail until step-14 wires
``resolve_route`` into ``_invoke`` and deletes ``_select_model_for_role``.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from _recording_event_store import _RecordingEventStore

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import IMPLEMENTER
from orchestrator.config import OrchestratorConfig, PriceEntry, RoutingRule, RuleMatch, RuleSet
from orchestrator.event_store import EventType
from orchestrator.routing import PlanShape, RoutingDecision
from orchestrator.workflow import TaskWorkflow

# Distinct, concrete values (not the role defaults) so a GREEN pass proves the
# routing_decision payload reflects the actually-resolved config, not a
# hardcoded fallback.
_BUDGET_USD = 10.0
_MAX_TURNS = 80

# Transcribes defaults.yaml's `rust-large-plan-implementer` rule (step-10)
# byte-for-byte, so tests that exercise the real (unpatched) resolve_route
# reproduce the same policy-rule behaviour production config ships.
_RUST_RULE = RoutingRule(
    id='rust-large-plan-implementer',
    match=RuleMatch(
        role=['implementer', 'debugger'],
        plan_min_steps=12,
        plan_min_modules=3,
        module_prefix='crates/',
    ),
    set=RuleSet(model='opus'),
)


def _make_workflow(
    *,
    event_store: _RecordingEventStore,
    cost_store: MagicMock | None = None,
) -> TaskWorkflow:
    """Minimal TaskWorkflow instance for ``_invoke`` routing-decision tests.

    Mirrors ``test_workflow_invocation_end_truthful.py``'s ``_make_workflow``
    (pydantic_spec spec_set MagicMock cfg, patched ``invoke_with_cap_retry`` +
    ``_build_agent_env``), plus concrete cfg.models/budgets/max_turns/effort/
    timeouts/backends.implementer values — needed here because the
    routing_decision payload asserts on the actual resolved model/effort/
    budget_usd/max_turns (a bare MagicMock attribute would satisfy the
    precedent test's assertions but not these, and a MagicMock timeout_val
    would blow up the ``'timeout=%.0fs'`` log-format call in ``_invoke``).
    ``scheduler.update_task`` is replaced with an ``AsyncMock`` so the
    metadata-mirror write is awaitable and assertable.

    ``cfg.routing.*`` is set to real (non-MagicMock) values — task epsilon's
    ``resolve_route`` does real membership/comparison ops
    (``candidate not in config.routing.allowed_models``, dict ``.get()``
    against ``per_model_daily_ceiling_usd``) that a bare MagicMock child
    attribute cannot satisfy (``pydantic_spec`` only constrains ``cfg``'s own
    top-level attribute names — ``cfg.routing`` itself is an unconstrained
    MagicMock unless configured here). ``rules`` carries the real
    ``rust-large-plan-implementer`` rule so pre-existing Rust-heuristic
    assertions below keep passing once ``_invoke`` calls the real resolver.
    ``per_model_daily_ceiling_usd`` defaults to ``{}`` (stock — no ceilings)
    so no test not opting into a ceiling pays a surprise cost_store read.
    """
    assignment = MagicMock()
    assignment.task_id = '2533'
    assignment.task = {'id': '2533', 'title': 'Test Task', 'description': 'd'}
    assignment.modules = []

    _spec = pydantic_spec(OrchestratorConfig)
    cfg = MagicMock(spec_set=_spec)
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.fused_memory.url = 'http://localhost:8002'
    cfg.max_review_cycles = 2
    cfg.max_amendment_rounds = 1
    cfg.lock_depth = 2
    cfg.steward_completion_timeout = 300.0
    cfg.timeouts.working_idle_secs = 999.0
    cfg.invocation_timeout = 8888.0

    cfg.models.implementer = 'sonnet'
    cfg.budgets.implementer = _BUDGET_USD
    cfg.max_turns.implementer = _MAX_TURNS
    cfg.effort.implementer = 'high'
    cfg.timeouts.implementer = 1200.0
    cfg.backends.implementer = 'claude'

    cfg.routing.allowed_models = ['haiku', 'sonnet', 'opus']
    cfg.routing.ladder = ['haiku', 'sonnet', 'opus']
    cfg.routing.per_model_daily_ceiling_usd = {}
    cfg.routing.rules = [_RUST_RULE]
    # These tests pass a plain tmp_path as cwd (not a real linked worktree) and
    # do not assert on sandbox wiring; disable the sandbox block so _invoke does
    # not call compute_write_set(cwd) on a non-worktree path (task 2905 α3).
    cfg.sandbox.enabled = False

    wf = TaskWorkflow(
        assignment=assignment,
        config=cfg,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
        event_store=event_store,  # type: ignore[arg-type]
        cost_store=cost_store,  # type: ignore[arg-type]
    )
    wf.scheduler.update_task = AsyncMock(return_value=True)
    return wf


def _stub_agent_result() -> AgentResult:
    return AgentResult(
        success=True,
        output='ok',
        timed_out=False,
        turns=3,
        cost_usd=1.0,
        duration_ms=1_000,
        transcript_turns=3,
    )


def _routing_decision_entries(rec: _RecordingEventStore) -> list[dict]:
    return [entry for (etype, entry) in rec.events if etype == EventType.routing_decision]


async def _invoke_implementer(wf: TaskWorkflow, cwd: Path, *, prompt: str = 'x') -> None:
    with (
        patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new=AsyncMock(return_value=_stub_agent_result()),
        ),
        patch.object(wf, '_build_agent_env', return_value=None),
    ):
        await wf._invoke(IMPLEMENTER, prompt=prompt, cwd=cwd)


@pytest.mark.asyncio
class TestInvokeRecordsRoutingDecisionEvent:
    """``_invoke`` emits exactly one ``routing_decision`` event per call."""

    async def test_config_layer_decision_is_recorded(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)

        await _invoke_implementer(wf, tmp_path)

        entries = _routing_decision_entries(rec)
        assert len(entries) == 1, (
            f'expected exactly one routing_decision event; got {rec.events!r}'
        )
        data = entries[0]['data']
        assert data['model'] == 'sonnet'
        assert data['effort'] == 'high'
        assert data['budget_usd'] == _BUDGET_USD
        assert data['max_turns'] == _MAX_TURNS
        assert data['source_layer'] == 'config'
        assert data['rule_id'] is None
        assert data['routing_tier'] == 0
        assert isinstance(data['inputs_digest'], str) and data['inputs_digest']

    async def test_rust_upgrade_case_records_policy_rule(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)
        wf.modules = ['crates/a', 'crates/b', 'crates/c']
        wf.plan = {'steps': [{}] * 12}

        await _invoke_implementer(wf, tmp_path)

        entries = _routing_decision_entries(rec)
        assert len(entries) == 1
        data = entries[0]['data']
        assert data['model'] == 'opus'
        assert data['source_layer'] == 'policy_rule'
        assert data['rule_id'] == 'rust-large-plan-implementer'


@pytest.mark.asyncio
class TestInvokeMirrorsRoutingMetadata:
    """``_invoke`` best-effort mirrors the decision onto ``metadata.routing``."""

    async def test_mirror_written_via_scheduler_and_in_memory(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)

        await _invoke_implementer(wf, tmp_path)

        update_task_mock = cast(AsyncMock, wf.scheduler.update_task)
        update_task_mock.assert_awaited_once()
        call_args = update_task_mock.call_args
        assert call_args.kwargs.get('metadata_mode') == 'merge'
        payload = call_args.args[1]
        assert payload['routing']['latest']['model'] == 'sonnet'

        assert wf.task['metadata']['routing']['latest']['model'] == 'sonnet'

    async def test_successive_invocations_accumulate_history(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)

        await _invoke_implementer(wf, tmp_path, prompt='first')
        await _invoke_implementer(wf, tmp_path, prompt='second')

        routing = wf.task['metadata']['routing']
        assert len(routing['history']) == 2
        assert routing['latest'] == routing['history'][-1]


@pytest.mark.asyncio
class TestInvokeRoutingDecisionGracefulDegradation:
    """Routing telemetry must never block or crash ``_invoke`` (PRD γ).

    ``_record_routing_decision`` is built around three independent failure
    boundaries: the scheduler mirror write, an absent event store, and the
    decision-record construction itself. Each must degrade silently without
    affecting the invocation's own success/failure.
    """

    async def test_scheduler_failure_still_updates_in_memory_metadata(
        self, tmp_path: Path,
    ) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)
        wf.scheduler.update_task = AsyncMock(side_effect=RuntimeError('boom'))

        await _invoke_implementer(wf, tmp_path)

        # The event still fires and the in-memory mirror is still updated
        # even though the awaited scheduler write raised.
        assert len(_routing_decision_entries(rec)) == 1
        assert wf.task['metadata']['routing']['latest']['model'] == 'sonnet'

    async def test_missing_event_store_does_not_raise(self, tmp_path: Path) -> None:
        wf = _make_workflow(event_store=_RecordingEventStore())
        wf.event_store = None

        await _invoke_implementer(wf, tmp_path)

        update_task_mock = cast(AsyncMock, wf.scheduler.update_task)
        update_task_mock.assert_awaited_once()
        assert wf.task['metadata']['routing']['latest']['model'] == 'sonnet'

    async def test_decision_build_failure_does_not_raise(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)

        with patch(
            'orchestrator.workflow.RoutingDecisionMirror',
            side_effect=RuntimeError('boom'),
        ):
            await _invoke_implementer(wf, tmp_path)

        # The build failure is caught before either write is attempted, so
        # neither the event nor the metadata mirror (in-memory or scheduler)
        # is touched.
        assert _routing_decision_entries(rec) == []
        cast(AsyncMock, wf.scheduler.update_task).assert_not_awaited()
        assert 'metadata' not in wf.task


@pytest.mark.asyncio
class TestInvokeAdoptsResolveRoute:
    """``_invoke`` resolves via ``orchestrator.routing.resolve_route`` BEFORE
    invoking, and the returned ``RoutingDecision`` feeds the invocation
    (PRD epsilon invariant 9 — resolution-before-invocation)."""

    async def test_resolve_route_called_and_feeds_the_invocation(
        self, tmp_path: Path,
    ) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)
        wf.modules = ['crates/a', 'crates/b']
        wf.plan = {'steps': [{}] * 7}

        fake_decision = RoutingDecision(
            model='haiku',
            effort='low',
            budget_usd=1.23,
            max_turns=17,
            source_layer='policy_rule',
            rule_id='some-rule',
            rejected=(),
        )
        with (
            patch(
                'orchestrator.workflow.resolve_route',
                return_value=fake_decision,
            ) as mock_resolve,
            patch(
                'orchestrator.workflow.invoke_with_cap_retry',
                new=AsyncMock(return_value=_stub_agent_result()),
            ) as mock_invoke,
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            await wf._invoke(IMPLEMENTER, prompt='x', cwd=tmp_path)

        mock_resolve.assert_called_once()
        route_inputs = mock_resolve.call_args.args[0]
        assert route_inputs.role_name == 'implementer'
        assert route_inputs.plan_shape == PlanShape(7, ('crates/a', 'crates/b'))
        assert route_inputs.routing_tier == 0
        assert mock_resolve.call_args.args[1] is wf.config

        assert mock_invoke.call_args.kwargs['model'] == 'haiku'
        assert mock_invoke.call_args.kwargs['effort'] == 'low'
        assert mock_invoke.call_args.kwargs['max_budget_usd'] == 1.23
        assert mock_invoke.call_args.kwargs['max_turns'] == 17


@pytest.mark.asyncio
class TestInvokeThreadsConfigPrices:
    """Defect (c) — ``_invoke`` threads ``config.prices`` into the
    invocation so the operator-tunable/hot-reloadable price table reaches
    the codex/gemini/pi cost estimators (claude ignores ``prices`` — it
    reports native cost; see orchestrator.agents.invoke._estimate_cost).
    ``prices`` rides the same **invoke_kwargs forwarding path already
    proven for ``backend`` (task 2457) — it is not an explicit
    ``invoke_with_cap_retry`` param.
    """

    async def test_prices_forwarded_to_invoke_with_cap_retry(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)
        wf.config.prices = {'gpt-5.4': PriceEntry(input_per_1m=1.0, output_per_1m=2.0)}

        with (
            patch(
                'orchestrator.workflow.invoke_with_cap_retry',
                new=AsyncMock(return_value=_stub_agent_result()),
            ) as mock_invoke,
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            await wf._invoke(IMPLEMENTER, prompt='x', cwd=tmp_path)

        assert mock_invoke.call_args.kwargs.get('prices') is wf.config.prices


@pytest.mark.asyncio
class TestInvokeRoutingDecisionRejectedField:
    """The ``routing_decision`` event + ``metadata.routing`` mirror carry
    the resolver's ``source_layer``/``rule_id``/``rejected`` verbatim."""

    async def test_rejected_and_provenance_flow_through(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)

        fake_decision = RoutingDecision(
            model='sonnet',
            effort='high',
            budget_usd=_BUDGET_USD,
            max_turns=_MAX_TURNS,
            source_layer='policy_rule',
            rule_id='some-rule',
            rejected=('metadata_override:model-not-in-allowlist',),
        )
        with (
            patch('orchestrator.workflow.resolve_route', return_value=fake_decision),
            patch(
                'orchestrator.workflow.invoke_with_cap_retry',
                new=AsyncMock(return_value=_stub_agent_result()),
            ),
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            await wf._invoke(IMPLEMENTER, prompt='x', cwd=tmp_path)

        entries = _routing_decision_entries(rec)
        assert len(entries) == 1
        data = entries[0]['data']
        assert data['source_layer'] == 'policy_rule'
        assert data['rule_id'] == 'some-rule'
        assert data['rejected'] == ['metadata_override:model-not-in-allowlist']

        latest = wf.task['metadata']['routing']['latest']
        assert latest['source_layer'] == 'policy_rule'
        assert latest['rule_id'] == 'some-rule'
        assert latest['rejected'] == ['metadata_override:model-not-in-allowlist']


@pytest.mark.asyncio
class TestInvokeCeilingSpendQuery:
    """``_invoke`` queries ``cost_store.model_cost_in_window`` only for
    models carrying a configured ``routing.per_model_daily_ceiling_usd``
    entry (PRD epsilon invariant 6 mechanics + byte-equivalence)."""

    async def test_stock_ceiling_config_fires_zero_queries(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        cost_store = MagicMock()
        cost_store.model_cost_in_window = AsyncMock(return_value=0.0)
        wf = _make_workflow(event_store=rec, cost_store=cost_store)
        # _make_workflow defaults routing.per_model_daily_ceiling_usd to {} —
        # stock config, no ceilings configured.

        await _invoke_implementer(wf, tmp_path)

        cost_store.model_cost_in_window.assert_not_awaited()

    async def test_configured_ceiling_awaits_query_for_that_model(
        self, tmp_path: Path,
    ) -> None:
        rec = _RecordingEventStore()
        cost_store = MagicMock()
        cost_store.model_cost_in_window = AsyncMock(return_value=0.0)
        wf = _make_workflow(event_store=rec, cost_store=cost_store)
        wf.config.routing.per_model_daily_ceiling_usd = {'opus': 50.0}

        await _invoke_implementer(wf, tmp_path)

        cost_store.model_cost_in_window.assert_awaited_once()
        call_args = cost_store.model_cost_in_window.call_args
        assert call_args.args[0] == 'opus'


@pytest.mark.asyncio
class TestInvokeThreadsScopeCapacity:
    """δ: ``_invoke`` threads the gate's advisory ``scope_capacity_snapshot()``
    into ``RouteInputs.scope_capacity`` (None when no gate is wired), and the
    resolver's ``model-capacity-exhausted`` rejection surfaces end-to-end
    (invariants S7/S8, PRD boundary test B7)."""

    async def test_gate_snapshot_threaded_into_route_inputs(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)
        gate = MagicMock()
        gate.scope_capacity_snapshot.return_value = {'claude-fable-5': False}
        wf.usage_gate = gate

        fake_decision = RoutingDecision(
            model='sonnet',
            effort='high',
            budget_usd=_BUDGET_USD,
            max_turns=_MAX_TURNS,
            source_layer='config',
            rule_id=None,
            rejected=(),
        )
        with (
            patch(
                'orchestrator.workflow.resolve_route',
                return_value=fake_decision,
            ) as mock_resolve,
            patch(
                'orchestrator.workflow.invoke_with_cap_retry',
                new=AsyncMock(return_value=_stub_agent_result()),
            ),
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            await wf._invoke(IMPLEMENTER, prompt='x', cwd=tmp_path)

        route_inputs = mock_resolve.call_args.args[0]
        assert route_inputs.scope_capacity == {'claude-fable-5': False}
        gate.scope_capacity_snapshot.assert_called_once()

    async def test_no_gate_threads_none_and_does_not_raise(self, tmp_path: Path) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)
        assert wf.usage_gate is None  # _make_workflow does not wire a gate

        fake_decision = RoutingDecision(
            model='sonnet',
            effort='high',
            budget_usd=_BUDGET_USD,
            max_turns=_MAX_TURNS,
            source_layer='config',
            rule_id=None,
            rejected=(),
        )
        with (
            patch(
                'orchestrator.workflow.resolve_route',
                return_value=fake_decision,
            ) as mock_resolve,
            patch(
                'orchestrator.workflow.invoke_with_cap_retry',
                new=AsyncMock(return_value=_stub_agent_result()),
            ),
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            await wf._invoke(IMPLEMENTER, prompt='x', cwd=tmp_path)

        route_inputs = mock_resolve.call_args.args[0]
        assert route_inputs.scope_capacity is None

    async def test_b7_capacity_exhausted_falls_to_config_end_to_end(
        self, tmp_path: Path,
    ) -> None:
        rec = _RecordingEventStore()
        wf = _make_workflow(event_store=rec)
        # Admit fable to the allowlist and clear policy rules so the config
        # model (`sonnet`) is the deterministic one-layer-down fallback.
        wf.config.routing.allowed_models = ['haiku', 'sonnet', 'opus', 'claude-fable-5']
        wf.config.routing.rules = []
        wf.task['metadata'] = {'model_overrides': {'implementer': 'claude-fable-5'}}
        gate = MagicMock()
        gate.scope_capacity_snapshot.return_value = {'claude-fable-5': False}
        wf.usage_gate = gate

        with (
            patch(
                'orchestrator.workflow.invoke_with_cap_retry',
                new=AsyncMock(return_value=_stub_agent_result()),
            ) as mock_invoke,
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            await wf._invoke(IMPLEMENTER, prompt='x', cwd=tmp_path)

        entries = _routing_decision_entries(rec)
        assert len(entries) == 1
        data = entries[0]['data']
        assert 'metadata_override:model-capacity-exhausted' in data['rejected']
        assert data['model'] == 'sonnet'
        assert data['source_layer'] == 'config'
        # Dispatch proceeded with the fallback model (S7: capacity never blocks).
        mock_invoke.assert_awaited_once()
        assert mock_invoke.call_args.kwargs['model'] == 'sonnet'


class TestSelectModelForRoleRetired:
    """``_select_model_for_role`` is retired — its Rust heuristic now ships
    as defaults.yaml's ``rust-large-plan-implementer`` policy rule, applied
    via ``resolve_route`` (task epsilon)."""

    def test_select_model_for_role_no_longer_exists(self) -> None:
        assert not hasattr(TaskWorkflow, '_select_model_for_role')
