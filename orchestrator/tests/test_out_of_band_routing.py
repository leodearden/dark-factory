"""Per-site adoption + parity tests for the out-of-band routing helper (task η,
plans/adaptive-model-routing-prd.md).

Each of the five out-of-band LLM dispatch sites now resolves its route through
``orchestrator.routing_dispatch.resolve_and_record_route`` (which calls
``orchestrator.routing.resolve_route`` and emits a ``routing_decision`` event),
instead of reading (model, effort, budget, max_turns) straight from config.
Backend stays each site's own ``config.backends.*`` / ``ua_cfg.backend`` read
(resolver-external, PRD boundary under harness-backend-reconnect-pi).

This file grows one section per site as task η's steps land:
* steward main invoke + inner triage (steps 3-4)
* deep_reviewer (steps 5-6)
* module_tagger (steps 7-8)
* unblock_auto (steps 9-10)
* PRD boundary test 9 + cross-site byte-equivalence (steps 11-12)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import (
    pydantic_spec,
    skip_role_config_layer,
    stamp_stock_routing_config,
)
from _recording_event_store import _RecordingEventStore
from escalation.models import Escalation

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.review_checkpoint import ReviewCheckpoint
from orchestrator.routing import RoutingDecision
from orchestrator.steward import TaskSteward
from orchestrator.verify import VerifyResult


def _routing_entries(rec: _RecordingEventStore) -> list[dict]:
    return [entry for (etype, entry) in rec.events if etype == EventType.routing_decision]


def _stub_result():
    from shared.cli_invoke import AgentResult
    return AgentResult(
        success=True,
        output='done',
        cost_usd=1.0,
        duration_ms=1000,
        turns=3,
        session_id=None,
        timed_out=False,
    )


def _fake_decision(*, model='sonnet', effort='low', budget_usd=1.23, max_turns=17,
                   source_layer='config') -> RoutingDecision:
    return RoutingDecision(
        model=model, effort=effort, budget_usd=budget_usd, max_turns=max_turns,
        source_layer=source_layer, rule_id=None, rejected=(),
    )


# ---------------------------------------------------------------------------
# Steward (steps 3-4)
# ---------------------------------------------------------------------------


def _steward_config():
    cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    cfg.project_root = Path('/tmp/fake-project')
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.fused_memory.url = 'http://localhost:8002'
    cfg.models.steward = 'opus'
    cfg.budgets.steward = 5.0
    cfg.max_turns.steward = 100
    cfg.effort.steward = 'high'
    cfg.backends.steward = 'claude'
    cfg.timeouts.steward = 1800.0
    cfg.models.triage = 'sonnet'
    cfg.budgets.triage = 2.0
    cfg.max_turns.triage = 25
    cfg.effort.triage = 'medium'
    cfg.backends.triage = 'claude'
    cfg.escalation.host = 'localhost'
    cfg.escalation.port = 8102
    stamp_stock_routing_config(cfg)
    return cfg


def _build_steward(worktree: Path, *, task: dict, event_store=None, cost_store=None):
    briefing = AsyncMock()
    briefing.build_steward_initial_prompt.return_value = 'Full briefing.'
    briefing.build_steward_continuation_prompt.return_value = 'Continuation.'
    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}
    queue = MagicMock()
    queue.get_by_task.return_value = []
    return TaskSteward(
        task_id=task['id'],
        task=task,
        worktree=worktree,
        config=_steward_config(),
        mcp=mcp,
        escalation_queue=queue,
        briefing=briefing,
        event_store=event_store,
        cost_store=cost_store,
    )


def _mk_escalation(**overrides) -> Escalation:
    defaults: dict = dict(
        id='esc-42-1', task_id='42', agent_role='orchestrator',
        severity='blocking', category='limit_exhausted', summary='s',
    )
    defaults.update(overrides)
    return Escalation(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestStewardMainAdoptsResolveRoute:
    """steward main invoke (`_invoke_with_session`) resolves via the helper."""

    async def test_decision_feeds_invoke_and_keeps_budget_and_backend(
        self, tmp_path: Path,
    ) -> None:
        wt = tmp_path / 'wt'
        wt.mkdir()
        task = {'id': '42', 'title': 't', 'description': 'd',
                'metadata': {'dispatch_count': 2}}
        steward = _build_steward(wt, task=task)
        fake = _fake_decision(model='sonnet', effort='low', max_turns=17)

        with (
            patch('orchestrator.steward.resolve_and_record_route',
                  new=AsyncMock(return_value=fake)) as mock_helper,
            patch('orchestrator.steward.invoke_with_cap_retry',
                  new=AsyncMock(return_value=_stub_result())) as mock_iwcr,
        ):
            await steward._invoke_with_session(
                prompt='p', cwd=wt, mcp_config={'mcpServers': {}},
                per_invocation_budget=3.0, escalation=_mk_escalation(),
            )

        mock_helper.assert_awaited_once()
        hk = mock_helper.call_args.kwargs
        assert hk['role_name'] == 'steward'
        assert hk['task_metadata'] == {'dispatch_count': 2}
        assert hk['task_id'] == '42'

        kwargs = mock_iwcr.call_args.kwargs
        assert kwargs['model'] == 'sonnet'
        assert kwargs['effort'] == 'low'
        assert kwargs['max_turns'] == 17
        # Budget (lifetime-capped per-invocation) + backend are NOT resolver-owned.
        assert kwargs['max_budget_usd'] == pytest.approx(3.0)
        assert kwargs['backend'] == 'claude'

    async def test_emits_one_routing_decision_event(self, tmp_path: Path) -> None:
        wt = tmp_path / 'wt'
        wt.mkdir()
        rec = _RecordingEventStore()
        task = {'id': '42', 'title': 't', 'description': 'd'}
        steward = _build_steward(wt, task=task, event_store=rec)

        with patch('orchestrator.steward.invoke_with_cap_retry',
                   new=AsyncMock(return_value=_stub_result())):
            await steward._invoke_with_session(
                prompt='p', cwd=wt, mcp_config={'mcpServers': {}},
                per_invocation_budget=3.0, escalation=_mk_escalation(),
            )

        entries = _routing_entries(rec)
        assert len(entries) == 1
        assert entries[0]['data']['role'] == 'steward'
        # Stock config, no override → config-layer model.
        assert entries[0]['data']['model'] == 'opus'

    async def test_model_override_wins_boundary_test_9(self, tmp_path: Path) -> None:
        wt = tmp_path / 'wt'
        wt.mkdir()
        rec = _RecordingEventStore()
        task = {'id': '42', 'title': 't', 'description': 'd',
                'metadata': {'model_overrides': {'steward': 'haiku'}}}
        steward = _build_steward(wt, task=task, event_store=rec)

        with patch('orchestrator.steward.invoke_with_cap_retry',
                   new=AsyncMock(return_value=_stub_result())) as mock_iwcr:
            await steward._invoke_with_session(
                prompt='p', cwd=wt, mcp_config={'mcpServers': {}},
                per_invocation_budget=3.0, escalation=_mk_escalation(),
            )

        assert mock_iwcr.call_args.kwargs['model'] == 'haiku'
        assert _routing_entries(rec)[0]['data']['source_layer'] == 'metadata_override'


@pytest.mark.asyncio
class TestStewardTriageAdoptsResolveRoute:
    """steward inner triage (`_pre_triage_suggestions`) resolves via the helper."""

    @pytest.fixture(autouse=True)
    def _mock_triage(self):
        from orchestrator.agents.triage import TRIAGE as _REAL_TRIAGE
        triage_mod = MagicMock()
        triage_mod.TRIAGE = _REAL_TRIAGE
        triage_mod.build_triage_prompt = MagicMock(return_value='triage prompt')
        triage_mod.extract_triage_verdict = MagicMock(return_value=None)
        triage_mod.format_pretriaged_detail = MagicMock(return_value='## Pre-triaged')
        with patch.dict(sys.modules, {'orchestrator.agents.triage': triage_mod}):
            yield triage_mod

    @staticmethod
    def _esc(n: int = 12) -> Escalation:
        suggestions = [
            {'description': f's{i}', 'location': f'f{i}.py',
             'reviewer': 'bot', 'category': 'style'}
            for i in range(n)
        ]
        return _mk_escalation(detail=json.dumps(suggestions), category='review_suggestions')

    def _wt(self, tmp_path: Path) -> Path:
        wt = tmp_path / 'worktree'
        wt.mkdir()
        (wt / '.task').mkdir()
        return wt

    async def test_decision_feeds_triage_invoke_and_keeps_backend(
        self, tmp_path: Path,
    ) -> None:
        wt = self._wt(tmp_path)
        task = {'id': '42', 'title': 't', 'description': 'd'}
        steward = _build_steward(wt, task=task)
        fake = _fake_decision(model='sonnet', effort='medium', budget_usd=2.5, max_turns=25)

        with (
            patch('orchestrator.steward.resolve_and_record_route',
                  new=AsyncMock(return_value=fake)) as mock_helper,
            patch('orchestrator.steward.invoke_with_cap_retry',
                  new=AsyncMock(return_value=_stub_result())) as mock_iwcr,
        ):
            await steward._pre_triage_suggestions(self._esc())

        mock_helper.assert_awaited_once()
        assert mock_helper.call_args.kwargs['role_name'] == 'triage'

        kwargs = mock_iwcr.call_args.kwargs
        assert kwargs['model'] == 'sonnet'
        assert kwargs['effort'] == 'medium'
        assert kwargs['max_budget_usd'] == 2.5
        assert kwargs['max_turns'] == 25
        assert kwargs['backend'] == 'claude'

    async def test_emits_one_routing_decision_event(self, tmp_path: Path) -> None:
        wt = self._wt(tmp_path)
        rec = _RecordingEventStore()
        task = {'id': '42', 'title': 't', 'description': 'd'}
        steward = _build_steward(wt, task=task, event_store=rec)

        with patch('orchestrator.steward.invoke_with_cap_retry',
                   new=AsyncMock(return_value=_stub_result())):
            await steward._pre_triage_suggestions(self._esc())

        entries = _routing_entries(rec)
        assert len(entries) == 1
        assert entries[0]['data']['role'] == 'triage'
        assert entries[0]['data']['model'] == 'sonnet'


# ---------------------------------------------------------------------------
# deep_reviewer (steps 5-6)
# ---------------------------------------------------------------------------


_REVIEW_PHASE1 = VerifyResult(
    passed=True, summary='OK', test_output='', lint_output='', type_output='',
)


def _review_config():
    cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    cfg.project_root = Path('/tmp/fake-project')
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.models.deep_reviewer = 'opus'
    cfg.budgets.deep_reviewer = 10.0
    cfg.max_turns.deep_reviewer = 50
    cfg.effort.deep_reviewer = 'max'
    cfg.backends.deep_reviewer = 'claude'
    cfg.escalation.host = 'localhost'
    cfg.escalation.port = 8102
    stamp_stock_routing_config(cfg)
    return cfg


def _build_checkpoint(*, event_store=None, cost_store=None) -> ReviewCheckpoint:
    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}
    cp = ReviewCheckpoint(_review_config(), mcp, None)
    cp.event_store = event_store
    cp.cost_store = cost_store
    return cp


@pytest.mark.asyncio
class TestDeepReviewerAdoptsResolveRoute:
    """deep_reviewer (`ReviewCheckpoint._run_review`) resolves via the helper.

    deep_reviewer is STRUCTURALLY project-level — ``_run_review(mode, modules)``
    has no task/task_metadata anywhere in scope (callers ``run_full()`` /
    ``run_focused()`` pass module-path lists, not tasks) — so the helper is
    called with no task_id/task_metadata/scheduler/in_memory_task (no
    ``metadata.routing`` mirror); it resolves at the config/role_default/
    policy_rule layers and emits a routing_decision when an event_store is
    wired. The ``metadata_override`` layer is unreachable here (proven at the
    shared-helper unit level in test_routing_dispatch.py instead).
    """

    async def test_decision_feeds_invoke_and_keeps_backend(self) -> None:
        cp = _build_checkpoint()
        fake = _fake_decision(model='sonnet', effort='low', budget_usd=3.5, max_turns=42)
        with (
            patch('orchestrator.review_checkpoint.resolve_and_record_route',
                  new=AsyncMock(return_value=fake)) as mock_helper,
            patch('orchestrator.review_checkpoint.invoke_with_cap_retry',
                  new=AsyncMock(return_value=_stub_result())) as mock_iwcr,
            patch('orchestrator.review_checkpoint.run_full_verification',
                  new=AsyncMock(return_value=_REVIEW_PHASE1)),
            patch.object(ReviewCheckpoint, '_save_report', lambda *a, **k: None),
        ):
            await cp._run_review('focused', ['orchestrator/foo.py'])

        mock_helper.assert_awaited_once()
        hk = mock_helper.call_args.kwargs
        assert hk['role_name'] == 'deep_reviewer'
        # Project-level: no task identity flows into resolution / no mirror sink.
        assert hk.get('task_id') is None
        assert hk.get('task_metadata') is None
        assert hk.get('scheduler') is None
        assert hk.get('in_memory_task') is None

        kwargs = mock_iwcr.call_args.kwargs
        assert kwargs['model'] == 'sonnet'
        assert kwargs['effort'] == 'low'
        assert kwargs['max_budget_usd'] == 3.5
        assert kwargs['max_turns'] == 42
        # backend stays the site's own config.backends.deep_reviewer read.
        assert kwargs['backend'] == 'claude'

    async def test_emits_one_routing_decision_event_stock_config(self) -> None:
        rec = _RecordingEventStore()
        cp = _build_checkpoint(event_store=rec)
        with (
            patch('orchestrator.review_checkpoint.invoke_with_cap_retry',
                  new=AsyncMock(return_value=_stub_result())) as mock_iwcr,
            patch('orchestrator.review_checkpoint.run_full_verification',
                  new=AsyncMock(return_value=_REVIEW_PHASE1)),
            patch.object(ReviewCheckpoint, '_save_report', lambda *a, **k: None),
        ):
            await cp._run_review('full', [])

        entries = _routing_entries(rec)
        assert len(entries) == 1
        assert entries[0]['data']['role'] == 'deep_reviewer'
        # Stock config, no override, no rule → config-layer resolution == pre-η.
        assert entries[0]['data']['model'] == 'opus'
        assert entries[0]['data']['source_layer'] == 'config'
        # Byte-equivalence: the resolved invoke values equal config.*.deep_reviewer.
        kwargs = mock_iwcr.call_args.kwargs
        assert kwargs['model'] == 'opus'
        assert kwargs['effort'] == 'max'
        assert kwargs['max_budget_usd'] == 10.0
        assert kwargs['max_turns'] == 50

    async def test_no_event_and_no_raise_when_event_store_none(self) -> None:
        cp = _build_checkpoint(event_store=None)
        with (
            patch('orchestrator.review_checkpoint.invoke_with_cap_retry',
                  new=AsyncMock(return_value=_stub_result())) as mock_iwcr,
            patch('orchestrator.review_checkpoint.run_full_verification',
                  new=AsyncMock(return_value=_REVIEW_PHASE1)),
            patch.object(ReviewCheckpoint, '_save_report', lambda *a, **k: None),
        ):
            # Must not raise despite no event_store wired.
            await cp._run_review('full', [])
        # Still resolves through the resolver to the stock config value.
        assert mock_iwcr.call_args.kwargs['model'] == 'opus'

    async def test_cost_row_model_equals_decision_model(self) -> None:
        cost_store = MagicMock()
        cost_store.save_invocation = AsyncMock()
        cp = _build_checkpoint(cost_store=cost_store)
        fake = _fake_decision(model='sonnet')
        with (
            patch('orchestrator.review_checkpoint.resolve_and_record_route',
                  new=AsyncMock(return_value=fake)),
            patch('orchestrator.review_checkpoint.invoke_with_cap_retry',
                  new=AsyncMock(return_value=_stub_result())),
            patch('orchestrator.review_checkpoint.run_full_verification',
                  new=AsyncMock(return_value=_REVIEW_PHASE1)),
            patch.object(ReviewCheckpoint, '_save_report', lambda *a, **k: None),
        ):
            await cp._run_review('full', [])

        cost_store.save_invocation.assert_awaited_once()
        # The persisted cost row's model is the RESOLVED model, not the raw
        # config.models.deep_reviewer (they diverge under a rule/override).
        assert cost_store.save_invocation.call_args.kwargs['model'] == 'sonnet'


class TestReviewCheckpointEventStoreAttribute:
    """ReviewCheckpoint gains ``event_store`` as a settable attr (default None),
    wired post-construction in the harness (like cost_store/run_id) — so the
    ``ReviewCheckpoint(config, mcp, usage_gate)`` constructor is unchanged."""

    def test_construction_unchanged_and_event_store_defaults_none(self) -> None:
        mcp = MagicMock()
        mcp.mcp_config_json.return_value = {'mcpServers': {}}
        # (config, mcp, usage_gate) positional signature is unchanged.
        cp = ReviewCheckpoint(_review_config(), mcp, None)
        assert cp.event_store is None


# ---------------------------------------------------------------------------
# unblock_auto (steps 9-10)
# ---------------------------------------------------------------------------


def _unblock_config():
    """Mock config for the autonomous dry-run unblock site.

    ``unblock_auto``'s model/effort/budget/turns live on the top-level
    ``config.unblock_auto`` block, NOT on the role-keyed ``config.models`` /
    ``budgets`` / ``max_turns`` / ``effort`` sub-models (which have no
    ``unblock_auto`` field) — so ``resolve_route``'s layer-3 config read is
    SKIPPED and the ``RoleDefaults(ua_cfg.*)`` base (layer-4) stands.
    ``skip_role_config_layer`` reproduces that production shape (empty-spec
    sub-models → ``hasattr(cfg.models,'unblock_auto')`` False, ``getattr(
    cfg.backends,'unblock_auto','claude')`` → 'claude'), so the site resolves
    at ``source_layer='role_default'`` — byte-equivalent to pre-η.
    """
    cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.unblock_auto.enabled = True
    cfg.unblock_auto.model = 'sonnet'
    cfg.unblock_auto.effort = 'high'
    cfg.unblock_auto.budget_usd = 5.0
    cfg.unblock_auto.max_turns = 50
    cfg.unblock_auto.backend = 'claude'
    cfg.unblock_auto.timeout_seconds = 600.0
    cfg.unblock_auto.b3_proposal_keep_last = 5
    cfg.timeouts.working_idle_secs = 1800.0
    cfg.invocation_timeout = 7200.0
    skip_role_config_layer(cfg)
    stamp_stock_routing_config(cfg)
    return cfg


async def _run_unblock(*, config, scheduler, worktree: Path,
                       event_store=None, cost_store=None, task_id='55'):
    from orchestrator.dry_run_unblock import run_dry_run_unblock
    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}
    await run_dry_run_unblock(
        task_id=task_id,
        worktree=str(worktree),
        reason='verify exhausted',
        detail='',
        scheduler=scheduler,
        mcp=mcp,
        config=config,
        event_store=event_store,
        cost_store=cost_store,
    )


def _routing_mirror_calls(scheduler) -> list:
    """The ``scheduler.update_task`` calls that mirror ``metadata.routing``
    (positional ``(task_id, {'routing': …})`` + ``metadata_mode='merge'``) —
    distinct from the dry-run proposal persist (``append=True``) and the
    keep-last trim (positional metadata, no mode)."""
    return [
        c for c in scheduler.update_task.call_args_list
        if c.kwargs.get('metadata_mode') == 'merge'
        and len(c.args) > 1
        and isinstance(c.args[1], dict)
        and 'routing' in c.args[1]
    ]


@pytest.mark.asyncio
class TestUnblockAutoAdoptsResolveRoute:
    """The autonomous dry-run unblock hook (`run_dry_run_unblock`) resolves its
    route through the shared helper.

    unblock_auto is genuinely per-task (has task_id + scheduler + event_store),
    so it best-effort fetches the task's metadata via ``scheduler.get_task``
    (fail-safe → {}) to honor ``model_overrides['unblock_auto']``/tier, emits a
    routing_decision event, and mirrors ``metadata.routing`` via
    ``scheduler.update_task(metadata_mode='merge')``. Backend stays the site's
    own ``ua_cfg.backend`` read (resolver-external).
    """

    async def test_decision_feeds_invoke_and_keeps_backend(
        self, tmp_path: Path,
    ) -> None:
        wt = tmp_path / 'wt'
        wt.mkdir()
        scheduler = MagicMock()
        scheduler.get_task = AsyncMock(return_value={'metadata': {'dispatch_count': 1}})
        scheduler.update_task = AsyncMock()
        fake = _fake_decision(model='sonnet', effort='low', budget_usd=4.2, max_turns=33)

        with (
            patch('orchestrator.dry_run_unblock.resolve_and_record_route',
                  new=AsyncMock(return_value=fake)) as mock_helper,
            patch('orchestrator.dry_run_unblock.invoke_with_cap_retry',
                  new=AsyncMock(return_value=_stub_result())) as mock_iwcr,
        ):
            await _run_unblock(config=_unblock_config(), scheduler=scheduler, worktree=wt)

        mock_helper.assert_awaited_once()
        hk = mock_helper.call_args.kwargs
        assert hk['role_name'] == 'unblock_auto'
        assert hk['task_id'] == '55'
        # Best-effort fetched task metadata flows into resolution (override/tier).
        assert hk['task_metadata'] == {'dispatch_count': 1}
        # Per-task → scheduler passed so the helper can mirror metadata.routing.
        assert hk['scheduler'] is scheduler

        kwargs = mock_iwcr.call_args.kwargs
        assert kwargs['model'] == 'sonnet'
        assert kwargs['effort'] == 'low'
        assert kwargs['max_budget_usd'] == 4.2
        assert kwargs['max_turns'] == 33
        # backend stays the site's own ua_cfg.backend read (resolver-external).
        assert kwargs['backend'] == 'claude'

    async def test_emits_routing_decision_and_mirrors_stock(
        self, tmp_path: Path,
    ) -> None:
        wt = tmp_path / 'wt'
        wt.mkdir()
        rec = _RecordingEventStore()
        scheduler = MagicMock()
        scheduler.get_task = AsyncMock(return_value={'metadata': {}})
        scheduler.update_task = AsyncMock()

        with patch('orchestrator.dry_run_unblock.invoke_with_cap_retry',
                   new=AsyncMock(return_value=_stub_result())) as mock_iwcr:
            await _run_unblock(config=_unblock_config(), scheduler=scheduler,
                               event_store=rec, worktree=wt)

        entries = _routing_entries(rec)
        assert len(entries) == 1
        data = entries[0]['data']
        assert data['role'] == 'unblock_auto'
        # No config.models.unblock_auto field → layer-3 skipped → RoleDefaults
        # (ua_cfg.*) base stands: role_default layer, byte-equivalent to pre-η.
        assert data['source_layer'] == 'role_default'
        assert data['model'] == 'sonnet'
        assert entries[0]['task_id'] == '55'

        # Byte-equivalence: the resolved invoke values equal ua_cfg.*.
        kwargs = mock_iwcr.call_args.kwargs
        assert kwargs['model'] == 'sonnet'
        assert kwargs['effort'] == 'high'
        assert kwargs['max_budget_usd'] == 5.0
        assert kwargs['max_turns'] == 50
        assert kwargs['backend'] == 'claude'

        # Per-task → metadata.routing mirrored via scheduler merge.
        assert len(_routing_mirror_calls(scheduler)) == 1

    async def test_model_override_wins_boundary_test_9(
        self, tmp_path: Path,
    ) -> None:
        wt = tmp_path / 'wt'
        wt.mkdir()
        rec = _RecordingEventStore()
        scheduler = MagicMock()
        scheduler.get_task = AsyncMock(return_value={
            'metadata': {'model_overrides': {'unblock_auto': 'haiku'}},
        })
        scheduler.update_task = AsyncMock()

        with patch('orchestrator.dry_run_unblock.invoke_with_cap_retry',
                   new=AsyncMock(return_value=_stub_result())) as mock_iwcr:
            await _run_unblock(config=_unblock_config(), scheduler=scheduler,
                               event_store=rec, worktree=wt)

        # The best-effort task fetch surfaces the override → invoke runs haiku.
        assert mock_iwcr.call_args.kwargs['model'] == 'haiku'
        entries = _routing_entries(rec)
        assert len(entries) == 1
        assert entries[0]['data']['source_layer'] == 'metadata_override'
        assert entries[0]['data']['model'] == 'haiku'

    async def test_scheduler_get_task_raising_degrades_to_stock(
        self, tmp_path: Path,
    ) -> None:
        wt = tmp_path / 'wt'
        wt.mkdir()
        rec = _RecordingEventStore()
        scheduler = MagicMock()
        scheduler.get_task = AsyncMock(side_effect=RuntimeError('scheduler hiccup'))
        scheduler.update_task = AsyncMock()

        # A scheduler.get_task that raises must degrade to empty metadata
        # (byte-equivalent stock resolution) without raising out of the
        # fire-and-forget investigation.
        with patch('orchestrator.dry_run_unblock.invoke_with_cap_retry',
                   new=AsyncMock(return_value=_stub_result())) as mock_iwcr:
            await _run_unblock(config=_unblock_config(), scheduler=scheduler,
                               event_store=rec, worktree=wt)

        assert mock_iwcr.call_args.kwargs['model'] == 'sonnet'
        entries = _routing_entries(rec)
        assert len(entries) == 1
        assert entries[0]['data']['source_layer'] == 'role_default'
