"""Tests for orchestrator.dry_run_unblock module.

Covers: config defaults, proposal schema shape, happy path, conservative
risk_label default, agent-failure fallback, budget-exhausted fallback,
and invocation_end event tagging.
"""

from __future__ import annotations

from importlib import resources as pkg_resources
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig

# ---------------------------------------------------------------------------
# step-3: config defaults
# ---------------------------------------------------------------------------

class TestUnblockAutoConfigDefaults:
    """Pins UnblockAutoConfig field names and default values."""

    def test_unblock_auto_config_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()

        ua = cfg.unblock_auto
        assert ua.enabled is True
        assert ua.budget_usd == pytest.approx(5.0)
        assert ua.timeout_seconds == pytest.approx(600.0)
        assert ua.model == 'sonnet'
        assert ua.max_turns == 50
        assert ua.effort == 'high'
        assert ua.backend == 'claude'
        # Three new fields (step-1)
        assert ua.attended_b3_enabled is False
        assert ua.b3_merge_cap_per_24h == 6
        assert ua.b3_proposal_keep_last == 5

    def test_defaults_yaml_has_unblock_auto_section(self):
        defaults_file = pkg_resources.files('orchestrator') / 'defaults.yaml'
        data = yaml.safe_load(defaults_file.read_text())
        assert 'unblock_auto' in data, (
            "defaults.yaml must have an 'unblock_auto' section"
        )
        ua = data['unblock_auto']
        assert ua['enabled'] is True
        assert ua['budget_usd'] == pytest.approx(5.0)
        assert ua['timeout_seconds'] == pytest.approx(600.0)
        assert ua['model'] == 'sonnet'
        assert ua['max_turns'] == 50
        assert ua['effort'] == 'high'
        assert ua['backend'] == 'claude'
        # Three new fields (step-1)
        assert ua['attended_b3_enabled'] is False
        assert ua['b3_merge_cap_per_24h'] == 6
        assert ua['b3_proposal_keep_last'] == 5


# ---------------------------------------------------------------------------
# step-5: proposal schema shape
# ---------------------------------------------------------------------------

class TestProposalSchemaShape:
    """DRY_RUN_PROPOSAL_SCHEMA has the required fields and risk_label enum."""

    def test_proposal_schema_shape(self):
        from orchestrator.dry_run_unblock import DRY_RUN_PROPOSAL_SCHEMA

        assert isinstance(DRY_RUN_PROPOSAL_SCHEMA, dict)
        required = DRY_RUN_PROPOSAL_SCHEMA.get('required', [])
        for field in ('proposal_text', 'risk_label', 'files_referenced'):
            assert field in required, f"'{field}' must be in schema required list"

        props = DRY_RUN_PROPOSAL_SCHEMA.get('properties', {})
        risk = props.get('risk_label', {})
        enum_vals = risk.get('enum', [])
        assert set(enum_vals) == {'low', 'medium', 'human-review-required'}, (
            f"risk_label enum must be exactly "
            f"{{'low', 'medium', 'human-review-required'}}, got {enum_vals}"
        )


# ---------------------------------------------------------------------------
# Helpers shared by steps 7-16
# ---------------------------------------------------------------------------

def _make_config(*, enabled=True, budget_usd=5.0, timeout_seconds=600.0,
                 model='sonnet', max_turns=50, effort='high', backend='claude'):
    cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    cfg.unblock_auto.enabled = enabled
    cfg.unblock_auto.budget_usd = budget_usd
    cfg.unblock_auto.timeout_seconds = timeout_seconds
    cfg.unblock_auto.model = model
    cfg.unblock_auto.max_turns = max_turns
    cfg.unblock_auto.effort = effort
    cfg.unblock_auto.backend = backend
    return cfg


def _make_agent_result(*, success=True, cost_usd=0.50, structured_output=None,
                       subtype='', output='', duration_ms=1000):
    r = MagicMock()
    r.success = success
    r.cost_usd = cost_usd
    r.structured_output = structured_output
    r.subtype = subtype
    r.output = output
    r.duration_ms = duration_ms
    return r


# ---------------------------------------------------------------------------
# step-7: happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    @pytest.mark.asyncio
    async def test_happy_path_writes_proposal_to_metadata(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        structured = {
            'proposal_text': 'Rebase on main and rerun verify',
            'risk_label': 'low',
            'files_referenced': ['orchestrator/src/orchestrator/workflow.py'],
        }
        agent_result = _make_agent_result(structured_output=structured)

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch(
            'orchestrator.dry_run_unblock.invoke_agent',
            new=AsyncMock(return_value=agent_result),
        ) as mock_invoke:
            await run_dry_run_unblock(
                task_id='42',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='All 5 attempts timed out',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        # scheduler.update_task called once with append=True
        scheduler.update_task.assert_awaited_once()
        call_args = scheduler.update_task.call_args
        assert call_args.args[0] == '42'
        metadata_arg = call_args.args[1]
        assert call_args.kwargs.get('append') is True

        # entry shape
        proposals = metadata_arg['dry_run_proposals']
        assert len(proposals) == 1
        entry = proposals[0]
        assert entry['proposal_text'] == 'Rebase on main and rerun verify'
        assert entry['risk_label'] == 'low'
        assert entry['files_referenced'] == ['orchestrator/src/orchestrator/workflow.py']
        assert entry['block_reason'] == 'verify exhausted'
        assert 'timestamp' in entry
        assert 'investigated_at' in entry

        # invoke_agent tool restrictions
        invoke_kwargs = mock_invoke.call_args.kwargs
        allowed = invoke_kwargs.get('allowed_tools', [])
        disallowed = invoke_kwargs.get('disallowed_tools', [])
        assert any('Read' in t for t in allowed)
        assert any('Glob' in t for t in allowed)
        assert any('Grep' in t for t in allowed)
        assert 'Edit' in disallowed
        assert 'Write' in disallowed
        assert 'mcp__fused-memory__set_task_status' in disallowed
        assert 'mcp__fused-memory__update_task' in disallowed


# ---------------------------------------------------------------------------
# step-9: missing risk_label defaults to human-review-required
# ---------------------------------------------------------------------------

class TestConservativeDefault:
    @pytest.mark.asyncio
    async def test_missing_risk_label_defaults_to_human_review_required(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        # structured_output without risk_label
        structured = {
            'proposal_text': 'Try rebasing',
            'files_referenced': [],
        }
        agent_result = _make_agent_result(structured_output=structured)

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='99',
                worktree=str(tmp_path),
                reason='review failed',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['risk_label'] == 'human-review-required'

    @pytest.mark.asyncio
    async def test_invalid_risk_label_defaults_to_human_review_required(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        structured = {
            'proposal_text': 'Some proposal',
            'risk_label': 'definitely-fine',  # invalid
            'files_referenced': [],
        }
        agent_result = _make_agent_result(structured_output=structured)

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='99',
                worktree=str(tmp_path),
                reason='merge conflict',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['risk_label'] == 'human-review-required'


# ---------------------------------------------------------------------------
# step-11: agent failure fallback
# ---------------------------------------------------------------------------

class TestAgentFailureFallback:
    @pytest.mark.asyncio
    async def test_agent_failure_writes_fallback_entry(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        agent_result = _make_agent_result(
            success=False,
            output='Exceeded max turns',
            subtype='error_max_turns',
            cost_usd=0.1,
            structured_output=None,
        )

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='77',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        scheduler.update_task.assert_awaited_once()
        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['risk_label'] == 'human-review-required'
        assert 'error_max_turns' in entry['proposal_text']
        assert entry['files_referenced'] == []
        assert 'investigated_at' in entry
        assert entry['block_reason'] == 'verify exhausted'
        # cost_usd must be present on all fallback entries for dashboard queries
        assert 'cost_usd' in entry
        assert entry['cost_usd'] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# step-13: budget exhausted fallback
# ---------------------------------------------------------------------------

class TestBudgetExhaustedFallback:
    @pytest.mark.asyncio
    async def test_budget_exhausted_writes_specific_entry(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        agent_result = _make_agent_result(
            success=False,
            cost_usd=5.0,
            subtype='error_max_budget_usd',
            output='',
            structured_output=None,
        )

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='88',
                worktree=str(tmp_path),
                reason='review failed',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(budget_usd=5.0),
            )

        scheduler.update_task.assert_awaited_once()
        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry.get('status') == 'budget_exhausted'
        assert 'budget exhausted' in entry['proposal_text'].lower()
        assert 'proposal incomplete' in entry['proposal_text'].lower()
        assert entry['risk_label'] == 'human-review-required'
        assert entry['cost_usd'] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# step-15: event tagging
# ---------------------------------------------------------------------------

class TestEventTagging:
    @pytest.mark.asyncio
    async def test_emits_event_with_dry_run_tag(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        structured = {
            'proposal_text': 'Rebase on main',
            'risk_label': 'medium',
            'files_referenced': [],
        }
        agent_result = _make_agent_result(structured_output=structured)

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)
        event_store = MagicMock()

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='55',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
                event_store=event_store,
            )

        event_store.emit.assert_called_once()
        emit_call = event_store.emit.call_args

        # First positional arg must be EventType.invocation_end
        from orchestrator.event_store import EventType
        assert emit_call.args[0] == EventType.invocation_end

        # Keyword args that operators filter on to find dry-run emissions
        assert emit_call.kwargs.get('phase') == 'blocked'
        assert emit_call.kwargs.get('role') == 'unblock_auto'

        # data payload
        data = emit_call.kwargs.get('data', {})
        assert data.get('dry_run') is True
        assert 'risk_label' in data
