"""Tests for orchestrator.dry_run_unblock module.

Covers: config defaults, proposal schema shape, happy path, conservative
risk_label default, agent-failure fallback, budget-exhausted fallback,
invocation_end event tagging, sha-stamping, and keep-last-N trim.
"""

from __future__ import annotations

import json as _json
import subprocess
from importlib import resources as pkg_resources
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from _orch_helpers import pydantic_spec
from shared.cli_invoke import AllAccountsCappedException
from shared.config_dir import TaskConfigDir

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import Scheduler

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
                 model='sonnet', max_turns=50, effort='high', backend='claude',
                 attended_b3_enabled=False, b3_merge_cap_per_24h=6,
                 b3_proposal_keep_last=5):
    cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    cfg.unblock_auto.enabled = enabled
    cfg.unblock_auto.budget_usd = budget_usd
    cfg.unblock_auto.timeout_seconds = timeout_seconds
    cfg.unblock_auto.model = model
    cfg.unblock_auto.max_turns = max_turns
    cfg.unblock_auto.effort = effort
    cfg.unblock_auto.backend = backend
    cfg.unblock_auto.attended_b3_enabled = attended_b3_enabled
    cfg.unblock_auto.b3_merge_cap_per_24h = b3_merge_cap_per_24h
    cfg.unblock_auto.b3_proposal_keep_last = b3_proposal_keep_last
    return cfg


def _init_git_repo(path) -> str:
    """Init a minimal git repo at *path* (must be a pathlib.Path), return HEAD sha."""
    p = str(path)
    subprocess.run(['git', 'init', '-b', 'main', p], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'config', 'user.name', 'Test User'], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'config', 'user.email', 'test@example.com'], check=True, capture_output=True)
    (path / 'README.md').write_text('init')
    subprocess.run(['git', '-C', p, 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'commit', '-m', 'initial commit'], check=True, capture_output=True)
    result = subprocess.run(['git', '-C', p, 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _make_agent_result(*, success=True, cost_usd=0.50, structured_output=None,
                       subtype='', output='', duration_ms=1000,
                       timed_out=False, transcript_turns=None, session_id='',
                       stderr='', turns=0, api_error_status=None,
                       schema_salvaged=False, output_tokens=None):
    # NOTE: duplicated verbatim in test_b3_gate.py (task 2020 review:
    # code_duplication). Both copies must be extended together whenever
    # shared.cli_invoke.AgentResult gains/renames a field consumed by
    # classify_agent_failure(). Out of this task's locked scope: hoisting
    # both into the existing orchestrator/tests/_orch_helpers.py shared-helper
    # module (the established convention for cross-suite test helpers in this
    # package) is a follow-up.
    r = MagicMock()
    r.success = success
    r.cost_usd = cost_usd
    r.structured_output = structured_output
    r.subtype = subtype
    r.output = output
    r.duration_ms = duration_ms
    # Faithful diagnostic-field defaults (mirrors shared.cli_invoke.AgentResult).
    # A bare MagicMock auto-vivifies every unset attribute as a truthy
    # MagicMock, which would silently corrupt classify_agent_failure()'s
    # decision rules: .timed_out truthy -> misclassifies every failure as
    # TIMED_OUT -> infra; .schema_salvaged truthy -> misclassifies a generic
    # failure as STRUCTURAL instead of UNKNOWN. Setting explicit defaults for
    # every field classify_agent_failure() consults (including
    # .output_tokens, read when formatting the MAX_TURNS summary) keeps
    # these mocks faithful stand-ins for its decision rules.
    r.timed_out = timed_out
    r.transcript_turns = transcript_turns
    r.session_id = session_id
    r.stderr = stderr
    r.turns = turns
    r.api_error_status = api_error_status
    r.schema_salvaged = schema_salvaged
    r.output_tokens = output_tokens
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
        # task 2020 step-5: diagnostic keys also present on investigation_failed.
        assert entry['timed_out'] is False
        assert entry['subtype'] == 'error_max_turns'
        assert 'transcript_turns' in entry
        assert 'session_id' in entry
        assert 'stderr_tail' in entry
        assert 'duration_ms' in entry


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
        # task 2020 step-5: diagnostic keys also present on budget_exhausted.
        assert entry['subtype'] == 'error_max_budget_usd'
        assert 'timed_out' in entry
        assert 'transcript_turns' in entry
        assert 'session_id' in entry
        assert 'stderr_tail' in entry
        assert 'duration_ms' in entry


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


# ---------------------------------------------------------------------------
# step-3: sha-stamping tests
# ---------------------------------------------------------------------------

class TestShaStamping:
    """head_sha/main_sha are stamped onto every entry shape; schema stays closed."""

    def test_schema_guard_no_sha_in_agent_schema(self):
        """DRY_RUN_PROPOSAL_SCHEMA must stay closed — agent cannot forge sha anchors."""
        from orchestrator.dry_run_unblock import DRY_RUN_PROPOSAL_SCHEMA

        assert DRY_RUN_PROPOSAL_SCHEMA['additionalProperties'] is False
        props = DRY_RUN_PROPOSAL_SCHEMA.get('properties', {})
        assert 'head_sha' not in props, 'head_sha must NOT be in agent output schema'
        assert 'main_sha' not in props, 'main_sha must NOT be in agent output schema'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('shape', [
        'ok',
        'investigation_failed',
        'budget_exhausted',
        'exception_fallback',
    ])
    async def test_sha_stamped_on_entry(self, shape, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        repo_dir = tmp_path / 'repo'
        repo_dir.mkdir()
        main_sha = _init_git_repo(repo_dir)
        # Diverge HEAD from main: create a feature branch, add a commit there.
        # After this, git rev-parse main == main_sha (initial commit) while
        # git rev-parse HEAD == head_sha (feature branch commit).
        # Without this divergence head_sha == main_sha and the test cannot
        # distinguish an impl bug that stamps HEAD into both anchor keys.
        p = str(repo_dir)
        subprocess.run(['git', '-C', p, 'checkout', '-b', 'feature'], check=True, capture_output=True)
        (repo_dir / 'extra.txt').write_text('feature work')
        subprocess.run(['git', '-C', p, 'add', '.'], check=True, capture_output=True)
        subprocess.run(['git', '-C', p, 'commit', '-m', 'feature commit'], check=True, capture_output=True)
        head_result = subprocess.run(
            ['git', '-C', p, 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True,
        )
        head_sha = head_result.stdout.strip()
        assert head_sha != main_sha, 'test precondition: HEAD must differ from main'

        structured = {
            'proposal_text': 'Fix the blockage',
            'risk_label': 'low',
            'files_referenced': [],
        }

        if shape == 'exception_fallback':
            agent_mock = AsyncMock(side_effect=RuntimeError('unexpected error'))
        elif shape == 'ok':
            agent_result = _make_agent_result(success=True, structured_output=structured)
            agent_mock = AsyncMock(return_value=agent_result)
        elif shape == 'investigation_failed':
            agent_result = _make_agent_result(
                success=False, subtype='error_max_turns',
                output='Exceeded max turns', structured_output=None,
            )
            agent_mock = AsyncMock(return_value=agent_result)
        else:  # budget_exhausted
            agent_result = _make_agent_result(
                success=False, subtype='error_max_budget_usd',
                cost_usd=5.0, structured_output=None,
            )
            agent_mock = AsyncMock(return_value=agent_result)

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch('orchestrator.dry_run_unblock.invoke_agent', new=agent_mock):
            await run_dry_run_unblock(
                task_id='42',
                worktree=str(repo_dir),
                reason='test blocked',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        # Find the append=True call (forward-compatible: step-6 adds append=False trim call)
        append_call = next(
            c for c in scheduler.update_task.call_args_list
            if c.kwargs.get('append') is True
        )
        entry = append_call.args[1]['dry_run_proposals'][0]

        assert entry['head_sha'] == head_sha, (
            f'shape={shape}: head_sha mismatch: {entry["head_sha"]!r} != {head_sha!r}'
        )
        assert entry['main_sha'] == main_sha, (
            f'shape={shape}: main_sha mismatch: {entry["main_sha"]!r} != {main_sha!r}'
        )

    @pytest.mark.asyncio
    async def test_sha_stamps_nonrepo_yields_none(self, tmp_path):
        """Non-git worktree: keys present in entry, values are None (graceful degradation)."""
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        # tmp_path is not a git repo — _capture_worktree_shas must return (None, None)
        structured = {
            'proposal_text': 'Fix the blockage',
            'risk_label': 'low',
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
                reason='test blocked',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        append_call = next(
            c for c in scheduler.update_task.call_args_list
            if c.kwargs.get('append') is True
        )
        entry = append_call.args[1]['dry_run_proposals'][0]

        assert 'head_sha' in entry, 'head_sha key must always be present'
        assert 'main_sha' in entry, 'main_sha key must always be present'
        assert entry['head_sha'] is None, f'Expected None, got {entry["head_sha"]!r}'
        assert entry['main_sha'] is None, f'Expected None, got {entry["main_sha"]!r}'


# ---------------------------------------------------------------------------
# task 2138 step-5: block_class stamping
# ---------------------------------------------------------------------------

class TestBlockClassStamping:
    """block_class is stamped onto every entry shape, derived from `reason`
    unless explicitly overridden; schema stays closed (task 2138, PRD:
    plans/verify-plan-prd.md task ζ). Mirrors TestShaStamping — block_class
    joins head_sha/main_sha as an orchestrator-stamped, non-forgeable field.
    """

    def test_schema_guard_no_block_class_in_agent_schema(self):
        """DRY_RUN_PROPOSAL_SCHEMA must stay closed — agent cannot forge block_class."""
        from orchestrator.dry_run_unblock import DRY_RUN_PROPOSAL_SCHEMA

        assert DRY_RUN_PROPOSAL_SCHEMA['additionalProperties'] is False
        props = DRY_RUN_PROPOSAL_SCHEMA.get('properties', {})
        assert 'block_class' not in props, 'block_class must NOT be in agent output schema'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('shape', [
        'success',
        'investigation_failed',
        'budget_exhausted',
        'infra_failure',
    ])
    async def test_block_class_derived_from_reason_on_every_shape(self, shape, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock
        from orchestrator.unblock_types import classify_block_reason

        reason = 'verify exhausted'

        if shape == 'success':
            structured = {
                'proposal_text': 'Rebase on main and rerun verify',
                'risk_label': 'low',
                'files_referenced': [],
            }
            agent_result = _make_agent_result(structured_output=structured)
            config = _make_config()
        elif shape == 'investigation_failed':
            agent_result = _make_agent_result(
                success=False, subtype='error_max_turns',
                output='Exceeded max turns', structured_output=None,
            )
            config = _make_config()
        elif shape == 'budget_exhausted':
            agent_result = _make_agent_result(
                success=False, cost_usd=5.0, subtype='error_max_budget_usd',
                output='', structured_output=None,
            )
            config = _make_config(budget_usd=5.0)
        else:  # infra_failure
            agent_result = _make_agent_result(
                success=False, subtype='error_empty_output',
                output='Agent produced no output', timed_out=True,
                structured_output=None,
            )
            config = _make_config()

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='42',
                worktree=str(tmp_path),
                reason=reason,
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=config,
            )

        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert 'block_class' in entry, f'shape={shape}: block_class key missing'
        assert entry['block_class'] == classify_block_reason(reason), (
            f'shape={shape}: block_class {entry["block_class"]!r} != '
            f'classify_block_reason({reason!r})={classify_block_reason(reason)!r}'
        )
        assert entry['block_class'] == 'agent_failure'

    @pytest.mark.asyncio
    async def test_block_class_reflects_post_merge_reason_on_success_shape(self, tmp_path):
        """Proves the stamp reflects the *reason*, not just a hardcoded
        default — a post-merge-red-main reason classifies accordingly even
        on an ordinary success entry.
        """
        from orchestrator.dry_run_unblock import run_dry_run_unblock
        from orchestrator.merge_gates import POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX

        reason = POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX + ': type-check failed for orchestrator'
        structured = {
            'proposal_text': 'Fix the type error',
            'risk_label': 'low',
            'files_referenced': ['foo.py'],
        }
        agent_result = _make_agent_result(structured_output=structured)

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='43',
                worktree=str(tmp_path),
                reason=reason,
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['block_class'] == 'post_merge_red_main'

    @pytest.mark.asyncio
    async def test_explicit_block_class_param_wins_over_derived(self, tmp_path):
        """An explicit block_class kwarg overrides the reason-derived default
        (forward-compat with task η, which constructs BlockRecord(MERGE_VERIFY_RED)
        for a generic un-constanted reason that classify_block_reason would
        otherwise default to AGENT_FAILURE).
        """
        from orchestrator.dry_run_unblock import run_dry_run_unblock
        from orchestrator.unblock_types import BlockClass

        # A plain reason that classify_block_reason would map to AGENT_FAILURE.
        reason = 'verify exhausted'
        structured = {
            'proposal_text': 'Fix it',
            'risk_label': 'low',
            'files_referenced': [],
        }
        agent_result = _make_agent_result(structured_output=structured)

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='44',
                worktree=str(tmp_path),
                reason=reason,
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
                block_class=BlockClass.MERGE_VERIFY_RED,
            )

        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['block_class'] == 'merge_verify_red'


# ---------------------------------------------------------------------------
# step-5: keep-last-N trim tests
# ---------------------------------------------------------------------------

class _RecordingScheduler:
    """Faithful fake scheduler for trim tests.

    Mirrors the #1827 metadata_mode contract:
    - metadata_mode='additive': recursive list-union, dict-recursive, scalar OLD-wins.
      Extends dry_run_proposals list and updates other keys.
    - metadata_mode='merge' (default/no-append): shallow last-write-wins.
      Supplied keys overwrite wholesale; omitted keys are preserved.
    - metadata_mode='replace': whole-blob overwrite.

    Legacy append kwarg accepted for back-compat: append=True → additive behaviour.

    get_task: returns {'metadata': <current blob>}.
    Tracks all update_task calls for assertion.
    """

    def __init__(self, initial_metadata: dict):
        self._meta: dict = dict(initial_metadata)
        self.update_task_calls: list[dict] = []

    async def update_task(self, task_id, metadata, *, append=False, metadata_mode=None):
        self.update_task_calls.append({
            'task_id': task_id,
            'metadata': metadata,
            'append': append,
            'metadata_mode': metadata_mode,
        })
        # Resolve effective mode: metadata_mode > append True→additive > default merge
        if metadata_mode is not None:
            effective = metadata_mode
        elif append:
            effective = 'additive'
        else:
            effective = 'merge'

        if effective == 'additive':
            # List-union, scalar overwrite (simplified — fake only needs list-union)
            for key, value in metadata.items():
                if key in self._meta and isinstance(self._meta[key], list) and isinstance(value, list):
                    self._meta[key] = self._meta[key] + value
                else:
                    self._meta[key] = value
        elif effective == 'replace':
            self._meta = dict(metadata)
        else:
            # merge: shallow last-write-wins (supplied keys overwrite, omitted preserved)
            self._meta = {**self._meta, **metadata}

    async def get_task(self, task_id):
        return {'metadata': dict(self._meta)}


class TestKeepLastNTrim:
    """keep-last-N trim bounds dry_run_proposals growth and preserves sibling keys."""

    @pytest.mark.asyncio
    async def test_trim_keeps_last_n_and_preserves_siblings(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        repo_dir = tmp_path / 'repo'
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        # Seed scheduler with sibling metadata keys that must survive trim
        initial_metadata = {
            'dry_run_proposals': [],
            'memory_hints': ['hint-A', 'hint-B'],
            'files': ['src/foo.py'],
        }
        scheduler = _RecordingScheduler(initial_metadata)

        keep_last = 5
        num_runs = 6  # one more than keep_last

        for i in range(num_runs):
            proposal_text = f'Proposal number {i}'
            structured = {
                'proposal_text': proposal_text,
                'risk_label': 'low',
                'files_referenced': [],
            }
            agent_result = _make_agent_result(structured_output=structured)

            with patch('orchestrator.dry_run_unblock.invoke_agent',
                       new=AsyncMock(return_value=agent_result)):
                await run_dry_run_unblock(
                    task_id='trim-test',
                    worktree=str(repo_dir),
                    reason='verify exhausted',
                    detail='',
                    scheduler=scheduler,
                    mcp=MagicMock(),
                    config=_make_config(b3_proposal_keep_last=keep_last),
                )

        proposals = scheduler._meta['dry_run_proposals']

        # After 6 runs with keep_last=5, exactly 5 remain
        assert len(proposals) == keep_last, (
            f'Expected {keep_last} proposals after trim, got {len(proposals)}'
        )

        # The retained entries are the LAST 5 (proposals 1-5, not proposal 0)
        retained_texts = [p['proposal_text'] for p in proposals]
        assert retained_texts == [f'Proposal number {i}' for i in range(1, num_runs)], (
            f'Expected proposals 1-5 to be retained, got: {retained_texts}'
        )

        # Sibling keys are intact after the full-blob trim write
        assert scheduler._meta.get('memory_hints') == ['hint-A', 'hint-B'], (
            'memory_hints sibling key must survive trim'
        )
        assert scheduler._meta.get('files') == ['src/foo.py'], (
            'files sibling key must survive trim'
        )

        # At least one update_task call was made with append=False (the RMW trim write)
        rmw_calls = [c for c in scheduler.update_task_calls if not c['append']]
        assert len(rmw_calls) >= 1, (
            'Expected at least one update_task(append=False) trim write, got none'
        )
        # The trim write must carry the full blob (including sibling keys)
        trim_call = rmw_calls[-1]
        assert 'memory_hints' in trim_call['metadata'], (
            'Trim write must carry the full blob (memory_hints missing)'
        )
        assert 'files' in trim_call['metadata'], (
            'Trim write must carry the full blob (files missing)'
        )


# ---------------------------------------------------------------------------
# step-4 (1828): wire-mode assertions — proposal=additive, trim=merge
# ---------------------------------------------------------------------------

class TestDryRunWireMode:
    """Verify that the two update_task callers in dry_run_unblock forward the
    correct metadata_mode on the wire.

    Uses a REAL Scheduler with monkeypatched mcp_call so we capture the raw
    MCP arguments exactly as they'll be received by fused-memory.

    RED today: proposal call forwards 'append' key instead of metadata_mode;
    trim call forwards nothing (no metadata_mode).
    GREEN after step-5 impl (scheduler.py update_task gains metadata_mode).
    """

    @pytest.mark.asyncio
    async def test_proposal_persist_and_trim_wire_modes(
        self, tmp_path, monkeypatch
    ):
        """Wire-mode assertions only: proposal persist → 'additive'; trim → 'merge'.

        Verifies the flags forwarded on the wire; behavioral verification
        (additive actually appends, merge actually trims) lives in the
        separate _RecordingScheduler / TestKeepLastNTrim suite.
        """
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        repo_dir = tmp_path / 'repo'
        repo_dir.mkdir()
        _init_git_repo(repo_dir)

        # Seed the task with 6 proposals + siblings so the trim is triggered
        # (keep_last=5 → len>5 triggers trim after get_task returns the blob).
        # We'll run once: proposal persist fires, then get_task returns 6 proposals
        # so that 6 > 5 triggers the trim write.
        six_proposals = [
            {'proposal_text': f'Proposal {i}', 'risk_label': 'low', 'files_referenced': []}
            for i in range(6)
        ]
        seeded_task_doc = {
            'id': 'wire-test',
            'metadata': {
                'dry_run_proposals': six_proposals,
                'memory_hints': ['hint-A'],
                'files': ['src/foo.py'],
            },
        }

        # All wire calls captured here (both update_task and get_task payloads).
        captured_payloads: list[dict] = []

        async def mock_mcp_call(url, method, payload, **kwargs):
            captured_payloads.append(payload)
            name = payload.get('name', '')
            if name == 'get_task':
                # Return full task doc so get_task's parser can find task['metadata'].
                return {
                    'result': {
                        'content': [{
                            'type': 'text',
                            'text': _json.dumps(seeded_task_doc),
                        }]
                    }
                }
            # update_task and any other tool → success
            return {}

        real_scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
        monkeypatch.setattr('orchestrator.scheduler.mcp_call', mock_mcp_call)

        structured = {
            'proposal_text': 'Rebase on main',
            'risk_label': 'low',
            'files_referenced': [],
        }
        agent_result = _make_agent_result(structured_output=structured)

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='wire-test',
                worktree=str(repo_dir),
                reason='verify exhausted',
                detail='',
                scheduler=real_scheduler,
                mcp=MagicMock(),
                config=_make_config(b3_proposal_keep_last=5),
            )

        update_calls = [p for p in captured_payloads if p.get('name') == 'update_task']
        assert len(update_calls) == 2, (
            f'Expected 2 update_task wire calls (proposal + trim); '
            f'got {len(update_calls)} '
            f'(all calls: {[p.get("name") for p in captured_payloads]})'
        )

        # First update_task call is the proposal persist (append=True → additive).
        proposal_args = update_calls[0]['arguments']
        assert proposal_args.get('metadata_mode') == 'additive', (
            f"Proposal persist must forward metadata_mode='additive'; got: {proposal_args}"
        )
        assert 'append' not in proposal_args, (
            f"'append' must not appear on the wire for proposal persist; got: {proposal_args}"
        )

        # Second update_task call is the trim write (append=False/omitted → merge).
        trim_args = update_calls[1]['arguments']
        assert trim_args.get('metadata_mode') == 'merge', (
            f"Trim write must forward metadata_mode='merge'; got: {trim_args}"
        )
        assert 'append' not in trim_args, (
            f"'append' must not appear on the wire for trim write; got: {trim_args}"
        )


# ---------------------------------------------------------------------------
# task 2020 step-1: infra vs. substantive failure classification
# ---------------------------------------------------------------------------

class TestInfraFailureClassification:
    """Transient HOST infra wedges (600s ceiling kill -> empty stdout +
    timed_out) are classified as a distinct 'infra_failure' status, separate
    from a substantive 'investigation_failed' agent conclusion.
    """

    @pytest.mark.asyncio
    async def test_host_wedge_classified_as_infra_failure(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        # Simulated HOST wedge: pre-turn-1 kill at the 600s UnblockAutoConfig
        # timeout ceiling -> empty stdout + timed_out=True.
        agent_result = _make_agent_result(
            success=False,
            subtype='error_empty_output',
            output='Agent produced no output',
            timed_out=True,
            duration_ms=600000,
            transcript_turns=0,
            session_id='sess-x',
            stderr='...Absolute ceiling reached after 600.0s...',
            structured_output=None,
        )

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='701',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        scheduler.update_task.assert_awaited_once()
        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['status'] == 'infra_failure'
        assert entry['risk_label'] == 'human-review-required'
        assert entry['files_referenced'] == []

        # task 2020 step-3: discrete diagnostic keys carried on the entry.
        assert entry['timed_out'] is True
        assert entry['duration_ms'] == 600000
        assert entry['subtype'] == 'error_empty_output'
        assert entry['transcript_turns'] == 0
        assert entry['session_id'] == 'sess-x'
        assert 'Absolute ceiling reached after 600.0s' in entry['stderr_tail']
        assert len(entry['stderr_tail']) <= 500

    @pytest.mark.asyncio
    async def test_substantive_failure_stays_investigation_failed(self, tmp_path):
        """Guard against over-broad infra classification: a non-timed-out,
        non-empty-output substantive failure must keep 'investigation_failed'.
        """
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        agent_result = _make_agent_result(
            success=False,
            subtype='error_max_turns',
            output='Exceeded max turns',
            timed_out=False,
            structured_output=None,
        )

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='702',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        scheduler.update_task.assert_awaited_once()
        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['status'] == 'investigation_failed'

    @pytest.mark.asyncio
    async def test_high_turn_timeout_still_classified_infra_failure(self, tmp_path):
        """Pin current behavior (review note, task 2020): classification is
        by AgentFailureKind alone (timed_out=True -> TIMED_OUT), NOT gated on
        transcript_turns. A genuine investigation that ran many turns before
        hitting the timeout ceiling is classified identically to a
        pre-turn-1 wedge — both land as 'infra_failure'. This is intentional:
        risk_label stays 'human-review-required' either way, so no
        auto-retry/auto-unblock path opens because of it (both statuses
        still ABORT at the B3 gate; see
        TestTwoWayBoundary.test_infra_failure_and_investigation_failed_both_aborted
        in test_b3_gate.py). A future consumer that auto-retries specifically
        on infra_failure should itself gate on transcript_turns rather than
        assume this status implies near-zero turns.
        """
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        agent_result = _make_agent_result(
            success=False,
            subtype='error_empty_output',
            output='',
            timed_out=True,
            duration_ms=600000,
            transcript_turns=47,
            session_id='sess-y',
            stderr='...Absolute ceiling reached after 600.0s...',
            structured_output=None,
        )

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch('orchestrator.dry_run_unblock.invoke_agent',
                   new=AsyncMock(return_value=agent_result)):
            await run_dry_run_unblock(
                task_id='704',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        scheduler.update_task.assert_awaited_once()
        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['status'] == 'infra_failure'
        assert entry['transcript_turns'] == 47


# ---------------------------------------------------------------------------
# task 2020 step-7: exception fallback carries None-safe diagnostic keys
# ---------------------------------------------------------------------------

class TestExceptionFallbackDiagnostics:
    """The exception-fallback entry (an orchestrator-side error, not a
    detected host wedge) stays 'investigation_failed' and carries all six
    diagnostic keys with None values — shape parity, since no AgentResult
    ever existed on this path.
    """

    @pytest.mark.asyncio
    async def test_exception_fallback_carries_none_diagnostics(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch(
            'orchestrator.dry_run_unblock.invoke_agent',
            new=AsyncMock(side_effect=RuntimeError('boom')),
        ):
            await run_dry_run_unblock(
                task_id='703',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        scheduler.update_task.assert_awaited_once()
        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['status'] == 'investigation_failed'
        assert entry['risk_label'] == 'human-review-required'
        for key in (
            'timed_out', 'duration_ms', 'subtype',
            'transcript_turns', 'session_id', 'stderr_tail',
        ):
            assert key in entry, f'{key} must be present on the exception fallback entry'
            assert entry[key] is None, f'{key} must be None on the exception fallback entry'


# ---------------------------------------------------------------------------
# task 2021 step-1: resilience scaffolding — config_dir + session_id revive
# the startup watchdog; cap-retry/failover wiring through invoke_with_cap_retry
# ---------------------------------------------------------------------------

class TestDryRunResilienceScaffolding:
    """run_dry_run_unblock must route through invoke_with_cap_retry with a
    per-investigation TaskConfigDir + pre-allocated session_id, so that the
    startup watchdog in shared/cli_invoke.py._run_subprocess is no longer
    structurally dead for this call site (config_dir and session_id were
    previously never passed, forcing `_grace_spent=True`).
    """

    @pytest.mark.asyncio
    async def test_watchdog_revival_and_cleanup_on_success(self, tmp_path):
        """invoke_agent (the real fast path, usage_gate omitted) receives a
        Path config_dir and a non-empty session_id; the per-investigation
        config dir is cleaned up after a successful run.
        """
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        structured = {
            'proposal_text': 'Rebase on main and rerun verify',
            'risk_label': 'low',
            'files_referenced': [],
        }
        agent_result = _make_agent_result(structured_output=structured)

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch(
            'orchestrator.dry_run_unblock.invoke_agent',
            new=AsyncMock(return_value=agent_result),
        ) as mock_invoke:
            await run_dry_run_unblock(
                task_id='2100',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        invoke_kwargs = mock_invoke.call_args.kwargs
        config_dir_arg = invoke_kwargs.get('config_dir')
        assert isinstance(config_dir_arg, Path), (
            f'Expected invoke_agent to receive a Path config_dir, got {config_dir_arg!r}'
        )
        session_id_arg = invoke_kwargs.get('session_id')
        assert isinstance(session_id_arg, str) and session_id_arg, (
            f'Expected a non-empty session_id string, got {session_id_arg!r}'
        )

        # Cleanup: no claude-config-*-unblock dir remains under tmp_path/.task
        task_dir = tmp_path / '.task'
        leftover = list(task_dir.glob('claude-config-*-unblock')) if task_dir.exists() else []
        assert leftover == [], f'Expected config dir to be cleaned up, found: {leftover}'

    @pytest.mark.asyncio
    async def test_cap_policy_wiring_forwards_usage_gate_and_cost_store(self, tmp_path):
        """invoke_with_cap_retry receives usage_gate/cost_store, a
        TaskConfigDir, the 1800s cap-wait sanity override, role='unblock_auto',
        invoke_fn=invoke_agent, and a session_id kwarg.
        """
        from orchestrator.dry_run_unblock import invoke_agent as real_invoke_agent
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        structured = {
            'proposal_text': 'Rebase on main and rerun verify',
            'risk_label': 'low',
            'files_referenced': [],
        }
        agent_result = _make_agent_result(structured_output=structured)

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        usage_gate_sentinel = object()
        cost_store_sentinel = object()

        with patch(
            'orchestrator.dry_run_unblock.invoke_with_cap_retry',
            new=AsyncMock(return_value=agent_result),
        ) as mock_cap_retry:
            await run_dry_run_unblock(
                task_id='2101',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
                usage_gate=usage_gate_sentinel,
                cost_store=cost_store_sentinel,
            )

        mock_cap_retry.assert_awaited_once()
        call_kwargs = mock_cap_retry.call_args.kwargs
        assert call_kwargs.get('usage_gate') is usage_gate_sentinel
        assert call_kwargs.get('cost_store') is cost_store_sentinel
        assert isinstance(call_kwargs.get('config_dir'), TaskConfigDir)
        assert call_kwargs.get('cap_wait_sanity_secs') == 1800
        assert call_kwargs.get('role') == 'unblock_auto'
        assert call_kwargs.get('invoke_fn') is real_invoke_agent
        session_id_arg = call_kwargs.get('session_id')
        assert isinstance(session_id_arg, str) and session_id_arg


# ---------------------------------------------------------------------------
# task 2021 step-3: exactly one fresh retry on a zero-output timeout wedge
# ---------------------------------------------------------------------------

class TestDryRunFreshRetry:
    """A pre-turn-1 host wedge (timed_out=True, transcript_turns==0) is
    retried exactly ONCE with a freshly-allocated session_id — never reusing
    a possibly-committed UUID (reify-3604 / _reset_for_fresh_retry semantics).
    usage_gate is omitted throughout so invoke_with_cap_retry takes its fast,
    no-gate path and forwards straight to the patched invoke_agent.
    """

    @pytest.mark.asyncio
    async def test_wedge_then_success_retries_once_with_fresh_session(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        wedge = _make_agent_result(
            success=False, timed_out=True, transcript_turns=0,
            subtype='error_empty_output', session_id='w1',
        )
        success = _make_agent_result(
            success=True,
            structured_output={
                'proposal_text': 'x', 'risk_label': 'low', 'files_referenced': [],
            },
        )

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        mock_invoke = AsyncMock(side_effect=[wedge, success])
        with patch('orchestrator.dry_run_unblock.invoke_agent', new=mock_invoke):
            await run_dry_run_unblock(
                task_id='2200',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        assert mock_invoke.await_count == 2, (
            f'Expected the wedge to trigger exactly one retry, got '
            f'{mock_invoke.await_count} invoke_agent calls'
        )
        first_session = mock_invoke.call_args_list[0].kwargs.get('session_id')
        second_session = mock_invoke.call_args_list[1].kwargs.get('session_id')
        assert first_session != second_session, (
            'retry must allocate a FRESH session_id, never reuse the wedged one'
        )

        scheduler.update_task.assert_awaited_once()
        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry.get('risk_label') == 'low'
        assert entry.get('status') is None, (
            f"successful retry must persist the plain success shape (no 'status' "
            f"key), got status={entry.get('status')!r}"
        )

    @pytest.mark.asyncio
    async def test_wedge_then_wedge_bounded_to_one_retry(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        wedge = _make_agent_result(
            success=False, timed_out=True, transcript_turns=0,
            subtype='error_empty_output', session_id='w1',
        )

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        mock_invoke = AsyncMock(side_effect=[wedge, wedge])
        with patch('orchestrator.dry_run_unblock.invoke_agent', new=mock_invoke):
            await run_dry_run_unblock(
                task_id='2201',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        assert mock_invoke.await_count == 2, (
            f'Expected exactly one retry (2 total attempts, bounded — not a '
            f'loop), got {mock_invoke.await_count}'
        )
        scheduler.update_task.assert_awaited_once()
        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['status'] == 'infra_failure'


# ---------------------------------------------------------------------------
# task 2021 step-5: cap exhaustion converts to a retryable infra_failure entry
# ---------------------------------------------------------------------------

class TestDryRunCapExhaustion:
    """AllAccountsCappedException (raised by invoke_with_cap_retry when the
    1800s cap-wait sanity bound is exceeded) is caught and converted into a
    retryable 'infra_failure' proposal entry — it must NOT propagate out of
    run_dry_run_unblock, and must NOT be written as the terminal
    'investigation_failed' shape (which would disable B3 auto-retry).
    """

    @pytest.mark.asyncio
    async def test_cap_exhaustion_writes_infra_failure_entry(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        cap_exc = AllAccountsCappedException(
            retries=3, elapsed_secs=1800.0, label='Task 42 [unblock_auto]',
        )

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch(
            'orchestrator.dry_run_unblock.invoke_with_cap_retry',
            new=AsyncMock(side_effect=cap_exc),
        ):
            # Must not raise.
            await run_dry_run_unblock(
                task_id='42',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        scheduler.update_task.assert_awaited_once()
        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['status'] == 'infra_failure', (
            f"cap exhaustion must land on 'infra_failure', not "
            f"{entry.get('status')!r} — a terminal 'investigation_failed' "
            f"would disable the B3 low-risk auto-unblock retry path"
        )
        assert entry['risk_label'] == 'human-review-required'
        assert entry['files_referenced'] == []
        assert 'capped' in entry['proposal_text'].lower()
        assert '3' in entry['proposal_text']
        for key in (
            'timed_out', 'duration_ms', 'subtype',
            'transcript_turns', 'session_id', 'stderr_tail',
        ):
            assert key in entry, f'{key} must be present on the cap-exhaustion entry'
            assert entry[key] is None, f'{key} must be None on the cap-exhaustion entry'


# ---------------------------------------------------------------------------
# task 2021 step-7: preserve the config dir on a doubly-wedged (forensic) run
# ---------------------------------------------------------------------------

class TestDryRunConfigDirPreserve:
    """The per-investigation TaskConfigDir is preserved (cleanup skipped)
    when the FINAL result — after the single retry — is still a zero-output
    timeout wedge, mirroring task-1739's _preserve_config_dir forensics
    pattern. A healthy run still cleans up normally (pinned both ways here
    so the preserve branch cannot regress into always-preserve).
    """

    @pytest.mark.asyncio
    async def test_doubly_wedged_run_preserves_config_dir(self, tmp_path):
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        wedge = _make_agent_result(
            success=False, timed_out=True, transcript_turns=0,
            subtype='error_empty_output', session_id='w1',
        )

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch(
            'orchestrator.dry_run_unblock.invoke_agent',
            new=AsyncMock(side_effect=[wedge, wedge]),
        ):
            await run_dry_run_unblock(
                task_id='2300',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        task_dir = tmp_path / '.task'
        preserved = list(task_dir.glob('claude-config-*-unblock')) if task_dir.exists() else []
        assert len(preserved) == 1, (
            f'Expected the config dir to be PRESERVED after a doubly-wedged '
            f'run (forensics), found: {preserved}'
        )

    @pytest.mark.asyncio
    async def test_successful_run_still_cleans_up_config_dir(self, tmp_path):
        """Counterpart pin: a clean success still cleans up — guards against
        a naive fix that preserves the config dir unconditionally.
        """
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        structured = {
            'proposal_text': 'Rebase on main',
            'risk_label': 'low',
            'files_referenced': [],
        }
        agent_result = _make_agent_result(structured_output=structured)

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch(
            'orchestrator.dry_run_unblock.invoke_agent',
            new=AsyncMock(return_value=agent_result),
        ):
            await run_dry_run_unblock(
                task_id='2301',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
            )

        task_dir = tmp_path / '.task'
        leftover = list(task_dir.glob('claude-config-*-unblock')) if task_dir.exists() else []
        assert leftover == [], f'Expected config dir cleanup on success, found: {leftover}'


# ---------------------------------------------------------------------------
# task 2021 amendment: pin the 1800s cap-wait bound to REAL loop behavior
# ---------------------------------------------------------------------------

class _ForcedCapSlot:
    """Fake ``invoke_slot()`` context manager that reports a cap hit every time."""

    account_name = 'forced-cap-account'
    token = 'tok'

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def detect_cap_hit(self, stderr, output, backend=None):
        return True

    def settle(self):
        pass

    def confirm(self, cost_usd):
        pass


class _ForcedCapUsageGate:
    """Minimal fake usage_gate whose every ``invoke_slot()`` reports a cap hit.

    Drives the REAL ``invoke_with_cap_retry`` loop (unlike
    TestDryRunCapExhaustion, which patches ``invoke_with_cap_retry`` itself
    and only pins the entry-conversion shape). Real accounting means the
    sanity-bound guard in ``invoke_with_cap_retry`` is what raises
    ``AllAccountsCappedException`` here, not a mocked side_effect.
    """

    account_count = 1
    active_account_name = 'forced-cap-account'
    soonest_resets_at = None

    def invoke_slot(self):
        return _ForcedCapSlot()


class TestDryRunCapExhaustionRealLoop:
    """End-to-end: a fake usage_gate that always reports a cap hit drives the
    real invoke_with_cap_retry cap-wait loop, which raises
    AllAccountsCappedException once elapsed time exceeds the sanity bound.
    Pins _DRY_RUN_CAP_WAIT_SANITY_SECS to actual loop behavior rather than a
    patched sentinel (TestDryRunCapExhaustion covers the entry-shape
    conversion in isolation).
    """

    @pytest.mark.asyncio
    async def test_forced_cap_hit_raises_and_converts_to_infra_failure(
        self, tmp_path, monkeypatch,
    ):
        from orchestrator import dry_run_unblock
        from orchestrator.dry_run_unblock import run_dry_run_unblock

        # A negative sanity bound guarantees `elapsed > cap_wait_sanity_secs`
        # trips on the very first cap hit regardless of actual wall-clock
        # jitter, so the test doesn't need to wait out the real 1800s (or
        # even the 5s base cooldown — the sanity check raises before the
        # loop ever reaches `await asyncio.sleep(cooldown)`).
        monkeypatch.setattr(dry_run_unblock, '_DRY_RUN_CAP_WAIT_SANITY_SECS', -1.0)

        agent_result = _make_agent_result(success=False, subtype='error_rate_limit')

        scheduler = MagicMock()
        scheduler.update_task = AsyncMock(return_value=True)

        with patch(
            'orchestrator.dry_run_unblock.invoke_agent',
            new=AsyncMock(return_value=agent_result),
        ):
            await run_dry_run_unblock(
                task_id='2400',
                worktree=str(tmp_path),
                reason='verify exhausted',
                detail='',
                scheduler=scheduler,
                mcp=MagicMock(),
                config=_make_config(),
                usage_gate=_ForcedCapUsageGate(),
            )

        scheduler.update_task.assert_awaited_once()
        entry = scheduler.update_task.call_args.args[1]['dry_run_proposals'][0]
        assert entry['status'] == 'infra_failure', (
            f"expected the real cap-wait loop to raise AllAccountsCappedException "
            f"and convert to 'infra_failure', got status={entry.get('status')!r}"
        )
        assert entry['risk_label'] == 'human-review-required'
        assert entry['files_referenced'] == []
        assert 'capped' in entry['proposal_text'].lower()
