"""Tests for orchestrator.dry_run_unblock module.

Covers: config defaults, proposal schema shape, happy path, conservative
risk_label default, agent-failure fallback, budget-exhausted fallback,
invocation_end event tagging, sha-stamping, and keep-last-N trim.
"""

from __future__ import annotations

import subprocess
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
        head_sha = _init_git_repo(repo_dir)
        # After init, main and HEAD point to the same commit
        main_sha = head_sha

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
# step-5: keep-last-N trim tests
# ---------------------------------------------------------------------------

class _RecordingScheduler:
    """Faithful fake scheduler for trim tests.

    append=True: recursive-merge — extends dry_run_proposals list
                  and updates any other supplied keys.
    append=False: replaces the whole metadata blob.
    get_task: returns {'metadata': <current blob>}.
    Tracks all update_task calls for assertion.
    """

    def __init__(self, initial_metadata: dict):
        self._meta: dict = dict(initial_metadata)
        self.update_task_calls: list[dict] = []

    async def update_task(self, task_id, metadata, *, append=False):
        self.update_task_calls.append({
            'task_id': task_id,
            'metadata': metadata,
            'append': append,
        })
        if append:
            # Recursive-merge: extend lists, update other keys
            for key, value in metadata.items():
                if key in self._meta and isinstance(self._meta[key], list) and isinstance(value, list):
                    self._meta[key] = self._meta[key] + value
                else:
                    self._meta[key] = value
        else:
            # Full blob replace
            self._meta = dict(metadata)

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
