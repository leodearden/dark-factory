"""Tests for the task curator gate."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from shared.cli_invoke import AgentResult, AllAccountsCappedException

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import (
    CURATOR_OUTPUT_SCHEMA,
    CandidateTask,
    CuratorDecision,
    CuratorFailureError,
    TaskCurator,
    _parse_decision,
    _PoolEntry,
    _task_dependencies,
    _task_files,
    _to_pool_entry,
    _trim_pool,
    flatten_task_tree,
)

# ----------------------------------------------------------------------
# Pure helper tests
# ----------------------------------------------------------------------


class TestTaskFiles:
    def test_direct_files_to_modify(self):
        task = {'id': '1', 'files_to_modify': ['src/a.py', 'src/b.py']}
        assert _task_files(task) == ['src/a.py', 'src/b.py']

    def test_fallback_to_metadata_modules(self):
        task = {'id': '1', 'metadata': {'modules': ['crates/reify-compiler/src']}}
        assert _task_files(task) == ['crates/reify-compiler/src']

    def test_metadata_files_to_modify_fallback(self):
        task = {'id': '1', 'metadata': {'files_to_modify': ['src/a.py']}}
        assert _task_files(task) == ['src/a.py']

    def test_string_coerced_to_single_item_list(self):
        task = {'id': '1', 'files_to_modify': 'src/only.py'}
        assert _task_files(task) == ['src/only.py']

    def test_empty(self):
        assert _task_files({'id': '1'}) == []

    def test_filters_empty_strings(self):
        task = {'id': '1', 'files_to_modify': ['', 'src/a.py', None, '']}
        assert _task_files(task) == ['src/a.py']


class TestTaskDependencies:
    def test_list(self):
        assert _task_dependencies({'dependencies': ['1', '2', '3']}) == ['1', '2', '3']

    def test_csv_fallback(self):
        assert _task_dependencies({'dependencies': '1, 2,3'}) == ['1', '2', '3']

    def test_empty(self):
        assert _task_dependencies({}) == []
        assert _task_dependencies({'dependencies': None}) == []


class TestToPoolEntry:
    def test_pending_is_combine_eligible(self):
        task = {
            'id': '42',
            'title': 'Fix parser',
            'description': 'The parser is broken',
            'details': 'Details here',
            'status': 'pending',
            'priority': 'high',
            'files_to_modify': ['src/parser.py'],
        }
        entry = _to_pool_entry(task, source='module', lock_depth=2)
        assert entry is not None
        assert entry.task_id == '42'
        assert entry.combine_eligible is True
        assert entry.source == 'module'
        assert entry.module_keys == ['src/parser.py']

    def test_done_is_not_combine_eligible(self):
        task = {'id': '1', 'title': 'x', 'status': 'done'}
        entry = _to_pool_entry(task, source='module', lock_depth=2)
        assert entry is not None
        assert entry.combine_eligible is False

    def test_missing_id_returns_none(self):
        entry = _to_pool_entry({'title': 'x'}, source='module', lock_depth=2)
        assert entry is None

    def test_none_returns_none(self):
        assert _to_pool_entry(None, source='module', lock_depth=2) is None


class TestFlattenTaskTree:
    def test_flat(self):
        tasks = {'tasks': [{'id': '1'}, {'id': '2'}]}
        assert [t['id'] for t in flatten_task_tree(tasks)] == ['1', '2']

    def test_nested(self):
        tasks = {
            'tasks': [
                {'id': '1', 'subtasks': [{'id': '1.1'}, {'id': '1.2'}]},
                {'id': '2'},
            ],
        }
        assert [t['id'] for t in flatten_task_tree(tasks)] == ['1', '1.1', '1.2', '2']

    def test_data_wrapper(self):
        tasks = {'data': {'tasks': [{'id': '1'}]}}
        assert [t['id'] for t in flatten_task_tree(tasks)] == ['1']

    def test_empty(self):
        assert flatten_task_tree({}) == []

class TestTrimPool:
    def _entry(self, task_id: str, source: str) -> _PoolEntry:
        return _PoolEntry(
            task_id=task_id,
            title='',
            description='',
            details='',
            files_to_modify=[],
            module_keys=[],
            status='pending',
            priority='medium',
            source=source,
            combine_eligible=True,
        )

    def test_no_trim_when_under_cap(self):
        pool = [self._entry(str(i), 'module') for i in range(5)]
        assert len(_trim_pool(pool, 10)) == 5

    def test_trims_dependency_first(self):
        pool = (
            [self._entry('a', 'anchor')]
            + [self._entry(f'm{i}', 'module') for i in range(3)]
            + [self._entry(f'e{i}', 'embedding') for i in range(3)]
            + [self._entry(f'd{i}', 'dependency') for i in range(3)]
        )
        result = _trim_pool(pool, 7)
        sources = [e.source for e in result]
        # dependency dropped first, so no dependency entries remain
        assert 'dependency' not in sources
        assert len(result) == 7

    def test_anchor_preserved(self):
        pool = [self._entry('a', 'anchor')] + [
            self._entry(f'm{i}', 'module') for i in range(20)
        ]
        result = _trim_pool(pool, 5)
        assert any(e.source == 'anchor' for e in result)


class TestCandidateHash:
    def test_same_payload_same_hash(self):
        a = CandidateTask(title='T', description='D', details='X', files_to_modify=['a'])
        b = CandidateTask(title='T', description='D', details='X', files_to_modify=['a'])
        assert a.payload_hash() == b.payload_hash()

    def test_different_details_different_hash(self):
        a = CandidateTask(title='T', description='D', details='X')
        b = CandidateTask(title='T', description='D', details='Y')
        assert a.payload_hash() != b.payload_hash()

    def test_file_order_insensitive(self):
        a = CandidateTask(title='T', files_to_modify=['a', 'b'])
        b = CandidateTask(title='T', files_to_modify=['b', 'a'])
        assert a.payload_hash() == b.payload_hash()

    def test_spawned_from_affects_hash(self):
        a = CandidateTask(title='T', spawned_from='42')
        b = CandidateTask(title='T', spawned_from='43')
        assert a.payload_hash() != b.payload_hash()


# ----------------------------------------------------------------------
# Decision parsing tests — test the pure _parse_decision helper
# ----------------------------------------------------------------------


def _agent_result(structured: dict | None = None, output: str = '') -> AgentResult:
    return AgentResult(
        success=True,
        output=output,
        structured_output=structured,
        cost_usd=0.01,
    )


def _pool_with_ids(*pairs: tuple[str, str]) -> list[_PoolEntry]:
    """Build a pool with the given (task_id, status) pairs."""
    return [
        _PoolEntry(
            task_id=tid,
            title='t',
            description='',
            details='',
            files_to_modify=[],
            module_keys=[],
            status=status,
            priority='medium',
            source='module',
            combine_eligible=(status == 'pending'),
        )
        for tid, status in pairs
    ]


class TestParseDecision:
    def test_create_no_target(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = _parse_decision(
            _agent_result({'action': 'create', 'justification': 'novel work'}),
            pool_sizes={'m': 1}, latency_ms=100, pool=pool,
        )
        assert result.action == 'create'
        assert result.target_id is None
        assert 'novel work' in result.justification

    def test_drop_valid_target(self):
        pool = _pool_with_ids(('10', 'pending'), ('11', 'done'))
        result = _parse_decision(
            _agent_result({
                'action': 'drop', 'target_id': '11',
                'justification': 'already done in 11',
            }),
            pool_sizes={'m': 2}, latency_ms=100, pool=pool,
        )
        assert result.action == 'drop'
        assert result.target_id == '11'

    def test_drop_invalid_target_degrades_to_create(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = _parse_decision(
            _agent_result({'action': 'drop', 'target_id': '999', 'justification': '...'}),
            pool_sizes={}, latency_ms=10, pool=pool,
        )
        assert result.action == 'create'
        assert 'invalid-target' in result.justification

    def test_combine_into_pending_target(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = _parse_decision(
            _agent_result({
                'action': 'combine',
                'target_id': '10',
                'justification': 'same scope',
                'rewritten_task': {
                    'title': 'Fixed parser',
                    'description': 'unified',
                    'details': 'all the details',
                    'files_to_modify': ['src/parser.py'],
                    'priority': 'high',
                },
            }),
            pool_sizes={}, latency_ms=10, pool=pool,
        )
        assert result.action == 'combine'
        assert result.target_id == '10'
        assert result.rewritten_task is not None
        assert result.rewritten_task.title == 'Fixed parser'

    def test_combine_into_non_pending_degrades_to_create(self):
        pool = _pool_with_ids(('10', 'in-progress'))
        result = _parse_decision(
            _agent_result({
                'action': 'combine',
                'target_id': '10',
                'justification': '...',
                'rewritten_task': {
                    'title': 'x', 'description': '', 'details': 'd',
                    'files_to_modify': [], 'priority': 'medium',
                },
            }),
            pool_sizes={}, latency_ms=10, pool=pool,
        )
        assert result.action == 'create'
        assert 'invalid-combine-target' in result.justification

    def test_combine_missing_rewrite_degrades(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = _parse_decision(
            _agent_result({
                'action': 'combine', 'target_id': '10', 'justification': '...',
            }),
            pool_sizes={}, latency_ms=10, pool=pool,
        )
        assert result.action == 'create'
        assert 'combine-missing-rewrite' in result.justification

    def test_combine_empty_title_degrades(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = _parse_decision(
            _agent_result({
                'action': 'combine', 'target_id': '10', 'justification': '...',
                'rewritten_task': {
                    'title': '', 'description': '', 'details': 'd',
                    'files_to_modify': [], 'priority': 'medium',
                },
            }),
            pool_sizes={}, latency_ms=10, pool=pool,
        )
        assert result.action == 'create'
        assert 'empty-title' in result.justification

    def test_invalid_action_string_degrades(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = _parse_decision(
            _agent_result({'action': 'split', 'justification': '...'}),
            pool_sizes={}, latency_ms=10, pool=pool,
        )
        assert result.action == 'create'
        assert 'invalid-action' in result.justification

    def test_parses_from_raw_output_json(self):
        pool = _pool_with_ids(('10', 'pending'))
        output = '{"action": "create", "justification": "new"}'
        result = _parse_decision(
            AgentResult(success=True, output=output, structured_output=None),
            pool_sizes={}, latency_ms=10, pool=pool,
        )
        assert result.action == 'create'

    def test_malformed_output_degrades(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = _parse_decision(
            AgentResult(success=True, output='not json', structured_output=None),
            pool_sizes={}, latency_ms=10, pool=pool,
        )
        assert result.action == 'create'
        assert 'parse-failed' in result.justification


# ----------------------------------------------------------------------
# Output schema sanity
# ----------------------------------------------------------------------


class TestCuratorOutputSchema:
    def test_schema_is_jsonable(self):
        import json

        blob = json.dumps(CURATOR_OUTPUT_SCHEMA)
        assert 'action' in blob
        assert 'drop' in blob
        assert 'combine' in blob
        assert 'create' in blob

    def test_schema_has_required_fields(self):
        assert 'action' in CURATOR_OUTPUT_SCHEMA['required']
        assert 'justification' in CURATOR_OUTPUT_SCHEMA['required']


# ----------------------------------------------------------------------
# TaskCurator.curate() — idempotency cache + fallback behavior
# ----------------------------------------------------------------------


def _make_config() -> FusedMemoryConfig:
    cfg = FusedMemoryConfig()
    cfg.curator = CuratorConfig()
    return cfg


class TestCurateIdempotency:
    @pytest.mark.asyncio
    async def test_same_payload_reuses_cached_decision(self):
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        candidate = CandidateTask(title='Fix x', description='y')

        # Seed cache with a synthetic decision
        decision = CuratorDecision(action='drop', target_id='42', justification='cached')
        curator._store_cache(candidate.payload_hash(), decision)

        result = await curator.curate(candidate, project_id='p', project_root='/x')
        assert result.action == 'drop'
        assert result.target_id == '42'
        assert result.justification == 'cached'


class TestCurateFallbacks:
    @pytest.mark.asyncio
    async def test_corpus_failure_degrades_to_create(self):
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        async def boom(*a, **k):
            raise RuntimeError('qdrant down')

        with patch.object(curator, '_build_corpus', side_effect=boom):
            result = await curator.curate(
                CandidateTask(title='T'), project_id='p', project_root='/x',
            )
        assert result.action == 'create'
        assert 'corpus-failed' in result.justification

    @pytest.mark.asyncio
    async def test_llm_failure_degrades_to_create(self):
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        async def boom(*a, **k):
            raise RuntimeError('llm down')

        with patch.object(curator, '_build_corpus', side_effect=empty_corpus), \
             patch.object(curator, '_call_llm', side_effect=boom):
            result = await curator.curate(
                CandidateTask(title='T'), project_id='p', project_root='/x',
            )
        assert result.action == 'create'
        assert 'llm-failed' in result.justification

    @pytest.mark.asyncio
    async def test_llm_failure_without_escalator_degrades_loudly(self):
        """R1+R2: CLI failure with no escalator → action='create' with
        the ``llm-error-escalated`` sentinel justification.

        Distinct from the old ``llm-error`` / ``llm-failed`` strings so
        logs can be grepped to confirm R2's wiring (see plan verification
        checklist step 2: journalctl should show zero ``llm-error``
        entries unpaired with a ``curator_failure`` escalation).
        """
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        failing_result = AgentResult(
            success=False, output='auth error', structured_output=None,
        )
        with patch.object(curator, '_build_corpus', side_effect=empty_corpus), \
             patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                   new=AsyncMock(return_value=failing_result)):
            result = await curator.curate(
                CandidateTask(title='T'), project_id='p', project_root='/x',
            )
        assert result.action == 'create'
        assert result.justification == 'llm-error-escalated'

    @pytest.mark.asyncio
    async def test_llm_failure_with_escalator_calls_report_failure(self):
        """R2: CuratorFailureError routes through escalator when wired."""
        config = _make_config()

        escalator = AsyncMock()
        escalator.report_failure = AsyncMock(return_value=None)

        curator = TaskCurator(
            config=config, taskmaster=None, escalator=escalator,
        )

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        failing_result = AgentResult(
            success=False, output='auth error', structured_output=None,
        )
        with patch.object(curator, '_build_corpus', side_effect=empty_corpus), \
             patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                   new=AsyncMock(return_value=failing_result)):
            result = await curator.curate(
                CandidateTask(title='T'), project_id='p', project_root='/x',
            )
        assert result.action == 'create'
        assert result.justification == 'llm-error-escalated'
        escalator.report_failure.assert_awaited_once()
        kwargs = escalator.report_failure.await_args.kwargs
        assert kwargs['project_id'] == 'p'
        assert kwargs['project_root'] == '/x'
        assert kwargs['candidate_title'] == 'T'

    @pytest.mark.asyncio
    async def test_escalator_raise_propagates_out_of_curate(self):
        """R2: interactive path — escalator re-raise is not swallowed.

        When no orchestrator is running, CuratorEscalator re-raises
        CuratorFailureError. That must reach the MCP boundary so the
        caller sees a loud failure instead of a silent
        ``action='create'``.
        """
        config = _make_config()

        async def escalator_raise(**kwargs):
            raise CuratorFailureError('no orchestrator')

        escalator = AsyncMock()
        escalator.report_failure = escalator_raise
        curator = TaskCurator(
            config=config, taskmaster=None, escalator=escalator,
        )

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        failing_result = AgentResult(
            success=False, output='err', structured_output=None,
        )
        with (
            patch.object(curator, '_build_corpus', side_effect=empty_corpus),
            patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                  new=AsyncMock(return_value=failing_result)),
            pytest.raises(CuratorFailureError),
        ):
            await curator.curate(
                CandidateTask(title='T'), project_id='p', project_root='/x',
            )

    @pytest.mark.asyncio
    async def test_call_llm_raises_curator_failure_error_directly(self):
        """Direct _call_llm path raises CuratorFailureError when is_error."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        failing_result = AgentResult(
            success=False, output='error_max_turns', structured_output=None,
            subtype='error_max_turns', turns=2,
        )
        with (
            patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                  new=AsyncMock(return_value=failing_result)),
            pytest.raises(CuratorFailureError),
        ):
            await curator._call_llm(
                CandidateTask(title='T'),
                pool=[],
                pool_sizes={'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0},
                start=0.0,
                project_id='p',
                project_root='/x',
            )

    @pytest.mark.asyncio
    async def test_call_llm_failure_message_surfaces_timeout_diagnostics(self):
        """CuratorFailureError message includes timed_out and the configured
        timeout so the L1 escalation reads as '240s timeout' instead of
        'produced no output'.
        """
        config = _make_config()
        config.curator.timeout_seconds = 240.0
        curator = TaskCurator(config=config, taskmaster=None)

        timed_out_result = AgentResult(
            success=False,
            output='Agent produced no output',
            subtype='error_empty_output',
            timed_out=True,
            duration_ms=240_003,
        )
        with patch(
            'fused_memory.middleware.task_curator.invoke_with_cap_retry',
            new=AsyncMock(return_value=timed_out_result),
        ), pytest.raises(CuratorFailureError) as exc_info:
            await curator._call_llm(
                CandidateTask(title='T'),
                pool=[],
                pool_sizes={'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0},
                start=0.0,
                project_id='p',
                project_root='/x',
            )
        msg = str(exc_info.value)
        assert 'timed_out=True' in msg
        assert 'duration_ms=240003' in msg
        assert 'configured_timeout_secs=240' in msg
        # Error attaches structured attributes for CuratorEscalator.
        assert exc_info.value.timed_out is True
        assert exc_info.value.duration_ms == 240_003

    @pytest.mark.asyncio
    async def test_call_llm_salvages_schema_payload(self):
        """R1: CLI is_error=True + valid structured_output → parsed decision.

        With schema_salvage in cli_invoke, AgentResult arrives at _call_llm
        with success=True even when the CLI reported error_max_turns. We
        should parse the structured output as normal instead of raising.
        """
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        salvaged_result = AgentResult(
            success=True,  # salvage already flipped it
            output='error_max_turns',  # raw error text preserved
            structured_output={
                'action': 'drop',
                'target_id': '42',
                'justification': 'already covered',
            },
            schema_salvaged=True,
            subtype='error_max_turns',
            turns=2,
        )
        pool = _pool_with_ids(('42', 'pending'))
        with patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                   new=AsyncMock(return_value=salvaged_result)):
            decision = await curator._call_llm(
                CandidateTask(title='T'),
                pool=pool,
                pool_sizes={'anchor': 0, 'module': 1, 'embedding': 0, 'dependency': 0},
                start=0.0,
                project_id='p',
                project_root='/x',
            )
        assert decision.action == 'drop'
        assert decision.target_id == '42'

    @pytest.mark.asyncio
    async def test_call_llm_passes_max_turns_three(self):
        """R1: max_turns=1 was incompatible with --json-schema; must be >= 3."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        empty_result = AgentResult(
            success=True,
            output='',
            structured_output={'action': 'create', 'justification': 'x'},
        )
        mock = AsyncMock(return_value=empty_result)
        with patch(
            'fused_memory.middleware.task_curator.invoke_with_cap_retry',
            new=mock,
        ):
            await curator._call_llm(
                CandidateTask(title='T'),
                pool=[],
                pool_sizes={'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0},
                start=0.0,
                project_id='p',
                project_root='/x',
            )
        _, kwargs = mock.call_args
        assert kwargs['max_turns'] >= 3


class TestCuratorCapHandling:
    """Verify TaskCurator.curate() handles AllAccountsCappedException gracefully."""

    @pytest.mark.asyncio
    async def test_curate_handles_all_accounts_capped_without_escalator(self):
        """Cap exception → action='create', justification='all-accounts-capped', no raise."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        cap_exc = AllAccountsCappedException(
            retries=3, elapsed_secs=90.0, label='task-curator[p]'
        )

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        with patch.object(curator, '_build_corpus', side_effect=empty_corpus), \
             patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                   new=AsyncMock(side_effect=cap_exc)):
            result = await curator.curate(
                CandidateTask(title='T'), project_id='p', project_root='/x',
            )

        assert result.action == 'create'
        assert result.justification == 'all-accounts-capped'

    @pytest.mark.asyncio
    async def test_curate_handles_all_accounts_capped_with_escalator(self):
        """Cap exception with escalator → escalator.report_failure called with cap justification."""
        config = _make_config()

        escalator = AsyncMock()
        escalator.report_failure = AsyncMock(return_value=None)

        curator = TaskCurator(config=config, taskmaster=None, escalator=escalator)

        cap_exc = AllAccountsCappedException(
            retries=3, elapsed_secs=90.0, label='task-curator[p]'
        )

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        with patch.object(curator, '_build_corpus', side_effect=empty_corpus), \
             patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                   new=AsyncMock(side_effect=cap_exc)):
            result = await curator.curate(
                CandidateTask(title='T'), project_id='p', project_root='/x',
            )

        assert result.action == 'create'
        assert result.justification == 'all-accounts-capped'
        escalator.report_failure.assert_awaited_once()
        kwargs = escalator.report_failure.await_args.kwargs
        assert 'all-accounts-capped' in kwargs['justification']
        assert kwargs['project_id'] == 'p'
        assert kwargs['candidate_title'] == 'T'


class TestCurateHappyPath:
    @pytest.mark.asyncio
    async def test_create_flows_through_llm(self):
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        llm_result = AgentResult(
            success=True,
            output='',
            structured_output={
                'action': 'create',
                'justification': 'genuinely new',
            },
            cost_usd=0.02,
        )
        with patch.object(curator, '_build_corpus', side_effect=empty_corpus), \
             patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                   new=AsyncMock(return_value=llm_result)):
            result = await curator.curate(
                CandidateTask(title='T'), project_id='p', project_root='/x',
            )
        assert result.action == 'create'
        assert result.cost_usd == 0.02
        assert result.pool_sizes == {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

    @pytest.mark.asyncio
    async def test_drop_returns_target(self):
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        async def corpus_with_match(*a, **k):
            return (
                _pool_with_ids(('99', 'done')),
                {'anchor': 0, 'module': 1, 'embedding': 0, 'dependency': 0},
            )

        llm_result = AgentResult(
            success=True,
            output='',
            structured_output={
                'action': 'drop',
                'target_id': '99',
                'justification': 'already done in task 99',
            },
        )
        with patch.object(curator, '_build_corpus', side_effect=corpus_with_match), \
             patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                   new=AsyncMock(return_value=llm_result)):
            result = await curator.curate(
                CandidateTask(title='T'), project_id='p', project_root='/x',
            )
        assert result.action == 'drop'
        assert result.target_id == '99'

    @pytest.mark.asyncio
    async def test_combine_returns_rewritten(self):
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        async def corpus_with_pending(*a, **k):
            return (
                _pool_with_ids(('50', 'pending')),
                {'anchor': 0, 'module': 1, 'embedding': 0, 'dependency': 0},
            )

        llm_result = AgentResult(
            success=True,
            output='',
            structured_output={
                'action': 'combine',
                'target_id': '50',
                'justification': 'same root cause',
                'rewritten_task': {
                    'title': 'Harden parser',
                    'description': 'unified',
                    'details': 'Fix line 42 + line 67; add test cases a,b,c',
                    'files_to_modify': ['src/parser.py', 'tests/test_parser.py'],
                    'priority': 'high',
                },
            },
        )
        with patch.object(curator, '_build_corpus', side_effect=corpus_with_pending), \
             patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                   new=AsyncMock(return_value=llm_result)):
            result = await curator.curate(
                CandidateTask(title='T'), project_id='p', project_root='/x',
            )
        assert result.action == 'combine'
        assert result.target_id == '50'
        assert result.rewritten_task is not None
        assert result.rewritten_task.title == 'Harden parser'
        assert 'line 42' in result.rewritten_task.details  # specifics preserved


# ----------------------------------------------------------------------
# Build-corpus integration (mocked taskmaster, skips embedder)
# ----------------------------------------------------------------------


class TestBuildCorpus:
    @pytest.mark.asyncio
    async def test_anchor_is_always_included_when_spawned_from(self):
        config = _make_config()
        taskmaster = AsyncMock()
        anchor_task = {
            'id': '100',
            'title': 'Original task under review',
            'description': 'Parent',
            'details': 'Parent details',
            'status': 'in-progress',
            'priority': 'high',
            'files_to_modify': ['src/a.py'],
        }
        taskmaster.get_task = AsyncMock(return_value=anchor_task)
        taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})

        curator = TaskCurator(config=config, taskmaster=taskmaster)

        # Make stream 3 a no-op by forcing embedder failure
        async def fail_collection(*a, **k):
            raise RuntimeError('no qdrant')

        with patch.object(curator, '_ensure_collection', side_effect=fail_collection):
            pool, sizes = await curator._build_corpus(
                CandidateTask(title='Follow-up', spawned_from='100'),
                project_id='p', project_root='/x',
            )

        assert sizes['anchor'] == 1
        assert any(e.task_id == '100' and e.source == 'anchor' for e in pool)

    @pytest.mark.asyncio
    async def test_module_pool_matches_on_file_overlap(self):
        config = _make_config()
        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value=None)
        taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': '200',
                    'title': 'Pending work in same module',
                    'status': 'pending',
                    'priority': 'medium',
                    'files_to_modify': ['src/parser.py'],
                },
                {
                    'id': '201',
                    'title': 'Unrelated work',
                    'status': 'pending',
                    'priority': 'medium',
                    'files_to_modify': ['src/other.py'],
                },
            ],
        })

        curator = TaskCurator(config=config, taskmaster=taskmaster)

        async def fail_collection(*a, **k):
            raise RuntimeError('no qdrant')

        with patch.object(curator, '_ensure_collection', side_effect=fail_collection):
            pool, sizes = await curator._build_corpus(
                CandidateTask(title='New bug', files_to_modify=['src/parser.py']),
                project_id='p', project_root='/x',
            )

        assert sizes['module'] >= 1
        module_ids = [e.task_id for e in pool if e.source == 'module']
        assert '200' in module_ids
        assert '201' not in module_ids  # unrelated file, not in pool

    @pytest.mark.asyncio
    async def test_cancelled_tasks_excluded_from_module_pool(self):
        config = _make_config()
        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value=None)
        taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': '300',
                    'title': 'Cancelled',
                    'status': 'cancelled',
                    'files_to_modify': ['src/parser.py'],
                },
            ],
        })
        curator = TaskCurator(config=config, taskmaster=taskmaster)

        async def fail_collection(*a, **k):
            raise RuntimeError('no qdrant')

        with patch.object(curator, '_ensure_collection', side_effect=fail_collection):
            pool, sizes = await curator._build_corpus(
                CandidateTask(title='T', files_to_modify=['src/parser.py']),
                project_id='p', project_root='/x',
            )
        assert sizes['module'] == 0
        assert not any(e.task_id == '300' for e in pool)


# ----------------------------------------------------------------------
# Batch config fields
# ----------------------------------------------------------------------


class TestCuratorConfigBatch:
    def test_curator_config_has_batch_fields_with_documented_defaults(self):
        config = CuratorConfig()
        assert config.batch_max == 5
        assert config.per_item_slack_seconds == 30.0
        assert config.per_item_turns == 1
        assert config.batch_timeout_cap_seconds == 540.0
        assert config.batch_turns_cap == 10


# ----------------------------------------------------------------------
# curate_batch
# ----------------------------------------------------------------------


class TestCurateBatch:
    @pytest.mark.asyncio
    async def test_curate_batch_empty_returns_empty_list(self):
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        result = await curator.curate_batch([], project_id='p', project_root='/x')
        assert result == []

    @pytest.mark.asyncio
    async def test_curate_batch_single_item_delegates_to_curate(self):
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        decision = CuratorDecision(action='drop', target_id='7', justification='cached')
        candidate = CandidateTask(title='Fix parser')
        curator.curate = AsyncMock(return_value=decision)
        result = await curator.curate_batch([candidate], project_id='p', project_root='/x')
        assert result == [decision]
        curator.curate.assert_awaited_once_with(candidate, 'p', '/x')
