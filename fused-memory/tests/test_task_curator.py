"""Tests for the task curator gate."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from shared.cli_invoke import AgentResult, AllAccountsCappedException

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import (
    CURATOR_BATCH_OUTPUT_SCHEMA,
    CURATOR_OUTPUT_SCHEMA,
    CandidateTask,
    CuratorDecision,
    CuratorFailureError,
    TaskCurator,
    normalize_title,
    _parse_decision,
    _parse_decision_dict,
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


class TestHashShapeContract16:
    """Shape-contract regression guard for the two task_curator.py sha256 mirror sites
    that lack an existing 16-char length assertion.

    The canonical 16-char sha256-hex shape is owned by
    ``orchestrator.agents.triage.sha256_16`` — see the docstring there for the full
    enumeration of mirror sites.  The third curator mirror site,
    ``TaskCurator._intra_batch_key``, is already guarded by
    ``TestIntraBatchKey.test_returns_16_hex_chars`` (line ~1868); do not add duplicate
    coverage here.  Anyone adding a fourth mirror site should add a corresponding
    assertion to that test or this class.

    These tests do NOT introspect docstrings — they verify actual hash output length,
    mirroring the ``TestSha256_16.test_length_is_16`` pattern in
    ``orchestrator/tests/test_triage_module.py``.
    """

    def test_payload_hash_is_16_chars(self):
        """CandidateTask.payload_hash() returns a 16-char hex string."""
        c = CandidateTask(title='t', description='d', details='x', files_to_modify=['f'])
        assert len(c.payload_hash()) == 16

    def test_normalize_key_is_16_chars(self):
        """TaskCurator._normalize_key() returns a 16-char hex string."""
        c = CandidateTask(title='t', files_to_modify=['f'])
        assert len(TaskCurator._normalize_key(c)) == 16


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


# ----------------------------------------------------------------------
# Batch output schema
# ----------------------------------------------------------------------


class TestCuratorBatchOutputSchema:
    def test_batch_schema_is_jsonable(self):
        import json
        raw = json.dumps(CURATOR_BATCH_OUTPUT_SCHEMA)
        assert 'decisions' in raw
        assert 'candidate_index' in raw
        assert 'batch_target_index' in raw
        assert 'action' in raw

    def test_batch_schema_requires_decisions(self):
        assert 'decisions' in CURATOR_BATCH_OUTPUT_SCHEMA['required']
        items_schema = CURATOR_BATCH_OUTPUT_SCHEMA['properties']['decisions']['items']
        assert 'action' in items_schema['required']
        assert 'candidate_index' in items_schema['required']
        assert 'justification' in items_schema['required']


# ----------------------------------------------------------------------
# _parse_decision_dict + CuratorDecision.batch_target_index
# ----------------------------------------------------------------------


class TestParseDecisionDict:
    """Mirrors TestParseDecision but calls _parse_decision_dict directly."""

    def _call(self, raw: dict, pool=None, *, pool_sizes=None, latency_ms=50, cost_usd=0.01):
        return _parse_decision_dict(
            raw,
            pool=pool or [],
            pool_sizes=pool_sizes or {},
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        )

    def test_create_no_target(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = self._call({'action': 'create', 'justification': 'novel work'}, pool)
        assert result.action == 'create'
        assert result.target_id is None
        assert 'novel work' in result.justification

    def test_drop_valid_target(self):
        pool = _pool_with_ids(('10', 'pending'), ('11', 'done'))
        result = self._call(
            {'action': 'drop', 'target_id': '11', 'justification': 'already done'},
            pool,
        )
        assert result.action == 'drop'
        assert result.target_id == '11'

    def test_drop_invalid_target_degrades(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = self._call({'action': 'drop', 'target_id': '999', 'justification': '...'}, pool)
        assert result.action == 'create'
        assert 'invalid-target' in result.justification

    def test_combine_into_pending(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = self._call(
            {
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
            },
            pool,
        )
        assert result.action == 'combine'
        assert result.rewritten_task is not None
        assert result.rewritten_task.title == 'Fixed parser'

    def test_combine_into_non_pending_degrades(self):
        pool = _pool_with_ids(('10', 'in-progress'))
        result = self._call(
            {
                'action': 'combine',
                'target_id': '10',
                'justification': '...',
                'rewritten_task': {
                    'title': 'x', 'description': '', 'details': 'd',
                    'files_to_modify': [], 'priority': 'medium',
                },
            },
            pool,
        )
        assert result.action == 'create'
        assert 'invalid-combine-target' in result.justification

    def test_combine_missing_rewrite_degrades(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = self._call({'action': 'combine', 'target_id': '10', 'justification': '...'}, pool)
        assert result.action == 'create'
        assert 'combine-missing-rewrite' in result.justification

    def test_combine_empty_title_degrades(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = self._call(
            {
                'action': 'combine', 'target_id': '10', 'justification': '...',
                'rewritten_task': {
                    'title': '', 'description': '', 'details': 'd',
                    'files_to_modify': [], 'priority': 'medium',
                },
            },
            pool,
        )
        assert result.action == 'create'
        assert 'empty-title' in result.justification

    def test_invalid_action_degrades(self):
        pool = _pool_with_ids(('10', 'pending'))
        result = self._call({'action': 'split', 'justification': '...'}, pool)
        assert result.action == 'create'
        assert 'invalid-action' in result.justification

    def test_parse_decision_dict_accepts_batch_target_index(self):
        """A drop with batch_target_index (no existing target_id) should preserve it."""
        # When a candidate drops to another batch item, target_id is None (no
        # existing task) and batch_target_index carries the sibling's index.
        result = self._call(
            {
                'action': 'drop',
                'target_id': None,
                'batch_target_index': 2,
                'justification': 'dup of batch item 2',
            },
            pool=[],  # no existing pool — target is within-batch
        )
        assert result.action == 'drop'
        assert result.batch_target_index == 2

    def test_parse_decision_dict_non_numeric_batch_target_index_does_not_raise(self):
        """A schema-drifted batch_target_index (e.g. 'foo') must not raise
        ValueError out of the single-item path; it should degrade to None so
        the parser's "invalid field → create" contract holds."""
        pool = _pool_with_ids(('10', 'pending'))
        # Non-numeric string → int() would raise ValueError; coercion must
        # swallow that and fall through to a safe create.
        result = self._call(
            {
                'action': 'create',
                'batch_target_index': 'foo',
                'justification': 'normal work',
            },
            pool,
        )
        assert result.action == 'create'
        assert result.batch_target_index is None

    def test_parse_decision_dict_float_batch_target_index_coerces(self):
        """float-as-string should also not raise — fall back to None."""
        result = self._call(
            {
                'action': 'create',
                'batch_target_index': '1.5',
                'justification': 'x',
            },
            pool=[],
        )
        assert result.action == 'create'
        assert result.batch_target_index is None

    def test_ambiguous_drop_degrades_to_create(self):
        """drop with BOTH target_id and batch_target_index set must degrade.

        The LLM violated the prompt by emitting both a pool reference and a
        within-batch reference.  Accepting it would silently discard one field
        downstream; degrading to 'create' is the safer fallback.
        """
        pool = _pool_with_ids(('task-99', 'pending'))
        result = self._call(
            {
                'action': 'drop',
                'target_id': 'task-99',
                'batch_target_index': 1,
                'justification': 'ambiguous',
            },
            pool,
        )
        assert result.action == 'create'
        assert 'ambiguous-drop' in result.justification


# ----------------------------------------------------------------------
# _parse_batch_decisions
# ----------------------------------------------------------------------


class TestParseBatchDecisions:
    def _make_result(self, decisions: list[dict]) -> AgentResult:
        return AgentResult(
            success=True,
            output='',
            structured_output={'decisions': decisions},
            cost_usd=0.03,
        )

    def test_returns_ordered_decisions_by_candidate_index(self):
        from fused_memory.middleware.task_curator import _parse_batch_decisions
        pool0 = _pool_with_ids(('10', 'pending'))
        pool1: list[_PoolEntry] = []
        pool2: list[_PoolEntry] = []
        result = _parse_batch_decisions(
            self._make_result([
                {'candidate_index': 2, 'action': 'create', 'justification': 'c'},
                {'candidate_index': 0, 'action': 'drop', 'target_id': '10', 'justification': 'd'},
                {'candidate_index': 1, 'action': 'create', 'justification': 'e'},
            ]),
            pools=[pool0, pool1, pool2],
            pool_sizes_list=[{'module': 1}, {}, {}],
            latency_ms=100,
        )
        assert len(result) == 3
        assert result[0].action == 'drop'
        assert result[1].action == 'create'
        assert result[2].action == 'create'

    def test_per_item_parse_failure_degrades_only_that_item(self):
        """A single bad decision dict (invalid action) degrades only that item."""
        from fused_memory.middleware.task_curator import _parse_batch_decisions
        result = _parse_batch_decisions(
            self._make_result([
                {'candidate_index': 0, 'action': 'create', 'justification': 'ok'},
                {'candidate_index': 1, 'action': 'garbage'},
                {'candidate_index': 2, 'action': 'create', 'justification': 'ok2'},
            ]),
            pools=[[], [], []],
            pool_sizes_list=[{}, {}, {}],
            latency_ms=50,
        )
        assert len(result) == 3
        assert result[0].action == 'create'
        assert result[0].justification == 'ok'
        assert result[1].action == 'create'
        assert 'invalid-action' in result[1].justification
        assert result[2].action == 'create'
        assert result[2].justification == 'ok2'

    def test_missing_item_index_degrades_only_that_item(self):
        """When a decision for index i is absent from the batch, only that item degrades."""
        from fused_memory.middleware.task_curator import _parse_batch_decisions
        # Decisions for indices 0 and 2 only — index 1 is missing.
        result = _parse_batch_decisions(
            self._make_result([
                {'candidate_index': 0, 'action': 'create', 'justification': 'first'},
                {'candidate_index': 2, 'action': 'create', 'justification': 'third'},
            ]),
            pools=[[], [], []],
            pool_sizes_list=[{}, {}, {}],
            latency_ms=50,
        )
        assert len(result) == 3
        assert result[0].action == 'create'
        assert result[0].justification == 'first'
        assert result[1].action == 'create'
        assert 'batch-item-missing' in result[1].justification
        assert result[2].action == 'create'
        assert result[2].justification == 'third'


# ----------------------------------------------------------------------
# TaskCurator._build_batch_user_prompt
# ----------------------------------------------------------------------


class TestBuildBatchUserPrompt:
    def test_renders_per_candidate_sections_with_pools(self):
        """Prompt contains a labelled block per candidate with its own pool."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        candidates = [
            CandidateTask(title='Alpha task', description='do alpha'),
            CandidateTask(title='Beta task', description='do beta'),
        ]
        pool0 = _pool_with_ids(('task-11', 'pending'), ('task-12', 'done'))
        pool1 = _pool_with_ids(('task-99', 'in-progress'))

        result = curator._build_batch_user_prompt(candidates, [pool0, pool1])

        # Section headers
        assert '# Candidate batch_index=0' in result
        assert '# Candidate batch_index=1' in result
        # Both candidate titles present
        assert 'Alpha task' in result
        assert 'Beta task' in result
        # Pool task ids rendered for each candidate's pool
        assert 'task-11' in result
        assert 'task-12' in result
        assert 'task-99' in result
        # Final instruction line references 'decisions'
        assert 'decisions' in result.lower()


# ----------------------------------------------------------------------
# TaskCurator._call_llm_batch — timeout / turns scaling
# ----------------------------------------------------------------------


class TestCallLlmBatchScaling:
    def _make_batch_result(self, n: int) -> AgentResult:
        """Minimal successful AgentResult with N create decisions."""
        decisions = [
            {'candidate_index': i, 'action': 'create', 'justification': f'c{i}'}
            for i in range(n)
        ]
        return AgentResult(
            success=True,
            output='',
            structured_output={'decisions': decisions},
            cost_usd=0.01 * n,
        )

    @pytest.mark.asyncio
    async def test_timeout_and_turns_scale_linearly(self):
        """N=3 → timeout=240+30*(3-1)=300, max_turns=3+1*(3-1)=5.

        The formula uses (n-1): baseline timeout/3-turns already cover item 0;
        each additional item beyond the first adds per_item_slack/per_item_turns.
        """
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        mock = AsyncMock(return_value=self._make_batch_result(3))
        candidates = [CandidateTask(title=f'T{i}') for i in range(3)]
        with (
            patch(
                'fused_memory.middleware.task_curator.invoke_with_cap_retry',
                new=mock,
            ),
            patch.object(curator, '_build_corpus', side_effect=Exception('no corpus')),
        ):
            await curator._call_llm_batch(
                candidates,
                pools=[[], [], []],
                pool_sizes_list=[{}, {}, {}],
                start=0.0,
                project_id='p',
                project_root='/x',
            )
        _, kwargs = mock.call_args
        assert kwargs['timeout_seconds'] == 300.0
        assert kwargs['max_turns'] == 5

    @pytest.mark.asyncio
    async def test_scaling_clamps_at_caps(self):
        """N=20 → timeout capped at 540.0, max_turns capped at 10."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        mock = AsyncMock(return_value=self._make_batch_result(20))
        candidates = [CandidateTask(title=f'T{i}') for i in range(20)]
        with (
            patch(
                'fused_memory.middleware.task_curator.invoke_with_cap_retry',
                new=mock,
            ),
            patch.object(curator, '_build_corpus', side_effect=Exception('no corpus')),
        ):
            await curator._call_llm_batch(
                candidates,
                pools=[[] for _ in range(20)],
                pool_sizes_list=[{} for _ in range(20)],
                start=0.0,
                project_id='p',
                project_root='/x',
            )
        _, kwargs = mock.call_args
        assert kwargs['timeout_seconds'] == 540.0
        assert kwargs['max_turns'] == 10


# ----------------------------------------------------------------------
# TaskCurator._call_llm_batch — failure error propagation
# ----------------------------------------------------------------------


class TestCallLlmBatchFailure:
    @pytest.mark.asyncio
    async def test_raises_curator_failure_error_when_agent_result_not_success(self):
        """Non-success AgentResult raises CuratorFailureError with batch diagnostics."""
        from fused_memory.middleware.task_curator import CuratorFailureError
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        failed_result = AgentResult(
            success=False,
            output='max_turns',
            subtype='error_max_turns',
            turns=8,
            timed_out=False,
            duration_ms=120000,
        )
        mock = AsyncMock(return_value=failed_result)
        candidates = [CandidateTask(title='A'), CandidateTask(title='B')]
        with (
            pytest.raises(CuratorFailureError) as exc_info,
            patch(
                'fused_memory.middleware.task_curator.invoke_with_cap_retry',
                new=mock,
            ),
        ):
            await curator._call_llm_batch(
                candidates,
                pools=[[], []],
                pool_sizes_list=[{}, {}],
                start=0.0,
                project_id='p',
                project_root='/x',
            )
        msg = str(exc_info.value)
        assert 'batch_size=2' in msg
        assert 'error_max_turns' in msg
        assert 'configured_timeout_secs' in msg


# ----------------------------------------------------------------------
# curate_batch N>1 happy path
# ----------------------------------------------------------------------


class TestCurateBatchHappyPath:
    @pytest.mark.asyncio
    async def test_curate_batch_n3_returns_ordered_decisions(self):
        """N=3 batch issues exactly one LLM call and returns per-candidate decisions."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        candidates = [
            CandidateTask(title='Alpha'),
            CandidateTask(title='Beta'),
            CandidateTask(title='Gamma'),
        ]
        # Pool for candidate 1 has task '50' so the drop target is valid.
        pools = [
            [],
            _pool_with_ids(('50', 'done')),
            [],
        ]
        sizes = [
            {'anchor': 0},
            {'anchor': 1},
            {'anchor': 0},
        ]

        call_idx = 0

        async def fake_corpus(candidate, project_id, project_root):
            nonlocal call_idx
            result = (pools[call_idx], sizes[call_idx])
            call_idx += 1
            return result

        batch_result = AgentResult(
            success=True,
            output='',
            structured_output={'decisions': [
                {'candidate_index': 0, 'action': 'create', 'justification': 'new'},
                {'candidate_index': 1, 'action': 'drop', 'target_id': '50',
                 'justification': 'dup'},
                {'candidate_index': 2, 'action': 'create', 'justification': 'newer'},
            ]},
            cost_usd=0.05,
        )
        mock = AsyncMock(return_value=batch_result)
        with patch.object(curator, '_build_corpus', side_effect=fake_corpus), \
             patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                   new=mock):
            result = await curator.curate_batch(candidates, 'p', '/x')

        assert len(result) == 3
        assert [d.action for d in result] == ['create', 'drop', 'create']
        assert result[1].target_id == '50'
        assert mock.await_count == 1


# ----------------------------------------------------------------------
# curate_batch: pre-batch-dedup must not pollute the payload-hash cache
# ----------------------------------------------------------------------


class TestCurateBatchPreDedupCachePollution:
    """Regression: synthetic pre-batch-dedup decisions share a payload_hash with
    the first sibling's real LLM decision.  Caching them would overwrite the
    real decision with a degenerate ``action='drop', target_id=None`` entry,
    and a later single-item ``curate()`` hit on that hash would then return
    the synthetic drop — which ``_process_add_ticket`` cannot safely dispatch
    and would interpret as "create a duplicate task".
    """

    @pytest.mark.asyncio
    async def test_duplicate_candidate_does_not_overwrite_real_decision_in_cache(self):
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        # Two candidates with IDENTICAL payload_hash
        c1 = CandidateTask(title='Same', description='same-body')
        c2 = CandidateTask(title='Same', description='same-body')
        assert c1.payload_hash() == c2.payload_hash()

        payload_hash = c1.payload_hash()

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        # The LLM only sees the unique candidate (the first one); emit a 'create'.
        batch_result = AgentResult(
            success=True,
            output='',
            structured_output={'decisions': [
                {'candidate_index': 0, 'action': 'create', 'justification': 'real'},
            ]},
            cost_usd=0.01,
        )
        with patch.object(curator, '_build_corpus', side_effect=empty_corpus), \
             patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                   new=AsyncMock(return_value=batch_result)):
            result = await curator.curate_batch([c1, c2], 'p', '/x')

        # Decision order preserved: c1 is the 'create', c2 is the synthetic drop.
        assert result[0].action == 'create'
        assert result[1].action == 'drop'
        assert result[1].batch_target_index == 0

        # The cache for the shared payload_hash must hold the REAL 'create'
        # decision, NOT the synthetic drop.
        # _decision_cache stores (decision, timestamp)
        cached = curator._decision_cache.get(payload_hash)
        cached_decision = cached[0] if isinstance(cached, tuple) else cached
        assert cached_decision is not None
        assert cached_decision.action == 'create'
        assert cached_decision.batch_target_index is None


# ----------------------------------------------------------------------
# curate_batch: whole-batch fallback on LLM failure
# ----------------------------------------------------------------------


class TestCurateBatchWholeBatchFailure:
    @pytest.mark.asyncio
    async def test_curator_failure_error_falls_back_to_per_candidate_curate(self):
        """CuratorFailureError from _call_llm_batch triggers N serial curate() calls."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        c1 = CandidateTask(title='Task One')
        c2 = CandidateTask(title='Task Two')

        fallback_decisions = [
            CuratorDecision(action='create', justification='fb1'),
            CuratorDecision(action='drop', target_id='9', justification='fb2'),
        ]

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        with patch.object(curator, '_build_corpus', side_effect=empty_corpus), \
             patch.object(
                 curator, '_call_llm_batch',
                 new=AsyncMock(side_effect=CuratorFailureError(
                     'whole-batch-failed', timed_out=False, duration_ms=5000,
                 )),
             ), \
             patch.object(
                 curator, 'curate',
                 new=AsyncMock(side_effect=fallback_decisions),
             ) as mock_curate:
            result = await curator.curate_batch([c1, c2], 'p', '/x')

        assert result == fallback_decisions
        assert mock_curate.await_count == 2
        # Called in order: c1 first, c2 second.
        assert mock_curate.call_args_list[0].args[0] is c1
        assert mock_curate.call_args_list[1].args[0] is c2


# ----------------------------------------------------------------------
# curate_batch: fallback on AllAccountsCappedException
# ----------------------------------------------------------------------


class TestCurateBatchAllAccountsCapped:
    @pytest.mark.asyncio
    async def test_cap_exception_falls_back_to_per_candidate_curate(self):
        """AllAccountsCappedException from _call_llm_batch triggers N serial curate() calls."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        c1 = CandidateTask(title='Cap One')
        c2 = CandidateTask(title='Cap Two')

        fallback_decisions = [
            CuratorDecision(action='create', justification='cap-fb1'),
            CuratorDecision(action='create', justification='cap-fb2'),
        ]

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        with patch.object(curator, '_build_corpus', side_effect=empty_corpus), \
             patch.object(
                 curator, '_call_llm_batch',
                 new=AsyncMock(side_effect=AllAccountsCappedException(
                     retries=3, elapsed_secs=90.0,
                     label='task-curator-batch[p]',
                 )),
             ), \
             patch.object(
                 curator, 'curate',
                 new=AsyncMock(side_effect=fallback_decisions),
             ) as mock_curate:
            result = await curator.curate_batch([c1, c2], 'p', '/x')

        assert result == fallback_decisions
        assert mock_curate.await_count == 2
        assert mock_curate.call_args_list[0].args[0] is c1
        assert mock_curate.call_args_list[1].args[0] is c2


# ----------------------------------------------------------------------
# batch_target_index validation and cycle detection
# ----------------------------------------------------------------------


class TestBatchTargetIndex:
    def _make_agent_result(self, decisions: list[dict]) -> AgentResult:
        return AgentResult(
            success=True,
            output='',
            structured_output={'decisions': decisions},
            cost_usd=0.02,
        )

    def test_within_batch_drop_preserves_batch_target_index(self):
        """A valid within-batch drop keeps action='drop', target_id=None, and
        batch_target_index correctly populated."""
        from fused_memory.middleware.task_curator import _parse_batch_decisions
        ar = self._make_agent_result([
            {'candidate_index': 0, 'action': 'create', 'justification': 'new'},
            {'candidate_index': 1, 'action': 'drop', 'batch_target_index': 0,
             'justification': 'dup of 0'},
        ])
        pools = [[], []]
        sizes = [{'anchor': 0}, {'anchor': 0}]
        result = _parse_batch_decisions(ar, pools=pools, pool_sizes_list=sizes,
                                        latency_ms=10)
        assert result[1].action == 'drop'
        assert result[1].target_id is None
        assert result[1].batch_target_index == 0

    def test_batch_target_cycle_degrades_both_to_create(self):
        """A→B and B→A cycle: both items must be degraded to 'create' with
        'batch-cycle' in their justification."""
        from fused_memory.middleware.task_curator import _parse_batch_decisions
        ar = self._make_agent_result([
            {'candidate_index': 0, 'action': 'drop', 'batch_target_index': 1,
             'justification': 'dup of 1'},
            {'candidate_index': 1, 'action': 'drop', 'batch_target_index': 0,
             'justification': 'dup of 0'},
        ])
        pools = [[], []]
        sizes = [{'anchor': 0}, {'anchor': 0}]
        result = _parse_batch_decisions(ar, pools=pools, pool_sizes_list=sizes,
                                        latency_ms=10)
        assert result[0].action == 'create'
        assert result[1].action == 'create'
        assert 'batch-cycle' in result[0].justification
        assert 'batch-cycle' in result[1].justification


# ----------------------------------------------------------------------
# curate_batch: batch_target_index remap from unique-space to original-space
# ----------------------------------------------------------------------


class TestCurateBatchBatchTargetIndexRemap:
    """Regression for reviewer_comprehensive/data_correctness finding.

    When ``unique_indices`` is non-contiguous (due to pre-batch dedup),
    a within-batch ``batch_target_index`` emitted by the LLM is in
    *unique-space* (positions 0..K-1).  The merge site must remap it to
    *original-space* (positions in the full ``candidates`` list) before
    returning, because ``_process_add_tickets_batch`` keys
    ``resolved_task_ids`` by original-space index.
    """

    @pytest.mark.asyncio
    async def test_batch_target_index_remapped_to_original_space_when_unique_indices_noncontiguous(
        self,
    ):
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)

        # Six candidates: [A, B, A(dup), C, B(dup), D]
        # payload_hash groups: A={0,2}, B={1,4}, C={3}, D={5}
        A1 = CandidateTask(title='A', description='group-a')
        B1 = CandidateTask(title='B', description='group-b')
        A2 = CandidateTask(title='A', description='group-a')  # dup of A1
        C = CandidateTask(title='C', description='group-c')
        B2 = CandidateTask(title='B', description='group-b')  # dup of B1
        D = CandidateTask(title='D', description='group-d')

        # Verify payload_hash groupings
        assert A1.payload_hash() == A2.payload_hash()
        assert B1.payload_hash() == B2.payload_hash()
        assert A1.payload_hash() != B1.payload_hash()
        assert C.payload_hash() not in {
            A1.payload_hash(), B1.payload_hash(), D.payload_hash(),
        }

        # Pre-batch dedup produces:
        #   unique_indices = [0, 1, 3, 5]
        #   pre_dedup_decisions = {2: drop→0, 4: drop→1}
        # The LLM sees unique_candidates = [A1, B1, C, D] (positions 0..3).
        #
        # LLM emits batch_target_index=2 for D, meaning "position 2 in
        # unique-space" = C.  After remapping, original-space index of C is
        # unique_indices[2] = 3.

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        batch_result = AgentResult(
            success=True,
            output='',
            structured_output={'decisions': [
                {'candidate_index': 0, 'action': 'create', 'justification': 'A'},
                {'candidate_index': 1, 'action': 'create', 'justification': 'B'},
                {'candidate_index': 2, 'action': 'create', 'justification': 'C'},
                {'candidate_index': 3, 'action': 'drop', 'batch_target_index': 2,
                 'justification': 'D dups C within batch'},
            ]},
            cost_usd=0.04,
        )
        with patch.object(curator, '_build_corpus', side_effect=empty_corpus), \
             patch('fused_memory.middleware.task_curator.invoke_with_cap_retry',
                   new=AsyncMock(return_value=batch_result)):
            decisions = await curator.curate_batch([A1, B1, A2, C, B2, D], 'p', '/x')

        # (a) correct length
        assert len(decisions) == 6

        # (b) D (original index 5): batch_target_index must be ORIGINAL-space
        #     index of C (=3), NOT unique-space 2 (which maps to A2, the
        #     duplicate at original index 2).
        assert decisions[5].action == 'drop'
        assert decisions[5].batch_target_index == 3, (
            f'Expected original-space index 3 (C), got {decisions[5].batch_target_index!r} '
            '(unique-space 2 = A2 is wrong)'
        )
        # (c) still a within-batch drop (no existing task)
        assert decisions[5].target_id is None

        # (d) pre-dedup decisions: already in original-space — must be untouched
        assert decisions[2].action == 'drop'
        assert decisions[2].batch_target_index == 0  # A2 → A1 (original index 0)
        assert decisions[4].action == 'drop'
        assert decisions[4].batch_target_index == 1  # B2 → B1 (original index 1)

        # (e) sanity on unique candidates
        assert decisions[0].action == 'create'
        assert decisions[1].action == 'create'
        assert decisions[3].action == 'create'

        # Second regression assertion: downstream consumer's lookup works correctly.
        resolved_task_ids = {3: 'task-for-C', 0: 'task-for-A'}
        assert resolved_task_ids.get(decisions[5].batch_target_index) == 'task-for-C'


# ----------------------------------------------------------------------
# curate_batch: decision cache check before LLM (suggestion 1)
# ----------------------------------------------------------------------


class TestCurateBatchCacheCheck:
    """curate_batch consults _check_cache before building corpus or calling LLM.

    Cache hits are returned directly; only uncached unique candidates are sent
    to _call_llm_batch.  If ALL unique candidates are cache hits, the LLM is
    never called.
    """

    @pytest.mark.asyncio
    async def test_all_cache_hits_skip_llm(self):
        """Three candidates all in cache → LLM is never called."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        c1 = CandidateTask(title='Alpha')
        c2 = CandidateTask(title='Beta')
        c3 = CandidateTask(title='Gamma')

        d1 = CuratorDecision(action='create', justification='cached-1')
        d2 = CuratorDecision(action='create', justification='cached-2')
        d3 = CuratorDecision(action='create', justification='cached-3')
        curator._store_cache(c1.payload_hash(), d1)
        curator._store_cache(c2.payload_hash(), d2)
        curator._store_cache(c3.payload_hash(), d3)

        mock_llm = AsyncMock()
        with patch.object(curator, '_call_llm_batch', new=mock_llm), \
             patch.object(curator, '_build_corpus', side_effect=Exception('no call expected')):
            result = await curator.curate_batch([c1, c2, c3], 'p', '/x')

        assert mock_llm.await_count == 0, 'LLM must not be called when all are cache hits'
        assert len(result) == 3
        assert result[0].justification == 'cached-1'
        assert result[1].justification == 'cached-2'
        assert result[2].justification == 'cached-3'

    @pytest.mark.asyncio
    async def test_partial_cache_hit_sends_only_uncached_to_llm(self):
        """Two candidates cached, one uncached → LLM call contains only the uncached one."""
        config = _make_config()
        curator = TaskCurator(config=config, taskmaster=None)
        c_cached1 = CandidateTask(title='Cached-One')
        c_new = CandidateTask(title='Brand-New')
        c_cached2 = CandidateTask(title='Cached-Two')

        d_cached1 = CuratorDecision(action='create', justification='c1-cached')
        d_cached2 = CuratorDecision(action='create', justification='c2-cached')
        curator._store_cache(c_cached1.payload_hash(), d_cached1)
        curator._store_cache(c_cached2.payload_hash(), d_cached2)

        llm_result = AgentResult(
            success=True,
            output='',
            structured_output={'decisions': [
                {'candidate_index': 0, 'action': 'create', 'justification': 'new-from-llm'},
            ]},
            cost_usd=0.01,
        )

        async def empty_corpus(*a, **k):
            return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}

        call_candidates: list = []

        async def capture_batch(candidates, pools, pool_sizes_list, start, pid, proot):
            call_candidates.extend(candidates)
            from fused_memory.middleware.task_curator import _parse_batch_decisions
            return _parse_batch_decisions(
                llm_result, pools=pools, pool_sizes_list=pool_sizes_list, latency_ms=0,
            )

        with patch.object(curator, '_build_corpus', side_effect=empty_corpus), \
             patch.object(curator, '_call_llm_batch', side_effect=capture_batch):
            result = await curator.curate_batch(
                [c_cached1, c_new, c_cached2], 'p', '/x',
            )

        # LLM should only see the one uncached candidate.
        assert len(call_candidates) == 1
        assert call_candidates[0] is c_new

        # Result order matches original candidate list.
        assert len(result) == 3
        assert result[0].justification == 'c1-cached'
        assert result[1].justification == 'new-from-llm'
        assert result[2].justification == 'c2-cached'


# ─────────────────────────────────────────────────────────────────────
# TaskCurator._intra_batch_key (step-1: tests written before impl)
# ─────────────────────────────────────────────────────────────────────


class TestIntraBatchKey:
    """Unit tests for the intra-batch dedup key helper.

    These tests are written before the implementation exists (TDD step-1);
    they must FAIL until _intra_batch_key is added in step-2.
    """

    def test_identical_inputs_produce_same_key(self):
        """(a) identical inputs → identical keys."""
        k1 = TaskCurator._intra_batch_key('Fix the parser', 'The parser is broken')
        k2 = TaskCurator._intra_batch_key('Fix the parser', 'The parser is broken')
        assert k1 == k2

    def test_title_case_normalised(self):
        """(b) case difference on title is normalised away."""
        k1 = TaskCurator._intra_batch_key('Fix Bug', 'same description')
        k2 = TaskCurator._intra_batch_key('fix bug', 'same description')
        assert k1 == k2

    def test_whitespace_drift_normalised(self):
        """(c) leading/trailing and internal whitespace collapse to same key."""
        k1 = TaskCurator._intra_batch_key('  Fix   bug ', 'desc')
        k2 = TaskCurator._intra_batch_key('fix bug', 'desc')
        assert k1 == k2

    def test_description_whitespace_normalised(self):
        """(c) description whitespace drift also normalised."""
        k1 = TaskCurator._intra_batch_key('title', '  do   something ')
        k2 = TaskCurator._intra_batch_key('title', 'do something')
        assert k1 == k2

    def test_different_description_yields_different_key(self):
        """(d) same title but different description → different keys."""
        k1 = TaskCurator._intra_batch_key('Fix parser', 'in module A')
        k2 = TaskCurator._intra_batch_key('Fix parser', 'in module B')
        assert k1 != k2

    def test_empty_description_does_not_raise(self):
        """(e) empty description is handled without raising."""
        k = TaskCurator._intra_batch_key('Some title', '')
        assert isinstance(k, str)

    def test_none_description_does_not_raise(self):
        """(e) None description is handled without raising."""
        k = TaskCurator._intra_batch_key('Some title', None)  # type: ignore[arg-type]
        assert isinstance(k, str)

    def test_returns_16_hex_chars(self):
        """(f) returns a stable 16-character hex string."""
        k = TaskCurator._intra_batch_key('hello', 'world')
        assert len(k) == 16
        assert all(c in '0123456789abcdef' for c in k)

    def test_empty_title_and_description(self):
        """(e) both empty — no raise, returns 16-char string."""
        k = TaskCurator._intra_batch_key('', '')
        assert len(k) == 16


# ─────────────────────────────────────────────────────────────────────
# normalize_title module-level helper
# ─────────────────────────────────────────────────────────────────────


class TestNormalizeTitle:
    """Unit tests for the module-level normalize_title helper."""

    def test_normalises_to_expected_output(self):
        """(a) normalised output is lowercase with collapsed whitespace."""
        assert normalize_title('Fix the parser') == 'fix the parser'

    def test_case_folding(self):
        """(b) case difference is normalised away."""
        assert normalize_title('Fix Bug') == normalize_title('fix bug')

    def test_whitespace_collapse(self):
        """(c) leading/trailing and internal whitespace collapse."""
        assert normalize_title('  Fix   bug ') == 'fix bug'
        assert normalize_title('  Fix   bug ') == normalize_title('fix bug')

    def test_none_input_returns_empty_string(self):
        """(d) None input → empty string (defensive None handling)."""
        assert normalize_title(None) == ''  # type: ignore[arg-type]

    def test_empty_input_returns_empty_string(self):
        """(e) empty string → empty string."""
        assert normalize_title('') == ''
