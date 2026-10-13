"""The task-creation journal: one write_op row per curator/planning-mode outcome.

Section 1 pins the row contract (``middleware/task_creation_journal.py``):
attribution parsed at the submit boundary, the operation vocabulary, and the
per-run reader Stage 2 grounds ``tasks_created`` / ``tasks_combined`` on.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.middleware.task_creation_journal import (
    TASK_COMBINED_OPERATION,
    TASK_CREATED_OPERATION,
    CreationAttribution,
    CreationOutcomes,
    creation_operation_for,
    read_run_creation_outcomes,
)

RUN = 'f9e49aef-0000-4000-8000-000000000001'
OTHER_RUN = '90d762d0-0000-4000-8000-000000000002'
STAGE2_AGENT = 'recon-stage-task_knowledge_sync'


# ── Section 1a: CreationAttribution.from_submission ─────────────────────────


class TestCreationAttributionFromSubmission:
    def test_dict_metadata_causation_is_kept_stripped(self):
        attribution = CreationAttribution.from_submission(
            {'_causation_id': f'  {RUN} ', 'source': 'x'}, STAGE2_AGENT,
        )

        assert attribution == CreationAttribution(agent_id=STAGE2_AGENT, causation_id=RUN)

    def test_json_object_string_metadata_is_parsed(self):
        attribution = CreationAttribution.from_submission(
            json.dumps({'_causation_id': RUN}), STAGE2_AGENT,
        )

        assert attribution.causation_id == RUN

    @pytest.mark.parametrize(
        'metadata',
        [
            None,
            {},
            {'_causation_id': ''},
            {'_causation_id': '   '},
            {'_causation_id': 42},
            {'_causation_id': None},
            'not json',
            '[1, 2]',
            json.dumps({'_causation_id': 7}),
            42,
        ],
        ids=[
            'none', 'empty', 'blank', 'whitespace', 'int', 'null', 'unparseable',
            'json-array', 'json-int-causation', 'int-metadata',
        ],
    )
    def test_unusable_causation_is_none(self, metadata):
        attribution = CreationAttribution.from_submission(metadata, STAGE2_AGENT)

        assert attribution.causation_id is None
        assert attribution.agent_id == STAGE2_AGENT

    def test_agent_id_none_is_allowed(self):
        attribution = CreationAttribution.from_submission({'_causation_id': RUN}, None)

        assert attribution == CreationAttribution(agent_id=None, causation_id=RUN)

    def test_instances_are_frozen(self):
        attribution = CreationAttribution(agent_id=STAGE2_AGENT, causation_id=RUN)

        with pytest.raises(AttributeError):
            attribution.causation_id = OTHER_RUN  # type: ignore[misc]


# ── Section 1b: blob round-trip ─────────────────────────────────────────────


class TestCreationAttributionBlobRoundTrip:
    @pytest.mark.parametrize(
        'attribution',
        [
            CreationAttribution(agent_id=STAGE2_AGENT, causation_id=RUN),
            CreationAttribution(agent_id=None, causation_id=RUN),
            CreationAttribution(agent_id=STAGE2_AGENT, causation_id=None),
            CreationAttribution(agent_id=None, causation_id=None),
        ],
    )
    def test_round_trips_through_json(self, attribution):
        blob_value = json.loads(json.dumps(attribution.to_blob_value()))

        assert CreationAttribution.from_blob_value(blob_value) == attribution

    @pytest.mark.parametrize(
        'value',
        [None, 'x', 42, ['a'], {}, {'agent_id': 5, 'causation_id': ['r']}],
        ids=['none', 'str', 'int', 'list', 'empty-dict', 'wrong-typed'],
    )
    def test_legacy_or_malformed_blob_value_is_unattributed(self, value):
        assert CreationAttribution.from_blob_value(value) == CreationAttribution(None, None)

    def test_each_field_is_validated_on_its_own(self):
        attribution = CreationAttribution.from_blob_value(
            {'agent_id': STAGE2_AGENT, 'causation_id': 99},
        )

        assert attribution == CreationAttribution(agent_id=STAGE2_AGENT, causation_id=None)


# ── Section 1c: operation vocabulary ────────────────────────────────────────


class TestCreationOperationFor:
    def test_created_and_combined_have_their_own_operations(self):
        assert creation_operation_for('created') == TASK_CREATED_OPERATION == 'task_created'
        assert creation_operation_for('combined') == TASK_COMBINED_OPERATION == 'task_combined'

    @pytest.mark.parametrize('status', ['failed', 'refused', 'cancelled', 'pending', None, ''])
    def test_every_other_outcome_has_no_creation_row(self, status):
        assert creation_operation_for(status) is None


# ── Section 1d: the per-run reader over a REAL WriteJournal ─────────────────


@pytest_asyncio.fixture
async def write_journal(tmp_path):
    from fused_memory.services.write_journal import WriteJournal

    journal = WriteJournal(tmp_path / 'wj')
    await journal.initialize()
    try:
        yield journal
    finally:
        await journal.close()


async def _log_row(
    journal,
    *,
    operation: str,
    task_id: object,
    project_id: str | None = 'solar_challenge',
    causation_id: str | None = RUN,
    success: bool = True,
    write_op_id: str | None = None,
) -> None:
    import uuid

    await journal.log_write_op(
        write_op_id=write_op_id or str(uuid.uuid4()),
        causation_id=causation_id,
        source='task_interceptor',
        operation=operation,
        project_id=project_id,
        agent_id=STAGE2_AGENT,
        params={'task_id': task_id},
        success=success,
    )


class TestReadRunCreationOutcomes:
    @pytest.mark.asyncio
    async def test_reads_created_and_combined_keys_for_the_run(self, write_journal):
        await _log_row(write_journal, operation=TASK_CREATED_OPERATION, task_id='105')
        await _log_row(
            write_journal, operation=TASK_COMBINED_OPERATION, task_id='42',
            project_id='dark_factory',
        )

        outcomes = await read_run_creation_outcomes(write_journal, RUN)

        assert outcomes == CreationOutcomes(
            created=frozenset({('solar_challenge', '105')}),
            combined=frozenset({('dark_factory', '42')}),
        )

    @pytest.mark.asyncio
    async def test_ignores_other_runs_failures_and_other_operations(self, write_journal):
        await _log_row(write_journal, operation=TASK_CREATED_OPERATION, task_id='105')
        await _log_row(
            write_journal, operation=TASK_CREATED_OPERATION, task_id='200',
            causation_id=OTHER_RUN,
        )
        await _log_row(
            write_journal, operation=TASK_CREATED_OPERATION, task_id='201', success=False,
        )
        await _log_row(write_journal, operation='update_task', task_id='202')
        await _log_row(write_journal, operation='add_memory', task_id='203')
        await write_journal.log_backend_op(
            causation_id=RUN, backend='taskmaster', operation=TASK_CREATED_OPERATION,
            payload={'task_id': '204'},
        )

        outcomes = await read_run_creation_outcomes(write_journal, RUN)

        assert outcomes == CreationOutcomes(
            created=frozenset({('solar_challenge', '105')}), combined=frozenset(),
        )

    @pytest.mark.asyncio
    async def test_task_id_forms_and_whitespace_collapse_to_one_key(self, write_journal):
        await _log_row(write_journal, operation=TASK_CREATED_OPERATION, task_id=105)
        await _log_row(write_journal, operation=TASK_CREATED_OPERATION, task_id=' 105 ')
        await _log_row(
            write_journal, operation=TASK_CREATED_OPERATION, task_id='105',
            project_id=' solar_challenge ',
        )

        outcomes = await read_run_creation_outcomes(write_journal, RUN)

        assert outcomes is not None
        assert outcomes.created == {('solar_challenge', '105')}

    @pytest.mark.asyncio
    @pytest.mark.parametrize('task_id', [None, '', '   '])
    async def test_row_without_a_task_id_is_skipped(self, write_journal, task_id):
        await _log_row(write_journal, operation=TASK_CREATED_OPERATION, task_id=task_id)

        outcomes = await read_run_creation_outcomes(write_journal, RUN)

        assert outcomes == CreationOutcomes(created=frozenset(), combined=frozenset())

    @pytest.mark.asyncio
    async def test_an_empty_run_is_empty_outcomes_not_unavailable(self, write_journal):
        assert await read_run_creation_outcomes(write_journal, RUN) == CreationOutcomes(
            created=frozenset(), combined=frozenset(),
        )

    @pytest.mark.asyncio
    async def test_row_with_undecodable_params_is_skipped(self):
        stub = MagicMock()
        stub.get_ops_by_causation = AsyncMock(return_value=[
            {'layer': 'write_op', 'operation': TASK_CREATED_OPERATION, 'causation_id': RUN,
             'project_id': 'p', 'params': 'not json', 'success': 1},
            {'layer': 'write_op', 'operation': TASK_CREATED_OPERATION, 'causation_id': RUN,
             'project_id': 'p', 'params': '[1]', 'success': 1},
            {'layer': 'write_op', 'operation': TASK_CREATED_OPERATION, 'causation_id': RUN,
             'project_id': 'p', 'params': {'task_id': '9'}, 'success': 1},
            'not-a-row',
        ])

        outcomes = await read_run_creation_outcomes(stub, RUN)

        assert outcomes == CreationOutcomes(created=frozenset({('p', '9')}), combined=frozenset())

    @pytest.mark.asyncio
    async def test_row_with_no_project_id_keys_as_none(self, write_journal):
        await _log_row(
            write_journal, operation=TASK_CREATED_OPERATION, task_id='5', project_id=None,
        )

        outcomes = await read_run_creation_outcomes(write_journal, RUN)

        assert outcomes is not None
        assert outcomes.created == {(None, '5')}


# ── Section 1e: journal unavailable ─────────────────────────────────────────


class TestReadRunCreationOutcomesUnavailable:
    @pytest.mark.asyncio
    async def test_no_journal_is_unavailable(self):
        assert await read_run_creation_outcomes(None, RUN) is None

    @pytest.mark.asyncio
    async def test_raising_read_is_unavailable(self):
        stub = MagicMock()
        stub.get_ops_by_causation = AsyncMock(side_effect=RuntimeError('db locked'))

        assert await read_run_creation_outcomes(stub, RUN) is None

    @pytest.mark.asyncio
    async def test_non_list_result_is_unavailable(self):
        """journal=AsyncMock() in tests/test_stages.py yields a MagicMock here."""
        journal = AsyncMock()

        assert await read_run_creation_outcomes(journal.write_journal, RUN) is None

    @pytest.mark.asyncio
    async def test_non_awaitable_read_is_unavailable(self):
        assert await read_run_creation_outcomes(MagicMock(), RUN) is None
