"""The task-creation journal: one write_op row per curator/planning-mode outcome.

Section 1 pins the row contract (``middleware/task_creation_journal.py``):
attribution parsed at the submit boundary, the operation vocabulary, and the
per-run reader Stage 2 grounds ``tasks_created`` / ``tasks_combined`` on.
Section 2 drives TaskInterceptor through every creation path against a real
WriteJournal and reads the rows back through that reader.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from _fm_helpers import poll_until

from fused_memory.backends.task_backend_errors import DuplicateCandidateKeyError
from fused_memory.middleware.task_creation_journal import (
    TASK_COMBINED_OPERATION,
    TASK_CREATED_OPERATION,
    CreationAttribution,
    CreationOutcomes,
    creation_operation_for,
    read_run_creation_outcomes,
)
from fused_memory.middleware.task_curator import CuratorDecision, RewrittenTask
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

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


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: TaskInterceptor writes the journal on every creation path
# ═══════════════════════════════════════════════════════════════════════════
#
# Fixtures replicated minimally from tests/test_ticket_worker.py (module-private
# fixtures are not cross-imported in this repo).

_CREATION_OPERATIONS = {TASK_CREATED_OPERATION, TASK_COMBINED_OPERATION}
_RUN_METADATA = {'_causation_id': RUN, 'source': 'stage2'}


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '42', 'title': 'New Task'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.ensure_connected = AsyncMock()
    return tm


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'journal_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest_asyncio.fixture
async def ticket_store(tmp_path):
    store = TicketStore(tmp_path / 'journal_tickets.db')
    await store.initialize()
    yield store
    await store.close()


async def _cancel_workers(interceptor: TaskInterceptor) -> None:
    workers = list(getattr(interceptor, '_worker_tasks', {}).values())
    for worker in workers:
        if not worker.done():
            worker.cancel()
    for worker in workers:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await worker


@pytest_asyncio.fixture
async def journaled_interceptor(taskmaster, event_buffer, ticket_store, write_journal):
    interceptor = TaskInterceptor(taskmaster, None, event_buffer, ticket_store=ticket_store)
    interceptor.set_write_journal(write_journal)
    yield interceptor
    await _cancel_workers(interceptor)


@pytest_asyncio.fixture
async def unjournaled_interceptor(taskmaster, event_buffer, ticket_store):
    interceptor = TaskInterceptor(taskmaster, None, event_buffer, ticket_store=ticket_store)
    yield interceptor
    await _cancel_workers(interceptor)


def _curator_returning(*decisions: CuratorDecision) -> MagicMock:
    """A curator whose single-item ``curate`` returns *decisions* in order."""
    curator = MagicMock()
    curator.prepare_candidate = AsyncMock(return_value=None)
    curator.curate = AsyncMock(side_effect=list(decisions))
    curator.note_created = MagicMock()
    curator.record_task = AsyncMock()
    curator.reembed_task = AsyncMock(return_value=None)
    return curator


async def _poll_ticket_resolved(ticket_store, ticket_id: str) -> dict:
    async def _resolved():
        row = await ticket_store.get(ticket_id)
        return row if row is not None and row['status'] != 'pending' else None

    return await poll_until(_resolved, message=f'worker did not resolve {ticket_id}')


async def _poll_outcomes(journal, run_id: str, predicate) -> CreationOutcomes:
    async def _read():
        outcomes = await read_run_creation_outcomes(journal, run_id)
        return outcomes if outcomes is not None and predicate(outcomes) else None

    return await poll_until(_read, message=f'no matching creation rows for {run_id}')


async def _creation_rows(journal) -> list[dict]:
    rows = await journal.get_ops_since('', limit=1000)
    return [row for row in rows if row['operation'] in _CREATION_OPERATIONS]


async def _submit(interceptor, curator, *, project_root='/project', metadata=None, **kwargs):
    with patch.object(type(interceptor), '_get_curator', new=AsyncMock(return_value=curator)):
        result = await interceptor.submit_task(
            project_root=project_root,
            agent_id=STAGE2_AGENT,
            title=kwargs.pop('title', 'Journal Task'),
            description=kwargs.pop('description', 'A task for the journal'),
            metadata=_RUN_METADATA if metadata is None else metadata,
            **kwargs,
        )
    return result


class TestPlanningModeJournal:
    @pytest.mark.asyncio
    async def test_planning_mode_create_writes_one_task_created_row(
        self, journaled_interceptor, write_journal,
    ):
        result = await journaled_interceptor.submit_task(
            project_root='/project', agent_id=STAGE2_AGENT, planning_mode=True,
            title='Planned', description='d', metadata=_RUN_METADATA,
        )

        assert result['task_id'] == '42'
        assert await read_run_creation_outcomes(write_journal, RUN) == CreationOutcomes(
            created=frozenset({('project', '42')}), combined=frozenset(),
        )
        [row] = await _creation_rows(write_journal)
        assert (row['agent_id'], row['causation_id'], row['project_id']) == (
            STAGE2_AGENT, RUN, 'project',
        )

    @pytest.mark.asyncio
    async def test_planning_mode_candidate_key_collision_writes_task_combined(
        self, journaled_interceptor, write_journal, taskmaster,
    ):
        taskmaster.add_task = AsyncMock(side_effect=DuplicateCandidateKeyError(existing_id=7))

        result = await journaled_interceptor.submit_task(
            project_root='/project', agent_id=STAGE2_AGENT, planning_mode=True,
            title='Planned', description='d', metadata=_RUN_METADATA,
        )

        assert result['combined'] is True
        assert await read_run_creation_outcomes(write_journal, RUN) == CreationOutcomes(
            created=frozenset(), combined=frozenset({('project', '7')}),
        )


class TestCuratorWorkerJournal:
    @pytest.mark.asyncio
    async def test_create_row_carries_the_submitters_attribution(
        self, journaled_interceptor, write_journal, ticket_store,
    ):
        result = await _submit(
            journaled_interceptor, _curator_returning(CuratorDecision(action='create')),
        )
        await _poll_ticket_resolved(ticket_store, result['ticket'])

        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.created)

        assert outcomes == CreationOutcomes(
            created=frozenset({('project', '42')}), combined=frozenset(),
        )
        [row] = await _creation_rows(write_journal)
        assert (row['agent_id'], row['causation_id']) == (STAGE2_AGENT, RUN)

    @pytest.mark.asyncio
    async def test_drop_writes_task_combined_for_the_target(
        self, journaled_interceptor, write_journal, taskmaster,
    ):
        curator = _curator_returning(
            CuratorDecision(action='drop', target_id='5', justification='dup'),
        )

        await _submit(journaled_interceptor, curator)
        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.combined)

        assert outcomes == CreationOutcomes(
            created=frozenset(), combined=frozenset({('project', '5')}),
        )
        taskmaster.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_combine_writes_task_combined_for_the_target(
        self, journaled_interceptor, write_journal,
    ):
        curator = _curator_returning(CuratorDecision(
            action='combine',
            target_id='42',
            rewritten_task=RewrittenTask(
                title='Combined', description='d', details='', files_to_modify=[],
                priority='medium',
            ),
            justification='similar scope',
        ))

        async def fake_execute_combine(project_root, decision, **_kwargs):
            return {'updated': True, 'target_id': decision.target_id}

        with patch.object(
            journaled_interceptor, '_execute_combine', side_effect=fake_execute_combine,
        ):
            await _submit(journaled_interceptor, curator)
            outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.combined)

        assert outcomes.combined == {('project', '42')}
        assert outcomes.created == frozenset()

    @pytest.mark.asyncio
    async def test_refused_and_failed_outcomes_write_no_creation_row(
        self, journaled_interceptor, write_journal, taskmaster, ticket_store,
    ):
        """Refuse, then a create whose add_task raises, then a sentinel create:
        tickets drain in order, so once the sentinel's row lands the first two
        have already been handled and must have left nothing."""
        taskmaster.add_task = AsyncMock(side_effect=[
            RuntimeError('backend down'),
            {'id': '43', 'title': 'Sentinel'},
        ])
        curator = _curator_returning(
            CuratorDecision(action='refuse', justification='blocked premise'),
            CuratorDecision(action='create'),
            CuratorDecision(action='create'),
        )

        refused = await _submit(journaled_interceptor, curator, title='Refused')
        failed = await _submit(journaled_interceptor, curator, title='Failed')
        await _submit(journaled_interceptor, curator, title='Sentinel')
        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.created)

        assert (await ticket_store.get(refused['ticket']))['status'] == 'refused'
        assert (await ticket_store.get(failed['ticket']))['status'] == 'failed'
        assert outcomes == CreationOutcomes(
            created=frozenset({('project', '43')}), combined=frozenset(),
        )
        assert len(await _creation_rows(write_journal)) == 1

    @pytest.mark.asyncio
    async def test_cross_project_row_names_the_target_project(
        self, journaled_interceptor, write_journal,
    ):
        """The f9e49aef / task 105 shape: Stage 2 of one project files into another."""
        await _submit(
            journaled_interceptor, _curator_returning(CuratorDecision(action='create')),
            project_root='/home/leo/src/solar_challenge',
        )

        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.created)

        assert outcomes.created == {('solar_challenge', '42')}

    @pytest.mark.asyncio
    async def test_add_task_never_sees_attribution_kwargs(
        self, journaled_interceptor, write_journal, taskmaster,
    ):
        await _submit(
            journaled_interceptor, _curator_returning(CuratorDecision(action='create')),
        )
        await _poll_outcomes(write_journal, RUN, lambda o: o.created)

        forwarded = taskmaster.add_task.call_args.kwargs
        assert 'agent_id' not in forwarded
        assert 'creation_attribution' not in forwarded

    @pytest.mark.asyncio
    async def test_submission_without_causation_still_writes_an_unattributed_row(
        self, journaled_interceptor, write_journal,
    ):
        await _submit(
            journaled_interceptor, _curator_returning(CuratorDecision(action='create')),
            metadata={'source': 'no-causation'},
        )

        async def _row():
            rows = await _creation_rows(write_journal)
            return rows or None

        [row] = await poll_until(_row, message='no unattributed creation row')

        assert row['causation_id'] is None
        assert row['agent_id'] == STAGE2_AGENT
        assert await read_run_creation_outcomes(write_journal, RUN) == CreationOutcomes(
            created=frozenset(), combined=frozenset(),
        )


class TestIdempotencyHitJournal:
    _EXISTING = {
        'id': '99',
        'title': 'Existing Escalation Task',
        'status': 'pending',
        'metadata': {'escalation_id': 'e1', 'suggestion_hash': 'h1'},
    }
    _METADATA = {**_RUN_METADATA, 'escalation_id': 'e1', 'suggestion_hash': 'h1'}

    @pytest.mark.asyncio
    async def test_single_ticket_path_writes_task_combined(
        self, journaled_interceptor, write_journal, taskmaster, ticket_store,
    ):
        taskmaster.get_tasks = AsyncMock(return_value={'tasks': [self._EXISTING]})
        ticket_id = await ticket_store.submit('project', json.dumps({
            'project_root': '/project',
            'kwargs': {'title': 'Escalation Task', 'description': 'd'},
            'metadata': self._METADATA,
            'creation_attribution': CreationAttribution(STAGE2_AGENT, RUN).to_blob_value(),
        }))

        with patch.object(
            type(journaled_interceptor), '_get_curator',
            new=AsyncMock(return_value=_curator_returning()),
        ):
            await journaled_interceptor._process_add_ticket(ticket_id)

        assert (await ticket_store.get(ticket_id))['reason'] == 'idempotency_hit'
        assert await read_run_creation_outcomes(write_journal, RUN) == CreationOutcomes(
            created=frozenset(), combined=frozenset({('project', '99')}),
        )

    @pytest.mark.asyncio
    async def test_batch_path_writes_task_combined(
        self, journaled_interceptor, write_journal, taskmaster, ticket_store,
    ):
        taskmaster.get_tasks = AsyncMock(return_value={'tasks': [self._EXISTING]})

        result = await _submit(
            journaled_interceptor, _curator_returning(), metadata=self._METADATA,
        )
        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.combined)

        assert (await ticket_store.get(result['ticket']))['reason'] == 'idempotency_hit'
        assert outcomes == CreationOutcomes(
            created=frozenset(), combined=frozenset({('project', '99')}),
        )
        taskmaster.add_task.assert_not_called()


class TestNoJournalWired:
    @pytest.mark.asyncio
    async def test_submit_and_resolve_behave_as_before(
        self, unjournaled_interceptor, ticket_store,
    ):
        result = await _submit(
            unjournaled_interceptor, _curator_returning(CuratorDecision(action='create')),
        )

        row = await _poll_ticket_resolved(ticket_store, result['ticket'])

        assert (row['status'], row['task_id']) == ('created', '42')

    @pytest.mark.asyncio
    async def test_planning_mode_behaves_as_before(self, unjournaled_interceptor):
        result = await unjournaled_interceptor.submit_task(
            project_root='/project', agent_id=STAGE2_AGENT, planning_mode=True,
            title='Planned', description='d', metadata=_RUN_METADATA,
        )

        assert result == {'task_id': '42', 'status': 'deferred', 'planning_mode': True}
