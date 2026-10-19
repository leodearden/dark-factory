"""The task-creation journal: one write_op row per curator/planning-mode outcome.

Section 1 pins the row contract (``middleware/task_creation_journal.py``):
attribution parsed at the submit boundary, the operation vocabulary, and the
per-run reader Stage 2 grounds ``tasks_created`` / ``tasks_combined`` on.
Section 2 drives TaskInterceptor through every creation path against a real
WriteJournal and reads the rows back through that reader.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock

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
    task_key,
)
from fused_memory.middleware.task_curator import CuratorDecision, RewrittenTask
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

RUN = 'f9e49aef-0000-4000-8000-000000000001'
OTHER_RUN = '90d762d0-0000-4000-8000-000000000002'
STAGE2_AGENT = 'recon-stage-task_knowledge_sync'
STAGE2_PROJECT = 'dark_factory'


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

        for field in dataclasses.fields(attribution):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(attribution, field.name, None)


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


# ── Section 1d: task_key — the one key rule for rows and records ────────────


class TestTaskKey:
    @pytest.mark.parametrize(
        'project_id, task_id, expected',
        [
            ('reify', '7', ('reify', '7')),
            (' reify ', ' 7 ', ('reify', '7')),
            ('reify', 7, ('reify', '7')),
            (None, '7', (STAGE2_PROJECT, '7')),
            ('', '7', (STAGE2_PROJECT, '7')),
            ('   ', '7', (STAGE2_PROJECT, '7')),
        ],
        ids=['plain', 'whitespace', 'int-id', 'no-project', 'empty-project', 'blank-project'],
    )
    def test_normalises_both_parts(self, project_id, task_id, expected):
        assert task_key(project_id, task_id, STAGE2_PROJECT) == expected

    @pytest.mark.parametrize('task_id', [None, '', '   '])
    def test_missing_task_id_is_no_key(self, task_id):
        assert task_key('reify', task_id, STAGE2_PROJECT) is None

    def test_default_project_may_itself_be_none(self):
        assert task_key(None, '7', None) == (None, '7')


# ── Section 1e: the per-run reader over a REAL WriteJournal ─────────────────


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


async def _read(journal, run_id: str = RUN) -> CreationOutcomes | None:
    """Read *run_id*'s outcomes as Stage 2 of ``STAGE2_PROJECT`` does."""
    return await read_run_creation_outcomes(journal, run_id, default_project_id=STAGE2_PROJECT)


class TestReadRunCreationOutcomes:
    @pytest.mark.asyncio
    async def test_reads_created_and_combined_keys_for_the_run(self, write_journal):
        await _log_row(write_journal, operation=TASK_CREATED_OPERATION, task_id='105')
        await _log_row(
            write_journal, operation=TASK_COMBINED_OPERATION, task_id='42',
            project_id='dark_factory',
        )

        outcomes = await _read(write_journal)

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

        outcomes = await _read(write_journal)

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

        outcomes = await _read(write_journal)

        assert outcomes is not None
        assert outcomes.created == {('solar_challenge', '105')}

    @pytest.mark.asyncio
    @pytest.mark.parametrize('task_id', [None, '', '   '])
    async def test_row_without_a_task_id_is_skipped(self, write_journal, task_id):
        await _log_row(write_journal, operation=TASK_CREATED_OPERATION, task_id=task_id)

        outcomes = await _read(write_journal)

        assert outcomes == CreationOutcomes(created=frozenset(), combined=frozenset())

    @pytest.mark.asyncio
    async def test_an_empty_run_is_empty_outcomes_not_unavailable(self, write_journal):
        assert await _read(write_journal) == CreationOutcomes(
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

        outcomes = await _read(stub)

        assert outcomes == CreationOutcomes(created=frozenset({('p', '9')}), combined=frozenset())

    @pytest.mark.asyncio
    @pytest.mark.parametrize('project_id', [None, '', '   '])
    async def test_row_with_no_project_id_keys_under_the_default_project(
        self, write_journal, project_id,
    ):
        await _log_row(
            write_journal, operation=TASK_CREATED_OPERATION, task_id='5', project_id=project_id,
        )

        outcomes = await _read(write_journal)

        assert outcomes is not None
        assert outcomes.created == {(STAGE2_PROJECT, '5')}

    @pytest.mark.asyncio
    async def test_rows_and_records_share_one_key_rule(self, write_journal):
        """A journal row and a Stage 2 record naming the same task normalise to
        the same key, so grounding never counts one task twice."""
        await _log_row(
            write_journal, operation=TASK_CREATED_OPERATION, task_id=' 5 ', project_id=None,
        )

        outcomes = await _read(write_journal)

        assert outcomes is not None
        assert outcomes.created == {task_key(None, 5, STAGE2_PROJECT)}


# ── Section 1f: journal unavailable ─────────────────────────────────────────


class TestReadRunCreationOutcomesUnavailable:
    @pytest.mark.asyncio
    async def test_no_journal_is_unavailable(self):
        assert await _read(None) is None

    @pytest.mark.asyncio
    async def test_raising_read_is_unavailable(self):
        stub = MagicMock()
        stub.get_ops_by_causation = AsyncMock(side_effect=RuntimeError('db locked'))

        assert await _read(stub) is None

    @pytest.mark.asyncio
    async def test_non_list_result_is_unavailable(self):
        """journal=AsyncMock() in tests/test_stages.py yields a MagicMock here."""
        journal = AsyncMock()

        assert await _read(journal.write_journal) is None

    @pytest.mark.asyncio
    async def test_non_awaitable_read_is_unavailable(self):
        assert await _read(MagicMock()) is None


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: TaskInterceptor writes the journal on every creation path
# ═══════════════════════════════════════════════════════════════════════════
#
# Driven through the public seams only: the curator is injected through the
# constructor, tickets go in through submit_task and come out through
# resolve_ticket, and rows are read back through read_run_creation_outcomes.

_CREATION_OPERATIONS = {TASK_CREATED_OPERATION, TASK_COMBINED_OPERATION}
_RUN_METADATA = {'_causation_id': RUN, 'source': 'stage2'}
_PROJECT_ROOT = '/project'


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


def _curator_returning(*decisions: CuratorDecision) -> MagicMock:
    """A curator whose single-item ``curate`` returns *decisions* in order."""
    curator = MagicMock()
    curator.prepare_candidate = AsyncMock(return_value=None)
    curator.curate = AsyncMock(side_effect=list(decisions))
    curator.note_created = MagicMock()
    curator.record_task = AsyncMock()
    curator.reembed_task = AsyncMock(return_value=None)
    curator.close = AsyncMock()
    return curator


@pytest_asyncio.fixture
async def build_interceptor(taskmaster, event_buffer, ticket_store, write_journal):
    """Build TaskInterceptors around an injected curator; each is closed at teardown."""
    built: list[TaskInterceptor] = []

    def _build(curator: MagicMock | None = None, *, journaled: bool = True) -> TaskInterceptor:
        interceptor = TaskInterceptor(
            taskmaster, None, event_buffer, ticket_store=ticket_store, curator=curator,
        )
        if journaled:
            interceptor.set_write_journal(write_journal)
        built.append(interceptor)
        return interceptor

    yield _build
    for interceptor in built:
        await interceptor.close()


async def _poll_outcomes(journal, run_id: str, predicate) -> CreationOutcomes:
    async def _matching():
        outcomes = await _read(journal, run_id)
        return outcomes if outcomes is not None and predicate(outcomes) else None

    return await poll_until(_matching, message=f'no matching creation rows for {run_id}')


async def _creation_rows(journal) -> list[dict]:
    rows = await journal.get_stored_ops_since('', limit=1000)
    return [row for row in rows if row['operation'] in _CREATION_OPERATIONS]


async def _submit(interceptor, *, project_root=_PROJECT_ROOT, metadata=None, **kwargs) -> str:
    """Submit a task as Stage 2 does and return its ticket id."""
    result = await interceptor.submit_task(
        project_root=project_root,
        agent_id=STAGE2_AGENT,
        title=kwargs.pop('title', 'Journal Task'),
        description=kwargs.pop('description', 'A task for the journal'),
        metadata=_RUN_METADATA if metadata is None else metadata,
        **kwargs,
    )
    return result['ticket']


async def _resolve(interceptor, ticket: str, project_root: str = _PROJECT_ROOT) -> dict:
    return await interceptor.resolve_ticket(ticket, project_root, timeout_seconds=10)


class TestPlanningModeJournal:
    @pytest.mark.asyncio
    async def test_planning_mode_create_writes_one_task_created_row(
        self, build_interceptor, write_journal,
    ):
        result = await build_interceptor().submit_task(
            project_root=_PROJECT_ROOT, agent_id=STAGE2_AGENT, planning_mode=True,
            title='Planned', description='d', metadata=_RUN_METADATA,
        )

        assert result['task_id'] == '42'
        assert await _read(write_journal) == CreationOutcomes(
            created=frozenset({('project', '42')}), combined=frozenset(),
        )
        [row] = await _creation_rows(write_journal)
        assert (row['agent_id'], row['causation_id'], row['project_id']) == (
            STAGE2_AGENT, RUN, 'project',
        )

    @pytest.mark.asyncio
    async def test_planning_mode_candidate_key_collision_writes_task_combined(
        self, build_interceptor, write_journal, taskmaster,
    ):
        taskmaster.add_task = AsyncMock(side_effect=DuplicateCandidateKeyError(existing_id=7))

        result = await build_interceptor().submit_task(
            project_root=_PROJECT_ROOT, agent_id=STAGE2_AGENT, planning_mode=True,
            title='Planned', description='d', metadata=_RUN_METADATA,
        )

        assert result['combined'] is True
        assert await _read(write_journal) == CreationOutcomes(
            created=frozenset(), combined=frozenset({('project', '7')}),
        )


class TestCuratorWorkerJournal:
    @pytest.mark.asyncio
    async def test_create_row_carries_the_submitters_attribution(
        self, build_interceptor, write_journal,
    ):
        interceptor = build_interceptor(_curator_returning(CuratorDecision(action='create')))

        resolved = await _resolve(interceptor, await _submit(interceptor))
        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.created)

        assert (resolved['status'], resolved['task_id']) == ('created', '42')
        assert outcomes == CreationOutcomes(
            created=frozenset({('project', '42')}), combined=frozenset(),
        )
        [row] = await _creation_rows(write_journal)
        assert (row['agent_id'], row['causation_id']) == (STAGE2_AGENT, RUN)

    @pytest.mark.asyncio
    async def test_drop_writes_task_combined_for_the_target(
        self, build_interceptor, write_journal, taskmaster,
    ):
        interceptor = build_interceptor(_curator_returning(
            CuratorDecision(action='drop', target_id='5', justification='dup'),
        ))

        resolved = await _resolve(interceptor, await _submit(interceptor))
        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.combined)

        assert resolved['status'] == 'combined'
        assert outcomes == CreationOutcomes(
            created=frozenset(), combined=frozenset({('project', '5')}),
        )
        taskmaster.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_combine_writes_task_combined_for_the_target(
        self, build_interceptor, write_journal, taskmaster, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv('DARK_FACTORY_DATA_DIR', str(tmp_path / 'data'))
        taskmaster.get_task = AsyncMock(return_value={
            'id': '42', 'title': 'Existing Task', 'status': 'pending', 'metadata': {},
        })
        interceptor = build_interceptor(_curator_returning(CuratorDecision(
            action='combine',
            target_id='42',
            target_fingerprint='Existing Task',
            rewritten_task=RewrittenTask(
                title='Combined', description='d', details='', files_to_modify=[],
                priority='medium',
            ),
            justification='similar scope',
        )))

        resolved = await _resolve(interceptor, await _submit(interceptor))
        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.combined)

        assert (resolved['status'], resolved['task_id']) == ('combined', '42')
        assert outcomes == CreationOutcomes(
            created=frozenset(), combined=frozenset({('project', '42')}),
        )
        taskmaster.add_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_refused_and_failed_outcomes_write_no_creation_row(
        self, build_interceptor, write_journal, taskmaster,
    ):
        """Refuse, then a create whose add_task raises, then a sentinel create:
        tickets drain in order, so once the sentinel's row lands the first two
        have already been handled and must have left nothing."""
        taskmaster.add_task = AsyncMock(side_effect=[
            RuntimeError('backend down'),
            {'id': '43', 'title': 'Sentinel'},
        ])
        interceptor = build_interceptor(_curator_returning(
            CuratorDecision(action='refuse', justification='blocked premise'),
            CuratorDecision(action='create'),
            CuratorDecision(action='create'),
        ))

        refused = await _submit(interceptor, title='Refused')
        failed = await _submit(interceptor, title='Failed')
        await _submit(interceptor, title='Sentinel')
        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.created)

        assert (await _resolve(interceptor, refused))['status'] == 'refused'
        assert (await _resolve(interceptor, failed))['status'] == 'failed'
        assert outcomes == CreationOutcomes(
            created=frozenset({('project', '43')}), combined=frozenset(),
        )
        assert len(await _creation_rows(write_journal)) == 1

    @pytest.mark.asyncio
    async def test_cross_project_row_names_the_target_project(
        self, build_interceptor, write_journal,
    ):
        """The f9e49aef / task 105 shape: Stage 2 of one project files into another."""
        interceptor = build_interceptor(_curator_returning(CuratorDecision(action='create')))

        await _submit(interceptor, project_root='/home/leo/src/solar_challenge')
        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.created)

        assert outcomes.created == {('solar_challenge', '42')}

    @pytest.mark.asyncio
    async def test_add_task_never_sees_attribution_kwargs(
        self, build_interceptor, write_journal, taskmaster,
    ):
        interceptor = build_interceptor(_curator_returning(CuratorDecision(action='create')))

        await _resolve(interceptor, await _submit(interceptor))

        forwarded = taskmaster.add_task.call_args.kwargs
        assert 'agent_id' not in forwarded
        assert 'creation_attribution' not in forwarded

    @pytest.mark.asyncio
    async def test_submission_without_causation_still_writes_an_unattributed_row(
        self, build_interceptor, write_journal,
    ):
        async def _row():
            rows = await _creation_rows(write_journal)
            return rows or None

        interceptor = build_interceptor(_curator_returning(CuratorDecision(action='create')))

        await _submit(interceptor, metadata={'source': 'no-causation'})
        [row] = await poll_until(_row, message='no unattributed creation row')

        assert row['causation_id'] is None
        assert row['agent_id'] == STAGE2_AGENT
        assert await _read(write_journal) == CreationOutcomes(
            created=frozenset(), combined=frozenset(),
        )


class TestOrphanRaceJournal:
    """``cancel_ticket`` wins the ticket row while the worker is mid-decision.

    The worker's own terminal write then finds the row no longer pending, but
    the backend write still happened, so the outcome's creation row must land
    with the submitter's attribution regardless.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'decision, expected',
        [
            (
                CuratorDecision(action='create'),
                CreationOutcomes(created=frozenset({('project', '42')}), combined=frozenset()),
            ),
            (
                CuratorDecision(action='drop', target_id='5', justification='dup'),
                CreationOutcomes(created=frozenset(), combined=frozenset({('project', '5')})),
            ),
        ],
        ids=['created', 'combined'],
    )
    async def test_cancel_winning_the_ticket_row_still_journals_the_outcome(
        self, build_interceptor, write_journal, decision, expected,
    ):
        curating = asyncio.Event()
        release = asyncio.Event()

        async def gated_curate(*_args, **_kwargs):
            curating.set()
            await release.wait()
            return decision

        curator = _curator_returning()
        curator.curate = AsyncMock(side_effect=gated_curate)
        interceptor = build_interceptor(curator)

        ticket = await _submit(interceptor)
        await poll_until(curating.is_set, message='the curator never started curating')
        cancelled = await interceptor.cancel_ticket(ticket)
        release.set()
        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.created or o.combined)

        assert cancelled == {'status': 'cancelled', 'ticket_id': ticket}
        assert (await _resolve(interceptor, ticket))['status'] == 'cancelled'
        assert outcomes == expected
        [row] = await _creation_rows(write_journal)
        assert (row['agent_id'], row['causation_id']) == (STAGE2_AGENT, RUN)


class TestIdempotencyHitJournal:
    _EXISTING = {
        'id': '99',
        'title': 'Existing Escalation Task',
        'status': 'pending',
        'metadata': {'escalation_id': 'e1', 'suggestion_hash': 'h1'},
    }
    _METADATA = {**_RUN_METADATA, 'escalation_id': 'e1', 'suggestion_hash': 'h1'}

    @pytest.mark.asyncio
    async def test_idempotency_hit_writes_task_combined(
        self, build_interceptor, write_journal, taskmaster,
    ):
        taskmaster.get_tasks = AsyncMock(return_value={'tasks': [self._EXISTING]})
        interceptor = build_interceptor(_curator_returning())

        resolved = await _resolve(interceptor, await _submit(interceptor, metadata=self._METADATA))
        outcomes = await _poll_outcomes(write_journal, RUN, lambda o: o.combined)

        assert (resolved['status'], resolved['reason']) == ('combined', 'idempotency_hit')
        assert outcomes == CreationOutcomes(
            created=frozenset(), combined=frozenset({('project', '99')}),
        )
        taskmaster.add_task.assert_not_called()


class TestNoJournalWired:
    @pytest.mark.asyncio
    async def test_submit_and_resolve_behave_as_before(self, build_interceptor):
        interceptor = build_interceptor(
            _curator_returning(CuratorDecision(action='create')), journaled=False,
        )

        resolved = await _resolve(interceptor, await _submit(interceptor))

        assert (resolved['status'], resolved['task_id']) == ('created', '42')

    @pytest.mark.asyncio
    async def test_planning_mode_behaves_as_before(self, build_interceptor):
        result = await build_interceptor(journaled=False).submit_task(
            project_root=_PROJECT_ROOT, agent_id=STAGE2_AGENT, planning_mode=True,
            title='Planned', description='d', metadata=_RUN_METADATA,
        )

        assert result == {'task_id': '42', 'status': 'deferred', 'planning_mode': True}
