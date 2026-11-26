"""Projecting the task store's allowlisted fields, and the rule for whom a task waits on."""

import dataclasses
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _defects import carrier_for, carrying, declared, gate_for
from _stores import EscalationQueueDir, RunsDb, TasksDb
from shared.project_layout import tasks_db

from client_feed.contract import Source, TaskStatus, WaitingOn
from client_feed.sources.escalations import OpenEscalations, read_open_escalations
from client_feed.sources.store import StoreProblem, StoreReadError
from client_feed.sources.tasks import (
    DefectCarrier,
    SkippedTask,
    TaskRow,
    TaskSkipReason,
    TasksRead,
    read_defect_carriers,
    read_tasks,
    waiting_on,
)

SHA = 'd' * 40
NO_ESCALATIONS = OpenEscalations(by_task={})
EARLY = '2026-09-30T08:00:00.000Z'
SINCE = '2026-10-01T00:00:00.000Z'
LATE = '2026-10-02T09:30:00.522233+00:00'
LATER = '2026-10-03T11:00:00.792Z'


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / 'root'
    path.mkdir()
    return path


@pytest.fixture
def tasks(root: Path) -> TasksDb:
    return TasksDb.create(root)


def read_all(root: Path, *, since: str | None = None) -> TasksRead:
    return read_tasks(root, since=since, escalations=NO_ESCALATIONS)


def only_row(read: TasksRead) -> TaskRow:
    (row,) = read.rows
    return row


def secret_task(tasks: TasksDb) -> None:
    tasks.add(
        7,
        title='Fix /home/leo/x crash',
        status='done',
        updated_at=LATE,
        description='SECRET-DESC',
        details='SECRET-DETAILS',
        test_strategy='SECRET-TS',
        metadata={
            'prd_path': 'plans/a-prd.md',
            'prd_task_label': 'alpha',
            'done_provenance': {'kind': 'merged', 'commit': SHA},
            'dry_run_proposals': 'SECRET-DRP',
            'external_deps': ['SECRET-EXT'],
            'files': ['x'],
        },
        depends_on=(3, 1),
    )


class TestAllowlist:
    def test_a_row_carries_exactly_the_allowlisted_fields(self) -> None:
        assert {field.name for field in dataclasses.fields(TaskRow)} == {
            'task_id', 'title', 'status', 'waiting_on', 'updated_at', 'depends_on',
            'prd_path', 'prd_task_label', 'landed_commit',
        }

    def test_a_task_projects_its_allowlisted_values_and_nothing_else(
        self, root: Path, tasks: TasksDb
    ) -> None:
        secret_task(tasks)

        read = read_all(root)

        assert read.rows == (
            TaskRow(
                task_id=7,
                title='Fix [path] crash',
                status=TaskStatus.DONE,
                waiting_on=None,
                updated_at=datetime(2026, 10, 2, 9, 30, 0, 522233, tzinfo=UTC),
                depends_on=(1, 3),
                prd_path='plans/a-prd.md',
                prd_task_label='alpha',
                landed_commit=SHA,
            ),
        )
        assert 'SECRET-' not in repr(read)

    def test_another_tag_is_never_read(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(7, title='master', updated_at=LATE, depends_on=(2,))
        tasks.add(7, title='feature', updated_at=LATER, tag='feature', depends_on=(5,))
        tasks.add(8, title='feature only', updated_at=LATER, tag='feature')

        read = read_all(root)

        assert [(row.task_id, row.title, row.depends_on) for row in read.rows] == [(7, 'master', (2,))]
        assert read.cursor == LATE


class TestCursor:
    def test_no_cursor_reads_every_row(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(1, updated_at=EARLY)
        tasks.add(2, updated_at=LATE)

        read = read_all(root)

        assert [row.task_id for row in read.rows] == [1, 2]
        assert read.cursor == LATE

    def test_a_cursor_reads_only_later_rows(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(1, updated_at=EARLY)
        tasks.add(2, updated_at=LATE)
        tasks.add(3, updated_at=LATER)

        read = read_all(root, since=SINCE)

        assert [row.task_id for row in read.rows] == [2, 3]
        assert read.cursor == LATER

    @pytest.mark.parametrize(
        ('status', 'updated_at'), [('archived', LATER), ('pending', 'yesterday')]
    )
    def test_a_skipped_row_never_moves_the_cursor(
        self, root: Path, tasks: TasksDb, status: str, updated_at: str
    ) -> None:
        tasks.add(1, updated_at=LATE)
        tasks.add(2, status=status, updated_at=updated_at)

        first = read_all(root)
        again = read_all(root, since=first.cursor)

        assert first.cursor == LATE
        assert (again.rows, again.cursor) == ((), LATE)
        assert [skipped.task_id for skipped in again.skipped] == [2]

    def test_a_later_projected_row_moves_the_cursor_past_a_skipped_one(
        self, root: Path, tasks: TasksDb
    ) -> None:
        tasks.add(1, status='archived', updated_at=LATE)
        tasks.add(2, updated_at=LATER)

        assert read_all(root).cursor == LATER

    def test_only_skipped_rows_leave_the_cursor_where_it_was(
        self, root: Path, tasks: TasksDb
    ) -> None:
        tasks.add(1, status='archived', updated_at=LATE)

        assert read_all(root).cursor is None
        assert read_all(root, since=SINCE).cursor == SINCE

    def test_nothing_new_leaves_the_cursor_where_it_was(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(1, updated_at=EARLY)

        assert read_all(root, since=SINCE) == TasksRead(rows=(), skipped=(), cursor=SINCE)

    def test_an_empty_store_has_no_cursor(self, root: Path, tasks: TasksDb) -> None:
        assert read_all(root) == TasksRead(rows=(), skipped=(), cursor=None)


class TestProjectedFields:
    def test_dependencies_are_sorted(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(7, updated_at=LATE, depends_on=(9, 2, 5))

        assert only_row(read_all(root)).depends_on == (2, 5, 9)

    def test_a_repo_relative_prd_path_is_kept(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(7, updated_at=LATE, metadata={'prd_path': 'plans/a-prd.md'})

        assert only_row(read_all(root)).prd_path == 'plans/a-prd.md'

    def test_an_absolute_prd_path_under_the_root_is_made_relative(
        self, root: Path, tasks: TasksDb
    ) -> None:
        tasks.add(7, updated_at=LATE, metadata={'prd_path': f'{root}/plans/b-prd.md'})

        assert only_row(read_all(root)).prd_path == 'plans/b-prd.md'

    @pytest.mark.parametrize(
        'prd_path', ['/home/leo/src/other/x.md', 'plans/../x.md', 42, {'path': 'plans/a.md'}]
    )
    def test_a_prd_path_that_is_not_in_the_repo_is_dropped(
        self, root: Path, tasks: TasksDb, prd_path: object
    ) -> None:
        tasks.add(7, updated_at=LATE, metadata={'prd_path': prd_path})

        assert only_row(read_all(root)).prd_path is None

    @pytest.mark.parametrize(
        'metadata',
        [
            {'done_provenance': {'kind': 'merged', 'commit': 'abc123'}},
            {'done_provenance': 'merged at ' + SHA},
            {'done_provenance': {'kind': 'reconciled'}},
        ],
    )
    def test_a_landed_commit_that_is_not_a_full_sha_is_dropped(
        self, root: Path, tasks: TasksDb, metadata: dict
    ) -> None:
        tasks.add(7, updated_at=LATE, metadata=metadata)

        assert only_row(read_all(root)).landed_commit is None

    @pytest.mark.parametrize('metadata', ['not json', None, ''])
    def test_unreadable_metadata_still_projects_the_row(
        self, root: Path, tasks: TasksDb, metadata: str | None
    ) -> None:
        tasks.add(7, title='kept', updated_at=LATE, metadata=metadata)

        row = only_row(read_all(root))

        assert (row.title, row.prd_path, row.prd_task_label, row.landed_commit) == (
            'kept', None, None, None
        )

    def test_a_prd_task_label_is_redacted(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(7, updated_at=LATE, metadata={'prd_task_label': 'β1 in ~/src/notes/x'})

        assert only_row(read_all(root)).prd_task_label == 'β1 in [path]'

    @pytest.mark.parametrize('label', [7, ['alpha'], {'label': 'alpha'}])
    def test_a_prd_task_label_that_is_not_a_string_is_dropped(
        self, root: Path, tasks: TasksDb, label: object
    ) -> None:
        tasks.add(7, updated_at=LATE, metadata={'prd_task_label': label})

        assert only_row(read_all(root)).prd_task_label is None


class TestSkippedTasks:
    @pytest.mark.parametrize('updated_at', ['', 'yesterday', '2026-10-02T09:30:00', '2026-10-02T11:30:00+02:00'])
    def test_an_updated_at_that_is_not_a_utc_instant_is_skipped(
        self, root: Path, tasks: TasksDb, updated_at: str
    ) -> None:
        tasks.add(105, updated_at=updated_at)

        read = read_all(root)

        assert read.rows == ()
        assert read.skipped == (SkippedTask(task_id=105, reason=TaskSkipReason.BAD_UPDATED_AT),)

    def test_an_unknown_status_is_skipped(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(7, status='archived', updated_at=LATE)
        tasks.add(8, updated_at=LATE)

        read = read_all(root)

        assert [row.task_id for row in read.rows] == [8]
        assert read.skipped == (SkippedTask(task_id=7, reason=TaskSkipReason.UNKNOWN_STATUS),)


@pytest.mark.parametrize(
    ('status', 'levels', 'expected'),
    [
        (TaskStatus.BLOCKED, (2,), WaitingOn.LEO),
        (TaskStatus.BLOCKED, (1,), WaitingOn.FACTORY),
        (TaskStatus.BLOCKED, (0,), WaitingOn.FACTORY),
        (TaskStatus.BLOCKED, (), WaitingOn.FACTORY),
        (TaskStatus.PENDING, (), None),
        (TaskStatus.IN_PROGRESS, (), None),
        (TaskStatus.IN_PROGRESS, (1,), WaitingOn.FACTORY),
        (TaskStatus.IN_PROGRESS, (1, 2), WaitingOn.LEO),
        (TaskStatus.DONE, (2,), None),
        (TaskStatus.CANCELLED, (1,), None),
    ],
)
def test_waiting_on(status: TaskStatus, levels: Sequence[int], expected: WaitingOn | None) -> None:
    assert waiting_on(status, levels) is expected


class TestWaitingOnFromTheStores:
    def waiting(self, root: Path, task_id: int) -> WaitingOn | None:
        read = read_tasks(root, since=None, escalations=read_open_escalations(root))
        return next(row.waiting_on for row in read.rows if row.task_id == task_id)

    def test_a_blocked_task_under_a_pending_level_two_waits_on_leo_until_it_is_archived(
        self, root: Path, tasks: TasksDb
    ) -> None:
        queue = EscalationQueueDir.create(root)
        tasks.add(9, status='blocked', updated_at=LATE)
        queue.write('esc-9-1', task_id='9', level=1)
        queue.write('esc-9-2', task_id='9', level=2, members=['esc-9-1'])

        assert self.waiting(root, 9) is WaitingOn.LEO

        queue.archive('esc-9-2', status='dismissed')
        queue.archive('esc-9-1', status='dismissed')

        assert self.waiting(root, 9) is WaitingOn.FACTORY

    def test_an_open_level_one_escalation_waits_on_the_factory(
        self, root: Path, tasks: TasksDb
    ) -> None:
        queue = EscalationQueueDir.create(root)
        tasks.add(10, status='in-progress', updated_at=LATE)
        queue.write('esc-10-1', task_id='10', level=1)

        assert self.waiting(root, 10) is WaitingOn.FACTORY

    def test_a_level_two_recorded_only_in_runs_db_no_longer_decides(
        self, root: Path, tasks: TasksDb
    ) -> None:
        EscalationQueueDir.create(root)
        tasks.add(9, status='blocked', updated_at=LATE)
        RunsDb.create(root).emit(
            'escalation_created', at=datetime(2026, 10, 2, tzinfo=UTC), task_id='9',
            data={'escalation_id': 'esc-9-1', 'level': 2},
        )

        assert self.waiting(root, 9) is WaitingOn.FACTORY


class TestStore:
    def test_a_missing_store_is_refused_and_never_created(self, root: Path) -> None:
        with pytest.raises(StoreReadError) as caught:
            read_all(root)

        assert (caught.value.source, caught.value.problem) == (Source.TASKS, StoreProblem.MISSING)
        assert not tasks_db(root).exists()

    def test_a_read_leaves_the_store_bytes_unchanged(self, root: Path, tasks: TasksDb) -> None:
        secret_task(tasks)
        before = tasks.path.read_bytes()

        read_all(root)

        assert tasks.path.read_bytes() == before


PROJECT = 'acme'
FIX = 'portal:acme:2:1'
FOREIGN_REFS = ('review:7:0123456789ab', 'followup:7:abc123', 'portal:other-client:2:1')


def carriers_of(root: Path) -> tuple[DefectCarrier, ...]:
    return read_defect_carriers(root, project_id=PROJECT)


def carrier(
    task_id: int, *, ref: str = FIX, gated: bool = False, waits_on_deferred_gate: bool = False
) -> DefectCarrier:
    return carrier_for(
        ref,
        project_id=PROJECT,
        task_id=task_id,
        gated=gated,
        waits_on_deferred_gate=waits_on_deferred_gate,
    )


class TestDefectCarriers:
    def test_a_carrier_carries_exactly_the_acknowledged_fields(self) -> None:
        assert {field.name for field in dataclasses.fields(DefectCarrier)} == {
            'ref', 'block', 'report', 'task_id', 'gated', 'waits_on_deferred_gate',
        }

    def test_a_pending_task_holding_a_portal_ref_reads_as_its_carrier(
        self, root: Path, tasks: TasksDb
    ) -> None:
        tasks.add(
            6,
            title='SECRET-TITLE',
            updated_at=LATE,
            description='SECRET-DESC',
            metadata={**carrying(FIX), 'files': ['SECRET-FILE']},
        )

        read = carriers_of(root)

        assert read == (
            DefectCarrier(
                ref=FIX, block=2, report=1, task_id=6, gated=False, waits_on_deferred_gate=False
            ),
        )
        assert 'SECRET-' not in repr(read)

    def test_carriers_come_sorted_by_ref(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(3, updated_at=LATE, metadata=carrying('portal:acme:3:1'))
        tasks.add(4, updated_at=LATE, metadata=carrying('portal:acme:2:7', 'portal:acme:10:2'))

        assert [c.ref for c in carriers_of(root)] == [
            'portal:acme:10:2', 'portal:acme:2:7', 'portal:acme:3:1',
        ]

    @pytest.mark.parametrize('status', [s for s in TaskStatus if s is not TaskStatus.DEFERRED])
    def test_every_status_but_deferred_makes_a_carrier(
        self, root: Path, tasks: TasksDb, status: TaskStatus
    ) -> None:
        tasks.add(6, status=status, updated_at=LATE, metadata=carrying(FIX))

        assert carriers_of(root) == (carrier(6),)

    @pytest.mark.parametrize('status', ['deferred', 'archived'])
    def test_a_deferred_or_unknown_status_is_no_carrier(
        self, root: Path, tasks: TasksDb, status: str
    ) -> None:
        tasks.add(6, status=status, updated_at=LATE, metadata=carrying(FIX))

        assert carriers_of(root) == ()

    @pytest.mark.parametrize('gate_status', ['pending', 'done', 'cancelled', 'deferred'])
    def test_a_carrier_depending_on_the_gate_for_its_ref_is_gated_whatever_the_gates_status(
        self, root: Path, tasks: TasksDb, gate_status: str
    ) -> None:
        tasks.add(5, status=gate_status, updated_at=LATE, metadata=gate_for(FIX))
        tasks.add(6, updated_at=LATE, metadata=carrying(FIX), depends_on=(5,))

        assert carriers_of(root) == (
            carrier(6, gated=True, waits_on_deferred_gate=(gate_status == 'deferred')),
        )

    @pytest.mark.parametrize('status', ['pending', 'in-progress', 'blocked', 'review'])
    def test_an_unfinished_carrier_behind_a_deferred_gate_for_its_ref_waits_on_it(
        self, root: Path, tasks: TasksDb, status: str
    ) -> None:
        tasks.add(5, status='deferred', updated_at=LATE, metadata=gate_for(FIX))
        tasks.add(6, status=status, updated_at=LATE, metadata=carrying(FIX), depends_on=(5,))

        assert carriers_of(root) == (carrier(6, gated=True, waits_on_deferred_gate=True),)

    @pytest.mark.parametrize('status', ['done', 'cancelled'])
    def test_a_finished_carrier_behind_a_deferred_gate_waits_on_nothing(
        self, root: Path, tasks: TasksDb, status: str
    ) -> None:
        tasks.add(5, status='deferred', updated_at=LATE, metadata=gate_for(FIX))
        tasks.add(6, status=status, updated_at=LATE, metadata=carrying(FIX), depends_on=(5,))

        assert carriers_of(root) == (carrier(6, gated=True),)

    def test_a_deferred_gate_for_another_ref_is_not_waited_on(
        self, root: Path, tasks: TasksDb
    ) -> None:
        tasks.add(5, status='deferred', updated_at=LATE, metadata=gate_for('portal:acme:2:9'))
        tasks.add(6, updated_at=LATE, metadata=carrying(FIX), depends_on=(5,))

        assert carriers_of(root) == (carrier(6),)

    def test_a_deferred_task_that_is_no_gate_is_not_waited_on(
        self, root: Path, tasks: TasksDb
    ) -> None:
        tasks.add(4, status='deferred', updated_at=LATE, metadata={'source': 'client-defect-filer'})
        tasks.add(6, updated_at=LATE, metadata=carrying(FIX), depends_on=(4,))

        assert carriers_of(root) == (carrier(6),)

    def test_a_gate_for_another_ref_does_not_gate(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(5, updated_at=LATE, metadata=gate_for('portal:acme:2:9'))
        tasks.add(6, updated_at=LATE, metadata=carrying(FIX), depends_on=(5,))

        assert carriers_of(root) == (carrier(6),)

    def test_a_gate_without_an_edge_does_not_gate(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(5, updated_at=LATE, metadata=gate_for(FIX))
        tasks.add(6, updated_at=LATE, metadata=carrying(FIX))

        assert carriers_of(root) == (carrier(6),)

    def test_an_edge_to_a_task_that_is_no_gate_does_not_gate(
        self, root: Path, tasks: TasksDb
    ) -> None:
        tasks.add(4, updated_at=LATE, metadata={'source': 'client-defect-filer'})
        tasks.add(6, updated_at=LATE, metadata=carrying(FIX), depends_on=(4,))

        assert carriers_of(root) == (carrier(6),)

    def test_of_several_holders_the_lowest_task_id_carries(
        self, root: Path, tasks: TasksDb
    ) -> None:
        tasks.add(9, updated_at=LATE, metadata=carrying(FIX))
        tasks.add(6, status='done', updated_at=LATE, metadata=carrying(FIX))

        assert carriers_of(root) == (carrier(6),)

    def test_a_deferred_lower_id_yields_to_a_pending_higher_one(
        self, root: Path, tasks: TasksDb
    ) -> None:
        tasks.add(6, status='deferred', updated_at=LATE, metadata=carrying(FIX))
        tasks.add(9, updated_at=LATE, metadata=carrying(FIX))

        assert carriers_of(root) == (carrier(9),)

    def test_no_carrier_comes_from_a_ref_this_project_would_not_mint(
        self, root: Path, tasks: TasksDb
    ) -> None:
        review, followup, other_project = FOREIGN_REFS
        tasks.add(1, updated_at=LATE, metadata={'defects': [declared(review, origin='review')]})
        followup_entry = declared(followup, origin='agent_followup')
        tasks.add(2, updated_at=LATE, metadata={'defects': [followup_entry]})
        tasks.add(3, updated_at=LATE, metadata=carrying(other_project))
        tasks.add(4, updated_at=LATE, metadata=carrying('portal:acme:check:1'))
        tasks.add(5, updated_at=LATE, metadata=gate_for(other_project))

        assert carriers_of(root) == ()

    def test_a_task_holding_a_foreign_ref_and_this_projects_yields_only_this_projects(
        self, root: Path, tasks: TasksDb
    ) -> None:
        tasks.add(5, updated_at=LATE, metadata=gate_for(FOREIGN_REFS[2]))
        tasks.add(
            6,
            updated_at=LATE,
            metadata={'defects': [declared(ref) for ref in (*FOREIGN_REFS, FIX)]},
            depends_on=(5,),
        )

        read = carriers_of(root)

        assert read == (carrier(6),)
        assert not any(ref in repr(read) for ref in FOREIGN_REFS)
        assert 'other-client' not in repr(read)

    @pytest.mark.parametrize(
        'metadata',
        [
            '{not json',
            {'defects': FIX},
            {'defects': {'ref': FIX}},
            {'defects': [FIX]},
            {'defects': [declared(21)]},
            {'defects': [declared(None)]},
            {'defects': [{'origin': 'client'}]},
        ],
        ids=[
            'not-json',
            'defects-a-string',
            'defects-an-object',
            'entry-a-string',
            'ref-a-number',
            'ref-null',
            'no-ref',
        ],
    )
    def test_metadata_off_the_shape_yields_no_carrier_and_raises_nothing(
        self, root: Path, tasks: TasksDb, metadata: dict[str, object] | str
    ) -> None:
        tasks.add(6, updated_at=LATE, metadata=metadata)

        assert carriers_of(root) == ()

    def test_a_gate_whose_metadata_is_not_json_does_not_gate(
        self, root: Path, tasks: TasksDb
    ) -> None:
        tasks.add(5, updated_at=LATE, metadata='{not json')
        tasks.add(6, updated_at=LATE, metadata=carrying(FIX), depends_on=(5,))

        assert carriers_of(root) == (carrier(6),)

    def test_another_tags_rows_are_never_read(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(5, updated_at=LATE)
        tasks.add(6, updated_at=LATE, metadata=carrying(FIX), depends_on=(5,))
        tasks.add(5, updated_at=LATE, tag='feature', metadata=gate_for(FIX))
        tasks.add(6, updated_at=LATE, tag='feature', metadata=carrying(FIX), depends_on=(5,))
        tasks.add(7, updated_at=LATE, tag='feature', metadata=carrying('portal:acme:3:1'))

        assert carriers_of(root) == (carrier(6),)

    def test_a_missing_store_is_refused_and_never_created(self, root: Path) -> None:
        with pytest.raises(StoreReadError) as caught:
            carriers_of(root)

        assert (caught.value.source, caught.value.problem) == (Source.TASKS, StoreProblem.MISSING)
        assert not tasks_db(root).exists()

    def test_a_read_leaves_the_store_bytes_unchanged(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(5, updated_at=LATE, metadata=gate_for(FIX))
        tasks.add(6, updated_at=LATE, metadata=carrying(FIX), depends_on=(5,))
        before = tasks.path.read_bytes()

        carriers_of(root)

        assert tasks.path.read_bytes() == before
