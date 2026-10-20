"""Projecting the task store's allowlisted fields, and the rule for whom a task waits on."""

import dataclasses
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _stores import RunsDb, TasksDb
from client_feed.sources.tasks import (
    SkippedTask,
    TaskRow,
    TaskSkipReason,
    TasksRead,
    read_tasks,
    waiting_on,
)
from shared.project_layout import tasks_db

from client_feed.contract import Source, TaskStatus, WaitingOn
from client_feed.sources.events import OpenEscalations, read_open_escalations
from client_feed.sources.store import StoreProblem, StoreReadError

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

    def test_the_cursor_passes_an_unprojectable_row(self, root: Path, tasks: TasksDb) -> None:
        tasks.add(1, updated_at=LATE)
        tasks.add(2, status='archived', updated_at=LATER)

        assert read_all(root).cursor == LATER

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
        (TaskStatus.BLOCKED, (None,), WaitingOn.FACTORY),
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
def test_waiting_on(status: TaskStatus, levels: Sequence[int | None], expected: WaitingOn | None) -> None:
    assert waiting_on(status, levels) is expected


class TestWaitingOnFromTheStores:
    def waiting(self, root: Path, task_id: int) -> WaitingOn | None:
        read = read_tasks(root, since=None, escalations=read_open_escalations(root))
        return next(row.waiting_on for row in read.rows if row.task_id == task_id)

    def test_a_blocked_task_with_an_open_level_two_waits_on_leo(
        self, root: Path, tasks: TasksDb
    ) -> None:
        runs = RunsDb.create(root)
        tasks.add(9, status='blocked', updated_at=LATE)
        runs.emit(
            'escalation_created', at=datetime(2026, 10, 2, tzinfo=UTC), task_id='9',
            data={'escalation_id': 'esc-9-1', 'level': 2, 'summary': 'SECRET-SUMMARY'},
        )

        assert self.waiting(root, 9) is WaitingOn.LEO

        runs.emit(
            'escalation_resolved', at=datetime(2026, 10, 3, tzinfo=UTC), task_id='9',
            data={'escalation_id': 'esc-9-1'},
        )

        assert self.waiting(root, 9) is WaitingOn.FACTORY

    def test_an_open_level_one_escalation_waits_on_the_factory(
        self, root: Path, tasks: TasksDb
    ) -> None:
        runs = RunsDb.create(root)
        tasks.add(10, status='in-progress', updated_at=LATE)
        runs.emit(
            'escalation_created', at=datetime(2026, 10, 2, tzinfo=UTC), task_id='10',
            data={'escalation_id': 'esc-10-1', 'level': 1},
        )

        assert self.waiting(root, 10) is WaitingOn.FACTORY


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
