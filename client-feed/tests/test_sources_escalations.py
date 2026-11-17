"""Reading the open escalations from the escalation queue's root records."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from _stores import EscalationQueueDir
from shared.project_layout import escalation_queue_dir

from client_feed.contract import Source, TaskStatus, WaitingOn
from client_feed.sources.escalations import (
    EscalationSkipReason,
    OpenEscalation,
    OpenEscalations,
    SkippedRecord,
    read_open_escalations,
)
from client_feed.sources.store import StoreProblem, StoreReadError
from client_feed.sources.tasks import waiting_on


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / 'root'
    path.mkdir()
    return path


@pytest.fixture
def queue(root: Path) -> EscalationQueueDir:
    return EscalationQueueDir.create(root)


def blocked_waits_on(escalations: OpenEscalations, task_id: int) -> WaitingOn | None:
    return waiting_on(TaskStatus.BLOCKED, escalations.levels(task_id))


def test_a_pending_record_is_open_at_its_level(root: Path, queue: EscalationQueueDir) -> None:
    queue.write('esc-9-1', task_id='9', level=2)

    assert read_open_escalations(root).levels(9) == (2,)


def test_a_promoted_then_resolved_l2_waits_on_leo_then_on_the_factory(
    root: Path, queue: EscalationQueueDir
) -> None:
    queue.write('esc-9-1', task_id='9', level=1)
    queue.write('esc-9-2', task_id='9', level=2, members=['esc-9-1'])

    promoted = read_open_escalations(root)

    assert sorted(promoted.levels(9)) == [1, 2]
    assert blocked_waits_on(promoted, 9) is WaitingOn.LEO

    queue.archive('esc-9-2', status='dismissed')
    queue.archive('esc-9-1', status='dismissed')
    resolved = read_open_escalations(root)

    assert resolved.levels(9) == ()
    assert blocked_waits_on(resolved, 9) is WaitingOn.FACTORY


def test_a_parked_record_waits_on_leo(root: Path, queue: EscalationQueueDir) -> None:
    queue.write('esc-9-1', task_id='9', level=2, resolution='parked', resolution_action='park')

    assert blocked_waits_on(read_open_escalations(root), 9) is WaitingOn.LEO


def test_an_l2_is_open_for_each_pending_members_task(root: Path, queue: EscalationQueueDir) -> None:
    queue.write('esc-6434-4', task_id='6434', level=1)
    queue.write('esc-x-1', task_id='write-journal-size-ceiling', level=1)
    queue.write(
        'esc-6084-5', task_id='6084', level=2, members=['esc-6434-4', 'esc-7000-1', 'esc-x-1']
    )

    assert read_open_escalations(root).by_task == {
        6084: (OpenEscalation('esc-6084-5', 2),),
        6434: (OpenEscalation('esc-6084-5', 2), OpenEscalation('esc-6434-4', 1)),
    }


def test_an_l2_without_a_task_number_is_open_for_its_pending_members_task(
    root: Path, queue: EscalationQueueDir
) -> None:
    queue.write('esc-9-1', task_id='9', level=1)
    queue.write('esc-x-2', task_id='write-journal-size-ceiling', level=2, members=['esc-9-1'])

    assert read_open_escalations(root).by_task == {
        9: (OpenEscalation('esc-9-1', 1), OpenEscalation('esc-x-2', 2)),
    }


def test_an_l2_listing_a_member_of_its_own_task_appears_once(
    root: Path, queue: EscalationQueueDir
) -> None:
    queue.write('esc-9-1', task_id='9', level=1)
    queue.write('esc-9-2', task_id='9', level=2, members=['esc-9-1'])

    assert read_open_escalations(root).by_task == {
        9: (OpenEscalation('esc-9-1', 1), OpenEscalation('esc-9-2', 2)),
    }


def test_a_tasks_escalations_come_back_in_id_order(root: Path, queue: EscalationQueueDir) -> None:
    for number, level in ((3, 0), (1, 2), (2, 1)):
        queue.write(f'esc-9-{number}', task_id='9', level=level)

    assert read_open_escalations(root).by_task[9] == (
        OpenEscalation('esc-9-1', 2),
        OpenEscalation('esc-9-2', 1),
        OpenEscalation('esc-9-3', 0),
    )


@pytest.mark.parametrize('status', ['resolved', 'dismissed'])
def test_a_closed_record_left_at_the_root_is_not_open(
    root: Path, queue: EscalationQueueDir, status: str
) -> None:
    queue.write('esc-9-1', task_id='9', level=2, status=status)

    escalations = read_open_escalations(root)

    assert (escalations.levels(9), escalations.skipped) == ((), ())


@pytest.mark.parametrize('task_id', [None, 'write-journal-size-ceiling', '0', '-3'])
def test_a_record_without_a_task_number_opens_nothing(
    root: Path, queue: EscalationQueueDir, task_id: str | None
) -> None:
    queue.write('esc-x-1', task_id=task_id, level=2)

    escalations = read_open_escalations(root)

    assert (escalations.by_task, escalations.skipped) == ({}, ())


def test_a_legacy_record_reads_as_the_producer_reads_it(
    root: Path, queue: EscalationQueueDir
) -> None:
    queue.write_raw(
        'esc-9-1.json',
        json.dumps({'id': 'esc-9-1', 'task_id': '9', 'status': 'pending', 'summary': 'old'}),
    )

    assert read_open_escalations(root).by_task == {9: (OpenEscalation('esc-9-1', 0),)}


def test_only_the_roots_records_are_read(root: Path, queue: EscalationQueueDir) -> None:
    for name in ('esc-9-1.json.lock', 'esc-9-2.seq', 'esc-9-1abc.tmp', 'notes.md'):
        queue.write_raw(name, json.dumps({'id': 'esc-9-1', 'task_id': '9', 'level': 2}))
    for subdirectory in ('recovered', 'archive/2026-11-01'):
        nested = EscalationQueueDir(queue.path / subdirectory)
        nested.path.mkdir(parents=True)
        nested.write('esc-9-3', task_id='9', level=2)

    escalations = read_open_escalations(root)

    assert (escalations.by_task, escalations.skipped) == ({}, ())


def _without_id(queue: EscalationQueueDir) -> None:
    queue.write_raw('esc-9-2.json', json.dumps({'task_id': '9', 'status': 'pending', 'level': 2}))


@pytest.mark.parametrize(
    'write_bad',
    [
        lambda queue: queue.write_raw('esc-9-2.json', '{not json'),
        lambda queue: queue.write('esc-9-2', task_id='9', level='2'),
        lambda queue: queue.write('esc-9-2', task_id='9', level=True),
        lambda queue: queue.write('esc-9-2', task_id='9', level=1.0),
        lambda queue: queue.write('esc-9-2', task_id='9', level=-1),
        lambda queue: queue.write('esc-9-2', task_id='9', level=2, status='snoozed'),
        _without_id,
    ],
    ids=[
        'not-json', 'level-str', 'level-bool', 'level-float', 'level-negative', 'status-unknown',
        'no-id',
    ],
)
def test_a_malformed_record_is_skipped_and_the_rest_still_read(
    root: Path, queue: EscalationQueueDir, write_bad: Callable[[EscalationQueueDir], object]
) -> None:
    queue.write('esc-9-1', task_id='9', level=2)
    write_bad(queue)

    escalations = read_open_escalations(root)

    assert escalations.levels(9) == (2,)
    assert escalations.skipped == (
        SkippedRecord('esc-9-2.json', EscalationSkipReason.MALFORMED_RECORD),
    )


def test_an_unreadable_record_is_skipped_and_the_rest_still_read(
    root: Path, queue: EscalationQueueDir
) -> None:
    queue.write('esc-9-1', task_id='9', level=2)
    (queue.path / 'esc-9-2.json').mkdir()

    escalations = read_open_escalations(root)

    assert escalations.levels(9) == (2,)
    assert escalations.skipped == (
        SkippedRecord('esc-9-2.json', EscalationSkipReason.UNREADABLE_RECORD),
    )


def test_a_record_archived_while_listing_is_neither_open_nor_skipped(
    root: Path, queue: EscalationQueueDir
) -> None:
    (queue.path / 'esc-9-7.json').symlink_to(queue.path / 'archived-meanwhile.json')

    escalations = read_open_escalations(root)

    assert (escalations.by_task, escalations.skipped) == ({}, ())


def test_no_record_text_reaches_the_result(root: Path, queue: EscalationQueueDir) -> None:
    queue.write('esc-9-1', task_id='9', level=2)
    queue.write_raw('esc-9-2.json', '{"summary": "SECRET-SUMMARY /home/leo/x"')

    rendered = repr(read_open_escalations(root))

    assert 'SECRET' not in rendered
    assert '/home/' not in rendered


def test_a_missing_queue_is_refused(root: Path) -> None:
    with pytest.raises(StoreReadError) as caught:
        read_open_escalations(root)

    assert (caught.value.source, caught.value.problem) == (Source.EVENTS, StoreProblem.MISSING)
    assert 'data/escalations' in str(caught.value)


def test_a_queue_that_is_not_a_directory_is_refused(root: Path) -> None:
    path = escalation_queue_dir(root)
    path.parent.mkdir(parents=True)
    path.write_text('not a directory')

    with pytest.raises(StoreReadError) as caught:
        read_open_escalations(root)

    assert (caught.value.source, caught.value.problem) == (Source.EVENTS, StoreProblem.UNREADABLE)


def _snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob('*'))
    }


def test_a_read_changes_nothing_under_the_root(root: Path, queue: EscalationQueueDir) -> None:
    queue.write('esc-9-1', task_id='9', level=1)
    queue.write('esc-9-2', task_id='9', level=2, members=['esc-9-1'])
    before = _snapshot(root)

    read_open_escalations(root)

    assert _snapshot(root) == before
