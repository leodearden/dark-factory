"""Current values that differ from the folded state become upsert and snapshot events, and nothing else does."""

from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import pytest

from client_feed.contract import (
    FEED_VERSION,
    FeedEvent,
    FeedState,
    PrdUpserted,
    TaskStatus,
    TaskUpserted,
    WaitingOn,
)
from client_feed.feed_io import Stamp
from client_feed.fold import fold
from client_feed.sources.events import OpenEscalation, OpenEscalations
from client_feed.sources.prds import PrdText
from client_feed.sources.tasks import TaskRow
from client_feed.value_events import prd_events, task_events

PROJECT = 'acme'
EPOCH = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
NOW = EPOCH + timedelta(hours=1)
X_PRD = 'plans/x-prd.md'
Y_PRD = 'plans/y-prd.md'
NO_ESCALATIONS = OpenEscalations(by_task={})


def at(minutes: float) -> datetime:
    return EPOCH + timedelta(minutes=minutes)


def stamp(first_seq: int = 100) -> Stamp:
    return Stamp(project_id=PROJECT, first_seq=first_seq)


def row(task_id: int = 7, **changes: object) -> TaskRow:
    base = TaskRow(
        task_id=task_id,
        title=f'Task {task_id}',
        status=TaskStatus.PENDING,
        waiting_on=None,
        updated_at=at(task_id),
        depends_on=(),
        prd_path=X_PRD,
        prd_task_label=None,
        landed_commit=None,
    )
    return replace(base, **changes)


def upserted(task: TaskRow, seq: int, *, minutes: float = 0) -> TaskUpserted:
    return TaskUpserted(
        seq=seq, at=at(minutes), project_id=PROJECT, feed_version=FEED_VERSION, **asdict(task)
    )


def state_of(*events: FeedEvent) -> FeedState:
    return fold(events, project_id=PROJECT)


def test_a_task_unknown_to_the_state_is_upserted_with_every_allowlisted_field() -> None:
    task = row(7, depends_on=(3, 4), prd_task_label='beta', landed_commit='a' * 40)

    events = task_events([task], state=state_of(), escalations=NO_ESCALATIONS, stamp=stamp(), now=NOW)

    assert events == (
        TaskUpserted(
            seq=100, at=task.updated_at, project_id=PROJECT, feed_version=FEED_VERSION, **asdict(task)
        ),
    )


def test_a_task_the_state_already_says_emits_nothing_whatever_its_envelope() -> None:
    task = row(7)
    state = state_of(upserted(task, 5, minutes=-30))

    assert task_events([task], state=state, escalations=NO_ESCALATIONS, stamp=stamp(), now=NOW) == ()


@pytest.mark.parametrize(
    'changes',
    [{'status': TaskStatus.BLOCKED}, {'title': 'Renamed'}, {'depends_on': (3,)}],
    ids=['status', 'title', 'depends_on'],
)
def test_a_task_differing_from_the_state_is_upserted(changes: dict[str, object]) -> None:
    state = state_of(upserted(row(7), 5))
    changed = row(7, **changes)

    events = task_events([changed], state=state, escalations=NO_ESCALATIONS, stamp=stamp(), now=NOW)

    assert [asdict(changed)] == [
        {name: getattr(event, name) for name in asdict(changed)} for event in events
    ]


def _waiting_change(
    before: WaitingOn | None, escalations: OpenEscalations | None
) -> tuple[TaskUpserted, ...]:
    blocked = row(9, status=TaskStatus.BLOCKED, waiting_on=before)
    state = state_of(upserted(row(7), 4), upserted(blocked, 5))
    return task_events([row(7)], state=state, escalations=escalations, stamp=stamp(), now=NOW)


def test_an_escalation_opening_on_an_unread_task_upserts_its_new_waiting_on() -> None:
    leo = OpenEscalations(by_task={9: (OpenEscalation(escalation_id='esc-9-1', level=2),)})

    events = _waiting_change(WaitingOn.FACTORY, leo)

    assert len(events) == 1
    assert events[0].task_id == 9
    assert events[0].waiting_on is WaitingOn.LEO
    assert events[0].at == NOW
    assert events[0].status is TaskStatus.BLOCKED
    assert events[0].title == 'Task 9'
    assert events[0].updated_at == at(9)


def test_an_escalation_resolving_on_an_unread_task_returns_it_to_the_factory() -> None:
    events = _waiting_change(WaitingOn.LEO, NO_ESCALATIONS)

    assert [(event.task_id, event.waiting_on) for event in events] == [(9, WaitingOn.FACTORY)]


def test_an_unchanged_waiting_on_emits_nothing() -> None:
    assert _waiting_change(WaitingOn.FACTORY, NO_ESCALATIONS) == ()


def test_without_escalations_no_waiting_on_is_recomputed() -> None:
    assert _waiting_change(WaitingOn.LEO, None) == ()


def tasks_view(*tasks: TaskRow) -> tuple[TaskUpserted, ...]:
    return tuple(upserted(task, seq) for seq, task in enumerate(tasks, 1))


def prd(path: str, title: str, excerpt: str, counts: dict[TaskStatus, int], seq: int = 50) -> PrdUpserted:
    return PrdUpserted(
        seq=seq,
        at=at(0),
        project_id=PROJECT,
        feed_version=FEED_VERSION,
        prd_path=path,
        title=title,
        excerpt=excerpt,
        task_counts=counts,
    )


X_TASKS = tasks_view(
    row(7, status=TaskStatus.DONE),
    row(8, status=TaskStatus.BLOCKED),
    row(9, status=TaskStatus.DONE),
    row(10, prd_path=None),
)
X_COUNTS = {TaskStatus.BLOCKED: 1, TaskStatus.DONE: 2}
X_TEXT = {X_PRD: PrdText(prd_path=X_PRD, title='X', excerpt='Why X.')}


def test_a_prd_first_seen_is_upserted_with_its_summary_and_task_counts() -> None:
    events = prd_events(
        tasks=X_TASKS, state=state_of(), summaries=X_TEXT, stamp=stamp(), now=NOW
    )

    assert events == (
        PrdUpserted(
            seq=100,
            at=NOW,
            project_id=PROJECT,
            feed_version=FEED_VERSION,
            prd_path=X_PRD,
            title='X',
            excerpt='Why X.',
            task_counts=X_COUNTS,
        ),
    )


def test_a_requested_prd_without_a_summary_reads_as_its_path() -> None:
    tasks = tasks_view(row(7, prd_path=Y_PRD))

    events = prd_events(tasks=tasks, state=state_of(), summaries={}, stamp=stamp(), now=NOW)

    assert [(event.prd_path, event.title, event.excerpt) for event in events] == [(Y_PRD, Y_PRD, '')]
    assert events[0].task_counts == {TaskStatus.PENDING: 1}


@pytest.mark.parametrize('summaries', [None, {}, X_TEXT], ids=['unread', 'not-requested', 'same'])
def test_an_unchanged_prd_emits_nothing(summaries: dict[str, PrdText] | None) -> None:
    state = state_of(prd(X_PRD, 'X', 'Why X.', X_COUNTS))

    assert prd_events(tasks=X_TASKS, state=state, summaries=summaries, stamp=stamp(), now=NOW) == ()


@pytest.mark.parametrize('summaries', [None, {}], ids=['unread', 'not-requested'])
def test_a_changed_count_keeps_the_previous_title_and_excerpt(
    summaries: dict[str, PrdText] | None,
) -> None:
    state = state_of(prd(X_PRD, 'Old X', 'Old why.', {TaskStatus.DONE: 3}))

    events = prd_events(tasks=X_TASKS, state=state, summaries=summaries, stamp=stamp(), now=NOW)

    assert [(event.title, event.excerpt, event.task_counts) for event in events] == [
        ('Old X', 'Old why.', X_COUNTS)
    ]


def test_a_changed_summary_is_upserted() -> None:
    state = state_of(prd(X_PRD, 'Old X', 'Old why.', X_COUNTS))

    events = prd_events(tasks=X_TASKS, state=state, summaries=X_TEXT, stamp=stamp(), now=NOW)

    assert [(event.title, event.excerpt) for event in events] == [('X', 'Why X.')]


def test_an_unread_prd_unknown_to_the_state_waits_for_its_summary() -> None:
    assert prd_events(tasks=X_TASKS, state=state_of(), summaries=None, stamp=stamp(), now=NOW) == ()
