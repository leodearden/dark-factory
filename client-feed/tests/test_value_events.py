"""Current values that differ from the folded state become upsert and snapshot events, and nothing else does."""

from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

import pytest

from client_feed.contract import (
    FEED_VERSION,
    Branch,
    BranchSnapshot,
    FactoryState,
    FeedEvent,
    FeedState,
    FileKind,
    PrdUpserted,
    RevealBudget,
    SchedulerState,
    TaskStatus,
    TaskUpserted,
    TreeFile,
    TreeSnapshot,
    TreeTotals,
    WaitingOn,
)
from client_feed.feed_io import Stamp
from client_feed.fold import fold
from client_feed.policy import Policy
from client_feed.sources.events import FactoryStatus, OpenEscalation, OpenEscalations
from client_feed.sources.prds import PrdText
from client_feed.sources.tasks import TaskRow
from client_feed.sources.tree import LineCounts, VisibleTree
from client_feed.value_events import (
    branch_event,
    factory_event,
    prd_events,
    task_events,
    tree_event,
)

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


def factory_status(
    *,
    scheduler: SchedulerState = SchedulerState.RUNNING,
    since: datetime | None = None,
    last_event_at: datetime | None = EPOCH,
    idle_seconds: int | None = 0,
) -> FactoryStatus:
    return FactoryStatus(
        scheduler=scheduler, since=since, last_event_at=last_event_at, idle_seconds=idle_seconds
    )


def emitted_factory(status: FactoryStatus, *, seq: int = 50) -> FactoryState:
    event = factory_event(status, state=state_of(), stamp=stamp(seq), now=NOW)
    assert event is not None
    return event


def test_the_first_factory_status_is_emitted_as_read() -> None:
    status = factory_status(scheduler=SchedulerState.PAUSED, since=at(-5), idle_seconds=300)

    event = factory_event(status, state=state_of(), stamp=stamp(), now=NOW)

    assert event == FactoryState(
        seq=100,
        at=NOW,
        project_id=PROJECT,
        feed_version=FEED_VERSION,
        scheduler=SchedulerState.PAUSED,
        since=at(-5),
        last_event_at=EPOCH,
        idle_seconds=300,
    )


def _next_factory(before: FactoryStatus, after: FactoryStatus) -> FactoryState | None:
    state = state_of(emitted_factory(before))
    return factory_event(after, state=state, stamp=stamp(), now=NOW)


def test_an_identical_factory_status_emits_nothing() -> None:
    assert _next_factory(factory_status(), factory_status()) is None


@pytest.mark.parametrize(
    'after',
    [
        factory_status(scheduler=SchedulerState.PAUSED, since=at(1)),
        factory_status(since=at(1)),
    ],
    ids=['scheduler', 'since'],
)
def test_a_scheduler_change_is_emitted_at_once(after: FactoryStatus) -> None:
    assert _next_factory(factory_status(), after) is not None


@pytest.mark.parametrize(
    ('before', 'after', 'emitted'),
    [
        (factory_status(idle_seconds=125), factory_status(idle_seconds=170), False),
        (factory_status(idle_seconds=170), factory_status(idle_seconds=185), True),
        (
            factory_status(last_event_at=EPOCH + timedelta(seconds=10)),
            factory_status(last_event_at=EPOCH + timedelta(seconds=50)),
            False,
        ),
        (
            factory_status(last_event_at=EPOCH + timedelta(seconds=50)),
            factory_status(last_event_at=EPOCH + timedelta(seconds=65)),
            True,
        ),
    ],
    ids=['idle-same-minute', 'idle-next-minute', 'last-event-same-minute', 'last-event-next-minute'],
)
def test_quiet_time_is_compared_by_the_minute(
    before: FactoryStatus, after: FactoryStatus, emitted: bool
) -> None:
    assert (_next_factory(before, after) is not None) is emitted


def test_b9_a_paused_quiet_factory_reports_three_hours_idle() -> None:
    quiet_since = EPOCH
    state, emitted = state_of(), 0
    for seconds in range(0, 3 * 3600 + 1, 5):
        status = factory_status(
            scheduler=SchedulerState.PAUSED,
            since=quiet_since,
            last_event_at=quiet_since,
            idle_seconds=seconds,
        )
        now = quiet_since + timedelta(seconds=seconds)
        event = factory_event(status, state=state, stamp=stamp(state.watermark_seq + 1), now=now)
        if event is not None:
            state, emitted = fold([event], project_id=PROJECT, start=state), emitted + 1

    assert state.factory is not None
    assert state.factory.scheduler is SchedulerState.PAUSED
    assert state.factory.idle_seconds is not None and state.factory.idle_seconds >= 10800
    assert emitted == 3 * 60 + 1


def branch(task_id: int, *, ahead: int = 1, minutes: float = 0) -> Branch:
    return Branch(name=f'task/{task_id}', task_id=task_id, ahead=ahead, last_commit_at=at(minutes))


BRANCHES = (branch(7), branch(9, ahead=3))


def test_the_first_branches_are_emitted() -> None:
    event = branch_event(BRANCHES, state=state_of(), stamp=stamp(), now=NOW)

    assert event == BranchSnapshot(
        seq=100, at=NOW, project_id=PROJECT, feed_version=FEED_VERSION, branches=BRANCHES
    )


@pytest.mark.parametrize(
    ('after', 'emitted'),
    [
        (BRANCHES, False),
        ((*BRANCHES, branch(11)), True),
        (BRANCHES[:1], True),
        ((branch(7), branch(9, ahead=4)), True),
        ((branch(7, minutes=5), branch(9, ahead=3)), True),
    ],
    ids=['equal', 'added', 'removed', 'ahead', 'last-commit-at'],
)
def test_branches_are_emitted_when_they_change(after: tuple[Branch, ...], emitted: bool) -> None:
    state = state_of(BranchSnapshot(**stamp(5)(at(0)), branches=BRANCHES))

    event = branch_event(after, state=state, stamp=stamp(), now=NOW)

    assert (event is not None) is emitted


def tree_file(path: str, kind: FileKind, lines: int) -> TreeFile:
    return TreeFile(path=path, kind=kind, lines=lines, bytes=lines * 10, last_changed_at=at(0))


def test_a_tree_snapshot_counts_only_production_lines_outside_free_and_never_reveal_globs() -> None:
    policy = Policy.model_validate(
        {
            'project_id': PROJECT,
            'project_root': '/srv/acme',
            'out_dir': '/srv/feed/out',
            'in_dir': '/srv/feed/in',
            'visible_globs': ['**'],
            'reveal_proportion': 0.3,
            'reveal_floor_lines': 40,
            'free_globs': ['core/src/free/**'],
            'never_reveal_globs': ['core/src/secret/**'],
        }
    )
    files = (
        tree_file('core/src/a.py', FileKind.PRODUCTION, 100),
        tree_file('core/src/b.py', FileKind.PRODUCTION, 33),
        tree_file('core/src/free/c.py', FileKind.PRODUCTION, 1000),
        tree_file('core/src/secret/d.py', FileKind.PRODUCTION, 2000),
        tree_file('core/tests/test_a.py', FileKind.TEST, 400),
        tree_file('docs/a.md', FileKind.DOCS, 500),
        tree_file('core/src/gen.py', FileKind.GENERATED, 600),
    )
    totals = TreeTotals(files=7, production_lines=3133, test_lines=400, docs_lines=500)
    line_counts = LineCounts(max_line_bytes=400, by_blob={})
    tree = VisibleTree(commit='c' * 40, files=files, totals=totals, line_counts=line_counts)

    event = tree_event(tree, policy=policy, stamp=stamp(), now=NOW)

    assert event == TreeSnapshot(
        seq=100,
        at=NOW,
        project_id=PROJECT,
        feed_version=FEED_VERSION,
        commit='c' * 40,
        totals=totals,
        files=files,
        budget=RevealBudget(
            proportion=0.3, counted_lines=133, budget_lines=39, spent_lines=0, floor_lines=40
        ),
    )
