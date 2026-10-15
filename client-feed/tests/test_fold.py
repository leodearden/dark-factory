"""fold: the one reducer from feed lines to FeedState, including resume from a watermark (PRD §4.2)."""

import copy
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from client_feed.contract import (
    EVENT_MODELS,
    FEED_VERSION,
    EventHeader,
    FeedError,
    FeedLine,
    FeedState,
    InFlightTask,
    RevealedFile,
    parse_feed_line,
)
from client_feed.fold import RECENT_LIMIT, CorruptFeed, ProjectMismatch, fold

SHA = 'c' * 40
OTHER_SHA = 'd' * 40
EPOCH = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def at(seq: int) -> datetime:
    return EPOCH + timedelta(minutes=seq)


def line(
    wire_type: str, seq: int, *, project_id: str = 'p', feed_version: str = '1.0', **payload: Any
) -> FeedLine:
    raw = {
        'seq': seq,
        'at': at(seq).isoformat(),
        'project_id': project_id,
        'feed_version': feed_version,
        'type': wire_type,
        **payload,
    }
    return parse_feed_line(json.dumps(raw), line_no=seq)


def heartbeat(
    seq: int,
    *,
    head: str | None = SHA,
    engagement_version: int | None = 1,
    degraded: tuple[dict[str, str], ...] = (),
    project_id: str = 'p',
) -> FeedLine:
    return line(
        'heartbeat',
        seq,
        project_id=project_id,
        head=head,
        as_of=at(seq).isoformat(),
        engagement_version=engagement_version,
        watermark_seq=max(seq - 1, 0),
        degraded=list(degraded),
    )


def factory_state(seq: int, scheduler: str = 'running') -> FeedLine:
    return line(
        'factory_state',
        seq,
        scheduler=scheduler,
        since=at(0).isoformat(),
        last_event_at=at(seq).isoformat(),
        idle_seconds=seq,
    )


def prd(seq: int, path: str = 'plans/a-prd.md', title: str = 'A') -> FeedLine:
    return line(
        'prd_upserted', seq, prd_path=path, title=title, excerpt='Why.', task_counts={'done': 1}
    )


def task(seq: int, task_id: int, status: str = 'pending', *, project_id: str = 'p') -> FeedLine:
    return line(
        'task_upserted',
        seq,
        project_id=project_id,
        task_id=task_id,
        title=f'Task {task_id}',
        status=status,
        waiting_on=None,
        updated_at=at(seq).isoformat(),
        depends_on=[],
        prd_path='plans/a-prd.md',
        prd_task_label=None,
        landed_commit=SHA if status == 'done' else None,
    )


def phase(seq: int, task_id: int, name: str, *, entered: bool = True) -> FeedLine:
    return line('task_phase', seq, task_id=task_id, phase=name, entered=entered)


def verify_run(seq: int, run_key: str = 'run-1', *, passed: bool = True) -> FeedLine:
    return line(
        'verify_run',
        seq,
        run_key=run_key,
        task_id=7,
        kind='task',
        passed=passed,
        duration_ms=10,
        commit=SHA,
        legs='not_recorded',
    )


def case_results(seq: int, run_key: str = 'run-1') -> FeedLine:
    return line(
        'test_results',
        seq,
        run_key=run_key,
        counts={'passed': 1, 'failed': 0, 'skipped': 0, 'errored': 0, 'truncated': 0},
        duration_ms=5,
        cases=[
            {
                'id': 't::ok',
                'outcome': 'passed',
                'duration_ms': 5,
                'criterion': None,
                'exception_type': None,
            }
        ],
    )


def merge(seq: int, commit: str = SHA) -> FeedLine:
    return line(
        'merge_landed',
        seq,
        commit=commit,
        task_ids=[7],
        titles=['Task 7'],
        files_changed=1,
        insertions=2,
        deletions=0,
        review='not_recorded',
    )


def branches(seq: int, ahead: int = 1) -> FeedLine:
    return line(
        'branch_snapshot',
        seq,
        branches=[
            {'name': 'task/7', 'task_id': 7, 'ahead': ahead, 'last_commit_at': at(seq).isoformat()}
        ],
    )


def tree(seq: int, commit: str = SHA) -> FeedLine:
    return line(
        'tree_snapshot',
        seq,
        commit=commit,
        totals={'files': 1, 'production_lines': 3, 'test_lines': 0, 'docs_lines': 0},
        files=[
            {
                'path': 'src/a.py',
                'kind': 'production',
                'lines': 3,
                'bytes': 30,
                'last_changed_at': None,
            }
        ],
        budget={
            'proportion': 0.25,
            'counted_lines': 3,
            'budget_lines': 0,
            'spent_lines': 0,
            'floor_lines': 50,
        },
    )


def revealed(
    seq: int,
    path: str,
    *,
    commit: str = SHA,
    lines: int = 3,
    deleted: bool = False,
    renamed_from: str | None = None,
) -> FeedLine:
    return line(
        'file_revealed',
        seq,
        path=path,
        commit=commit,
        lines=lines,
        content=None if deleted else 'x = 1\n',
        deleted=deleted,
        renamed_from=renamed_from,
    )


def refused(seq: int, path: str = '../x') -> FeedLine:
    return line('reveal_refused', seq, path=path, reason='not_in_tree')


def criteria(seq: int, run_kind: str = 'verify') -> FeedLine:
    return line(
        'criteria_results',
        seq,
        run_kind=run_kind,
        commit=SHA,
        run_at=at(seq).isoformat(),
        results=[{'id': 'C-1', 'outcome': 'passed', 'tests': ['t::ok']}],
    )


def unknown(seq: int) -> FeedLine:
    return line('future_minor_event', seq, feed_version='1.1', payload={'n': seq})


def mixed_feed() -> list[FeedLine]:
    """About thirty lines touching every event type, every applier rule and an unknown type."""
    return [
        heartbeat(1),
        factory_state(2),
        prd(3),
        prd(4, 'plans/b-prd.md', 'B'),
        task(5, 1),
        task(6, 2),
        phase(7, 1, 'plan'),
        phase(8, 2, 'plan'),
        phase(9, 1, 'plan', entered=False),
        phase(10, 1, 'implement'),
        verify_run(11),
        case_results(12),
        merge(13),
        task(14, 1, 'done'),
        branches(15),
        tree(16),
        revealed(17, 'src/a.py'),
        revealed(18, 'src/b.py'),
        refused(19),
        criteria(20),
        unknown(21),
        heartbeat(22, degraded=({'source': 'archive', 'reason': 'unreadable'},)),
        revealed(23, 'src/c.py', renamed_from='src/a.py'),
        revealed(24, 'src/b.py', deleted=True),
        criteria(25, 'staging'),
        prd(26, title='A again'),
        factory_state(27, 'paused'),
        phase(28, 2, 'plan', entered=False),
        verify_run(29, passed=False),
        merge(30),
        branches(31, ahead=2),
        tree(32, OTHER_SHA),
    ]


def empty_state(project_id: str = 'p') -> FeedState:
    return FeedState(
        project_id=project_id,
        feed_version=FEED_VERSION,
        watermark_seq=0,
        as_of=None,
        head=None,
        engagement_version=None,
        degraded=(),
        factory=None,
        prds={},
        tasks={},
        in_flight=(),
        branches=None,
        tree=None,
        revealed=(),
        criteria={},
        recent=(),
    )


def test_an_empty_feed_folds_to_an_empty_state() -> None:
    assert fold([], project_id='p') == empty_state()
    assert RECENT_LIMIT == 200


def test_a_heartbeat_sets_the_publisher_health_and_a_later_one_replaces_it() -> None:
    first = heartbeat(1, degraded=({'source': 'archive', 'reason': 'gone'},))
    state = fold([first], project_id='p')
    assert (state.as_of, state.head, state.engagement_version) == (at(1), SHA, 1)
    assert [d.source for d in state.degraded] == ['archive']
    later = fold([first, heartbeat(2, head=OTHER_SHA, engagement_version=4)], project_id='p')
    assert (later.as_of, later.head, later.engagement_version, later.degraded) == (
        at(2),
        OTHER_SHA,
        4,
        (),
    )
    assert later.recent == ()


def test_a_task_re_upsert_replaces_its_state_and_its_timeline_entry() -> None:
    first, between, last = task(3, 5, 'pending'), refused(5), task(9, 5, 'blocked')
    state = fold([first, between, last], project_id='p')
    assert state.tasks == {5: last}
    assert state.recent == (between, last)


@pytest.mark.parametrize(
    'build',
    [prd, verify_run, case_results, merge, lambda seq: revealed(seq, 'src/a.py')],
    ids=['prd_upserted', 'verify_run', 'test_results', 'merge_landed', 'file_revealed'],
)
def test_a_keyed_re_emission_does_not_duplicate_in_recent(build: Any) -> None:
    first, between, again = build(1), refused(2), build(3)
    state = fold([first, between, again], project_id='p')
    assert state.recent == (between, again)


@pytest.mark.parametrize(('seqs', 'previous'), [((1, 3, 3), 3), ((1, 5, 3), 5)])
def test_a_non_increasing_seq_is_a_corrupt_feed(seqs: tuple[int, ...], previous: int) -> None:
    with pytest.raises(CorruptFeed) as caught:
        fold([heartbeat(seq) for seq in seqs], project_id='p')
    assert (caught.value.seq, caught.value.previous_seq) == (3, previous)
    assert '3' in str(caught.value)
    assert str(previous) in str(caught.value)


@pytest.mark.parametrize('seq', [4, 2])
def test_resuming_at_or_below_the_start_watermark_is_a_corrupt_feed(seq: int) -> None:
    start = fold([heartbeat(1), heartbeat(4)], project_id='p')
    with pytest.raises(CorruptFeed) as caught:
        fold([heartbeat(seq)], project_id='p', start=start)
    assert (caught.value.seq, caught.value.previous_seq) == (seq, 4)


def test_a_line_of_another_project_is_a_project_mismatch() -> None:
    feed = [heartbeat(1), task(2, 1, project_id='canary-b'), heartbeat(3)]
    with pytest.raises(ProjectMismatch) as caught:
        fold(feed, project_id='p')
    error = caught.value
    assert (error.seq, error.found, error.expected) == (2, 'canary-b', 'p')
    assert 'canary-b' in str(error)


def test_a_start_state_of_another_project_is_a_project_mismatch() -> None:
    with pytest.raises(ProjectMismatch) as caught:
        fold([heartbeat(1)], project_id='p', start=empty_state('q'))
    assert (caught.value.seq, caught.value.found, caught.value.expected) == (None, 'q', 'p')


def test_both_fold_refusals_are_feed_errors() -> None:
    assert issubclass(CorruptFeed, FeedError)
    assert issubclass(ProjectMismatch, FeedError)


@pytest.mark.parametrize(('upto_seq', 'watermark'), [(3, 3), (4, 3), (1, 1), (99, 5)])
def test_upto_seq_applies_only_lines_at_or_below_it(upto_seq: int, watermark: int) -> None:
    feed = [heartbeat(1), task(3, 1), task(5, 2)]
    state = fold(feed, project_id='p', upto_seq=upto_seq)
    assert state.watermark_seq == watermark
    assert state == fold([ln for ln in feed if ln.seq <= upto_seq], project_id='p')


def test_an_unknown_type_only_advances_the_watermark() -> None:
    without = [heartbeat(1), task(2, 1), refused(4)]
    with_unknown = [heartbeat(1), task(2, 1), unknown(3), refused(4), unknown(5)]
    state = fold(with_unknown, project_id='p')
    assert state.watermark_seq == 5
    assert state.model_dump(exclude={'watermark_seq'}) == fold(without, project_id='p').model_dump(
        exclude={'watermark_seq'}
    )


def test_in_flight_follows_the_current_phase_of_each_task() -> None:
    enter_plan = phase(1, 7, 'plan')
    assert fold([enter_plan], project_id='p').in_flight == (
        InFlightTask(task_id=7, phase='plan', since=at(1)),
    )
    feed = [enter_plan, phase(2, 7, 'implement')]
    assert fold(feed, project_id='p').in_flight == (
        InFlightTask(task_id=7, phase='implement', since=at(2)),
    )
    feed.append(phase(3, 7, 'plan', entered=False))
    assert [t.phase for t in fold(feed, project_id='p').in_flight] == ['implement']
    feed.append(phase(4, 7, 'implement', entered=False))
    assert fold(feed, project_id='p').in_flight == ()


def test_in_flight_is_sorted_by_task_id() -> None:
    state = fold([phase(1, 9, 'plan'), phase(2, 3, 'plan'), phase(3, 5, 'plan')], project_id='p')
    assert [t.task_id for t in state.in_flight] == [3, 5, 9]


def test_revealed_follows_grants_re_reveals_renames_and_deletes() -> None:
    feed = [revealed(1, 'src/b.py')]
    assert fold(feed, project_id='p').revealed == (
        RevealedFile(path='src/b.py', commit=SHA, lines=3),
    )
    feed.append(revealed(2, 'src/b.py', commit=OTHER_SHA, lines=5))
    assert fold(feed, project_id='p').revealed == (
        RevealedFile(path='src/b.py', commit=OTHER_SHA, lines=5),
    )
    feed.append(revealed(3, 'src/c.py', renamed_from='src/b.py'))
    assert [r.path for r in fold(feed, project_id='p').revealed] == ['src/c.py']
    feed.append(revealed(4, 'src/c.py', deleted=True))
    assert fold(feed, project_id='p').revealed == ()


def test_revealed_is_sorted_by_path() -> None:
    state = fold([revealed(1, 'src/z.py'), revealed(2, 'src/a.py')], project_id='p')
    assert [r.path for r in state.revealed] == ['src/a.py', 'src/z.py']


def test_criteria_keep_the_latest_result_per_run_kind() -> None:
    first_verify, staging, last_verify = criteria(1), criteria(2, 'staging'), criteria(3)
    state = fold([first_verify, staging, last_verify], project_id='p')
    assert state.criteria == {'verify': last_verify, 'staging': staging}


def test_snapshots_keep_the_latest() -> None:
    last_factory, last_branches, last_tree = factory_state(4, 'paused'), branches(5, 3), tree(6)
    feed = [factory_state(1), branches(2), tree(3, OTHER_SHA), last_factory, last_branches, last_tree]
    state = fold(feed, project_id='p')
    assert (state.factory, state.branches, state.tree) == (last_factory, last_branches, last_tree)


def test_recent_holds_timeline_events_only() -> None:
    timeline = [
        phase(2, 1, 'plan'),
        phase(3, 1, 'plan', entered=False),
        refused(5),
        refused(6),
        criteria(8),
        criteria(9),
    ]
    snapshots = [heartbeat(1), factory_state(4), branches(7), tree(10), unknown(11)]
    feed = sorted(timeline + snapshots, key=lambda ln: ln.seq)
    assert fold(feed, project_id='p').recent == tuple(timeline)


def test_recent_is_capped_and_drops_the_oldest_first() -> None:
    feed = [phase(seq, 1, 'plan', entered=seq % 2 == 1) for seq in range(1, 251)]
    recent = fold(feed, project_id='p').recent
    assert len(recent) == RECENT_LIMIT
    assert recent[0].seq == 51
    assert recent[-1].seq == 250


def test_the_mixed_feed_touches_every_known_type() -> None:
    kinds = {type(ln) for ln in mixed_feed()}
    assert kinds == set(EVENT_MODELS.values()) | {EventHeader}


@pytest.mark.parametrize('split', range(len(mixed_feed()) + 1))
def test_folding_from_a_watermark_equals_folding_the_whole_feed(split: int) -> None:
    feed = mixed_feed()
    start = fold(feed[:split], project_id='p')
    resumed = fold(feed[split:], project_id='p', start=start)
    whole = fold(feed, project_id='p')
    assert resumed == whole
    assert resumed.model_dump_json() == whole.model_dump_json()


def test_fold_never_mutates_its_start() -> None:
    feed = mixed_feed()
    start = fold(feed[:15], project_id='p')
    before = copy.deepcopy(start)
    fold(feed[15:], project_id='p', start=start)
    assert start == before
