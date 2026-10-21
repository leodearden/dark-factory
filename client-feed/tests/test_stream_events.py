"""Each runs.db stream row becomes its feed events once: phases, verify windows and merge verdicts, and landed merges."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _archive_files import (
    SECRET_CMD,
    SECRET_HINT,
    SECRET_MESSAGE,
    SECRET_TEXT,
    TASK_ID,
    junit_xml,
    passed,
    red,
    write_junit,
    write_plan,
    write_summary,
)
from _git_repo import make_repo
from _stores import RunsDb
from shared.project_layout import runs_db

from client_feed.contract import (
    FEED_VERSION,
    NOT_RECORDED,
    CaseCounts,
    CaseOutcome,
    FeedEvent,
    FeedState,
    MergeLanded,
    ReviewCounts,
    Source,
    TaskPhase,
    TaskStatus,
    TaskUpserted,
    VerifyKind,
    VerifyLeg,
    VerifyRun,
    VerifyTestResults,
)
from client_feed.feed_io import Stamp
from client_feed.fold import fold
from client_feed.sources.events import LandedCommit, PhaseChange, StreamEvent, VerifyVerdict
from client_feed.sources.git import GitError, merge_stats
from client_feed.sources.store import StoreReadError
from client_feed.stream_events import Identity, StreamOutput, identity, stream_events

PROJECT = 'acme'
BASE = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
MERGE_SHA = 'c' * 40
GREEN = (('test', 0), ('lint', 0), ('type', 0))
RED_TEST = (('test', 1), ('lint', 0), ('type', 0))


def at(minutes: float) -> datetime:
    return BASE + timedelta(minutes=minutes)


def phase(event_id: int, minutes: float, name: str, *, entered: bool, task_id: int = TASK_ID) -> PhaseChange:
    return PhaseChange(event_id=event_id, at=at(minutes), task_id=task_id, phase=name, entered=entered)


def verify_enter(event_id: int, minutes: float) -> PhaseChange:
    return phase(event_id, minutes, 'verify', entered=True)


def verify_exit(event_id: int, minutes: float) -> PhaseChange:
    return phase(event_id, minutes, 'verify', entered=False)


def merge_verdict(
    event_id: int,
    minutes: float,
    *,
    duration_ms: int | None = 10 * 60 * 1000,
    task_id: int | None = TASK_ID,
    passed_: bool = True,
) -> VerifyVerdict:
    return VerifyVerdict(
        event_id=event_id,
        at=at(minutes),
        kind=VerifyKind.MERGE,
        passed=passed_,
        duration_ms=duration_ms,
        commit=MERGE_SHA,
        task_id=task_id,
    )


def empty_state() -> FeedState:
    return fold((), project_id=PROJECT)


def run(
    root: Path,
    *stream: StreamEvent,
    state: FeedState | None = None,
    published: frozenset[Identity] = frozenset(),
    first_seq: int = 1,
) -> StreamOutput:
    return stream_events(
        root,
        stream,
        state=empty_state() if state is None else state,
        max_cases=100,
        published=published,
        stamp=Stamp(project_id=PROJECT, first_seq=first_seq),
    )


def verify_runs(output: StreamOutput) -> list[VerifyRun]:
    return [event for event in output.events if isinstance(event, VerifyRun)]


def results_of(output: StreamOutput) -> list[VerifyTestResults]:
    return [event for event in output.events if isinstance(event, VerifyTestResults)]


def task_phase(seq: int, change: PhaseChange) -> TaskPhase:
    return TaskPhase(
        seq=seq,
        at=change.at,
        project_id=PROJECT,
        feed_version=FEED_VERSION,
        task_id=change.task_id,
        phase=change.phase,
        entered=change.entered,
    )


def test_each_phase_change_becomes_one_task_phase(tmp_path: Path) -> None:
    changes = (
        phase(11, 0, 'plan', entered=True),
        phase(12, 5, 'plan', entered=False),
        phase(13, 6, 'implement', entered=True, task_id=9),
    )

    output = run(tmp_path, *changes)

    assert output == StreamOutput(
        events=tuple(task_phase(seq, change) for seq, change in enumerate(changes, start=1)),
        degraded=(),
    )


def test_a_red_verify_window_is_a_failed_task_run_with_its_test_results(tmp_path: Path) -> None:
    write_plan(tmp_path, at(1))
    write_summary(tmp_path, at(9), commands=RED_TEST)
    write_junit(
        tmp_path, at(9), junit_xml(passed('test_a'), red('test_b', 'failure', 'AssertionError'))
    )
    enter, exit_ = verify_enter(20, 0), verify_exit(21, 10)

    output = run(tmp_path, enter, exit_)

    assert output.events[:2] == (task_phase(1, enter), task_phase(2, exit_))
    assert output.events[2] == VerifyRun(
        seq=3,
        at=exit_.at,
        project_id=PROJECT,
        feed_version=FEED_VERSION,
        run_key='task-21',
        task_id=TASK_ID,
        kind=VerifyKind.TASK,
        passed=False,
        duration_ms=None,
        commit=None,
        legs=tuple(VerifyLeg(name=label, category='test_failure', rc=rc) for label, rc in RED_TEST),
    )
    results = output.events[3]
    assert isinstance(results, VerifyTestResults)
    assert (results.seq, results.at, results.run_key) == (4, exit_.at, 'task-21')
    assert results.counts == CaseCounts(passed=1, failed=1, skipped=0, errored=0, truncated=0)
    failed = [case for case in results.cases if case.outcome is CaseOutcome.FAILED]
    assert [case.exception_type for case in failed] == ['AssertionError']
    assert len(output.events) == 4
    assert output.degraded == ()


def test_no_secret_from_the_archive_reaches_the_events(tmp_path: Path) -> None:
    write_summary(tmp_path, at(9), commands=RED_TEST)
    write_junit(tmp_path, at(9), junit_xml(red('test_b', 'failure', 'AssertionError')))

    output = run(tmp_path, verify_enter(20, 0), verify_exit(21, 10))

    assert results_of(output)
    published = ''.join(event.model_dump_json() for event in output.events)
    for secret in (SECRET_MESSAGE, SECRET_TEXT, SECRET_CMD, SECRET_HINT):
        assert secret not in published


def test_an_all_zero_summary_is_a_passed_task_run(tmp_path: Path) -> None:
    write_summary(tmp_path, at(9), category='passed', commands=GREEN)

    (verify,) = verify_runs(run(tmp_path, verify_enter(20, 0), verify_exit(21, 10)))

    assert verify.passed is True


def test_a_window_with_no_summary_emits_no_verify_run(tmp_path: Path) -> None:
    write_junit(tmp_path, at(9), junit_xml(passed('test_a')))
    enter, exit_ = verify_enter(20, 0), verify_exit(21, 10)

    output = run(tmp_path, enter, exit_)

    assert output == StreamOutput(events=(task_phase(1, enter), task_phase(2, exit_)), degraded=())


def test_an_exit_whose_entry_was_folded_earlier_is_windowed_from_the_in_flight_entry(
    tmp_path: Path,
) -> None:
    write_summary(tmp_path, at(-5), qualifiers='stale', commands=RED_TEST)
    write_summary(tmp_path, at(9), qualifiers='alpha', category='passed', commands=GREEN)
    state = fold([task_phase(40, verify_enter(20, 0))], project_id=PROJECT)

    (verify,) = verify_runs(run(tmp_path, verify_exit(21, 10), state=state))

    assert verify.passed is True
    assert verify.legs == tuple(VerifyLeg(name=label, category='passed', rc=rc) for label, rc in GREEN)


def test_an_exit_with_no_known_entry_emits_only_its_task_phase(tmp_path: Path) -> None:
    write_summary(tmp_path, at(9), commands=RED_TEST)
    exit_ = verify_exit(21, 10)

    assert run(tmp_path, exit_) == StreamOutput(events=(task_phase(1, exit_),), degraded=())


def test_a_workflow_verify_verdict_emits_nothing(tmp_path: Path) -> None:
    write_summary(tmp_path, at(9), commands=GREEN)
    verdict = VerifyVerdict(
        event_id=30,
        at=at(10),
        kind=VerifyKind.TASK,
        passed=True,
        duration_ms=None,
        commit='d' * 40,
        task_id=TASK_ID,
    )

    assert run(tmp_path, verdict) == StreamOutput(events=(), degraded=())


def test_a_merge_verdict_reads_the_archive_over_its_duration(tmp_path: Path) -> None:
    write_summary(tmp_path, at(-15), qualifiers='stale', commands=RED_TEST)
    write_summary(tmp_path, at(5), qualifiers='alpha', category='passed', commands=GREEN)
    write_junit(tmp_path, at(5), junit_xml(passed('test_a'), passed('test_c')))
    verdict = merge_verdict(50, 10, duration_ms=10 * 60 * 1000)

    output = run(tmp_path, verdict)

    assert output.events[0] == VerifyRun(
        seq=1,
        at=verdict.at,
        project_id=PROJECT,
        feed_version=FEED_VERSION,
        run_key='merge-50',
        task_id=TASK_ID,
        kind=VerifyKind.MERGE,
        passed=True,
        duration_ms=10 * 60 * 1000,
        commit=MERGE_SHA,
        legs=tuple(VerifyLeg(name=label, category='passed', rc=rc) for label, rc in GREEN),
    )
    (results,) = results_of(output)
    assert (results.seq, results.run_key, results.at) == (2, 'merge-50', verdict.at)
    assert results.counts == CaseCounts(passed=2, failed=0, skipped=0, errored=0, truncated=0)
    assert len(output.events) == 2


def test_a_failed_merge_verdict_is_a_failed_merge_run(tmp_path: Path) -> None:
    (verify,) = verify_runs(run(tmp_path, merge_verdict(50, 10, passed_=False)))

    assert (verify.kind, verify.passed, verify.legs) == (VerifyKind.MERGE, False, NOT_RECORDED)


def test_a_merge_verdict_without_a_duration_or_a_task_is_not_windowed(tmp_path: Path) -> None:
    write_summary(tmp_path, at(5), category='passed', commands=GREEN)
    write_junit(tmp_path, at(5), junit_xml(passed('test_a')))

    output = run(
        tmp_path,
        merge_verdict(50, 10, duration_ms=None),
        merge_verdict(51, 10, task_id=None),
    )

    assert [(event.run_key, event.task_id, event.duration_ms, event.legs) for event in verify_runs(output)] == [
        ('merge-50', TASK_ID, None, NOT_RECORDED),
        ('merge-51', None, 10 * 60 * 1000, NOT_RECORDED),
    ]
    assert results_of(output) == []
    assert output.degraded == ()


def test_a_malformed_junit_degrades_the_archive_and_drops_that_task_run(tmp_path: Path) -> None:
    write_summary(tmp_path, at(9), commands=RED_TEST)
    junit = write_junit(tmp_path, at(9), b'<testsuites><testsuite')
    enter, exit_ = verify_enter(20, 0), verify_exit(21, 10)

    output = run(tmp_path, enter, exit_)

    assert output.events == (task_phase(1, enter), task_phase(2, exit_))
    (degradation,) = output.degraded
    assert degradation.source is Source.ARCHIVE
    assert str(junit.relative_to(tmp_path)) in degradation.reason
    assert str(tmp_path) not in degradation.reason


def test_a_malformed_junit_leaves_a_merge_run_with_no_legs(tmp_path: Path) -> None:
    write_summary(tmp_path, at(5), category='passed', commands=GREEN)
    write_junit(tmp_path, at(5), b'<testsuites><testsuite')

    output = run(tmp_path, merge_verdict(50, 10))

    (verify,) = verify_runs(output)
    assert (verify.run_key, verify.passed, verify.legs) == ('merge-50', True, NOT_RECORDED)
    assert results_of(output) == []
    assert [degradation.source for degradation in output.degraded] == [Source.ARCHIVE]


TITLE = 'Ship the widget'


@dataclass(frozen=True)
class Landed:
    root: Path
    runs: RunsDb
    merge: str


@pytest.fixture
def landed(tmp_path: Path) -> Landed:
    """A root whose main holds `Merge task/7 into main`, with an empty runs.db."""
    repo = make_repo(tmp_path / 'root')
    repo.write('README.md', 'hello\n')
    repo.commit('Initial', when=at(-60))
    repo.branch('task/7')
    repo.checkout('task/7')
    repo.write('src/widget.py', 'a = 1\nb = 2\nc = 3\n')
    repo.write('README.md', 'hello widget\n')
    repo.commit('Add the widget', when=at(-30))
    repo.checkout('main')
    merge = repo.merge_no_ff('task/7', 'Merge task/7 into main', when=at(-20))
    return Landed(root=repo.root, runs=RunsDb.create(repo.root), merge=merge)


def state_holding_task_7() -> FeedState:
    task = TaskUpserted(
        seq=1,
        at=at(-60),
        project_id=PROJECT,
        feed_version=FEED_VERSION,
        task_id=TASK_ID,
        title=TITLE,
        status=TaskStatus.DONE,
        waiting_on=None,
        updated_at=at(-60),
        depends_on=(),
        prd_path=None,
        prd_task_label=None,
        landed_commit=None,
    )
    return fold([task], project_id=PROJECT)


def landing(event_id: int, commit: str, *task_ids: int, minutes: float = 30) -> LandedCommit:
    return LandedCommit(event_id=event_id, at=at(minutes), commit=commit, task_ids=task_ids)


def review(cycle: int, blocking: int, suggestions: int, by_category: dict[str, int]) -> dict:
    """The keys of the orchestrator's review_outcome.py::review_completed_payload that the counts read."""
    return {
        'cycle': cycle,
        'blocking_count': blocking,
        'suggestion_count': suggestions,
        'by_category': by_category,
    }


def merges(output: StreamOutput) -> list[MergeLanded]:
    return [event for event in output.events if isinstance(event, MergeLanded)]


def test_a_landed_merge_carries_its_titles_its_stats_and_its_last_review(landed: Landed) -> None:
    landed.runs.emit(
        'review_completed', at=at(-40), task_id=7, phase='review',
        data=review(1, 2, 1, {'correctness': 2, 'style': 1}),
    )
    landed.runs.emit(
        'review_completed', at=at(-35), task_id=7, phase='review',
        data=review(2, 0, 3, {'style': 3}),
    )
    commit = landing(60, landed.merge, 7)

    output = run(landed.root, commit, state=state_holding_task_7())

    assert output == StreamOutput(
        events=(
            MergeLanded(
                seq=1,
                at=commit.at,
                project_id=PROJECT,
                feed_version=FEED_VERSION,
                commit=landed.merge,
                task_ids=(7,),
                titles=(TITLE,),
                **asdict(merge_stats(landed.root, landed.merge)),
                review=ReviewCounts(
                    cycles=2, blocking_count=0, suggestion_count=3, by_category={'style': 3}
                ),
            ),
        ),
        degraded=(),
    )


def test_a_merge_with_no_review_row_has_its_review_not_recorded(landed: Landed) -> None:
    (merge,) = merges(run(landed.root, landing(60, landed.merge, 7), state=state_holding_task_7()))

    assert merge.review == NOT_RECORDED


def test_a_train_landing_has_its_review_not_recorded(landed: Landed) -> None:
    landed.runs.emit('review_completed', at=at(-35), task_id=7, data=review(1, 0, 0, {}))

    (merge,) = merges(run(landed.root, landing(60, landed.merge, 7, 8), state=state_holding_task_7()))

    assert (merge.task_ids, merge.review) == ((7, 8), NOT_RECORDED)


def test_a_task_the_state_does_not_hold_is_titled_empty(landed: Landed) -> None:
    (merge,) = merges(run(landed.root, landing(60, landed.merge, 9), state=state_holding_task_7()))

    assert (merge.task_ids, merge.titles) == ((9,), ('',))


def test_two_landings_of_one_commit_are_one_merge_landed_at_the_first(landed: Landed) -> None:
    first = landing(60, landed.merge, 7, minutes=30)
    between = phase(61, 31, 'review', entered=True, task_id=9)
    second = landing(62, landed.merge, 8, 7, minutes=32)

    output = run(landed.root, first, between, second, state=state_holding_task_7())

    merge, after = output.events
    assert isinstance(merge, MergeLanded)
    assert (merge.seq, merge.at, merge.task_ids, merge.titles) == (1, first.at, (7, 8), (TITLE, ''))
    assert after == task_phase(2, between)


def identities(*events: FeedEvent) -> frozenset[Identity]:
    keys = [identity(event) for event in events]
    assert None not in keys
    return frozenset(key for key in keys if key is not None)


def test_identity_names_stream_events_only(landed: Landed) -> None:
    write_summary(landed.root, at(9), category='passed', commands=GREEN)
    write_junit(landed.root, at(9), junit_xml(passed('test_a')))
    stream = (verify_enter(20, 0), verify_exit(21, 10), landing(60, landed.merge, 7))
    enter, exit_, verify, results, merge = run(landed.root, *stream).events
    renumbered = run(landed.root, *stream, first_seq=50).events

    assert len(identities(enter, exit_, verify, results, merge)) == 5
    assert [identity(event) for event in renumbered] == [
        identity(event) for event in (enter, exit_, verify, results, merge)
    ]
    assert identity(state_holding_task_7().tasks[TASK_ID]) is None


def test_an_event_already_published_is_not_emitted_again(landed: Landed) -> None:
    write_summary(landed.root, at(9), category='passed', commands=GREEN)
    stream = (
        verify_enter(20, 0),
        verify_exit(21, 10),
        landing(60, landed.merge, 7),
        phase(63, 31, 'review', entered=True),
    )
    enter, exit_, verify, merge, review_entry = run(
        landed.root, *stream, state=state_holding_task_7()
    ).events

    again = run(
        landed.root,
        *stream,
        state=state_holding_task_7(),
        published=identities(enter, verify, merge),
        first_seq=100,
    )

    assert [identity(event) for event in again.events] == [identity(exit_), identity(review_entry)]
    seqs = [event.seq for event in again.events]
    assert seqs == sorted(set(seqs))
    assert seqs[0] >= 100


def test_a_merge_commit_unknown_to_git_raises(landed: Landed) -> None:
    with pytest.raises(GitError):
        run(landed.root, landing(60, 'e' * 40, 7), state=state_holding_task_7())


def test_an_unreadable_runs_db_raises(landed: Landed) -> None:
    runs_db(landed.root).write_bytes(b'this is not a sqlite database at all' * 100)

    with pytest.raises(StoreReadError):
        run(landed.root, landing(60, landed.merge, 7), state=state_holding_task_7())
