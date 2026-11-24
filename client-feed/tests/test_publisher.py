"""One publisher pass: every source read as its own unit, the events built in a fixed order, a heartbeat last."""

import json
import logging
import os
import shutil
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from _archive_files import (
    junit_xml,
    passed,
    red,
    task_dir,
    write_junit,
    write_plan,
    write_summary,
)
from _git_repo import Repo, make_repo, path_with_git_answering, path_with_git_killed_on
from _in_dir import write_engagement
from _stores import EscalationQueueDir, RunsDb, TasksDb
from shared.project_layout import runs_db, tasks_db

from client_feed.contract import (
    BranchSnapshot,
    CriteriaDeclared,
    CriteriaResults,
    CriterionOutcome,
    CriterionResult,
    Degradation,
    FactoryState,
    FeedLine,
    Heartbeat,
    MergeLanded,
    PrdUpserted,
    ReviewCounts,
    RunKind,
    SchedulerState,
    Source,
    TaskPhase,
    TaskStatus,
    TaskUpserted,
    TreeSnapshot,
    VerifyKind,
    VerifyRun,
    VerifyTestResults,
    WaitingOn,
    parse_state,
)
from client_feed.criteria import CRITERIA_RELPATH, STAGING_FILENAME, CriteriaRun, write_staging
from client_feed.feed_io import CURSOR_FILENAME, Cursor, OutDir, read_feed_lines
from client_feed.fold import fold
from client_feed.policy import Policy
from client_feed.publisher import (
    TASKS_RESCAN_PASSES,
    UNKNOWN_MERGE_GRACE,
    PassReport,
    publish,
    run_pass,
)
from client_feed.sources.collection_map import collection_map_path
from client_feed.validate import validate_out_dir

PROJECT = 'acme'
BASE = datetime(2026, 10, 21, 10, 0, tzinfo=UTC)
NOW = BASE + timedelta(hours=1)
X_PRD = 'plans/x-prd.md'
TITLE_7 = 'Ship the widget from /home/leo/x'
REDACTED_TITLE_7 = 'Ship the widget from [path]'
UPDATED_7 = '2026-10-21T09:00:00.000Z'
UPDATED_9 = '2026-10-21T09:30:00.000Z'
RED_TEST = (('test', 1), ('lint', 0), ('type', 0))
FIRST_PASS_TYPES = [
    'task_upserted',
    'task_upserted',
    'task_phase',
    'task_phase',
    'verify_run',
    'test_results',
    'merge_landed',
    'factory_state',
    'branch_snapshot',
    'tree_snapshot',
    'prd_upserted',
    'heartbeat',
]


def at(minutes: float) -> datetime:
    return BASE + timedelta(minutes=minutes)


def prd_markdown(title: str) -> str:
    return f'# {title}\n\n## 1. Goal\n\nThe goal paragraph.\n'


@dataclass(frozen=True)
class Project:
    """A fixture root as the publisher meets it: main, task branches, the two stores, the escalation queue, a verify archive."""

    repo: Repo
    tasks: TasksDb
    runs: RunsDb
    queue: EscalationQueueDir
    policy: Policy
    merge: str
    last_stream_id: int

    @property
    def root(self) -> Path:
        return self.repo.root

    @property
    def out_dir(self) -> Path:
        return self.policy.out_dir

    def open(self) -> OutDir:
        return OutDir.open(self.out_dir, project_id=PROJECT)


def _git_history(root: Path) -> tuple[Repo, str]:
    """main with an initial commit and `Merge task/7 into main`; task/9 one commit ahead, unmerged."""
    repo = make_repo(root)
    repo.write('.gitignore', '.taskmaster/\ndata/\n')
    repo.write('README.md', 'hello\n')
    repo.write(X_PRD, prd_markdown('X plan'))
    repo.commit('Initial', when=at(-60))
    repo.branch('task/7')
    repo.checkout('task/7')
    repo.write('src/widget.py', 'a = 1\nb = 2\n')
    repo.commit('Add the widget', when=at(-30))
    repo.checkout('main')
    merge = repo.merge_no_ff('task/7', 'Merge task/7 into main', when=at(18))
    repo.branch('task/9')
    repo.checkout('task/9')
    repo.write('src/gadget.py', 'g = 1\n')
    repo.commit('Start the gadget', when=at(-20))
    repo.checkout('main')
    return repo, merge


def _red_verify_attempt(root: Path) -> None:
    """Task 7's archived attempt inside its verify window [at(0), at(10)]: the test leg red."""
    write_plan(root, at(1))
    write_summary(root, at(5), commands=RED_TEST)
    write_junit(root, at(5), junit_xml(passed('test_a'), red('test_b', 'failure', 'AssertionError')))


def _review(cycle: int) -> dict:
    return {'cycle': cycle, 'blocking_count': 0, 'suggestion_count': 2, 'by_category': {'style': 2}}


def policy_over(root: Path, out_dir: Path, in_dir: Path) -> Policy:
    return Policy.model_validate({
        'project_id': PROJECT,
        'project_root': str(root),
        'out_dir': str(out_dir),
        'in_dir': str(in_dir),
        'visible_globs': ['src/**', 'plans/**', 'README.md'],
    })


@pytest.fixture
def project(tmp_path: Path) -> Project:
    repo, merge = _git_history(tmp_path / 'root')
    tasks = TasksDb.create(repo.root)
    tasks.add(7, title=TITLE_7, status='done', updated_at=UPDATED_7, metadata={'prd_path': X_PRD})
    tasks.add(
        9, title='Build the gadget', status='blocked', updated_at=UPDATED_9, metadata={'prd_path': X_PRD}
    )
    _red_verify_attempt(repo.root)
    runs = RunsDb.create(repo.root)
    runs.emit('phase_enter', at=at(0), task_id=7, phase='verify')
    runs.emit('phase_exit', at=at(10), task_id=7, phase='verify')
    runs.emit('review_completed', at=at(12), task_id=7, phase='review', data=_review(1))
    landed = runs.emit(
        'merge_finalized', at=at(20), task_id=7, phase='merge', data={'state': 'done', 'merge_sha': merge}
    )
    runs.emit('scheduler_paused', at=at(30))
    queue = EscalationQueueDir.create(repo.root)
    queue.write('esc-9-1', task_id='9', level=2)
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    policy = policy_over(repo.root, out_dir, tmp_path / 'in')
    write_engagement(policy.in_dir, project_id=PROJECT)
    return Project(
        repo=repo, tasks=tasks, runs=runs, queue=queue, policy=policy, merge=merge,
        last_stream_id=landed,
    )


def feed(project: Project) -> tuple[FeedLine, ...]:
    return read_feed_lines(project.out_dir)


def stored_cursor(project: Project) -> Cursor:
    return Cursor.model_validate_json((project.out_dir / CURSOR_FILENAME).read_text())


def typed(lines: tuple[FeedLine, ...]) -> list[tuple[str, int | None]]:
    return [(line.type, getattr(line, 'task_id', None)) for line in lines]


def test_a_first_pass_publishes_every_source_in_order_with_the_heartbeat_last(project: Project) -> None:
    run_pass(project.policy, project.open(), now=NOW)

    lines = feed(project)
    assert [line.type for line in lines] == FIRST_PASS_TYPES
    assert [line.seq for line in lines] == list(range(1, len(FIRST_PASS_TYPES) + 1))
    task_7, task_9, enter, exit_, verify, results, merge, factory, branches, tree, prd, heartbeat = lines
    assert isinstance(task_7, TaskUpserted) and isinstance(task_9, TaskUpserted)
    assert (task_7.task_id, task_7.title, task_7.status, task_7.waiting_on) == (
        7, REDACTED_TITLE_7, TaskStatus.DONE, None,
    )
    assert (task_9.task_id, task_9.status, task_9.waiting_on) == (9, TaskStatus.BLOCKED, WaitingOn.LEO)
    assert isinstance(enter, TaskPhase) and isinstance(exit_, TaskPhase)
    assert [(p.task_id, p.phase, p.entered, p.at) for p in (enter, exit_)] == [
        (7, 'verify', True, at(0)),
        (7, 'verify', False, at(10)),
    ]
    assert isinstance(verify, VerifyRun)
    assert (verify.task_id, verify.kind, verify.passed, verify.at) == (7, VerifyKind.TASK, False, at(10))
    assert isinstance(results, VerifyTestResults) and results.run_key == verify.run_key
    assert isinstance(merge, MergeLanded)
    assert (merge.commit, merge.task_ids, merge.titles, merge.at) == (
        project.merge, (7,), (REDACTED_TITLE_7,), at(20),
    )
    assert merge.review == ReviewCounts(
        cycles=1, blocking_count=0, suggestion_count=2, by_category={'style': 2}
    )
    assert isinstance(factory, FactoryState)
    assert (factory.scheduler, factory.since, factory.last_event_at, factory.idle_seconds) == (
        SchedulerState.PAUSED, at(30), at(30), 30 * 60,
    )
    assert isinstance(branches, BranchSnapshot)
    assert [(b.name, b.task_id, b.ahead) for b in branches.branches] == [('task/7', 7, 0), ('task/9', 9, 1)]
    assert isinstance(tree, TreeSnapshot)
    assert tree.commit == project.repo.rev('main')
    assert [file.path for file in tree.files] == ['README.md', X_PRD, 'src/widget.py']
    assert isinstance(prd, PrdUpserted)
    assert (prd.prd_path, prd.title, prd.task_counts) == (
        X_PRD, 'X plan', {TaskStatus.BLOCKED: 1, TaskStatus.DONE: 1},
    )
    assert isinstance(heartbeat, Heartbeat)
    assert (
        heartbeat.at, heartbeat.head, heartbeat.as_of, heartbeat.engagement_version,
        heartbeat.watermark_seq, heartbeat.degraded,
    ) == (NOW, project.repo.rev('main'), NOW, 1, 0, ())


def test_the_pass_leaves_a_valid_out_dir_whose_state_is_the_fold_of_its_feed(project: Project) -> None:
    run_pass(project.policy, project.open(), now=NOW)

    assert validate_out_dir(project.out_dir, expected_project_id=PROJECT).ok
    state = parse_state((project.out_dir / 'state.json').read_text())
    assert state == fold(feed(project), project_id=PROJECT)


def test_the_cursor_records_how_far_each_source_and_the_feed_were_committed(project: Project) -> None:
    run_pass(project.policy, project.open(), now=NOW)

    cursor = stored_cursor(project)
    lines = feed(project)
    assert cursor.events_id == project.last_stream_id
    assert cursor.tasks_updated_at == UPDATED_9
    assert cursor.head == project.repo.rev('main')
    assert cursor.line_counts is not None
    assert cursor.line_counts.max_line_bytes == project.policy.max_line_bytes
    assert len(cursor.line_counts.by_blob) == 3
    assert cursor.through_seq == lines[-1].seq
    assert cursor.feed_bytes == (project.out_dir / 'feed.jsonl').stat().st_size


def test_a_second_pass_over_unchanged_sources_appends_only_a_heartbeat(project: Project) -> None:
    out = project.open()
    run_pass(project.policy, out, now=NOW)
    first = feed(project)

    run_pass(project.policy, out, now=NOW)

    (heartbeat,) = feed(project)[len(first) :]
    assert isinstance(heartbeat, Heartbeat)
    assert heartbeat.watermark_seq == first[-1].seq
    assert validate_out_dir(project.out_dir, expected_project_id=PROJECT).ok


def test_a_new_main_commit_publishes_its_tree_and_rereads_the_prds(project: Project) -> None:
    out = project.open()
    run_pass(project.policy, out, now=NOW)
    first = feed(project)
    project.repo.write(X_PRD, prd_markdown('X plan, revised'))
    head = project.repo.commit('Revise the plan', when=at(40))

    run_pass(project.policy, out, now=NOW)

    tree, prd, heartbeat = feed(project)[len(first) :]
    assert isinstance(tree, TreeSnapshot) and tree.commit == head
    assert isinstance(prd, PrdUpserted) and (prd.prd_path, prd.title) == (X_PRD, 'X plan, revised')
    assert isinstance(heartbeat, Heartbeat) and heartbeat.head == head
    assert stored_cursor(project).head == head


def test_a_new_task_row_and_a_new_stream_row_publish_without_a_rescan(project: Project) -> None:
    out = project.open()
    run_pass(project.policy, out, now=NOW)
    first = feed(project)
    project.tasks.add(11, title='Polish', status='in-progress', updated_at='2026-10-21T10:45:00.000Z')
    phase_id = project.runs.emit('phase_enter', at=at(40), task_id=11, phase='implement')

    run_pass(project.policy, out, now=NOW, rescan_tasks=False)

    assert typed(feed(project)[len(first) :]) == [
        ('task_upserted', 11),
        ('task_phase', 11),
        ('factory_state', None),
        ('heartbeat', None),
    ]
    cursor = stored_cursor(project)
    assert (cursor.events_id, cursor.tasks_updated_at) == (phase_id, '2026-10-21T10:45:00.000Z')


def test_the_report_carries_what_the_pass_appended(project: Project) -> None:
    out = project.open()

    first = run_pass(project.policy, out, now=NOW)
    second = run_pass(project.policy, out, now=NOW)

    count = len(FIRST_PASS_TYPES)
    assert first == PassReport(appended=count, watermark_seq=count, degraded=())
    assert second == PassReport(appended=1, watermark_seq=count + 1, degraded=())


NOT_A_DATABASE = b'this is not a sqlite database at all' * 100
UNKNOWN_SHA = 'e' * 40


def assert_valid(project: Project) -> None:
    assert validate_out_dir(project.out_dir, expected_project_id=PROJECT).ok


def types_of(lines: tuple[FeedLine, ...]) -> set[str]:
    return {line.type for line in lines}


def last_heartbeat(project: Project) -> Heartbeat:
    heartbeat = feed(project)[-1]
    assert isinstance(heartbeat, Heartbeat)
    return heartbeat


def warnings_of(caplog: pytest.LogCaptureFixture) -> str:
    return '\n'.join(
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    )


def test_an_unreadable_runs_db_is_named_events_and_every_other_source_still_publishes(
    project: Project, caplog: pytest.LogCaptureFixture
) -> None:
    runs_db(project.root).write_bytes(NOT_A_DATABASE)

    with caplog.at_level(logging.WARNING):
        report = run_pass(project.policy, project.open(), now=NOW)

    (degradation,) = last_heartbeat(project).degraded
    assert degradation.source is Source.EVENTS
    assert runs_db(project.root).relative_to(project.root).as_posix() in degradation.reason
    assert '/home/' not in degradation.reason
    assert str(project.root.parent) not in degradation.reason
    assert report.degraded == (degradation,)
    published = types_of(feed(project))
    assert {'task_upserted', 'branch_snapshot', 'tree_snapshot', 'prd_upserted'} <= published
    assert not published & {'task_phase', 'verify_run', 'merge_landed'}
    assert stored_cursor(project).events_id == 0
    assert 'events' in warnings_of(caplog)
    assert_valid(project)


def upserted_task(lines: tuple[FeedLine, ...], task_id: int) -> TaskUpserted:
    (task,) = (line for line in lines if isinstance(line, TaskUpserted) and line.task_id == task_id)
    return task


def test_an_unreadable_runs_db_holds_the_stream_but_not_who_the_escalated_task_waits_on(
    project: Project,
) -> None:
    good = runs_db(project.root).read_bytes()
    runs_db(project.root).write_bytes(NOT_A_DATABASE)
    out = project.open()

    blind = run_pass(project.policy, out, now=NOW)

    held = feed(project)
    assert [degradation.source for degradation in blind.degraded] == [Source.EVENTS]
    assert upserted_task(held, 9).waiting_on is WaitingOn.LEO
    assert_valid(project)
    runs_db(project.root).write_bytes(good)

    report = run_pass(project.policy, out, now=NOW, rescan_tasks=False)

    recovered = feed(project)[len(held) :]
    assert report.degraded == ()
    assert {'task_phase', 'verify_run', 'merge_landed'} <= types_of(recovered)
    assert stored_cursor(project).events_id == project.last_stream_id
    assert_valid(project)


def test_a_missing_escalation_queue_is_named_events_and_its_return_rewaits_the_escalated_task(
    project: Project,
) -> None:
    shutil.rmtree(project.queue.path)
    out = project.open()

    blind = run_pass(project.policy, out, now=NOW)

    first = feed(project)
    (degradation,) = blind.degraded
    assert degradation.source is Source.EVENTS
    assert 'data/escalations' in degradation.reason
    assert upserted_task(first, 9).waiting_on is WaitingOn.FACTORY
    assert_valid(project)
    EscalationQueueDir.create(project.root).write('esc-9-1', task_id='9', level=2)

    report = run_pass(project.policy, out, now=NOW, rescan_tasks=False)

    upserts = [line for line in feed(project)[len(first) :] if isinstance(line, TaskUpserted)]
    assert report.degraded == ()
    assert [(task.task_id, task.waiting_on) for task in upserts] == [(9, WaitingOn.LEO)]
    assert_valid(project)


def _queue_replaced_by_a_file(queue: Path) -> None:
    shutil.rmtree(queue)
    queue.write_text('not a directory')


@pytest.mark.parametrize(
    'break_queue', [shutil.rmtree, _queue_replaced_by_a_file], ids=['removed', 'replaced-by-a-file']
)
def test_a_rescan_while_the_escalation_queue_is_unreadable_holds_the_escalated_tasks_waiting_on(
    project: Project, break_queue: Callable[[Path], object]
) -> None:
    out = project.open()
    run_pass(project.policy, out, now=NOW)
    assert out.state.tasks[9].waiting_on is WaitingOn.LEO
    first = feed(project)
    break_queue(project.queue.path)

    report = run_pass(project.policy, out, now=NOW, rescan_tasks=True)

    assert [degradation.source for degradation in report.degraded] == [Source.EVENTS]
    assert 'task_upserted' not in types_of(feed(project)[len(first) :])
    assert out.state.tasks[9].waiting_on is WaitingOn.LEO
    assert_valid(project)


def test_a_malformed_escalation_record_is_counted_and_the_escalated_task_still_waits_on_leo(
    project: Project,
) -> None:
    project.queue.write_raw('esc-9-2.json', '{not json')

    report = run_pass(project.policy, project.open(), now=NOW)

    assert report.degraded == (
        Degradation(
            source=Source.EVENTS,
            reason='skipped 1 records in data/escalations: malformed_record=1',
        ),
    )
    assert upserted_task(feed(project), 9).waiting_on is WaitingOn.LEO
    assert_valid(project)


def test_a_missing_tasks_db_is_named_tasks_and_the_events_and_git_still_publish(
    project: Project,
) -> None:
    tasks_db(project.root).unlink()

    report = run_pass(project.policy, project.open(), now=NOW)

    (degradation,) = report.degraded
    assert degradation.source is Source.TASKS
    assert 'missing' in degradation.reason
    published = types_of(feed(project))
    assert {'task_phase', 'verify_run', 'merge_landed', 'branch_snapshot', 'tree_snapshot'} <= published
    assert 'task_upserted' not in published
    assert stored_cursor(project).tasks_updated_at is None
    assert_valid(project)


def test_skipped_rows_are_named_with_their_counts_and_the_other_rows_publish(
    project: Project, caplog: pytest.LogCaptureFixture
) -> None:
    project.runs.emit('merge_verify', at=at(31), task_id=7, phase='merge', data='{not json')
    project.tasks.add(12, title='Odd', status='exploding', updated_at='2026-10-21T09:45:00.000Z')

    with caplog.at_level(logging.WARNING):
        report = run_pass(project.policy, project.open(), now=NOW)

    assert report.degraded == (
        Degradation(source=Source.EVENTS, reason='skipped 1 rows: malformed_payload=1'),
        Degradation(source=Source.TASKS, reason='skipped 1 rows: unknown_status=1'),
    )
    assert [line.type for line in feed(project)] == FIRST_PASS_TYPES
    logged = warnings_of(caplog)
    assert 'malformed_payload' in logged and 'unknown_status' in logged
    assert_valid(project)


def test_a_merge_unknown_to_git_holds_the_whole_events_batch(project: Project) -> None:
    project.runs.emit(
        'merge_finalized', at=at(31), task_id=9, phase='merge',
        data={'state': 'done', 'merge_sha': UNKNOWN_SHA},
    )
    out = project.open()

    first = run_pass(project.policy, out, now=NOW)
    second = run_pass(project.policy, out, now=NOW)

    assert [degradation.source for degradation in first.degraded] == [Source.GIT]
    assert second.degraded == first.degraded
    published = types_of(feed(project))
    assert not published & {'task_phase', 'verify_run', 'test_results', 'merge_landed'}
    assert {'task_upserted', 'branch_snapshot', 'tree_snapshot'} <= published
    assert stored_cursor(project).events_id == 0
    assert_valid(project)


def errors_of(caplog: pytest.LogCaptureFixture) -> str:
    return '\n'.join(
        record.getMessage() for record in caplog.records if record.levelno == logging.ERROR
    )


def test_a_merge_missing_from_the_repository_is_skipped_once_it_landed_longer_ago_than_the_grace(
    project: Project, caplog: pytest.LogCaptureFixture
) -> None:
    unknown_id = project.runs.emit(
        'merge_finalized', at=at(31), task_id=9, phase='merge',
        data={'state': 'done', 'merge_sha': UNKNOWN_SHA},
    )
    out = project.open()
    run_pass(project.policy, out, now=NOW)
    assert stored_cursor(project).events_id == 0
    past_the_grace = at(31) + UNKNOWN_MERGE_GRACE + timedelta(minutes=1)

    with caplog.at_level(logging.ERROR):
        report = run_pass(project.policy, out, now=past_the_grace)

    assert report.degraded == (
        Degradation(
            source=Source.GIT, reason='skipped 1 landing whose merge commit is not in the repository'
        ),
    )
    assert [line.commit for line in feed(project) if isinstance(line, MergeLanded)] == [project.merge]
    assert {'task_phase', 'verify_run', 'test_results'} <= types_of(feed(project))
    assert stored_cursor(project).events_id == unknown_id
    assert UNKNOWN_SHA in errors_of(caplog)
    assert run_pass(project.policy, out, now=past_the_grace).degraded == ()
    assert_valid(project)


def test_a_merge_git_fails_to_read_is_held_however_old_it_is(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_path = os.environ['PATH']
    monkeypatch.setenv('PATH', path_with_git_killed_on(tmp_path / 'bin', 'diff-tree'))
    out = project.open()
    a_month_on = NOW + timedelta(days=30)

    held = run_pass(project.policy, out, now=a_month_on)

    (degradation,) = held.degraded
    assert degradation.source is Source.GIT
    assert 'diff-tree' in degradation.reason
    assert not degradation.reason.startswith('skipped')
    assert 'merge_landed' not in types_of(feed(project))
    assert stored_cursor(project).events_id == 0
    assert_valid(project)

    monkeypatch.setenv('PATH', real_path)
    delivered = run_pass(project.policy, out, now=a_month_on)

    assert [line.commit for line in feed(project) if isinstance(line, MergeLanded)] == [project.merge]
    assert stored_cursor(project).events_id >= project.last_stream_id
    assert not any(degradation.reason.startswith('skipped') for degradation in delivered.degraded)
    assert_valid(project)


def test_a_merge_held_while_git_cannot_read_main_is_never_skipped(
    project: Project, tmp_path: Path
) -> None:
    shutil.move(project.root / '.git', tmp_path / 'moved.git')
    (project.root / '.git').write_text('gitdir: /nonexistent/gitdir\n')

    report = run_pass(project.policy, project.open(), now=NOW + timedelta(days=30))

    assert not any(degradation.reason.startswith('skipped') for degradation in report.degraded)
    assert 'merge_landed' not in types_of(feed(project))
    assert stored_cursor(project).events_id == 0
    assert_valid(project)


def test_an_unusable_git_directory_is_named_git_and_the_heartbeat_has_no_head(
    project: Project, tmp_path: Path
) -> None:
    shutil.move(project.root / '.git', tmp_path / 'moved.git')
    (project.root / '.git').write_text('gitdir: /nonexistent/gitdir\n')

    report = run_pass(project.policy, project.open(), now=NOW)

    assert last_heartbeat(project).head is None
    assert report.degraded
    assert {degradation.source for degradation in report.degraded} == {Source.GIT}
    published = types_of(feed(project))
    assert 'task_upserted' in published
    assert not published & {'branch_snapshot', 'tree_snapshot'}
    assert stored_cursor(project).head is None
    assert_valid(project)


def test_git_output_that_does_not_parse_is_named_git_and_the_pass_still_commits(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('PATH', path_with_git_answering(tmp_path / 'bin', 'ls-tree', b'garbage\0'))

    report = run_pass(project.policy, project.open(), now=NOW)

    assert {degradation.source for degradation in report.degraded} == {Source.GIT}
    assert 'task_upserted' in types_of(feed(project))
    assert_valid(project)


@dataclass(frozen=True)
class Landing:
    merge: str
    last_event_id: int


def land_task_9(project: Project) -> Landing:
    """`Merge task/9 into main` in git, with the merge verify and the landing runs.db records after it."""
    merge = project.repo.merge_no_ff('task/9', 'Merge task/9 into main', when=at(35))
    project.runs.emit(
        'merge_verify', at=at(39), task_id=9, phase='merge',
        data={'passed': True, 'duration_ms': 60_000, 'merge_sha': merge},
    )
    landed = project.runs.emit(
        'merge_finalized', at=at(40), task_id=9, phase='merge', data={'state': 'done', 'merge_sha': merge}
    )
    return Landing(merge=merge, last_event_id=landed)


def committed_files(project: Project) -> dict[str, bytes]:
    return {name: (project.out_dir / name).read_bytes() for name in (CURSOR_FILENAME, 'state.json')}


def restore(project: Project, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        (project.out_dir / name).write_bytes(data)


def cut_feed(project: Project, *, lines_kept: int, torn: bool) -> None:
    """The feed as a kill left it: its first `lines_kept` lines, and half of the next when the append was torn."""
    path = project.out_dir / 'feed.jsonl'
    texts = path.read_bytes().split(b'\n')
    kept = b''.join(text + b'\n' for text in texts[:lines_kept])
    path.write_bytes(kept + (texts[lines_kept][: len(texts[lines_kept]) // 2] if torn else b''))


def crashed_in_pass_2(project: Project, *, cut_after_merge: bool) -> Landing:
    """Pass 1 committed; pass 2 appended over a second landing, then was killed before state.json was replaced."""
    out = project.open()
    run_pass(project.policy, out, now=NOW)
    landing = land_task_9(project)
    before_pass_2 = committed_files(project)
    run_pass(project.policy, out, now=NOW)
    lines = feed(project)
    landed_at = next(
        index for index, line in enumerate(lines)
        if isinstance(line, MergeLanded) and line.commit == landing.merge
    )
    assert landed_at + 1 < len(lines)
    restore(project, before_pass_2)
    if cut_after_merge:
        cut_feed(project, lines_kept=landed_at + 1, torn=True)
    return landing


def assert_recovered(project: Project, landing: Landing) -> None:
    lines = feed(project)
    seqs = [line.seq for line in lines]
    assert seqs == sorted(set(seqs))
    assert_valid(project)
    merges = Counter(line.commit for line in lines if isinstance(line, MergeLanded))
    assert merges == {project.merge: 1, landing.merge: 1}
    verify_runs = Counter(line.run_key for line in lines if isinstance(line, VerifyRun))
    assert len(verify_runs) == 2 and set(verify_runs.values()) == {1}
    cursor = stored_cursor(project)
    assert cursor.through_seq == lines[-1].seq
    assert cursor.feed_bytes == (project.out_dir / 'feed.jsonl').stat().st_size
    assert cursor.events_id == landing.last_event_id
    assert cursor.head == project.repo.rev('main') == landing.merge


def test_a_kill_mid_append_cuts_the_torn_line_and_republishes_nothing_twice(
    project: Project, caplog: pytest.LogCaptureFixture
) -> None:
    landing = crashed_in_pass_2(project, cut_after_merge=True)

    with caplog.at_level(logging.WARNING, logger='client_feed.feed_io'):
        out = project.open()
    run_pass(project.policy, out, now=NOW)

    assert any(
        record.name == 'client_feed.feed_io' and 'torn' in record.getMessage()
        for record in caplog.records
    )
    assert_recovered(project, landing)


def test_a_batch_appended_whole_whose_cursor_was_never_written_is_not_republished(
    project: Project,
) -> None:
    landing = crashed_in_pass_2(project, cut_after_merge=False)

    run_pass(project.policy, project.open(), now=NOW)

    assert_recovered(project, landing)


def test_a_restart_without_a_crash_reemits_nothing_but_a_heartbeat(project: Project) -> None:
    run_pass(project.policy, project.open(), now=NOW)
    first = feed(project)

    run_pass(project.policy, project.open(), now=NOW)

    (heartbeat,) = feed(project)[len(first) :]
    assert isinstance(heartbeat, Heartbeat)
    assert heartbeat.watermark_seq == first[-1].seq
    assert_valid(project)


class StopAfterWaits(threading.Event):
    """A stop Event that never sleeps and is set by the `waits`-th wait between passes."""

    def __init__(self, waits: int) -> None:
        super().__init__()
        self._left = waits

    def wait(self, timeout: float | None = None) -> bool:
        self._left -= 1
        if self._left == 0:
            self.set()
        return self.is_set()


def test_publish_rereads_every_task_on_its_first_pass_and_then_every_rescan_cadence(
    project: Project,
) -> None:
    passes: list[datetime] = []

    def clock() -> datetime:
        passes.append(NOW)
        if len(passes) == 2:
            project.tasks.add(13, title='Late', status='pending', updated_at='2026-10-21T08:00:00.000Z')
        return NOW

    publish(project.policy, stop=StopAfterWaits(TASKS_RESCAN_PASSES + 1), clock=clock)

    lines = feed(project)
    assert len(passes) == TASKS_RESCAN_PASSES + 1
    assert [line.task_id for line in lines if isinstance(line, TaskUpserted)] == [7, 9, 13]
    (late_at,) = (
        index for index, line in enumerate(lines)
        if isinstance(line, TaskUpserted) and line.task_id == 13
    )
    passes_before = sum(isinstance(line, Heartbeat) for line in lines[:late_at])
    assert passes_before == TASKS_RESCAN_PASSES
    assert stored_cursor(project).tasks_updated_at == UPDATED_9
    assert_valid(project)


# acceptance criteria

CRITERION_IDS = ('C-1', 'C-2', 'C-3')
STAGED = CriteriaRun(
    commit='f' * 40,
    run_at=at(40),
    results=(CriterionResult(id='C-1', outcome=CriterionOutcome.FAILED, tests=('tests.e2e::t',)),),
)


def _declare_criteria(project: Project, text: str | None = None) -> None:
    """acceptance/criteria.yaml declaring C-1..C-3, committed on main."""
    declared = {
        'format': 1,
        'criteria': [
            {
                'id': criterion_id,
                'prd': X_PRD,
                'capability': 'Widget',
                'statement': f'The widget meets {criterion_id}.',
                'kind': 'success',
            }
            for criterion_id in CRITERION_IDS
        ],
    }
    project.repo.write(CRITERIA_RELPATH, yaml.safe_dump(declared) if text is None else text)
    project.repo.commit('Declare the acceptance criteria', when=at(19))


def _bind(junit: Path) -> None:
    """A defect-tests collection map beside `junit` binding test_a to C-1 and test_b to C-2."""
    collection_map = {
        'tests/test_a.py::test_a': {'defects': [], 'criterion': 'C-1'},
        'tests/test_a.py::test_b': {'defects': [], 'criterion': 'C-2'},
    }
    collection_map_path(junit).write_text(json.dumps(collection_map))


def _map_beside_the_archived_junit(project: Project) -> None:
    (junit,) = task_dir(project.root).glob('*.junit-*.xml.gz')
    _bind(junit)


def _task_9_verify_run_with_a_map(project: Project) -> None:
    """Task 9's verify window [at(40), at(50)]: test_a red and test_b green, with a map beside its junit."""
    write_plan(project.root, at(41), task_id=9)
    write_summary(project.root, at(45), commands=RED_TEST, task_id=9)
    junit = write_junit(
        project.root,
        at(45),
        junit_xml(red('test_a', 'failure', 'AssertionError'), passed('test_b')),
        task_id=9,
    )
    _bind(junit)
    project.runs.emit('phase_enter', at=at(40), task_id=9, phase='verify')
    project.runs.emit('phase_exit', at=at(50), task_id=9, phase='verify')


TASK_9_RESULTS = (
    CriterionResult(id='C-1', outcome=CriterionOutcome.FAILED, tests=('tests.test_a::test_a',)),
    CriterionResult(id='C-2', outcome=CriterionOutcome.PASSED, tests=('tests.test_a::test_b',)),
    CriterionResult(id='C-3', outcome=CriterionOutcome.NOT_RUN, tests=()),
)


def _criteria_events(project: Project) -> list[CriteriaDeclared | CriteriaResults]:
    return [
        line for line in feed(project) if isinstance(line, CriteriaDeclared | CriteriaResults)
    ]


def _results_of(project: Project, kind: RunKind) -> list[CriteriaResults]:
    return [
        line
        for line in feed(project)
        if isinstance(line, CriteriaResults) and line.run_kind is kind
    ]


def test_b20_a_pass_publishes_the_declared_criteria_and_the_verify_runs_outcomes(
    project: Project,
) -> None:
    _declare_criteria(project)
    _map_beside_the_archived_junit(project)

    run_pass(project.policy, project.open(), now=NOW)

    lines = feed(project)
    assert [line.type for line in lines[-4:]] == [
        'prd_upserted',
        'criteria_declared',
        'criteria_results',
        'heartbeat',
    ]
    declared, results = lines[-3], lines[-2]
    assert isinstance(declared, CriteriaDeclared)
    assert [criterion.id for criterion in declared.criteria] == list(CRITERION_IDS)
    assert isinstance(results, CriteriaResults)
    assert (results.run_kind, results.at, results.run_at) == (RunKind.VERIFY, at(10), at(10))
    assert results.results == (
        CriterionResult(id='C-1', outcome=CriterionOutcome.PASSED, tests=('tests.test_a::test_a',)),
        CriterionResult(id='C-2', outcome=CriterionOutcome.FAILED, tests=('tests.test_a::test_b',)),
        CriterionResult(id='C-3', outcome=CriterionOutcome.NOT_RUN, tests=()),
    )
    assert_valid(project)


def test_a_second_pass_appends_no_criteria_event_again(project: Project) -> None:
    _declare_criteria(project)
    _map_beside_the_archived_junit(project)
    out = project.open()
    run_pass(project.policy, out, now=NOW)
    first = len(feed(project))

    run_pass(project.policy, out, now=NOW)

    assert [line.type for line in feed(project)[first:]] == ['heartbeat']


def test_the_criteria_join_reads_every_case_whatever_the_case_cap(project: Project) -> None:
    _declare_criteria(project)
    _map_beside_the_archived_junit(project)
    capped = project.policy.model_copy(update={'max_cases_per_run': 1})

    run_pass(capped, project.open(), now=NOW)

    (test_results,) = [line for line in feed(project) if isinstance(line, VerifyTestResults)]
    assert [case.id for case in test_results.cases] == ['tests.test_a::test_b']
    (verify,) = _results_of(project, RunKind.VERIFY)
    assert verify.results[0].outcome is CriterionOutcome.PASSED


def test_a_run_whose_junit_has_no_map_records_no_criteria_results(project: Project) -> None:
    _declare_criteria(project)

    run_pass(project.policy, project.open(), now=NOW)

    assert [event.type for event in _criteria_events(project)] == ['criteria_declared']


def test_a_staging_file_is_published_once_as_the_staging_results(project: Project) -> None:
    _declare_criteria(project)
    write_staging(project.out_dir, STAGED)
    out = project.open()

    run_pass(project.policy, out, now=NOW)
    run_pass(project.policy, out, now=NOW)

    (staging,) = _results_of(project, RunKind.STAGING)
    assert (staging.commit, staging.run_at, staging.results) == (
        STAGED.commit,
        STAGED.run_at,
        STAGED.results,
    )
    assert_valid(project)


def test_an_invalid_criteria_file_is_named_criteria_and_every_other_source_still_publishes(
    project: Project,
) -> None:
    _declare_criteria(project, text='format: 2\ncriteria: []\n')
    _map_beside_the_archived_junit(project)

    report = run_pass(project.policy, project.open(), now=NOW)

    assert [degradation.source for degradation in report.degraded] == [Source.CRITERIA]
    assert CRITERIA_RELPATH in report.degraded[0].reason
    assert last_heartbeat(project).degraded == report.degraded
    assert _criteria_events(project) == []
    assert [line.type for line in feed(project)] == FIRST_PASS_TYPES


def test_a_malformed_staging_file_is_named_criteria_and_the_verify_criteria_still_publish(
    project: Project,
) -> None:
    _declare_criteria(project)
    _map_beside_the_archived_junit(project)
    (project.out_dir / STAGING_FILENAME).write_text('not json')

    report = run_pass(project.policy, project.open(), now=NOW)

    assert [degradation.source for degradation in report.degraded] == [Source.CRITERIA]
    assert STAGING_FILENAME in report.degraded[0].reason
    assert [event.type for event in _criteria_events(project)] == [
        'criteria_declared',
        'criteria_results',
    ]
    assert _results_of(project, RunKind.STAGING) == []


def test_of_two_tested_runs_in_one_pass_only_the_newest_is_joined(project: Project) -> None:
    _declare_criteria(project)
    _map_beside_the_archived_junit(project)
    _task_9_verify_run_with_a_map(project)

    run_pass(project.policy, project.open(), now=NOW)

    (verify,) = _results_of(project, RunKind.VERIFY)
    assert (verify.run_at, verify.results) == (at(50), TASK_9_RESULTS)
    assert_valid(project)


def test_a_malformed_archived_map_is_named_criteria_and_the_declaration_still_publishes(
    project: Project,
) -> None:
    _declare_criteria(project)
    (junit,) = task_dir(project.root).glob('*.junit-*.xml.gz')
    collection_map_path(junit).write_text('[]')

    report = run_pass(project.policy, project.open(), now=NOW)

    assert [degradation.source for degradation in report.degraded] == [Source.CRITERIA]
    assert 'invalid_map' in report.degraded[0].reason
    assert [event.type for event in _criteria_events(project)] == ['criteria_declared']
    assert_valid(project)


def test_a_new_verify_run_joins_the_states_declaration_while_the_criteria_file_is_invalid(
    project: Project,
) -> None:
    _declare_criteria(project)
    _map_beside_the_archived_junit(project)
    out = project.open()
    run_pass(project.policy, out, now=NOW)
    first = len(feed(project))
    project.repo.write(CRITERIA_RELPATH, 'format: 2\ncriteria: []\n')
    project.repo.commit('Break the acceptance criteria', when=at(39))
    _task_9_verify_run_with_a_map(project)

    report = run_pass(project.policy, out, now=NOW)

    assert [degradation.source for degradation in report.degraded] == [Source.CRITERIA]
    assert CRITERIA_RELPATH in report.degraded[0].reason
    (verify,) = [
        line for line in feed(project)[first:] if isinstance(line, CriteriaDeclared | CriteriaResults)
    ]
    assert isinstance(verify, CriteriaResults)
    assert (verify.run_kind, verify.run_at, verify.results) == (RunKind.VERIFY, at(50), TASK_9_RESULTS)
    assert_valid(project)
