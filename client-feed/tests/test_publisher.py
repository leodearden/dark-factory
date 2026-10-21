"""One publisher pass: every source read as its own unit, the events built in a fixed order, a heartbeat last."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _archive_files import junit_xml, passed, red, write_junit, write_plan, write_summary
from _git_repo import Repo, make_repo
from _stores import RunsDb, TasksDb

from client_feed.contract import (
    BranchSnapshot,
    FactoryState,
    FeedLine,
    Heartbeat,
    MergeLanded,
    PrdUpserted,
    ReviewCounts,
    SchedulerState,
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
from client_feed.feed_io import CURSOR_FILENAME, Cursor, OutDir, read_feed_lines
from client_feed.fold import fold
from client_feed.policy import Policy
from client_feed.publisher import PassReport, run_pass
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
    """A fixture root as the publisher meets it: main, task branches, the two stores, a verify archive."""

    repo: Repo
    tasks: TasksDb
    runs: RunsDb
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
    runs.emit('escalation_created', at=at(25), task_id=9, data={'escalation_id': 'esc-9-1', 'level': 2})
    runs.emit('scheduler_paused', at=at(30))
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    policy = policy_over(repo.root, out_dir, tmp_path / 'in')
    return Project(repo=repo, tasks=tasks, runs=runs, policy=policy, merge=merge, last_stream_id=landed)


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
    ) == (NOW, project.repo.rev('main'), NOW, None, 0, ())


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
