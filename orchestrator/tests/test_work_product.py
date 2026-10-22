"""work_product: the one-git-child task-branch probe and the continuity predicate
(task 3820; plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C3).
"""

from __future__ import annotations

import asyncio
import dataclasses
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from _call_recorder import BlockingCall, ThreadRecordingCall
from _git_fixtures import README_SEED, _git, build_repo

from orchestrator.sentinel_l1 import SentinelL1
from orchestrator.work_product import (
    FAILURE_STREAK_ESCALATION_THRESHOLD,
    PROBE_SENTINEL,
    BranchProbeError,
    BranchWork,
    WorkProductProbe,
    WorkProductSnapshot,
    parse_branch_work,
    read_branch_work,
)

MAIN_TIP_AT = 1_760_000_000
TASK_TIP_AT = 1_760_003_600


def _committer_date(monkeypatch: pytest.MonkeyPatch, epoch: int) -> None:
    monkeypatch.setenv('GIT_COMMITTER_DATE', f'{epoch} +0000')


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _committer_date(monkeypatch, MAIN_TIP_AT)
    return build_repo(tmp_path / 'repo', README_SEED)


def _branch_with_commits(
    repo: Path, monkeypatch: pytest.MonkeyPatch, branch: str, *, commits: int, at: int = TASK_TIP_AT,
) -> None:
    _committer_date(monkeypatch, at)
    _git(repo, 'checkout', '-q', '-b', branch, 'main')
    for n in range(commits):
        _git(repo, 'commit', '-q', '--allow-empty', '-m', f'{branch} work {n}')
    _git(repo, 'checkout', '-q', 'main')


def _read(project_root: Path, **kwargs) -> dict[str, BranchWork]:
    return read_branch_work(project_root, main_branch='main', branch_prefix='task/', **kwargs)


class TestReadBranchWork:
    def test_branches_are_keyed_by_task_id_with_commits_ahead_and_tip_time(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/7', commits=2)
        _git(repo, 'branch', 'task/8', 'main')

        assert _read(repo) == {
            '7': BranchWork(commits_ahead=2, tip_committed_at=TASK_TIP_AT),
            '8': BranchWork(commits_ahead=0, tip_committed_at=MAIN_TIP_AT),
        }

    def test_a_same_named_tag_does_not_change_the_keys(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/7', commits=1)
        _git(repo, 'tag', 'task/7', 'main')

        assert set(_read(repo)) == {'7'}

    def test_suffixed_and_nested_branches_keep_their_whole_suffix(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/9-retry', commits=1)
        _branch_with_commits(repo, monkeypatch, 'task/nested/4', commits=1)

        assert set(_read(repo)) == {'9-retry', 'nested/4'}

    def test_a_repo_without_task_branches_reads_as_empty(self, repo):
        assert _read(repo) == {}

    def test_a_missing_main_branch_is_a_probe_error_quoting_git(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/7', commits=1)

        with pytest.raises(BranchProbeError) as excinfo:
            read_branch_work(repo, main_branch='trunk', branch_prefix='task/')

        assert excinfo.value.returncode == 128
        rendered = str(excinfo.value)
        assert 'failed to find' in rendered
        assert 'for-each-ref' in rendered

    def test_a_subdirectory_of_a_repo_is_not_read_as_that_repo(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/7', commits=1)
        subdir = repo / 'sub'
        subdir.mkdir()

        with pytest.raises(BranchProbeError):
            _read(subdir)

    def test_one_read_forks_exactly_one_ceilinged_for_each_ref(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/7', commits=1)
        calls: list[tuple[tuple, dict]] = []

        def counting_run(*args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.run(*args, **kwargs)

        _read(repo, run=counting_run)

        assert len(calls) == 1
        args, kwargs = calls[0]
        argv = list(args[0] if args else kwargs['args'])
        assert argv[0] == 'git'
        assert 'for-each-ref' in argv
        assert any('%(ahead-behind:main)' in arg for arg in argv)
        assert 'refs/heads/task/' in argv
        assert kwargs['env']['GIT_CEILING_DIRECTORIES'] == str(repo.resolve().parent)

    def test_a_timed_out_git_is_a_probe_error(self, repo):
        def hung_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0] if args else kwargs['args'], timeout=60.0)

        with pytest.raises(BranchProbeError, match='timed out'):
            _read(repo, run=hung_run)

    def test_a_missing_git_binary_is_a_probe_error(self, repo):
        def no_git(*args, **kwargs):
            raise FileNotFoundError(2, 'No such file or directory', 'git')

        with pytest.raises(BranchProbeError):
            _read(repo, run=no_git)


class TestParseBranchWork:
    def test_blank_lines_are_ignored(self):
        stdout = (
            'refs/heads/task/7 2 0 1760003600\n'
            '\n'
            '   \n'
            'refs/heads/task/8 0 3 1760000000\n'
        )

        assert parse_branch_work(stdout, branch_prefix='task/') == {
            '7': BranchWork(commits_ahead=2, tip_committed_at=1760003600),
            '8': BranchWork(commits_ahead=0, tip_committed_at=1760000000),
        }

    @pytest.mark.parametrize(
        'line',
        [
            'refs/heads/task/7 x 0 1760003600',
            'refs/heads/task/7 2 0',
            'refs/heads/task/7 2 0 1760003600 extra',
            'refs/heads/feature/7 2 0 1760003600',
            'refs/tags/task/7 2 0 1760003600',
        ],
    )
    def test_an_unparseable_line_is_a_probe_error_naming_it(self, line):
        with pytest.raises(BranchProbeError) as excinfo:
            parse_branch_work(f'refs/heads/task/1 1 0 1760000000\n{line}\n', branch_prefix='task/')

        assert line in str(excinfo.value)


NOW = 1_760_100_000.0
MAX_AGE = 604800.0


def _snapshot(
    *, ahead: int = 2, tip: float = NOW - 3600, zero_progress: Iterable[str] = (),
) -> WorkProductSnapshot:
    return WorkProductSnapshot(
        branches={'10': BranchWork(commits_ahead=ahead, tip_committed_at=tip)},
        zero_progress=frozenset(zero_progress),
    )


def _credited(snapshot: WorkProductSnapshot, task_id: str = '10') -> bool:
    return snapshot.has_work_product(task_id, now=NOW, max_age_secs=MAX_AGE)


class TestHasWorkProduct:
    def test_a_fresh_branch_ahead_of_main_is_work_product(self):
        assert _credited(_snapshot(ahead=2))

    def test_a_branch_with_nothing_ahead_is_not(self):
        assert not _credited(_snapshot(ahead=0))

    def test_a_task_without_a_branch_is_not(self):
        assert not _credited(_snapshot(), task_id='11')

    def test_the_zero_progress_gate_wins_over_fresh_work(self):
        assert not _credited(_snapshot(zero_progress={'10'}))

    def test_another_tasks_zero_progress_does_not_gate_this_one(self):
        assert _credited(_snapshot(zero_progress={'99'}))

    def test_credit_lapses_only_after_the_max_age(self):
        assert _credited(_snapshot(tip=NOW - MAX_AGE))
        assert not _credited(_snapshot(tip=NOW - MAX_AGE - 1))

    def test_a_tip_in_the_future_is_credited(self):
        assert _credited(_snapshot(tip=NOW + 600))

    @pytest.mark.parametrize('task_id', ['10', '11', ''])
    def test_the_empty_snapshot_credits_nothing(self, task_id):
        assert not _credited(WorkProductSnapshot.EMPTY, task_id=task_id)

    @pytest.mark.parametrize('field_name', ['branches', 'zero_progress'])
    def test_the_snapshot_is_immutable(self, field_name):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(_snapshot(), field_name, frozenset({'10'}))


TTL = 300.0
FRESH = WorkProductSnapshot(
    branches={'10': BranchWork(commits_ahead=1, tip_committed_at=NOW)}, zero_progress=frozenset(),
)
PROBE_FAILURE = BranchProbeError(
    reason='git for-each-ref exited 128',
    cmd=('git', '-C', '/srv/project', 'for-each-ref', 'refs/heads/task/'),
    cwd='/srv/project',
    returncode=128,
    stderr="fatal: failed to find 'main'",
)


def _counting_load() -> ThreadRecordingCall[WorkProductSnapshot]:
    return ThreadRecordingCall(FRESH)


def _failing(exc: Exception = PROBE_FAILURE) -> Callable[[], WorkProductSnapshot]:
    def load() -> WorkProductSnapshot:
        raise exc
    return load


class _Rig:
    """A probe on a hand-wound monotonic clock and a hot-reloadable TTL."""

    def __init__(self) -> None:
        self.clock = [0.0]
        self.ttl = TTL
        self.filed: list[SentinelL1] = []
        self.resolved: list[str] = []
        self.probe = WorkProductProbe(
            ttl_secs=lambda: self.ttl,
            clock=lambda: self.clock[0],
            escalate=self.filed.append,
            resolve=self._record_resolution,
        )

    def _record_resolution(self, sentinel: str, resolution: str) -> None:
        assert resolution
        self.resolved.append(sentinel)

    async def refresh(self, load: Callable[[], WorkProductSnapshot]) -> None:
        self.probe.poll(load)
        await self.probe.join()

    async def refresh_next_window(self, load: Callable[[], WorkProductSnapshot]) -> None:
        self.clock[0] += self.ttl
        await self.refresh(load)


@pytest.mark.asyncio
class TestWorkProductProbe:
    async def test_nothing_is_credited_before_the_first_refresh(self):
        assert _Rig().probe.snapshot is WorkProductSnapshot.EMPTY

    async def test_a_refresh_publishes_the_loaded_snapshot(self):
        rig, load = _Rig(), _counting_load()

        await rig.refresh(load)

        assert load.calls == 1
        assert rig.probe.snapshot is FRESH

    async def test_at_most_one_load_per_ttl_window(self):
        rig, load = _Rig(), _counting_load()
        await rig.refresh(load)

        for _ in range(50):
            await rig.refresh(load)
        rig.clock[0] = TTL - 1
        await rig.refresh(load)
        assert load.calls == 1

        rig.clock[0] = TTL
        await rig.refresh(load)
        assert load.calls == 2

    async def test_the_window_is_anchored_at_the_attempt_not_the_completion(self):
        rig, counting = _Rig(), _counting_load()

        def slow_load() -> WorkProductSnapshot:
            rig.clock[0] += 100
            return FRESH

        await rig.refresh(slow_load)
        rig.clock[0] = TTL
        await rig.refresh(counting)

        assert counting.calls == 1

    async def test_the_load_runs_off_the_event_loop_thread(self):
        rig, load = _Rig(), _counting_load()

        await rig.refresh(load)

        assert load.threads and load.threads[0] != threading.get_ident()

    async def test_poll_never_waits_and_never_overlaps_refreshes(self):
        rig, counting, load = _Rig(), _counting_load(), BlockingCall(FRESH)

        try:
            started_at = time.monotonic()
            rig.probe.poll(load)
            assert time.monotonic() - started_at < 1.0
            await load.entered()
            assert rig.probe.snapshot is WorkProductSnapshot.EMPTY

            rig.clock[0] = TTL + 1
            rig.probe.poll(counting)
            assert load.blocked
        finally:
            load.release()
        await rig.probe.join()

        assert rig.probe.snapshot is FRESH
        assert counting.calls == 0

    async def test_a_failed_refresh_withdraws_all_credit(self):
        rig = _Rig()
        await rig.refresh(_counting_load())

        await rig.refresh_next_window(_failing())

        assert rig.probe.snapshot is WorkProductSnapshot.EMPTY

    async def test_a_failure_streak_escalates_once_it_reaches_the_threshold(self):
        rig = _Rig()
        await rig.refresh(_failing())
        for _ in range(FAILURE_STREAK_ESCALATION_THRESHOLD - 2):
            await rig.refresh_next_window(_failing())
        assert rig.filed == []

        await rig.refresh_next_window(_failing())

        assert len(rig.filed) == 1
        l1 = rig.filed[0]
        assert l1.sentinel == PROBE_SENTINEL
        assert l1.category == 'infra_issue'
        assert f'{FAILURE_STREAK_ESCALATION_THRESHOLD} consecutive' in l1.detail
        assert f'{TTL:g}' in l1.detail
        assert str(PROBE_FAILURE) in l1.detail

    async def test_each_further_failure_re_files_for_the_queue_to_dedup(self):
        rig = _Rig()
        await rig.refresh(_failing())
        for _ in range(FAILURE_STREAK_ESCALATION_THRESHOLD):
            await rig.refresh_next_window(_failing())

        assert len(rig.filed) == 2

    async def test_one_success_resets_the_streak(self):
        rig = _Rig()
        await rig.refresh(_failing())
        for _ in range(FAILURE_STREAK_ESCALATION_THRESHOLD - 2):
            await rig.refresh_next_window(_failing())
        await rig.refresh_next_window(_counting_load())
        for _ in range(FAILURE_STREAK_ESCALATION_THRESHOLD - 1):
            await rig.refresh_next_window(_failing())

        assert rig.filed == []

    async def test_an_unexpected_load_error_is_a_counted_failure_that_never_propagates(self):
        rig = _Rig()
        boom = _failing(RuntimeError('sqlite went away'))
        await rig.refresh(boom)
        for _ in range(FAILURE_STREAK_ESCALATION_THRESHOLD - 1):
            await rig.refresh_next_window(boom)

        assert rig.probe.snapshot is WorkProductSnapshot.EMPTY
        assert len(rig.filed) == 1
        assert 'sqlite went away' in rig.filed[0].detail

    async def test_a_hot_ttl_retune_moves_the_window_on_the_next_poll(self):
        rig, load = _Rig(), _counting_load()
        await rig.refresh(load)

        rig.ttl = 60.0
        rig.clock[0] = 60.0
        await rig.refresh(load)

        assert load.calls == 2


@pytest.mark.asyncio
class TestWorkProductProbeClose:
    async def test_close_cancels_an_in_flight_refresh_without_waiting_for_git(self):
        rig, load = _Rig(), BlockingCall(FRESH)

        try:
            rig.probe.poll(load)
            await load.entered()
            await asyncio.wait_for(rig.probe.aclose(), timeout=5)
            assert load.blocked
        finally:
            load.release()

        assert rig.probe.snapshot is WorkProductSnapshot.EMPTY
        assert rig.filed == []

    async def test_closing_before_the_refresh_first_runs_leaves_nothing_to_join(self):
        rig, load = _Rig(), _counting_load()
        rig.probe.poll(load)

        await rig.probe.aclose()
        await rig.probe.join()

        assert load.calls == 0

    async def test_a_refresh_cancelled_before_it_first_runs_does_not_wedge_the_next_window(self):
        rig, load = _Rig(), _counting_load()
        rig.probe.poll(load)
        current = asyncio.current_task()
        stragglers = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        assert stragglers, 'poll started no refresh for the sweep to cancel'

        for straggler in stragglers:
            straggler.cancel()
        await asyncio.wait(stragglers)
        await rig.refresh_next_window(load)

        assert load.calls == 1

    async def test_a_closed_probe_starts_no_further_refresh(self):
        rig, load = _Rig(), _counting_load()
        await rig.probe.aclose()

        rig.clock[0] = 10 * TTL
        await rig.refresh(load)

        assert load.calls == 0

    async def test_close_keeps_the_last_published_snapshot(self):
        rig = _Rig()
        await rig.refresh(_counting_load())

        await rig.probe.aclose()

        assert rig.probe.snapshot is FRESH


@pytest.mark.asyncio
class TestWorkProductProbeRecovery:
    """The probe's L1 is resolved by the first refresh that succeeds after it may have been filed."""

    async def test_a_new_probe_resolves_an_escalation_a_previous_process_may_have_left(self):
        rig = _Rig()

        await rig.refresh(_counting_load())
        await rig.refresh_next_window(_counting_load())

        assert rig.resolved == [PROBE_SENTINEL]

    async def test_the_first_success_after_a_filed_streak_resolves_it(self):
        rig = _Rig()
        await rig.refresh(_counting_load())
        for _ in range(FAILURE_STREAK_ESCALATION_THRESHOLD):
            await rig.refresh_next_window(_failing())
        assert len(rig.filed) == 1

        await rig.refresh_next_window(_counting_load())
        await rig.refresh_next_window(_counting_load())

        assert rig.resolved == [PROBE_SENTINEL, PROBE_SENTINEL]

    async def test_a_streak_too_short_to_file_resolves_nothing(self):
        rig = _Rig()
        await rig.refresh(_counting_load())
        for _ in range(FAILURE_STREAK_ESCALATION_THRESHOLD - 1):
            await rig.refresh_next_window(_failing())

        await rig.refresh_next_window(_counting_load())

        assert rig.resolved == [PROBE_SENTINEL]
