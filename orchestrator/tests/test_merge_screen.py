"""The merge pre-verify screen, through ``orchestrator.merge_lane.screen``'s public seams.

plans/merge-pre-verify-screen-prd.md decisions 4-8: the screen runs the
manifest's node ids on the merged tree in a warm lane, attributes a finding
base-first, bounces only the item's own defect, and fails open, loudly, on
every instrument fault.

The repo is real git and the lanes are real worktrees. Most cases script the
node-id runner from the checked-out tree's content; the own-defect and
red-base cases use the real pytest runner.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import subprocess
import sys
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import RepoSeed, _git, seed_repo
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT, git_env_with_ceiling
from _verify_artifact_fixtures import registry_config
from escalation.categories import EscalationCategory

from orchestrator.config import GitConfig, MergeScreenConfig, ModuleConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import (
    EphemeralWorktreeError,
    GitOps,
    ScreenLane,
    ScreenLaneUnavailable,
    WorktreeKind,
)
from orchestrator.merge_lane.outcome_facts import MergeBlockCategory
from orchestrator.merge_lane.screen import (
    MERGE_SCREEN_ALARM_TASK_ID,
    Bounce,
    MergeScreen,
    Proceed,
    ScreenSite,
    run_screen,
    screen_or_proceed,
)
from orchestrator.merge_lane.types import MergeOutcome
from orchestrator.verify import NodeIdResult, NodeIdStatus
from orchestrator.verify_cancel import lane_lock_path

pytestmark = pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)

MANIFEST = 'ci/screen.yaml'
CLEAN_ID = 'pkg/tests/test_screen_probe.py::test_no_defect'
OTHER_ID = 'pkg/tests/test_screen_probe.py::test_always_passes'
DEFECT_TEXT = 'DEFECT:1 unowned suppression'

_PROBE = f'''\
from pathlib import Path


def test_no_defect():
    assert not (Path(__file__).resolve().parents[2] / 'DEFECT').exists(), {DEFECT_TEXT!r}


def test_always_passes():
    pass
'''

_MANIFEST_TEXT = f'''\
checks:
  - id: clean
    module: pkg
    node_ids: [{CLEAN_ID}]
  - id: other
    module: pkg
    node_ids: [{OTHER_ID}]
'''

_SEED = RepoSeed(
    files=(
        ('pkg/pytest.ini', '[pytest]\n'),
        ('pkg/tests/test_screen_probe.py', _PROBE),
        (MANIFEST, _MANIFEST_TEXT),
    ),
    message='base: clean',
)


def _head(cwd: Path) -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=cwd, env=git_env_with_ceiling(cwd),
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit_file(repo: Path, relpath: str, text: str, message: str) -> str:
    (repo / relpath).write_text(text, encoding='utf-8')
    _git(repo, 'add', relpath)
    _git(repo, 'commit', '-m', message)
    return _head(repo)


@dataclass(frozen=True)
class History:
    """base is clean; item and item2 each add the defect on base; on_red adds a file on item."""

    base: str
    item: str
    item2: str
    on_red: str


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo', _SEED)


@pytest.fixture
def history(repo: Path) -> History:
    base = _head(repo)
    item = _commit_file(repo, 'DEFECT', 'defect\n', 'item: add defect')
    on_red = _commit_file(repo, 'unrelated.txt', 'x\n', 'on red: unrelated')
    _git(repo, 'checkout', '-q', '-b', 'second', base)
    _commit_file(repo, 'DEFECT', 'defect\n', 'item2: add defect')
    item2 = _commit_file(repo, 'extra.txt', 'y\n', 'item2: extra')
    _git(repo, 'checkout', '-q', '-')
    return History(base=base, item=item, item2=item2, on_red=on_red)


@pytest.fixture
def git_ops(repo: Path) -> GitOps:
    config = GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )
    return GitOps(config, repo)


def _module_config(prefix: str) -> ModuleConfig:
    return ModuleConfig(prefix=prefix, test_command='pytest', lint_command=None, type_check_command=None)


def _config(
    repo: Path, *, registry: Sequence[str] = ('pkg',), **screen: Any,
) -> OrchestratorConfig:
    return registry_config(
        [_module_config(prefix) for prefix in registry],
        project_root=repo,
        max_concurrent_tasks=1,
        git=GitConfig(main_branch='main', branch_prefix='task/', remote='origin', worktree_dir='.worktrees'),
        merge_verify_breadth='full',
        verify_admission_enabled=False,
        verify_cold_preprovision_command='',
        verify_env={'PATH': f'{Path(sys.executable).parent}{os.pathsep}{os.environ["PATH"]}'},
        merge_screen=MergeScreenConfig(**{'manifest': MANIFEST, **screen}),
    )


@dataclass(frozen=True)
class RunnerCall:
    tree: Path
    sha: str
    ids: tuple[str, ...]
    prefixes: tuple[str, ...]
    concurrent_groups: bool


def _defect_fails(tree: Path, node_id: str) -> NodeIdResult:
    if node_id == CLEAN_ID and (tree / 'DEFECT').exists():
        return NodeIdResult('fail', f'AssertionError: {DEFECT_TEXT}')
    return NodeIdResult('pass')


@dataclass
class ScriptedRunner:
    """A node-id runner that answers from the checked-out tree, recording each call."""

    verdict: Any = _defect_fails
    hangs: bool = False
    calls: list[RunnerCall] = field(default_factory=list)

    async def __call__(
        self, tree: Path, config: OrchestratorConfig, module_configs: Sequence[ModuleConfig],
        ids: Collection[str], *, concurrent_groups: bool = False,
    ) -> Mapping[str, NodeIdResult]:
        sha = _head(tree) if (tree / '.git').exists() else ''
        self.calls.append(RunnerCall(
            tree, sha, tuple(ids), tuple(mc.prefix for mc in module_configs), concurrent_groups,
        ))
        if self.hangs:
            await asyncio.Event().wait()
        return {node_id: self.verdict(tree, node_id) for node_id in ids}

    def runs_at(self, sha: str) -> list[RunnerCall]:
        return [call for call in self.calls if call.sha == sha]


@dataclass
class RecordingEvents:
    emitted: list[tuple[EventType, str | None, dict[str, Any]]] = field(default_factory=list)

    def emit(self, event_type: EventType, *, task_id: str | None = None, data: dict | None = None, **_: Any) -> int:
        self.emitted.append((event_type, task_id, dict(data or {})))
        return len(self.emitted)

    def screens(self) -> list[dict[str, Any]]:
        return [data for event_type, _, data in self.emitted if event_type is EventType.merge_screen]


@dataclass
class RecordingEscalationQueue:
    """The escalation-queue surface the screen's alarm uses: file, dedupe, resolve."""

    filed: list[Any] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)

    def make_id(self, key: str) -> str:
        return f'esc-{key}-{len(self.filed) + 1}'

    def submit(self, escalation: Any) -> None:
        self.filed.append(escalation)

    def has_open_l1(self, task_id: str, *, category: str | None = None) -> bool:
        return bool(self.get_by_task(task_id, status='pending', level=1))

    def get_by_task(self, task_id: str, status: str | None = None, level: int | None = None) -> list[Any]:
        return [
            esc for esc in self.filed
            if esc.task_id == task_id
            and (status is None or esc.status == status)
            and (level is None or esc.level == level)
        ]

    def resolve(self, escalation_id: str, resolution: str, **_: Any) -> Any:
        for esc in self.filed:
            if esc.id == escalation_id:
                esc.status = 'resolved'
                self.resolved.append(escalation_id)
                return esc
        return None


def _screen(
    git_ops: Any, *, run_ids: Any = None, escalations: Any = None,
) -> tuple[MergeScreen, RecordingEvents]:
    events = RecordingEvents()
    kwargs = {} if run_ids is None else {'run_ids': run_ids}
    screen = MergeScreen(git_ops, events, escalations or RecordingEscalationQueue(), **kwargs)
    return screen, events


async def _screen_item(
    screen: MergeScreen, config: OrchestratorConfig, merged: str, base: str | None, task_id: str = '42',
) -> Any:
    return await screen.screen_item(
        task_id=task_id, merged_sha=merged, base_sha=base, config=config, site=ScreenSite.MERGER,
    )


@pytest.fixture
def held_merger_lane(git_ops: GitOps) -> Iterator[None]:
    lock = lane_lock_path(git_ops.worktree_base / ScreenLane.MERGER.value)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


@dataclass
class _RaisingEntry:
    """An async context manager whose entry records *label* and raises *error*."""

    error: BaseException
    label: str
    entered: list[str]

    async def __aenter__(self) -> Path:
        self.entered.append(self.label)
        raise self.error

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _UnenterableTrees:
    """A GitOps stand-in whose lane and fallback both raise on entry, recording each attempt."""

    def __init__(self, *, lane_error: BaseException, fallback_error: BaseException) -> None:
        self.lane_error = lane_error
        self.fallback_error = fallback_error
        self.entered: list[str] = []

    def screen_lane(self, lane: ScreenLane, sha: str, **_: Any) -> _RaisingEntry:
        return _RaisingEntry(self.lane_error, f'lane:{sha}', self.entered)

    def ephemeral_worktree(self, kind: WorktreeKind, sha: str, **_: Any) -> _RaisingEntry:
        return _RaisingEntry(self.fallback_error, f'fallback:{sha}', self.entered)


# ---------------------------------------------------------------------------
# (a) (b) attribution with the real runner


@pytest.mark.asyncio
async def test_own_defect_bounces(repo: Path, history: History, git_ops: GitOps) -> None:
    screen, events = _screen(git_ops)

    decision = await _screen_item(screen, _config(repo), history.item, history.base)

    assert isinstance(decision, Bounce)
    outcome = decision.outcome
    assert outcome.status == 'blocked'
    assert outcome.failure_category == 'merge_screen'
    assert outcome.block_category is MergeBlockCategory.MERGE_SCREEN
    assert outcome.failure_cause_hint == 'clean'
    assert decision.check_ids == ('clean',)
    assert 'clean' in outcome.reason
    assert 'DEFECT:1' in outcome.reason
    [event] = events.screens()
    assert (event['status'], event['lane'], event['site']) == ('bounce', 'warm', 'merger')
    assert event['merged_sha'] == history.item
    assert event['base_sha'] == history.base
    assert {check['id']: check['status'] for check in event['checks']} == {
        'clean': 'finding', 'other': 'pass',
    }
    assert [task_id for _, task_id, _ in events.emitted] == ['42']


@pytest.mark.asyncio
async def test_red_base_is_not_blamed(repo: Path, history: History, git_ops: GitOps) -> None:
    screen, events = _screen(git_ops)

    decision = await _screen_item(screen, _config(repo), history.on_red, history.item)

    assert isinstance(decision, Proceed)
    [event] = events.screens()
    assert event['status'] == 'base_red'


# ---------------------------------------------------------------------------
# (c) base verdict cache


@pytest.mark.asyncio
async def test_base_verdict_is_cached_per_sha(repo: Path, history: History, git_ops: GitOps) -> None:
    runner = ScriptedRunner()
    screen, events = _screen(git_ops, run_ids=runner)
    config = _config(repo)

    first = await _screen_item(screen, config, history.item, history.base)
    second = await _screen_item(screen, config, history.item2, history.base, task_id='43')

    assert isinstance(first, Bounce) and isinstance(second, Bounce)
    [base_run] = runner.runs_at(history.base)
    assert base_run.ids == (CLEAN_ID,)
    assert len(runner.calls) == 3
    assert [event['status'] for event in events.screens()] == ['bounce', 'bounce']


# ---------------------------------------------------------------------------
# (d) instrument errors fail open


def _status_for_clean(status: NodeIdStatus) -> Any:
    def verdict(tree: Path, node_id: str) -> NodeIdResult:
        return NodeIdResult(status, 'probe') if node_id == CLEAN_ID else NodeIdResult('pass')
    return verdict


def _red_item_erroring_base(base_tree_marker: str) -> Any:
    def verdict(tree: Path, node_id: str) -> NodeIdResult:
        if node_id == CLEAN_ID and not (tree / 'DEFECT').exists():
            return NodeIdResult('error', base_tree_marker)
        return _defect_fails(tree, node_id)
    return verdict


async def _assert_fails_open(
    screen: MergeScreen, events: RecordingEvents, config: OrchestratorConfig, history: History,
    *, cause: str,
) -> None:
    decision = await _screen_item(screen, config, history.item, history.base)
    assert isinstance(decision, Proceed)
    [event] = events.screens()
    assert event['status'] == 'error'
    assert any(cause in text for text in event['errors']), event['errors']


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['error', 'skipped'])
async def test_a_non_verdict_id_is_an_error_not_a_pass(
    repo: Path, history: History, git_ops: GitOps, status: NodeIdStatus,
) -> None:
    screen, events = _screen(git_ops, run_ids=ScriptedRunner(verdict=_status_for_clean(status)))
    await _assert_fails_open(screen, events, _config(repo), history, cause='clean')


@pytest.mark.asyncio
async def test_a_manifest_missing_from_the_tree_fails_open(
    repo: Path, history: History, git_ops: GitOps,
) -> None:
    runner = ScriptedRunner()
    screen, events = _screen(git_ops, run_ids=runner)
    config = _config(repo, manifest='ci/absent.yaml')
    await _assert_fails_open(screen, events, config, history, cause='ci/absent.yaml')
    assert runner.calls == []


@pytest.mark.asyncio
async def test_an_unregistered_manifest_module_fails_open(
    repo: Path, history: History, git_ops: GitOps,
) -> None:
    screen, events = _screen(git_ops, run_ids=ScriptedRunner())
    config = _config(repo, registry=('elsewhere',))
    await _assert_fails_open(screen, events, config, history, cause='not a registered module')


@pytest.mark.asyncio
async def test_a_runner_past_the_timeout_fails_open(
    repo: Path, history: History, git_ops: GitOps,
) -> None:
    screen, events = _screen(git_ops, run_ids=ScriptedRunner(hangs=True))
    config = _config(repo, timeout_secs=1.0)
    await _assert_fails_open(screen, events, config, history, cause='timed out')


@pytest.mark.asyncio
async def test_a_finding_with_an_erroring_base_fails_open(
    repo: Path, history: History, git_ops: GitOps,
) -> None:
    runner = ScriptedRunner(verdict=_red_item_erroring_base('base probe broke'))
    screen, events = _screen(git_ops, run_ids=runner)
    await _assert_fails_open(screen, events, _config(repo), history, cause='clean')
    assert runner.runs_at(history.base), 'the base re-run happened and errored'


@pytest.mark.asyncio
async def test_an_unexpected_exception_fails_open(repo: Path, history: History, git_ops: GitOps) -> None:
    def explode(tree: Path, node_id: str) -> NodeIdResult:
        raise ValueError('runner exploded')

    screen, events = _screen(git_ops, run_ids=ScriptedRunner(verdict=explode))
    await _assert_fails_open(screen, events, _config(repo), history, cause='runner exploded')


# ---------------------------------------------------------------------------
# (e) error streak alarm


@pytest.mark.asyncio
async def test_error_streak_files_one_l1_and_a_clean_screen_closes_it(
    repo: Path, history: History, git_ops: GitOps,
) -> None:
    erroring = ScriptedRunner(verdict=_status_for_clean('error'))
    escalations = RecordingEscalationQueue()
    screen, events = _screen(git_ops, run_ids=erroring, escalations=escalations)
    config = _config(repo, error_streak_alarm=3)

    for _ in range(3):
        await _screen_item(screen, config, history.item, history.base)

    [alarm] = escalations.filed
    assert alarm.level == 1
    assert alarm.severity == 'info'
    assert alarm.category == EscalationCategory.MERGE_SCREEN_UNAVAILABLE.value
    assert alarm.task_id == MERGE_SCREEN_ALARM_TASK_ID
    assert '3' in alarm.summary and 'clean' in alarm.summary
    assert 'merge_screen.manifest' in alarm.suggested_action

    await _screen_item(screen, config, history.item, history.base)
    assert len(escalations.filed) == 1

    erroring.verdict = _defect_fails
    await _screen_item(screen, config, history.base, history.base)
    assert escalations.resolved == [alarm.id]

    erroring.verdict = _status_for_clean('error')
    await _screen_item(screen, config, history.item, history.base)
    assert [event['error_streak'] for event in events.screens()] == [1, 2, 3, 4, 0, 1]


# ---------------------------------------------------------------------------
# (f) kill switch


@pytest.mark.asyncio
async def test_kill_switch_runs_no_screen_code(repo: Path, history: History) -> None:
    trees = _UnenterableTrees(
        lane_error=AssertionError('lane entered'), fallback_error=AssertionError('fallback entered'),
    )
    runner = ScriptedRunner()
    screen, events = _screen(trees, run_ids=runner)
    config = _config(repo, manifest=None)

    decision = await _screen_item(screen, config, history.item, history.base)

    assert isinstance(decision, Proceed)
    assert (runner.calls, trees.entered, events.emitted) == ([], [], [])


# ---------------------------------------------------------------------------
# (g) fallback


@pytest.mark.asyncio
@pytest.mark.usefixtures('held_merger_lane')
async def test_a_held_lane_falls_back_to_an_ephemeral_worktree(
    repo: Path, history: History, git_ops: GitOps,
) -> None:
    runner = ScriptedRunner()
    screen, events = _screen(git_ops, run_ids=runner)

    decision = await _screen_item(screen, _config(repo), history.item, history.base)

    assert isinstance(decision, Bounce)
    assert runner.calls and all(
        call.tree.name.startswith(WorktreeKind.MERGE_SCREEN.value) for call in runner.calls
    )
    [event] = events.screens()
    assert event['lane'] == 'fallback'
    assert ScreenLane.MERGER.value in event['fallback_reason']


@pytest.mark.asyncio
async def test_no_lane_and_no_fallback_fails_open(repo: Path, history: History) -> None:
    trees = _UnenterableTrees(
        lane_error=ScreenLaneUnavailable(ScreenLane.MERGER, repo / 'x.lock', 0.1),
        fallback_error=EphemeralWorktreeError('worktree add failed'),
    )
    runner = ScriptedRunner()
    screen, events = _screen(trees, run_ids=runner)

    decision = await _screen_item(screen, _config(repo), history.item, history.base)

    assert isinstance(decision, Proceed)
    assert runner.calls == []
    [event] = events.screens()
    assert (event['status'], event['lane']) == ('error', 'fallback')
    assert any('worktree add failed' in text for text in event['errors'])


# ---------------------------------------------------------------------------
# (h) run_screen's status mapping

_MAPPING_MANIFEST = '''\
checks:
  - id: finds
    module: pkg
    node_ids: [pkg/t.py::fails, pkg/t.py::passes]
  - id: errs
    module: pkg
    node_ids: [pkg/t.py::errors]
  - id: skips
    module: lib
    node_ids: [lib/t.py::skips]
  - id: vanishes
    module: lib
    node_ids: [lib/t.py::unreported]
  - id: passes
    module: pkg
    node_ids: [pkg/t.py::passes, pkg/t.py::also_passes]
'''

_MAPPING_RESULTS = {
    'pkg/t.py::fails': NodeIdResult('fail', 'assert 1 == 2 at scripts/x.py:3'),
    'pkg/t.py::passes': NodeIdResult('pass'),
    'pkg/t.py::errors': NodeIdResult('error', 'ImportError: no module'),
    'lib/t.py::skips': NodeIdResult('skipped'),
    'pkg/t.py::also_passes': NodeIdResult('pass'),
}


def _mapping_runner() -> ScriptedRunner:
    return ScriptedRunner(verdict=lambda tree, node_id: _MAPPING_RESULTS.get(node_id))


def _drop_unreported(runner: ScriptedRunner) -> Any:
    async def run(*args: Any, **kwargs: Any) -> Mapping[str, NodeIdResult]:
        results = await runner(*args, **kwargs)
        return {node_id: result for node_id, result in results.items() if result is not None}
    return run


@pytest.fixture
def mapping_tree(tmp_path: Path) -> Path:
    tree = tmp_path / 'tree'
    (tree / 'ci').mkdir(parents=True)
    (tree / MANIFEST).write_text(_MAPPING_MANIFEST, encoding='utf-8')
    return tree


@pytest.mark.asyncio
async def test_run_screen_maps_id_statuses_to_check_statuses(mapping_tree: Path, repo: Path) -> None:
    runner = _mapping_runner()
    config = _config(repo, registry=('pkg', 'lib', 'unused'))

    verdict = await run_screen(mapping_tree, config, tree_sha='abc', run_ids=_drop_unreported(runner))

    assert verdict.tree_sha == 'abc'
    statuses = {check.id: check.status for check in verdict.checks}
    assert statuses == {
        'finds': 'finding', 'errs': 'error', 'skips': 'error', 'vanishes': 'error', 'passes': 'pass',
    }
    [finding] = verdict.findings
    assert finding.id == 'finds'
    assert any('scripts/x.py:3' in text for text in finding.findings)
    assert {check.id for check in verdict.errors} == {'errs', 'skips', 'vanishes'}
    [call] = runner.calls
    assert call.concurrent_groups is True
    assert sorted(call.ids) == sorted({*_MAPPING_RESULTS, 'lib/t.py::unreported'})
    assert sorted(call.prefixes) == ['lib', 'pkg']


@pytest.mark.asyncio
async def test_run_screen_only_restricts_to_the_named_checks(mapping_tree: Path, repo: Path) -> None:
    runner = _mapping_runner()
    config = _config(repo, registry=('pkg', 'lib'))

    verdict = await run_screen(
        mapping_tree, config, tree_sha='abc', only=frozenset({'finds'}), run_ids=_drop_unreported(runner),
    )

    assert [check.id for check in verdict.checks] == ['finds']
    [call] = runner.calls
    assert sorted(call.ids) == ['pkg/t.py::fails', 'pkg/t.py::passes']
    assert call.prefixes == ('pkg',)
    assert call.concurrent_groups is True


# ---------------------------------------------------------------------------
# fail-open robustness: absent and raising collaborators, the port-boundary guard

_LANE_LOGGER = 'orchestrator.merge_lane'


def _lane_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _LANE_LOGGER and r.levelno == logging.WARNING]


@pytest.mark.asyncio
async def test_absent_collaborators_still_bounce_an_own_defect(
    repo: Path, history: History, git_ops: GitOps,
) -> None:
    screen = MergeScreen(git_ops, None, None)

    decision = await _screen_item(screen, _config(repo), history.item, history.base)

    assert isinstance(decision, Bounce)
    assert decision.outcome.failure_category == 'merge_screen'
    assert decision.check_ids == ('clean',)


@pytest.mark.asyncio
async def test_absent_collaborators_raise_and_close_no_alarm(
    repo: Path, history: History, git_ops: GitOps,
) -> None:
    runner = ScriptedRunner(verdict=_status_for_clean('error'))
    screen = MergeScreen(git_ops, None, None, run_ids=runner)
    config = _config(repo, error_streak_alarm=1)

    decisions = [await _screen_item(screen, config, history.item, history.base) for _ in range(3)]
    runner.verdict = _defect_fails
    clean = await _screen_item(screen, config, history.base, history.base)

    assert all(isinstance(decision, Proceed) for decision in [*decisions, clean])


class _RaisingEvents:
    def emit(self, event_type: EventType, *, task_id: str | None = None, data: dict | None = None) -> int:
        raise OSError('event store disk full')


class _RaisingEscalationQueue:
    """Every call the alarm makes fails, as the file-backed queue can."""

    def make_id(self, key: str) -> str:
        raise OSError('queue: make_id')

    def submit(self, escalation: Any) -> object:
        raise OSError('queue: submit')

    def has_open_l1(self, task_id: str) -> bool:
        raise OSError('queue: has_open_l1')

    def get_by_task(self, task_id: str, status: str | None = None) -> list[Any]:
        raise OSError('queue: get_by_task')

    def resolve(
        self, escalation_id: str, resolution: str, *,
        resolved_by: str | None = None, resolution_class: str | None = None,
    ) -> object:
        raise OSError('queue: resolve')


@pytest.mark.asyncio
async def test_raising_collaborators_lose_no_decision(
    repo: Path, history: History, git_ops: GitOps,
) -> None:
    runner = ScriptedRunner()
    screen = MergeScreen(git_ops, _RaisingEvents(), _RaisingEscalationQueue(), run_ids=runner)
    config = _config(repo, error_streak_alarm=1)

    bounced = await _screen_item(screen, config, history.item, history.base)
    runner.verdict = _status_for_clean('error')
    errored = await _screen_item(screen, config, history.item, history.base)

    assert isinstance(bounced, Bounce)
    assert bounced.check_ids == ('clean',)
    assert isinstance(errored, Proceed)


@pytest.mark.asyncio
async def test_a_raising_queue_still_lets_the_event_emit(
    repo: Path, history: History, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
) -> None:
    events = RecordingEvents()
    runner = ScriptedRunner(verdict=_status_for_clean('error'))
    screen = MergeScreen(git_ops, events, _RaisingEscalationQueue(), run_ids=runner)
    config = _config(repo, error_streak_alarm=1)

    with caplog.at_level(logging.WARNING, logger=_LANE_LOGGER):
        decisions = [await _screen_item(screen, config, history.item, history.base) for _ in range(2)]

    assert all(isinstance(decision, Proceed) for decision in decisions)
    assert [event['error_streak'] for event in events.screens()] == [1, 2]
    warnings = _lane_warnings(caplog)
    assert len(warnings) == 2
    assert all(r.exc_info is not None and isinstance(r.exc_info[1], OSError) for r in warnings)


@pytest.mark.asyncio
async def test_a_raising_event_sink_is_logged_and_swallowed(
    repo: Path, history: History, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
) -> None:
    screen = MergeScreen(git_ops, _RaisingEvents(), RecordingEscalationQueue(), run_ids=ScriptedRunner())

    with caplog.at_level(logging.WARNING, logger=_LANE_LOGGER):
        decision = await _screen_item(screen, _config(repo), history.item, history.base)

    assert isinstance(decision, Bounce)
    [warning] = _lane_warnings(caplog)
    assert warning.exc_info is not None and isinstance(warning.exc_info[1], OSError)


@dataclass
class _FixedPort:
    """A ScreenPort that returns *decision*, or raises *raises* when it is set."""

    decision: Any = None
    raises: BaseException | None = None

    async def screen_item(
        self, *, task_id: str, merged_sha: str, base_sha: str | None,
        config: OrchestratorConfig, site: ScreenSite,
    ) -> Any:
        if self.raises is not None:
            raise self.raises
        return self.decision


async def _through_guard(port: _FixedPort, config: OrchestratorConfig) -> Any:
    return await screen_or_proceed(
        port, task_id='42', merged_sha='m' * 40, base_sha='b' * 40, config=config, site=ScreenSite.MERGER,
    )


@pytest.mark.asyncio
async def test_screen_or_proceed_turns_a_raising_port_into_proceed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LANE_LOGGER):
        decision = await _through_guard(_FixedPort(raises=RuntimeError('boom')), _config(tmp_path))

    assert isinstance(decision, Proceed)
    [warning] = _lane_warnings(caplog)
    assert warning.exc_info is not None and isinstance(warning.exc_info[1], RuntimeError)
    assert '42' in warning.getMessage()


@pytest.mark.asyncio
async def test_screen_or_proceed_passes_a_bounce_through(tmp_path: Path) -> None:
    bounce = Bounce(MergeOutcome('blocked', reason='x'), ('clean',))

    assert await _through_guard(_FixedPort(decision=bounce), _config(tmp_path)) is bounce


@pytest.mark.asyncio
async def test_screen_or_proceed_lets_cancellation_through(tmp_path: Path) -> None:
    with pytest.raises(asyncio.CancelledError):
        await _through_guard(_FixedPort(raises=asyncio.CancelledError()), _config(tmp_path))
