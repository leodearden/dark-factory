"""The merge screen's two dispatch sites, driven through the lane's public surfaces.

plans/merge-pre-verify-screen-prd.md decisions 1 and 7: the merger screens
each merged item before it reaches verify, and a declared train screens its
merged tip. A bounce is an ordinary blocked outcome with no verify and no
host lease; ``screen=None`` and a manifest of null are byte-identical to no
screen at all; the remerge path never screens.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import (
    FakeVerifier,
    branch_merge_request,
    lane_scene_config,
    make_lane,
    merge_through_lane,
    running_lane,
)
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT, git_env_with_ceiling

from orchestrator.config import GitConfig, MergeScreenConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.outcome_facts import MergeBlockCategory
from orchestrator.merge_lane.screen import (
    MERGE_SCREEN_FAILURE_CATEGORY,
    MERGE_SCREEN_REASON_PREFIX,
    Bounce,
    MergeScreen,
    Proceed,
    ScreenDecision,
    ScreenSite,
)
from orchestrator.merge_lane.types import MergedOk, MergeOutcome, QueuedBranch
from orchestrator.merge_lane.worker import (
    TRAIN_VERIFY_FAILED_REASON_PREFIX,
    GroupMergeRequest,
    classify_and_merge,
)

pytestmark = pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)

FINDING = 'inline_suppressions: scripts/x.py:3 unowned suppression'


@dataclass(frozen=True)
class ScreenCall:
    task_id: str
    merged_sha: str
    base_sha: str | None
    site: ScreenSite


@dataclass
class ScriptedScreen:
    """A ScreenPort that records each call and bounces, proceeds or raises as told."""

    bounce: bool = False
    raises: BaseException | None = None
    calls: list[ScreenCall] = field(default_factory=list)

    async def screen_item(
        self, *, task_id: str, merged_sha: str, base_sha: str | None,
        config: OrchestratorConfig, site: ScreenSite,
    ) -> ScreenDecision:
        self.calls.append(ScreenCall(task_id, merged_sha, base_sha, site))
        if self.raises is not None:
            raise self.raises
        if not self.bounce:
            return Proceed()
        return Bounce(
            MergeOutcome(
                'blocked',
                reason=f'{MERGE_SCREEN_REASON_PREFIX} the item\'s own defect in inline_suppressions:\n{FINDING}',
                failure_category=MERGE_SCREEN_FAILURE_CATEGORY,
                failure_cause_hint='inline_suppressions',
                block_category=MergeBlockCategory.MERGE_SCREEN,
            ),
            ('inline_suppressions',),
        )


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture
def git_ops(git_config: GitConfig, repo: Path) -> GitOps:
    return GitOps(git_config, repo)


@pytest.fixture
def config(repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return lane_scene_config(repo, git_config)


@pytest.fixture
def event_store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'events.db', 'run-screen')


def _rows(store: EventStore) -> list[tuple[str, str | None, dict[str, Any]]]:
    conn = sqlite3.connect(str(store.db_path))
    try:
        rows = conn.execute('SELECT event_type, task_id, data FROM events ORDER BY id').fetchall()
    finally:
        conn.close()
    return [(event_type, task_id, json.loads(data) if data else {}) for event_type, task_id, data in rows]


def _events(store: EventStore, event_type: str, task_id: str | None = None) -> list[dict[str, Any]]:
    return [
        data for kind, row_task, data in _rows(store)
        if kind == event_type and (task_id is None or row_task == task_id)
    ]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, env=git_env_with_ceiling(repo),
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _merge_worktrees(repo: Path) -> list[str]:
    listing = _git(repo, 'worktree', 'list', '--porcelain')
    return [line for line in listing.splitlines() if line.startswith('worktree ') and '/_merge-' in line]


# ---------------------------------------------------------------------------
# (a) (b) the merger site


@pytest.mark.asyncio
async def test_an_own_defect_bounces_before_verify(
    git_ops: GitOps, config: OrchestratorConfig, event_store: EventStore, repo: Path,
) -> None:
    req = await branch_merge_request(git_ops, config, 'a1', filename='a1.md')
    branch_tip = _git(repo, 'rev-parse', 'task/a1')
    main_before = await git_ops.get_main_sha()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    verifier = FakeVerifier()
    screen = ScriptedScreen(bounce=True)
    lane = make_lane(git_ops, queue, verifier=verifier, event_store=event_store, screen=screen)

    outcome = await merge_through_lane(lane, queue, req)

    assert outcome.status == 'blocked', outcome
    assert outcome.failure_category == 'merge_screen'
    assert outcome.block_category is MergeBlockCategory.MERGE_SCREEN
    assert FINDING in outcome.reason
    assert [data['outcome'] for data in _events(event_store, 'merge_attempt', 'a1')] == ['screen_failed']
    assert _events(event_store, 'merge_verify', 'a1') == []
    assert 'a1' not in verifier.verified
    assert _merge_worktrees(repo) == []
    assert lane.snapshot()['entries'] == []
    assert await git_ops.get_main_sha() == main_before
    [call] = screen.calls
    assert (call.task_id, call.site) == ('a1', ScreenSite.MERGER)
    assert _git(repo, 'rev-parse', f'{call.merged_sha}^2') == branch_tip
    assert call.base_sha == _git(repo, 'rev-parse', f'{call.merged_sha}^1') == main_before


@pytest.mark.asyncio
async def test_proceed_verifies_and_lands(
    git_ops: GitOps, config: OrchestratorConfig, event_store: EventStore,
) -> None:
    req = await branch_merge_request(git_ops, config, 'b1', filename='b1.md')
    queue: asyncio.Queue[Any] = asyncio.Queue()
    verifier = FakeVerifier()
    screen = ScriptedScreen()
    lane = make_lane(git_ops, queue, verifier=verifier, event_store=event_store, screen=screen)

    outcome = await merge_through_lane(lane, queue, req)

    assert outcome.status == 'done', outcome
    assert 'b1' in verifier.verified
    assert await git_ops.get_main_sha() == outcome.merge_sha
    assert [call.task_id for call in screen.calls] == ['b1']
    assert [data['outcome'] for data in _events(event_store, 'merge_attempt', 'b1')] == ['done']


@pytest.mark.asyncio
async def test_a_raising_screen_fails_open_at_the_merger(
    git_ops: GitOps, config: OrchestratorConfig, event_store: EventStore, repo: Path,
) -> None:
    req = await branch_merge_request(git_ops, config, 'r1', filename='r1.md')
    queue: asyncio.Queue[Any] = asyncio.Queue()
    verifier = FakeVerifier()
    screen = ScriptedScreen(raises=RuntimeError('screen exploded'))
    lane = make_lane(git_ops, queue, verifier=verifier, event_store=event_store, screen=screen)

    outcome = await merge_through_lane(lane, queue, req)

    assert outcome.status == 'done', outcome
    assert 'r1' in verifier.verified
    assert [data['outcome'] for data in _events(event_store, 'merge_attempt', 'r1')] == ['done']
    assert _merge_worktrees(repo) == []
    assert lane.snapshot()['entries'] == []
    assert [call.site for call in screen.calls] == [ScreenSite.MERGER]


# ---------------------------------------------------------------------------
# (c) the kill switch is byte-identical

_VOLATILE_KEY_SUFFIXES = ('_ms', '_secs', '_s')


def _scrubbed(value: Any, request_ids: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrubbed(item, request_ids)
            for key, item in value.items()
            if not key.endswith(_VOLATILE_KEY_SUFFIXES)
        }
    if isinstance(value, list):
        return [_scrubbed(item, request_ids) for item in value]
    if isinstance(value, str) and value.startswith('mr-'):
        return request_ids.setdefault(value, f'mr-{len(request_ids)}')
    return value


def _comparable_events(store: EventStore) -> list[tuple[str, str | None, Any]]:
    request_ids: dict[str, str] = {}
    return [
        (kind, task_id, _scrubbed(data, request_ids))
        for kind, task_id, data in _rows(store)
        if kind != 'merge_heartbeat'
    ]


async def _two_request_scene(root: Path, git_config: GitConfig, screen_for: Any) -> tuple[list[Any], list[Any]]:
    repo = seed_repo(root / 'repo')
    git_ops = GitOps(git_config, repo)
    config = lane_scene_config(repo, git_config, merge_screen=MergeScreenConfig(manifest=None))
    store = EventStore(root / 'events.db', 'run-scene')
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(git_ops, queue, event_store=store, screen=screen_for(git_ops, store))
    requests = [
        await branch_merge_request(git_ops, config, task_id, filename=f'{task_id}.md')
        for task_id in ('k1', 'k2')
    ]
    outcomes = []
    async with running_lane(lane) as run:
        for req in requests:
            await queue.put(req)
            outcome = await run.outcome(req)
            outcomes.append((outcome.status, outcome.reason, outcome.merge_sha))
    return outcomes, _comparable_events(store)


@pytest.mark.asyncio
async def test_kill_switch_is_byte_identical_to_no_screen(
    tmp_path: Path, git_config: GitConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('GIT_AUTHOR_DATE', '2026-01-01T00:00:00Z')
    monkeypatch.setenv('GIT_COMMITTER_DATE', '2026-01-01T00:00:00Z')

    without = await _two_request_scene(tmp_path / 'none', git_config, lambda git_ops, store: None)
    killed = await _two_request_scene(
        tmp_path / 'killed', git_config,
        lambda git_ops, store: MergeScreen(git_ops, store, None),
    )

    assert [status for status, _, _ in without[0]] == ['done', 'done']
    assert killed == without
    assert all(kind != 'merge_screen' for kind, _, _ in killed[1])


# ---------------------------------------------------------------------------
# (d) the remerge path never screens


@pytest.mark.asyncio
async def test_classify_and_merge_itself_never_screens(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    req = await branch_merge_request(git_ops, config, 'd1', filename='d1.md')
    screen = ScriptedScreen(bounce=True)
    worker = make_lane(git_ops, asyncio.Queue(), screen=screen)

    result = await classify_and_merge(
        worker, req, await git_ops.get_main_sha(), speculative=False, started_monotonic=None,
    )

    assert isinstance(result, MergedOk)
    assert screen.calls == []
    assert result.merge_wt is not None
    await git_ops.cleanup_merge_worktree(result.merge_wt)


# ---------------------------------------------------------------------------
# (e) a declared train's merged tip


async def _stacked_member(git_ops: GitOps, name: str, base_ref: str) -> Path:
    worktree = git_ops.worktree_base / name
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _git(git_ops.project_root, 'worktree', 'add', '-q', '-b', f'task/{name}', str(worktree), base_ref)
    (worktree / f'{name}.txt').write_text(f'{name}\n')
    await git_ops.commit(worktree, f'Add {name}')
    return worktree


async def _train(git_ops: GitOps, config: OrchestratorConfig, train_id: str) -> GroupMergeRequest:
    members = [f'{train_id}-a', f'{train_id}-b', f'{train_id}-c']
    base = await git_ops.get_main_sha()
    tip = None
    for name in members:
        tip = await _stacked_member(git_ops, name, base)
        base = f'task/{name}'
    assert tip is not None
    return GroupMergeRequest(
        task_id=members[-1],
        branch=QueuedBranch.parse(members[-1], config.git.branch_prefix),
        worktree=tip,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        train_id=train_id,
        member_task_ids=members,
        tip_branch=QueuedBranch.parse(members[-1], config.git.branch_prefix),
        tip_task_id=members[-1],
        status_check=AsyncMock(return_value=dict.fromkeys(members, 'merge-deferred')),
        mark_member_done=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_a_train_finding_derails_without_verify(
    git_ops: GitOps, config: OrchestratorConfig, event_store: EventStore, repo: Path,
) -> None:
    req = await _train(git_ops, config, 'T1')
    main_before = await git_ops.get_main_sha()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    verifier = FakeVerifier()
    screen = ScriptedScreen(bounce=True)
    lane = make_lane(git_ops, queue, verifier=verifier, event_store=event_store, screen=screen)

    outcome = await merge_through_lane(lane, queue, req)

    assert outcome.status == 'blocked', outcome
    assert not outcome.reason.startswith(TRAIN_VERIFY_FAILED_REASON_PREFIX)
    assert outcome.failure_category == 'merge_screen'
    assert verifier.verified == []
    assert _events(event_store, 'merge_verify') == []
    assert req.mark_member_done.call_count == 0
    attempts = _events(event_store, 'merge_attempt', req.task_id)
    assert [(data['outcome'], data.get('train_id')) for data in attempts] == [('screen_failed', 'T1')]
    assert len(_events(event_store, 'train_derailed', req.task_id)) == 1
    assert _merge_worktrees(repo) == []
    assert await git_ops.get_main_sha() == main_before
    [call] = screen.calls
    assert call.site == ScreenSite.TRAIN
    assert call.base_sha == _git(repo, 'rev-parse', f'{call.merged_sha}^1')


@pytest.mark.asyncio
async def test_a_proceeding_train_verifies_and_lands(
    git_ops: GitOps, config: OrchestratorConfig, event_store: EventStore,
) -> None:
    req = await _train(git_ops, config, 'T2')
    queue: asyncio.Queue[Any] = asyncio.Queue()
    verifier = FakeVerifier()
    screen = ScriptedScreen()
    lane = make_lane(git_ops, queue, verifier=verifier, event_store=event_store, screen=screen)

    outcome = await merge_through_lane(lane, queue, req)

    assert outcome.status == 'done', outcome
    assert verifier.verified == [req.task_id]
    assert [call.site for call in screen.calls] == [ScreenSite.TRAIN]
    assert req.mark_member_done.call_count == 3
    assert [data['outcome'] for data in _events(event_store, 'merge_attempt', req.task_id)] == ['done']


@pytest.mark.asyncio
async def test_a_raising_screen_fails_open_at_the_train_tip(
    git_ops: GitOps, config: OrchestratorConfig, event_store: EventStore, repo: Path,
) -> None:
    req = await _train(git_ops, config, 'T3')
    queue: asyncio.Queue[Any] = asyncio.Queue()
    verifier = FakeVerifier()
    screen = ScriptedScreen(raises=RuntimeError('screen exploded'))
    lane = make_lane(git_ops, queue, verifier=verifier, event_store=event_store, screen=screen)

    outcome = await merge_through_lane(lane, queue, req)

    assert outcome.status == 'done', outcome
    assert verifier.verified == [req.task_id]
    assert req.mark_member_done.call_count == 3
    assert _merge_worktrees(repo) == []
    assert [call.site for call in screen.calls] == [ScreenSite.TRAIN]
