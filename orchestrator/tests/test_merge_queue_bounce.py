"""Tests for the η=1892 bounce/rebase layer of the two-layer merge queue.

Covers:
  * the bounce primitives — ``NEEDS_REBASE_REASON_PREFIX``,
    ``MERGE_BOUNCE_CAP``, ``MergeBounceRegistry`` — as pure values;
  * the bounce as a RUNNING lane performs it over a real git repo: an item
    that conflicts with the frozen-prefix tip is diverted before it can
    consume a verify slot, and an item that is clean against that tip is
    left alone and lands;
  * the bookkeeping a bounce leaves behind — the per-branch bounce count,
    the untouched enqueue time, the ``needs_rebase`` log line, and the
    ``get_main_sha`` fail-open — read off ``SuffixConflictTracker``'s own
    public state.

Task 5030 (PRD ``plans/merge-lane-quality-prd.md`` task γ7) replaced this
file's drive mechanism. Every test here used to reach into the worker: it
seeded ``_inflight`` with a hand-built ``InflightEntry`` carrying a fake
merge commit, pushed requests straight into ``_lane_buffers``, assigned a
``_suffix_conflict_graph``, then called the private
``_bounce_conflicting_suffix_items()`` / ``_acquire_next_request()`` — one
test going as far as replacing three private methods with stubs purely to
pin the ORDER they are called in. That froze the lane's internals rather
than its behaviour: none of it would have noticed the bounce being wired
into a different code path, and all of it would have broken on a rename.

The lane-level tests below now submit real requests on the public queue and
read only ``MergeOutcome`` futures, ``MergeLane.snapshot()`` and what the
injected ``FakeVerifier`` was asked to verify. The frozen prefix is a real
one: the first request's verify is held open by ``hangs_until``, which is
what puts its merge commit at the tip while the second request is probed
against it. The bookkeeping tests drive ``SuffixConflictTracker`` — the
bounce layer's own class (``orchestrator.suffix_graph``) — through the four
accessor callables it is constructed with, so the lane buffers they assert
on belong to the test.

The core bounce paths (clean rebase re-queues, real conflict escalates, cap
exceeded escalates without rebasing, trains are skipped, an empty
``conflicts_with_main`` is a no-op, the TOCTOU signature reuse) are pinned
once, at that same seam, in ``test_suffix_conflict_tracker.py``; what stays
here is the coverage that file does not carry.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _merge_lane_fakes import FakeVerifier, hangs_until
from _orch_helpers import wait_responsive

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane import MergeLane
from orchestrator.merge_queue import (
    MERGE_BOUNCE_CAP,
    NEEDS_REBASE_REASON_PREFIX,
    MergeBounceRegistry,
    MergeOutcome,
    MergeRequest,
)
from orchestrator.merge_types import QueuedBranch
from orchestrator.suffix_graph import SuffixConflictGraph, SuffixConflictTracker

_STOP_TIMEOUT = 30.0

#: How long to wait for the first request to reach the frozen prefix. Generous
#: because it covers a real merge (worktree + git merge) on a cold tmp repo.
_FROZEN_PREFIX_TIMEOUT = 30.0


# ── Fixtures ──────────────────────────────────────────────────────────────────


async def _setup_repo(repo: Path) -> None:
    """Initialise a git repo with shared.txt and README.md on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    (repo / 'shared.txt').write_text('line1\nline2\nline3\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# ── Helpers: a running lane over a real repo ─────────────────────────────────


async def _prepare(
    git_ops: GitOps,
    config: OrchestratorConfig,
    task_id: str,
    filename: str,
    content: str,
) -> MergeRequest:
    """Commit *content* to *filename* on a fresh branch off main, unsubmitted.

    Preparing and submitting are separate so two requests can be branched off
    the SAME main: the bounce is only interesting when the second item was cut
    before the first one's merge commit existed.

    ``create_worktree`` prefixes the branch itself, so it takes the bare task
    id -- passing ``task/<id>`` would create ``task/task/<id>`` and the
    conflict graph's ``resolve_branch_sha(req.branch.full_name)`` would then
    find no ref and skip the item.
    """
    branch = f'task/{task_id}'
    worktree = (await git_ops.create_worktree(task_id)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'{task_id}: edit {filename}')
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


@contextlib.asynccontextmanager
async def _running_lane(git_ops: GitOps, verifier: FakeVerifier):
    """A running single-host lane whose verify outcomes *verifier* scripts.

    Teardown goes through ``stop()`` -- the lane's own shutdown protocol, which
    resolves in-flight request futures, drains its queues, cleans merge
    worktrees and releases leases, and is internally bounded so it cannot hang.
    """
    queue: asyncio.Queue = asyncio.Queue()
    lane = MergeLane(git_ops, queue, verifier=verifier)
    lane_task = asyncio.ensure_future(lane.run())
    try:
        yield lane, queue
    finally:
        # Exception, not BaseException: this must not swallow a CancelledError
        # aimed at the enclosing test task (or a KeyboardInterrupt).
        with contextlib.suppress(Exception):
            await asyncio.wait_for(lane.stop(), timeout=_STOP_TIMEOUT)
        lane_task.cancel()
        await asyncio.gather(lane_task, return_exceptions=True)


async def _await_frozen_tip(lane, timeout: float = _FROZEN_PREFIX_TIMEOUT) -> str:
    """Wait until one item is verifying, and return the frozen-prefix tip SHA.

    That tip is the merge commit the held-open verify froze at the head of the
    pipeline -- the base a suffix item is probed and rebased against.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        frozen = lane.snapshot()['frozen_prefix']
        if frozen['verify_depth'] >= 1 and frozen['tip_merge_commit']:
            return frozen['tip_merge_commit']
        await asyncio.sleep(0.05)
    raise AssertionError(
        f'no item reached the frozen prefix within {timeout}s; '
        f'snapshot={lane.snapshot()["frozen_prefix"]!r}'
    )


# ── Helpers: the bounce layer's own seam ─────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    *,
    merge_first_enqueued_at: float | None = 1000.0,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
    """
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
        merge_first_enqueued_at=merge_first_enqueued_at,
    )


def _make_tracker(
    git_ops: GitOps,
    *,
    lane_buffers: dict[str, collections.deque],
    frozen_tip: str,
) -> SuffixConflictTracker:
    """A tracker over the TEST's lane buffers and a fixed frozen-prefix tip.

    The four accessors the tracker is constructed with are the whole seam: no
    worker is involved, so every piece of state these tests assert on is one
    the test owns or one the tracker publishes.
    """
    return SuffixConflictTracker(
        git_ops=lambda: git_ops,
        lane_buffers=lambda: lane_buffers,
        frozen_prefix=lambda: (),
        frozen_prefix_tip=lambda _main_sha: frozen_tip,
    )


def _conflicting(req: MergeRequest) -> SuffixConflictGraph:
    """A graph whose single node conflicts with the frozen tip."""
    return SuffixConflictGraph(
        nodes=(req.request_id,),
        textual_edges=frozenset(),
        footprint_edges=frozenset(),
        conflicts_with_main=frozenset({req.request_id}),
    )


# ── The bounce primitives ────────────────────────────────────────────────────


class TestBouncePrimitives:
    """NEEDS_REBASE_REASON_PREFIX, MERGE_BOUNCE_CAP, MergeBounceRegistry."""

    def test_needs_rebase_reason_prefix_is_non_empty_str(self) -> None:
        """NEEDS_REBASE_REASON_PREFIX is a non-empty string constant."""
        assert isinstance(NEEDS_REBASE_REASON_PREFIX, str)
        assert len(NEEDS_REBASE_REASON_PREFIX) > 0

    def test_needs_rebase_reason_prefix_is_distinct(self) -> None:
        """NEEDS_REBASE_REASON_PREFIX differs from the other *_REASON_PREFIX constants."""
        from orchestrator.merge_queue import (
            DROPPED_PLAN_TARGETS_REASON_PREFIX,
            PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
        )
        others = {
            DROPPED_PLAN_TARGETS_REASON_PREFIX,
            PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
        }
        assert NEEDS_REBASE_REASON_PREFIX not in others

    def test_merge_outcome_with_needs_rebase_prefix_round_trips(self) -> None:
        """A blocked MergeOutcome carrying the prefix round-trips via startswith."""
        outcome = MergeOutcome(
            status='blocked',
            reason=NEEDS_REBASE_REASON_PREFIX + ' branch task/591 bounced',
        )
        assert outcome.status == 'blocked'
        assert outcome.reason is not None
        assert outcome.reason.startswith(NEEDS_REBASE_REASON_PREFIX)

    def test_merge_bounce_cap_is_positive_int(self) -> None:
        """MERGE_BOUNCE_CAP is an int >= 1."""
        assert isinstance(MERGE_BOUNCE_CAP, int)
        assert MERGE_BOUNCE_CAP >= 1

    def test_bounce_registry_fresh_returns_zero(self) -> None:
        """A fresh MergeBounceRegistry returns count 0 for any unseen branch."""
        reg = MergeBounceRegistry()
        assert reg.count('task/591') == 0
        assert reg.count('task/999') == 0

    def test_bounce_registry_record_bounce_increments(self) -> None:
        """record_bounce returns the new count and increments monotonically."""
        reg = MergeBounceRegistry()
        assert reg.record_bounce('task/591') == 1
        assert reg.record_bounce('task/591') == 2
        assert reg.count('task/591') == 2

    def test_bounce_registry_two_keys_are_independent(self) -> None:
        """Two different branch keys are tracked independently."""
        reg = MergeBounceRegistry()
        reg.record_bounce('task/591')
        reg.record_bounce('task/591')
        reg.record_bounce('task/592')
        assert reg.count('task/591') == 2
        assert reg.count('task/592') == 1


# ── The bounce as a running lane performs it ─────────────────────────────────


@pytest.mark.asyncio
class TestBounceDrivenByTheLane:
    """What a running lane does with a suffix item that collides with the
    frozen-prefix tip, submitted on the public queue and observed through the
    request's own outcome.

    Both tests hold the first request's verify open with ``hangs_until``: that
    is what keeps its merge commit at the frozen-prefix tip (an item is frozen
    exactly while it is verifying) and keeps main behind it, so the second
    request is probed against a tip that is genuinely ahead of main. The
    second request is cut from the same main as the first, before either has
    landed -- a branch cut afterwards would carry the first one's change and
    could not collide with it.
    """

    async def test_item_conflicting_with_frozen_tip_is_bounced_before_it_verifies(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """The conflicting item is escalated, and never reaches the verifier.

        The suffix branch is CLEAN against bare main -- nothing else has
        touched that line on main -- and conflicts only with the frozen tip.
        So this pins the §7 probe base as well as the bounce: were the lane
        still probing bare main, the item would sail through unbounced.
        """
        release = asyncio.Event()
        verifier = FakeVerifier(scripts={'frozen-a': hangs_until(release)})
        async with _running_lane(git_ops, verifier) as (lane, queue):
            frozen = await _prepare(
                git_ops, config, 'frozen-a', 'shared.txt', 'line1\nFROZEN-LINE2\nline3\n',
            )
            suffix = await _prepare(
                git_ops, config, 'suffix-b', 'shared.txt', 'line1\nSUFFIX-LINE2\nline3\n',
            )

            await queue.put(frozen)
            tip = await _await_frozen_tip(lane)

            await queue.put(suffix)
            outcome = await wait_responsive(suffix.result, label='conflicting suffix bounce')

            assert outcome.status == 'blocked', (
                f'expected the conflicting suffix item to be bounced, got {outcome!r}'
            )
            assert outcome.reason is not None
            assert outcome.reason.startswith(NEEDS_REBASE_REASON_PREFIX), (
                f'expected a needs_rebase escalation, got reason={outcome.reason!r}'
            )
            assert tip in outcome.reason, (
                f'the bounce probed {outcome.reason!r}, not the frozen-prefix tip '
                f'{tip!r} -- a suffix item must be stacked onto the tip, not bare main'
            )
            assert 'suffix-b' not in verifier.verified, (
                'the bounced item consumed a verify slot: the bounce must divert it '
                f'before dispatch, but the verifier was asked for {verifier.verified!r}'
            )

            release.set()

    async def test_item_clean_against_frozen_tip_is_not_bounced(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """CONTROL: a suffix item that does not collide is left to land.

        Same shape as the test above, but the suffix branch touches a
        different file. It must not be diverted -- it waits for the verify
        slot the frozen item is holding and lands once that item releases it.
        """
        release = asyncio.Event()
        verifier = FakeVerifier(scripts={'frozen-a': hangs_until(release)})
        async with _running_lane(git_ops, verifier) as (lane, queue):
            frozen = await _prepare(
                git_ops, config, 'frozen-a', 'shared.txt', 'line1\nFROZEN-LINE2\nline3\n',
            )
            clean = await _prepare(
                git_ops, config, 'suffix-c', 'README.md', '# clean suffix\n',
            )

            await queue.put(frozen)
            await _await_frozen_tip(lane)
            await queue.put(clean)
            release.set()

            frozen_outcome = await wait_responsive(
                frozen.result, label='frozen-prefix item lands',
            )
            clean_outcome = await wait_responsive(
                clean.result, label='non-conflicting item lands',
            )

            assert frozen_outcome.status == 'done', (
                f'expected the frozen item to land, got {frozen_outcome!r}'
            )
            assert clean_outcome.status == 'done', (
                f'expected the clean suffix item to land unbounced, got {clean_outcome!r}'
            )
            assert 'suffix-c' in verifier.verified, (
                'the clean suffix item never reached the verifier, so it was '
                f'diverted after all (verified={verifier.verified!r})'
            )


# ── What a bounce records ────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestBounceBookkeeping:
    """The state a bounce leaves behind, at the bounce layer's own seam.

    ``test_suffix_conflict_tracker.py`` pins where each item ENDS UP (re-queued
    on a clean rebase, escalated on a real conflict or a busted cap, skipped
    for a train). These pin what the bounce records while doing it -- the
    per-branch count, the untouched enqueue time, the log line an operator
    greps for, and the fail-open when the tip cannot be established at all.
    """

    async def test_clean_rebase_records_one_bounce_and_keeps_the_enqueue_time(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A clean rebase counts against the cap but costs the item nothing else.

        ``merge_first_enqueued_at`` drives the queue-age alarms, so a bounced
        item that keeps its place in the buffer must keep its original enqueue
        time too -- otherwise repeated bounces would hide a starving branch.
        """
        req = _make_req('591', '591', config, git_repo, merge_first_enqueued_at=9999.0)
        buffers = {'high': collections.deque(), 'normal': collections.deque([req])}
        tracker = _make_tracker(
            git_ops, lane_buffers=buffers, frozen_tip='deadbeefcafe0000',
        )
        tracker.graph = _conflicting(req)
        tracker.signature = ((req.request_id,), 'abc123')
        git_ops.rebase_onto_main = AsyncMock(return_value=True)  # type: ignore[method-assign]

        with caplog.at_level(logging.INFO):
            await tracker.bounce_conflicting_suffix_items()

        assert tracker.bounce_registry.count('591') == 1, (
            'a clean rebase must still count against MERGE_BOUNCE_CAP, or a '
            'branch that rebases cleanly every cycle would bounce forever'
        )
        assert req.merge_first_enqueued_at == 9999.0, (
            'merge_first_enqueued_at must survive a bounce'
        )
        assert req in buffers['normal'], 'a cleanly rebased item stays queued'
        assert any('needs_rebase' in r.message.lower() for r in caplog.records), (
            'no needs_rebase log line was emitted; records='
            f'{[r.message for r in caplog.records]}'
        )

    async def test_escalation_clears_the_bounce_count(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """An escalated branch is handed back with a clean slate.

        The bounce that escalates is recorded and then cleared, so the branch
        is not born at the cap when the steward resubmits it.
        """
        req = _make_req('591', '591', config, git_repo)
        buffers = {'high': collections.deque(), 'normal': collections.deque([req])}
        tracker = _make_tracker(
            git_ops, lane_buffers=buffers, frozen_tip='deadbeefcafe1111',
        )
        tracker.graph = _conflicting(req)
        tracker.signature = ((req.request_id,), 'abc123')
        git_ops.rebase_onto_main = AsyncMock(return_value=False)  # type: ignore[method-assign]

        await tracker.bounce_conflicting_suffix_items()

        assert req.result.done(), 'expected the escalation to resolve the future'
        assert req.result.result().status == 'blocked'
        assert tracker.bounce_registry.count('591') == 0, (
            'bounce count must be 0 after escalation: record_bounce() increments, '
            'then clear() resets so a later resubmission starts fresh'
        )

    async def test_get_main_sha_failure_leaves_items_untouched(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With no cached signature, a failing get_main_sha() abandons the bounce.

        The signature is the main SHA the graph was computed against; a prior
        bounce in the same acquire cycle clears it, and the fresh fetch is the
        only remaining source. When that fetch fails there is no tip to rebase
        onto, so nothing may be rebased, escalated or counted.
        """
        req = _make_req('591', '591', config, git_repo)
        buffers = {'high': collections.deque(), 'normal': collections.deque([req])}
        tracker = _make_tracker(
            git_ops, lane_buffers=buffers, frozen_tip='deadbeefcafe2222',
        )
        tracker.graph = _conflicting(req)
        tracker.signature = None
        git_ops.get_main_sha = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError('network error'),
        )
        rebase_spy = AsyncMock(return_value=True)
        git_ops.rebase_onto_main = rebase_spy  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING):
            await tracker.bounce_conflicting_suffix_items()

        rebase_spy.assert_not_awaited()
        assert req in buffers['normal'], 'req was removed despite get_main_sha failing'
        assert not req.result.done(), 'req was escalated despite get_main_sha failing'
        assert tracker.bounce_registry.count('591') == 0, (
            f'bounce count must stay 0 after a failed fetch, got '
            f'{tracker.bounce_registry.count("591")}'
        )
        assert tracker.signature is None, 'a failed bounce must not cache a signature'
        assert any('get_main_sha' in r.message for r in caplog.records), (
            'expected a warning naming get_main_sha but none was logged'
        )
