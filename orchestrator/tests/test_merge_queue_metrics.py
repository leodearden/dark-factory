"""Tests for ι=1894: retries-per-landing + drift-at-detection metrics.

Covers:
  * ``MergeMetrics`` as a pure accumulator (landings, retries, drift window).
  * The lane's public ``snapshot()['metrics']`` key: present on a fresh lane,
    zero-valued before any work, and additive beside the pre-existing keys.
  * The call-site wiring, driven end-to-end: a real merge that lands moves
    ``landings_total``, and a real merge conflict records a drift sample.
  * Per-request drift-base isolation, driven end-to-end: a conflict counts
    only the landings since ITS OWN merge-start, so a landing that consumed
    another in-flight request's drift base is caught.

Task 5030 (PRD ``plans/merge-lane-quality-prd.md`` task γ7) replaced this
file's former drive mechanism. The wiring used to be exercised by calling the
lane's private notifiers (``_note_merge_started``/``_note_merge_landing``/
``_note_merge_retry``/``_note_conflict_detected``) and asserting on the
private ``_merge_metrics``/``_drift_base`` they write, which pinned the
implementation rather than the behaviour. Every lane-level test here now
submits real requests on the public queue and reads only
``MergeLane.snapshot()``; the accumulator's arithmetic stays where it belongs,
in the ``MergeMetrics`` unit tests below.
"""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from _merge_lane_fakes import FakeVerifier, hangs_until, passes
from _orch_helpers import (
    MERGE_GATE_BARRIER_TIMEOUT,
    MERGE_RESULT_TIMEOUT,
    wait_responsive,
)

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_lane import MergeLane
from orchestrator.merge_lane.ports import VerifyPort
from orchestrator.merge_queue import MergeMetrics, MergeRequest
from orchestrator.merge_types import QueuedBranch

_STOP_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Fixtures (per-file duplication convention — see
# test_merge_queue_permit_conservation.py)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a git repo with one commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
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
    """Single-host (no verify_runners) OrchestratorConfig."""
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# ---------------------------------------------------------------------------
# Lane helpers
# ---------------------------------------------------------------------------


def _bare_lane() -> Any:
    """A lane over a mocked GitOps — enough to read snapshot(), never run."""
    git_ops = MagicMock(spec=GitOps)
    git_ops.project_root = None  # None-safe: __init__ guards on this
    return MergeLane(git_ops, asyncio.Queue(), verifier=FakeVerifier())


async def _prepare(
    git_ops: GitOps,
    config: OrchestratorConfig,
    task_id: str,
    filename: str,
    content: str,
) -> MergeRequest:
    """Commit *filename* on a fresh branch off the CURRENT main, unsubmitted.

    Preparing and submitting are separate so a test can branch two requests off
    the same base (the add/add conflict below needs that) rather than always
    branching off whatever has landed by then.
    """
    branch = f'task/{task_id}'
    worktree = (await git_ops.create_worktree(branch)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    request = MergeRequest(
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
    return request


class _MergeGatedGitOps(GitOps):
    """A ``GitOps`` whose ``merge_to_main`` parks on an event for ONE branch.

    A SUBCLASS rather than a ``patch``/``setattr``: ``git_ops`` is a
    constructor parameter of ``MergeLane``, so scripting a PUBLIC method of an
    injected collaborator is the same category of seam as the injected
    ``FakeVerifier`` -- it needs no patching and no private access, and keeps
    this file at its ratchet baseline of 0 patch targets and 0 private reads.

    Parking inside ``merge_to_main`` is what makes a drift drive
    deterministic: the lane stashes a request's drift base immediately before
    merging it, and that window is otherwise only a handful of awaits wide.
    ``at_gate`` being set is proof the base has already been stashed, so a
    test can hold the gated request there for as long as it needs while it
    drives another request all the way to a landing.

    The override mirrors the parent signature EXACTLY rather than absorbing
    a ``**kwargs``: a true substitution keeps a future positional
    ``base_sha`` caller binding here as it does in production, and the
    declared ``MergeResult`` return lets pyright catch a fake that stops
    handing back a merge result at this seam instead of downstream.
    """

    def __init__(self, config: GitConfig, root: Path, *, branch: str) -> None:
        super().__init__(config, root)
        self._gated_branch = branch
        self.at_gate = asyncio.Event()
        self.release_merge = asyncio.Event()

    async def merge_to_main(
        self, worktree: Path, branch: str, base_sha: str | None = None,
    ) -> MergeResult:
        if branch == self._gated_branch:
            self.at_gate.set()
            await self.release_merge.wait()
        return await super().merge_to_main(worktree, branch, base_sha=base_sha)


@contextlib.asynccontextmanager
async def _running_lane(
    git_ops: GitOps,
    *,
    verifier: VerifyPort | None = None,
    speculation_depth: int = 1,
    gates: tuple[asyncio.Event, ...] = (),
):
    """A running single-host lane; its scoped verify passes unless scripted.

    Every keyword defaults to today's behaviour, so the plain
    ``_running_lane(git_ops)`` call sites are unchanged: an always-passing
    ``FakeVerifier``, one merge ahead, and no gates.

    Teardown releases every gate in *gates* BEFORE stopping, so a failing
    assertion can never leave a verify or a merge parked and hang the stop.
    It then goes through ``stop()`` -- the lane's own shutdown protocol, which
    resolves in-flight request futures, drains its queues, cleans merge
    worktrees and releases leases, and is internally bounded so it cannot hang.
    """
    queue: asyncio.Queue = asyncio.Queue()
    lane = MergeLane(
        git_ops, queue,
        speculation_depth=speculation_depth,
        verifier=FakeVerifier() if verifier is None else verifier,
    )
    lane_task = asyncio.ensure_future(lane.run())
    try:
        yield lane, queue
    finally:
        for gate in gates:
            gate.set()
        # Exception, not BaseException: this must not swallow a CancelledError
        # aimed at the enclosing test task (or a KeyboardInterrupt).
        with contextlib.suppress(Exception):
            await asyncio.wait_for(lane.stop(), timeout=_STOP_TIMEOUT)
        lane_task.cancel()
        await asyncio.gather(lane_task, return_exceptions=True)


# ---------------------------------------------------------------------------
# MergeMetrics pure accumulator
# ---------------------------------------------------------------------------


class TestMergeMetrics:
    """Pure unit tests for the MergeMetrics accumulator.

    RED until step-02 GREEN adds MergeMetrics to merge_queue.py.
    """

    def test_initial_state_zero_counts(self):
        """Fresh accumulator has zero landings, retries, and empty drift."""
        m = MergeMetrics()
        assert m.landings == 0
        assert m.retries == 0
        assert m.main_position == 0

    def test_main_position_equals_landings(self):
        """main_position tracks the number of landings (each landing advances main by 1)."""
        m = MergeMetrics()
        m.record_landing()
        assert m.main_position == 1
        m.record_landing()
        m.record_landing()
        assert m.main_position == 3

    def test_retries_per_landing_none_when_no_landings(self):
        """retries_per_landing is None when landings == 0 (no div-by-zero)."""
        m = MergeMetrics()
        assert m.retries_per_landing is None

    def test_retries_per_landing_zero_when_no_retries(self):
        """retries_per_landing is 0.0 when there are landings but no retries."""
        m = MergeMetrics()
        m.record_landing()
        assert m.retries_per_landing == 0.0

    def test_retries_per_landing_exact_arithmetic(self):
        """retries_per_landing = retries / landings exactly (3 retries / 2 landings == 1.5)."""
        m = MergeMetrics()
        m.record_landing()
        m.record_landing()
        m.record_retry()
        m.record_retry()
        m.record_retry()
        assert m.retries_per_landing == 1.5

    def test_record_drift_populates_summary(self):
        """record_drift(n) feeds the drift summary with correct values."""
        m = MergeMetrics()
        m.record_drift(3)
        s = m.drift_summary()
        assert s['count'] == 1
        assert s['last'] == 3
        assert s['mean'] == 3.0
        assert s['max'] == 3

    def test_drift_summary_multi_sample(self):
        """drift_summary() returns correct count/last/mean/max across multiple samples."""
        m = MergeMetrics()
        m.record_drift(2)
        m.record_drift(6)
        m.record_drift(4)
        s = m.drift_summary()
        assert s['count'] == 3
        assert s['last'] == 4
        assert s['mean'] == pytest.approx(4.0)
        assert s['max'] == 6

    def test_drift_summary_empty_when_no_drifts(self):
        """drift_summary() returns all-zero/None values when no drifts recorded."""
        m = MergeMetrics()
        s = m.drift_summary()
        assert s['count'] == 0
        assert s['last'] is None
        assert s['mean'] is None
        assert s['max'] is None

    def test_drift_buffer_is_bounded(self):
        """Drift samples are bounded — oldest samples drop past the window cap."""
        m = MergeMetrics(drift_window=5)
        for i in range(10):
            m.record_drift(i)
        s = m.drift_summary()
        # Only the 5 most-recent samples (5..9) remain
        assert s['count'] == 5
        assert s['max'] == 9
        # The 5 oldest (0..4) are gone — mean of 5..9 = 7.0
        assert s['mean'] == pytest.approx(7.0)

    def test_as_snapshot_shape(self):
        """as_snapshot() returns the expected dict keys."""
        m = MergeMetrics()
        m.record_landing()
        m.record_retry()
        m.record_drift(2)
        snap = m.as_snapshot()
        assert 'retries_per_landing' in snap
        assert 'drift_at_detection' in snap
        assert 'landings_total' in snap
        assert 'retries_total' in snap
        assert snap['landings_total'] == 1
        assert snap['retries_total'] == 1
        assert snap['retries_per_landing'] == 1.0
        assert snap['drift_at_detection']['count'] == 1
        assert snap['drift_at_detection']['last'] == 2

    def test_as_snapshot_none_rpl_with_no_landings(self):
        """as_snapshot() carries retries_per_landing=None when landings==0."""
        m = MergeMetrics()
        snap = m.as_snapshot()
        assert snap['retries_per_landing'] is None



# ---------------------------------------------------------------------------
# snapshot()'s 'metrics' key
# ---------------------------------------------------------------------------


class TestLaneSnapshotMetricsKey:
    """``snapshot()`` carries a top-level 'metrics' key, additively."""

    def test_snapshot_has_metrics_key(self) -> None:
        assert 'metrics' in _bare_lane().snapshot()

    def test_snapshot_metrics_zero_state(self) -> None:
        """A lane that has done no work reports no landings, retries or drift."""
        metrics = _bare_lane().snapshot()['metrics']

        assert metrics['retries_per_landing'] is None
        assert metrics['landings_total'] == 0
        assert metrics['retries_total'] == 0
        assert metrics['drift_at_detection']['count'] == 0

    def test_snapshot_backward_compat_keys_present(self) -> None:
        """Pre-existing snapshot keys are still present alongside 'metrics'."""
        snap = _bare_lane().snapshot()

        for key in ('entries', 'depth', 'head_of_line', 'suffix_conflict_graph'):
            assert key in snap, f"pre-existing key '{key}' missing from snapshot"


# ---------------------------------------------------------------------------
# Call-site wiring, driven end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMetricsFromRealMerges:
    """The metrics a REAL merge produces, read off the public snapshot.

    These replace the former private-notifier tests: a regression that removes
    or misplaces a ``_note_merge_landing`` / ``_note_conflict_detected``
    call-site fails here, and so does one that stops surfacing the counter --
    neither is visible to a test that calls the notifier itself.
    """

    async def test_clean_landing_increments_landings_total(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """One request merged, verified and landed moves landings_total 0 -> 1."""
        async with _running_lane(git_ops) as (lane, queue):
            assert lane.snapshot()['metrics']['landings_total'] == 0

            request = await _prepare(
                git_ops, config, 'metrics-land-a', 'land_a.py', 'x = 1\n',
            )
            await queue.put(request)
            outcome = await wait_responsive(request.result, label='clean landing')
            assert outcome.status == 'done', f'expected a clean landing, got {outcome!r}'

            metrics = lane.snapshot()['metrics']
            assert metrics['landings_total'] == 1, (
                '_note_merge_landing wiring missing: landings_total did not increment'
            )
            assert metrics['retries_total'] == 0
            assert metrics['retries_per_landing'] == 0.0

    async def test_conflicting_merge_records_a_drift_sample(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """A real merge conflict records one drift sample.

        The sample's VALUE is ``main_position - <position at merge-start>``,
        which is 0 here by construction: this drive lands A before B is even
        dequeued, so no landing intervenes between B's merge-start and its
        conflict. The subtraction itself is covered by ``TestMergeMetrics``
        (``record_drift`` + ``main_position``); what this test pins is the
        wiring -- that a conflict reaches the drift recorder at all, and that
        the sample surfaces on the snapshot.
        """
        async with _running_lane(git_ops) as (lane, queue):
            # BOTH branches are cut from the same base, before either lands, so
            # they each ADD clash.py -- an add/add conflict. Branching B after A
            # landed would make B a clean modification of A's file instead.
            first = await _prepare(
                git_ops, config, 'metrics-drift-a', 'clash.py', 'x = 1\n',
            )
            second = await _prepare(
                git_ops, config, 'metrics-drift-b', 'clash.py', 'x = 2\n',
            )

            await queue.put(first)
            outcome_first = await wait_responsive(first.result, label='item A lands')
            assert outcome_first.status == 'done', f'expected A to land, got {outcome_first!r}'
            assert lane.snapshot()['metrics']['drift_at_detection']['count'] == 0

            await queue.put(second)
            outcome_second = await wait_responsive(second.result, label='item B conflicts with A')
            assert outcome_second.status == 'conflict', (
                f'expected B to conflict with A, got {outcome_second!r}'
            )

            drift = lane.snapshot()['metrics']['drift_at_detection']
            assert drift['count'] == 1, (
                '_note_conflict_detected wiring missing: the conflict recorded '
                f'no drift sample ({drift!r})'
            )
            assert drift['last'] == 0


# ---------------------------------------------------------------------------
# Per-request drift-base isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDriftBaseIsolation:
    """A landing must not consume another in-flight request's drift base."""

    async def test_drift_counts_only_the_landings_since_its_own_merge_start(
        self, git_config: GitConfig, git_repo: Path, config: OrchestratorConfig,
    ) -> None:
        """A conflict after ONE intervening landing records a drift of 1.

        The sibling ``test_conflicting_merge_records_a_drift_sample`` records a
        drift of 0 by construction -- it lands A before B is dequeued, so no
        landing falls between B's merge-start and its conflict. At 0, "pop MY
        entry" and "clear EVERY entry" are indistinguishable. Only an
        intervening landing separates them: here B's base is stashed while
        main_position is still 0, A lands (main_position -> 1), and only then
        does B conflict. A landing that consumed B's base instead of its own
        would drop the sample back to 0.

        The window between B's ``_note_merge_started`` and its merge is a
        handful of awaits wide, so it is not stably observable on the public
        snapshot. Parking B INSIDE ``merge_to_main`` holds it open for as long
        as the test needs: ``at_gate`` being set is proof the base was already
        stashed, and A's landing is then driven to completion and confirmed on
        the public counter before B's merge is released.
        """
        verify_gate = asyncio.Event()
        git_ops = _MergeGatedGitOps(git_config, git_repo, branch='task/drift-blocked')
        verifier = FakeVerifier(
            default=passes(), scripts={'drift-lander': hangs_until(verify_gate)},
        )

        async with _running_lane(
            git_ops,
            verifier=verifier,
            speculation_depth=2,
            gates=(verify_gate, git_ops.release_merge),
        ) as (lane, queue):
            # Both branches are cut from the same base, before either lands, so
            # they each ADD clash.py -- an add/add conflict.
            lander = await _prepare(
                git_ops, config, 'drift-lander', 'clash.py', 'x = 1\n',
            )
            blocked = await _prepare(
                git_ops, config, 'drift-blocked', 'clash.py', 'x = 2\n',
            )

            await queue.put(lander)
            await wait_responsive(
                verifier.await_entry(1),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='lander: verify entered',
            )

            await queue.put(blocked)
            await wait_responsive(
                git_ops.at_gate.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='blocked request parked inside merge_to_main',
            )
            # Load-bearing precondition: the blocked request is parked INSIDE
            # merge_to_main, so its drift base was stashed while main_position
            # was still 0. Without this, the whole drive proves nothing.
            assert lane.snapshot()['metrics']['landings_total'] == 0, (
                'the lander landed before the blocked request reached its '
                'merge -- the blocked base was stashed too late to be at 0'
            )

            verify_gate.set()
            outcome_lander = await wait_responsive(
                lander.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='lander lands',
            )
            assert outcome_lander.status == 'done', (
                f'expected the lander to land, got {outcome_lander!r}'
            )
            async def _lander_counted() -> None:
                while lane.snapshot()['metrics']['landings_total'] < 1:
                    await asyncio.sleep(0.01)

            await wait_responsive(
                _lander_counted(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='lander counted on the public landings_total',
            )

            git_ops.release_merge.set()
            outcome_blocked = await wait_responsive(
                blocked.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='blocked request conflicts',
            )
            assert outcome_blocked.status == 'conflict', (
                f'expected the blocked request to conflict, got {outcome_blocked!r}'
            )

            metrics = lane.snapshot()['metrics']
            assert metrics['landings_total'] == 1
            drift = metrics['drift_at_detection']
            assert drift['count'] == 1, (
                f'the conflict recorded no drift sample ({drift!r})'
            )
            assert drift['last'] == 1, (
                'a landing consumed another request\'s drift base: the blocked '
                'request started merging at main_position 0 and conflicted at 1, '
                f'so its drift is 1, not {drift["last"]!r}'
            )
