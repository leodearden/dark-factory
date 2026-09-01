"""Tests for merge-verify depth telemetry (task 2340).

Covers:
  step-9  RED   — SpeculativeMergeWorker._verify_frontier_depth() pure unit test
  step-10 GREEN — _verify_frontier_depth() implementation
  step-11 RED   — depth/speculative plumbing: _run_post_merge_verify's
                  pool.dispatch forwarding + _run_inflight_verify caller wiring
  step-12 GREEN — thread depth/speculative through _run_post_merge_verify /
                  _run_inflight_verify / _dispatch_item
  step-13 RED   — speculative_merge event carries depth
  step-14 GREEN — classify_and_merge threads worker._verify_frontier_depth()
                  into the speculative_merge _emit_speculative call
  amend         — TestDispatchItemDepthAheadOfItem: _dispatch_item reads the
                  frozen-frontier count BEFORE the dispatching item joins it
                  (review followup: reviewer_comprehensive/test_coverage)

DEPTH DEFINITION (ε=1890 verify-frontier stack height): depth 0 = a head
verify against real main; depth d = d speculated items already
frozen/verifying ahead of the item joining the frontier.  See
_verify_frontier_depth()'s docstring and test_merge_queue_frozen_prefix.py
for the underlying frozen-prefix model this helper reuses.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    InflightEntry,
    MergedOk,
    MergeRequest,
    RealMergeItem,
    SpeculativeMergeWorker,
    classify_and_merge,
)
from orchestrator.merge_types import (
    InflightStatus,
    InflightVerifyResult,
    QueuedBranch,
)
from orchestrator.verify import VerifyResult


def _sentinel_entry() -> InflightEntry:
    """Opaque, identity-only stand-in for a real ``InflightEntry``.

    Mirrors test_merge_queue_single_writer_asserts.py's ``_sentinel_entry``:
    the tests below only care about the *count* of frozen entries
    (``_verify_frontier_depth`` is a pure ``len()`` delegation), never their
    fields, so a bare ``object()`` cast to ``InflightEntry`` is enough at
    runtime without constructing a real merge-result/lease graph.
    """
    return cast(InflightEntry, object())


def _make_bare_worker() -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for pure unit tests.

    No event loop or real git_ops required — mirrors test_halt_owner.py's
    ``SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())``
    construction style.
    """
    return SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())


# ---------------------------------------------------------------------------
# step-9 RED / step-10 GREEN: _verify_frontier_depth()
# ---------------------------------------------------------------------------


class TestVerifyFrontierDepth:
    """_verify_frontier_depth() == len(_frozen_inflight_entries()) (ε=1890).

    Pure/synchronous delegation test — lightly stubs _frozen_inflight_entries()
    so this test is isolated from the frozen-prefix computation itself
    (already covered by test_merge_queue_frozen_prefix.py) and asserts only
    the depth helper's own wiring.

    RED until step-10 GREEN adds the method.
    """

    def test_empty_frontier_returns_zero(self) -> None:
        """No frozen/verifying entries ahead -> depth 0 (head verify vs real main)."""
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: []
        assert worker._verify_frontier_depth() == 0

    def test_one_frozen_entry_returns_one(self) -> None:
        """One speculated item ahead -> depth 1."""
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: [_sentinel_entry()]
        assert worker._verify_frontier_depth() == 1

    def test_three_frozen_entries_returns_three(self) -> None:
        """Three speculated items ahead -> depth 3."""
        worker = _make_bare_worker()
        worker._frozen_inflight_entries = lambda: [_sentinel_entry() for _ in range(3)]
        assert worker._verify_frontier_depth() == 3


# ---------------------------------------------------------------------------
# step-11 RED / step-12 GREEN: depth/speculative plumbing
# ---------------------------------------------------------------------------


def _make_bare_config() -> OrchestratorConfig:
    return OrchestratorConfig(git=GitConfig(main_branch='main'))


def _make_request(
    task_id: str, branch: str, worktree: Path, config: OrchestratorConfig,
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop.

    Callers must be inside an async test (all tests in this class are).
    """
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


def _make_git_ops_mock() -> MagicMock:
    """Mock git_ops sufficient for _run_post_merge_verify's pre-verify guards.

    Mirrors TestRunPostMergeVerifyRunnerParam._make_git_ops_mock in
    test_merge_queue_concurrent_verify.py.
    """
    mock = MagicMock()
    mock.get_main_sha = AsyncMock(return_value='main-sha')
    mock.get_free_disk_bytes = AsyncMock(return_value=100 * 1024 ** 3)
    mock.cleanup_merge_worktree = AsyncMock()
    mock.create_throwaway_verify_worktree = AsyncMock(return_value='/repo/_throwaway')
    return mock


def _fake_pass_runner(name: str = 'fake-runner') -> MagicMock:
    """Fake RemoteRunner-shaped object whose run_merge_verify always passes."""
    fake = MagicMock()
    fake.name = name
    fake.is_local = False
    fake.run_merge_verify = AsyncMock(return_value=VerifyResult(
        passed=True, test_output='ok', lint_output='', type_output='',
        summary='ok', category='',
    ))
    fake.cancel_verify = AsyncMock(return_value=0)
    fake.probe_clean = AsyncMock(return_value=True)
    return fake


class _CapturingEventStore(EventStore):
    """Capturing EventStore -- records emit() calls without touching sqlite.

    Mirrors _LateArrivalFakeEventStore (test_merge_speculation.py) /
    _FakeEventStore (test_merge_queue_concurrent_verify.py).
    """

    def __init__(self) -> None:
        object.__init__(self)
        self.emitted: list[dict] = []

    def emit(  # type: ignore[override]
        self, event_type, *, task_id=None, phase=None, role=None,
        data=None, cost_usd=None, duration_ms=None,
    ) -> None:
        self.emitted.append({'event_type': event_type, 'data': data or {}})

    def events_of(self, event_type: EventType) -> list[dict]:
        return [e for e in self.emitted if e['event_type'] == event_type]


@pytest.mark.asyncio
class TestRunPostMergeVerifyDepthPlumbing:
    """_run_post_merge_verify threads depth/speculative into pool.dispatch
    (task 2340), which threads them into the emitted merge_verify event.

    RED until step-12 GREEN adds the depth/speculative kwargs.
    """

    async def test_depth_and_speculative_propagate_into_merge_verify_event(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import _run_post_merge_verify

        config = _make_bare_config()
        req = _make_request('t-depth', 'task/t-depth', tmp_path, config)
        git_ops = _make_git_ops_mock()
        es = _CapturingEventStore()

        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={}, max_timeouts=2, max_enospc=1,
            merge_sha='abc123', runner=_fake_pass_runner(), event_store=es,
            depth=3, speculative=True,  # RED: these kwargs don't exist yet
        )

        assert outcome is None  # verify passed
        events = es.events_of(EventType.merge_verify)
        assert len(events) == 1
        assert events[0]['data']['depth'] == 3
        assert events[0]['data']['speculative'] is True

    async def test_default_call_omits_depth_and_speculative_byte_identical(
        self, tmp_path: Path,
    ) -> None:
        """Legacy call form (no depth/speculative kwargs) -> both None in the event."""
        from orchestrator.merge_queue import _run_post_merge_verify

        config = _make_bare_config()
        req = _make_request('t-nodepth', 'task/t-nodepth', tmp_path, config)
        git_ops = _make_git_ops_mock()
        es = _CapturingEventStore()

        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={}, max_timeouts=2, max_enospc=1,
            merge_sha='abc123', runner=_fake_pass_runner(), event_store=es,
        )

        assert outcome is None
        events = es.events_of(EventType.merge_verify)
        assert len(events) == 1
        assert events[0]['data']['depth'] is None
        assert events[0]['data']['speculative'] is None


# ---------------------------------------------------------------------------
# step-7 RED / step-8 GREEN (task 3185, PRD γ): _run_post_merge_verify forwards
# chain_items to ALL THREE pool.dispatch sites.
#
# Unlike depth/speculative (which default to None), chain_items defaults to 1 —
# the smallest TRUTHFUL 1-indexed count of items in a verified tree — so the
# legacy callers (reverify_member_solo, _do_train_merge, merge_gates'
# _reverify_rebased_tree) keep emitting an honest count with no edit.
# ---------------------------------------------------------------------------


def _fail_result(category: str = '', test_output: str = 'boom') -> VerifyResult:
    """A failing VerifyResult with a caller-chosen category/output."""
    return VerifyResult(
        passed=False, test_output=test_output, lint_output='', type_output='',
        summary='fail', category=category,
    )


def _pass_result() -> VerifyResult:
    return VerifyResult(
        passed=True, test_output='ok', lint_output='', type_output='',
        summary='ok', category='',
    )


@pytest.mark.asyncio
class TestRunPostMergeVerifyChainItemsPlumbing:
    """chain_items reaches the initial dispatch AND both retry dispatches.

    A retry re-verifies the SAME tree, so it must carry the SAME chain_items as
    its attempt-0 dispatch.  A retry that silently dropped back to the default
    would understate depth in exactly the rows ε's deep-fail-rate reader keys
    on — it would read a deep retry as an ordinary single-item verify.
    """

    async def _dispatch_calls(
        self, tmp_path: Path, results: list[VerifyResult], **kwargs,
    ) -> list[dict]:
        """Run _run_post_merge_verify with a scripted dispatch and return kwargs.

        *results* is consumed one per ``pool.dispatch`` call (the last is
        repeated if the code dispatches more times than scripted).
        """
        from orchestrator.merge_queue import _run_post_merge_verify

        config = _make_bare_config()
        req = _make_request('t-chain', 'task/t-chain', tmp_path, config)
        git_ops = _make_git_ops_mock()
        # The ENOSPC arm prunes stale merge worktrees before its retry; the
        # shared mock does not stub it, and a bare MagicMock is not awaitable.
        git_ops.prune_stale_merge_worktrees = AsyncMock(return_value=[])
        captured: list[dict] = []
        seq = list(results)

        async def _fake_dispatch(_self, merge_sha, spec, **dispatch_kwargs):
            captured.append(dispatch_kwargs)
            return seq.pop(0) if len(seq) > 1 else seq[0]

        with patch(
            'orchestrator.verify_runner.VerifyRunnerPool.dispatch', _fake_dispatch,
        ):
            await _run_post_merge_verify(
                git_ops, req, tmp_path,
                timeouts={}, enospc_retries={}, max_timeouts=2, max_enospc=1,
                merge_sha='abc123', runner=_fake_pass_runner(),
                event_store=_CapturingEventStore(),
                **kwargs,
            )
        return captured

    async def test_initial_dispatch_carries_chain_items(self, tmp_path: Path) -> None:
        calls = await self._dispatch_calls(
            tmp_path, [_pass_result()], chain_items=4,
        )

        assert len(calls) == 1
        assert calls[0]['chain_items'] == 4

    async def test_enospc_retry_carries_the_same_chain_items(
        self, tmp_path: Path,
    ) -> None:
        """The ENOSPC retry dispatch (attempt=1) must not drop to the default."""
        enospc = _fail_result(test_output='fatal: No space left on device')
        calls = await self._dispatch_calls(
            tmp_path, [enospc, _pass_result()], chain_items=5,
        )

        assert len(calls) == 2, 'expected an ENOSPC retry dispatch'
        assert calls[1]['attempt'] == 1
        assert [c['chain_items'] for c in calls] == [5, 5]

    async def test_infra_transient_retry_carries_the_same_chain_items(
        self, tmp_path: Path,
    ) -> None:
        """The classified-infra-transient retry dispatch likewise."""
        from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES

        category = sorted(str(c) for c in INFRA_TRANSIENT_CATEGORIES)[0]
        transient = _fail_result(category=category)
        calls = await self._dispatch_calls(
            tmp_path, [transient, _pass_result()], chain_items=3,
        )

        assert len(calls) == 2, f'expected an infra-transient retry for {category!r}'
        assert calls[1]['attempt'] >= 1
        assert [c['chain_items'] for c in calls] == [3, 3]

    async def test_legacy_call_form_defaults_to_one(self, tmp_path: Path) -> None:
        """No chain_items kwarg → 1, keeping the train / merge_gates callers
        byte-identical in MEANING: each verifies exactly one item's tree."""
        calls = await self._dispatch_calls(tmp_path, [_pass_result()])

        assert len(calls) == 1
        assert calls[0]['chain_items'] == 1


@pytest.mark.asyncio
class TestRunInflightVerifyDepthWiring:
    """_run_inflight_verify forwards depth + item.speculative into
    _run_post_merge_verify (task 2340).

    RED until step-12 GREEN adds the depth kwarg on _run_inflight_verify
    and forwards depth/speculative into its _run_post_merge_verify call.
    """

    async def test_depth_and_speculative_forwarded_to_run_post_merge_verify(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.verify_runner import HostLease

        config = _make_bare_config()
        req = _make_request('t-wire', 'task/t-wire', tmp_path, config)
        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(
                success=True, merge_commit='deadbeef', merge_worktree=tmp_path,
            ),
            merge_wt=tmp_path,
            base_sha='dead' * 10,
            speculative=True,
        )
        lease = HostLease(name='laptop', runner=_fake_pass_runner(), is_local=False)

        captured: dict = {}

        async def _fake_run_post_merge_verify(*_args, **kwargs):
            captured.update(kwargs)
            return None  # pass

        worker = SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            _fake_run_post_merge_verify,
        ):
            await worker._run_inflight_verify(item, lease, depth=2)  # RED: no depth kwarg yet

        assert captured.get('depth') == 2
        assert captured.get('speculative') is True


# ---------------------------------------------------------------------------
# step-9 RED / step-10 GREEN (task 3185, PRD γ): _dispatch_item computes a
# TRUTHFUL chain_items and threads it through _run_inflight_verify.
#
# chain_items is a fact about the TREE ACTUALLY VERIFIED, so it is derived from
# _verify_frontier_depth() directly and NEVER from the local `depth` variable,
# which a firing ProbePlacement may have relabelled into an attribution fact
# about a stack that was never verified.  That divergence is the whole point of
# the field — see ProbePlacement's KNOWN PHASE-1 LIMITATION note.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunInflightVerifyChainItemsWiring:
    """_run_inflight_verify forwards chain_items into _run_post_merge_verify."""

    def _item_and_lease(self, tmp_path: Path, *, speculative: bool = True):
        from orchestrator.verify_runner import HostLease

        config = _make_bare_config()
        req = _make_request('t-ci-wire', 'task/t-ci-wire', tmp_path, config)
        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(
                success=True, merge_commit='deadbeef', merge_worktree=tmp_path,
            ),
            merge_wt=tmp_path,
            base_sha='dead' * 10,
            speculative=speculative,
        )
        lease = HostLease(name='laptop', runner=_fake_pass_runner(), is_local=False)
        return item, lease

    async def _captured_kwargs(self, tmp_path: Path, **verify_kwargs) -> dict:
        item, lease = self._item_and_lease(tmp_path)
        captured: dict = {}

        async def _fake_run_post_merge_verify(*_args, **kwargs):
            captured.update(kwargs)
            return None  # pass

        worker = SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())
        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            _fake_run_post_merge_verify,
        ):
            await worker._run_inflight_verify(item, lease, **verify_kwargs)
        return captured

    async def test_chain_items_is_forwarded(self, tmp_path: Path) -> None:
        captured = await self._captured_kwargs(tmp_path, depth=2, chain_items=3)

        assert captured.get('chain_items') == 3

    async def test_omitted_chain_items_defaults_to_one(self, tmp_path: Path) -> None:
        """Keeps _merge_queue_harness.drive_verify_and_advance and the other
        direct-call test paths byte-identical."""
        captured = await self._captured_kwargs(tmp_path, depth=2)

        assert captured.get('chain_items') == 1


@pytest.mark.asyncio
class TestDispatchItemComputesTruthfulChainItems:
    """_dispatch_item emits chain_items in CHAIN-ITEM units: a flat 1.

    A dispatch contributes exactly one chain item — the dispatching item
    itself, chain item #1 — and only the deep-chain arm of
    ``_run_inflight_verify`` adds more (``1 + len(chain.links)``).  That holds
    for slot 1 (merged onto REAL MAIN) and slot 2 (merged onto the frozen tip)
    alike, and at every verify frontier: the value is deliberately
    frontier-INDEPENDENT so that ``chain_items >= 2`` is a sound
    deep-verify discriminator and ``chain_cap=0`` can never emit it.

    The frozen-prefix height is not lost — it is what the separate,
    always-present ``depth`` field has always carried.
    """

    def _worker_with_frontier(self, frontier: int) -> SpeculativeMergeWorker:
        worker = SpeculativeMergeWorker(git_ops=MagicMock(), queue=asyncio.Queue())
        worker._frozen_inflight_entries = lambda: [  # type: ignore[method-assign]
            _sentinel_entry() for _ in range(frontier)
        ]
        return worker

    async def _dispatch_kwargs(
        self, tmp_path: Path, *, frontier: int, speculative: bool, probe=None,
    ) -> dict:
        """Drive _dispatch_item far enough to capture _run_inflight_verify's kwargs."""
        from orchestrator.verify_runner import HostLease

        config = _make_bare_config()
        req = _make_request('t-ci-disp', 'task/t-ci-disp', tmp_path, config)
        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(
                success=True, merge_commit='cafebabe', merge_worktree=tmp_path,
            ),
            merge_wt=tmp_path,
            base_sha='dead' * 10,
            speculative=speculative,
        )
        worker = self._worker_with_frontier(frontier)
        if probe is not None:
            worker._probe_verify_placement = lambda _i: probe  # type: ignore[method-assign]

        captured: dict = {}

        async def _fake_run_inflight_verify(_item, _lease, **kwargs):
            captured.update(kwargs)
            return InflightVerifyResult(
                outcome=None, merge_wt=None, status=InflightStatus.REQUEUED,
            )

        worker._run_inflight_verify = _fake_run_inflight_verify  # type: ignore[method-assign]
        lease = HostLease(name='laptop', runner=_fake_pass_runner(), is_local=False)
        worker._acquire_host_lease_for = AsyncMock(  # type: ignore[method-assign]
            return_value=(lease, None),
        )

        entry = await worker._dispatch_item(item)
        if entry is not None and entry.verify_task is not None:
            await entry.verify_task
        return captured

    async def test_non_speculative_head_item_is_one(self, tmp_path: Path) -> None:
        """Slot 1 with an empty frontier: its tree is exactly itself."""
        captured = await self._dispatch_kwargs(
            tmp_path, frontier=0, speculative=False,
        )

        assert captured['chain_items'] == 1

    async def test_non_speculative_item_with_a_verify_in_flight_is_still_one(
        self, tmp_path: Path,
    ) -> None:
        """The case a naive ``depth + 1`` gets WRONG.

        A non-speculative re-merge dispatched while another verify is in flight
        has frontier == 1, but it is merged onto REAL MAIN — its tree contains
        only itself, so chain_items is 1, not 2.
        """
        captured = await self._dispatch_kwargs(
            tmp_path, frontier=1, speculative=False,
        )

        assert captured['chain_items'] == 1

    async def test_speculative_item_is_one_in_chain_item_units(
        self, tmp_path: Path,
    ) -> None:
        """CHAIN-ITEM units, frontier-INDEPENDENT.

        A dispatch that builds no chain contributes exactly one chain item —
        itself — so a slot-2 item emits 1 at every frontier, exactly as a
        slot-1 item does.  Folding the frozen-prefix height in here would make
        ``chain_items >= 2`` fire on ordinary adjacent verifies and destroy its
        value as the deep-verify discriminator that
        scripts/merge-deep-canary-predicate.sh:84 keys on.  The frontier height
        is not lost: it is carried, unchanged, by the separate ``depth`` field.
        """
        for frontier in (0, 1, 3):
            captured = await self._dispatch_kwargs(
                tmp_path, frontier=frontier, speculative=True,
            )
            assert captured['chain_items'] == 1, f'frontier={frontier}'
            assert captured['depth'] == frontier, 'the height still rides `depth`'

    async def test_a_firing_probe_relabels_depth_but_not_chain_items(
        self, tmp_path: Path,
    ) -> None:
        """The assertion that encodes "supersedes reliance on the broken label".

        A firing ProbePlacement overrides the dispatched ``depth`` (that is its
        entire job — an attribution fact about an already-built stack), but the
        probe never redirected what is VERIFIED, so no chain item was added and
        ``chain_items`` stays at the one item this dispatch actually exercises.

        The point is if anything sharper in chain-item units: ``depth`` is
        relabelled all the way to the probe's 5 while ``chain_items`` does not
        move off 1 — the two fields visibly answer different questions.
        """
        from orchestrator.merge_queue import ProbePlacement

        captured = await self._dispatch_kwargs(
            tmp_path, frontier=1, speculative=True,
            probe=ProbePlacement(depth=5, base='f' * 40),
        )

        assert captured['depth'] == 5             # relabelled, as today
        assert captured['chain_items'] == 1       # one item verified, untouched


# ---------------------------------------------------------------------------
# step-13 RED / step-14 GREEN: speculative_merge event carries depth
#
# classify_and_merge drives real git operations (branch-presence guard,
# already-merged check, merge_to_main), so this needs a real GitOps against a
# tmp git repo rather than the MagicMock git_ops used by the plumbing tests
# above.  Fixture quartet (git_repo/git_config/git_ops/config) + the
# _make_branch_with_file helper are copied from test_merge_guard_pipeline.py
# — there is no shared conftest git_ops fixture; per-file duplication is the
# established convention (see test_merge_queue_resource_audit.py's module
# docstring).
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        # Tests use a tmp repo with no real remote; disabling the push avoids
        # per-test subprocess noise.
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path."""
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


@pytest.mark.asyncio
class TestSpeculativeMergeEventDepth:
    """classify_and_merge's speculative_merge event carries depth (task 2340).

    RED until step-14 GREEN adds depth=worker._verify_frontier_depth() to the
    _emit_speculative(EventType.speculative_merge, ...) call in
    classify_and_merge (merge_queue.py, inside the
    `if speculative and isinstance(worker, SpeculativeMergeWorker):` guard).
    """

    async def test_speculative_merge_event_carries_depth(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        worktree = await _make_branch_with_file(
            git_ops, 'spec-depth-1', 'f.py', 'x = 1\n',
        )
        es = _CapturingEventStore()
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=es)
        # Two speculated items already frozen/verifying ahead of this one
        # joining the frontier -> depth 2 (see _verify_frontier_depth()).
        worker._frozen_inflight_entries = lambda: [_sentinel_entry() for _ in range(2)]

        req = _make_request('spec-depth-1', 'spec-depth-1', worktree, config)
        main_sha = await git_ops.get_main_sha()

        result = await classify_and_merge(
            worker, req, main_sha, speculative=True, started_monotonic=time.monotonic(),
        )

        assert isinstance(result, MergedOk)
        events = es.events_of(EventType.speculative_merge)
        assert len(events) == 1
        # _emit_speculative str-converts every data value.
        assert events[0]['data']['depth'] == '2'
        assert events[0]['data']['base_sha'] == main_sha

        if result.merge_wt:
            await git_ops.cleanup_merge_worktree(result.merge_wt)
