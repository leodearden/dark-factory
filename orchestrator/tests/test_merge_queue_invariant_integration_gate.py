"""Integration gate π: fault-injection pipeline run asserting all new
invariant surfaces (B+H leaf).

This file is the G2 boundary-test leaf for the merge-queue modularization
invariants PRD (plans/merge-queue-modularization-invariants-prd.md task π;
the §Boundary-test sketch rows 1-12 below are this task's spec).  It drives a
REAL SpeculativeMergeWorker against a real tmp git repo with fake/instrumented
runners (no real ssh/build) through merge bursts with injected faults, and
after EACH scenario asserts the 5-surface QUIESCENCE contract (see
``_assert_quiescent``):

  (a) every tracked request's result Future has resolved (done or cancelled).
  (b) worker.speculation_accounting_violations() == [] (I4, task ι=1994).
  (c) worker.worktree_ledger_violations() == [] (I6, task ι=1994).
  (d) the request-liveness ledger is empty after sweep_resolved() (η=1992).
  (e) worker.two_layer_invariants(main_sha) == [] with a REAL main_sha
      (λ=1895, extended by ξ=1999 I5).

SCOPE — TEST-ONLY.  merge_queue.py and its split modules (merge_types,
merge_request_ledger, merge_gates, merge_liveness, merge_shadow, merge_drift,
suffix_graph, merge_speculation_controller) are BEHAVIOUR-FROZEN for this
batch: every surface exercised below was already shipped by the prerequisite
tasks α through ο (task ο / 2000, the immediate prereq, is DONE).  This is a
COMPOSITION gate — if a scenario surfaces a GENUINE production defect,
escalate (category=design_concern or scope_violation); do NOT edit production
code here — that would widen the concurrency lock and conflict with the
frozen seam.

STALE-OFFSET WARNING: the PRD's own ``:NNNN`` line citations (e.g. ``:5927``,
``:4936``) are STALE by thousands of lines — merge_queue.py shrank from
~12.9k to ~8.9k lines across the α-ο split tasks.  Always locate symbols BY
NAME (grep/search), never trust a PRD line offset.

§Boundary-test sketch rows 1-12 (task π's spec)
------------------------------------------------
  1.  Speculative head-failure cascade re-lands N+1 with both Futures
      resolved (submission order preserved).
  2.  RunnerUnavailable quarantines the host and re-dispatches (not a chain
      failure).
  3.  Operator-halt mid-verify → REQUEUED (req re-queued, Future still
      pending); unhalt + re-verify → 'done'.
  4.  Abandoned sole-waiter (Future cancelled mid-verify) → DROPPED, merge_wt
      cleaned, no leaked speculation slot.
  5.  Wedged verify (armed-but-unresolved ledger entry) → liveness escalation
      (category='merge_request_stuck') within the heartbeat window [η=1992].
  6.  Forced permit leak → speculation_accounting_violations() non-empty +
      streak-gated resource-leak escalation (category='merge_resource_leak')
      [ι=1994].
  7.  Ill-formed item shape raises at CONSTRUCTION time (structural TypeError
      post-ο; RealMergeItem/DecidedItem are now disjoint dataclasses) — plus
      the surviving InflightEntry passthrough_outcome ValueError [ε=1890].
  8.  CAS-retry rebased landing carries merged_branch_tip through the
      dataclasses.replace rebuild — no phantom POST_MERGE_EQUIVALENCE_FAILED
      (1928 regression pin) [ε=1890 I3].
  9.  Guard-matrix equivalence: the MERGER path (classify_and_merge via
      worker.run()) and the _remerge path produce identical MergeOutcome +
      merge_attempt event streams [κ=1995].
  10. Injected verify-base/frozen-tip mismatch surfaces a violation string
      that is a member of two_layer_invariants() [ξ=1999 I5].
  11. snapshot() keys are additive-only across the whole module-split batch
      (13 keys total, resource_audit being the newest).
  12. InFlightMergeRegistry.acquire/attach fan-out mirror + the
      coalesce_or_enqueue_merge_request dispatched/in_flight/alias contract
      are unchanged by the split.

Fake/instrumented runners only — no real ssh/build/merge-to-main:
  - run_scoped_verification / run_merge_verify / advance_main / rebase_onto_main
    stubbed or AsyncMock'd per scenario.
  - Three documented worker-driving styles (mirrors the λ=1895 precedent):
      A) full worker.run() feeding the public worker._queue (row 1 cascade).
      B) direct _verifier_loop() with a trailing None sentinel via
         worker._verifier_queue (row 4 DROPPED chokepoint).
      C) direct _run_inflight_verify(item, lease) / _finalize_inflight(entry)
         / _verify_and_advance(item) compat shim (rows 2, 3, 4, 8).

Pattern precedent: test_merge_queue_two_layer_integration.py (λ=1895) — the
fixtures/builders/harness factory below are ported/adapted from there; the
gated-runner + two-host fault-injection helpers are ported from
test_merge_queue_concurrent_verify.py (γ=1735).
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.harness import Harness
from orchestrator.merge_queue import (
    DecidedItem,
    InflightEntry,
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
    SpeculativeItem,
    SpeculativeMergeWorker,
)
from orchestrator.run_store import RunStore
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import HostAllocator

# ── Module-level sentinel ─────────────────────────────────────────────────────

_SENTINEL_VERIFY_TASK = object()  # noqa: PD901
"""Sentinel for verify_task in pure unit tests.

frozen_prefix() checks ``e.verify_task is not None`` only, so any non-None
object doubles as a verifying-entry marker in tests that do not need a real
asyncio.Task.
"""


# ── Repo seeding ──────────────────────────────────────────────────────────────


async def _setup_repo(repo: Path) -> None:
    """Initialise a git repo with a single commit (README.md) on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


# ── Fixtures ──────────────────────────────────────────────────────────────────


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


# ── Request / item / entry builders ──────────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    lane: Literal['normal', 'high'] = 'normal',
    *,
    merge_first_enqueued_at: float | None = 1000.0,
    request_id: str | None = None,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
    The optional *request_id* kwarg lets tests pin a stable identity for
    registry/alias assertions; when omitted a fresh UUID is auto-generated.
    """
    kwargs: dict = {}
    if request_id is not None:
        kwargs['request_id'] = request_id
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane=lane,
        merge_first_enqueued_at=merge_first_enqueued_at,
        **kwargs,
    )


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring)."""
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


def _make_fake_item(
    task_id: str,
    *,
    base_sha: str = 'aaaa0000',
    merge_commit: str | None = 'bbbb1111',
    config: OrchestratorConfig,
    git_repo: Path,
) -> tuple[MergeRequest, SpeculativeItem]:
    """Build a (MergeRequest, SpeculativeItem) pair from fake SHAs.

    Does NOT create real git branches — suitable for pure-unit tests that only
    exercise frozen-prefix / frozen_prefix_tip accessor methods without real
    git operations.  Must be called from an async context (for the future).
    """
    req = _make_req(task_id, f'task/{task_id}', config, git_repo)
    item: SpeculativeItem
    if merge_commit is not None:
        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(success=True, merge_commit=merge_commit),
            merge_wt=Path('/fake/merge-wt'),
            base_sha=base_sha,
            speculative=False,
        )
    else:
        item = DecidedItem(
            request=req,
            immediate_outcome=MergeOutcome('conflict', reason='fake-no-merge-commit'),
            base_sha=base_sha,
            speculative=False,
        )
    return req, item


def _make_inflight_entry(
    item: SpeculativeItem,
    *,
    verifying: bool = True,
) -> InflightEntry:
    """Build an InflightEntry for unit tests.

    verifying=True  → verify_task set to _SENTINEL_VERIFY_TASK (any non-None
                       object; frozen_prefix() only checks ``is not None``).
    verifying=False → verify_task=None (passthrough entry; excluded from the
                       frozen prefix by §5.3 definition).
    """
    return InflightEntry(
        item=item,
        lease=None,
        verify_task=_SENTINEL_VERIFY_TASK if verifying else None,  # type: ignore[arg-type]
        merge_wt=None,
        was_speculative=False,
        phase='verifying',
    )


# ── Branch / commit helpers ───────────────────────────────────────────────────


async def _create_branch_editing(
    repo: Path,
    branch_name: str,
    filename: str,
    content: str,
    base_branch: str = 'main',
) -> str:
    """Create a branch editing *filename* with *content*; return the branch SHA."""
    await _run(['git', 'checkout', '-b', branch_name], cwd=repo)
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Edit {filename} in {branch_name}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    await _run(['git', 'checkout', base_branch], cwd=repo)
    return sha.strip()


async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path.

    Ported from test_merge_queue_concurrent_verify.py (per-file duplication
    convention) — complements ``_create_branch_editing`` above for scenarios
    that need a real materialised worktree rather than just a branch SHA.
    """
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


# ── Gated-runner + fake-remote + two-host allocator injection ────────────────
# Ported from test_merge_queue_concurrent_verify.py (γ=1735).


def _mock_verify_result(passed: bool) -> VerifyResult:
    """Return a VerifyResult with the given pass/fail status."""
    return VerifyResult(
        passed=passed,
        test_output='ok' if passed else 'FAILED',
        lint_output='',
        type_output='',
        summary='ok' if passed else 'fail',
        category='' if passed else 'test_failure',
    )


def _gated_runner(
    gate_release: asyncio.Event,
    gate_entered: asyncio.Event | None = None,
    *,
    passed: bool = True,
    name: str = 'gated',
) -> MagicMock:
    """Return a fake runner whose run_merge_verify blocks until gate_release is set.

    *gate_entered* (optional): set when the first call starts, so tests can
    await both enters before releasing.  Subsequent calls pass immediately
    (gate is already set after first call).

    This is a fake RemoteRunner shaped object: name, is_local=False,
    run_merge_verify/cancel_verify/probe_clean as AsyncMocks.
    """
    _first_blocked = [False]

    async def _side_effect(*args: Any, **kwargs: Any) -> VerifyResult:
        if not _first_blocked[0]:
            _first_blocked[0] = True
            if gate_entered is not None:
                gate_entered.set()
            await gate_release.wait()
        return _mock_verify_result(passed)

    runner = MagicMock()
    runner.name = name
    runner.is_local = False
    runner.run_merge_verify = AsyncMock(side_effect=_side_effect)
    runner.cancel_verify = AsyncMock(return_value=0)
    runner.probe_clean = AsyncMock(return_value=True)
    return runner


def _make_fake_remote(name: str = 'laptop') -> MagicMock:
    """Build a fake RemoteRunner (MagicMock) with async cancel/probe/verify."""
    fake = MagicMock()
    fake.name = name
    fake.is_local = False
    fake.run_merge_verify = AsyncMock(return_value=_mock_verify_result(True))
    fake.cancel_verify = AsyncMock(return_value=0)  # 0 = clean cancel
    fake.probe_clean = AsyncMock(return_value=True)
    return fake


def _inject_two_host_allocator(
    worker: SpeculativeMergeWorker,
    fake_remote: Any,
) -> HostAllocator:
    """Inject a two-host HostAllocator (local + fake_remote) onto a worker.

    Returns the allocator so callers can introspect slot state or verify calls.
    """
    allocator = HostAllocator([fake_remote], quarantine=worker._runner_quarantine)
    worker._host_allocator = allocator
    return allocator


# ── Harness factory + real-worker attachment ─────────────────────────────────
# Ported from test_merge_queue_two_layer_integration.py (λ=1895).


def _make_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Harness with a real Scheduler + EscalationQueue and a spy RunStore.

    Mirrors test_harness_no_landings_breaker._make_harness.
    Returns (harness, mock_run_store).
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-integration-0001'
    harness.event_store = EventStore(tmp_path / 'events.db', 'run-integration-0001')
    harness._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    return harness, mock_run_store


def _attach_real_worker(harness: Harness, git_ops: GitOps) -> SpeculativeMergeWorker:
    """Attach a REAL SpeculativeMergeWorker to the harness as harness._merge_worker.

    Wires the harness's real EscalationQueue into the worker so any escalation
    surfaced by the worker (liveness/resource-audit) lands in the same queue
    the harness observes.

    Returns the attached worker for direct manipulation in tests.
    """
    worker = SpeculativeMergeWorker(
        git_ops,
        asyncio.Queue(),
        escalation_queue=harness._escalation_queue,
    )
    harness._merge_worker = worker
    return worker


# ── Fake escalation queue ──────────────────────────────────────────────────────
# Ported verbatim from test_merge_queue_request_liveness.py /
# test_merge_queue_resource_audit.py (per-file duplication convention).


class _FakeEscalationQueue:
    """Minimal fake escalation queue with a ``.submitted`` list for assertions."""

    def __init__(self, *, open_l1: bool = False):
        self._open_l1 = open_l1
        self._seq = 0
        self.submitted: list = []

    def has_open_l1(self, task_id: str) -> bool:  # noqa: ARG002
        return self._open_l1

    def make_id(self, task_id: str) -> str:
        self._seq += 1
        return f'esc-{self._seq}'

    def submit(self, esc) -> None:
        self.submitted.append(esc)

    def open_it(self):
        """Simulate a prior open L1 (for dedup tests)."""
        self._open_l1 = True


# ── Shared quiescence helper ──────────────────────────────────────────────────


def _assert_quiescent(
    worker: SpeculativeMergeWorker,
    main_sha: str,
    requests: list[MergeRequest],
) -> None:
    """Assert the 5-surface QUIESCENCE contract holds for *worker*.

    Called after each scenario to confirm the pipeline returned to a clean
    resting state with no leaked permits, worktrees, ledger entries, or
    unresolved Futures:

      (a) every request in *requests* has resolved (done or cancelled) — no
          dangling in-flight work left over from the scenario.
      (b) worker.speculation_accounting_violations() == [] — I4 permit/cap
          conservation holds.  Requires *worker* to still be ``_running``
          (both accounting methods short-circuit to [] when not running —
          see their docstrings — so a stopped worker would make this
          assertion vacuous, not meaningful).
      (c) worker.worktree_ledger_violations() == [] — I6 on-disk
          ``_merge-*`` worktree ledger is fully accounted for.  Also
          requires a running worker (same short-circuit).
      (d) the request-liveness ledger is empty AFTER sweeping resolved
          entries.  Resolution is detected PASSIVELY — RequestLedger has no
          on-resolve hook, so a resolved request stays armed until
          ``sweep_resolved()`` runs; calling it here before ``is_empty()``
          is required, not optional.
      (e) worker.two_layer_invariants(main_sha) == [] — *main_sha* MUST be a
          REAL sha, never 'unknown': the base-chain and verify-base
          sub-checks are silently skipped for the 'unknown' sentinel, which
          would make this assertion pass vacuously rather than meaningfully.
    """
    for req in requests:
        assert req.result.done() or req.result.cancelled(), (
            f'Expected request {req.request_id!r} (task {req.task_id!r}) to '
            f'have resolved (done or cancelled) at quiescence, but it is '
            f'still pending'
        )

    spec_violations = worker.speculation_accounting_violations()
    assert spec_violations == [], (
        f'speculation_accounting_violations() non-empty at quiescence: {spec_violations!r}'
    )

    wt_violations = worker.worktree_ledger_violations()
    assert wt_violations == [], (
        f'worktree_ledger_violations() non-empty at quiescence: {wt_violations!r}'
    )

    # Resolution is passive — sweep before asserting emptiness (see
    # RequestLedger.sweep_resolved's docstring).
    worker._request_ledger.sweep_resolved()
    assert worker._request_ledger.is_empty(), (
        f'request-liveness ledger non-empty at quiescence: '
        f'{worker._request_ledger.open_request_ids()!r}'
    )

    assert main_sha and main_sha != 'unknown', (
        f'_assert_quiescent requires a REAL main_sha, got {main_sha!r}'
    )
    tli_violations = worker.two_layer_invariants(main_sha)
    assert tli_violations == [], (
        f'two_layer_invariants({main_sha!r}) non-empty at quiescence: {tli_violations!r}'
    )


# ── step-01 RED: shared quiescence helper + healthy-at-rest ─────────────────


@pytest.mark.asyncio
class TestQuiescenceContract:
    """The 5-surface quiescence contract asserted by _assert_quiescent.

    RED until the helper/module lands. Every §Boundary-test scenario below
    calls _assert_quiescent after driving its fault — this class pins the
    trivial healthy-at-rest baseline: a freshly-built, still-running worker
    with no submitted work is already quiescent.
    """

    async def test_healthy_running_worker_is_quiescent(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """A freshly-built running worker with no work is quiescent."""
        worker = _make_worker(git_ops)
        assert worker._running is True, 'precondition: a fresh worker starts running'

        main_sha = await git_ops.get_main_sha()

        _assert_quiescent(worker, main_sha, [])

        snap = worker.snapshot()
        assert snap['resource_audit'] == {
            'speculation_accounting': [],
            'worktree_ledger': [],
        }


# ── step-03 RED / step-04 GREEN: Row 1 — speculative head-failure cascade ───


@pytest.mark.asyncio
class TestScenario1SpeculativeCascade:
    """Row 1: speculative head-failure cascade re-lands N+1 (submission order
    preserved), both Futures resolved.

    N (head, LOCAL verify) FAILS with a failing VerifyResult (not an
    exception) — a genuine chain failure, so N resolves permanently
    'blocked' (``_finalize_inflight``'s FAIL/skip branch; NOT retried — only
    RUNNER_UNAVAILABLE is transient/re-dispatched).  N+1 (speculative
    downstream, REMOTE verify) is still in-flight when N fails: the
    _verifier_loop head-failure cascade must cancel N+1's in-flight remote
    verify, re-merge N+1 onto ACTUAL main (not N's now-abandoned speculative
    commit / N's failure), and re-dispatch it for a fresh verify — N+1 lands
    'done' independently of N's failure.

    Style A: drives worker.run() over the public worker._queue, with a
    two-host allocator (local + gated remote) so N and N+1 verify
    concurrently — the precondition for the cascade to matter.

    Modelled on TestCascadeFiresRemoteCancel (failing-VerifyResult head, not
    RunnerUnavailable — including its ``outcome_a.status not in ('done',
    'already_merged')`` assertion shape) and TestRunnerUnavailableHeadCascade
    (gated-runner remote + file-presence assertions), both in
    test_merge_queue_concurrent_verify.py.  The cascade's remote-cancel call
    is asserted directly (``cancel_verify.assert_called()``) rather than via
    ``worker._remerge_occurred`` — that flag is dispatch-transient (reset to
    False by the immediate clean re-dispatch of the re-merged N+1; see
    _dispatch_item's Mechanism-2 staleness check) so it is not a stable
    post-hoc signal once the scenario has fully drained.
    """

    async def test_head_failure_cascade_relands_both_requests(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """N fails permanently (local, gated); N+1 (remote) survives via cascade."""
        # ── N's gated local verify: FAILS (failing VerifyResult) on call 0 ───
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        _local_calls: list[int] = [0]

        async def _local_verify_side_effect(*args: Any, **kwargs: Any) -> VerifyResult:
            call = _local_calls[0]
            _local_calls[0] += 1
            if call == 0:
                # N's initial verify: wait for gate, then FAIL (not raise).
                gate_a_entered.set()
                await gate_a_release.wait()
                return _mock_verify_result(False)
            # Re-dispatched verifies (N and N+1 after cascade re-merge): pass.
            return _mock_verify_result(True)

        # ── N+1's remote verify: gated, then passes ──────────────────────────
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='remote-cascade1',
        )

        # ── Branches ─────────────────────────────────────────────────────────
        wt_a = await _make_branch_with_file(git_ops, 'task/casc1-a', 'casc1_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/casc1-b', 'casc1_b.py', 'b = 2\n')

        # ── Worker + two-host allocator (N local, N+1 remote) ────────────────
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        _inject_two_host_allocator(worker, gated_remote)

        loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='casc1-a', branch='task/casc1-a', worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='casc1-b', branch='task/casc1-b', worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        outcome_a: MergeOutcome | None = None
        outcome_b: MergeOutcome | None = None

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _local_verify_side_effect,
        ):
            worker_task = asyncio.create_task(worker.run())

            await q.put(req_a)
            await q.put(req_b)

            # Wait for BOTH verifies to enter — confirms true concurrent
            # overlap (the cascade only matters if N+1 is genuinely in-flight
            # when N fails).
            await asyncio.wait_for(gate_a_entered.wait(), timeout=15.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=15.0)

            # N fails → _finalize_inflight(N) returns False → head-failure
            # cascade fires for N+1: cancels its in-flight verify, re-merges
            # it onto actual main, and re-dispatches both.
            gate_a_release.set()
            # Release N+1's gate too, so the (possibly still-running) gated
            # task can unblock even if cancel() arrives slightly late.
            gate_b_release.set()

            with contextlib.suppress(TimeoutError):
                outcome_a = await asyncio.wait_for(req_a.result, timeout=15.0)
            with contextlib.suppress(TimeoutError):
                outcome_b = await asyncio.wait_for(req_b.result, timeout=15.0)

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)

        # ── N (head): must FAIL permanently — a genuine VerifyResult failure
        # is a real chain failure, not a transient RUNNER_UNAVAILABLE that
        # gets re-dispatched (see _finalize_inflight's FAIL/skip branch).
        assert outcome_a is not None and outcome_a.status not in ('done', 'already_merged'), (
            f'Expected N to fail (genuine VerifyResult failure, not '
            f'RunnerUnavailable), got status={outcome_a.status!r}.'
        )

        # ── N+1 (speculative downstream): must land 'done' after cascade ────
        # even though its "head" (N) permanently failed — the cascade
        # re-merges N+1 against ACTUAL main independently of N's fate.
        assert outcome_b is not None and outcome_b.status == 'done', (
            f'Expected N+1 (speculative downstream) to resolve "done" after '
            f'cascade re-merge re-dispatch, got {outcome_b!r}. If the cascade '
            f'did not fire, N+1 would stay in _inflight with a stale '
            f'speculative commit and eventually fail CAS or time out.'
        )

        # ── Cascade cancelled N+1's in-flight remote verify ──────────────────
        gated_remote.cancel_verify.assert_called()

        # ── Main has N+1's file but NOT N's (N failed and was never merged;
        # N+1 landed independently via the cascade re-merge) ─────────────────
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'casc1_a.py' not in main_files, (
            'N (casc1_a.py) failed verification and must NOT be on main'
        )
        assert 'casc1_b.py' in main_files, (
            'N+1 (casc1_b.py) must be on main after cascade re-merge/re-verify'
        )

        # ── Quiescence: I4 (speculation accounting) and the other 4 surfaces
        # all hold once the cascade has fully drained.
        main_sha = await git_ops.get_main_sha()
        _assert_quiescent(worker, main_sha, [req_a, req_b])
