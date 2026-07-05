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
import logging
from pathlib import Path
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import AdvanceOutcome, GitOps, MergeResult, _run
from orchestrator.harness import Harness
from orchestrator.merge_queue import (
    DecidedItem,
    InflightEntry,
    InflightStatus,
    InflightVerifyResult,
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
    SpeculativeItem,
    SpeculativeMergeWorker,
    item_merge_wt,
)
from orchestrator.run_store import RunStore
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import HostAllocator, HostLease, RunnerUnavailable

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


async def _make_merged_item(
    git_ops: GitOps,
    config: OrchestratorConfig,
    branch: str,
    filename: str,
    content: str,
    *,
    speculative: bool = False,
) -> tuple[MergeRequest, RealMergeItem]:
    """Create a REAL merged RealMergeItem on a fresh branch (style C driving).

    Ported from TestRunInflightVerifyRemoteCancelOnAbort._make_merged_item
    (test_merge_queue_concurrent_verify.py) — merges *branch* to main via a
    real ``git_ops.merge_to_main`` call (no queue/dispatch involved) so rows
    2-4/8 can drive ``_run_inflight_verify``/``_finalize_inflight`` directly
    against a real merge commit.  Does NOT advance main — the caller decides
    whether/how to finalize.
    """
    wt = await _make_branch_with_file(git_ops, branch, filename, content)
    loop = asyncio.get_running_loop()
    req = MergeRequest(
        task_id=branch, branch=branch, worktree=wt,
        pre_rebased=False, task_files=None, module_configs=[],
        config=config, result=loop.create_future(), lane='normal',
    )
    merge_result = await git_ops.merge_to_main(wt, branch)
    assert merge_result.success and merge_result.merge_commit
    assert merge_result.merge_worktree is not None
    base_sha = await git_ops.get_main_sha()
    item = RealMergeItem(
        request=req,
        merge_result=merge_result,
        merge_wt=merge_result.merge_worktree,
        base_sha=base_sha,
        speculative=speculative,
    )
    return req, item


def _wrap_verify_result(vr: InflightVerifyResult) -> asyncio.Future:
    """Wrap a pre-computed InflightVerifyResult in an already-resolved Future.

    Lets style-C tests call ``_run_inflight_verify`` for the real vr, then
    build the InflightEntry ``_finalize_inflight`` expects without re-running
    the verify (mirrors test_merge_speculation.py's ``_build_entry`` helper).
    """

    async def _resolved() -> InflightVerifyResult:
        return vr

    return asyncio.ensure_future(_resolved())


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


# ── step-05 RED / step-06 GREEN: Rows 2,3,4 — verifier lifecycle faults ─────


@pytest.mark.asyncio
class TestScenario234VerifierLifecycleFaults:
    """Rows 2-4: _run_inflight_verify / _finalize_inflight lifecycle faults.

    Style C: direct _run_inflight_verify(item, lease) [+ _finalize_inflight
    for the release-half assertions], driven against REAL merged items
    (``_make_merged_item``) rather than the full worker.run() loop — these
    rows test the verify/finalize halves of ``_verify_and_advance`` in
    isolation, per the task's documented style-C driving convention.
    """

    async def test_runner_unavailable_quarantines_and_redispatches(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """Row 2: RunnerUnavailable quarantines the host and re-dispatches —
        NOT a chain failure — and the re-dispatched item eventually lands.
        """
        worker = _make_worker(git_ops)

        fake_remote = _make_fake_remote(name='ru2-remote')
        fake_remote.run_merge_verify = AsyncMock(
            side_effect=RunnerUnavailable('simulated host failure'),
        )
        allocator = _inject_two_host_allocator(worker, fake_remote)
        lease = allocator.acquire_remote()
        assert lease is not None, 'precondition: remote slot must be free'

        req, item = await _make_merged_item(
            git_ops, config, 'task/casc2-ru', 'casc2_ru.py', 'r = 1\n',
        )
        worker._register_owned_merge_worktree(item.merge_wt)

        result = await worker._run_inflight_verify(item, lease)
        assert result.status == 'RUNNER_UNAVAILABLE', (
            f'Expected RUNNER_UNAVAILABLE, got status={result.status!r}'
        )
        assert result.merge_wt == item.merge_wt, (
            'RUNNER_UNAVAILABLE must preserve merge_wt (re-dispatched, not cleaned)'
        )

        entry = InflightEntry(
            item=item, lease=lease, verify_task=_wrap_verify_result(result),
            merge_wt=None, was_speculative=False, phase='verifying',
        )
        advanced = await worker._finalize_inflight(entry)
        assert advanced is False

        assert 'ru2-remote' in worker._runner_quarantine, (
            'RUNNER_UNAVAILABLE must quarantine the failed remote host'
        )
        assert worker._n_failed is False, (
            'RUNNER_UNAVAILABLE is transient, NOT a chain failure — '
            '_n_failed must stay False'
        )
        assert len(worker._redispatch) == 1, (
            f'Expected the re-merged item on _redispatch, got {worker._redispatch!r}'
        )
        redispatched = worker._redispatch.popleft()
        assert redispatched.request is req

        assert not req.result.done(), (
            'req.result must still be pending immediately after the '
            'RUNNER_UNAVAILABLE finalize (no premature resolution)'
        )

        # ── "Future eventually resolvable": drive the re-dispatched item to
        # a clean landing via the compat shim (a fresh, healthy verify) —
        # confirms the item is genuinely re-verifiable, not stuck.
        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            AsyncMock(return_value=_mock_verify_result(True)),
        ):
            landed = await worker._verify_and_advance(redispatched)
        assert landed is True
        assert req.result.done() and req.result.result().status == 'done', (
            f'Expected the re-dispatched item to eventually land "done", got '
            f'{req.result.result() if req.result.done() else "<pending>"!r}'
        )

    async def test_operator_halt_requeues_and_reverifies(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """Row 3: operator-halt mid-verify -> REQUEUED; unhalt + re-verify -> 'done'."""
        worker = _make_worker(git_ops)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        gated = _gated_runner(gate_release, gate_entered, passed=True, name='casc3-halt')

        # Signal when the abort-poll's remote cancel actually fires, so the
        # test can wait for the halt to be NOTICED before releasing the
        # gate — releasing it immediately after operator_halt() races the
        # gated runner's own completion (via gate_release) against the
        # poll's next halt check, and the runner usually wins (asyncio.wait
        # returns as soon as verify_task completes, which can be sooner than
        # the halt is ever checked).
        cancel_fired = asyncio.Event()

        async def _signaling_cancel() -> int:
            cancel_fired.set()
            return 0

        gated.cancel_verify = AsyncMock(side_effect=_signaling_cancel)

        req, item = await _make_merged_item(
            git_ops, config, 'task/casc3-halt', 'casc3_halt.py', 'h = 1\n',
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        lease = HostLease(name='casc3-halt', runner=gated, is_local=False)

        cas_retries_before = dict(worker._cas_retries)
        gate_retries_before = dict(worker._gate_retries)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
        await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

        # Trigger operator halt mid-verify; wait for the abort-poll to
        # actually notice it (remote cancel fired) before releasing the gate.
        worker.operator_halt('test row-3 halt')
        await asyncio.wait_for(cancel_fired.wait(), timeout=2.0)
        gate_release.set()

        result = await asyncio.wait_for(verify_future, timeout=5.0)
        assert result.status == 'REQUEUED', (
            f'Expected REQUEUED, got status={result.status!r}'
        )
        assert not req.result.done(), (
            'operator-halt REQUEUED must leave req.result pending — a halt is not a failure'
        )
        assert worker._queue.qsize() == 1, 'req must be back on worker._queue'
        assert worker._cas_retries == cas_retries_before, (
            'per-task CAS retry counters must be untouched by a halt'
        )
        assert worker._gate_retries == gate_retries_before, (
            'per-task gate retry counters must be untouched by a halt'
        )

        requeued_req = worker._queue.get_nowait()
        assert requeued_req is req

        # Unhalt, then re-verify the SAME item (main hasn't moved; the merge
        # commit is still valid) — must land 'done'.
        worker.unhalt_all_lanes('test row-3 unhalt')

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            AsyncMock(return_value=_mock_verify_result(True)),
        ):
            landed = await worker._verify_and_advance(item)
        assert landed is True
        assert req.result.done() and req.result.result().status == 'done', (
            f'Expected re-verify after unhalt to land "done", got '
            f'{req.result.result() if req.result.done() else "<pending>"!r}'
        )

    async def test_abandoned_waiter_dropped_and_cleaned(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """Row 4: sole-waiter abandon (Future cancelled) mid-verify -> DROPPED,
        merge_wt cleaned, no leaked speculation slot.
        """
        worker = _make_worker(git_ops)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02

        req, item = await _make_merged_item(
            git_ops, config, 'task/casc4-drop', 'casc4_drop.py', 'd = 1\n',
            speculative=True,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        assert item.merge_wt in worker._owned_merge_worktrees

        # Simulate this item holding a speculation permit the way a real
        # speculative dispatch would — _run_inflight_verify/_finalize_inflight
        # never ACQUIRE the semaphore themselves (only _finalize_inflight's
        # finally block releases it, gated on entry.was_speculative), so the
        # direct-call style needs to arm it manually to make the release
        # observable.
        await worker._speculation_slot.acquire()

        gate_entered = asyncio.Event()

        async def _hang_verify(*args: Any, **kwargs: Any) -> Any:
            gate_entered.set()
            await asyncio.sleep(100)  # abandon wins long before this returns
            return None  # pragma: no cover — unreachable

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)

        with patch('orchestrator.merge_queue._run_post_merge_verify', _hang_verify):
            verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
            await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

            # Sole waiter gives up: cancel the result Future mid-verify.
            req.result.cancel()

            result = await asyncio.wait_for(verify_future, timeout=5.0)

        assert result.status == 'DROPPED', (
            f'Expected DROPPED, got status={result.status!r}'
        )
        assert result.merge_wt is None, 'DROPPED must report merge_wt cleaned (None)'
        assert req.result.cancelled() is True, (
            'abandon must never set_result on the already-cancelled future'
        )

        entry = InflightEntry(
            item=item, lease=lease, verify_task=_wrap_verify_result(result),
            merge_wt=None, was_speculative=True, phase='verifying',
        )
        advanced = await worker._finalize_inflight(entry)
        assert advanced is False

        assert worker.speculation_accounting_violations() == [], (
            'the speculation slot must be released on the DROPPED path — no leak'
        )
        assert item.merge_wt not in worker._owned_merge_worktrees, (
            'merge_wt must be deregistered from the owned-worktree ledger on '
            'DROPPED cleanup'
        )

        main_sha = await git_ops.get_main_sha()
        _assert_quiescent(worker, main_sha, [req])


# ── step-07 RED / step-08 GREEN: Rows 5,6 — new invariant-surface escalations ──


@pytest.mark.asyncio
class TestScenario56NewSurfaceEscalations:
    """Rows 5-6: the two NEW invariant-surface escalations introduced by the
    module-split batch — request-liveness (eta=1992, I1) and resource-
    conservation audit (iota=1994, I4). Both are OBSERVATION + ESCALATION
    ONLY (PRD design decision 4): neither resolves a Future nor mutates
    queue/inflight/worktree state.

    Driven directly (sync, clock-injectable) against a worker wired to a
    _FakeEscalationQueue so the filed escalation's category/summary and the
    WARNING log line can be asserted on — mirrors
    test_merge_queue_request_liveness.py's TestCheckRequestLiveness and
    test_merge_queue_resource_audit.py's TestCheckResourceAudit.
    """

    async def test_wedged_verify_escalates_liveness(
        self,
        git_ops: GitOps,
        git_repo: Path,
        config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Row 5: a dequeued-but-never-resolved request ages past the stuck
        threshold -> exactly one WARNING naming request_id/branch, plus a
        dedup'd category='merge_request_stuck' L1 escalation. Resolving the
        request and sweeping empties the ledger; full quiescence then holds.
        """
        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), escalation_queue=fake_eq)

        req = _make_req('row5-wedged', 'task/row5-wedged', config, git_repo)

        t0 = 1_000_000.0
        # Arm the ledger the way the merger loop does on dequeue. This
        # request is never pushed onto worker._queue/_inflight, so it is
        # absent from snapshot()['entries'] entirely — the leaked/unowned
        # shape, not merely a slow-but-visible in-flight verify.
        worker._request_ledger.on_dequeue(req, now=t0)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            worker._check_request_liveness(t0 + 2000, threshold_s=1000)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, f'expected exactly one WARNING, got: {caplog.text}'
        assert req.request_id in warnings[0].message
        assert req.branch in warnings[0].message

        assert len(fake_eq.submitted) == 1
        esc = fake_eq.submitted[0]
        assert esc.category == 'merge_request_stuck'
        assert req.request_id in esc.summary

        # Resolve the request and sweep -> the ledger empties passively
        # (RequestLedger has no on-resolve hook; see its module docstring).
        req.result.set_result(MergeOutcome('done'))
        worker._request_ledger.sweep_resolved()
        assert worker._request_ledger.is_empty()

        main_sha = await git_ops.get_main_sha()
        _assert_quiescent(worker, main_sha, [req])

    async def test_forced_permit_leak_surfaces_resource_audit(
        self,
        git_ops: GitOps,
    ) -> None:
        """Row 6: a forced speculation-slot permit leak surfaces as a
        non-empty speculation_accounting_violations() /
        snapshot()['resource_audit']['speculation_accounting'], and
        persisting across RESOURCE_AUDIT_ESCALATION_STREAK consecutive
        _check_resource_audit calls escalates category='merge_resource_leak'.
        Restoring the permit clears the violation; full quiescence then holds.
        """
        fake_eq = _FakeEscalationQueue(open_l1=False)
        worker = SpeculativeMergeWorker(
            git_ops, asyncio.Queue(), escalation_queue=fake_eq, speculation_depth=2,
        )
        assert worker._running is True, (
            'both audit surfaces short-circuit to [] when not running — see '
            'speculation_accounting_violations\' docstring'
        )

        # Force a leak: a speculation-slot permit vanished without being
        # recorded as held-by-merger, transferred to the verifier, or
        # genuinely still available — breaks the I4 conservation identity.
        worker._speculation_slot._value -= 1

        violations = worker.speculation_accounting_violations()
        assert violations, 'expected the forced permit leak to surface a violation'
        assert any('speculation-slot conservation violated' in v for v in violations)
        assert worker.snapshot()['resource_audit']['speculation_accounting'] == violations

        n = worker.RESOURCE_AUDIT_ESCALATION_STREAK
        for i in range(1, n + 1):
            worker._check_resource_audit(1000.0 + i)

        assert worker._resource_audit_violation_streak == n
        assert len(fake_eq.submitted) == 1, (
            f'expected exactly one escalation after {n} consecutive violating '
            f'heartbeats, got {len(fake_eq.submitted)}'
        )
        esc = fake_eq.submitted[0]
        assert esc.category == 'merge_resource_leak'

        # Restore the permit -> the identity holds again and full quiescence
        # returns (all 5 surfaces, not just this one).
        worker._speculation_slot._value += 1

        main_sha = await git_ops.get_main_sha()
        _assert_quiescent(worker, main_sha, [])


# ── step-09 RED / step-10 GREEN: Row 7 — ill-formed item raises at construction ──


def _real_kwargs() -> dict:
    """Minimal well-formed RealMergeItem kwargs (mirrors test_merge_types_invariants.py)."""
    return dict(
        request=MagicMock(),
        merge_result=MergeResult(success=True, merge_commit='deadbeef'),
        merge_wt=Path('/fake/merge-wt'),
        base_sha='aabbccdd',
        speculative=False,
    )


def _decided_kwargs() -> dict:
    """Minimal well-formed DecidedItem kwargs (mirrors test_merge_types_invariants.py)."""
    return dict(
        request=MagicMock(),
        base_sha='aabbccdd',
        speculative=False,
        immediate_outcome=MergeOutcome('blocked', reason='test'),
    )


class TestScenario7ItemShape:
    """Row 7: ill-formed item shape raises at CONSTRUCTION time.

    Post-ο (task 2000 / ο, this task's immediate prereq) RealMergeItem and
    DecidedItem are structurally disjoint dataclasses: RealMergeItem has no
    immediate_outcome/already_delivered/failure_diagnostic fields and
    DecidedItem has no merge_result/merge_wt/merged_branch_tip/
    counts_against_cap fields. The former task-1990 I2 "REAL xor DECIDED"
    __post_init__ check therefore retires into type structure — the illegal
    cross-variant shape can no longer be constructed at all. Passing a field
    from the other variant is an unknown-kwarg TypeError raised by the
    dataclass constructor itself, never a runtime ValueError, so this class
    deliberately does NOT pin a message substring (neither does the
    precedent module). Model: test_merge_types_invariants.py.

    The InflightEntry.passthrough_outcome shadow invariant (ε=1890) survives
    this split unchanged as a runtime __post_init__ ValueError — pinned here
    too, alongside item_merge_wt's exhaustive RealMergeItem/DecidedItem match.
    """

    def test_real_item_rejects_decided_only_kwarg(self) -> None:
        """A DECIDED-only field on RealMergeItem is an unknown-kwarg TypeError."""
        kwargs = _real_kwargs()
        kwargs['immediate_outcome'] = MergeOutcome('conflict')
        with pytest.raises(TypeError):
            RealMergeItem(**kwargs)

    def test_decided_item_rejects_real_only_kwarg(self) -> None:
        """A REAL-only field on DecidedItem is an unknown-kwarg TypeError."""
        kwargs = _decided_kwargs()
        kwargs['merge_result'] = MergeResult(success=True, merge_commit='deadbeef')
        with pytest.raises(TypeError):
            DecidedItem(**kwargs)

    def test_valid_shapes_construct_and_item_merge_wt_is_exhaustive(self) -> None:
        """Both well-formed shapes construct; item_merge_wt matches exhaustively:
        a RealMergeItem always owns a merge worktree, a DecidedItem never does.
        """
        real_kwargs = _real_kwargs()
        real_item = RealMergeItem(**real_kwargs)
        decided_item = DecidedItem(**_decided_kwargs())

        assert item_merge_wt(real_item) == real_kwargs['merge_wt']
        assert item_merge_wt(decided_item) is None

    def test_passthrough_outcome_wrapping_real_item_raises_value_error(self) -> None:
        """The surviving runtime shape check: passthrough_outcome requires a
        DecidedItem — wrapping a RealMergeItem is a __post_init__ ValueError.
        """
        real_item = RealMergeItem(**_real_kwargs())
        with pytest.raises(ValueError):
            InflightEntry(
                item=real_item,
                lease=None,
                verify_task=None,
                merge_wt=None,
                was_speculative=False,
                phase='passthrough',
                passthrough_outcome=MergeOutcome('conflict'),
            )


# ── step-11 RED / step-12 GREEN: Row 8 — CAS-retry rebased landing carries merged_branch_tip ──


@pytest.mark.asyncio
class TestScenario8CasRetryTipCarry:
    """Row 8: a CAS-retry rebased landing (advance_main -> 'rebased_pending_reverify')
    must carry item.merged_branch_tip through the I3 replace-only rebuild so
    the post-merge equivalence gate receives the REAL branch tip instead of
    None — no phantom POST_MERGE_EQUIVALENCE_FAILED 'blocked' on
    byte-identical work (task-1928 regression pin; ε=1890 I3).

    Driven directly via worker._verify_and_advance(item) (style C). Models
    test_merge_queue.py's TestMergedBranchTipCarryThroughRebuild — reuses its
    test-(C) mocked-``advance_main`` technique (a simple branch + a fully
    mocked CAS loop, no real overlap/rebase needed) but targets the
    'rebased_pending_reverify' branch (``dataclasses.replace(item,
    base_sha=rebased_onto)``) instead of 'cas_failed'.

    advance_main is fully mocked, so the fabricated advanced_sha returned on
    the second call is not a real git object: ``_resolve_second_parent``
    (fallback term-1) fails open to None against a nonexistent SHA (see its
    docstring), leaving item.merged_branch_tip (term-2) as the only
    surviving trusted-tip source — req.snapshot_tip (term-3) is None by
    default (_make_req does not set it). This isolates exactly what I3's
    dataclasses.replace guarantee is responsible for.
    """

    async def test_rebased_pending_reverify_carries_merged_branch_tip(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        branch = 'row8-cas-retry'
        worker = _make_worker(git_ops)

        # Simple branch; no real overlap/rebase needed (advance_main is mocked).
        branch_wt = (await git_ops.create_worktree(branch)).path
        (branch_wt / 'row8.py').write_text('row8 = 1\n')
        await git_ops.commit(branch_wt, 'Add row8.py')
        _, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=branch_wt)
        ORIGINAL_TIP = head_out.strip()

        base_sha = await git_ops.get_main_sha()
        merge_result = await git_ops.merge_to_main(branch_wt, branch)
        assert merge_result.success and merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None
        worker._register_owned_merge_worktree(merge_result.merge_worktree)

        req = _make_req(branch, branch, config, git_repo)
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
            merged_branch_tip=ORIGINAL_TIP,
        )

        # advance_main mock: call 1 -> rebased_pending_reverify (forces the
        # dataclasses.replace(item, base_sha=rebased_onto) rebuild); call 2+
        # -> advanced, with a FABRICATED advanced_sha (not a real git object
        # — see class docstring for why that isolates term-2).
        advance_calls = [0]

        async def _fake_advance(sha: str, wt: Path, **kw: Any) -> AdvanceOutcome:
            advance_calls[0] += 1
            if advance_calls[0] == 1:
                return AdvanceOutcome(
                    'rebased_pending_reverify',
                    advanced_sha='f' * 40,
                    rebased_from=base_sha,
                    rebased_onto='e' * 40,
                )
            return AdvanceOutcome('advanced', advanced_sha='a' * 40)

        captured_equiv_kw: dict = {}

        async def _spy_equiv(*args: Any, **kwargs: Any) -> list[str]:
            captured_equiv_kw.update(kwargs)
            return []  # clean -> outcome proceeds to 'done'

        pyright_clean = MagicMock(broken=False, failing_subprojects=[], detail='')
        with (
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                AsyncMock(return_value=_mock_verify_result(True)),
            ),
            patch.object(git_ops, 'advance_main', side_effect=_fake_advance),
            patch(
                'orchestrator.merge_queue._reverify_rebased_tree',
                AsyncMock(return_value=None),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_equivalence',
                side_effect=_spy_equiv,
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                AsyncMock(return_value=pyright_clean),
            ),
        ):
            advanced = await worker._verify_and_advance(item)

        assert advanced is True, 'Expected True (main advanced)'
        outcome = req.result.result()
        assert outcome.status == 'done', (
            f'no phantom POST_MERGE_EQUIVALENCE_FAILED blocked; got {outcome!r}'
        )
        assert captured_equiv_kw.get('merged_tip') == ORIGINAL_TIP, (
            f'_check_post_merge_equivalence must be called with '
            f'merged_tip={ORIGINAL_TIP!r} (merged_branch_tip carried through '
            f'the rebased_pending_reverify replace-only rebuild); got '
            f'{captured_equiv_kw.get("merged_tip")!r}'
        )

        main_sha = await git_ops.get_main_sha()
        _assert_quiescent(worker, main_sha, [req])
