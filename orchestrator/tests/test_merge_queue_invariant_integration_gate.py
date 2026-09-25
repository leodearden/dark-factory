"""Integration gate π: fault-injection pipeline run asserting all new
invariant surfaces (B+H leaf).

This file is the G2 boundary-test leaf for the merge-queue modularization
invariants PRD (plans/merge-queue-modularization-invariants-prd.md task π;
the §Boundary-test sketch rows below are this task's spec).  It drives a
REAL merge lane against a real tmp git repo with injected faults through
merge bursts, and after EACH scenario asserts the QUIESCENCE contract (see
``_assert_quiescent``):

  (a) every tracked request's result Future has resolved (done or cancelled).
  (b) lane.speculation_accounting_violations() == [] (I4, task ι=1994).
  (c) lane.worktree_ledger_violations() == [] (I6, task ι=1994).
  (d) snapshot()['entries'] == [] — the live-item census has retired every
      request (κ=1995 / task 2435 registry, η=1992 liveness ledger).
  (e) lane.two_layer_invariants(main_sha) == [] with a REAL main_sha
      (λ=1895, extended by ξ=1999 I5).

SCOPE — TEST-ONLY.  merge_queue.py and its split modules (merge_types,
merge_request_ledger, merge_gates, merge_liveness, merge_shadow, merge_drift,
suffix_graph, merge_speculation_controller) are BEHAVIOUR-FROZEN for this
batch: every surface exercised below was already shipped by the prerequisite
tasks α through ο.  This is a COMPOSITION gate — if a scenario surfaces a
GENUINE production defect, escalate (category=design_concern or
scope_violation); do NOT edit production code here.

STALE-OFFSET WARNING: the PRD's own ``:NNNN`` line citations (e.g. ``:5927``,
``:4936``) are STALE by thousands of lines — merge_queue.py shrank from
~12.9k to ~8.9k lines across the α-ο split tasks.  Always locate symbols BY
NAME (grep/search), never trust a PRD line offset.

§Boundary-test sketch rows (task π's spec)
------------------------------------------
  1.  Speculative head-failure cascade re-lands N+1 with both Futures
      resolved (submission order preserved).
  2.  RunnerUnavailable is transient: the item is re-dispatched and lands,
      rather than failing the chain.
  3.  Operator-halt mid-verify → REQUEUED (req re-queued, Future still
      pending); unhalt + re-verify → 'done'.
  4.  Abandoned sole-waiter (Future cancelled mid-verify) → dropped, merge_wt
      cleaned, no leaked speculation slot.
  5.  Wedged verify aged past the stuck threshold → liveness escalation
      (category='merge_request_stuck') off the heartbeat [η=1992].
  7.  Ill-formed item shape raises at CONSTRUCTION time (structural TypeError
      post-ο; RealMergeItem/DecidedItem are now disjoint dataclasses) — plus
      the surviving InflightEntry passthrough_outcome ValueError [ε=1890].
  8.  CAS-retry rebased landing completes without a phantom
      POST_MERGE_EQUIVALENCE_FAILED (1928 regression pin) [ε=1890 I3].
  10. A main tip that advances under an in-flight verify surfaces a
      verify-base violation that is a member of two_layer_invariants()
      [ξ=1999 I5], and clears once the item drains.
  11. snapshot() keys are additive-only across the whole module-split batch
      (13 keys total, resource_audit being the newest).
  12. InFlightMergeRegistry.acquire/attach fan-out mirror + the
      coalesce_or_enqueue_merge_request dispatched/in_flight/alias contract
      are unchanged by the split.

Task 5030 (PRD ``plans/merge-lane-quality-prd.md`` task γ7) re-seated every
scenario on the lane's PUBLIC surface. The faults used to be injected by
hand into the lane's internals — items appended to ``_inflight``, verify
driven by calling ``_run_inflight_verify``/``_finalize_inflight``/
``_remerge`` directly, the heartbeat's ``_check_request_liveness`` /
``_check_resource_audit`` invoked as methods, and five lane functions
replaced by ``patch('orchestrator.merge_queue.…')``. A gate asserting
composition cannot be built out of calls that bypass the composition, so
every surviving scenario now submits real requests on the public queue of a
running lane and injects its fault through an injected port (``FakeVerifier``
raises / hangs / fails), an injected host (a fake remote runner), the public
operator controls (``operator_halt`` / ``unhalt_all_lanes``), the waiter's
own future, or real git state (a foreign landing on main). Observations are
the ``MergeOutcome`` futures, ``snapshot()``, the public audit methods, and
the escalation queue.

Rows 6 and 9 are gone with that change — see the task-5030 commit body: an
I4 permit leak can only be forged by taking the semaphore out of band, and
the merger-vs-``_remerge`` guard-matrix equivalence has no public entry for
its second path. Both are pinned at their own seams, in
test_merge_queue_resource_audit.py::TestCheckResourceAudit and
test_merge_guard_pipeline.py::TestPathEquivalence.

Fault-injection inventory (no real ssh/build/merge-to-main):
  - the verify port (``verifier=FakeVerifier(...)``) scripts every scoped
    verify: pass, fail, raise ``RunnerUnavailable``, or park on an event.
  - a fake remote runner injected through the two-host allocator gives the
    speculative downstream its own host.
  - ``clock=FakeClock`` ages the heartbeat for row 5 without a real wait.
  - ``patch.object(git_ops, 'advance_main')`` forges the CAS-retry outcome
    row 8 needs (a git_ops method, not a lane internal).
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest
from _merge_lane_fakes import (
    FakeClock,
    FakeVerifier,
    VerifyScript,
    fails,
    main_health_probe_spawned,
    passes,
)
from _orch_helpers import MERGE_RESULT_TIMEOUT
from test_merge_queue_concurrent_verify import (
    _gated_runner,
    _inject_two_host_allocator,
    _make_branch_with_file,
)

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import AdvanceOutcome, GitOps, MergeResult, _run
from orchestrator.merge_lane import MergeLane
from orchestrator.merge_queue import (
    DecidedItem,
    InflightEntry,
    InFlightMergeRegistry,
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
    WaiterRecord,
    coalesce_or_enqueue_merge_request,
    item_merge_wt,
)
from orchestrator.merge_types import QueuedBranch
from orchestrator.verify_runner import RunnerUnavailable

_STOP_TIMEOUT = 30.0

#: Poll ceiling for a state the lane reaches on its own (a verify entering, a
#: worktree being cleaned). Generous: these run real git against a cold tmp repo.
_SETTLE_TIMEOUT = 30.0


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
    return OrchestratorConfig(
        project_root=git_repo, git=git_config,
        escalate_preexisting_main_break=False,
    )


# ── Request builder ───────────────────────────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    worktree: Path,
    lane: Literal['normal', 'high'] = 'normal',
    *,
    merge_first_enqueued_at: float | None = 1000.0,
) -> MergeRequest:
    """Build a MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
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
        lane=lane,
        merge_first_enqueued_at=merge_first_enqueued_at,
    )


async def _submitted(
    git_ops: GitOps, config: OrchestratorConfig, task_id: str, filename: str, content: str,
) -> MergeRequest:
    """A request for a fresh branch carrying one committed file, unsubmitted.

    ``create_worktree`` applies the branch prefix itself, so it takes the BARE
    task id — passing ``task/<id>`` would create ``task/task/<id>`` and leave
    ``req.branch.full_name`` pointing at a ref that does not exist.
    """
    worktree = await _make_branch_with_file(git_ops, task_id, filename, content)
    return _make_req(task_id, f'task/{task_id}', config, worktree)


# ── Lane lifecycle + polling ─────────────────────────────────────────────────


@contextlib.asynccontextmanager
async def _running_lane(git_ops: GitOps, **ports: Any):
    """A running lane over *git_ops*, stopped on the way out.

    Teardown goes through ``stop()`` -- the lane's own shutdown protocol,
    which resolves in-flight request futures, drains its queues, cleans merge
    worktrees and releases leases, and is internally bounded so it cannot hang.
    """
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    lane = MergeLane(git_ops, queue, **ports)
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


async def _until(predicate, *, what: str, timeout: float = _SETTLE_TIMEOUT) -> None:
    """Wait for *predicate* to hold, or fail naming *what* was expected."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f'timed out after {timeout}s waiting for {what}')


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


# ── Shared quiescence helper ──────────────────────────────────────────────────


def _assert_quiescent(
    lane: MergeLane,
    main_sha: str,
    requests: list[MergeRequest],
) -> None:
    """Assert the QUIESCENCE contract holds for *lane*.

    Called after each scenario to confirm the pipeline returned to a clean
    resting state with no leaked permits, worktrees, census entries or
    unresolved Futures:

      (a) every request in *requests* has resolved (done or cancelled) — no
          dangling in-flight work left over from the scenario.
      (b) lane.speculation_accounting_violations() == [] — I4 permit/cap
          conservation holds.  Requires the lane to still be RUNNING (both
          accounting methods short-circuit to [] once stopped — see their
          docstrings — so a stopped lane would make this assertion vacuous),
          which is why every caller samples before leaving ``_running_lane``.
      (c) lane.worktree_ledger_violations() == [] — I6 on-disk ``_merge-*``
          worktree ledger is fully accounted for.  Same running requirement.
      (d) snapshot()['entries'] == [] — the live-item census is empty. The
          census is built from the ItemLifecycle registry's non-terminal set
          plus the in-flight/finalizing entries, so a chokepoint that moved a
          request out of LANE_BUFFERED without ever reaching _retire_item's
          TERMINAL hop shows up here as a leaked entry.
      (e) lane.two_layer_invariants(main_sha) == [] — *main_sha* MUST be a
          REAL sha, never 'unknown': the base-chain and verify-base
          sub-checks are silently skipped for the 'unknown' sentinel, which
          would make this assertion pass vacuously rather than meaningfully.

    Task 5030 dropped a sixth surface: the request-liveness ledger's own
    emptiness, which was asserted as ``_request_ledger.sweep_resolved()``
    followed by ``is_empty()``. Neither the sweep nor the ledger has a public
    accessor, and the sweep only retires entries whose request has resolved --
    which (a) already asserts. What the ledger would have caught beyond that
    is a request the scenario never tracked, and (d) catches that from the
    census side.
    """
    for req in requests:
        assert req.result.done() or req.result.cancelled(), (
            f'Expected request {req.request_id!r} (task {req.task_id!r}) to '
            f'have resolved (done or cancelled) at quiescence, but it is '
            f'still pending'
        )

    spec_violations = lane.speculation_accounting_violations()
    assert spec_violations == [], (
        f'speculation_accounting_violations() non-empty at quiescence: {spec_violations!r}'
    )

    wt_violations = lane.worktree_ledger_violations()
    assert wt_violations == [], (
        f'worktree_ledger_violations() non-empty at quiescence: {wt_violations!r}'
    )

    entries = lane.snapshot()['entries']
    assert entries == [], (
        f'live-item census non-empty at quiescence: '
        f'{[(e["task_id"], e["state"]) for e in entries]!r}'
    )

    assert main_sha and main_sha != 'unknown', (
        f'_assert_quiescent requires a REAL main_sha, got {main_sha!r}'
    )
    tli_violations = lane.two_layer_invariants(main_sha)
    assert tli_violations == [], (
        f'two_layer_invariants({main_sha!r}) non-empty at quiescence: {tli_violations!r}'
    )


# ── Healthy at rest ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestQuiescenceContract:
    """The baseline every §Boundary-test scenario is measured against: a
    freshly-built, still-running lane with no submitted work is already
    quiescent, and its audit surfaces read clean off the snapshot.
    """

    async def test_healthy_running_lane_is_quiescent(
        self, git_ops: GitOps,
    ) -> None:
        """A freshly-built running lane with no work is quiescent."""
        async with _running_lane(git_ops, verifier=FakeVerifier()) as (lane, _queue):
            main_sha = await git_ops.get_main_sha()

            _assert_quiescent(lane, main_sha, [])

            assert lane.snapshot()['resource_audit'] == {
                'speculation_accounting': [],
                'worktree_ledger': [],
            }


# ── Row 1 — speculative head-failure cascade ─────────────────────────────────


@pytest.mark.asyncio
class TestScenario1SpeculativeCascade:
    """Row 1: speculative head-failure cascade re-lands N+1 (submission order
    preserved), both Futures resolved.

    N (head, LOCAL verify) FAILS with a failing VerifyResult (not an
    exception) — a genuine chain failure, so N resolves permanently and is
    NOT retried (only RUNNER_UNAVAILABLE is transient).  N+1 (speculative
    downstream, REMOTE verify) is still in-flight when N fails: the
    head-failure cascade must cancel N+1's in-flight remote verify, re-merge
    it onto ACTUAL main (not N's now-abandoned speculative commit), and
    re-dispatch it — N+1 lands 'done' independently of N's failure.

    The two verifies overlap by construction: N's scoped verify parks on an
    event inside the injected verifier, and N+1 is dispatched to a gated fake
    remote, so both are genuinely in flight when N's gate releases.
    """

    async def test_head_failure_cascade_relands_both_requests(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """N fails permanently (local, parked); N+1 (remote) survives via cascade."""
        head_gate = asyncio.Event()
        follower_release = asyncio.Event()
        follower_entered = asyncio.Event()
        gated_remote = _gated_runner(
            follower_release, follower_entered, passed=True, name='remote-cascade1',
        )
        verifier = FakeVerifier(scripts={
            # Park until the test releases it, then FAIL (not raise).
            'casc1-a': VerifyScript(
                result=fails(category='test_failure', summary='head verify failed').result,
                release=head_gate,
            ),
        })

        async with _running_lane(git_ops, verifier=verifier) as (lane, queue):
            _inject_two_host_allocator(lane, gated_remote)

            req_a = await _submitted(git_ops, config, 'casc1-a', 'casc1_a.py', 'a = 1\n')
            req_b = await _submitted(git_ops, config, 'casc1-b', 'casc1_b.py', 'b = 2\n')
            await queue.put(req_a)
            await queue.put(req_b)

            # Wait for BOTH verifies to enter — the cascade only matters if
            # N+1 is genuinely in-flight when N fails.
            await _until(lambda: 'casc1-a' in verifier.verified, what="N's verify to enter")
            await asyncio.wait_for(follower_entered.wait(), timeout=_SETTLE_TIMEOUT)

            head_gate.set()
            # Release N+1's gate too, so the (possibly still-running) gated
            # task can unblock even if cancel() arrives slightly late.
            follower_release.set()

            outcome_a = await asyncio.wait_for(req_a.result, timeout=MERGE_RESULT_TIMEOUT)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=MERGE_RESULT_TIMEOUT)

            assert outcome_a.status not in ('done', 'already_merged'), (
                f'Expected N to fail (genuine VerifyResult failure, not '
                f'RunnerUnavailable), got {outcome_a!r}.'
            )
            assert not main_health_probe_spawned(outcome_a), outcome_a.reason
            assert outcome_b.status == 'done', (
                f'Expected N+1 (speculative downstream) to resolve "done" after '
                f'cascade re-merge re-dispatch, got {outcome_b!r}. If the cascade '
                f'did not fire, N+1 would stay in flight on a stale speculative '
                f'commit and eventually fail CAS or time out.'
            )

            gated_remote.cancel_verify.assert_called()

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

            main_sha = await git_ops.get_main_sha()
            _assert_quiescent(lane, main_sha, [req_a, req_b])


# ── Rows 2,3,4 — verifier lifecycle faults ───────────────────────────────────


@pytest.mark.asyncio
class TestScenario234VerifierLifecycleFaults:
    """Rows 2-4: the three verify-phase faults, each injected through the
    seam a real one would arrive on — the verify port raising, the operator
    halting, the waiter walking away — and each observed as the outcome the
    request ends up with.
    """

    async def test_runner_unavailable_is_transient_and_redispatches(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Row 2: a RunnerUnavailable verify is re-dispatched, not failed.

        The host-quarantine classification that accompanies it is pinned in
        test_merge_queue_concurrent_verify.py, which owns the host-fault
        tests; what this gate adds is that the burst still reaches quiescence
        with the request landed rather than stuck or chain-failed.
        """
        attempts: list[str | None] = []

        class _UnavailableOnce(FakeVerifier):
            async def run_scoped(self, *args: Any, **options: Any):
                attempts.append(options.get('task_id'))
                if len(attempts) == 1:
                    raise RunnerUnavailable('simulated host failure')
                return await super().run_scoped(*args, **options)

        verifier = _UnavailableOnce()
        async with _running_lane(git_ops, verifier=verifier) as (lane, queue):
            req = await _submitted(git_ops, config, 'ru2', 'ru2.py', 'r = 1\n')
            await queue.put(req)

            outcome = await asyncio.wait_for(req.result, timeout=MERGE_RESULT_TIMEOUT)

            assert outcome.status == 'done', (
                f'RUNNER_UNAVAILABLE is transient — the item must be re-dispatched '
                f'and land, got {outcome!r}'
            )
            assert len(attempts) >= 2, (
                f'expected a second verify attempt after the unavailable host, '
                f'got attempts={attempts!r}'
            )

            main_sha = await git_ops.get_main_sha()
            _assert_quiescent(lane, main_sha, [req])

    async def test_operator_halt_requeues_and_reverifies(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Row 3: operator-halt mid-verify -> requeued, not failed; unhalt -> 'done'."""
        verify_gate = asyncio.Event()
        verifier = FakeVerifier(scripts={
            'halt3': VerifyScript(result=passes().result, release=verify_gate),
        })

        async with _running_lane(git_ops, verifier=verifier) as (lane, queue):
            lane.VERIFY_ABANDON_POLL_SECS = 0.02
            req = await _submitted(git_ops, config, 'halt3', 'halt3.py', 'h = 1\n')
            await queue.put(req)
            await _until(lambda: 'halt3' in verifier.verified, what='the verify to enter')

            lane.operator_halt('row-3 halt')
            assert lane.snapshot()['is_wip_halted'] is True

            # The abort poll notices the halt and abandons the in-flight
            # verify: the item leaves the verify frontier without its future
            # resolving. Waiting for that (rather than releasing the parked
            # verify first) is what makes the halt, not the verify's own
            # completion, the thing this test observes.
            await _until(
                lambda: lane.snapshot()['frozen_prefix']['verify_depth'] == 0,
                what='the halt to abandon the in-flight verify',
            )
            assert not req.result.done(), (
                'an operator halt must leave req.result pending — a halt is not a failure'
            )

            lane.unhalt_all_lanes('row-3 unhalt')
            verify_gate.set()
            outcome = await asyncio.wait_for(req.result, timeout=MERGE_RESULT_TIMEOUT)
            assert outcome.status == 'done', (
                f'expected the re-verify after unhalt to land "done", got {outcome!r}'
            )
            assert verifier.verified.count('halt3') >= 2, (
                f'expected a second verify after the unhalt, got {verifier.verified!r}'
            )

            main_sha = await git_ops.get_main_sha()
            _assert_quiescent(lane, main_sha, [req])

    async def test_abandoned_waiter_dropped_and_cleaned(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Row 4: the sole waiter cancels mid-verify -> the merge worktree is
        cleaned and no speculation slot leaks.
        """
        verify_gate = asyncio.Event()
        verifier = FakeVerifier(scripts={
            'drop4': VerifyScript(result=passes().result, release=verify_gate),
        })

        async with _running_lane(git_ops, verifier=verifier) as (lane, queue):
            lane.VERIFY_ABANDON_POLL_SECS = 0.02
            req = await _submitted(git_ops, config, 'drop4', 'drop4.py', 'd = 1\n')
            await queue.put(req)
            await _until(lambda: 'drop4' in verifier.verified, what='the verify to enter')
            assert lane.snapshot()['owned_merge_worktrees'], (
                'precondition: the merge under verify owns a worktree'
            )

            # Sole waiter gives up.
            req.result.cancel()

            await _until(
                lambda: not lane.snapshot()['owned_merge_worktrees'],
                what='the abandoned merge worktree to be cleaned',
            )
            assert req.result.cancelled() is True, (
                'abandon must never set_result on the already-cancelled future'
            )
            assert lane.snapshot()['resource_audit'] == {
                'speculation_accounting': [],
                'worktree_ledger': [],
            }

            verify_gate.set()
            await _until(
                lambda: not lane.snapshot()['entries'],
                what='the dropped item to retire from the census',
            )
            main_sha = await git_ops.get_main_sha()
            _assert_quiescent(lane, main_sha, [req])


# ── Row 5 — request-liveness escalation ──────────────────────────────────────


@pytest.mark.asyncio
class TestScenario5LivenessEscalation:
    """Row 5: a verify that never returns ages past the stuck threshold and
    the heartbeat files a dedup'd category='merge_request_stuck' escalation
    [η=1992].

    The ageing is done by the injected clock, not by waiting: ``FakeClock``
    advances by whatever the heartbeat asks to sleep, so the lane's own
    ``_HEARTBEAT_POLL_S`` cadence carries wall-clock time past the threshold
    within a few event-loop turns. The escalation therefore comes off the
    real heartbeat wiring rather than a direct call to the check.
    """

    async def test_wedged_verify_escalates_liveness(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """A wedged request surfaces one merge_request_stuck escalation naming it."""
        verify_gate = asyncio.Event()
        verifier = FakeVerifier(scripts={
            'wedge5': VerifyScript(result=passes().result, release=verify_gate),
        })
        escalations = _FakeEscalationQueue(open_l1=False)

        async with _running_lane(
            git_ops, verifier=verifier, escalation_queue=escalations, clock=FakeClock(),
        ) as (_lane, queue):
            req = await _submitted(git_ops, config, 'wedge5', 'wedge5.py', 'w = 1\n')
            await queue.put(req)

            await _until(
                lambda: bool(escalations.submitted),
                what='the heartbeat to escalate the wedged request',
            )

            esc = escalations.submitted[0]
            assert esc.category == 'merge_request_stuck', (
                f'expected a liveness escalation, got category={esc.category!r}'
            )
            assert req.request_id in esc.summary, (
                f'expected the escalation to name {req.request_id!r}, got {esc.summary!r}'
            )
            assert req.branch.bare_id in esc.summary, (
                f'expected the escalation to name branch {req.branch.bare_id!r}, '
                f'got {esc.summary!r}'
            )

            verify_gate.set()


# ── Row 7 — ill-formed item shape raises at construction ─────────────────────


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

    Post-ο (task 2000 / ο) RealMergeItem and DecidedItem are structurally
    disjoint dataclasses: RealMergeItem has no immediate_outcome/
    already_delivered/failure_diagnostic fields and DecidedItem has no
    merge_result/merge_wt/merged_branch_tip/counts_against_cap fields. The
    former task-1990 I2 "REAL xor DECIDED" __post_init__ check therefore
    retires into type structure — the illegal cross-variant shape can no
    longer be constructed at all. Passing a field from the other variant is
    an unknown-kwarg TypeError raised by the dataclass constructor itself,
    never a runtime ValueError, so this class deliberately does NOT pin a
    message substring (neither does the precedent module). Model:
    test_merge_types_invariants.py.

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
                passthrough_outcome=MergeOutcome('conflict'),
            )


# ── Row 8 — CAS-retry rebased landing ────────────────────────────────────────


@pytest.mark.asyncio
class TestScenario8CasRetryTipCarry:
    """Row 8: a CAS-retry rebased landing (advance_main -> 'rebased_pending_reverify')
    completes as a clean landing — no phantom POST_MERGE_EQUIVALENCE_FAILED
    'blocked' on byte-identical work (task-1928 regression pin; ε=1890 I3).

    The rebuild the regression was about is ``dataclasses.replace(item,
    base_sha=rebased_onto)`` dropping ``merged_branch_tip``, which left the
    post-merge equivalence gate comparing against a drifted HEAD. That the
    gate is CALLED with ``merged_tip == ORIGINAL_TIP`` is pinned argument-wise
    in test_merge_queue.py::TestMergedBranchTipCarryThroughRebuild — it can
    only be seen by patching the gate function, which is not on the verify
    port. What this gate adds is the composition: a request driven through
    the real pipeline across that rebuild still lands, with the burst
    quiescent afterwards.
    """

    async def test_rebased_pending_reverify_lands_cleanly(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        base_sha = await git_ops.get_main_sha()
        advance_calls: list[str] = []

        async def _fake_advance(sha: str, worktree: Path, **kwargs: Any) -> AdvanceOutcome:
            # Call 1 forces the replace-only rebuild; call 2 lands. The
            # fabricated advanced_sha is deliberate: it is not a real git
            # object, so the equivalence gate's own tip resolution fails open
            # and cannot mask a dropped merged_branch_tip.
            advance_calls.append(sha)
            if len(advance_calls) == 1:
                return AdvanceOutcome(
                    'rebased_pending_reverify',
                    advanced_sha='f' * 40,
                    rebased_from=base_sha,
                    rebased_onto='e' * 40,
                )
            return AdvanceOutcome('advanced', advanced_sha='a' * 40)

        async with _running_lane(git_ops, verifier=FakeVerifier()) as (lane, queue):
            req = await _submitted(git_ops, config, 'row8', 'row8.py', 'row8 = 1\n')
            with patch.object(git_ops, 'advance_main', side_effect=_fake_advance):
                await queue.put(req)
                outcome = await asyncio.wait_for(req.result, timeout=MERGE_RESULT_TIMEOUT)

            assert len(advance_calls) >= 2, (
                f'expected the CAS retry to re-advance after the rebase, got '
                f'{len(advance_calls)} advance_main call(s)'
            )
            assert outcome.status == 'done', (
                f'no phantom POST_MERGE_EQUIVALENCE_FAILED blocked expected; got {outcome!r}'
            )

            main_sha = await git_ops.get_main_sha()
            _assert_quiescent(lane, main_sha, [req])


# ── Row 10 — verify-base / frozen-tip mismatch ───────────────────────────────


@pytest.mark.asyncio
class TestScenario10VerifyBaseMismatch:
    """Row 10: a verify-base/frozen-tip mismatch surfaces a violation string
    that is a structural member of ``two_layer_invariants()`` [ξ=1999 I5],
    promoted from the ε=1890 log-only WARN.

    The mismatch is produced the way a real one is: a foreign lander moves
    main while an item is mid-verify against the older tip. Asserts
    STRUCTURAL containment (the same violation strings appear in
    ``two_layer_invariants()`` and ``snapshot()['two_layer_invariants']``),
    never violation wording.
    """

    async def test_main_advancing_under_verify_surfaces_and_clears(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        verify_gate = asyncio.Event()
        verifier = FakeVerifier(scripts={
            'vb10': VerifyScript(result=passes().result, release=verify_gate),
        })

        async with _running_lane(git_ops, verifier=verifier) as (lane, queue):
            req = await _submitted(git_ops, config, 'vb10', 'vb10.py', 'v = 1\n')
            await queue.put(req)
            await _until(
                lambda: lane.snapshot()['frozen_prefix']['verify_depth'] >= 1,
                what='the item to reach the frozen prefix',
            )

            main_before = await git_ops.get_main_sha()
            assert lane.two_layer_invariants(main_before) == [], (
                'precondition: the pipeline is healthy against the base it merged on'
            )

            # A foreign lander moves main out from under the in-flight verify.
            (git_repo / 'foreign.txt').write_text('landed elsewhere\n')
            await _run(['git', 'add', '-A'], cwd=git_repo)
            await _run(['git', 'commit', '-m', 'Foreign landing'], cwd=git_repo)
            main_after = await git_ops.get_main_sha()
            assert main_after != main_before

            violations = lane.two_layer_invariants(main_after)
            assert violations, (
                f'expected the stale verify base to be flagged against main '
                f'{main_after!r}, got: {violations}'
            )
            assert any(req.request_id in v for v in violations), (
                f'expected a violation naming {req.request_id!r}, got: {violations}'
            )

            # snapshot() reads the same surface against the main SHA cached by
            # the lane's own recompute — refresh it and the violations must
            # match verbatim.
            await lane.recompute_suffix_conflict_graph()
            snap_violations = lane.snapshot()['two_layer_invariants']
            assert all(v in snap_violations for v in violations), (
                f"expected snapshot()['two_layer_invariants'] to surface the same "
                f'violation(s), got: {snap_violations}'
            )

            # Draining the item clears the surface: quiescence is measured
            # against the main the pipeline actually ends on.
            verify_gate.set()
            await asyncio.wait_for(req.result, timeout=MERGE_RESULT_TIMEOUT)
            main_final = await git_ops.get_main_sha()
            _assert_quiescent(lane, main_final, [req])


# ── Rows 11,12 — preservation ────────────────────────────────────────────────


@pytest.mark.asyncio
class TestScenario1112Preservation:
    """Rows 11-12: snapshot() keys are additive-only across the whole
    module-split batch, and the InFlightMergeRegistry acquire/attach fan-out
    mirror + coalesce_or_enqueue_merge_request dispatched/in_flight/alias
    contract are unchanged by the split.
    """

    async def test_snapshot_keys_stable(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Row 11: the 12 pre-existing snapshot() keys are all still present,
        plus the newest additive key 'resource_audit' (13 total); per-entry
        keys are unchanged.
        """
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        lane = MergeLane(git_ops, queue, verifier=FakeVerifier())
        snap = lane.snapshot()

        pre_existing_keys = {
            'entries', 'depth', 'head_of_line', 'verify_in_progress',
            'is_wip_halted', 'halt_owner_esc_id', 'occupancy',
            'suffix_conflict_graph', 'metrics', 'frozen_prefix',
            'two_layer_invariants', 'speculation',
        }
        expected_keys = pre_existing_keys | {'resource_audit'}
        missing = expected_keys - snap.keys()
        # Additive-only contract: every expected key must be present
        # (expected_keys <= snap.keys()), but a future genuinely-additive key
        # must NOT fail this gate (the class docstring explicitly allows
        # additive extension) — so this checks the subset relationship, not
        # an exact key count.
        assert not missing, (
            f'expected snapshot() keys missing: {missing!r}. '
            f'Keys present: {sorted(snap.keys())!r}'
        )

        assert snap['resource_audit'] == {
            'speculation_accounting': [],
            'worktree_ledger': [],
        }

        # Per-entry keys stable: submit one request on the lane's own queue
        # (the lane is not running, so it stays queued) and check its shape.
        req = _make_req('row11-entry', 'task/row11-entry', config, git_repo)
        queue.put_nowait(req)
        snap2 = lane.snapshot()
        assert snap2['depth'] == 1
        entry_keys = {
            'task_id', 'branch', 'state', 'enqueued_at', 'age_secs', 'position',
            'waiter_alive', 'worktree', 'pre_rebased', 'request_id', 'lane',
            'host', 'verify_started_at', 'verify_age_secs',
        }
        missing_entry_keys = entry_keys - snap2['entries'][0].keys()
        assert not missing_entry_keys, (
            f'expected per-entry keys missing: {missing_entry_keys!r}. '
            f'Entry: {snap2["entries"][0]!r}'
        )

    async def test_registry_attach_fanout_and_coalesce_alias_unchanged(
        self, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """Row 12: InFlightMergeRegistry.acquire/attach fan-out mirror, and
        the coalesce_or_enqueue_merge_request dispatched/in_flight/alias
        contract, are unchanged by the module split.
        """
        loop = asyncio.get_running_loop()

        # ── acquire + attach fan-out mirror ──────────────────────────────────
        registry = InFlightMergeRegistry()
        f1: asyncio.Future = loop.create_future()
        assert registry.acquire(
            'row12-branch', 'row12-task', f1, request_id='mr-row12-1',
        ) is True

        f2: asyncio.Future = loop.create_future()
        attached = registry.attach(
            'row12-branch',
            WaiterRecord(request_id='mr-row12-2', future=f2, source='mcp'),
        )
        assert attached is True
        entry = registry.entry('row12-branch')
        assert entry is not None and len(entry.waiters) == 2

        outcome = MergeOutcome(status='done', merge_sha='row12sha')
        f1.set_result(outcome)
        await asyncio.sleep(0)
        assert f2.done() and f2.result() is outcome, (
            'expected the fan-out mirror to propagate the primary result '
            'onto the attached waiter future'
        )

        # ── coalesce dispatched/in_flight/alias contract ─────────────────────
        queue: asyncio.Queue = asyncio.Queue()
        registry2 = InFlightMergeRegistry()
        event_store = EventStore(tmp_path / 'row12_events.db', 'row12-coalesce')

        req1 = _make_req('row12-coalesce-1', 'row12-coalesce-branch', config, tmp_path)
        result1 = await coalesce_or_enqueue_merge_request(queue, req1, event_store, registry2)
        assert result1.dispatched is True, f'expected dispatched=True for req1, got {result1}'

        req2 = _make_req('row12-coalesce-2', 'row12-coalesce-branch', config, tmp_path)
        result2 = await coalesce_or_enqueue_merge_request(queue, req2, event_store, registry2)
        assert result2.in_flight is True, f'expected in_flight=True (coalesced), got {result2}'
        assert result2.inflight_request_id == req1.request_id, (
            f'expected coalesced result to alias to primary request_id '
            f'{req1.request_id!r}, got {result2.inflight_request_id!r}'
        )


# ── B8 (MQ-reliability kappa-b / task 2435): mid-pipeline census ────────────
# Not one of the π-task's original rows above — this is the boundary test for
# the SEPARATE merge-queue-reliability PRD scope-4 kappa-b task (single-read
# conversion: snapshot()/liveness repointed onto the ItemLifecycle registry).
# Reuses Row 1's two-item speculative-cascade setup as its structural
# template; the addition is a LIVE sampling point taken while both items are
# still mid-verify, before either Future resolves.


@pytest.mark.asyncio
class TestB8MidPipelineCensus:
    """B8: the snapshot census and the resource audit hold mid-pipeline, not
    merely at rest — sampled while both items are genuinely verifying and
    neither Future has resolved, then again as full quiescence once the
    pipeline drains.

    Task 5030 dropped the two assertions that compared the census against the
    ItemLifecycle registry and the liveness ledger directly
    (``_lifecycle.non_terminal_items()`` / ``_request_ledger.open_request_ids()``).
    Both restated, in test-visible form, that the census is DERIVED from
    those structures — an implementation identity with no public surface. The
    behaviour that identity exists to deliver is what survives here: every
    in-flight request appears in the census, with a phase, while it is in
    flight.
    """

    async def test_census_and_audit_agree_mid_pipeline(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Two-item cascade sampled mid-verify, then drained to quiescence."""
        head_gate = asyncio.Event()
        follower_release = asyncio.Event()
        follower_entered = asyncio.Event()
        gated_remote = _gated_runner(
            follower_release, follower_entered, passed=True, name='remote-b8',
        )
        verifier = FakeVerifier(scripts={
            'b8-a': VerifyScript(
                result=fails(category='test_failure', summary='head verify failed').result,
                release=head_gate,
            ),
        })

        async with _running_lane(git_ops, verifier=verifier) as (lane, queue):
            _inject_two_host_allocator(lane, gated_remote)

            req_a = await _submitted(git_ops, config, 'b8-a', 'b8_a.py', 'a = 1\n')
            req_b = await _submitted(git_ops, config, 'b8-b', 'b8_b.py', 'b = 2\n')
            await queue.put(req_a)
            await queue.put(req_b)

            await _until(lambda: 'b8-a' in verifier.verified, what="N's verify to enter")
            await asyncio.wait_for(follower_entered.wait(), timeout=_SETTLE_TIMEOUT)

            # ── Live sample: both items in flight, neither Future resolved ──
            assert not req_a.result.done() and not req_b.result.done(), (
                'precondition: the sample must be taken before either outcome resolves'
            )
            snap = lane.snapshot()
            census = {entry['request_id']: entry['state'] for entry in snap['entries']}
            assert {req_a.request_id, req_b.request_id} <= census.keys(), (
                f'expected both in-flight requests in the mid-pipeline census, '
                f'got {census!r}'
            )
            assert all(census[rid] for rid in (req_a.request_id, req_b.request_id)), (
                f'expected every in-flight request to carry a phase, got {census!r}'
            )
            assert snap['resource_audit']['speculation_accounting'] == [], (
                f'speculation_accounting non-empty mid-pipeline: '
                f'{snap["resource_audit"]["speculation_accounting"]!r}'
            )

            # ── Let the cascade run to completion (mirrors Row 1) ───────────
            head_gate.set()
            follower_release.set()

            outcome_a = await asyncio.wait_for(req_a.result, timeout=MERGE_RESULT_TIMEOUT)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=MERGE_RESULT_TIMEOUT)
            assert outcome_a.status not in ('done', 'already_merged'), (
                f'Expected N to fail (genuine VerifyResult failure), got {outcome_a!r}.'
            )
            assert outcome_b.status == 'done', (
                f'Expected N+1 to resolve "done" after cascade re-merge, got {outcome_b!r}.'
            )

            main_sha = await git_ops.get_main_sha()
            _assert_quiescent(lane, main_sha, [req_a, req_b])
