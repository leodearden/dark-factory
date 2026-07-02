"""Integration gate for the infra extension of the offline deep-test lane
(task 1960, IE2 — the ζ-analog for infra).

Stands up the REAL trigger→worker→warm-worktree→infra-sub-run→failure-
handling chain end-to-end and RUNS (does not tabulate) the infra boundary
scenarios IB1-IB6 below. No production module is touched by this task —
every scenario asserts already-landed behaviour from:

* β1 (task 1951) — ``Harness._offline_lane_notifiee`` / ``_note_merge_all`` /
  ``_note_offline_lane`` fan-out (``harness.py``) — the real merge-landed
  hot-path callback, shared by numeric AND infra sub-runs (one coalesced
  trigger, two sub-runs).
* IE1 (task 1959) — ``OfflineLaneWorker``'s ``infra_runner`` /
  ``infra_confirmation_runner`` seams (``offline_lane.py``): ``_run_once``
  runs BOTH the numeric (``suite_runner``) and infra (``infra_runner``)
  sub-runs at the SAME snapshot head in the SAME ``_offline-deep`` worktree
  when ``config.git.offline_lane_infra_enabled`` is True; the infra red leg
  reuses ``_handle_red_run(wt, head,
  confirmation_runner=self.infra_confirmation_runner)``; the default seams
  ``_default_run_infra``/``_default_infra_confirmation_run`` shell out to
  ``tests/infra/run_all.sh --scope host-infra`` (reify H9) and parse
  ``RESULT: FAIL (<name>)`` lines via module-level ``_parse_infra_failures``.
* β3 (task 1954) — ``OfflineLaneWorker._handle_red_run`` (confirmation,
  fingerprinting, dedup'd fix-task file/update, L0/L2 escalation staging) —
  content-agnostic, so infra test-file-names dedup exactly like numeric IDs.

STRATEGY — injected-seam integration harness (mirrors the numeric ζ-gate,
``test_offline_lane_integration.py``, task 1955): the reify H9 subprocess
boundary is faked ONLY at the ``infra_runner``/``infra_confirmation_runner``
seam; every other component in the chain (the git repo, ``GitOps``,
``Harness`` fan-out, ``OfflineLaneWorker``, ``EscalationQueue``) is real.
``config.git.offline_lane_infra_enabled`` is set True so the infra sub-run
actually executes; the numeric ``suite_runner`` is injected GREEN in every
scenario so the infra red is the SOLE signal — isolating the
``confirmation_runner=infra_confirmation_runner`` path through
``_handle_red_run``.

PRECONDITION (bound at the dependency-graph level, not executable here):
host-exclusive infra tests are host-global-unsafe by definition (real
cgroup burn / reflink FS), so they cannot run inside a hermetic pytest. The
"live reify checkout with H1/H9 landed" precondition is instead bound at
the dependency-graph level: reify:4929 (H9 ``run_all --scope host-infra``
runner) and reify:4921 (H1 classification manifest) are both ``done`` on
reify ``main`` (verified via ``get_external_statuses`` at plan time — see
this task's ``.task/plan.json`` analysis for the full rationale).

Scenario map:

  IB1 - FROM-HEAD — an advance triggers a from-head infra sub-run
  IB2 - NEVER A GATE (C7/§6) — an infra sub-run in-flight never gates a
        merge: prompt return, fail-open, no halt/escalation
  IB3 - CONFIRMED RED — numeric green + infra red confirmed → a normal
        pending fix task + L0 info escalation, merge queue untouched
  IB4 - DEDUP across advances — a same-infra-failing-set recurrence updates
        the open fix task, never duplicates it or double-escalates
  IB5 - FLAKE FILTER — infra fail-then-pass on confirmation is intermittent
        nondeterminism: no task, no escalation, no open fingerprint
  IB6 - REAL H9 WIRE-FORMAT — the REAL default infra seams shell out to a
        committed stub ``tests/infra/run_all.sh``, exercising
        ``_parse_infra_failures`` end-to-end into a filed fix task

Each scenario is framed as a negative control: it would RED if IE1's infra
wiring were reverted, and is expected GREEN against landed IE1.

OUT OF SCOPE: real reify infra test timing/content and host-exclusive test
execution itself (owned by reify H9), and the flip to always-on
(``offline_lane_infra_enabled`` stays default-False here) — owned by IE4.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.offline_lane import OfflineLaneWorker

logger = logging.getLogger(__name__)

# Default git_overrides for _build_infra_worker — declared with an explicit
# dict[str, Any] value type (rather than left as an inline dict literal,
# which pyright would infer as the concrete dict[str, bool] and then reject
# unpacking against GitConfig's other, non-bool fields) so **-unpacking it
# into GitConfig(...) type-checks cleanly regardless of which fields a
# caller-supplied git_overrides sets.
_DEFAULT_INFRA_GIT_OVERRIDES: dict[str, Any] = {'offline_lane_infra_enabled': True}

# ---------------------------------------------------------------------------
# Shared end-to-end scaffolding (prerequisite P1)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Create a real git repo with an initial commit (mirrors test_git_ops.py)."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _advance_main(repo: Path, message: str = 'advance') -> str:
    """Commit a trivial, targeted change on main and return the new head SHA.

    Stages ONLY the dedicated counter file (never ``git add -A``) so that a
    real ``.worktrees/_offline-deep`` worktree nested under *repo* (created
    mid-test by
    :meth:`~orchestrator.git_ops.GitOps.reset_persistent_offline_deep_worktree`)
    is never swept into a same-repo commit as an embedded repository.
    """
    counter_path = repo / '_advance_counter.txt'
    n = int(counter_path.read_text()) + 1 if counter_path.exists() else 1
    counter_path.write_text(str(n))
    await _run(['git', 'add', str(counter_path)], cwd=repo)
    await _run(['git', 'commit', '-m', f'{message} {n}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'main'], cwd=repo)
    return sha.strip()


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        # No remote configured on the tmp repo; never push in this suite.
        push_after_advance=False,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real tmp git repo (initial commit only) — never a MagicMock git_ops."""
    repo_path = tmp_path / 'repo'
    repo_path.mkdir()
    asyncio.run(_setup_repo(repo_path))
    return repo_path


@pytest.fixture
def git_ops(git_config: GitConfig, repo: Path) -> GitOps:
    """Real GitOps bound to the real tmp repo (IB1 from-head snapshots)."""
    return GitOps(git_config, repo)


@pytest.fixture
def harness(git_config: GitConfig, git_ops: GitOps, mock_orch_config) -> Harness:
    """Minimally-constructed real Harness wired to the real tmp-repo GitOps.

    Mirrors ``test_offline_lane_integration.py``'s ``harness`` fixture
    verbatim: ``h.git_ops`` is replaced wholesale by the real tmp-repo-backed
    :func:`git_ops` fixture (rather than left as Harness's own internally-
    constructed GitOps over an un-git-initialized ``tmp_path``), so that real
    head snapshots / worktree resets are actually exercised end-to-end.
    """
    mock_orch_config.git = git_config
    mock_orch_config.project_root = git_ops.project_root
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.event_store = MagicMock()
    h.git_ops = git_ops
    h.git_ops.get_merge_diff_files = AsyncMock(return_value=([], None))
    h._service_restart_coordinators = []
    return h


async def _default_green_suite_runner(wt: Path, head: str, threads: int) -> tuple[int, str]:
    """Trivial always-green numeric ``suite_runner`` — the default numeric
    leg for every infra scenario built via :func:`_build_infra_worker`.

    The tmp repo has no ``scripts/run-offline-deep.sh``, so the REAL default
    numeric seam would spuriously red (subprocess rc=127) and file its own
    unrelated fix task. Forcing green here isolates the infra red path
    (``_handle_red_run(confirmation_runner=infra_confirmation_runner)``) as
    the SOLE signal in every scenario below (see the design decision in
    this task's ``.task/plan.json``).
    """
    return (0, '')


def _build_infra_worker(
    git_ops: GitOps,
    tmp_path: Path,
    *,
    suite_runner=None,
    infra_runner=None,
    infra_confirmation_runner=None,
    task_client=None,
    escalation_queue: EscalationQueue | None = None,
    git_overrides: dict | None = None,
) -> OfflineLaneWorker:
    """Build a real OfflineLaneWorker wired for the INFRA integration harness.

    Adapts ``test_offline_lane_integration.py``'s ``_build_worker`` for the
    infra sub-run (task 1959, IE1): the reify-subprocess boundary THIS
    harness fakes is ``infra_runner``/``infra_confirmation_runner`` — the
    infra seams — NOT ``suite_runner``/``confirmation_runner`` (the numeric
    seams, which stay real-shaped but forced green — see below). *git_ops*
    is the shared real tmp-repo GitOps (see the ``git_ops`` fixture) so the
    worker's head snapshots / worktree resets land in the SAME repo
    :func:`_advance_main` commits to; *escalation_queue* defaults to a real
    :class:`EscalationQueue` (never a MagicMock) so IB3/IB4 can assert
    against real on-disk escalation records; *task_client* defaults to a
    fake ``OfflineLaneTaskClient`` (a bare ``AsyncMock`` — its
    ``submit_fix_task``/``append_suspect_range``/``get_status`` children
    auto-vivify as awaitable ``AsyncMock``s, per the β3 unit-test convention
    in ``test_offline_lane.py``) with ``get_status`` pre-set to the steady
    in-flight ``'in-progress'`` state and ``submit_fix_task`` pre-set to
    return a stable ``'fix-task-1'`` id (IB3/IB4/IB5 only ever have one
    fingerprint open at a time, so one stable id suffices).

    Two infra-specific defaults isolate the infra red path as the SOLE
    signal in every scenario:

    * ``git_overrides`` defaults to ``{'offline_lane_infra_enabled': True}``
      (rather than ``{}``) so the infra sub-run actually executes in
      ``_run_once`` without every call site having to opt in explicitly. A
      caller-supplied ``git_overrides`` REPLACES this default wholesale
      (same replace-not-merge semantics as the numeric ``_build_worker``) —
      it must set ``offline_lane_infra_enabled=True`` itself if the infra
      leg is still needed.
    * ``suite_runner`` defaults to :func:`_default_green_suite_runner` (a
      trivial always-``(0, '')`` stub) rather than the real
      ``scripts/run-offline-deep.sh`` invocation — the numeric leg has no
      script in this tmp repo and would otherwise spuriously red (rc=127),
      polluting the infra assertions with an unrelated fix task.

    ``infra_runner``/``infra_confirmation_runner`` default to ``None`` and
    are passed straight through to ``OfflineLaneWorker`` — when left
    unset, the worker falls back to its OWN real default seams
    (``_default_run_infra``/``_default_infra_confirmation_run``), which is
    exactly what IB6 needs to drive the real H9 wire-format parse over a
    committed stub script.
    """
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = git_ops.project_root
    config.git = GitConfig(
        **(git_overrides if git_overrides is not None else _DEFAULT_INFRA_GIT_OVERRIDES)
    )

    if task_client is None:
        task_client = AsyncMock()
        task_client.get_status = AsyncMock(return_value='in-progress')
        task_client.submit_fix_task = AsyncMock(return_value='fix-task-1')

    if escalation_queue is None:
        escalation_queue = EscalationQueue(tmp_path / 'escalations')

    return OfflineLaneWorker(
        git_ops,
        config,
        lock_path=tmp_path / 'offline_lane.lock',
        suite_runner=suite_runner if suite_runner is not None else _default_green_suite_runner,
        infra_runner=infra_runner,
        infra_confirmation_runner=infra_confirmation_runner,
        task_client=task_client,
        escalation_queue=escalation_queue,
    )


def _wire_lane(harness: Harness, worker: OfflineLaneWorker) -> None:
    """Register the worker's real on_post_merge as the harness's lane notifiee.

    Mirrors the direct-attribute registration ``Harness._start_offline_lane``
    performs in production (``harness.py:5119``), without going through the
    enable-gated launch path itself — these tests drive the worker loop
    under their own control rather than as a live background task.
    """
    harness._offline_lane_notifiee = worker.on_post_merge


async def _drive_advance(
    harness: Harness,
    repo: Path,
    task_id: str = 'task-1',
) -> tuple[str, str]:
    """Advance main and await the REAL on_merge_landed fan-out.

    ``harness._note_merge_all`` IS the exact callback the
    ``SpeculativeMergeWorker`` invokes as ``on_merge_landed`` in production
    (see ``harness.py:4979``) — driving it directly here (rather than
    standing up a full merge worker) exercises the real fan-out without
    pulling the whole speculative-merge machinery into this integration
    gate. Returns ``(base_sha, head_sha)`` — the pre- and post-advance main
    SHAs — for the caller's own assertions.
    """
    base_sha = await harness.git_ops.get_main_sha()
    head_sha = await _advance_main(repo)
    await harness._note_merge_all(task_id, base_sha, head_sha)
    return base_sha, head_sha


async def _run_lane(
    worker: OfflineLaneWorker,
    expected_passes: int,
    *,
    timeout: float = 5.0,
) -> None:
    """Drive worker.run() as a real background task until *expected_passes*
    full passes (``_run_once`` calls) have COMPLETED, then cancel the loop
    task.

    Wraps ``worker._run_once`` itself — NOT ``suite_runner``/``infra_runner``
    alone — with a call counter, so ``done`` is only set once a pass,
    INCLUDING any ``_handle_red_run`` confirmation/fix-task/escalation work
    it triggers (numeric OR infra), has fully returned. Counting at either
    sub-run boundary instead would race the red path: every seam this
    harness injects (a bare ``async def ...: return ...``, or an
    ``AsyncMock``) happens not to yield to the event loop, so today
    ``_handle_red_run`` always finishes before the loop next suspends at
    ``_wake.wait()`` — but that is an accident of the fakes, not a
    guarantee, and cancelling right after a sub-run returns would race
    ``_handle_red_run`` mid-flight the moment a seam awaits real I/O.
    Restores the original ``_run_once`` in the ``finally`` block so repeated
    calls (``_drive_infra_reds`` drives 2+ passes for IB4) never nest
    counting wrappers.

    Adapted verbatim from ``test_offline_lane_integration.py``'s
    ``_run_lane``, generalized from a fixed N=1 to an arbitrary pass count
    (IB1's single pass via :func:`_run_one_lane_pass`; IB4's dedup sequence
    uses N=2 via :func:`_drive_infra_reds`). Requires ``worker._dirty`` (or
    the wake event) to already be set by a prior trigger, exactly as
    production wiring would leave it.
    """
    inner_run_once = worker._run_once
    done = asyncio.Event()
    count = {'n': 0}

    async def _counting_run_once():
        result = await inner_run_once()
        count['n'] += 1
        if count['n'] >= expected_passes:
            done.set()
        return result

    worker._run_once = _counting_run_once
    task = asyncio.create_task(worker.run())
    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        worker._run_once = inner_run_once


async def _run_one_lane_pass(worker: OfflineLaneWorker, *, timeout: float = 5.0) -> None:
    """Drive worker.run() as a real background task for exactly one pass
    (IB1/IB3/IB5/IB6)."""
    await _run_lane(worker, 1, timeout=timeout)


class _ControllableInfraRunner:
    """An infra_runner whose FIRST call blocks in-flight until released (IB2).

    Mirrors ``test_offline_lane_integration.py``'s ``_ControllableSuiteRunner``
    verbatim in shape, used here as the INFRA seam so the infra sub-run
    (not the numeric one) is the leg held in-flight: records every head it
    is invoked with, in call order. Only the first call blocks (on an
    internal ``asyncio.Event`` gate) — later calls return immediately — so a
    single instance can drive both the held initial pass and any subsequent
    coalesced re-run in one scenario.
    """

    def __init__(self) -> None:
        self.heads: list[str] = []
        self._hold = asyncio.Event()
        self._entered = asyncio.Event()
        self._held_once = False

    async def __call__(self, wt: Path, head: str, threads: int) -> tuple[int, str]:
        self.heads.append(head)
        if not self._held_once:
            self._held_once = True
            self._entered.set()
            await self._hold.wait()
        return (0, '')

    def release(self) -> None:
        """Release the held first call."""
        self._hold.set()

    async def wait_entered(self, timeout: float = 5.0) -> None:
        """Block until the held first call has actually started (is in-flight)."""
        await asyncio.wait_for(self._entered.wait(), timeout=timeout)


def _inject_infra_red(worker: OfflineLaneWorker, failing_ids: list[str]) -> None:
    """Wire *worker* to simulate one confirmed-red INFRA sub-run (IB3/IB4).

    ``infra_runner`` reports a normal FAILED pass (``rc=1``);
    ``infra_confirmation_runner`` confirms the SAME *failing_ids* still fail
    in isolation — a genuine break, never the IB5 flake case (see
    :func:`_inject_infra_flake`). The numeric ``suite_runner`` is left
    untouched (stays whatever :func:`_build_infra_worker` wired it to —
    green by default), so this is the ONLY point this integration harness
    fakes the reify-subprocess boundary for an infra red pass; everything
    downstream (fingerprinting, dedup, fix-task file/update, escalation
    staging) is the real ``OfflineLaneWorker._handle_red_run`` chain.
    """

    async def _infra_runner(wt: Path, head: str, threads: int) -> tuple[int, str]:
        return (1, 'FAILED (injected, infra)')

    async def _infra_confirmation_runner(wt: Path, head: str) -> list[str]:
        return list(failing_ids)

    worker.infra_runner = _infra_runner
    worker.infra_confirmation_runner = _infra_confirmation_runner


def _inject_infra_flake(worker: OfflineLaneWorker) -> None:
    """Wire *worker* to simulate an INFRA flake: fails once, confirms clean (IB5).

    ``infra_runner`` reports a normal FAILED pass (``rc=1``); the isolated
    infra confirmation re-run finds NOTHING still failing (empty list) —
    intermittent nondeterminism, never the genuine break
    :func:`_inject_infra_red` simulates. The numeric ``suite_runner`` is
    left untouched (stays green).
    """

    async def _infra_runner(wt: Path, head: str, threads: int) -> tuple[int, str]:
        return (1, 'FAILED (injected, infra flake)')

    async def _infra_confirmation_runner(wt: Path, head: str) -> list[str]:
        return []

    worker.infra_runner = _infra_runner
    worker.infra_confirmation_runner = _infra_confirmation_runner


def _submitted_fix_task_arguments(task_client) -> dict:
    """Assert ``submit_fix_task`` was awaited exactly once; return its sole argument block."""
    task_client.submit_fix_task.assert_awaited_once()
    return task_client.submit_fix_task.await_args.args[0]


def _l0_info_escalations(escalation_queue: EscalationQueue, task_id: str) -> list:
    """Pending L0 (agent→steward) info escalations filed for *task_id*."""
    return escalation_queue.get_by_task(task_id, status='pending', level=0)


def _l2_blocker_escalations(escalation_queue: EscalationQueue, task_id: str) -> list:
    """Pending L2 (born-at-L2) blocker escalations filed for *task_id*."""
    return escalation_queue.get_by_task(task_id, status='pending', level=2)


async def _drive_infra_reds(
    harness: Harness,
    repo: Path,
    worker: OfflineLaneWorker,
    failing_ids: list[str],
    n: int,
) -> None:
    """Drive *n* same-failing-set confirmed-INFRA-red advances through the real chain.

    Injects the SAME confirmed infra break once (:func:`_inject_infra_red`)
    then, for *n* iterations, drives a real advance + the real
    on_merge_landed fan-out (:func:`_drive_advance`) followed by one full
    pass (:func:`_run_one_lane_pass`) — the same-fingerprint repeat-red
    sequence IB4's dedup scenario needs.
    """
    _inject_infra_red(worker, failing_ids)
    for _ in range(n):
        await _drive_advance(harness, repo)
        await _run_one_lane_pass(worker)


# ---------------------------------------------------------------------------
# IB1 — advance triggers a from-head infra sub-run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ib1_advance_triggers_from_head_infra_sub_run(
    harness,
    git_ops,
    repo,
    tmp_path,
    caplog,
):
    """IB1 — a landed advance triggers an infra sub-run snapshotted from the
    CURRENT main head, never the advisory trigger SHA passed to
    on_post_merge; the numeric sub-run runs too, at that SAME head.

    Negative control: reverting IE1's infra leg in ``_run_once`` (the infra
    sub-run block that awaits ``self.infra_runner`` when
    ``offline_lane_infra_enabled`` is True) would leave ``infra_calls``
    empty and fail this test.
    """
    infra_calls: list[tuple] = []
    suite_calls: list[tuple] = []

    async def _infra_runner(wt, head, threads):
        infra_calls.append((wt, head, threads))
        return (0, '')

    async def _suite_runner(wt, head, threads):
        suite_calls.append((wt, head, threads))
        return (0, '')

    worker = _build_infra_worker(
        git_ops,
        tmp_path,
        suite_runner=_suite_runner,
        infra_runner=_infra_runner,
    )
    _wire_lane(harness, worker)

    with caplog.at_level(logging.INFO):
        base_sha, head_sha = await _drive_advance(harness, repo)
        await _run_one_lane_pass(worker)

    real_head = await git_ops.get_main_sha()
    assert real_head == head_sha
    assert len(infra_calls) == 1
    assert infra_calls[0][1] == real_head, (
        'infra_runner must be invoked with the real from-head snapshot, '
        'never the advisory trigger SHA'
    )
    assert len(suite_calls) == 1
    assert suite_calls[0][1] == real_head, (
        'the numeric sub-run must also run, at the SAME snapshot head as the infra sub-run'
    )

    assert 'offline-lane: on_post_merge' in caplog.text
    assert base_sha[:12] in caplog.text
    assert head_sha[:12] in caplog.text


# ---------------------------------------------------------------------------
# IB2 — never a gate (C7/§6, load-bearing): infra sub-run in-flight
# ---------------------------------------------------------------------------


async def _assert_infra_never_a_gate(
    harness: Harness,
    worker: OfflineLaneWorker,
    runner: _ControllableInfraRunner,
    repo: Path,
    git_ops: GitOps,
) -> None:
    """Assert the never-a-gate invariant (C7/§6) while an INFRA run is held in-flight.

    Called with the FIRST infra pass already confirmed in-flight (the
    caller awaited ``runner.wait_entered()``) and released only after this
    returns, so the bounded ``wait_for`` below proves the synchronous
    on_merge_landed fan-out never blocks on it.

    Adapts ``test_offline_lane_integration.py``'s ``_assert_never_a_gate``
    for the infra seam — same four assertions, held on the INFRA sub-run
    in-flight rather than the numeric one:

    (1)+(2) ``harness._note_merge_all`` — the exact ``on_merge_landed``
    callback ``SpeculativeMergeWorker`` invokes (``harness.py:4979``) —
    must return promptly and must set ``worker._dirty`` (arming a
    coalesced re-run).
    (3) A raising notifiee is fail-open: ``_note_offline_lane``'s own
    try/except (``harness.py:5072-5079``) swallows it — the SAME shape
    ``SpeculativeMergeWorker`` independently wraps this exact call in
    (``merge_queue.py:10578-10596``, belt-and-suspenders) — so the merge
    still lands (proven behaviorally below via the service-restart
    coordinator fan-out still running, NOT by log text).
    (4) Neither case ever halts the merge queue or files an escalation —
    the only halt/gate-adjacent side effects reachable from this call path.
    """
    assert harness.get_merge_halt_status() == {'wired': False}

    base_sha = await git_ops.get_main_sha()
    head_sha = await _advance_main(repo)
    await asyncio.wait_for(
        harness._note_merge_all('task-2', base_sha, head_sha),
        timeout=0.5,
    )
    assert worker._dirty is True, (
        'a landed advance during an in-flight infra run must arm a coalesced re-run'
    )
    assert harness.get_merge_halt_status() == {'wired': False}
    assert cast(EscalationQueue, worker.escalation_queue).get_pending() == []

    async def _raising_notifiee(task_id: str, base: str, head: str) -> None:
        await worker.on_post_merge(task_id, base, head)
        raise RuntimeError('infra lane boom')

    harness._offline_lane_notifiee = _raising_notifiee
    coord = MagicMock()
    coord.note_merge = AsyncMock()
    harness._service_restart_coordinators = [coord]

    base2 = head_sha
    head2 = await _advance_main(repo)
    await harness._note_merge_all('task-3', base2, head2)  # must not raise

    # Fail-open is proven behaviorally: the merge still landed and ran the
    # service-restart coordinator fan-out despite the raising notifiee.
    coord.note_merge.assert_awaited_once_with(
        'task-3',
        base2,
        head2,
        prefetched_diff=[],
    )
    assert harness.get_merge_halt_status() == {'wired': False}
    assert cast(EscalationQueue, worker.escalation_queue).get_pending() == []


@pytest.mark.asyncio
async def test_ib2_infra_run_in_flight_never_gates_merge(harness, git_ops, repo, tmp_path):
    """IB2 (C7/§6, load-bearing) — a merge-landed notification while the
    INFRA sub-run is in-flight must never gate the merge: it returns
    promptly (never blocks the synchronous notifiee call on the in-flight
    infra run), sets _dirty for a coalesced re-run, is fail-open when the
    notifiee raises, and never files an escalation (the only halt/gate-
    adjacent side-effect reachable from this call path).

    The numeric suite_runner returns green immediately (the
    :func:`_build_infra_worker` default), so the numeric leg completes and
    the INFRA leg is the one held in-flight by :class:`_ControllableInfraRunner`.

    Negative control: any inline/blocking infra run on the notifiee path,
    or a halt/escalation reachable from this call path, makes this RED.
    """
    runner = _ControllableInfraRunner()
    worker = _build_infra_worker(git_ops, tmp_path, infra_runner=runner)
    _wire_lane(harness, worker)

    await _drive_advance(harness, repo)
    lane_task = asyncio.create_task(_run_lane(worker, expected_passes=1))
    await runner.wait_entered()

    await _assert_infra_never_a_gate(harness, worker, runner, repo, git_ops)

    runner.release()
    await lane_task


# ---------------------------------------------------------------------------
# IB3 — confirmed red files a normal fix task + L0 info escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ib3_confirmed_infra_red_files_normal_fix_task_and_info_escalation(
    harness,
    git_ops,
    repo,
    tmp_path,
):
    """IB3 — a confirmed-red INFRA run files a NORMAL pending fix task
    (never the b3_gate red-main fix-forward path) plus a non-blocking L0
    info escalation, leaving the merge queue itself completely untouched.

    Proves the infra red routes through β3's ``_handle_red_run`` with
    ``confirmation_runner=infra_confirmation_runner`` injected (task 1959,
    IE1). Negative control: routing an infra red to the red-main
    fix-forward path, or touching the merge queue from this call path,
    makes this RED.
    """
    from orchestrator.workflow import compute_failing_test_set_fingerprint

    failing_ids = ['infra::test_cgroup_burn.sh', 'infra::test_reflink.sh']
    worker = _build_infra_worker(git_ops, tmp_path)
    _inject_infra_red(worker, failing_ids)
    _wire_lane(harness, worker)

    _, head_sha = await _drive_advance(harness, repo)
    await _run_one_lane_pass(worker)

    arguments = _submitted_fix_task_arguments(worker.task_client)
    assert arguments['status'] == 'pending'
    assert arguments['metadata']['merge_lane'] == 'normal', (
        'a confirmed-red infra fix task must route through the standard '
        'TDD→PR→merge gate, never the b3_gate red-main fix-forward path'
    )
    assert arguments['metadata']['failing_tests'] == sorted(failing_ids)
    assert arguments['metadata']['suspect_ranges'] == [head_sha], (
        'no prior green run is recorded yet, so the suspect range is just the head'
    )

    fp = compute_failing_test_set_fingerprint(failing_ids)
    task_id = worker.open_fix_tasks[fp]
    escalations = _l0_info_escalations(cast(EscalationQueue, worker.escalation_queue), task_id)
    assert len(escalations) == 1
    assert escalations[0].severity == 'info'
    assert escalations[0].agent_role == 'orchestrator-offline-lane'

    # The merge queue itself is left completely untouched by the infra red path.
    assert harness.get_merge_halt_status() == {'wired': False}


# ---------------------------------------------------------------------------
# IB4 — dedup across advances: same-set recurrence updates, never duplicates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ib4_same_infra_set_recurrence_updates_not_duplicates(
    harness,
    git_ops,
    repo,
    tmp_path,
):
    """IB4 — a SECOND infra-red advance with the SAME failing-test set
    updates the existing fix task (appends a suspect range) rather than
    filing a duplicate task or raising a second escalation.

    Confirms the content-agnostic failing-test-set fingerprint from β3
    dedups infra failures exactly like numeric IDs. Negative control:
    re-keying on main_sha or filing a duplicate makes this RED.
    """
    from orchestrator.workflow import compute_failing_test_set_fingerprint

    failing_ids = ['infra::test_cgroup_burn.sh']
    worker = _build_infra_worker(git_ops, tmp_path)
    _wire_lane(harness, worker)

    await _drive_infra_reds(harness, repo, worker, failing_ids, 2)

    cast(AsyncMock, worker.task_client).submit_fix_task.assert_awaited_once()
    cast(AsyncMock, worker.task_client).append_suspect_range.assert_awaited_once()

    fp = compute_failing_test_set_fingerprint(failing_ids)
    task_id = worker.open_fix_tasks[fp]
    escalations = _l0_info_escalations(cast(EscalationQueue, worker.escalation_queue), task_id)
    assert len(escalations) == 1, 'a same-set recurrence must not raise a second L0 escalation'
