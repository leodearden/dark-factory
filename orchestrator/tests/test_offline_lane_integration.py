"""Integration gate for the offline deep-test lane (task 1955, ζ).

Stands up the REAL trigger→worker→warm-worktree→failure-handling chain
end-to-end and runs PRD §8 boundary scenarios B1-B8 for
``offline-deep-test-lane-worker.md``.  No production module is touched by
this task — every scenario below asserts already-landed behaviour from:

* β1 (task 1951) — ``Harness._offline_lane_notifiee`` / ``_note_merge_all`` /
  ``_note_offline_lane`` fan-out (``harness.py``).
* δ  (task 1952) — the persistent ``_offline-deep`` warm worktree
  (``git_ops.py``: ``persistent_offline_deep_worktree_path``,
  ``reset_persistent_offline_deep_worktree``, the ``cleanup_merge_worktree``
  prune exemption).
* β2 (task 1953) — ``OfflineLaneWorker`` (``offline_lane.py``): the
  coalescing ``run()`` loop, the always-from-head ``_run_once`` snapshot,
  and the injectable ``suite_runner`` / ``confirmation_runner`` seams.
* β3 (task 1954) — ``OfflineLaneWorker._handle_red_run`` (confirmation,
  fingerprinting, dedup'd fix-task file/update, L0/L2 escalation staging).

STRATEGY — injected-seam integration harness: the real reify-side heavy
suite run (Part A, cross-project, not yet on reify ``main``) is faked ONLY
at the ``suite_runner`` / ``confirmation_runner`` boundary — every other
component in the chain (the git repo, ``GitOps``, ``Harness`` fan-out,
``OfflineLaneWorker``, ``EscalationQueue``) is real.  See the task's
``.task/plan.json`` analysis for the full rationale.

Scenario map (PRD §8):

  B1 - advance triggers a from-head run
  B2 - coalescing under a burst of advances
  B3 - never-a-gate (C7): prompt return, fail-open, no halt/gate
  B4 - confirmed red files a normal fix task + L0 info escalation
  B5 - dedup: a same-set recurrence updates rather than duplicates
  B6 - flake filtered: fail-then-pass on confirmation is not a red
  B7 - stall promotes to a born-at-L2 escalate_blocker
  B8 - target/worktree ISOLATION (offline-deep path != merge-verify path)

OUT OF SCOPE:

* B8 asserts isolation only — never real reify compile timing, which is a
  reify-build capability owned by reify:4916/4913 (not deliverable here).
* B9 (flip-live-and-atomic) is owned by ε2 (deps ζ + reify:A4).
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import logging
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.offline_lane import OfflineLaneWorker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout bounds (task 4030) — named constants for the bounded waits below,
# replacing six near-identical inline 30.0 literals (the duplication that let
# a single false citation drift into three near-verbatim docstring copies).
# Modeled on the repo's own constant-plus-derivation-comment convention:
# _orch_helpers.py's CANCEL_SCOPE_BARRIER_TIMEOUT / CANCEL_SCOPE_PURE_UNIT_TIMEOUT
# and test_lane_lock_leak_guard.py's _FOREIGN_HOLDER_* block. The full floor
# (task 3451's measured spawn latency) / ceiling (the pyproject 60s global)
# derivation lives in `_run_lane`'s docstring below; the executable pins are
# `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling` and
# `test_every_composing_caller_carries_a_timeout_override`.
# ---------------------------------------------------------------------------

# Bound for a single `_run_lane` pass / `_ControllableSuiteRunner.wait_entered`
# call. See `_run_lane`'s docstring for the full derivation.
_LANE_PASS_BOUND_SECS = 30.0

# Bound for `_assert_never_a_gate`'s `_note_offline_lane` promptness check.
_NOTE_OFFLINE_LANE_BOUND_SECS = 0.5

# Bound for `_assert_never_a_gate`'s `_note_merge_all` promptness check.
_NOTE_MERGE_ALL_BOUND_SECS = 15.0

# Task 3451's measured worst-case happy-path subprocess spawn latency (n=3:
# 2.13/3.10/4.71, load-per-core 6.6) -- the FLOOR authority
# _LANE_PASS_BOUND_SECS is sized against. See `_run_lane`'s docstring for the
# full derivation.
_MEASURED_SPAWN_LATENCY_SECS = 4.71

# task 3836's own fraction, reused rather than re-derived: the CEILING a
# marker-less test's worst-case bounded-wait budget must stay under, as a
# fraction of the effective per-test pytest-timeout. The remaining 40% is
# headroom for real-git fixture setup/teardown sharing the same per-test
# budget (pytest-timeout times the whole `pytest_runtest_protocol`, not just
# the call phase, whenever `func_only` is unset -- true everywhere in this
# repo). See `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling`
# for the full derivation.
_MARKERLESS_CEILING_FRACTION = 0.6

# The measured worst-case count of subprocess spawns inside a single
# `_run_lane` window -- the multiplier `_LANE_PASS_BOUND_SECS`'s floor check
# is sized against. Measured in the SIBLING infra module's
# `test_ib6_real_infra_runner_over_stub_classified_set_files_fix_task`: 3 git
# spawns from `_run_once` plus 2 real infra `run_all.sh` spawns. See
# `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling` for the
# full derivation.
_SPAWNS_PER_LANE_PASS_WORST_CASE = 5

# Task 4203 — the multiplicands `_required_timeout_secs` below prices at
# `_MEASURED_SPAWN_LATENCY_SECS` per spawn. MEASURED by
# `test_out_of_bound_spawn_counts_are_measured_not_asserted`, not
# hand-counted: the `repo` fixture's `_setup_repo` spawns `init`, `config`
# x2, `add`, `commit`.
_SPAWNS_PER_REPO_FIXTURE = 5

# Task 4203 — the other multiplicand of `_required_timeout_secs`. MEASURED,
# not hand-counted (see `test_out_of_bound_spawn_counts_are_measured_not_asserted`):
# `_drive_advance` costs `get_main_sha` (1) plus `_advance_main`'s `add` /
# `commit` / `rev-parse` (3) = 4. This is NOT the naive "5" a reading of
# `_note_merge_all`'s production source would suggest (`harness.py:10377`
# unconditionally calls `get_merge_diff_files`, a real `git diff` spawn in
# production) — under THIS module's `harness` fixture, `h.git_ops
# .get_merge_diff_files` is replaced wholesale with an
# `AsyncMock(return_value=([], None))`, so that leg never reaches a
# subprocess at all here. `_maybe_pipeline_landing_tripwire` (also reached
# from `_note_merge_all`) is confirmed to no-op whenever
# `config.git.load_bearing_oracle_cmd` is falsy (`harness.py:10438-10444`),
# which it is here (the `git_config` fixture never sets it).
_SPAWNS_PER_DRIVE_ADVANCE = 4

# Task 4203 (reviewer amendment) — the THIRD multiplicand, needed only by
# `test_b3_never_a_gate`'s `out_of_bound_spawns` row. MEASURED, not
# hand-counted (see `test_out_of_bound_spawn_counts_are_measured_not_asserted`):
# `_assert_never_a_gate` performs one `git_ops.get_main_sha()` (1) plus TWO
# `_advance_main` rounds (3 each: `add` / `commit` / `rev-parse`) = 7 — NOT
# `_SPAWNS_PER_DRIVE_ADVANCE` (4), because it drives an extra `_advance_main`
# beyond a single `_drive_advance`'s shape, and NOT 8 (2x
# `_SPAWNS_PER_DRIVE_ADVANCE`) either, since only ONE `get_main_sha` is
# fetched up front (the second round reuses its own returned head as the new
# base). Both `_note_merge_all` calls inside it contribute zero spawns here,
# same stub as `_SPAWNS_PER_DRIVE_ADVANCE`'s. An earlier hand approximation
# priced this whole function as "one _drive_advance-equivalent round" (4),
# undercounting the true value by 3 — exactly the drift a measuring test
# exists to catch.
_SPAWNS_PER_ASSERT_NEVER_A_GATE = 7


def _required_timeout_secs(bounded_secs: float, out_of_bound_spawns: int) -> float:
    """Task 4203 — THE canonical sizing model for `@pytest.mark.timeout`
    OVERRIDES on tests that compose past a single `_run_lane` pass. This is
    the single, callable statement of the model; both
    `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling` and
    `test_every_composing_caller_carries_a_timeout_override` call it rather
    than re-deriving or re-stating it in prose, so the rule cannot drift into
    per-callsite copies the way `_drive_advance`'s spawn count already had
    (two landed docstrings undercounted it, inconsistently, before this
    task).

    THE MODEL: a composing test's effective per-test pytest-timeout must
    cover its bounded-wait sum (*bounded_secs* — sums of
    `_LANE_PASS_BOUND_SECS`-style waits the test's own body composes) PLUS
    its counted out-of-bound real-git subprocess spawns (*out_of_bound_spawns*
    — real git work the test does OUTSIDE any bounded `wait_for`/`_run_lane`
    window), each spawn priced at the worst-case measured latency
    `_MEASURED_SPAWN_LATENCY_SECS`. Every term is an already-measured,
    already-pinned quantity — task 3451's 4.71s, and the
    `_SPAWNS_PER_REPO_FIXTURE` / `_SPAWNS_PER_DRIVE_ADVANCE` counts this
    task's own measuring test pins — nothing guessed, matching the bar
    `test_lane_bounds_...`'s own floor check already sets for itself ("the
    multiplier is DERIVED, not guessed").

    WHY THE ADDITIVE TERM EXISTS: pytest-timeout 2.4.0 installs its timer in
    `pytest_runtest_protocol` whenever `func_only` is False — unset
    repo-wide, so true for every test here — meaning the per-test budget
    covers fixture setup/teardown and all real-git test-body work, not just
    the bounded waits. Before this task, the marker-carrying guard
    (`test_every_composing_caller_carries_a_timeout_override`) compared a
    marker only against the bounded-wait sum (a bare `value > worst_case`),
    reserving ZERO headroom for that real-git work — while the marker-less
    guard (immediately above) reserved 40% of the budget for exactly it. That
    asymmetry, not any one test's marker value, is what this task fixes:
    `test_b7_stall_promotes_to_blocker` is simply the first test where the
    missing term became numerically visible (see its updated
    `@pytest.mark.timeout` comment for the worked derivation).

    SCOPE — applies to the marker-carrying OVERRIDES only, i.e. a number this
    suite CHOOSES. The marker-less population deliberately keeps
    `_MARKERLESS_CEILING_FRACTION` as its proxy instead of this model — see
    the reconciliation assertion and comment in
    `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling` for
    why (not restated here, to avoid recreating the near-verbatim docstring
    copies task 4030 removed).
    """
    return bounded_secs + out_of_bound_spawns * _MEASURED_SPAWN_LATENCY_SECS


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
    """Real GitOps bound to the real tmp repo (B1 from-head snapshots, B8 isolation)."""
    return GitOps(git_config, repo)


@pytest.fixture
def harness(git_config: GitConfig, git_ops: GitOps, mock_orch_config) -> Harness:
    """Minimally-constructed real Harness wired to the real tmp-repo GitOps.

    Modeled on ``test_harness_offline_lane_trigger.py``'s ``harness``
    fixture; the delta is that ``h.git_ops`` is replaced wholesale by the
    real tmp-repo-backed :func:`git_ops` fixture (rather than left as
    Harness's own internally-constructed GitOps over an un-git-initialized
    ``tmp_path``), so that real head snapshots / worktree resets are
    actually exercised end-to-end.
    """
    mock_orch_config.git = git_config
    mock_orch_config.project_root = git_ops.project_root
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.event_store = MagicMock()
    h.git_ops = git_ops
    h.git_ops.get_merge_diff_files = AsyncMock(return_value=([], None))
    h._service_restart_coordinators = []
    return h


def _build_worker(
    git_ops: GitOps,
    tmp_path: Path,
    *,
    suite_runner=None,
    confirmation_runner=None,
    task_client=None,
    escalation_queue: EscalationQueue | None = None,
    git_overrides: dict | None = None,
) -> OfflineLaneWorker:
    """Build a real OfflineLaneWorker wired for the integration harness.

    The ONLY injection point is the reify-subprocess boundary
    (``suite_runner`` / ``confirmation_runner``) — everything else is real:
    *git_ops* is the shared real tmp-repo GitOps (see the ``git_ops``
    fixture) so the worker's head snapshots / worktree resets land in the
    SAME repo :func:`_advance_main` commits to; *escalation_queue* defaults
    to a real :class:`EscalationQueue` (never a MagicMock) so B4/B5/B7 can
    assert against real on-disk escalation records; *task_client* defaults
    to a fake ``OfflineLaneTaskClient`` (a bare ``AsyncMock`` — its
    ``submit_fix_task`` / ``append_suspect_range`` / ``get_status``
    children auto-vivify as awaitable ``AsyncMock``s, per the β3 unit-test
    convention in ``test_offline_lane.py``) with ``get_status`` pre-set to
    the steady in-flight ``'in-progress'`` state and ``submit_fix_task``
    pre-set to return a stable ``'fix-task-1'`` id (B4/B5/B7 only ever have
    one fingerprint open at a time, so one stable id suffices).
    """
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = git_ops.project_root
    config.git = GitConfig(**(git_overrides or {}))

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
        suite_runner=suite_runner,
        confirmation_runner=confirmation_runner,
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
    harness: Harness, repo: Path, task_id: str = 'task-1',
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
    worker: OfflineLaneWorker, expected_passes: int, *, timeout: float = _LANE_PASS_BOUND_SECS,
) -> None:
    """Drive worker.run() as a real background task until *expected_passes*
    full passes (``_run_once`` calls) have COMPLETED, then cancel the loop
    task.

    Wraps ``worker._run_once`` itself — NOT ``suite_runner`` alone — with a
    call counter, so ``done`` is only set once a pass, INCLUDING any
    ``_handle_red_run`` confirmation/fix-task/escalation work it triggers,
    has fully returned. Counting at the ``suite_runner`` boundary instead
    would race the red path: every seam this harness injects (a bare
    ``async def ...: return ...``, or an ``AsyncMock``) happens not to
    yield to the event loop, so today ``_handle_red_run`` always finishes
    before the loop next suspends at ``_wake.wait()`` — but that is an
    accident of the fakes, not a guarantee, and cancelling right after
    ``suite_runner`` returns would race ``_handle_red_run`` mid-flight the
    moment a seam awaits real I/O. Restores the original ``_run_once`` in
    the ``finally`` block so repeated calls (``_drive_reds`` drives 2-3
    passes per scenario) never nest counting wrappers.

    Modeled on ``test_offline_lane.py``'s
    ``test_loop_coalesces_to_exactly_one_rerun`` cancel-after-N pattern,
    generalized from a fixed N=1 to an arbitrary pass count (B1's single
    pass via :func:`_run_one_lane_pass`; B2's held-pass-plus-one-coalesced-
    rerun burst uses N=2). Requires ``worker._dirty`` (or the wake event) to
    already be set by a prior trigger, exactly as production wiring would
    leave it.

    ``timeout`` defaults to 30.0, not a tight bound: the clock starts at
    ``asyncio.create_task`` above, so it also covers the real-git test-body
    work the caller does between entering the held pass and releasing it
    (each :func:`_drive_advance` costs `_SPAWNS_PER_DRIVE_ADVANCE` subprocess
    spawns — MEASURED, not hand-counted, by
    `test_out_of_bound_spawn_counts_are_measured_not_asserted`) — not just
    the lane pass(es) themselves. Task 3451
    measured worst-case single subprocess spawn latency at 4.71s on this
    host under load; a caller can easily need several spawns inside this
    window. Same load-sensitive full-suite-flake class as
    1335/1836/2819/3451/3491 (task 3832). FLOOR (task 3451): worst-case
    happy-path subprocess spawn latency measured at 4.71s (n=3:
    2.13/3.10/4.71, load-per-core 6.6) — this, not task 3491, is the genuine
    precedent for ``_LANE_PASS_BOUND_SECS``. CEILING: the 60s global
    pytest-timeout (``orchestrator/pyproject.toml``, ``timeout_method =
    "thread"``, ``--max-worker-restart=0``). Task 3491 is precedent AGAINST
    a bound near that ceiling, not for one: it REJECTED a 30s ceiling for
    this exact collision and landed ``asyncio.wait_for(..., timeout=5)``
    instead (``test_usage_gate.py``:328,790) — its scope was exclusively
    ``test_usage_gate.py``; it never touched any offline-lane module. Both
    halves are pinned executably by
    ``test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling``
    below. Widening THIS bound alone can never make a broken
    staging pass — it only lengthens how long a genuinely broken SINGLE pass
    takes to fail. That safety property does not compose for free: callers
    chaining N passes (:func:`_drive_reds`) sum this bound N times, so a
    worst case can now approach or exceed this module's pyproject-configured
    60s per-test timeout before this function's own ``wait_for`` ever fires —
    trading a clean, well-located ``TimeoutError`` here for pytest-timeout's
    blunter thread-mode worker kill instead (task 3832 review; see
    ``orchestrator/pyproject.toml``'s ``timeout``/``timeout_method``
    comment). Multi-pass callers whose worst-case sum is at or above that
    ceiling carry their own ``@pytest.mark.timeout`` override — see
    ``test_b5_same_set_recurrence_updates_not_duplicates``,
    ``test_b7_stall_promotes_to_blocker``, and ``test_b3_never_a_gate``
    below — a rule ``test_every_composing_caller_carries_a_timeout_override``
    now enforces rather than merely describes.

    The six 30.0 defaults rest on task 3451's measurement above, not on the
    retracted task 3491 claim — narrowing them would re-introduce the flake
    commits 289ae708bb / 80ae271a34 fixed. The gap this task (4030) closed
    was composition coverage (the override rule the preceding paragraph
    describes), not the bound value itself.
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


async def _run_one_lane_pass(worker: OfflineLaneWorker, *, timeout: float = _LANE_PASS_BOUND_SECS) -> None:
    """Drive worker.run() as a real background task for exactly one pass (B1)."""
    await _run_lane(worker, 1, timeout=timeout)


class _ControllableSuiteRunner:
    """A suite_runner whose FIRST call blocks in-flight until released (B2/B3).

    Records every head it is invoked with, in call order. Only the first
    call blocks (on an internal ``asyncio.Event`` gate) — later calls
    return immediately — so a single instance can drive both the held
    initial pass and the subsequent coalesced re-run(s) in one scenario.
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

    async def wait_entered(self, timeout: float = _LANE_PASS_BOUND_SECS) -> None:
        """Block until the held first call has actually started (is in-flight).

        ``timeout`` defaults to 30.0 for the same reason as ``_run_lane``'s
        default above (see that function's docstring for the full FLOOR/
        CEILING derivation): the ``_run_lane`` task created just before this
        call races the SAME clock, and this call is entered only after
        ``OfflineLaneWorker._run_once`` has already done its own real-git
        work (``get_main_sha`` plus a persistent-worktree reset — 3-5
        subprocess spawns) — a tighter bound here would fire before
        ``_run_lane``'s ever could. Same safety property: widening can never
        make a broken staging pass, it only lengthens how long a genuinely
        wedged runner takes to fail.
        """
        await asyncio.wait_for(self._entered.wait(), timeout=timeout)


async def _assert_never_a_gate(
    harness: Harness,
    worker: OfflineLaneWorker,
    runner: _ControllableSuiteRunner,
    repo: Path,
    git_ops: GitOps,
) -> None:
    """Assert the never-a-gate invariant (C7) while a lane run is held in-flight.

    Called with the FIRST pass already confirmed in-flight (the caller
    awaited ``runner.wait_entered()``) and released only after this
    returns, so the bounded ``wait_for`` below proves the synchronous
    on_merge_landed fan-out never blocks on it.

    (1)+(2) ``harness._note_offline_lane`` — the synchronous notifiee call
    ``_note_merge_all`` (the exact ``on_merge_landed`` callback
    ``SpeculativeMergeWorker`` invokes, ``harness.py:4979``) awaits AHEAD of
    its diff fetch — must return promptly. ``_note_merge_all`` itself is
    then exercised immediately after, at a loose 15.0s bound (well outside
    the flake band, but still catching a genuine block), with ``_dirty``
    explicitly reset just before so the ``worker._dirty`` assertion below
    re-proves ITS OWN fan-out sets it, not merely the ``_note_offline_lane``
    call above (task 3832 review).
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
    # Bounded at 0.5s (task 3832 assessment, NOT the _run_lane flake class):
    # this window covers only ``harness._note_offline_lane`` ->
    # ``worker.on_post_merge`` (harness.py:9333-9349, offline_lane.py:352-
    # 366), a bare enqueue-and-return (sets ``_dirty`` + an event, no
    # subprocess spawn) — exactly the documented "MUST enqueue-and-return
    # promptly ... rather than perform the deep-test run inline" contract
    # (harness.py:9342-9348). This is a genuine promptness assertion (proves
    # the synchronous on_merge_landed fan-out never blocks on the in-flight
    # run); widening it would weaken what it's testing.
    #
    # ``_note_merge_all`` is called separately, right after, at a much
    # looser 15.0s bound rather than left fully unbounded (task 3832
    # review): unlike ``_note_offline_lane`` alone, it unconditionally runs
    # ``git_ops.get_merge_diff_files`` (a real ``git diff`` subprocess spawn)
    # once ``_note_offline_lane`` returns, so bounding IT at 0.5s would sit
    # in the same load-sensitive flake class this task (3832) exists to
    # remove (task 3451 measured 4.71s worst-case single-spawn latency under
    # load). 15.0s sits comfortably outside that flake band (>3x the
    # measured worst-case single spawn) while still catching a genuine block
    # in the production ``on_merge_landed`` callback this call IS
    # (harness.py:4979) — restoring promptness coverage of the real
    # callback that narrowing the FIRST bound to ``_note_offline_lane``
    # alone would otherwise drop entirely. The resulting double-invoke of
    # ``on_post_merge`` is safe — it is idempotent (sets ``_dirty = True``
    # and an already-set ``asyncio.Event``) — so the assertions below still
    # exercise the full fan-out unchanged. ``_dirty`` is explicitly reset
    # just before this call so the assertion below re-proves
    # ``_note_merge_all``'s OWN fan-out sets it, rather than passing
    # vacuously on the strength of the ``_note_offline_lane`` call above.
    await asyncio.wait_for(
        harness._note_offline_lane('task-2', base_sha, head_sha),
        timeout=_NOTE_OFFLINE_LANE_BOUND_SECS,
    )
    worker._dirty = False
    await asyncio.wait_for(
        harness._note_merge_all('task-2', base_sha, head_sha),
        timeout=_NOTE_MERGE_ALL_BOUND_SECS,
    )
    assert worker._dirty is True, (
        'a landed advance during an in-flight run must arm a coalesced re-run'
    )
    assert harness.get_merge_halt_status() == {'wired': False}
    assert cast(EscalationQueue, worker.escalation_queue).get_pending() == []

    async def _raising_notifiee(task_id: str, base: str, head: str) -> None:
        await worker.on_post_merge(task_id, base, head)
        raise RuntimeError('lane boom')

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
        'task-3', base2, head2, prefetched_diff=[],
    )
    assert harness.get_merge_halt_status() == {'wired': False}
    assert cast(EscalationQueue, worker.escalation_queue).get_pending() == []


def _inject_red(worker: OfflineLaneWorker, failing_ids: list[str]) -> None:
    """Wire *worker* to simulate one confirmed-red suite pass (B4/B5/B7).

    ``suite_runner`` reports a normal FAILED pass (``rc=1``);
    ``confirmation_runner`` confirms the SAME *failing_ids* still fail in
    isolation — a genuine break, never the B6 flake case (see
    :func:`_inject_flake`). This is the ONLY point this integration harness
    fakes the reify-subprocess boundary for a red pass; everything
    downstream (fingerprinting, dedup, fix-task file/update, escalation
    staging) is the real ``OfflineLaneWorker._handle_red_run`` chain.
    """

    async def _suite_runner(wt: Path, head: str, threads: int) -> tuple[int, str]:
        return (1, 'FAILED (injected)')

    async def _confirmation_runner(wt: Path, head: str) -> list[str]:
        return list(failing_ids)

    worker.suite_runner = _suite_runner
    worker.confirmation_runner = _confirmation_runner


def _inject_flake(worker: OfflineLaneWorker) -> None:
    """Wire *worker* to simulate a flake: fails once, but confirms clean (B6).

    ``suite_runner`` reports a normal FAILED pass (``rc=1``); the isolated
    confirmation re-run finds NOTHING still failing (empty list) —
    intermittent nondeterminism, never the genuine break :func:`_inject_red`
    simulates.
    """

    async def _suite_runner(wt: Path, head: str, threads: int) -> tuple[int, str]:
        return (1, 'FAILED (injected, flake)')

    async def _confirmation_runner(wt: Path, head: str) -> list[str]:
        return []

    worker.suite_runner = _suite_runner
    worker.confirmation_runner = _confirmation_runner


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


async def _drive_reds(
    harness: Harness, repo: Path, worker: OfflineLaneWorker, failing_ids: list[str], n: int,
) -> None:
    """Drive *n* same-failing-set confirmed-red advances through the real chain.

    Injects the SAME confirmed break once (:func:`_inject_red`) then, for
    *n* iterations, drives a real advance + the real on_merge_landed fan-out
    (:func:`_drive_advance`, mirroring B1-B4) followed by one full pass
    (:func:`_run_one_lane_pass`) — the same-fingerprint repeat-red sequence
    B5's dedup and B7's stall-promotion scenarios both need.

    Each pass sums its own 30s :func:`_run_lane` bound (see that function's
    docstring); a caller whose ``n`` (or count of calls) pushes the total at
    or above the 60s pyproject per-test timeout MUST carry its own
    ``@pytest.mark.timeout`` override (task 3832 review) — see B5/B7 below.
    This rule is enforced, not merely described, by
    ``test_every_composing_caller_carries_a_timeout_override``.
    """
    _inject_red(worker, failing_ids)
    for _ in range(n):
        await _drive_advance(harness, repo)
        await _run_one_lane_pass(worker)


def _terminal_status_task_client(status: str) -> AsyncMock:
    """A fake ``OfflineLaneTaskClient`` whose open fix task already reports a
    terminal non-done *status* (B7 arm 2 — ``'cancelled'`` / ``'deferred'``).

    :meth:`OfflineLaneWorker._maybe_promote_blocker` promotes on this arm
    immediately, regardless of the configured advance-count threshold.
    ``submit_fix_task`` / ``append_suspect_range`` behave as the
    ``_build_worker`` default (see its docstring) — only ``get_status``
    diverges.
    """
    task_client = AsyncMock()
    task_client.get_status = AsyncMock(return_value=status)
    task_client.submit_fix_task = AsyncMock(return_value='fix-task-1')
    return task_client


def _materialized_worktree_names(worktree_base: Path) -> set[str]:
    """Top-level entry names materialized under *worktree_base* so far (B8).

    A before/after diff of this set proves a lane reset touches ONLY its
    own persistent worktree directory, never the other lane's. Returns the
    empty set when *worktree_base* itself has not been created yet (the
    "before" case, prior to any worktree ever being registered).
    """
    if not worktree_base.exists():
        return set()
    return {p.name for p in worktree_base.iterdir()}


# ---------------------------------------------------------------------------
# Timeout-bound invariants (task 4030)
# ---------------------------------------------------------------------------

# Helper names whose presence in a test's OWN source means that test drives
# bounded waits its own body does not spell out. Widened by task 4203's
# review remediation from "drives at least one `_run_lane`-bounded pass" to
# that broader membership rule: `_run_lane` / `_run_one_lane_pass` /
# `_drive_reds` each compose `_LANE_PASS_BOUND_SECS`-bounded passes, and
# `_assert_never_a_gate` composes its own `_NOTE_OFFLINE_LANE_BOUND_SECS` +
# `_NOTE_MERGE_ALL_BOUND_SECS` pair (0.5 + 15.0 = 15.5s) without any lane
# pass at all. Used by `_lane_pass_composing_callers` below to DISCOVER the
# set of tests `test_every_composing_caller_carries_a_timeout_override` must
# classify, rather than relying solely on a hand-maintained table a new test
# can silently fall outside of.
#
# `_assert_never_a_gate`'s absence from this set was a real hole, not a
# hypothetical one: task 4203's own
# `test_out_of_bound_spawn_counts_are_measured_not_asserted` reaches 15.5s of
# bounded waits through that helper and NOTHING else, so the pre-remediation
# net never discovered it and it ran marker-less under the 60s pyproject
# default while requiring more than twice that.
_LANE_PASS_COMPOSING_HELPER_NAMES = frozenset(
    {'_run_lane', '_run_one_lane_pass', '_drive_reds', '_assert_never_a_gate'},
)


def _lane_pass_composing_callers(module_globals: dict) -> set[str]:
    """Names of module-level `test_*` functions whose OWN source calls one of
    `_LANE_PASS_COMPOSING_HELPER_NAMES` at least once.

    AST-based rather than a text/regex scan, so a PROSE mention of a helper
    name (this module's own docstrings name every member of the set in
    backticks) can never register as a false positive the way a file-wide
    regex could —
    only an actual `ast.Call` node naming the helper does. Deliberately does
    not resolve call arguments (e.g. whether a `_run_lane(...,
    expected_passes=N)` call's N is 1) — this is a coverage NET, not a
    worst-case calculator: it exists only to guarantee every test that
    touches these helpers is explicitly classified below, into either
    `worst_case_secs` or `single_pass_exempt`, so a newly added composing
    test can no longer silently land in neither.
    """
    callers: set[str] = set()
    for name, obj in module_globals.items():
        if not name.startswith('test_') or not inspect.isfunction(obj):
            continue
        tree = ast.parse(inspect.getsource(obj))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _LANE_PASS_COMPOSING_HELPER_NAMES
            ):
                callers.add(name)
                break
    return callers


def test_every_composing_caller_carries_a_timeout_override() -> None:
    """Every test that composes bounded waits PAST a single `_run_lane` pass
    must carry its own `@pytest.mark.timeout` override wide enough to cover
    its own worst-case bounded-wait sum — otherwise it can silently collide
    with this module's pyproject-configured 60s per-test default.

    Makes WORK item 2's open re-verification question ("does EVERY chaining
    caller carry such an override?") executable rather than prose, in two
    layers. COVERAGE: `_lane_pass_composing_callers` discovers every test
    whose own source calls any member of `_LANE_PASS_COMPOSING_HELPER_NAMES`
    at all — deliberately named there and not re-spelled here, so widening
    that set (as task 4203's remediation did, adding `_assert_never_a_gate`)
    cannot leave a stale hand-typed list behind in this docstring or in the
    `undeclared` assertion message below; every discovered name must
    appear in EITHER `worst_case_secs` below OR `single_pass_exempt` — so a
    test added tomorrow that calls e.g. `_drive_reds(..., 3)` without a
    `@pytest.mark.timeout` marker fails this test immediately instead of
    silently passing because nobody added a row for it. VALUE: for every
    `worst_case_secs` entry, its marker must exist and its value must
    actually clear the REQUIRED timeout (task 4203's `_required_timeout_secs`
    model) — the bounded-wait sum plus its out-of-bound real-git spawn
    allowance, not the bounded sum alone.

    `worst_case_secs` maps each composing test FUNCTION OBJECT — never a
    string, so a rename breaks this test loudly instead of silently dropping
    a row — to its worst-case bounded-wait sum, expressed in terms of this
    module's named timeout constants (never bare literals).

    `out_of_bound_spawns` (task 4203) maps the SAME function objects to their
    counted real-git subprocess spawns that happen OUTSIDE any bounded wait
    — the other multiplicand `_required_timeout_secs` prices at
    `_MEASURED_SPAWN_LATENCY_SECS` per spawn, expressed in this module's
    `_SPAWNS_PER_REPO_FIXTURE` / `_SPAWNS_PER_DRIVE_ADVANCE` constants, never
    bare literals. Every function classified in `worst_case_secs` must also
    appear here (enforced by a key-set equality assertion below) — a row
    added to one table without its counterpart in the other would silently
    size the requirement wrong.

    `single_pass_exempt` holds tests that call a composing helper but are
    proven to never exceed a single `_LANE_PASS_BOUND_SECS`-bounded pass —
    the marker-less budget
    `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling` pins.
    ``test_b2_coalesces_burst_of_advances_to_one_rerun`` is the only member:
    it starts `_run_lane(expected_passes=2)` via `asyncio.create_task` and
    then awaits `runner.wait_entered()` — those two bounded waits race the
    SAME wall clock (max, not sum). The rest — b1/b4/b6 — each drive exactly
    one `_run_one_lane_pass` call.

    Correlation between a marker and a SPECIFIC test's value is done via
    `fn.pytestmark` introspection — pytest's own applied-marker list for a
    given test function — rather than the coarse `_TIMEOUT_MARKER_RE` text
    scan `test_lane_lock_leak_guard.py`'s sibling invariant
    (`test_no_foreign_holder_consumer_opts_out_of_the_global_timeout`) uses.
    That invariant only needs to know whether ANY `@pytest.mark.timeout`
    appears ANYWHERE in a file, so a deliberately coarse regex is an
    acceptable, cheap check for it. This invariant is finer-grained: it must
    correlate a marker to a SPECIFIC test and compare its VALUE against that
    test's own computed worst case. `fn.pytestmark` gives that directly, with
    no false positives from this module's own prose mentions of the marker.
    Where two `@pytest.mark.timeout` markers are stacked on one test,
    `markers[0]` (not `markers[-1]`) is used — verified against the
    installed pytest/pytest-timeout: ``Item.get_closest_marker('timeout')``
    resolves to the FIRST-applied mark (the decorator closest to `def`),
    since `store_mark` appends each newly-applied decorator's mark to the
    END of the existing list and decorators apply bottom-up.
    """
    worst_case_secs = {
        test_b5_same_set_recurrence_updates_not_duplicates: 2 * _LANE_PASS_BOUND_SECS,
        test_b7_stall_promotes_to_blocker: 6 * _LANE_PASS_BOUND_SECS,
        test_b3_never_a_gate: (
            _LANE_PASS_BOUND_SECS
            + _NOTE_OFFLINE_LANE_BOUND_SECS
            + _NOTE_MERGE_ALL_BOUND_SECS
        ),
        # Task 4203 (review remediation) — the only row whose bounded waits
        # come entirely from `_assert_never_a_gate`: this test composes no
        # `_run_lane` pass at all, so there is no `_LANE_PASS_BOUND_SECS`
        # term. Discovered by the net only once `_assert_never_a_gate`
        # joined `_LANE_PASS_COMPOSING_HELPER_NAMES`.
        test_out_of_bound_spawn_counts_are_measured_not_asserted: (
            _NOTE_OFFLINE_LANE_BOUND_SECS + _NOTE_MERGE_ALL_BOUND_SECS
        ),
    }
    # Task 4203 — the other multiplicand `_required_timeout_secs` needs per
    # composing test: its counted out-of-bound real-git spawns (never
    # `_run_one_lane_pass`'s spawns, which occur INSIDE the `_run_lane` 30s
    # window and are already carried by worst_case_secs above). b5 and b7
    # drive exactly n x `_drive_advance` via `_drive_reds` (`for _ in
    # range(n)`) — b7's four `_drive_reds` calls sum n = 2+1+2+1 = 6. b3
    # drives one direct `_drive_advance` plus `_assert_never_a_gate`'s own
    # unbounded advance-and-notify work — MEASURED (reviewer amendment,
    # task 4203) at `_SPAWNS_PER_ASSERT_NEVER_A_GATE` = 7, not the earlier
    # hand approximation of one `_drive_advance`-equivalent round (4), which
    # undercounted by 3 (see that constant's comment for the breakdown).
    out_of_bound_spawns = {
        test_b5_same_set_recurrence_updates_not_duplicates: (
            _SPAWNS_PER_REPO_FIXTURE + 2 * _SPAWNS_PER_DRIVE_ADVANCE
        ),
        test_b7_stall_promotes_to_blocker: (
            _SPAWNS_PER_REPO_FIXTURE + 6 * _SPAWNS_PER_DRIVE_ADVANCE
        ),
        test_b3_never_a_gate: (
            _SPAWNS_PER_REPO_FIXTURE
            + _SPAWNS_PER_DRIVE_ADVANCE
            + _SPAWNS_PER_ASSERT_NEVER_A_GATE
        ),
        # Task 4203 (review remediation) — the ONLY row in either table that
        # pays `_SPAWNS_PER_REPO_FIXTURE` TWICE: once for the `repo` fixture
        # every test in this module transitively pulls, and once more for the
        # fresh `_setup_repo` this test deliberately runs on a clean tmp_path
        # in order to MEASURE that very constant. Its `_ControllableSuiteRunner`
        # / `_build_worker` / `_wire_lane` setup contributes nothing further —
        # measured to be spawn-free inside that test, not assumed.
        test_out_of_bound_spawn_counts_are_measured_not_asserted: (
            2 * _SPAWNS_PER_REPO_FIXTURE
            + _SPAWNS_PER_DRIVE_ADVANCE
            + _SPAWNS_PER_ASSERT_NEVER_A_GATE
        ),
    }
    assert set(worst_case_secs) == set(out_of_bound_spawns), (
        f'worst_case_secs and out_of_bound_spawns must classify exactly the '
        f'same composing tests — '
        f'{sorted(fn.__name__ for fn in set(worst_case_secs) ^ set(out_of_bound_spawns))} '
        f'appear in only one of the two tables, so a row added to one table '
        f'silently omits half of the _required_timeout_secs inputs for that '
        f'test.'
    )
    single_pass_exempt = {
        test_b1_advance_triggers_from_head_run,
        test_b2_coalesces_burst_of_advances_to_one_rerun,
        test_b4_confirmed_red_files_fix_task_and_info_escalation,
        test_b6_flake_filtered_by_confirmation_rerun,
    }

    discovered = _lane_pass_composing_callers(globals())
    classified = {fn.__name__ for fn in worst_case_secs} | {
        fn.__name__ for fn in single_pass_exempt
    }
    undeclared = discovered - classified
    assert not undeclared, (
        f'{sorted(undeclared)} call a bounded-wait-composing helper '
        f'({" / ".join(sorted(_LANE_PASS_COMPOSING_HELPER_NAMES))}) but are '
        f'classified in neither '
        f'worst_case_secs (composes past a single pass; needs its own '
        f'@pytest.mark.timeout override) nor single_pass_exempt (proven to '
        f'never exceed one bounded pass), in '
        f'test_every_composing_caller_carries_a_timeout_override. This is '
        f'exactly the drift this test exists to catch — classify it as one '
        f'or the other.'
    )

    for fn, worst_case in worst_case_secs.items():
        out_of_bound = out_of_bound_spawns[fn]
        required = _required_timeout_secs(worst_case, out_of_bound)
        spawn_allowance = out_of_bound * _MEASURED_SPAWN_LATENCY_SECS
        markers = [m for m in getattr(fn, 'pytestmark', []) if m.name == 'timeout']
        assert markers, (
            f'{fn.__name__} composes bounded waits to a worst case of '
            f'{worst_case}s plus {out_of_bound} out-of-bound real-git spawns '
            f'({out_of_bound} x {_MEASURED_SPAWN_LATENCY_SECS}s = '
            f'{spawn_allowance}s) — required = {required}s — but carries no '
            f'@pytest.mark.timeout override. Left uncovered, this can '
            f'silently collide with the 60s orchestrator/pyproject.toml '
            f'per-test default — under timeout_method="thread" with '
            f'--max-worker-restart=0, pytest-timeout os._exit()s the xdist '
            f"worker instead of failing cleanly, discarding _run_lane's own "
            f'well-located TimeoutError. Add @pytest.mark.timeout(N) with '
            f'N >= {required} ({worst_case}s bounded + {out_of_bound} spawns '
            f'x {_MEASURED_SPAWN_LATENCY_SECS}s).'
        )
        value = markers[0].args[0] if markers[0].args else markers[0].kwargs.get('timeout')
        assert value is not None, (
            f'{fn.__name__} carries @pytest.mark.timeout(...) but no timeout '
            f'value could be extracted from it (args={markers[0].args!r}, '
            f'kwargs={markers[0].kwargs!r}) — cannot verify it clears the '
            f'{required}s required timeout.'
        )
        assert value >= required, (
            f'{fn.__name__} carries @pytest.mark.timeout({value}) but the '
            f'required timeout is {required}s — {worst_case}s bounded-wait '
            f'sum + {out_of_bound} out-of-bound real-git spawns x '
            f'{_MEASURED_SPAWN_LATENCY_SECS}s = {spawn_allowance}s — the '
            f'override does not actually clear what it exists to cover. '
            f'Left uncovered, this can silently collide with the 60s '
            f'orchestrator/pyproject.toml per-test default — under '
            f'timeout_method="thread" with --max-worker-restart=0, '
            f'pytest-timeout os._exit()s the xdist worker instead of '
            f"failing cleanly, discarding _run_lane's own well-located "
            f'TimeoutError.'
        )


def test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling(pytestconfig) -> None:
    """`_LANE_PASS_BOUND_SECS` must clear a MEASURED floor and stay under a
    stated fraction of the effective per-test CEILING — the two-sided
    contract task 3836 landed for the sibling foreign-holder constants
    (``test_lane_lock_leak_guard.py``'s
    `test_foreign_holder_bounds_clear_measured_spawn_latency` +
    `test_foreign_holder_bounds_stay_clear_of_the_global_pytest_timeout`),
    adapted here for this module's own lane-pass bound.

    FLOOR — `_LANE_PASS_BOUND_SECS` must be at least
    `_SPAWNS_PER_LANE_PASS_WORST_CASE`x `_MEASURED_SPAWN_LATENCY_SECS` (task
    3451's measured worst-case happy-path subprocess spawn latency: n=3,
    2.13/3.10/4.71s, load-per-core 6.6). The multiplier is DERIVED, not
    guessed: it is the measured worst-case count of subprocess spawns that
    occur INSIDE a single `_run_lane` window, and that worst case lives in
    the SIBLING infra module (``test_offline_lane_infra_integration.py``'s
    `test_ib6_real_infra_runner_over_stub_classified_set_files_fix_task`):
    `_run_once`'s 3 git spawns (`get_main_sha`, `worktree list`, `worktree
    add`) plus the real infra seam spawning `run_all.sh` twice (the initial
    run and the isolated confirmation re-run). Later passes swap `worktree
    add` for `reset --hard` + `clean -xfd` — 4 spawns, still under 5. This
    module's own tests sit well under that shared worst case (the numeric
    `suite_runner` seam here is always a bare injected callable, never a
    real subprocess); the floor is still asserted here, AND additionally
    checked for equality against the infra module's own copy immediately
    below — the floor/ceiling relation alone permits any value in
    [5x measured spawn latency, 0.6x effective timeout] for EACH module
    independently, so only the explicit equality check actually prevents
    the two modules' `_LANE_PASS_BOUND_SECS` from silently diverging within
    that shared range.

    CEILING — the marker-less worst case (one `_LANE_PASS_BOUND_SECS`, since
    steps 2/4 of task 4030 covered every test that composes past a single
    pass — a premise `test_every_composing_caller_carries_a_timeout_override`
    makes executable rather than prose-only, named here the same way
    `test_no_foreign_holder_consumer_opts_out_of_the_global_timeout` backs
    its own sibling ceiling invariant) must stay at or under
    `_MARKERLESS_CEILING_FRACTION` (0.6, task 3836's own fraction) of the
    EFFECTIVE per-test timeout, resolved via `_effective_per_test_timeout`
    (imported function-locally from `test_lane_lock_leak_guard` — never
    duplicated or re-derived from a bare `tomllib` read; that module is task
    3836's and already correct, and importing it inherits its correctness).
    Why 60%, made concrete rather than left abstract: pytest-timeout 2.4.0
    installs its timer in `pytest_runtest_protocol` whenever `func_only` is
    False (the default; `timeout_func_only` is not set anywhere in this
    repo), so the 60s budget covers FIXTURE SETUP AND TEARDOWN, not just the
    call phase. Every test in both offline-lane modules transitively pulls
    the `repo` fixture, whose `_setup_repo` costs `_SPAWNS_PER_REPO_FIXTURE`
    git spawns (`init`, `config` x2, `add`, `commit`), and each
    `_drive_advance` costs `_SPAWNS_PER_DRIVE_ADVANCE` more (`get_main_sha`,
    `add`, `commit`, `rev-parse` — NOT `_note_merge_all`'s `git diff`, which
    this module's `harness` fixture stubs to an `AsyncMock` and so never
    reaches a subprocess; see `_SPAWNS_PER_DRIVE_ADVANCE`'s own comment and
    `test_out_of_bound_spawn_counts_are_measured_not_asserted`) — the 24s of
    headroom the 0.6 fraction leaves at a 60s timeout is what pays for that
    out-of-bound work.

    When no timeout is in effect at all, this either fails loudly (this
    module's own `orchestrator/pyproject.toml` is the governing inifile —
    genuine drift, the premise these bounds are sized against no longer
    holds) or skips, naming the governing inifile (an invocation artifact,
    e.g. a root-bound run).
    """
    from test_lane_lock_leak_guard import _effective_per_test_timeout  # noqa: PLC0415

    floor = _SPAWNS_PER_LANE_PASS_WORST_CASE * _MEASURED_SPAWN_LATENCY_SECS
    assert floor <= _LANE_PASS_BOUND_SECS, (
        f'_LANE_PASS_BOUND_SECS ({_LANE_PASS_BOUND_SECS}) must clear '
        f'{_SPAWNS_PER_LANE_PASS_WORST_CASE}x the measured worst-case happy-path '
        f'subprocess spawn latency ({_MEASURED_SPAWN_LATENCY_SECS}s, task 3451: n=3 '
        f'2.13/3.10/4.71, load-per-core 6.6) = {floor}s. The '
        f'{_SPAWNS_PER_LANE_PASS_WORST_CASE}-spawn multiplier is the measured worst-case '
        f'count of subprocess spawns inside a single _run_lane window (see '
        f'test_ib6_real_infra_runner_over_stub_classified_set_files_fix_task in '
        f'test_offline_lane_infra_integration.py: 3 git spawns from _run_once plus 2 '
        f'real infra run_all.sh spawns).'
    )

    from test_offline_lane_infra_integration import (  # noqa: PLC0415
        _LANE_PASS_BOUND_SECS as _infra_lane_pass_bound_secs,
    )
    from test_offline_lane_infra_integration import (  # noqa: PLC0415
        _SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE as _infra_spawns_per_assert_never_a_gate,
    )
    from test_offline_lane_infra_integration import (  # noqa: PLC0415
        _SPAWNS_PER_DRIVE_ADVANCE as _infra_spawns_per_drive_advance,
    )
    from test_offline_lane_infra_integration import (  # noqa: PLC0415
        _SPAWNS_PER_REPO_FIXTURE as _infra_spawns_per_repo_fixture,
    )

    assert _infra_lane_pass_bound_secs == _LANE_PASS_BOUND_SECS, (
        f"this module's _LANE_PASS_BOUND_SECS ({_LANE_PASS_BOUND_SECS}) has drifted "
        f"from test_offline_lane_infra_integration.py's copy "
        f'({_infra_lane_pass_bound_secs}) — each module independently permits any '
        f'value in [{floor}, {_MARKERLESS_CEILING_FRACTION} * effective_timeout], so '
        f'only this explicit cross-module equality check catches the two silently '
        f'diverging within that shared range.'
    )

    # Task 4203 — the same anti-drift rationale applies verbatim to the two
    # new _required_timeout_secs multiplicands: each module independently
    # measures its own copy (test_out_of_bound_spawn_counts_are_measured_not_asserted),
    # so only an explicit equality check catches the two silently diverging.
    assert _infra_spawns_per_repo_fixture == _SPAWNS_PER_REPO_FIXTURE, (
        f"this module's _SPAWNS_PER_REPO_FIXTURE ({_SPAWNS_PER_REPO_FIXTURE}) has "
        f"drifted from test_offline_lane_infra_integration.py's copy "
        f'({_infra_spawns_per_repo_fixture}) — both modules share byte-identical '
        f'_setup_repo bodies and harness fixture stubs, so a real divergence here '
        f"means one module's measurement (or fixture) has silently changed."
    )
    assert _infra_spawns_per_drive_advance == _SPAWNS_PER_DRIVE_ADVANCE, (
        f"this module's _SPAWNS_PER_DRIVE_ADVANCE ({_SPAWNS_PER_DRIVE_ADVANCE}) has "
        f"drifted from test_offline_lane_infra_integration.py's copy "
        f'({_infra_spawns_per_drive_advance}) — both modules share byte-identical '
        f'_drive_advance bodies and harness fixture stubs, so a real divergence here '
        f"means one module's measurement (or fixture) has silently changed."
    )
    # Reviewer amendment (task 4203) — same anti-drift rationale, for the third
    # multiplicand test_b3_never_a_gate / test_ib2_infra_run_in_flight_never_gates_merge
    # each need beyond their own _drive_advance call.
    assert _infra_spawns_per_assert_never_a_gate == _SPAWNS_PER_ASSERT_NEVER_A_GATE, (
        f"this module's _SPAWNS_PER_ASSERT_NEVER_A_GATE ({_SPAWNS_PER_ASSERT_NEVER_A_GATE}) "
        f"has drifted from test_offline_lane_infra_integration.py's copy, "
        f'_SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE ({_infra_spawns_per_assert_never_a_gate}) — '
        f'both modules share byte-identical _assert_never_a_gate / _assert_infra_never_a_gate '
        f"bodies and harness fixture stubs, so a real divergence here means one module's "
        f'measurement (or fixture) has silently changed.'
    )

    global_timeout = _effective_per_test_timeout(pytestconfig)
    if global_timeout is None:
        orchestrator_pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
        governing_inifile = pytestconfig.inipath
        if governing_inifile is not None and governing_inifile.resolve() == orchestrator_pyproject:
            pytest.fail(
                f'{governing_inifile} is the governing inifile for this run, yet no '
                f'per-test timeout is in effect (checked --timeout, PYTEST_TIMEOUT, and '
                f'[tool.pytest.ini_options].timeout) — this is a genuine regression: the '
                f'pytest-timeout / os._exit() premise _LANE_PASS_BOUND_SECS is sized '
                f'against no longer holds. Either restore the timeout key or re-derive '
                f'this bound without it.'
            )
        pytest.skip(
            f'no per-test timeout is in effect under the governing inifile '
            f'{governing_inifile!r} — this is not {orchestrator_pyproject}, so this is '
            f'an invocation artifact (e.g. a root-bound run), not drift; the '
            f'pytest-timeout ceiling this invariant checks does not apply to this run'
        )

    ceiling = _MARKERLESS_CEILING_FRACTION * global_timeout
    assert ceiling >= _LANE_PASS_BOUND_SECS, (
        f'_LANE_PASS_BOUND_SECS ({_LANE_PASS_BOUND_SECS}s) exceeds '
        f'{_MARKERLESS_CEILING_FRACTION * 100:.0f}% of the effective per-test timeout '
        f'({global_timeout}s, resolved from {pytestconfig.inipath}) = {ceiling}s. Every '
        f'test composing past a single _run_lane pass carries its own '
        f'@pytest.mark.timeout override (enforced by '
        f'test_every_composing_caller_carries_a_timeout_override), so this bounds ONLY '
        f"the marker-less budget — a single _run_lane pass. The remaining "
        f'{(1 - _MARKERLESS_CEILING_FRACTION) * 100:.0f}% is headroom for the real-git '
        f"fixture work (the repo fixture's _setup_repo, and each _drive_advance) that "
        f'shares the same per-test budget, since pytest-timeout installs its timer in '
        f'pytest_runtest_protocol (func_only=False, unset in this repo) and so times '
        f'fixture setup/teardown too, not just the call phase.'
    )

    # Task 4203 — reconcile the retained 0.6 fraction (marker-less budget)
    # against the additive `_required_timeout_secs` model this task adopts
    # for @pytest.mark.timeout OVERRIDES. The two models are not applied to
    # the same population: an override is a number this suite CHOOSES, so
    # `_required_timeout_secs` prices it at every out-of-bound spawn's
    # worst-case latency; the marker-less budget is bounded by the FIXED 60s
    # pyproject global, which this suite cannot move, so it keeps the 0.6
    # fraction as a cheaper proxy instead. Gating the marker-less population
    # additively would be unsatisfiable: b1/b4/b6 (this module) and
    # ib1/ib3/ib5 (the sibling infra module) each do the repo fixture plus at
    # least one _drive_advance = _SPAWNS_PER_REPO_FIXTURE +
    # _SPAWNS_PER_DRIVE_ADVANCE = 9 spawns = 42.39s of out-of-bound work
    # against the 24s the 0.6 fraction leaves at a 60s timeout — precisely
    # task 4030's "10 of the 15 tests exceed the 60s global" observation.
    # Pricing every spawn at the worst-case SINGLE-spawn 4.71s is a sound
    # upper bound for sizing a number this suite chooses (an override) and an
    # unsound hard gate for one it cannot move (the 60s global) — no observed
    # run actually pays that price on every spawn.
    #
    # What IS pinnable, and is asserted here executably rather than left as
    # prose: the 40% the 0.6 fraction holds back at the effective timeout
    # must cover at least the repo fixture's own out-of-bound spawns (every
    # test in both modules transitively pays this cost, marker-less or not),
    # priced by the SAME additive model. This is the fixture-only baseline
    # both models agree on.
    fixture_only_required = _required_timeout_secs(0.0, _SPAWNS_PER_REPO_FIXTURE)
    marker_less_headroom = (1 - _MARKERLESS_CEILING_FRACTION) * global_timeout
    assert marker_less_headroom >= fixture_only_required, (
        f'the {(1 - _MARKERLESS_CEILING_FRACTION) * 100:.0f}% of the effective per-test '
        f'timeout ({global_timeout}s) the {_MARKERLESS_CEILING_FRACTION} fraction holds '
        f'back as marker-less headroom = {marker_less_headroom}s, but the repo fixture '
        f'alone already requires {fixture_only_required}s '
        f'(_SPAWNS_PER_REPO_FIXTURE={_SPAWNS_PER_REPO_FIXTURE} spawns x '
        f'{_MEASURED_SPAWN_LATENCY_SECS}s) under the additive model this task adopts for '
        f'@pytest.mark.timeout overrides. The 0.6 fraction is deliberately kept (not '
        f'replaced by the additive model) for the marker-less population, since a '
        f"marker-less test's timeout is the fixed 60s global and cannot be widened — but "
        f'that choice is only sound while this fixture-only floor holds; a failure here '
        f'means _MEASURED_SPAWN_LATENCY_SECS has been re-measured upward enough that even '
        f'the cheapest marker-less test (fixture only, zero _drive_advance calls) no '
        f'longer fits, and the 0.6 fraction itself needs re-deriving.'
    )


@pytest.mark.asyncio
async def test_out_of_bound_spawn_counts_are_measured_not_asserted(
    harness: Harness, git_ops: GitOps, repo: Path, tmp_path: Path,
) -> None:
    """`_SPAWNS_PER_DRIVE_ADVANCE` / `_SPAWNS_PER_REPO_FIXTURE` are the
    multiplicands `_required_timeout_secs` (task 4203) prices at
    `_MEASURED_SPAWN_LATENCY_SECS` per spawn — this test MEASURES them by
    counting real git subprocess spawns, rather than trusting a
    hand-maintained comment. That distinction is not academic: the two
    landed comments this task corrects (`test_lane_bounds_...`'s and
    `_run_lane`'s) had already drifted from each other AND from the truth,
    which is exactly the failure mode a measuring test — instead of a third
    hand count — closes off.

    REVIEWER AMENDMENT (task 4203) also measures `_SPAWNS_PER_ASSERT_NEVER_A_GATE`
    — the third multiplicand `test_b3_never_a_gate`'s `out_of_bound_spawns` row
    needs beyond its own single `_drive_advance` call. An earlier hand
    approximation treated the whole of `_assert_never_a_gate` as "one
    `_drive_advance`-equivalent round" (4 spawns); measured under the SAME
    counting wrapper, it is actually 7.

    Deliberately calls `_assert_never_a_gate` WITHOUT first driving it through
    `test_b3_never_a_gate`'s full held-in-flight `_run_lane` scaffolding:
    `_assert_never_a_gate` accepts a `runner` argument but never reads it in
    its own body (the parameter only documents an already-satisfied
    precondition of its real caller) — its git-spawn count is therefore
    unaffected by whether a pass is actually held in flight. Composing an
    actual `_run_lane` pass here would also wrongly register THIS test in
    `_lane_pass_composing_callers`' AST net (it scans for literal calls to
    `_run_lane` / `_run_one_lane_pass` / `_drive_reds` by name), forcing it
    into `worst_case_secs` and a `@pytest.mark.timeout` override it does not
    need — this test's own real-git footprint stays the marker-less shape
    the docstring below describes.

    SEAM VERIFICATION (must hold before a single monkeypatch target can be
    trusted to observe every spawn): this module does ``from
    orchestrator.git_ops import GitOps, _run`` at the top of the file. That
    `import` statement copies a REFERENCE to the `_run` function object into
    this module's own globals — a name distinct from ``orchestrator.git_ops
    ._run``, the attribute a `unittest.mock.patch` target string resolves
    against. `_setup_repo` / `_advance_main` (defined in THIS module) call
    the bare name `_run`, which Python resolves through THIS module's own
    globals at call time — so patching only ``'orchestrator.git_ops._run'``
    silently misses them (confirmed directly: doing so undercounts
    `_drive_advance` at 1 spawn instead of 4). `GitOps.get_main_sha`, by
    contrast, calls `_run` via ``orchestrator.git_ops``'s OWN globals, so it
    is only ever observed by patching ``'orchestrator.git_ops._run'``. Both
    targets are patched below with the same counting wrapper so neither leg
    is silently missed.

    THIRD LEG, MEASURED RATHER THAN ASSUMED: in PRODUCTION, `_note_merge_all`
    (`harness.py:10377`) unconditionally calls
    ``self.git_ops.get_merge_diff_files(...)`` — a real ``git diff`` spawn.
    But the `harness` fixture above replaces ``h.git_ops.get_merge_diff_files``
    wholesale with an ``AsyncMock(return_value=([], None))``, so under THIS
    integration harness that leg reaches neither `_run` nor any subprocess at
    all — it is a pure in-memory stub. (`_maybe_pipeline_landing_tripwire`,
    also reached from `_note_merge_all` after the diff fetch, is confirmed
    separately to no-op whenever ``config.git.load_bearing_oracle_cmd`` is
    falsy — `harness.py:10438-10444` — and the `git_config` fixture never
    sets it.) So `_drive_advance`'s measured spawn count is 4 — `get_main_sha`
    plus `_advance_main`'s `add`/`commit`/`rev-parse` — not 5: not because
    `get_main_sha` is uncounted (it IS, unlike the superseded `_run_lane`
    docstring's "3+"), but because the diff fetch contributes zero spawns
    under this harness's stub, unlike the superseded `test_lane_bounds_...`
    docstring's assumption that it is real. If a future change to the
    `harness` fixture ever makes `get_merge_diff_files` real again, THIS
    assertion fails with the call list attached, rather than the sizing
    model silently mispricing every composing test that calls
    `_drive_advance` (directly or via `_drive_reds`).

    No `@pytest.mark.timeout` override: this test has zero bounded waits,
    and its real-git footprint (one `_drive_advance` plus one fresh
    `_setup_repo`, on top of the `repo` fixture's own `_setup_repo`) is the
    same real-git shape — fixture plus a handful of git spawns — the
    marker-less B1/B4/B6 tests already carry unmarked; see this module's
    `_MARKERLESS_CEILING_FRACTION` comment and
    `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling`'s own
    docstring for why that shape is accepted for the marker-less population
    as a whole rather than gated per-test.
    """
    import orchestrator.git_ops as git_ops_mod

    calls: list[list[str]] = []
    orig_run = git_ops_mod._run

    async def _counting_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get('cmd')
        assert cmd is not None, '_run must be called with a cmd list, positional or keyword'
        calls.append(list(cmd))
        return await orig_run(*args, **kwargs)

    with patch('orchestrator.git_ops._run', side_effect=_counting_run), \
         patch(f'{__name__}._run', side_effect=_counting_run):
        await _drive_advance(harness, repo)

    assert len(calls) == _SPAWNS_PER_DRIVE_ADVANCE, (
        f'_drive_advance spawned {len(calls)} real git subprocess(es) '
        f'({calls}), not the pinned _SPAWNS_PER_DRIVE_ADVANCE = '
        f'{_SPAWNS_PER_DRIVE_ADVANCE}. This count is a multiplicand of '
        f'_required_timeout_secs (the @pytest.mark.timeout override sizing '
        f'model) — a drift here silently mis-prices every composing test '
        f'that calls _drive_advance, directly or via _drive_reds.'
    )

    calls.clear()
    fresh_repo = tmp_path / 'fresh_repo_for_spawn_count'
    fresh_repo.mkdir()
    with patch('orchestrator.git_ops._run', side_effect=_counting_run), \
         patch(f'{__name__}._run', side_effect=_counting_run):
        await _setup_repo(fresh_repo)

    assert len(calls) == _SPAWNS_PER_REPO_FIXTURE, (
        f'_setup_repo spawned {len(calls)} real git subprocess(es) '
        f'({calls}), not the pinned _SPAWNS_PER_REPO_FIXTURE = '
        f'{_SPAWNS_PER_REPO_FIXTURE}. This count is a multiplicand of '
        f'_required_timeout_secs — every test in this module transitively '
        f'pulls the repo fixture, so a drift here mis-prices every '
        f'composing test, not just one.'
    )

    # Reviewer amendment (task 4203) — measure `_assert_never_a_gate` itself.
    # `runner` is required by its signature but never read in its body (see
    # this test's docstring), so a bare, never-driven instance is enough —
    # no held-in-flight `_run_lane` pass is composed here.
    #
    # The lane wiring itself is MEASURED to be spawn-free rather than assumed
    # (task 4203 review remediation): this test's own `out_of_bound_spawns`
    # row prices exactly 2x`_SPAWNS_PER_REPO_FIXTURE` +
    # `_SPAWNS_PER_DRIVE_ADVANCE` + `_SPAWNS_PER_ASSERT_NEVER_A_GATE`, so a
    # constructor or wiring call that quietly spawned git would inflate this
    # test's real footprint past the timeout its own guard row sizes for it.
    calls.clear()
    with patch('orchestrator.git_ops._run', side_effect=_counting_run), \
         patch(f'{__name__}._run', side_effect=_counting_run):
        runner = _ControllableSuiteRunner()
        worker = _build_worker(git_ops, tmp_path, suite_runner=runner)
        _wire_lane(harness, worker)

    assert calls == [], (
        f'constructing _ControllableSuiteRunner / _build_worker / _wire_lane '
        f'spawned {len(calls)} real git subprocess(es) ({calls}), but this '
        f"test's out_of_bound_spawns row in "
        f'test_every_composing_caller_carries_a_timeout_override prices the '
        f'lane wiring at zero. Pin the measured count in a named constant '
        f'and add it to that row rather than letting it be absorbed '
        f'silently — an unpriced spawn under-sizes this test\'s own '
        f'@pytest.mark.timeout override.'
    )

    with patch('orchestrator.git_ops._run', side_effect=_counting_run), \
         patch(f'{__name__}._run', side_effect=_counting_run):
        await _assert_never_a_gate(harness, worker, runner, repo, git_ops)

    assert len(calls) == _SPAWNS_PER_ASSERT_NEVER_A_GATE, (
        f'_assert_never_a_gate spawned {len(calls)} real git subprocess(es) '
        f'({calls}), not the pinned _SPAWNS_PER_ASSERT_NEVER_A_GATE = '
        f'{_SPAWNS_PER_ASSERT_NEVER_A_GATE}. This count is the second '
        f'out_of_bound_spawns multiplicand test_b3_never_a_gate needs beyond '
        f'its own _drive_advance call — a drift here mis-prices that test '
        f"(and the infra module's mirror, "
        f'test_ib2_infra_run_in_flight_never_gates_merge).'
    )


# ---------------------------------------------------------------------------
# B1 — advance triggers a from-head run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b1_advance_triggers_from_head_run(harness, git_ops, repo, tmp_path, caplog):
    """B1 (PRD §8) — a landed advance triggers a lane run snapshotted from the
    CURRENT main head, never the advisory trigger SHA passed to on_post_merge."""
    calls: list[tuple] = []

    async def _suite_runner(wt, head, threads):
        calls.append((wt, head, threads))
        return (0, '')

    worker = _build_worker(git_ops, tmp_path, suite_runner=_suite_runner)
    _wire_lane(harness, worker)

    with caplog.at_level(logging.INFO):
        base_sha, head_sha = await _drive_advance(harness, repo)
        await _run_one_lane_pass(worker)

    real_head = await git_ops.get_main_sha()
    assert real_head == head_sha
    assert len(calls) == 1
    assert calls[0][1] == real_head, (
        'suite_runner must be invoked with the real from-head snapshot'
    )

    assert 'offline-lane: on_post_merge' in caplog.text
    assert base_sha[:12] in caplog.text
    assert head_sha[:12] in caplog.text


# ---------------------------------------------------------------------------
# B2 — coalescing under a burst of advances
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b2_coalesces_burst_of_advances_to_one_rerun(harness, git_ops, repo, tmp_path):
    """B2 (PRD §8) — advances landing while a run is in-flight coalesce into
    exactly ONE re-run at the newest head, never a queue of stale re-runs."""
    runner = _ControllableSuiteRunner()
    worker = _build_worker(git_ops, tmp_path, suite_runner=runner)
    _wire_lane(harness, worker)

    _, first_head = await _drive_advance(harness, repo)

    lane_task = asyncio.create_task(_run_lane(worker, expected_passes=2))
    await runner.wait_entered()

    newest_head = first_head
    for _ in range(3):
        _, newest_head = await _drive_advance(harness, repo)

    runner.release()
    await lane_task

    assert runner.heads == [first_head, newest_head], (
        'a burst of advances during an in-flight run must coalesce into '
        'exactly one re-run at the newest head, never a queue of stale runs'
    )


# ---------------------------------------------------------------------------
# B3 — never-a-gate (C7): prompt return, fail-open, no halt/gate
# ---------------------------------------------------------------------------


@pytest.mark.timeout(150)  # task 4030: NOT a _drive_reds chainer, which is why the task-3832
# review's marker sweep missed it -- but it composes anyway: a 30s _run_lane bound raced
# concurrently by wait_entered (max, not sum), then _assert_never_a_gate's 0.5 + 15.0
# SEQUENTIALLY after it = 45.5s bounded, on top of unbounded real-git spawns: repo init
# (_SPAWNS_PER_REPO_FIXTURE=5), one _drive_advance (4), and _assert_never_a_gate's own
# get_main_sha + two _advance_main rounds (_SPAWNS_PER_ASSERT_NEVER_A_GATE=7, task 4203
# reviewer amendment -- MEASURED, supersedes an earlier "two _drive_advance/_advance_main
# rounds" approximation) = 16 spawns x 4.71s = 75.36s, required = 120.86s, comfortably under
# this marker. Clear the 60s pyproject default with margin so a genuinely wedged pass fails
# via _run_lane's own TimeoutError, not pytest-timeout's blunter worker kill.
@pytest.mark.asyncio
async def test_b3_never_a_gate(harness, git_ops, repo, tmp_path):
    """B3 (PRD §8, C7) — a merge-landed notification while the lane is
    in-flight must never gate the merge: it returns promptly (never blocks
    the synchronous notifiee call on the in-flight run), sets _dirty for a
    coalesced re-run, is fail-open when the notifiee raises, and never
    files an escalation (the only halt/gate-adjacent side-effect reachable
    from this call path)."""
    runner = _ControllableSuiteRunner()
    worker = _build_worker(git_ops, tmp_path, suite_runner=runner)
    _wire_lane(harness, worker)

    await _drive_advance(harness, repo)
    lane_task = asyncio.create_task(_run_lane(worker, expected_passes=1))
    await runner.wait_entered()

    await _assert_never_a_gate(harness, worker, runner, repo, git_ops)

    runner.release()
    await lane_task


# ---------------------------------------------------------------------------
# B4 — confirmed red files a normal fix task + L0 info escalation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b4_confirmed_red_files_fix_task_and_info_escalation(harness, git_ops, repo, tmp_path):
    """B4 (PRD §8) — a confirmed-red run files a NORMAL pending fix task
    (never the b3_gate red-main fix-forward path) plus a non-blocking L0
    info escalation, leaving the merge queue itself completely untouched."""
    from orchestrator.workflow import compute_failing_test_set_fingerprint

    failing_ids = ['t::b', 't::a']
    worker = _build_worker(git_ops, tmp_path)
    _inject_red(worker, failing_ids)
    _wire_lane(harness, worker)

    _, head_sha = await _drive_advance(harness, repo)
    await _run_one_lane_pass(worker)

    arguments = _submitted_fix_task_arguments(worker.task_client)
    assert arguments['status'] == 'pending'
    assert arguments['metadata']['merge_lane'] == 'normal', (
        'a confirmed-red fix task must route through the standard '
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

    # The merge queue itself is left completely untouched by the red path.
    assert harness.get_merge_halt_status() == {'wired': False}


# ---------------------------------------------------------------------------
# B5 — dedup: a same-set recurrence updates rather than duplicates
# ---------------------------------------------------------------------------


@pytest.mark.timeout(150)  # task 3832 review: _drive_reds(n=2) chains 2 30s-bounded
# _run_one_lane_pass calls (60s alone) plus real-git _drive_advance overhead --
# clear the 60s pyproject default with margin so a genuinely wedged pass fails
# via _run_lane's own TimeoutError, not pytest-timeout's blunter worker kill.
@pytest.mark.asyncio
async def test_b5_same_set_recurrence_updates_not_duplicates(harness, git_ops, repo, tmp_path):
    """B5 (PRD §8) — a SECOND red advance with the SAME failing-test set
    updates the existing fix task (appends a suspect range) rather than
    filing a duplicate task or raising a second escalation."""
    from orchestrator.workflow import compute_failing_test_set_fingerprint

    failing_ids = ['t::a', 't::b']
    worker = _build_worker(git_ops, tmp_path)
    _wire_lane(harness, worker)

    await _drive_reds(harness, repo, worker, failing_ids, 2)

    cast(AsyncMock, worker.task_client).submit_fix_task.assert_awaited_once()
    cast(AsyncMock, worker.task_client).append_suspect_range.assert_awaited_once()

    fp = compute_failing_test_set_fingerprint(failing_ids)
    task_id = worker.open_fix_tasks[fp]
    escalations = _l0_info_escalations(cast(EscalationQueue, worker.escalation_queue), task_id)
    assert len(escalations) == 1, (
        'a same-set recurrence must not raise a second L0 escalation'
    )


# ---------------------------------------------------------------------------
# B6 — flake filtered: fail-then-pass on confirmation is not a red
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b6_flake_filtered_by_confirmation_rerun(harness, git_ops, repo, tmp_path):
    """B6 (PRD §8) — a run that fails, then passes on the isolated
    confirmation re-run, is intermittent nondeterminism, never a genuine
    break: no fix task and no escalation are ever raised."""
    worker = _build_worker(git_ops, tmp_path)
    _wire_lane(harness, worker)
    _inject_flake(worker)

    await _drive_advance(harness, repo)
    await _run_one_lane_pass(worker)

    # Intermittent-nondeterminism handling is proven behaviorally below
    # (no fix task, no open fingerprint, no escalation) rather than by log
    # text.
    cast(AsyncMock, worker.task_client).submit_fix_task.assert_not_awaited()
    assert worker.open_fix_tasks == {}
    assert cast(EscalationQueue, worker.escalation_queue).get_pending() == []


# ---------------------------------------------------------------------------
# B7 — stall promotes to a born-at-L2 escalate_blocker
# ---------------------------------------------------------------------------


@pytest.mark.timeout(360)  # task 4203: widened from 300 under the now-adopted
# _required_timeout_secs model (see its docstring for the full derivation).
# Bounded term (task 3832 review, still correct): 4 _drive_reds calls, n =
# 2+1+2+1 = 6 _run_one_lane_pass calls at _LANE_PASS_BOUND_SECS = 180s.
# Out-of-bound term (previously only gestured at as "real-git overhead", now
# counted): _SPAWNS_PER_REPO_FIXTURE + 6 * _SPAWNS_PER_DRIVE_ADVANCE = 29
# spawns x _MEASURED_SPAWN_LATENCY_SECS = 136.59s -- the most out-of-bound
# real-git work of any test in either module, which is why this is the one
# marker the model moves (b3/b5/ib2/ib4 all clear it unchanged). required =
# 316.59s; 360 is the next multiple of the 60s pyproject grid at or above
# that. Erring high is deliberate: the marker is a backstop that must never
# fire before _run_lane's own 30s TimeoutError, since under
# timeout_method="thread" with --max-worker-restart=0 pytest-timeout
# os._exit()s the xdist worker instead of failing cleanly.
@pytest.mark.asyncio
async def test_b7_stall_promotes_to_blocker(harness, git_ops, repo, tmp_path):
    """B7 (PRD §8, C4) — a fix task that stalls promotes the L0 info signal
    to a real, born-at-L2 escalate_blocker via either arm: the confirmed-red
    advance count reaching the configured threshold (arm 1), or the fix task
    itself landing in a terminal non-done status (arm 2, cancelled/deferred).
    Promotion is idempotent per fingerprint and never touches the merge
    queue."""
    from orchestrator.workflow import compute_failing_test_set_fingerprint

    # Arm 1 — N-advances: the confirmed-red count reaches the (small,
    # explicitly configured) threshold.
    failing_a = ['t::stall_a']
    worker_a = _build_worker(
        git_ops, tmp_path / 'arm1',
        git_overrides={'offline_lane_red_advances_before_blocker': 2},
    )
    _wire_lane(harness, worker_a)
    await _drive_reds(harness, repo, worker_a, failing_a, 2)

    fp_a = compute_failing_test_set_fingerprint(failing_a)
    task_a = worker_a.open_fix_tasks[fp_a]
    blockers_a = _l2_blocker_escalations(cast(EscalationQueue, worker_a.escalation_queue), task_a)
    assert len(blockers_a) == 1
    assert blockers_a[0].severity == 'critical'
    assert blockers_a[0].agent_role == 'orchestrator-offline-lane'

    # A further same-set red must not re-promote (idempotent per fingerprint).
    await _drive_reds(harness, repo, worker_a, failing_a, 1)
    assert len(_l2_blocker_escalations(cast(EscalationQueue, worker_a.escalation_queue), task_a)) == 1
    assert harness.get_merge_halt_status() == {'wired': False}

    # Arm 2 — terminal non-done status: promotes immediately regardless of
    # the count (threshold set deliberately high so arm 1 cannot explain it).
    failing_b = ['t::stall_b']
    worker_b = _build_worker(
        git_ops, tmp_path / 'arm2',
        task_client=_terminal_status_task_client('cancelled'),
        git_overrides={'offline_lane_red_advances_before_blocker': 100},
    )
    _wire_lane(harness, worker_b)
    await _drive_reds(harness, repo, worker_b, failing_b, 2)

    fp_b = compute_failing_test_set_fingerprint(failing_b)
    task_b = worker_b.open_fix_tasks[fp_b]
    blockers_b = _l2_blocker_escalations(cast(EscalationQueue, worker_b.escalation_queue), task_b)
    assert len(blockers_b) == 1, (
        'a terminal non-done fix-task status must promote immediately, '
        'regardless of the configured advance-count threshold'
    )

    await _drive_reds(harness, repo, worker_b, failing_b, 1)
    assert len(_l2_blocker_escalations(cast(EscalationQueue, worker_b.escalation_queue), task_b)) == 1
    assert harness.get_merge_halt_status() == {'wired': False}


# ---------------------------------------------------------------------------
# B8 — target/worktree ISOLATION (offline-deep path != merge-verify path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b8_offline_deep_worktree_isolated_from_merge_lane(git_ops, repo):
    """B8 (PRD §8) — the offline-deep lane's persistent worktree is isolated
    from the merge lane's: distinct fixed paths; a real reset materializes
    and registers ONLY the offline-deep path (the merge lane's _merge-verify
    path is never created or touched by the lane reset); and
    cleanup_merge_worktree is prune-exempt (a no-op) on the offline-deep
    path. Real reify compile timing is NOT asserted here — self-warming
    timing is a reify-build capability (reify:4916/4913) validated at
    production activation, out of ζ's executable scope."""
    assert (
        git_ops.persistent_offline_deep_worktree_path
        != git_ops.persistent_merge_worktree_path
    )

    before = _materialized_worktree_names(git_ops.worktree_base)
    head = await git_ops.get_main_sha()
    offline_wt = await git_ops.reset_persistent_offline_deep_worktree(head)
    after = _materialized_worktree_names(git_ops.worktree_base)

    assert offline_wt == git_ops.persistent_offline_deep_worktree_path
    assert after - before == {git_ops.persistent_offline_deep_worktree_path.name}, (
        'a lane reset must materialize ONLY its own _offline-deep worktree, '
        'never the merge lane _merge-verify path'
    )
    assert not git_ops.persistent_merge_worktree_path.exists(), (
        'the merge lane worktree must never be created by an offline-deep reset'
    )
    assert await git_ops._is_registered_worktree(offline_wt)

    await git_ops.cleanup_merge_worktree(offline_wt)
    assert offline_wt.exists(), (
        'cleanup_merge_worktree must be prune-exempt (a no-op) on the '
        'persistent offline-deep worktree'
    )
    assert await git_ops._is_registered_worktree(offline_wt), (
        'the offline-deep worktree must remain a registered git worktree '
        'after a cleanup_merge_worktree no-op'
    )


# ---------------------------------------------------------------------------
# Incident reproduction (task 5308) — a verify.sh usage dump files NOTHING
# ---------------------------------------------------------------------------


# A faithful reproduction of reify scripts/verify.sh's error+usage() dump
# (reify:5308 incident; twin of the fixture in test_offline_lane.py): an
# em-dash ``verify.sh: ERROR — <msg>`` banner followed by the full usage()
# block (bare ``Usage:``/``Options:`` section headers + invocation/option
# lines). run-offline-deep.sh merges stderr into stdout, so this whole blob
# reaches the numeric confirmation seam's parser.
_VERIFY_USAGE_DUMP = (
    "verify.sh: ERROR — unknown argument '--test-threads=1'\n"
    "\n"
    "scripts/verify.sh — unified verification entrypoint for Reify.\n"
    "\n"
    "Usage:\n"
    "  verify.sh <test|lint|typecheck|all> [options]\n"
    "\n"
    "Options:\n"
    "  --scope <all|rust|gui|infra>  Restrict verification to a subsystem.\n"
    "  --profile <fast|full|both>    Select the verification profile.\n"
    "  -h|--help  Show usage.\n"
)


@pytest.mark.asyncio
async def test_verify_usage_dump_confirmation_files_no_task_or_escalation(git_ops, tmp_path):
    """Incident reproduction (task 5308; reify:5264) — when the DEFAULT numeric
    confirmation seam ingests a reify verify.sh usage/help/error dump (stderr
    merged into stdout), the full red path files ZERO fix tasks and ZERO
    escalations, never one fake fix task built from usage lines.

    Unlike B4 (which injects a fake confirmation_runner), this leaves the
    numeric confirmation seam at its DEFAULT so the REAL
    ``_default_confirmation_run`` parser runs end-to-end — the reify-subprocess
    boundary is faked one level deeper, at ``create_subprocess_exec``, which is
    patched to emit the usage dump with returncode 64. Everything downstream of
    the parse (the empty-``confirmed`` guard, fingerprinting, fix-task filing,
    escalation staging) is the real ``_handle_red_run`` chain, driven directly
    (the confirmed-list → ``_handle_red_run`` corruption flow's entry point).

    RED pre-fix: the seam's inline comprehension turns every non-blank usage
    line into a confirmed test ID, filing a fix task + an L0 info escalation.
    """
    worker = _build_worker(git_ops, tmp_path)  # DEFAULT confirmation seam, real queue + spy client
    head = await git_ops.get_main_sha()
    wt_path = git_ops.persistent_offline_deep_worktree_path

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(_VERIFY_USAGE_DUMP.encode(), b''))
    mock_proc.returncode = 64

    with patch(
        'orchestrator.offline_lane.asyncio.create_subprocess_exec',
        return_value=mock_proc,
    ):
        await worker._handle_red_run(wt_path, head)

    # Zero fix tasks filed, zero fingerprints opened.
    cast(AsyncMock, worker.task_client).submit_fix_task.assert_not_awaited()
    assert worker.open_fix_tasks == {}
    # Zero escalations of ANY kind reached the real queue (no
    # orchestrator-offline-lane info escalation either).
    assert cast(EscalationQueue, worker.escalation_queue).get_pending() == []
