"""Integration gate for the offline deep-test lane (task 1955, ζ; red path, task 6861).

Stands up the REAL trigger→worker→warm-worktree→red-path chain end-to-end and
runs PRD §8 boundary scenarios B1-B8 for ``offline-deep-test-lane-worker.md``.
Since task 6861, B4-B7 and the usage-dump row assert the red path of
``plans/offline-lane-red-path-prd.md`` (its §6 scenarios in brackets below)
through the same chain:

* β1 (task 1951) — ``Harness._offline_lane_notifiee`` / ``_note_merge_all`` /
  ``_note_offline_lane`` fan-out (``harness.py``).
* δ  (task 1952) — the persistent ``_offline-deep`` warm worktree
  (``git_ops.py``: ``persistent_offline_deep_worktree_path``,
  ``reset_persistent_offline_deep_worktree``, the ``cleanup_merge_worktree``
  prune exemption).
* β2 (task 1953) — ``OfflineLaneWorker`` (``offline_lane.py``): the
  coalescing ``run()`` loop, the always-from-head ``_run_once`` snapshot,
  and the injectable detect seams.
* δ2 (task 6861) — each pass is ``offline_lane_pass.py::run_pass`` (detect,
  confirm, the engine), and ``offline_lane_episodes.py::LaneEngine`` decides
  what a red pass holds, files or escalates.

STRATEGY — B1-B3 inject the numeric ``suite_runner`` seam.  The red-path rows
inject nothing at the run boundary: each commits a fixture command to main (a
real pytest project whose ``test_target`` reads its verdict from a control
directory, or a stub ``scripts/run-offline-deep.sh``) and the worker's default
seams run it in the real ``_offline-deep`` worktree.  What a red-path row fakes
is the task store (``_lane_episode_fakes.py::FakeLedgerClient``, the seam
``chronic_flake.py::SchedulerChronicFlakeTaskClient`` fills in production) and
the clock (the worker's ``now``, so passes are spaced without sleeping); the
escalation queue is real.

Scenario map (PRD §8; red-path §6 scenarios in brackets):

  B1 - advance triggers a from-head run
  B2 - coalescing under a burst of advances
  B3 - never-a-gate (C7): prompt return, fail-open, no halt/gate
  B4 - a confirmed red is held at one counted pass; a reproduction 30 minutes
       later files exactly one task [3, 4]
  B5 - a persistent red under an open task files no second task and raises
       one escalation at 24 h; a covered test green then red again files
       nothing [13, 14]
  B6 - a red the narrowed confirm clears ends its episode, files nothing [6]
  B7 - a cancelled covering task: one escalation at 24 h; once it is resolved
       the next red is held and a second spaced red files a task naming the
       prior [13, 15]
  B8 - target/worktree ISOLATION (offline-deep path != merge-verify path)
  usage dump - a numeric confirm that prints verify.sh's usage is
       unconfirmable on two spaced passes: one command-level task, naming no
       test [8]

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
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _lane_episode_fakes import Clock, FakeLedgerClient
from _orch_helpers import (
    MEASURED_SPAWN_LATENCY_SECS,
    VERIFY_CLI_PER_TEST_TIMEOUT,
    pydantic_spec,
    required_timeout_secs,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, LaneCommand, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.offline_lane import OfflineLaneWorker
from orchestrator.offline_lane_episode_state import (
    CommandRecord,
    Episode,
    EpisodeStore,
    unconfirmable_node,
)
from orchestrator.offline_lane_escalations import UNLANDED_RED_CATEGORY, sentinel_task_id
from orchestrator.offline_lane_records import LEGACY_NUMERIC_COMMAND
from orchestrator.offline_lane_runs import RUN_OFFLINE_DEEP_SCRIPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout bounds (task 4030) — named constants for the bounded waits below,
# replacing six near-identical inline 30.0 literals (the duplication that let
# a single false citation drift into three near-verbatim docstring copies).
# Modeled on the repo's own constant-plus-derivation-comment convention:
# _orch_helpers.py's CANCEL_SCOPE_BARRIER_TIMEOUT / CANCEL_SCOPE_PURE_UNIT_TIMEOUT
# and test_lane_lock_leak_guard.py's _FOREIGN_HOLDER_* block. The full floor
# (task 3451's measured spawn latency) / ceiling (the effective per-test pytest-timeout)
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
# `test_ib3_a_confirmed_infra_red_is_held_then_a_spaced_reproduction_files_one_task`: 3 git
# spawns from `_run_once` plus 2 real infra `run_all.sh` spawns. See
# `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling` for the
# full derivation.
_SPAWNS_PER_LANE_PASS_WORST_CASE = 5

# Task 4203 — the multiplicands `_orch_helpers.py::required_timeout_secs` prices at
# `MEASURED_SPAWN_LATENCY_SECS` per spawn. MEASURED by
# `test_out_of_bound_spawn_counts_are_measured_not_asserted`, not
# hand-counted: the `repo` fixture's `_setup_repo` spawns `init`, `config`
# x2, `add`, `commit`.
_SPAWNS_PER_REPO_FIXTURE = 5

# Task 4203 — the other multiplicand of `required_timeout_secs`. MEASURED,
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

# Task 6861 — the fourth multiplicand: a red-path row commits its fixture
# command to main before its first pass. MEASURED, not hand-counted (see
# `test_out_of_bound_spawn_counts_are_measured_not_asserted`): `_commit_files`
# spawns one `add` and one `commit`.
_SPAWNS_PER_FIXTURE_COMMIT = 2


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
    task_client=None,
    escalation_queue: EscalationQueue | None = None,
    now=None,
    git_overrides: dict[str, Any] | None = None,
) -> OfflineLaneWorker:
    """Build a real OfflineLaneWorker over the shared tmp-repo GitOps.

    *git_ops* is the ``git_ops`` fixture, so the worker's head snapshots and
    worktree resets land in the SAME repo :func:`_advance_main` commits to.
    Every seam left ``None`` is the worker's own default: B1-B3 pass only a
    numeric ``suite_runner``; a red-path row (:func:`_red_lane`) passes the
    fake task store, a real escalation queue and the clock it spaces passes
    with, and leaves every run seam at its default.
    """
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = git_ops.project_root
    config.git = GitConfig(**(git_overrides or {}))
    return OfflineLaneWorker(
        git_ops,
        config,
        lock_path=tmp_path / 'offline_lane.lock',
        suite_runner=suite_runner,
        task_client=task_client,
        escalation_queue=escalation_queue,
        now=now,
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
    call counter, so ``done`` is only set once a pass, INCLUDING its
    confirm, the engine's observe and any filing or escalation it makes,
    has fully returned. Counting at the detect seam instead would race the
    red path, which awaits real I/O after detection returns. Restores the
    original ``_run_once`` in the ``finally`` block so repeated calls
    (:func:`_drive_lane_pass` is called up to six times per scenario) never
    nest counting wrappers.

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
    precedent for ``_LANE_PASS_BOUND_SECS``. CEILING: the effective per-test
    pytest-timeout (`_orch_helpers.py::PYPROJECT_DEFAULT_TIMEOUT` under a
    bare run, `_orch_helpers.py::VERIFY_CLI_PER_TEST_TIMEOUT` under verify;
    ``timeout_method = "thread"``, ``--max-worker-restart=0``). Task 3491 is
    precedent AGAINST a bound near that ceiling, not for one: it REJECTED a
    30s ceiling for
    this exact collision and landed ``asyncio.wait_for(..., timeout=5)``
    instead (``test_usage_gate.py``:328,790) — its scope was exclusively
    ``test_usage_gate.py``; it never touched any offline-lane module. Both
    halves are pinned executably by
    ``test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling``
    below. Widening THIS bound alone can never make a broken
    staging pass — it only lengthens how long a genuinely broken SINGLE pass
    takes to fail. That safety property does not compose for free: callers
    chaining N passes (:func:`_drive_lane_pass`) sum this bound N times, so a
    worst case can now approach or exceed the effective per-test timeout
    before this function's own ``wait_for`` ever fires —
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
    escalation_queue: EscalationQueue,
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
    assert escalation_queue.get_pending() == []

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
    assert escalation_queue.get_pending() == []


# ---------------------------------------------------------------------------
# The red-path rows' fixture command (task 6861)
# ---------------------------------------------------------------------------

_FIXTURE_DIR = 'lane'
_TARGET = 'tests/test_lane.py::test_target'
_FIXTURE_COMMAND = LaneCommand(name='lane-fixture', command='pytest -q tests', cwd=_FIXTURE_DIR)
_FIXTURE_GIT: dict[str, Any] = {
    'offline_lane_legacy_numeric_enabled': False,
    'offline_lane_commands': [_FIXTURE_COMMAND],
}

# `test_target`'s verdict is the word in `<control>/test_target`: `red`,
# `green`, or `flaky` (red on the pass's first run, green after it: the
# marker it leaves is untracked, so the next worktree reset removes it).
_FIXTURE_TESTS = '''\
import pathlib

_CONTROL = pathlib.Path({control!r})
_HERE = pathlib.Path(__file__).parent


def _red(name):
    verdict = (_CONTROL / name).read_text().strip()
    if verdict == 'flaky':
        mark = _HERE / f'.{{name}}.ran'
        first = not mark.exists()
        mark.touch()
        return first
    return verdict == 'red'


def test_target():
    assert not _red('test_target'), 'the control directory says red'


def test_steady():
    pass
'''

_FIXTURE_PYPROJECT = '[tool.pytest.ini_options]\naddopts = "-p no:cacheprovider"\n'

# A stub of reify's numeric runner: detection fails naming no test; the
# confirm (`--confirm-failed`) prints verify.sh's usage dump and exits 64.
_NUMERIC_USAGE_STUB = '''\
#!/bin/sh
case " $* " in
  *" --confirm-failed "*) cat "$(dirname "$0")/verify-usage.txt"; exit 64 ;;
esac
echo 'offline-deep: 1 failed'
exit 1
'''


@pytest.fixture
def pytest_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fixture command runs ``pytest`` by name, as a lane command does."""
    monkeypatch.setenv('PATH', f'{Path(sys.executable).parent}{os.pathsep}{os.environ["PATH"]}')


async def _commit_files(repo: Path, files: dict[str, str]) -> None:
    """Write *files* (repo-relative path -> text) and commit them to main; a ``.sh`` file is executable."""
    for relative, text in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if path.suffix == '.sh':
            path.chmod(0o755)
    await _run(['git', 'add', *files], cwd=repo)
    await _run(['git', 'commit', '-m', 'add the lane fixture'], cwd=repo)


@dataclass(frozen=True)
class _RedLane:
    """A worker running one fixture command, and the task store, queue and clock it files through."""

    harness: Harness
    project_root: Path
    worker: OfflineLaneWorker
    client: FakeLedgerClient
    queue: EscalationQueue
    clock: Clock
    command: str
    control: Path

    def record(self) -> CommandRecord:
        return EpisodeStore(self.project_root, clock=self.clock).load(self.command)

    def episode(self, node: str = _TARGET) -> Episode:
        return self.record().episodes[node]

    def unlanded(self) -> list[Escalation]:
        """The pending D11 records on the command's sentinel id."""
        sentinel = sentinel_task_id(self.command)
        return [
            escalation for escalation in self.queue.get_pending()
            if escalation.task_id == sentinel and escalation.category == UNLANDED_RED_CATEGORY
        ]


def _red_lane(
    harness: Harness, git_ops: GitOps, tmp_path: Path, *, command: str, git_overrides: dict[str, Any],
) -> _RedLane:
    clock = Clock()
    client = FakeLedgerClient([])
    queue = EscalationQueue(tmp_path / 'escalations')
    worker = _build_worker(
        git_ops, tmp_path, task_client=client, escalation_queue=queue, now=clock, git_overrides=git_overrides,
    )
    _wire_lane(harness, worker)
    control = tmp_path / 'control'
    control.mkdir()
    return _RedLane(harness, git_ops.project_root, worker, client, queue, clock, command, control)


async def _pytest_fixture_lane(harness: Harness, git_ops: GitOps, tmp_path: Path) -> _RedLane:
    """A lane whose one command is a real pytest project on main, its ``test_target`` steered by the control dir."""
    lane = _red_lane(harness, git_ops, tmp_path, command=_FIXTURE_COMMAND.name, git_overrides=_FIXTURE_GIT)
    await _commit_files(git_ops.project_root, {
        f'{_FIXTURE_DIR}/pyproject.toml': _FIXTURE_PYPROJECT,
        f'{_FIXTURE_DIR}/tests/test_lane.py': _FIXTURE_TESTS.format(control=str(lane.control)),
    })
    return lane


@dataclass(frozen=True)
class _PassBlocks:
    """One pass's id and the outcome and effect blocks its run record keeps."""

    pass_id: str
    outcome: dict[str, Any] | None
    effect: dict[str, Any]


async def _drive_lane_pass(lane: _RedLane, target: str | None = None, *, minutes: float = 30) -> _PassBlocks:
    """One lane pass *minutes* after the last, with ``test_target`` set to *target*; returns its pass blocks.

    The pass is a real advance and the real on_merge_landed fan-out
    (:func:`_drive_advance`) followed by one full ``run()`` pass
    (:func:`_run_one_lane_pass`), so a caller composing several carries its own
    ``@pytest.mark.timeout`` (enforced by
    ``test_every_composing_caller_carries_a_timeout_override``).
    """
    if target is not None:
        (lane.control / 'test_target').write_text(target)
    lane.clock.advance(minutes=minutes)
    await _drive_advance(lane.harness, lane.project_root)
    await _run_one_lane_pass(lane.worker)
    pass_id = lane.record().last_pass_id
    assert pass_id is not None, 'the engine records every pass it observes'
    record = lane.worker.records.load_pass(lane.command, pass_id)
    assert record is not None and record.effect is not None, 'run_pass writes the effect last'
    return _PassBlocks(record.pass_id, record.outcome, record.effect)


def _confirmed(*, reproduced=(), cleared=(), inconclusive=()) -> dict[str, Any]:
    """A narrowed confirm's outcome block, as the pass record keeps it."""
    return {
        'arm': 'confirmed',
        'mode': 'narrowed',
        'reproduced': list(reproduced),
        'cleared': list(cleared),
        'inconclusive': list(inconclusive),
    }


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
# `_drive_lane_pass` each compose `_LANE_PASS_BOUND_SECS`-bounded passes, and
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
# net never discovered it and it ran marker-less under the then-60s pyproject
# default while requiring more than twice that.
_LANE_PASS_COMPOSING_HELPER_NAMES = frozenset(
    {'_run_lane', '_run_one_lane_pass', '_drive_lane_pass', '_assert_never_a_gate'},
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
    with the effective per-test timeout.

    Makes WORK item 2's open re-verification question ("does EVERY chaining
    caller carry such an override?") executable rather than prose, in two
    layers. COVERAGE: `_lane_pass_composing_callers` discovers every test
    whose own source calls any member of `_LANE_PASS_COMPOSING_HELPER_NAMES`
    at all — deliberately named there and not re-spelled here, so widening
    that set (as task 4203's remediation did, adding `_assert_never_a_gate`)
    cannot leave a stale hand-typed list behind in this docstring or in the
    `undeclared` assertion message below; every discovered name must
    appear in EITHER `worst_case_secs` below OR `single_pass_exempt` — so a
    test added tomorrow that calls e.g. `_drive_lane_pass` three times without a
    `@pytest.mark.timeout` marker fails this test immediately instead of
    silently passing because nobody added a row for it. VALUE: for every
    `worst_case_secs` entry, its marker must exist and its value must
    actually clear the REQUIRED timeout (task 4203's `required_timeout_secs`
    model) — the bounded-wait sum plus its out-of-bound real-git spawn
    allowance, not the bounded sum alone.

    `worst_case_secs` maps each composing test FUNCTION OBJECT — never a
    string, so a rename breaks this test loudly instead of silently dropping
    a row — to its worst-case bounded-wait sum, expressed in terms of this
    module's named timeout constants (never bare literals).

    `out_of_bound_spawns` (task 4203) maps the SAME function objects to their
    counted real-git subprocess spawns that happen OUTSIDE any bounded wait
    — the other multiplicand `required_timeout_secs` prices at
    `MEASURED_SPAWN_LATENCY_SECS` per spawn, expressed in this module's
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
    SAME wall clock (max, not sum). The other, b1, drives exactly one
    `_run_one_lane_pass` call.

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
        test_b4_a_confirmed_red_is_held_then_a_spaced_reproduction_files_one_task: 2 * _LANE_PASS_BOUND_SECS,
        test_b5_a_persistent_red_under_an_open_task_files_once_and_escalates_once: 6 * _LANE_PASS_BOUND_SECS,
        test_b6_a_red_the_narrowed_confirm_clears_ends_its_episode_and_files_nothing: 2 * _LANE_PASS_BOUND_SECS,
        test_b7_a_cancelled_task_escalates_once_and_after_resolution_a_new_task_names_it: (
            6 * _LANE_PASS_BOUND_SECS
        ),
        test_verify_usage_dump_on_two_spaced_passes_files_one_command_level_task: 2 * _LANE_PASS_BOUND_SECS,
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
    # Task 4203 — the other multiplicand `required_timeout_secs` needs per
    # composing test: its counted out-of-bound real-git spawns (never
    # `_run_one_lane_pass`'s spawns, which occur INSIDE the `_run_lane` 30s
    # window and are already carried by worst_case_secs above). Each
    # red-path row (task 6861) pays the repo fixture, one fixture commit and
    # one `_drive_advance` per `_drive_lane_pass` call. b3 drives one direct
    # `_drive_advance` plus `_assert_never_a_gate`'s own unbounded
    # advance-and-notify work — MEASURED (reviewer amendment, task 4203) at
    # `_SPAWNS_PER_ASSERT_NEVER_A_GATE` = 7, not the earlier hand
    # approximation of one `_drive_advance`-equivalent round (4), which
    # undercounted by 3 (see that constant's comment for the breakdown).
    two_pass_row = _SPAWNS_PER_REPO_FIXTURE + _SPAWNS_PER_FIXTURE_COMMIT + 2 * _SPAWNS_PER_DRIVE_ADVANCE
    six_pass_row = _SPAWNS_PER_REPO_FIXTURE + _SPAWNS_PER_FIXTURE_COMMIT + 6 * _SPAWNS_PER_DRIVE_ADVANCE
    out_of_bound_spawns = {
        test_b4_a_confirmed_red_is_held_then_a_spaced_reproduction_files_one_task: two_pass_row,
        test_b5_a_persistent_red_under_an_open_task_files_once_and_escalates_once: six_pass_row,
        test_b6_a_red_the_narrowed_confirm_clears_ends_its_episode_and_files_nothing: two_pass_row,
        test_b7_a_cancelled_task_escalates_once_and_after_resolution_a_new_task_names_it: six_pass_row,
        test_verify_usage_dump_on_two_spaced_passes_files_one_command_level_task: two_pass_row,
        test_b3_never_a_gate: (
            _SPAWNS_PER_REPO_FIXTURE
            + _SPAWNS_PER_DRIVE_ADVANCE
            + _SPAWNS_PER_ASSERT_NEVER_A_GATE
        ),
        # Task 4203 (review remediation) — the ONLY row in either table that
        # pays `_SPAWNS_PER_REPO_FIXTURE` TWICE: once for the `repo` fixture
        # every test in this module transitively pulls, and once more for the
        # fresh `_setup_repo` this test deliberately runs on a clean tmp_path
        # in order to MEASURE that very constant (it measures
        # `_SPAWNS_PER_FIXTURE_COMMIT` on that fresh repo too). Its
        # `_ControllableSuiteRunner` / `_build_worker` / `_wire_lane` setup
        # contributes nothing further — measured to be spawn-free inside
        # that test, not assumed.
        test_out_of_bound_spawn_counts_are_measured_not_asserted: (
            2 * _SPAWNS_PER_REPO_FIXTURE
            + _SPAWNS_PER_FIXTURE_COMMIT
            + _SPAWNS_PER_DRIVE_ADVANCE
            + _SPAWNS_PER_ASSERT_NEVER_A_GATE
        ),
    }
    assert set(worst_case_secs) == set(out_of_bound_spawns), (
        f'worst_case_secs and out_of_bound_spawns must classify exactly the '
        f'same composing tests — '
        f'{sorted(fn.__name__ for fn in set(worst_case_secs) ^ set(out_of_bound_spawns))} '
        f'appear in only one of the two tables, so a row added to one table '
        f'silently omits half of the required_timeout_secs inputs for that '
        f'test.'
    )
    single_pass_exempt = {
        test_b1_advance_triggers_from_head_run,
        test_b2_coalesces_burst_of_advances_to_one_rerun,
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
        required = required_timeout_secs(worst_case, out_of_bound)
        spawn_allowance = out_of_bound * MEASURED_SPAWN_LATENCY_SECS
        markers = [m for m in getattr(fn, 'pytestmark', []) if m.name == 'timeout']
        assert markers, (
            f'{fn.__name__} composes bounded waits to a worst case of '
            f'{worst_case}s plus {out_of_bound} out-of-bound real-git spawns '
            f'({out_of_bound} x {MEASURED_SPAWN_LATENCY_SECS}s = '
            f'{spawn_allowance}s) — required = {required}s — but carries no '
            f'@pytest.mark.timeout override. Without one it inherits the '
            f'effective per-test timeout (the CLI --timeout under verify, else '
            f'the pyproject default), a budget this module neither sets nor '
            f'checks against its worst case. Were that budget ever below '
            f'{required}s, then under timeout_method="thread" with '
            f'--max-worker-restart=0, pytest-timeout os._exit()s the xdist '
            f"worker instead of failing cleanly, discarding _run_lane's own "
            f'well-located TimeoutError. Add @pytest.mark.timeout(N) with '
            f'N >= {required} ({worst_case}s bounded + {out_of_bound} spawns '
            f'x {MEASURED_SPAWN_LATENCY_SECS}s).'
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
            f'{MEASURED_SPAWN_LATENCY_SECS}s = {spawn_allowance}s — the '
            f'override does not actually clear what it exists to cover: it '
            f'fires before that worst case completes, and under '
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
    `_SPAWNS_PER_LANE_PASS_WORST_CASE`x `MEASURED_SPAWN_LATENCY_SECS` (task
    3451's measured worst-case happy-path subprocess spawn latency: n=3,
    2.13/3.10/4.71s, load-per-core 6.6). The multiplier is DERIVED, not
    guessed: it is the measured worst-case count of subprocess spawns that
    occur INSIDE a single `_run_lane` window, and that worst case lives in
    the SIBLING infra module (``test_offline_lane_infra_integration.py``'s
    `test_ib3_a_confirmed_infra_red_is_held_then_a_spaced_reproduction_files_one_task`):
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
    repo), so the per-test budget covers FIXTURE SETUP AND TEARDOWN, not just the
    call phase. Every test in both offline-lane modules transitively pulls
    the `repo` fixture, whose `_setup_repo` costs `_SPAWNS_PER_REPO_FIXTURE`
    git spawns (`init`, `config` x2, `add`, `commit`), and each
    `_drive_advance` costs `_SPAWNS_PER_DRIVE_ADVANCE` more (`get_main_sha`,
    `add`, `commit`, `rev-parse` — NOT `_note_merge_all`'s `git diff`, which
    this module's `harness` fixture stubs to an `AsyncMock` and so never
    reaches a subprocess; see `_SPAWNS_PER_DRIVE_ADVANCE`'s own comment and
    `test_out_of_bound_spawn_counts_are_measured_not_asserted`) — the 40% of
    the effective per-test timeout the 0.6 fraction holds back is what pays
    for that out-of-bound work.

    When no timeout is in effect at all, this either fails loudly (this
    module's own `orchestrator/pyproject.toml` is the governing inifile —
    genuine drift, the premise these bounds are sized against no longer
    holds) or skips, naming the governing inifile (an invocation artifact,
    e.g. a root-bound run).
    """
    from test_lane_lock_leak_guard import _effective_per_test_timeout  # noqa: PLC0415

    floor = _SPAWNS_PER_LANE_PASS_WORST_CASE * MEASURED_SPAWN_LATENCY_SECS
    assert floor <= _LANE_PASS_BOUND_SECS, (
        f'_LANE_PASS_BOUND_SECS ({_LANE_PASS_BOUND_SECS}) must clear '
        f'{_SPAWNS_PER_LANE_PASS_WORST_CASE}x the measured worst-case happy-path '
        f'subprocess spawn latency ({MEASURED_SPAWN_LATENCY_SECS}s, task 3451: n=3 '
        f'2.13/3.10/4.71, load-per-core 6.6) = {floor}s. The '
        f'{_SPAWNS_PER_LANE_PASS_WORST_CASE}-spawn multiplier is the measured worst-case '
        f'count of subprocess spawns inside a single _run_lane window (see '
        f'test_ib3_a_confirmed_infra_red_is_held_then_a_spaced_reproduction_files_one_task in '
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
        _SPAWNS_PER_FIXTURE_COMMIT as _infra_spawns_per_fixture_commit,
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
    # new required_timeout_secs multiplicands: each module independently
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
    # Task 6861 — the same anti-drift rationale, for the fixture commit both
    # modules' red-path rows pay once.
    assert _infra_spawns_per_fixture_commit == _SPAWNS_PER_FIXTURE_COMMIT, (
        f"this module's _SPAWNS_PER_FIXTURE_COMMIT ({_SPAWNS_PER_FIXTURE_COMMIT}) has "
        f"drifted from test_offline_lane_infra_integration.py's copy "
        f'({_infra_spawns_per_fixture_commit}) — both modules commit their fixture '
        f"with one add and one commit, so a real divergence here means one module's "
        f'measurement (or helper) has silently changed.'
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
    # against the additive `required_timeout_secs` model this task adopts
    # for @pytest.mark.timeout OVERRIDES. The two models are not applied to
    # the same population: an override is a number this suite CHOOSES, so
    # `required_timeout_secs` prices it at every out-of-bound spawn's
    # worst-case latency; the marker-less budget is the ambient effective
    # per-test timeout (`_orch_helpers.py::PYPROJECT_DEFAULT_TIMEOUT` under a bare run,
    # `_orch_helpers.py::VERIFY_CLI_PER_TEST_TIMEOUT` under verify), which
    # this suite cannot move, so it keeps the 0.6 fraction as a cheaper proxy
    # instead. When that default was 60s (tasks 4030/4203), gating the
    # marker-less population additively was unsatisfiable: b1/b4/b6 (this
    # module) and ib1/ib3/ib5 (the sibling infra module) each do the repo
    # fixture plus at least one _drive_advance = _SPAWNS_PER_REPO_FIXTURE +
    # _SPAWNS_PER_DRIVE_ADVANCE = 9 spawns = 42.39s of out-of-bound work
    # against the 24s the 0.6 fraction then left — precisely task 4030's "10
    # of the 15 tests exceed the 60s global" observation. Pricing every spawn
    # at the worst-case SINGLE-spawn 4.71s is a sound upper bound for sizing a
    # number this suite chooses (an override) and an unsound hard gate for one
    # it cannot move (the ambient budget) — no observed run actually pays that
    # price on every spawn.
    #
    # What IS pinnable, and is asserted here executably rather than left as
    # prose: the 40% the 0.6 fraction holds back at the effective timeout
    # must cover at least the repo fixture's own out-of-bound spawns (every
    # test in both modules transitively pays this cost, marker-less or not),
    # priced by the SAME additive model. This is the fixture-only baseline
    # both models agree on.
    fixture_only_required = required_timeout_secs(0.0, _SPAWNS_PER_REPO_FIXTURE)
    marker_less_headroom = (1 - _MARKERLESS_CEILING_FRACTION) * global_timeout
    assert marker_less_headroom >= fixture_only_required, (
        f'the {(1 - _MARKERLESS_CEILING_FRACTION) * 100:.0f}% of the effective per-test '
        f'timeout ({global_timeout}s) the {_MARKERLESS_CEILING_FRACTION} fraction holds '
        f'back as marker-less headroom = {marker_less_headroom}s, but the repo fixture '
        f'alone already requires {fixture_only_required}s '
        f'(_SPAWNS_PER_REPO_FIXTURE={_SPAWNS_PER_REPO_FIXTURE} spawns x '
        f'{MEASURED_SPAWN_LATENCY_SECS}s) under the additive model this task adopts for '
        f'@pytest.mark.timeout overrides. The 0.6 fraction is deliberately kept (not '
        f'replaced by the additive model) for the marker-less population, since a '
        f"marker-less test's timeout is the ambient effective per-test timeout "
        f'({global_timeout}s for this run), which this suite cannot widen — but '
        f'that choice is only sound while this fixture-only floor holds; a failure here '
        f'means MEASURED_SPAWN_LATENCY_SECS has been re-measured upward enough that even '
        f'the cheapest marker-less test (fixture only, zero _drive_advance calls) no '
        f'longer fits, and the 0.6 fraction itself needs re-deriving.'
    )


@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)  # task 4203 review remediation: the ONLY
# marker-carrying test in this module composing NO _run_lane pass -- its whole bounded term
# is _assert_never_a_gate's _NOTE_OFFLINE_LANE_BOUND_SECS + _NOTE_MERGE_ALL_BOUND_SECS =
# 15.5s. Out-of-bound term = 23 spawns: 2 x _SPAWNS_PER_REPO_FIXTURE (the repo fixture
# AND the fresh _setup_repo measured against below -- the only doubled fixture term in
# either module), plus _SPAWNS_PER_FIXTURE_COMMIT (2), _SPAWNS_PER_DRIVE_ADVANCE (4) and
# _SPAWNS_PER_ASSERT_NEVER_A_GATE (7); 23 x MEASURED_SPAWN_LATENCY_SECS = 108.33s,
# required = 123.83s. The verify budget
# clears that requirement, and the value is ENFORCED by this test's row in
# test_every_composing_caller_carries_a_timeout_override, which fails if the marker ever
# drops below required.
@pytest.mark.asyncio
async def test_out_of_bound_spawn_counts_are_measured_not_asserted(
    harness: Harness, git_ops: GitOps, repo: Path, tmp_path: Path,
) -> None:
    """`_SPAWNS_PER_DRIVE_ADVANCE` / `_SPAWNS_PER_REPO_FIXTURE` are the
    multiplicands `required_timeout_secs` (task 4203) prices at
    `MEASURED_SPAWN_LATENCY_SECS` per spawn — this test MEASURES them by
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
    unaffected by whether a pass is actually held in flight. Composing a real
    pass here would only add a further `_LANE_PASS_BOUND_SECS` of bounded wait
    plus its in-window spawns, for zero additional measurement value.

    (An earlier revision justified that choice differently — a real pass
    "would also wrongly register THIS test in `_lane_pass_composing_callers`'
    AST net, forcing it into `worst_case_secs` and a `@pytest.mark.timeout`
    override it does not need". That reasoning inverted once the reviewer
    amendment added the `_assert_never_a_gate` measurement: THAT helper is
    itself in `_LANE_PASS_COMPOSING_HELPER_NAMES`, so this test is already in
    the net, already classified in `worst_case_secs`, and does need the
    override it now carries. The decision survives on the reason above; only
    its justification changed.)

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
    `_drive_advance` (directly or via `_drive_lane_pass`).

    CARRIES `@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)` — it is NOT marker-less. An earlier
    revision of this paragraph claimed this test had "zero bounded waits" and
    the same real-git shape the marker-less B1 test carries unmarked.
    The reviewer amendment that added the `_assert_never_a_gate` measurement
    above falsified BOTH clauses: that helper composes 15.5s of bounded waits
    (`_NOTE_OFFLINE_LANE_BOUND_SECS` + `_NOTE_MERGE_ALL_BOUND_SECS`), and at
    23 out-of-bound spawns this test does ~2.6x the real-git work of a
    B1 test (9: the `repo` fixture plus one `_drive_advance`).
    `required_timeout_secs` puts the requirement at 123.83s. See the marker
    comment above for the worked derivation; the value is enforced by
    `test_every_composing_caller_carries_a_timeout_override`'s row for this
    test, not by this prose.
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
        f'required_timeout_secs (the @pytest.mark.timeout override sizing '
        f'model) — a drift here silently mis-prices every composing test '
        f'that calls _drive_advance, directly or via _drive_lane_pass.'
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
        f'required_timeout_secs — every test in this module transitively '
        f'pulls the repo fixture, so a drift here mis-prices every '
        f'composing test, not just one.'
    )

    calls.clear()
    with patch('orchestrator.git_ops._run', side_effect=_counting_run), \
         patch(f'{__name__}._run', side_effect=_counting_run):
        await _commit_files(fresh_repo, {'fixture.txt': 'fixture\n'})

    assert len(calls) == _SPAWNS_PER_FIXTURE_COMMIT, (
        f'_commit_files spawned {len(calls)} real git subprocess(es) '
        f'({calls}), not the pinned _SPAWNS_PER_FIXTURE_COMMIT = '
        f'{_SPAWNS_PER_FIXTURE_COMMIT}. This count is a multiplicand of '
        f'required_timeout_secs — every red-path row commits its fixture '
        f'command once, so a drift here mis-prices each of them.'
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
        queue = EscalationQueue(tmp_path / 'escalations')
        worker = _build_worker(git_ops, tmp_path, suite_runner=runner, escalation_queue=queue)
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
        await _assert_never_a_gate(harness, worker, runner, repo, git_ops, queue)

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


@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)  # task 4030: NOT a _drive_lane_pass chainer,
# which is why the task-3832 review's marker sweep missed it -- but it composes anyway: a 30s
# _run_lane bound raced concurrently by wait_entered (max, not sum), then
# _assert_never_a_gate's 0.5 + 15.0 SEQUENTIALLY after it = 45.5s bounded, on top of
# unbounded real-git spawns: repo init (_SPAWNS_PER_REPO_FIXTURE=5), one _drive_advance (4),
# and _assert_never_a_gate's own get_main_sha + two _advance_main rounds
# (_SPAWNS_PER_ASSERT_NEVER_A_GATE=7, task 4203 reviewer amendment -- MEASURED, supersedes an
# earlier "two _drive_advance/_advance_main rounds" approximation) = 16 spawns x 4.71s =
# 75.36s, required = 120.86s, comfortably under this marker.
@pytest.mark.asyncio
async def test_b3_never_a_gate(harness, git_ops, repo, tmp_path):
    """B3 (PRD §8, C7) — a merge-landed notification while the lane is
    in-flight must never gate the merge: it returns promptly (never blocks
    the synchronous notifiee call on the in-flight run), sets _dirty for a
    coalesced re-run, is fail-open when the notifiee raises, and never
    files an escalation (the only halt/gate-adjacent side-effect reachable
    from this call path)."""
    runner = _ControllableSuiteRunner()
    queue = EscalationQueue(tmp_path / 'escalations')
    worker = _build_worker(git_ops, tmp_path, suite_runner=runner, escalation_queue=queue)
    _wire_lane(harness, worker)

    await _drive_advance(harness, repo)
    lane_task = asyncio.create_task(_run_lane(worker, expected_passes=1))
    await runner.wait_entered()

    await _assert_never_a_gate(harness, worker, runner, repo, git_ops, queue)

    runner.release()
    await lane_task


# ---------------------------------------------------------------------------
# B4 — a confirmed red is held, then a spaced reproduction files one task
# ---------------------------------------------------------------------------


# Two _drive_lane_pass calls: 2 x _LANE_PASS_BOUND_SECS = 60s bounded, plus the repo fixture,
# the fixture commit and two _drive_advance rounds out of bound (5 + 2 + 2 x 4 = 15 spawns x
# 4.71s = 70.65s); required = 130.65s, enforced by this test's rows in
# test_every_composing_caller_carries_a_timeout_override.
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
@pytest.mark.usefixtures('pytest_on_path')
@pytest.mark.asyncio
async def test_b4_a_confirmed_red_is_held_then_a_spaced_reproduction_files_one_task(
    harness, git_ops, repo, tmp_path,
):
    """B4 (red-path §6 scenarios 3 and 4) — a test that fails detection and the
    narrowed confirm files nothing on its first pass and holds one counted
    pass.  Reproduced on a pass 30 minutes later, it files exactly one NORMAL
    task, committed out of planning, whose id the episode and that pass's
    effect both carry; the digest shows both passes.  Filing raises no
    escalation and leaves the merge queue untouched."""
    lane = await _pytest_fixture_lane(harness, git_ops, tmp_path)

    first = await _drive_lane_pass(lane, 'red')

    assert first.outcome == _confirmed(reproduced=[_TARGET])
    assert first.effect['tasks_filed'] == []
    assert lane.client.submitted == []
    assert lane.episode().counted_passes == (first.pass_id,)

    second = await _drive_lane_pass(lane, 'red')

    (arguments,) = lane.client.submitted
    episode = lane.episode()
    assert episode.covering_task_id is not None
    assert episode.planning_committed
    assert second.effect['tasks_filed'] == [episode.covering_task_id]
    task, error = await lane.client.get_task(episode.covering_task_id)
    assert error is None and task is not None and task['status'] == 'pending'
    for counted in (first, second):
        assert f'pass {counted.pass_id} detect' in arguments['details']
    assert arguments['metadata']['merge_lane'] == 'normal', (
        'a lane fix task routes through the standard TDD→PR→merge gate, never '
        'the b3_gate red-main fix-forward path'
    )
    assert lane.queue.get_pending() == []
    assert harness.get_merge_halt_status() == {'wired': False}


# ---------------------------------------------------------------------------
# B5 — a persistent red under an open task: one task, one escalation at 24 h
# ---------------------------------------------------------------------------


# Six _drive_lane_pass calls: 6 x _LANE_PASS_BOUND_SECS = 180s bounded, plus 5 + 2 + 6 x 4 =
# 31 out-of-bound spawns x 4.71s = 146.01s; required = 326.01s, so 360 (the next multiple of
# the 60s grid), enforced by this test's rows in
# test_every_composing_caller_carries_a_timeout_override.
@pytest.mark.timeout(360)
@pytest.mark.usefixtures('pytest_on_path')
@pytest.mark.asyncio
async def test_b5_a_persistent_red_under_an_open_task_files_once_and_escalates_once(
    harness, git_ops, repo, tmp_path,
):
    """B5 (red-path §6 scenarios 13 and 14) — once a task covers the test, the
    test going green and then red twice files nothing new (14), and staying
    red files no second task: no escalation 12 hours after filing, exactly one
    level-1 record on the command's sentinel id once a counted pass lands 24
    hours after filing, and none after that (13)."""
    lane = await _pytest_fixture_lane(harness, git_ops, tmp_path)
    await _drive_lane_pass(lane, 'red')
    await _drive_lane_pass(lane, 'red')
    task_id = lane.episode().covering_task_id
    assert task_id is not None

    await _drive_lane_pass(lane, 'green')
    await _drive_lane_pass(lane, 'red')
    await _drive_lane_pass(lane, 'red', minutes=11 * 60)

    assert len(lane.client.submitted) == 1
    assert lane.episode().covering_task_id == task_id
    assert lane.unlanded() == []

    await _drive_lane_pass(lane, 'red', minutes=12 * 60)

    (record,) = lane.unlanded()
    assert task_id in record.summary
    assert (record.level, record.severity, record.agent_role) == (1, 'blocking', 'orchestrator-offline-lane')
    assert lane.episode().escalation_id == record.id

    await _drive_lane_pass(lane, 'red')

    assert lane.unlanded() == [record]
    assert len(lane.client.submitted) == 1
    assert harness.get_merge_halt_status() == {'wired': False}


# ---------------------------------------------------------------------------
# B6 — a red the narrowed confirm clears ends its episode and files nothing
# ---------------------------------------------------------------------------


# Two _drive_lane_pass calls; the same 130.65s requirement as B4.
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
@pytest.mark.usefixtures('pytest_on_path')
@pytest.mark.asyncio
async def test_b6_a_red_the_narrowed_confirm_clears_ends_its_episode_and_files_nothing(
    harness, git_ops, repo, tmp_path,
):
    """B6 (red-path §6 scenario 6) — a held test that fails detection and then
    passes the narrowed confirm is cleared: its episode ends and no fix task
    or escalation is raised."""
    lane = await _pytest_fixture_lane(harness, git_ops, tmp_path)
    await _drive_lane_pass(lane, 'red')

    flaked = await _drive_lane_pass(lane, 'flaky')

    assert flaked.outcome == _confirmed(cleared=[_TARGET])
    assert [(ended['node'], ended['reason']) for ended in flaked.effect['episodes_ended']] == [
        (_TARGET, 'cleared'),
    ]
    assert not lane.episode().live
    assert lane.client.submitted == []
    assert lane.queue.get_pending() == []


# ---------------------------------------------------------------------------
# B7 — a cancelled covering task: one escalation, then a refile once resolved
# ---------------------------------------------------------------------------


# Six _drive_lane_pass calls; the same 326.01s requirement as B5.
@pytest.mark.timeout(360)
@pytest.mark.usefixtures('pytest_on_path')
@pytest.mark.asyncio
async def test_b7_a_cancelled_task_escalates_once_and_after_resolution_a_new_task_names_it(
    harness, git_ops, repo, tmp_path,
):
    """B7 (red-path §6 scenarios 13 and 15) — the test stays red after its
    covering task is cancelled: no refile, one escalation once a counted pass
    lands 24 hours after filing.  Once that escalation is resolved the next
    red starts a new episode held at one pass, and a second spaced red files
    a new task naming the cancelled one."""
    lane = await _pytest_fixture_lane(harness, git_ops, tmp_path)
    await _drive_lane_pass(lane, 'red')
    await _drive_lane_pass(lane, 'red')
    cancelled = lane.episode().covering_task_id
    assert cancelled is not None
    lane.client.set_status(cancelled, 'cancelled')

    await _drive_lane_pass(lane, 'red', minutes=12 * 60)
    assert lane.unlanded() == []
    await _drive_lane_pass(lane, 'red', minutes=12 * 60)

    (record,) = lane.unlanded()
    assert cancelled in record.summary
    assert len(lane.client.submitted) == 1
    lane.queue.resolve(record.id, 'seen; refile when it persists', resolved_by='operator')

    held = await _drive_lane_pass(lane, 'red')
    refiled = await _drive_lane_pass(lane, 'red')

    assert held.effect['episodes_started'] == [_TARGET]
    assert held.effect['tasks_filed'] == []
    assert len(lane.client.submitted) == 2
    assert refiled.effect['tasks_filed'] == [lane.episode().covering_task_id]
    assert lane.client.submitted[-1]['metadata']['x_offline_lane']['prior_task_ids'] == [cancelled]
    assert lane.unlanded() == []
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
# Incident reproduction (task 5308) — a verify.sh usage dump names no test
# ---------------------------------------------------------------------------


# A faithful reproduction of reify scripts/verify.sh's error+usage() dump
# (reify:5308 incident; twin of the fixture in test_offline_lane.py): an
# em-dash ``verify.sh: ERROR — <msg>`` banner followed by the full usage()
# block (bare ``Usage:``/``Options:`` section headers + invocation/option
# lines). run-offline-deep.sh merges stderr into stdout, so this whole blob
# reaches the numeric confirm.
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


# Two _drive_lane_pass calls; the same 130.65s requirement as B4 (the stub commit is the
# fixture commit).
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
@pytest.mark.asyncio
async def test_verify_usage_dump_on_two_spaced_passes_files_one_command_level_task(
    harness, git_ops, repo, tmp_path,
):
    """Incident reproduction (task 5308; reify:5264), as red-path §6 scenario 8 —
    the numeric leg's DEFAULT seams run a committed ``run-offline-deep.sh``
    stub whose confirm prints reify verify.sh's usage dump and exits 64.  Each
    pass is unconfirmable for the usage text, so no usage line ever becomes a
    test id: the command's pseudo-test is the only episode, and the second
    spaced pass files one command-level task saying why.  No escalation."""
    await _commit_files(repo, {
        RUN_OFFLINE_DEEP_SCRIPT: _NUMERIC_USAGE_STUB,
        'scripts/verify-usage.txt': _VERIFY_USAGE_DUMP,
    })
    lane = _red_lane(harness, git_ops, tmp_path, command=LEGACY_NUMERIC_COMMAND, git_overrides={})

    first = await _drive_lane_pass(lane)
    second = await _drive_lane_pass(lane)

    usage_text = {
        'arm': 'unconfirmable',
        'reason': 'usage_text',
        'detail': "verify.sh: ERROR — unknown argument '--test-threads=1'",
    }
    assert first.outcome == second.outcome == usage_text
    pseudo_test = unconfirmable_node(LEGACY_NUMERIC_COMMAND)
    assert list(lane.record().episodes) == [pseudo_test]
    (arguments,) = lane.client.submitted
    assert arguments['metadata']['x_offline_lane']['nodes'] == [pseudo_test]
    assert 'exits non-zero with no reproducible failing test' in arguments['title']
    assert 'usage_text' in arguments['description']
    assert second.effect['tasks_filed'] == [lane.episode(pseudo_test).covering_task_id]
    assert lane.queue.get_pending() == []
