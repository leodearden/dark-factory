"""Integration gate for the infra extension of the offline deep-test lane
(task 1960, IE2 — the ζ-analog for infra; red path, task 6861).

Stands up the REAL trigger→worker→warm-worktree→infra-sub-run→red-path chain
end-to-end and RUNS (does not tabulate) the infra boundary scenarios IB1-IB6
below, asserting landed behaviour from:

* β1 (task 1951) — ``Harness._offline_lane_notifiee`` / ``_note_merge_all`` /
  ``_note_offline_lane`` fan-out (``harness.py``) — the real merge-landed
  hot-path callback, shared by numeric AND infra sub-runs (one coalesced
  trigger, two sub-runs).
* IE1 (task 1959) — ``OfflineLaneWorker``'s ``infra_runner`` seam
  (``offline_lane.py``): ``_run_once`` runs BOTH the numeric
  (``suite_runner``) and infra (``infra_runner``) legs at the SAME snapshot
  head in the SAME ``_offline-deep`` worktree when
  ``config.git.offline_lane_infra_enabled`` is True; the default infra seam
  ``offline_lane_runs.py::infra_runner`` shells out to
  ``tests/infra/run_all.sh --scope host-infra`` (reify H9) and parses its
  ``RESULT: <verdict> (<name>)`` lines (``offline_lane_runs.py::INFRA_RESULT_RE``).
* δ2 (task 6861) — the infra leg's pass is ``offline_lane_pass.py::run_pass``
  like every other: the confirm re-runs the same script and compares the two
  runs (``offline_lane_confirm.py::confirm``), with a test's PASS or SKIP line
  as its per-test evidence, and ``offline_lane_episodes.py::LaneEngine``
  decides what the pass holds, files or escalates — the red path of
  ``plans/offline-lane-red-path-prd.md``, whose §6 scenarios IB3-IB6 assert
  (in brackets below).

STRATEGY — the numeric ``suite_runner`` is injected GREEN in every scenario
so the infra leg is the SOLE signal.  IB1/IB2 inject the ``infra_runner``
seam.  IB3-IB6 inject nothing at the run boundary: they commit a stub
``tests/infra/run_all.sh`` that prints one ``RESULT:`` line per file in a
control directory (the file's word is the test's verdict), and the worker's
default infra seam runs it for detection and again for the confirm.  What
those rows fake is the task store (``_lane_episode_fakes.py::FakeLedgerClient``)
and the clock (the worker's ``now``); the escalation queue is real.

PRECONDITION (bound at the dependency-graph level, not executable here):
host-exclusive infra tests are host-global-unsafe by definition (real
cgroup burn / reflink FS), so they cannot run inside a hermetic pytest. The
"live reify checkout with H1/H9 landed" precondition is instead bound at
the dependency-graph level: reify:4929 (H9 ``run_all --scope host-infra``
runner) and reify:4921 (H1 classification manifest) are both ``done`` on
reify ``main`` (verified via ``get_external_statuses`` at plan time — see
this task's ``.task/plan.json`` analysis for the full rationale).

Scenario map (red-path §6 scenarios in brackets):

  IB1 - FROM-HEAD — an advance triggers a from-head infra sub-run
  IB2 - NEVER A GATE (C7/§6) — an infra sub-run in-flight never gates a
        merge: prompt return, fail-open, no halt/escalation
  IB3 - a confirmed infra red is held at one counted pass; a reproduction
        30 minutes later files exactly one task naming only the FAILed test
        [3, 4]
  IB4 - a persistent infra red under an open task files no second task and
        raises one escalation at 24 h; green then red again files nothing
        [13, 14]
  IB5 - an infra red whose confirm prints PASS for the test is cleared: its
        episode ends and nothing is filed [6]
  IB6 - a cancelled covering task: one escalation at 24 h; once it is
        resolved the next red is held and a second spaced red files a task
        naming the prior [13, 15]

OUT OF SCOPE: real reify infra test timing/content and host-exclusive test
execution itself (owned by reify H9), and the flip to always-on
(``offline_lane_infra_enabled`` stays default-False here) — owned by IE4.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import logging
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

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.offline_lane import OfflineLaneWorker
from orchestrator.offline_lane_episode_state import CommandRecord, Episode, EpisodeStore
from orchestrator.offline_lane_escalations import UNLANDED_RED_CATEGORY, sentinel_task_id
from orchestrator.offline_lane_records import LEGACY_INFRA_COMMAND
from orchestrator.offline_lane_runs import INFRA_SCOPE, RUN_ALL_INFRA_SCRIPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout bounds (task 4030) — this module's copy of
# test_offline_lane_integration.py's named constants for the bounded waits
# below, replacing near-identical inline 30.0 literals (the duplication that
# let a single false citation drift into three near-verbatim docstring
# copies). Modeled on the repo's own constant-plus-derivation-comment
# convention: _orch_helpers.py's CANCEL_SCOPE_BARRIER_TIMEOUT /
# CANCEL_SCOPE_PURE_UNIT_TIMEOUT and test_lane_lock_leak_guard.py's
# _FOREIGN_HOLDER_* block. The full floor (task 3451's measured spawn
# latency) / ceiling (the effective per-test pytest-timeout) derivation lives in
# `_run_lane`'s docstring below; the executable pins are
# `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling` and
# `test_every_composing_caller_carries_a_timeout_override`.
# ---------------------------------------------------------------------------

# Bound for a single `_run_lane` pass / `_ControllableInfraRunner.wait_entered`
# call. See `_run_lane`'s docstring for the full derivation.
_LANE_PASS_BOUND_SECS = 30.0

# Bound for `_assert_infra_never_a_gate`'s `_note_offline_lane` promptness check.
_NOTE_OFFLINE_LANE_BOUND_SECS = 0.5

# Bound for `_assert_infra_never_a_gate`'s `_note_merge_all` promptness check.
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
# is sized against. Measured IN THIS MODULE's
# `test_ib3_a_confirmed_infra_red_is_held_then_a_spaced_reproduction_files_one_task`: 3 git
# spawns from `_run_once` plus 2 real infra `run_all.sh` spawns. See
# `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling` for the
# full derivation.
_SPAWNS_PER_LANE_PASS_WORST_CASE = 5

# Task 4203 — this module's own copy of the sibling numeric module's
# constant of the same name (see that module for the full rationale): a
# multiplicand `_orch_helpers.py::required_timeout_secs` prices at
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
# production) — under THIS module's `harness` fixture (identical to the
# sibling's), `h.git_ops.get_merge_diff_files` is replaced wholesale with an
# `AsyncMock(return_value=([], None))`, so that leg never reaches a
# subprocess at all here.
_SPAWNS_PER_DRIVE_ADVANCE = 4

# Task 4203 (reviewer amendment) — this module's own copy of the sibling's
# constant of the same name (see that module for the full rationale), needed
# only by `test_ib2_infra_run_in_flight_never_gates_merge`'s
# `out_of_bound_spawns` row. MEASURED, not hand-counted (see
# `test_out_of_bound_spawn_counts_are_measured_not_asserted`):
# `_assert_infra_never_a_gate` performs one `git_ops.get_main_sha()` (1) plus
# TWO `_advance_main` rounds (3 each) = 7. Both `_note_merge_all` calls inside
# it contribute zero spawns here, same stub as `_SPAWNS_PER_DRIVE_ADVANCE`'s.
# An earlier hand approximation priced this whole function as "one
# _drive_advance-equivalent round" (4), undercounting the true value by 3.
_SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE = 7

# Task 6861 — the fourth multiplicand: IB3-IB6 commit their stub
# `tests/infra/run_all.sh` to main before their first pass. MEASURED, not
# hand-counted (see `test_out_of_bound_spawn_counts_are_measured_not_asserted`):
# `_commit_infra_stub` spawns one `add` and one `commit`.
_SPAWNS_PER_FIXTURE_COMMIT = 2


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
    numeric seam would go red (subprocess rc=127) on every pass. Forcing it
    green leaves the infra leg as the SOLE signal in every scenario below.
    """
    return (0, '')


def _build_infra_worker(
    git_ops: GitOps,
    tmp_path: Path,
    *,
    suite_runner=None,
    infra_runner=None,
    task_client=None,
    escalation_queue: EscalationQueue | None = None,
    now=None,
    git_overrides: dict[str, Any] | None = None,
) -> OfflineLaneWorker:
    """Build a real OfflineLaneWorker wired for the INFRA integration harness.

    Adapts ``test_offline_lane_integration.py``'s ``_build_worker``: *git_ops*
    is the shared real tmp-repo GitOps, so the worker's head snapshots and
    worktree resets land in the SAME repo :func:`_advance_main` commits to.
    ``git_overrides`` defaults to ``{'offline_lane_infra_enabled': True}`` (a
    caller-supplied dict REPLACES it wholesale), and ``suite_runner`` to
    :func:`_default_green_suite_runner`.  ``infra_runner`` left ``None`` is the
    worker's own default seam, which IB3-IB6 run over their committed stub;
    a red-path row (:func:`_red_infra_lane`) also passes the fake task store,
    a real escalation queue and the clock it spaces passes with.
    """
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = git_ops.project_root
    config.git = GitConfig(
        **(git_overrides if git_overrides is not None else _DEFAULT_INFRA_GIT_OVERRIDES)
    )
    return OfflineLaneWorker(
        git_ops,
        config,
        lock_path=tmp_path / 'offline_lane.lock',
        suite_runner=suite_runner if suite_runner is not None else _default_green_suite_runner,
        infra_runner=infra_runner,
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
    timeout: float = _LANE_PASS_BOUND_SECS,
) -> None:
    """Drive worker.run() as a real background task until *expected_passes*
    full passes (``_run_once`` calls) have COMPLETED, then cancel the loop
    task.

    Wraps ``worker._run_once`` itself — NOT ``suite_runner``/``infra_runner``
    alone — with a call counter, so ``done`` is only set once a pass,
    INCLUDING its confirm, the engine's observe and any filing or
    escalation it makes (numeric OR infra), has fully returned. Counting at
    either detect seam instead would race the red path, which awaits real
    I/O after detection returns. Restores the original ``_run_once`` in the
    ``finally`` block so repeated calls (:func:`_drive_infra_pass` is called
    up to six times per scenario) never nest counting wrappers.

    Adapted verbatim from ``test_offline_lane_integration.py``'s
    ``_run_lane``, generalized from a fixed N=1 to an arbitrary pass count
    (IB1's single pass via :func:`_run_one_lane_pass`; IB3-IB6 one per
    :func:`_drive_infra_pass` call). Requires ``worker._dirty`` (or
    the wake event) to already be set by a prior trigger, exactly as
    production wiring would leave it.

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
    chaining N passes (:func:`_drive_infra_pass`) sum this bound N times, so
    a worst case can now approach or exceed the effective per-test timeout
    before this function's own ``wait_for`` ever fires — trading a clean,
    well-located ``TimeoutError`` here for
    pytest-timeout's blunter thread-mode worker kill instead (task 3832
    review; see ``orchestrator/pyproject.toml``'s ``timeout``/
    ``timeout_method`` comment). Multi-pass callers whose worst-case sum is
    at or above that ceiling carry their own ``@pytest.mark.timeout``
    override — see
    ``test_ib4_same_infra_set_recurrence_updates_not_duplicates`` and
    ``test_ib2_infra_run_in_flight_never_gates_merge`` below — a rule
    ``test_every_composing_caller_carries_a_timeout_override`` now enforces
    rather than merely describes.

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
    """Drive worker.run() as a real background task for exactly one pass
    (IB1, and each :func:`_drive_infra_pass` call)."""
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

    async def wait_entered(self, timeout: float = _LANE_PASS_BOUND_SECS) -> None:
        """Block until the held first call has actually started (is in-flight).

        ``timeout`` defaults to 30.0 for the same reason as ``_run_lane``'s
        default above (see that function's docstring for the full FLOOR/
        CEILING derivation, canonically stated in
        ``test_offline_lane_integration.py``'s own ``_run_lane``): the
        ``_run_lane`` task created just before this call races the SAME
        clock, and this call is entered only after
        ``OfflineLaneWorker._run_once`` has already done its own real-git
        work (``get_main_sha`` plus a persistent-worktree reset — 3-5
        subprocess spawns) — a tighter bound here would fire before
        ``_run_lane``'s ever could. Same safety property: widening can never
        make a broken staging pass, it only lengthens how long a genuinely
        wedged runner takes to fail.
        """
        await asyncio.wait_for(self._entered.wait(), timeout=timeout)


# ---------------------------------------------------------------------------
# The red-path rows' stub infra runner (task 6861)
# ---------------------------------------------------------------------------

_X = 'infra_test_x.sh'
_Y = 'infra_test_y.sh'

# One `RESULT:` line per file in the control directory, in name order; the
# file's word is the verdict: `red`, `green`, `skip`, or `flaky` (red on the
# pass's first run, green after it: the marker it leaves in the worktree is
# untracked, so the next worktree reset removes it).
_INFRA_STUB = '''\
#!/bin/sh
control='{control}'
rc=0
for verdict_file in "$control"/*; do
  name=$(basename "$verdict_file")
  verdict=$(cat "$verdict_file")
  if [ "$verdict" = flaky ]; then
    if [ -e ".$name.ran" ]; then verdict=green; else verdict=red; touch ".$name.ran"; fi
  fi
  case "$verdict" in
    red) echo "RESULT: FAIL ($name)"; rc=1 ;;
    green) echo "RESULT: PASS ($name)" ;;
    skip) echo "RESULT: SKIP ($name)" ;;
  esac
done
exit $rc
'''


async def _commit_infra_stub(repo: Path, control: Path) -> None:
    """Commit the stub ``tests/infra/run_all.sh`` reading *control*: one ``git add`` and one ``git commit``.

    Committed, so every ``_offline-deep`` reset carries it at
    ``<wt>/tests/infra/run_all.sh`` for the worker's DEFAULT infra seam to run.
    """
    script = repo / RUN_ALL_INFRA_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_INFRA_STUB.format(control=control))
    script.chmod(0o755)
    await _run(['git', 'add', str(script)], cwd=repo)
    await _run(['git', 'commit', '-m', 'add stub tests/infra/run_all.sh'], cwd=repo)


@dataclass(frozen=True)
class _RedLane:
    """A worker whose infra leg runs the stub, and the task store, queue and clock it files through."""

    harness: Harness
    project_root: Path
    worker: OfflineLaneWorker
    client: FakeLedgerClient
    queue: EscalationQueue
    clock: Clock
    control: Path

    def record(self) -> CommandRecord:
        return EpisodeStore(self.project_root, clock=self.clock).load(LEGACY_INFRA_COMMAND)

    def episode(self, node: str = _X) -> Episode:
        return self.record().episodes[node]

    def unlanded(self) -> list[Escalation]:
        """The pending D11 records on the infra leg's sentinel id."""
        sentinel = sentinel_task_id(LEGACY_INFRA_COMMAND)
        return [
            escalation for escalation in self.queue.get_pending()
            if escalation.task_id == sentinel and escalation.category == UNLANDED_RED_CATEGORY
        ]


async def _red_infra_lane(harness: Harness, git_ops: GitOps, tmp_path: Path) -> _RedLane:
    """A lane over the committed stub: ``_Y`` stays green, each pass sets ``_X``."""
    control = tmp_path / 'control'
    control.mkdir()
    (control / _Y).write_text('green')
    await _commit_infra_stub(git_ops.project_root, control)
    clock = Clock()
    client = FakeLedgerClient([])
    queue = EscalationQueue(tmp_path / 'escalations')
    worker = _build_infra_worker(git_ops, tmp_path, task_client=client, escalation_queue=queue, now=clock)
    _wire_lane(harness, worker)
    return _RedLane(harness, git_ops.project_root, worker, client, queue, clock, control)


@dataclass(frozen=True)
class _PassBlocks:
    """One pass's id and the outcome and effect blocks its run record keeps."""

    pass_id: str
    outcome: dict[str, Any] | None
    effect: dict[str, Any]


async def _drive_infra_pass(lane: _RedLane, verdict: str, *, minutes: float = 30) -> _PassBlocks:
    """One lane pass *minutes* after the last, with ``_X``'s verdict set to *verdict*; returns its infra pass blocks.

    The pass is a real advance and the real on_merge_landed fan-out
    (:func:`_drive_advance`) followed by one full ``run()`` pass
    (:func:`_run_one_lane_pass`), so a caller composing several carries its own
    ``@pytest.mark.timeout`` (enforced by
    ``test_every_composing_caller_carries_a_timeout_override``).
    """
    (lane.control / _X).write_text(verdict)
    lane.clock.advance(minutes=minutes)
    await _drive_advance(lane.harness, lane.project_root)
    await _run_one_lane_pass(lane.worker)
    pass_id = lane.record().last_pass_id
    assert pass_id is not None, 'the engine records every pass it observes'
    record = lane.worker.records.load_pass(LEGACY_INFRA_COMMAND, pass_id)
    assert record is not None and record.effect is not None, 'run_pass writes the effect last'
    return _PassBlocks(record.pass_id, record.outcome, record.effect)


def _intersected(*, reproduced=(), cleared=(), inconclusive=()) -> dict[str, Any]:
    """The infra confirm's outcome block, as the pass record keeps it."""
    return {
        'arm': 'confirmed',
        'mode': 'intersected',
        'reproduced': list(reproduced),
        'cleared': list(cleared),
        'inconclusive': list(inconclusive),
    }


# ---------------------------------------------------------------------------
# Timeout-bound invariants (task 4030)
# ---------------------------------------------------------------------------

# This module's own copy of `test_offline_lane_integration.py`'s helper-name
# set (see that module for the full rationale, including task 4203's review
# remediation widening its membership rule from "drives at least one
# `_run_lane`-bounded pass" to "drives bounded waits its own body does not
# spell out"). Two members are substituted for this module's own spellings:
# `_drive_infra_pass` replaces `_drive_lane_pass` (this module's
# per-pass driver), and `_assert_infra_never_a_gate` replaces
# `_assert_never_a_gate` (this module's copy of the never-a-gate assertion,
# carrying the same `_NOTE_OFFLINE_LANE_BOUND_SECS` +
# `_NOTE_MERGE_ALL_BOUND_SECS` = 15.5s of bounded waits without any lane pass
# at all).
_LANE_PASS_COMPOSING_HELPER_NAMES = frozenset(
    {
        '_run_lane',
        '_run_one_lane_pass',
        '_drive_infra_pass',
        '_assert_infra_never_a_gate',
    },
)


def _lane_pass_composing_callers(module_globals: dict) -> set[str]:
    """Names of module-level `test_*` functions whose OWN source calls one of
    `_LANE_PASS_COMPOSING_HELPER_NAMES` at least once.

    This module's own copy of ``test_offline_lane_integration.py``'s helper
    of the same name — see that sibling module for the full rationale on why
    this is AST-based (avoids the false-positive risk of a text/regex scan
    matching a prose mention) and why it deliberately does not resolve call
    arguments (a coverage NET, not a worst-case calculator).
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

    This module's own copy of ``test_offline_lane_integration.py``'s
    invariant of the same name (these two modules are deliberate lock-step
    siblings — this module's own `_run_lane` docstring says it was "Adapted
    verbatim from test_offline_lane_integration.py"). See that sibling test
    for the full rationale on: why coverage is DISCOVERED via
    `_lane_pass_composing_callers` — over every member of
    `_LANE_PASS_COMPOSING_HELPER_NAMES`, deliberately named there rather than
    re-spelled here so widening that set cannot strand a stale hand-typed
    list in this docstring or in the `undeclared` assertion message below —
    rather than relying solely on a hand-maintained table (a test added
    tomorrow that calls `_drive_infra_pass` three times without a marker must
    fail this test, not pass it silently); why correlation between a marker and a SPECIFIC
    test's value is done via `fn.pytestmark` introspection rather than the
    coarse `_TIMEOUT_MARKER_RE` file-wide regex scan
    `test_lane_lock_leak_guard.py`'s sibling invariant uses; and why a
    stacked-marker tie-break uses `markers[0]`, matching pytest's own
    `get_closest_marker` resolution, not `markers[-1]`.

    `worst_case_secs` maps each composing test FUNCTION OBJECT — never a
    string, so a rename breaks this test loudly instead of silently dropping
    a row — to its worst-case bounded-wait sum, expressed in terms of this
    module's named timeout constants (never bare literals).
    ``test_ib2_infra_run_in_flight_never_gates_merge`` has the identical
    uncovered shape as the sibling module's ``test_b3_never_a_gate``: it
    races a 30s `_run_lane` bound concurrently against `wait_entered` (max,
    not sum), then runs `_assert_infra_never_a_gate`'s 0.5 + 15.0 bounds
    SEQUENTIALLY after it.

    `out_of_bound_spawns` (task 4203) maps the SAME function objects to
    their counted real-git subprocess spawns that happen OUTSIDE any bounded
    wait — the other multiplicand `_orch_helpers.py::required_timeout_secs`
    prices at `MEASURED_SPAWN_LATENCY_SECS` per spawn. This module's own copy of the
    sibling's table of the same name; see that module for the full
    rationale. Every function classified in `worst_case_secs` must also
    appear here (enforced by a key-set equality assertion below).

    `single_pass_exempt` holds the remaining composing test, ib1 — it drives
    exactly one `_run_one_lane_pass` call, so its worst case is a single
    `_LANE_PASS_BOUND_SECS`, the marker-less budget
    `test_lane_bounds_clear_the_measured_floor_and_the_global_ceiling` pins.
    """
    worst_case_secs = {
        test_ib3_a_confirmed_infra_red_is_held_then_a_spaced_reproduction_files_one_task: (
            2 * _LANE_PASS_BOUND_SECS
        ),
        test_ib4_a_persistent_infra_red_under_an_open_task_files_once_and_escalates_once: (
            6 * _LANE_PASS_BOUND_SECS
        ),
        test_ib5_an_infra_red_the_confirm_clears_ends_its_episode_and_files_nothing: 2 * _LANE_PASS_BOUND_SECS,
        test_ib6_a_cancelled_infra_task_escalates_once_and_after_resolution_a_new_task_names_it: (
            6 * _LANE_PASS_BOUND_SECS
        ),
        test_ib2_infra_run_in_flight_never_gates_merge: (
            _LANE_PASS_BOUND_SECS
            + _NOTE_OFFLINE_LANE_BOUND_SECS
            + _NOTE_MERGE_ALL_BOUND_SECS
        ),
        # Task 4203 (review remediation) — this module's copy of the
        # sibling's row for the same test: the only row here whose bounded
        # waits come entirely from `_assert_infra_never_a_gate`, with no
        # `_LANE_PASS_BOUND_SECS` term at all. Discovered by the net only
        # once that helper joined `_LANE_PASS_COMPOSING_HELPER_NAMES`.
        test_out_of_bound_spawn_counts_are_measured_not_asserted: (
            _NOTE_OFFLINE_LANE_BOUND_SECS + _NOTE_MERGE_ALL_BOUND_SECS
        ),
    }
    # Task 4203 — this module's own copy of the sibling's out_of_bound_spawns
    # table (see that module for the full rationale). Each red-path row (task
    # 6861) pays the repo fixture, one stub commit and one _drive_advance per
    # _drive_infra_pass call; ib2 drives one direct _drive_advance plus
    # _assert_infra_never_a_gate's own unbounded advance-and-notify work —
    # MEASURED (reviewer amendment, task 4203) at
    # _SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE = 7, not the earlier hand
    # approximation of one _drive_advance-equivalent round (4), which
    # undercounted by 3 (see that constant's comment for the breakdown).
    two_pass_row = _SPAWNS_PER_REPO_FIXTURE + _SPAWNS_PER_FIXTURE_COMMIT + 2 * _SPAWNS_PER_DRIVE_ADVANCE
    six_pass_row = _SPAWNS_PER_REPO_FIXTURE + _SPAWNS_PER_FIXTURE_COMMIT + 6 * _SPAWNS_PER_DRIVE_ADVANCE
    out_of_bound_spawns = {
        test_ib3_a_confirmed_infra_red_is_held_then_a_spaced_reproduction_files_one_task: two_pass_row,
        test_ib4_a_persistent_infra_red_under_an_open_task_files_once_and_escalates_once: six_pass_row,
        test_ib5_an_infra_red_the_confirm_clears_ends_its_episode_and_files_nothing: two_pass_row,
        test_ib6_a_cancelled_infra_task_escalates_once_and_after_resolution_a_new_task_names_it: six_pass_row,
        test_ib2_infra_run_in_flight_never_gates_merge: (
            _SPAWNS_PER_REPO_FIXTURE
            + _SPAWNS_PER_DRIVE_ADVANCE
            + _SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE
        ),
        # Task 4203 (review remediation) — the ONLY row in either module's
        # table that pays `_SPAWNS_PER_REPO_FIXTURE` TWICE: once for the
        # `repo` fixture every test here transitively pulls, and once more
        # for the fresh `_setup_repo` this test deliberately runs on a clean
        # tmp_path in order to MEASURE that very constant (it measures
        # `_SPAWNS_PER_FIXTURE_COMMIT` on that fresh repo too). Its
        # `_build_infra_worker` / `_wire_lane` setup contributes nothing
        # further — measured to be spawn-free inside that test, not assumed.
        test_out_of_bound_spawn_counts_are_measured_not_asserted: (
            2 * _SPAWNS_PER_REPO_FIXTURE
            + _SPAWNS_PER_FIXTURE_COMMIT
            + _SPAWNS_PER_DRIVE_ADVANCE
            + _SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE
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
        test_ib1_advance_triggers_from_head_infra_sub_run,
    }

    discovered = _lane_pass_composing_callers(globals())
    classified = {fn.__name__ for fn in worst_case_secs} | {
        fn.__name__ for fn in single_pass_exempt
    }
    undeclared = discovered - classified
    assert not undeclared, (
        f'{sorted(undeclared)} call a bounded-wait-composing helper '
        f'({" / ".join(sorted(_LANE_PASS_COMPOSING_HELPER_NAMES))}) but are '
        f'classified in '
        f'neither worst_case_secs (composes past a single pass; needs its '
        f'own @pytest.mark.timeout override) nor single_pass_exempt (proven '
        f'to never exceed one bounded pass), in '
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
    adapted here for this module's own lane-pass bound. This module's own
    copy of ``test_offline_lane_integration.py``'s invariant of the same
    name (these two modules are deliberate lock-step siblings).

    FLOOR — `_LANE_PASS_BOUND_SECS` must be at least
    `_SPAWNS_PER_LANE_PASS_WORST_CASE`x `MEASURED_SPAWN_LATENCY_SECS` (task
    3451's measured worst-case happy-path subprocess spawn latency: n=3,
    2.13/3.10/4.71s, load-per-core 6.6). The multiplier is DERIVED, not
    guessed, and the measurement lives IN THIS MODULE:
    `test_ib3_a_confirmed_infra_red_is_held_then_a_spaced_reproduction_files_one_task`
    below is the measured worst-case count of subprocess spawns inside a
    single `_run_lane` window — `_run_once`'s 3 git spawns (`get_main_sha`,
    `worktree list`, `worktree add`) plus the real infra seam spawning
    `run_all.sh` twice (the initial run and the isolated confirmation
    re-run). Later passes swap `worktree add` for `reset --hard` + `clean
    -xfd` — 4 spawns, still under 5. The sibling numeric module
    (``test_offline_lane_integration.py``) asserts this SAME relation AND
    additionally imports this module's `_LANE_PASS_BOUND_SECS` to assert
    equality against its own copy — see that module's version of this test
    for the check itself — so the two modules' `_LANE_PASS_BOUND_SECS`
    copies cannot silently drift apart: the floor/ceiling relation alone
    cannot catch that, since both independently permit the same range.

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
        f'test_ib3_a_confirmed_infra_red_is_held_then_a_spaced_reproduction_files_one_task below: 3 '
        f'git spawns from _run_once plus 2 real infra run_all.sh spawns).'
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
    # for @pytest.mark.timeout OVERRIDES. This module's own copy of the
    # sibling numeric module's reconciliation assertion — see that module's
    # version of this test for the full rationale (not restated here, to
    # avoid recreating the near-verbatim docstring copies task 4030
    # removed).
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


@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)  # task 4203 review remediation: this
# module's twin of the sibling's test of the same name, its requirement derived independently
# from THIS module's own constants. The only marker-carrying test here composing NO _run_lane
# pass -- its whole bounded term is _assert_infra_never_a_gate's
# _NOTE_OFFLINE_LANE_BOUND_SECS + _NOTE_MERGE_ALL_BOUND_SECS = 15.5s. Out-of-bound term
# = 23 spawns: 2 x _SPAWNS_PER_REPO_FIXTURE (the repo fixture AND the fresh _setup_repo
# measured against below -- the only doubled fixture term in either module), plus
# _SPAWNS_PER_FIXTURE_COMMIT (2), _SPAWNS_PER_DRIVE_ADVANCE (4) and
# _SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE (7); 23 x MEASURED_SPAWN_LATENCY_SECS = 108.33s,
# required = 123.83s. The verify budget clears that
# requirement, and the value is ENFORCED by this test's row in
# test_every_composing_caller_carries_a_timeout_override.
@pytest.mark.asyncio
async def test_out_of_bound_spawn_counts_are_measured_not_asserted(
    harness: Harness, git_ops: GitOps, repo: Path, tmp_path: Path,
) -> None:
    """`_SPAWNS_PER_DRIVE_ADVANCE` / `_SPAWNS_PER_REPO_FIXTURE` are the
    multiplicands `required_timeout_secs` (task 4203) prices at
    `MEASURED_SPAWN_LATENCY_SECS` per spawn — this test MEASURES them by
    counting real git subprocess spawns, rather than trusting a
    hand-maintained comment. This module's own copy of the sibling numeric
    module's test of the same name — see that module for the full
    rationale. This module's `_setup_repo` / `_advance_main` / `_drive_advance`
    bodies are byte-identical to the sibling's and this module's `harness`
    fixture stubs `get_merge_diff_files` the same way, but the count itself
    is MEASURED here, independently, per this task's own "measure, don't
    trust" rule — not copied from the sibling's result.

    REVIEWER AMENDMENT (task 4203) also measures
    `_SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE` — this module's own copy of the
    sibling's third multiplicand, needed only by
    `test_ib2_infra_run_in_flight_never_gates_merge`'s `out_of_bound_spawns`
    row. Measured under the SAME counting wrapper.

    Deliberately calls `_assert_infra_never_a_gate` WITHOUT first driving it
    through `test_ib2_infra_run_in_flight_never_gates_merge`'s full
    held-in-flight `_run_lane` scaffolding — this module's own copy accepts
    no `runner` argument at all (unlike the sibling's `_assert_never_a_gate`),
    so there is nothing to hold in flight for the measurement to skip.
    Composing a real pass here would only add a further
    `_LANE_PASS_BOUND_SECS` of bounded wait plus its in-window spawns, for
    zero additional measurement value.

    (An earlier revision justified that choice differently — a real pass
    "would also wrongly register THIS test in `_lane_pass_composing_callers`'
    AST net, forcing it into `worst_case_secs` and a `@pytest.mark.timeout`
    override it does not need". That reasoning inverted once the reviewer
    amendment added the `_assert_infra_never_a_gate` measurement: THAT helper
    is itself in `_LANE_PASS_COMPOSING_HELPER_NAMES`, so this test is already
    in the net, already classified in `worst_case_secs`, and does need the
    override it now carries. The decision survives on the reason above; only
    its justification changed.)

    SEAM VERIFICATION: identical to the sibling module's — this module also
    does ``from orchestrator.git_ops import GitOps, _run`` at module scope,
    and ``_setup_repo`` / ``_advance_main`` (defined in THIS module) resolve
    the bare name ``_run`` through THIS module's own globals at call time, a
    reference distinct from ``orchestrator.git_ops._run`` itself that a
    single-target patch would silently miss. Both targets are patched below
    with the same counting wrapper so neither leg is silently missed.

    THIRD LEG: this module's `harness` fixture ALSO replaces
    ``h.git_ops.get_merge_diff_files`` wholesale with an
    ``AsyncMock(return_value=([], None))``, so `_note_merge_all`'s `git
    diff` leg contributes zero spawns here too.

    CARRIES `@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)` — it is NOT marker-less. An earlier
    revision of this paragraph claimed this test had "zero bounded waits" and
    the same real-git shape the marker-less ib1 carries unmarked. The
    reviewer amendment that added the `_assert_infra_never_a_gate`
    measurement above falsified both clauses — and this copy was wrong in one
    further respect the sibling's was not: the footprint it enumerated ("one
    `_drive_advance` plus one fresh `_setup_repo`") omitted the `repo`
    fixture's OWN `_setup_repo`, undercounting by a further
    `_SPAWNS_PER_REPO_FIXTURE`. Actual: 15.5s of bounded waits
    (`_NOTE_OFFLINE_LANE_BOUND_SECS` + `_NOTE_MERGE_ALL_BOUND_SECS`) and 21
    out-of-bound spawns (23 since task 6861 added the stub commit), ~2.6x
    the real-git work of the ib1 test (9: the `repo` fixture plus one
    `_drive_advance`), for a `required_timeout_secs` of 123.83s. See the
    marker comment above for the
    derivation; the value is enforced by
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
        f'that calls _drive_advance, directly or via _drive_infra_pass.'
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
    control = tmp_path / 'control_for_spawn_count'
    control.mkdir()
    with patch('orchestrator.git_ops._run', side_effect=_counting_run), \
         patch(f'{__name__}._run', side_effect=_counting_run):
        await _commit_infra_stub(fresh_repo, control)

    assert len(calls) == _SPAWNS_PER_FIXTURE_COMMIT, (
        f'_commit_infra_stub spawned {len(calls)} real git subprocess(es) '
        f'({calls}), not the pinned _SPAWNS_PER_FIXTURE_COMMIT = '
        f'{_SPAWNS_PER_FIXTURE_COMMIT}. This count is a multiplicand of '
        f'required_timeout_secs — every red-path row commits the stub once, '
        f'so a drift here mis-prices each of them.'
    )

    # Reviewer amendment (task 4203) — measure `_assert_infra_never_a_gate`
    # itself. No held-in-flight `_run_lane` pass is composed here (see this
    # test's docstring), so `_build_infra_worker` is left on its default
    # `infra_runner` — this measurement never triggers a run, so which seam
    # it would have used is immaterial.
    #
    # The lane wiring is MEASURED to be spawn-free rather than assumed (task
    # 4203 review remediation), and measured HERE rather than inherited from
    # the sibling's result: this module's copy constructs
    # `_build_infra_worker` on its DEFAULT `infra_runner`, where the sibling
    # passes an explicit `_ControllableSuiteRunner` — a genuine difference in
    # what gets constructed, so symmetry of the spawn count is exactly the
    # thing that must not be assumed. This test's own `out_of_bound_spawns`
    # row prices the wiring at zero.
    calls.clear()
    with patch('orchestrator.git_ops._run', side_effect=_counting_run), \
         patch(f'{__name__}._run', side_effect=_counting_run):
        queue = EscalationQueue(tmp_path / 'escalations')
        worker = _build_infra_worker(git_ops, tmp_path, escalation_queue=queue)
        _wire_lane(harness, worker)

    assert calls == [], (
        f'constructing _build_infra_worker (on its default infra_runner) / '
        f'_wire_lane spawned {len(calls)} real git subprocess(es) ({calls}), '
        f"but this test's out_of_bound_spawns row in "
        f'test_every_composing_caller_carries_a_timeout_override prices the '
        f'lane wiring at zero. Pin the measured count in a named constant '
        f'and add it to that row rather than letting it be absorbed '
        f"silently — an unpriced spawn under-sizes this test's own "
        f'@pytest.mark.timeout override.'
    )

    with patch('orchestrator.git_ops._run', side_effect=_counting_run), \
         patch(f'{__name__}._run', side_effect=_counting_run):
        await _assert_infra_never_a_gate(harness, worker, repo, git_ops, queue)

    assert len(calls) == _SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE, (
        f'_assert_infra_never_a_gate spawned {len(calls)} real git '
        f'subprocess(es) ({calls}), not the pinned '
        f'_SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE = '
        f'{_SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE}. This count is the second '
        f'out_of_bound_spawns multiplicand '
        f'test_ib2_infra_run_in_flight_never_gates_merge needs beyond its '
        f'own _drive_advance call.'
    )


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
    repo: Path,
    git_ops: GitOps,
    escalation_queue: EscalationQueue,
) -> None:
    """Assert the never-a-gate invariant (C7/§6) while an INFRA run is held in-flight.

    Called with the FIRST infra pass already confirmed in-flight (the
    caller awaited ``runner.wait_entered()``) and released only after this
    returns, so the bounded ``wait_for`` below proves the synchronous
    on_merge_landed fan-out never blocks on it.

    Adapts ``test_offline_lane_integration.py``'s ``_assert_never_a_gate``
    for the infra seam — same four assertions, held on the INFRA sub-run
    in-flight rather than the numeric one:

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
    # infra run); widening it would weaken what it's testing.
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
        'a landed advance during an in-flight infra run must arm a coalesced re-run'
    )
    assert harness.get_merge_halt_status() == {'wired': False}
    assert escalation_queue.get_pending() == []

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
    assert escalation_queue.get_pending() == []


@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)  # task 4030: NOT a _drive_infra_pass
# chainer, which is why the task-3832 review's marker sweep missed it -- but it composes
# anyway: a 30s _run_lane bound raced concurrently by wait_entered (max, not sum), then
# _assert_infra_never_a_gate's 0.5 + 15.0 SEQUENTIALLY after it = 45.5s bounded, on top of
# unbounded real-git spawns: repo init (_SPAWNS_PER_REPO_FIXTURE=5), one _drive_advance (4),
# and _assert_infra_never_a_gate's own get_main_sha + two _advance_main rounds
# (_SPAWNS_PER_ASSERT_INFRA_NEVER_A_GATE=7, task 4203 reviewer amendment -- MEASURED,
# supersedes an earlier "two _drive_advance/_advance_main rounds" approximation) = 16 spawns
# x 4.71s = 75.36s, required = 120.86s, comfortably under this marker.
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
    queue = EscalationQueue(tmp_path / 'escalations')
    worker = _build_infra_worker(git_ops, tmp_path, infra_runner=runner, escalation_queue=queue)
    _wire_lane(harness, worker)

    await _drive_advance(harness, repo)
    lane_task = asyncio.create_task(_run_lane(worker, expected_passes=1))
    await runner.wait_entered()

    await _assert_infra_never_a_gate(harness, worker, repo, git_ops, queue)

    runner.release()
    await lane_task


# ---------------------------------------------------------------------------
# IB3 — a confirmed infra red is held, then a spaced reproduction files one task
# ---------------------------------------------------------------------------


# Two _drive_infra_pass calls: 2 x _LANE_PASS_BOUND_SECS = 60s bounded, plus the repo fixture,
# the stub commit and two _drive_advance rounds out of bound (5 + 2 + 2 x 4 = 15 spawns x
# 4.71s = 70.65s); required = 130.65s, enforced by this test's rows in
# test_every_composing_caller_carries_a_timeout_override.
@pytest.mark.timeout(180)
@pytest.mark.asyncio
async def test_ib3_a_confirmed_infra_red_is_held_then_a_spaced_reproduction_files_one_task(
    harness, git_ops, repo, tmp_path,
):
    """IB3 (red-path §6 scenarios 3 and 4, infra leg) — a test the stub FAILs
    on detection and again on the confirm files nothing on its first pass and
    holds one counted pass.  Reproduced 30 minutes later, it files exactly
    one NORMAL high-priority task naming only that test (the PASS line's test
    is never filed), committed out of planning, whose id the episode and the
    pass's effect carry.  No escalation; the merge queue is untouched.

    Negative control: a broken ``RESULT:`` parse, an infra red routed past
    the confirm, or a filing on the first pass makes this RED.
    """
    lane = await _red_infra_lane(harness, git_ops, tmp_path)

    first = await _drive_infra_pass(lane, 'red')

    assert first.outcome == _intersected(reproduced=[_X])
    assert first.effect['tasks_filed'] == []
    assert lane.client.submitted == []
    assert lane.episode().counted_passes == (first.pass_id,)

    second = await _drive_infra_pass(lane, 'red')

    (arguments,) = lane.client.submitted
    episode = lane.episode()
    assert episode.covering_task_id is not None
    assert episode.planning_committed
    assert second.effect['tasks_filed'] == [episode.covering_task_id]
    assert arguments['metadata']['x_offline_lane']['nodes'] == [_X]
    assert arguments['metadata']['merge_lane'] == 'normal', (
        'an infra lane fix task routes through the standard TDD→PR→merge '
        'gate, never the b3_gate red-main fix-forward path'
    )
    assert arguments['priority'] == 'high'
    assert f'Reproduce: {RUN_ALL_INFRA_SCRIPT} --scope {INFRA_SCOPE}' in arguments['description']
    assert lane.queue.get_pending() == []
    assert harness.get_merge_halt_status() == {'wired': False}


# ---------------------------------------------------------------------------
# IB4 — a persistent infra red under an open task: one task, one escalation
# ---------------------------------------------------------------------------


# Six _drive_infra_pass calls: 6 x _LANE_PASS_BOUND_SECS = 180s bounded, plus 5 + 2 + 6 x 4 =
# 31 out-of-bound spawns x 4.71s = 146.01s; required = 326.01s, so 360 (the next multiple of
# the 60s grid), enforced by this test's rows in
# test_every_composing_caller_carries_a_timeout_override.
@pytest.mark.timeout(360)
@pytest.mark.asyncio
async def test_ib4_a_persistent_infra_red_under_an_open_task_files_once_and_escalates_once(
    harness, git_ops, repo, tmp_path,
):
    """IB4 (red-path §6 scenarios 13 and 14, infra leg) — once a task covers the
    test, a PASS line for it and then two red passes file nothing new, and
    staying red files no second task: no escalation 12 hours after filing,
    exactly one once a counted pass lands 24 hours after filing, none later.

    Negative control: deduplicating on anything but the covering task's
    binding, or escalating per pass, makes this RED.
    """
    lane = await _red_infra_lane(harness, git_ops, tmp_path)
    await _drive_infra_pass(lane, 'red')
    await _drive_infra_pass(lane, 'red')
    task_id = lane.episode().covering_task_id
    assert task_id is not None

    green = await _drive_infra_pass(lane, 'green')
    await _drive_infra_pass(lane, 'red')
    await _drive_infra_pass(lane, 'red', minutes=11 * 60)

    assert [(ended['node'], ended['reason']) for ended in green.effect['episodes_ended']] == [(_X, 'passed')]
    assert len(lane.client.submitted) == 1
    assert lane.episode().covering_task_id == task_id
    assert lane.unlanded() == []

    await _drive_infra_pass(lane, 'red', minutes=12 * 60)

    (record,) = lane.unlanded()
    assert task_id in record.summary
    assert (record.level, record.severity, record.agent_role) == (1, 'blocking', 'orchestrator-offline-lane')

    await _drive_infra_pass(lane, 'red')

    assert lane.unlanded() == [record]
    assert len(lane.client.submitted) == 1
    assert harness.get_merge_halt_status() == {'wired': False}


# ---------------------------------------------------------------------------
# IB5 — an infra red whose confirm prints PASS for the test is cleared
# ---------------------------------------------------------------------------


# Two _drive_infra_pass calls; the same 130.65s requirement as IB3.
@pytest.mark.timeout(180)
@pytest.mark.asyncio
async def test_ib5_an_infra_red_the_confirm_clears_ends_its_episode_and_files_nothing(
    harness, git_ops, repo, tmp_path,
):
    """IB5 (red-path §6 scenario 6, infra leg) — a held test the stub FAILs on
    detection and PASSes on the confirm is cleared by that PASS line, its
    per-test evidence: the episode ends, and no fix task or escalation is
    raised.

    Negative control: filing on an unconfirmed (flaky) infra fail, or
    clearing without per-test evidence, makes this RED.
    """
    lane = await _red_infra_lane(harness, git_ops, tmp_path)
    await _drive_infra_pass(lane, 'red')

    flaked = await _drive_infra_pass(lane, 'flaky')

    assert flaked.outcome == _intersected(cleared=[_X])
    assert [(ended['node'], ended['reason']) for ended in flaked.effect['episodes_ended']] == [(_X, 'cleared')]
    assert not lane.episode().live
    assert lane.client.submitted == []
    assert lane.queue.get_pending() == []


# ---------------------------------------------------------------------------
# IB6 — a cancelled covering task: one escalation, then a refile once resolved
# ---------------------------------------------------------------------------


# Six _drive_infra_pass calls; the same 326.01s requirement as IB4.
@pytest.mark.timeout(360)
@pytest.mark.asyncio
async def test_ib6_a_cancelled_infra_task_escalates_once_and_after_resolution_a_new_task_names_it(
    harness, git_ops, repo, tmp_path,
):
    """IB6 (red-path §6 scenarios 13 and 15, infra leg) — the test stays red
    after its covering task is cancelled: no refile, one escalation once a
    counted pass lands 24 hours after filing.  Once it is resolved the next
    red starts a new episode held at one pass, and a second spaced red files a
    new task naming the cancelled one.

    Negative control: refiling under a cancelled task, or never filing again
    after the escalation is resolved, makes this RED.
    """
    lane = await _red_infra_lane(harness, git_ops, tmp_path)
    await _drive_infra_pass(lane, 'red')
    await _drive_infra_pass(lane, 'red')
    cancelled = lane.episode().covering_task_id
    assert cancelled is not None
    lane.client.set_status(cancelled, 'cancelled')

    await _drive_infra_pass(lane, 'red', minutes=12 * 60)
    assert lane.unlanded() == []
    await _drive_infra_pass(lane, 'red', minutes=12 * 60)

    (record,) = lane.unlanded()
    assert cancelled in record.summary
    assert len(lane.client.submitted) == 1
    lane.queue.resolve(record.id, 'seen; refile when it persists', resolved_by='operator')

    held = await _drive_infra_pass(lane, 'red')
    refiled = await _drive_infra_pass(lane, 'red')

    assert held.effect['episodes_started'] == [_X]
    assert held.effect['tasks_filed'] == []
    assert len(lane.client.submitted) == 2
    assert refiled.effect['tasks_filed'] == [lane.episode().covering_task_id]
    assert lane.client.submitted[-1]['metadata']['x_offline_lane']['prior_task_ids'] == [cancelled]
    assert lane.unlanded() == []
