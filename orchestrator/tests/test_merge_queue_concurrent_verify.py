"""Concurrent-verify tests for γ (task 1735).

This module covers γ-specific tests that require multi-host verify infrastructure.
Tests are grouped per plan step:

  pre-1  SCAFFOLD — shared fixtures/helpers (git repo, gated runner, fake remote)
  step-1 RED      — _run_post_merge_verify(runner=<fake>) + runner=None byte-identical
  step-3 RED      — _inflight/_redispatch/_n_failed/_remerge_occurred attrs + InflightEntry
  step-5 RED      — _run_inflight_verify happy paths (LOCAL + REMOTE lease)
  step-7 RED      — _run_inflight_verify abort-poll (abandon + halt)
  step-9 RED      — _run_inflight_verify RUNNER_UNAVAILABLE sentinel
  step-11 RED     — _finalize_inflight PASS path (CAS advance, lease release)
  step-13 RED     — _finalize_inflight non-pass paths
  step-15 RED     — single-host serial byte-identical via restructured loop
  step-17 RED     — OVERLAP SIGNAL (two hosts, both verifies enter before either released)
  step-19 RED     — CHAIN-INVALIDATION UNDER OVERLAP
  step-21 RED     — operator-halt aborts all in-flight + RUNNER_UNAVAILABLE quarantine
  step-23 RED     — stop() drains inflight + snapshot() surfaces inflight entries
  step-27 RED     — warm_results threading into shadow compare

NOTE: Individual test classes are added in their respective RED steps.
This file starts with shared scaffolding only (pre-1).
"""

from __future__ import annotations

import ast
import asyncio
import collections
import contextlib
import dataclasses
import inspect
import logging
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeGuard
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _merge_lane_fakes import (
    FakeVerifier,
    VerifyScript,
    fails,
    hangs_until,
    lane_scene_config,
    main_health_probe_spawned,
    raises,
)
from _orch_helpers import (
    MERGE_GATE_BARRIER_TIMEOUT,
    MERGE_RESULT_TIMEOUT,
    # task 4215 moved this constant to _orch_helpers.py, where a runtime
    # `tomllib` read of orchestrator/pyproject.toml now pins it against the
    # real `[tool.pytest.ini_options].timeout`
    # (test_whole_tree_scan_timeout_guard.py).
    PYPROJECT_DEFAULT_TIMEOUT,
    STOP_WORKER_JOIN_TIMEOUT,
    UNDER_LOAD_HEADROOM_FACTOR,
    wait_responsive,
)
from _wait_hygiene_scan import (
    suppressed_result_wait_items,
    unguarded_worker_teardown_items,
)

from orchestrator.config import GitConfig, OrchestratorConfig, VerifyRunnerConfig
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_lane.types import QueuedBranch
from orchestrator.merge_lane.worker import (
    MERGE_WORKER_SHUTDOWN_REASON,
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
    SpeculativeMergeWorker,
)
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import HostAllocator, HostLease

# ---------------------------------------------------------------------------
# (a) Temp-repo GitOps + branch-with-file builders
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
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
    return _make_config_no_runners(git_repo, git_config)


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


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    pre_rebased: bool = False,
    lane: Literal['normal', 'high'] = 'normal',
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop."""
    try:
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    except RuntimeError:
        from _orch_helpers import make_placeholder_future
        future = make_placeholder_future()
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=pre_rebased,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        lane=lane,
    )


def _make_real_item(
    git_ops: GitOps,
    config: OrchestratorConfig,
    task_id: str,
    base_sha: str,
) -> RealMergeItem:
    """Build a dispatch-ready, non-speculative RealMergeItem over a fresh
    MergeRequest whose merge has notionally succeeded -- the single
    construction site for tests that hand a worker an item without running
    a real merge.
    """
    req = _make_request(task_id, f'task/{task_id}', git_ops.project_root, config)
    wt = git_ops.project_root / '.worktrees' / task_id
    return RealMergeItem(
        request=req,
        merge_result=MergeResult(success=True, merge_commit='deadbeef', merge_worktree=wt),
        merge_wt=wt,
        base_sha=base_sha,
        speculative=False,
    )


# THIS FILE CARRIES DEBT under rule `wall-clock-deadline` in
# fused-memory/scripts/check_bare_magicmock_config.py (task 4246), which checks
# the SHAPE of a load-bearing wait: a `MergeRequest.result` future or a
# zero-argument `.wait()` barrier must be awaited through `wait_responsive(...)`
# with a descriptive `label=`, its bound derived from MERGE_RESULT_TIMEOUT rather
# than written as a number. This file is on that rule's shrink-only, opt-out
# `_WALL_CLOCK_DEADLINE_DEBT` baseline: its pre-existing sites are
# grandfathered, but the budget is CHECKED -- add one more offending wait and
# the gate reports the overrun. New load-bearing waits here must therefore use
# `wait_responsive(...)`, and the recorded number may only be lowered as sites
# migrate, never raised.


async def _await_outcome(
    fut: asyncio.Future[MergeOutcome],
    *,
    label: str,
    timeout: float = MERGE_RESULT_TIMEOUT,
) -> MergeOutcome:
    """Await a ``MergeRequest.result`` future, failing LOUDLY -- by name --
    on deadline instead of letting a real timeout masquerade as ``None``
    (task 3477).

    The shape this replaces --
    ``with contextlib.suppress(TimeoutError): outcome = await
    asyncio.wait_for(fut, timeout=N)`` followed by
    ``assert outcome is not None and outcome.status == ...`` -- converts a
    genuine pipeline hang or a missing cascade into a confusing
    ``outcome is None`` assertion failure. That shape is exactly what
    flaked in TestRunnerUnavailableHeadCascade and
    TestCascadeFiresRemoteCancel: under load, the fixed real-time deadline
    was reached before the merge pipeline resolved, TimeoutError was
    swallowed, and the next hard assertion fired on a still-None outcome --
    reporting a timeout as if it were a wrong-value bug.

    The default *timeout* (45s) is ``_orch_helpers.MERGE_RESULT_TIMEOUT``
    (task 2376's shared, never-narrow merge-pipeline result-wait ceiling),
    not a fresh literal -- so callers inherit the same generous,
    load-tolerant budget already established for test_merge_queue.py,
    test_merge_speculation.py and test_concurrent_verify_boundary.py.
    """
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except TimeoutError:
        pytest.fail(
            f'{label}: MergeOutcome never resolved within {timeout}s -- '
            f'the merge pipeline never completed this request. This is a '
            f'real hang or a genuinely missing cascade, NOT a None outcome.'
        )


async def _stop_worker(
    worker: SpeculativeMergeWorker,
    worker_task: asyncio.Task[None],
    *,
    join_timeout: float = STOP_WORKER_JOIN_TIMEOUT,
) -> None:
    """Shut *worker* down and join its run task -- the ONE teardown shape
    shared by this module and every sibling merge-worker test module that
    imports it, so no site can drift or be forgotten.

    ALWAYS call this from a ``finally:`` whose ``try:`` encloses every wait in
    the test body (task 3980 amendment, esc-3980-4). ``wait_responsive``
    gives up by raising ``_pytest.outcomes.Failed``, and an assertion
    mid-body raises too; on the old straight-line shape either one skipped
    ``stop()`` entirely and leaked a live merge worker plus its run task into
    pytest-asyncio teardown. That leak is why one red test used to cascade
    into unrelated failures elsewhere in the session.
    ``_wait_hygiene_scan.py::unguarded_worker_teardown_items`` enforces
    the ``finally:``.

    Safe on the give-up path even when a gate was never released: ``stop()``
    cancels every in-flight verify task rather than awaiting it
    (``orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.stop``),
    so it cannot itself block on an unreleased ``asyncio.Event``.  Whatever
    the lane still has to unwind on the way out keeps seeing the injected
    verifier, which is the lane's own collaborator for its whole lifetime
    rather than a binding swapped in for a block.

    The join stays best-effort (``suppress(Exception)``): it asserts nothing,
    and a slow join must not convert a real failure above into a confusing
    second one. It is also why this wait is exempt from the shared
    ``wall-clock-deadline`` rule
    (fused-memory/scripts/check_bare_magicmock_config.py): its target is a
    bare Name, not a ``.result`` future or a zero-argument ``.wait()`` barrier, so the
    exemption is structural rather than a listed name. It is still a real
    wait, so ``_wait_budget_audit.py::_call_wait_budget`` bills its
    *join_timeout*.
    """
    await worker.stop()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(worker_task, timeout=join_timeout)


# The shared per-test ceiling for every class whose waits outgrow the ambient
# budget. SHARED, not per-class, so the marks cannot drift apart (task 3477
# found two that had; task 3492 generalized CASCADE_TEST_TIMEOUT into this).
#
# No test's budget is written here or beside its mark:
# test_wait_budget_coverage_guard.py recomputes each one from source on every
# run and checks it against the closest timeout mark, or against the ambient
# budget for an unmarked test -- so leaving a class unmarked is a checked
# decision too.
#
# A literal, pinned by TestHeavyBarrierTimeoutConstant against the measured
# worst case below x UNDER_LOAD_HEADROOM_FACTOR and against
# PYPROJECT_DEFAULT_TIMEOUT (never-narrow).  Derivation:
# plans/pytest-per-test-timeout-measurement-2026-09-17.md.
HEAVY_BARRIER_TEST_TIMEOUT = 540

# Worst per-test wall clock -- setup+call+teardown, what a pytest-timeout mark
# bounds -- of any HEAVY_BARRIER- or capstone-marked test: task 5572, 2026-10-08,
# `-n auto` at loadavg 152 -> 101,
# test_merge_speculation.py::TestSpecLaneAbortReleasesLane::test_operator_halt_releases_spec_lane.
# Corpus: plans/pytest-per-test-timeout-measurement-2026-09-17.md.
_HEAVY_BARRIER_MEASURED_UNDER_LOAD_WORST_CASE = 14.59


# ---------------------------------------------------------------------------
# (d) OrchestratorConfig builders: single-host vs two-host
# ---------------------------------------------------------------------------


def _make_config_no_runners(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    """Single-host OrchestratorConfig (no verify_runners)."""
    return lane_scene_config(git_repo, git_config)


def _make_config_with_runner(
    git_repo: Path,
    git_config: GitConfig,
    runner_name: str = 'laptop',
) -> OrchestratorConfig:
    """Two-host OrchestratorConfig with one enabled verify_runner."""
    runner_cfg = VerifyRunnerConfig(
        name=runner_name,
        ssh_host='h.local',
        git_remote='origin',
        enabled=True,
    )
    return OrchestratorConfig(
        project_root=git_repo,
        git=git_config,
        verify_runners=[runner_cfg],
        verify_host_policy='prefer_local',
    )


# ---------------------------------------------------------------------------
# (b) _gated_runner factory: slow fake verify gated on asyncio.Event
# ---------------------------------------------------------------------------


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


_VERIFY_RESULT_FIELD_NAMES = {f.name for f in dataclasses.fields(VerifyResult)}


def _fake_verify_result(*, passed: bool, **overrides: Any) -> MagicMock:
    """Return a MagicMock(spec=VerifyResult) seeded from a REAL VerifyResult's
    field defaults, with **overrides layered on top (task 3477).

    Why not a bare ``MagicMock(passed=..., summary=..., ...)``: an
    unconfigured attribute READ on a bare MagicMock auto-vivifies a truthy
    child Mock rather than returning a real default. Every one of this
    file's inline VerifyResult-shaped doubles omitted ``cause_hint``, so
    merge_disposition.py's ``_extract_failing_tests_and_candidate_files``
    (:218-221) joins ``verify_result.cause_hint`` and
    ``verify_result.test_output`` via ``str.join`` — which raised
    ``TypeError: sequence item 0: expected str instance, MagicMock found``
    on every FAILING double. ``classify_merge_failure_disposition``
    (:710-719) swallows that into a silent fail-open (WARNING +
    INDETERMINATE), so the affected tests only APPEARED to exercise
    disposition classification.

    Why ``spec=VerifyResult``: it makes a future unknown-attribute READ
    fail loudly (``AttributeError``) instead of auto-vivifying — the
    root-cause fix, not just a ``cause_hint=''`` patch for today's one
    known field. Seeding from ``dataclasses.fields(VerifyResult)`` (rather
    than a hardcoded attr list) means a field added to VerifyResult in
    future is picked up automatically, so this factory does not itself
    drift out of sync the way the file's inline doubles did.

    Why this stays a MagicMock and not a real VerifyResult instance: a real
    dataclass would also make ``.failure_report()`` return a real string
    instead of a Mock. merge_queue.py:703's ``isinstance(s, str)`` filter
    currently discards the Mock; a real string would pass through and
    append a ``## Test Failures`` block to the rendered ``reason`` at
    merge_queue.py:3020/3041/3078 — assertion churn well beyond this
    factory's fidelity-only scope. Seeding a MagicMock from real defaults
    confines the behaviour delta to exactly the fields callers read.

    Deliberately does NOT special-case a ``verify_skipped`` kwarg: that
    field lives on MergeOutcome (merge_types.py:945), not VerifyResult, and
    is never read off a VerifyResult anywhere in src/ — callers migrating
    off the old inline doubles should simply drop it rather than pass it
    through **overrides.

    **overrides are validated against VerifyResult's real field names
    before being applied (task 3477 amend). ``spec=`` alone only makes
    unconfigured *reads* fail loudly; a ``spec=`` (not ``spec_set=``) mock
    still accepts an unknown attribute WRITE via plain ``setattr`` — so a
    typo'd override (or a leftover ``verify_skipped=``) would silently
    produce a double whose real field keeps its seeded default instead of
    raising, which is exactly the class of silent drift this factory
    exists to eliminate.
    """
    if unknown := set(overrides) - _VERIFY_RESULT_FIELD_NAMES:
        raise TypeError(
            f'_fake_verify_result: not VerifyResult fields: {sorted(unknown)}'
        )
    real = _mock_verify_result(passed)
    fake = MagicMock(spec=VerifyResult)
    for f in dataclasses.fields(VerifyResult):
        setattr(fake, f.name, getattr(real, f.name))
    for key, value in overrides.items():
        setattr(fake, key, value)
    return fake


# ---------------------------------------------------------------------------
# Disposition fail-open predicate (task 3980 amendment) — the companion guard
# to _fake_verify_result above, and deliberately colocated with it.
#
# The classifier has TWO fail-open sites, both emitting the same substring on
# the lane's one logger, 'orchestrator.merge_lane':
#   1. merge_lane/disposition.py — the classifier's own internal fail-open.
#   2. merge_lane/worker.py::_classify_disposition_for_outcome — the
#      belt-and-suspenders catch for anything site 1 re-raises past.
# Keying on the shared substring catches both; filtering by any narrower
# signal than that silently excludes one of them.
#
# Lives HERE rather than in test_merge_speculation.py so there is exactly one
# predicate for one production invariant. Task 3980 first wrote a second copy
# over there, and the two had already diverged before the amendment pass: this
# file's older inline version keyed on a longer substring that matched only
# site 1. Two copies of a drift-detector is how the drift gets back in.
_FAIL_OPEN_SUBSTRING = 'degrading to INDETERMINATE (fail-open, I3)'
_FAIL_OPEN_LOGGERS = ('orchestrator.merge_lane',)


def _fail_open_records(
    records: list[logging.LogRecord],
) -> list[logging.LogRecord]:
    """Return the disposition fail-open WARNINGs among *records* (both sites).

    NOT symmetric with "the classifier ran": a classifier that SUCCEEDS emits
    NOTHING at WARNING (it logs only on the degrade paths,
    merge_disposition.py:695 and :711), so an empty return here means "no
    fail-open", never "the classifier was reached". Proving the classifier was
    reached needs a delegating SPY on the callable — see
    test_merge_speculation.py's ``TestLateArrivalFailCascade`` for the live one
    and ``TestDispositionDoubleFidelity`` for the isolated two-sided proof.
    Do not read log silence here as evidence the classifier never ran: a
    successful classification is precisely the silent case.

    Deliberately does NOT match merge_disposition.py:695's *other* degrade
    ('degrading implicated landings to INDETERMINATE (...)'), which is a
    legitimate evidence-absent verdict rather than a swallowed exception.
    """
    return [
        r for r in records
        if r.name in _FAIL_OPEN_LOGGERS and _FAIL_OPEN_SUBSTRING in r.getMessage()
    ]


def _format_fail_open_records(records: list[logging.LogRecord]) -> str:
    """Render fail-open records for an assertion message, naming the underlying
    exception (the actionable part — the WARNING text alone never says WHY)."""
    return '\n'.join(
        f'  - [{r.name}] {r.getMessage()}'
        + (
            f'\n    underlying: {r.exc_info[1]!r}\n'
            + ''.join(traceback.format_exception(*r.exc_info))
            if r.exc_info
            else ''
        )
        for r in records
    )


class _GatedVerifier(FakeVerifier):
    """``FakeVerifier`` plus the two things every gated test here needs.

    * ``entered`` -- the ARRIVAL half of a gate.  A script carries only the
      release half (``hangs_until``, or ``dataclasses.replace(fails(...),
      release=...)``), so a test otherwise has no way to know a verify is
      genuinely in flight before it releases that gate.
      ``entered[task_id]`` -- or the ``None`` catch-all, for a test that
      gates whichever verify arrives first -- is set the moment that task's
      scoped verify starts, ahead of the script's own release wait.
    * ``first`` -- a script governing ONLY the first scoped verify of the run;
      every later one falls through to ``scripts``/``default``.  That is what
      these tests mean by "N's verify fails, and the re-dispatch passes": the
      head's initial attempt is the gated one, and the cascade re-dispatches
      the SAME task as often as its successor, so keying the failure on task
      id instead would re-fail the re-dispatched head.
    """

    def __init__(
        self,
        *,
        entered: Mapping[str | None, asyncio.Event] | None = None,
        first: VerifyScript | None = None,
        **scripting: Any,
    ) -> None:
        super().__init__(**scripting)
        self.entered = dict(entered or {})
        self.first = first

    async def run_scoped(self, *args: Any, **options: Any) -> VerifyResult:
        task_id = options.get('task_id')
        gate = self.entered.get(task_id) or self.entered.get(None)
        if gate is not None:
            gate.set()
        script = self.first if not self.verified else None
        if script is None:
            return await super().run_scoped(*args, **options)
        # Inlined rather than delegated: ``first`` is keyed on call ORDER,
        # which the shared fake's per-task-id dispatch cannot express.
        self.verified.append(task_id)
        if script.release is not None:
            await script.release.wait()
        if script.error is not None:
            raise script.error
        return script.result


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


# ---------------------------------------------------------------------------
# (c) Fake RemoteRunner + two-host HostAllocator injection
# ---------------------------------------------------------------------------


def _make_fake_remote(name: str = 'laptop') -> MagicMock:
    """Build a fake RemoteRunner (MagicMock) with async cancel/probe/verify."""
    fake = MagicMock()
    fake.name = name
    fake.is_local = False
    fake.run_merge_verify = AsyncMock(return_value=_mock_verify_result(True))
    fake.cancel_verify = AsyncMock(return_value=0)  # 0 = clean cancel
    fake.probe_clean = AsyncMock(return_value=True)
    return fake


def _id_liveness_fake_runner(
    gate_release: asyncio.Event,
    gate_entered: asyncio.Event,
    name: str = 'id-live-laptop',
) -> MagicMock:
    """Faithful fake RemoteRunner that models verify_runner.py:799/814.

    ``run_merge_verify``: sets ``live=True`` on entry (after signalling
    *gate_entered*), awaits *gate_release*, then clears ``live=False`` in
    a ``finally``.  This mirrors the real RemoteRunner's
    ``finally: self._inflight_request_id = None`` — so that task cancellation
    (direct or via an enclosing task) clears the id exactly like production.

    ``cancel_verify``: appends ``1`` to ``remote_cancels_while_live`` ONLY
    while ``live is True``, mirroring verify_runner.py:814 returning 0
    (no-op) when ``_inflight_request_id is None``.  Always returns 0.

    Expose ``remote_cancels_while_live`` on the mock for assertions:
      - ``== [1]`` GREEN: _abort_remote_verify fired before task.cancel()
      - ``== []``  RED:   cancel reached a dead id (neutered)
    """
    state: dict[str, bool] = {'live': False}
    remote_cancels_while_live: list[int] = []

    async def _run_merge_verify_side(*args: Any, **kwargs: Any) -> Any:
        state['live'] = True
        gate_entered.set()
        try:
            await gate_release.wait()
            return _mock_verify_result(True)
        finally:
            state['live'] = False

    async def _cancel_verify_side(*args: Any, **kwargs: Any) -> int:
        if state['live']:
            remote_cancels_while_live.append(1)
        return 0

    runner = MagicMock()
    runner.name = name
    runner.is_local = False
    runner.run_merge_verify = AsyncMock(side_effect=_run_merge_verify_side)
    runner.cancel_verify = AsyncMock(side_effect=_cancel_verify_side)
    runner.probe_clean = AsyncMock(return_value=True)
    runner.remote_cancels_while_live = remote_cancels_while_live
    return runner


def _inject_two_host_allocator(
    worker: SpeculativeMergeWorker,
    fake_remote: Any,
    *,
    allocator_cls: type[HostAllocator] = HostAllocator,
) -> HostAllocator:
    """Inject a two-host HostAllocator (local + fake_remote) onto a worker.

    *allocator_cls* lets a test substitute a HostAllocator subclass that
    models a concurrent acquirer.  Returns the allocator so callers can
    introspect slot state or verify calls.
    """
    return _inject_host_allocator(worker, fake_remote, allocator_cls=allocator_cls)


def _inject_host_allocator(
    worker: SpeculativeMergeWorker,
    *fake_remotes: Any,
    allocator_cls: type[HostAllocator] = HostAllocator,
) -> HostAllocator:
    """Inject a HostAllocator over local plus every one of *fake_remotes*."""
    allocator = allocator_cls(list(fake_remotes), quarantine=worker._runner_quarantine)
    worker._host_allocator = allocator
    return allocator


# ---------------------------------------------------------------------------
# step-1 RED: _run_post_merge_verify(runner=<fake>) wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunPostMergeVerifyRunnerParam:
    """_run_post_merge_verify(runner=<fake>) dispatches on the injected runner;
    runner=None (default) stays LOCAL-ONLY byte-identical.

    RED until step-2 GREEN adds the runner= keyword arg.
    """

    def _make_git_ops_mock(self) -> MagicMock:
        mock = MagicMock()
        mock.get_main_sha = AsyncMock(return_value='main-sha')
        mock.get_free_disk_bytes = AsyncMock(return_value=100 * 1024 ** 3)
        mock.cleanup_merge_worktree = AsyncMock()
        mock.create_ephemeral_verify_worktree = AsyncMock(return_value='/repo/_throwaway')
        return mock

    async def test_remote_runner_dispatched_when_runner_provided(self, tmp_path: Path) -> None:
        """runner=<fake remote> → pool is [runner], fake.run_merge_verify called."""
        from orchestrator.merge_lane.worker import _run_post_merge_verify

        fake_remote = _make_fake_remote('laptop')
        config = OrchestratorConfig(git=GitConfig(main_branch='main'))
        req = _make_request('t1', 'task/t1', tmp_path, config)
        git_ops = self._make_git_ops_mock()

        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha='abc123',
            runner=fake_remote,  # RED: this param doesn't exist yet
        )

        assert outcome is None  # verify passed
        fake_remote.run_merge_verify.assert_called_once()

    async def test_runner_none_uses_local_runner_byte_identical(self, tmp_path: Path) -> None:
        """runner=None (default) → LocalRunner pool, byte-identical to today."""
        from orchestrator.merge_lane.worker import _run_post_merge_verify

        config = OrchestratorConfig(git=GitConfig(main_branch='main'))
        req = _make_request('t2', 'task/t2', tmp_path, config)
        git_ops = self._make_git_ops_mock()

        outcome = await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha='abc123',
            runner=None,  # RED: this param doesn't exist yet
            verifier=FakeVerifier(),
        )

        assert outcome is None

    async def test_merge_verify_event_carries_runner_name_when_remote(self, tmp_path: Path) -> None:
        """merge_verify event runner field == injected runner's name, not 'local'."""
        from orchestrator.event_store import EventStore
        from orchestrator.merge_lane.worker import _run_post_merge_verify

        fake_remote = _make_fake_remote('laptop')
        config = OrchestratorConfig(git=GitConfig(main_branch='main'))
        req = _make_request('t3', 'task/t3', tmp_path, config)
        git_ops = self._make_git_ops_mock()

        emitted: list[dict] = []

        class _FakeEventStore(EventStore):
            def __init__(self) -> None:
                object.__init__(self)

            def emit(self, event_type, *, task_id=None, phase=None, data=None, **kw):  # type: ignore[override]
                emitted.append({'event_type': event_type, 'data': data or {}})

        await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha='abc123',
            runner=fake_remote,   # RED: this param doesn't exist yet
            event_store=_FakeEventStore(),
        )

        merge_verify_events = [
            e for e in emitted
            if hasattr(e['event_type'], 'value') and e['event_type'].value == 'merge_verify'
        ]
        assert len(merge_verify_events) >= 1
        assert merge_verify_events[0]['data']['runner'] == 'laptop'


# ---------------------------------------------------------------------------
# step-3 RED: InflightEntry dataclass + new instance attrs
# ---------------------------------------------------------------------------


class TestInflightStateAttrs:
    """SpeculativeMergeWorker exposes _inflight/_redispatch/_n_failed/_remerge_occurred.
    InflightEntry dataclass is importable from merge_queue.

    RED until step-4 GREEN adds the dataclass and initialises the attrs.
    """

    def _make_worker(self, git_ops: GitOps) -> SpeculativeMergeWorker:
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        return SpeculativeMergeWorker(git_ops, q)

    def test_inflight_is_empty_deque(self, git_ops: GitOps) -> None:
        """_inflight starts as an empty collections.deque."""
        import collections
        worker = self._make_worker(git_ops)
        assert isinstance(worker._inflight, collections.deque)
        assert len(worker._inflight) == 0

    def test_redispatch_is_empty_deque(self, git_ops: GitOps) -> None:
        """_redispatch starts as an empty collections.deque."""
        import collections
        worker = self._make_worker(git_ops)
        assert isinstance(worker._redispatch, collections.deque)
        assert len(worker._redispatch) == 0

    def test_n_failed_is_false(self, git_ops: GitOps) -> None:
        """_n_failed starts False."""
        worker = self._make_worker(git_ops)
        assert worker._n_failed is False

    def test_remerge_occurred_is_false(self, git_ops: GitOps) -> None:
        """_remerge_occurred starts False."""
        worker = self._make_worker(git_ops)
        assert worker._remerge_occurred is False

    def test_inflight_entry_importable(self) -> None:
        """InflightEntry is importable from merge_queue."""
        from orchestrator.merge_lane.worker import InflightEntry
        assert isinstance(InflightEntry, type)


# ---------------------------------------------------------------------------
# step-5 RED: _run_inflight_verify happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunInflightVerifyHappyPath:
    """_run_inflight_verify(item, lease) happy paths: LOCAL lease and REMOTE lease.

    RED until step-6 GREEN adds the method.
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        """Helper: create a branch, merge it to main, return (req, item)."""
        from orchestrator.merge_lane.worker import RealMergeItem
        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_running_loop()
        req = MergeRequest(
            task_id=branch,
            branch=QueuedBranch.parse(branch, config.git.branch_prefix),
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        # A successful merge always has a worktree; RealMergeItem.merge_wt is
        # required non-None (task ο), so narrow explicitly (mirrors merge_queue.py).
        assert merge_result.merge_worktree is not None
        base_sha = await git_ops.get_main_sha()
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
        )
        return req, item

    async def test_local_lease_increments_attempt_count(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """LOCAL lease: _verify_attempt_count incremented by one."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(git_ops, config, 'inv-local-a', 'fa.py', 'a=1\n')
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())

        fake_local = _make_fake_remote('local-fake')
        fake_local.is_local = True
        lease = HostLease(name='local', runner=fake_local, is_local=True)

        count_before = worker._verify_attempt_count
        result = await worker._run_inflight_verify(item, lease)  # RED: method doesn't exist

        assert worker._verify_attempt_count == count_before + 1
        assert result.outcome is None  # pass

    async def test_local_lease_verify_phase_set_to_verifying(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """LOCAL lease: _verify_phase transitions to 'verifying' during the call."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(git_ops, config, 'inv-local-b', 'fb.py', 'b=2\n')
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())

        fake_local = _make_fake_remote('local-fake2')
        fake_local.is_local = True
        lease = HostLease(name='local', runner=fake_local, is_local=True)


        result = await worker._run_inflight_verify(item, lease)

        # After completion result.outcome is None (pass)
        assert result.outcome is None
        assert result.status is None or result.status == 'done'

    async def test_remote_lease_no_warm_swap(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """REMOTE lease: _verify_attempt_count NOT incremented (no warm-swap),
        lease.runner.run_merge_verify called, outcome=None on pass.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(git_ops, config, 'inv-remote-a', 'fc.py', 'c=3\n')
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        fake_remote = _make_fake_remote('laptop')
        lease = HostLease(name='laptop', runner=fake_remote, is_local=False)

        count_before = worker._verify_attempt_count
        result = await worker._run_inflight_verify(item, lease)  # RED: method doesn't exist

        # REMOTE path: warm-swap NOT run → attempt count unchanged
        assert worker._verify_attempt_count == count_before
        # runner.run_merge_verify called (remote dispatch path)
        fake_remote.run_merge_verify.assert_called_once()
        # outcome is None (pass)
        assert result.outcome is None

    async def test_run_inflight_verify_returns_merge_wt(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """_run_inflight_verify returns result with non-None merge_wt."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(git_ops, config, 'inv-wt', 'fd.py', 'd=4\n')
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        fake_remote = _make_fake_remote('laptop2')
        lease = HostLease(name='laptop2', runner=fake_remote, is_local=False)

        result = await worker._run_inflight_verify(item, lease)

        assert result.merge_wt is not None


# ---------------------------------------------------------------------------
# step-7 RED: _run_inflight_verify abort-poll (abandon + operator halt)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunInflightVerifyAbortPoll:
    """_run_inflight_verify abort-poll: abandon → DROPPED, halt → REQUEUED.

    RED until step-8 GREEN ports the abort-poll loop into _run_inflight_verify.
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        """Create a merged SpeculativeItem on the given branch."""
        from orchestrator.merge_lane.worker import RealMergeItem

        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_running_loop()
        req = MergeRequest(
            task_id=branch,
            branch=QueuedBranch.parse(branch, config.git.branch_prefix),
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        # A successful merge always has a worktree; RealMergeItem.merge_wt is
        # required non-None (task ο), so narrow explicitly (mirrors merge_queue.py).
        assert merge_result.merge_worktree is not None
        base_sha = await git_ops.get_main_sha()
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
        )
        return req, item

    async def test_abandon_mid_verify_returns_dropped(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Sole-waiter abandon mid-verify → inner task cancelled, merge_wt cleaned,
        returns status=DROPPED.

        RED (step-6): current code has no poll loop — verify runs to completion
        when gate is released, result.status is None.
        GREEN (step-8): poll loop detects abandon before gate released → DROPPED.
        """
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        gated = _gated_runner(gate_release, gate_entered, passed=True, name='slow-abandon')

        req, item = await self._make_merged_item(
            git_ops, config, 'abort-poll-a', 'pa.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02  # fast polling for tests
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='slow-abandon', runner=gated, is_local=False)

        # Start verify in background (gated runner blocks until gate_release)
        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))

        # Wait for the gated runner to enter
        await asyncio.wait_for(gate_entered.wait(), timeout=45.0)

        # Trigger sole-waiter abandon
        req.result.cancel()

        # Give poll loop a chance to fire (step-6: will not detect — no loop)
        await asyncio.sleep(worker.VERIFY_ABANDON_POLL_SECS * 2)

        # Release gate so RED case doesn't hang
        gate_release.set()

        result = await verify_future

        # GREEN: poll loop detected abandon before gate → DROPPED
        # RED: verify ran to completion (gate released) → status=None
        assert result.status == 'DROPPED'  # RED: fails (None != 'DROPPED')

    async def test_abandon_mid_verify_cleans_merge_wt(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Abandon → merge_wt cleaned: result.merge_wt is None on DROPPED."""
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        gated = _gated_runner(gate_release, gate_entered, passed=True, name='slow-wt')

        req, item = await self._make_merged_item(
            git_ops, config, 'abort-poll-b', 'pb.py', 'y=2\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='slow-wt', runner=gated, is_local=False)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
        await asyncio.wait_for(gate_entered.wait(), timeout=45.0)
        req.result.cancel()
        await asyncio.sleep(worker.VERIFY_ABANDON_POLL_SECS * 2)
        gate_release.set()

        result = await verify_future

        # GREEN: wt cleaned by abort handler → result.merge_wt is None
        # RED: verify ran normally → result.merge_wt is not None
        assert result.merge_wt is None  # RED: fails (merge_wt is not None)

    async def test_operator_halt_mid_verify_returns_requeued(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Operator halt mid-verify → inner task cancelled, req re-queued, status=REQUEUED.

        RED (step-6): current code has no poll loop — verify completes normally.
        GREEN (step-8): poll loop detects halt, requeues req, returns REQUEUED.
        """
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        gated = _gated_runner(gate_release, gate_entered, passed=True, name='slow-halt')

        req, item = await self._make_merged_item(
            git_ops, config, 'abort-poll-c', 'pc.py', 'z=3\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='slow-halt', runner=gated, is_local=False)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
        await asyncio.wait_for(gate_entered.wait(), timeout=45.0)

        worker.operator_halt('test: halt mid-verify')

        await asyncio.sleep(worker.VERIFY_ABANDON_POLL_SECS * 2)
        # Release gate so RED case doesn't hang
        gate_release.set()

        result = await verify_future

        # GREEN: poll loop detected halt → REQUEUED
        # RED: verify ran to completion → status=None
        assert result.status == 'REQUEUED'  # RED: fails (None != 'REQUEUED')

    async def test_operator_halt_requeues_request_on_queue(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Operator halt → req is put back on worker._queue."""
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        gated = _gated_runner(gate_release, gate_entered, passed=True, name='slow-halt2')

        req, item = await self._make_merged_item(
            git_ops, config, 'abort-poll-d', 'pd.py', 'w=4\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='slow-halt2', runner=gated, is_local=False)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
        await asyncio.wait_for(gate_entered.wait(), timeout=45.0)

        worker.operator_halt('test: halt mid-verify')
        await asyncio.sleep(worker.VERIFY_ABANDON_POLL_SECS * 2)
        gate_release.set()

        await verify_future

        # GREEN: req put_nowait onto _queue (which is worker._queue == q)
        # RED: _queue is empty
        assert not q.empty(), 'req should be back on _queue after halt'  # RED: fails


# ---------------------------------------------------------------------------
# step-9 RED: _run_inflight_verify with RUNNER_UNAVAILABLE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunInflightVerifyRunnerUnavailable:
    """_run_inflight_verify with a REMOTE lease whose run_merge_verify raises
    RunnerUnavailable → returns status=RUNNER_UNAVAILABLE, merge_wt NOT cleaned.

    RED until step-10 GREEN catches RunnerUnavailable separately.
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        from orchestrator.merge_lane.worker import RealMergeItem

        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_running_loop()
        req = MergeRequest(
            task_id=branch,
            branch=QueuedBranch.parse(branch, config.git.branch_prefix),
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        # A successful merge always has a worktree; RealMergeItem.merge_wt is
        # required non-None (task ο), so narrow explicitly (mirrors merge_queue.py).
        assert merge_result.merge_worktree is not None
        base_sha = await git_ops.get_main_sha()
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
        )
        return req, item

    async def test_runner_unavailable_returns_sentinel(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """REMOTE run_merge_verify raises RunnerUnavailable → status=RUNNER_UNAVAILABLE,
        no exception escapes.

        RED (step-9): caught by generic except Exception → status=None, merge_wt=None.
        GREEN (step-10): caught by specific except RunnerUnavailable → status set.
        """
        from orchestrator.verify_runner import HostLease, RunnerUnavailable

        dead_remote = _make_fake_remote('dead-laptop')
        dead_remote.run_merge_verify = AsyncMock(
            side_effect=RunnerUnavailable('connection refused')
        )

        req, item = await self._make_merged_item(
            git_ops, config, 'runner-unavail-a', 'ra.py', 'a=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='dead-laptop', runner=dead_remote, is_local=False)

        # Should not raise — RunnerUnavailable is caught internally
        result = await worker._run_inflight_verify(item, lease)

        # GREEN: status sentinel set, merge_wt preserved for re-dispatch
        # RED: status=None (generic except catches RunnerUnavailable)
        assert result.status == 'RUNNER_UNAVAILABLE'  # RED: fails (None != ...)

    async def test_runner_unavailable_merge_wt_not_cleaned(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """REMOTE RunnerUnavailable → merge_wt NOT cleaned (preserved for re-dispatch)."""
        from orchestrator.verify_runner import HostLease, RunnerUnavailable

        dead_remote = _make_fake_remote('dead-laptop2')
        dead_remote.run_merge_verify = AsyncMock(
            side_effect=RunnerUnavailable('timeout')
        )

        req, item = await self._make_merged_item(
            git_ops, config, 'runner-unavail-b', 'rb.py', 'b=2\n',
        )
        merge_wt_path = item.merge_wt
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='dead-laptop2', runner=dead_remote, is_local=False)

        result = await worker._run_inflight_verify(item, lease)

        # GREEN: merge_wt intact — item re-dispatched on another host with wt intact
        # RED: merge_wt=None (generic exception handler cleans it)
        assert result.merge_wt is not None  # RED: fails (merge_wt is None)
        assert result.merge_wt == merge_wt_path


# ---------------------------------------------------------------------------
# step-11 RED: _finalize_inflight PASS path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinalizeInflightPass:
    """_finalize_inflight(entry) PASS path: CAS advance, lease release, _n_failed=False.

    RED until step-12 GREEN adds _finalize_inflight.
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        from orchestrator.merge_lane.worker import RealMergeItem

        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_running_loop()
        req = MergeRequest(
            task_id=branch,
            branch=QueuedBranch.parse(branch, config.git.branch_prefix),
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        # A successful merge always has a worktree; RealMergeItem.merge_wt is
        # required non-None (task ο), so narrow explicitly (mirrors merge_queue.py).
        assert merge_result.merge_worktree is not None
        base_sha = await git_ops.get_main_sha()
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
        )
        return req, item

    def _make_pass_entry(self, item, lease, was_speculative: bool = False):
        """Build an InflightEntry representing a completed, passing verify."""
        from orchestrator.merge_lane.worker import InflightEntry
        return InflightEntry(
            item=item,
            lease=lease,
            verify_task=None,
            merge_wt=item.merge_wt,
            was_speculative=was_speculative,
            passthrough_outcome=None,
            verify_result=None,  # None = pass
            status=None,
        )

    def _make_mock_allocator(self):
        """Return a MagicMock with async release/cancel_and_release."""
        alloc = MagicMock()
        alloc.release = AsyncMock()
        alloc.cancel_and_release = AsyncMock()
        return alloc

    async def test_finalize_pass_returns_true(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """PASS entry → _finalize_inflight returns True (main advanced).

        RED: method doesn't exist yet.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-a', 'fa.py', 'a=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease)

        advanced = await worker._finalize_inflight(entry)  # RED: method missing

        assert advanced is True

    async def test_finalize_pass_resolves_req_done(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """PASS entry → req.result resolved with outcome.status == 'done'."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-b', 'fb.py', 'b=2\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease)

        await worker._finalize_inflight(entry)

        assert req.result.done()
        assert req.result.result().status == 'done'

    async def test_finalize_pass_releases_lease(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """PASS entry → allocator.release called with the entry's lease."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-c', 'fc.py', 'c=3\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())
        mock_alloc = self._make_mock_allocator()
        worker._host_allocator = mock_alloc
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease)

        await worker._finalize_inflight(entry)

        mock_alloc.release.assert_called_once_with(lease)

    async def test_finalize_pass_sets_n_failed_false(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """PASS entry → _n_failed = False after finalize."""
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-d', 'fd.py', 'd=4\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._n_failed = True  # set to True to check it's reset to False

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease)

        await worker._finalize_inflight(entry)

        assert worker._n_failed is False

    async def test_finalize_pass_releases_speculation_slot_iff_speculative(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """was_speculative=True → the threaded permit is released via the ledger.

        η: _finalize_inflight's release guard is `entry.permit is not None`
        (not the raw was_speculative flag), so the entry must carry a real
        ledger-issued token for the release to fire.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-e', 'fe.py', 'e=5\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        # Acquire one permit through the ledger to simulate that a
        # speculative item is in-flight.
        permit = await worker._speculation_ledger.acquire()
        slot_value_before = worker._speculation_ledger.slot_available

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease, was_speculative=True)
        entry.permit = permit

        await worker._finalize_inflight(entry)

        # Slot should be released back
        assert worker._speculation_ledger.slot_available == slot_value_before + 1

    async def test_finalize_pass_does_not_release_slot_if_not_speculative(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """was_speculative=False, permit=None → _speculation_slot NOT released.

        η: the release gate is `entry.permit is not None`, not
        `was_speculative`. This covers the common case where both fields
        move together (permit-less, non-speculative entry); see
        `test_finalize_pass_does_not_release_slot_when_speculative_flag_true_but_no_permit`
        and `test_finalize_pass_releases_slot_when_permit_set_but_speculative_flag_false`
        below for the divergent combinations that actually exercise the gate
        being keyed on the permit rather than the flag.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-f', 'ff.py', 'f=6\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        slot_value_before = worker._speculation_ledger.slot_available

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease, was_speculative=False)

        await worker._finalize_inflight(entry)

        # Slot value unchanged (no release)
        assert worker._speculation_ledger.slot_available == slot_value_before

    async def test_finalize_pass_does_not_release_slot_when_speculative_flag_true_but_no_permit(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """was_speculative=True but permit=None → _speculation_slot NOT released.

        η: proves the release gate is genuinely keyed on `entry.permit`, not
        `was_speculative` — a regression that gated on the flag again would
        have nothing to distinguish this from the iff-speculative-true case,
        but there is no token to release here.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-g', 'fg.py', 'g=7\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        slot_value_before = worker._speculation_ledger.slot_available

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease, was_speculative=True)
        assert entry.permit is None

        await worker._finalize_inflight(entry)

        # Slot value unchanged: was_speculative=True alone releases nothing.
        assert worker._speculation_ledger.slot_available == slot_value_before

    async def test_finalize_pass_releases_slot_when_permit_set_but_speculative_flag_false(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """was_speculative=False but permit set → release fires anyway.

        η: the inverse divergence — proves a real ledger token is released
        through `_finalize_inflight` regardless of the (now-informational)
        was_speculative flag.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fin-pass-h', 'fh.py', 'h=8\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        permit = await worker._speculation_ledger.acquire()
        slot_value_before = worker._speculation_ledger.slot_available

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_pass_entry(item, lease, was_speculative=False)
        entry.permit = permit

        await worker._finalize_inflight(entry)

        # Slot released despite was_speculative=False, because a real token
        # was threaded onto the entry.
        assert worker._speculation_ledger.slot_available == slot_value_before + 1


# ---------------------------------------------------------------------------
# step-13 RED: _finalize_inflight non-pass paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinalizeInflightNonPass:
    """_finalize_inflight non-pass paths: FAIL, PASSTHROUGH, DROPPED, REQUEUED.

    RED until step-14 GREEN extends _finalize_inflight.
    """

    # ------------------------------------------------------------------
    # Shared helpers (duplicated from TestFinalizeInflightPass for isolation)
    # ------------------------------------------------------------------

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        from orchestrator.merge_lane.worker import RealMergeItem

        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_running_loop()
        req = MergeRequest(
            task_id=branch,
            branch=QueuedBranch.parse(branch, config.git.branch_prefix),
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        # A successful merge always has a worktree; RealMergeItem.merge_wt is
        # required non-None (task ο), so narrow explicitly (mirrors merge_queue.py).
        assert merge_result.merge_worktree is not None
        base_sha = await git_ops.get_main_sha()
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
        )
        return req, item

    def _make_passthrough_item(
        self,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
        outcome: MergeOutcome,
        already_delivered: bool = False,
    ):
        """Build a DecidedItem with immediate_outcome (no real merge)."""
        from orchestrator.merge_lane.worker import DecidedItem

        loop = asyncio.get_running_loop()
        req = MergeRequest(
            task_id='pt-task',
            branch=QueuedBranch.parse('pt-branch', config.git.branch_prefix),
            worktree=git_repo / 'pt-wt',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        item = DecidedItem(
            request=req,
            immediate_outcome=outcome,
            base_sha='deadbeef' * 5,
            speculative=False,
            already_delivered=already_delivered,
        )
        return req, item

    def _make_mock_allocator(self):
        """Return a MagicMock with async release/cancel_and_release."""
        alloc = MagicMock()
        alloc.release = AsyncMock()
        alloc.cancel_and_release = AsyncMock()
        return alloc

    def _make_fail_entry(self, item, lease, fail_outcome, merge_wt_val):
        """Build an InflightEntry with a verify_task that returns a FAIL result."""
        from orchestrator.merge_lane.worker import InflightEntry, InflightVerifyResult

        async def _fake_fail_verify():
            return InflightVerifyResult(
                outcome=fail_outcome,
                merge_wt=merge_wt_val,
                status=None,
            )

        verify_task = asyncio.ensure_future(_fake_fail_verify())
        return InflightEntry(
            item=item,
            lease=lease,
            verify_task=verify_task,
            merge_wt=merge_wt_val,
            was_speculative=False,
        )

    def _make_sentinel_entry(self, item, lease, sentinel_status, merge_wt_val=None):
        """Build an InflightEntry with a verify_task returning a DROPPED/REQUEUED sentinel."""
        from orchestrator.merge_lane.worker import InflightEntry, InflightVerifyResult

        async def _fake_sentinel_verify():
            return InflightVerifyResult(
                outcome=None,
                merge_wt=merge_wt_val,
                status=sentinel_status,
            )

        verify_task = asyncio.ensure_future(_fake_sentinel_verify())
        return InflightEntry(
            item=item,
            lease=lease,
            verify_task=verify_task,
            merge_wt=merge_wt_val,
            was_speculative=False,
        )

    # ------------------------------------------------------------------
    # (a) FAIL path
    # ------------------------------------------------------------------

    async def test_finalize_fail_returns_false(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """FAIL verify outcome → _finalize_inflight returns False.

        RED: current step-12 code ignores verify_task and runs CAS → returns True.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fail-a', 'fa.py', 'a=1\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        fail_outcome = MergeOutcome('blocked', reason='test verify fail')
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_fail_entry(item, lease, fail_outcome, item.merge_wt)

        result = await worker._finalize_inflight(entry)
        assert result is False  # RED: current code returns True

    async def test_finalize_fail_resolves_req_with_fail_outcome(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """FAIL outcome → req.result resolved with the fail MergeOutcome.

        RED: current code advances main and resolves with 'done'.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fail-b', 'fb.py', 'b=2\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        fail_outcome = MergeOutcome('blocked', reason='verify error')
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_fail_entry(item, lease, fail_outcome, item.merge_wt)

        await worker._finalize_inflight(entry)
        assert req.result.done()
        assert req.result.result().status == 'blocked'

    async def test_finalize_fail_releases_lease(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """FAIL outcome → allocator.release called with the lease.

        RED: current code may call release (in finally), but wrong behavior before that.
        This test verifies lease is released specifically for the fail path.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fail-c', 'fc.py', 'c=3\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        mock_alloc = self._make_mock_allocator()
        worker._host_allocator = mock_alloc
        worker._register_owned_merge_worktree(item.merge_wt)

        fail_outcome = MergeOutcome('blocked', reason='fail')
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_fail_entry(item, lease, fail_outcome, item.merge_wt)

        with contextlib.suppress(Exception):
            await worker._finalize_inflight(entry)
        mock_alloc.release.assert_called_once_with(lease)

    async def test_finalize_fail_sets_n_failed_true(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """FAIL outcome → self._n_failed = True.

        RED: current code sets _n_failed = not advanced = False (wrong).
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'fail-d', 'fd.py', 'd=4\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._n_failed = False  # start at False to detect the change
        worker._register_owned_merge_worktree(item.merge_wt)

        fail_outcome = MergeOutcome('blocked', reason='fail')
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_fail_entry(item, lease, fail_outcome, item.merge_wt)

        with contextlib.suppress(Exception):
            await worker._finalize_inflight(entry)
        assert worker._n_failed is True  # RED: current code sets False

    # ------------------------------------------------------------------
    # (b) PASSTHROUGH path (immediate_outcome / conflict / already_merged)
    # ------------------------------------------------------------------

    async def test_finalize_passthrough_conflict_resolves_req(
        self, git_repo: Path, git_config: GitConfig, config: OrchestratorConfig,
    ) -> None:
        """PASSTHROUGH (conflict) → req.result resolved with the passthrough outcome.

        RED: current code asserts merge_wt is not None → AssertionError.
        """
        from orchestrator.merge_lane.worker import InflightEntry

        conflict_outcome = MergeOutcome('conflict', reason='merge conflict')
        req, item = self._make_passthrough_item(
            git_repo, git_config, config, conflict_outcome, already_delivered=False,
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            GitOps(git_config, git_repo), q,
        )
        worker._host_allocator = self._make_mock_allocator()

        entry = InflightEntry(
            item=item,
            lease=None,
            verify_task=None,
            merge_wt=None,
            was_speculative=False,
            passthrough_outcome=conflict_outcome,
        )
        await worker._finalize_inflight(entry)  # RED: raises AssertionError
        assert req.result.done()
        assert req.result.result().status == 'conflict'

    async def test_finalize_passthrough_n_failed_true_for_conflict(
        self, git_repo: Path, git_config: GitConfig, config: OrchestratorConfig,
    ) -> None:
        """PASSTHROUGH conflict → _n_failed = True (conflict is not 'done'/'already_merged').

        RED: current code raises AssertionError before reaching _n_failed.
        """
        from orchestrator.merge_lane.worker import InflightEntry

        conflict_outcome = MergeOutcome('conflict', reason='conflict')
        req, item = self._make_passthrough_item(
            git_repo, git_config, config, conflict_outcome,
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(GitOps(git_config, git_repo), q)
        worker._host_allocator = self._make_mock_allocator()
        worker._n_failed = False

        entry = InflightEntry(
            item=item, lease=None, verify_task=None, merge_wt=None,
            was_speculative=False,
            passthrough_outcome=conflict_outcome,
        )
        with contextlib.suppress(Exception):
            await worker._finalize_inflight(entry)
        assert worker._n_failed is True

    async def test_finalize_passthrough_n_failed_false_for_already_merged(
        self, git_repo: Path, git_config: GitConfig, config: OrchestratorConfig,
    ) -> None:
        """PASSTHROUGH already_merged → _n_failed = False (chain not broken).

        RED: same AssertionError as above.
        """
        from orchestrator.merge_lane.worker import InflightEntry

        am_outcome = MergeOutcome('already_merged')
        req, item = self._make_passthrough_item(
            git_repo, git_config, config, am_outcome,
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(GitOps(git_config, git_repo), q)
        worker._host_allocator = self._make_mock_allocator()
        worker._n_failed = True  # start True to detect reset

        entry = InflightEntry(
            item=item, lease=None, verify_task=None, merge_wt=None,
            was_speculative=False,
            passthrough_outcome=am_outcome,
        )
        with contextlib.suppress(Exception):
            await worker._finalize_inflight(entry)
        assert worker._n_failed is False  # already_merged → not n_failed

    async def test_finalize_passthrough_already_delivered_skips_resolve(
        self, git_repo: Path, git_config: GitConfig, config: OrchestratorConfig,
    ) -> None:
        """PASSTHROUGH with already_delivered=True → req.result NOT set by finalize.

        RED: current code raises AssertionError before reaching the delivery check.
        """
        from orchestrator.merge_lane.worker import InflightEntry

        conflict_outcome = MergeOutcome('conflict', reason='conflict')
        req, item = self._make_passthrough_item(
            git_repo, git_config, config, conflict_outcome, already_delivered=True,
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(GitOps(git_config, git_repo), q)
        worker._host_allocator = self._make_mock_allocator()

        entry = InflightEntry(
            item=item, lease=None, verify_task=None, merge_wt=None,
            was_speculative=False,
            passthrough_outcome=conflict_outcome,
        )
        # req.result must NOT be set because already_delivered=True
        with contextlib.suppress(Exception):
            await worker._finalize_inflight(entry)
        # already_delivered means the merger already set the result OOB
        assert not req.result.done()

    # ------------------------------------------------------------------
    # (c) DROPPED / REQUEUED sentinel paths
    # ------------------------------------------------------------------

    async def test_finalize_dropped_calls_cancel_and_release(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """DROPPED sentinel → allocator.cancel_and_release(lease) called.

        RED: current code raises AssertionError (merge_wt=None).
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'drop-a', 'da.py', 'a=1\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        mock_alloc = self._make_mock_allocator()
        worker._host_allocator = mock_alloc

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_sentinel_entry(item, lease, 'DROPPED', merge_wt_val=None)

        with contextlib.suppress(Exception):
            await worker._finalize_inflight(entry)
        mock_alloc.cancel_and_release.assert_called_once_with(lease)

    async def test_finalize_dropped_sets_n_failed_true(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """DROPPED sentinel → _n_failed = True (chain stale: sole waiter abandoned).

        RED: current code raises before reaching _n_failed.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'drop-b', 'db.py', 'b=2\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()
        worker._n_failed = False

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_sentinel_entry(item, lease, 'DROPPED', merge_wt_val=None)

        with contextlib.suppress(Exception):
            await worker._finalize_inflight(entry)
        assert worker._n_failed is True

    async def test_finalize_dropped_returns_false(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """DROPPED sentinel → _finalize_inflight returns False.

        RED: current code raises before returning.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'drop-c', 'dc.py', 'c=3\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_sentinel_entry(item, lease, 'DROPPED', merge_wt_val=None)

        result = await worker._finalize_inflight(entry)
        assert result is False  # RED: current code raises AssertionError

    async def test_finalize_requeued_returns_false(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """REQUEUED sentinel → _finalize_inflight returns False.

        RED: current code raises AssertionError (merge_wt=None).
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'requeue-a', 'ra.py', 'a=1\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = self._make_mock_allocator()

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_sentinel_entry(item, lease, 'REQUEUED', merge_wt_val=None)

        result = await worker._finalize_inflight(entry)
        assert result is False  # RED: current raises AssertionError

    async def test_finalize_requeued_calls_cancel_and_release(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """REQUEUED sentinel → allocator.cancel_and_release(lease) called.

        RED: current code raises before cancel_and_release.
        """
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'requeue-b', 'rb.py', 'b=2\n',
        )
        q = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        mock_alloc = self._make_mock_allocator()
        worker._host_allocator = mock_alloc

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = self._make_sentinel_entry(item, lease, 'REQUEUED', merge_wt_val=None)

        with contextlib.suppress(Exception):
            await worker._finalize_inflight(entry)
        mock_alloc.cancel_and_release.assert_called_once_with(lease)


# ---------------------------------------------------------------------------
# step-15 RED: SINGLE-HOST serial byte-identical via the restructured loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestSingleHostSerialByteIdentical:
    """SINGLE-HOST serial byte-identical via the restructured _verifier_loop.

    RED until step-16 GREEN restructures _verifier_loop, adds _dispatch_item,
    and makes _verify_and_advance a thin compat shim.

    The two structural tests (test_dispatch_item_method_exists and
    test_verifier_loop_sets_n_failed_via_instance_attr) are the RED markers.
    The end-to-end tests document the byte-identical oracle that must stay
    green before and after the restructuring.
    """

    # ------------------------------------------------------------------
    # RED marker 1: _dispatch_item must exist on the class
    # ------------------------------------------------------------------

    async def test_dispatch_item_method_exists(self, git_ops: GitOps) -> None:
        """_dispatch_item is a callable on SpeculativeMergeWorker.

        RED: method not yet added (step-16 adds it).
        GREEN: step-16 adds _dispatch_item.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        assert callable(getattr(worker, '_dispatch_item', None)), (
            '_dispatch_item not found on SpeculativeMergeWorker — '
            'restructured loop not yet added (step-16). '
            'RED: None (missing). GREEN: callable after step-16.'
        )

    # ------------------------------------------------------------------
    # RED marker 2: _verifier_loop must update self._n_failed on the instance
    # ------------------------------------------------------------------

    async def test_verifier_loop_sets_n_failed_via_instance_attr(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Restructured _verifier_loop sets self._n_failed (not just a loop-local).

        The OLD _verifier_loop keeps n_failed as a loop-local variable and
        NEVER writes to self._n_failed.  The NEW loop delegates to
        _finalize_inflight which always sets self._n_failed in its finally.

        Use an immediate_outcome CONFLICT token to trigger the non-done path:
          - conflict.status='conflict' → _n_failed_val=True in _finalize_inflight
          - OLD: self._n_failed stays False (never set by old loop)
          - NEW: self._n_failed becomes True after finalize

        RED: old loop never writes self._n_failed → stays False after conflict.
        GREEN: step-16 restructured loop → _finalize_inflight sets self._n_failed=True.
        """
        from orchestrator.merge_lane.worker import DecidedItem

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req = _make_request('nf-conflict', 'nf-conflict', git_ops.project_root, config)
        conflict_outcome = MergeOutcome('conflict', reason='merge conflict')

        token = DecidedItem(
            request=req,
            immediate_outcome=conflict_outcome,
            base_sha='deadbeef',
            speculative=False,
            already_delivered=False,
        )

        await worker._verifier_queue.put(token)
        await worker._verifier_queue.put(None)  # sentinel

        # Drive _verifier_loop directly (same pattern as existing 3514/3571/3605 tests)
        await worker._verifier_loop()

        # OLD: self._n_failed stays False (loop-local n_failed only)
        # NEW: _finalize_inflight sets self._n_failed=True for conflict status
        assert worker._n_failed is True, (
            '_n_failed should be True after a conflict token. '
            'OLD _verifier_loop never updates self._n_failed (loop-local only). '
            'NEW restructured loop uses _finalize_inflight which sets self._n_failed. '
            'RED: stays False with old loop. GREEN: True after step-16.'
        )

    # ------------------------------------------------------------------
    # Byte-identical oracle: end-to-end run with two real items (no verify_runners)
    # ------------------------------------------------------------------

    async def test_single_host_two_items_both_done(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Single-host config + two real items → both resolve done, main advanced in order.

        This test is the byte-identical oracle: it must stay GREEN both before
        (old serial loop) and after (new dispatch-fill + finalize-head) step-16.

        Submission order: N is submitted first and must land on main first.
        Both items use real git repos so advance_main genuinely runs.
        """
        wt_a = await _make_branch_with_file(git_ops, 'task/sh-a', 'sha.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/sh-b', 'shb.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())
        worker_task = asyncio.create_task(worker.run())

        loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='sh-a',
            branch=QueuedBranch.parse('task/sh-a', config.git.branch_prefix),
            worktree=wt_a,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        req_b = MergeRequest(
            task_id='sh-b',
            branch=QueuedBranch.parse('task/sh-b', config.git.branch_prefix),
            worktree=wt_b,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )

        await q.put(req_a)
        outcome_a = await asyncio.wait_for(req_a.result, timeout=30)
        await q.put(req_b)
        outcome_b = await asyncio.wait_for(req_b.result, timeout=30)

        await worker.stop()
        await worker_task

        assert outcome_a.status == 'done', f'Item A: expected done, got {outcome_a}'
        assert outcome_b.status == 'done', f'Item B: expected done, got {outcome_b}'

        # Both files must be on main
        from orchestrator.git_ops import _run as _git_run
        _, main_files, _ = await _git_run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'sha.py' in main_files, 'Item A file not on main'
        assert 'shb.py' in main_files, 'Item B file not on main'


# ---------------------------------------------------------------------------
# step-17 RED: OVERLAP SIGNAL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestOverlapSignal:
    """Two-host overlap: both verifies enter before either is released.

    RED (step-17) until step-18 GREEN wires the fill loop to continue
    dispatching while a verify slot is free.

    RED markers:
    1. gate_b_entered times out — N+1 is NOT dispatched while N is verifying
       (serial blocking-get path finalizes N before fetching N+1).
    2. Peak _inflight length == 1 (never 2 simultaneously in serial mode).

    GREEN after step-18: dispatch-fill loop continues picking up N+1 while N
    is still running → both gates enter → peak == 2 → submission-order finalize
    confirmed even when N+1's verify completes first.
    """

    class _TrackingDeque(collections.deque):  # type: ignore[type-arg]
        """deque subclass that records peak length via append tracking."""

        def __init__(self) -> None:
            super().__init__()
            self.max_len: int = 0

        def append(self, item: Any) -> None:  # type: ignore[override]
            super().append(item)
            if len(self) > self.max_len:
                self.max_len = len(self)

        def appendleft(self, item: Any) -> None:  # type: ignore[override]
            super().appendleft(item)
            if len(self) > self.max_len:
                self.max_len = len(self)

    async def test_both_verifies_enter_before_either_released(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """N+1 starts verifying while N is still in-flight (concurrent overlap).

        RED: gate_b_entered times out (N+1 not dispatched until N finalizes).
        GREEN (step-18): both gates enter → overlap confirmed → submission-order
        finalize confirmed (N lands first even when N+1 verify completes first).
        """
        import contextlib

        # ── Gated verifies ─────────────────────────────────────────────────
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()

        # Remote runner (N+1's verify host) — gated on gate_b
        gated_remote = _gated_runner(gate_b_release, gate_b_entered, name='laptop')

        # Local verify (N's verify host) — gated through the injected port
        verifier = _GatedVerifier(
            entered={None: gate_a_entered}, default=hangs_until(gate_a_release),
        )

        # ── Branches ───────────────────────────────────────────────────────
        wt_a = await _make_branch_with_file(git_ops, 'task/ov-a', 'ov_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/ov-b', 'ov_b.py', 'b = 2\n')

        # ── Worker setup ───────────────────────────────────────────────────
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=verifier)

        # Replace _inflight with a tracking deque (captures peak length)
        tracked = self._TrackingDeque()
        worker._inflight = tracked  # type: ignore[assignment]

        # Inject two-host allocator: local + gated_remote
        _inject_two_host_allocator(worker, gated_remote)

        loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='ov-a', branch=QueuedBranch.parse('task/ov-a', config.git.branch_prefix), worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='ov-b', branch=QueuedBranch.parse('task/ov-b', config.git.branch_prefix), worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        worker_task = asyncio.create_task(worker.run())
        try:
            # Submit both requests; merger will process them while verifier runs
            await q.put(req_a)
            await q.put(req_b)

            # N (local) should enter its gated verify quickly
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='ov-a: local verify entered',
            )

            # While N is STILL verifying (gate_a NOT released), N+1 should ALSO
            # start verifying (remote slot is free → dispatch-fill overlap).
            # RED: times out — serial loop only dispatches N+1 after N finalizes.
            # GREEN (step-18): concurrent fill → gate_b_entered fires quickly.
            try:
                # task 2376: widened from 3.0s (host-oversubscription flake — a
                # starved worker missed the deadline while the invariant held;
                # see the 45.0s convention already used at lines ~2251/4185).
                await asyncio.wait_for(gate_b_entered.wait(), timeout=45.0)
            except TimeoutError:
                # Cleanup: release gates so worker can drain cleanly
                gate_a_release.set()
                gate_b_release.set()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(req_a.result, timeout=10.0)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(req_b.result, timeout=10.0)
                pytest.fail(
                    'OVERLAP not observed: gate_b timed out while gate_a was set. '
                    'RED: serial blocking-get path finalizes N before fetching N+1 '
                    '— N+1 only dispatched after N fully completes. '
                    'GREEN (step-18): blocking-get path loops back to fill → '
                    'dispatch-fill picks up N+1 to remote slot while N is in-flight.'
                )

            # ── Both verifies entered — overlap confirmed ──────────────────
            # gate_a is still blocking, gate_b is still blocking → true overlap
            assert gate_a_entered.is_set(), 'N (local) gate not entered'
            assert gate_b_entered.is_set(), 'N+1 (remote) gate not entered'
            assert not gate_a_release.is_set(), 'gate_a already released (test bug)'
            assert not gate_b_release.is_set(), 'gate_b already released (test bug)'

            # Peak _inflight == 2: both entries in deque before head was popped
            # RED: max_len == 1 (N appended, popped, finalized; then N+1 appended)
            # GREEN: max_len == 2 (N appended, N+1 appended, then head popped)
            assert tracked.max_len == 2, (
                f'Expected peak _inflight length 2 (both entries before popleft), '
                f'got {tracked.max_len}. '
                'RED: serial dispatch never fills two slots simultaneously. '
                'GREEN (step-18): fill loop appends N then N+1 before any popleft.'
            )

            # ── Submission-order finalize: release N+1 FIRST ──────────────
            # Even though N+1's verify finishes before N's, main must advance N first.
            gate_b_release.set()   # N+1 (remote) verify completes first
            gate_a_release.set()   # N (local) verify completes second

            # Both should resolve 'done'
            outcome_a = await wait_responsive(
                req_a.result,
                label='ov-a: MergeOutcome (submission-order finalize)',
            )
            outcome_b = await wait_responsive(
                req_b.result,
                label='ov-b: MergeOutcome (submission-order finalize)',
            )
        finally:
            await _stop_worker(worker, worker_task, join_timeout=45.0)

        assert outcome_a.status == 'done', f'N: expected done, got {outcome_a}'
        assert outcome_b.status == 'done', f'N+1: expected done, got {outcome_b}'

        # Both files on main (submission order preserved)
        from orchestrator.git_ops import _run as _git_run
        _, main_files, _ = await _git_run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'ov_a.py' in main_files, 'N (ov_a.py) not on main'
        assert 'ov_b.py' in main_files, 'N+1 (ov_b.py) not on main'


# ---------------------------------------------------------------------------
# esc-1735-5 REGRESSION: last-item-of-burst must not hang on the fill-loop's
# blocking queue.get() when a verify completes with no follow-on item.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLastItemOfBurstFinalizes:
    """A single (last-of-burst) item must finalize even with a free remote slot.

    Repro for the DISPATCH-FILL liveness bug (esc-1735-5): on multi-host, after
    dispatching N to the LOCAL slot the fill loop continues (a remote slot is
    free), finds the queue empty while N is still verifying, and — in the buggy
    version — blocks on a bare ``await self._verifier_queue.get()``.  Nothing
    wakes that get() when N's verify completes, so FINALIZE-HEAD never runs and
    N's result Future is never resolved: ``merge()`` hangs until an unrelated
    request happens to flow through.

    RED (buggy): ``req.result`` times out after the gate is released.
    GREEN (fixed): the fill loop races the persistent getter against the running
    verify task (asyncio.wait FIRST_COMPLETED); the completing verify wakes it,
    fill stops, FINALIZE-HEAD resolves N → 'done'.
    """

    async def test_single_item_resolves_with_free_remote_slot(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        # Gated LOCAL verify: enters, then blocks until released — guarantees the
        # fill loop reaches the empty-queue branch while N is still running.
        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()

        verifier = _GatedVerifier(
            entered={None: gate_entered}, default=hangs_until(gate_release),
        )

        # Remote runner exists (→ free_host_count()>1 so the fill loop continues
        # past the local dispatch) but is never used by this single-item burst.
        remote = _gated_runner(asyncio.Event(), name='laptop')

        wt = await _make_branch_with_file(git_ops, 'task/burst-n', 'burst_n.py', 'n = 1\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=verifier)
        _inject_two_host_allocator(worker, remote)

        loop = asyncio.get_running_loop()
        req = MergeRequest(
            task_id='burst-n', branch=QueuedBranch.parse('task/burst-n', config.git.branch_prefix), worktree=wt,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        worker_task = asyncio.create_task(worker.run())
        try:
            # Submit ONLY this one item — the last (and only) item of the burst.
            await q.put(req)

            # N enters its gated verify; the dispatch-fill loop is now parked
            # waiting for either a new queue item or N's verify to complete.
            await wait_responsive(
                gate_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='burst-n: local verify entered',
            )

            # Release N's verify.  With the bug, FINALIZE-HEAD is unreachable
            # (loop blocked on get()) and this Future never resolves.
            gate_release.set()
            try:
                outcome = await asyncio.wait_for(req.result, timeout=10.0)
            except TimeoutError:
                pytest.fail(
                    'LAST-ITEM HANG: single-item burst never finalized. '
                    'RED: fill loop blocked on bare _verifier_queue.get() — the '
                    'completing verify did not wake it, so FINALIZE-HEAD never ran. '
                    'GREEN: race persistent getter vs running verify tasks.'
                )
        finally:
            await _stop_worker(worker, worker_task)

        assert outcome.status == 'done', f'expected done, got {outcome}'

        from orchestrator.git_ops import _run as _git_run
        _, main_files, _ = await _git_run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'burst_n.py' in main_files, 'N (burst_n.py) not on main'


# ---------------------------------------------------------------------------
# step-19 RED: CHAIN-INVALIDATION UNDER OVERLAP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestChainInvalidationUnderOverlap:
    """N's verify fails while N+1 is in-flight: N+1 aborted, re-merged, re-verifies done.

    RED (step-19) until step-20 GREEN implements the head-failure cascade.

    RED markers:
    1. req_b times out — the fill-ahead blocking-get deadlocks: after N fails,
       the loop re-enters DISPATCH-FILL with _has_running_inflight=True and
       free_host_count=1, so it blocks on queue.get() waiting for new items.
       N+1's verify task finishes (gate_b released) but its result is never
       consumed because the loop is waiting for QUEUE items, not verify tasks.
    2. remote cancel_verify NOT called — N+1's verify runs to completion
       (or is left dangling) without the cancel signal.

    GREEN after step-20: head-failure cascade in _verifier_loop:
      · cancels N+1's in-flight verify task
      · cancel_and_release(N+1's lease) → remote cancel_verify called
      · cleans N+1's stale merge worktree
      · _remerge(N+1's req) → fresh SpeculativeItem on actual main
      · appends to _redispatch → re-dispatched on next fill iteration
    N+1 resolves 'done' and main has ci_b.py; N (ci_a.py) never landed.
    """

    async def test_n_fail_aborts_downstream_verify_reruns_remerge(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """N's local verify fails; N+1's remote verify is aborted and re-verified.

        RED: loop deadlocks on fill-ahead queue.get() → req_b times out.
        GREEN (step-20): head-failure cascade aborts N+1, re-merges onto
        actual main, re-dispatches → N+1 resolves 'done'.
        """
        # ── N's local verify: gated (fails when gate_a_release is set) ──────
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()

        verifier = _GatedVerifier(
            entered={None: gate_a_entered},
            first=dataclasses.replace(
                fails(category='test_failure', summary='test_failure'),
                release=gate_a_release,
            ),
        )

        # ── N+1's remote verify: gated (passes when gate_b_release is set) ──
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='laptop',
        )

        # ── Branches ─────────────────────────────────────────────────────────
        wt_a = await _make_branch_with_file(
            git_ops, 'task/ci-a', 'ci_a.py', 'a = 1\n'
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/ci-b', 'ci_b.py', 'b = 2\n'
        )

        # ── Worker setup ──────────────────────────────────────────────────────
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=verifier)
        _inject_two_host_allocator(worker, gated_remote)

        loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='ci-a', branch=QueuedBranch.parse('task/ci-a', config.git.branch_prefix), worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='ci-b', branch=QueuedBranch.parse('task/ci-b', config.git.branch_prefix), worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        worker_task = asyncio.create_task(worker.run())

        try:
            await q.put(req_a)
            await q.put(req_b)

            # Wait for both verifies to enter (true concurrent overlap)
            # NOTE (task 2350): widened from 15.0s -- fixed real-time deadlines
            # starve under heavy shared-host xdist contention even though the
            # underlying cascade logic is correct (timing flake, not a bug).
            await asyncio.wait_for(gate_a_entered.wait(), timeout=45.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=45.0)

            # N's verify fails
            gate_a_release.set()

            # N must resolve with a fail status
            outcome_a = await asyncio.wait_for(req_a.result, timeout=45.0)
            assert outcome_a.status not in ('done', 'already_merged'), (
                f'Expected N to fail, got status={outcome_a.status!r}.'
            )

            # Release N+1's gate so the test can complete in both paths:
            # RED: N+1's inner verify task unblocks, but the loop is still
            #      stuck on fill-ahead queue.get() → req_b never resolves.
            # GREEN: cascade already cancelled N+1's task; gate_b unblocks
            #        only the leaked inner task (result ignored).
            gate_b_release.set()

            # Wait for N+1 to resolve:
            # GREEN: cascade → re-merge → re-verify → 'done' (fast)
            # RED: deadlock → the wait gives up loudly, by label
            outcome_b = await wait_responsive(
                req_b.result,
                timeout=20.0,
                label='ci-b: MergeOutcome (re-merged and re-verified after N failed)',
            )
        finally:
            await _stop_worker(worker, worker_task, join_timeout=20.0)

        # A starved-or-hung wait fails loudly by label at the wait_responsive
        # call above; this assertion only ever fires on a WRONG STATUS.
        assert outcome_b.status == 'done', (
            f'Expected N+1 to resolve "done" after re-merge/re-verify, '
            f'got {outcome_b!r}. '
            'RED: fill-ahead blocking-get deadlocks after N fails — '
            '_inflight still has N+1 but loop waits for queue items. '
            'GREEN (step-20): head-failure cascade clears _inflight, '
            're-dispatches N+1 via _redispatch → "done".'
        )

        # ── RED: fails here (cancel_verify not called) ───────────────────────
        # Exactly 2 cancel_verify calls on the clean-cancel (rc=0) path:
        # (a) _abort_remote_verify (pre-cancel, id live → reaches remote process)
        # (b) cancel_and_release (post-cancel, id dead → harmless no-op in prod)
        # assert_called_once() was relaxed in task-1762 step-4; strengthened
        # to call_count == 2 (task-1762 amend) because the count is deterministic
        # on this path — catches both a missing and a spurious extra call.
        assert gated_remote.cancel_verify.call_count == 2, (
            f'Expected exactly 2 cancel_verify calls '
            f'(pre-cancel _abort_remote_verify + post-cancel cancel_and_release), '
            f'got {gated_remote.cancel_verify.call_count}'
        )

        # ── Main state: N+1 landed, N never did ─────────────────────────────
        from orchestrator.git_ops import _run as _git_run
        _, main_files, _ = await _git_run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'ci_b.py' in main_files, (
            'N+1 (ci_b.py) not on main after re-merge/re-verify'
        )
        assert 'ci_a.py' not in main_files, (
            'N (ci_a.py) must NOT be on main (verify failed)'
        )

    async def test_head_failure_cascade_remerge_registers_merging_then_redispatch_parked(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """The head-failure cascade's remerge-and-park of a downstream entry
        must reconcile the ItemLifecycle registry (VERIFYING -> MERGING ->
        REDISPATCH_PARKED), not just move the item onto ``_redispatch``.

        task 2441: merge_queue.py:9456-9460 re-merges N+1 and appends it to
        ``_redispatch`` with NO ``_note_transition`` calls, so the registry
        stays at VERIFYING while the item is physically parked on
        ``_redispatch``. The later DISPATCH-FILL drain's
        REDISPATCH_PARKED -> DISPATCHING ``_note_transition`` (9204-9209)
        then fails its from_state cross-check against the diverged VERIFYING
        registry state, firing a dedup'd L1
        ``_alarm_illegal_lifecycle_transition`` escalation.

        The bug is INVISIBLE to functional assertions: ``_note_transition``
        is best-effort (never raises), so N+1 still resolves 'done' even on
        RED. The RED signal is therefore the missing registry transitions /
        spurious escalation, NOT the outcome — mirrors the sibling
        ``test_runner_unavailable_cascade_finalizing_merging_redispatch_parked``
        (test_merge_queue_lifecycle_registry.py:1219).

        GREEN (task 2441): the cascade's remerge-and-park site wraps
        ``_remerge`` with ``_note_transition(VERIFYING -> MERGING)`` before
        and ``_note_transition(MERGING -> REDISPATCH_PARKED)`` after,
        mirroring the RUNNER_UNAVAILABLE sibling template
        (merge_queue.py:10532-10543).
        """
        from orchestrator.merge_lane.worker import ItemLifecycleState

        # ── N's local verify: gated (fails when gate_a_release is set) ──────
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()

        verifier = _GatedVerifier(
            entered={None: gate_a_entered},
            first=dataclasses.replace(
                fails(category='test_failure', summary='test_failure'),
                release=gate_a_release,
            ),
        )

        # ── N+1's remote verify: gated (passes when gate_b_release is set) ──
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='laptop',
        )

        # ── Branches ─────────────────────────────────────────────────────────
        wt_a = await _make_branch_with_file(
            git_ops, 'task/ci-a-lifecycle', 'ci_a_lifecycle.py', 'a = 1\n'
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/ci-b-lifecycle', 'ci_b_lifecycle.py', 'b = 2\n'
        )

        # ── Worker setup ──────────────────────────────────────────────────────
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=verifier)
        _inject_two_host_allocator(worker, gated_remote)

        loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='ci-a-lifecycle', branch=QueuedBranch.parse('task/ci-a-lifecycle', config.git.branch_prefix), worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='ci-b-lifecycle', branch=QueuedBranch.parse('task/ci-b-lifecycle', config.git.branch_prefix), worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        # ── Spies (installed BEFORE worker.run()) ───────────────────────────
        # (a) _note_transition: record req_b's (from_state, to_state) pairs.
        observed_b: list[tuple[Any, Any]] = []
        original_note_transition = worker._note_transition

        def _spy_note_transition(
            rid: str, from_state: Any, to_state: Any, **kwargs: Any,
        ) -> None:
            if rid == req_b.request_id:
                observed_b.append((from_state, to_state))
            return original_note_transition(rid, from_state, to_state, **kwargs)

        worker._note_transition = _spy_note_transition  # type: ignore[method-assign]

        # (b) _remerge: capture the registry's current state for req_b mid-call,
        # then delegate to the original so N+1 still re-merges and resolves.
        mid_remerge_state: list[Any] = []
        original_remerge = worker._remerge

        async def _spying_remerge(req_: Any, started: Any) -> Any:
            if req_.request_id == req_b.request_id:
                mid_remerge_state.append(worker._lifecycle.current(req_b.request_id))
            return await original_remerge(req_, started)

        worker._remerge = _spying_remerge  # type: ignore[method-assign]

        # (c) _alarm_illegal_lifecycle_transition: record request_ids of every
        # rejected-transition alarm (2nd positional arg, per the call at
        # merge_queue.py:5390-5393).
        illegal_alarms: list[str] = []

        def _spy_illegal_alarm(
            escalation_queue: Any, request_id: str, *args: Any, **kwargs: Any,
        ) -> None:
            illegal_alarms.append(request_id)

        outcome_b: MergeOutcome | None = None

        with patch(
            'orchestrator.merge_lane.worker._alarm_illegal_lifecycle_transition',
            _spy_illegal_alarm,
        ):
            worker_task = asyncio.create_task(worker.run())

            await q.put(req_a)
            await q.put(req_b)

            # Wait for both verifies to enter (true concurrent overlap).
            # NOTE (task 2350): widened timeouts -- fixed real-time deadlines
            # starve under heavy shared-host xdist contention.
            await asyncio.wait_for(gate_a_entered.wait(), timeout=45.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=45.0)

            # N's verify fails
            gate_a_release.set()

            outcome_a = await asyncio.wait_for(req_a.result, timeout=45.0)
            assert outcome_a.status not in ('done', 'already_merged'), (
                f'Expected N to fail, got status={outcome_a.status!r}.'
            )

            # Release N+1's gate so the cascade's re-dispatched verify can pass.
            gate_b_release.set()

            outcome_b = await asyncio.wait_for(req_b.result, timeout=45.0)

            await worker.stop()

        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=20.0)

        # ── Functional sanity: the bug self-recovers, so this passes on both
        # RED and GREEN. It is NOT the signal under test. ───────────────────
        assert outcome_b is not None and outcome_b.status == 'done', (
            f'Expected N+1 to resolve "done" after re-merge/re-verify, got {outcome_b!r}.'
        )

        # ── RED signal 1: the registry must read MERGING (and never VERIFYING)
        # while the cascade's _remerge() call for req_b is in flight (RED
        # reads [VERIFYING] because merge_queue.py:9456-9460 omits the
        # _note_transition wiring). Asserted on intent rather than exact list
        # equality/call-count: in this harness N+1 re-merges exactly once (N
        # failed, so main never advanced and the re-dispatch's Mechanism-2
        # staleness check finds base_sha==main and skips a second remerge),
        # but a future change that causes an extra remerge shouldn't fail
        # this assertion when the lifecycle wiring under test is otherwise
        # correct — only ever observing VERIFYING (the pre-fix stale read)
        # is the actual defect signal.
        assert ItemLifecycleState.MERGING in mid_remerge_state, (
            f'expected the registry to read MERGING at least once while the '
            f'cascade _remerge() ran for req_b, got {mid_remerge_state!r}'
        )
        assert ItemLifecycleState.VERIFYING not in mid_remerge_state, (
            f'registry must never read VERIFYING (the pre-fix stale state) '
            f'while the cascade _remerge() ran for req_b, got {mid_remerge_state!r}'
        )

        # ── RED signal 2: both cascade transitions must be observed for req_b.
        assert (ItemLifecycleState.VERIFYING, ItemLifecycleState.MERGING) in observed_b, (
            f'expected VERIFYING -> MERGING in req_b transitions, got {observed_b!r}'
        )
        assert (
            ItemLifecycleState.MERGING, ItemLifecycleState.REDISPATCH_PARKED,
        ) in observed_b, (
            f'expected MERGING -> REDISPATCH_PARKED in req_b transitions, got {observed_b!r}'
        )

        # ── RED signal 3: no spurious illegal-transition escalation for req_b.
        # (Do NOT assert (REDISPATCH_PARKED, DISPATCHING) in observed_b as a RED
        # signal -- the spy records the attempted transition even when the
        # underlying registry call rejects it, so that pair appears on both
        # RED and GREEN.)
        assert req_b.request_id not in illegal_alarms, (
            f'unexpected illegal-lifecycle-transition alarm for req_b '
            f'(registry diverged from pipeline state): {illegal_alarms!r}'
        )


# ---------------------------------------------------------------------------
# step-21 RED: OPERATOR-HALT aborts all in-flight + RUNNER_UNAVAILABLE quarantine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestHaltAndUnavailable:
    """step-21 RED tests: halt aborts all in-flight; RUNNER_UNAVAILABLE quarantine.

    RED until step-22 GREEN:
      (a) cascade incorrectly re-merges REQUEUED items (each item self-requeues
          via abort-poll, but the head-failure cascade fires and re-merges them)
      (b) RUNNER_UNAVAILABLE falls through to PASS path; host not quarantined
    """

    async def test_operator_halt_aborts_all_inflight(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """OPERATOR-HALT aborts both in-flight verifies and requeues both requests.

        After halt each _run_inflight_verify abort-poll detects _operator_halt
        and returns REQUEUED (req already back on _queue).  The head-failure
        cascade MUST NOT re-merge REQUEUED items — they are already handled.

        RED: cascade fires for N+1 (because N's finalize returns False for
        REQUEUED), calls _remerge(req_b) and puts the re-merged item in
        _redispatch.  We detect this by spying on _remerge.
        GREEN (step-22): cascade skips REQUEUED entries; _remerge is never
        called; both reqs land on _queue exactly once and are resolved cleanly.
        """
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()

        # N's local verify: gated (passes when released)
        verifier = _GatedVerifier(
            entered={None: gate_a_entered}, default=hangs_until(gate_a_release),
        )

        # N+1's remote verify: gated (passes when released)
        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='laptop',
        )

        wt_a = await _make_branch_with_file(git_ops, 'task/halt-a', 'halt_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/halt-b', 'halt_b.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=verifier)
        _inject_two_host_allocator(worker, gated_remote)
        worker.VERIFY_ABANDON_POLL_SECS = 0.01  # fast abort-poll for determinism

        loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='halt-a', branch=QueuedBranch.parse('task/halt-a', config.git.branch_prefix), worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='halt-b', branch=QueuedBranch.parse('task/halt-b', config.git.branch_prefix), worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        # Spy on _remerge to detect incorrect cascade re-merges.
        _original_remerge = worker._remerge
        remerge_task_ids: list[str] = []

        async def _spy_remerge(req: Any, started_mono: Any) -> Any:
            remerge_task_ids.append(req.task_id)
            return await _original_remerge(req, started_mono)

        worker._remerge = _spy_remerge  # type: ignore[method-assign]

        worker_task = asyncio.create_task(worker.run())
        try:
            await q.put(req_a)
            await q.put(req_b)

            # Wait for both verifies to enter (true concurrent overlap)
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='halt-a: local verify entered',
            )
            await wait_responsive(
                gate_b_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='halt-b: remote verify entered',
            )

            # Set operator halt — both abort-polls fire within 0.01s
            worker._operator_halt.set()

            # Give abort-polls time to fire and requeue
            await asyncio.sleep(0.15)

            # Release gates so the leaked inner tasks can complete (harmlessly)
            gate_a_release.set()
            gate_b_release.set()
        finally:
            await _stop_worker(worker, worker_task)

        # Both futures should be resolved (by stop()→shutdown, not by double-dispatch)
        assert req_a.result.done(), 'req_a.result should be resolved by stop()'
        assert req_b.result.done(), 'req_b.result should be resolved by stop()'

        # cancel_verify should be called for the remote runner (N+1's lease).
        gated_remote.cancel_verify.assert_called()

        # ── RED: cascade incorrectly calls _remerge for REQUEUED N+1 ───────────
        assert remerge_task_ids == [], (
            f'Expected no _remerge calls after halt (each item self-requeues), '
            f'but _remerge was called for: {remerge_task_ids!r}. '
            'RED: head-failure cascade fires for REQUEUED N+1 and calls _remerge. '
            'GREEN (step-22): cascade skips REQUEUED entries.'
        )

    async def test_runner_unavailable_quarantine_fallback(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """RUNNER_UNAVAILABLE mid-overlap: host quarantined, item re-dispatched on LOCAL.

        N (local) verifies successfully.  N+1's remote runner raises
        RunnerUnavailable.  _finalize_inflight must call quarantine_and_release
        (adding the dead runner's name to _runner_quarantine) and re-dispatch
        the item on the local fallback so it eventually resolves 'done'.

        RED: RUNNER_UNAVAILABLE falls through to the PASS path in
        _finalize_inflight.  advance_main succeeds with the unverified merge
        commit (req_b resolves 'done' BUT without verification) and the runner
        is NOT quarantined.
        GREEN (step-22): quarantine_and_release is called, runner name appears
        in _runner_quarantine, item is re-dispatched on local, req_b resolves
        'done' after a proper re-verify.
        """
        from orchestrator.verify_runner import RunnerUnavailable

        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        gate_b_entered = asyncio.Event()

        # N's local verify: gated (passes when released)
        verifier = _GatedVerifier(
            entered={None: gate_a_entered}, default=hangs_until(gate_a_release),
        )

        # N+1's remote runner: gates on entered, then raises RunnerUnavailable
        async def _unavailable_side(*args: Any, **kwargs: Any) -> Any:
            gate_b_entered.set()
            # Simulate a transient network failure after entering
            raise RunnerUnavailable('host unreachable')

        dead_remote = MagicMock()
        dead_remote.name = 'dead-laptop'
        dead_remote.is_local = False
        dead_remote.run_merge_verify = AsyncMock(side_effect=_unavailable_side)
        dead_remote.cancel_verify = AsyncMock(return_value=0)
        dead_remote.probe_clean = AsyncMock(return_value=True)

        wt_a = await _make_branch_with_file(git_ops, 'task/unav-a', 'unav_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/unav-b', 'unav_b.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=verifier)
        _inject_two_host_allocator(worker, dead_remote)

        loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='unav-a', branch=QueuedBranch.parse('task/unav-a', config.git.branch_prefix), worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='unav-b', branch=QueuedBranch.parse('task/unav-b', config.git.branch_prefix), worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        worker_task = asyncio.create_task(worker.run())

        try:
            await q.put(req_a)
            await q.put(req_b)

            # Wait for both verifies to enter
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='unav-a: local verify entered',
            )
            await wait_responsive(
                gate_b_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='unav-b: remote verify entered',
            )

            # N+1's runner already raised RunnerUnavailable (no gate; fires
            # immediately). Release N's gate so it can complete.
            gate_a_release.set()

            outcome_a = await wait_responsive(
                req_a.result,
                label='unav-a: MergeOutcome (local verify passed)',
            )
            outcome_b = await wait_responsive(
                req_b.result,
                label='unav-b: MergeOutcome (re-dispatched on local after quarantine)',
            )
        finally:
            await _stop_worker(worker, worker_task)

        # N should resolve 'done' (local verify passed)
        assert outcome_a.status == 'done', (
            f'Expected N to resolve "done", got {outcome_a!r}'
        )

        # N+1 should also resolve 'done' (re-dispatched on local fallback)
        assert outcome_b.status == 'done', (
            f'Expected N+1 to resolve "done" after re-dispatch on local, got {outcome_b!r}. '
            'RED: RUNNER_UNAVAILABLE falls through to PASS → unverified advance.'
        )

        # ── RED: dead runner not quarantined ─────────────────────────────────
        hosts = worker.snapshot()['hosts']
        quarantined = {h['name'] for h in hosts if h['quarantined']}
        assert dead_remote.name in quarantined, (
            f'Expected dead_remote.name={dead_remote.name!r} among the hosts '
            f'snapshot() reports quarantined; got {quarantined!r} out of {hosts!r}. '
            'RED: _finalize_inflight falls through to PASS for RUNNER_UNAVAILABLE '
            '— quarantine_and_release is never called. '
            'GREEN (step-22): RUNNER_UNAVAILABLE handled explicitly; '
            'quarantine_and_release called, runner quarantined.'
        )


# ---------------------------------------------------------------------------
# step-23 RED: stop() drains _inflight + snapshot() surfaces _inflight entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStopDrainsInflight:
    """stop() with items in-flight: verify tasks cancelled, futures resolved.

    RED until step-24 GREEN extends stop() to drain self._inflight.

    RED markers:
    1. req_a.result / req_b.result still pending after stop() — the current
       stop() drains _queue and _verifier_queue but ignores _inflight.
    2. In-flight verify tasks are not cancelled (still blocked on their gates).
    """

    async def test_stop_resolves_inflight_futures_without_running_worker(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """stop() called with two manual InflightEntries → both futures resolved.

        This test injects entries directly into _inflight without running
        worker.run(), so stop() is exercised in isolation.

        RED: stop() doesn't drain _inflight → futures stay pending.
        GREEN (step-24): stop() cancels each verify_task, resolves each
        pending req.result with the shutdown outcome.
        """
        from orchestrator.merge_lane.worker import InflightEntry, InflightVerifyResult

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        # Two gated verify coroutines that block until the gate is set.
        gate_a = asyncio.Event()
        gate_b = asyncio.Event()

        async def _gated_verify_a() -> InflightVerifyResult:
            await gate_a.wait()
            return InflightVerifyResult(outcome=None, merge_wt=None, status=None)

        async def _gated_verify_b() -> InflightVerifyResult:
            await gate_b.wait()
            return InflightVerifyResult(outcome=None, merge_wt=None, status=None)

        verify_task_a = asyncio.ensure_future(_gated_verify_a())
        verify_task_b = asyncio.ensure_future(_gated_verify_b())

        # Build two fake RealMergeItems with fresh futures.
        item_a = _make_real_item(git_ops, config, 'stop-a', 'aaa')
        item_b = _make_real_item(git_ops, config, 'stop-b', 'bbb')

        entry_a = InflightEntry(
            item=item_a,
            lease=None,
            verify_task=verify_task_a,
            merge_wt=None,
            was_speculative=False,
        )
        entry_b = InflightEntry(
            item=item_b,
            lease=None,
            verify_task=verify_task_b,
            merge_wt=None,
            was_speculative=False,
        )

        # Inject directly into _inflight
        worker._inflight.append(entry_a)
        worker._inflight.append(entry_b)

        # stop() must resolve both futures (gates still blocked).
        await worker.stop()

        # RED: futures still pending (stop() ignored _inflight).
        # GREEN (step-24): stop() cancels tasks + resolves futures with shutdown.
        assert item_a.request.result.done(), (
            'req_a.result not resolved by stop(). '
            'RED: stop() does not drain _inflight. '
            'GREEN (step-24): stop() cancels each verify_task and resolves the future.'
        )
        assert item_b.request.result.done(), (
            'req_b.result not resolved by stop(). '
            'RED: stop() does not drain _inflight. '
            'GREEN (step-24): stop() cancels each verify_task and resolves the future.'
        )

    async def test_stop_cancels_inflight_verify_tasks(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """stop() cancels all in-flight verify tasks (no leaked tasks).

        RED: tasks remain running after stop() (not cancelled).
        GREEN (step-24): tasks are cancelled by stop().
        """
        from orchestrator.merge_lane.worker import DecidedItem, InflightEntry, InflightVerifyResult

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        gate = asyncio.Event()

        async def _blocking_verify() -> InflightVerifyResult:
            await gate.wait()
            return InflightVerifyResult(outcome=None, merge_wt=None, status=None)

        verify_task = asyncio.ensure_future(_blocking_verify())
        req = _make_request('stop-ct', 'task/stop-ct', git_ops.project_root, config)
        item = DecidedItem(
            request=req,
            immediate_outcome=MergeOutcome('blocked', reason='test-filler'),
            base_sha='ccc',
            speculative=False,
        )
        entry = InflightEntry(
            item=item,
            lease=None,
            verify_task=verify_task,
            merge_wt=None,
            was_speculative=False,
        )
        worker._inflight.append(entry)

        await worker.stop()

        # RED: verify_task.cancelled() → False (stop() didn't cancel it).
        # GREEN (step-24): stop() cancels each task → done() and cancelled() True.
        assert verify_task.done(), (
            'verify_task not done after stop(). '
            'RED: stop() ignores _inflight. '
            'GREEN (step-24): stop() cancels each verify_task.'
        )

    async def test_stop_drains_redispatch(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """stop() also drains _redispatch (pending re-dispatch items).

        RED: _redispatch items are not drained; their futures stay pending.
        GREEN (step-24): _redispatch drained + futures resolved.
        """
        from orchestrator.merge_lane.worker import DecidedItem

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req = _make_request('stop-rd', 'task/stop-rd', git_ops.project_root, config)
        item = DecidedItem(
            request=req,
            immediate_outcome=MergeOutcome('blocked', reason='test-filler'),
            base_sha='ddd',
            speculative=False,
        )
        worker._redispatch.append(item)

        await worker.stop()

        assert req.result.done(), (
            'req.result not resolved after stop() with item in _redispatch. '
            'RED: stop() does not drain _redispatch. '
            'GREEN (step-24): _redispatch drained; future resolved with shutdown.'
        )


@pytest.mark.asyncio
class TestSnapshotInflight:
    """snapshot() surfaces all _inflight entries (not just the back-compat head).

    RED until step-24 GREEN extends snapshot() to enumerate self._inflight.

    RED markers:
    1. Only the head (via _verify_item) appears; second _inflight entry invisible.
    2. verify_in_progress matches the head (back-compat unchanged).
    Single-host (≤1 in-flight) is byte-identical to today.
    """

    async def test_snapshot_second_inflight_item_missing_in_red(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """With two in-flight entries, snapshot() currently only shows the head.

        After step-24 GREEN, both items appear in snapshot()['entries'].

        RED: second item's task_id ('snap-b') NOT in entries.
        GREEN: both 'snap-a' and 'snap-b' in entries (with state 'verifying').
        """
        from orchestrator.merge_lane.worker import DecidedItem, InflightEntry

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req_a = _make_request('snap-a', 'task/snap-a', git_ops.project_root, config)
        req_b = _make_request('snap-b', 'task/snap-b', git_ops.project_root, config)

        item_a = DecidedItem(
            request=req_a,
            immediate_outcome=MergeOutcome('blocked', reason='test-filler'),
            base_sha='aaa',
            speculative=False,
        )
        item_b = DecidedItem(
            request=req_b,
            immediate_outcome=MergeOutcome('blocked', reason='test-filler'),
            base_sha='bbb',
            speculative=False,
        )

        entry_a = InflightEntry(
            item=item_a,
            lease=None,
            verify_task=None,
            merge_wt=None,
            was_speculative=False,
        )
        entry_b = InflightEntry(
            item=item_b,
            lease=None,
            verify_task=None,
            merge_wt=None,
            was_speculative=False,
        )

        # Both entries in _inflight
        worker._inflight.append(entry_a)
        worker._inflight.append(entry_b)

        snap = worker.snapshot()
        entry_task_ids = [e['task_id'] for e in snap['entries']]

        # GREEN: both appear
        assert 'snap-a' in entry_task_ids, (
            f'HEAD (snap-a) missing from snapshot entries: {entry_task_ids}.'
        )
        assert 'snap-b' in entry_task_ids, (
            f'snap-b (second in-flight) not in snapshot entries: {entry_task_ids}. '
            'RED: snapshot() uses only _verify_item (head); _inflight[1:] invisible. '
            'GREEN (step-24): enumerate _inflight for additional verifying entries.'
        )

    async def test_snapshot_single_host_byte_identical(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """Single-host (≤1 in-flight): snapshot unchanged from today.

        With _inflight having exactly one entry, the snapshot must produce
        the SAME output as the pre-γ code.

        GREEN both before and after step-24 (byte-identical oracle).
        """
        from orchestrator.merge_lane.worker import DecidedItem, InflightEntry

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req = _make_request('snap-sh', 'task/snap-sh', git_ops.project_root, config)
        item = DecidedItem(
            request=req,
            immediate_outcome=MergeOutcome('blocked', reason='test-filler'),
            base_sha='aaa',
            speculative=False,
        )
        entry = InflightEntry(
            item=item,
            lease=None,
            verify_task=None,
            merge_wt=None,
            was_speculative=False,
        )

        # Single-host scenario: one _inflight entry
        worker._inflight.append(entry)

        snap = worker.snapshot()
        verifying_entries = [e for e in snap['entries'] if e['state'] == 'verifying']

        # Must have EXACTLY 1 verifying entry (no double-count)
        assert len(verifying_entries) == 1, (
            f'Expected exactly 1 verifying entry for single-host, '
            f'got {len(verifying_entries)}: {verifying_entries}. '
            'Single-host must be byte-identical before and after step-24.'
        )
        assert verifying_entries[0]['task_id'] == 'snap-sh'

    async def test_snapshot_verify_in_progress_reflects_head(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """verify_in_progress still mirrors the head (back-compat unchanged).

        After step-24, verify_in_progress must still come from the _inflight
        head (back-compat); additional _inflight entries appear only in
        'entries'.
        """
        from orchestrator.merge_lane.worker import DecidedItem, InflightEntry

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req_a = _make_request('snap-vip-a', 'task/snap-vip-a', git_ops.project_root, config)
        req_b = _make_request('snap-vip-b', 'task/snap-vip-b', git_ops.project_root, config)

        item_a = DecidedItem(
            request=req_a,
            immediate_outcome=MergeOutcome('blocked', reason='test-filler'),
            base_sha='aaa', speculative=False,
        )
        item_b = DecidedItem(
            request=req_b,
            immediate_outcome=MergeOutcome('blocked', reason='test-filler'),
            base_sha='bbb', speculative=False,
        )
        entry_a = InflightEntry(
            item=item_a, lease=None, verify_task=None, merge_wt=None,
            was_speculative=False,
        )
        entry_b = InflightEntry(
            item=item_b, lease=None, verify_task=None, merge_wt=None,
            was_speculative=False,
        )

        worker._inflight.append(entry_a)
        worker._inflight.append(entry_b)

        snap = worker.snapshot()

        # verify_in_progress must still point to head (back-compat)
        assert snap['verify_in_progress'] is not None
        assert snap['verify_in_progress']['task_id'] == 'snap-vip-a', (
            f"verify_in_progress['task_id'] should be 'snap-vip-a' (head), "
            f"got {snap['verify_in_progress'].get('task_id')!r}."
        )


# ---------------------------------------------------------------------------
# step-27 RED: warm_results threading into shadow compare
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinalizeInflightWarmResultsThreading:
    """_finalize_inflight PASS branch must thread vr.warm_results into
    _maybe_schedule_shadow_compare instead of passing a hardcoded empty dict.

    RED until step-28 GREEN replaces:
        _warm_results: dict[str, bool] = {}
    with:
        _warm_results = vr.warm_results if vr is not None else {}

    (merge_queue.py ~:7218-7219 in the PASS branch of _finalize_inflight)
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        from orchestrator.merge_lane.worker import RealMergeItem

        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_running_loop()
        req = MergeRequest(
            task_id=branch,
            branch=QueuedBranch.parse(branch, config.git.branch_prefix),
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        # A successful merge always has a worktree; RealMergeItem.merge_wt is
        # required non-None (task ο), so narrow explicitly (mirrors merge_queue.py).
        assert merge_result.merge_worktree is not None
        base_sha = await git_ops.get_main_sha()
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
        )
        return req, item

    def _make_mock_allocator(self):
        """Return a MagicMock with async release/cancel_and_release."""
        alloc = MagicMock()
        alloc.release = AsyncMock()
        alloc.cancel_and_release = AsyncMock()
        return alloc

    async def test_warm_results_threaded_into_shadow_compare(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """PASS entry with a completed verify_task carrying warm_results must
        pass those warm_results to _maybe_schedule_shadow_compare at arg[4].

        RED: the PASS branch hardcodes _warm_results = {} so the captured arg
        at index 4 is always {} regardless of vr.warm_results.
        """
        from orchestrator.merge_lane.worker import InflightEntry, InflightVerifyResult
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'wr-thread-a', 'wra.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)

        # Build a real asyncio.Task (not just a Future) wrapping a coroutine that
        # immediately returns a passing InflightVerifyResult with a non-empty
        # warm_results map — the data the shadow compare needs.
        warm_map: dict[str, str] = {'crate::test_x': 'pass', 'crate::test_y': 'fail'}
        merge_wt_cap = item.merge_wt  # capture for closure

        async def _immediate_vr() -> InflightVerifyResult:
            return InflightVerifyResult(
                outcome=None,
                merge_wt=merge_wt_cap,
                warm_results=warm_map,
            )

        vr_task: asyncio.Task[InflightVerifyResult] = asyncio.ensure_future(_immediate_vr())
        # Drain the event loop so the task has already completed before we pass
        # it to _finalize_inflight (mirrors how real dispatch works: the task
        # finishes before finalize awaits the deque head).
        await asyncio.sleep(0)

        entry = InflightEntry(
            item=item,
            lease=lease,
            verify_task=vr_task,
            merge_wt=item.merge_wt,
            was_speculative=False,
        )

        shadow_mock = AsyncMock()
        with patch(
            'orchestrator.merge_lane.worker._maybe_schedule_shadow_compare', shadow_mock,
        ):
            advanced = await worker._finalize_inflight(entry)

        assert advanced is True, "Expected PASS to return True (main advanced)."
        assert shadow_mock.called, "_maybe_schedule_shadow_compare was not awaited."
        actual_warm = shadow_mock.call_args.kwargs['warm_results']
        assert actual_warm == warm_map, (
            f"Expected warm_results {warm_map!r} threaded into shadow compare "
            f"(vr.warm_results pass-through), got {actual_warm!r}. "
            f"This is RED because _finalize_inflight hardcodes _warm_results = {{}}."
        )

    async def test_warm_results_empty_for_compat_shim_path(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Compat-shim path: verify_task=None → _maybe_schedule_shadow_compare
        receives {} (the 'vr is None → else {}' branch).

        This test pins byte-identical compat-shim behaviour: _verify_and_advance
        builds InflightEntries with verify_task=None from a LOCAL lease where
        warm_results came from the local _run_inflight_verify path already stored
        in vr; without a verify_task there is no vr, so {} is correct.
        """
        from orchestrator.merge_lane.worker import InflightEntry
        from orchestrator.verify_runner import HostLease

        req, item = await self._make_merged_item(
            git_ops, config, 'wr-compat-b', 'wrb.py', 'y=2\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=FakeVerifier())
        worker._host_allocator = self._make_mock_allocator()
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)

        # verify_task=None: pre-established PASS (compat shim / old-style pass).
        entry = InflightEntry(
            item=item,
            lease=lease,
            verify_task=None,
            merge_wt=item.merge_wt,
            was_speculative=False,
        )

        shadow_mock = AsyncMock()
        with patch(
            'orchestrator.merge_lane.worker._maybe_schedule_shadow_compare', shadow_mock,
        ):
            advanced = await worker._finalize_inflight(entry)

        assert advanced is True, "Expected PASS to return True (main advanced)."
        assert shadow_mock.called, "_maybe_schedule_shadow_compare was not awaited."
        actual_warm = shadow_mock.call_args.kwargs['warm_results']
        assert actual_warm == {}, (
            f"Expected empty warm_results for verify_task=None (compat path), "
            f"got {actual_warm!r}."
        )


# ---------------------------------------------------------------------------
# amend: RUNNER_UNAVAILABLE on HEAD — cascade re-merges speculative downstream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)  # task 3477 amend: see HEAVY_BARRIER_TEST_TIMEOUT -- derived so this cannot drift from TestCascadeFiresRemoteCancel's identical wait profile
class TestRunnerUnavailableHeadCascade:
    """RUNNER_UNAVAILABLE on the HEAD triggers the head-failure cascade.

    When _finalize_inflight(N) returns False because N's verify returned
    RUNNER_UNAVAILABLE, the _verifier_loop's head-failure cascade must fire
    for any downstream speculative entries still in _inflight.

    The downstream entry (N+1) is NOT left in _inflight with a stale
    speculative commit — the cascade is the correctness mechanism.
    This test directly confirms:
      · cascade cancels N+1's in-flight verify
      · cascade re-merges N+1 onto actual main
      · both N (re-merged) and N+1 (re-merged) eventually land 'done'
      · quarantine_and_release is called for the lease whose verify failed

    See the RUNNER_UNAVAILABLE comment in _finalize_inflight for the
    design rationale.
    """

    async def test_ru_head_cascade_reruns_speculative_downstream(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """N's verify raises RunnerUnavailable while N+1 (speculative) is in-flight.

        The head-failure cascade must cancel N+1, re-merge both N and N+1, and
        eventually resolve both 'done' in submission order.

        RunnerUnavailable is raised from the injected verifier port so that
        both items can go through the local verify path; the cascade and quarantine
        behaviour is identical regardless of whether the failure comes from a local
        or remote runner (quarantine_and_release with is_local=True just frees the
        slot without adding to _runner_quarantine).
        """
        from orchestrator.verify_runner import RunnerUnavailable

        # ── N's verify: gated, then raises RunnerUnavailable ─────────────────
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()

        # ── N+1's remote verify: gated, then passes ───────────────────────────
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        gated_remote = _gated_runner(gate_b_release, gate_b_entered, passed=True, name='remote-b')

        verifier = _GatedVerifier(
            entered={None: gate_a_entered},
            first=dataclasses.replace(
                raises(RunnerUnavailable('simulated host failure for cascade test')),
                release=gate_a_release,
            ),
        )

        # ── Branches ─────────────────────────────────────────────────────────
        wt_a = await _make_branch_with_file(git_ops, 'task/rucascade-a', 'rucascade_a.py', 'a=1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/rucascade-b', 'rucascade_b.py', 'b=2\n')

        # ── Worker + two-host allocator ───────────────────────────────────────
        # N gets the LOCAL slot (prefer-local), N+1 gets the REMOTE slot.
        # When N's local verify raises RunnerUnavailable, _finalize_inflight
        # calls quarantine_and_release on N's local lease (which just frees the
        # local slot since is_local=True bypasses runner quarantine).
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=verifier)
        _inject_two_host_allocator(worker, gated_remote)

        loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='rucascade-a', branch=QueuedBranch.parse('task/rucascade-a', config.git.branch_prefix), worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='rucascade-b', branch=QueuedBranch.parse('task/rucascade-b', config.git.branch_prefix), worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        worker_task = asyncio.create_task(worker.run())

        await q.put(req_a)
        await q.put(req_b)

        # Wait for N's local verify to enter AND N+1's remote verify to enter.
        # This confirms both are in-flight simultaneously.
        # NOTE (task 3477): widened from 15.0s -- fixed real-time deadlines
        # starve under heavy shared-host xdist contention even though the
        # underlying cascade logic is correct (timing flake, not a bug).
        await asyncio.wait_for(gate_a_entered.wait(), timeout=45.0)
        await asyncio.wait_for(gate_b_entered.wait(), timeout=45.0)

        # Both verifies are now in-flight.  Release N's gate → RunnerUnavailable.
        # _finalize_inflight(N) returns False → head-failure cascade fires:
        #   · N+1's verify task is cancelled (cascade calls cancel_and_release)
        #   · N+1 is re-merged onto actual main
        #   · both N and N+1 are re-dispatched via _redispatch
        gate_a_release.set()

        # Release N+1's gate so the (possibly still-running) gated task
        # can unblock even if cancel() arrives slightly late.
        gate_b_release.set()

        # Wait for both to resolve. A real timeout now fails loudly by
        # name (task 3477) instead of leaving outcome_X as a confusing
        # None that the assertions below would misreport.
        outcome_a = await _await_outcome(req_a.result, label='N (RUNNER_UNAVAILABLE head)')
        outcome_b = await _await_outcome(req_b.result, label='N+1 (speculative downstream)')

        await worker.stop()

        # NOTE (task 3477): widened from 5.0s, matching the gate/outcome waits above.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=45.0)

        # ── N (RUNNER_UNAVAILABLE head): must land 'done' after re-dispatch ──
        assert outcome_a.status == 'done', (
            f'Expected N (RUNNER_UNAVAILABLE head) to resolve "done" after '
            f're-merge re-dispatch, got {outcome_a!r}. '
            'The head-failure cascade should re-merge N onto actual main '
            'and re-dispatch it for a clean verify.'
        )

        # ── N+1 (speculative downstream): must land 'done' after cascade ─────
        assert outcome_b.status == 'done', (
            f'Expected N+1 (speculative downstream) to resolve "done" after '
            f'cascade re-merge re-dispatch, got {outcome_b!r}. '
            'If the cascade did not fire, N+1 would stay in _inflight with a '
            'stale speculative commit and eventually fail CAS or time out.'
        )

        # ── Cascade called cancel_and_release for N+1's remote lease ─────────
        gated_remote.cancel_verify.assert_called()

        # ── Main has both files (submission order preserved) ──────────────────
        from orchestrator.git_ops import _run as _git_run
        _, main_files, _ = await _git_run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'rucascade_a.py' in main_files, (
            'N (rucascade_a.py) must be on main after re-merge/re-verify'
        )
        assert 'rucascade_b.py' in main_files, (
            'N+1 (rucascade_b.py) must be on main after cascade re-merge/re-verify'
        )


# ---------------------------------------------------------------------------
# task-1757 RED step-1: remote cancel fired on abandon abort
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestRunInflightVerifyRemoteCancelOnAbort:
    """_run_inflight_verify abort paths fire remote cancel before cancelling verify.

    RED (step-1 / step-3) until the fix inserts cancel_verify() before
    verify_task.cancel() in both abort branches.
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        """Create a merged SpeculativeItem on the given branch."""
        from orchestrator.merge_lane.worker import RealMergeItem

        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        loop = asyncio.get_running_loop()
        req = MergeRequest(
            task_id=branch,
            branch=QueuedBranch.parse(branch, config.git.branch_prefix),
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        # A successful merge always has a worktree; RealMergeItem.merge_wt is
        # required non-None (task ο), so narrow explicitly (mirrors merge_queue.py).
        assert merge_result.merge_worktree is not None
        base_sha = await git_ops.get_main_sha()
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
        )
        return req, item

    async def test_abandon_mid_verify_fires_remote_cancel(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Sole-waiter abandon with a REMOTE lease → cancel_verify() fired AND
        status == 'DROPPED'.

        RED (step-1): the abandon branch (merge_queue.py ~:7371) calls
        verify_task.cancel() first, which clears _inflight_request_id; any
        subsequent cancel_verify is a no-op.  cancel_verify.await_count == 0.
        GREEN (step-2): cancel_verify fired BEFORE verify_task.cancel() while
        id is still live → await_count == 1.
        """
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        gated = _gated_runner(gate_release, gate_entered, passed=True, name='rca-abandon')

        req, item = await self._make_merged_item(
            git_ops, config, 'rca-abandon-a', 'rca_a.py', 'x=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='rca-abandon', runner=gated, is_local=False)

        # Wrap cancel_verify to signal when it fires so the test can wait
        # deterministically rather than relying on a fixed timing sleep.
        cancel_fired = asyncio.Event()
        async def _signaling_cancel() -> int:
            cancel_fired.set()
            return 0
        gated.cancel_verify = AsyncMock(side_effect=_signaling_cancel)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))

        # Wait for the gated runner to enter verify
        await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

        # Trigger sole-waiter abandon
        req.result.cancel()

        # Wait until cancel_verify fires (observable signal replaces timing sleep)
        await asyncio.wait_for(cancel_fired.wait(), timeout=2.0)

        # Release gate (belt-and-suspenders; verify_task is already being
        # cancelled by the abort branch but gate_release is harmless here)
        gate_release.set()

        result = await verify_future

        # Must return DROPPED (already passes with step-7 GREEN)
        assert result.status == 'DROPPED'

        # RED: cancel_verify not called before fix (await_count == 0)
        # GREEN: cancel_verify called once (await_count == 1)
        gated.cancel_verify.assert_awaited_once()

    async def test_operator_halt_mid_verify_fires_remote_cancel_before_requeue(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Operator halt with a REMOTE lease → cancel_verify() fired AND fired
        BEFORE req is put back on the queue AND status == 'REQUEUED'.

        RED (step-3a): the halt branch calls verify_task.cancel() first
        (cancel_verify await_count == 0); even after step-2 fix only the abandon
        branch was patched.
        GREEN (step-4): cancel_verify inserted before verify_task.cancel() in
        the halt branch (trigger 2), and before put_nowait so queue is empty at
        cancel time.
        """
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        gated = _gated_runner(gate_release, gate_entered, passed=True, name='rca-halt')

        req, item = await self._make_merged_item(
            git_ops, config, 'rca-halt-a', 'rca_halt_a.py', 'h=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker._register_owned_merge_worktree(item.merge_wt)

        lease = HostLease(name='rca-halt', runner=gated, is_local=False)

        # Patch cancel_verify to record whether queue was empty at call time
        # and to signal deterministically when the cancel fires.
        _queue_empty_at_cancel: list[bool] = []
        cancel_fired = asyncio.Event()

        async def _record_cancel() -> int:
            _queue_empty_at_cancel.append(q.empty())
            cancel_fired.set()
            return 0

        gated.cancel_verify = AsyncMock(side_effect=_record_cancel)

        verify_future = asyncio.ensure_future(worker._run_inflight_verify(item, lease))
        await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

        worker.operator_halt('test: halt mid-verify')

        # Wait until cancel_verify fires (observable signal replaces timing sleep)
        await asyncio.wait_for(cancel_fired.wait(), timeout=2.0)
        gate_release.set()

        result = await verify_future

        # Status must be REQUEUED
        assert result.status == 'REQUEUED'

        # RED: cancel_verify not fired → await_count == 0
        # GREEN: cancel_verify called once
        assert gated.cancel_verify.await_count == 1, (
            f'Expected cancel_verify awaited once; got {gated.cancel_verify.await_count}'
        )

        # GREEN: cancel was fired BEFORE re-queue (queue was empty at cancel time)
        assert _queue_empty_at_cancel == [True], (
            f'cancel_verify must fire before put_nowait; '
            f'queue-empty-at-cancel={_queue_empty_at_cancel}'
        )

        # req was put back onto the queue after cancel
        assert not q.empty(), 'req should be back on queue after halt'

    async def test_local_lease_abort_does_not_fire_remote_cancel(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """LOCAL lease abandon → cancel_verify is NOT called (no remote process).

        Regression guard for the `if not lease.is_local` guard introduced in
        step-2 / step-4.  LocalRunner has no cancel_verify; an unguarded call
        would AttributeError.  Even with a MagicMock runner, verify that
        cancel_verify.await_count == 0 on the local path.

        Uses a local HostLease + a gated verifier injected at the worker's verifier
        port, so the abort-poll fires before the verify completes.
        """
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()

        # Gated scoped verify, injected through the worker's verifier port, so
        # the local arm blocks inside _run_post_merge_verify until released.
        verifier = _GatedVerifier(
            entered={None: gate_entered}, default=hangs_until(gate_release),
        )

        req, item = await self._make_merged_item(
            git_ops, config, 'rca-local-a', 'rca_local_a.py', 'l=1\n',
        )
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=verifier)
        worker.VERIFY_ABANDON_POLL_SECS = 0.02
        worker._register_owned_merge_worktree(item.merge_wt)

        # LOCAL lease — runner is a MagicMock that has cancel_verify as AsyncMock
        # so we can assert it is NOT called.
        mock_runner = MagicMock()
        mock_runner.cancel_verify = AsyncMock(return_value=0)
        lease = HostLease(name='local', runner=mock_runner, is_local=True)

        verify_future = asyncio.ensure_future(
            worker._run_inflight_verify(item, lease)
        )
        await asyncio.wait_for(gate_entered.wait(), timeout=45.0)

        # Trigger abandon.  The abort poll will cancel verify_task
        # (propagating CancelledError into the gated verifier's release wait),
        # so verify_future completes with DROPPED without needing the
        # gate to be released.  Await directly with a timeout instead
        # of a fixed timing sleep.
        req.result.cancel()
        # task 2376: widened from 2.0s — host oversubscription can starve
        # this poll past a short deadline while the invariant still holds.
        result = await asyncio.wait_for(verify_future, timeout=45.0)
        gate_release.set()  # cleanup: no-op if already cancelled

        assert result.status == 'DROPPED'

        # The local path must NOT call cancel_verify
        mock_runner.cancel_verify.assert_not_awaited()


# ---------------------------------------------------------------------------
# task-1762 step-1 RED: stop() fires remote cancel before task.cancel()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStopDrainFiresRemoteCancel:
    """stop() fires remote cancel-verify BEFORE cancelling in-flight verify tasks.

    RED (step-1) until step-2 GREEN inserts ``_abort_remote_verify`` before
    ``verify_task.cancel()`` in the _inflight drain loop inside stop().

    RED marker: ``fake.remote_cancels_while_live == []`` — stop() cancels the
    verify_task first, which clears live=False in the fake's finally (mirroring
    verify_runner.py:799 clearing _inflight_request_id), so the subsequent
    cancel_verify call finds a dead id and appends nothing.

    GREEN: ``_abort_remote_verify`` is called while live=True →
    ``remote_cancels_while_live == [1]``.
    """

    async def test_stop_fires_remote_cancel_before_task_cancel(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """stop() remote-cancels an in-flight REMOTE verify BEFORE task.cancel().

        Uses an id-liveness fake (pre-1) that models verify_runner.py:799/814:
          run_merge_verify:  live=True on entry, live=False in finally (mirrors
                             _inflight_request_id clearing on CancelledError).
          cancel_verify:     appends 1 only while live=True.

        worker._host_allocator is None so cancel_and_release is never called —
        the ONLY possible cancel_verify path is the new _abort_remote_verify.

        RED: stop() cancels verify_task before any remote cancel → live cleared
             → remote_cancels_while_live stays [].
        GREEN (step-2): _abort_remote_verify fires while live=True → [1].
        """
        from orchestrator.merge_lane.worker import DecidedItem, InflightEntry
        from orchestrator.verify_runner import HostLease

        gate_release = asyncio.Event()
        gate_entered = asyncio.Event()
        fake_remote = _id_liveness_fake_runner(
            gate_release, gate_entered, name='id-live-laptop',
        )

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = None     # cancel_and_release never called

        # ── REMOTE entry: id-liveness fake ──────────────────────────────────
        req_remote = _make_request(
            'stop-rc-remote', 'task/stop-rc-remote',
            git_ops.project_root, config,
        )
        item_remote = DecidedItem(
            request=req_remote,
            immediate_outcome=MergeOutcome('blocked', reason='test-filler'),
            base_sha='abc',
            speculative=False,
        )
        # Start run_merge_verify so the id is live (gate_entered set, awaits release).
        verify_task = asyncio.ensure_future(
            fake_remote.run_merge_verify(
                'branch', '/wt', 'sha', {}, 0, 0, 'id-live-laptop',
            )
        )
        await asyncio.wait_for(gate_entered.wait(), timeout=5.0)

        remote_lease = HostLease(
            name=fake_remote.name, runner=fake_remote, is_local=False,
        )
        entry_remote = InflightEntry(
            item=item_remote,
            lease=remote_lease,
            verify_task=verify_task,
            merge_wt=None,
            was_speculative=False,
        )

        # ── LOCAL entry: negative control — _abort_remote_verify must skip ──
        local_runner = MagicMock()
        local_runner.cancel_verify = AsyncMock(return_value=0)
        local_lease = HostLease(name='local', runner=local_runner, is_local=True)
        req_local = _make_request(
            'stop-rc-local', 'task/stop-rc-local',
            git_ops.project_root, config,
        )
        item_local = DecidedItem(
            request=req_local,
            immediate_outcome=MergeOutcome('blocked', reason='test-filler'),
            base_sha='def',
            speculative=False,
        )
        verify_task_local = asyncio.ensure_future(asyncio.sleep(999))
        entry_local = InflightEntry(
            item=item_local,
            lease=local_lease,
            verify_task=verify_task_local,
            merge_wt=None,
            was_speculative=False,
        )

        worker._inflight.append(entry_remote)
        worker._inflight.append(entry_local)

        await worker.stop()
        gate_release.set()   # cleanup: unblock leaked inner task if any

        # (a) Remote cancel must have reached the LIVE id.
        # RED: remote_cancels_while_live == [] (stop() cancelled verify_task first
        #      → live cleared → cancel_verify sees dead id).
        # GREEN (step-2): _abort_remote_verify fires before task.cancel() → [1].
        assert fake_remote.remote_cancels_while_live == [1], (
            f'Expected remote_cancels_while_live==[1] (cancel reached live id), '
            f'got {fake_remote.remote_cancels_while_live!r}. '
            'RED: stop() cancels verify_task before remote cancel → live cleared '
            '→ cancel_verify sees dead id → no entry appended. '
            'GREEN (step-2): _abort_remote_verify fires before task.cancel().'
        )

        # (b) Local lease must NOT trigger cancel_verify (is_local guard).
        local_runner.cancel_verify.assert_not_awaited()


# ---------------------------------------------------------------------------
# task-1762 step-3 RED: head-failure cascade fires remote cancel before cancel()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)  # task 3477 amend: see HEAVY_BARRIER_TEST_TIMEOUT -- derived so this cannot drift from TestRunnerUnavailableHeadCascade's identical wait profile
class TestCascadeFiresRemoteCancel:
    """_verifier_loop head-failure cascade fires remote cancel BEFORE task.cancel().

    RED (step-3) until step-4 GREEN inserts _abort_remote_verify before
    _entry.verify_task.cancel() in the cascade loop inside _verifier_loop.

    RED marker: ``fake.remote_cancels_while_live == []`` — the cascade calls
    _entry.verify_task.cancel() first, which clears live=False in the fake's
    finally; the subsequent cancel_and_release→cancel_verify sees a dead id.

    GREEN: ``_abort_remote_verify`` fires while live=True →
    ``remote_cancels_while_live == [1]``.

    Template: test_n_fail_aborts_downstream_verify_reruns_remerge (:2018).
    N is a gated LOCAL verify that fails; N+1's remote runner is the
    id-liveness fake (pre-1) injected via _inject_two_host_allocator.
    """

    async def test_cascade_fires_remote_cancel_before_task_cancel(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """Head-fail cascade remote-cancels N+1's LIVE id BEFORE task.cancel().

        N (local) fails → head-failure cascade fires against N+1 (remote,
        id-liveness fake).

        RED: cascade calls _entry.verify_task.cancel() first → live cleared
             → cancel_and_release→cancel_verify sees dead id →
             remote_cancels_while_live stays [].
        GREEN (step-4): _abort_remote_verify fires before task.cancel() → [1].
        """
        # ── N's gated local verify: fails when gate_a_release is set ─────────
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        verifier = _GatedVerifier(
            entered={None: gate_a_entered},
            first=dataclasses.replace(
                fails(category='test_failure', summary='test_failure'),
                release=gate_a_release,
            ),
        )

        # ── N+1's remote verify: id-liveness fake ────────────────────────────
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        fake_remote = _id_liveness_fake_runner(
            gate_b_release, gate_b_entered, name='id-live-cascade',
        )

        # ── Branches ─────────────────────────────────────────────────────────
        wt_a = await _make_branch_with_file(
            git_ops, 'task/casc-a', 'casc_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'task/casc-b', 'casc_b.py', 'b = 2\n',
        )

        # ── Worker setup ─────────────────────────────────────────────────────
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=verifier)
        _inject_two_host_allocator(worker, fake_remote)

        loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='casc-a', branch=QueuedBranch.parse('task/casc-a', config.git.branch_prefix), worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='casc-b', branch=QueuedBranch.parse('task/casc-b', config.git.branch_prefix), worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=loop.create_future(), lane='normal',
        )

        worker_task = asyncio.create_task(worker.run())

        await q.put(req_a)
        await q.put(req_b)

        # Wait for both verifies to enter (true concurrent overlap)
        # NOTE (task 3477): widened from 15.0s -- fixed real-time deadlines
        # starve under heavy shared-host xdist contention even though the
        # underlying cascade logic is correct (timing flake, not a bug).
        await asyncio.wait_for(gate_a_entered.wait(), timeout=45.0)
        await asyncio.wait_for(gate_b_entered.wait(), timeout=45.0)

        # N's verify fails → triggers head-failure cascade for N+1
        gate_a_release.set()

        # N must resolve with a fail status. Routed through _await_outcome
        # (task 3477) so a real timeout reports by name instead of raising
        # a bare TimeoutError, matching outcome_b's wait below.
        outcome_a = await _await_outcome(req_a.result, label='N (gated local fail)')
        assert outcome_a.status not in ('done', 'already_merged'), (
            f'Expected N to fail, got status={outcome_a.status!r}.'
        )

        # Release N+1's gate to unblock the leaked inner task (the cascade
        # already cancelled the outer verify_task; this unblocks the inner
        # run_merge_verify coroutine so the test can complete cleanly).
        gate_b_release.set()

        # Wait for N+1 to resolve 'done' after cascade re-merge/re-verify.
        # A real timeout now fails loudly by name (task 3477) instead of
        # leaving outcome_b as a confusing None that the assertion below
        # would misreport.
        outcome_b = await _await_outcome(req_b.result, label='N+1 (cascade re-merge/re-verify)')

        await worker.stop()

        # NOTE (task 3477): widened from 5.0s, matching the gate/outcome waits above.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=45.0)

        # ── N+1 must resolve done (cascade re-merged and re-verified) ────────
        assert outcome_b.status == 'done', (
            f'Expected N+1 to resolve "done" after cascade re-merge/re-verify, '
            f'got {outcome_b!r}.'
        )

        # ── Remote cancel must have reached N+1's LIVE id ────────────────────
        # RED: cascade calls _entry.verify_task.cancel() first → live cleared
        #      → cancel_verify (from cancel_and_release) sees dead id →
        #      remote_cancels_while_live stays [].
        # GREEN (step-4): _abort_remote_verify fires before task.cancel() → [1].
        assert fake_remote.remote_cancels_while_live == [1], (
            f'Expected remote_cancels_while_live==[1] (cancel reached live id), '
            f'got {fake_remote.remote_cancels_while_live!r}. '
            'RED: cascade calls verify_task.cancel() before remote cancel → '
            'live cleared → cancel_verify sees dead id → no entry appended. '
            'GREEN (step-4): _abort_remote_verify fires before task.cancel().'
        )


# ---------------------------------------------------------------------------
# step-1856 RED: CASCADE ERROR CONTAINMENT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestCascadeErrorContainment:
    """Per-entry exception in the head-failure cascade MUST NOT kill _verifier_loop.

    RED until 1856 GREEN guards each downstream entry in the cascade for-loop
    with a try/except block, mirroring the four sibling _verifier_loop branches.

    RED markers:
    1. req_c (third request, queued post-cascade) times out — _verifier_loop is
       dead (RuntimeError from _remerge propagated out of the unguarded cascade
       for-loop → _verifier_task terminated; asyncio.gather re-raises →
       worker_task carries RuntimeError).
    2. req_b.result is never set — RuntimeError escaped before the guard could
       resolve it as MergeOutcome('blocked', ...).
    3. worker_task.exception() is the RuntimeError (not None).

    GREEN after 1856: per-entry try/except in the cascade for-loop:
      · CancelledError/KeyboardInterrupt still re-raised (sentinel shutdown works)
      · Any other BaseException → logger.exception; req_b resolved as
        MergeOutcome('blocked', reason='Verifier cascade error: ...');
        lease+slot released exactly once (_entry_released flag); _n_failed=True;
        continue to the next downstream entry.  The loop survives and
        a subsequent request (req_c) dispatches on the local host and
        resolves 'done'.
    """

    async def test_cascade_remerge_error_does_not_kill_verifier_loop(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """RuntimeError from cascade _remerge must not propagate out of _verifier_loop.

        RED: unguarded _remerge raises → RuntimeError exits the cascade for-loop
             → _verifier_loop terminates → req_c queued post-cascade times out.
        GREEN (1856): per-entry try/except resolves req_b 'blocked', continues
             the loop → req_c dispatches on local host and resolves 'done'.
        """
        # ── N's local verify: gated (fails on first call, passes thereafter) ─
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        verifier = _GatedVerifier(
            entered={None: gate_a_entered},
            first=dataclasses.replace(
                fails(category='test_failure', summary='test_failure'),
                release=gate_a_release,
            ),
        )

        # ── N+1's remote verify: gated (passes when gate_b_release is set) ──
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='laptop',
        )

        # ── Branches ─────────────────────────────────────────────────────────
        wt_a = await _make_branch_with_file(git_ops, 'task/cas-a', 'cas_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/cas-b', 'cas_b.py', 'b = 2\n')

        # ── Worker setup ─────────────────────────────────────────────────────
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=verifier)
        _inject_two_host_allocator(worker, gated_remote)

        # Capture initial slot depth so we can verify exact-once release later.
        depth0 = worker._speculation_ledger.slot_available

        event_loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='cas-a', branch=QueuedBranch.parse('task/cas-a', config.git.branch_prefix), worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=event_loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='cas-b', branch=QueuedBranch.parse('task/cas-b', config.git.branch_prefix), worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=event_loop.create_future(), lane='normal',
        )

        # ── Inject killer: _remerge raises RuntimeError for req_b's task_id ─
        # Simulates merge_to_main → _create_merge_worktree failing under ENOSPC.
        _original_remerge = worker._remerge

        async def _killer_remerge(req: Any, started_mono: Any) -> Any:
            if req.task_id == 'cas-b':
                raise RuntimeError(
                    'Failed to create merge worktree: '
                    '/repo/.worktrees/cas-b No space left on device'
                )
            return await _original_remerge(req, started_mono)

        worker._remerge = _killer_remerge  # type: ignore[method-assign]

        worker_task = asyncio.create_task(worker.run())
        try:
            await q.put(req_a)
            await q.put(req_b)

            # Wait for both verifies to enter (true concurrent overlap)
            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='cascade-err-a: gate_a_entered',
            )
            await wait_responsive(
                gate_b_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='cascade-err-b: gate_b_entered',
            )

            # N fails → head-failure cascade fires → _killer_remerge raises for N+1
            gate_a_release.set()

            outcome_a = await wait_responsive(
                req_a.result,
                label='cascade-err-a: MergeOutcome (N head verify fails)',
            )
            assert outcome_a.status not in ('done', 'already_merged'), (
                f'Expected N to fail, got status={outcome_a.status!r}.'
            )
            assert not main_health_probe_spawned(outcome_a), outcome_a.reason

            # Unblock N+1's inner verify coroutine (the cascade already cancelled
            # the outer verify_task; this releases any coroutine still awaiting
            # gate_b so the test completes cleanly on both RED and GREEN paths).
            gate_b_release.set()

            # Wait for req_b to resolve. nominal PRESERVED at 5.0 -- never
            # measured to fail.
            # GREEN: except handler → MergeOutcome('blocked', 'Verifier cascade error: ...')
            # RED:   RuntimeError escaped before handler → req_b.result never
            #        set → wait_responsive fails loudly by label instead of a
            #        silent None.
            outcome_b = await wait_responsive(
                req_b.result,
                timeout=5.0,
                label='cascade-err-b: MergeOutcome (cascade error -> blocked)',
            )

            # (3a) SLOT EXACT-ONCE: check BEFORE stop() (which over-releases for safety).
            # The cascade releases the downstream speculative permit exactly once
            # (in-body, BEFORE the _remerge call); _entry_released=True then prevents
            # the except handler from re-releasing it on the _remerge-raises path.
            #
            # MEASUREMENT NOTE (task 1907): the merger is parked at its speculative
            # look-ahead acquire() — _speculation_depth=1 and the downstream entry
            # held the only permit — so it is a *waiter* on the slot.
            # asyncio.Semaphore.release() hands the freed permit straight to that
            # waiter (_wake_up_first decrements _value back), so a correct single
            # release leaves _value at depth0 - 1, NOT depth0; the merger holds that
            # one look-ahead permit for the rest of the test. A naive unconditional
            # re-release in the except handler (double-release) would push _value up
            # to depth0 — so this assertion still catches it. An under-release (leak)
            # leaves the merger waiter blocked → req_c never dispatched → caught by
            # the loop-survival assertion below.
            expected_slot = depth0 - 1  # one permit held by the merger look-ahead
            assert worker._speculation_ledger.slot_available == expected_slot, (
                f'Expected speculation slot at depth0-1={expected_slot} '
                f'(merger holds one look-ahead permit), '
                f'got {worker._speculation_ledger.slot_available!r}. '
                'A higher value means the except handler over-released '
                '(naive unconditional re-release).'
            )

            # Queue a third request as the loop-survival signal.
            # GREEN: verifier_loop alive → req_c dispatched on local → 'done'.
            # RED:   verifier_loop dead (RuntimeError terminated it) → req_c
            #        never dispatched → wait_responsive fails loudly by label
            #        instead of a silent None.
            wt_c = await _make_branch_with_file(
                git_ops, 'task/cas-c', 'cas_c.py', 'c = 3\n'
            )
            req_c = MergeRequest(
                task_id='cas-c', branch=QueuedBranch.parse('task/cas-c', config.git.branch_prefix), worktree=wt_c,
                pre_rebased=False, task_files=None, module_configs=[],
                config=config, result=event_loop.create_future(), lane='normal',
            )
            await q.put(req_c)

            # NAMED WIDENING 10.0 -> MERGE_RESULT_TIMEOUT (45s): the one
            # MEASURED failure site -- the xdist-load flake this task
            # fixes (see data/verify-logs/2493/attempt-1.orchestrator.
            # summary-20260720T122511_413502Z.json and the task-3980
            # branch observation). suppress(TimeoutError) removed -- a
            # give-up now fails loudly by label instead of leaving
            # outcome_c at a stale None.
            outcome_c = await wait_responsive(
                req_c.result,
                timeout=MERGE_RESULT_TIMEOUT,
                label='cascade-err-c: MergeOutcome (loop-survival signal)',
            )
        finally:
            # esc-3980-4 (see _stop_worker): wait_responsive gives up by
            # raising _pytest.outcomes.Failed, and the two hard asserts above
            # raise too -- so on the old straight-line shape any of those
            # could skip worker.stop() and leak a live worker plus its run
            # task into pytest-asyncio teardown.
            await _stop_worker(worker, worker_task)

        # ── (1) LOOP SURVIVES ────────────────────────────────────────────────
        # A starved-or-hung wait now fails loudly by label at the
        # wait_responsive call itself, above; this assertion only ever fires
        # on a WRONG STATUS.
        assert outcome_c.status == 'done', (
            f'Expected req_c to resolve "done" (loop survived cascade error), '
            f'got {outcome_c!r}. '
            'RED: RuntimeError in _remerge exits the unguarded cascade for-loop '
            '→ _verifier_loop terminates → req_c never dispatched. '
            'GREEN (1856): per-entry try/except contains the error; loop continues.'
        )

        # ── (2) OFFENDING REQ BLOCKED ────────────────────────────────────────
        # A starved-or-hung wait now fails loudly by label at the
        # wait_responsive call itself, above; this assertion only ever fires
        # on a WRONG STATUS.
        assert outcome_b.status == 'blocked', (
            f'Expected cascade error to resolve req_b as "blocked", '
            f'got {outcome_b!r}. '
            'RED: RuntimeError propagated before the handler set req_b.result. '
            'GREEN (1856): except sets MergeOutcome("blocked", "Verifier cascade error: ...").'
        )

        # ── (3b) LEASE RELEASED: cancel_verify called ───────────────────────
        assert gated_remote.cancel_verify.call_count >= 1, (
            f'Expected at least one cancel_verify call for N+1 lease release, '
            f'got call_count={gated_remote.cancel_verify.call_count}.'
        )

        # ── (4) NO EXCEPTION ESCAPES ─────────────────────────────────────────
        # RED: RuntimeError from _remerge propagated out of _verifier_loop;
        #      asyncio.gather() re-raises → worker_task carries the exception.
        assert worker_task.done(), 'Worker task should be done after stop() + await'
        assert worker_task.exception() is None, (
            f'Expected worker task to exit cleanly (exception() is None), '
            f'got {worker_task.exception()!r}. '
            'RED: RuntimeError propagated out of _verifier_loop; '
            'asyncio.gather re-raises → worker_task.exception() is RuntimeError. '
            'GREEN (1856): per-entry guard contains error; loop exits via sentinel.'
        )

    async def test_cascade_cancel_and_release_raises_contained(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
        config: OrchestratorConfig,
    ) -> None:
        """cancel_and_release raising in-body keeps _entry_released=False;
        the not-_entry_released handler must still release lease+slot and
        the loop must survive.

        This covers the secondary error path (cascade body raises before the
        in-body release completes) that the primary test cannot reach.  The
        not-_entry_released branch in the except handler:
          · best-effort cancel_and_release (contextlib.suppress)
          · best-effort _cleanup_owned_merge_worktree (contextlib.suppress)
          · unconditional _speculation_slot.release() for speculative entries

        Without this test a regression — e.g. removing contextlib.suppress from
        the retry, or a double slot-release — would pass the existing suite.
        """
        # ── N's local verify: gated (fails on first call, passes thereafter) ─
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        verifier = _GatedVerifier(
            entered={None: gate_a_entered},
            first=dataclasses.replace(
                fails(category='test_failure', summary='test_failure'),
                release=gate_a_release,
            ),
        )

        # ── N+1's remote verify: gated ────────────────────────────────────────
        gate_b_release = asyncio.Event()
        gate_b_entered = asyncio.Event()
        gated_remote = _gated_runner(
            gate_b_release, gate_b_entered, passed=True, name='laptop',
        )

        # ── Branches ─────────────────────────────────────────────────────────
        wt_a = await _make_branch_with_file(git_ops, 'task/cr-a', 'cr_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/cr-b', 'cr_b.py', 'b = 2\n')

        # ── Worker setup ─────────────────────────────────────────────────────
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, verifier=verifier)
        allocator = _inject_two_host_allocator(worker, gated_remote)

        # Capture initial slot depth for exact-once release assertion.
        depth0 = worker._speculation_ledger.slot_available

        event_loop = asyncio.get_running_loop()
        req_a = MergeRequest(
            task_id='cr-a', branch=QueuedBranch.parse('task/cr-a', config.git.branch_prefix), worktree=wt_a,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=event_loop.create_future(), lane='normal',
        )
        req_b = MergeRequest(
            task_id='cr-b', branch=QueuedBranch.parse('task/cr-b', config.git.branch_prefix), worktree=wt_b,
            pre_rebased=False, task_files=None, module_configs=[],
            config=config, result=event_loop.create_future(), lane='normal',
        )

        # ── Inject: allocator.cancel_and_release raises on its first call ────
        # Simulates a failure during the in-body lease release at the start of
        # the cascade except path.  _entry_released stays False → the except
        # handler takes the not-_entry_released branch: best-effort retry of
        # cancel_and_release + merge_wt cleanup + speculation_slot.release().
        _original_cancel_and_release = allocator.cancel_and_release
        _cancel_call_count: list[int] = [0]

        async def _raising_cancel_and_release(lease: Any, **kwargs: Any) -> Any:
            _cancel_call_count[0] += 1
            if _cancel_call_count[0] == 1:
                raise RuntimeError('Simulated cancel_and_release failure')
            return await _original_cancel_and_release(lease, **kwargs)

        allocator.cancel_and_release = _raising_cancel_and_release  # type: ignore[method-assign]

        worker_task = asyncio.create_task(worker.run())

        try:
            await q.put(req_a)
            await q.put(req_b)

            # Wait for both verifies to enter (true concurrent overlap).
            # NOTE (task 2350): widened from 15.0s -- fixed real-time deadlines
            # starve under heavy shared-host xdist contention even though the
            # underlying cascade logic is correct (timing flake, not a bug).
            await asyncio.wait_for(gate_a_entered.wait(), timeout=45.0)
            await asyncio.wait_for(gate_b_entered.wait(), timeout=45.0)

            # N fails → head-failure cascade fires → in-body cancel_and_release raises.
            gate_a_release.set()

            outcome_a = await asyncio.wait_for(req_a.result, timeout=45.0)
            assert outcome_a.status not in ('done', 'already_merged'), (
                f'Expected N to fail, got status={outcome_a.status!r}.'
            )

            # Unblock N+1's inner verify coroutine so it exits cleanly on both
            # RED and GREEN paths (cascade already cancelled the outer verify_task).
            gate_b_release.set()

            # GREEN: except handler (not-_entry_released branch) →
            #   MergeOutcome('blocked', 'Verifier cascade error: ...').
            outcome_b = await wait_responsive(
                req_b.result,
                timeout=20.0,
                label='cr-b: MergeOutcome (blocked by the contained cascade error)',
            )

            # Loop-survival signal: queue a third request that should dispatch
            # on the local host and resolve "done".
            # NOTE: the slot assertion is placed AFTER outcome_c (below), not here.
            # In this secondary-failure path, set_result is called before the
            # except handler releases the slot (unlike the primary path where the
            # in-body release precedes set_result).  Waiting for outcome_c
            # ensures the cascade iteration has fully completed so the slot is
            # guaranteed to be released before we measure it.
            wt_c = await _make_branch_with_file(
                git_ops, 'task/cr-c', 'cr_c.py', 'c = 3\n'
            )
            req_c = MergeRequest(
                task_id='cr-c', branch=QueuedBranch.parse('task/cr-c', config.git.branch_prefix), worktree=wt_c,
                pre_rebased=False, task_files=None, module_configs=[],
                config=config, result=event_loop.create_future(), lane='normal',
            )
            await q.put(req_c)

            outcome_c = await wait_responsive(
                req_c.result,
                timeout=30.0,
                label='cr-c: MergeOutcome (loop-survival signal)',
            )

            # (3) SLOT EXACT-ONCE: check AFTER outcome_c, BEFORE stop().
            # req_c is non-speculative (no in-flight head), so it never acquires
            # the speculation slot.  Once req_c resolves, the verifier_loop has
            # completed the cascade iteration and the not-_entry_released handler
            # has released the downstream speculative permit exactly once.
            #
            # MEASUREMENT NOTE (task 1907): as in the sibling test, the merger is
            # perpetually a waiter/holder of one speculative look-ahead permit
            # (_speculation_depth=1), so a correct single release leaves _value at
            # depth0 - 1: release() hands the freed permit straight to the merger's
            # pending acquire(). A double-release in the not-_entry_released branch
            # would push _value up to depth0 — caught here.
            expected_slot = depth0 - 1  # one permit held by the merger look-ahead
            assert worker._speculation_ledger.slot_available == expected_slot, (
                f'Expected speculation slot at depth0-1={expected_slot} '
                f'(merger holds one look-ahead permit), '
                f'got {worker._speculation_ledger.slot_available!r}. '
                'A higher value means the not-_entry_released handler over-released.'
            )
        finally:
            await _stop_worker(worker, worker_task, join_timeout=20.0)

        # ── (1) LOOP SURVIVES ────────────────────────────────────────────────
        # A starved-or-hung wait fails loudly by label at the wait_responsive
        # call above; this assertion only ever fires on a WRONG STATUS.
        assert outcome_c.status == 'done', (
            f'Expected req_c to resolve "done" (loop survived cascade error), '
            f'got {outcome_c!r}. '
            'The not-_entry_released except branch must continue the cascade loop.'
        )

        # ── (2) OFFENDING REQ BLOCKED ────────────────────────────────────────
        assert outcome_b.status == 'blocked', (
            f'Expected cascade error to resolve req_b as "blocked", '
            f'got {outcome_b!r}. '
            'Except handler must set MergeOutcome("blocked", "Verifier cascade error: ...").'
        )

        # ── (3b) CANCEL_AND_RELEASE RETRIED ─────────────────────────────────
        # Once in-body (raises) + once in except handler (suppressed).
        assert _cancel_call_count[0] >= 2, (
            f'Expected cancel_and_release called at least twice '
            f'(in-body: raises; not-_entry_released handler: suppressed), '
            f'got {_cancel_call_count[0]} call(s). '
            'The not-_entry_released branch must retry cancel_and_release.'
        )

        # ── (4) NO EXCEPTION ESCAPES ─────────────────────────────────────────
        assert worker_task.done(), 'Worker task should be done after stop() + await'
        assert worker_task.exception() is None, (
            f'Expected worker task to exit cleanly (exception() is None), '
            f'got {worker_task.exception()!r}.'
        )


# ---------------------------------------------------------------------------
# task 3204 step-3 RED / step-4 GREEN: the verify-abort teardown has a single
# CHOKEPOINT — `SpeculativeMergeWorker._teardown_verify_task`.
#
# The remote-abort-BEFORE-cancel ordering is currently stated only in prose on
# `_abort_remote_verify`, and re-typed at six sites. Because the helper always
# cancels AFTER aborting, making it the SOLE caller of `_abort_remote_verify`
# turns that ordering from a convention into a property of the code: there is
# no code path that can abort without also cancelling afterwards.
#
# The guard scans BOTH halves of the recipe — the abort AND
# `verify_task.cancel()` — because sole-callership of the abort alone only
# closes one direction. The converse (a local cancel with no prior remote
# abort, the exact defect task 1757 fixed) needs the cancel half guarded too;
# see `_teardown_chokepoint_offenders`.
#
# The `cancel_verify` call-count/ordering classes above (
# TestRunInflightVerifyRemoteCancelOnAbort, TestStopDrainFiresRemoteCancel,
# TestCascadeFiresRemoteCancel, TestCascadeErrorContainment) are deliberately
# left UNTOUCHED — they are the behavioural regression net that proves the
# step-4 extraction is behaviour-neutral.
# ---------------------------------------------------------------------------

_MQ_CHOKEPOINT_CLASS = 'SpeculativeMergeWorker'
_MQ_TEARDOWN_CHOKEPOINT = '_teardown_verify_task'
_MQ_GUARDED_ABORT = '_abort_remote_verify'
_MQ_GUARDED_CANCEL_RECEIVER = 'verify_task'


def _chokepoint_ranges(tree: ast.Module, cls_name: str, method_name: str) -> list[tuple[int, int]]:
    """Line ranges of every *method_name* defined DIRECTLY on ``class cls_name``.

    Copied from test_lock_release_single_writer_guard.py:134 per the per-file
    duplication convention for orchestrator test helpers (the same recursion
    now also backs the requeue chokepoint guard in
    test_merge_queue_lifecycle_registry.py — duplicated rather than
    cross-imported between test modules, which this repo does not do).

    Tracks the nearest *directly enclosing* ClassDef and resets it to None
    inside a function body, so neither a same-named method on another class
    nor a closure nested within the method can accidentally sanction a site.
    """
    ranges: list[tuple[int, int]] = []

    def visit(node: ast.AST, enclosing_class: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name == method_name and enclosing_class == cls_name:
                    ranges.append((child.lineno, getattr(child, 'end_lineno', child.lineno)))
                visit(child, None)
            else:
                visit(child, enclosing_class)

    visit(tree, None)
    return ranges


def _is_teardown_step_call(node: ast.AST) -> TypeGuard[ast.Call]:
    """True for either HALF of the verify-abort teardown recipe.

    The abort half is any ``<something>._abort_remote_verify(...)``; the name
    exists solely for this recipe, so it needs no receiver anchor. The cancel
    half is anchored on the receiver instead — ``verify_task.cancel()`` or
    ``<expr>.verify_task.cancel()`` — because a bare ``.cancel()`` is far too
    common in this module (loop tasks, heartbeat/reprobe tasks, request
    futures) to guard unanchored. Exact-matching ``cancel`` also keeps the
    neighbouring ``verify_task.cancelled()`` / ``.done()`` / ``.result()``
    peeks out of the offender list.

    Typed as a ``TypeGuard`` (a plain ``bool`` at runtime) so a True result
    narrows the walked ``ast.AST`` to ``ast.Call`` at the call site — the
    offender list reads ``.lineno``, which lives on ``ast.expr``, not on the
    ``ast.AST`` base that :func:`ast.walk` yields.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr == _MQ_GUARDED_ABORT:
        return True
    if node.func.attr != 'cancel':
        return False
    receiver = node.func.value
    if isinstance(receiver, ast.Name):
        return receiver.id == _MQ_GUARDED_CANCEL_RECEIVER
    if isinstance(receiver, ast.Attribute):
        return receiver.attr == _MQ_GUARDED_CANCEL_RECEIVER
    return False


def _teardown_chokepoint_offenders(source: str) -> list[str]:
    """Every verify-abort-teardown call site outside the teardown helper, as
    ``'<enclosing_fn>:<lineno> <call_src>'``.

    Scans BOTH halves of the recipe — ``self._abort_remote_verify(...)`` and
    ``<...>verify_task.cancel()`` — exactly as the sibling requeue guard in
    test_merge_queue_lifecycle_registry.py scans both ``_queue.put_nowait``
    and ``on_requeued``. Scanning only the abort half would enforce "every
    abort is followed by a cancel" while leaving the CONVERSE — the actual
    defect class task 1757 fixed — unguarded: a ``verify_task.cancel()`` fired
    without a prior remote abort silently orphans the remote verify-merge
    process, because the verify coroutine's finally clause has by then cleared
    ``_inflight_request_id``. Guarding both halves is what makes the pairing
    AND the ordering properties of the code (task 3204 amend, reviewer finding
    test-coverage).

    ``ast.Call``-only, so ``_abort_remote_verify``'s own ``async def``
    DEFINITION is not reported — the invariant is about call sites, not about
    the implementation they call. Pure (takes source text, not a path) so the
    scanner self-tests can drive it on synthetic modules; returns ``[]`` rather
    than raising on source that cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    ranges = _chokepoint_ranges(tree, _MQ_CHOKEPOINT_CLASS, _MQ_TEARDOWN_CHOKEPOINT)

    parent: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node

    def _enclosing_fn(node: ast.AST) -> str:
        cur: ast.AST | None = node
        while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cur = parent.get(id(cur))
        return getattr(cur, 'name', '<module>')

    calls = [node for node in ast.walk(tree) if _is_teardown_step_call(node)]
    offenders = [
        f'{_enclosing_fn(call)}:{call.lineno} {ast.unparse(call)}'
        for call in calls
        if not any(start <= call.lineno <= end for start, end in ranges)
    ]
    # ast.walk is breadth-first; sort so the offender list reads top-to-bottom.
    offenders.sort(key=lambda entry: int(entry.split(':', 1)[1].split(' ', 1)[0]))
    return offenders


def _never_finishes() -> asyncio.Task:
    """A real, never-completing asyncio.Task to stand in for a live verify."""
    return asyncio.get_running_loop().create_task(asyncio.Event().wait())


class TestVerifyTeardownChokepoint:
    """``_teardown_verify_task`` is the SOLE caller of BOTH teardown steps —
    ``_abort_remote_verify`` and ``verify_task.cancel()`` (task 3204 step-3
    RED; the cancel half added by the step-3 amend).

    Sync-only, and deliberately NOT marked ``@pytest.mark.asyncio``: this
    repo runs pytest warnings-as-errors, and pytest-asyncio raises on a sync
    test carrying that mark. The async helper-contract tests live in
    :class:`TestVerifyTeardownHelper` below.
    """

    # ── scanner self-tests: the ratchet must never pass vacuously ──────────

    def test_scanner_accepts_a_compliant_chokepoint(self) -> None:
        """POSITIVE control: both steps are inside the helper -> no offenders."""
        src = (
            'class SpeculativeMergeWorker:\n'
            '    async def _teardown_verify_task(self, lease, verify_task, task_id):\n'
            '        if lease is not None:\n'
            '            await self._abort_remote_verify(lease, task_id)\n'
            '        if verify_task is not None:\n'
            '            verify_task.cancel()\n'
        )
        assert _teardown_chokepoint_offenders(src) == []

    def test_scanner_flags_an_abort_bypass_outside_the_helper(self) -> None:
        """SYNTHETIC BYPASS (abort half): an abort typed at a call site is flagged."""
        src = (
            'class SpeculativeMergeWorker:\n'
            '    async def _teardown_verify_task(self, lease, verify_task, task_id):\n'
            '        await self._abort_remote_verify(lease, task_id)\n'
            '        verify_task.cancel()\n'
            '\n'
            '    async def _sneaky(self, entry):\n'
            '        await self._abort_remote_verify(entry.lease, entry.task_id)\n'
        )
        offenders = _teardown_chokepoint_offenders(src)
        assert len(offenders) == 1, offenders
        assert offenders[0].startswith('_sneaky:'), offenders

    def test_scanner_flags_a_cancel_bypass_outside_the_helper(self) -> None:
        """SYNTHETIC BYPASS (cancel half): this is the task-1757 defect class —
        a local cancel with NO prior remote abort orphans the remote verify.

        Both receiver shapes must be caught: the bare ``verify_task`` local
        the helper itself uses, and the ``<entry>.verify_task`` attribute the
        call sites hold.
        """
        src = (
            'class SpeculativeMergeWorker:\n'
            '    async def _teardown_verify_task(self, lease, verify_task, task_id):\n'
            '        await self._abort_remote_verify(lease, task_id)\n'
            '        verify_task.cancel()\n'
            '\n'
            '    async def _sneaky(self, entry, verify_task):\n'
            '        entry.verify_task.cancel()\n'
            '        verify_task.cancel()\n'
        )
        offenders = _teardown_chokepoint_offenders(src)
        assert len(offenders) == 2, offenders
        assert all(entry.startswith('_sneaky:') for entry in offenders), offenders

    def test_scanner_ignores_unrelated_cancels_and_verify_task_peeks(self) -> None:
        """ANCHORING: the cancel half is receiver-anchored on ``verify_task``.

        merge_queue.py cancels plenty of OTHER tasks (merger/verifier loop
        tasks, heartbeat, reprobe, request futures) and peeks at a verify
        task's ``.cancelled()`` / ``.done()`` / ``.result()`` right next to the
        teardown — none of those are teardown steps, and flagging them would
        make the ratchet unsatisfiable.
        """
        src = (
            'class SpeculativeMergeWorker:\n'
            '    async def _elsewhere(self, entry, t):\n'
            '        t.cancel()\n'
            '        self._heartbeat_task.cancel()\n'
            '        entry.item.request.result.cancel()\n'
            '        if entry.verify_task.done() and not entry.verify_task.cancelled():\n'
            '            entry.verify_task.result()\n'
        )
        assert _teardown_chokepoint_offenders(src) == []

    def test_scanner_flags_a_same_named_method_on_another_class(self) -> None:
        """CLASS-QUALIFICATION: a ``_teardown_verify_task`` on some OTHER class
        does not sanction its body.
        """
        src = (
            'class SomethingElse:\n'
            '    async def _teardown_verify_task(self, lease, task_id):\n'
            '        await self._abort_remote_verify(lease, task_id)\n'
        )
        offenders = _teardown_chokepoint_offenders(src)
        assert len(offenders) == 1, offenders
        assert offenders[0].startswith('_teardown_verify_task:'), offenders

    def test_scanner_ignores_docstring_and_comment_mentions(self) -> None:
        """AST-not-regex: prose naming a guarded symbol is not a call site.

        merge_queue.py genuinely carries such mentions (the module-level
        teardown narrative and the back-pointer on ``_abort_remote_verify``,
        which spells ``verify_task.cancel()`` in its own docstring), so a
        grep-based guard would flag them forever. The bare ``async def``
        definition must not be flagged either.
        """
        src = (
            '"""Prose naming self._abort_remote_verify(lease, task_id)."""\n'
            '\n'
            '# self._abort_remote_verify(lease, task_id) in a comment.\n'
            'class SpeculativeMergeWorker:\n'
            '    async def _abort_remote_verify(self, lease, task_id):\n'
            '        """See self._abort_remote_verify, called before '
            'verify_task.cancel()."""\n'
            '        return None\n'
        )
        assert _teardown_chokepoint_offenders(src) == []

    def test_scanner_returns_empty_on_unparseable_source(self) -> None:
        """SYNTAX ERROR: return [] rather than raising."""
        assert _teardown_chokepoint_offenders('async def broken(:\n') == []

    # ── the ratchet ────────────────────────────────────────────────────────

    def test_teardown_steps_are_only_called_from_the_teardown_helper(self) -> None:
        """STRUCTURAL: sole-callership of BOTH steps is what makes the pairing
        and the ordering unviolatable.

        ``_abort_remote_verify`` must fire BEFORE ``verify_task.cancel()``,
        because the verify coroutine's finally clause clears
        ``_inflight_request_id`` on cancellation, which makes a later
        ``cancel_verify()`` a silent no-op that orphans the remote
        verify-merge process (task 1757). Since the helper always cancels
        after aborting, a caller that cannot abort on its own cannot get the
        order wrong — and since it cannot cancel on its own either, it cannot
        skip the abort altogether, which is the direction the abort-only
        form of this guard left open.
        """
        import orchestrator.merge_lane.worker as _mq  # module-local import convention

        offenders = _teardown_chokepoint_offenders(Path(_mq.__file__).read_text())
        assert offenders == [], (
            f'`{_MQ_GUARDED_ABORT}` and `{_MQ_GUARDED_CANCEL_RECEIVER}.cancel()` '
            f'must only be called from `{_MQ_CHOKEPOINT_CLASS}.'
            f'{_MQ_TEARDOWN_CHOKEPOINT}`, which always aborts the remote verify '
            'FIRST and cancels the local task second — that sole-callership is '
            'what makes the abort-before-cancel ordering a property of the code '
            'rather than a convention re-typed at every site. Route these '
            'through `await self._teardown_verify_task(lease, verify_task, '
            f'task_id)` instead. Offenders: {offenders!r}'
        )


@pytest.mark.asyncio
class TestVerifyTeardownHelper:
    """``_teardown_verify_task`` aborts the remote verify, THEN cancels and
    awaits the local task, no-oping either step whose argument is None
    (task 3204 step-3 RED).
    """

    async def test_aborts_before_cancelling_the_local_task(self, git_ops: GitOps) -> None:
        """ORDER: the remote abort is awaited while verify_task is still live."""
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        calls: list[str] = []
        verify_task = _never_finishes()
        cancel_requested_at_abort: int | None = None

        runner = MagicMock()

        async def _cancel_verify() -> int:
            nonlocal cancel_requested_at_abort
            calls.append('abort')
            # OBSERVE ONLY — never assert in here (task 3204 amend, reviewer
            # finding test-quality). This coroutine is awaited from inside
            # `_abort_remote_verify`'s `try/except Exception` block, and an
            # AssertionError IS an Exception: an inline assertion is swallowed
            # and merely logged as a WARNING, so it can never fail the test.
            # `.cancelled()` would also be the wrong probe — cancellation is
            # not DELIVERED until the task next runs, so it reads False right
            # after `.cancel()`. `.cancelling()` counts cancel REQUESTS and is
            # non-zero the moment `.cancel()` returns, which is exactly the
            # ordering fact this test is about: the remote cancel must land
            # while the local task is still live, or it arrives after the
            # verify coroutine's finally has cleared _inflight_request_id.
            cancel_requested_at_abort = verify_task.cancelling()
            return 0

        runner.cancel_verify = _cancel_verify
        lease = HostLease(name='remote', runner=runner, is_local=False)

        real_cancel = verify_task.cancel

        def _spy_cancel(*a: Any, **k: Any) -> bool:
            calls.append('cancel')
            return real_cancel(*a, **k)

        verify_task.cancel = _spy_cancel  # type: ignore[method-assign]

        await worker._teardown_verify_task(lease, verify_task, 'df3204-order')

        assert calls == ['abort', 'cancel'], calls
        # Asserted OUT HERE, where nothing can swallow it.
        assert cancel_requested_at_abort == 0, (
            f'the remote abort must land while the local verify task is still '
            f'live; verify_task.cancelling() was {cancel_requested_at_abort!r} '
            f'when cancel_verify() ran (None = the abort never fired at all), '
            f'so the remote cancel would be a no-op against an already-cleared '
            f'_inflight_request_id'
        )
        assert verify_task.done(), 'the helper must AWAIT the cancelled task'

    async def test_awaits_the_cancelled_task(self, git_ops: GitOps) -> None:
        """AWAITED: no pending task is left behind after the helper returns."""
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        verify_task = _never_finishes()
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)

        await worker._teardown_verify_task(lease, verify_task, 'df3204-awaited')

        assert verify_task.done()
        assert verify_task.cancelled()

    async def test_none_lease_skips_only_the_remote_abort(self, git_ops: GitOps) -> None:
        """``lease=None`` no-ops THAT STEP internally — the caller must not
        have to branch (task 3204 requirement 2) — but still cancels + awaits.
        """
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker._abort_remote_verify = AsyncMock()  # type: ignore[method-assign]
        verify_task = _never_finishes()

        await worker._teardown_verify_task(None, verify_task, 'df3204-nolease')

        worker._abort_remote_verify.assert_not_awaited()
        assert verify_task.cancelled()

    async def test_none_verify_task_skips_only_the_cancel(self, git_ops: GitOps) -> None:
        """``verify_task=None`` no-ops THAT STEP — the abort still fires and
        there is no AttributeError.
        """
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        runner = MagicMock()
        runner.cancel_verify = AsyncMock(return_value=0)
        lease = HostLease(name='remote', runner=runner, is_local=False)

        await worker._teardown_verify_task(lease, None, 'df3204-notask')

        runner.cancel_verify.assert_awaited_once()

    async def test_both_none_is_a_clean_no_op(self, git_ops: GitOps) -> None:
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker._abort_remote_verify = AsyncMock()  # type: ignore[method-assign]

        await worker._teardown_verify_task(None, None, 'df3204-both-none')

        worker._abort_remote_verify.assert_not_awaited()

    async def test_swallows_an_exception_from_awaiting_the_verify_task(
        self, git_ops: GitOps,
    ) -> None:
        """The ``await verify_task`` is unconditionally suppressed — as it is at
        all six sites today — so a verify that died with an exception cannot
        derail the teardown.
        """
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())

        async def _boom() -> None:
            raise RuntimeError('verify blew up')

        verify_task = asyncio.get_running_loop().create_task(_boom())
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)

        await worker._teardown_verify_task(lease, verify_task, 'df3204-raiser')

        assert verify_task.done()

    async def test_shutdown_defensive_swallows_a_raising_abort(
        self, git_ops: GitOps,
    ) -> None:
        """``shutdown_defensive=True``: a BaseException out of the remote abort
        (e.g. a SIGTERM-driven CancelledError) must not abort the drain loop,
        and the teardown still proceeds to cancel.

        This is the one genuine per-site difference: the two stop() drain sites
        wrap the abort in ``contextlib.suppress(BaseException)``.
        """
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker._abort_remote_verify = AsyncMock(  # type: ignore[method-assign]
            side_effect=asyncio.CancelledError(),
        )
        verify_task = _never_finishes()
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)

        await worker._teardown_verify_task(
            lease, verify_task, 'df3204-defensive', shutdown_defensive=True,
        )

        worker._abort_remote_verify.assert_awaited_once()
        assert verify_task.cancelled(), 'must still cancel after swallowing the abort'

    async def test_default_lets_a_raising_abort_propagate(self, git_ops: GitOps) -> None:
        """DEFAULT (``shutdown_defensive=False``): the exception PROPAGATES.

        The head-failure cascade deliberately does not suppress — a
        CancelledError there must stop the verifier loop, which is correct
        behaviour. Collapsing the flag either way would be a silent behaviour
        change on one path or the other.
        """
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())
        worker._abort_remote_verify = AsyncMock(  # type: ignore[method-assign]
            side_effect=asyncio.CancelledError(),
        )
        verify_task = _never_finishes()
        lease = HostLease(name='local', runner=MagicMock(), is_local=True)

        with pytest.raises(asyncio.CancelledError):
            await worker._teardown_verify_task(lease, verify_task, 'df3204-propagates')

        # Clean up the task this deliberately-aborted teardown left behind.
        verify_task.cancel()
        with contextlib.suppress(BaseException):
            await verify_task


# ---------------------------------------------------------------------------
# task 4164: stop()'s shutdown-drain teardowns must pass shutdown_defensive=True
# ---------------------------------------------------------------------------

_MQ_SHUTDOWN_DRAIN_METHODS: tuple[str, ...] = ('stop',)
_MQ_SHUTDOWN_DEFENSIVE_KWARG = 'shutdown_defensive'
_MQ_SHUTDOWN_DRAIN_SITE_FLOOR = 2


def _shutdown_drain_teardown_calls(tree: ast.Module) -> list[tuple[str, ast.Call]]:
    """Every ``<x>._teardown_verify_task(...)`` call lexically inside one of
    the ``_MQ_SHUTDOWN_DRAIN_METHODS`` defined directly on
    ``SpeculativeMergeWorker``, paired with that method's name, in source order.
    """
    teardown_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == _MQ_TEARDOWN_CHOKEPOINT
    ]
    drain_calls = [
        (method, call)
        for method in _MQ_SHUTDOWN_DRAIN_METHODS
        for start, end in _chokepoint_ranges(tree, _MQ_CHOKEPOINT_CLASS, method)
        for call in teardown_calls
        if start <= call.lineno <= end
    ]
    return sorted(drain_calls, key=lambda pair: pair[1].lineno)


def _passes_literal_shutdown_defensive(call: ast.Call) -> bool:
    """True only for a literal ``shutdown_defensive=True`` keyword — the one
    spelling whose value is provable from source text alone.
    """
    return any(
        kw.arg == _MQ_SHUTDOWN_DEFENSIVE_KWARG
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is True
        for kw in call.keywords
    )


def _undefended_shutdown_teardown_offenders(source: str) -> list[str]:
    """Every shutdown-drain teardown call that does not pass a literal
    ``shutdown_defensive=True``, as ``'<method>:<lineno> <call_src>'``.

    Pure (takes source text, not a path) like
    :func:`_teardown_chokepoint_offenders`, so the self-tests can drive it on
    synthetic modules; returns ``[]`` rather than raising on unparseable source.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        f'{method}:{call.lineno} {ast.unparse(call)}'
        for method, call in _shutdown_drain_teardown_calls(tree)
        if not _passes_literal_shutdown_defensive(call)
    ]


class TestShutdownDrainTeardownIsDefensive:
    """Every ``_teardown_verify_task`` call inside ``SpeculativeMergeWorker.stop``
    passes ``shutdown_defensive=True`` (task 4164). Without it a SIGTERM-driven
    CancelledError escaping ``_abort_remote_verify`` aborts the drain loop and
    leaves the remaining ``_inflight`` entries un-drained — the task-1757
    orphaned-remote-verify class. The helper's two arms are pinned separately
    by :class:`TestVerifyTeardownHelper`, and the drain behaviour end to end by
    :class:`TestStopDrainSurvivesARaisingRemoteAbort`; this scan is the cheap
    signal that names the offending site. Sync-only for the same reason as
    :class:`TestVerifyTeardownChokepoint`.
    """

    # ── scanner self-tests: the ratchet must never pass vacuously ──────────

    def test_scanner_accepts_drain_sites_that_pass_the_flag(self) -> None:
        """POSITIVE control: both drain sites pass the flag -> no offenders,
        and both calls were actually seen.
        """
        src = (
            'class SpeculativeMergeWorker:\n'
            '    async def stop(self):\n'
            '        if self.head is not None:\n'
            '            await self._teardown_verify_task(a, b, c, shutdown_defensive=True)\n'
            '        for entry in self.inflight:\n'
            '            await self._teardown_verify_task(a, b, c, shutdown_defensive=True)\n'
        )
        assert _undefended_shutdown_teardown_offenders(src) == []
        assert len(_shutdown_drain_teardown_calls(ast.parse(src))) == 2

    def test_scanner_flags_a_drain_site_that_omits_the_flag(self) -> None:
        """SYNTHETIC BYPASS: the flagless second call (line 6) is the one flagged."""
        src = (
            'class SpeculativeMergeWorker:\n'
            '    async def stop(self):\n'
            '        if self.head is not None:\n'
            '            await self._teardown_verify_task(a, b, c, shutdown_defensive=True)\n'
            '        for entry in self.inflight:\n'
            '            await self._teardown_verify_task(a, b, c)\n'
        )
        offenders = _undefended_shutdown_teardown_offenders(src)
        assert len(offenders) == 1, offenders
        assert offenders[0].startswith('stop:6 '), offenders

    def test_scanner_flags_anything_but_a_literal_true(self) -> None:
        """LITERAL-ONLY: ``False``, a variable, or a ``**kwargs`` splat cannot be
        proven True from source text, so each is an offender.
        """
        src = (
            'class SpeculativeMergeWorker:\n'
            '    async def stop(self, defensive, kwargs):\n'
            '        await self._teardown_verify_task(a, b, c, shutdown_defensive=False)\n'
            '        await self._teardown_verify_task(a, b, c, shutdown_defensive=defensive)\n'
            '        await self._teardown_verify_task(a, b, c, **kwargs)\n'
        )
        offenders = _undefended_shutdown_teardown_offenders(src)
        assert len(offenders) == 3, offenders

    def test_scanner_sees_a_teardown_nested_in_a_wrapper_call(self) -> None:
        """NESTING: a teardown wrapped in shield/wait_for is still a drain site."""
        src = (
            'class SpeculativeMergeWorker:\n'
            '    async def stop(self):\n'
            '        await asyncio.wait_for('
            'asyncio.shield(self._teardown_verify_task(a, b, c)), timeout=1)\n'
        )
        offenders = _undefended_shutdown_teardown_offenders(src)
        assert len(offenders) == 1, offenders

    def test_scanner_ignores_teardown_calls_outside_the_drain_path(self) -> None:
        """SCOPE: the cascade and abort triggers deliberately propagate by
        default (see ``test_default_lets_a_raising_abort_propagate``).
        """
        src = (
            'class SpeculativeMergeWorker:\n'
            '    async def _verifier_loop(self):\n'
            '        await self._teardown_verify_task(a, b, c)\n'
            '\n'
            '    async def _run_inflight_verify(self):\n'
            '        await self._teardown_verify_task(a, b, c)\n'
        )
        assert _undefended_shutdown_teardown_offenders(src) == []

    def test_scanner_ignores_a_stop_on_another_class(self) -> None:
        """CLASS-QUALIFICATION: only ``SpeculativeMergeWorker.stop`` is scanned."""
        src = (
            'class SomethingElse:\n'
            '    async def stop(self):\n'
            '        await self._teardown_verify_task(a, b, c)\n'
        )
        assert _undefended_shutdown_teardown_offenders(src) == []
        assert _shutdown_drain_teardown_calls(ast.parse(src)) == []

    def test_scanner_returns_empty_on_unparseable_source(self) -> None:
        """SYNTAX ERROR: return [] rather than raising."""
        assert _undefended_shutdown_teardown_offenders('async def broken(:\n') == []

    # ── the ratchet ────────────────────────────────────────────────────────

    def test_stop_drain_teardowns_all_pass_shutdown_defensive(self) -> None:
        """STRUCTURAL: deleting the flag at either stop() drain site must fail
        here, not pass silently.
        """
        source_file = inspect.getsourcefile(SpeculativeMergeWorker)
        assert source_file is not None
        src = Path(source_file).read_text()

        drain_calls = _shutdown_drain_teardown_calls(ast.parse(src))
        assert len(drain_calls) >= _MQ_SHUTDOWN_DRAIN_SITE_FLOOR, (
            f'expected at least {_MQ_SHUTDOWN_DRAIN_SITE_FLOOR} '
            f'`{_MQ_TEARDOWN_CHOKEPOINT}` calls inside {_MQ_SHUTDOWN_DRAIN_METHODS!r} '
            f'on `{_MQ_CHOKEPOINT_CLASS}`, found {len(drain_calls)}. If a drain '
            'site moved out of stop(), add its method to '
            '`_MQ_SHUTDOWN_DRAIN_METHODS`, or this ratchet goes vacuous.'
        )

        offenders = _undefended_shutdown_teardown_offenders(src)
        assert offenders == [], (
            f'every `{_MQ_TEARDOWN_CHOKEPOINT}` call on the shutdown drain path '
            f'must pass `{_MQ_SHUTDOWN_DEFENSIVE_KWARG}=True`: otherwise a '
            'SIGTERM-driven CancelledError escaping the remote abort aborts the '
            'drain loop and leaves the remaining in-flight verifies un-drained, '
            'orphaning their remote verify-merge processes (task 1757). Pass '
            f'`{_MQ_SHUTDOWN_DEFENSIVE_KWARG}=True` literally. '
            f'Offenders: {offenders!r}'
        )


def _remote_whose_abort_raises(
    name: str, *, entered: asyncio.Event, reaped: asyncio.Event,
) -> MagicMock:
    """A fake remote whose verify hangs until cancelled and whose
    ``cancel_verify`` raises ``CancelledError``, as when a SIGTERM lands
    mid-shutdown. *entered* is set when its verify starts, *reaped* when that
    verify exits by any route.
    """

    async def _verify_until_cancelled(*args: Any, **kwargs: Any) -> VerifyResult:
        entered.set()
        try:
            await asyncio.Event().wait()
            return _mock_verify_result(True)
        finally:
            reaped.set()

    runner = _make_fake_remote(name)
    runner.run_merge_verify = AsyncMock(side_effect=_verify_until_cancelled)
    runner.cancel_verify = AsyncMock(side_effect=asyncio.CancelledError)
    return runner


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)
class TestStopDrainSurvivesARaisingRemoteAbort:
    """BEHAVIOUR behind :class:`TestShutdownDrainTeardownIsDefensive` (task
    4164 amend): when every remote abort raises ``CancelledError``, stop()
    still drains EVERY in-flight verify. Driven through the worker's public
    surface: two remote hosts under ``prefer_remote`` put both in-flight
    entries on remote leases, so both stop() drain sites, the finalizing head
    and then the ``_inflight`` loop, meet a raising abort.
    """

    async def test_stop_drains_every_inflight_verify_when_remote_aborts_raise(
        self,
        git_ops: GitOps,
        git_repo: Path,
        git_config: GitConfig,
    ) -> None:
        entered_a, reaped_a = asyncio.Event(), asyncio.Event()
        entered_b, reaped_b = asyncio.Event(), asyncio.Event()
        remote_a = _remote_whose_abort_raises('remote-a', entered=entered_a, reaped=reaped_a)
        remote_b = _remote_whose_abort_raises('remote-b', entered=entered_b, reaped=reaped_b)

        class _AlsoManagesRemoteB(HostAllocator):
            def __init__(self, remote_runners: list, *, quarantine: set[str] | None = None) -> None:
                super().__init__([*remote_runners, remote_b], quarantine=quarantine)

        config = lane_scene_config(git_repo, git_config, verify_host_policy='prefer_remote')
        wt_a = await _make_branch_with_file(git_ops, 'task/drain-a', 'drain_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/drain-b', 'drain_b.py', 'b = 2\n')
        req_a = _make_request('drain-a', 'task/drain-a', wt_a, config)
        req_b = _make_request('drain-b', 'task/drain-b', wt_b, config)

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        _inject_two_host_allocator(worker, remote_a, allocator_cls=_AlsoManagesRemoteB)
        worker_task = asyncio.create_task(worker.run())

        try:
            await q.put(req_a)
            await q.put(req_b)
            await wait_responsive(
                entered_a.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='drain-a: remote verify entered',
            )
            await wait_responsive(
                entered_b.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='drain-b: remote verify entered',
            )
        finally:
            await _stop_worker(worker, worker_task)

        remote_a.cancel_verify.assert_awaited()
        remote_b.cancel_verify.assert_awaited()
        shutdown = MergeOutcome('blocked', reason=MERGE_WORKER_SHUTDOWN_REASON)
        outcomes = {
            req.task_id: req.result.result() if req.result.done() else 'PENDING'
            for req in (req_a, req_b)
        }
        assert outcomes == {'drain-a': shutdown, 'drain-b': shutdown}, (
            f'stop() must resolve every in-flight request with the shutdown '
            f'outcome even when the remote abort raises; got {outcomes!r}'
        )
        assert reaped_a.is_set() and reaped_b.is_set(), (
            f'stop() must cancel every in-flight remote verify even when the '
            f'remote abort raises; reaped: a={reaped_a.is_set()}, '
            f'b={reaped_b.is_set()}'
        )


# ---------------------------------------------------------------------------
# task 2063: _redispatch-parked speculative permit accounting (signal test)
# ---------------------------------------------------------------------------


class _FakeEscalationQueue:
    """Minimal fake escalation queue (per-file duplication convention — see
    the identical helper in test_merge_queue_resource_audit.py).
    """

    def __init__(self) -> None:
        self._open_l1 = False
        self._seq = 0
        self.submitted: list[Any] = []

    def has_open_l1(self, task_id: str) -> bool:  # noqa: ARG002
        return self._open_l1

    def make_id(self, task_id: str) -> str:
        self._seq += 1
        return f'esc-{self._seq}'

    def submit(self, esc: Any) -> None:
        self.submitted.append(esc)


class _DriftCheckTakesRemoteAllocator(HostAllocator):
    """A HostAllocator where a concurrent drift check wins the remote slot.

    The first ``acquire()`` made while local is busy finds the remote already
    taken through the public ``acquire_remote()`` — what
    ``merge_drift.py::_run_drift_check`` does from its own task — so it
    returns None after the caller's ``free_host_count()`` guard passed.
    """

    drift_check_lease: HostLease | None = None

    async def acquire(self, local_factory: Any, *, policy: Any) -> HostLease | None:
        if self.drift_check_lease is None and self.is_busy(self.local_name):
            self.drift_check_lease = self.acquire_remote()
        return await super().acquire(local_factory, policy=policy)


@pytest.mark.asyncio
@pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)  # task 3492: a deliberate over-mark -- a loop.time()-deadline poll loop here is invisible to _wait_budget_audit.py::item_wait_budgets
class TestRedispatchSpeculativeConservation:
    """Signal/e2e regression lock (task 2063): a genuinely speculative item
    parked on ``_redispatch`` must stay counted by
    ``_inflight_speculative_count()`` throughout the REAL merger/verifier
    pipeline — not just in the hand-constructed unit tests in
    test_merge_queue_resource_audit.py.

    MECHANISM — the production route to a speculative item on
    ``_redispatch``: ``_dispatch_item`` passes its ``free_host_count()``
    guard, then a concurrent non-loop acquirer takes the free slot before its
    ``acquire()``.  That acquirer is
    ``orchestrator/src/orchestrator/merge_drift.py::_run_drift_check``, which
    takes slots from its own task through the public ``acquire_local`` /
    ``acquire_remote``.  :class:`_DriftCheckTakesRemoteAllocator` models it
    deterministically: the first ``acquire()`` made while local is busy finds
    the remote already taken, so the dispatch returns None and B parks,
    STILL ``speculative=True``.

    SCENARIO: A's local verify is gated then FAILS. While gated, the merger
    speculatively merges B against A's tentative commit and hands it to the
    verifier; dispatching B loses the remote to the modelled drift check and
    B parks on ``_redispatch``. When A's gate releases (A fails), the
    existing 'previous_failed' chain-invalidation path re-merges B against
    actual main (speculative -> False) and re-dispatches it; B's second
    local verify passes -> 'done'.
    """

    async def test_speculative_redispatch_item_stays_counted_through_real_pipeline(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The I4 speculation-slot identity holds at every heartbeat-observable
        point while a genuinely speculative item sits parked on
        ``_redispatch``, and no ``merge_resource_leak`` escalation ever fires.

        Also (task 3477): the disposition classifier must not silently
        degrade to its fail-open path while processing this test's failing
        VerifyResult double — see the caplog guard at the end of this test.
        """
        caplog.set_level(logging.WARNING, logger='orchestrator.merge_lane')
        gate_a_release = asyncio.Event()
        gate_a_entered = asyncio.Event()
        verifier = _GatedVerifier(
            entered={None: gate_a_entered},
            first=dataclasses.replace(
                fails(category='test_failure', summary='test_failure'),
                release=gate_a_release,
            ),
        )

        wt_a = await _make_branch_with_file(git_ops, 'task/rc-a', 'rc_a.py', 'a = 1\n')
        wt_b = await _make_branch_with_file(git_ops, 'task/rc-b', 'rc_b.py', 'b = 2\n')

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        fake_esc_queue = _FakeEscalationQueue()
        worker = SpeculativeMergeWorker(
            git_ops, q, speculation_depth=2, escalation_queue=fake_esc_queue,
            verifier=verifier,
        )
        # Fast/deterministic: alarm on the very first violating heartbeat
        # (documented test override — merge_queue.py's
        # RESOURCE_AUDIT_ESCALATION_STREAK class-attr docstring).
        worker.RESOURCE_AUDIT_ESCALATION_STREAK = 1

        fake_remote = _make_fake_remote('laptop')
        _inject_two_host_allocator(
            worker, fake_remote, allocator_cls=_DriftCheckTakesRemoteAllocator,
        )

        recorded_violations: list[list[str]] = []
        _orig_violations = worker.speculation_accounting_violations

        def _traced_violations() -> list[str]:
            result = _orig_violations()
            recorded_violations.append(result)
            return result

        worker.speculation_accounting_violations = _traced_violations  # type: ignore[method-assign]

        req_a = _make_request('rc-a', 'task/rc-a', wt_a, config)
        req_b = _make_request('rc-b', 'task/rc-b', wt_b, config)

        worker_task = asyncio.create_task(worker.run())
        try:
            await q.put(req_a)
            await q.put(req_b)

            await wait_responsive(
                gate_a_entered.wait(),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='rc-a: local verify entered',
            )

            # Bounded poll (not a flat sleep): the merger's real git worktree
            # + commit calls have variable latency, so wait until B has
            # actually landed on _redispatch rather than guessing a delay.
            deadline = asyncio.get_running_loop().time() + 15.0
            while not worker._redispatch and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)

            parked = list(worker._redispatch)
            assert len(parked) == 1 and parked[0].speculative, (
                f'Expected exactly one speculative item parked on _redispatch '
                f'while A is gated (proves the fix is actually exercised); '
                f'got {parked!r}.'
            )
            inflight_speculative = worker.snapshot()['speculation']['inflight_speculative']
            assert inflight_speculative == 1, (
                f'the parked speculative B must stay counted as verifier-owned; '
                f'snapshot speculation.inflight_speculative={inflight_speculative}'
            )

            # NOTE(test-determinism): from here on this test calls
            # worker._maybe_log_queue_heartbeat(...) directly from the test
            # coroutine while worker_task (the real worker.run() loop) keeps
            # running concurrently. That manual interleaving is only safe
            # because _maybe_log_queue_heartbeat — and the
            # _check_resource_audit / snapshot() calls it makes — are fully
            # synchronous (no `await` anywhere in that call chain), so each
            # call is atomic with respect to worker_task's own event-loop
            # iterations and can never observe state torn mid-dispatch. If
            # any of those methods ever gains an `await`, this interleaving
            # could start observing torn state and flake intermittently; a
            # future async refactor of the heartbeat/audit path must
            # revisit this test.
            #
            # Sample the audit at several heartbeat-observable points while
            # the speculative item sits parked (the multi-heartbeat window
            # the original false positive persisted across). Check the
            # recorded_violations delta AROUND EACH INDIVIDUAL call (rather
            # than leaning on the blanket non-empty check further below,
            # which the explicit speculation_accounting_violations() call a
            # few lines down would satisfy all on its own) so the test
            # proves _maybe_log_queue_heartbeat itself drives the audit
            # (via _check_resource_audit and snapshot()'s own
            # resource_audit key) on every one of these calls while B is
            # parked — not merely that *some* call somewhere recorded a
            # sample. Asserts a non-empty delta rather than a hardcoded
            # multiplier: the exact number of speculation_accounting_
            # violations() calls per heartbeat (currently 2 — once via
            # _check_resource_audit, once via snapshot()'s resource_audit
            # key) is an incidental implementation detail, not the contract
            # under test.
            heartbeat_samples: list[list[str]] = []
            for heartbeat_ts in (1_000_000.0, 1_000_100.0, 1_000_200.0):
                _pre_count = len(recorded_violations)
                worker._maybe_log_queue_heartbeat(heartbeat_ts)
                _new_samples = recorded_violations[_pre_count:]
                assert _new_samples, (
                    f'expected _maybe_log_queue_heartbeat({heartbeat_ts}) to '
                    f'invoke speculation_accounting_violations at least once '
                    f'via _check_resource_audit — proves the audit genuinely '
                    f'runs on the heartbeat path itself while B is parked, '
                    f'not just via the explicit call below; got no new '
                    f'recorded samples'
                )
                heartbeat_samples.extend(_new_samples)
            assert all(v == [] for v in heartbeat_samples), (
                f'speculation-slot identity must hold on every '
                f'heartbeat-triggered audit call while the speculative item '
                f'is parked on _redispatch; got {heartbeat_samples!r}'
            )

            assert worker.speculation_accounting_violations() == [], (
                'speculation-slot identity must hold while the speculative '
                'item is parked on _redispatch'
            )

            # A fails -> the existing 'previous_failed' chain-invalidation
            # path re-merges B against actual main and re-dispatches it.
            gate_a_release.set()

            outcome_a = await wait_responsive(
                req_a.result,
                label='rc-a: MergeOutcome (head verify failed)',
            )
            outcome_b = await wait_responsive(
                req_b.result,
                label='rc-b: MergeOutcome (re-merged and re-verified after A failed)',
            )

            # Same source-specific proof for the post-resolution heartbeat.
            _pre_final_heartbeat_samples = len(recorded_violations)
            worker._maybe_log_queue_heartbeat(1_000_300.0)
            final_heartbeat_samples = recorded_violations[_pre_final_heartbeat_samples:]
            assert final_heartbeat_samples and all(
                v == [] for v in final_heartbeat_samples
            ), (
                f'expected the post-resolution heartbeat call to invoke the '
                f'audit at least once and find it clean; got '
                f'{final_heartbeat_samples!r}'
            )
        finally:
            await _stop_worker(worker, worker_task)

        assert outcome_a.status != 'done', (
            f'Expected A to fail (gated verify returns passed=False), got {outcome_a!r}'
        )
        assert outcome_b is not None and outcome_b.status == 'done', (
            f'Expected B to land after remerge + re-verify, got {outcome_b!r}'
        )

        assert recorded_violations, 'expected at least one recorded audit sample'
        assert all(v == [] for v in recorded_violations), (
            f'speculation-slot identity must hold at every heartbeat-observable '
            f'point (identity broken pre-fix while B sat uncounted on '
            f'_redispatch); got samples: {recorded_violations!r}'
        )

        leak_escalations = [
            esc for esc in fake_esc_queue.submitted
            if getattr(esc, 'category', None) == 'merge_resource_leak'
        ]
        assert leak_escalations == [], (
            f'Expected zero merge_resource_leak escalations, got {leak_escalations!r}'
        )

        from orchestrator.git_ops import _run as _git_run
        _, main_files, _ = await _git_run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'], cwd=git_ops.project_root,
        )
        assert 'rc_b.py' in main_files, 'B (rc_b.py) must land on main after remerge'
        assert 'rc_a.py' not in main_files, 'A (rc_a.py) must NOT land (verify failed)'

        # task 3477: A's VerifyResult double (passed=False) reaches
        # classify_merge_failure_disposition via the real post-merge-verify
        # path. That classifier must complete honestly, not silently degrade
        # to its fail-open path because the double's cause_hint/test_output
        # shape confused _extract_failing_tests_and_candidate_files. This guard
        # is what proves the (b) fidelity fix actually reached this call site —
        # any future double built without a real cause_hint re-trips it.
        #
        # task 3980 amendment: routed through the shared `_fail_open_records`
        # predicate above. The inline version this replaces keyed on a longer
        # substring ('internal error; degrading to ...') that matched only
        # merge_disposition.py:711, so the second fail-open site
        # (worker.py::_classify_disposition_for_outcome) could swallow
        # the same defect here undetected.
        fail_open_records = _fail_open_records(caplog.records)
        assert fail_open_records == [], (
            'classify_merge_failure_disposition degraded to the silent '
            'fail-open path while classifying A\'s failing verify — the '
            'VerifyResult double reached _extract_failing_tests_and_'
            'candidate_files and raised instead of returning a real '
            'disposition. Offending record(s):\n'
            + _format_fail_open_records(fail_open_records)
        )


# ---------------------------------------------------------------------------
# task 2096: _finalizing_head speculative-permit accounting THROUGHOUT the
# real finalize-head verify-duration window.
#
# TestFinalizingHeadLifecycle (test_merge_queue_finalize_head_visibility.py)
# already proves _finalizing_head is set/cleared around `await
# entry.verify_task`. This test proves the I4 speculation-slot identity
# (speculation_accounting_violations()) stays CLEAN for the entire duration
# of that await when the finalizing head itself is speculative — the real
# window esc-__merge_resource_leak__-23/-24 (reify) observed as a spurious
# violation lasting the full verify (20-60 min), not the "single-tick"
# transient task 2063 assumed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinalizeHeadSpeculativeAccountingThroughout:
    """Ledger-based accounting for a speculative permit held by a finalizing
    head, THROUGHOUT the real `await entry.verify_task` window (task 2096 →
    task 2160/η).

    The entries below carry real ledger-issued tokens (``entry.permit``)
    rather than relying on structural location, and
    ``_inflight_speculative_count()`` derives its count from
    ``len(ledger.live) - held_by_merger`` (η step-6) instead of scanning the
    five old ``was_speculative``/``.speculative`` locations. This asserts
    that identity holds — and the finalizing-head permit stays counted —
    for the entire duration of the real `await entry.verify_task` window,
    not just at entry/exit.
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        """Build a real merged (speculative) item so _finalize_inflight's
        merge_result path works — mirrors TestFinalizeInflightPass /
        TestFinalizingHeadLifecycle's identically-named per-class helper.
        """
        from orchestrator.merge_lane.worker import RealMergeItem

        wt = await _make_branch_with_file(git_ops, branch, filename, content)
        req = _make_request(branch, branch, wt, config)
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success and merge_result.merge_commit
        assert merge_result.merge_worktree is not None
        base_sha = await git_ops.get_main_sha()
        item = RealMergeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=True,
        )
        return req, item

    async def test_speculation_accounting_clean_throughout_speculative_finalize_head_await(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        from orchestrator.merge_lane.worker import (
            InflightEntry,
            InflightVerifyResult,
            ItemLifecycleState,
        )
        from orchestrator.verify_runner import HostLease

        gate = asyncio.Event()

        async def _gated_fail_verify() -> InflightVerifyResult:
            await gate.wait()
            return InflightVerifyResult(
                outcome=MergeOutcome('blocked', reason='gated-test'),
                merge_wt=None,
            )

        req, item = await self._make_merged_item(
            git_ops, config, 'fh-spec-gate', 'fhspec.py', 'x = 1\n',
        )

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q, speculation_depth=2)

        alloc = MagicMock()
        alloc.release = AsyncMock()
        alloc.cancel_and_release = AsyncMock()
        worker._host_allocator = alloc

        # Acquire both permits THROUGH the ledger up front: the gated
        # finalizing head (below) holds one, a second speculative entry
        # parked in _inflight holds the other.
        second_req, second_item = await self._make_merged_item(
            git_ops, config, 'fh-spec-second', 'fhspec2.py', 'y = 2\n',
        )
        worker._inflight.append(InflightEntry(
            item=second_item,
            lease=HostLease(name='local2', runner=MagicMock(), is_local=True),
            verify_task=None,
            merge_wt=second_item.merge_wt,
            was_speculative=True,
            permit=await worker._speculation_ledger.acquire(),
        ))

        local_lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        verify_task = asyncio.ensure_future(_gated_fail_verify())
        entry = InflightEntry(
            item=item,
            lease=local_lease,
            verify_task=verify_task,
            merge_wt=None,
            was_speculative=True,
            permit=await worker._speculation_ledger.acquire(),
        )

        # Register at VERIFYING before invoking _finalize_inflight directly —
        # mirrors what the real _inflight_append chokepoint already does before
        # any entry reaches _finalize_inflight in production (task 2435
        # kappa-b: the finalize-head is now derived from _live_items via
        # _finalizing_head_entry(), not the deleted _finalizing_head field, so
        # the entry must be tracked in the registry to be discoverable).
        worker._register_item(entry, initial=ItemLifecycleState.VERIFYING)

        # Launch _finalize_inflight; it will pause at `await entry.verify_task`.
        fin = asyncio.ensure_future(worker._finalize_inflight(entry))
        await asyncio.sleep(0)  # yield to let _finalize_inflight reach the await

        snap_mid = worker.snapshot()
        assert snap_mid['head_of_line'] == req.task_id, (
            f'Expected head_of_line=={req.task_id!r} mid-await — the gated '
            f'finalizing head, not the parked speculative entry; '
            f'got {snap_mid["head_of_line"]!r}.'
        )

        violations_mid_await = worker.speculation_accounting_violations()
        assert violations_mid_await == [], (
            f'speculation-slot identity must hold for the ENTIRE '
            f'`await entry.verify_task` window when the finalizing head is '
            f'itself speculative (task 2096); got {violations_mid_await!r}. '
            f'_inflight_speculative_count() derives from ledger.live, so the '
            f'finalizing head permit stays counted (not drained-but-uncounted) '
            f'for the whole verify duration.'
        )
        assert worker._inflight_speculative_count() == 2, (
            f'Expected both the finalizing head and the parked _inflight '
            f'entry counted; got {worker._inflight_speculative_count()!r}.'
        )

        # Release the gate so _finalize_inflight can complete.
        gate.set()
        await fin

        snap_after = worker.snapshot()
        assert snap_after['head_of_line'] == second_req.task_id, (
            f'Expected the finalizing head to have retired after '
            f'_finalize_inflight, leaving the parked entry '
            f'{second_req.task_id!r} as head_of_line; '
            f'got {snap_after["head_of_line"]!r}.'
        )
        assert req.result.done(), 'Expected req.result to be resolved after FAIL path.'
        assert worker.speculation_accounting_violations() == [], (
            'speculation-slot identity must hold after the finalizing head '
            'releases its permit in the finally clause.'
        )


# ---------------------------------------------------------------------------
# task 3477 step-1 RED: _fake_verify_result factory contract
# ---------------------------------------------------------------------------
#
# None of the 18 inline VerifyResult-shaped `MagicMock(passed=..., summary=...,
# test_output=..., ...)` doubles in this file set `cause_hint`, so an
# unconfigured attribute read on a bare MagicMock auto-vivifies a truthy
# child Mock instead of a real string. Every FAILING double reaches
# merge_disposition.py:218-221's
# `'\n'.join(part for part in (verify_result.cause_hint, verify_result.test_output) if part)`,
# which raises `TypeError: sequence item 0: expected str instance, MagicMock
# found`. classify_merge_failure_disposition (merge_disposition.py:710-719)
# swallows that into a silent fail-open — WARNING
# "classify_merge_failure_disposition: internal error; degrading to
# INDETERMINATE (fail-open, I3)" — so the affected tests only APPEAR to
# exercise disposition classification. `_fake_verify_result` (GREEN:
# task 3477 step-2, added beside `_mock_verify_result` above) fixes this at
# the source: a `MagicMock(spec=VerifyResult)` seeded from a REAL
# VerifyResult's field defaults, so `cause_hint` (and every other field) is
# authoritative and future fields are picked up automatically.
# ---------------------------------------------------------------------------


class TestFakeVerifyResultFidelity:
    """`_fake_verify_result` doubles must survive the real disposition-
    classification extraction path, unlike a bare `MagicMock(passed=...)`.
    """

    def test_failing_double_survives_disposition_extraction(self) -> None:
        """A failing `_fake_verify_result` double must not TypeError inside
        `_extract_failing_tests_and_candidate_files`.

        RED (pre task-3477 step-2): NameError: name '_fake_verify_result' is
        not defined.
        """
        from orchestrator.merge_lane.disposition import (
            _extract_failing_tests_and_candidate_files,
        )

        double = _fake_verify_result(
            passed=False,
            summary='test_failure',
            test_output='FAILED',
            category='test_failure',
        )

        failing_tests, candidate_files = _extract_failing_tests_and_candidate_files(
            double,
        )
        assert (failing_tests, candidate_files) == ((), ()), (
            f'Expected an uninformative failing double to degrade to the '
            f'honest empty-candidate-set return, not raise or manufacture a '
            f'false candidate; got {(failing_tests, candidate_files)!r}.'
        )

        # Root-cause pin: a bare MagicMock auto-vivifies `.cause_hint` as a
        # truthy child Mock (not a str) — that is the whole bug. The factory
        # must seed a real string default (VerifyResult's own default: '')
        # instead.
        assert isinstance(double.cause_hint, str), (
            f'Expected double.cause_hint to be a real str (the VerifyResult '
            f'default is \'\'), not an auto-vivified MagicMock child; got '
            f'{double.cause_hint!r} ({type(double.cause_hint)!r}). This is '
            f'the root cause of the file-wide fail-open: '
            f'_extract_failing_tests_and_candidate_files joins '
            f'(cause_hint, test_output) and str.join rejects a non-str item.'
        )

    def test_rejects_unknown_override_field(self) -> None:
        """An override key that is not a real VerifyResult field must raise
        immediately, not silently no-op (task 3477 amend).

        ``spec=VerifyResult`` alone only guards unconfigured *reads*; a
        plain ``setattr`` on a ``spec=`` (not ``spec_set=``) mock still
        accepts an unknown attribute WRITE. Without this validation, a
        typo'd override (e.g. ``categoy=`` for ``category=``) or a
        leftover ``verify_skipped=`` would silently produce a double whose
        real field keeps its seeded default instead of raising -- the same
        class of silent drift this factory exists to eliminate.
        """
        with pytest.raises(TypeError, match='verify_skipped'):
            _fake_verify_result(passed=False, verify_skipped=True)


# ---------------------------------------------------------------------------
# task 3477 step-5: _await_outcome loud-timeout contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAwaitOutcomeHelper:
    """`_await_outcome` must report a real timeout AS a timeout.

    The previous shape in this file --
    ``with contextlib.suppress(TimeoutError): outcome = await
    asyncio.wait_for(fut, timeout=N)`` followed by
    ``assert outcome is not None and outcome.status == ...`` -- converts a
    genuine pipeline hang into a confusing ``outcome is None`` assertion
    failure, hiding whether the pipeline actually hung or the cascade never
    fired. ``_await_outcome`` closes that gap by failing loudly, by name,
    at the deadline instead.
    """

    async def test_await_outcome_returns_resolved_outcome(self) -> None:
        """An already-resolved future's value is returned unchanged."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        sentinel = object()
        fut.set_result(sentinel)

        result = await _await_outcome(fut, label='N (already resolved)')

        assert result is sentinel, (
            f'Expected _await_outcome to return the resolved future value '
            f'unchanged, got {result!r}.'
        )

    async def test_await_outcome_fails_loudly_on_deadline(self) -> None:
        """A future that never resolves must pytest.fail -- by name, with
        the deadline -- rather than let the caller observe a bare None.

        RED (pre task-3477 step-6): NameError: name '_await_outcome' is not
        defined.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()

        with pytest.raises(pytest.fail.Exception, match='cascade-N-await') as exc_info:
            await _await_outcome(fut, label='cascade-N-await', timeout=0.05)

        assert '0.05' in str(exc_info.value), (
            f'Expected the failure message to name the deadline (0.05s), '
            f'got {exc_info.value!r}. A real timeout must report AS a '
            f'timeout -- naming the label and the deadline -- not surface '
            f'as a confusing `outcome is None`.'
        )


class TestLoudWaitMigrationRatchet:
    """Enforced invariant, task 4219, ratcheted to zero by task 4846: THIS
    module's own source carries no suppressed result wait -- neither the
    `with contextlib.suppress(TimeoutError):` spelling nor the swallowing
    `try/except TimeoutError:` spelling -- and every method that waits
    through `wait_responsive` stops its worker from a `finally:`.

    No `@pytest.mark.timeout` mark needed: neither of these two methods
    performs a real await, so `_wait_budget_audit.py::item_wait_budgets`
    bills each of them 0.0.
    """

    def test_no_suppressed_result_waits(self) -> None:
        """`suppressed_result_wait_items` finds nothing in this module's
        own source -- a fresh suppressed wait, in either recognised
        spelling, or a migrated method that regressed, fails loudly, naming
        the class and method and pointing at the two blessed replacements.
        """
        source = Path(__file__).read_text()
        scanned = suppressed_result_wait_items(source)

        assert not scanned, (
            'The following test methods contain a suppressed result wait '
            '(either `with contextlib.suppress(TimeoutError): await '
            'asyncio.wait_for(...)`, or `try: ... await asyncio.wait_for(...) '
            'except TimeoutError: ... = None`):\n'
            + '\n'.join(f'  - {item}' for item in scanned)
            + '\n\nThis is the `outcome is None` anti-pattern: a genuine '
            'timeout is swallowed and a downstream assertion reports a '
            'confusing None/wrong-value failure instead of a loud, '
            'by-label timeout. Migrate the wait to '
            '`_orch_helpers.wait_responsive` (or, for a bare result future '
            'with no starvation accounting needed, `_await_outcome`).'
        )

    def test_migrated_methods_stop_worker_in_finally(self) -> None:
        """No method that calls `wait_responsive` may leave `worker.stop()`
        outside a `finally:` block (esc-3980-4): `wait_responsive` gives up
        by raising `_pytest.outcomes.Failed`, so a straight-line body that
        skips `worker.stop()` on give-up leaks a live merge worker plus its
        run task into pytest-asyncio teardown -- how one red test used to
        cascade into unrelated failures elsewhere in the session.
        """
        source = Path(__file__).read_text()
        offenders = unguarded_worker_teardown_items(source)

        assert not offenders, (
            'The following methods call wait_responsive but do not call '
            'worker.stop() from inside a finally: block (esc-3980-4 leak '
            'invariant -- see '
            'test_merge_queue_concurrent_verify.py::_stop_worker):\n'
            + '\n'.join(f'  - {offender}' for offender in offenders)
        )


class TestHeavyBarrierTimeoutConstant:
    """HEAVY_BARRIER_TEST_TIMEOUT is a literal held up by two floors.

    Whether each marked test's waits fit under it is checked per item by
    test_wait_budget_coverage_guard.py, not here.
    """

    def test_clears_the_measured_wall_clock_with_headroom(self) -> None:
        """The constant keeps UNDER_LOAD_HEADROOM_FACTOR over the measured worst case."""
        required = _HEAVY_BARRIER_MEASURED_UNDER_LOAD_WORST_CASE * UNDER_LOAD_HEADROOM_FACTOR
        assert required <= HEAVY_BARRIER_TEST_TIMEOUT, (
            f'HEAVY_BARRIER_TEST_TIMEOUT ({HEAVY_BARRIER_TEST_TIMEOUT}s) no longer '
            f'clears {UNDER_LOAD_HEADROOM_FACTOR}x the measured-under-load worst '
            f'case ({_HEAVY_BARRIER_MEASURED_UNDER_LOAD_WORST_CASE}s = {required}s). '
            'Raise the constant (and re-record the measurement) rather than '
            'relaxing this arithmetic.'
        )

    def test_never_narrows_the_ini_default(self) -> None:
        """The constant never falls below PYPROJECT_DEFAULT_TIMEOUT."""
        assert HEAVY_BARRIER_TEST_TIMEOUT >= PYPROJECT_DEFAULT_TIMEOUT, (
            f'HEAVY_BARRIER_TEST_TIMEOUT ({HEAVY_BARRIER_TEST_TIMEOUT}s) is below '
            f'the pyproject default ({PYPROJECT_DEFAULT_TIMEOUT}s). A timeout mark '
            'REPLACES the ambient budget rather than flooring it (see '
            '_orch_helpers.VERIFY_CLI_PER_TEST_TIMEOUT), so below the ini default '
            'the merge-barrier classes carrying this mark run TIGHTER than every '
            'unmarked test on a bare local or agent run. Raise the constant.'
        )
