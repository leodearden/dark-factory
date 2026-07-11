"""Tests for the I7 single-writer debug-assert flag + helper (task 1999).

Covers:
  step-03 RED — module-level ``_DEBUG_ASSERTS`` flag and the
                ``_assert_single_writer`` owner-task-identity guard, wired at
                the lane-buffer mutation choke points (``_buffer_owned_request``'s
                append, ``_pop_next_pickable``'s ``del buf[i]`` removal).
                Exercises the full semantics matrix:

                  (1) flag OFF                              → always a no-op
                  (2) flag ON + running + foreign owner task → raises AssertionError
                  (3) flag ON + owner task is None           → no-op (direct-call /
                                                                unstarted-loop tolerance)
                  (4) flag ON + not running                  → no-op (shutdown
                                                                drain tolerance)
                  (5) flag ON + owner task IS the caller      → no-op

See ``_lane_buffers``' "Accessed only from the merger coroutine" docstring
comment and ``_spawn_loop`` (stores ``self._merger_task``) for the
single-writer discipline this enforces.

RED until step-04 GREEN adds ``merge_queue._DEBUG_ASSERTS`` +
``SpeculativeMergeWorker._assert_single_writer`` and wires it at the two
lane-buffer mutation call sites.

  step-05 RED — the verifier-owned ``_inflight`` choke points. Unlike the
                lane-buffer side, the real ``_inflight.append`` /
                ``.popleft`` / ``.clear`` call sites are inlined directly in
                ``_verifier_loop`` (no pre-existing dedicated methods), so
                step-06 GREEN introduces thin wrapper choke-point methods —
                ``_inflight_append`` / ``_inflight_popleft`` / ``_inflight_clear``
                — that ``_verifier_loop`` calls instead of mutating the deque
                inline, each asserting ownership via ``self._verifier_task``
                before mutating. RED because these methods do not exist yet.

  step-07 RED — suite-wide enablement + end-to-end no-false-positive
                conformance. (a) asserts that ``merge_queue._DEBUG_ASSERTS``
                is ``True`` with no per-test monkeypatching in effect,
                proving conftest.py turns the discipline on for the whole
                suite (RED until step-08 GREEN). (b) drives a REAL
                ``worker.run()`` (merger + verifier loops) through two
                independent real branches to resolution with the flag
                forced on, proving the steps 04/06 wiring never raises a
                false-positive ``AssertionError`` against the production
                coroutines.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator import merge_queue
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import InflightEntry, MergeRequest, SpeculativeMergeWorker

# ── Fixtures ──────────────────────────────────────────────────────────────
#
# _buffer_owned_request / _pop_next_pickable are pure in-memory lane-buffer
# operations that never touch the filesystem or shell out to git, so (unlike
# the frozen-prefix / two-layer-invariant test files) no real git repo is
# needed — a bare tmp_path stands in for project_root.


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
def config(tmp_path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path, git=git_config)


@pytest.fixture
def git_ops(git_config: GitConfig, tmp_path) -> GitOps:
    return GitOps(git_config, tmp_path)


# ── step-07 fixtures: a REAL git repo ───────────────────────────────────────
#
# Only the end-to-end conformance test needs one — the lane-buffer/_inflight
# choke-point tests above are pure in-memory operations that never shell out
# to git, so they keep using the bare-tmp_path git_ops/config fixtures above.


async def _setup_e2e_repo(repo: Path) -> None:
    """Initialise a minimal real git repo with one initial commit on main.

    Mirrors test_merge_queue.py's module-level ``_setup_repo``.
    """
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def e2e_git_repo(tmp_path: Path) -> Path:
    """Real initialised git repo for the step-07 end-to-end conformance test."""
    repo = tmp_path / 'e2e-repo'
    repo.mkdir()
    asyncio.run(_setup_e2e_repo(repo))
    return repo


@pytest.fixture
def e2e_git_ops(git_config: GitConfig, e2e_git_repo: Path) -> GitOps:
    return GitOps(git_config, e2e_git_repo)


@pytest.fixture
def e2e_config(e2e_git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=e2e_git_repo, git=git_config)


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring).

    Mirrors test_merge_queue_verify_base_invariant.py's helper of the same
    name. ``_merger_task``/``_verifier_task`` default to None (never-started
    loops) and ``_running`` defaults to True (see merge_queue.py ``__init__``).
    """
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


def _make_req(
    config: OrchestratorConfig, tmp_path, *, lane: Literal['normal', 'high'] = 'normal',
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
    """
    return MergeRequest(
        task_id='t-1',
        branch='task/t-1',
        worktree=tmp_path,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane=lane,
    )


async def _make_foreign_task() -> asyncio.Task:  # type: ignore[type-arg]
    """Return a distinct, already-finished dummy Task — 'foreign' to the caller.

    ``_assert_single_writer`` compares object identity against
    ``asyncio.current_task()``, so any Task other than the calling test's own
    task works as a stand-in for "some other coroutine mutated this
    structure". Awaited to completion before returning so it is never left
    pending at test teardown (a pending Task garbage-collected mid-test emits
    "Task was destroyed but it is pending!").
    """
    async def _noop() -> None:
        return None

    task = asyncio.ensure_future(_noop())
    await task
    return task


def _sentinel_entry() -> InflightEntry:
    """Opaque, identity-only stand-in for a real ``InflightEntry``.

    The ``_inflight`` choke-point tests below exercise pure queue mechanics
    (raises vs. no-op, ``is``-identity of whatever gets popped) and never
    inspect an entry's fields, so a lightweight ``SimpleNamespace`` stub is
    enough at runtime. ``cast`` tells the type checker to treat it as an
    ``InflightEntry`` without constructing an irrelevant
    ``SpeculativeItem``/``MergeResult`` graph — contrast with
    test_merge_queue_verify_base_invariant.py's ``_make_inflight_entry``,
    which builds a real one because that test's assertions inspect
    ``base_sha``/``merge_commit``.

    The ``.item.request.request_id`` attribute chain is required (not just
    identity) because MQ-reliability kappa/2169 (commit 56132176382) made
    ``_inflight_append`` call
    ``_note_transition(entry.item.request.request_id, DISPATCHING, VERIFYING)``
    unconditionally, ahead of where the ownership no-op in
    ``_assert_single_writer`` can short-circuit — a bare ``object()`` has no
    ``.item`` and raises ``AttributeError`` there. The request_id
    (``'sentinel-request-id'``) is an intentionally UNREGISTERED
    placeholder: the worker's fresh ``ItemLifecycle`` has never seen it, so
    the transition raises ``IllegalLifecycleTransition``, which
    ``_note_transition`` swallows (WARNING log only) and routes to
    ``_alarm_illegal_lifecycle_transition`` — a no-op under
    ``_make_worker``'s default ``escalation_queue=None``. So the no-op
    tests above neither raise nor escalate; they just absorb a harmless
    WARNING.
    """
    return cast(
        InflightEntry,
        SimpleNamespace(
            item=SimpleNamespace(
                request=SimpleNamespace(request_id='sentinel-request-id'),
            ),
        ),
    )


# ── step-03 RED: _assert_single_writer semantics at the lane-buffer sites ───


@pytest.mark.asyncio
class TestAssertSingleWriterLaneBufferWiring:
    """``_assert_single_writer`` at the merger-owned lane-buffer choke points (task 1999 I7).

    RED until step-04 GREEN adds the module flag, the helper, and wires it at
    ``_buffer_owned_request``'s append and the ``del buf[i]`` removal inside
    ``_pop_next_pickable``.
    """

    async def test_flag_off_is_noop_for_append_and_pop(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path, monkeypatch,
    ) -> None:
        """(1) Flag OFF: a foreign merger task never raises — zero-overhead no-op."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', False)
        worker = _make_worker(git_ops)
        worker._running = True
        worker._merger_task = await _make_foreign_task()

        req = _make_req(config, tmp_path)
        worker._buffer_owned_request(req)  # must not raise
        popped = worker._pop_next_pickable()  # must not raise
        assert popped is req

    async def test_flag_on_foreign_owner_raises_on_append(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path, monkeypatch,
    ) -> None:
        """(2) Flag ON + running + foreign merger task: append raises, naming the structure."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker = _make_worker(git_ops)
        worker._running = True
        worker._merger_task = await _make_foreign_task()

        req = _make_req(config, tmp_path)
        with pytest.raises(AssertionError, match='_lane_buffers'):
            worker._buffer_owned_request(req)

    async def test_flag_on_foreign_owner_raises_on_pop(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path, monkeypatch,
    ) -> None:
        """(2) Flag ON + running + foreign merger task: pop raises, naming the structure.

        Seeds the buffer directly (bypassing ``_buffer_owned_request``) so the
        append and pop choke points are exercised independently.
        """
        worker = _make_worker(git_ops)
        req = _make_req(config, tmp_path)
        worker._lane_buffers['normal'].append(req)

        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker._running = True
        worker._merger_task = await _make_foreign_task()

        with pytest.raises(AssertionError, match='_lane_buffers'):
            worker._pop_next_pickable()

    async def test_flag_on_owner_task_none_is_noop(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path, monkeypatch,
    ) -> None:
        """(3) Flag ON but no owner task recorded yet: no-op (direct-call / unstarted-loop)."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker = _make_worker(git_ops)
        worker._running = True
        worker._merger_task = None

        req = _make_req(config, tmp_path)
        worker._buffer_owned_request(req)  # must not raise
        popped = worker._pop_next_pickable()  # must not raise
        assert popped is req

    async def test_flag_on_not_running_is_noop(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path, monkeypatch,
    ) -> None:
        """(4) Flag ON but worker stopped: no-op regardless of owner (shutdown-drain tolerance)."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker = _make_worker(git_ops)
        worker._merger_task = await _make_foreign_task()
        worker._running = False

        req = _make_req(config, tmp_path)
        worker._buffer_owned_request(req)  # must not raise
        popped = worker._pop_next_pickable()  # must not raise
        assert popped is req

    async def test_flag_on_owner_is_current_task_is_noop(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path, monkeypatch,
    ) -> None:
        """(5) Flag ON + running + the CALLER is the recorded owner: no-op."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker = _make_worker(git_ops)
        worker._running = True
        worker._merger_task = asyncio.current_task()

        req = _make_req(config, tmp_path)
        worker._buffer_owned_request(req)  # must not raise
        popped = worker._pop_next_pickable()  # must not raise
        assert popped is req


# ── step-05 RED: _assert_single_writer semantics at the _inflight sites ─────


@pytest.mark.asyncio
class TestAssertSingleWriterInflightWiring:
    """``_assert_single_writer`` at the verifier-owned ``_inflight`` choke points (task 1999 I7).

    RED until step-06 GREEN adds ``_inflight_append`` / ``_inflight_popleft`` /
    ``_inflight_clear`` — thin wrapper choke-point methods that assert
    ownership via ``self._verifier_task`` before mutating ``self._inflight``
    — and rewires ``_verifier_loop`` to call them instead of mutating the
    deque inline.
    """

    async def test_flag_off_is_noop_for_append_pop_and_clear(
        self, git_ops: GitOps, monkeypatch,
    ) -> None:
        """(1) Flag OFF: a foreign verifier task never raises — zero-overhead no-op."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', False)
        worker = _make_worker(git_ops)
        worker._running = True
        worker._verifier_task = await _make_foreign_task()

        entry = _sentinel_entry()
        worker._inflight_append(entry)  # must not raise
        popped = worker._inflight_popleft()  # must not raise
        assert popped is entry
        worker._inflight_append(_sentinel_entry())
        worker._inflight_clear()  # must not raise
        assert len(worker._inflight) == 0

    async def test_flag_on_foreign_owner_raises_on_append(
        self, git_ops: GitOps, monkeypatch,
    ) -> None:
        """(2) Flag ON + running + foreign verifier task: append raises, naming the structure."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker = _make_worker(git_ops)
        worker._running = True
        worker._verifier_task = await _make_foreign_task()

        with pytest.raises(AssertionError, match='_inflight'):
            worker._inflight_append(_sentinel_entry())

    async def test_flag_on_foreign_owner_raises_on_popleft(
        self, git_ops: GitOps, monkeypatch,
    ) -> None:
        """(2) Flag ON + running + foreign verifier task: popleft raises, naming the structure.

        Seeds the deque directly (bypassing ``_inflight_append``) so the
        append and popleft choke points are exercised independently.
        """
        worker = _make_worker(git_ops)
        worker._inflight.append(_sentinel_entry())

        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker._running = True
        worker._verifier_task = await _make_foreign_task()

        with pytest.raises(AssertionError, match='_inflight'):
            worker._inflight_popleft()

    async def test_flag_on_foreign_owner_raises_on_clear(
        self, git_ops: GitOps, monkeypatch,
    ) -> None:
        """(2) Flag ON + running + foreign verifier task: clear raises, naming the structure."""
        worker = _make_worker(git_ops)
        worker._inflight.append(_sentinel_entry())

        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker._running = True
        worker._verifier_task = await _make_foreign_task()

        with pytest.raises(AssertionError, match='_inflight'):
            worker._inflight_clear()

    async def test_flag_on_owner_task_none_is_noop(
        self, git_ops: GitOps, monkeypatch,
    ) -> None:
        """(3) Flag ON but no owner task recorded yet: no-op (direct-call / unstarted-loop)."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker = _make_worker(git_ops)
        worker._running = True
        worker._verifier_task = None

        entry = _sentinel_entry()
        worker._inflight_append(entry)  # must not raise
        popped = worker._inflight_popleft()  # must not raise
        assert popped is entry
        worker._inflight_append(_sentinel_entry())
        worker._inflight_clear()  # must not raise

    async def test_flag_on_not_running_is_noop(
        self, git_ops: GitOps, monkeypatch,
    ) -> None:
        """(4) Flag ON but worker stopped: no-op regardless of owner (matches stop()'s drain)."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker = _make_worker(git_ops)
        worker._inflight.append(_sentinel_entry())
        worker._verifier_task = await _make_foreign_task()
        worker._running = False

        popped = worker._inflight_popleft()  # must not raise
        assert popped is not None
        worker._inflight_append(_sentinel_entry())
        worker._inflight_clear()  # must not raise

    async def test_flag_on_owner_is_current_task_is_noop(
        self, git_ops: GitOps, monkeypatch,
    ) -> None:
        """(5) Flag ON + running + the CALLER is the recorded owner: no-op."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker = _make_worker(git_ops)
        worker._running = True
        worker._verifier_task = asyncio.current_task()

        entry = _sentinel_entry()
        worker._inflight_append(entry)  # must not raise
        popped = worker._inflight_popleft()  # must not raise
        assert popped is entry
        worker._inflight_append(_sentinel_entry())
        worker._inflight_clear()  # must not raise


# ── Regression: _sentinel_entry() choke-point contract (task 2445) ─────────


@pytest.mark.asyncio
class TestSentinelEntryChokePointContract:
    """Regression guard for ``_sentinel_entry()``'s shape (task 2445).

    MQ-reliability kappa (task 2169, commit 56132176382) made
    ``_inflight_append`` call
    ``_note_transition(entry.item.request.request_id, ...)`` unconditionally
    — ahead of where ``_assert_single_writer``'s no-op could short-circuit.
    A bare ``object()`` sentinel has no ``.item``, so every no-op test in
    ``TestAssertSingleWriterInflightWiring`` above that appends a sentinel
    raised ``AttributeError: 'object' object has no attribute 'item'`` at
    that dereference (merge_queue.py:8992). This pins the fix so it cannot
    regress silently.
    """

    async def test_sentinel_entry_survives_inflight_append_chokepoint(
        self, git_ops: GitOps,
    ) -> None:
        """``_inflight_append`` must not raise on a bare ``_sentinel_entry()``.

        Regression guard for ``AttributeError: 'object' object has no
        attribute 'item'`` previously raised at ``_note_transition``'s
        ``entry.item.request.request_id`` dereference (merge_queue.py:8992).
        ``_make_worker`` builds a worker whose ``_verifier_task`` defaults to
        None — so ``_assert_single_writer`` is a no-op regardless of
        ``_DEBUG_ASSERTS`` — and whose ``_escalation_queue`` defaults to
        None, pinned below as the precondition that makes the sentinel's
        intentionally-unregistered request_id raise-and-swallow an
        ``IllegalLifecycleTransition`` inside ``_note_transition`` (a
        harmless WARNING log) rather than escalate anything: the None-guard
        on ``_alarm_illegal_lifecycle_transition`` returns immediately. Do
        NOT wire a non-None spy escalation queue here — with an unregistered
        request_id that would spuriously fail, since the alarm would then
        actually submit.
        """
        worker = _make_worker(git_ops)
        assert worker._escalation_queue is None

        entry = _sentinel_entry()
        worker._inflight_append(entry)  # must not raise

        assert worker._inflight[-1] is entry


# ── step-07 RED: suite-wide enablement + end-to-end no-false-positive ───────
# conformance ────────────────────────────────────────────────────────────────


async def _make_branch_with_file(
    git_ops: GitOps, branch_name: str, filename: str, content: str,
) -> Path:
    """Create a worktree branch with one committed file; return its path.

    Mirrors the identically-named helper in test_merge_queue.py — only the
    step-07 end-to-end conformance test needs a real branch.
    """
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


def _make_e2e_request(
    task_id: str, branch: str, worktree: Path, config: OrchestratorConfig,
) -> MergeRequest:
    """Build a MergeRequest against a real worktree for the step-07 e2e test.

    Must be called from within an async context (asyncio.get_running_loop()).
    """
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


@pytest.mark.asyncio
class TestDebugAssertsSuiteWideAndConformance:
    """step-07 (task 1999 I7): suite-wide enablement + real-pipeline conformance.

    (a) The whole test suite must run with ``ORCH_DEBUG_ASSERTS`` enabled —
        proves conftest.py turns the single-writer discipline asserts on for
        every test, not just this file's own explicitly-monkeypatched cases
        above.
    (b) A real end-to-end merge — two independent real branches pushed
        through a LIVE ``worker.run()`` (merger + verifier loops) to
        resolution — must complete cleanly with the debug asserts forced ON,
        proving the production merger/verifier coroutines only ever mutate
        ``_lane_buffers``/``_inflight`` from their own recorded owner task
        (``self._merger_task`` / ``self._verifier_task``) — i.e. the steps
        04/06 wiring produces no false-positive ``AssertionError`` under a
        real run.

    RED (part a only) until step-08 GREEN sets
    ``os.environ.setdefault('ORCH_DEBUG_ASSERTS', '1')`` in conftest.py (plus
    a belt-and-braces direct assignment pinning
    ``merge_queue._DEBUG_ASSERTS = True``, to defeat any import-order race).
    """

    async def test_suite_wide_debug_asserts_enabled(self) -> None:
        """(a) conftest.py must enable ORCH_DEBUG_ASSERTS suite-wide.

        RED: nothing sets ORCH_DEBUG_ASSERTS today, so the module flag —
        seeded once at import from the env var — is False.
        """
        assert merge_queue._DEBUG_ASSERTS is True, (
            'Expected merge_queue._DEBUG_ASSERTS is True suite-wide (conftest.py '
            'should set ORCH_DEBUG_ASSERTS=1 before importing orchestrator, per '
            f'task 1999 I7 / PRD Open Q3) but got {merge_queue._DEBUG_ASSERTS!r} — '
            'the single-writer discipline asserts wired in steps 04/06 are not '
            'being exercised by the suite.'
        )

    async def test_real_pipeline_no_false_positive_with_asserts_on(
        self,
        e2e_git_ops: GitOps,
        e2e_config: OrchestratorConfig,
        monkeypatch,
    ) -> None:
        """(b) Two real requests flow through a live worker to resolution, asserts ON.

        Forces ``merge_queue._DEBUG_ASSERTS = True`` directly so this
        assertion holds regardless of part (a)'s conftest wiring, then drives
        the REAL merger + verifier loops via ``worker.run()`` — the same
        real-pipeline shape as
        ``TestSpeculativeMergeWorker.test_speculative_basic_throughput`` in
        test_merge_queue.py.  No ``AssertionError`` may propagate from either
        request's resolution, and both must reach a terminal ``'done'``
        outcome.
        """
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)

        wt_n = await _make_branch_with_file(
            e2e_git_ops, 'sw-e2e-n', 'file_sw_e2e_n.py', 'n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            e2e_git_ops, 'sw-e2e-n1', 'file_sw_e2e_n1.py', 'n1 = 2\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(e2e_git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            AsyncMock(return_value=MagicMock(passed=True, summary='')),
        ):
            req_n = _make_e2e_request('sw-e2e-n', 'sw-e2e-n', wt_n, e2e_config)
            req_n1 = _make_e2e_request('sw-e2e-n1', 'sw-e2e-n1', wt_n1, e2e_config)

            # Submit both before the worker processes them, matching
            # test_speculative_basic_throughput's real-pipeline shape.
            await queue.put(req_n)
            await queue.put(req_n1)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)

        assert outcome_n.status == 'done', f'N failed: {outcome_n}'
        assert outcome_n1.status == 'done', f'N+1 failed: {outcome_n1}'

        await worker.stop()
        await worker_task
