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
"""

from __future__ import annotations

import asyncio

import pytest

from orchestrator import merge_queue
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_queue import MergeRequest, SpeculativeMergeWorker

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


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring).

    Mirrors test_merge_queue_verify_base_invariant.py's helper of the same
    name. ``_merger_task``/``_verifier_task`` default to None (never-started
    loops) and ``_running`` defaults to True (see merge_queue.py ``__init__``).
    """
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


def _make_req(config: OrchestratorConfig, tmp_path, *, lane: str = 'normal') -> MergeRequest:
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

        entry = object()
        worker._inflight_append(entry)  # must not raise
        popped = worker._inflight_popleft()  # must not raise
        assert popped is entry
        worker._inflight_append(object())
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
            worker._inflight_append(object())

    async def test_flag_on_foreign_owner_raises_on_popleft(
        self, git_ops: GitOps, monkeypatch,
    ) -> None:
        """(2) Flag ON + running + foreign verifier task: popleft raises, naming the structure.

        Seeds the deque directly (bypassing ``_inflight_append``) so the
        append and popleft choke points are exercised independently.
        """
        worker = _make_worker(git_ops)
        worker._inflight.append(object())

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
        worker._inflight.append(object())

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

        entry = object()
        worker._inflight_append(entry)  # must not raise
        popped = worker._inflight_popleft()  # must not raise
        assert popped is entry
        worker._inflight_append(object())
        worker._inflight_clear()  # must not raise

    async def test_flag_on_not_running_is_noop(
        self, git_ops: GitOps, monkeypatch,
    ) -> None:
        """(4) Flag ON but worker stopped: no-op regardless of owner (matches stop()'s drain)."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker = _make_worker(git_ops)
        worker._inflight.append(object())
        worker._verifier_task = await _make_foreign_task()
        worker._running = False

        popped = worker._inflight_popleft()  # must not raise
        assert popped is not None
        worker._inflight_append(object())
        worker._inflight_clear()  # must not raise

    async def test_flag_on_owner_is_current_task_is_noop(
        self, git_ops: GitOps, monkeypatch,
    ) -> None:
        """(5) Flag ON + running + the CALLER is the recorded owner: no-op."""
        monkeypatch.setattr(merge_queue, '_DEBUG_ASSERTS', True)
        worker = _make_worker(git_ops)
        worker._running = True
        worker._verifier_task = asyncio.current_task()

        entry = object()
        worker._inflight_append(entry)  # must not raise
        popped = worker._inflight_popleft()  # must not raise
        assert popped is entry
        worker._inflight_append(object())
        worker._inflight_clear()  # must not raise
