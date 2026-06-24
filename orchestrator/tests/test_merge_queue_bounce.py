"""Tests for the η=1892 bounce/rebase layer of the two-layer merge queue.

Covers:
  pre-1        — scaffolding: fixtures + helpers (no test bodies yet)
  step-01 RED  — recompute probes suffix vs FROZEN TIP, not bare main
  step-03 RED  — bounce primitives exist (NEEDS_REBASE_REASON_PREFIX,
                 MERGE_BOUNCE_CAP, MergeBounceRegistry)
  step-05 RED  — _bounce_conflicting_suffix_items() clean-rebase path re-queues
  step-07 RED  — _bounce_conflicting_suffix_items() escalation paths
  step-09 RED  — _acquire_next_request() wires bounce at graph time
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    MERGE_BOUNCE_CAP,
    NEEDS_REBASE_REASON_PREFIX,
    InflightEntry,
    MergeBounceRegistry,
    MergeOutcome,
    MergeRequest,
    SpeculativeItem,
    SpeculativeMergeWorker,
    SuffixConflictGraph,
)

# ── Module-level sentinel (mirrors test_merge_queue_frozen_prefix.py) ────────
_SENTINEL_VERIFY_TASK = object()  # noqa: PD901


# ── Fixtures ──────────────────────────────────────────────────────────────────


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repo with shared.txt and README.md on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    (repo / 'shared.txt').write_text('line1\nline2\nline3\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


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


# ── Module-level helpers ──────────────────────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    lane: Literal['normal', 'high'] = 'normal',
    *,
    merge_first_enqueued_at: float | None = 1000.0,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
    """
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

    Does NOT create real git branches — suitable for pure-unit tests that
    only exercise frozen-prefix/frozen_prefix_tip accessor methods without
    real git operations. Must be called from an async context (for the future).
    """
    req = _make_req(task_id, f'task/{task_id}', config, git_repo)
    fake_merge_result = (
        MergeResult(
            success=True,
            merge_commit=merge_commit,
        )
        if merge_commit is not None
        else None
    )
    item = SpeculativeItem(
        request=req,
        merge_result=fake_merge_result,
        merge_wt=None,
        base_sha=base_sha,
        speculative=False,
        skip_verify=False,
    )
    return req, item


def _make_inflight_entry(
    item: SpeculativeItem,
    *,
    verifying: bool = True,
) -> InflightEntry:
    """Build an InflightEntry for unit tests.

    verifying=True  → verify_task set to _SENTINEL_VERIFY_TASK (any non-None
                       object; frozen_prefix() only checks `is not None`).
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


async def _create_branch_editing(
    repo: Path,
    branch_name: str,
    filename: str,
    content: str,
    base_branch: str = 'main',
) -> str:
    """Create a branch editing filename with content; return the branch SHA."""
    await _run(['git', 'checkout', '-b', branch_name], cwd=repo)
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Edit {filename} in {branch_name}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    await _run(['git', 'checkout', base_branch], cwd=repo)
    return sha.strip()


async def _create_frozen_tip_commit(
    repo: Path,
    filename: str,
    content: str,
) -> str:
    """Commit a change to main (simulating a frozen-prefix merge commit).

    Directly commits on main (no branch switch needed — we're already on main
    after _setup_repo).  Returns the resulting commit SHA.
    """
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Frozen-tip commit: {filename}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    return sha.strip()


# ── step-01 RED: recompute probes suffix vs FROZEN TIP, not bare main ─────────


@pytest.mark.asyncio
class TestFrozenTipProbe:
    """recompute_suffix_conflict_graph() §7 must use frozen_prefix_tip() as the
    probe base, not bare main_sha.

    The RED case: frozen prefix has an entry whose merge_commit T is NOT the
    same as the current main HEAD (M0).  A suffix item S is clean vs M0 but
    conflicts with T.

    - Current (bare-main) probe: S vs M0 → clean → NOT in conflicts_with_main.
      The test asserts IN → FAILS (RED) until step-02 repoints the base.
    - After fix (frozen-tip) probe: S vs T → conflict → IN conflicts_with_main
      → PASSES (GREEN).

    CONTROL: empty frozen prefix → frozen_prefix_tip == main_sha == M0 → S
    clean vs M0 → NOT in conflicts_with_main (both before and after fix).

    RED until step-02 GREEN repoints the §7 probe base to frozen_prefix_tip().
    """

    async def test_suffix_item_conflicts_with_frozen_tip(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """S is clean vs bare main (M0) but conflicts with the frozen tip T.

        Setup:
          M0 = initial commit (shared.txt = 'line1\\nline2\\nline3\\n')
          T  = commit on a side branch 'frozen-tip' that edits shared.txt line2
               → 'FROZEN-LINE2'.  T's SHA is used as the frozen item's
               merge_commit.  main HEAD stays at M0.
          S  = suffix branch off M0 editing shared.txt line2 → 'SUFFIX-LINE2'.

        S vs M0 → CLEAN (only S changed line2; M0 has the original).
        S vs T  → CONFLICT (both S and T changed line2 to different values).
        """
        # Record the original main SHA (M0).
        _, m0_sha_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        m0_sha = m0_sha_raw.strip()

        # 1. Create T on a side branch (NOT on main) so main_sha stays == M0.
        await _run(['git', 'checkout', '-b', 'frozen-tip'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('line1\nFROZEN-LINE2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'Frozen-tip: edit line2'], cwd=git_repo)
        _, t_sha_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        frozen_tip_sha = t_sha_raw.strip()
        # Return to main — main HEAD is still M0.
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        # Sanity: main must still be at M0 (frozen-tip commit is NOT on main).
        _, cur_main_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        assert cur_main_raw.strip() == m0_sha, 'main HEAD must stay at M0 for this test'

        # 2. Create suffix branch S from M0 that edits line2 to 'SUFFIX-LINE2'.
        #    S vs M0 = clean (no conflicting edit on M0).
        #    S vs T  = conflict (both S and T edited line2).
        await _run(['git', 'checkout', '-b', 'task/591'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('line1\nSUFFIX-LINE2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'S edits line2'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        # 3. Build worker; populate frozen prefix with T as merge_commit.
        worker = _make_worker(git_ops)
        _, item_t = _make_fake_item(
            't-frozen', merge_commit=frozen_tip_sha, config=config, git_repo=git_repo,
        )
        entry_t = _make_inflight_entry(item_t, verifying=True)
        worker._inflight.append(entry_t)

        # 4. Put suffix request S into the lane buffer.
        req_s = _make_req('591', '591', config, git_repo)
        worker._lane_buffers['normal'].append(req_s)

        # 5. Recompute.
        await worker.recompute_suffix_conflict_graph()

        # 6. S conflicts with the FROZEN TIP → must appear in conflicts_with_main.
        #    RED: current code probes S vs bare M0 → S is clean → assertion fails.
        #    GREEN: probe repointed to T → S conflicts → assertion passes.
        assert req_s.request_id in worker._suffix_conflict_graph.conflicts_with_main, (
            'Expected S to conflict with the frozen tip T but it was NOT in '
            'conflicts_with_main.  The §7 probe base must be frozen_prefix_tip(), '
            'not bare main_sha.  S is clean vs M0 (bare main) but conflicts with T.'
        )

    async def test_empty_frozen_prefix_uses_bare_main(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """CONTROL: empty frozen prefix → frozen_prefix_tip() == main_sha → a
        suffix item clean vs bare main is NOT in conflicts_with_main (both
        before and after the step-02 repoint).
        """
        # Create branch S editing only README.md — clean vs shared.txt on main.
        await _create_branch_editing(
            git_repo, 'task/591-clean', 'README.md', 'clean-suffix\n',
        )

        worker = _make_worker(git_ops)  # no _inflight → frozen prefix empty
        req_s = _make_req('591-clean', '591-clean', config, git_repo)
        worker._lane_buffers['normal'].append(req_s)

        await worker.recompute_suffix_conflict_graph()

        # S is clean vs bare main == frozen_prefix_tip (empty prefix).
        assert req_s.request_id not in worker._suffix_conflict_graph.conflicts_with_main, (
            'S edits only README.md and is clean vs bare main; with an empty '
            'frozen prefix, frozen_prefix_tip() == main_sha, so S must NOT be '
            'flagged in conflicts_with_main.'
        )


# ── step-03 RED: bounce primitives exist ─────────────────────────────────────


class TestBouncePrimitives:
    """NEEDS_REBASE_REASON_PREFIX, MERGE_BOUNCE_CAP, MergeBounceRegistry exist.

    RED until step-04 GREEN adds these to merge_queue.py.
    """

    def test_needs_rebase_reason_prefix_is_non_empty_str(self) -> None:
        """NEEDS_REBASE_REASON_PREFIX is a non-empty string constant."""
        assert isinstance(NEEDS_REBASE_REASON_PREFIX, str)
        assert len(NEEDS_REBASE_REASON_PREFIX) > 0

    def test_needs_rebase_reason_prefix_is_distinct(self) -> None:
        """NEEDS_REBASE_REASON_PREFIX differs from the other *_REASON_PREFIX constants."""
        from orchestrator.merge_queue import (
            DROPPED_PLAN_TARGETS_REASON_PREFIX,
            PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
        )
        others = {
            DROPPED_PLAN_TARGETS_REASON_PREFIX,
            PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
        }
        assert NEEDS_REBASE_REASON_PREFIX not in others

    def test_merge_outcome_with_needs_rebase_prefix_round_trips(self) -> None:
        """A blocked MergeOutcome carrying the prefix round-trips via startswith."""
        outcome = MergeOutcome(
            status='blocked',
            reason=NEEDS_REBASE_REASON_PREFIX + ' branch task/591 bounced',
        )
        assert outcome.status == 'blocked'
        assert outcome.reason is not None
        assert outcome.reason.startswith(NEEDS_REBASE_REASON_PREFIX)

    def test_merge_bounce_cap_is_positive_int(self) -> None:
        """MERGE_BOUNCE_CAP is an int >= 1."""
        assert isinstance(MERGE_BOUNCE_CAP, int)
        assert MERGE_BOUNCE_CAP >= 1

    def test_bounce_registry_fresh_returns_zero(self) -> None:
        """A fresh MergeBounceRegistry returns count 0 for any unseen branch."""
        reg = MergeBounceRegistry()
        assert reg.count('task/591') == 0
        assert reg.count('task/999') == 0

    def test_bounce_registry_record_bounce_increments(self) -> None:
        """record_bounce returns the new count and increments monotonically."""
        reg = MergeBounceRegistry()
        assert reg.record_bounce('task/591') == 1
        assert reg.record_bounce('task/591') == 2
        assert reg.count('task/591') == 2

    def test_bounce_registry_two_keys_are_independent(self) -> None:
        """Two different branch keys are tracked independently."""
        reg = MergeBounceRegistry()
        reg.record_bounce('task/591')
        reg.record_bounce('task/591')
        reg.record_bounce('task/592')
        assert reg.count('task/591') == 2
        assert reg.count('task/592') == 1


# ── step-05 RED: _bounce_conflicting_suffix_items() clean-rebase path ─────────


@pytest.mark.asyncio
class TestBounceCleanRebase:
    """_bounce_conflicting_suffix_items(): clean-rebase path re-queues the item.

    A clean rebase (rebase_onto_main returns True) must:
    - leave the request in _lane_buffers (work preserved, no agent)
    - not resolve req.result (future untouched)
    - not mutate merge_first_enqueued_at
    - increment the bounce registry counter to 1
    - set _suffix_conflict_signature to None (debounce invalidated)
    - emit a needs_rebase log line

    RED until step-06 GREEN implements _bounce_conflicting_suffix_items().
    """

    async def test_clean_rebase_requeues_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Clean rebase → item stays in lane buffer; future untouched; registry++."""
        import logging

        worker = _make_worker(git_ops)

        # Build a frozen InflightEntry with a fake merge_commit SHA.
        frozen_mc = 'deadbeefcafe0000'
        _, item_frozen = _make_fake_item(
            'frozen-task', merge_commit=frozen_mc, config=config, git_repo=git_repo,
        )
        entry_frozen = _make_inflight_entry(item_frozen, verifying=True)
        worker._inflight.append(entry_frozen)

        # Put a MergeRequest for branch '591' in the lane buffer.
        req = _make_req('591', '591', config, git_repo, merge_first_enqueued_at=9999.0)
        worker._lane_buffers['normal'].append(req)

        # Seed the conflict graph: req is flagged as conflicting with the frozen tip.
        worker._suffix_conflict_graph = SuffixConflictGraph(
            nodes=(req.request_id,),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset({req.request_id}),
        )
        # Give the debounce signature a non-None value so we can assert it's cleared.
        worker._suffix_conflict_signature = (('some-rid',), 'abc123')

        # Stub rebase_onto_main → True (clean rebase, no actual git op needed).
        worker._git_ops.rebase_onto_main = AsyncMock(return_value=True)  # type: ignore[method-assign]

        with caplog.at_level(logging.INFO):
            await worker._bounce_conflicting_suffix_items()

        # rebase_onto_main was called with onto=frozen_mc.
        worker._git_ops.rebase_onto_main.assert_awaited_once()
        call_kwargs = worker._git_ops.rebase_onto_main.call_args
        assert call_kwargs.kwargs.get('onto') == frozen_mc or (
            len(call_kwargs.args) >= 2 and call_kwargs.args[1] == frozen_mc
        ), f'Expected onto={frozen_mc!r}, got {call_kwargs}'

        # Item is STILL in the lane buffer (re-queued, not removed).
        assert req in worker._lane_buffers['normal'], (
            'Expected req to remain in lane buffer after a clean rebase (re-queue).'
        )

        # Future is NOT resolved (no agent dispatched).
        assert not req.result.done(), (
            'Expected req.result future to remain pending after a clean rebase.'
        )

        # merge_first_enqueued_at is unchanged.
        assert req.merge_first_enqueued_at == 9999.0, (
            'merge_first_enqueued_at must not be mutated by the bounce.'
        )

        # Bounce registry is incremented.
        assert worker._bounce_registry.count('591') == 1, (
            'Expected bounce registry count for branch "591" to be 1 after one bounce.'
        )

        # Debounce signature was invalidated.
        assert worker._suffix_conflict_signature is None, (
            'Expected _suffix_conflict_signature to be None after a bounce '
            '(so the next recompute re-probes reality).'
        )

        # A needs_rebase log line was emitted.
        relevant = [r for r in caplog.records if 'needs_rebase' in r.message.lower()
                    or NEEDS_REBASE_REASON_PREFIX.lower()[:10] in r.message.lower()]
        assert relevant, (
            f'Expected a needs_rebase log line but none found. '
            f'Log records: {[r.message for r in caplog.records]}'
        )


# ── step-07 RED: _bounce_conflicting_suffix_items() escalation paths ──────────


@pytest.mark.asyncio
class TestBounceEscalation:
    """_bounce_conflicting_suffix_items(): escalation paths.

    CASE A: real rebase conflict → item removed from buffer, future resolved
            'blocked' with NEEDS_REBASE_REASON_PREFIX.
    CASE B: bounce-cap exceeded → NO rebase attempted, item removed + escalated.

    RED until step-08 GREEN adds the escalation branches.
    """

    async def _make_worker_with_conflict_setup(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        branch: str = '591',
    ) -> tuple[SpeculativeMergeWorker, MergeRequest]:
        """Build a worker with a frozen entry and a conflicting suffix request."""
        worker = _make_worker(git_ops)
        frozen_mc = 'deadbeefcafe1111'
        _, item_frozen = _make_fake_item(
            'frozen-task', merge_commit=frozen_mc, config=config, git_repo=git_repo,
        )
        entry_frozen = _make_inflight_entry(item_frozen, verifying=True)
        worker._inflight.append(entry_frozen)

        req = _make_req('591', branch, config, git_repo)
        worker._lane_buffers['normal'].append(req)
        worker._suffix_conflict_graph = SuffixConflictGraph(
            nodes=(req.request_id,),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset({req.request_id}),
        )
        return worker, req

    async def test_case_a_real_conflict_escalates(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """CASE A: rebase_onto_main returns False → item removed, blocked escalated."""
        worker, req = await self._make_worker_with_conflict_setup(git_ops, config, git_repo)

        # Stub rebase → False (real conflict).
        worker._git_ops.rebase_onto_main = AsyncMock(return_value=False)  # type: ignore[method-assign]

        await worker._bounce_conflicting_suffix_items()

        # Item must be REMOVED from the lane buffer.
        assert req not in worker._lane_buffers['normal'], (
            'Expected req to be removed from lane buffer after a rebase conflict.'
        )

        # Future is resolved as 'blocked' with NEEDS_REBASE_REASON_PREFIX.
        assert req.result.done(), 'Expected req.result to be resolved after escalation.'
        outcome: MergeOutcome = req.result.result()
        assert outcome.status == 'blocked', (
            f'Expected status="blocked" but got {outcome.status!r}.'
        )
        assert outcome.reason is not None
        assert outcome.reason.startswith(NEEDS_REBASE_REASON_PREFIX), (
            f'Expected reason to start with NEEDS_REBASE_REASON_PREFIX but got '
            f'{outcome.reason!r}.'
        )

        # Registry incremented to 1.
        assert worker._bounce_registry.count('591') == 1

        # Debounce signature invalidated.
        assert worker._suffix_conflict_signature is None

    async def test_case_b_cap_exceeded_no_rebase(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """CASE B: bounce cap already at MERGE_BOUNCE_CAP → escalate WITHOUT rebase."""
        worker, req = await self._make_worker_with_conflict_setup(git_ops, config, git_repo)

        # Pre-seed registry to MERGE_BOUNCE_CAP (so the NEXT record_bounce exceeds it).
        for _ in range(MERGE_BOUNCE_CAP):
            worker._bounce_registry.record_bounce('591')
        assert worker._bounce_registry.count('591') == MERGE_BOUNCE_CAP

        # Spy on rebase_onto_main — it must NOT be called when cap is exceeded.
        rebase_spy = AsyncMock(return_value=True)
        worker._git_ops.rebase_onto_main = rebase_spy  # type: ignore[method-assign]

        await worker._bounce_conflicting_suffix_items()

        # rebase_onto_main must NOT have been called.
        rebase_spy.assert_not_awaited()

        # Item removed from lane buffer.
        assert req not in worker._lane_buffers['normal'], (
            'Expected req to be removed from lane buffer when bounce cap is exceeded.'
        )

        # Future resolved 'blocked' with NEEDS_REBASE_REASON_PREFIX.
        assert req.result.done(), 'Expected req.result to be resolved after cap escalation.'
        outcome: MergeOutcome = req.result.result()
        assert outcome.status == 'blocked'
        assert outcome.reason is not None
        assert outcome.reason.startswith(NEEDS_REBASE_REASON_PREFIX), (
            f'Expected cap-exceeded reason to start with NEEDS_REBASE_REASON_PREFIX, '
            f'got {outcome.reason!r}.'
        )

        # Debounce signature invalidated.
        assert worker._suffix_conflict_signature is None


# ── step-09 RED: _acquire_next_request() wires bounce at graph time ───────────


@pytest.mark.asyncio
class TestAcquireWiring:
    """_acquire_next_request() must call _bounce_conflicting_suffix_items()
    AFTER recompute_suffix_conflict_graph() and BEFORE _pop_next_pickable(),
    so a conflicting item is diverted before consuming a verify slot.

    RED until step-10 GREEN inserts the await into _acquire_next_request().
    """

    async def test_bounce_called_after_recompute_before_pop(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """ORDER: bounce is called after recompute and before _pop_next_pickable."""
        worker = _make_worker(git_ops)
        worker._shutdown_signaled = True  # exit after one cycle

        call_order: list[str] = []

        async def fake_recompute() -> None:
            call_order.append('recompute')

        async def fake_bounce() -> None:
            call_order.append('bounce')

        def fake_pop() -> MergeRequest | None:
            call_order.append('pop')
            return None  # nothing to pick; shutdown will return None

        worker.recompute_suffix_conflict_graph = fake_recompute  # type: ignore[method-assign]
        worker._bounce_conflicting_suffix_items = fake_bounce  # type: ignore[method-assign]
        worker._pop_next_pickable = fake_pop  # type: ignore[method-assign]

        result = await worker._acquire_next_request()

        # Should have returned None (shutdown signaled, nothing pickable).
        assert result is None

        # Bounce must appear AFTER recompute and BEFORE pop.
        assert 'bounce' in call_order, (
            f'_bounce_conflicting_suffix_items was not called at all. '
            f'call_order={call_order!r}. RED until step-10 wires the bounce.'
        )
        recompute_idx = call_order.index('recompute')
        bounce_idx = call_order.index('bounce')
        pop_idx = call_order.index('pop')
        assert recompute_idx < bounce_idx < pop_idx, (
            f'Expected recompute < bounce < pop but got call_order={call_order!r}.'
        )

    async def test_conflicting_item_diverted_before_verify_slot(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """NO VERIFY SLOT: a conflicting item is escalated before _pop_next_pickable
        can return it for verify dispatch.

        Setup: one conflicting req in lane buffer; rebase→False (escalate path);
        _shutdown_signaled=True so the method exits without blocking.

        Expected WITH wiring (GREEN): bounce removes req from lane buffer →
        _pop_next_pickable returns None → _acquire_next_request() returns None.
        req.result is done with status='blocked'.

        Expected WITHOUT wiring (RED): bounce not called → _pop_next_pickable
        returns req → _acquire_next_request() returns req (wrong!).
        """
        worker = _make_worker(git_ops)

        # Build a frozen entry so frozen_prefix_tip is not bare main.
        _, item_frozen = _make_fake_item(
            'frozen-task', merge_commit='deadbeef1234', config=config, git_repo=git_repo,
        )
        entry_frozen = _make_inflight_entry(item_frozen, verifying=True)
        worker._inflight.append(entry_frozen)

        # Put one conflicting req in lane buffer.
        req = _make_req('591', '591', config, git_repo)
        worker._lane_buffers['normal'].append(req)

        # Seed the conflict graph so the bounce will process req.
        worker._suffix_conflict_graph = SuffixConflictGraph(
            nodes=(req.request_id,),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset({req.request_id}),
        )

        # Stub rebase → False (escalate; req removed + future resolved 'blocked').
        worker._git_ops.rebase_onto_main = AsyncMock(return_value=False)  # type: ignore[method-assign]

        # Signal shutdown so _acquire_next_request() exits instead of blocking.
        worker._shutdown_signaled = True

        # Override recompute to be a no-op (graph already seeded above).
        async def _noop_recompute() -> None:
            pass
        worker.recompute_suffix_conflict_graph = _noop_recompute  # type: ignore[method-assign]

        result = await worker._acquire_next_request()

        # WITH wiring (GREEN): result is None (item diverted, nothing pickable).
        # WITHOUT wiring (RED): result is req (item NOT diverted, picked by pop).
        assert result is None, (
            f'Expected _acquire_next_request() to return None (conflicting item '
            f'diverted before verify slot) but got {result!r}. This is RED until '
            f'step-10 wires _bounce_conflicting_suffix_items() into _acquire_next_request().'
        )

        # The item's future must be resolved as blocked.
        assert req.result.done(), (
            'Expected req.result to be resolved as blocked after graph-time bounce.'
        )
        outcome: MergeOutcome = req.result.result()
        assert outcome.status == 'blocked'
        assert outcome.reason is not None
        assert outcome.reason.startswith(NEEDS_REBASE_REASON_PREFIX)
