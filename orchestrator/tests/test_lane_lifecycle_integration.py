"""B+H integration gate (mechanisms 1+2) — boundary suite B1-B5.

PRD W11 omega (``plans/worktree-lane-lifecycle-prd.md``). TEST-ONLY gate: the
behaviors below (gamma acquire-writer, delta crash-recovery reader,
epsilon1 contamination structural prevention, epsilon2 clean-survival) are
already merged. These tests drive the REAL writer (GitOps.acquire_warm_lane)
and the REAL reader (Harness._recover_crashed_tasks) over a real local git
repo, proving they agree on the durable-record location/format/semantics
through their production code paths — not the mocked-git unit tests already
covered by test_crash_recovery.py::TestRecordDrivenRecovery.

Boundary suite:
    B1 — writer->reader round-trip (adopt on exact git-reality match)
    B2 — crash->quarantine (orphaned admin-entry divergence, 2097/2098)
    B3 — illegal transition escalates (born-at-L2, real EscalationQueue)
    B4 — hostile `git add -A` stages ZERO task-meta (STRUCTURAL, load-bearing
         for task theta's later guard deletion)
    B5 — durable record + .task-meta survive `git clean -xfd` + `checkout -f`
    B7 — REUSE dispatch refreshes updated_at; crash-recovery restore does not
         (task 7863)

B6 (dashboard reader resolves a relocated lane) was RETIRED by task
gamma/2636 (``plans/dashboard-task-runtime-endpoint-prd.md``): that task
deleted the dashboard's hand-rolled worktree reader
(dashboard.data.orchestrator.read_task_artifacts) in favor of consuming
per-task runtime state over MCP (get_task_runtime_state), by design severing
the very format coupling B6 existed to prove. With no dashboard-side disk
reader left, the two-way writer<->dashboard-reader boundary B6 checked no
longer exists, so its two test classes and the dashboard-src sys.path
plumbing were removed here alongside that deletion.

Each test is expected GREEN once its fixture wiring converges: this task's
impl steps exercise the existing seam rather than add production code.

Module-local fixtures only (no conftest.py edit) — a conftest edit trips
verify.py's has_conftest full-suite fallback for merge-time verify.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _lane_test_helpers import (
    acquire_lane,
    add_warm_lane_scripts,
    build_harness,
    get_head,
    make_orch_config,
    make_plan,
    wire_recovery_scheduler,
)

from orchestrator.artifacts import TaskArtifacts
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps, WorktreeInfo, _run
from orchestrator.lane_lifecycle import (
    ESCALATION_SENTINEL_ROLE,
    IllegalLaneTransition,
    LaneLifecycle,
    LaneRecord,
)
from orchestrator.lane_lifecycle import LaneState as DurableLaneState
from orchestrator.warm_lane_pool import LaneState, WarmLanePool

# ── Module-local fixtures + helpers ─────────────────────────────────────


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def ig_git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit + a resolvable warm base.

    Pre-creates the default derived warm-lane base (task 2061 gate) so the
    acquire_warm_lane pre-acquire base-health gate sees WarmBaseHealth.OK,
    and marks `.worktrees/.pool-root` present (task 2099 create-once guard) —
    mirrors test_lane_lifecycle_gitops.py's git_repo / test_warm_lane_
    integration_gate.py's ig_git_repo fixtures.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    default_base = repo / '.worktrees' / '_merge-verify' / 'target'
    default_base.mkdir(parents=True, exist_ok=True)
    (default_base / '.keep').write_text('warm base sentinel\n')
    (repo / '.worktrees' / '.pool-root').touch()
    return repo


def _backdate_record(lifecycle: LaneLifecycle, lane: Path, *, days: int) -> str:
    """Rewrite *lane*'s on-disk updated_at to *days* ago; return that stamp."""
    path = lifecycle.state_dir / f'{lane.name}.json'
    record = LaneRecord.from_json(path.read_text(encoding='utf-8'))
    stamp = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    path.write_text(replace(record, updated_at=stamp).to_json(), encoding='utf-8')
    return stamp


# ── B1 — writer->reader round-trip (adopt) ──────────────────────────────


@pytest.mark.asyncio
class TestB1WriterReaderRoundTrip:
    """B1: acquire a lane for a task, crash (fresh in-memory pool cache,
    durable record + git worktree persist), recover — the reader must adopt
    the lane by reading the SAME durable record the REAL writer produced.
    """

    async def test_acquire_then_recover_adopts_lane(self, ig_git_repo: Path):
        repo = ig_git_repo
        await add_warm_lane_scripts(repo)
        harness = build_harness(make_orch_config(repo))
        wire_recovery_scheduler(harness)
        # Strengthen the round-trip claim: exercise the identity guard's
        # genuine BOTH-present, equal branch (harness.py:2151-2190) rather
        # than the fail-open default `wire_recovery_scheduler` sets up —
        # return the SAME title the writer stores below via `ta.init`, so
        # `identities_match` actually compares 'B1 task' == 'B1 task'
        # instead of short-circuiting on a title-less dict.
        harness.scheduler.get_task = AsyncMock(return_value={'title': 'B1 task'})

        # WRITER: real acquire_warm_lane — real `git worktree add`.
        head = await get_head(repo)
        lane = await acquire_lane(harness.git_ops, '42', head, expected_title='B1 task')
        assert lane.name.startswith('_lane-')

        # Write crashed progress under the sibling .task-meta roots, as a
        # dispatch of '42' into this lane leaves them.
        ta = TaskArtifacts.for_task(lane, '42')
        ta.init('42', 'B1 task', 'd')
        ta.write_plan(make_plan(3, 5, '42'))

        # Simulate restart: fresh in-memory pool cache; durable record +
        # git worktree persist on disk.
        old_pool = harness.git_ops.warm_lane_pool
        assert old_pool is not None
        harness.git_ops.warm_lane_pool = WarmLanePool(
            worktree_base=harness.git_ops.worktree_base,
            size=old_pool.size,
        )

        # READER: real crash recovery.
        await harness._recover_crashed_tasks()

        pool = harness.git_ops.warm_lane_pool
        assert pool.assignment_for('42') == lane
        assert pool.state(lane) == LaneState.ASSIGNED

        rec = harness.git_ops._lane_lifecycle.read(lane)
        assert rec is not None
        assert rec.state == DurableLaneState.ASSIGNED
        assert rec.task_id == '42'

        assert '42' in harness._recovered_plans
        recovered = harness._recovered_plans['42']
        assert len(recovered['steps']) == 5
        assert sum(1 for s in recovered['steps'] if s['status'] == 'done') == 3
        assert lane.exists()
        assert rec.state != DurableLaneState.QUARANTINED


# ── B2 — crash->quarantine (orphaned admin-entry divergence) ───────────


async def _assert_parked_once(git_ops: GitOps, lane: Path) -> None:
    """*lane* left worktree_base for exactly one quarantine parking, its branch renamed to match.

    Task 4845: a full ``task/42`` record branch must not double-prefix, and an
    unregistered lane must still be moved.
    """
    assert not lane.exists()
    parkings = list(git_ops.quarantine_base.glob('42-*'))
    assert len(parkings) == 1, parkings
    assert await git_ops.resolve_branch_sha('task/42') is None
    assert await git_ops.resolve_branch_sha(f'task/{parkings[0].name}') is not None


@pytest.mark.asyncio
class TestB2CrashQuarantine:
    """B2: crash-recovery divergence between the durable record and git
    reality must QUARANTINE, never re-pin. Two distinct divergence shapes
    share this class:

    - ``test_orphaned_registration_quarantines_never_repins`` — the lane's
      git admin entry (.git/worktrees/<lane>) is gone (the 2097/2098
      orphaned-registration divergence).
    - ``test_identity_mismatch_quarantines_never_repins`` — the admin entry
      and checked-out branch are BOTH intact (the checks gating the
      identity guard, harness.py:2140, both pass), but the live task's
      title diverges from the lane's stored .task-meta title (a
      recycled-id collision, reify 3770) — the identity-mismatch branch at
      harness.py:2151-2190, otherwise only exercised via B1's positive
      match / the shared helper's fail-open default.
    """

    async def test_orphaned_registration_quarantines_never_repins(
        self, ig_git_repo: Path, caplog,
    ):
        import logging

        repo = ig_git_repo
        await add_warm_lane_scripts(repo)
        harness = build_harness(make_orch_config(repo))
        wire_recovery_scheduler(harness)

        head = await get_head(repo)
        lane = await acquire_lane(harness.git_ops, '42', head)
        assert (repo / '.git' / 'worktrees' / lane.name).exists()

        old_pool = harness.git_ops.warm_lane_pool
        assert old_pool is not None
        harness.git_ops.warm_lane_pool = WarmLanePool(
            worktree_base=harness.git_ops.worktree_base,
            size=old_pool.size,
        )

        # Simulate the 2097/2098 orphan: the admin entry is gone but the
        # lane dir + durable record survive.
        shutil.rmtree(repo / '.git' / 'worktrees' / lane.name)
        assert await harness.git_ops._is_registered_worktree(lane) is False

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._recover_crashed_tasks()

        rec = harness.git_ops._lane_lifecycle.read(lane)
        assert rec is not None
        assert rec.state == DurableLaneState.QUARANTINED

        pool = harness.git_ops.warm_lane_pool
        assert pool.assignment_for('42') is None
        assert pool.state(lane) == LaneState.FREE
        assert '42' not in harness._recovered_plans

        # Match on the event type across both positional and keyword call
        # forms — resilient to a future `emit(event_type=...)` call-site
        # change, which would otherwise raise IndexError here instead of
        # cleanly failing the membership assertion below.
        emitted = {
            (c.args[0] if c.args else c.kwargs.get('event_type'))
            for c in harness.event_store.emit.call_args_list  # type: ignore[union-attr]
        }
        assert EventType.worktree_quarantined in emitted

        assert any(
            log_rec.levelno == logging.WARNING
            and '42' in log_rec.getMessage()
            and 'quarantin' in log_rec.getMessage().lower()
            for log_rec in caplog.records
        ), f'expected a quarantine warning naming task 42; got: {caplog.text!r}'

        await _assert_parked_once(harness.git_ops, lane)

        # Next dispatch is clean: a different task can still acquire a lane.
        # Unpack the tuple rather than a bare `is not None` smoke check —
        # `acquire_for` always returns a (Path, bool) tuple when it succeeds,
        # so `is not None` would pass even for a stale/reused lane; assert
        # directly on a fresh (non-reused) FREE-lane acquisition instead.
        acquired = await pool.acquire_for('task/99')
        assert acquired is not None
        lane_path, reused = acquired
        assert lane_path is not None
        assert not reused

    async def test_identity_mismatch_quarantines_never_repins(
        self, ig_git_repo: Path, caplog,
    ):
        """Boundary case for the identity-mismatch branch (harness.py
        :2151-2190): the admin entry AND checked-out branch both still
        match (unlike the orphan case above), but the live task's title
        diverges from the lane's stored `.task-meta` title — a
        recycled-id collision (reify 3770). Recovery must quarantine via
        the identity guard itself, not the git-reality checks that gate it.
        """
        import logging

        repo = ig_git_repo
        await add_warm_lane_scripts(repo)
        harness = build_harness(make_orch_config(repo))
        wire_recovery_scheduler(harness)
        # Diverging live title — the guard must catch this even though the
        # admin-entry + branch checks that gate it (harness.py:2140) both
        # pass below.
        harness.scheduler.get_task = AsyncMock(
            return_value={'title': 'Unrelated live task'},
        )

        head = await get_head(repo)
        lane = await acquire_lane(harness.git_ops, '42', head, expected_title='B2 identity')
        ta = TaskArtifacts.for_task(lane, '42')
        ta.init('42', 'B2 identity', 'd')
        ta.write_plan(make_plan(1, 2, '42'))

        # Admin entry + branch stay intact — the divergence here is
        # identity, not registration (contrast with the orphan case above).
        assert (repo / '.git' / 'worktrees' / lane.name).exists()

        old_pool = harness.git_ops.warm_lane_pool
        assert old_pool is not None
        harness.git_ops.warm_lane_pool = WarmLanePool(
            worktree_base=harness.git_ops.worktree_base,
            size=old_pool.size,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            await harness._recover_crashed_tasks()

        rec = harness.git_ops._lane_lifecycle.read(lane)
        assert rec is not None
        assert rec.state == DurableLaneState.QUARANTINED

        pool = harness.git_ops.warm_lane_pool
        assert pool.assignment_for('42') is None
        assert pool.state(lane) == LaneState.FREE
        assert '42' not in harness._recovered_plans

        # Match the SPECIFIC identity-mismatch reason, not just the event
        # type, so this stays distinguishable from the orphan-registration
        # quarantine above (`recovery-record-divergence`).
        emitted_reasons = [
            (c.kwargs.get('data') or {}).get('reason')
            for c in harness.event_store.emit.call_args_list  # type: ignore[union-attr]
            if (c.args[0] if c.args else c.kwargs.get('event_type'))
            == EventType.worktree_quarantined
        ]
        assert 'recovery-identity-mismatch' in emitted_reasons

        assert any(
            log_rec.levelno == logging.WARNING
            and '42' in log_rec.getMessage()
            and 'identity' in log_rec.getMessage().lower()
            for log_rec in caplog.records
        ), f'expected an identity-mismatch warning naming task 42; got: {caplog.text!r}'

        await _assert_parked_once(harness.git_ops, lane)

        # Next dispatch is clean: a different task can still acquire a lane.
        # Unpack the tuple rather than a bare `is not None` smoke check —
        # `acquire_for` always returns a (Path, bool) tuple when it succeeds,
        # so `is not None` would pass even for a stale/reused lane; assert
        # directly on a fresh (non-reused) FREE-lane acquisition instead.
        acquired = await pool.acquire_for('task/99')
        assert acquired is not None
        lane_path, reused = acquired
        assert lane_path is not None
        assert not reused


# ── B3 — illegal transition escalates (real EscalationQueue) ───────────


class TestB3IllegalTransitionEscalates:
    """B3: forcing RELEASED->IN_USE raises IllegalLaneTransition and files a
    born-at-L2 escalation via a REAL EscalationQueue; the durable record is
    left unchanged. Standalone — no Harness/GitOps needed.
    """

    def test_released_to_in_use_raises_and_escalates(self, tmp_path: Path):
        from escalation.models import BORN_AT_L2_SEVERITIES
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path / 'esc')
        lc = LaneLifecycle(tmp_path / '.worktrees', escalation_queue=queue)
        lane = tmp_path / '.worktrees' / '_lane-0'

        # Drive the legal ladder to RELEASED.
        lc.transition(lane, DurableLaneState.SEED, seeded_from_sha='abc')
        lc.transition(lane, DurableLaneState.REGISTERED, branch='task/42')
        lc.transition(lane, DurableLaneState.ASSIGNED, task_id='42', branch='task/42')
        lc.transition(lane, DurableLaneState.RELEASED)
        released_rec = lc.read(lane)
        assert released_rec is not None
        assert released_rec.state == DurableLaneState.RELEASED

        with pytest.raises(IllegalLaneTransition):
            lc.transition(lane, DurableLaneState.IN_USE)

        # Record left unchanged (never silent-heal).
        unchanged_rec = lc.read(lane)
        assert unchanged_rec is not None
        assert unchanged_rec.state == DurableLaneState.RELEASED

        pending = queue.get_by_task(f'lane-lifecycle-{lane.name}', status='pending')
        assert pending
        esc = pending[0]
        assert esc.agent_role == ESCALATION_SENTINEL_ROLE
        assert esc.level == 2
        assert esc.severity in BORN_AT_L2_SEVERITIES


# ── B4 — hostile `git add -A` stages ZERO task-meta (STRUCTURAL) ───────


@pytest.mark.asyncio
class TestB4Contamination:
    """B4 (load-bearing): a hostile agent runs raw `git add -A && git
    commit` directly in the lane, deliberately bypassing GitOps.commit()
    (and every scrub guard it runs). The commit must stage ZERO task-meta
    paths — contamination prevention here is STRUCTURAL (the .task-meta
    sidecar lives OUTSIDE the worktree), not guard-defended, so this must
    keep passing after task theta later deletes the scrub guards.
    """

    async def test_hostile_add_all_stages_zero_task_meta(self, ig_git_repo: Path):
        repo = ig_git_repo
        await add_warm_lane_scripts(repo)
        harness = build_harness(make_orch_config(repo))
        wire_recovery_scheduler(harness)

        lane = await acquire_lane(harness.git_ops, '42', await get_head(repo))
        ta = TaskArtifacts.for_task(lane, '42')
        ta.init('42', 'B4', 'd')
        ta.write_plan(make_plan(1, 2, '42'))
        ta.append_iteration_log({'k': 1})

        # Metadata lives OUTSIDE the worktree: TaskArtifacts was pointed at
        # the sibling `.task-meta` roots, never at `<lane>/.task`.  Nothing
        # writes `<lane>/.task` any more, either — acquire's
        # _ensure_task_gitignore defense-in-depth guard, the last thing that
        # unconditionally created it, is gone.
        assert ta.artifact_path('plan.json').exists()
        assert not (lane / '.task' / 'plan.json').exists()
        assert not (lane / '.task-meta').exists()

        # Hostile agent uses RAW git — deliberately bypasses GitOps.commit
        # so NO scrub guard participates.
        (lane / 'hello.py').write_text('print(1)\n')
        await _run(['git', 'add', '-A'], cwd=lane)
        await _run(['git', 'commit', '-m', 'hostile add -A'], cwd=lane)

        rc, out, _ = await _run(['git', 'ls-tree', '-r', '--name-only', 'HEAD'], cwd=lane)
        assert rc == 0
        assert 'hello.py' in out
        for forbidden in ('.task', '.task-meta', 'plan.json', 'metadata.json', 'iterations.jsonl'):
            assert forbidden not in out, f'{forbidden!r} leaked into the hostile commit tree'

        # The sidecar is untouched by the commit.
        assert ta.artifact_path('plan.json').exists()


# ── B5 — survives cleaning ──────────────────────────────────────────────


@pytest.mark.asyncio
class TestB5SurvivesCleaning:
    """B5: `git clean -xfd` + `git checkout -f` run INSIDE the lane must
    not touch the durable .lane-state record or the .task-meta artifacts —
    both live outside the worktree as siblings of it.
    """

    async def test_lane_state_and_task_meta_survive_clean_and_checkout(
        self, ig_git_repo: Path,
    ):
        repo = ig_git_repo
        await add_warm_lane_scripts(repo)
        harness = build_harness(make_orch_config(repo))
        wire_recovery_scheduler(harness)

        base = harness.git_ops.worktree_base
        lane = await acquire_lane(harness.git_ops, '42', await get_head(repo))
        ta = TaskArtifacts.for_task(lane, '42')
        ta.init('42', 'B5', 'd')
        ta.write_plan(make_plan(1, 2, '42'))

        record_path = base / '.lane-state' / f'{lane.name}.json'
        assert record_path.exists()

        # Hostile clean INSIDE the lane.
        await _run(['git', 'clean', '-xfd'], cwd=lane)
        await _run(['git', 'checkout', '-f'], cwd=lane)

        # Both siblings survive — they live outside the worktree.
        assert record_path.exists()
        rec = harness.git_ops._lane_lifecycle.read(lane)
        assert rec is not None
        assert rec.state == DurableLaneState.ASSIGNED
        assert rec.task_id == '42'

        assert ta.artifact_path('plan.json').exists()


# ── B7 — REUSE dispatch refreshes the record's age (task 7863) ─────────


@pytest.mark.asyncio
class TestB7ReuseDispatchRefreshesRecordAge:
    """B7: the durable record's updated_at is warm-lane-gc.sh's record-gate
    age clock. A restart's crash-recovery restore must leave it alone (or
    abandoned lanes stay young), while the next REUSE dispatch of the same
    task must move it to now (or an actively re-dispatched task ages out of
    GC protection).
    """

    async def test_restore_keeps_age_and_reuse_dispatch_refreshes_it(
        self, ig_git_repo: Path, caplog,
    ):
        repo = ig_git_repo
        await add_warm_lane_scripts(repo)
        harness = build_harness(make_orch_config(repo))
        wire_recovery_scheduler(harness)
        harness.scheduler.get_task = AsyncMock(return_value={'title': 'B7 task'})

        head = await get_head(repo)
        lane = await acquire_lane(harness.git_ops, '42', head, expected_title='B7 task')
        ta = TaskArtifacts.for_task(lane, '42')
        ta.init('42', 'B7 task', 'd')
        ta.write_plan(make_plan(1, 3, '42'))

        lifecycle = LaneLifecycle(harness.git_ops.worktree_base)
        old = _backdate_record(lifecycle, lane, days=30)

        old_pool = harness.git_ops.warm_lane_pool
        assert old_pool is not None
        harness.git_ops.warm_lane_pool = WarmLanePool(
            worktree_base=harness.git_ops.worktree_base,
            size=old_pool.size,
        )

        await harness._recover_crashed_tasks()

        pool = harness.git_ops.warm_lane_pool
        assert pool.assignment_for('42') == lane
        restored = lifecycle.read(lane)
        assert restored is not None
        assert restored.updated_at == old

        with caplog.at_level(logging.INFO, logger='orchestrator.git_ops'):
            wt = await harness.git_ops.acquire_warm_lane(
                '42', head, expected_title='B7 task',
            )

        assert isinstance(wt, WorktreeInfo), f'acquire_warm_lane failed: {wt!r}'
        assert wt.path == lane
        assert 'route=reuse edge=' in caplog.text, caplog.text
        rec = lifecycle.read(lane)
        assert rec is not None
        assert (rec.state, rec.task_id) == (DurableLaneState.ASSIGNED, '42')
        refreshed_at = datetime.fromisoformat(rec.updated_at)
        assert refreshed_at > datetime.fromisoformat(old)
        assert abs(datetime.now(UTC) - refreshed_at) < timedelta(minutes=5)
