"""B+H integration gate (mechanisms 1+2) — two-way boundary suite B1-B6.

PRD W11 omega (``plans/worktree-lane-lifecycle-prd.md``). TEST-ONLY gate: the
six behaviors below (gamma acquire-writer, delta crash-recovery reader,
epsilon1 contamination structural prevention, epsilon2 clean-survival) are
already merged. These tests drive the REAL writer (GitOps.acquire_warm_lane),
the REAL reader (Harness._recover_crashed_tasks), and the REAL dashboard
reader (dashboard.data.orchestrator.read_task_artifacts) over a real local
git repo, proving they agree on the durable-record location/format/semantics
through their production code paths — not the mocked-git unit tests already
covered by test_crash_recovery.py::TestRecordDrivenRecovery.

Boundary suite:
    B1 — writer->reader round-trip (adopt on exact git-reality match)
    B2 — crash->quarantine (orphaned admin-entry divergence, 2097/2098)
    B3 — illegal transition escalates (born-at-L2, real EscalationQueue)
    B4 — hostile `git add -A` stages ZERO task-meta (STRUCTURAL, load-bearing
         for task theta's later guard deletion)
    B5 — durable record + .task-meta survive `git clean -xfd` + `checkout -f`
    B6 — dashboard reader resolves a relocated lane (new-then-old)

Each test is expected GREEN once its fixture wiring converges: this task's
impl steps exercise the existing seam rather than add production code.

Module-local fixtures only (no conftest.py edit) — a conftest edit trips
verify.py's has_conftest full-suite fallback for merge-time verify.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps, WorktreeInfo, _run
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import (
    ESCALATION_SENTINEL_ROLE,
    IllegalLaneTransition,
    LaneLifecycle,
)
from orchestrator.lane_lifecycle import LaneState as DurableLaneState
from orchestrator.warm_lane_pool import LaneState, WarmLanePool

# B6 imports `dashboard.data.orchestrator` from inside the test bodies below
# (deferred to avoid F401 — see module docstring). `orchestrator/tests/
# conftest.py` puts orchestrator/shared/escalation `src/` dirs on sys.path
# but deliberately not dashboard's, and the root conftest.py that *does* wire
# up dashboard/src is never loaded here — verify.py (and pytest's own
# confcutdir) roots the run at `orchestrator/`, above which pytest does not
# ascend for conftest discovery. Rather than mutate sys.path at
# module-import/collection time (which would leak dashboard/src onto the
# import path for the rest of the pytest session — every other orchestrator
# test module collected in the same run — risking module-name shadowing or
# cross-test contamination), the `_dashboard_on_path` fixture below scopes
# the insertion to just the two B6 tests that need it via
# `monkeypatch.syspath_prepend`, which reverts automatically at fixture
# teardown. This still avoids editing orchestrator/tests/conftest.py (which
# would trip verify.py's has_conftest full-suite fallback per this task's
# design decisions).
_DASHBOARD_SRC = Path(__file__).parent.parent.parent / 'dashboard' / 'src'


@pytest.fixture
def _dashboard_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put dashboard/src on sys.path for the duration of one B6 test only."""
    monkeypatch.syspath_prepend(str(_DASHBOARD_SRC))


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


async def _add_warm_lane_scripts(repo: Path, port: int = 39411) -> None:
    """Commit stub seed-warm-lane.sh + setup-worktree-debug-port.sh into repo.

    No real ssh/reify — the seed stub only needs to mkdir target/ and drop a
    marker file so acquire_warm_lane's seed step exits 0.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed_script = scripts_dir / 'seed-warm-lane.sh'
    seed_script.write_text(
        '#!/usr/bin/env bash\nmkdir -p "$2/target"\necho "seeded" > "$2/target/seeded.bin"\n'
    )
    seed_script.chmod(0o755)
    debug_script = scripts_dir / 'setup-worktree-debug-port.sh'
    debug_script.write_text(f'#!/usr/bin/env bash\necho {port}\n')
    debug_script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add warm-lane scripts'], cwd=repo)


def _make_orch_config(repo: Path, *, max_concurrent_tasks: int = 1) -> OrchestratorConfig:
    """Build a minimal real OrchestratorConfig with the warm-lane pool on."""
    return OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=max_concurrent_tasks,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
            warm_lane_pool=True,
        ),
    )


def _build_harness(config: OrchestratorConfig) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    GitOps is deliberately left REAL (not patched) — this gate drives the
    real writer/reader seam. Mirrors test_warm_lane_integration_gate.py /
    test_harness_warm_lane_wiring.py's _build_harness.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        return Harness(config)


async def _get_head(repo: Path) -> str:
    """Return the HEAD commit SHA of the repo."""
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'git rev-parse HEAD failed (rc={rc})'
    return out.strip()


def _make_plan(done: int, total: int, task_id: str = 'test') -> dict:
    """Build a plan dict with the given step-completion counts."""
    steps = []
    for i in range(total):
        steps.append({
            'id': f'step-{i + 1}',
            'description': f'Step {i + 1}',
            'status': 'done' if i < done else 'pending',
            'commit': f'abc{i}' if i < done else None,
        })
    return {'task_id': task_id, 'title': 'Test Task', 'steps': steps}


async def _acquire_lane(
    git_ops: GitOps,
    task_id: str,
    start_ref: str,
    *,
    expected_title: str | None = None,
) -> Path:
    """Acquire a warm lane for *task_id*, asserting success, and return its path."""
    wt = await git_ops.acquire_warm_lane(task_id, start_ref, expected_title=expected_title)
    assert isinstance(wt, WorktreeInfo), f'acquire_warm_lane failed: {wt!r}'
    return wt.path


def _wire_recovery_scheduler(harness: Harness) -> None:
    """Stub the scheduler + event_store surface _recover_crashed_tasks reads.

    get_task returns a title-less (but non-None) dict so the identity guard
    (config.worktree_identity_guard_enabled defaults True) fails OPEN
    (identities_match treats an empty title as a match) instead of deferring.
    """
    harness.scheduler.get_status = AsyncMock(return_value=None)
    harness.scheduler.get_task = AsyncMock(return_value={})
    harness.scheduler.get_tasks = AsyncMock(return_value=[])
    harness.scheduler.is_deterministic = MagicMock(return_value=False)
    harness.event_store = MagicMock()


# ── B1 — writer->reader round-trip (adopt) ──────────────────────────────


@pytest.mark.asyncio
class TestB1WriterReaderRoundTrip:
    """B1: acquire a lane for a task, crash (fresh in-memory pool cache,
    durable record + git worktree persist), recover — the reader must adopt
    the lane by reading the SAME durable record the REAL writer produced.
    """

    async def test_acquire_then_recover_adopts_lane(self, ig_git_repo: Path):
        repo = ig_git_repo
        await _add_warm_lane_scripts(repo)
        harness = _build_harness(_make_orch_config(repo))
        _wire_recovery_scheduler(harness)
        # Strengthen the round-trip claim: exercise the identity guard's
        # genuine BOTH-present, equal branch (harness.py:2151-2190) rather
        # than the fail-open default `_wire_recovery_scheduler` sets up —
        # return the SAME title the writer stores below via `ta.init`, so
        # `identities_match` actually compares 'B1 task' == 'B1 task'
        # instead of short-circuiting on a title-less dict.
        harness.scheduler.get_task = AsyncMock(return_value={'title': 'B1 task'})

        # WRITER: real acquire_warm_lane — real `git worktree add`.
        head = await _get_head(repo)
        lane = await _acquire_lane(harness.git_ops, '42', head, expected_title='B1 task')
        assert lane.name.startswith('_lane-')

        # Write crashed progress under the sibling .task-meta root.
        meta = TaskArtifacts.meta_root_for(harness.git_ops.worktree_base, lane.name)
        ta = TaskArtifacts(lane, meta)
        ta.init('42', 'B1 task', 'd')
        ta.write_plan(_make_plan(3, 5, '42'))

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
        await _add_warm_lane_scripts(repo)
        harness = _build_harness(_make_orch_config(repo))
        _wire_recovery_scheduler(harness)

        head = await _get_head(repo)
        lane = await _acquire_lane(harness.git_ops, '42', head)
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
        await _add_warm_lane_scripts(repo)
        harness = _build_harness(_make_orch_config(repo))
        _wire_recovery_scheduler(harness)
        # Diverging live title — the guard must catch this even though the
        # admin-entry + branch checks that gate it (harness.py:2140) both
        # pass below.
        harness.scheduler.get_task = AsyncMock(
            return_value={'title': 'Unrelated live task'},
        )

        head = await _get_head(repo)
        lane = await _acquire_lane(harness.git_ops, '42', head, expected_title='B2 identity')
        meta = TaskArtifacts.meta_root_for(harness.git_ops.worktree_base, lane.name)
        ta = TaskArtifacts(lane, meta)
        ta.init('42', 'B2 identity', 'd')
        ta.write_plan(_make_plan(1, 2, '42'))

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
        await _add_warm_lane_scripts(repo)
        harness = _build_harness(_make_orch_config(repo))
        _wire_recovery_scheduler(harness)

        lane = await _acquire_lane(harness.git_ops, '42', await _get_head(repo))
        meta = TaskArtifacts.meta_root_for(harness.git_ops.worktree_base, lane.name)
        ta = TaskArtifacts(lane, meta)
        ta.init('42', 'B4', 'd')
        ta.write_plan(_make_plan(1, 2, '42'))
        ta.append_iteration_log({'k': 1})

        # Metadata lives OUTSIDE the worktree: TaskArtifacts was pointed at
        # the sibling `meta` root, never at `<lane>/.task`.  Nothing writes
        # `<lane>/.task` any more, either — acquire's _ensure_task_gitignore
        # defense-in-depth guard, the last thing that unconditionally
        # created it, is gone.
        assert (meta / 'plan.json').exists()
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
        assert (meta / 'plan.json').exists()


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
        await _add_warm_lane_scripts(repo)
        harness = _build_harness(_make_orch_config(repo))
        _wire_recovery_scheduler(harness)

        base = harness.git_ops.worktree_base
        lane = await _acquire_lane(harness.git_ops, '42', await _get_head(repo))
        meta = TaskArtifacts.meta_root_for(base, lane.name)
        ta = TaskArtifacts(lane, meta)
        ta.init('42', 'B5', 'd')
        ta.write_plan(_make_plan(1, 2, '42'))

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

        assert (meta / 'plan.json').exists()
        assert TaskArtifacts(lane, meta).read_plan()['steps']


# ── B6 — dashboard reader (new-then-old) ────────────────────────────────


@pytest.mark.asyncio
class TestB6DashboardReaderNewPath:
    """B6 (new path): dashboard's read_task_artifacts resolves an acquired
    lane's relocated .task-meta sidecar (worktree_path.parent / '.task-meta'
    / worktree_path.name), exercising the same meta_root_for path shape the
    real writer (GitOps/TaskArtifacts) uses.
    """

    async def test_reads_relocated_task_meta(
        self, ig_git_repo: Path, _dashboard_on_path: None,
    ):
        from dashboard.data.orchestrator import (  # pyright: ignore[reportMissingImports]
            read_task_artifacts,
        )

        repo = ig_git_repo
        await _add_warm_lane_scripts(repo)
        harness = _build_harness(_make_orch_config(repo))
        _wire_recovery_scheduler(harness)

        base = harness.git_ops.worktree_base
        lane = await _acquire_lane(harness.git_ops, '42', await _get_head(repo))
        meta = TaskArtifacts.meta_root_for(base, lane.name)
        ta = TaskArtifacts(lane, meta)
        ta.init('42', 'B6 dash', 'd')
        ta.write_plan(_make_plan(3, 5, '42'))
        ta.append_iteration_log({'a': 1})
        ta.append_iteration_log({'b': 2})

        result = read_task_artifacts(lane)
        assert result['phase'] == 'EXECUTE'
        assert result['plan_progress'] == {'done': 3, 'total': 5}
        assert result['iteration_count'] == 2
        assert result['metadata']['task_id'] == '42'


class TestB6DashboardReaderLegacyPath:
    """B6 (old path, compat): read_task_artifacts still parses a legacy
    `<worktree>/.task/` layout when no `.task-meta` sibling exists.
    """

    def test_reads_legacy_task_dir(self, tmp_path: Path, _dashboard_on_path: None):
        from dashboard.data.orchestrator import (  # pyright: ignore[reportMissingImports]
            read_task_artifacts,
        )

        base = tmp_path / '.worktrees'
        legacy = base / '_legacy'
        (legacy / '.task').mkdir(parents=True)
        (legacy / '.task' / 'metadata.json').write_text(json.dumps({'task_id': 'legacy'}))
        (legacy / '.task' / 'plan.json').write_text(json.dumps({
            'steps': [
                {'id': 'step-1', 'status': 'done'},
                {'id': 'step-2', 'status': 'pending'},
            ],
        }))

        result = read_task_artifacts(legacy)
        assert result['plan_progress']['total'] > 0
        assert result['metadata'] is not None
        assert result['metadata']['task_id'] == 'legacy'
