"""Warm-lane session-resume INTEGRATION GATE (ω, task 2775).

The G2 user-observable, two-way integration gate for the warm-lane
session-resume feature (plans/warm-lane-session-resume-prd.md §8 boundary
matrix B1–B11). Its dependencies are all landed on this worktree:

  α (2771) sidecar v2 + preserve-on-cancel,
  β (2772) recovery adopts sessions,
  γ (2774) guarded injection + config + events + prompt note.

Where the α/β/γ UNIT tests in test_crash_recovery.py each assert ONE side of a
seam, ω proves the COMPOSED, two-way chain: it drives the REAL
``_recover_crashed_tasks`` to populate ``_recovered_sessions`` /
``_recovered_session_config_dirs`` / ``_recovered_plans`` (β's output), then
feeds that SAME recovered state through the REAL ``_run_slot`` γ guard (and,
for B1/B2/B11, the REAL ``_invoke`` α path) — proving β's output is exactly
γ's input, and that the resumed invocation carries the same session id, the
architect is skipped, and the WIP commit survives.

Fakes live ONLY at the true CLI boundary (``build_workflow``,
``invoke_with_cap_retry``); ω never runs a full ``TaskWorkflow.run()`` (that
would invoke real Claude agents and merge machinery). "Architect skipped" is
proven at the ``_run_slot``→``build_workflow`` seam via the injected
``initial_plan`` (the ratified complete-plan-skips-architect invariant), not by
observing zero live architect invocations.

Fixtures/helpers are module-local (no shared conftest) per the established
convention; only the repo-wide ``mock_orch_config`` conftest fixture and the
``_workflow_helpers`` fakes are reused. The autouse
``_derive_meta_root_like_production`` fixture is deliberately NOT imported so
the invoke-driver's ``TaskArtifacts(cwd)`` uses the legacy ``.task`` root that
``_config_dir`` and the sidecar read/write target.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import namedtuple
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from shared.config_dir import TaskConfigDir

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import IMPLEMENTER
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig, SessionResumeConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import LaneLifecycle
from orchestrator.lane_lifecycle import LaneState as DurableLaneState
from orchestrator.scheduler import TaskAssignment
from orchestrator.warm_lane_pool import WarmLanePool
from orchestrator.workflow import TaskWorkflow


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """A Harness with mocked internals for driving REAL recovery + dispatch.

    Adapted verbatim from test_crash_recovery.py's ``harness`` fixture: heavy
    constructors (McpLifecycle/Scheduler/BriefingAssembler) are patched out,
    ``git_ops`` keeps a REAL worktree_base under ``tmp/.worktrees`` (so
    ``_recover_crashed_tasks`` and ``_run_slot`` run for real), and the
    scheduler/event_store are mocks. ``_lane_lifecycle`` is rebound onto the
    reassigned worktree_base so the record-driven recovery path and the
    ``_seed_lane_record`` helper target the same ``.lane-state`` dir.
    """
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.set_task_status = AsyncMock()
    # Title-less dict → non-None (no defer) + identities_match fails open (adopt).
    h.scheduler.get_task = AsyncMock(return_value={})
    # None status → transient/None → fall through to restore (never terminal-release).
    h.scheduler.get_status = AsyncMock(return_value=None)
    h.scheduler._dispatched = set()
    # is_deterministic is a sync @staticmethod predicate at the top of _run_slot.
    h.scheduler.is_deterministic = MagicMock(return_value=False)

    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    # Mount-presence guard: mark storage present so recovery does not defer.
    h.git_ops.mark_pool_storage_present()
    h.git_ops.cleanup_worktree = AsyncMock()
    h.git_ops.quarantine_worktree = AsyncMock(return_value=None)
    # Rebind lane lifecycle onto the reassigned worktree_base (W11 delta).
    h.git_ops._lane_lifecycle = LaneLifecycle(
        h.git_ops.worktree_base, quarantine_worktree=h.git_ops.quarantine_worktree,
    )
    # Default "still registered" so fabricated lanes take the restore path.
    h.git_ops._is_registered_worktree = AsyncMock(return_value=True)
    # Exercise the best-effort event emits without a real store.
    h.event_store = MagicMock()

    return h


def _make_plan(
    steps_done: int,
    steps_total: int,
    task_id: str = 'test',
    *,
    session_id: str | None = None,
) -> dict:
    """Build a plan dict with the given step-completion counts.

    Mirrors test_crash_recovery.py's ``_make_plan``: when ``session_id`` is
    given the plan is provenance-stamped (signals the recovery path that the
    architect already produced this plan). Recovery keys a plan into
    ``_recovered_plans`` when it has ≥1 ``done`` step.
    """
    steps = []
    for i in range(steps_total):
        steps.append({
            'id': f'step-{i + 1}',
            'description': f'Step {i + 1}',
            'status': 'done' if i < steps_done else 'pending',
            'commit': f'abc{i}' if i < steps_done else None,
        })
    plan: dict = {
        'task_id': task_id,
        'title': 'Test Task',
        'steps': steps,
    }
    if session_id is not None:
        plan['_session_id'] = session_id
    return plan


# ── Composition scaffolding (recover → dispatch seam) ────────────────────────
_DispatchCapture = namedtuple(
    '_DispatchCapture', ['resume_session_id', 'initial_plan', 'emits'],
)


def _attach_pool(harness: Harness, size: int = 2) -> WarmLanePool:
    """Attach a WarmLanePool on the harness's (reassigned) worktree_base so
    ``pool.is_lane('_lane-k')`` recognises a fabricated lane as a lane."""
    base = harness.git_ops.worktree_base
    base.mkdir(parents=True, exist_ok=True)
    pool = WarmLanePool(worktree_base=base, size=size)
    harness.git_ops.warm_lane_pool = pool
    return pool


def _make_transcript(base: Path, session_id: str) -> Path:
    """Create ``<base>/claude-config-<sid>/projects/<slug>/<sid>.jsonl`` and
    return the ``claude-config-<sid>`` dir.

    Mirrors test_crash_recovery.py. Recovery globs
    ``<entry>/.task/claude-config-*`` at boot and ``transcript_exists`` re-globs
    ``<cfg>/projects/*/<sid>.jsonl`` at dispatch — placing the transcript under
    the lane's ``.task/`` satisfies BOTH the boot glob and the dispatch re-glob,
    so a composed recover→dispatch reaches ``(True, 'eligible')``.
    """
    cfg = base / f'claude-config-{session_id}'
    proj = cfg / 'projects' / 'some-slug'
    proj.mkdir(parents=True, exist_ok=True)
    (proj / f'{session_id}.jsonl').write_text('{"type": "summary"}\n')
    return cfg


def _sidecar(
    session_id: str, role: str, *, task_id: str, fresh: bool,
    sidecar_version: int, resume_count: int,
) -> dict:
    """Build a v1 or v2 ``agent_session.json`` payload.

    ``fresh=False`` back-dates ``started_at`` to 2× the freshness window so the
    γ guard rejects it as 'stale'. A v1 sidecar carries only the legacy keys
    (no ``task_id`` / ``resume_count`` / ``schema_version``) — the pre-deploy
    shape (B11).
    """
    if fresh:
        started_at = datetime.now(UTC).isoformat()
    else:
        window = SessionResumeConfig().freshness_window_secs
        started_at = (datetime.now(UTC) - timedelta(seconds=2 * window)).isoformat()
    payload: dict = {
        'session_id': session_id,
        'role': role,
        'started_at': started_at,
        'owner_pid': 4242,
    }
    if sidecar_version >= 2:
        payload['task_id'] = task_id
        payload['resume_count'] = resume_count
        payload['schema_version'] = 2
    return payload


def _setup_warm_lane_session(
    harness: Harness,
    task_id: str,
    session_id: str,
    *,
    role: str = 'implementer',
    steps_done: int = 3,
    steps_total: int = 5,
    fresh: bool = True,
    with_transcript: bool = True,
    with_plan: bool = True,
    with_sidecar: bool = True,
    sidecar_version: int = 2,
    resume_count: int = 0,
    lane: bool = True,
) -> Path:
    """Lay down an on-disk warm lane (or cold worktree) mid-invocation.

    Writes ``<dir>/.task/plan.json`` (stamped w/ the real task_id + partial
    progress, when ``with_plan``), a v1/v2 ``agent_session.json`` sidecar (when
    ``with_sidecar``), and — when ``with_transcript`` — the corroborating
    transcript under ``<dir>/.task/claude-config-<sid>/...``. Returns the
    lane/worktree dir.

    ``with_sidecar=False`` models the POST-COMPLETION on-disk state (B10): the
    plan survives but α's ``_invoke`` ``finally`` already cleared the sidecar,
    so recovery finds a plan to recover but NO session to adopt.

    For a warm lane (``lane=True``) a pool is attached and the dir is named
    ``_lane-0`` (≠ the real task_id, by pool-slot design); for a cold worktree
    (``lane=False``, B9) the dir is named after the task_id and no pool is
    attached.
    """
    base = harness.git_ops.worktree_base
    if lane:
        _attach_pool(harness)
        wt = base / '_lane-0'
    else:
        wt = base / task_id
    task_dir = wt / '.task'
    task_dir.mkdir(parents=True, exist_ok=True)
    if with_plan:
        (task_dir / 'plan.json').write_text(
            json.dumps(_make_plan(steps_done, steps_total, task_id))
        )
    if with_sidecar:
        (task_dir / 'agent_session.json').write_text(json.dumps(_sidecar(
            session_id, role, task_id=task_id, fresh=fresh,
            sidecar_version=sidecar_version, resume_count=resume_count,
        )))
    if with_transcript:
        _make_transcript(task_dir, session_id)
    return wt


def _session_resume_emits(harness: Harness) -> list[tuple]:
    """Return ``[(event_type, kwargs), ...]`` for every session_resume* emit.

    EventType members are referenced here at CALL time (never module scope) —
    matching the test_crash_recovery.py convention.
    """
    wanted = {
        EventType.session_resume,
        EventType.session_resume_fallback,
        EventType.session_resume_capped,
    }
    out: list[tuple] = []
    for call in harness.event_store.emit.call_args_list:  # type: ignore[attr-defined]
        if call.args and call.args[0] in wanted:
            out.append((call.args[0], call.kwargs))
    return out


async def _dispatch_capture(
    harness: Harness, task_id: str, *, task: dict | None = None,
) -> _DispatchCapture:
    """Drive REAL ``_run_slot`` with ``build_workflow`` patched; capture the
    ``resume_session_id`` / ``initial_plan`` kwargs it built + the
    session_resume* emits.

    The assignment's task dict deliberately carries no ``metadata`` key so the
    D4 substrate gate is skipped (its predicate is key-presence).
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = task if task is not None else {'title': f'task {task_id}'}
    sem = MagicMock()
    sem.release = MagicMock()
    with patch('orchestrator.harness.build_workflow') as MockWorkflow:
        mock_wf = AsyncMock()
        mock_wf.run.return_value = MagicMock(value='done')
        mock_wf.metrics = MagicMock(
            total_cost_usd=0.0, total_duration_ms=0, agent_invocations=0,
        )
        MockWorkflow.return_value = mock_wf
        await harness._run_slot(assignment, sem)
        kwargs = MockWorkflow.call_args.kwargs
    return _DispatchCapture(
        resume_session_id=kwargs['resume_session_id'],
        initial_plan=kwargs['initial_plan'],
        emits=_session_resume_emits(harness),
    )


def _seed_lane_record(
    lifecycle: LaneLifecycle, lane: Path, *, task_id: str, branch: str | None = None,
) -> None:
    """Bring *lane*'s durable record to ASSIGNED:*task_id* via the legal
    seed-up ladder (None → SEED → REGISTERED → ASSIGNED).

    Cloned from test_crash_recovery.py:711-725 — mirrors
    ``GitOps._note_assigned_via_route``'s climb. ``branch=None`` seeds a
    branchless record, which trivially satisfies the record-driven recovery
    path's ``rec.branch is None or …`` branch-match check regardless of what
    ``lane_branch_checkouts()`` reports, so a real ``git worktree list`` read
    is never load-bearing here.
    """
    lifecycle.transition(lane, DurableLaneState.SEED, seeded_from_sha='abc')
    lifecycle.transition(lane, DurableLaneState.REGISTERED, branch=branch)
    lifecycle.transition(
        lane, DurableLaneState.ASSIGNED, task_id=task_id, branch=branch,
    )


async def _make_real_git_lane(
    harness: Harness,
    task_id: str,
    session_id: str,
    *,
    role: str = 'implementer',
    steps_done: int = 3,
    steps_total: int = 5,
) -> tuple[Path, str]:
    """Build a REAL git-initialised warm lane mid-implementer; return
    ``(lane_dir, wip_sha)``.

    Unlike ``_setup_warm_lane_session`` (which fabricates a lane via a bare
    ``mkdir`` and so takes recovery's HEURISTIC plan path), this seeds a
    durable ASSIGNED lane record so recovery takes the RECORD-DRIVEN adopt
    path, AND the lane is a real git repo carrying a committed WIP so a
    post-recovery ``git rev-parse HEAD`` proves the exact WIP sha survives the
    recovery pass — not merely that ``cleanup_worktree`` was skipped.

    Layout: ``git init -b main`` + user config + an initial commit + a WIP
    commit. The WIP sha is captured BEFORE the on-disk ``.task`` artifacts are
    laid down, so ``HEAD`` is the WIP commit and the (untracked) ``.task`` dir
    is merely present on disk for recovery to read. The stamped
    ``.task/plan.json`` (partial progress — ≥1 done step, so recovery
    pre-loads it into ``_recovered_plans``), the v2 sidecar, and the
    corroborating transcript mirror ``_setup_warm_lane_session``. The durable
    record is seeded branchless so the record-driven branch-match check is
    trivially satisfied.
    """
    _attach_pool(harness)
    lane = harness.git_ops.worktree_base / '_lane-0'
    lane.mkdir(parents=True, exist_ok=True)
    # Real git repo: an initial commit, then a WIP commit whose sha survives.
    await _run(['git', 'init', '-b', 'main'], cwd=lane)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=lane)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=lane)
    (lane / 'work.py').write_text('def f():\n    return 0\n')
    await _run(['git', 'add', '-A'], cwd=lane)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=lane)
    (lane / 'work.py').write_text('def f():\n    return 1  # WIP mid-implementer\n')
    await _run(['git', 'add', '-A'], cwd=lane)
    await _run(['git', 'commit', '-m', 'WIP: mid-implementer'], cwd=lane)
    rc, wip_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane)
    assert rc == 0
    wip_sha = wip_sha.strip()
    # On-disk .task artifacts (untracked — HEAD stays at the WIP commit).
    task_dir = lane / '.task'
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / 'plan.json').write_text(
        json.dumps(_make_plan(steps_done, steps_total, task_id))
    )
    (task_dir / 'agent_session.json').write_text(json.dumps(_sidecar(
        session_id, role, task_id=task_id, fresh=True,
        sidecar_version=2, resume_count=0,
    )))
    _make_transcript(task_dir, session_id)
    # Durable ASSIGNED record → record-driven adopt path (branchless).
    _seed_lane_record(
        harness.git_ops._lane_lifecycle, lane, task_id=task_id, branch=None,
    )
    # Hermetic, deterministic branch-checkout read: the record is branchless so
    # the value is irrelevant to branch_ok, but pinning it avoids depending on
    # the host's real `git worktree list` (mirrors the record-driven recovery
    # tests in test_crash_recovery.py, which all stub this).
    harness.git_ops.lane_branch_checkouts = AsyncMock(return_value={})
    return lane, wip_sha


def _storm_queue() -> MagicMock:
    """A stand-in escalation queue for the B8 fallback-storm test.

    Mirrors test_crash_recovery.py's ``TestSessionResumeStorm._queue``:
    ``has_open_l1`` → False (no existing L1 to dedupe against), ``make_id`` →
    a fixed id, and a default-MagicMock ``submit`` whose ``call_count`` /
    ``call_args`` the storm assertion inspects.
    """
    q = MagicMock()
    q.has_open_l1 = MagicMock(return_value=False)
    q.make_id = MagicMock(return_value='sr-storm')
    return q


# ── B1: clean SIGTERM mid-implementer, warm lane ─────────────────────────────
@pytest.mark.asyncio
async def test_b1_warm_lane_adopts_then_injects_same_session(harness: Harness):
    """B1 — the flagship two-way seam: REAL recovery adopts the session/plan,
    then that SAME recovered state flows through REAL ``_run_slot`` as
    ``resume_session_id`` + ``initial_plan`` with one ``session_resume`` event.

    Chains β's output (``_recovered_sessions`` / ``_recovered_session_config_dirs``
    / ``_recovered_plans``) straight into γ's input, proving they are the same
    dict (``is`` identity) — not merely equal — across the boot→dispatch seam.
    """
    task_id, session_id = '42', 'uuid-b1-implementer'
    _setup_warm_lane_session(harness, task_id, session_id, role='implementer')
    harness.config.session_resume = SessionResumeConfig()

    await harness._recover_crashed_tasks()

    # ── ADOPT side (β) ──
    assert task_id in harness._recovered_sessions
    assert harness._recovered_sessions[task_id]['session_id'] == session_id
    assert task_id in harness._recovered_session_config_dirs
    assert task_id in harness._recovered_plans
    harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    adopted = harness._recovered_sessions[task_id]
    recovered_plan = harness._recovered_plans[task_id]

    # ── INJECT side (γ) ──
    cap = await _dispatch_capture(harness, task_id)
    assert cap.resume_session_id is adopted
    assert cap.resume_session_id['session_id'] == session_id
    assert cap.initial_plan is recovered_plan
    assert [et for et, _ in cap.emits] == [EventType.session_resume]


# ── Invocation-level driver (γ → invocation seam; α sidecar rewrite) ─────────
# Cloned from test_workflow_agent_session_preserve.py: a REAL-git TaskWorkflow
# over a real worktree + real on-disk TaskArtifacts, driving the REAL _invoke
# with only invoke_with_cap_retry patched. Used by B1 (journal proof), B10
# (completion clears the sidecar) and B11 (v1→v2 rewrite on next write).
_InvokeCapture = namedtuple(
    '_InvokeCapture', ['kwargs', 'workflow', 'cwd', 'sidecar_midflight'],
)


async def _init_resume_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'lib.py').write_text('def greet(name): return name\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


def _config(git_repo: Path, **overrides) -> OrchestratorConfig:
    """A hermetic OrchestratorConfig rooted at *git_repo* (transcript-archive
    disabled so _invoke's finally hook is a no-op)."""
    kwargs: dict = dict(
        project_root=git_repo,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
        transcript_archive={'enabled': False},
    )
    kwargs.update(overrides)
    return OrchestratorConfig(**kwargs)


def _resume_assignment(task_id: str) -> TaskAssignment:
    return TaskAssignment(
        task_id=task_id,
        task={
            'id': task_id, 'title': 'X', 'description': 'Y', 'status': 'pending',
            'metadata': {'files': ['lib']}, 'dependencies': [],
        },
        modules=['lib'],
    )


async def _make_resumed_workflow(
    config: OrchestratorConfig, git_ops: GitOps, task_assignment: TaskAssignment,
    resume_session_id,
) -> tuple[TaskWorkflow, Path]:
    """Build a probe TaskWorkflow with a REAL on-disk TaskArtifacts + config_dir.

    Mirrors test_workflow_agent_session_preserve.py:_make_workflow — direct
    _invoke skips run()'s setup, so _config_dir is set MANUALLY at the legacy
    ``.task`` root (the autouse meta-root fixture is deliberately un-imported).
    """
    wt_info = await git_ops.create_worktree(task_assignment.task_id)
    cwd = wt_info.path
    workflow = TaskWorkflow(
        assignment=task_assignment,
        config=config,
        git_ops=git_ops,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
        resume_session_id=resume_session_id,
    )
    artifacts = TaskArtifacts(cwd)
    artifacts.init(task_assignment.task_id, 'X', 'Y')
    workflow.artifacts = artifacts
    workflow._config_dir = TaskConfigDir(task_assignment.task_id, base_dir=cwd / '.task')
    return workflow, cwd


async def _drive_resumed_invoke(
    tmp_path: Path, recovered_session, role, caplog, *, task_id: str = '42',
) -> _InvokeCapture:
    """Feed *recovered_session* as ``resume_session_id`` into a REAL
    ``TaskWorkflow`` and drive ``_invoke(role, 'p', cwd)`` with
    ``invoke_with_cap_retry`` patched. Snapshots the sidecar MID-invocation
    (the finally clears it on completion) and captures the iwcr kwargs.
    """
    caplog.set_level(logging.INFO)
    sid = (recovered_session or {}).get('session_id', 'x')
    repo = tmp_path / f'resume-repo-{sid}'
    repo.mkdir()
    await _init_resume_repo(repo)
    git_ops = GitOps(
        GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
        repo,
    )
    with patch.dict(os.environ, {'ORCH_CONFIG_PATH': ''}):
        config = _config(repo)
    assignment = _resume_assignment(task_id)
    workflow, cwd = await _make_resumed_workflow(
        config, git_ops, assignment, recovered_session,
    )
    snapshot: dict = {}

    def _side_effect(**kwargs):
        assert workflow.artifacts is not None
        snapshot['sidecar'] = workflow.artifacts.read_agent_session()
        return AgentResult(success=True, output='')

    with patch(
        'orchestrator.workflow.invoke_with_cap_retry',
        new_callable=AsyncMock, side_effect=_side_effect,
    ) as mock_iwcr:
        await workflow._invoke(role, 'p', cwd)

    return _InvokeCapture(
        kwargs=mock_iwcr.call_args.kwargs,
        workflow=workflow,
        cwd=cwd,
        sidecar_midflight=snapshot.get('sidecar'),
    )


# ── B1: invocation-level journal proof (γ → invocation seam) ─────────────────
@pytest.mark.asyncio
async def test_b1_resumed_invocation_journals_prior_session(
    harness: Harness, tmp_path: Path, caplog,
):
    """B1 (γ→invocation): the adopted session dict, fed as ``resume_session_id``
    into a REAL ``TaskWorkflow``, drives a REAL ``_invoke`` that passes the SAME
    session id to ``invoke_with_cap_retry`` and journals ``resuming prior
    session <sid>``.

    Chains β's adopted dict straight into α's resumed invocation — the strongest
    "the resumed invocation actually used the session" proof ω can make without
    a live agent.
    """
    task_id, session_id = '77', 'uuid-b1-journal'
    _setup_warm_lane_session(harness, task_id, session_id, role='implementer')
    harness.config.session_resume = SessionResumeConfig()
    await harness._recover_crashed_tasks()
    recovered = harness._recovered_sessions[task_id]

    cap = await _drive_resumed_invoke(tmp_path, recovered, IMPLEMENTER, caplog)

    assert cap.kwargs['resume_session_id'] == session_id
    assert f'resuming prior session {session_id}' in caplog.text


# ── B1: WIP commit preserved across recovery + architect skipped ─────────────
@pytest.mark.asyncio
async def test_b1_recovery_preserves_wip_commit_and_skips_architect(
    harness: Harness,
):
    """B1 (WIP-preserve + architect-skip): a REAL git lane carrying a committed
    WIP is record-adopted at recovery — its WIP commit stays reachable and the
    lane is NEVER cleaned up — and the recovered plan flows through REAL
    ``_run_slot`` as ``initial_plan`` (the ratified complete-plan invariant
    skips the architect).

    Preserving committed WIP is the precondition for DISK_BACKSTOP_REUSE at
    re-acquire: recovery adopts the lane in place rather than resetting or
    removing it, so the next dispatch resumes atop the surviving commit. The
    lane is a REAL git repo so ``git rev-parse HEAD`` proves the exact WIP sha
    survives the recovery pass (not merely that ``cleanup_worktree`` was
    skipped).
    """
    task_id, session_id = '55', 'uuid-b1-wip'
    lane, wip_sha = await _make_real_git_lane(harness, task_id, session_id)
    harness.config.session_resume = SessionResumeConfig()

    await harness._recover_crashed_tasks()

    # ── ADOPT side (β): the real git lane is record-adopted, not cleaned ──
    assert task_id in harness._recovered_plans
    assert task_id in harness._recovered_sessions
    assert harness._recovered_sessions[task_id]['session_id'] == session_id
    harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    # ── WIP commit survives: still the exact sha, reachable via git ──
    rc, head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=lane)
    assert rc == 0
    assert head.strip() == wip_sha

    # ── INJECT side (γ): architect skipped — recovered plan is initial_plan ──
    recovered_plan = harness._recovered_plans[task_id]
    cap = await _dispatch_capture(harness, task_id)
    assert cap.initial_plan is recovered_plan


# ── B2: SIGKILL crash mid-implementer, warm lane — uniform-scope proof ───────
@pytest.mark.asyncio
async def test_b2_sigkill_warm_lane_identical_to_b1(harness: Harness):
    """B2 — a SIGKILL crash mid-implementer yields the SAME adopt+inject
    outcome as B1's clean SIGTERM.

    The mechanism keys on sidecar PRESENCE, not on how the process died: a
    SIGKILL (the ``_invoke`` ``finally`` that clears the sidecar NEVER ran) and
    a clean SIGTERM (α's ``agent_session_preserved`` re-writes it on the
    CancelledError path) both leave an IDENTICAL on-disk v2 sidecar, and the
    shutdown mode is simply not observable at recovery time — it is α's
    concern, covered by test_workflow_agent_session_preserve.py::
    TestPreserveOnCancellation. ω therefore proves B2 ≡ B1: session adopted,
    config_dir stashed, plan recovered, ``resume_session_id`` is the same
    session, and exactly one ``session_resume`` event.
    """
    task_id, session_id = '43', 'uuid-b2-sigkill'
    _setup_warm_lane_session(harness, task_id, session_id, role='implementer')
    harness.config.session_resume = SessionResumeConfig()

    await harness._recover_crashed_tasks()

    # ── ADOPT side (β) — identical to B1 ──
    assert task_id in harness._recovered_sessions
    assert harness._recovered_sessions[task_id]['session_id'] == session_id
    assert task_id in harness._recovered_session_config_dirs
    assert task_id in harness._recovered_plans
    harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]

    adopted = harness._recovered_sessions[task_id]
    recovered_plan = harness._recovered_plans[task_id]

    # ── INJECT side (γ) — identical to B1 ──
    cap = await _dispatch_capture(harness, task_id)
    assert cap.resume_session_id is adopted
    assert cap.resume_session_id['session_id'] == session_id
    assert cap.initial_plan is recovered_plan
    assert [et for et, _ in cap.emits] == [EventType.session_resume]


# ── B3: restart mid-architect, warm lane, no plan yet ────────────────────────
@pytest.mark.asyncio
async def test_b3_mid_architect_no_plan_adopts_session_lane_preserved(
    harness: Harness,
):
    """B3 — a restart MID-ARCHITECT (before any plan.json was written) still
    adopts the architect session, and at dispatch resumes it WITHOUT skipping
    the architect (no recovered plan → ``initial_plan is None``).

    Two-way: β adopts the architect session (keyed by the v2 sidecar's own
    ``task_id``, the only identity source on a no-plan lane) with NO recovered
    plan, and γ injects that SAME session as ``resume_session_id`` while leaving
    ``initial_plan`` None — so the resumed invocation re-enters the architect
    (the complete-plan-skips-architect invariant does NOT fire without a plan).

    LANDED-BEHAVIOR NOTE (mirrors ω design_decision #3 for B6): the PRD-prose
    framing "lane preserved / cleanup not called" describes the COLD-worktree
    no-plan branch (harness.py ~2899: ``_preserved_worktrees.add`` + no
    cleanup). A WARM LANE with no plan takes a DIFFERENT landed branch
    (harness.py ~2877-2897): it RELEASES the lane back to the pool
    (``cleanup_worktree`` IS called — a release, not a destroy) and never
    touches ``_preserved_worktrees``. Crucially, session adoption is keyed by
    the REAL task_id (the v2 sidecar's own ``task_id``), NOT the lane dir name,
    so the resume survives the lane going back to the pool — the adoption and
    the lane's pool membership are independent. ω asserts this landed seam
    behavior (GREEN-on-write against merged β), not the RED-doomed cold-path
    prose.
    """
    task_id, session_id = '49', 'uuid-b3-architect'
    _setup_warm_lane_session(
        harness, task_id, session_id, role='architect', with_plan=False,
    )
    harness.config.session_resume = SessionResumeConfig()

    await harness._recover_crashed_tasks()

    # ── ADOPT side (β): architect session adopted; NO plan; lane released ──
    assert task_id in harness._recovered_sessions
    assert harness._recovered_sessions[task_id]['role'] == 'architect'
    assert harness._recovered_sessions[task_id]['session_id'] == session_id
    assert task_id in harness._recovered_session_config_dirs
    assert task_id not in harness._recovered_plans  # mid-architect: no plan yet
    # Landed warm-lane no-plan disposition: released to the pool, NOT preserved.
    assert task_id not in harness._preserved_worktrees
    harness.git_ops.cleanup_worktree.assert_called_once()  # type: ignore[attr-defined]

    adopted = harness._recovered_sessions[task_id]

    # ── INJECT side (γ): same session resumed; architect NOT skipped ──
    cap = await _dispatch_capture(harness, task_id)
    assert cap.resume_session_id is adopted
    assert cap.resume_session_id['session_id'] == session_id
    assert cap.initial_plan is None  # no plan → architect re-runs, not skipped
    assert [et for et, _ in cap.emits] == [EventType.session_resume]


# ── B4: foreign task acquires the lane first → corroboration fails ───────────
@pytest.mark.asyncio
async def test_b4_foreign_acquire_falls_back_no_transcript(harness: Harness):
    """B4 — a foreign task reseeds the lane between crash and re-dispatch,
    destroying A's transcript: the session is STILL adopted at boot (plan +
    sidecar survive) but the dispatch-time transcript re-glob finds nothing, so
    the γ guard corroborates-and-rejects → fresh dispatch WITH the recovered
    plan + one ``session_resume_fallback`` carrying ``reason='no_transcript'``.

    Two-way: adoption genuinely happened (the recovered plan flows through as
    ``initial_plan``) AND the guard independently rejected the resume — proving
    β's output reached γ and γ's corroboration is load-bearing, not a rubber
    stamp. The wiped transcript is modelled with ``with_transcript=False`` so
    ``_adopt_recovered_session`` stashes no config-dir.
    """
    task_id, session_id = '44', 'uuid-b4-noxscript'
    _setup_warm_lane_session(
        harness, task_id, session_id, role='implementer', with_transcript=False,
    )
    harness.config.session_resume = SessionResumeConfig()

    await harness._recover_crashed_tasks()

    # ── ADOPT side (β): session + plan recovered; NO config-dir corroboration ──
    assert task_id in harness._recovered_sessions
    assert task_id in harness._recovered_plans
    assert task_id not in harness._recovered_session_config_dirs
    recovered_plan = harness._recovered_plans[task_id]

    # ── INJECT side (γ): corroboration fails → fallback, plan kept ──
    cap = await _dispatch_capture(harness, task_id)
    assert cap.resume_session_id is None
    assert cap.initial_plan is recovered_plan
    assert len(cap.emits) == 1
    et, kwargs = cap.emits[0]
    assert et == EventType.session_resume_fallback
    assert kwargs['data']['reason'] == 'no_transcript'


# ── B4': the lane was reseeded → EXPECTED fallback, no escalation ────────────
@pytest.mark.asyncio
async def test_b4b_reseeded_lane_is_expected_fallback_no_escalation(harness: Harness):
    """B4' — the reseeded-lane case is an EXPECTED fallback, not an escalation
    (task 3256).

    Unlike B4 (where the transcript never existed, so boot recovery stashed no
    config dir at all), here the transcript DOES corroborate at boot — the
    config dir is genuinely stashed — and the whole ``<lane>/.task/`` tree is
    then wiped between adoption and re-dispatch, exactly as warm-lane acquire's
    always-reseed-from-base invariant does (``git clean -xfd`` on RECYCLE,
    ``rmtree(lane/'.task')`` on RESET_IN_PLACE_REATTACH).

    Two-way: β genuinely stashed the config dir AND γ independently reclassified
    the vanished store as ``reason='reseeded'`` — emitting the measurement event
    but filing NO L1, even with ``fallback_storm_threshold=1`` where any genuine
    fallback would fire immediately. The recovered plan still flows through as
    ``initial_plan`` (I3 — the wipe costs the resume, never the plan).
    """
    task_id, session_id = '444', 'uuid-b4b-reseeded'
    lane = _setup_warm_lane_session(
        harness, task_id, session_id, role='implementer', with_transcript=True,
    )
    harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=1)
    harness._escalation_queue = _storm_queue()

    await harness._recover_crashed_tasks()

    # ── ADOPT side (β): the transcript corroborated, so a config dir IS stashed ──
    assert task_id in harness._recovered_sessions
    assert task_id in harness._recovered_session_config_dirs
    recovered_plan = harness._recovered_plans[task_id]

    # The next acquire re-seeds the lane from base, wiping .task/ wholesale —
    # the stashed config dir now points at a path that no longer exists.
    shutil.rmtree(lane / '.task')
    assert not Path(harness._recovered_session_config_dirs[task_id]).exists()

    # ── INJECT side (γ): expected fallback — event yes, escalation no ──
    cap = await _dispatch_capture(harness, task_id)
    assert cap.resume_session_id is None
    assert cap.initial_plan is recovered_plan
    assert len(cap.emits) == 1
    et, kwargs = cap.emits[0]
    assert et == EventType.session_resume_fallback
    assert kwargs['data']['reason'] == 'reseeded'
    assert harness._escalation_queue.submit.call_count == 0


# ── B5: stale sidecar beyond the freshness window ────────────────────────────
@pytest.mark.asyncio
async def test_b5_stale_sidecar_falls_back(harness: Harness):
    """B5 — a sidecar whose ``started_at`` is older than the freshness window
    is adopted at boot but rejected at dispatch as ``stale`` → fresh dispatch
    WITH the recovered plan + one ``session_resume_fallback`` (reason
    ``stale``).

    The transcript is present here (so ``no_transcript`` is NOT the
    disqualifier), isolating STALENESS as the sole rejection cause: the γ guard
    checks freshness BEFORE transcript corroboration, so an old-but-corroborated
    session still degrades to a fresh dispatch rather than resuming a
    long-dead agent.
    """
    task_id, session_id = '45', 'uuid-b5-stale'
    _setup_warm_lane_session(
        harness, task_id, session_id, role='implementer', fresh=False,
    )
    harness.config.session_resume = SessionResumeConfig()

    await harness._recover_crashed_tasks()

    # ── ADOPT side (β): session + plan + config-dir all recovered ──
    assert task_id in harness._recovered_sessions
    assert task_id in harness._recovered_plans
    assert task_id in harness._recovered_session_config_dirs
    recovered_plan = harness._recovered_plans[task_id]

    # ── INJECT side (γ): stale → fallback, plan kept, no --resume ──
    cap = await _dispatch_capture(harness, task_id)
    assert cap.resume_session_id is None
    assert cap.initial_plan is recovered_plan
    assert len(cap.emits) == 1
    et, kwargs = cap.emits[0]
    assert et == EventType.session_resume_fallback
    assert kwargs['data']['reason'] == 'stale'


# ── B6: kill switch (session_resume.enabled = false) ─────────────────────────
@pytest.mark.asyncio
async def test_b6_kill_switch_fresh_dispatch_no_event(harness: Harness):
    """B6 — with the kill switch off, an otherwise-eligible session is STILL
    adopted at boot but degrades SILENTLY at dispatch: ``resume_session_id`` is
    None, the recovered plan is kept, and NO session_resume* event of any kind
    is emitted.

    ω asserts the LANDED inject-time-only semantics (ω design_decision #3):
    ``_adopt_recovered_session`` has no ``enabled`` gate, so recovery populates
    ``_recovered_sessions`` regardless; only ``_session_resume_eligible``
    returns ``(False, 'disabled')`` and ``_run_slot`` degrades without an
    event or a storm-streak bump. This deliberately does NOT assert the PRD §8
    prose "no adoption at boot" — that would be a RED-doomed false premise
    against merged γ.
    """
    task_id, session_id = '46', 'uuid-b6-killswitch'
    _setup_warm_lane_session(harness, task_id, session_id, role='implementer')
    harness.config.session_resume = SessionResumeConfig(enabled=False)

    await harness._recover_crashed_tasks()

    # ── ADOPT side (β): the kill switch does NOT gate boot-time adoption ──
    assert task_id in harness._recovered_sessions
    assert task_id in harness._recovered_plans
    recovered_plan = harness._recovered_plans[task_id]

    # ── INJECT side (γ): disabled → silent fresh dispatch, plan kept, NO event ──
    cap = await _dispatch_capture(harness, task_id)
    assert cap.resume_session_id is None
    assert cap.initial_plan is recovered_plan
    assert cap.emits == []
    assert _session_resume_emits(harness) == []


# ── B7: resume cap (resume_count == max_resumes_per_task) ────────────────────
@pytest.mark.asyncio
async def test_b7_resume_cap_emits_capped(harness: Harness):
    """B7 — a session that has already been resumed ``max_resumes_per_task``
    times is adopted at boot but throttled at dispatch: ``resume_session_id``
    is None, the recovered plan is kept, and exactly one
    ``session_resume_capped`` event (a DISTINCT by-design-throttling signal,
    not a corroboration ``fallback``) is emitted.

    The sidecar is fresh with a present transcript so freshness and
    corroboration both PASS — the per-task cap is the sole disqualifier,
    isolating the ``capped`` arm (which, unlike ``stale``/``no_transcript``,
    does NOT feed the fallback-storm streak).
    """
    task_id, session_id = '47', 'uuid-b7-capped'
    _setup_warm_lane_session(
        harness, task_id, session_id, role='implementer', resume_count=3,
    )
    harness.config.session_resume = SessionResumeConfig(max_resumes_per_task=3)

    await harness._recover_crashed_tasks()

    # ── ADOPT side (β): session (resume_count=3) + plan recovered ──
    assert task_id in harness._recovered_sessions
    assert harness._recovered_sessions[task_id]['resume_count'] == 3
    assert task_id in harness._recovered_plans
    recovered_plan = harness._recovered_plans[task_id]

    # ── INJECT side (γ): capped → throttled, plan kept, capped event ──
    cap = await _dispatch_capture(harness, task_id)
    assert cap.resume_session_id is None
    assert cap.initial_plan is recovered_plan
    assert [et for et, _ in cap.emits] == [EventType.session_resume_capped]


# ── B8: resume-failure fallback storm → one L1, streak not per-event ─────────
@pytest.mark.asyncio
async def test_b8_fallback_storm_files_one_l1(harness: Harness):
    """B8 — ``fallback_storm_threshold`` CONSECUTIVE stale dispatches in one
    boot file exactly ONE level-1 escalation (not one per fallback): the streak
    is a per-boot RUN counter, and a run this long is the signature of a
    SYSTEMATIC corroboration break (clock skew / mass reseed), not isolated
    foreign-acquires.

    The FIRST stale dispatch flows through REAL ``_recover_crashed_tasks`` (so
    this stays a composition, not a hand-populated unit test); the remaining
    two seed the recovered dicts directly — the streak is a harness-level
    counter, so it accumulates identically either way. Reuses ``_storm_queue``.
    """
    harness.config.session_resume = SessionResumeConfig(fallback_storm_threshold=3)
    harness._escalation_queue = _storm_queue()

    # 1st stale dispatch — REAL recovery → adopt → stale fallback (streak=1).
    _setup_warm_lane_session(harness, 'st0', 'uuid-st0', fresh=False)
    await harness._recover_crashed_tasks()
    assert 'st0' in harness._recovered_sessions  # β genuinely adopted it
    cap0 = await _dispatch_capture(harness, 'st0')
    assert cap0.resume_session_id is None

    # 2nd + 3rd stale dispatches — leaner hand-populated recovered state.
    for i in (1, 2):
        tid = f'st{i}'
        harness._recovered_sessions[tid] = _sidecar(
            f'uuid-{tid}', 'implementer', task_id=tid, fresh=False,
            sidecar_version=2, resume_count=0,
        )
        harness._recovered_plans[tid] = _make_plan(3, 5, tid)
        cap = await _dispatch_capture(harness, tid)
        assert cap.resume_session_id is None

    # Exactly one L1 filed at the threshold — not one per fallback.
    assert harness._escalation_queue.submit.call_count == 1
    esc = harness._escalation_queue.submit.call_args.args[0]
    assert esc.level == 1
    assert 'resume' in esc.summary.lower()


# ── B9: cold worktree, plan present — β widens the cold path too ─────────────
@pytest.mark.asyncio
async def test_b9_cold_worktree_plan_present_adopts_and_injects(harness: Harness):
    """B9 — a COLD worktree (``<base>/<task_id>``, NOT a pool lane) with a plan
    + v2 sidecar + transcript is adopted-and-injected exactly like the warm
    lane: β widened both paths, so the cold worktree also resumes.

    The dir is named after the real task_id (cold path: ``recovery_id ==
    entry.name``) and no pool is attached, so ``is_lane`` is False and recovery
    takes the heuristic cold branch — yet the two-way outcome is identical to
    B1: session adopted, plan recovered, and at dispatch ``resume_session_id``
    is the same session with the recovered plan and one ``session_resume``
    event.
    """
    task_id, session_id = '48', 'uuid-b9-cold'
    _setup_warm_lane_session(
        harness, task_id, session_id, role='implementer', lane=False,
    )
    harness.config.session_resume = SessionResumeConfig()

    await harness._recover_crashed_tasks()

    # ── ADOPT side (β): cold worktree adopted + plan recovered ──
    assert task_id in harness._recovered_sessions
    assert harness._recovered_sessions[task_id]['session_id'] == session_id
    assert task_id in harness._recovered_plans
    adopted = harness._recovered_sessions[task_id]
    recovered_plan = harness._recovered_plans[task_id]

    # ── INJECT side (γ): same session + plan + one resume event ──
    cap = await _dispatch_capture(harness, task_id)
    assert cap.resume_session_id is adopted
    assert cap.resume_session_id['session_id'] == session_id
    assert cap.initial_plan is recovered_plan
    assert [et for et, _ in cap.emits] == [EventType.session_resume]


# ── B10: completion still clears → no adoption on next boot ──────────────────
@pytest.mark.asyncio
async def test_b10_completion_clears_sidecar_no_adoption_next_boot(
    harness: Harness, tmp_path: Path, caplog,
):
    """B10 — a COMPLETED invocation clears its sidecar (α), so the next boot
    finds a plan to recover but NO session to adopt: a clean finish never
    leaves a resumable session behind.

    Two-way (producer + consumer):
      PRODUCER (α): a REAL ``_invoke`` driven to COMPLETION (patched
        ``invoke_with_cap_retry`` returns success — no CancelledError) writes
        the sidecar in-flight, then its ``finally`` CLEARS it (``session_pre-
        served`` stays False) — so ``read_agent_session()`` is None afterwards.
      CONSUMER (β/γ): the post-completion on-disk state (``plan.json`` present,
        ``agent_session.json`` absent) is recovered as a plan-only task —
        ``_recovered_plans`` populated, ``_recovered_sessions`` NOT — and at
        dispatch degrades to a plain fresh dispatch WITH the recovered plan and
        NO session_resume* event of any kind (there is simply nothing to
        resume, distinct from an ineligible fallback).
    """
    # ── PRODUCER (α): a COMPLETING invoke clears the sidecar it wrote ──
    cap = await _drive_resumed_invoke(tmp_path, None, IMPLEMENTER, caplog)
    assert cap.sidecar_midflight is not None            # written in-flight
    assert cap.workflow.artifacts is not None
    assert cap.workflow.artifacts.read_agent_session() is None  # cleared on done

    # ── CONSUMER (β/γ): plan-but-no-sidecar → plan recovered, no session ──
    task_id, session_id = '50', 'uuid-b10-cleared'
    _setup_warm_lane_session(
        harness, task_id, session_id, role='implementer', with_sidecar=False,
    )
    harness.config.session_resume = SessionResumeConfig()

    await harness._recover_crashed_tasks()

    assert task_id in harness._recovered_plans
    assert task_id not in harness._recovered_sessions  # completion cleared it
    recovered_plan = harness._recovered_plans[task_id]

    cap2 = await _dispatch_capture(harness, task_id)
    assert cap2.resume_session_id is None
    assert cap2.initial_plan is recovered_plan
    assert cap2.emits == []
    assert _session_resume_emits(harness) == []


# ── B11: v1 sidecar compat — adopt via plan.json, rewrite v2 on next write ───
@pytest.mark.asyncio
async def test_b11_v1_sidecar_adopts_via_plan_and_rewrites_v2(
    harness: Harness, tmp_path: Path, caplog,
):
    """B11 — a PRE-DEPLOY v1 sidecar (no ``task_id`` / ``schema_version``) on a
    plan-bearing lane is still adopted (keyed via plan.json's task_id), and the
    resumed invocation REWRITES it as v2 while preserving the prior session_id.

    Two-way (consumer + producer):
      CONSUMER (β): the v1 sidecar carries no ``task_id`` of its own, so β keys
        the adoption off plan.json's task_id — the stored recovered dict is the
        RAW v1 payload (session_id preserved, still NO ``task_id`` key).
      PRODUCER (α): feeding that adopted v1 dict back in as ``resume_session_id``
        drives a REAL ``_invoke`` that resumes the SAME session_id (not a fresh
        uuid) and, on its next ``write_agent_session``, stamps the durable v2
        shape — ``schema_version == 2`` and ``task_id == str(task_id)`` — so a
        pre-deploy sidecar self-upgrades on first resume. The rewritten sidecar
        is captured MID-FLIGHT (α's ``finally`` clears it on completion).
    """
    task_id, session_id = '51', 'uuid-b11-v1compat'

    # ── CONSUMER (β): v1 sidecar adopted, keyed via plan.json's task_id ──
    _setup_warm_lane_session(
        harness, task_id, session_id, role='implementer', sidecar_version=1,
    )
    harness.config.session_resume = SessionResumeConfig()

    await harness._recover_crashed_tasks()

    assert task_id in harness._recovered_sessions
    adopted_v1 = harness._recovered_sessions[task_id]
    assert adopted_v1['session_id'] == session_id
    assert 'task_id' not in adopted_v1        # v1 payload carries no task_id
    assert 'schema_version' not in adopted_v1  # …nor a version discriminator

    # ── PRODUCER (α): resume the v1 dict → next write is v2, id preserved ──
    cap = await _drive_resumed_invoke(
        tmp_path, adopted_v1, IMPLEMENTER, caplog, task_id=task_id,
    )
    # Resumed against the PRIOR id (not a fresh uuid).
    assert cap.kwargs['resume_session_id'] == session_id
    assert f'resuming prior session {session_id}' in caplog.text
    # The next sidecar write self-upgrades v1 → v2, preserving the session id.
    rewritten = cap.sidecar_midflight
    assert rewritten is not None
    assert rewritten.schema_version == 2
    assert rewritten.task_id == str(task_id)
    assert rewritten.session_id == session_id
