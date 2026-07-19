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
from collections import namedtuple
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from shared.config_dir import TaskConfigDir

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import ARCHITECT, IMPLEMENTER, SIMPLE_TASK
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
    sidecar_version: int = 2,
    resume_count: int = 0,
    lane: bool = True,
) -> Path:
    """Lay down an on-disk warm lane (or cold worktree) mid-invocation.

    Writes ``<dir>/.task/plan.json`` (stamped w/ the real task_id + partial
    progress, when ``with_plan``), a v1/v2 ``agent_session.json`` sidecar, and
    — when ``with_transcript`` — the corroborating transcript under
    ``<dir>/.task/claude-config-<sid>/...``. Returns the lane/worktree dir.

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
