"""W10-ι: B+H supervision composition integration gate (task 2244).

Stream W10 (harness-supervision), PRD ``plans/harness-supervision-prd.md``
§7 boundary sketch. This is a TEST-ONLY integration gate roping the
already-landed RestartPlan (``proc_supervision.py``), DeployState
(``shared``/``orchestrator`` ``deploy_state.py``), TaskGroundTruth
(``task_ground_truth.py``), LifecycleRegistry/BackgroundService
(``background_service.py``), and MergeProvenance/LandedOutbox
(``landed_outbox.py``) into ONE green composition suite exercising them
across their seams — the 2064/2105/1900/2059/2066 incident chain.

COMPOSES the REAL objects across their seams — real producer + real
consumer, not synthetic-input unit tests. Each cell's own unit behavior is
already covered in its mechanism's own suite (test_proc_supervision.py,
test_task_ground_truth.py, test_reconcile_stranded.py,
test_background_service.py); this module adds no new unit coverage and no
production source change. Follows the established
test_*_integration_gate.py convention (coalesce/config_reload/warm_lane/
milestone).

Crashes are simulated by injected fakes (FakeRunner, make_fake_inspector, a
metadata-state fault point) — never real process kills (PRD §7).

Composition cells (observable signal — all GREEN in CI):
  D1a     RestartPlan x DeployState x TaskGroundTruth — a real own-unit
          self-restart schedules deferred (SCHEDULED), a real None->RAN
          DeployState write lands, the deploy then 'crashes' (no VERIFIED
          write), and TaskGroundTruth's recovery_for sees the NAMED
          deploy_phase=ran and re-files an escalation — never phantom-done.
  D1b/R4  RestartPlan x DeployState — a real cross-unit blocking verify
          passes (DEPLOYED_AND_VERIFIED) and the real DeployState chain
          advances RAN->VERIFIED->DONE, every edge _LEGAL.
  R3      RestartPlan x EscalationQueue — the 2064 self-kill guard: an
          unset own_unit refuses a blocking restart and escalates, with
          zero runner/inspector calls.
  R2      RestartPlan (RP-3) — the 2105 exit-127 fix: a relative script +
          absolute cwd yields a systemd-run argv carrying
          --working-directory plus an absolute script token.
  S1      LifecycleRegistry x BackgroundService — a wedging service is
          abandoned bounded, mid-ladder, with the reverse-order shutdown
          ladder still completing (no hang).
  G1      TaskGroundTruth x MergeProvenance — a bound LandedOutbox row
          recovers a stranded task to done{merged,sha} JOURNAL-FIRST, with
          git archaeology never awaited (TG-1).

Sibling test scaffolding (FakeRunner, make_fake_inspector, read_escalations,
_reset_merge_provenance, _bind_landed_row, _make_ground_truth,
_fake_git_ops/_fake_scheduler, _RecordingService/wedging-service idiom) is
reused by LOCAL re-declaration below — there is no conftest-shared version
of these helpers to import.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue

from orchestrator.landed_outbox import LandedOutbox, LandedRow, MergeProvenance
from orchestrator.task_ground_truth import TaskGroundTruth

# ---------------------------------------------------------------------------
# Prerequisite-1: shared scaffolding (no behaviour assertions here) — reused
# by LOCAL re-declaration (per the design decision above) from
# test_proc_supervision.py, test_task_ground_truth.py / test_reconcile_stranded.py,
# and test_background_service.py.
# ---------------------------------------------------------------------------


class FakeRunner:
    """Async callable matching ``asyncio.create_subprocess_exec``'s signature.

    Records every ``(argv, kwargs)`` call in ``self.calls`` so tests can
    assert on exact positional argv and keyword args (cwd=, stdout=, ...).
    Returns a ``MagicMock`` proc whose ``communicate()`` is an
    ``AsyncMock(return_value=(stdout, None))`` and whose ``returncode`` is
    configurable (default 0). Copied verbatim from test_proc_supervision.py.
    """

    def __init__(self, returncode: int = 0, stdout: bytes = b'') -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.calls: list[tuple[tuple, dict]] = []
        # Every fake proc this runner has handed back, in call order — lets a
        # test assert on a spawned proc's post-return state without execute()
        # needing to hand the proc back to the caller.
        self.procs: list[MagicMock] = []

    async def __call__(self, *args: object, **kwargs: object):
        self.calls.append((args, kwargs))
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(self.stdout, None))
        proc.returncode = self.returncode
        self.procs.append(proc)
        return proc


def make_fake_inspector(state: dict):
    """Build an async inspector callable: ``(unit, *, timeout_secs, reap_grace_secs=5.0) -> dict``.

    Returns a shallow copy of *state* on every call regardless of args, and
    records each call's kwargs on the returned callable's ``.calls`` list.
    Copied verbatim from test_proc_supervision.py.
    """
    calls: list[dict] = []

    async def _inspector(unit: str, *, timeout_secs: float, reap_grace_secs: float = 5.0) -> dict:
        calls.append({
            'unit': unit,
            'timeout_secs': timeout_secs,
            'reap_grace_secs': reap_grace_secs,
        })
        return dict(state)

    _inspector.calls = calls  # type: ignore[attr-defined]
    return _inspector


@pytest.fixture
def tmp_queue_dir(tmp_path: Path) -> Path:
    """A ``tmp_path`` subdirectory usable as an ``EscalationQueue`` queue_dir.

    Not pre-created — ``EscalationQueue.__init__`` mkdir(parents=True,
    exist_ok=True)'s it, matching production. Copied verbatim from
    test_proc_supervision.py.
    """
    return tmp_path / 'escalations'


def read_escalations(
    queue_dir: str | Path,
    task_id: str,
    *,
    status: str | None = None,
    level: int | None = None,
    agent_role: str | None = None,
) -> list:
    """Construct ``EscalationQueue(queue_dir)`` and return ``get_by_task(task_id, ...)``.

    Product read path (PRD Sec 7): tests assert filed escalations through
    this helper rather than inspecting on-disk JSON directly. Copied
    verbatim from test_proc_supervision.py.
    """
    return EscalationQueue(Path(queue_dir)).get_by_task(
        task_id, status=status, level=level, agent_role=agent_role,
    )


@pytest.fixture(autouse=True)
def _reset_merge_provenance():
    """``MergeProvenance._outbox`` is a process-global — never leak a bound
    outbox into another test (mirrors test_task_ground_truth.py's /
    test_reconcile_stranded.py's identically-named fixture)."""
    MergeProvenance._outbox = None
    yield
    MergeProvenance._outbox = None


def _bind_landed_row(tmp_path: Path, *, task_id: str, advanced_sha: str) -> None:
    """Bind a real LandedOutbox (via MergeProvenance.bind) holding a row for
    *task_id* (mirrors test_task_ground_truth.py's identically-named helper)."""
    outbox = LandedOutbox(tmp_path / 'landed.json')
    outbox.record(LandedRow(
        task_id=task_id, branch_tip_sha='branchtip', advanced_sha=advanced_sha,
        landed_at=1.0,
    ))
    MergeProvenance.bind(outbox)


def _fake_git_ops(
    *,
    branch_prefix: str = 'task/',
    main_branch: str = 'main',
    is_ancestor: bool = False,
    branch_sha: str | None = None,
    marker_sha: str | None = None,
) -> MagicMock:
    """A minimal git_ops double exposing exactly the async surface
    TaskGroundTruth's branch-state archaeology consumes, plus the
    ``config.branch_prefix`` / ``config.main_branch`` attributes it derives
    the branch name and main-ref from. Copied verbatim from
    test_task_ground_truth.py."""
    git_ops = MagicMock()
    git_ops.config = MagicMock(branch_prefix=branch_prefix, main_branch=main_branch)
    git_ops.is_ancestor = AsyncMock(return_value=is_ancestor)
    git_ops.resolve_branch_sha = AsyncMock(return_value=branch_sha)
    git_ops.find_merge_marker = AsyncMock(return_value=marker_sha)
    return git_ops


def _fake_scheduler(
    *,
    is_actively_held: bool = False,
    task: dict | None = None,
) -> MagicMock:
    """A minimal scheduler double exposing exactly the TG-3 liveness surface
    (``is_actively_held`` — sync) and the task-fetch surface (``get_task`` —
    async) TaskGroundTruth's live_claimant resolution consumes. Copied
    verbatim from test_task_ground_truth.py."""
    scheduler = MagicMock()
    scheduler.is_actively_held = MagicMock(return_value=is_actively_held)
    scheduler.get_task = AsyncMock(return_value=task)
    return scheduler


def _make_ground_truth(
    *,
    git_ops: MagicMock | None = None,
    scheduler: MagicMock | None = None,
    escalation_queue: MagicMock | None = None,
    worktree_resolver=None,
    now_fn=None,
    heartbeat_ttl: timedelta | None = None,
) -> TaskGroundTruth:
    """Build a REAL TaskGroundTruth with only its collaborator surface faked.
    Copied verbatim from test_task_ground_truth.py."""
    kwargs = {}
    if now_fn is not None:
        kwargs['now_fn'] = now_fn
    if heartbeat_ttl is not None:
        kwargs['heartbeat_ttl'] = heartbeat_ttl
    return TaskGroundTruth(
        git_ops or _fake_git_ops(),
        scheduler or _fake_scheduler(),
        escalation_queue,
        worktree_resolver or (lambda tid: Path('/nonexistent-worktree') / tid),
        **kwargs,
    )


class _RecordingService:
    """Fake lifecycle service recording start()/stop() calls into a shared
    list, in call order.

    Duck-types the ``LifecycleService`` Protocol (name, async start(), async
    stop(), stop_timeout_secs) without depending on it. Copied verbatim from
    test_background_service.py.
    """

    def __init__(self, name: str, calls: list[str], stop_timeout_secs: float = 1.0) -> None:
        self.name = name
        self._calls = calls
        self.stop_timeout_secs = stop_timeout_secs

    async def start(self) -> None:
        self._calls.append(f'{self.name}:start')

    async def stop(self) -> None:
        self._calls.append(f'{self.name}:stop')


# ---------------------------------------------------------------------------
# step-1/2 — D1a: RestartPlan x DeployState x TaskGroundTruth. A real
# own-unit self-restart schedules deferred (SCHEDULED); a real None->RAN
# DeployState write lands; the deploy then 'crashes' (no VERIFIED write ever
# happens); TaskGroundTruth.recovery_for must see the NAMED deploy_phase=ran
# and re-file an escalation — never phantom-done.
# ---------------------------------------------------------------------------


async def _scheduled_self_restart_then_crash_at_ran(tmp_queue_dir: Path):
    """D1a driver: a real own-unit self-restart schedules deferred
    (SCHEDULED — never blocks/self-kills), then a real None->RAN DeployState
    write lands via enforce_transition + DeployState.to_metadata(). The
    fault: no further VERIFIED write ever happens — the deploy 'crashed'
    with metadata frozen at phase=ran. A real TaskGroundTruth then reads
    that frozen metadata back through recovery_for(). Returns
    (restart_outcome, truth_report, recovery_action)."""
    from orchestrator.deploy_state import DeployPhase, DeployState, enforce_transition
    from orchestrator.proc_supervision import EscalationSpec, RestartPlan

    tid = 'task-2244-d1a'
    plan = RestartPlan(
        script=Path('/proj/scripts/restart-orchestrator.sh'),
        args=[],
        cwd=Path('/proj'),
        target_unit='orch.service',
        own_unit='orch.service',
        on_failure_escalation=EscalationSpec(
            queue_dir=str(tmp_queue_dir),
            task_id=tid,
            summary='Self-restart fire-time failure',
        ),
        verify=None,
        transient_unit=f'orch-redeploy-restart-{tid}.service',
        on_active_secs=10,
    )
    restart_outcome = await plan.execute(runner=FakeRunner(returncode=0))

    metadata: dict = {}
    sink_calls: list[tuple[str, DeployPhase | None, DeployPhase]] = []

    def recording_sink(task_id: str, old: DeployPhase | None, new: DeployPhase) -> None:
        sink_calls.append((task_id, old, new))

    # old is None -> always legal (an initial write, not a transition).
    enforce_transition(None, DeployPhase.RAN, task_id=tid, escalation_sink=recording_sink)
    metadata.update(
        DeployState(phase=DeployPhase.RAN, ran_at='2026-07-15T00:00:00+00:00').to_metadata()
    )
    # Fault point: the deploy 'crashes' here — no VERIFIED write ever happens.

    resolver = _make_ground_truth(
        git_ops=_fake_git_ops(is_ancestor=False, branch_sha=None, marker_sha=None),
        scheduler=_fake_scheduler(is_actively_held=False, task={
            'status': 'in-progress',
            'claimant_run_id': None,
            'heartbeat_at': None,
            'metadata': metadata,
        }),
        escalation_queue=None,
    )
    truth_report, action = await resolver.recovery_for(tid)
    return restart_outcome, truth_report, action


@pytest.mark.asyncio
class TestD1SelfRestartDeployCrash:
    """D1a composition cell (the 2059/2066 crashed-mid-deploy case)."""

    async def test_ran_phase_crash_recovers_to_refile_not_phantom_done(
        self, tmp_queue_dir: Path,
    ) -> None:
        from shared.deploy_state import DeployPhase

        from orchestrator.proc_supervision import RestartDisposition
        from orchestrator.task_ground_truth import RecoveryAction

        restart_outcome, truth_report, action = await _scheduled_self_restart_then_crash_at_ran(
            tmp_queue_dir,
        )

        assert restart_outcome.disposition == RestartDisposition.SCHEDULED
        assert truth_report.deploy_phase == DeployPhase.RAN
        assert action == RecoveryAction.RE_FILE_ESCALATION
        assert action != RecoveryAction.MARK_DONE_WITH_PROVENANCE
