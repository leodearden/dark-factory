"""δ B+H boundary gate — orchestrator-consumer half.

This module is the orchestrator-consumer half of the δ end-to-end liveness boundary
gate (PRD plans/agent-liveness-telemetry-resume-prd.md §8, Appendix B).  It locks the
cli_invoke ↔ orchestrator/workflow seam contract from the orchestrator's perspective:

  - B1: progress result does NOT trip the zero-output circuit breaker.
  - B2: pre-turn-1 wedge: fast kill (watchdog) + breaker increments (consumer).
  - B3: resume across the wall carries the killed session id (headline integrated path).
  - B5: zero-output evidence carries the transcript (not proc_tree only).
  - B6: long synchronous tool after turn-1 NOT false-killed → progress → resume.
  - B7: unreadable/absent transcript → conservative degrade (no early kill → legacy → breaker).

See shared/tests/test_liveness_boundary_gate.py for the shared-contract half
(B1/B2/B4/B7 predicate and reader assertions).

The gate uses REAL on-disk JSONL transcripts driven through REAL count_transcript_turns /
_run_subprocess watchdog / predicates / orchestrator consumers — the genuine two-way
integration that per-component unit tests (which mock one side of the seam) do not provide.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from shared.cli_invoke import (
    _run_subprocess,
    count_transcript_turns,
    is_timed_out_with_progress,
    is_zero_output_timeout,
)
from shared.config_dir import TaskConfigDir

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import IMPLEMENTER
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import ZERO_OUTPUT_HANG_REASON, TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# Transcript helper (matches shared/tests/test_liveness_boundary_gate.py)
# ---------------------------------------------------------------------------


def _write_transcript(
    config_dir: Path,
    session_id: str,
    *,
    n_assistant: int,
    last_tool: str | None = None,
    cwd_slug: str = 'proj',
) -> Path:
    """Write a real nested-schema JSONL transcript and return its path.

    Creates ``<config_dir>/projects/<cwd_slug>/<session_id>.jsonl`` (mkdir parents).

    Schema: the REAL nested Claude CLI transcript format —

      assistant records have NO top-level 'content' key; the content-block list
      lives at rec['message']['content'].

    Layout:
    - One leading ``{"type":"user", ...}`` record so 0-assistant cases still have
      a non-empty file (count_transcript_turns skips type!="assistant").
    - ``n_assistant`` assistant records.  When ``last_tool`` is set, the **last**
      assistant record carries a ``{"type":"tool_use","name":last_tool}`` block in
      its ``message.content`` list; all others carry a plain text block.
    """
    transcript_dir = config_dir / 'projects' / cwd_slug
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f'{session_id}.jsonl'

    records: list[dict] = [
        {
            'type': 'user',
            'message': {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'Start task.'}],
            },
            'uuid': 'u-0',
            'parentUuid': None,
        },
    ]

    for i in range(n_assistant):
        is_last = (i == n_assistant - 1)
        if is_last and last_tool is not None:
            content = [{'type': 'tool_use', 'name': last_tool, 'id': f'tu-{i}', 'input': {}}]
        else:
            content = [{'type': 'text', 'text': f'Step {i}.'}]
        records.append({
            'type': 'assistant',
            'message': {'role': 'assistant', 'content': content},
            'uuid': f'a-{i}',
            'parentUuid': f'a-{i - 1}' if i > 0 else 'u-0',
        })

    transcript_path.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    return transcript_path


# ---------------------------------------------------------------------------
# Workflow harness (adapted from test_workflow_resume_on_progress.py:38-162)
# ---------------------------------------------------------------------------


def _make_workflow(
    *,
    tmp_path: Path,
    task_id: str = '1781',
    max_execute_iterations: int = 10,
    max_consecutive_zero_output_timeouts: int = 2,
) -> TaskWorkflow:
    """Minimal TaskWorkflow harness for liveness boundary gate tests.

    Adapted verbatim from test_workflow_resume_on_progress.py:_make_workflow.
    Includes the zero-output-hang config attributes required by _execute_iterations.
    """
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = []

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'
    config.merge_train_former_enabled = False
    config.merge_train_max_members = 3
    config.max_execute_iterations = max_execute_iterations
    config.max_consecutive_zero_output_timeouts = max_consecutive_zero_output_timeouts
    config.recycle_config_dir_on_zero_output = False
    config.judge_after_each_iteration = False
    config.inter_iteration_rebase = False

    scheduler = MagicMock()
    git_ops = MagicMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / '.task').mkdir(exist_ok=True)
    wf.artifacts = TaskArtifacts(worktree)
    wf.worktree = worktree
    wf.merge_queue = MagicMock()
    wf.plan = {
        'files': [],
        'prerequisites': [],
        'steps': [{'id': 'step-1', 'status': 'pending'}],
    }
    wf._base_commit = 'base_sha'
    wf._module_configs = []
    return wf


def _progress_timeout_agent_result() -> AgentResult:
    """AgentResult satisfying is_timed_out_with_progress() — transcript_turns=5 > 0."""
    return AgentResult(
        success=False,
        output='',
        timed_out=True,
        turns=0,
        cost_usd=0.0,
        duration_ms=1_200_000,
        transcript_turns=5,  # > 0 ⇒ is_timed_out_with_progress() == True
    )


def _zero_output_agent_result() -> AgentResult:
    """AgentResult satisfying is_zero_output_timeout() — transcript_turns=0."""
    return AgentResult(
        success=False,
        output='Agent produced no output',
        timed_out=True,
        turns=0,
        cost_usd=0.0,
        duration_ms=1_200_000,
        transcript_turns=0,  # == 0 ⇒ is_zero_output_timeout() == True
    )


def _stub_iteration_helpers(wf: TaskWorkflow, invoke_result: AgentResult) -> AsyncMock:
    """Stub per-iteration helpers so _execute_iterations runs without real I/O.

    Returns the mocked _invoke AsyncMock so callers can configure side effects.
    Adapted verbatim from test_workflow_resume_on_progress.py:_stub_iteration_helpers.
    """
    wf.artifacts.get_pending_steps = MagicMock(  # type: ignore[method-assign]
        return_value=[{'id': 'step-1'}],
    )
    wf.artifacts.validate_plan_owner = MagicMock(return_value=True)  # type: ignore[method-assign]
    wf.artifacts.read_plan = MagicMock(  # type: ignore[method-assign]
        return_value={
            'prerequisites': [],
            'steps': [{'id': 'step-1', 'status': 'pending'}],
        },
    )
    wf.artifacts.read_iteration_log = MagicMock(return_value=([], []))  # type: ignore[method-assign]
    wf.artifacts.append_iteration_log = MagicMock()  # type: ignore[method-assign]
    wf.artifacts.stamp_plan_provenance = MagicMock()  # type: ignore[method-assign]

    mock_invoke = AsyncMock(return_value=invoke_result)
    wf._invoke = mock_invoke  # type: ignore[method-assign]
    wf._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    wf._get_head_commit = AsyncMock(return_value='abc123')  # type: ignore[method-assign]
    wf.briefing.build_implementer_prompt = AsyncMock(return_value='test-prompt')

    return mock_invoke


def _make_hanging_proc():
    """Return (proc, call_count_ref) where communicate hangs on call 1, raises TimeoutError on call 2.

    Ported verbatim from shared/tests/test_cli_invoke.py::TestRunSubprocessWatchdog._make_hanging_proc.

    The fake communicate pattern:
      - First call: hangs via asyncio.Event().wait() until cancelled by the watchdog
        kill path (raises CancelledError on cancellation).
      - Second call (post-SIGTERM): raises TimeoutError immediately to trigger the
        SIGKILL branch so terminate_process_group is called.
    """
    call_count = [0]

    async def communicate_side_effect(input=None):  # noqa: A002
        call_count[0] += 1
        if call_count[0] == 1:
            # Hang until the watchdog cancels comm_task — CancelledError is raised here.
            await asyncio.Event().wait()
        # Second call (inside SIGTERM grace window): raise immediately → SIGKILL path.
        raise TimeoutError

    proc = MagicMock()
    proc.communicate = communicate_side_effect
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()
    proc.returncode = None
    proc.pid = 12345
    return proc, call_count


# ---------------------------------------------------------------------------
# Step-5: B1 — progress does NOT trip the zero-output circuit breaker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB1BreakerNotTrippedOnProgress:
    """B1: a progress-timeout result routes to the resume branch, not the wedge breaker.

    Cross-seam: the REAL is_timed_out_with_progress / is_zero_output_timeout predicates
    drive the REAL orchestrator breaker.
    """

    async def test_progress_does_not_increment_or_trip_breaker(self, tmp_path: Path) -> None:
        """Progress timeout sets pending resume and does NOT trip the zero-output breaker.

        - max_execute_iterations=1 → loop runs one iteration then caps out.
        - max_consecutive_zero_output_timeouts=2 → breaker would fire on 2nd wedge.
        - With a progress result on iteration 1, the elif branch sets _pending_resume_session_id
          and continues; the cap check fires → BLOCKED (via max_execute_iterations).
        - _zero_output_hang_info must remain falsy (breaker did NOT trip).
        - _pending_resume_session_id must == 'killed-sid' and _pending_resume_role == 'implementer'.
        """
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=1,
            max_consecutive_zero_output_timeouts=2,
        )
        _stub_iteration_helpers(wf, _progress_timeout_agent_result())
        # Pre-set the session-id stash (_invoke normally sets this; mocked _invoke skips it)
        wf._last_invoke_session_id = 'killed-sid'  # type: ignore[attr-defined]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED, (
            f'Expected BLOCKED (via max_execute_iterations cap); got {outcome!r}'
        )
        # Breaker must NOT have fired — zero_output_hang_info stays falsy
        assert not wf._zero_output_hang_info, (  # type: ignore[attr-defined]
            f'Expected _zero_output_hang_info falsy (breaker not tripped); '
            f'got {wf._zero_output_hang_info!r}'
        )
        # Progress routed to resume branch, not wedge
        assert wf._pending_resume_session_id == 'killed-sid', (  # type: ignore[attr-defined]
            f"Expected _pending_resume_session_id='killed-sid'; "
            f"got {wf._pending_resume_session_id!r}"
        )
        assert wf._pending_resume_role == IMPLEMENTER.name, (  # type: ignore[attr-defined]
            f"Expected _pending_resume_role={IMPLEMENTER.name!r}; "
            f"got {wf._pending_resume_role!r}"
        )


# ---------------------------------------------------------------------------
# Step-6: B2 — pre-turn-1 wedge: fast kill (watchdog) + breaker increments (consumer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB2PreTurn1Wedge:
    """B2: a zero-turn wedge is killed fast at the grace bound AND trips the breaker.

    Two sub-tests (two-way seam):
      (a) Watchdog fast kill: real _run_subprocess with 0-turn transcript → killed at grace.
      (b) Consumer: zero-output AgentResult → breaker trips → BLOCKED, no resume.
    """

    async def test_zero_turn_wedge_killed_at_grace_not_ceiling(self, tmp_path: Path) -> None:
        """(a) Watchdog: 0-turn real transcript → fast kill at startup_grace_secs, not ceiling.

        Uses REAL count_transcript_turns reading a real 0-assistant JSONL transcript.
        grace=0.05s, ceiling=5.0s, assert wall<2.0s (killed at grace, not ceiling).
        """
        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        # Write a REAL 0-assistant transcript so count_transcript_turns returns 0
        _write_transcript(cfg_dir, sid, n_assistant=0)
        assert count_transcript_turns(cfg_dir, sid) == 0, (
            'Sanity: REAL count_transcript_turns must read 0 for 0-assistant transcript'
        )

        proc, _ = _make_hanging_proc()
        terminate_pg_mock = AsyncMock()

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch('shared.cli_invoke.terminate_process_group', terminate_pg_mock),
        ):
            t0 = time.monotonic()
            result = await _run_subprocess(
                ['fake'],
                cwd=tmp_path,
                env={},
                model='opus',
                timeout_seconds=5.0,
                startup_grace_secs=0.05,
                session_id=sid,
                config_dir=cfg_dir,
            )
            wall = time.monotonic() - t0

        assert result.timed_out is True, 'Expected timed_out=True for startup wedge kill'
        assert result.transcript_turns == 0, (
            f'Expected transcript_turns=0; got {result.transcript_turns!r}'
        )
        terminate_pg_mock.assert_called_once()
        # Upper bound generous (2s) to avoid CI flakiness; key assertion: killed at grace.
        assert wall < 2.0, (
            f'Expected fast kill (<2s), got {wall:.3f}s — wedge not killed at grace bound'
        )

    async def test_zero_output_result_trips_breaker(self, tmp_path: Path) -> None:
        """(b) Consumer: zero-output AgentResult trips the circuit breaker → BLOCKED, no resume.

        max_consecutive_zero_output_timeouts=1 → single zero-output result trips the breaker.
        _zero_output_hang_info must be set with ZERO_OUTPUT_HANG_REASON.
        _pending_resume_session_id must be None (no resume for a wedge).
        """
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=5,
            max_consecutive_zero_output_timeouts=1,
        )
        _stub_iteration_helpers(wf, _zero_output_agent_result())

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED, (
            f'Expected BLOCKED (breaker tripped); got {outcome!r}'
        )
        assert wf._zero_output_hang_info, (  # type: ignore[attr-defined]
            'Expected _zero_output_hang_info set (breaker tripped)'
        )
        assert ZERO_OUTPUT_HANG_REASON in wf._zero_output_hang_info.get('reason', ''), (  # type: ignore[attr-defined]
            f'Expected ZERO_OUTPUT_HANG_REASON in reason; got {wf._zero_output_hang_info!r}'
        )
        assert wf._pending_resume_session_id is None, (  # type: ignore[attr-defined]
            f'Expected _pending_resume_session_id=None for wedge (no resume); '
            f'got {wf._pending_resume_session_id!r}'
        )


# ---------------------------------------------------------------------------
# Step-7: B3 — resume across the wall carries the killed session id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB3ResumeAcrossWall:
    """B3: headline integrated path — resume carries killed session id across the wall.

    Two phases:
      Phase 1 — _execute_iterations routes progress to resume fields.
      Phase 2 — _invoke consumes the pending fields, passes resume_session_id to
                 invoke_with_cap_retry, logs the resume, writes the sidecar.

    NEW vs γ unit tests: the journal-log assertion and the sidecar assertion across
    the full _execute_iterations → _invoke handoff.
    """

    async def test_progress_iteration_redispatches_with_same_session_id(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Full B3 path: progress timeout → pending fields set → _invoke resumes with same sid."""
        # ── Phase 1: _execute_iterations sets pending fields ──────────────────
        wf = _make_workflow(tmp_path=tmp_path, max_execute_iterations=1)
        _stub_iteration_helpers(wf, _progress_timeout_agent_result())
        wf._last_invoke_session_id = 'killed-sid'  # type: ignore[attr-defined]

        await wf._execute_iterations()

        assert wf._pending_resume_session_id == 'killed-sid', (  # type: ignore[attr-defined]
            f"Phase 1: expected _pending_resume_session_id='killed-sid'; "
            f"got {wf._pending_resume_session_id!r}"
        )
        assert wf._pending_resume_role == IMPLEMENTER.name, (  # type: ignore[attr-defined]
            f"Phase 1: expected _pending_resume_role={IMPLEMENTER.name!r}; "
            f"got {wf._pending_resume_role!r}"
        )

        # ── Phase 2: restore the real _invoke and call it directly ───────────
        # Phase 1 stubs wf._invoke as an instance attribute via _stub_iteration_helpers.
        # Delete the instance attribute to restore the class method for Phase 2.
        del wf._invoke  # type: ignore[attr-defined]

        success_result = AgentResult(
            success=True,
            output='Done!',
            timed_out=False,
            turns=3,
            cost_usd=0.25,
            duration_ms=5_000,
            session_id='killed-sid',
        )

        caplog.set_level(logging.INFO)

        # Spy on write_agent_session to capture sidecar args before _invoke's finally clears it.
        # The sidecar is written before the subprocess starts and cleared in the finally block,
        # so read_agent_session() after _invoke returns always returns None.
        sidecar_calls: list[dict] = []
        assert wf.artifacts is not None
        original_write = wf.artifacts.write_agent_session

        def spy_write(session_id: str, role: str, started_at: str) -> None:
            sidecar_calls.append({'session_id': session_id, 'role': role})
            original_write(session_id, role, started_at)

        wf.artifacts.write_agent_session = spy_write  # type: ignore[method-assign]

        with (
            patch('orchestrator.workflow.invoke_with_cap_retry', new_callable=AsyncMock) as mock_iwcr,
            patch.object(wf, '_build_agent_env', return_value=None),
        ):
            mock_iwcr.return_value = success_result
            await wf._invoke(IMPLEMENTER, 'test-prompt', tmp_path)

        # resume_session_id passed to invoke_with_cap_retry
        assert mock_iwcr.call_args is not None, 'Expected invoke_with_cap_retry to be called'
        iwcr_kwargs = mock_iwcr.call_args.kwargs
        assert iwcr_kwargs.get('resume_session_id') == 'killed-sid', (
            f"Phase 2: expected resume_session_id='killed-sid' in invoke_with_cap_retry call; "
            f"got {iwcr_kwargs.get('resume_session_id')!r}"
        )

        # Journal log contains 'resuming prior session killed-sid'
        assert 'resuming prior session killed-sid' in caplog.text, (
            f"Phase 2: expected 'resuming prior session killed-sid' in log; "
            f"caplog.text snippet: {caplog.text[-500:]!r}"
        )

        # Sidecar was written with role='implementer' and session 'killed-sid'
        # (Note: _invoke's finally block clears the sidecar after subprocess completes,
        # so we verify via the spy rather than read_agent_session() after _invoke returns.)
        assert sidecar_calls, 'Phase 2: expected write_agent_session to be called'
        sidecar = sidecar_calls[0]
        assert sidecar.get('session_id') == 'killed-sid', (
            f"Phase 2: expected sidecar session_id='killed-sid'; got {sidecar!r}"
        )
        assert sidecar.get('role') == IMPLEMENTER.name, (
            f"Phase 2: expected sidecar role={IMPLEMENTER.name!r}; got {sidecar!r}"
        )

        # Pending fields consumed (both None after _invoke)
        assert wf._pending_resume_session_id is None, (  # type: ignore[attr-defined]
            f'Phase 2: expected _pending_resume_session_id=None after _invoke; '
            f'got {wf._pending_resume_session_id!r}'
        )
        assert wf._pending_resume_role is None, (  # type: ignore[attr-defined]
            f'Phase 2: expected _pending_resume_role=None after _invoke; '
            f'got {wf._pending_resume_role!r}'
        )


# ---------------------------------------------------------------------------
# Step-8: B5 — zero-output evidence carries the transcript (not proc_tree only)
# ---------------------------------------------------------------------------


class TestB5EvidenceCarriesTranscript:
    """B5: _capture_zero_output_evidence enriches evidence with transcript_turns, last_tool,
    last_records AND retains proc_tree — demonstrating it is no longer proc-tree-only.

    PRD §2.3 misdiagnosis root cause: evidence was proc_tree-only, hiding the real
    transcript turn count.
    """

    def test_evidence_json_has_transcript_fields(self, tmp_path: Path) -> None:
        """Evidence JSON has transcript_turns, last_tool, last_records, AND proc_tree."""
        wf = _make_workflow(tmp_path=tmp_path)
        assert wf.artifacts is not None

        # Build a real TaskConfigDir so _capture_zero_output_evidence can locate the transcript
        config_dir = TaskConfigDir(wf.task_id, base_dir=wf.artifacts.root)
        wf._config_dir = config_dir  # type: ignore[attr-defined]

        sid = 'sid-b5-evidence'
        # Write a 2-assistant transcript with last_tool='Bash' in the REAL nested schema
        _write_transcript(config_dir.path, sid, n_assistant=2, last_tool='Bash')
        wf._last_invoke_session_id = sid  # type: ignore[attr-defined]

        result = AgentResult(
            success=False,
            output='',
            timed_out=True,
            turns=0,
            cost_usd=0.0,
            duration_ms=1_200_000,
            transcript_turns=0,
            proc_tree='<proc-snapshot>',
        )

        wf._capture_zero_output_evidence(result, iteration=3)  # type: ignore[attr-defined]

        evidence_path = wf.artifacts.root / 'zero_output_evidence-iter3.json'
        assert evidence_path.exists(), f'Expected evidence file at {evidence_path}'

        data = json.loads(evidence_path.read_text())

        # transcript_turns from result
        assert 'transcript_turns' in data, (
            f'Expected "transcript_turns" key in evidence; got {list(data.keys())!r}'
        )
        assert data['transcript_turns'] == 0, (
            f'Expected transcript_turns==0; got {data["transcript_turns"]!r}'
        )

        # last_tool extracted from the transcript's nested message.content blocks
        assert 'last_tool' in data, (
            f'Expected "last_tool" key in evidence; got {list(data.keys())!r}'
        )
        assert data['last_tool'] == 'Bash', (
            f'Expected last_tool=="Bash" (from nested message.content); got {data["last_tool"]!r}'
        )

        # last_records — last ≤5 records from the transcript, preserving nested shape
        assert 'last_records' in data, (
            f'Expected "last_records" key in evidence; got {list(data.keys())!r}'
        )
        assert isinstance(data['last_records'], list), (
            f'Expected last_records to be a list; got {type(data["last_records"])!r}'
        )
        assert len(data['last_records']) <= 5, (
            f'Expected at most 5 last_records; got {len(data["last_records"])}'
        )
        # Entries must preserve the nested schema (each has a 'message' key)
        for rec in data['last_records']:
            assert 'message' in rec, (
                f'Expected each last_records entry to have a "message" key (nested schema); '
                f'got {list(rec.keys())!r}'
            )

        # proc_tree retained — evidence is transcript-enriched AND proc_tree preserved
        assert 'proc_tree' in data, (
            f'Expected "proc_tree" key in evidence; got {list(data.keys())!r}'
        )
        assert data['proc_tree'] == '<proc-snapshot>', (
            f'Expected proc_tree=="<proc-snapshot>"; got {data["proc_tree"]!r}'
        )


# ---------------------------------------------------------------------------
# Step-9: B6 — long sync tool after turn-1 NOT false-killed → progress → resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB6LongSyncToolSurvives:
    """B6 headline integrated path: watchdog survival → progress classification → orchestrator resume.

    A process that has emitted turn-1 must NOT be false-killed at the startup_grace bound.
    The working regime latches once seen_turn is True; the ceiling (timeout_seconds) fires.
    This simulates a long-running synchronous tool (Bash) still in flight after turn-1.
    """

    async def test_seen_turn_survives_grace_then_classifies_progress(self, tmp_path: Path) -> None:
        """B6: 1-turn real transcript → liveness latched → no early kill → progress → resume."""
        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()

        # Write a REAL 1-assistant transcript (turn-1 emitted, then transcript silent)
        _write_transcript(cfg_dir, sid, n_assistant=1, last_tool='Bash')
        assert count_transcript_turns(cfg_dir, sid) == 1, (
            'Sanity: REAL count_transcript_turns must read 1 for 1-assistant transcript'
        )

        proc, _ = _make_hanging_proc()
        terminate_pg_mock = AsyncMock()

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch('shared.cli_invoke.terminate_process_group', terminate_pg_mock),
        ):
            t0 = time.monotonic()
            result = await _run_subprocess(
                ['fake'],
                cwd=tmp_path,
                env={},
                model='opus',
                timeout_seconds=0.3,
                startup_grace_secs=0.05,
                session_id=sid,
                config_dir=cfg_dir,
            )
            wall = time.monotonic() - t0

        # Must NOT have been killed at the 0.05s grace — must reach the 0.3s ceiling
        assert wall >= 0.2, (
            f'Expected kill at ceiling (~0.3s), but killed early at {wall:.3f}s '
            f'— seen turn should prevent grace kill (working regime)'
        )
        assert result.transcript_turns == 1, (
            f'Expected transcript_turns=1 after ceiling kill; got {result.transcript_turns!r}'
        )

        # Classify as progress
        ar = AgentResult(
            success=False,
            output='',
            timed_out=True,
            turns=0,
            cost_usd=0.0,
            duration_ms=int(wall * 1000),
            transcript_turns=result.transcript_turns,
        )
        assert is_timed_out_with_progress(ar) is True, (
            'Expected is_timed_out_with_progress=True for transcript_turns=1'
        )
        assert is_zero_output_timeout(ar) is False, (
            'Expected is_zero_output_timeout=False for transcript_turns=1'
        )

        # Orchestrator routes to resume, not breaker
        wf = _make_workflow(tmp_path=tmp_path, max_execute_iterations=1)
        _stub_iteration_helpers(wf, ar)
        wf._last_invoke_session_id = 'killed-sid'  # type: ignore[attr-defined]

        await wf._execute_iterations()

        assert wf._pending_resume_session_id == 'killed-sid', (  # type: ignore[attr-defined]
            f"B6: expected _pending_resume_session_id='killed-sid' (long-tool run resumes); "
            f"got {wf._pending_resume_session_id!r}"
        )
        # Breaker did NOT trip
        assert not wf._zero_output_hang_info, (  # type: ignore[attr-defined]
            f'B6: expected _zero_output_hang_info falsy (no wedge); got {wf._zero_output_hang_info!r}'
        )


# ---------------------------------------------------------------------------
# Step-10: B7 — unreadable/absent transcript → conservative degrade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestB7UnreadableTranscriptConservative:
    """B7: absent transcript → no early kill → legacy fallback → breaker.

    When count_transcript_turns returns None (transcript file absent), the watchdog
    cannot prove a wedge and must NOT early-kill at startup_grace.  The result is
    classified via the legacy heuristic (turns==0 and cost==0.0), then the consumer
    breaker routes it to BLOCKED (not resume).

    Conservative degrade: no new behavior, no false progress.
    """

    async def test_absent_transcript_no_early_kill_then_legacy_wedge(self, tmp_path: Path) -> None:
        """B7: no transcript file → count_transcript_turns returns None → ceiling kill → BLOCKED."""
        sid = str(uuid.uuid4())
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()
        # Do NOT write any transcript — cfg_dir/projects/<sid>.jsonl does not exist

        # Verify the REAL reader returns None for an absent file
        assert count_transcript_turns(cfg_dir, sid) is None, (
            'Sanity: REAL count_transcript_turns must return None for absent transcript'
        )

        proc, _ = _make_hanging_proc()
        terminate_pg_mock = AsyncMock()

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch('shared.cli_invoke.asyncio.create_subprocess_exec', side_effect=fake_exec),
            patch('shared.cli_invoke.terminate_process_group', terminate_pg_mock),
        ):
            t0 = time.monotonic()
            result = await _run_subprocess(
                ['fake'],
                cwd=tmp_path,
                env={},
                model='opus',
                timeout_seconds=0.3,
                startup_grace_secs=0.05,
                session_id=sid,
                config_dir=cfg_dir,
            )
            wall = time.monotonic() - t0

        # β did NOT early-kill (cannot prove wedge on None); must reach the ceiling
        assert wall >= 0.2, (
            f'Expected kill at ceiling (~0.3s); killed early at {wall:.3f}s '
            f'— None transcript should degrade to ceiling, not trigger fast kill'
        )
        assert result.transcript_turns is None, (
            f'Expected transcript_turns=None for absent transcript; got {result.transcript_turns!r}'
        )

        # Legacy fallback: turns==0 and cost_usd==0.0 → is_zero_output_timeout True
        ar = AgentResult(
            success=False,
            output='',
            timed_out=True,
            turns=0,
            cost_usd=0.0,
            duration_ms=int(wall * 1000),
            transcript_turns=None,
        )
        assert is_zero_output_timeout(ar) is True, (
            'Expected is_zero_output_timeout=True via legacy fallback (turns=0, cost=0.0)'
        )

        # Consumer: breaker trips → BLOCKED, no resume
        wf = _make_workflow(
            tmp_path=tmp_path,
            max_execute_iterations=5,
            max_consecutive_zero_output_timeouts=1,
        )
        _stub_iteration_helpers(wf, ar)

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED, (
            f'B7: expected BLOCKED (conservative degrade via legacy wedge + breaker); got {outcome!r}'
        )
        assert wf._zero_output_hang_info, (  # type: ignore[attr-defined]
            'B7: expected _zero_output_hang_info set (breaker tripped on legacy wedge)'
        )
