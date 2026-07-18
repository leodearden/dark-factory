"""Base class for reconciliation pipeline stages — CLI-native MCP execution."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from shared.cli_invoke import read_transcript_records
from shared.config_dir import TaskConfigDir
from shared.safe_io import load_json_or_warn

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    ReconciliationEvent,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.cli_stage_runner import (
    STAGE_REPORT_SCHEMA,
    recon_config_base_dir,
    run_stage_via_cli,
)
from fused_memory.utils.validation import require_project_id, require_run_id

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
    from fused_memory.models.scope import ProjectScope
    from fused_memory.reconciliation.journal import ReconciliationJournal
    from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class BaseStage:
    """Base class for reconciliation pipeline stages.

    Each stage runs as a Claude CLI subprocess with access to fused-memory
    MCP tools.  Per-stage permissions are enforced via ``--disallowedTools``.
    """

    def __init__(
        self,
        stage_id: StageId,
        memory_service: MemoryService,
        taskmaster: TaskBackendProtocol | None,
        journal: ReconciliationJournal,
        config: ReconciliationConfig,
        usage_gate=None,
        *,
        scope: ProjectScope,
        known_projects: dict[str, str] | None = None,
        recon_report_port: int = 8003,
        recon_report_state=None,
    ):
        self.stage_id = stage_id
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.config = config
        self.scope = scope
        # Cross-project routing: dict[project_id, project_root] of every
        # project the harness knows about (including this one).  Stage 2
        # surfaces this to the LLM so it can re-route a finding to the
        # correct project_root when the scope demands.  Empty until the
        # harness sets it per cycle.
        self.known_projects: dict[str, str] = known_projects or {}
        self._usage_gate = usage_gate
        self._escalation_url: str | None = None
        self._escalation_queue: EscalationQueue | None = None
        # PRD γ: recon_report channel threading
        self._recon_report_port: int = recon_report_port
        self._recon_report_state = recon_report_state

    @property
    def project_id(self) -> str:
        """Read-only view over ``self.scope.project_id`` (task 2146 β).

        ``project_id``/``project_root`` are derived from ``self.scope`` — the
        single source of truth threaded in at construction — so they cannot
        drift apart via independent post-construction assignment.
        """
        return self.scope.project_id

    @property
    def project_root(self) -> str:
        """Read-only view over ``self.scope.project_root`` (task 2146 β)."""
        return self.scope.project_root

    def get_disallowed_tools(self) -> list[str]:
        """Override in subclass — return MCP tool names this stage may NOT use."""
        raise NotImplementedError

    def get_system_prompt(self) -> str:
        """Override in subclass."""
        raise NotImplementedError

    def get_report_schema(self) -> dict:
        """Return the JSON schema for this stage's output report.

        Override in subclass to use a stage-specific schema.
        """
        return STAGE_REPORT_SCHEMA

    async def assemble_payload(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
    ) -> str:
        """Override in subclass — build structured initial context."""
        raise NotImplementedError

    async def run(
        self,
        events: list[ReconciliationEvent],
        watermark: Watermark,
        prior_reports: list[StageReport],
        run_id: str,
        model: str | None = None,
        resume_session_id: str | None = None,
    ) -> StageReport:
        """Execute this stage via Claude CLI with MCP tools.

        ``resume_session_id`` (task 2717 σ) is set only by the startup
        adopt-and-resume pass, to ``--resume`` an in-flight session that a
        restart interrupted mid-stage. When set AND its transcript is still on
        disk in this run's per-run config dir (record_run_session's snapshot is
        best-effort — see its docstring warning), the stage subprocess resumes
        that session and is handed a purpose-written recovery prompt; otherwise
        --resume is dropped and the stage runs fresh. Defaults to None — every
        non-resume caller keeps today's fresh-launch behaviour.
        """
        require_project_id(self.project_id)
        require_run_id(run_id)

        # Validate watermark.project_id consistency (skip if watermark has no project_id)
        # Defensive strip: model_construct() bypasses the Pydantic field_validator that
        # normally normalizes this.  The `or ''` coerces None (and PydanticUndefined,
        # also falsy) to empty string so .strip() is always called on a str.
        wm_pid = (watermark.project_id or '').strip()
        if wm_pid:
            if wm_pid != self.project_id:
                raise ValueError(
                    f'project_id mismatch: stage has {self.project_id!r} but '
                    f'watermark has {wm_pid!r}'
                )
        else:
            logger.debug(
                'Watermark has no project_id — skipping mismatch check (stage project_id=%r)',
                self.project_id,
            )

        payload = await self.assemble_payload(events, watermark, prior_reports)

        # Inject reconciliation context so CLI agents include causation_id in writes
        recon_context = (
            f'\n\n## Reconciliation Context\n'
            f'run_id: {run_id}\n'
            f'stage: {self.stage_id.value}\n'
            f'agent_id: recon-stage-{self.stage_id.value}\n'
            f'- `project_id`: "{self.project_id}"\n\n'
            f'**IMPORTANT**: For every fused-memory write call, include:\n'
            f'- `agent_id`: "recon-stage-{self.stage_id.value}"\n'
            f'- In `metadata`: include `"_causation_id": "{run_id}"`\n'
        )
        payload = payload + recon_context

        disallowed = self.get_disallowed_tools()
        mcp_config = self._build_mcp_config()

        # PRD γ: signal the recon_report state machine that this stage is starting.
        # The CLI agent will call recon_report.add_finding / cite_* / complete via MCP;
        # the harness then reads back the assembled report after the CLI returns.
        #
        # _active_rrs is a run-local alias for self._recon_report_state so that a
        # start_report failure can degrade to None for this run only, without
        # mutating the shared state object (which must survive across runs).
        _active_rrs = self._recon_report_state
        start_report_failed = False
        if _active_rrs is not None:
            try:
                _active_rrs.start_report(
                    run_id=run_id,
                    stage=self.stage_id.value,
                    project_id=self.project_id,
                )
            except Exception as exc:  # noqa: BLE001
                # start_report is on the critical path — a transient error
                # (e.g. duplicate run_id from a crashed prior run, TTL-reaper race)
                # must not abort the whole reconciliation cycle.  Short-circuit:
                # the stage prompt has been switched to recon_report mode and
                # tells the agent NOT to produce a structured JSON response, so
                # invoking the CLI now would either time out or fail schema
                # validation, costing a full agent budget per stage on every
                # recon_report outage.  Return an empty StageReport instead.
                logger.warning(
                    'Stage %s: start_report raised %r for run_id=%s; '
                    'short-circuiting to empty StageReport (no LLM invocation)',
                    self.stage_id.value, exc, run_id,
                )
                _active_rrs = None
                start_report_failed = True

        started = datetime.now(UTC)

        if start_report_failed:
            completed = datetime.now(UTC)
            return StageReport(
                stage=self.stage_id,
                started_at=started,
                completed_at=completed,
                items_flagged=[],
                stats={},
                llm_calls=0,
                tokens_used=0,
            )

        # When recon_report_state is active, drop the output_schema requirement —
        # the assembled in-process state is the source of truth, not the LLM's JSON.
        effective_output_schema = (
            None if _active_rrs is not None
            else self.get_report_schema()
        )

        # Session-capture substrate (task 2744): mint a CLI session and a per-run
        # isolated config dir, persist the (session_id, stage) snapshot on the run
        # row BEFORE launching the stage subprocess (mint-before-spawn, mirroring
        # the orchestrator), then clear the snapshot once the subprocess exits so
        # the row carries a session_id only while a stage is actually in flight.
        # The config dir is derived deterministically from (data_dir, run_id) so
        # all three stages of a run share it and the harness GCs the same path;
        # attempt auto-increments per launch as durable history for task σ (resume).
        session_id = str(uuid4())
        config_dir = TaskConfigDir(
            task_id=run_id,
            base_dir=recon_config_base_dir(self.journal.data_dir),
        )
        await self.journal.record_run_session(
            run_id, session_id=session_id, stage_cursor=self.stage_id.value
        )

        # Task σ resume seam: when the startup adopt-and-resume pass supplies the
        # in-flight session id, --resume it IFF its transcript is still on disk in
        # this run's per-run config dir (config_dir is keyed on run_id, so it is
        # the SAME dir across the interrupt). record_run_session's snapshot is
        # best-effort — it can go stale after an internal cap retry — so validate
        # the transcript before trusting it to --resume. On a valid transcript,
        # deliver a purpose-written recovery prompt (resume_delivers_prompt=True)
        # that ALSO embeds the full stage payload, so a resume failure falls back
        # to a fully-contexted fresh invocation via invoke_with_cap_retry's
        # original_prompt. On a missing/empty transcript, drop --resume and run
        # the stage fresh.
        effective_payload = payload
        effective_resume_session_id: str | None = None
        resume_delivers_prompt = False
        if resume_session_id:
            transcript = read_transcript_records(config_dir.path, resume_session_id)
            if transcript:
                effective_resume_session_id = resume_session_id
                resume_delivers_prompt = True
                effective_payload = self._build_recovery_prompt(run_id, payload)
            else:
                logger.warning(
                    'Stage %s run_id=%s: resume_session_id=%s has no transcript in '
                    '%s — dropping --resume, running the stage fresh',
                    self.stage_id.value, run_id, resume_session_id, config_dir.path,
                )

        # Clear the session snapshot on a normal stage exit (and on non-cancel
        # errors) but PRESERVE it on asyncio.CancelledError: a restart cancelling
        # the in-flight stage must leave (session_id, stage_cursor) on the run row
        # so the startup adopt-and-resume pass (task σ) can --resume this run.
        # A non-cancel exception still clears (the run is failing, not resuming).
        cancelled = False
        try:
            stage_result = await run_stage_via_cli(
                system_prompt=self.get_system_prompt(),
                payload=effective_payload,
                disallowed_tools=disallowed,
                config=self.config,
                mcp_config=mcp_config,
                usage_gate=self._usage_gate,
                model=model,
                cwd=Path(self.config.explore_codebase_root),
                output_schema=effective_output_schema,
                session_id=session_id,
                config_dir=config_dir,
                resume_session_id=effective_resume_session_id,
                resume_delivers_prompt=resume_delivers_prompt,
            )
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if not cancelled:
                await self.journal.clear_run_session(run_id)

        completed = datetime.now(UTC)

        # PRD γ: read assembled report from in-process state (overrides stage_result.report).
        # Falls back to stage_result.report when recon_report_state is not configured
        # (back-compat path for non-recon stages) or when start_report degraded to None.
        if _active_rrs is not None:
            assembled = _active_rrs.get_assembled_report(
                run_id, self.stage_id.value
            )
            if assembled is not None:
                stage_result.report = assembled
            else:
                # Agent crashed / timed out before calling recon_report.complete.
                # Substitute empty assembled dict — matches today's malformed-JSON outcome.
                logger.warning(
                    'Stage %s: get_assembled_report returned None for run_id=%s '
                    '(agent may have crashed before calling recon_report.complete)',
                    self.stage_id.value, run_id,
                )
                stage_result.report = {'summary': '', 'stats': {}, 'flagged_items': []}

        # Schema consistency (reviewer finding schema_consistency): warn when a
        # flagged item lacks both dedup keys — compute_flag_signature will return
        # None for these findings, silently bypassing cross-run recurrence tracking.
        for _item in stage_result.report.get('flagged_items', []):
            if not _item.get('task_id') and not _item.get('cited_tasks'):
                logger.warning(
                    'Stage %s run_id=%s: finding lacks task_id and cited_tasks — '
                    'dedup signature cannot be computed (finding: %.80s)',
                    self.stage_id.value, run_id,
                    _item.get('description', '<no description>'),
                )

        report_data = stage_result.report
        stage_report = StageReport(
            stage=self.stage_id,
            started_at=started,
            completed_at=completed,
            items_flagged=report_data.get('flagged_items', []),
            stats=report_data.get('stats', {}),
            llm_calls=stage_result.llm_calls,
            tokens_used=stage_result.tokens_used,
        )

        duration = (completed - started).total_seconds()
        logger.info(
            'reconciliation.stage_completed',
            extra={
                'run_id': run_id,
                'stage': self.stage_id.value,
                'duration_seconds': round(duration, 1),
                'model': stage_result.model,
                'cost_usd': stage_result.cost_usd,
                'llm_calls': stage_result.llm_calls,
                'success': stage_result.success,
                'error': stage_result.error or None,
            },
        )

        if not stage_result.success:
            logger.error(
                f'Stage {self.stage_id.value} failed: {stage_result.error}'
            )

        return stage_report

    def _build_recovery_prompt(self, run_id: str, stage_payload: str) -> str:
        """Build the continuation prompt delivered to a ``--resume``d stage (task σ).

        Unlike the generic CRASH_RECOVERY placeholder ('continue where you left
        off'), this is purpose-written for reconciliation: it frames the interrupt
        as a fused-memory restart, notes the MCP reset, and — critically — points
        the agent at the write journal (``causation_id={run_id}``) so it can see
        which writes it already applied this run before repeating them. The full
        original stage payload is appended verbatim so the same string doubles as
        the fresh-fallback prompt (invoke_with_cap_retry keeps ``prompt`` as
        ``original_prompt`` when a resume fails and it restarts fresh).
        """
        return (
            '## Reconciliation run RESUMED after a restart\n\n'
            f'This reconciliation stage (run_id={run_id}, stage={self.stage_id.value}) '
            'was INTERRUPTED by a fused-memory restart mid-run and has now been '
            'resumed via --resume. Your MCP tool connections were reset by the '
            'restart, so re-establish any state you need before continuing.\n\n'
            '**Some of the writes you made before the interrupt may ALREADY have '
            'been applied.** Every fused-memory write in this run was recorded in '
            f'the write_journal under causation_id={run_id}. Before repeating any '
            'write, check what you already recorded this run (the write_journal '
            f'entries with causation_id={run_id}, and the current memory/task '
            'state they produced) so you do not duplicate work already committed. '
            'Then continue exactly where you left off and finish the stage.\n\n'
            '--- Original stage task (unchanged; also used verbatim if this stage '
            'has to restart fresh) ---\n\n'
            f'{stage_payload}'
        )

    def _build_mcp_config(self) -> dict:
        """Assemble MCP server config for Claude CLI.

        Includes the fused-memory server (HTTP or stdio), and optionally
        the escalation HTTP server if an escalation URL is configured.
        """
        fm_config = _find_fused_memory_server()

        if fm_config.get('type') == 'http':
            # HTTP: just a URL, no env needed (server has its own env)
            fm_entry: dict = {
                'type': 'http',
                'url': fm_config['url'],
            }
        else:
            # Stdio: command + args + env
            fm_env = dict(fm_config.get('env', {}))
            if 'OPENAI_API_KEY' not in fm_env:
                fm_env['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')
            fm_entry = {
                'command': fm_config['command'],
                'args': fm_config['args'],
                'env': fm_env,
            }

        servers: dict = {
            'fused-memory': fm_entry,
            'jcodemunch': {
                'command': 'uvx',
                'args': ['jcodemunch-mcp'],
            },
            # PRD γ: recon_report MCP server — in-process only, not in any
            # disallow list because mcp__recon-report__* tools only mutate
            # in-process state (not Graphiti / Mem0 / Taskmaster).
            # See STAGE3_DISALLOWED comment in cli_stage_runner.py for rationale.
            'recon-report': {
                'type': 'http',
                'url': f'http://127.0.0.1:{self._recon_report_port}/mcp/',
            },
        }

        # Add escalation server if URL is available
        if self._escalation_url:
            servers['escalation'] = {
                'type': 'http',
                'url': self._escalation_url,
            }

        return {'mcpServers': servers}


def _find_fused_memory_server(candidates: list[Path] | None = None) -> dict:
    """Locate the fused-memory MCP server config.

    Reads from .mcp.json — supports both HTTP and stdio transport types.
    Returns dict with either ``{type, url}`` for HTTP or
    ``{command, args, env?}`` for stdio.

    Args:
        candidates: Ordered list of .mcp.json paths to try.  Defaults to the
            production list [/home/leo/src/dark-factory/.mcp.json, cwd/.mcp.json].
            Injectable for hermetic testing.
    """
    if candidates is None:
        candidates = [
            Path('/home/leo/src/dark-factory/.mcp.json'),
            Path.cwd() / '.mcp.json',
        ]
    for candidate in candidates:
        mcp_data, _ok = load_json_or_warn(candidate, default={}, on_corrupt='warn')
        fm_config = mcp_data.get('mcpServers', {}).get('fused-memory', {})

        # HTTP transport
        if fm_config.get('type') == 'http' and fm_config.get('url'):
            return {'type': 'http', 'url': fm_config['url']}

        # Stdio transport
        if fm_config.get('command'):
            result: dict = {
                'command': fm_config['command'],
                'args': fm_config.get('args', []),
            }
            if 'env' in fm_config:
                result['env'] = fm_config['env']
            return result

    # Default: assume uv-managed fused-memory stdio
    return {
        'command': 'uv',
        'args': [
            'run', '--project', 'fused-memory',
            'python', '-m', 'fused_memory.server.main',
        ],
    }
