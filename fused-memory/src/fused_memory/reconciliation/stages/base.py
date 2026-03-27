"""Base class for reconciliation pipeline stages — CLI-native MCP execution."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    ReconciliationEvent,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.cli_stage_runner import (
    STAGE_REPORT_SCHEMA,
    run_stage_via_cli,
)
from fused_memory.utils.validation import validate_project_id

if TYPE_CHECKING:
    from fused_memory.backends.taskmaster_client import TaskmasterBackend
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
        taskmaster: TaskmasterBackend | None,
        journal: ReconciliationJournal,
        config: ReconciliationConfig,
        usage_gate=None,
    ):
        self.stage_id = stage_id
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.config = config
        self.project_id: str = ''
        self.project_root: str = ''
        self._usage_gate = usage_gate
        self._escalation_url: str | None = None

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
    ) -> StageReport:
        """Execute this stage via Claude CLI with MCP tools."""
        # Validate project_id
        if err := validate_project_id(self.project_id):
            raise ValueError(err['error'])

        # Validate watermark.project_id consistency (skip if watermark has no project_id)
        if watermark.project_id and watermark.project_id.strip() and watermark.project_id != self.project_id:
            raise ValueError(
                f'project_id mismatch: stage has {self.project_id!r} but '
                f'watermark has {watermark.project_id!r}'
            )
        elif not (watermark.project_id and watermark.project_id.strip()):
            logger.debug(
                'Watermark has no project_id — skipping mismatch check '
                '(stage project_id=%r)',
                self.project_id,
            )

        payload = await self.assemble_payload(events, watermark, prior_reports)

        # Inject reconciliation context so CLI agents include causation_id in writes
        recon_context = (
            f'\n\n## Reconciliation Context\n'
            f'run_id: {run_id}\n'
            f'stage: {self.stage_id.value}\n'
            f'agent_id: recon-stage-{self.stage_id.value}\n'
            f'project_id: {self.project_id}\n\n'
            f'**IMPORTANT**: For every fused-memory write call, include:\n'
            f'- `agent_id`: "recon-stage-{self.stage_id.value}"\n'
            f'- In `metadata`: include `"_causation_id": "{run_id}"`\n'
        )
        payload = payload + recon_context

        disallowed = self.get_disallowed_tools()
        mcp_config = self._build_mcp_config()

        started = datetime.now(UTC)

        stage_result = await run_stage_via_cli(
            system_prompt=self.get_system_prompt(),
            payload=payload,
            disallowed_tools=disallowed,
            config=self.config,
            mcp_config=mcp_config,
            usage_gate=self._usage_gate,
            model=model,
            cwd=Path(self.config.explore_codebase_root),
            output_schema=self.get_report_schema(),
        )

        completed = datetime.now(UTC)

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

        servers: dict = {'fused-memory': fm_entry}

        # Add escalation server if URL is available
        if self._escalation_url:
            servers['escalation'] = {
                'type': 'http',
                'url': self._escalation_url,
            }

        return {'mcpServers': servers}


def _find_fused_memory_server() -> dict:
    """Locate the fused-memory MCP server config.

    Reads from .mcp.json — supports both HTTP and stdio transport types.
    Returns dict with either ``{type, url}`` for HTTP or
    ``{command, args, env?}`` for stdio.
    """
    for candidate in [
        Path('/home/leo/src/dark-factory/.mcp.json'),
        Path.cwd() / '.mcp.json',
    ]:
        if candidate.exists():
            try:
                mcp_data = json.loads(candidate.read_text())
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
            except (json.JSONDecodeError, KeyError):
                pass

    # Default: assume uv-managed fused-memory stdio
    return {
        'command': 'uv',
        'args': [
            'run', '--project', 'fused-memory',
            'python', '-m', 'fused_memory.server.main',
        ],
    }
