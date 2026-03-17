"""Thin wrapper over the Claude Code CLI for agent invocation."""

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    success: bool
    output: str
    cost_usd: float = 0.0
    duration_ms: int = 0
    turns: int = 0
    session_id: str = ''
    structured_output: Any = None
    subtype: str = ''


async def invoke_agent(
    prompt: str,
    system_prompt: str,
    cwd: Path,
    model: str = 'opus',
    max_turns: int = 50,
    max_budget_usd: float = 5.0,
    allowed_tools: list[str] | None = None,
    disallowed_tools: list[str] | None = None,
    mcp_config: dict | None = None,
    output_schema: dict | None = None,
    permission_mode: str = 'bypassPermissions',
    sandbox_modules: list[str] | None = None,
) -> AgentResult:
    """Invoke a Claude Code instance via CLI and return structured result.

    Uses `claude --print --output-format json` for non-interactive execution.
    """
    cmd = ['claude', '--print', '--output-format', 'json']

    # Model
    cmd.extend(['--model', model])

    # Limits
    cmd.extend(['--max-budget-usd', str(max_budget_usd)])

    # System prompt
    cmd.extend(['--system-prompt', system_prompt])

    # Permission mode
    cmd.extend(['--permission-mode', permission_mode])

    # Tools
    if allowed_tools:
        cmd.extend(['--allowed-tools', *allowed_tools])
    if disallowed_tools:
        cmd.extend(['--disallowed-tools', *disallowed_tools])

    # MCP config — write to temp file
    mcp_config_path = None
    if mcp_config:
        fd, mcp_config_path = tempfile.mkstemp(suffix='.json', prefix='mcp_')
        with open(fd, 'w') as f:
            json.dump(mcp_config, f)
        cmd.extend(['--mcp-config', mcp_config_path])

    # Structured output
    if output_schema:
        cmd.extend(['--json-schema', json.dumps(output_schema)])

    # Prompt goes last
    cmd.extend(['--', prompt])

    # Wrap command in bwrap sandbox if modules are specified
    if sandbox_modules is not None:
        from orchestrator.agents.sandbox import build_bwrap_command
        cmd = build_bwrap_command(cmd, cwd, sandbox_modules)

    logger.info(f'Invoking agent: model={model} cwd={cwd} budget=${max_budget_usd}')
    logger.info(f'Command: {" ".join(cmd[:15])}...')

    try:
        # Strip ANTHROPIC_API_KEY so `claude` falls back to the Max
        # subscription (OAuth) instead of billing against the API key.
        env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if stderr:
            logger.info(f'Agent stderr (last 1000): {stderr.decode()[-1000:]}')
        logger.info(f'Agent exit code: {proc.returncode}')
        logger.info(f'Agent stdout length: {len(stdout)} bytes, first 500: {stdout.decode()[:500]}')

        raw = stdout.decode()
        if not raw.strip():
            return AgentResult(
                success=False,
                output='Agent produced no output',
                subtype='error_empty_output',
            )

        # Parse JSON result
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # If output format is not JSON, treat raw text as output
            return AgentResult(
                success=proc.returncode == 0,
                output=raw,
                subtype='text_output',
            )

        # Extract fields from SDK result message
        cost = result.get('cost_usd', result.get('total_cost_usd', 0.0))
        duration = result.get('duration_ms', 0)
        turns = result.get('num_turns', 0)
        session_id = result.get('session_id', '')
        subtype = result.get('subtype', '')
        structured = result.get('structured_output')

        # Extract text output from result
        output_text = result.get('result', '')
        if not output_text and isinstance(result.get('messages'), list):
            # Collect assistant text blocks
            parts = []
            for msg in result['messages']:
                if msg.get('type') == 'assistant':
                    for block in msg.get('content', []):
                        if isinstance(block, dict) and block.get('type') == 'text':
                            parts.append(block['text'])
                        elif isinstance(block, str):
                            parts.append(block)
            output_text = '\n'.join(parts)

        is_success = subtype == 'success' or proc.returncode == 0

        return AgentResult(
            success=is_success,
            output=output_text,
            cost_usd=cost,
            duration_ms=duration,
            turns=turns,
            session_id=session_id,
            structured_output=structured,
            subtype=subtype,
        )

    finally:
        # Cleanup temp MCP config
        if mcp_config_path:
            Path(mcp_config_path).unlink(missing_ok=True)
