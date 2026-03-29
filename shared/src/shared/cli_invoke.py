"""Shared Claude CLI invocation with cap-retry and structured output parsing."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.usage_gate import UsageGate

logger = logging.getLogger(__name__)

_CAP_HIT_COOLDOWN_SECS = 5.0

__all__ = [
    'AgentResult',
    'invoke_claude_agent',
    'invoke_with_cap_retry',
]


@dataclass
class AgentResult:
    success: bool
    output: str
    cost_usd: float = 0.0
    duration_ms: int = 0
    turns: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_create_tokens: int | None = None
    session_id: str = ''
    structured_output: Any = None
    subtype: str = ''
    stderr: str = ''


@dataclass
class _SubprocessResult:
    stdout: str
    stderr: str
    returncode: int
    duration_ms: int


async def invoke_claude_agent(
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
    effort: str | None = None,
    oauth_token: str | None = None,
    timeout_seconds: float | None = None,
    resume_session_id: str | None = None,
) -> AgentResult:
    """Invoke Claude Code CLI and return structured result.

    *oauth_token*, when set, overrides the Claude CLI's default credentials
    via the ``CLAUDE_CODE_OAUTH_TOKEN`` env var (multi-account failover).

    *resume_session_id*, when set, resumes an existing session via
    ``--resume <id>`` instead of starting a new one.  The system prompt is
    skipped on resume (it was already set in the initial session).
    """
    return await _invoke_claude(
        prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
        max_turns=max_turns, max_budget_usd=max_budget_usd,
        allowed_tools=allowed_tools, disallowed_tools=disallowed_tools,
        mcp_config=mcp_config, output_schema=output_schema,
        permission_mode=permission_mode, effort=effort,
        oauth_token=oauth_token, timeout_seconds=timeout_seconds,
        resume_session_id=resume_session_id,
    )


async def invoke_with_cap_retry(
    usage_gate: UsageGate | None,
    label: str,
    **invoke_kwargs,
) -> AgentResult:
    """Invoke an agent, retrying on usage-cap hits with account failover.

    *label* identifies the caller in log messages (e.g. "Module tagging",
    "Task 7 [implementer]").

    All keyword arguments are forwarded to ``invoke_claude_agent()``.
    """
    while True:
        oauth_token = None
        if usage_gate:
            oauth_token = await usage_gate.before_invoke()

        result = await invoke_claude_agent(**invoke_kwargs, oauth_token=oauth_token)

        if usage_gate and usage_gate.detect_cap_hit(
            result.stderr, result.output, 'claude', oauth_token=oauth_token,
        ):
            acct_name = usage_gate.active_account_name
            if acct_name:
                logger.warning(
                    f'{label}: cap hit, sleeping {_CAP_HIT_COOLDOWN_SECS}s '
                    f'then switching to account {acct_name}',
                )
            else:
                logger.warning(
                    f'{label}: cap hit on all accounts, sleeping '
                    f'{_CAP_HIT_COOLDOWN_SECS}s then waiting for reset',
                )
            await asyncio.sleep(_CAP_HIT_COOLDOWN_SECS)
            continue

        if usage_gate:
            usage_gate.on_agent_complete(result.cost_usd)
        break

    return result


async def _invoke_claude(
    prompt: str,
    system_prompt: str,
    cwd: Path,
    model: str,
    max_turns: int,
    max_budget_usd: float,
    allowed_tools: list[str] | None,
    disallowed_tools: list[str] | None,
    mcp_config: dict | None,
    output_schema: dict | None,
    permission_mode: str,
    effort: str | None,
    oauth_token: str | None = None,
    timeout_seconds: float | None = None,
    resume_session_id: str | None = None,
) -> AgentResult:
    """Invoke Claude Code CLI."""
    cmd = ['claude', '--print', '--output-format', 'json']

    cmd.extend(['--model', model])
    cmd.extend(['--max-budget-usd', str(max_budget_usd)])

    if resume_session_id:
        # Resume an existing session — skip --system-prompt (incompatible)
        cmd.extend(['--resume', resume_session_id])
    else:
        cmd.extend(['--system-prompt', system_prompt])

    cmd.extend(['--permission-mode', permission_mode])
    cmd.extend(['--max-turns', str(max_turns)])

    if effort:
        cmd.extend(['--effort', effort])

    if allowed_tools:
        cmd.extend(['--allowed-tools', *allowed_tools])
    if disallowed_tools:
        cmd.extend(['--disallowed-tools', *disallowed_tools])

    mcp_config_path = None
    if mcp_config:
        fd, mcp_config_path = tempfile.mkstemp(suffix='.json', prefix='mcp_')
        with open(fd, 'w') as f:
            json.dump(mcp_config, f)
        cmd.extend(['--mcp-config', mcp_config_path])

    if output_schema:
        cmd.extend(['--json-schema', json.dumps(output_schema)])

    cmd.extend(['--', prompt])

    # Strip ANTHROPIC_API_KEY so `claude` falls back to OAuth
    env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
    # Multi-account failover: inject per-invocation OAuth token
    if oauth_token:
        env['CLAUDE_CODE_OAUTH_TOKEN'] = oauth_token

    try:
        result = await _run_subprocess(cmd, cwd, env, model, timeout_seconds)
        return _parse_claude_output(result)
    finally:
        if mcp_config_path:
            Path(mcp_config_path).unlink(missing_ok=True)


def _parse_claude_output(result: _SubprocessResult) -> AgentResult:
    """Parse Claude Code JSON output into AgentResult."""
    if not result.stdout.strip():
        return AgentResult(
            success=False,
            output='Agent produced no output',
            subtype='error_empty_output',
            stderr=result.stderr,
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return AgentResult(
            success=result.returncode == 0,
            output=result.stdout,
            subtype='text_output',
            stderr=result.stderr,
        )

    cost = data.get('cost_usd', data.get('total_cost_usd', 0.0))
    duration = data.get('duration_ms', 0)
    turns = data.get('num_turns', 0)
    session_id = data.get('session_id', '')
    subtype = data.get('subtype', '')
    structured = data.get('structured_output')

    usage = data.get('usage') or {}
    input_tokens = usage.get('input_tokens')
    output_tokens = usage.get('output_tokens')
    cache_read_tokens = usage.get('cache_read_input_tokens')
    cache_create_tokens = usage.get('cache_creation_input_tokens')

    output_text = data.get('result', '')
    if not output_text and isinstance(data.get('messages'), list):
        parts = []
        for msg in data['messages']:
            if msg.get('type') == 'assistant':
                for block in msg.get('content', []):
                    if isinstance(block, dict) and block.get('type') == 'text':
                        parts.append(block['text'])
                    elif isinstance(block, str):
                        parts.append(block)
        output_text = '\n'.join(parts)

    is_success = subtype == 'success' or result.returncode == 0

    return AgentResult(
        success=is_success,
        output=output_text,
        cost_usd=cost,
        duration_ms=duration,
        turns=turns,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_create_tokens=cache_create_tokens,
        session_id=session_id,
        structured_output=structured,
        subtype=subtype,
        stderr=result.stderr,
    )


async def _run_subprocess(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    model: str,
    timeout_seconds: float | None = None,
) -> _SubprocessResult:
    """Run a subprocess, log output."""
    logger.info(f'Invoking claude agent: model={model} cwd={cwd}')
    logger.info(f'Command: {" ".join(cmd[:15])}...')

    start_ms = int(time.monotonic() * 1000)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        duration_ms = int(time.monotonic() * 1000) - start_ms
        return _SubprocessResult(
            stdout='',
            stderr=f'Process killed after {timeout_seconds}s timeout',
            returncode=1,
            duration_ms=duration_ms,
        )

    duration_ms = int(time.monotonic() * 1000) - start_ms

    stderr_text = stderr.decode()[-2000:] if stderr else ''
    if stderr_text:
        logger.info(f'Agent stderr (last 1000): {stderr_text[-1000:]}')
    logger.info(f'Agent exit code: {proc.returncode}')
    logger.info(f'Agent stdout length: {len(stdout)} bytes, first 500: {stdout.decode()[:500]}')

    return _SubprocessResult(
        stdout=stdout.decode(),
        stderr=stderr_text,
        returncode=proc.returncode if proc.returncode is not None else 1,
        duration_ms=duration_ms,
    )
