"""Shared Claude CLI invocation with cap-retry and structured output parsing."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.cost_store import CostStore
    from shared.usage_gate import UsageGate

logger = logging.getLogger(__name__)

_CAP_HIT_COOLDOWN_SECS = 5.0
_MAX_CAP_COOLDOWN_SECS = 300.0

__all__ = [
    'AgentResult',
    'invoke_claude_agent',
    'invoke_with_cap_retry',
]


@dataclass
class AgentResult:
    """Structured result from a CLI agent invocation.

    Fields:
    - ``success``: whether the agent completed without error
    - ``output``: the primary text response from the agent
    - ``cost_usd``: total cost in USD (0.0 if not reported by the provider)
    - ``duration_ms``: wall-clock time in milliseconds
    - ``turns``: number of agentic turns (0 if not tracked)
    - ``session_id``: provider session identifier for resumption
    - ``structured_output``: parsed JSON output if an output schema was requested
    - ``subtype``: provider-specific result subtype (e.g. ``"success"``, ``"error"``)
    - ``stderr``: captured stderr from the CLI process
    - ``account_name``: the OAuth account used for this invocation
    """

    success: bool
    output: str
    cost_usd: float = 0.0
    duration_ms: int = 0
    turns: int = 0
    session_id: str = ''
    structured_output: Any = None
    subtype: str = ''
    stderr: str = ''
    account_name: str = ''
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_create_tokens: int | None = None


def _to_token_count(v: int | None) -> int | None:
    """Normalise a raw token count from a provider response to ``Optional[int]``.

    ``None`` means the provider did not report a value.  Both ``0`` and
    ``None`` are normalised to ``None`` because zero tokens are impossible in
    practice — if a field is zero it means the provider omitted it.

    This convention prevents silent cost under-reporting caused by treating an
    absent field as ``0`` when summing token counts.

    Usage guidance: use this helper when you need ``Optional[int]`` semantics
    (e.g. accumulating across multiple turns where absence must be distinguished
    from zero).  At arithmetic sites that immediately discard ``None`` via
    ``or 0``, prefer ``value.get('field') or 0`` directly — the roundtrip
    through this helper adds no value there.
    """
    return v or None


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
    *,
    cost_store: CostStore | None = None,
    run_id: str = '',
    task_id: str = '',
    project_id: str = '',
    role: str = '',
    **invoke_kwargs,
) -> AgentResult:
    """Invoke an agent, retrying on usage-cap hits with account failover.

    Uses exponential backoff: the first pass through all accounts uses the
    base cooldown (5 s).  After each full cycle through every account, the
    cooldown doubles, capped at ``_MAX_CAP_COOLDOWN_SECS`` (300 s).

    *label* identifies the caller in log messages (e.g. "Module tagging",
    "Task 7 [implementer]").

    When *cost_store* is provided, successful invocations are recorded via
    ``save_invocation()`` and cap-hit events via ``save_account_event()``.

    All keyword arguments are forwarded to ``invoke_claude_agent()``.
    """
    model = invoke_kwargs.get('model', 'opus')
    consecutive_cap_hits = 0
    num_accounts = max(usage_gate.account_count, 1) if usage_gate else 1
    while True:
        oauth_token = None
        account_name = ''
        if usage_gate:
            oauth_token = await usage_gate.before_invoke()
            account_name = usage_gate.active_account_name or ''

        started_at = datetime.now(UTC).isoformat()
        result = await invoke_claude_agent(**invoke_kwargs, oauth_token=oauth_token)
        completed_at = datetime.now(UTC).isoformat()

        if usage_gate and usage_gate.detect_cap_hit(
            result.stderr, result.output, 'claude', oauth_token=oauth_token,
        ):
            consecutive_cap_hits += 1
            full_cycles = (consecutive_cap_hits - 1) // num_accounts
            cooldown = min(
                _CAP_HIT_COOLDOWN_SECS * (2 ** full_cycles),
                _MAX_CAP_COOLDOWN_SECS,
            )

            acct_name = usage_gate.active_account_name
            if cost_store:
                try:
                    await cost_store.save_account_event(
                        account_name=account_name,
                        event_type='cap_hit',
                        project_id=project_id or None,
                        run_id=run_id or None,
                        details=label,
                        created_at=datetime.now(UTC).isoformat(),
                    )
                except Exception:
                    logger.warning('Failed to save cap_hit event', exc_info=True)
            if acct_name:
                logger.warning(
                    f'{label}: cap hit ({consecutive_cap_hits} consecutive), '
                    f'sleeping {cooldown:.0f}s then switching to account {acct_name}',
                )
            else:
                logger.warning(
                    f'{label}: cap hit on all accounts ({consecutive_cap_hits} consecutive), '
                    f'sleeping {cooldown:.0f}s then waiting for reset',
                )
            await asyncio.sleep(cooldown)
            continue

        if usage_gate:
            usage_gate.on_agent_complete(result.cost_usd)
        break

    result.account_name = account_name
    if cost_store:
        try:
            await cost_store.save_invocation(
                run_id=run_id,
                task_id=task_id or None,
                project_id=project_id,
                account_name=account_name,
                model=model,
                role=role,
                cost_usd=result.cost_usd,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cache_read_tokens=result.cache_read_tokens,
                cache_create_tokens=result.cache_create_tokens,
                duration_ms=result.duration_ms,
                capped=False,
                started_at=started_at,
                completed_at=completed_at,
            )
        except Exception:
            logger.warning('Failed to save invocation cost', exc_info=True)
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
    input_tokens = _to_token_count(usage.get('input_tokens'))
    output_tokens = _to_token_count(usage.get('output_tokens'))
    cache_read_tokens = _to_token_count(usage.get('cache_read_input_tokens'))
    cache_create_tokens = _to_token_count(usage.get('cache_creation_input_tokens'))

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
        session_id=session_id,
        structured_output=structured,
        subtype=subtype,
        stderr=result.stderr,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_create_tokens=cache_create_tokens,
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
