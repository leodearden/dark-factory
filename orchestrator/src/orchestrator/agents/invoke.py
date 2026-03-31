"""Multi-backend agent invocation: Claude Code, Codex, and Gemini CLIs.

Claude-specific invocation is delegated to ``shared.cli_invoke`` so that
other subsystems (e.g. fused-memory reconciliation) can reuse it without
depending on the orchestrator package.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Re-export shared Claude invocation primitives for backwards compatibility
from shared.cli_invoke import (  # noqa: F401
    AgentResult,
    _parse_claude_output,
    _run_subprocess,
    _SubprocessResult,
    invoke_claude_agent,
)

if TYPE_CHECKING:
    from shared.usage_gate import UsageGate

logger = logging.getLogger(__name__)

_CAP_HIT_COOLDOWN_SECS = 5.0

# Approximate cost per million tokens by model (for backends without native cost reporting)
_MODEL_COSTS: dict[str, dict[str, float]] = {
    # OpenAI models: {input_per_1m, output_per_1m}
    'gpt-5.4': {'input': 2.50, 'output': 10.00},
    'o4-mini': {'input': 1.10, 'output': 4.40},
    # Google models
    'gemini-3.1-pro-preview': {'input': 1.25, 'output': 5.00},
    'gemini-3-flash': {'input': 0.075, 'output': 0.30},
}


async def invoke_with_cap_retry(
    usage_gate: UsageGate | None,
    label: str,
    **invoke_kwargs,
) -> AgentResult:
    """Invoke a multi-backend agent, retrying on usage-cap hits with account failover.

    *label* identifies the caller in log messages (e.g. "Module tagging",
    "Task 7 [implementer]").

    All keyword arguments are forwarded to ``invoke_agent()``.
    """
    backend = invoke_kwargs.get('backend', 'claude')
    account_name = ''

    while True:
        oauth_token = None
        if usage_gate:
            oauth_token = await usage_gate.before_invoke()
            account_name = usage_gate.active_account_name or ''

        result = await invoke_agent(**invoke_kwargs, oauth_token=oauth_token)

        if usage_gate and usage_gate.detect_cap_hit(
            result.stderr, result.output, backend, oauth_token=oauth_token,
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

    result.account_name = account_name
    return result


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
    effort: str | None = None,
    backend: str = 'claude',
    oauth_token: str | None = None,
    resume_session_id: str | None = None,
    timeout_seconds: float | None = None,
) -> AgentResult:
    """Invoke an agent via CLI and return structured result.

    Dispatches to the appropriate backend (claude, codex, gemini).
    *oauth_token*, when set, overrides the Claude CLI's default credentials
    via the ``CLAUDE_CODE_OAUTH_TOKEN`` env var (multi-account failover).
    *resume_session_id*, when set, resumes an existing Claude session.
    *timeout_seconds*, when set, kills the subprocess after this many seconds.
    """
    if backend == 'claude':
        return await _invoke_claude_with_sandbox(
            prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
            max_turns=max_turns, max_budget_usd=max_budget_usd,
            allowed_tools=allowed_tools, disallowed_tools=disallowed_tools,
            mcp_config=mcp_config, output_schema=output_schema,
            permission_mode=permission_mode, sandbox_modules=sandbox_modules,
            effort=effort, oauth_token=oauth_token,
            resume_session_id=resume_session_id,
            timeout_seconds=timeout_seconds,
        )
    elif backend == 'codex':
        return await _invoke_codex(
            prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
            max_budget_usd=max_budget_usd, mcp_config=mcp_config,
            sandbox_modules=sandbox_modules, effort=effort,
            timeout_seconds=timeout_seconds,
        )
    elif backend == 'gemini':
        return await _invoke_gemini(
            prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
            max_budget_usd=max_budget_usd, mcp_config=mcp_config,
            sandbox_modules=sandbox_modules, effort=effort,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(f'Unknown backend: {backend!r}')


# ---------------------------------------------------------------------------
# Claude backend — thin wrapper adding sandbox support over shared.cli_invoke
# ---------------------------------------------------------------------------

async def _invoke_claude_with_sandbox(
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
    sandbox_modules: list[str] | None,
    effort: str | None,
    oauth_token: str | None = None,
    resume_session_id: str | None = None,
    timeout_seconds: float | None = None,
) -> AgentResult:
    """Invoke Claude Code CLI with optional bwrap sandboxing.

    Delegates to shared.cli_invoke for the core invocation; adds
    orchestrator-specific sandbox wrapping.
    """
    # For sandboxed invocations we need to build the command ourselves
    # and use the lower-level shared primitives
    if sandbox_modules is not None:
        from orchestrator.agents.sandbox import build_bwrap_command, is_bwrap_available
        if is_bwrap_available():
            # Build command manually to wrap with bwrap
            cmd = ['claude', '--print', '--output-format', 'json']
            cmd.extend(['--model', model])
            cmd.extend(['--max-budget-usd', str(max_budget_usd)])

            if resume_session_id:
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

            cmd = build_bwrap_command(cmd, cwd, sandbox_modules)

            env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
            if oauth_token:
                env['CLAUDE_CODE_OAUTH_TOKEN'] = oauth_token

            try:
                result = await _run_subprocess(cmd, cwd, env, model, timeout_seconds)
                return _parse_claude_output(result)
            finally:
                if mcp_config_path:
                    Path(mcp_config_path).unlink(missing_ok=True)
        else:
            logger.warning('Sandbox requested but bwrap unavailable — running unsandboxed')

    # No sandbox or bwrap unavailable — use shared invocation
    return await invoke_claude_agent(
        prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
        max_turns=max_turns, max_budget_usd=max_budget_usd,
        allowed_tools=allowed_tools, disallowed_tools=disallowed_tools,
        mcp_config=mcp_config, output_schema=output_schema,
        permission_mode=permission_mode, effort=effort,
        oauth_token=oauth_token, resume_session_id=resume_session_id,
        timeout_seconds=timeout_seconds,
    )


# ---------------------------------------------------------------------------
# Codex backend
# ---------------------------------------------------------------------------

async def _invoke_codex(
    prompt: str,
    system_prompt: str,
    cwd: Path,
    model: str,
    max_budget_usd: float,
    mcp_config: dict | None,
    sandbox_modules: list[str] | None,
    effort: str | None,
    timeout_seconds: float | None = None,
) -> AgentResult:
    """Invoke OpenAI Codex CLI."""
    temp_files: list[Path] = []
    try:
        # Write system prompt as AGENTS.md
        agents_md = cwd / 'AGENTS.md'
        _write_temp_instruction_file(agents_md, system_prompt)
        temp_files.append(agents_md)

        # Write MCP config
        if mcp_config:
            codex_dir = cwd / '.codex'
            codex_dir.mkdir(exist_ok=True)
            config_toml = codex_dir / 'config.toml'
            _write_codex_mcp_config(config_toml, mcp_config)
            temp_files.append(config_toml)

        cmd = ['codex', 'exec', '-m', model, '--json',
               '--dangerously-bypass-approvals-and-sandbox']

        if effort:
            cmd.extend(['-c', f'model_reasoning_effort={effort}'])

        cmd.append(prompt)

        if sandbox_modules is not None:
            from orchestrator.agents.sandbox import build_bwrap_command, is_bwrap_available
            if is_bwrap_available():
                cmd = build_bwrap_command(cmd, cwd, sandbox_modules)
            else:
                logger.warning('Sandbox requested but bwrap unavailable — running unsandboxed')

        # Strip OPENAI_API_KEY if using OAuth
        env = dict(os.environ)

        result = await _run_subprocess_local(cmd, cwd, env, 'codex', model, max_budget_usd, timeout_seconds)
        return _parse_codex_output(result, model)

    finally:
        for f in temp_files:
            f.unlink(missing_ok=True)


def _parse_codex_output(result: _SubprocessResult, model: str) -> AgentResult:
    """Parse Codex JSONL output into AgentResult.

    Codex outputs multiple JSON lines (JSONL): thread.started, messages,
    thread.completed, etc. We find the completion event for results.
    """
    if not result.stdout.strip():
        return AgentResult(
            success=False,
            output='Agent produced no output',
            subtype='error_empty_output',
            stderr=result.stderr,
        )

    # Parse JSONL — collect all events
    events = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # If no events parsed, try as single JSON
    if not events:
        try:
            events = [json.loads(result.stdout)]
        except json.JSONDecodeError:
            return AgentResult(
                success=result.returncode == 0,
                output=result.stdout,
                subtype='text_output',
                stderr=result.stderr,
            )

    # Find completion events and collect text
    # Codex JSONL format:
    #   {"type":"thread.started","thread_id":"..."}
    #   {"type":"turn.started"}
    #   {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
    #   {"type":"item.completed","item":{"type":"command_execution","command":"...","exit_code":0}}
    #   {"type":"turn.completed","usage":{"input_tokens":N,"output_tokens":N}}
    output_parts = []
    total_input_tokens = 0
    total_output_tokens = 0
    session_id = ''
    is_error = False
    num_turns = 0

    for event in events:
        event_type = event.get('type', '')

        if event_type == 'thread.started':
            session_id = event.get('thread_id', session_id)

        elif event_type == 'item.completed':
            item = event.get('item', {})
            if item.get('type') == 'agent_message':
                text = item.get('text', '')
                if text:
                    output_parts.append(text)
            elif item.get('type') == 'command_execution':
                # Track command executions for debugging
                pass

        elif event_type == 'turn.completed':
            num_turns += 1
            usage = event.get('usage') or {}
            total_input_tokens += usage.get('input_tokens') or 0
            total_output_tokens += usage.get('output_tokens') or 0

        elif event_type == 'error':
            is_error = True
            output_parts.append(event.get('message', 'Unknown error'))

    output_text = '\n'.join(output_parts) if output_parts else result.stdout[:2000]

    # Estimate cost from token counts
    rates = _MODEL_COSTS.get(model, {'input': 2.0, 'output': 8.0})
    cost = (total_input_tokens * rates['input'] + total_output_tokens * rates['output']) / 1_000_000

    return AgentResult(
        success=result.returncode == 0 and not is_error,
        output=output_text,
        cost_usd=cost,
        duration_ms=result.duration_ms,
        turns=num_turns,
        session_id=session_id,
        subtype='success' if result.returncode == 0 and not is_error else 'error',
        stderr=result.stderr,
    )


def _write_codex_mcp_config(config_path: Path, mcp_config: dict) -> None:
    """Write MCP server config as .codex/config.toml."""
    lines = []
    servers = mcp_config.get('mcpServers', {})
    for name, cfg in servers.items():
        lines.append('[[mcp_servers]]')
        lines.append(f'name = "{name}"')
        command = cfg.get('command', '')
        args = cfg.get('args', [])
        full_cmd = f'{command} {" ".join(args)}'.strip()
        lines.append(f'command = "{full_cmd}"')
        env_vars = cfg.get('env', {})
        if env_vars:
            lines.append('[mcp_servers.env]')
            for k, v in env_vars.items():
                lines.append(f'{k} = "{v}"')
        lines.append('')
    config_path.write_text('\n'.join(lines))


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

async def _invoke_gemini(
    prompt: str,
    system_prompt: str,
    cwd: Path,
    model: str,
    max_budget_usd: float,
    mcp_config: dict | None,
    sandbox_modules: list[str] | None,
    effort: str | None,
    timeout_seconds: float | None = None,
) -> AgentResult:
    """Invoke Google Gemini CLI."""
    temp_files: list[Path] = []
    try:
        # Write system prompt as GEMINI.md
        gemini_md = cwd / 'GEMINI.md'
        _write_temp_instruction_file(gemini_md, system_prompt)
        temp_files.append(gemini_md)

        # Write .gemini/settings.json (MCP config + thinking level)
        gemini_dir = cwd / '.gemini'
        gemini_dir.mkdir(exist_ok=True)
        settings_path = gemini_dir / 'settings.json'
        _write_gemini_settings(settings_path, mcp_config, effort)
        temp_files.append(settings_path)

        cmd = ['gemini', '-p', prompt, '-m', model, '-o', 'json', '--yolo']

        if sandbox_modules is not None:
            from orchestrator.agents.sandbox import build_bwrap_command, is_bwrap_available
            if is_bwrap_available():
                cmd = build_bwrap_command(cmd, cwd, sandbox_modules)
            else:
                logger.warning('Sandbox requested but bwrap unavailable — running unsandboxed')

        env = dict(os.environ)

        result = await _run_subprocess_local(cmd, cwd, env, 'gemini', model, max_budget_usd, timeout_seconds)
        return _parse_gemini_output(result, model)

    finally:
        for f in temp_files:
            f.unlink(missing_ok=True)


def _parse_gemini_output(result: _SubprocessResult, model: str) -> AgentResult:
    """Parse Gemini JSON output into AgentResult."""
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

    # Gemini output: {"response": "...", "stats": {"input_tokens": N, "output_tokens": N}}
    output_text = data.get('response', data.get('result', ''))
    stats = data.get('stats') or {}
    input_tokens = stats.get('input_tokens') or 0
    output_tokens = stats.get('output_tokens') or 0

    rates = _MODEL_COSTS.get(model, {'input': 1.0, 'output': 4.0})
    cost = (input_tokens * rates['input'] + output_tokens * rates['output']) / 1_000_000

    return AgentResult(
        success=result.returncode == 0,
        output=output_text,
        cost_usd=cost,
        duration_ms=result.duration_ms,
        turns=stats.get('turns', 0),
        session_id=data.get('session_id', ''),
        structured_output=data.get('structured_output'),
        subtype='success' if result.returncode == 0 else 'error',
        stderr=result.stderr,
    )


def _write_gemini_settings(
    settings_path: Path,
    mcp_config: dict | None,
    effort: str | None,
) -> None:
    """Write .gemini/settings.json with MCP servers and thinking level."""
    settings: dict[str, Any] = {}

    # Map effort to Gemini thinking levels
    if effort:
        thinking_map = {
            'low': 'low',
            'medium': 'medium',
            'high': 'high',
            'max': 'max',
        }
        settings['thinkingLevel'] = thinking_map.get(effort, 'high')

    # Convert MCP config to Gemini format
    if mcp_config:
        mcp_servers = {}
        for name, cfg in mcp_config.get('mcpServers', {}).items():
            mcp_servers[name] = {
                'command': cfg.get('command', ''),
                'args': cfg.get('args', []),
            }
            if 'env' in cfg:
                mcp_servers[name]['env'] = cfg['env']
        if mcp_servers:
            settings['mcpServers'] = mcp_servers

    settings_path.write_text(json.dumps(settings, indent=2))


# ---------------------------------------------------------------------------
# Shared helpers (orchestrator-local, for non-Claude backends)
# ---------------------------------------------------------------------------

async def _run_subprocess_local(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    backend: str,
    model: str,
    max_budget_usd: float,
    timeout_seconds: float | None = None,
) -> _SubprocessResult:
    """Run a subprocess, log output, enforce budget timeout."""
    logger.info(f'Invoking agent: backend={backend} model={model} cwd={cwd} budget=${max_budget_usd}')
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


def _write_temp_instruction_file(path: Path, content: str) -> None:
    """Write a temporary instruction file (AGENTS.md, GEMINI.md, etc.)."""
    path.write_text(content)
