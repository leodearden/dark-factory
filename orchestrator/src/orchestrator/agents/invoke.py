"""Multi-backend agent invocation: Claude Code, Codex, and Gemini CLIs.

Claude-specific invocation is delegated to ``shared.cli_invoke`` so that
other subsystems (e.g. fused-memory reconciliation) can reuse it without
depending on the orchestrator package. The cap-retry loop also lives in
``shared.cli_invoke``; orchestrator callers pass ``invoke_fn=invoke_agent``
to route through the multi-backend dispatcher below.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

# Re-export shared Claude invocation primitives for backwards compatibility
from shared.cli_invoke import (  # noqa: F401
    CAP_HIT_RESUME_PROMPT,
    CRASH_RECOVERY_RESUME_PROMPT,
    AgentResult,
    _parse_claude_output,
    _run_subprocess,
    _SubprocessResult,
    build_claude_argv,
    invoke_claude_agent,
    invoke_with_cap_retry,
)

# Process-group termination helper for subprocess tree cleanup
from shared.proc_group import terminate_process_group

# MCP startup-timeout injection (bounded drop-on-fail for all projects)
from orchestrator.mcp_lifecycle import apply_mcp_startup_env

logger = logging.getLogger(__name__)

# Fallback USD/1M-token rate used ONLY when a model has no configured price
# (neither in a threaded-in prices map nor the packaged default_price_table()).
# Unifies the two former silent per-backend fallbacks (codex 2.0/8.0, gemini
# 1.0/4.0) into one DEFINED, logged rate — see _estimate_cost.
_FALLBACK_PRICE: dict[str, float] = {'input_per_1m': 2.0, 'output_per_1m': 8.0}


def _rate(entry: Any, key: str) -> float:
    """Read *key* off a price *entry*, which may be a PriceEntry (attribute)
    or a plain dict (item) — so both ``config.prices`` values and plain
    dicts (e.g. _FALLBACK_PRICE, test fixtures) work through the same
    accessor. Uses hasattr (not a getattr default) so a legitimate 0.0 rate
    is read correctly rather than masked by a fallback default.
    """
    return getattr(entry, key) if hasattr(entry, key) else entry[key]


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    prices: dict[str, Any] | None = None,
) -> float:
    """Estimate USD cost for *model* from token counts (backends without
    native cost reporting: codex, gemini).

    *prices*, when provided, is a config-shaped map (e.g. ``config.prices``,
    or a plain dict of dicts) consulted first. When None, resolves to the
    packaged ``default_price_table()`` (lazy import — this low-level module
    does not depend on orchestrator.config at module-load time). A model
    absent from the resolved table falls back to _FALLBACK_PRICE.
    """
    if prices is None:
        from orchestrator.config import default_price_table
        prices = default_price_table()
    entry = prices.get(model)
    if entry is None:
        logger.warning(
            'No configured price for model %r; falling back to $%.2f/1M input, '
            '$%.2f/1M output (add it to config.prices to silence)',
            model, _FALLBACK_PRICE['input_per_1m'], _FALLBACK_PRICE['output_per_1m'],
        )
        entry = _FALLBACK_PRICE
    return (
        input_tokens * _rate(entry, 'input_per_1m')
        + output_tokens * _rate(entry, 'output_per_1m')
    ) / 1_000_000


async def invoke_agent(
    prompt: str,
    system_prompt: str,
    cwd: Path,
    model: str = 'opus',
    max_turns: int = 50,
    max_budget_usd: float = 5.0,
    prices: dict[str, Any] | None = None,
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
    session_id: str | None = None,
    timeout_seconds: float | None = None,
    config_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    startup_grace_secs: float = 120.0,
    working_idle_secs: float | None = None,
    absolute_cap_secs: float | None = None,
) -> AgentResult:
    """Invoke an agent via CLI and return structured result.

    Dispatches to the appropriate backend (claude, codex, gemini).
    *oauth_token*, when set, overrides the Claude CLI's default credentials
    via the ``CLAUDE_CODE_OAUTH_TOKEN`` env var (multi-account failover).
    *resume_session_id*, when set, resumes an existing Claude session.
    *session_id*, when set and *resume_session_id* is not, pre-allocates the
    session UUID via ``--session-id``.
    *timeout_seconds*, when set, kills the subprocess after this many seconds.
    *env_overrides*, when set, are merged into the subprocess environment.
    *working_idle_secs* / *absolute_cap_secs*, when BOTH set (claude backend
    only), extend the working-regime watchdog past *timeout_seconds* while
    the transcript keeps advancing. Default ``None`` for both → no extension.
    *prices*, when set (codex/gemini backends only — claude reports native
    cost), is forwarded to the parser's cost estimator in place of the
    packaged default price table; see ``_estimate_cost``.
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
            session_id=session_id,
            timeout_seconds=timeout_seconds,
            config_dir=config_dir,
            env_overrides=env_overrides,
            startup_grace_secs=startup_grace_secs,
            working_idle_secs=working_idle_secs,
            absolute_cap_secs=absolute_cap_secs,
        )
    elif backend == 'codex':
        return await _invoke_codex(
            prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
            max_budget_usd=max_budget_usd, mcp_config=mcp_config,
            sandbox_modules=sandbox_modules, effort=effort,
            timeout_seconds=timeout_seconds, prices=prices,
        )
    elif backend == 'gemini':
        return await _invoke_gemini(
            prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
            max_budget_usd=max_budget_usd, mcp_config=mcp_config,
            sandbox_modules=sandbox_modules, effort=effort,
            timeout_seconds=timeout_seconds, prices=prices,
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
    session_id: str | None = None,
    timeout_seconds: float | None = None,
    config_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    startup_grace_secs: float = 120.0,
    working_idle_secs: float | None = None,
    absolute_cap_secs: float | None = None,
) -> AgentResult:
    """Invoke Claude Code CLI with optional bwrap sandboxing.

    Delegates to shared.cli_invoke for the core invocation; adds
    orchestrator-specific sandbox wrapping.
    """
    # Inject MCP_TIMEOUT into every claude agent's subprocess env so a slow,
    # hung, or incompatible stdio MCP is dropped within a bounded time and the
    # agent still reaches turn 1 (bounded drop-on-fail; reify esc-4415-232).
    # Both sub-paths below consume env_overrides, so this single transform is
    # project-agnostic and cannot miss a dispatch site.  setdefault lets
    # explicit caller values win.
    env_overrides = apply_mcp_startup_env(env_overrides)

    # For sandboxed invocations we need to build the command ourselves
    # and use the lower-level shared primitives
    if sandbox_modules is not None:
        from orchestrator.agents.sandbox_dispatch import resolve_active_backend, wrap_command
        active = resolve_active_backend()
        if active != 'none':
            # Build command via the shared single source of truth (task 2465
            # dedup) so this sandboxed path stays in lockstep with the
            # non-sandbox path in shared.cli_invoke._invoke_claude — including
            # the CLI-2.1.168 StructuredOutput deny-list expansion.
            cmd, temp_files = build_claude_argv(
                model=model,
                max_budget_usd=max_budget_usd,
                system_prompt=system_prompt,
                max_turns=max_turns,
                permission_mode=permission_mode,
                allowed_tools=allowed_tools,
                disallowed_tools=disallowed_tools,
                mcp_config=mcp_config,
                output_schema=output_schema,
                effort=effort,
                resume_session_id=resume_session_id,
                session_id=session_id,
            )

            # User prompt piped via stdin to avoid ARG_MAX
            stdin_data = prompt.encode()

            cmd = wrap_command(cmd, cwd, sandbox_modules)

            env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
            if env_overrides:
                env.update(env_overrides)
            if oauth_token:
                env['CLAUDE_CODE_OAUTH_TOKEN'] = oauth_token
            if config_dir:
                env['CLAUDE_CONFIG_DIR'] = str(config_dir)

            try:
                result = await _run_subprocess(
                    cmd, cwd, env, model, timeout_seconds,
                    stdin_data=stdin_data,
                    session_id=(resume_session_id or session_id),
                    config_dir=config_dir,
                    startup_grace_secs=startup_grace_secs,
                    working_idle_secs=working_idle_secs,
                    absolute_cap_secs=absolute_cap_secs,
                )
                return _parse_claude_output(result)
            finally:
                for path in temp_files:
                    Path(path).unlink(missing_ok=True)

    # No sandbox or bwrap unavailable — use shared invocation
    return await invoke_claude_agent(
        prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
        max_turns=max_turns, max_budget_usd=max_budget_usd,
        allowed_tools=allowed_tools, disallowed_tools=disallowed_tools,
        mcp_config=mcp_config, output_schema=output_schema,
        permission_mode=permission_mode, effort=effort,
        oauth_token=oauth_token, resume_session_id=resume_session_id,
        session_id=session_id,
        timeout_seconds=timeout_seconds, config_dir=config_dir,
        env_overrides=env_overrides,
        startup_grace_secs=startup_grace_secs,
        working_idle_secs=working_idle_secs,
        absolute_cap_secs=absolute_cap_secs,
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
    prices: dict[str, Any] | None = None,
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
            from orchestrator.agents.sandbox_dispatch import wrap_command
            cmd = wrap_command(cmd, cwd, sandbox_modules)

        # Strip OPENAI_API_KEY if using OAuth
        env = dict(os.environ)

        result = await _run_subprocess_local(cmd, cwd, env, 'codex', model, max_budget_usd, timeout_seconds)
        return _parse_codex_output(result, model, prices)

    finally:
        for f in temp_files:
            f.unlink(missing_ok=True)


def _parse_codex_output(
    result: _SubprocessResult,
    model: str,
    prices: dict[str, Any] | None = None,
) -> AgentResult:
    """Parse Codex JSONL output into AgentResult.

    Codex outputs multiple JSON lines (JSONL): thread.started, messages,
    thread.completed, etc. We find the completion event for results.

    timed_out is propagated directly from result.timed_out on every return path.
    *prices*, when provided, is forwarded to _estimate_cost (see there for the
    resolution order when None).
    """
    if not result.stdout.strip():
        return AgentResult(
            success=False,
            output='Agent produced no output',
            subtype='error_empty_output',
            stderr=result.stderr,
            timed_out=result.timed_out,
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
                timed_out=result.timed_out,
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
    cost = _estimate_cost(model, total_input_tokens, total_output_tokens, prices)

    return AgentResult(
        success=result.returncode == 0 and not is_error,
        output=output_text,
        cost_usd=cost,
        duration_ms=result.duration_ms,
        turns=num_turns,
        session_id=session_id,
        subtype='success' if result.returncode == 0 and not is_error else 'error',
        stderr=result.stderr,
        timed_out=result.timed_out,
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
    prices: dict[str, Any] | None = None,
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
            from orchestrator.agents.sandbox_dispatch import wrap_command
            cmd = wrap_command(cmd, cwd, sandbox_modules)

        env = dict(os.environ)

        result = await _run_subprocess_local(cmd, cwd, env, 'gemini', model, max_budget_usd, timeout_seconds)
        return _parse_gemini_output(result, model, prices)

    finally:
        for f in temp_files:
            f.unlink(missing_ok=True)


def _parse_gemini_output(
    result: _SubprocessResult,
    model: str,
    prices: dict[str, Any] | None = None,
) -> AgentResult:
    """Parse Gemini JSON output into AgentResult.

    timed_out is propagated directly from result.timed_out on every return path.
    *prices*, when provided, is forwarded to _estimate_cost (see there for the
    resolution order when None).
    """
    if not result.stdout.strip():
        return AgentResult(
            success=False,
            output='Agent produced no output',
            subtype='error_empty_output',
            stderr=result.stderr,
            timed_out=result.timed_out,
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return AgentResult(
            success=result.returncode == 0,
            output=result.stdout,
            subtype='text_output',
            stderr=result.stderr,
            timed_out=result.timed_out,
        )

    # Gemini output: {"response": "...", "stats": {"input_tokens": N, "output_tokens": N}}
    output_text = data.get('response', data.get('result', ''))
    stats = data.get('stats') or {}
    input_tokens = stats.get('input_tokens') or 0
    output_tokens = stats.get('output_tokens') or 0

    cost = _estimate_cost(model, input_tokens, output_tokens, prices)

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
        timed_out=result.timed_out,
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
# Pi backend (@earendil-works/pi-coding-agent) — wired from the T2 empirical
# spike, plans/pi-spike-findings.md. See that doc for the observed evidence
# behind every divergence from the codex/gemini shape below.
# ---------------------------------------------------------------------------

_PI_BUILTIN_TOOL_MAP: dict[str, str] = {
    'Read': 'read',
    'Write': 'write',
    'Edit': 'edit',
    'Bash': 'bash',
    'Glob': 'glob',
    'Grep': 'grep',
}


def _pi_tool_name(spec: str) -> str | None:
    """Map a Claude-style tool spec to pi's direct-tool name (spike Q3).

    Concrete MCP tools (``mcp__<server>__<tool>``) map to
    ``<server '-'->'_'>_<tool>`` — server-key hyphens become underscores,
    the tool name is used verbatim. This is pi-mcp-adapter's default
    ``toolPrefix: "server"`` formula (``formatToolName``/``getServerPrefix``),
    live-confirmed by the spike (``spike-demo`` + ``echo_it`` ->
    ``spike_demo_echo_it``).

    A wildcard MCP spec (``mcp__<server>__*``) is not expressible as a
    single direct-tool name — it returns None; callers must drop it from
    ``--tools``/``--exclude-tools`` and log a warning (no silent cap).

    Built-in Claude tool names, optionally carrying a ``(...)`` argument
    suffix (e.g. ``Bash(git:*)`` -> ``Bash``), map to pi's lowercase
    built-in names; an unrecognized built-in falls through to
    ``spec.lower()``.
    """
    if spec.startswith('mcp__'):
        remainder = spec[len('mcp__'):]
        server, _, tool = remainder.partition('__')
        if not tool or '*' in tool:
            return None
        return f"{server.replace('-', '_')}_{tool}"
    base = spec.split('(', 1)[0]
    return _PI_BUILTIN_TOOL_MAP.get(base, base.lower())


def _parse_pi_output(
    result: _SubprocessResult,
    model: str,
    prices: dict[str, Any] | None = None,
) -> AgentResult:
    """Parse pi's `--mode json` JSONL stream into AgentResult.

    Line-parses each JSONL event (skipping blank/unparseable lines, like
    _parse_codex_output). ``session_id`` comes from the first `session`
    event's `id`. Prefers the single `agent_end` event's `messages` list
    (spike-recommended, authoritative); falls back to accumulating over
    `turn_end` events when no `agent_end` was seen. `turns` counts
    `turn_end` events. `message_update`/`tool_execution_update` (streaming
    deltas) are ignored. cost_usd/token totals sum pi's NATIVE per-message
    `usage` over assistant-role messages. output = the joined
    content[type=="text"].text of the *terminal* (last) assistant message.

    timed_out/duration_ms are propagated from *result* unchanged, exactly
    like _parse_codex_output / _parse_gemini_output.
    """
    if not result.stdout.strip():
        return AgentResult(
            success=False,
            output='Agent produced no output',
            subtype='error_empty_output',
            stderr=result.stderr,
            timed_out=result.timed_out,
        )

    events = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    session_id = ''
    turns = 0
    agent_end_messages: list[Any] | None = None
    turn_end_messages: list[Any] = []

    for event in events:
        event_type = event.get('type', '')

        if event_type == 'session':
            session_id = event.get('id', session_id)

        elif event_type == 'turn_end':
            turns += 1
            message = event.get('message')
            if isinstance(message, dict):
                turn_end_messages.append(message)

        elif event_type == 'agent_end':
            messages = event.get('messages')
            if isinstance(messages, list):
                agent_end_messages = messages

        # message_update / tool_execution_update (streaming deltas) and any
        # other event types are intentionally ignored.

    source_messages = agent_end_messages if agent_end_messages is not None else turn_end_messages
    assistant_messages = [
        m for m in source_messages
        if isinstance(m, dict) and m.get('role') == 'assistant'
    ]

    total_input_tokens = 0
    total_output_tokens = 0
    native_cost = 0.0
    for message in assistant_messages:
        usage = message.get('usage') or {}
        total_input_tokens += usage.get('input') or 0
        total_output_tokens += usage.get('output') or 0
        cost = usage.get('cost') or {}
        native_cost += cost.get('total') or 0.0

    # spike Q2: pi reports native per-message USD cost, unlike codex/gemini
    # (tokens-only). Use it directly; fall back to the config price table
    # (_estimate_cost, task 2459) only for the exotic custom-provider/
    # base-url case where pi has no price data of its own (native cost
    # absent or reported as exactly 0.0).
    cost_usd = native_cost if native_cost else _estimate_cost(
        model, total_input_tokens, total_output_tokens, prices,
    )

    terminal = assistant_messages[-1] if assistant_messages else None
    if terminal is not None:
        content = terminal.get('content') or []
        output_text = '\n'.join(
            part.get('text', '') for part in content
            if isinstance(part, dict) and part.get('type') == 'text'
        )
    else:
        output_text = ''

    # spike Q1 (LOAD-BEARING): pi's `--mode json -p` exits 0 even on a hard
    # API failure (401/400/out-of-quota) — the failure lives IN the stream
    # as the terminal assistant message's stopReason=='error'/errorMessage,
    # not the process exit code. Contrast _parse_codex_output, which leans
    # on returncode==0 alone. Do NOT derive success from returncode alone.
    error_message = terminal.get('errorMessage') if terminal is not None else None
    is_error = (
        terminal is None
        or terminal.get('stopReason') == 'error'
        or bool(error_message)
    )
    success = result.returncode == 0 and not is_error

    if success:
        return AgentResult(
            success=True,
            output=output_text,
            cost_usd=cost_usd,
            duration_ms=result.duration_ms,
            turns=turns,
            session_id=session_id,
            subtype='success',
            stderr=result.stderr,
            timed_out=result.timed_out,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    return AgentResult(
        success=False,
        output=error_message or output_text or 'pi agent reported an error',
        cost_usd=cost_usd,
        duration_ms=result.duration_ms,
        turns=turns,
        session_id=session_id,
        subtype='error',
        stderr=result.stderr,
        timed_out=result.timed_out,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )


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
        start_new_session=True,
    )
    # Capture pgid at spawn; start_new_session guarantees pgid == pid.
    pgid = proc.pid

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        await terminate_process_group(proc, pgid)
        duration_ms = int(time.monotonic() * 1000) - start_ms
        return _SubprocessResult(
            stdout='',
            stderr=f'Process killed after {timeout_seconds}s timeout (SIGTERM+SIGKILL)',
            returncode=1,
            duration_ms=duration_ms,
            timed_out=True,
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
