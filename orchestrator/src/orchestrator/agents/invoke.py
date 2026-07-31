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
from collections.abc import Iterable
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
    apply_spawn_env,
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
    sandbox_extras: list[str] | None = None,
    effort: str | None = None,
    backend: str = 'claude',
    oauth_token: str | None = None,
    resume_session_id: str | None = None,
    session_id: str | None = None,
    timeout_seconds: float | None = None,
    config_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    spawn_env: dict[str, str] | None = None,
    startup_grace_secs: float = 120.0,
    working_idle_secs: float | None = None,
    absolute_cap_secs: float | None = None,
    strict_mcp_config: bool = False,
) -> AgentResult:
    """Invoke an agent via CLI and return structured result.

    Dispatches to the appropriate backend (claude, codex, gemini, pi).
    *strict_mcp_config* (claude backend ONLY): when True, ``--strict-mcp-config``
    is emitted alongside ``--mcp-config`` so the invocation is scoped to only
    the servers in *mcp_config*, ignoring the ambient project ``.mcp.json``
    merge (the recon-watch isolation pattern; task 2796, THREAD 2). The
    codex/gemini/pi backends ignore it — ``--strict-mcp-config`` is claude-only.
    *oauth_token*, when set, overrides the Claude CLI's default credentials
    via the ``CLAUDE_CODE_OAUTH_TOKEN`` env var (multi-account failover).
    *resume_session_id*, when set, resumes an existing Claude session.
    *session_id*, when set and *resume_session_id* is not, pre-allocates the
    session UUID via ``--session-id``.
    *timeout_seconds*, when set, kills the subprocess after this many seconds.
    *env_overrides*, when set, are merged into the subprocess environment.
    *spawn_env*, when set (claude backend only), carries ``CLAUDE_SPAWN_*``
    spawn-identity vars (role/project/task/parent) for the SessionStart hook;
    any inherited ``CLAUDE_SPAWN_SESSION_ID``/``CLAUDE_SPAWN_LAUNCHER_PID``
    is scrubbed at the same time (see shared.cli_invoke._invoke_claude).
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
            sandbox_extras=sandbox_extras,
            effort=effort, oauth_token=oauth_token,
            resume_session_id=resume_session_id,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
            config_dir=config_dir,
            env_overrides=env_overrides,
            spawn_env=spawn_env,
            startup_grace_secs=startup_grace_secs,
            working_idle_secs=working_idle_secs,
            absolute_cap_secs=absolute_cap_secs,
            strict_mcp_config=strict_mcp_config,
        )
    elif backend == 'codex':
        return await _invoke_codex(
            prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
            max_budget_usd=max_budget_usd, mcp_config=mcp_config,
            sandbox_modules=sandbox_modules, sandbox_extras=sandbox_extras,
            effort=effort,
            timeout_seconds=timeout_seconds, prices=prices,
            max_turns=max_turns,
        )
    elif backend == 'gemini':
        return await _invoke_gemini(
            prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
            max_budget_usd=max_budget_usd, mcp_config=mcp_config,
            sandbox_modules=sandbox_modules, sandbox_extras=sandbox_extras,
            effort=effort,
            timeout_seconds=timeout_seconds, prices=prices,
        )
    elif backend == 'pi':
        return await _invoke_pi(
            prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
            max_budget_usd=max_budget_usd, allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools, mcp_config=mcp_config,
            sandbox_modules=sandbox_modules, sandbox_extras=sandbox_extras,
            effort=effort,
            oauth_token=oauth_token, resume_session_id=resume_session_id,
            session_id=session_id, timeout_seconds=timeout_seconds,
            env_overrides=env_overrides, prices=prices,
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
    sandbox_extras: list[str] | None = None,
    oauth_token: str | None = None,
    resume_session_id: str | None = None,
    session_id: str | None = None,
    timeout_seconds: float | None = None,
    config_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    spawn_env: dict[str, str] | None = None,
    startup_grace_secs: float = 120.0,
    working_idle_secs: float | None = None,
    absolute_cap_secs: float | None = None,
    strict_mcp_config: bool = False,
) -> AgentResult:
    """Invoke Claude Code CLI with optional bwrap sandboxing.

    Delegates to shared.cli_invoke for the core invocation; adds
    orchestrator-specific sandbox wrapping. *strict_mcp_config* threads through
    to ``build_claude_argv`` (both the sandboxed and non-sandbox sub-paths),
    emitting ``--strict-mcp-config`` alongside ``--mcp-config`` (task 2796).
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
                strict_mcp_config=strict_mcp_config,
            )

            # User prompt piped via stdin to avoid ARG_MAX
            stdin_data = prompt.encode()

            cmd = wrap_command(cmd, cwd, sandbox_modules, writable_extras=sandbox_extras)

            env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
            if env_overrides:
                env.update(env_overrides)
            if oauth_token:
                env['CLAUDE_CODE_OAUTH_TOKEN'] = oauth_token
            if config_dir:
                env['CLAUDE_CONFIG_DIR'] = str(config_dir)
            # Mirrors shared.cli_invoke._invoke_claude's non-sandbox
            # injection (task 2512 dedup) -- see apply_spawn_env's docstring
            # for the full rationale on the CLAUDE_SPAWN_SESSION_ID/
            # LAUNCHER_PID scrub.
            apply_spawn_env(env, spawn_env)

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
        spawn_env=spawn_env,
        startup_grace_secs=startup_grace_secs,
        working_idle_secs=working_idle_secs,
        absolute_cap_secs=absolute_cap_secs,
        strict_mcp_config=strict_mcp_config,
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
    sandbox_extras: list[str] | None = None,
    timeout_seconds: float | None = None,
    prices: dict[str, Any] | None = None,
    max_turns: int | None = None,
) -> AgentResult:
    """Invoke OpenAI Codex CLI.

    No instruction file (AGENTS.md) is ever written into the worktree —
    system_prompt and prompt are folded together and delivered entirely via
    stdin (cmd arg '-' tells codex exec to read instructions from stdin;
    verified codex-cli 0.143.0 `codex exec --help`). This mirrors the
    claude backend's stdin piping (used there to avoid ARG_MAX) and is
    strictly stronger than relocating/excluding the file: with no file on
    disk at all, no staging command run against this worktree — including
    the single-implementer-commit path's `git add -A -- . :!.claude`
    (git_ops.py:5320) — can ever pick it up.

    *max_turns* and *max_budget_usd* are accepted for dispatcher-signature
    uniformity/observability only: codex-cli (verified 0.143.0, full `codex
    exec --help`) has NO native --max-turns flag and NO budget flag, so
    neither cap is natively enforceable here. The wall-clock watchdog
    (*timeout_seconds*, enforced by `_run_subprocess_local`'s
    `asyncio.wait_for`) is the sole ceiling on a codex invocation — parity
    with the pi backend (PRD open-Q3). Do not read cap enforcement into the
    mere presence of these params.
    """
    # debug (not info): this fires on every codex invocation, and codex is a
    # per-task-workflow backend that can be invoked many times — an
    # unconditional INFO line here would be hot-path log noise rather than
    # an event worth recording each time.
    logger.debug(
        'codex backend: requested max_turns=%r max_budget_usd=%r are NOT '
        'natively enforced by codex-cli (no --max-turns/budget flag); only '
        'the wall-clock watchdog (timeout_seconds=%r) is enforced',
        max_turns, max_budget_usd, timeout_seconds,
    )
    temp_files: list[Path] = []
    try:
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

        # '-' tells codex exec to read the prompt from stdin instead of argv.
        cmd.append('-')
        codex_input = f'{system_prompt}\n\n{prompt}'.encode()

        if sandbox_modules is not None:
            from orchestrator.agents.sandbox_dispatch import wrap_command
            cmd = wrap_command(cmd, cwd, sandbox_modules, writable_extras=sandbox_extras)

        # Strip OPENAI_API_KEY if using OAuth
        env = dict(os.environ)

        result = await _run_subprocess_local(
            cmd, cwd, env, 'codex', model, max_budget_usd, timeout_seconds,
            stdin_data=codex_input,
        )
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
    sandbox_extras: list[str] | None = None,
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
            cmd = wrap_command(cmd, cwd, sandbox_modules, writable_extras=sandbox_extras)

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


_PI_THINKING_LEVELS = {'off', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}


def _pi_thinking(effort: str | None) -> str | None:
    """Map a Claude-style effort level to pi's ``--thinking`` level.

    pi's thinking vocabulary (off/minimal/low/medium/high/xhigh/max) is a
    superset of the Claude effort levels used elsewhere in this codebase
    (low/medium/high/max — see ``EffortConfig`` in config.py), so a
    recognized effort passes through unchanged. An unset/empty effort
    returns None (caller omits ``--thinking`` entirely); an unrecognized
    non-empty effort string falls back to pi's 'high' level rather than
    being silently dropped, mirroring _write_gemini_settings'
    ``thinking_map.get(effort, 'high')`` pattern.
    """
    if not effort:
        return None
    return effort if effort in _PI_THINKING_LEVELS else 'high'


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


def _warn_if_argv_near_arg_max(cmd: list[str]) -> None:
    """Log a warning when *cmd*'s total size approaches the OS exec() argv
    ceiling (``ARG_MAX``).

    ``_invoke_pi`` writes the (usually larger) system prompt to a file and
    passes ``--system-prompt @<path>`` precisely to keep it off argv — see
    there — but the spike template carries the user *prompt* as a plain
    positional argv element, and the spike doc does not document a
    stdin/file form for it. A sufficiently large prompt can still overflow
    ``ARG_MAX`` and fail ``exec()`` with ``E2BIG``, a failure mode invisible
    in unit tests (which use tiny fixture strings). This does not prevent
    that failure; it makes an approaching ceiling observable via a log line
    instead of a bare, cryptic subprocess-exec error.
    """
    try:
        arg_max = os.sysconf('SC_ARG_MAX')
    except (ValueError, OSError, AttributeError):
        arg_max = 2 * 1024 * 1024  # conservative fallback (Linux default is 2MB)
    # +1 per argument approximates the NUL terminator each C-string argv
    # entry carries — a rough but serviceable proxy for the real exec() cost.
    total_bytes = sum(len(arg.encode('utf-8')) + 1 for arg in cmd)
    # ARG_MAX is shared between argv and envp on Linux, so budget only a
    # quarter of it to argv as a conservative safety margin.
    threshold = arg_max // 4
    if total_bytes > threshold:
        logger.warning(
            'pi backend: argv is %d bytes, approaching the OS ARG_MAX (%d '
            'bytes) — a large prompt risks exec() failing with E2BIG',
            total_bytes, arg_max,
        )


async def _invoke_pi(
    prompt: str,
    system_prompt: str,
    cwd: Path,
    model: str,
    max_budget_usd: float,
    allowed_tools: list[str] | None,
    disallowed_tools: list[str] | None,
    mcp_config: dict | None,
    sandbox_modules: list[str] | None,
    effort: str | None,
    sandbox_extras: list[str] | None = None,
    oauth_token: str | None = None,
    resume_session_id: str | None = None,
    session_id: str | None = None,
    timeout_seconds: float | None = None,
    env_overrides: dict[str, str] | None = None,
    prices: dict[str, Any] | None = None,
) -> AgentResult:
    """Invoke the pi coding agent CLI (@earendil-works/pi-coding-agent).

    Mirrors _invoke_codex's shape; every divergence is wired STRICTLY from
    the T2 empirical spike (plans/pi-spike-findings.md) — see that doc and
    the _parse_pi_output/_pi_tool_name docstrings above for the observed
    evidence behind each choice.
    """
    temp_files: list[Path] = []
    mcp_backup: tuple[Path, Path] | None = None
    try:
        # --session-dir: a git-excluded per-task dir (.gitignore already
        # excludes .task/ wholesale) as the durable JSONL liveness/telemetry
        # surface (spike Q5). pi writes no instruction file into the
        # worktree at all, so the codex AGENTS.md-in-diff hazard (T3)
        # simply does not apply here.
        session_dir = cwd / '.task' / 'pi-sessions'
        session_dir.mkdir(parents=True, exist_ok=True)

        # Large system prompts (architect/deep-reviewer roles especially),
        # combined with a large user prompt, risk exceeding the OS ARG_MAX
        # and failing exec() with E2BIG if embedded directly on argv — a
        # failure invisible in unit tests, which use tiny fixture strings.
        # pi's `--system-prompt` accepts `<text-or-@file>` (spike template,
        # plans/pi-spike-findings.md:201), so write it to a file under the
        # same git-excluded --session-dir used above and pass `@<path>`
        # instead of the raw text — this keeps the (usually larger) system
        # prompt off argv entirely. See _warn_if_argv_near_arg_max below for
        # the remaining (spike-undocumented) user-prompt argv risk.
        system_prompt_path = session_dir / 'system-prompt.txt'
        _write_temp_instruction_file(system_prompt_path, system_prompt)
        temp_files.append(system_prompt_path)

        cmd = [
            'pi', '--model', model, '--mode', 'json', '-p',
            '--session-dir', str(session_dir),
            '--system-prompt', f'@{system_prompt_path}',
            # Don't auto-discover the repo's AGENTS.md/CLAUDE.md — the
            # system prompt above is the sole instruction source, mirroring
            # how codex/gemini are driven (no double-instruction surface).
            '--no-context-files',
        ]

        if resume_session_id:
            cmd.extend(['--session', resume_session_id])
        elif session_id:
            cmd.extend(['--session-id', session_id])

        thinking_level = _pi_thinking(effort)
        if thinking_level:
            cmd.extend(['--thinking', thinking_level])

        allowed_specs = allowed_tools or []
        tool_names = [_pi_tool_name(t) for t in allowed_specs]
        dropped_specs = [spec for spec, name in zip(allowed_specs, tool_names, strict=True) if not name]
        tools_csv = ','.join(name for name in tool_names if name)
        if tools_csv:
            cmd.extend(['--tools', tools_csv])

        disallowed_specs = disallowed_tools or []
        exclude_names = [_pi_tool_name(t) for t in disallowed_specs]
        dropped_specs += [spec for spec, name in zip(disallowed_specs, exclude_names, strict=True) if not name]
        exclude_csv = ','.join(name for name in exclude_names if name)
        if exclude_csv:
            cmd.extend(['--exclude-tools', exclude_csv])

        if dropped_specs:
            # spike Q3: a wildcard mcp__server__* spec isn't expressible as
            # a single pi direct-tool name — drop it from
            # --tools/--exclude-tools but never silently: log exactly which
            # spec(s) were lost (no silent cap, matches the repo's
            # no-silent-truncation ethos).
            logger.warning(
                f'pi backend: dropped tool spec(s) not expressible as a pi '
                f'direct-tool name (wildcards are unmappable): {", ".join(dropped_specs)}'
            )

        if mcp_config:
            # pi (per the spike) reads MCP servers from .mcp.json in its
            # cwd — no non-cwd config-path flag was observed in the spike
            # (see _write_pi_mcp_config). Back up any pre-existing
            # .mcp.json (e.g. the repo's own committed one) and restore it
            # in `finally` so this run's config never clobbers or leaks
            # into the worktree.
            mcp_json_path = cwd / '.mcp.json'
            if mcp_json_path.exists():
                backup_path = cwd / '.mcp.json.pi-invoke-backup'
                mcp_json_path.replace(backup_path)
                mcp_backup = (mcp_json_path, backup_path)
            else:
                temp_files.append(mcp_json_path)
            _write_pi_mcp_config(mcp_json_path, mcp_config)
            _warn_if_pi_mcp_cache_unwarmed(mcp_config.get('mcpServers', {}).keys())

        cmd.append(prompt)

        if sandbox_modules is not None:
            from orchestrator.agents.sandbox_dispatch import wrap_command
            cmd = wrap_command(cmd, cwd, sandbox_modules, writable_extras=sandbox_extras)

        _warn_if_argv_near_arg_max(cmd)

        env = dict(os.environ)
        if oauth_token:
            # Strip a lingering ANTHROPIC_API_KEY so it can't shadow the
            # OAuth token pi is being asked to use (mirrors
            # _invoke_claude_with_sandbox's credential-precedence guard) —
            # otherwise the multi-account OAuth failover oauth_token exists
            # for is silently defeated whenever both are present in the
            # inherited environment. Left untouched when oauth_token is
            # unset so a direct ANTHROPIC_API_KEY credential (pi's
            # non-OAuth Anthropic auth path) keeps working.
            env.pop('ANTHROPIC_API_KEY', None)
            env['ANTHROPIC_OAUTH_TOKEN'] = oauth_token
        if env_overrides:
            env.update(env_overrides)

        result = await _run_subprocess_local(cmd, cwd, env, 'pi', model, max_budget_usd, timeout_seconds)
        return _parse_pi_output(result, model, prices)

    finally:
        if mcp_backup is not None:
            target, backup = mcp_backup
            target.unlink(missing_ok=True)
            backup.replace(target)
        for f in temp_files:
            f.unlink(missing_ok=True)


def _write_pi_mcp_config(config_path: Path, mcp_config: dict | None) -> None:
    """Write pi's MCP server config in .mcp.json shape, injecting
    ``"directTools": true`` on every server (spike Q3 — required so each
    MCP tool is individually allowlistable via --tools/--exclude-tools;
    without it, pi-mcp-adapter exposes only the proxy `mcp` tool). A
    None/empty *mcp_config* writes an empty-servers config rather than
    raising.

    ⚠ CRITICAL GOTCHA (spike Q3, not handled by this function): direct-tools
    registration also needs a pre-built metadata cache
    (``~/.pi/agent/mcp-cache.json``). Until that cache is populated for a
    given server, pi-mcp-adapter exposes only the proxy `mcp` tool and the
    ``directTools``-generated ``--tools``/``--exclude-tools`` names for that
    server don't resolve — the call silently falls back to proxy (or fails
    the allowlist) even though this file is correctly shaped. The cache is
    populated by one real adapter *connection* (e.g. a proxy
    ``mcp({search})`` call, or the adapter connecting on first use) —
    running `pi-mcp-adapter init` alone does not reliably warm it for a
    project-local server. Callers relying on a fresh server's direct-tool
    names should ensure the cache has been warmed at least once first.
    """
    servers = (mcp_config or {}).get('mcpServers', {})
    out_servers = {name: {**cfg, 'directTools': True} for name, cfg in servers.items()}
    config_path.write_text(json.dumps({'mcpServers': out_servers}, indent=2))


def _pi_mcp_cache_path() -> Path:
    """Location of pi-mcp-adapter's direct-tools metadata cache (spike Q3
    CRITICAL GOTCHA — see ``_write_pi_mcp_config``'s docstring). Factored
    out to a function so tests can patch it to a tmp_path instead of
    touching the real ``~/.pi`` directory.
    """
    return Path.home() / '.pi' / 'agent' / 'mcp-cache.json'


def _warn_if_pi_mcp_cache_unwarmed(server_names: Iterable[str]) -> None:
    """Emit a warning when direct-tool MCP names are requested for servers
    but pi-mcp-adapter's metadata cache hasn't been warmed yet (spike Q3
    CRITICAL GOTCHA): until a real connection populates
    ``~/.pi/agent/mcp-cache.json``, direct-tools names silently fall back
    to the proxy `mcp` tool (or fail the --tools allowlist) even though
    the on-disk config _write_pi_mcp_config produced is correctly shaped.

    The cache's per-server internal schema isn't documented by the spike,
    so this checks only for the cache file's *existence* — a coarser but
    spike-grounded signal — rather than guessing its shape; a missing file
    means no server's direct-tool names can have resolved yet, so the
    silent proxy-tool fallback stays observable instead of surfacing later
    as mysterious 'tool not available' behavior.
    """
    names = sorted(set(server_names))
    if not names:
        return
    cache_path = _pi_mcp_cache_path()
    if not cache_path.exists():
        logger.warning(
            'pi backend: MCP direct-tools cache not found at %s — direct-tool '
            'names for server(s) %s may not resolve yet (pi-mcp-adapter falls '
            'back to the proxy `mcp` tool until warmed by one real connection; '
            'see _write_pi_mcp_config)',
            cache_path, ', '.join(names),
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
    stdin_data: bytes | None = None,
) -> _SubprocessResult:
    """Run a subprocess, log output, enforce budget timeout.

    *stdin_data*, when set, is piped to the process's stdin (mirrors
    shared.cli_invoke._run_subprocess's stdin_data param, used by the
    codex backend to deliver instructions without a worktree file — see
    _invoke_codex). When None (gemini/pi callers), behavior is
    byte-identical to before this param existed.
    """
    logger.info(f'Invoking agent: backend={backend} model={model} cwd={cwd} budget=${max_budget_usd}')
    logger.info(f'Command: {" ".join(cmd[:15])}...')

    start_ms = int(time.monotonic() * 1000)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    # Capture pgid at spawn; start_new_session guarantees pgid == pid.
    pgid = proc.pid

    # Outer try: a cancel landing *inside* the TimeoutError handler's
    # terminate_process_group await (SIGTERM already sent, SIGKILL escalation
    # pending) must still be caught here too — an exception raised inside an
    # except block is not caught by a sibling handler of the same try, so a
    # CancelledError handler that was only a sibling of `except TimeoutError:`
    # would let that escalation be abandoned with a SIGTERM-ignoring child
    # left alive.
    try:
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
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
    except asyncio.CancelledError:
        # Orchestrator shutdown / steward-stop path: the awaiting task was
        # cancelled.  Without this, proc.communicate() is cancelled but the
        # child process GROUP is never signalled and the agent survives as an
        # orphan still editing/committing in its worktree.  Mirrors the
        # already-blessed handler in shared/src/shared/cli_invoke.py (:2433)
        # and orchestrator/src/orchestrator/steward.py (:405).
        # No comm_task to reap: asyncio.wait_for already cancels and awaits
        # its inner task before propagating CancelledError.
        if proc.returncode is None:
            logger.warning(
                f'Subprocess cancelled — terminating process group for pid {proc.pid}'
            )
            await terminate_process_group(proc, pgid, grace_secs=5.0)
        raise

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
