"""Shared Claude CLI invocation with cap-retry and structured output parsing."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# VllmBridge depends on aiohttp, which is not installed in every consumer
# environment (e.g. dashboard's venv).  Tolerate ImportError so that callers
# that never set ANTHROPIC_BASE_URL can still import shared.cli_invoke.
try:
    from shared.vllm_bridge import VllmBridge
except ImportError:  # pragma: no cover - exercised only when aiohttp absent
    VllmBridge = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from shared.config_dir import TaskConfigDir
    from shared.cost_store import CostStore
    from shared.usage_gate import UsageGate

logger = logging.getLogger(__name__)

_CAP_HIT_COOLDOWN_SECS = 5.0
_MAX_CAP_COOLDOWN_SECS = 300.0
_DEFAULT_MAX_CAP_RETRIES = 20
_DEFAULT_CAP_RETRY_DEADLINE_SECS = 3600.0
CAP_HIT_RESUME_PROMPT = (
    'Your previous run was interrupted by a usage limit. '
    'Continue where you left off and complete your task.'
)

__all__ = [
    'CAP_HIT_RESUME_PROMPT',
    'AgentResult',
    'AllAccountsCappedException',
    'invoke_claude_agent',
    'invoke_with_cap_retry',
]


class AllAccountsCappedException(Exception):
    """Raised when the cap-hit retry loop exceeds max retries or wall-clock deadline.

    Attributes:
    - ``retries``: number of consecutive cap hits before giving up
    - ``elapsed_secs``: wall-clock seconds elapsed since first cap hit
    - ``label``: caller label from invoke_with_cap_retry (e.g. "Task 7 [impl]")
    """

    def __init__(self, retries: int, elapsed_secs: float, label: str) -> None:
        self.retries = retries
        self.elapsed_secs = elapsed_secs
        self.label = label
        super().__init__(
            f'{label}: all accounts capped after {retries} retries '
            f'({elapsed_secs:.1f}s elapsed)'
        )


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
    - ``timed_out``: True when the subprocess was killed by a wall-clock timeout
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
    timed_out: bool = False


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
    timed_out: bool = False


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
    config_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> AgentResult:
    """Invoke Claude Code CLI and return structured result.

    *oauth_token*, when set, overrides the Claude CLI's default credentials
    via the ``CLAUDE_CODE_OAUTH_TOKEN`` env var (multi-account failover).

    *resume_session_id*, when set, resumes an existing session via
    ``--resume <id>`` instead of starting a new one.  The system prompt is
    skipped on resume (it was already set in the initial session).

    *env_overrides*, when set, are merged into the subprocess environment.
    Used to point Claude Code at a vLLM endpoint via ``ANTHROPIC_BASE_URL``.
    """
    return await _invoke_claude(
        prompt=prompt, system_prompt=system_prompt, cwd=cwd, model=model,
        max_turns=max_turns, max_budget_usd=max_budget_usd,
        allowed_tools=allowed_tools, disallowed_tools=disallowed_tools,
        mcp_config=mcp_config, output_schema=output_schema,
        permission_mode=permission_mode, effort=effort,
        oauth_token=oauth_token, timeout_seconds=timeout_seconds,
        resume_session_id=resume_session_id, config_dir=config_dir,
        env_overrides=env_overrides,
    )


async def invoke_with_cap_retry(
    usage_gate: UsageGate | None,
    label: str,
    *,
    config_dir: TaskConfigDir | None = None,
    cost_store: CostStore | None = None,
    run_id: str = '',
    task_id: str = '',
    project_id: str = '',
    role: str = '',
    max_cap_retries: int | None = _DEFAULT_MAX_CAP_RETRIES,
    cap_retry_deadline_secs: float | None = _DEFAULT_CAP_RETRY_DEADLINE_SECS,
    **invoke_kwargs,
) -> AgentResult:
    """Invoke an agent, retrying on usage-cap hits with account failover.

    Uses exponential backoff: the first pass through all accounts uses the
    base cooldown (5 s).  After each full cycle through every account, the
    cooldown doubles, capped at ``_MAX_CAP_COOLDOWN_SECS`` (300 s).

    On cap hit, if the capped invocation produced a ``session_id``, the
    retry resumes that session via ``--resume`` instead of starting fresh.
    This preserves all agent progress (tool calls, reasoning) across
    account switches.  If resume itself fails (non-cap-hit error), falls
    back to a fresh invocation with the original prompt.

    *label* identifies the caller in log messages (e.g. "Module tagging",
    "Task 7 [implementer]").

    When *cost_store* is provided, successful invocations are recorded via
    ``save_invocation()`` and cap-hit events via ``save_account_event()``.

    All keyword arguments are forwarded to ``invoke_claude_agent()``.
    """
    model = invoke_kwargs.get('model', 'opus')
    original_prompt = invoke_kwargs.get('prompt', '')
    consecutive_cap_hits = 0
    num_accounts = max(usage_gate.account_count, 1) if usage_gate else 1
    retry_start = time.monotonic()
    while True:
        oauth_token = None
        account_name = ''
        skip_confirm = False  # set True when heuristic fires but account unresolvable
        if usage_gate:
            oauth_token = await usage_gate.before_invoke()
            account_name = usage_gate.active_account_name or ''

        if config_dir and oauth_token:
            config_dir.write_credentials(oauth_token)

        started_at = datetime.now(UTC).isoformat()
        try:
            result = await invoke_claude_agent(
                **invoke_kwargs,
                oauth_token=oauth_token,
                config_dir=config_dir.path if config_dir else None,
            )
        except BaseException:
            # Safety net: release probe slot on any exception so probe_in_flight
            # never leaks. See also: orchestrator/agents/invoke.py:invoke_with_cap_retry,
            # orchestrator/steward.py:_invoke_with_session
            if usage_gate is not None:
                try:
                    usage_gate.release_probe_slot(oauth_token)
                except Exception:
                    logger.warning('release_probe_slot failed', exc_info=True)
            raise
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

            # Resume the capped session on the next account if possible
            if result.session_id:
                invoke_kwargs['resume_session_id'] = result.session_id
                invoke_kwargs['prompt'] = CAP_HIT_RESUME_PROMPT
                resume_or_fresh = 'resuming'
            else:
                invoke_kwargs.pop('resume_session_id', None)
                invoke_kwargs['prompt'] = original_prompt
                resume_or_fresh = 'fresh'

            if acct_name:
                logger.warning(
                    f'{label}: cap hit ({consecutive_cap_hits} consecutive), '
                    f'sleeping {cooldown:.0f}s then {resume_or_fresh} on account {acct_name}',
                )
            else:
                logger.warning(
                    f'{label}: cap hit on all accounts ({consecutive_cap_hits} consecutive), '
                    f'sleeping {cooldown:.0f}s then waiting for reset ({resume_or_fresh})',
                )

            # Guard: raise before sleeping if retry limit or deadline exceeded
            elapsed = time.monotonic() - retry_start
            if max_cap_retries is not None and consecutive_cap_hits >= max_cap_retries:
                logger.error(
                    f'{label}: giving up after {consecutive_cap_hits} consecutive cap hits '
                    f'({elapsed:.1f}s elapsed, {num_accounts} account(s))',
                )
                raise AllAccountsCappedException(
                    retries=consecutive_cap_hits,
                    elapsed_secs=elapsed,
                    label=label,
                )
            if cap_retry_deadline_secs is not None and elapsed > cap_retry_deadline_secs:
                logger.error(
                    f'{label}: cap retry deadline exceeded after {elapsed:.1f}s '
                    f'({consecutive_cap_hits} retries, {num_accounts} account(s))',
                )
                raise AllAccountsCappedException(
                    retries=consecutive_cap_hits,
                    elapsed_secs=elapsed,
                    label=label,
                )

            await asyncio.sleep(cooldown)
            continue

        # Heuristic safety net: a zero-cost, near-instant, ≤1-turn result
        # that wasn't caught by pattern matching is almost certainly a cap
        # hit with an unrecognised message format.  Treat it as a cap hit so
        # the retry loop can wait / fail over instead of silently returning a
        # useless "success" to the caller.
        if (
            usage_gate
            and not result.success  # is_error=true → success=False after fix 2
            and result.cost_usd == 0
            and result.turns <= 1
            and result.duration_ms < 5000
        ):
            logger.warning(
                f'{label}: suspicious zero-cost instant exit (turns={result.turns}, '
                f'duration={result.duration_ms}ms) — treating as cap hit. '
                f'Output: {result.output[:200]!r}',
            )
            cap_marked = usage_gate._handle_cap_detected(
                f'Heuristic cap: zero-cost instant exit — {result.output[:120]}',
                None,
                oauth_token,
            )
            if not cap_marked:
                logger.warning(
                    f'{label}: heuristic cap suspected but no account could be marked '
                    f'(token unresolved) — treating as normal failure',
                )
                # Calling confirm_account_ok with an unresolvable token would be
                # semantically misleading (we never confirmed the account is ok —
                # we simply couldn't identify it).  Skip it for this iteration.
                skip_confirm = True
            else:
                consecutive_cap_hits += 1
                full_cycles = (consecutive_cap_hits - 1) // num_accounts
                cooldown = min(
                    _CAP_HIT_COOLDOWN_SECS * (2 ** full_cycles),
                    _MAX_CAP_COOLDOWN_SECS,
                )
                # Cannot resume a session that never ran
                invoke_kwargs.pop('resume_session_id', None)
                invoke_kwargs['prompt'] = original_prompt
                acct_name = usage_gate.active_account_name
                logger.warning(
                    f'{label}: sleeping {cooldown:.0f}s then retrying fresh on {acct_name or "next account"}',
                )

                # Guard: raise before sleeping if retry limit or deadline exceeded
                elapsed = time.monotonic() - retry_start
                if max_cap_retries is not None and consecutive_cap_hits >= max_cap_retries:
                    logger.error(
                        f'{label}: giving up after {consecutive_cap_hits} consecutive heuristic cap hits '
                        f'({elapsed:.1f}s elapsed, {num_accounts} account(s))',
                    )
                    raise AllAccountsCappedException(
                        retries=consecutive_cap_hits,
                        elapsed_secs=elapsed,
                        label=label,
                    )
                if cap_retry_deadline_secs is not None and elapsed > cap_retry_deadline_secs:
                    logger.error(
                        f'{label}: cap retry deadline exceeded after {elapsed:.1f}s '
                        f'(heuristic branch, {consecutive_cap_hits} retries, {num_accounts} account(s))',
                    )
                    raise AllAccountsCappedException(
                        retries=consecutive_cap_hits,
                        elapsed_secs=elapsed,
                        label=label,
                    )

                await asyncio.sleep(cooldown)
                continue

        # Non-cap-hit failure while resuming → fall back to fresh invocation
        if not result.success and invoke_kwargs.get('resume_session_id'):
            logger.warning(
                f'{label}: resume failed (session_id={invoke_kwargs["resume_session_id"]}), '
                f'retrying fresh',
            )
            invoke_kwargs.pop('resume_session_id', None)
            invoke_kwargs['prompt'] = original_prompt
            continue

        if usage_gate and not skip_confirm:
            usage_gate.confirm_account_ok(oauth_token)
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
    config_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> AgentResult:
    """Invoke Claude Code CLI."""
    cmd = ['claude', '--print', '--output-format', 'json']

    cmd.extend(['--model', model])
    cmd.extend(['--max-budget-usd', str(max_budget_usd)])

    temp_files: list[str] = []

    if resume_session_id:
        # Resume an existing session — skip --system-prompt (incompatible)
        cmd.extend(['--resume', resume_session_id])
    else:
        # Write system prompt to temp file to avoid ARG_MAX on large payloads
        fd, sysprompt_path = tempfile.mkstemp(suffix='.txt', prefix='sysprompt_')
        with open(fd, 'w') as f:
            f.write(system_prompt)
        temp_files.append(sysprompt_path)
        cmd.extend(['--system-prompt-file', sysprompt_path])

    cmd.extend(['--permission-mode', permission_mode])
    cmd.extend(['--max-turns', str(max_turns)])

    if effort:
        cmd.extend(['--effort', effort])

    if allowed_tools:
        cmd.extend(['--allowed-tools', *allowed_tools])
    if disallowed_tools:
        cmd.extend(['--disallowed-tools', *disallowed_tools])

    if mcp_config:
        fd, mcp_config_path = tempfile.mkstemp(suffix='.json', prefix='mcp_')
        with open(fd, 'w') as f:
            json.dump(mcp_config, f)
        temp_files.append(mcp_config_path)
        cmd.extend(['--mcp-config', mcp_config_path])

    if output_schema:
        cmd.extend(['--json-schema', json.dumps(output_schema)])

    # User prompt is piped via stdin to avoid ARG_MAX on large payloads
    stdin_data = prompt.encode()

    # Strip ANTHROPIC_API_KEY so `claude` falls back to OAuth
    env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
    # Merge caller-supplied overrides (e.g. ANTHROPIC_BASE_URL for vLLM)
    if env_overrides:
        env.update(env_overrides)
    # Multi-account failover: inject per-invocation OAuth token
    if oauth_token:
        env['CLAUDE_CODE_OAUTH_TOKEN'] = oauth_token
    # Per-task config dir: credential file + session isolation
    if config_dir:
        env['CLAUDE_CONFIG_DIR'] = str(config_dir)

    # Start a per-invocation vLLM bridge when ANTHROPIC_BASE_URL is set so that
    # Claude CLI talks to the local bridge (which translates vLLM tool_use format)
    # rather than the upstream endpoint directly.
    # NOTE: sentinel must be declared BEFORE the try block so the finally clause
    # has the variable in scope.  Instantiation and start() happen INSIDE the try
    # so that if start() raises mid-init (e.g. AppRunner setup succeeds but
    # TCPSite.start() fails), the finally clause still calls stop() to release
    # any partially-initialised AppRunner resources.
    bridge: VllmBridge | None = None
    try:
        if env_overrides and env_overrides.get('ANTHROPIC_BASE_URL'):
            if VllmBridge is None:
                raise RuntimeError(
                    'ANTHROPIC_BASE_URL is set but aiohttp is not installed; '
                    'install dark-factory-shared with the vllm extras to use VllmBridge.'
                )
            bridge = VllmBridge(upstream_url=env_overrides['ANTHROPIC_BASE_URL'])
            await bridge.start()
            env['ANTHROPIC_BASE_URL'] = bridge.url

        result = await _run_subprocess(cmd, cwd, env, model, timeout_seconds, stdin_data=stdin_data)
        return _parse_claude_output(result)
    finally:
        for path in temp_files:
            Path(path).unlink(missing_ok=True)
        if bridge is not None:
            await bridge.stop()


def _parse_claude_output(result: _SubprocessResult) -> AgentResult:
    """Parse Claude Code JSON output into AgentResult."""
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

    # The CLI may report subtype='success' even when is_error is true (e.g.
    # usage cap hit).  Trust is_error as an authoritative override.
    is_error = data.get('is_error', False)
    is_success = (subtype == 'success' or result.returncode == 0) and not is_error

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
        timed_out=result.timed_out,
    )


async def _run_subprocess(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    model: str,
    timeout_seconds: float | None = None,
    stdin_data: bytes | None = None,
) -> _SubprocessResult:
    """Run a subprocess, log output.

    *stdin_data*, when set, is piped to the process's stdin.  This avoids
    passing large payloads as command-line arguments (which hit ARG_MAX).
    """
    logger.info(f'Invoking claude agent: model={model} cwd={cwd}')
    logger.info(f'Command: {" ".join(cmd[:15])}...')

    start_ms = int(time.monotonic() * 1000)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            # Graceful shutdown: SIGTERM first, then SIGKILL after grace period.
            # SIGTERM lets the Claude CLI flush its final JSON output to stdout
            # (including session_id and token counts) before exiting.
            _SIGTERM_GRACE_SECS = 5
            proc.terminate()  # SIGTERM
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=_SIGTERM_GRACE_SECS,
                )
            except TimeoutError:
                # Still alive after grace period — force kill
                proc.kill()
                await proc.wait()
                stdout_text = ''
                stderr_text = f'Process killed after {timeout_seconds}s timeout (SIGTERM+SIGKILL)'
            else:
                stdout_text = stdout.decode() if stdout else ''
                stderr_text = stderr.decode()[-2000:] if stderr else ''
                if stdout_text:
                    logger.info(
                        f'Agent produced {len(stdout_text)} bytes after SIGTERM '
                        f'(first 500): {stdout_text[:500]}'
                    )
                stderr_text = (
                    f'Process terminated after {timeout_seconds}s timeout (SIGTERM); '
                    + stderr_text
                )
            duration_ms = int(time.monotonic() * 1000) - start_ms
            return _SubprocessResult(
                stdout=stdout_text,
                stderr=stderr_text,
                returncode=proc.returncode if proc.returncode is not None else 1,
                duration_ms=duration_ms,
                timed_out=True,
            )
    except asyncio.CancelledError:
        # Orchestrator shutdown path: the awaiting task was cancelled. The
        # subprocess itself is *not* automatically killed — we must reap it
        # here or its asyncio child-watcher thread keeps the event loop
        # alive after the main coroutine returns. Shield the final wait so
        # a second cancel can't abandon the waitpid mid-flight.
        if proc.returncode is None:
            logger.warning(f'Subprocess cancelled — terminating pid {proc.pid}')
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.shield(asyncio.wait_for(proc.wait(), timeout=5.0))
            except TimeoutError:
                logger.warning(
                    f'pid {proc.pid} did not exit on SIGTERM — killing'
                )
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                try:
                    await asyncio.shield(
                        asyncio.wait_for(proc.wait(), timeout=5.0)
                    )
                except TimeoutError:
                    logger.error(f'pid {proc.pid} unresponsive to SIGKILL')
        raise

    duration_ms = int(time.monotonic() * 1000) - start_ms

    stderr_text = stderr.decode()[-2000:] if stderr else ''
    if stderr_text:
        logger.info(f'Agent stderr (last 1000): {stderr_text[-1000:]}')
    logger.info(f'Agent exit code: {proc.returncode}')
    stdout_text_for_log = stdout.decode()
    if proc.returncode != 0:
        # On failure, dump the full stdout so downstream debugging can see
        # the actual messages array (tool_use blocks, error details) instead
        # of only the truncated result envelope.
        logger.info(
            f'Agent stdout length: {len(stdout)} bytes (full, returncode={proc.returncode}):\n{stdout_text_for_log}'
        )
    else:
        logger.info(f'Agent stdout length: {len(stdout)} bytes, first 500: {stdout_text_for_log[:500]}')

    return _SubprocessResult(
        stdout=stdout.decode(),
        stderr=stderr_text,
        returncode=proc.returncode if proc.returncode is not None else 1,
        duration_ms=duration_ms,
    )
