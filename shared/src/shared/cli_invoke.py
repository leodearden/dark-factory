"""Shared Claude CLI invocation with cap-retry and structured output parsing."""

from __future__ import annotations

import asyncio
import contextlib
import enum
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# VllmBridge depends on aiohttp, which is not installed in every consumer
# environment (e.g. dashboard's venv).  Tolerate ImportError so that callers
# that never set ANTHROPIC_BASE_URL can still import shared.cli_invoke.
from shared.proc_group import snapshot_process_group, terminate_process_group

try:
    from shared.vllm_bridge import VllmBridge as _VllmBridgeRuntime
except ImportError:  # pragma: no cover - exercised only when aiohttp absent
    _VllmBridgeRuntime = None

if TYPE_CHECKING:
    from shared.config_dir import TaskConfigDir
    from shared.cost_store import CostStore
    from shared.usage_gate import UsageGate
    from shared.vllm_bridge import VllmBridge
else:
    # Re-expose the runtime binding under the public name so callers (and test
    # patchers) can refer to ``shared.cli_invoke.VllmBridge``.  This is the
    # Optional form: ``None`` when aiohttp isn't installed.
    VllmBridge = _VllmBridgeRuntime

logger = logging.getLogger(__name__)

_CAP_HIT_COOLDOWN_SECS = 5.0
_MAX_CAP_COOLDOWN_SECS = 300.0
# Poll interval for the two-regime liveness watchdog in _run_subprocess.
# Each tick reads the on-disk transcript to check for assistant turns; the
# actual sleep per tick is min(_WATCHDOG_POLL_SECS, time_to_grace, time_to_ceiling)
# so grace and ceiling are never overshot by a full poll interval.
_WATCHDOG_POLL_SECS = 5.0
# Minimum poll duration — prevents the poll from degenerating to 0.0 when both
# time_to_grace and time_to_ceiling have already elapsed (would otherwise cause an
# asyncio.wait(timeout=0) tight-spin hammering count_transcript_turns).
_WATCHDOG_MIN_POLL_SECS = 0.01
# Per-caller cap-wait policy (post-1365 audit, task 1401)
# ─────────────────────────────────────────────────────────────────────────────
# _DEFAULT_CAP_WAIT_SANITY_SECS (14 days) is inherited by callers that do NOT
# pass an explicit cap_wait_sanity_secs= override.  Each call site below has
# been audited; the policy for each is documented here so future readers know
# why an override is or is not present.
#
# Caller                                  Policy / WHY
# ───────────────────────────────────────────────────────────────────────────
# orchestrator/workflow.py                14-day default OK.  Per-task AFK
#   (implementer/debugger invocation)     implementer/debugger; 14-day patient
#                                         wait is the documented AFK A1 intent.
#
# orchestrator/steward.py                 14-day default OK.  Pre-triage; the
#   (pre-triage invocation)               AllAccountsCappedException handler
#                                         logs and falls back to inline triage.
#
# orchestrator/review_checkpoint.py       14-day default OK.  Deep reviewer;
#   (deep reviewer invocation)            AllAccountsCappedException handler
#                                         returns an empty report, no queue
#                                         stall.
#
# orchestrator/harness.py                 14-day default OK for both sites.
#   (module tagging + watcher rotation)   AllAccountsCappedException handlers
#                                         log and return / supervisor-driven
#                                         restart; neither blocks a queue.
#
# fused_memory/middleware/task_curator.py _CURATOR_CAP_WAIT_SANITY_SECS=120s.
#   (LLM triage calls)                    Best-effort middleware, fast-fail/
#                                         defer contract; 120 s is intentional.
#
# fused_memory/reconciliation             _RECONCILIATION_STAGE_CAP_WAIT_SANITY_SECS = 1800 s (30 min).
#   (agent_loop, judge, cli_stage_runner) Shared by all three stage runners.
#   (reconciliation stage runners)        Short-lived stage runners; expected
#                                         to complete promptly within the
#                                         reconciliation cycle.  14-day default
#                                         would stall the queue indefinitely
#                                         under sustained cap.  1800 s lets a
#                                         brief cap window resolve in-band.
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_CAP_WAIT_SANITY_SECS = 14 * 86400  # 14 days: outer sanity bound for patient cap waits
_CAP_WAIT_LOG_INTERVAL_SECS = 600.0        # emit at most one cap_wait log per ~10 min
CAP_HIT_RESUME_PROMPT = (
    'Your previous run was interrupted by a usage limit. '
    'Continue where you left off and complete your task.'
)
# Prompt sent to a session that the orchestrator is resuming after a crash.
# Kept separate from CAP_HIT_RESUME_PROMPT because the cause differs
# (orchestrator restart, not a usage-cap interrupt) and the agent message
# should stay a plain "continue" rather than mentioning usage limits.
CRASH_RECOVERY_RESUME_PROMPT = 'continue'

# Concrete CLI/usage errors that exit instantly with no output but are NOT usage
# caps. Matched case-insensitively against stderr/output so the zero-cost cap
# heuristic doesn't misfire (and loop forever) on a local CLI failure.
NON_CAP_CLI_ERROR_MARKERS = [
    'is already in use',        # --session-id collision (reify-3604)
    'unrecognized arguments',
    'unknown option',
    'invalid value',
    'no such file or directory',
    'permission denied',
]


def _is_non_cap_cli_error(stderr: str, output: str) -> bool:
    blob = f'{stderr or ""}\n{output or ""}'.lower()
    return any(m in blob for m in NON_CAP_CLI_ERROR_MARKERS)


__all__ = [
    'CAP_HIT_RESUME_PROMPT',
    'CRASH_RECOVERY_RESUME_PROMPT',
    'AgentFailureClass',
    'AgentFailureKind',
    'AgentResult',
    'AllAccountsCappedException',
    'build_failure_message',
    'classify_agent_failure',
    'count_transcript_turns',
    'invoke_claude_agent',
    'invoke_with_cap_retry',
    'is_timed_out_with_progress',
    'is_zero_output_timeout',
    'read_transcript_records',
]


class AllAccountsCappedException(Exception):
    """Raised when the cap-hit retry loop exceeds the sanity-bound wall-clock deadline.

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


# ─────────────────────────────────────────────────────────────────────────────
# StructuredOutput schema-tool deny-list (CLI 2.1.168 regression guard)
# ─────────────────────────────────────────────────────────────────────────────
# CLI 2.1.168 delivers ``--json-schema`` structured output through a *synthetic
# tool* named ``StructuredOutput``.  A ``disallowed_tools=['*']`` wildcard — used
# by pure-classifier callers (curator single/batch, recon agent_loop) that want
# no real tool access — now ALSO matches and denies that schema tool, so every
# structured answer is permission-denied → ``error_max_structured_output_retries``
# (or ``error_max_budget_usd``) with no salvageable payload.
#
# Fix (central, in ``_invoke_claude``): when an ``output_schema`` is requested AND
# ``'*'`` is in ``disallowed_tools``, expand the ``'*'`` into this explicit
# deny-list of real built-in tools — which deliberately OMITS ``StructuredOutput``
# — preserving the "no real (file/bash/web/MCP) tool access" guarantee while
# letting the schema tool through.  Whitelisting ``StructuredOutput`` while keeping
# ``'*'`` does NOT work: deny precedence beats allow, so the wildcard must be
# removed entirely (confirmed against live CLI 2.1.168).
#
# KEEP IN SYNC with the CLI's built-in tool names: a *future new* built-in tool
# would not be auto-denied by this list.  Accepted because (a) these prompts forbid
# tool use, (b) no ``mcp_config`` is wired for these callers (MCP tools absent), and
# (c) a future change to the CLI's tool-exclusion semantics is caught loudly by the
# ``schema_tool_denied`` detection below rather than degrading silently.
_SCHEMA_OUTPUT_TOOL = 'StructuredOutput'
_REAL_BUILTIN_TOOLS_DENYLIST = [
    'Bash',
    'BashOutput',
    'KillShell',
    'KillBash',
    'Read',
    'Edit',
    'Write',
    'MultiEdit',
    'NotebookEdit',
    'Glob',
    'Grep',
    'Task',
    'Agent',
    'WebFetch',
    'WebSearch',
    'TodoWrite',
    'ExitPlanMode',
    'SlashCommand',
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
    - ``timed_out``: True when the subprocess was killed by a wall-clock timeout
    - ``schema_salvaged``: True when the CLI reported is_error=True but a valid
      ``structured_output`` was present — commonly ``error_max_turns`` paired
      with a completed JSON schema tool-use turn. Callers treat this as success.
    - ``schema_tool_denied``: True when the CLI reported is_error=True with NO
      structured payload AND a ``StructuredOutput`` permission denial — i.e. the
      schema tool itself was blocked.  This is a systemic config break (the
      cli_invoke deny-list no longer permits the schema tool), NOT a flaky
      candidate.  ``success`` stays False (NOT salvaged); callers should raise a
      loud, un-suppressed escalation so the deny-list gets fixed.
    - ``proc_tree``: human-readable snapshot of the subprocess process group
      captured by ``snapshot_process_group(pgid)`` at the top of the
      ``TimeoutError`` handler in ``_run_subprocess`` — i.e. while the wedged
      children are still alive and their ``/proc`` entries are readable.
      Empty string when the invocation did not time out.  Persisted to
      ``.task/zero_output_evidence-iter{N}.json`` by the workflow's
      ``_capture_zero_output_evidence`` helper (task 1739).
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
    schema_salvaged: bool = False
    schema_tool_denied: bool = False
    api_error_status: int | None = None
    proc_tree: str = ''
    transcript_turns: int | None = None
    """Number of assistant turns found in the on-disk JSONL transcript, or None
    when the transcript was not read (non-timeout paths) or could not be located.
    Stamped only on the SIGTERM/SIGKILL timeout path via count_transcript_turns."""


def _resolve_transcript_path(config_dir: Path, session_id: str) -> Path | None:
    """Locate the transcript file for *session_id* under *config_dir*.

    Globs ``<config_dir>/projects/*/<session_id>.jsonl`` and returns the first
    match, or None if nothing is found or an error occurs.  Uses session_id
    (a unique UUID) as the glob anchor — version-robust per PRD decision #2.
    """
    try:
        matches = list(config_dir.glob(f'projects/*/{session_id}.jsonl'))
        return matches[0] if matches else None
    except Exception:
        logger.debug(
            f'_resolve_transcript_path: failed to glob for session {session_id} under {config_dir}'
        )
        return None


def read_transcript_records(
    config_dir: Path,
    session_id: str,
) -> list[dict] | None:
    """Read and return all parsed records from the transcript for *session_id*.

    Locates ``<config_dir>/projects/*/<session_id>.jsonl`` via glob and parses
    each line as JSON, returning a list of dicts in order.  Parsing is TOLERANT:
    unparseable lines (e.g. a truncated final line left by SIGKILL) are silently
    skipped.

    Returns:
    - ``list[dict]`` — all successfully-parsed records (may be empty).
    - ``None`` — transcript file could not be located, or the whole read raised
      a catastrophic error.  Never raises; logs at debug/warning on failure.
    """
    try:
        path = _resolve_transcript_path(config_dir, session_id)
        if path is None:
            return None
        records: list[dict] = []
        with path.open(encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        records.append(record)
                except json.JSONDecodeError:
                    logger.debug(
                        f'read_transcript_records: skipping unparseable line in {path}'
                    )
        return records
    except Exception:
        logger.warning(
            f'read_transcript_records: failed to read transcript for session {session_id}'
        )
        return None


def count_transcript_turns(
    config_dir: Path,
    session_id: str,
) -> int | None:
    """Count assistant turns in the on-disk JSONL transcript for *session_id*.

    Delegates to ``read_transcript_records`` and counts records with
    ``type == "assistant"``.  Inherits the same TOLERANT parsing semantics
    (truncated lines skipped, None on file-not-found or catastrophic error).

    Returns:
    - ``int`` — number of assistant records found (may be 0).
    - ``None`` — transcript file could not be located, or the whole read raised
      a catastrophic error.  Never raises; logs at debug/warning on failure.
    """
    records = read_transcript_records(config_dir, session_id)
    if records is None:
        return None
    return sum(1 for r in records if r.get('type') == 'assistant')


def is_zero_output_timeout(result: AgentResult) -> bool:
    """Return True when *result* is a fresh-invocation zero-output CLI wedge.

    Classification is transcript-authoritative when ``transcript_turns`` is
    available (i.e. not None):

    - ``transcript_turns == 0`` → True  (no real work; genuine pre-turn wedge)
    - ``transcript_turns > 0``  → False (work was done; reify-4415 case)

    When ``transcript_turns is None`` (transcript not read or not available),
    falls back to the legacy heuristic: ``turns == 0 and cost_usd == 0.0``.
    An unreadable transcript NEVER upgrades a wedge to "progress" — the None
    case always degrades to today's conservative behavior (PRD decision #3).

    The predicate is the single canonical definition shared by:

    - The RESUME-variant wedge guard in ``invoke_with_cap_retry`` (~line 644,
      task 1532): clears the wedged ``resume_session_id`` so the cap-retry
      loop does not re-resume an orphaned provider session.

    - The FRESH-invocation circuit breaker in ``workflow._execute_iterations``
      (task 1739): fast-fails to BLOCKED after
      ``config.max_consecutive_zero_output_timeouts`` consecutive such results
      instead of burning the full ``max_execute_iterations`` budget (~3.3h).

    Root causes:
    - reify-4429 (2026-06-11): 10/10 implementer iterations hung pre-first-turn
      with no recoverable session state → zero_output_timeout True (correct).
    - reify-4415: 43 assistant turns over 1198s, killed mid-work → zero turns
      and cost in JSON output → previously mis-classified as wedge (now fixed
      when transcript_turns is stamped by _run_subprocess).
    """
    if not result.timed_out:
        return False
    if result.transcript_turns is not None:
        return result.transcript_turns == 0
    # Legacy fallback: transcript not available — use JSON-output fields.
    # Log so that 'transcript not located for a possibly-productive run' is
    # diagnosable rather than silently degrading to the conservative wedge path.
    logger.debug(
        'is_zero_output_timeout: timed_out=True but transcript_turns=None '
        '(transcript could not be located for session %r) — '
        'falling back to legacy turns==0 and cost_usd==0.0 heuristic',
        result.session_id,
    )
    return result.turns == 0 and result.cost_usd == 0.0


def is_timed_out_with_progress(result: AgentResult) -> bool:
    """Return True when *result* is a timed-out run that did real agentic work.

    Specifically: ``timed_out=True`` and ``transcript_turns > 0``.  This is the
    complement of ``is_zero_output_timeout`` when ``transcript_turns is not
    None``.

    Mutual-exclusivity invariant: when ``result.timed_out`` is True and
    ``result.transcript_turns is not None``, exactly one of
    {``is_zero_output_timeout``, ``is_timed_out_with_progress``} returns True.

    Callers:
    - ``steward._is_empty_output``: guards against misclassifying a productive
      SIGTERM-killed run (subtype=error_empty_output) as "no real work done".
    - ``workflow._capture_zero_output_evidence``: enriches the evidence JSON.
    - task γ: decides whether to resume a killed productive run.
    """
    return result.timed_out and (result.transcript_turns or 0) > 0


class AgentFailureKind(enum.StrEnum):
    """Classification of an AgentResult.  SUCCESS is the non-failure case."""

    SUCCESS = 'success'
    MAX_TURNS = 'max_turns'
    EMPTY_OUTPUT = 'empty_output'
    API_ERROR = 'api_error'
    TIMED_OUT = 'timed_out'
    STRUCTURAL = 'structural'
    UNKNOWN = 'unknown'


@dataclass
class AgentFailureClass:
    """Operator-facing classification of a failed (or succeeded-via-salvage) agent run.

    - ``kind``: machine-readable classification; drives retry/escalation policy.
    - ``summary``: one-line description suitable for escalation summaries.
    - ``diagnostic_detail``: multi-line dump of every diagnostic signal
      available on the underlying ``AgentResult``, for escalation detail
      fields.  Never silently drops signals, even when ``kind == UNKNOWN``.
    """

    kind: AgentFailureKind
    summary: str
    diagnostic_detail: str


def classify_agent_failure(result: AgentResult) -> AgentFailureClass:
    """Classify an ``AgentResult`` for the steward / escalation layer.

    The decision rules fire in order — the first match wins:

    1. ``result.success`` → ``SUCCESS``.
    2. ``result.timed_out`` → ``TIMED_OUT``.
    3. ``result.subtype == 'error_max_turns'`` → ``MAX_TURNS``
       (high ``turns`` + non-zero ``output_tokens`` but empty ``output``).
    4. ``result.api_error_status`` set → ``API_ERROR`` (includes status code
       in the summary; transient — worth retrying against another account).
    5. ``result.subtype == 'error_empty_output'`` → ``EMPTY_OUTPUT``
       (may be transient).
    6. ``result.schema_salvaged`` → ``STRUCTURAL`` (schema-salvage: the
       subtype looked like an error but a valid structured output was
       recovered; callers usually treat as success).
    7. otherwise → ``UNKNOWN``.

    ``diagnostic_detail`` always includes: subtype, turns, cost_usd,
    duration_ms, timed_out, api_error_status, output length, last 500 chars
    of stdout output, and last 500 chars of stderr.
    """
    tail_out = result.output[-500:] if result.output else ''
    tail_err = result.stderr[-500:] if result.stderr else ''
    diagnostic_detail = (
        f'subtype={result.subtype!r}\n'
        f'turns={result.turns}\n'
        f'cost_usd={result.cost_usd}\n'
        f'duration_ms={result.duration_ms}\n'
        f'timed_out={result.timed_out}\n'
        f'api_error_status={result.api_error_status}\n'
        f'len(output)={len(result.output)}\n'
        f'output (last 500 chars):\n{tail_out}\n'
        f'stderr (last 500 chars):\n{tail_err}'
    )

    if result.success:
        return AgentFailureClass(
            kind=AgentFailureKind.SUCCESS,
            summary='agent succeeded',
            diagnostic_detail=diagnostic_detail,
        )
    if result.timed_out:
        return AgentFailureClass(
            kind=AgentFailureKind.TIMED_OUT,
            summary=(
                f'agent timed out after {result.duration_ms}ms '
                f'({result.turns} turns)'
            ),
            diagnostic_detail=diagnostic_detail,
        )
    if result.subtype == 'error_max_turns':
        return AgentFailureClass(
            kind=AgentFailureKind.MAX_TURNS,
            summary=(
                f'agent hit max_turns ({result.turns} turns, '
                f'output_tokens={result.output_tokens})'
            ),
            diagnostic_detail=diagnostic_detail,
        )
    if result.api_error_status is not None:
        return AgentFailureClass(
            kind=AgentFailureKind.API_ERROR,
            summary=f'agent API error: HTTP {result.api_error_status}',
            diagnostic_detail=diagnostic_detail,
        )
    if result.subtype == 'error_empty_output':
        return AgentFailureClass(
            kind=AgentFailureKind.EMPTY_OUTPUT,
            summary='agent returned empty output',
            diagnostic_detail=diagnostic_detail,
        )
    if result.schema_salvaged:
        return AgentFailureClass(
            kind=AgentFailureKind.STRUCTURAL,
            summary='agent succeeded via schema salvage',
            diagnostic_detail=diagnostic_detail,
        )
    return AgentFailureClass(
        kind=AgentFailureKind.UNKNOWN,
        summary=(
            f'agent failed: subtype={result.subtype!r} '
            f'(no specific failure signal)'
        ),
        diagnostic_detail=diagnostic_detail,
    )


def build_failure_message(label: str, result: AgentResult) -> str:
    """Format the canonical '{label} failed: {summary}\\n{diagnostic_detail}' message.

    Thin wrapper around classify_agent_failure that pins the message format
    used by RuntimeError-raising call sites in the reconciliation loop and
    the CLI judge, so the format evolves in one place as
    AgentFailureClass.diagnostic_detail grows.
    """
    cls = classify_agent_failure(result)
    return f'{label} failed: {cls.summary}\n{cls.diagnostic_detail}'


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


def _reset_for_fresh_retry(invoke_kwargs: dict[str, Any], original_prompt: str) -> None:
    """Switch the retry loop to a fresh (non-resume) invocation.

    Drops the resume session, restores the real task prompt, and regenerates a
    pre-allocated session_id *when one was set* — the prior failed attempt may have
    already committed that UUID to disk, and reusing it on a fresh `--session-id`
    makes the CLI exit instantly with 'Session ID … is already in use'
    (the 2026-05-26 reify-3604 wedge). Callers that pass no session_id keep none.
    """
    invoke_kwargs.pop('resume_session_id', None)
    invoke_kwargs['prompt'] = original_prompt
    if invoke_kwargs.get('session_id'):
        invoke_kwargs['session_id'] = str(uuid.uuid4())


@dataclass
class _SubprocessResult:
    stdout: str
    stderr: str
    returncode: int
    duration_ms: int
    timed_out: bool = False
    proc_tree: str = ''
    transcript_turns: int | None = None


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
    session_id: str | None = None,
    config_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    startup_grace_secs: float = 120.0,
) -> AgentResult:
    """Invoke Claude Code CLI and return structured result.

    *oauth_token*, when set, overrides the Claude CLI's default credentials
    via the ``CLAUDE_CODE_OAUTH_TOKEN`` env var (multi-account failover).

    *resume_session_id*, when set, resumes an existing session via
    ``--resume <id>`` instead of starting a new one.  The system prompt is
    skipped on resume (it was already set in the initial session).

    *session_id*, when set and *resume_session_id* is not, pre-allocates the
    session UUID via ``--session-id <id>`` so callers can resume the same
    session later (orchestrator crash-recovery sidecar).  Mutually exclusive
    with *resume_session_id* — when both are set, *resume_session_id* wins.

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
        resume_session_id=resume_session_id, session_id=session_id,
        config_dir=config_dir,
        env_overrides=env_overrides,
        startup_grace_secs=startup_grace_secs,
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
    cap_wait_sanity_secs: float | None = _DEFAULT_CAP_WAIT_SANITY_SECS,
    invoke_fn: Callable[..., Awaitable[AgentResult]] | None = None,
    backend: str = 'claude',
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

    *cap_wait_sanity_secs* is the outer wall-clock bound for cap-hit patience.
    When total elapsed time since the first cap hit exceeds this value,
    ``AllAccountsCappedException`` is raised so the caller can escalate.
    Defaults to 14 days (``_DEFAULT_CAP_WAIT_SANITY_SECS``).  Pass ``None``
    to wait indefinitely.

    *label* identifies the caller in log messages (e.g. "Module tagging",
    "Task 7 [implementer]").

    When *cost_store* is provided, successful invocations are recorded via
    ``save_invocation()`` and cap-hit events via ``save_account_event()``.

    All keyword arguments are forwarded to ``invoke_claude_agent()``.
    """
    model = invoke_kwargs.get('model', 'opus')
    original_prompt = invoke_kwargs.get('prompt', '')
    # Caller-initiated resume contract: when the caller pre-sets resume_session_id
    # (e.g. orchestrator crash recovery), they pass the real task prompt as `prompt`
    # so that `original_prompt` captures it for fresh-fallback restoration.  We
    # immediately overwrite `invoke_kwargs['prompt']` with CRASH_RECOVERY_RESUME_PROMPT
    # so the first subprocess invocation uses the short continuation string.  The
    # existing non-cap-hit resume-failure branch (below) then correctly restores
    # `original_prompt` (the real task prompt) for any subsequent fresh invocation.
    if invoke_kwargs.get('resume_session_id'):
        if not original_prompt:
            raise TypeError(
                "invoke_with_cap_retry: 'prompt' must be a non-empty string when "
                "'resume_session_id' is set.  The prompt is the real task context used "
                "for fresh-fallback recovery if the resume invocation fails; passing an "
                "empty or missing prompt silently corrupts that fallback."
            )
        invoke_kwargs['prompt'] = CRASH_RECOVERY_RESUME_PROMPT
    consecutive_cap_hits = 0
    num_accounts = max(usage_gate.account_count, 1) if usage_gate else 1
    retry_start = time.monotonic()
    last_cap_wait_log_at: float | None = None

    def _check_cap_wait(now: float, elapsed: float, cooldown: float, hits: int) -> None:
        """Guard and throttled log for cap-wait iterations.

        Raises AllAccountsCappedException when the 14-day sanity bound is exceeded.
        Emits a structured 'cap_wait' JSON log at most once per _CAP_WAIT_LOG_INTERVAL_SECS.
        Closes over: cap_wait_sanity_secs, label, num_accounts, usage_gate,
        last_cap_wait_log_at (nonlocal write).
        """
        nonlocal last_cap_wait_log_at
        if cap_wait_sanity_secs is not None and elapsed > cap_wait_sanity_secs:
            logger.error(
                f'{label}: cap-wait sanity bound exceeded after {elapsed:.1f}s '
                f'({hits} retries, {num_accounts} account(s))',
            )
            raise AllAccountsCappedException(
                retries=hits,
                elapsed_secs=elapsed,
                label=label,
            )
        if last_cap_wait_log_at is None or now - last_cap_wait_log_at >= _CAP_WAIT_LOG_INTERVAL_SECS:
            logger.warning(json.dumps({
                'event': 'cap_wait',
                'label': label,
                'elapsed_s': round(elapsed, 1),
                'soonest_open_at': (
                    usage_gate.soonest_resets_at.isoformat()
                    if usage_gate and usage_gate.soonest_resets_at else None
                ),
                'next_probe_in_s': round(cooldown, 1),
            }, default=str))
            last_cap_wait_log_at = now

    account_name = ''
    unattributed_cap = False  # True when heuristic fires but token is unresolvable;
    # controls: (1) skip confirm, (2) mark capped=True in cost_store
    started_at = ''
    completed_at = ''

    # Default to Claude-specific invocation when no invoke_fn was provided
    invoke: Callable[..., Awaitable[AgentResult]] = invoke_fn or invoke_claude_agent

    # Fast path: no usage gate → single invocation, no cap retry.
    # NOTE: if `resume_session_id` was set by the caller (crash-recovery path),
    # this fast path will attempt the resume but cannot fall back to a fresh
    # invocation on failure — the non-cap-hit resume→fresh-fallback branch in
    # the while-loop below only runs when usage_gate is provided.  In practice
    # the orchestrator always supplies a gate, but callers without one should be
    # aware that a failed resume returns the failure result directly.
    if not usage_gate:
        started_at = datetime.now(UTC).isoformat()
        result = await invoke(
            **invoke_kwargs,
            config_dir=config_dir.path if config_dir else None,
        )
        completed_at = datetime.now(UTC).isoformat()
    else:
        while True:
            async with usage_gate.invoke_slot() as slot:
                account_name = slot.account_name

                if config_dir and slot.token:
                    config_dir.write_credentials(slot.token)

                started_at = datetime.now(UTC).isoformat()
                logger.info(f'{label}: dispatching on account {account_name!r}')
                result = await invoke(
                    **invoke_kwargs,
                    oauth_token=slot.token,
                    config_dir=config_dir.path if config_dir else None,
                )
                completed_at = datetime.now(UTC).isoformat()

                # Auth-failure routing (401/403): distinct from cap hits.
                # Mark the account auth_failed and fail over; don't count
                # toward consecutive_cap_hits so the cooldown doesn't compound.
                # The narrow {401, 403} is load-bearing: 429 carries a real
                # cap-message body ("You're out of extra usage · resets ...")
                # that the cap-hit detector below already recognises, but if
                # we route 429 here the slot.detect_cap_hit() call never runs,
                # AllAccountsCappedException never fires, and the curator
                # worker's cap-defer machinery silently never engages.
                if result.api_error_status in (401, 403):
                    auth_marked = usage_gate._handle_auth_failure(
                        f'HTTP {result.api_error_status}: {result.output[:120]}',
                        slot.token,
                    )
                    if auth_marked:
                        slot.settle()
                        _reset_for_fresh_retry(invoke_kwargs, original_prompt)
                        logger.warning(
                            f'{label}: account {account_name} auth-failed '
                            f'(HTTP {result.api_error_status}) — failing over',
                        )
                        continue
                    # Couldn't attribute — fall through; downstream heuristic
                    # or return handles it.

                # Wedge guard: a full-timeout CLI call (timed_out=True with zero
                # turns and zero cost) means the subprocess never executed any
                # agentic work.  Its provider-side session is orphaned; re-resuming
                # it just perpetuates the same hang (observed:
                # esc-task-curator-3/-5/-6, session
                # c5d446f5-6339-4291-81d6-1d26b5e2f199, 2026-05-27).  Without this
                # guard, the cap-hit branch below would re-set
                # invoke_kwargs['resume_session_id'] = result.session_id from the
                # wedge's session_id, perpetuating the wedge across every retry.
                # Analogous to the fused_memory/reconciliation/agent_loop.py:368
                # "clear stale session id" defence, applied at the cap-retry layer.
                #
                # NOTE — cap-as-wedge edge case: if a genuine rate-cap manifests as
                # a full-timeout zero-output result (no stderr cap-pattern, but the
                # account is capped), this branch fires before detect_cap_hit and
                # retries fresh without incrementing consecutive_cap_hits or applying
                # exponential cooldown.  This is intentional: the fresh retry against
                # the still-capped account will produce a fast (sub-5 s) zero-cost
                # response whose cap-pattern IS detectable, so cap accounting resumes
                # on the very next iteration.  At most one extra full-timeout
                # (~configured_timeout_ms) is incurred before the cap is re-detected.
                if (
                    is_zero_output_timeout(result)
                    and invoke_kwargs.get('resume_session_id')
                ):
                    logger.warning(
                        f'{label}: zero-output timed-out invocation '
                        f'(duration_ms={result.duration_ms}) — clearing wedged '
                        f'resume_session_id={invoke_kwargs["resume_session_id"]} '
                        f'before retry',
                    )
                    _reset_for_fresh_retry(invoke_kwargs, original_prompt)
                    continue  # __aexit__ releases probe slot

                if slot.detect_cap_hit(result.stderr, result.output, backend=backend):
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
                        _reset_for_fresh_retry(invoke_kwargs, original_prompt)
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

                    # Guard + periodic log: raise on 14-day sanity bound, emit throttled JSON.
                    now = time.monotonic()
                    elapsed = now - retry_start
                    _check_cap_wait(now, elapsed, cooldown, consecutive_cap_hits)

                    await asyncio.sleep(cooldown)
                    continue

                # Heuristic safety net: a zero-cost, near-instant, ≤1-turn result
                # that wasn't caught by pattern matching is almost certainly a cap
                # hit with an unrecognised message format.  Treat it as a cap hit so
                # the retry loop can wait / fail over instead of silently returning a
                # useless "success" to the caller.
                if (
                    not result.success  # is_error=true → success=False after fix 2
                    and result.cost_usd == 0
                    and result.turns <= 1
                    and result.duration_ms < 5000
                ):
                    if _is_non_cap_cli_error(result.stderr, result.output):
                        # A recognised local CLI/usage error (e.g. --session-id
                        # collision) exits zero-cost and instantly, but it is NOT a
                        # usage cap.  Counting it as a cap loops forever (reify-3604).
                        # Fall through: Branch C retries fresh when resuming, else the
                        # failed result is returned for normal verify/steward handling.
                        logger.warning(
                            f'{label}: zero-cost instant exit is a CLI error, not a cap '
                            f'(stderr={result.stderr[:160]!r}) — not counting as cap hit',
                        )
                    else:
                        logger.warning(
                            f'{label}: suspicious zero-cost instant exit (turns={result.turns}, '
                            f'duration={result.duration_ms}ms) — treating as cap hit. '
                            f'Output: {result.output[:200]!r}',
                        )
                        cap_marked = usage_gate._handle_cap_detected(
                            f'Heuristic cap: zero-cost instant exit — {result.output[:120]}',
                            None,
                            slot.token,
                        )
                        if not cap_marked:
                            logger.warning(
                                f'{label}: heuristic cap suspected but no account could be marked '
                                f'(token unresolved) — treating as normal failure',
                            )
                            unattributed_cap = True
                        else:
                            slot.settle()  # _handle_cap_detected already cleared probe_in_flight
                            consecutive_cap_hits += 1
                            full_cycles = (consecutive_cap_hits - 1) // num_accounts
                            cooldown = min(
                                _CAP_HIT_COOLDOWN_SECS * (2 ** full_cycles),
                                _MAX_CAP_COOLDOWN_SECS,
                            )
                            # Cannot resume a session that never ran
                            _reset_for_fresh_retry(invoke_kwargs, original_prompt)
                            acct_name = usage_gate.active_account_name
                            logger.warning(
                                f'{label}: sleeping {cooldown:.0f}s then retrying fresh on {acct_name or "next account"}',
                            )

                            # Guard + periodic log: raise on 14-day sanity bound, emit throttled JSON.
                            now = time.monotonic()
                            elapsed = now - retry_start
                            _check_cap_wait(now, elapsed, cooldown, consecutive_cap_hits)

                            await asyncio.sleep(cooldown)
                            continue

                # Non-cap-hit failure while resuming → fall back to fresh invocation
                if not result.success and invoke_kwargs.get('resume_session_id'):
                    logger.warning(
                        f'{label}: resume failed (session_id={invoke_kwargs["resume_session_id"]}), '
                        f'retrying fresh',
                    )
                    _reset_for_fresh_retry(invoke_kwargs, original_prompt)
                    continue  # __aexit__ releases probe slot

                if not unattributed_cap:
                    slot.confirm(result.cost_usd)
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
                capped=unattributed_cap,
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
    session_id: str | None = None,
    config_dir: Path | None = None,
    env_overrides: dict[str, str] | None = None,
    startup_grace_secs: float = 120.0,
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
        # Pre-allocate the session UUID so future --resume can find it.
        # --session-id and --resume are mutually exclusive at the CLI level.
        if session_id:
            cmd.extend(['--session-id', session_id])

    cmd.extend(['--permission-mode', permission_mode])
    cmd.extend(['--max-turns', str(max_turns)])

    if effort:
        cmd.extend(['--effort', effort])

    if allowed_tools:
        cmd.extend(['--allowed-tools', *allowed_tools])
    if disallowed_tools:
        # CLI 2.1.168: ``--json-schema`` is delivered via a synthetic
        # ``StructuredOutput`` tool that a ``'*'`` deny wildcard would block,
        # failing every structured-output call.  When a schema IS requested,
        # expand the wildcard into an explicit real-builtins deny-list that omits
        # ``StructuredOutput`` — keeping "no real tool access" while letting the
        # schema tool through.  Callers without an output_schema (e.g. judge.py)
        # keep ``'*'`` verbatim, so all tools stay blocked.  See the deny-list
        # constant above for the keep-in-sync caveat.
        if output_schema and '*' in disallowed_tools:
            disallowed_tools = [
                t for t in disallowed_tools if t != '*'
            ] + _REAL_BUILTIN_TOOLS_DENYLIST
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
            BridgeCls = VllmBridge
            if BridgeCls is None:
                raise RuntimeError(
                    'ANTHROPIC_BASE_URL is set but aiohttp is not installed; '
                    'install dark-factory-shared with the vllm extras to use VllmBridge.'
                )
            bridge = BridgeCls(upstream_url=env_overrides['ANTHROPIC_BASE_URL'])
            await bridge.start()
            env['ANTHROPIC_BASE_URL'] = bridge.url

        result = await _run_subprocess(
            cmd, cwd, env, model, timeout_seconds, stdin_data=stdin_data,
            session_id=(resume_session_id or session_id),
            config_dir=config_dir,
            startup_grace_secs=startup_grace_secs,
        )
        return _parse_claude_output(result)
    finally:
        for path in temp_files:
            Path(path).unlink(missing_ok=True)
        if bridge is not None:
            await bridge.stop()


def _parse_claude_output(result: _SubprocessResult) -> AgentResult:
    """Parse Claude Code JSON output into AgentResult.

    timed_out and transcript_turns are propagated directly from result on every
    return path.
    """
    if not result.stdout.strip():
        return AgentResult(
            success=False,
            output='Agent produced no output',
            subtype='error_empty_output',
            stderr=result.stderr,
            timed_out=result.timed_out,
            duration_ms=result.duration_ms,
            proc_tree=result.proc_tree,
            transcript_turns=result.transcript_turns,
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
            proc_tree=result.proc_tree,
            transcript_turns=result.transcript_turns,
        )

    cost = data.get('cost_usd', data.get('total_cost_usd', 0.0))
    duration = data.get('duration_ms', 0)
    turns = data.get('num_turns', 0)
    session_id = data.get('session_id', '')
    subtype = data.get('subtype', '')
    structured = data.get('structured_output')
    api_error_status = data.get('api_error_status')

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

    # Schema salvage: when the CLI reports is_error=True but the schema tool
    # already produced a valid structured payload (common with error_max_turns
    # + --json-schema), trust the payload and report success. The raw error
    # text stays in ``output`` for diagnostics.
    schema_salvaged = False
    if is_error and isinstance(structured, dict):
        is_success = True
        schema_salvaged = True

    # Schema-tool-denied detection (CLI 2.1.168 regression guard): is_error with
    # NO structured payload AND a ``StructuredOutput`` permission denial means the
    # synthetic schema tool itself was blocked — a systemic config break, not a
    # flaky candidate.  We deliberately do NOT salvage to success: ``success``
    # stays False and ``schema_tool_denied`` is flagged so callers raise a loud,
    # un-suppressed escalation to get the cli_invoke deny-list fixed.  Priority is
    # "get it fixed"; silent recovery is exactly the trap that hid the outage.
    schema_tool_denied = False
    if not is_success and not isinstance(structured, dict):
        denials = data.get('permission_denials')
        if isinstance(denials, list) and any(
            isinstance(d, dict) and d.get('tool_name') == _SCHEMA_OUTPUT_TOOL
            for d in denials
        ):
            schema_tool_denied = True

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
        schema_salvaged=schema_salvaged,
        schema_tool_denied=schema_tool_denied,
        api_error_status=api_error_status,
        proc_tree=result.proc_tree,
        transcript_turns=result.transcript_turns,
    )


def _cpu_govern_prefix(env: dict[str, str]) -> list[str]:
    """Return a cgroup-governance prefix list if DF_AGENT_CPU_GOVERN is set to an executable.

    Pops ``DF_AGENT_CPU_GOVERN`` from *env* (mutates the dict in place) so the
    variable does not leak into the child process environment and a nested
    invocation cannot double-wrap in a second cgroup scope.

    Returns ``[<path>, '--role', 'task', '--']`` when the popped value is a
    non-empty string AND ``shutil.which(<value>)`` confirms it is executable,
    else ``[]``.

    The value is the absolute path to reify's ``cpu-governed-exec.sh`` script.
    ``'--role' 'task' '--'`` is the dark-factory-side CLI contract for
    agent-launch invocations (merge-verify uses ``--role merge`` via a
    separate DF-3 path in ``verify.py``).

    Fail-safe: absent key, empty string, or non-executable/missing path all
    return ``[]`` so no spawn ever fails due to a bad or missing govern script.

    ``cpu-governed-exec.sh`` execs in place on both its governed
    (``exec systemd-run --user --scope``) and fail-open (``exec nice/exec cmd``)
    paths, so ``start_new_session=True`` / ``pgid=proc.pid`` and the
    process-group kill logic in ``_run_subprocess`` are unaffected.  Cargo and
    rustc children inherit the cgroup scope via fork, which is the intended
    effect for DF-1.
    """
    raw = env.pop('DF_AGENT_CPU_GOVERN', None)
    if not raw:
        return []
    if not shutil.which(raw):
        return []
    return [raw, '--role', 'task', '--']


def _cpu_priority_prefix(env: dict[str, str]) -> list[str]:
    """Return a ``nice`` prefix list if DF_AGENT_CPU_NICE is set to a valid value.

    Pops ``DF_AGENT_CPU_NICE`` from *env* (mutates the dict in place) so the
    variable does not leak into the child process environment and a hypothetical
    nested invocation cannot double-renice.

    Returns ``['nice', '-n', str(n)]`` when the popped value parses as an int in
    the privilege-free de-prioritizing range 1..19 AND ``nice`` is on PATH, else
    ``[]``.

    Fail-safe: absent key, empty string, zero, negative, out-of-range (>19),
    malformed value, or absent ``nice`` coreutil all return ``[]`` so no spawn
    ever fails due to a bad DF_AGENT_CPU_NICE value or a missing binary.

    The ``nice`` coreutil execvp's into the target binary in the same PID, so
    ``start_new_session=True`` / ``pgid=proc.pid`` and the process-group kill
    logic in _run_subprocess are unaffected.  cargo/rustc inherit the niceness
    via fork, which is the intended effect.
    """
    raw = env.pop('DF_AGENT_CPU_NICE', None)
    if not raw:
        return []
    try:
        n = int(raw)
    except ValueError:
        return []
    if not (1 <= n <= 19):
        return []
    if not shutil.which('nice'):
        return []
    return ['nice', '-n', str(n)]


async def _run_subprocess(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    model: str,
    timeout_seconds: float | None = None,
    stdin_data: bytes | None = None,
    session_id: str | None = None,
    config_dir: Path | None = None,
    startup_grace_secs: float = 120.0,
) -> _SubprocessResult:
    """Run a subprocess, log output.

    *stdin_data*, when set, is piped to the process's stdin.  This avoids
    passing large payloads as command-line arguments (which hit ARG_MAX).
    """
    logger.info(f'Invoking claude agent: model={model} cwd={cwd}')
    logger.info(f'Command: {" ".join(cmd[:15])}...')

    start_ms = int(time.monotonic() * 1000)

    # Compose the spawn prefix (govern OUTERMOST, then nice, then cmd):
    #   _cpu_govern_prefix  — places the agent and its inherited cargo/rustc/test
    #     subtree into a reify cpu.weight-weighted cgroup scope via
    #     cpu-governed-exec.sh (reify-owned).  Pops DF_AGENT_CPU_GOVERN so it
    #     does not leak to the child and cannot re-wrap.  cpu-governed-exec.sh
    #     execs in place, so start_new_session/pgid kill logic below is unaffected.
    #   _cpu_priority_prefix — prepends `nice -n N` to de-prioritize the agent.
    #     Pops DF_AGENT_CPU_NICE.  nice also execvp's in place.
    # Govern is outermost so the cgroup scope wraps nice wraps the Claude CLI
    # (PRD C-G1).
    spawn_cmd = _cpu_govern_prefix(env) + _cpu_priority_prefix(env) + cmd

    proc = await asyncio.create_subprocess_exec(
        *spawn_cmd,
        cwd=str(cwd),
        env=env,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    # Capture pgid at spawn (pgid == pid under start_new_session).  Never
    # refresh via os.getpgid() later — the PID may be reused post-reap.
    pgid = proc.pid

    comm_task: asyncio.Task[tuple[bytes, bytes]] | None = None
    try:
        try:
            # ── Two-regime liveness watchdog ────────────────────────────────
            # STARTUP regime (pre-turn-1): if no assistant turn appears within
            # startup_grace_secs, kill fast — catches from-source-build / uv /
            # MCP-startup wedges.
            # WORKING regime (≥1 turn seen): liveness proven; only the absolute
            # ceiling (timeout_seconds) triggers a kill.
            #
            # Conservative degrade: kill fires ONLY on an explicit observed
            # live_turns == 0 (a successful read returning 0).  A None /
            # unreadable transcript never triggers the fast kill — the watchdog
            # cannot prove a wedge and degrades through to the ceiling.
            #
            # asyncio.wait({comm_task}, timeout=poll) does NOT raise on a task
            # exception — it returns the task in `done`; comm_task.result()
            # re-raises.  Existing tests that mock communicate(side_effect=
            # TimeoutError) land in `done` on the first poll, result() re-raises,
            # and the unchanged except-TimeoutError kill block runs as today.
            watchdog_start = time.monotonic()
            seen_turn = False  # latched True once ≥1 assistant turn observed
            live_turns: int | None = None  # last non-None turn count read

            comm_task = asyncio.ensure_future(
                proc.communicate(input=stdin_data)
            )

            while True:
                elapsed = time.monotonic() - watchdog_start
                # How long until the next mandatory check-point?
                #
                # time_to_grace: collapse to inf once the startup-grace kill can
                # no longer fire — avoids a 0.0-poll tight-spin after grace expires.
                # The kill is permanently non-actionable when:
                #   • seen_turn: ≥1 assistant turn seen → working regime, never kills
                #   • elapsed >= startup_grace_secs: grace already passed; the kill
                #     check at the bottom of the loop fires on the first read that
                #     returns live_turns==0, no early wake-up needed
                #   • not (config_dir and session_id): transcript cannot be read →
                #     live_turns stays None → startup-kill requires live_turns==0 →
                #     can never trigger
                _grace_spent = (
                    seen_turn
                    or elapsed >= startup_grace_secs
                    or not (config_dir and session_id)
                )
                time_to_grace = (
                    float('inf') if _grace_spent
                    else max(0.0, startup_grace_secs - elapsed)
                )
                time_to_ceiling = (
                    max(0.0, timeout_seconds - elapsed)
                    if timeout_seconds is not None
                    else float('inf')
                )
                # Floor at _WATCHDOG_MIN_POLL_SECS so the poll never degenerates
                # to 0.0 (which would make asyncio.wait return immediately and
                # tight-spin, hammering count_transcript_turns and starving the
                # event loop).
                poll = max(
                    min(_WATCHDOG_POLL_SECS, time_to_grace, time_to_ceiling),
                    _WATCHDOG_MIN_POLL_SECS,
                )

                done, _ = await asyncio.wait({comm_task}, timeout=poll)

                if comm_task in done:
                    # Process exited (or communicate raised) — retrieve result.
                    # result() re-raises any exception the coroutine completed with
                    # (e.g. TimeoutError from mocked communicate in existing tests).
                    stdout, stderr = comm_task.result()
                    break  # normal exit → skip the kill block

                # Comm task still pending — check liveness.
                # Short-circuit once seen_turn is latched True: the startup-kill
                # guard requires `not seen_turn`, so live_turns is never consulted
                # again in the working regime.  Skip the on-disk read to avoid
                # redundant FS I/O for the (potentially 20-40 min) post-turn-1
                # lifetime of a healthy long-running agent.
                # The post-kill transcript_turns re-read in the except block is
                # unaffected — it is a separate, one-shot read outside this loop.
                if not seen_turn and config_dir and session_id:
                    n = count_transcript_turns(config_dir, session_id)
                    if n is not None:
                        live_turns = n
                        if n >= 1:
                            seen_turn = True

                elapsed = time.monotonic() - watchdog_start

                # Startup-regime kill: explicit 0-turn read AND grace expired.
                # NEVER kill on None (unreadable transcript) — conservative degrade.
                if (
                    not seen_turn
                    and live_turns == 0
                    and elapsed >= startup_grace_secs
                ):
                    logger.warning(
                        f'Startup wedge detected after {elapsed:.1f}s '
                        f'(grace={startup_grace_secs}s, turns=0): '
                        f'model={model} — cancelling comm_task and killing'
                    )
                    comm_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await comm_task
                    raise TimeoutError

                # Absolute-ceiling kill.
                if timeout_seconds is not None and elapsed >= timeout_seconds:
                    logger.warning(
                        f'Absolute ceiling reached after {elapsed:.1f}s '
                        f'(ceiling={timeout_seconds}s): model={model} — killing'
                    )
                    comm_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await comm_task
                    raise TimeoutError

        except TimeoutError:
            # Snapshot the process group FIRST — before terminate() — while the
            # wedged children are still alive and their /proc entries readable.
            # This is the sole place where pgid is in scope and the group is
            # guaranteed live; after the kill the snapshot would be empty/stale.
            proc_tree = snapshot_process_group(pgid)
            # Graceful shutdown: SIGTERM first, then SIGKILL after grace period.
            # SIGTERM lets the Claude CLI flush its final JSON output to stdout
            # (including session_id and token counts) before exiting.
            _SIGTERM_GRACE_SECS = 5
            # Try to capture final output — preserve existing stdout-capture
            # optimisation before falling back to process-group kill.
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.terminate()  # SIGTERM to direct child for output flush
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=_SIGTERM_GRACE_SECS,
                )
            except TimeoutError:
                # Still alive after grace period — kill entire process group
                # (bash → cargo → rustc grandchildren included).
                await terminate_process_group(proc, pgid, grace_secs=_SIGTERM_GRACE_SECS)
                stdout_text = ''
                stderr_text = f'Process killed after {timeout_seconds}s timeout (SIGTERM+SIGKILL)'
                logger.warning(
                    f'Subprocess SIGKILLed after {timeout_seconds}s timeout: '
                    f'model={model} pgid={pgid} — no stdout produced before kill'
                )
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
            tt = (
                count_transcript_turns(config_dir, session_id)
                if (config_dir and session_id)
                else None
            )
            return _SubprocessResult(
                stdout=stdout_text,
                stderr=stderr_text,
                returncode=proc.returncode if proc.returncode is not None else 1,
                duration_ms=duration_ms,
                timed_out=True,
                proc_tree=proc_tree,
                transcript_turns=tt,
            )
    except asyncio.CancelledError:
        # Orchestrator shutdown path: the awaiting task was cancelled. Kill the
        # entire process group (not just the direct child) so cargo/rustc
        # grandchildren are also reaped.
        # Also cancel comm_task (initialised to None above) so the communicate
        # coroutine is not left dangling after the process group is killed.
        if comm_task is not None and not comm_task.done():
            comm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await comm_task
        if proc.returncode is None:
            logger.warning(f'Subprocess cancelled — terminating process group for pid {proc.pid}')
            await terminate_process_group(proc, pgid, grace_secs=5.0)
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
