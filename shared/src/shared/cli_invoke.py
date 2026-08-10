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
from typing import TYPE_CHECKING, Any, TypeGuard

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
# Bounded retry budget for a pre-turn CLI rejection (task 3143 / esc-3118-1):
# the CLI exited on argument validation before contacting the API, so nothing
# was billed and no work was lost — a free retry.  ONE is deliberate: a single
# rejection is consistent with a transient race delivering the prompt to the
# child's stdin, but a SECOND consecutive one is deterministic (a genuinely
# blank prompt, a broken argv, a wrapper that never pipes stdin) and must reach
# a human via the normal steward/escalation path instead of looping.
_MAX_CLI_INPUT_REJECTED_RETRIES = 1
# Poll interval for the two-regime liveness watchdog in _run_subprocess.
# Each tick reads the on-disk transcript to check for assistant turns; the
# actual sleep per tick is min(_WATCHDOG_POLL_SECS, time_to_grace, time_to_ceiling)
# so grace and ceiling are never overshot by a full poll interval.
_WATCHDOG_POLL_SECS = 5.0
# Minimum poll duration — prevents the poll from degenerating to 0.0 when both
# time_to_grace and time_to_ceiling have already elapsed (would otherwise cause an
# asyncio.wait(timeout=0) tight-spin hammering count_transcript_turns).
_WATCHDOG_MIN_POLL_SECS = 0.01
# Coarse poll cadence for the WORKING-regime progress extension (task 2360).
# Once seen_turn latches AND working_idle_secs/absolute_cap_secs are both set,
# the watchdog keeps polling count_transcript_turns — but at this much coarser
# cadence than _WATCHDOG_POLL_SECS, since a healthy working session can run for
# 20-40 minutes and there is no need to hammer the transcript file every 5s.
# Still floored by _WATCHDOG_MIN_POLL_SECS and clamped by time-to-idle-kill /
# time-to-absolute-cap so a kill boundary is never overshot by a full poll.
_WATCHDOG_WORKING_POLL_SECS = 60.0
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
#
# orchestrator/dry_run_unblock.py         _DRY_RUN_CAP_WAIT_SANITY_SECS = 1800 s (30 min).
#   (block-time investigation invocation) Fire-and-forget background
#                                         investigation spawned from
#                                         _mark_blocked; 14-day default would
#                                         leave it pending for weeks under a
#                                         cap storm. AllAccountsCappedException
#                                         is caught and converted to a
#                                         retryable 'infra_failure' proposal
#                                         entry instead of raising.
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_CAP_WAIT_SANITY_SECS = 14 * 86400  # 14 days: outer sanity bound for patient cap waits
_CAP_WAIT_LOG_INTERVAL_SECS = 600.0  # emit at most one cap_wait log per ~10 min
CAP_HIT_RESUME_PROMPT = (
    'Your previous run was interrupted by a usage limit. '
    'Continue where you left off and complete your task.'
)
# Prompt sent to a session that the orchestrator is resuming after a crash.
# Kept separate from CAP_HIT_RESUME_PROMPT because the cause differs
# (orchestrator restart, not a usage-cap interrupt) and the agent message
# should stay a short crash-recovery continuation prompt rather than mentioning usage limits.
CRASH_RECOVERY_RESUME_PROMPT = (
    'You were interrupted by an orchestrator restart. '
    'Re-check your working context (plan.json if present) and current '
    'git/worktree state, then continue where you left off. '
    'Any escalation you filed before the restart may have been auto-dismissed '
    'as stale — re-raise it if it is still relevant.'
)

# The NON_CAP_CLI_ERROR_MARKERS table and its _is_non_cap_cli_error scanner
# that used to live here have moved to shared.invocation_outcome (task
# W4-beta single-source collapse); this module now consumes the verdict
# indirectly via classify_invocation's CliLocalError variant.


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
    'detect_ended_awaiting_background',
    'ended_awaiting_background_for_session',
    'invoke_claude_agent',
    'invoke_with_cap_retry',
    'is_cli_invocation_rejected',
    'is_server_error_status',
    'is_timed_out_with_progress',
    'is_zero_output_timeout',
    'read_transcript_records',
    'require_non_blank_prompt',
    'transcript_exists',
]


class AllAccountsCappedException(Exception):
    """Raised when the cap-hit retry loop exceeds its patience bound.

    Two independent bounds share this exception, both raised from the same
    ``_check_cap_wait`` choke point inside ``invoke_with_cap_retry``:
    - the wall-clock ``cap_wait_sanity_secs`` sanity deadline (default 14
      days), and
    - the count-based ``max_cap_retries`` bound, when the caller opts in.

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
            f'{label}: all accounts capped after {retries} retries ({elapsed_secs:.1f}s elapsed)'
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
    - ``ended_awaiting_background``: True when the run ended its turn while a
      backgrounded Bash command was still pending (launched via
      ``run_in_background`` and never subsequently polled/killed).  The headless
      one-shot ``claude --print`` session exits subtype=success and silently
      abandons the pending work (Reify-5164 RCA).  ``_parse_claude_output``
      downgrades ``success`` to False when this is set on an otherwise-successful
      run, so existing non-success handling retries/resumes.  Stamped by
      ``_run_subprocess`` on the normal-exit path from the parsed transcript
      records (via ``detect_ended_awaiting_background``; the
      ``ended_awaiting_background_for_session`` seam is the equivalent
      standalone helper).
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
    ended_awaiting_background: bool = False
    api_error_status: int | None = None
    proc_tree: str = ''
    transcript_turns: int | None = None
    """Number of assistant turns found in the on-disk JSONL transcript, or None
    when the transcript could not be read or located.  Stamped on the
    SIGTERM/SIGKILL timeout path (via count_transcript_turns) AND on the
    normal-exit path (task 2761 — derived from the same records read for the
    ended_awaiting_background check, at no extra I/O)."""


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


def transcript_exists(config_dir: Path, session_id: str) -> bool:
    """Return ``True`` iff a transcript for *session_id* exists under *config_dir*.

    A public, boolean existence check wrapping :func:`_resolve_transcript_path`
    (the version-robust glob-by-session-id locator).  Exported so cross-package
    callers (e.g. the orchestrator's session-resume eligibility guard) get a
    pure existence signal without importing the underscore-prefixed locator.
    Total — never raises: any glob error or absent file yields ``False``.
    """
    return _resolve_transcript_path(config_dir, session_id) is not None


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
                    logger.debug(f'read_transcript_records: skipping unparseable line in {path}')
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


# Background-management tool names that "reap" a launched background task — a
# poll (``BashOutput``) or a kill (``KillShell`` / ``KillBash``, the latter an
# older CLI spelling).  Any of these AFTER the last background launch clears the
# abandonment verdict.
_BACKGROUND_REAP_TOOLS = frozenset({'BashOutput', 'KillShell', 'KillBash'})


def detect_ended_awaiting_background(records: list[dict]) -> bool:
    """Return True when the transcript's final background action was an unreaped
    launch — i.e. the agent ended its turn while a backgrounded Bash command was
    still pending, silently abandoning the work (the Reify-5164 RCA).

    Contract (high-precision, no fragile parsing).  Over the ordered transcript
    *records*:

    - a **launch** = a ``Bash`` tool_use whose ``input.run_in_background`` is
      truthy;
    - a **reap** = any ``BashOutput`` / ``KillShell`` / ``KillBash`` tool_use.

    Fire (True) iff ``index(last launch) > index(last reap)`` — the session's
    final background-management action was a launch never followed by a
    poll/kill.  Any engagement with a background task (a poll or kill after it)
    clears the verdict, keeping precision high and avoiding fragile shell-id /
    result-text parsing that differs across CLI versions.

    Fail-safe / conservative by construction — a ``success``→failure downgrade
    must NEVER re-run a genuinely complete task on ambiguous data:

    - no launches, an empty/None/garbage record list, or a reap positioned after
      the last launch → False;
    - malformed records/blocks (non-dict, missing keys, wrong nesting) are
      skipped, never raise.

    Tolerant to both transcript content nestings:
    ``record['message']['content']`` (the real CLI shape) and a flat
    ``record['content']``.

    Known conservative false negative (accepted): an agent that polls a
    still-running task once with ``BashOutput`` and THEN ends its turn anyway
    reaps after the launch → False.  Erring toward NOT downgrading a possibly
    complete run is the safe direction for a success→failure flip.
    """
    if not records:
        return False
    last_launch_idx = -1
    last_reap_idx = -1
    pos = 0  # strictly-increasing position over tool_use blocks (record- then block-order)
    for record in records:
        if not isinstance(record, dict) or record.get('type') != 'assistant':
            continue
        message = record.get('message')
        if isinstance(message, dict) and isinstance(message.get('content'), list):
            blocks = message['content']
        elif isinstance(record.get('content'), list):
            blocks = record['content']
        else:
            continue
        for block in blocks:
            if not isinstance(block, dict) or block.get('type') != 'tool_use':
                continue
            pos += 1
            name = block.get('name')
            if name == 'Bash':
                inp = block.get('input')
                if isinstance(inp, dict) and inp.get('run_in_background'):
                    last_launch_idx = pos
            elif name in _BACKGROUND_REAP_TOOLS:
                last_reap_idx = pos
    return last_launch_idx != -1 and last_launch_idx > last_reap_idx


def ended_awaiting_background_for_session(
    config_dir: Path,
    session_id: str,
) -> bool:
    """Return True when *session_id*'s on-disk transcript ended its turn with a
    still-pending backgrounded Bash command.

    Mirrors ``count_transcript_turns``' shape: delegate to
    ``read_transcript_records``; if it returns None (transcript not located or a
    catastrophic read error) return False (fail-safe — an unreadable transcript
    must never downgrade a success on ambiguous data); otherwise apply the pure
    ``detect_ended_awaiting_background`` detector.  Never raises.
    """
    records = read_transcript_records(config_dir, session_id)
    if records is None:
        return False
    return detect_ended_awaiting_background(records)


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


def _stderr_has_cli_input_required(stderr: str) -> bool:
    """Case-insensitive scan of *stderr* for the CLI's input-required error.

    The single place the marker table is consulted, shared by
    ``is_cli_invocation_rejected`` (which takes an ``AgentResult``) and
    ``_parse_claude_output`` (which only has a ``_SubprocessResult``), so the
    subtype and the predicate can never disagree about what the marker is.
    """
    if not stderr:
        return False
    # Lazy (function-local) import — see the identical note in
    # classify_agent_failure: invocation_outcome imports cli_invoke at module
    # top, so a module-top import here would create a circular import.
    from shared.invocation_outcome import CLI_INPUT_REQUIRED_MARKERS

    stderr_lower = stderr.lower()
    return any(marker in stderr_lower for marker in CLI_INPUT_REQUIRED_MARKERS)


def _cli_input_rejection_cause(stderr: str) -> str:
    """Extract the operator-facing CAUSE line for a pre-turn CLI rejection.

    The CLI's own stderr is the ONLY evidence of what actually happened, so
    this returns REAL OBSERVED TEXT or an explicit absence marker — never an
    invented explanation.  Preference order:

    1. the first stderr line carrying one of ``CLI_INPUT_REQUIRED_MARKERS``
       (the CLI's verbatim argument-validation error);
    2. the LAST non-empty stderr line (defensive: a caller stamped the subtype
       by hand, or the CLI's wording drifted off the marker table — either way
       the tail is the closest thing to a real cause we observed);
    3. ``'<no stderr captured>'`` when stderr is empty — an explicit statement
       that nothing was observed, which is the honest degradation.  Fabricating
       a plausible-sounding cause here would be exactly the laundering this
       task exists to remove.
    """
    # Lazy (function-local) import — see _stderr_has_cli_input_required.
    from shared.invocation_outcome import CLI_INPUT_REQUIRED_MARKERS

    lines = [line.strip() for line in (stderr or '').splitlines()]
    non_empty = [line for line in lines if line]
    for line in non_empty:
        line_lower = line.lower()
        if any(marker in line_lower for marker in CLI_INPUT_REQUIRED_MARKERS):
            return line
    if non_empty:
        return non_empty[-1]
    return '<no stderr captured>'


def is_cli_invocation_rejected(result: AgentResult) -> bool:
    """Return True when the CLI rejected the invocation BEFORE any model turn.

    The signature is a pre-first-turn *transport* rejection: ``success=False``,
    ``timed_out=False``, ``turns == 0``, ``cost_usd == 0.0``, and a stderr
    carrying one of ``CLI_INPUT_REQUIRED_MARKERS``.  The agent was never asked
    anything — nothing was billed and no work was done — so the run is a free
    retry candidate rather than an agent failure.

    Observed payload (esc-3118-1, 2026-07-28 ~16:31Z)::

        Warning: no stdin data received in 3s, proceeding without it. ...
        Error: Input must be provided either through stdin or as a prompt
        argument when using --print

    with ``turns=0``, ``cost_usd=0.0``, ``duration_ms=17331``,
    ``timed_out=False``, and empty stdout.

    Deliberately contrasted with ``is_zero_output_timeout``: that predicate is
    keyed to the TIMEOUT family (it returns False immediately unless
    ``result.timed_out``), so it always misses this failure — which is exactly
    why no timeout-keyed consumer (the resume wedge guard, the workflow
    circuit breaker) ever caught the observed incident.  A killed run is not a
    pre-turn rejection: when ``timed_out`` is set the timeout predicates stay
    authoritative and this one returns False even if the marker text is
    present on stderr.

    EVIDENCE (either suffices, both mean the same thing):

    - ``result.subtype == 'error_cli_input_rejected'`` — the subtype
      ``_parse_claude_output`` mints for exactly this shape, so a result that
      has already been adjudicated stays adjudicated;
    - the stderr marker scan — catches a rejection that ``_parse_claude_output``
      could not label, e.g. one that happened to emit some stdout and so never
      entered the empty-stdout branch where the subtype is minted.

    Accepting BOTH is what keeps this predicate (the retry policy) and
    ``classify_agent_failure``'s CLI_INPUT_REJECTED rule (the taxonomy) from
    disagreeing about what happened: that rule consults this predicate, and
    this predicate accepts that rule's subtype, so neither can claim a result
    the other rejects.  See ``TestPredicateAndClassifierAgree``.

    ONE deliberate asymmetry remains, pinned there too: a hand-stamped subtype
    on a run that observably BILLED (turns/cost above zero — unreachable from
    ``_parse_claude_output``, which mints that subtype only when stdout is
    empty and therefore turns/cost were never parsed) is still classified
    CLI_INPUT_REJECTED but is NOT retried.  This predicate gates an ACTION
    with a cost, so on contradictory evidence it declines; stricter on the
    acting side is the safe direction.
    """
    if result.success or result.timed_out:
        return False
    if result.turns != 0 or result.cost_usd != 0.0:
        return False
    return (
        result.subtype == 'error_cli_input_rejected'
        or _stderr_has_cli_input_required(result.stderr)
    )


def require_non_blank_prompt(
    prompt: str | None, *, context: str, detail: str = ''
) -> None:
    """Raise ``ValueError`` when *prompt* is None, empty, or whitespace-only.

    The other half of the esc-3118-1 fix: make the "prompt never reached the
    CLI" failure impossible to cause from OUR side.

    The claude backend is 100% stdin-dependent.  ``build_claude_argv`` emits
    ``cmd = ['claude', '--print', '--output-format', 'json']`` and NEVER
    appends a positional prompt or a ``-`` stdin marker (unlike the codex
    backend in ``orchestrator/agents/invoke.py``, which passes its own input
    argument).  The prompt is delivered solely by
    ``stdin_data = prompt.encode()``, and a blank one pipes happily — the CLI
    then exits on argument validation with an opaque
    "Input must be provided either through stdin or as a prompt argument"
    error, zero-cost and zero-turn, with no indication that WE sent nothing.

    Called at every boundary that can originate an invocation, so the failure
    surfaces at the caller that built the blank prompt — with *context* naming
    it — instead of as an unattributable CLI error many layers away.

    THE SINGLE raise site for "blank prompt" across this module, deliberately:
    a caller that wants to defensively handle "I built a blank prompt" catches
    ONE exception type, and it does not vary with an unrelated flag.  The
    ``invoke_with_cap_retry`` resume branch — which previously raised its own
    hand-rolled ``TypeError`` for the same caller bug — delegates here and
    passes its resume-specific rationale as *detail* rather than forking the
    type.  *detail*, when given, is appended to the standard message.

    None is accepted and rejected loudly (not an ``AttributeError``): an
    explicitly-passed ``prompt=None`` is precisely the shape this guard exists
    to catch.
    """
    if prompt is None or not prompt.strip():
        message = (
            f'{context}: prompt must be a non-empty, non-whitespace string. '
            f'The claude CLI receives the prompt ONLY via stdin (the argv carries '
            f'no positional prompt), so a blank prompt is piped silently and the '
            f'CLI rejects the invocation before any model turn with an opaque '
            f'argument error (esc-3118-1). Got {prompt!r}.'
        )
        if detail:
            message = f'{message}  {detail}'
        raise ValueError(message)


def _should_retry_cli_input_rejected(result: AgentResult, retries_used: int) -> bool:
    """The SINGLE definition of the pre-turn-rejection retry policy.

    Called by both ``invoke_with_cap_retry`` dispatch sites — the gated
    ``while True`` loop and the ``usage_gate is None`` fast path — so the
    ceiling can never drift between them.
    """
    return is_cli_invocation_rejected(result) and retries_used < _MAX_CLI_INPUT_REJECTED_RETRIES


def is_server_error_status(status: int | None) -> TypeGuard[int]:
    """Return True when *status* is a server-side HTTP error (5xx).

    PRD contract C1 (plans/server-side-api-error-handling-prd.md): a 5xx —
    including 529 "Overloaded" — is a PROVIDER-side failure.  It is not
    account-scoped and not caused by anything local, so it must be routed to
    the transient-requeue lane rather than to cap/auth accounting.

    This is the single canonical definition (INV-5) shared by:

    - The ``ServerError`` tier in ``shared.invocation_outcome.
      classify_invocation`` (ranked below CapHit/NearCap, above
      ZeroOutputWedge).
    - ``classify_agent_failure``'s 5xx rule, which emits the verbatim
      ``agent API error: HTTP <status>`` marker.
    - Via ``shared``'s re-export, the orchestrator scheduler / workflow
      consumers landing in PRD tasks beta/gamma/delta.

    Every one of those callers must call THIS function rather than inline a
    ``500 <= n <= 599`` check, so the band has exactly one definition.

    ``None`` means "no structured status was reported" and is False — the
    absence of evidence is never evidence of a server error.

    4xx statuses deliberately fall OUTSIDE this band so the existing routing is
    untouched: 401/403 stay with ``AuthFailed``, 404 with ``ModelNotFound``,
    and 429 keeps its cap carve-out.

    Typed as a ``TypeGuard`` (a plain ``bool`` at runtime) so a True result
    also narrows ``int | None`` to ``int`` for the caller — the same narrowing
    the inline ``== 404`` / ``in (401, 403)`` status checks in
    ``classify_invocation`` already give, which is what lets the ``ServerError``
    tier pass ``result.api_error_status`` straight to ``ServerError(status=...)``.
    """
    return status is not None and 500 <= status <= 599


class AgentFailureKind(enum.StrEnum):
    """Classification of an AgentResult.  SUCCESS is the non-failure case."""

    SUCCESS = 'success'
    ENDED_AWAITING_BACKGROUND = 'ended_awaiting_background'
    MAX_TURNS = 'max_turns'
    EMPTY_OUTPUT = 'empty_output'
    CLI_INPUT_REJECTED = 'cli_input_rejected'
    API_ERROR = 'api_error'
    MODEL_NOT_FOUND = 'model_not_found'
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

    1. ``classify_invocation(result, strict_confirm=True)`` is ``OK``
       (mirrors ``result.success``) → ``SUCCESS``.
    2. ``result.ended_awaiting_background`` → ``ENDED_AWAITING_BACKGROUND``
       (the run ended its turn while a backgrounded Bash command was still
       pending; ``_parse_claude_output`` already downgraded ``success`` to
       False, so this sits immediately below the OK check — it can never
       shadow a genuine success — and above the timeout rule, which it cannot
       shadow either since the two flags are mutually exclusive by
       construction: the timeout path never sets this flag).
    3. ``is_server_error_status(result.api_error_status)`` → ``API_ERROR``
       (task 3314, plans/server-side-api-error-handling-prd.md). Placed
       ABOVE the timeout rule: a watchdog SIGTERM kill flushes the CLI's
       result JSON with ``api_error_status`` set (2026-07-29 incident), so
       ranking ``timed_out`` first discarded the 5xx evidence and misfiled a
       provider outage as a zero-output wedge — and the
       ``agent API error: HTTP <status>`` marker the scheduler's transient
       requeue lane keys on was never produced. The summary's prefix is that
       verbatim marker; a free-form kill-context suffix is appended only when
       ``result.timed_out``, so the non-timed-out summary is unchanged.
       This rule outranks rule 4 ONLY — three negative guards keep it from
       shadowing the rules that sit below rule 4 in source order but ABOVE
       it in precedence, i.e. it behaves as rule "3.5":
       ``is_timed_out_with_progress`` (defers to rule 4's productive-kill
       branch), ``subtype == 'error_max_turns'`` (defers to rule 5) and
       ``ModelNotFound`` (defers to rule 6, matching
       ``classify_invocation``'s ModelNotFound > ServerError ranking —
       INV-5).
    4. ``result.timed_out`` → ``TIMED_OUT`` (summary distinguishes a
       PRODUCTIVE kill — ``transcript_turns > 0`` — from a no-progress wedge;
       see ``is_timed_out_with_progress``/reify-4827). A productive kill
       keeps this kind even when a 5xx status rode along (rule 3's first
       guard): the wall-clock ceiling, not the provider, is what ended a run
       that did real agentic work, and downstream consumers key on
       ``TIMED_OUT`` for that (``dry_run_unblock``'s infra-failure kinds,
       the scheduler's genuine — not transient — requeue lane).
    5. ``result.subtype == 'error_max_turns'`` → ``MAX_TURNS``
       (high ``turns`` + non-zero ``output_tokens`` but empty ``output``).
       Rule 3 defers to this so a saturated run carrying an incidental 5xx
       is still detectable as saturation (workflow ``_stamp_simple_saturated``).
    6. the outcome is ``ModelNotFound`` → ``MODEL_NOT_FOUND`` (TERMINAL —
       no cross-account retry; placed ABOVE the ``api_error_status`` rule
       below because a 404 also sets ``api_error_status`` and would
       otherwise be mis-tagged as transient ``API_ERROR``. Rule 3 defers to
       this for the same reason: a ModelNotFound marker alongside a 5xx
       status is terminal for ``invoke_with_cap_retry``, so emitting the
       transient marker for it would tell the scheduler to requeue a run
       the retry loop already gave up on).
    7. ``result.api_error_status`` set, OR the outcome is ``AuthFailed`` →
       ``API_ERROR`` (includes status code in the summary; transient — worth
       retrying against another account). ``AuthFailed`` ({401, 403}) is a
       strict subset of "api_error_status is not None", so the ``OR`` never
       changes the verdict — it keeps this rule visibly tied to the
       InvocationOutcome contract without narrowing API_ERROR away from
       429, which InvocationOutcome does not model. Since rule 3 landed, only
       NON-5xx statuses ever reach here (a 5xx is claimed above), so this
       rule now covers 4xx/429 exclusively — its verdict for those is
       unchanged.
    8. ``result.subtype == 'error_cli_input_rejected'`` OR
       ``is_cli_invocation_rejected(result)`` → ``CLI_INPUT_REJECTED``
       (task 3143 / esc-3118-1): the CLI rejected the
       invocation on ARGUMENT VALIDATION before any model turn, because no
       prompt ever reached it.  The predicate disjunct keeps this rule and
       the retry policy in ``invoke_with_cap_retry`` from telling an operator
       two different stories about one run — the subtype is minted only
       inside ``_parse_claude_output``'s empty-stdout branch, so a rejection
       that emitted some stdout carries no subtype yet still satisfies the
       predicate.  Precedence is unchanged by the disjunct: the predicate
       requires ``not timed_out``, so rule 2 still claims every killed run,
       and rules 3/6/7 still claim a run carrying a server error, a
       ModelNotFound marker or an ``api_error_status``.  Placed immediately ABOVE rule 9 because a
       rejection is a strict, MORE SPECIFIC subset of "empty stdout": both
       arrive with empty output, but only this one means the agent was never
       asked anything.  Ranked below it, the generic rule would claim every
       such run first and launder a transport rejection into the transient
       agent-failure bucket, replacing the only evidence there is (the CLI's
       own stderr line) with the fixed, actively-wrong string 'agent returned
       empty output'.  The summary embeds that stderr line verbatim.
    9. ``result.subtype == 'error_empty_output'`` → ``EMPTY_OUTPUT``
       (may be transient).
    10. ``result.schema_salvaged`` → ``STRUCTURAL`` (schema-salvage: the
       subtype looked like an error but a valid structured output was
       recovered; callers usually treat as success).
    11. otherwise → ``UNKNOWN``.

    ``diagnostic_detail`` always includes: subtype, turns, cost_usd,
    duration_ms, timed_out, transcript_turns, api_error_status, output
    length, last 500 chars of stdout output, and last 500 chars of stderr.
    """
    # Lazy (function-local) import — see the identical note in
    # invoke_with_cap_retry: a module-top import here would create a
    # cli_invoke<->invocation_outcome circular import.
    from shared.invocation_outcome import OK, AuthFailed, ModelNotFound, classify_invocation

    outcome = classify_invocation(result, strict_confirm=True)

    tail_out = result.output[-500:] if result.output else ''
    tail_err = result.stderr[-500:] if result.stderr else ''
    diagnostic_detail = (
        f'subtype={result.subtype!r}\n'
        f'turns={result.turns}\n'
        f'cost_usd={result.cost_usd}\n'
        f'duration_ms={result.duration_ms}\n'
        f'timed_out={result.timed_out}\n'
        f'transcript_turns={result.transcript_turns}\n'
        f'api_error_status={result.api_error_status}\n'
        f'len(output)={len(result.output)}\n'
        f'output (last 500 chars):\n{tail_out}\n'
        f'stderr (last 500 chars):\n{tail_err}'
    )

    if isinstance(outcome, OK):
        return AgentFailureClass(
            kind=AgentFailureKind.SUCCESS,
            summary='agent succeeded',
            diagnostic_detail=diagnostic_detail,
        )
    if result.ended_awaiting_background:
        return AgentFailureClass(
            kind=AgentFailureKind.ENDED_AWAITING_BACKGROUND,
            summary=(
                'agent ended its turn awaiting a still-pending backgrounded '
                'task (work abandoned mid-turn)'
            ),
            diagnostic_detail=diagnostic_detail,
        )
    # Server-side (5xx) API failure — ranked ABOVE the timeout rule below.
    # A watchdog SIGTERM kill flushes the CLI's result JSON on the way out, so
    # a timed-out result can still carry hard 5xx evidence; ranking timed_out
    # first threw that evidence away and reported a provider outage as a local
    # zero-output wedge (2026-07-29 incident). Reads is_server_error_status
    # rather than an inline range check so the 5xx band has exactly one
    # definition (INV-5) shared with invocation_outcome's ServerError tier.
    #
    # The `agent API error: HTTP <status>` PREFIX is a cross-module contract:
    # orchestrator scheduler.py's _API_ERROR_REASON_RE searches block_reason
    # for it and reads the status out to route the transient requeue lane, so
    # it must stay verbatim and leading. The kill-context SUFFIX is free-form
    # operator forensics (PRD open question 5) and is emitted only on the
    # timed-out path, which keeps the non-timed-out summary byte-identical to
    # what callers already assert on.
    #
    # The three guards make this rule outrank the timeout rule ONLY. Source
    # order alone would also put it above max_turns and ModelNotFound, which
    # sit BELOW the timeout rule but ABOVE this one in precedence:
    # - is_timed_out_with_progress: a PRODUCTIVE kill (transcript_turns > 0)
    #   is a wall-clock timeout that happened to carry a 5xx, not a
    #   pre-first-token outage. Claiming otherwise both mis-phrases the
    #   summary and routes a productive kill into the transient-requeue lane
    #   / out of dry_run_unblock's infra-failure kinds. The 5xx is still in
    #   diagnostic_detail. NOTE this is a deliberate, benign divergence from
    #   classify_invocation, which still returns ServerError for that shape:
    #   in invoke_with_cap_retry the ServerError branch and the
    #   progress-timeout guard below it both confirm the slot and break, so
    #   the retry loop's behaviour is identical either way.
    # - error_max_turns: saturation detection (workflow's
    #   _stamp_simple_saturated) must survive an incidental 5xx.
    # - ModelNotFound: classify_invocation ranks ModelNotFound ABOVE
    #   ServerError, and the cap-retry loop treats it as TERMINAL. Emitting
    #   the transient marker here would have the scheduler requeue a run the
    #   retry loop already gave up on — the exact mis-tagging rule 6's
    #   placement exists to prevent (INV-5: the two classifiers agree).
    if (
        is_server_error_status(result.api_error_status)
        and not is_timed_out_with_progress(result)
        and result.subtype != 'error_max_turns'
        and not isinstance(outcome, ModelNotFound)
    ):
        summary = f'agent API error: HTTP {result.api_error_status}'
        if result.timed_out:
            # Guarded above, so transcript_turns is 0 or None here — never a
            # positive count. Only the 0 case can truthfully claim the kill
            # landed before the first token; None means the transcript was
            # never read, which is not evidence of either.
            elapsed_secs = result.duration_ms // 1000
            if result.transcript_turns == 0:
                summary += f' (killed at {elapsed_secs}s pre-first-token; transcript_turns=0)'
            else:
                summary += f' (killed at {elapsed_secs}s; transcript_turns=unknown)'
        return AgentFailureClass(
            kind=AgentFailureKind.API_ERROR,
            summary=summary,
            diagnostic_detail=diagnostic_detail,
        )
    if result.timed_out:
        # Truthful reporting (task 2360 fix #3): result.turns is always 0 on
        # the empty-stdout timeout path by construction (the CLI's JSON is
        # never parsed), so "(N turns)" was vacuous there — replace it with
        # the transcript-authoritative signal and a productive/wedge
        # distinction so a killed-but-productive run (reify-4827) is never
        # reported as indistinguishable from a genuine no-progress wedge.
        if result.transcript_turns:
            progress_desc = f'{result.transcript_turns} transcript turns (productive; not a wedge)'
        else:
            progress_desc = 'no transcript turns (wedge — no progress made)'
        return AgentFailureClass(
            kind=AgentFailureKind.TIMED_OUT,
            summary=(f'agent timed out after {result.duration_ms}ms with {progress_desc}'),
            diagnostic_detail=diagnostic_detail,
        )
    if result.subtype == 'error_max_turns':
        return AgentFailureClass(
            kind=AgentFailureKind.MAX_TURNS,
            summary=(
                f'agent hit max_turns ({result.turns} turns, output_tokens={result.output_tokens})'
            ),
            diagnostic_detail=diagnostic_detail,
        )
    if isinstance(outcome, ModelNotFound):
        return AgentFailureClass(
            kind=AgentFailureKind.MODEL_NOT_FOUND,
            summary=(
                f'agent model not available / not found ({outcome.reason}) — '
                f'terminal, no cross-account retry'
            ),
            diagnostic_detail=diagnostic_detail,
        )
    if result.api_error_status is not None or isinstance(outcome, AuthFailed):
        return AgentFailureClass(
            kind=AgentFailureKind.API_ERROR,
            summary=f'agent API error: HTTP {result.api_error_status}',
            diagnostic_detail=diagnostic_detail,
        )
    # Pre-turn CLI rejection — ranked immediately ABOVE the generic
    # empty-output rule below.  A rejection is a strict subset of "empty
    # stdout" (both land here with no output), so ordering these the other way
    # round would have the generic rule claim every rejection first — exactly
    # the laundering task 3143 exists to remove: 'agent returned empty output'
    # asserts we asked the agent something and got nothing back, when in fact
    # the prompt never reached the CLI and no model turn ever ran.  The cause
    # is carried through verbatim from stderr (the only evidence there is)
    # rather than replaced by a fixed string.
    #
    # The `or is_cli_invocation_rejected(...)` disjunct is what keeps the
    # TAXONOMY (this rule) and the RETRY POLICY (that predicate) from
    # disagreeing about what happened: the subtype is minted only inside
    # _parse_claude_output's empty-stdout branch, so a rejection that emitted
    # some stdout carries no subtype at all — it would be retried by
    # invoke_with_cap_retry and then reported downstream as EMPTY_OUTPUT,
    # i.e. the two layers telling an operator two different stories about one
    # run.  Consulting the predicate here closes that direction; the predicate
    # accepting this rule's subtype closes the other.
    if result.subtype == 'error_cli_input_rejected' or is_cli_invocation_rejected(result):
        cause = _cli_input_rejection_cause(result.stderr)
        return AgentFailureClass(
            kind=AgentFailureKind.CLI_INPUT_REJECTED,
            summary=(
                'CLI rejected the invocation before any model turn '
                f'(no prompt reached the CLI): {cause}'
            ),
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
        summary=(f'agent failed: subtype={result.subtype!r} (no specific failure signal)'),
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
    ended_awaiting_background: bool = False
    """True when the normal-exit transcript ended its turn with a still-pending
    backgrounded Bash command; carried into AgentResult and used by
    _parse_claude_output to downgrade success→failure.  Never set on the
    timeout path (a timed-out run is already non-success)."""


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
    spawn_env: dict[str, str] | None = None,
    startup_grace_secs: float = 120.0,
    sandbox_wrap: Callable[[list[str]], list[str]] | None = None,
    working_idle_secs: float | None = None,
    absolute_cap_secs: float | None = None,
    strict_mcp_config: bool = False,
) -> AgentResult:
    """Invoke Claude Code CLI and return structured result.

    *strict_mcp_config*, when True (and an *mcp_config* is set), emits
    ``--strict-mcp-config`` so the invocation is scoped to only *mcp_config*'s
    servers, ignoring the ambient ``.mcp.json`` merge (task 2796, THREAD 2);
    forwarded verbatim to ``build_claude_argv``. Default ``False`` keeps every
    existing caller byte-identical.

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

    *spawn_env*, when set, carries ``CLAUDE_SPAWN_*`` spawn-identity vars
    (role/project/task/parent) for the SessionStart hook.  Merged into the
    subprocess environment AFTER *env_overrides* so per-agent spawn identity
    always wins over any inherited ``CLAUDE_SPAWN_*`` value; keys with an
    empty/falsy value are skipped so a blank never clobbers an inherited one.
    Also scrubs any ``CLAUDE_SPAWN_SESSION_ID``/``CLAUDE_SPAWN_LAUNCHER_PID``
    this process itself inherited (e.g. if the orchestrator was itself
    fleet-spawned) — ``session_hooks.hook_session_slug`` prefers an inherited
    ``CLAUDE_SPAWN_SESSION_ID`` outright over reconstructing from
    role/project/task_id, so leaving it in place would collapse every
    spawned agent onto ONE registry record instead of each getting its own.

    *sandbox_wrap*, when set, is applied to the built claude argv immediately
    before the subprocess is spawned.  The callable receives the full cmd list
    (e.g. ``['claude', '--print', ...]``) and returns a replacement list (e.g.
    ``['python', '/path/to/landlock_exec.py', '--', 'claude', '--print', ...]``).
    Keeps ``shared`` policy-agnostic: callers supply the confinement closure;
    the subprocess sees the wrapped argv.  Default ``None`` → no wrap (today's
    behavior).

    *working_idle_secs* / *absolute_cap_secs*, when BOTH set, extend the
    working-regime watchdog past *timeout_seconds* while the transcript keeps
    advancing: the subprocess is killed only after no new transcript turn for
    ``max(working_idle_secs, timeout_seconds)``, or at *absolute_cap_secs*,
    whichever comes first.  Default ``None`` for both → no extension,
    *timeout_seconds* stays the hard wall (today's exact behavior).
    """
    return await _invoke_claude(
        prompt=prompt,
        system_prompt=system_prompt,
        cwd=cwd,
        model=model,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        mcp_config=mcp_config,
        output_schema=output_schema,
        permission_mode=permission_mode,
        effort=effort,
        oauth_token=oauth_token,
        timeout_seconds=timeout_seconds,
        resume_session_id=resume_session_id,
        session_id=session_id,
        config_dir=config_dir,
        env_overrides=env_overrides,
        spawn_env=spawn_env,
        startup_grace_secs=startup_grace_secs,
        sandbox_wrap=sandbox_wrap,
        working_idle_secs=working_idle_secs,
        absolute_cap_secs=absolute_cap_secs,
        strict_mcp_config=strict_mcp_config,
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
    max_cap_retries: int | None = None,
    rebuild_prompt: Callable[[bool], Awaitable[str]] | None = None,
    resume_delivers_prompt: bool = False,
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

    *max_cap_retries* is an optional count-based sibling of
    *cap_wait_sanity_secs*: when the number of consecutive cap hits reaches
    this value, ``AllAccountsCappedException`` is raised (same exception as
    the time-based bound) before the next cooldown sleep.  Defaults to
    ``None``, which preserves the existing patient, count-unbounded wait —
    only *cap_wait_sanity_secs* bounds the retry loop.

    *rebuild_prompt*, when provided, is awaited as ``rebuild_prompt(True)``
    on a cap retry whose session cannot be resumed (no ``session_id`` on the
    capped result) — ``True`` signals ``session_lost``.  Its return value
    replaces ``prompt`` for the next invocation, letting the caller rebuild
    fresh context (e.g. re-gathered pending escalations) instead of reusing
    the stale original prompt.  The resumable path (capped result carries a
    ``session_id``) is unaffected — it keeps resuming with
    ``CAP_HIT_RESUME_PROMPT`` and never calls this hook.  Defaults to
    ``None``, which preserves the existing fresh-retry behaviour (reuse the
    original prompt unchanged).

    *resume_delivers_prompt*, when ``True``, delivers the caller's real
    ``prompt`` on a caller-initiated resume (``resume_session_id`` pre-set
    before the first invocation) instead of overwriting it with
    ``CRASH_RECOVERY_RESUME_PROMPT``.  This is for a *live continuation*,
    where the resumed session must receive NEW content it has not seen yet
    (e.g. the steward's per-escalation continuation prompt).  Defaults to
    ``False``, which preserves the crash-recovery contract used by
    ``workflow._invoke``: a crash-recovered session already holds the full
    task context, so the short crash-recovery continuation prompt is sufficient
    and the real prompt is kept only as ``original_prompt`` for fresh-fallback.

    The ``session_lost`` argument is currently always ``True`` — every wired
    call site is an unresumable cap retry.  It is kept as an explicit
    parameter (rather than a no-arg callable) as a forward-compat placeholder
    matching the caller contract in PRD §7.4, so a future resumable-path call
    (if ever wired) needs no signature change.  If the hook itself raises,
    the failure is caught and logged, and the retry falls back to the
    already-restored original prompt rather than aborting the retry loop.

    *label* identifies the caller in log messages (e.g. "Module tagging",
    "Task 7 [implementer]").

    When *cost_store* is provided, successful invocations are recorded via
    ``save_invocation()`` and cap-hit events via ``save_account_event()``.

    All keyword arguments are forwarded to ``invoke_claude_agent()``.  When a
    custom *invoke_fn* is supplied, it additionally receives *backend*
    (multi-backend reconnect, PRD harness-backend-reconnect-pi T1) — the
    default ``invoke_claude_agent`` path never does, since it has no
    ``backend`` parameter.

    A pre-turn CLI REJECTION (``is_cli_invocation_rejected``: the prompt never
    reached the child's stdin, so the CLI exited on argument validation before
    contacting the API) is retried FRESH at most
    ``_MAX_CLI_INPUT_REJECTED_RETRIES`` (1) time — nothing was billed and no
    transcript exists, so the retry is free and loses nothing.  The branch sits
    ABOVE the heuristic cap safety-net deliberately: the CLI's stdin wait is
    only 3s, so a fast-exit rejection falls inside that net's sub-5s window and
    would otherwise be converted into a synthetic cap hit, churning the whole
    account pool for a local argument error the API never saw.  Once the budget
    is spent the failed result is returned unchanged for normal steward
    handling — a second consecutive rejection is deterministic, not a glitch.
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
    # `resume_delivers_prompt` opts a live-continuation caller (the steward) out of
    # this swap: its resumed session must receive the real prompt, not the short
    # crash-recovery continuation prompt.
    if invoke_kwargs.get('resume_session_id'):
        # ONE exception type for "blank prompt" (task 3143 amendment): this
        # branch used to raise its own hand-rolled TypeError, so the SAME
        # caller bug surfaced as TypeError or ValueError depending on an
        # unrelated flag — a caller defending against it had to catch both.
        # Delegating also fixes the None shape: `original_prompt` is
        # `invoke_kwargs.get('prompt', '')`, so an explicitly-passed
        # `prompt=None` used to die on `None.strip()` with an incidental
        # AttributeError instead of this deliberate, well-messaged raise —
        # precisely the shape the guard exists to catch loudly.  The
        # resume-specific rationale rides along as `detail` so nothing is lost.
        require_non_blank_prompt(
            original_prompt,
            context=f'{label} (resume_session_id set)',
            detail=(
                'On a resume invocation the prompt is the real task context '
                'kept for fresh-fallback recovery if the resume fails; passing '
                'an empty or missing prompt silently corrupts that fallback.'
            ),
        )
        if not resume_delivers_prompt:
            invoke_kwargs['prompt'] = CRASH_RECOVERY_RESUME_PROMPT
    else:
        # Non-resume invocation: the prompt IS the whole request, so a blank one
        # is always a caller bug.  Raised here — before any invoke_slot is
        # acquired — so a blank prompt never consumes an account slot, never
        # burns a dispatch, and never reaches the CLI as an opaque argument
        # error (esc-3118-1).  Same guard, same exception type as the resume
        # branch above; only the context/detail differ, since that branch
        # legitimately overwrites `prompt` with a short continuation string.
        require_non_blank_prompt(invoke_kwargs.get('prompt'), context=f'{label}')
    consecutive_cap_hits = 0
    cli_input_rejected_retries = 0
    num_accounts = max(usage_gate.account_count, 1) if usage_gate else 1
    retry_start = time.monotonic()
    last_cap_wait_log_at: float | None = None

    def _check_cap_wait(now: float, elapsed: float, cooldown: float, hits: int) -> None:
        """Guard and throttled log for cap-wait iterations.

        Raises AllAccountsCappedException when the 14-day sanity bound is exceeded,
        or when the count-based max_cap_retries bound is reached.
        Emits a structured 'cap_wait' JSON log at most once per _CAP_WAIT_LOG_INTERVAL_SECS.
        Closes over: cap_wait_sanity_secs, max_cap_retries, label, num_accounts,
        usage_gate, last_cap_wait_log_at (nonlocal write).
        """
        nonlocal last_cap_wait_log_at
        if max_cap_retries is not None and hits >= max_cap_retries:
            logger.error(
                f'{label}: max cap-retries bound ({max_cap_retries}) reached after {hits} retries',
            )
            raise AllAccountsCappedException(
                retries=hits,
                elapsed_secs=elapsed,
                label=label,
            )
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
        if (
            last_cap_wait_log_at is None
            or now - last_cap_wait_log_at >= _CAP_WAIT_LOG_INTERVAL_SECS
        ):
            logger.warning(
                json.dumps(
                    {
                        'event': 'cap_wait',
                        'label': label,
                        'elapsed_s': round(elapsed, 1),
                        'soonest_open_at': (
                            usage_gate.soonest_resets_at.isoformat()
                            if usage_gate and usage_gate.soonest_resets_at
                            else None
                        ),
                        'next_probe_in_s': round(cooldown, 1),
                    },
                    default=str,
                )
            )
            last_cap_wait_log_at = now

    async def _rebuild_fresh_prompt() -> None:
        """Let the caller rebuild the prompt for an unresumable cap retry.

        session_lost is always True at both call sites (exact-detect and
        heuristic FRESH cap paths) — this helper only runs where the capped
        session cannot be resumed, per the rebuild_prompt hook contract
        (PRD §7.4 / task W4-zeta). Closes over: rebuild_prompt, invoke_kwargs,
        label.

        A failure raised by the caller's hook (e.g. a transient I/O/MCP error
        while re-gathering pending escalations) is caught and logged rather
        than propagated: propagating would abort the entire patient cap-retry
        loop over a transient hook failure.  ``invoke_kwargs['prompt']`` was
        already restored to ``original_prompt`` by the preceding
        ``_reset_for_fresh_retry`` call, so on failure it is simply left as
        that already-restored value — the retry degrades to the stale
        original prompt instead of dying.
        """
        if rebuild_prompt is None:
            return
        try:
            invoke_kwargs['prompt'] = await rebuild_prompt(True)
        except Exception:
            logger.warning(
                f'{label}: rebuild_prompt hook raised — falling back to original prompt',
                exc_info=True,
            )

    account_name = ''
    unattributed_cap = False  # True when heuristic fires but token is unresolvable;
    # controls: (1) skip confirm, (2) mark capped=True in cost_store
    started_at = ''
    completed_at = ''

    # Default to Claude-specific invocation when no invoke_fn was provided
    invoke: Callable[..., Awaitable[AgentResult]] = invoke_fn or invoke_claude_agent

    # Multi-backend reconnect (PRD T1, harness-backend-reconnect-pi): forward
    # `backend` into the dispatched call ONLY when a custom invoke_fn (the
    # multi-backend invoke_agent) is supplied. The default invoke_claude_agent
    # path (fused-memory recon/curator, cli_stage_runner.py) has NO `backend`
    # parameter and must NEVER receive the kwarg (Invariant 3, PRD Appendix A).
    if invoke_fn is not None:
        # setdefault (not `=`) is purely defensive: `backend` is a
        # keyword-only parameter of this function, so a caller can never
        # smuggle a pre-existing `backend` key into invoke_kwargs — there is
        # no live override path today. Kept as setdefault so a future
        # invoke_kwargs source that does pre-populate `backend` isn't
        # silently clobbered.
        invoke_kwargs.setdefault('backend', backend)

    # Fast path: no usage gate → no cap retry (there is no account pool to fail
    # over to), but the pre-turn-rejection retry below still applies: it is not
    # an account failure, and this path is taken by the gate-less fused-memory
    # callers (reconciliation, judge, curator) that would otherwise keep eating
    # the failure silently.  Bounded by the SAME
    # _should_retry_cli_input_rejected policy the gated loop uses, so the
    # ceiling can never drift between the two dispatch sites.
    #
    # NOTE: if `resume_session_id` was set by the caller (crash-recovery path),
    # this fast path will attempt the resume but cannot fall back to a fresh
    # invocation on a general failure — the non-cap-hit resume→fresh-fallback
    # branch in the while-loop below only runs when usage_gate is provided.  In
    # practice the orchestrator always supplies a gate, but callers without one
    # should be aware that a failed resume returns the failure result directly.
    # A pre-turn REJECTION is the one exception, and it is not really an
    # exception to that rule: the CLI exited before contacting the API, so
    # there is no session to preserve and _reset_for_fresh_retry's switch to a
    # fresh invocation discards nothing.
    if not usage_gate:
        while True:
            started_at = datetime.now(UTC).isoformat()
            result = await invoke(
                **invoke_kwargs,
                config_dir=config_dir.path if config_dir else None,
            )
            completed_at = datetime.now(UTC).isoformat()
            # Re-stamped per attempt (above), so started_at/completed_at always
            # describe the attempt actually RETURNED — leaving the discarded
            # attempt's stamps in place would write a silently false window
            # into the cost_store invocations row.
            if not _should_retry_cli_input_rejected(result, cli_input_rejected_retries):
                break
            cli_input_rejected_retries += 1
            logger.warning(
                f'{label}: CLI rejected the invocation before any model turn '
                f'(no prompt reached the CLI) — retrying fresh '
                f'({cli_input_rejected_retries}/{_MAX_CLI_INPUT_REJECTED_RETRIES}). '
                f'Cause: {_cli_input_rejection_cause(result.stderr)}',
            )
            _reset_for_fresh_retry(invoke_kwargs, original_prompt)
    else:
        # Derive this invocation's cap scope once (PRD task β, write half of
        # boundary B1): the invoked model when it is a scoped-cap model, else
        # None (the general scope). Scoped ONLY for the claude backend
        # (decision 7 "claude-backend scope only"): a non-claude backend
        # (codex/gemini) never touches the OAuth account pool's model-scoped
        # caps, satisfying B8. Lazy import mirrors the invocation_outcome
        # import below to avoid the usage_gate<->cli_invoke cycle.
        # getattr(...,'_config',None) is None for a spec'd MagicMock gate and
        # scope_for(m, <no config>) is None, so every existing mock-based suite
        # derives scope None → byte-equivalent.
        from shared.usage_gate import scope_for

        _cfg = getattr(usage_gate, '_config', None)
        scope = scope_for(model, _cfg) if (backend == 'claude' and _cfg is not None) else None
        while True:
            async with usage_gate.invoke_slot(scope=scope) as slot:
                # slot.account_name is derived from slot.lease — the SAME
                # account slot.token came from (task W4-δ, PRD §7.4). This
                # is what makes the attribution below (and the save_invocation
                # call at the bottom of this function) name the account
                # actually invoked, not a differently-phased one (finding 3 /
                # boundary test B5).
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

                # Lazy (function-local) import: invocation_outcome.py imports
                # shared.cli_invoke at module top (for is_zero_output_timeout),
                # so a module-top import here would create a circular import.
                # Importing inside the loop body runs after both modules are
                # fully loaded, breaking the cycle at negligible cost (the
                # module is already in sys.modules).
                from shared.invocation_outcome import (
                    AuthFailed,
                    CapHit,
                    CliLocalError,
                    ModelNotFound,
                    ServerError,
                    ZeroOutputWedge,
                    classify_invocation,
                )

                outcome = classify_invocation(result, strict_confirm=True, backend=backend)

                # Auth-failure routing (401/403): distinct from cap hits.
                # Mark the account auth_failed and fail over; don't count
                # toward consecutive_cap_hits so the cooldown doesn't compound.
                # The narrow {401, 403} is load-bearing: 429 carries a real
                # cap-message body ("You're out of extra usage · resets ...")
                # that the cap-hit detector below already recognises, but if
                # we route 429 here the slot.detect_cap_hit() call never runs,
                # AllAccountsCappedException never fires, and the curator
                # worker's cap-defer machinery silently never engages.
                # classify_invocation mirrors this exact narrowing: AuthFailed
                # is only returned for {401, 403} (see invocation_outcome.py).
                # slot.report(outcome) applies the AUTH_FAILED transition and
                # settles the slot atomically (task W4-ε) — always failover
                # unconditionally; an unresolvable token means the account
                # vanished/refreshed, so failing over is safe and
                # self-terminating (no separate unattributed fall-through).
                # Rebuild via the caller's hook (no-op when rebuild_prompt is
                # None): a live-continuation caller's original_prompt
                # (resume_delivers_prompt=True) is only valid inside the
                # resumed session, not the brand-new one this failover starts.
                if isinstance(outcome, AuthFailed):
                    slot.report(outcome)
                    _reset_for_fresh_retry(invoke_kwargs, original_prompt)
                    await _rebuild_fresh_prompt()
                    logger.warning(
                        f'{label}: account {account_name} auth-failed '
                        f'(HTTP {result.api_error_status}) — failing over',
                    )
                    continue

                # Model-not-found is TERMINAL: the requested model doesn't exist /
                # isn't available to any account, so cross-account failover can
                # only repeat the same zero-cost, near-instant failure on every
                # other account.  Without this branch, the result falls through to
                # the heuristic cap-hit safety net below (zero-cost/near-instant/
                # ≤1-turn) which misclassifies it as a cap hit — cycling the
                # entire pool through compounding cooldowns until
                # AllAccountsCappedException fires (the "TRANSIENT → whole-pool
                # churn" bug this branch exists to prevent).  slot.confirm settles
                # the slot as a normal (zero-cost) completion — no account is
                # marked capped/auth_failed — and the failed result is returned to
                # the caller directly for genuine terminal handling.
                if isinstance(outcome, ModelNotFound):
                    logger.warning(
                        f'{label}: model not found/available on account '
                        f'{account_name} ({outcome.reason}) — terminal, no '
                        f'cross-account failover',
                    )
                    if not unattributed_cap:
                        slot.confirm(result.cost_usd)
                    break

                # Pre-turn CLI REJECTION (task 3143 / esc-3118-1): the prompt
                # never reached the child's stdin, so the CLI exited on
                # argument validation BEFORE contacting the API.  The agent was
                # never asked anything, nothing was billed and no transcript
                # exists — so this is a free retry, not an agent failure.
                #
                # POSITIONING is load-bearing in two directions:
                # - ABOVE the heuristic cap safety-net below: the CLI's stdin
                #   wait is only 3s, so a fast-exit rejection lands inside that
                #   net's `duration_ms < 5000` window.  Reaching it would
                #   convert a local argument error into a SYNTHETIC CapHit and
                #   churn the whole account pool through compounding cooldowns.
                #   (The CliLocalError escape added with CLI_INPUT_REQUIRED_MARKERS
                #   also covers this — defence in depth, not redundancy: this
                #   branch retries, that escape merely declines to cap.)
                # - BELOW the ModelNotFound/AuthFailed branches: those are
                #   account- or model-scoped verdicts that must keep their own
                #   terminal/failover handling.
                #
                # Retried FRESH, never resumed: no session was ever created, so
                # there is nothing to resume, and reusing the prior attempt's
                # pre-allocated session_id would hit the reify-3604 'Session ID
                # ... is already in use' wedge.  _reset_for_fresh_retry
                # regenerates it.  The caller's rebuild_prompt hook is
                # deliberately NOT invoked: it signals session_lost, and here no
                # context was ever built, let alone lost — the original prompt
                # is still exactly the right thing to send.
                #
                # slot.confirm settles the slot as a normal zero-cost
                # completion (mirroring the ModelNotFound branch's shape): no
                # account is marked capped or auth_failed, because nothing about
                # the ACCOUNT failed.
                if _should_retry_cli_input_rejected(result, cli_input_rejected_retries):
                    cli_input_rejected_retries += 1
                    if not unattributed_cap:
                        slot.confirm(result.cost_usd)
                    logger.warning(
                        f'{label}: CLI rejected the invocation before any model turn '
                        f'(no prompt reached the CLI) on account {account_name} — '
                        f'retrying fresh '
                        f'({cli_input_rejected_retries}/{_MAX_CLI_INPUT_REJECTED_RETRIES}). '
                        f'Cause: {_cli_input_rejection_cause(result.stderr)}',
                    )
                    _reset_for_fresh_retry(invoke_kwargs, original_prompt)
                    continue

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
                # Rebuild via the caller's hook (no-op when rebuild_prompt is
                # None): a live-continuation caller's original_prompt
                # (resume_delivers_prompt=True) is only valid inside the
                # wedged session, not the brand-new one this retry starts.
                if isinstance(outcome, ZeroOutputWedge) and invoke_kwargs.get('resume_session_id'):
                    logger.warning(
                        f'{label}: zero-output timed-out invocation '
                        f'(duration_ms={result.duration_ms}) — clearing wedged '
                        f'resume_session_id={invoke_kwargs["resume_session_id"]} '
                        f'before retry',
                    )
                    _reset_for_fresh_retry(invoke_kwargs, original_prompt)
                    await _rebuild_fresh_prompt()
                    continue  # __aexit__ releases probe slot

                if slot.detect_cap_hit(result.stderr, result.output, backend=backend):
                    consecutive_cap_hits += 1
                    full_cycles = (consecutive_cap_hits - 1) // num_accounts
                    cooldown = min(
                        _CAP_HIT_COOLDOWN_SECS * (2**full_cycles),
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

                    # ------------------------------------------------------------------
                    # MEASURED 2026-08-01 (task 3454, claude CLI 2.1.220), and
                    # RE-MEASURED to a verdict 2026-08-05 (task 3484, CLI
                    # 2.1.222).
                    #
                    # MECHANISM (confirmed empirically, and it reframes the
                    # question).  Claude CLI sessions are LOCAL JSONL transcripts
                    # at <config_dir>/projects/<cwd-slug>/<session_id>.jsonl — not
                    # server-side, account-scoped objects.  `--resume` replays that
                    # local file.  So what governs a cross-account resume is
                    # TRANSCRIPT REACHABILITY, not OAuth identity.  Observed: a
                    # session started on one account wrote
                    # .../projects/-tmp/<sid>.jsonl (slug from cwd=/tmp) under the
                    # EFFECTIVE config dir — the ambient CLAUDE_CONFIG_DIR, since
                    # invoke_claude_agent inherits os.environ when config_dir is
                    # None — and NOT under ~/.claude.  A resume issued on a
                    # DIFFERENT account appended its turn to that same file
                    # (12 -> 20 records), i.e. the resume attached locally across
                    # the account switch.
                    #
                    # The retry loop below keeps that reachable on purpose: it
                    # reuses ONE TaskConfigDir across rotations and rewrites
                    # .credentials.json in place (see the write_credentials call
                    # further down), passing the same config_dir.path every
                    # attempt.  The guard below ENFORCES that invariant instead of
                    # assuming it.
                    #
                    # VERDICT (2026-08-05, task 3484): a cross-account resume
                    # DOES PRESERVE conversation context.  Measured
                    # 20:04:11–20:05:15Z on claude CLI 2.1.222 with accounts
                    # CLAUDE_OAUTH_TOKEN_F (r1 — starts the session) ->
                    # CLAUDE_OAUTH_TOKEN_C (r2 — issues the --resume), both
                    # probed healthy 6 minutes earlier.  3 valid runs, 0 void,
                    # 3 distinct r1 sessions:
                    #   6a259899-315b-4cd3-94cd-8448c982daaf
                    #   75f9c167-7e91-4743-995e-5d943fac2326
                    #   362bb71e-0489-4f61-b930-0cf772982d04
                    # In every one: transcript present after r1 (11 records);
                    # r2 succeeded on the OTHER account (subtype='success',
                    # empty stderr) and answered "ZEPPELIN" — the codeword
                    # planted in r1; and the same-account control PASSED in the
                    # same pytest process, so the harness was sound while the
                    # cross-account result was taken.  The transcripts confirm
                    # the mechanism above carried it: r2 appended to r1's own
                    # local file (11 -> 19 records), r2's turn and its answer
                    # among the appended records.
                    # Full record, with the verbatim per-run evidence and the
                    # pre-1 gate: plans/cross-account-resume-measurement.md.
                    #
                    # CONSEQUENCE: the resume below is doing what it intends,
                    # and the reachability guard that follows is load-bearing
                    # for the OTHER failure mode — a transcript that is GONE,
                    # not an account that changed.
                    #
                    # SCOPE of the claim: this is a property of the CURRENT
                    # mechanism (local transcript + --resume) on CLI 2.1.222,
                    # not a guarantee from the API.  If a future CLI moves
                    # sessions server-side and scopes them per account, the
                    # answer can change with it.  The regression guard is
                    # tests/test_cli_invoke_integration.py::TestCrossAccountResume
                    # — re-run it after a CLI upgrade that touches session
                    # handling (needs `-m integration`, which pyproject
                    # deselects by default, and two simultaneously-uncapped
                    # accounts aimed at via CROSS_ACCOUNT_RESUME_TOKENS).
                    #
                    # THE 2026-08-01 ROUND (task 3454), kept because it is why
                    # the skip guard exists.  Same-account control PASSED
                    # (CLAUDE_OAUTH_TOKEN_B, r1
                    # sid=8e4d1819-db90-4b69-8f42-f8ef09facd52, 12 records,
                    # codeword recalled).  The single cross-account attempt
                    # (A=CLAUDE_OAUTH_TOKEN_B, B=CLAUDE_OAUTH_TOKEN_C, r1
                    # sid=eeec059e-be5d-413d-bae7-15274dd758c3, transcript
                    # PRESENT after r1, 12 records) did NOT recall the codeword
                    # — but it was VOID, not negative: r2's transcript turn is
                    # literally "You've hit your weekly limit · resets Aug 5,
                    # 11am", i.e. account B was CAPPED and no model turn ever
                    # ran.  It read as context loss only because the test
                    # module's skip guard matched "you've hit your usage" while
                    # the real text is "you've hit your weekly limit".  Task
                    # 3483 closed that gap — the corpus now lives single-homed
                    # in tests/_capacity_skip.py, pinned against this exact
                    # string and cross-checked against
                    # invocation_outcome.classify_invocation so the two cannot
                    # drift apart again.  A capped account SKIPS, and task 3484
                    # added a second void class (verdict='void_error') so a
                    # budget abort or API error cannot masquerade as context
                    # loss either.
                    # ------------------------------------------------------------------
                    # Resume the capped session on the next account if possible.
                    #
                    # A session is resumable only if its transcript is actually
                    # REACHABLE: Claude CLI sessions are local JSONL files at
                    # <config_dir>/projects/*/<session_id>.jsonl (see
                    # _resolve_transcript_path), and --resume replays that file.
                    # Resuming a session whose transcript is gone (cleaned-up
                    # TaskConfigDir, a different config dir, a swept temp dir)
                    # starts an effectively EMPTY session, and the agent then
                    # restarts on CAP_HIT_RESUME_PROMPT with no context to
                    # continue from — silent context loss.  Mirrors the
                    # orchestrator's own resume-eligibility guard
                    # (harness.py, 'no_transcript').
                    #
                    # config_dir is None -> resume as today: without a concrete
                    # directory there is no correct place to glob (the process
                    # default ~/.claude would be wrong for any caller under an
                    # isolated CLAUDE_CONFIG_DIR), so the veto is scoped to "we
                    # have a directory and the transcript is provably not in it".
                    #
                    # resume_or_fresh carries the REASON, not just the verdict:
                    # it is interpolated into both cap-hit warnings below, so a
                    # fresh retry that dropped context is distinguishable in the
                    # logs from one that never had a session to keep.  The
                    # 'resuming'/'fresh' prefix stays first so existing log
                    # greps keep matching.
                    if not result.session_id:
                        _reset_for_fresh_retry(invoke_kwargs, original_prompt)
                        await _rebuild_fresh_prompt()
                        resume_or_fresh = 'fresh (no session_id)'
                    elif config_dir is not None and not transcript_exists(
                        config_dir.path, result.session_id
                    ):
                        logger.warning(
                            f'{label}: capped session {result.session_id} has no transcript '
                            f'under {config_dir.path} — retrying FRESH instead of resuming '
                            f'into an empty session (context from this attempt is lost)',
                        )
                        _reset_for_fresh_retry(invoke_kwargs, original_prompt)
                        await _rebuild_fresh_prompt()
                        resume_or_fresh = 'fresh (transcript unreachable)'
                    else:
                        invoke_kwargs['resume_session_id'] = result.session_id
                        invoke_kwargs['prompt'] = CAP_HIT_RESUME_PROMPT
                        # Distinguish a VERIFIED resume from an unverified one:
                        # with no config_dir the transcript was never checked,
                        # so claiming 'transcript present' would be a false
                        # statement in the log.
                        resume_or_fresh = (
                            'resuming (transcript unchecked — no config dir)'
                            if config_dir is None
                            else 'resuming (transcript present)'
                        )

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
                    if isinstance(outcome, (CliLocalError, ServerError)):
                        # Two different causes, one mechanism: a zero-cost instant
                        # exit that we can POSITIVELY attribute to something other
                        # than a cap must not be counted as a cap.
                        #
                        # CliLocalError — a recognised local CLI/usage error (e.g.
                        # --session-id collision) exits zero-cost and instantly, but
                        # it is NOT a usage cap.  Counting it as a cap loops forever
                        # (reify-3604).  Falls through: Branch C retries fresh when
                        # resuming, else the failed result is returned for normal
                        # verify/steward handling.
                        #
                        # ServerError — a fast 5xx (e.g. 529 Overloaded) has exactly
                        # the same zero-cost / <=1-turn / sub-5s shape, so without
                        # this escape the net marks a perfectly HEALTHY account
                        # CAPPED and fails over pointlessly (2026-07-29 incident).
                        # This escape is what prevents that.  It does NOT fall
                        # through to Branch C: the terminal ServerError branch
                        # immediately below exits the loop first, so a 5xx never
                        # reaches the resume-fresh fallback.
                        if isinstance(outcome, ServerError):
                            logger.warning(
                                f'{label}: zero-cost instant exit is a server-side API '
                                f'error (HTTP {outcome.status}), not a cap — not '
                                f'counting as cap hit, not mutating account state',
                            )
                        else:
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
                        # attributed must be captured BEFORE slot.report(): report()
                        # bumps the account's generation, which would make the lease
                        # read as stale even for the very account it just mutated.
                        # slot.lease is None when the token was never resolved to an
                        # account (mirrors InvokeSlot.report()'s own guard at the
                        # lease_is_current call site) — treat that as unattributed
                        # rather than passing None into lease_is_current.
                        attributed = slot.lease is not None and usage_gate.lease_is_current(
                            slot.lease
                        )
                        synthetic = CapHit(
                            resets_at=None,
                            reason=f'Heuristic cap: zero-cost instant exit — {result.output[:120]}',
                        )
                        slot.report(synthetic)
                        if not attributed:
                            logger.warning(
                                f'{label}: heuristic cap suspected but no account could be marked '
                                f'(token unresolved) — treating as normal failure',
                            )
                            unattributed_cap = True
                        else:
                            consecutive_cap_hits += 1
                            full_cycles = (consecutive_cap_hits - 1) // num_accounts
                            cooldown = min(
                                _CAP_HIT_COOLDOWN_SECS * (2**full_cycles),
                                _MAX_CAP_COOLDOWN_SECS,
                            )
                            # Cannot resume a session that never ran
                            _reset_for_fresh_retry(invoke_kwargs, original_prompt)
                            await _rebuild_fresh_prompt()
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

                # Server-side API error is TERMINAL for this loop (task 3314,
                # PRD decision 4).  Server errors are NOT account-scoped — the
                # 2026-07-29 incident data showed the FRESHEST account carrying
                # the HIGHEST failure rate — so cross-account failover only
                # multiplies load on an already-degraded provider without ever
                # finding a healthy account.  The failed result goes straight
                # back to the caller, and the workflow/scheduler (PRD tasks
                # γ/β) owns the requeue, with pacing.
                #
                # `slot.confirm` (mirroring the ModelNotFound terminal branch
                # above) settles the slot as a normal completion WITHOUT any
                # cap/auth transition: "no account mutation" means no phase
                # change, not an unsettled slot.
                #
                # Placement is load-bearing in three directions:
                # - AFTER slot.detect_cap_hit, so the loop's control flow
                #   mirrors the sum type's CapHit > ServerError precedence
                #   exactly and a 429/cap-body result keeps today's cap-and-
                #   failover path byte-for-byte.
                # - AFTER the heuristic net, which keeps that net's ServerError
                #   escape live as defence-in-depth (it, not this break, is
                #   what stops a fast 529 from marking a healthy account
                #   CAPPED).
                # - BEFORE the "resume failed → retry fresh" fallback below,
                #   which would otherwise restart the invocation on a new slot
                #   — an implicit failover the PRD forbids.
                #
                # Note: because ServerError now outranks ZeroOutputWedge, the
                # wedge resume-guard above no longer fires for a timed-out 5xx.
                # This branch exits the loop instead, so the orphaned provider
                # session is still never re-resumed (PRD decision 2's intent) —
                # which is why is_zero_output_timeout itself stays deliberately
                # shape-based and untouched.
                #
                # RESIDUAL GAP (deliberately not closed here): the result this
                # branch returns still satisfies is_zero_output_timeout(), so
                # workflow.py's zero-output hang circuit breaker — which keys on
                # that predicate — still counts a 5xx-caused timeout toward
                # consecutive_zero_output and can block the task as an
                # infra_issue.  Making that consumer cause-aware is PRD task γ's
                # job, not this loop's; the hazard is closed at the cap-retry
                # layer only.
                if isinstance(outcome, ServerError):
                    logger.warning(
                        f'{label}: server-side API error (HTTP {outcome.status}) on '
                        f'account {account_name} — not account-scoped, no '
                        f'cross-account failover; returning result to caller for '
                        f'transient requeue',
                    )
                    if not unattributed_cap:
                        slot.confirm(result.cost_usd)
                    break

                # Progress-timeout guard (reify-4827, task 2360 fix #2): a
                # RESUMED invocation that hit the working-regime ceiling but
                # made real agentic progress (transcript_turns > 0) must be
                # returned to the caller, not silently discarded into the
                # generic non-cap-hit resume-failure branch below — that
                # branch restarts from the ORIGINAL prompt, throwing away the
                # transcript and all agent progress. The workflow gamma
                # branch (is_timed_out_with_progress) owns re-resuming this
                # session with its own continuation prompt. Mutually
                # exclusive with the ZeroOutputWedge guard above
                # (transcript_turns 0 vs >0), so zero-output wedges are
                # unaffected and still take the existing fresh-fallback path.
                if invoke_kwargs.get('resume_session_id') and is_timed_out_with_progress(result):
                    logger.warning(
                        f'{label}: resumed invocation timed out WITH progress '
                        f'(transcript_turns={result.transcript_turns}, '
                        f'duration_ms={result.duration_ms}) — returning to '
                        f'caller for γ re-resume instead of discarding into '
                        f'a fresh retry',
                    )
                    if not unattributed_cap:
                        slot.confirm(result.cost_usd)
                    break

                # Non-cap-hit failure while resuming → fall back to fresh invocation.
                # Rebuild via the caller's hook (no-op when rebuild_prompt is None):
                # mirrors the two cap-hit fresh-fallback paths above, since a
                # live-continuation caller's original_prompt (resume_delivers_prompt=True)
                # is only valid inside the resumed session, not a brand-new one.
                if not result.success and invoke_kwargs.get('resume_session_id'):
                    logger.warning(
                        f'{label}: resume failed (session_id={invoke_kwargs["resume_session_id"]}), '
                        f'retrying fresh',
                    )
                    _reset_for_fresh_retry(invoke_kwargs, original_prompt)
                    await _rebuild_fresh_prompt()
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


def build_claude_argv(
    *,
    model: str,
    max_budget_usd: float,
    system_prompt: str,
    max_turns: int,
    permission_mode: str,
    allowed_tools: list[str] | None,
    disallowed_tools: list[str] | None,
    mcp_config: dict | None,
    output_schema: dict | None,
    effort: str | None,
    resume_session_id: str | None,
    session_id: str | None,
    strict_mcp_config: bool = False,
) -> tuple[list[str], list[str]]:
    """Assemble the Claude CLI argv — the single source of truth shared by the
    non-sandbox (``_invoke_claude``) and sandbox (``_invoke_claude_with_sandbox``)
    invocation paths (task 2465 dedup).

    Builds the argv up to (but NOT including) any sandbox wrap, creating the
    on-disk system-prompt / mcp-config temp files it references along the way.

    ``strict_mcp_config`` (default ``False``): when ``True`` AND an
    ``mcp_config`` is supplied, ``--strict-mcp-config`` is appended right after
    the ``--mcp-config <path>`` pair. This scopes the invocation to ONLY the
    servers in the ``--mcp-config`` file, ignoring the ambient project
    ``.mcp.json`` merge — the recon-watch isolation pattern. It is the
    supervised auto-watcher rotation's guard against its capped ``escalation``
    connection (identical server name + URL as the interactive header-less
    block) bleeding into a concurrent interactive session under the non-strict
    ambient merge (task 2796, THREAD 2). The flag is emitted ONLY inside the
    ``if mcp_config:`` block, so ``strict_mcp_config=True`` with no
    ``mcp_config`` is a no-op (``--strict-mcp-config`` is meaningless with no
    ``--mcp-config``). The default ``False`` keeps every existing caller's argv
    byte-identical.

    Returns ``(cmd, temp_files)``: ``cmd`` is the assembled argv list;
    ``temp_files`` lists the temp file paths created (empty when resuming and
    no ``mcp_config`` is set).  The caller owns cleanup of a successful
    return, typically via
    ``finally: for p in temp_files: Path(p).unlink(missing_ok=True)``.

    On exception (e.g. a non-serializable ``mcp_config``), any temp files
    already created during this call are unlinked before the exception
    propagates — callers never need to clean up after a raised call.
    """
    cmd = ['claude', '--print', '--output-format', 'json']

    cmd.extend(['--model', model])
    cmd.extend(['--max-budget-usd', str(max_budget_usd)])

    temp_files: list[str] = []

    # Everything below may create on-disk temp files before raising (e.g. a
    # non-serializable mcp_config blowing up json.dump after the sysprompt
    # file already exists). Track each temp file in `temp_files` the instant
    # it's created — before writing to it — so any exception can be traced
    # back to a clean unlink of everything created so far, leaving no
    # orphaned temp files for the caller to worry about.
    try:
        # Write system prompt to temp file to avoid ARG_MAX on large payloads.
        #
        # UNCONDITIONAL — including on resume (task 3983).  This used to live in
        # the `else` below, on the belief that --system-prompt-file and --resume
        # were incompatible.  They are NOT: probed on CLI 2.1.226, the pair is
        # accepted and fails only on a nonexistent session id, i.e. past argument
        # validation.  CLI CHANGELOG 2.0.64 — "Fixed --system-prompt being ignored
        # when using --continue or --resume flags" — makes re-passing the intended
        # usage.  The system prompt is a process-invocation parameter that is never
        # persisted with the session, so omitting it on resume dropped the role
        # charter entirely and the agent silently ran under the stock Claude Code
        # prompt.
        #
        # REPLACE, not append: roles are RESTRICTIVE charters, and
        # --append-system-prompt-file would layer them over the stock
        # general-purpose identity that produced the role-disowning behaviour in
        # the first place.  Replace also keeps fresh and resumed argv
        # byte-identical.  Re-passing is a prompt-cache HIT; omitting it was a
        # total cache MISS — this is cheaper than the status quo, not costlier.
        #
        # A resumed session gets the CURRENT role prompt, not a byte-replay of the
        # original.  That is the intended semantics: no role prompt is templated
        # with per-invocation task context (that lives in the USER prompt), though
        # a few are built from live inputs that can shift between invocations —
        # recon Stage 2 branches on `project_id`, reviewer/curator prompts are
        # model-keyed artifacts, Stage 1/3 introspect live FastMCP signatures, and
        # recon-verify is tool-list templated.
        fd, sysprompt_path = tempfile.mkstemp(suffix='.txt', prefix='sysprompt_')
        temp_files.append(sysprompt_path)
        with open(fd, 'w') as f:
            f.write(system_prompt)
        cmd.extend(['--system-prompt-file', sysprompt_path])

        if resume_session_id:
            # Resume an existing session.
            cmd.extend(['--resume', resume_session_id])
        elif session_id:
            # Pre-allocate the session UUID so future --resume can find it.
            # Unlike --system-prompt-file, --session-id IS genuinely exclusive
            # with --resume, verbatim from the CLI: "--session-id can only be
            # used with --continue or --resume if --fork-session is also
            # specified."  So it stays in the branch while the system prompt
            # does not.
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
            # schema tool through.  A caller that passes no output_schema keeps
            # ``'*'`` verbatim, so all tools stay blocked.  See the deny-list
            # constant above for the keep-in-sync caveat.
            if output_schema and '*' in disallowed_tools:
                disallowed_tools = [
                    t for t in disallowed_tools if t != '*'
                ] + _REAL_BUILTIN_TOOLS_DENYLIST
            cmd.extend(['--disallowed-tools', *disallowed_tools])

        if mcp_config:
            fd, mcp_config_path = tempfile.mkstemp(suffix='.json', prefix='mcp_')
            temp_files.append(mcp_config_path)
            with open(fd, 'w') as f:
                json.dump(mcp_config, f)
            cmd.extend(['--mcp-config', mcp_config_path])
            if strict_mcp_config:
                # Scope the invocation to ONLY the --mcp-config servers,
                # ignoring the ambient .mcp.json merge (recon-watch isolation
                # pattern). Emitted here, inside `if mcp_config:`, so it is a
                # no-op with no --mcp-config to strict-scope. See the docstring.
                cmd.append('--strict-mcp-config')

        if output_schema:
            cmd.extend(['--json-schema', json.dumps(output_schema)])
    except Exception:
        for path in temp_files:
            Path(path).unlink(missing_ok=True)
        raise

    return cmd, temp_files


def apply_spawn_env(env: dict[str, str], spawn_env: dict[str, str] | None) -> None:
    """Merge spawn-identity vars into a subprocess env dict, in place.

    The single source of truth for the ``CLAUDE_SPAWN_*`` injection shared by
    the non-sandbox (``_invoke_claude``) and sandbox
    (``orchestrator.agents.invoke._invoke_claude_with_sandbox``) invocation
    paths (task 2512 dedup — mirrors the ``build_claude_argv`` split above).

    When *spawn_env* is set, its truthy-valued keys (``CLAUDE_SPAWN_ROLE`` /
    ``PROJECT`` / ``TASK_ID`` / ``PARENT_ID``) are merged into *env*; empty
    values are skipped so a blank never clobbers an inherited
    ``CLAUDE_SPAWN_*`` value. Any ``CLAUDE_SPAWN_SESSION_ID`` /
    ``CLAUDE_SPAWN_LAUNCHER_PID`` this process itself inherited (e.g. if the
    orchestrator was itself fleet-spawned) is then scrubbed from *env* —
    ``session_hooks.hook_session_slug`` prefers an inherited
    ``CLAUDE_SPAWN_SESSION_ID`` outright over reconstructing a slug from
    role/project/task_id, the exact branch ``Workflow._build_spawn_env``'s
    ``CLAUDE_SPAWN_PARENT_ID`` reconstruction assumes; leaving an inherited
    value in place would collapse every spawned agent onto ONE registry
    record instead of each getting its own.

    A no-op (including the scrub) when *spawn_env* is falsy, so callers can
    invoke this unconditionally after building the base env.
    """
    if not spawn_env:
        return
    env.update({k: v for k, v in spawn_env.items() if v})
    env.pop('CLAUDE_SPAWN_SESSION_ID', None)
    env.pop('CLAUDE_SPAWN_LAUNCHER_PID', None)


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
    spawn_env: dict[str, str] | None = None,
    startup_grace_secs: float = 120.0,
    sandbox_wrap: Callable[[list[str]], list[str]] | None = None,
    working_idle_secs: float | None = None,
    absolute_cap_secs: float | None = None,
    strict_mcp_config: bool = False,
) -> AgentResult:
    """Invoke Claude Code CLI."""
    # BEFORE build_claude_argv, which writes system-prompt / mcp-config temp
    # files: a blank prompt can never produce a useful run, so failing here
    # leaves nothing to clean up and never spawns a subprocess.
    require_non_blank_prompt(prompt, context='_invoke_claude')
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

    # User prompt is piped via stdin to avoid ARG_MAX on large payloads
    stdin_data = prompt.encode()

    # Strip ANTHROPIC_API_KEY so `claude` falls back to OAuth
    env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
    # Merge caller-supplied overrides (e.g. ANTHROPIC_BASE_URL for vLLM)
    if env_overrides:
        env.update(env_overrides)
    # Merge spawn-identity vars (CLAUDE_SPAWN_ROLE/PROJECT/TASK_ID/PARENT_ID) for
    # the SessionStart hook, and scrub any inherited CLAUDE_SPAWN_SESSION_ID/
    # LAUNCHER_PID. Applied after env_overrides so per-agent spawn identity
    # always wins. See apply_spawn_env's docstring for the full rationale
    # (shared with the sandbox path in orchestrator.agents.invoke).
    apply_spawn_env(env, spawn_env)
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

        # Apply the caller-supplied sandbox wrap (e.g. Landlock or bwrap confinement)
        # immediately before spawning.  The wrap transforms the full claude argv
        # (e.g. ['claude','--print',...]) into a sandboxed command
        # (e.g. ['python','/path/landlock_exec.py','--','claude','--print',...]).
        # Applying it here — after temp files are created — ensures the wrapped
        # command can reference those file paths.  None → no wrap.
        if sandbox_wrap is not None:
            cmd = sandbox_wrap(cmd)

        result = await _run_subprocess(
            cmd,
            cwd,
            env,
            model,
            timeout_seconds,
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
        if bridge is not None:
            await bridge.stop()


def _parse_claude_output(result: _SubprocessResult) -> AgentResult:
    """Parse Claude Code JSON output into AgentResult.

    timed_out and transcript_turns are propagated directly from result on every
    return path.
    """
    if not result.stdout.strip():
        # Distinct subtype (task 2360 fix #3): a wall-clock timeout that DID
        # make real agentic progress (transcript_turns>0) is not the same
        # failure as a genuine pre-turn wedge (transcript_turns==0/None) —
        # conflating them under 'error_empty_output' fabricates "no real work
        # done" for a productive run (reify-4827). Mirrors
        # is_timed_out_with_progress's condition inline since that predicate
        # takes an AgentResult, not this _SubprocessResult.
        #
        # Third arm (task 3143 / esc-3118-1): a NOT-timed-out fast exit whose
        # stderr carries the CLI's input-required error is a pre-turn
        # invocation REJECTION — the prompt never reached the child's stdin, so
        # the CLI exited on argument validation before contacting the API.
        # Extends the same argument: conflating a rejection ("we never asked
        # the agent anything") with an empty output ("we asked and got
        # nothing") fabricates a false narrative and makes the fixed summary
        # 'agent returned empty output' actively misdescribe the cause.
        empty_output_subtype = (
            'error_timeout_killed_with_progress'
            if result.timed_out and (result.transcript_turns or 0) > 0
            else 'error_cli_input_rejected'
            if not result.timed_out and _stderr_has_cli_input_required(result.stderr)
            else 'error_empty_output'
        )
        return AgentResult(
            success=False,
            output='Agent produced no output',
            subtype=empty_output_subtype,
            stderr=result.stderr,
            timed_out=result.timed_out,
            duration_ms=result.duration_ms,
            proc_tree=result.proc_tree,
            transcript_turns=result.transcript_turns,
            ended_awaiting_background=result.ended_awaiting_background,
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
            ended_awaiting_background=result.ended_awaiting_background,
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

    # Ended-awaiting-background downgrade (task 2761): the run exited
    # subtype=success but its transcript tail launched a backgrounded Bash
    # command that was never polled/killed — the headless one-shot session
    # abandoned still-pending work (Reify-5164 RCA).  Flip an otherwise-success
    # verdict to failure so existing non-success handling retries/resumes rather
    # than proceeding on a half-done tree.  Guarded on ``is_success`` so it is an
    # idempotent no-op on an already-failing result (the flag still propagates
    # below for classification).
    if result.ended_awaiting_background and is_success:
        is_success = False

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
            isinstance(d, dict) and d.get('tool_name') == _SCHEMA_OUTPUT_TOOL for d in denials
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
        ended_awaiting_background=result.ended_awaiting_background,
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
    # Belt-and-suspenders executability check: DF_AGENT_CPU_GOVERN is always
    # populated from CpuGovernConfig.resolved_exec_path(), which already
    # validated the path with os.access(path, os.X_OK).  shutil.which() is a
    # second layer of defence for the edge case where the path was constructed
    # outside that gate (e.g. injected directly in tests or a future caller).
    # The two predicates differ subtly (os.access honours real-UID/ACL
    # semantics; shutil.which uses the effective UID) — this is intentional
    # belt-and-suspenders redundancy, not an attempt at equivalence.  Both
    # always fail-open: a rejected path returns [] and never breaks the spawn.
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
    working_idle_secs: float | None = None,
    absolute_cap_secs: float | None = None,
) -> _SubprocessResult:
    """Run a subprocess, log output.

    *stdin_data*, when set, is piped to the process's stdin.  This avoids
    passing large payloads as command-line arguments (which hit ARG_MAX).

    *working_idle_secs* / *absolute_cap_secs*, when BOTH set, extend the
    WORKING regime past *timeout_seconds* while the transcript keeps
    advancing.  Default ``None`` for both → today's exact behavior:
    *timeout_seconds* is the flat WORKING-regime ceiling.
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
            # WORKING-regime progress extension (task 2360): last observed
            # turn count and the monotonic time it was observed increasing.
            # Both are set together, the moment seen_turn first latches, and
            # updated together whenever a later poll observes MORE turns.
            last_progress_turns: int | None = None
            last_progress_monotonic: float | None = None

            comm_task = asyncio.ensure_future(proc.communicate(input=stdin_data))

            while True:
                elapsed = time.monotonic() - watchdog_start
                # Extension engages once liveness is proven (seen_turn) AND the
                # caller opted in (both params set).  Monotonic: seen_turn only
                # ever goes False→True, so this can only turn on, never off.
                extension_engaged = (
                    seen_turn and working_idle_secs is not None and absolute_cap_secs is not None
                )
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
                    seen_turn or elapsed >= startup_grace_secs or not (config_dir and session_id)
                )
                time_to_grace = (
                    float('inf') if _grace_spent else max(0.0, startup_grace_secs - elapsed)
                )
                if extension_engaged:
                    # extension_engaged's own definition requires both params to
                    # be set (see derivation above/below) — narrow for the type
                    # checker, which cannot infer that from the bool flag alone.
                    assert working_idle_secs is not None and absolute_cap_secs is not None
                    # idle_bound: the per-role ceiling is the FLOOR of the idle
                    # window (B6 long-tool-call safety) — never smaller than
                    # today's ceiling.
                    idle_bound = (
                        max(working_idle_secs, timeout_seconds)
                        if timeout_seconds is not None
                        else working_idle_secs
                    )
                    time_to_idle_kill = (
                        max(0.0, idle_bound - (time.monotonic() - last_progress_monotonic))
                        if last_progress_monotonic is not None
                        else idle_bound
                    )
                    time_to_abs_cap = max(0.0, absolute_cap_secs - elapsed)
                    poll = max(
                        min(_WATCHDOG_WORKING_POLL_SECS, time_to_idle_kill, time_to_abs_cap),
                        _WATCHDOG_MIN_POLL_SECS,
                    )
                else:
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
                # lifetime of a healthy long-running agent — UNLESS the progress
                # extension is engaged, in which case the read is the extension's
                # own (coarse-cadence) liveness signal.
                # The post-kill transcript_turns re-read in the except block is
                # unaffected — it is a separate, one-shot read outside this loop.
                if not seen_turn and config_dir and session_id:
                    n = count_transcript_turns(config_dir, session_id)
                    if n is not None:
                        live_turns = n
                        if n >= 1:
                            seen_turn = True
                            last_progress_turns = n
                            last_progress_monotonic = time.monotonic()
                elif extension_engaged and config_dir and session_id:
                    n = count_transcript_turns(config_dir, session_id)
                    if n is not None and (last_progress_turns is None or n > last_progress_turns):
                        last_progress_turns = n
                        last_progress_monotonic = time.monotonic()

                elapsed = time.monotonic() - watchdog_start
                # Re-derive fresh (not the top-of-loop value) so a seen_turn
                # transition earlier in THIS iteration is reflected immediately.
                extension_engaged = (
                    seen_turn and working_idle_secs is not None and absolute_cap_secs is not None
                )

                # Startup-regime kill: explicit 0-turn read AND grace expired.
                # NEVER kill on None (unreadable transcript) — conservative degrade.
                if not seen_turn and live_turns == 0 and elapsed >= startup_grace_secs:
                    logger.warning(
                        f'Startup wedge detected after {elapsed:.1f}s '
                        f'(grace={startup_grace_secs}s, turns=0): '
                        f'model={model} — cancelling comm_task and killing'
                    )
                    comm_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await comm_task
                    raise TimeoutError

                if extension_engaged:
                    # extension_engaged ⟹ seen_turn ⟹ last_progress_turns/
                    # last_progress_monotonic were set atomically with seen_turn
                    # (above) and are never reset to None; working_idle_secs/
                    # absolute_cap_secs are part of extension_engaged's own
                    # definition — narrow all three for the type checker.
                    assert (
                        working_idle_secs is not None
                        and absolute_cap_secs is not None
                        and last_progress_monotonic is not None
                    )
                    idle_bound = (
                        max(working_idle_secs, timeout_seconds)
                        if timeout_seconds is not None
                        else working_idle_secs
                    )
                    idle_elapsed = time.monotonic() - last_progress_monotonic
                    if idle_elapsed >= idle_bound:
                        logger.warning(
                            f'Working-regime idle bound reached after {idle_elapsed:.1f}s '
                            f'with no new transcript turn (idle_bound={idle_bound}s, '
                            f'last progress at {last_progress_turns} turns): '
                            f'model={model} — cancelling comm_task and killing'
                        )
                        comm_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await comm_task
                        raise TimeoutError

                    if elapsed >= absolute_cap_secs:
                        logger.warning(
                            f'Working-regime absolute cap reached after {elapsed:.1f}s '
                            f'(cap={absolute_cap_secs}s): model={model} — killing'
                        )
                        comm_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await comm_task
                        raise TimeoutError
                else:
                    # Absolute-ceiling kill — today's exact behavior.  Fires
                    # only when the extension is not engaged: either param is
                    # None, OR seen_turn hasn't latched, OR (transitively) the
                    # transcript never proved readable (B7 conservative degrade).
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
                    f'Process terminated after {timeout_seconds}s timeout (SIGTERM); ' + stderr_text
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
        logger.info(
            f'Agent stdout length: {len(stdout)} bytes, first 500: {stdout_text_for_log[:500]}'
        )

    # Re-read the on-disk transcript ONCE on the normal-exit path and derive
    # BOTH signals from the same parsed records — no double file I/O (task 2761
    # amendment):
    #   • transcript_turns — the assistant-turn count surfaced in
    #     classify_agent_failure's diagnostic_detail.  Previously stamped only on
    #     the timeout path, so a normal-exit ENDED_AWAITING_BACKGROUND
    #     classification lost the turn-count signal (carried None); deriving it
    #     from the records we already parse here restores it at zero extra I/O.
    #   • ended_awaiting_background — a run that exited subtype=success but whose
    #     transcript tail launched a still-pending backgrounded Bash command
    #     silently abandoned the work.  Symmetric to the timeout path's
    #     transcript re-read above; _parse_claude_output owns the actual
    #     success→failure downgrade.
    # Both fail safe when the transcript can't be located (records None →
    # transcript_turns None, ended_awaiting_background False).
    transcript_records = (
        read_transcript_records(config_dir, session_id) if (config_dir and session_id) else None
    )
    if transcript_records is None:
        transcript_turns = None
        ended_awaiting_background = False
    else:
        transcript_turns = sum(1 for r in transcript_records if r.get('type') == 'assistant')
        ended_awaiting_background = detect_ended_awaiting_background(transcript_records)

    return _SubprocessResult(
        stdout=stdout.decode(),
        stderr=stderr_text,
        returncode=proc.returncode if proc.returncode is not None else 1,
        duration_ms=duration_ms,
        transcript_turns=transcript_turns,
        ended_awaiting_background=ended_awaiting_background,
    )
