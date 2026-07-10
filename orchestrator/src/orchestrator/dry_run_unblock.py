"""Autonomous dry-run unblock — fire-and-forget investigation at block time.

Invoked by workflow._mark_blocked when a task transitions to blocked.
Runs the unblock-auto skill read-only, then persists a proposal entry to
metadata.dry_run_proposals[] via scheduler.update_task(metadata_mode='additive')
(list-union — preserves prior entries) and trims the list to keep_last entries
via a separate update_task(default merge — overwrites dry_run_proposals wholesale
while preserving sibling keys like memory_hints/files).
Never mutates task state; all writes go through the parent process.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shared.cli_invoke import (
    AgentFailureKind,
    AllAccountsCappedException,
    classify_agent_failure,
    invoke_with_cap_retry,
    is_timed_out_with_progress,
    is_zero_output_timeout,
)
from shared.config_dir import TaskConfigDir

from orchestrator.agents.invoke import invoke_agent  # noqa: E402
from orchestrator.agents.skill_prompt import load_skill_system_prompt
from orchestrator.unblock_types import BlockClass, BlockRecord, classify_block_reason

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema and constants
# ---------------------------------------------------------------------------

DRY_RUN_PROPOSAL_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'required': ['proposal_text', 'risk_label', 'files_referenced'],
    'properties': {
        'proposal_text': {
            'type': 'string',
            'description': 'Concrete description of the proposed action and root cause.',
        },
        'risk_label': {
            'type': 'string',
            'enum': ['low', 'medium', 'human-review-required'],
        },
        'files_referenced': {
            'type': 'array',
            'items': {'type': 'string'},
            'description': 'Files that the proposed action would need to modify.',
        },
    },
    'additionalProperties': False,
}

_RISK_LABELS: frozenset[str] = frozenset({'low', 'medium', 'human-review-required'})
_HUMAN_REVIEW_REQUIRED: str = 'human-review-required'

# Stable subtype values emitted by the CLI when the cost cap fires.
# Mirrors the canonical value used in shared/src/shared/usage_gate.py and
# fused-memory/src/fused_memory/middleware/task_curator.py — the Claude CLI
# emits ``error_max_budget_usd`` (with the ``_usd`` suffix) on cost-cap
# exhaustion, not ``error_max_budget``.
_BUDGET_SUBTYPES: frozenset[str] = frozenset({'error_max_budget_usd'})

# Transient HOST infra wedges (e.g. a pre-turn-1 kill at the
# UnblockAutoConfig.timeout ceiling -> empty stdout + timed_out) are
# retryable and operationally distinct from a substantive
# 'investigation_failed' agent conclusion. api_error is deliberately NOT
# included: it is handled at another layer, not treated as infra here.
#
# NOTE: classification is by AgentFailureKind alone (timed_out=True ->
# TIMED_OUT) and is intentionally NOT gated on transcript_turns — a genuine
# investigation that ran many turns before hitting the timeout ceiling is
# classified identically to a pre-turn-1 wedge (both land here as
# 'infra_failure'). This is safe today because risk_label stays
# 'human-review-required' either way, so no auto-retry/auto-unblock path is
# opened by this status alone (see TestTwoWayBoundary.
# test_infra_failure_and_investigation_failed_both_aborted in
# test_b3_gate.py). A future consumer that auto-retries specifically on
# infra_failure should itself gate on transcript_turns rather than assume
# near-zero turns. Current behavior is pinned by TestInfraFailureClassification.
# test_high_turn_timeout_still_classified_infra_failure in
# test_dry_run_unblock.py.
_INFRA_FAILURE_KINDS: frozenset[AgentFailureKind] = frozenset({
    AgentFailureKind.TIMED_OUT,
    AgentFailureKind.EMPTY_OUTPUT,
})

# Outer wall-clock sanity bound for invoke_with_cap_retry's cap-hit patience
# (shared/src/shared/cli_invoke.py cap-wait policy table). This is a
# fire-and-forget, best-effort block-time investigation — it must NOT inherit
# the 14-day default, which would leave a background task pending for weeks
# under a sustained cap storm. Mirrors the 1800s reconciliation-stage
# precedent (short-lived stage runners; see cli_invoke.py:82-94).
_DRY_RUN_CAP_WAIT_SANITY_SECS: float = 1800.0


def _is_budget_exhausted(result: Any, budget_usd: float) -> bool:  # noqa: ARG001
    """Return True only when the agent's subtype explicitly signals budget exhaustion.

    We avoid the ``cost >= budget_usd`` heuristic: an agent can spend close to
    the cap before failing for an unrelated reason (e.g. max_turns, tool error),
    which would mis-classify those failures as 'budget_exhausted' and produce
    misleading operator-visible status fields in metadata.
    """
    subtype = getattr(result, 'subtype', '') or ''
    return subtype in _BUDGET_SUBTYPES


def _failure_diagnostics(result: Any) -> dict[str, Any]:
    """Build the discrete, queryable diagnostic fields stamped on failure entries.

    None-safe: when *result* is None (the orchestrator-side exception fallback
    in ``run_dry_run_unblock``, where no ``AgentResult`` was ever produced),
    every key is present with a ``None`` value — preserving shape parity
    across all four entry shapes (ok, investigation_failed, budget_exhausted,
    exception fallback).

    ``stderr_tail`` mirrors ``classify_agent_failure``'s 500-char tail
    convention (``shared/src/shared/cli_invoke.py``).

    These keys are orchestrator-stamped OUTSIDE ``DRY_RUN_PROPOSAL_SCHEMA``
    (``additionalProperties: False``) — the same pattern as head_sha/main_sha
    (task 1613) — so the agent cannot forge them.
    """
    if result is None:
        return {
            'timed_out': None,
            'duration_ms': None,
            'subtype': None,
            'transcript_turns': None,
            'session_id': None,
            'stderr_tail': None,
        }
    stderr = getattr(result, 'stderr', '') or ''
    return {
        'timed_out': bool(getattr(result, 'timed_out', False)),
        'duration_ms': getattr(result, 'duration_ms', None),
        'subtype': getattr(result, 'subtype', '') or '',
        'transcript_turns': getattr(result, 'transcript_turns', None),
        'session_id': getattr(result, 'session_id', '') or '',
        'stderr_tail': stderr[-500:],
    }


# Read-only tools the dry-run agent is allowed to use.
# Bash(pytest:*) and Bash(cargo:*) are intentionally omitted: both can
# write .pyc/__pycache__/target/ files or fetch from the network, which
# contradicts the read-only safety contract advertised in SKILL.md.
# The agent can read existing test output from .task/iterations.jsonl instead.
# Explicit read-only git subcommands instead of a wildcard — Bash(git:*) would
# permit mutating subcommands (commit, push, reset, checkout, restore, clean)
# which contradict the read-only contract for this autonomous, no-human-
# checkpoint invocation where the threat profile is higher than interactive use.
_ALLOWED_TOOLS: list[str] = [
    'Read',
    'Glob',
    'Grep',
    'Bash(git log:*)',
    'Bash(git diff:*)',
    'Bash(git status:*)',
    'Bash(git show:*)',
    'Bash(git rev-parse:*)',
    'Bash(git branch:*)',
    'Bash(git ls-files:*)',
    'mcp__fused-memory__search',
    'mcp__fused-memory__get_task',
    'mcp__fused-memory__get_tasks',
]

# Mutating tools explicitly blocked.
# Defence-in-depth: list the most dangerous git mutations alongside the
# write-capable MCP tools so the allowlist and denylist are consistent.
_DISALLOWED_TOOLS: list[str] = [
    'Edit',
    'Write',
    'Bash(git commit:*)',
    'Bash(git push:*)',
    'Bash(git reset:*)',
    'Bash(git checkout:*)',
    'Bash(git restore:*)',
    'Bash(git clean:*)',
    'Bash(git merge:*)',
    'Bash(git rebase:*)',
    'mcp__fused-memory__set_task_status',
    'mcp__fused-memory__update_task',
    'mcp__fused-memory__delete_memory',
    'mcp__fused-memory__remove_task',
]

# ---------------------------------------------------------------------------
# SHA capture helper
# ---------------------------------------------------------------------------

async def _capture_worktree_shas(worktree: str) -> tuple[str | None, str | None]:
    """Capture HEAD and main branch SHAs from the worktree via git.

    Fully defensive: any non-zero exit code, subprocess exception, or
    non-repo worktree yields None for that sha — never raises.
    Both shas are always returned (keys always present in the stamped entry)
    so the entry shape stays consistent across repo and non-repo worktrees.
    """
    async def _rev_parse(ref: str) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                'git', '-C', worktree, 'rev-parse', ref,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return None
            return stdout.decode().strip() or None
        except Exception:
            return None

    head_sha, main_sha = await asyncio.gather(_rev_parse('HEAD'), _rev_parse('main'))
    return head_sha, main_sha


# ---------------------------------------------------------------------------
# SKILL.md loader
# ---------------------------------------------------------------------------

def _load_skill_system_prompt() -> str:
    """Load skills/unblock-auto/SKILL.md from repo root, strip frontmatter.

    Delegates to the shared ``load_skill_system_prompt`` helper so the
    parent-walk and frontmatter-strip logic is not duplicated here.
    Raises FileNotFoundError with a clear message if the file cannot be found.
    """
    return load_skill_system_prompt('unblock-auto')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_dry_run_unblock(
    *,
    task_id: str,
    worktree: str,
    reason: str,
    detail: str,
    scheduler: Any,
    mcp: Any,
    config: Any,
    event_store: Any = None,
    usage_gate: Any = None,
    cost_store: Any = None,
    block_class: BlockClass | None = None,
) -> None:
    """Investigate a blocked task and write a proposal to metadata.

    Fire-and-forget: called via asyncio.create_task from _mark_blocked.
    Any exception is caught and written as a fallback entry so failures
    are always visible in metadata.dry_run_proposals[].

    *block_class* lets the caller supply the entry's typed BlockClass
    explicitly (e.g. workflow._spawn_dry_run_unblock classifies `reason`
    up front); when omitted (None), it is derived via
    ``classify_block_reason(reason)`` as a defensive fallback.
    """
    import time

    ua_cfg = config.unblock_auto
    start_time = time.monotonic()
    project_id = getattr(getattr(config, 'fused_memory', None), 'project_id', '') or ''
    run_id = getattr(event_store, 'run_id', '') or ''

    # Capture worktree SHAs before invoke_agent so both the success and
    # exception paths can stamp the same anchor values.
    head_sha, main_sha = await _capture_worktree_shas(worktree)

    config_dir: TaskConfigDir | None = None
    preserve_config_dir = False
    try:
        # Per-investigation isolated CLAUDE_CONFIG_DIR, named distinctly from
        # the main task's `claude-config-{task_id}` dir so this background
        # investigation cannot collide with the blocked task's own session/
        # credentials. Passing config_dir + session_id to invoke_with_cap_retry
        # (below) revives the startup-wedge fast-kill in
        # shared/cli_invoke.py._run_subprocess — without both, `_grace_spent`
        # is forced True and a pre-turn-1 wedge burns the full timeout ceiling.
        config_dir = TaskConfigDir(f'{task_id}-unblock', base_dir=Path(worktree) / '.task')

        system_prompt = _load_skill_system_prompt()
        user_prompt = (
            f'Task ID: {task_id}\n'
            f'Worktree: {worktree}\n'
            f'Block reason: {reason}\n'
            f'Detail: {detail or "(none)"}\n\n'
            'Investigate and emit your structured proposal.'
        )

        mcp_config = mcp.mcp_config_json() if mcp is not None else None

        async def _one_attempt(session_id: str) -> Any:
            return await invoke_with_cap_retry(
                usage_gate=usage_gate,
                label=f'Task {task_id} [unblock_auto]',
                config_dir=config_dir,
                cost_store=cost_store,
                run_id=run_id,
                task_id=task_id,
                project_id=project_id,
                role='unblock_auto',
                cap_wait_sanity_secs=_DRY_RUN_CAP_WAIT_SANITY_SECS,
                invoke_fn=invoke_agent,
                backend=ua_cfg.backend,
                session_id=session_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
                cwd=Path(worktree),
                model=ua_cfg.model,
                max_turns=ua_cfg.max_turns,
                max_budget_usd=ua_cfg.budget_usd,
                allowed_tools=_ALLOWED_TOOLS,
                disallowed_tools=_DISALLOWED_TOOLS,
                mcp_config=mcp_config,
                output_schema=DRY_RUN_PROPOSAL_SCHEMA,
                effort=ua_cfg.effort,
                timeout_seconds=ua_cfg.timeout_seconds,
                # Working-regime progress extension (task 2360, reify-4827):
                # config_dir + session_id (above) already revive the startup
                # watchdog, so once the transcript shows turn-1 the watchdog
                # can poll it — these two params let a genuinely productive
                # investigation run past ua_cfg.timeout_seconds instead of
                # being ceiling-killed and reproducing the very failure it is
                # meant to diagnose.
                working_idle_secs=config.timeouts.working_idle_secs,
                absolute_cap_secs=config.invocation_timeout,
            )

        try:
            result = await _one_attempt(str(uuid.uuid4()))
            if is_zero_output_timeout(result):
                # Transcript-authoritative pre-turn-1 wedge: retry EXACTLY once
                # with a freshly-allocated session_id. Never reuse the wedged
                # UUID — a possibly-committed session id makes the CLI's
                # --session-id exit instantly with 'already in use' (reify-3604 /
                # _reset_for_fresh_retry semantics, shared/cli_invoke.py:562).
                logger.warning(
                    'dry_run_unblock: zero-output timeout wedge for task %s — '
                    'retrying once with a fresh session',
                    task_id,
                )
                result = await _one_attempt(str(uuid.uuid4()))

            entry = _build_entry(result, reason=reason, budget_usd=ua_cfg.budget_usd)
        except AllAccountsCappedException as cap_exc:
            # A retryable infra wedge, not a substantive investigation
            # conclusion — must NOT surface as 'investigation_failed' (that
            # shape is terminal for the B3 low-risk auto-unblock path).
            logger.warning(
                'dry_run_unblock: all accounts capped for task %s: %s',
                task_id, cap_exc,
            )
            entry = _cap_exhausted_entry(reason=reason, exc=cap_exc)
            result = None

        preserve_config_dir = result is not None and is_zero_output_timeout(result)

    except Exception as exc:
        logger.warning('dry_run_unblock: unexpected error for task %s: %s', task_id, exc)
        entry = {
            'status': 'investigation_failed',
            'proposal_text': f'Investigation failed (unexpected error): {exc}',
            'risk_label': _HUMAN_REVIEW_REQUIRED,
            'files_referenced': [],
            'block_reason': reason,
            'cost_usd': 0.0,
            'investigated_at': _now_iso(),
            'timestamp': _now_iso(),
            **_failure_diagnostics(None),
        }
        result = None
    finally:
        if config_dir is not None:
            if preserve_config_dir:
                # Not independently reaped here: this dir lives under
                # <worktree>/.task/, which GitOps.cleanup_worktree() removes
                # wholesale (`git worktree remove --force`) when the worktree
                # is torn down — see the .task/ contamination-prevention
                # notes atop git_ops.py. The forensic window is bounded by
                # the worktree's lifetime, not unbounded.
                logger.warning(
                    'dry_run_unblock: config dir preserved for forensic analysis '
                    '(doubly-wedged investigation) for task %s → %s',
                    task_id, config_dir.path,
                )
            else:
                try:
                    config_dir.cleanup()
                except Exception as exc:
                    logger.warning(
                        'dry_run_unblock: failed to clean up config dir for task %s: %s',
                        task_id, exc,
                    )

    # Stamp the git anchor and typed block_class onto the entry at a single
    # point so all six shapes (ok, investigation_failed, budget_exhausted,
    # infra_failure, exception fallback, cap-exhausted) carry them.
    # block_class joins head_sha/main_sha as an orchestrator-stamped,
    # non-forgeable field: DRY_RUN_PROPOSAL_SCHEMA stays unchanged
    # (additionalProperties:False, no block_class/sha keys) so the agent
    # cannot forge any of these. entry.update(...) is additive — sibling
    # keys (proposal_text/block_reason/timestamp/status/cost_usd/
    # diagnostics) are left untouched; risk_label/files_referenced/
    # investigated_at are idempotently re-set to the values already on entry.
    resolved_class = block_class if block_class is not None else classify_block_reason(reason)
    record = BlockRecord(
        block_class=resolved_class,
        risk_label=entry.get('risk_label', _HUMAN_REVIEW_REQUIRED),
        head_sha=head_sha,
        main_sha=main_sha,
        files_referenced=entry.get('files_referenced', []),
        investigated_at=entry.get('investigated_at', ''),
    )
    if 'status' in entry and record.risk_label == 'low':
        # Cross-module producer invariant that b3_gate.check_proposal's typed
        # path depends on (see b3_gate.py check_proposal step 3 docstring):
        # every failure shape ('status' key present) must set risk_label !=
        # 'low', because on the typed path (block_class is not None) step 3's
        # status-sniff is intentionally skipped in favor of step 2 (risk_label
        # != 'low' -> abort). All current failure shapes here (_build_entry,
        # _cap_exhausted_entry, the exception fallback above) hardcode
        # risk_label=_HUMAN_REVIEW_REQUIRED, so this should be unreachable —
        # log loudly (not raise/assert) if a future producer shape violates
        # it, since this is a fire-and-forget background investigation that
        # must not crash on its own bug, but a silent violation would let a
        # failure entry slip past check_proposal straight to FRESH.
        logger.error(
            'dry_run_unblock: invariant violated for task %s — failure entry '
            "(status=%r) has risk_label='low'; b3_gate.check_proposal's typed "
            'dual-read path (step 3) relies on failure shapes never being '
            'low-risk',
            task_id, entry.get('status'),
        )
    entry.update(record.to_dict())

    try:
        await scheduler.update_task(
            task_id,
            {'dry_run_proposals': [entry]},
            append=True,
        )
    except Exception as exc:
        logger.error('dry_run_unblock: failed to persist proposal for task %s: %s', task_id, exc)

    # Best-effort keep-last-N trim: read the full metadata blob, slice
    # dry_run_proposals to the most recent keep_last entries, and rewrite the
    # whole blob via the default merge mode (shallow last-write-wins).  Merge
    # overwrites the dry_run_proposals list wholesale (achieving the trim) while
    # preserving sibling keys like memory_hints/files — strictly safer than
    # replace (which would delete-by-omission any sibling written concurrently).
    # Wrapped in its own try/except — a trim failure never crashes the hook.
    # keep_last <= 0 disables trimming.
    # Note: existing MagicMock-scheduler tests stay green because their
    # scheduler.get_task is non-awaitable — raises TypeError, is caught here,
    # and only the single additive call persists.
    #
    # Concurrency note: this read-modify-write is not atomic. If another writer
    # mutates the task metadata between get_task and the trim update_task, that
    # concurrent change is clobbered (stale-plus-trimmed blob wins).
    # Concurrent dry-runs on the same blocked task are extremely unlikely — the
    # scheduler dispatches at most one hook per task — so this is acceptable as
    # a best-effort trim. A missed trim cycle is preferable to coordination
    # overhead.
    try:
        keep_last = ua_cfg.b3_proposal_keep_last
        if keep_last and keep_last > 0:
            task = await scheduler.get_task(task_id)
            if task:
                metadata = task.get('metadata') or {}
                proposals = metadata.get('dry_run_proposals') or []
                if len(proposals) > keep_last:
                    metadata['dry_run_proposals'] = proposals[-keep_last:]
                    await scheduler.update_task(task_id, metadata)
    except Exception as exc:
        logger.warning(
            'dry_run_unblock: trim failed for task %s (best-effort, continuing): %s',
            task_id, exc,
        )

    if event_store and result is not None:
        try:
            from orchestrator.event_store import EventType
            event_store.emit(
                EventType.invocation_end,
                task_id=task_id,
                phase='blocked',
                role='unblock_auto',
                cost_usd=getattr(result, 'cost_usd', 0.0),
                duration_ms=getattr(result, 'duration_ms', int((time.monotonic() - start_time) * 1000)),
                data={
                    'dry_run': True,
                    'risk_label': entry.get('risk_label', _HUMAN_REVIEW_REQUIRED),
                    'success': getattr(result, 'success', False),
                    'status': entry.get('status', 'ok'),
                },
            )
        except Exception as exc:
            logger.warning('dry_run_unblock: failed to emit event for task %s: %s', task_id, exc)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _build_entry(result: Any, *, reason: str, budget_usd: float) -> dict[str, Any]:
    """Convert an AgentResult to a proposal entry dict."""
    now = _now_iso()

    if not result.success:
        diagnostics = _failure_diagnostics(result)
        if _is_budget_exhausted(result, budget_usd):
            return {
                'status': 'budget_exhausted',
                'proposal_text': (
                    f'Budget exhausted, proposal incomplete. '
                    f'cost_usd={result.cost_usd:.3f}, subtype={result.subtype}'
                ),
                'risk_label': _HUMAN_REVIEW_REQUIRED,
                'files_referenced': [],
                'block_reason': reason,
                'cost_usd': result.cost_usd,
                'investigated_at': now,
                'timestamp': now,
                **diagnostics,
            }
        failure_cls = classify_agent_failure(result)
        if failure_cls.kind in _INFRA_FAILURE_KINDS:
            proposal_text = (
                f'Infra wedge (retryable, not a human-review conclusion): '
                f'{failure_cls.summary}; subtype={result.subtype}'
            )
            if is_timed_out_with_progress(result):
                # Truthful reporting (task 2360 fix #3/reify-4827): the bare
                # "Infra wedge (retryable...)" framing above reads as a
                # contradiction next to a many-turn productive run — append
                # an explicit marker that routes the reader toward "raise
                # the wall / task is slow" rather than "retryable infra".
                proposal_text += (
                    f' (timed_out=True, transcript_turns={result.transcript_turns} '
                    f'— productive wall-clock kill, not a wedge; raise the '
                    f'wall / task is slow)'
                )
            return {
                'status': 'infra_failure',
                'proposal_text': proposal_text,
                'risk_label': _HUMAN_REVIEW_REQUIRED,
                'files_referenced': [],
                'block_reason': reason,
                'cost_usd': getattr(result, 'cost_usd', 0.0),
                'investigated_at': now,
                'timestamp': now,
                **diagnostics,
            }
        return {
            'status': 'investigation_failed',
            'proposal_text': (
                f'Investigation failed: subtype={result.subtype}; '
                f'output={result.output[:200]}'
            ),
            'risk_label': _HUMAN_REVIEW_REQUIRED,
            'files_referenced': [],
            'block_reason': reason,
            'cost_usd': getattr(result, 'cost_usd', 0.0),
            'investigated_at': now,
            'timestamp': now,
            **diagnostics,
        }

    # Success path — parse structured_output
    raw = result.structured_output or {}
    risk = raw.get('risk_label', '')
    if risk not in _RISK_LABELS:
        risk = _HUMAN_REVIEW_REQUIRED

    return {
        'proposal_text': raw.get('proposal_text', ''),
        'risk_label': risk,
        'files_referenced': raw.get('files_referenced', []),
        'block_reason': reason,
        'investigated_at': now,
        'timestamp': now,
    }


def _cap_exhausted_entry(*, reason: str, exc: AllAccountsCappedException) -> dict[str, Any]:
    """Build the retryable infra_failure entry for a cap-exhausted investigation.

    AllAccountsCappedException means invoke_with_cap_retry exceeded the
    ``_DRY_RUN_CAP_WAIT_SANITY_SECS`` bound (all accounts capped) — no
    AgentResult was ever produced, so this reuses the None-safe
    ``_failure_diagnostics(None)`` shape rather than the AgentResult-derived
    one, matching the task-2020 infra_failure entry shape.
    """
    now = _now_iso()
    return {
        'status': 'infra_failure',
        'proposal_text': (
            f'Infra wedge (retryable, not a human-review conclusion): '
            f'all accounts capped after {exc.retries} retries '
            f'({exc.elapsed_secs:.1f}s elapsed)'
        ),
        'risk_label': _HUMAN_REVIEW_REQUIRED,
        'files_referenced': [],
        'block_reason': reason,
        'cost_usd': 0.0,
        'investigated_at': now,
        'timestamp': now,
        **_failure_diagnostics(None),
    }
