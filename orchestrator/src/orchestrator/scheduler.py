"""Task selection and module lock management."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from shared.locking import files_to_modules, modules_conflict, normalize_lock
from shared.mcp_envelope import parse_tool_result, resolver_failed

from orchestrator.config import (
    DEFAULT_TIER,
    PRIORITY_RANK,
    PRIORITY_TIERS,
    TIER_BASE,
    TIER_WIDTH,
    OrchestratorConfig,
    coerce_tier,
)
from orchestrator.event_store import EventStore, EventType
from orchestrator.mcp_lifecycle import mcp_call
from orchestrator.overrides import OverrideRow, OverrideStore
from orchestrator.task_status import ACTIVE_TASK_STATUSES, TERMINAL_STATUSES

# task_skipped events for "effectively infinite" skip thresholds (>= this
# value) are rate-limited to a geometric schedule so the event store is not
# flooded with diagnostics for tasks that will perpetually lose the race.
_INF_SKIP_THRESHOLD: int = 1000
_GEOMETRIC_SKIP_EMIT_COUNTS: frozenset[int] = frozenset({1, 10, 100, 1000, 10000})

# set_task_status transient-failure retry parameters.  Three attempts with
# 1.5s, 3s gaps lets the taskmaster child finish a typical reconnect window
# (~2-4s observed) before giving up.
_TRANSIENT_RETRIES: int = 3
_TRANSIENT_BACKOFF_BASE: float = 1.5

logger = logging.getLogger(__name__)

__all__ = [
    'normalize_lock',
    'files_to_modules',
    'McpSessionLike',
    'RequeueRecord',
    'TaskAssignment',
    'ModuleLockTable',
    'Scheduler',
    'SetTaskStatusRejected',
    'TerminalExitRejection',
    'DoneGateRejection',
    'ProvenanceValidationRejection',
    'ExternalResolverError',
    'extract_rejection',
    'extract_structured_rejection',
    'is_transient_rejection',
    'is_transient_api_requeue',
]


# Server-side terminal statuses (mirrors fused-memory's TERMINAL_STATUSES).
# Used to classify a ``terminal_exit_rejected`` response as a logical
# contradiction (terminal -> non-terminal with no reopen_reason) versus a
# benign side-effect of an idempotent terminal -> terminal write.
_TERMINAL_STATUSES = frozenset({'done', 'cancelled'})


class SetTaskStatusRejected(Exception):
    """Base class for non-transient set_task_status rejections.

    Catch this to handle the whole family (terminal-exit, phantom-done,
    provenance) uniformly. Subclasses carry typed fields for callers that
    need to react to a specific kind. Callers that previously relied on
    the silent-WARNING-and-return behaviour must now wrap the call.
    """

    def __init__(self, task_id: str, error_code: str, raw: str, message: str | None = None):
        self.task_id = task_id
        self.error_code = error_code
        self.raw = raw
        super().__init__(
            message
            or f'set_task_status({task_id!r}) rejected: {error_code} — {raw}'
        )


class TerminalExitRejection(SetTaskStatusRejected):
    """Server's terminal-exit gate refused a non-terminal write because the row is terminal.

    Raised only when the rejection is a logical contradiction (caller asked
    for a non-terminal status with no reopen_reason on a terminal row).
    Callers that need to write 'pending' / 'blocked' / 'in-progress' must
    catch this and decide whether to reopen explicitly or accept the
    terminal state. The most important caller is
    ``workflow._mark_blocked`` which uses the exception as the trip-wire
    to detect an out-of-band ``update_task(status='done')`` bypass.
    """

    def __init__(self, task_id: str, old_status: str, target_status: str, raw: str):
        self.old_status = old_status
        self.target_status = target_status
        super().__init__(
            task_id=task_id,
            error_code='terminal_exit_rejected',
            raw=raw,
            message=(
                f'set_task_status({task_id!r}, {target_status!r}) refused: '
                f'task is currently {old_status!r} (terminal-exit gate)'
            ),
        )


class DoneGateRejection(SetTaskStatusRejected):
    """Phantom-done gate refused: ``metadata.files`` lists missing paths.

    Raised when fused-memory returns ``error == 'done_gate_missing_files'``.
    The workflow's happy-path handler catches this, logs honestly, and
    routes the task to ``_mark_blocked`` so the architect's claim does not
    silently disagree with the persistence layer.
    """

    def __init__(self, task_id: str, missing_files: list[str], raw: str):
        self.missing_files = list(missing_files)
        super().__init__(
            task_id=task_id,
            error_code='done_gate_missing_files',
            raw=raw,
            message=(
                f'set_task_status({task_id!r}, "done") refused: '
                f'phantom-done gate — missing files: {missing_files!r}'
            ),
        )


class ProvenanceValidationRejection(SetTaskStatusRejected):
    """done_provenance validation refused the transition.

    Raised for both ``done_provenance_required`` and
    ``done_provenance_invalid`` (unresolved commit, branch-only SHA, etc).
    The error_code field distinguishes the two server-side codes.
    """

    def __init__(self, task_id: str, error_code: str, raw: str):
        super().__init__(
            task_id=task_id,
            error_code=error_code,
            raw=raw,
            message=(
                f'set_task_status({task_id!r}, "done") refused: '
                f'{error_code} — {raw}'
            ),
        )

class ExternalResolverError(RuntimeError):
    """Synthesised error: ``get_external_statuses`` returned an unusable result.

    Raised (into the error slot of the ``(statuses, error)`` tuple) in two cases:

    1. **Non-dict / unparseable result** — ``parse_tool_result`` returned an
       error (missing 'statuses' key, wrong type, or JSON parse failure).
       The returned statuses dict is ``{}``.

    2. **Partial-result (missing keys)** — the 'statuses' dict was present but did
       not contain a key for every requested dep string.  The returned statuses dict
       is the partial dict (not ``{}``) so callers can log what was received, but
       the error flag forces a fail-safe wait (do not silently treat missing keys
       as non-done statuses).

    In both cases ``_external_resolver_failed`` becomes ``True`` via the existing
    ``external_err is not None`` plumbing — no gate-logic changes needed.
    """


# Error-type names that indicate a transient backend failure (taskmaster
# child reconnecting, fused-memory crashed mid-call, network blip).  Matches
# substring so wrapper-formatted strings like ``"TimeoutError(...)"`` and
# ``"asyncio.TimeoutError"`` both classify as transient.
TRANSIENT_ERROR_TYPES = (
    'TimeoutError',
    'ConnectionError',
    'ConnectError',
    'ReadTimeout',
    'WriteTimeout',
    'asyncio.TimeoutError',
)


def extract_rejection(response: Any) -> str | None:
    """Return a one-line description of a fused-memory rejection, or None on success.

    fused-memory's set_task_status returns structured error dicts (terminal-exit
    gate, phantom-done gate, done_provenance validation) without raising. The
    payload arrives wrapped in MCP envelope:
    ``{result: {structuredContent: {...}, content: [{type: text, text: <json>}], isError: bool}}``.

    A rejection has either ``error`` (truthy) or ``success: False`` in the payload.
    A no-op (same-status) returns ``success: True, no_op: True`` and is treated
    as success.
    """
    if not isinstance(response, dict):
        return None
    result = response.get('result', response)
    if not isinstance(result, dict):
        return None
    payload = result.get('structuredContent')
    if not isinstance(payload, dict):
        # Fall back to parsing the text content block.
        for block in result.get('content', []) or []:
            if isinstance(block, dict) and block.get('type') == 'text':
                try:
                    payload = json.loads(block.get('text') or '')
                except (ValueError, TypeError):
                    payload = None
                break
    if not isinstance(payload, dict):
        return None
    if payload.get('error'):
        hint = payload.get('hint') or payload.get('error_type') or ''
        return f'{payload["error"]}{" — " + hint if hint else ""}'
    if payload.get('success') is False:
        return f'success=False payload={payload!r}'
    return None


def extract_structured_rejection(response: Any) -> dict | None:
    """Return the structured rejection payload as a dict, or None on success.

    Parallel to :func:`extract_rejection`, but returns the raw payload so
    callers can inspect typed fields (``error``, ``from_status``, …) without
    re-parsing the rendered string. Used by ``set_task_status`` to classify
    ``terminal_exit_rejected`` responses as logical contradictions worthy of
    raising ``TerminalExitRejection``.
    """
    if not isinstance(response, dict):
        return None
    result = response.get('result', response)
    if not isinstance(result, dict):
        return None
    payload = result.get('structuredContent')
    if not isinstance(payload, dict):
        for block in result.get('content', []) or []:
            if isinstance(block, dict) and block.get('type') == 'text':
                try:
                    payload = json.loads(block.get('text') or '')
                except (ValueError, TypeError):
                    payload = None
                break
    if not isinstance(payload, dict):
        return None
    if payload.get('error'):
        return payload
    if payload.get('success') is False:
        return payload
    return None


def is_transient_rejection(rejection: str | None) -> bool:
    """True when ``rejection`` text names a known transient backend error.

    Used by the ``set_task_status`` retry loop to distinguish recoverable
    backend failures (taskmaster child restarting, fused-memory crashed
    mid-call) from terminal-business rejections (phantom-done gate,
    done_provenance validation, terminal-exit gate).  Transient rejections
    are retried; everything else is surfaced as a warning and returned.
    """
    if not rejection:
        return False
    return any(name in rejection for name in TRANSIENT_ERROR_TYPES)


# Marker produced solely by shared.cli_invoke.classify_agent_failure for
# AgentFailureKind.API_ERROR (cli_invoke.py:362-366).  The planning and
# execution phases write it into block_reason (workflow.py:2222/2298);
# _run_simple_task (workflow.py:2599) uses it only as an internal REQUEUED
# fall-through sentinel and does NOT write block_reason directly — that is
# done later by the architect phase (workflow.py:2222/2298).
_API_ERROR_REASON_RE = re.compile(r'agent API error: HTTP (\d{3})')


def is_transient_api_requeue(reason: str | None) -> bool:
    """True when *reason* encodes a transient server-side (HTTP 5xx) API error.

    Matches the ``"agent API error: HTTP <status>"`` marker (present in
    block_reason regardless of workflow phase) and classifies HTTP 5xx
    (500-599, including 529 Overloaded) as transient.  HTTP 4xx (client/auth
    errors) and non-API reasons return False and still count against
    ``requeue_cap``.

    Note: HTTP 429 (rate-limit / too-many-requests) is intentionally
    classified as non-transient.  Unlike a server-side 5xx overload that
    resolves on its own, a 429 signals a quota or rate-limiting configuration
    problem that benefits from human review.  To change this policy, also
    update the ``test_false_for_non_transient`` parametrize list and the
    design decision in plan.json.
    """
    if not reason:
        return False
    m = _API_ERROR_REASON_RE.search(reason)
    if m is None:
        return False
    return 500 <= int(m.group(1)) <= 599


@dataclass(frozen=True)
class RequeueRecord:
    """A single REQUEUED outcome for a task, tracked for the retry cap.

    Instances accumulate in ``Scheduler._requeue_history`` until a DONE
    outcome clears them or the cap is hit and a cap-exhaust escalation is
    submitted.  Fields mirror the cap-exhaust report layout.
    """

    attempt: int
    phase: str
    reason: str
    detail: str
    run_id: str
    cost_usd: float
    timestamp: float


def _render_retry_cap_report(
    *,
    task_id: str,
    run_id: str,
    cap: int,
    history: list[RequeueRecord],
    cost_usd: float,
) -> str:
    """Render the markdown report artifact for a retry-cap exhaustion event.

    Progressive-disclosure layout: header with totals, per-attempt timeline,
    then a documented SQL query for digging deeper into the event_store.
    """
    exhausted_at = datetime.now(UTC).isoformat()
    lines = [
        f'# Retry Cap Exhausted: task {task_id}',
        '',
        f'- **Run ID:** {run_id}',
        f'- **Cap:** {cap}',
        f'- **Attempts recorded:** {len(history)}',
        f'- **Cost-to-date (this run):** ${cost_usd:.2f}',
        f'- **Exhausted at:** {exhausted_at}',
        '',
        '## Timeline',
        '',
    ]
    if not history:
        lines.append('_No attempts recorded._')
        lines.append('')
    for record in history:
        iso = datetime.fromtimestamp(record.timestamp, UTC).isoformat()
        lines.extend([
            f'### Attempt {record.attempt} — phase={record.phase} at {iso}',
            '- **Outcome:** REQUEUED',
            f'- **Block reason:** {record.reason}',
            f'- **Block detail:** {record.detail[:500]}',
            f'- **Cost (this attempt):** ${record.cost_usd:.2f}',
            '',
        ])
    lines.extend([
        '## Dig deeper',
        '',
        f'Full phase/outcome stream for task `{task_id}` in run `{run_id}`:',
        '',
        '```sql',
        '-- Run from the project root:',
        'sqlite3 data/orchestrator/runs.db \\',
        "  \"SELECT timestamp, event_type, phase, role, data \\",
        "   FROM events \\",
        f"   WHERE task_id='{task_id}' AND run_id='{run_id}' \\",
        '   ORDER BY timestamp\"',
        '```',
        '',
    ])
    return '\n'.join(lines)


class McpSessionLike(Protocol):
    """Structural interface for MCP session objects.

    Both ``McpSession`` (production) and ``_StubMcpSession`` (eval mode)
    conform to this interface.  Typing the optional *mcp_session* kwarg on
    ``Scheduler.__init__`` against this Protocol lets pyright verify that any
    injected stub actually provides the expected API — a mis-shaped stub will
    be caught at type-check time rather than at runtime.
    """

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        timeout: float = ...,
    ) -> dict: ...


@dataclass
class TaskAssignment:
    """A task that has been assigned to a workflow slot, with module locks held."""

    task_id: str
    task: dict
    modules: list[str]


def _resolve_time_source(ts: Callable[[], float] | None) -> Callable[[], float]:
    """Return *ts* if provided, else the stdlib ``time.monotonic`` callable.

    Centralises the ``None``-fallback logic so ``ModuleLockTable`` and
    ``Scheduler`` don't each inline a redundant lambda.
    """
    return ts if ts is not None else (lambda: time.monotonic())


class ModuleLockTable:
    """Hierarchical module locking — two modules conflict if one is a prefix
    of the other (parent/child), but siblings are independent.

    Examples (all conflict):
        autopilot/analyze  <->  autopilot/analyze/asr   (parent <-> child)
        src/server         <->  src/server               (exact match)

    Examples (no conflict):
        autopilot/analyze/asr  <->  autopilot/analyze/speech  (siblings)
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        time_source: Callable[[], float] | None = None,
    ):
        self._limits: dict[str, int] = {}
        self._held: dict[str, set[str]] = {}  # task_id -> set of held modules
        # normalized_module -> (owner_task_id, priority_rank)
        self._parked: dict[str, tuple[str, int]] = {}
        # task_id -> ISO8601 timestamp of first install_parks call for that owner
        self._park_install_at: dict[str, str] = {}
        self._config = config
        self._time_source: Callable[[], float] = _resolve_time_source(time_source)

    # --- Hierarchy helpers ---

    @staticmethod
    def _conflicts(a: str, b: str) -> bool:
        """Two modules conflict if one is a prefix of the other (or exact match).

        Thin delegate to ``shared.locking.modules_conflict`` so the prefix rule
        has a single definition shared with the dashboard's holder lookup.
        """
        return modules_conflict(a, b)

    def _count_conflicts(self, module: str, exclude_task: str | None = None) -> int:
        """Count how many *other* tasks hold a lock that conflicts with ``module``."""
        count = 0
        for task_id, task_modules in self._held.items():
            if task_id == exclude_task:
                continue
            if any(self._conflicts(held, module) for held in task_modules):
                count += 1
        return count

    # --- Park (reservation) helpers ---

    def _is_parked_blocks(self, module: str, task_id: str) -> bool:
        """Return True iff any active park hierarchically conflicts with *module*
        and is owned by a different task.
        """
        for parked_module, (owner, _rank) in self._parked.items():
            if owner == task_id:
                continue
            if self._conflicts(parked_module, module):
                return True
        return False

    def has_parks(self, task_id: str) -> bool:
        """Return True if *task_id* currently owns any reservation."""
        return any(owner == task_id for owner, _ in self._parked.values())

    def install_parks(
        self, task_id: str, modules: list[str], priority: str
    ) -> tuple[list[str], list[tuple[str, list[str]]]]:
        """Install reservations on the normalized form of *modules* for *task_id*.

        Returns ``(installed, evicted)`` where *installed* is the list of
        normalized modules actually parked and *evicted* is a list of
        ``(owner_id, modules_lost)`` for any lower-priority parks displaced.

        Cross-tier preemption: a park with ``existing_rank > new_rank``
        (i.e. *strictly* lower priority) is evicted. Same-tier or
        higher-priority existing parks block installation of the new park on
        that module (no eviction, no install for that module).
        """
        depth = self._config.lock_depth
        rank = PRIORITY_RANK[coerce_tier(priority)]
        installed: list[str] = []
        # Accumulate evictions: victim_owner -> list of modules lost.
        eviction_acc: dict[str, list[str]] = {}
        # Track insertion order of evicted owners for stable output.
        eviction_order: list[str] = []
        for m in modules:
            normalized = normalize_lock(m, depth)
            if not normalized:
                continue
            # Find all conflicting parks from other owners.
            to_evict: list[str] = []  # parked_module keys to pop
            blocked = False
            for parked_m, (owner, existing_rank) in list(self._parked.items()):
                if owner == task_id:
                    continue
                if not self._conflicts(parked_m, normalized):
                    continue
                # Hierarchical conflict with a different owner.
                if existing_rank > rank:
                    # Strictly lower priority → evict.
                    to_evict.append(parked_m)
                else:
                    # Same or higher priority → block our install.
                    blocked = True
                    break
            if blocked:
                continue
            # Perform evictions for this module.
            for parked_m in to_evict:
                owner, _ = self._parked.pop(parked_m)
                if owner not in eviction_acc:
                    eviction_acc[owner] = []
                    eviction_order.append(owner)
                eviction_acc[owner].append(parked_m)
            self._parked[normalized] = (task_id, rank)
            installed.append(normalized)
        # Drop fully-evicted owners from _park_install_at so the dict stays
        # bounded under preemption churn and a later re-install records a
        # fresh installed_at instead of the stale setdefault-preserved value.
        for owner in eviction_acc:
            if not self.has_parks(owner):
                self._park_install_at.pop(owner, None)
        # Build evicted list in first-seen owner order, modules sorted.
        evicted = [
            (owner, sorted(eviction_acc[owner]))
            for owner in eviction_order
        ]
        if installed:
            self._park_install_at.setdefault(task_id, datetime.now(UTC).isoformat())
        return installed, evicted

    def clear_parks_for(self, task_id: str) -> None:
        """Remove every reservation owned by *task_id*."""
        self._parked = {
            m: (owner, rank)
            for m, (owner, rank) in self._parked.items()
            if owner != task_id
        }
        self._park_install_at.pop(task_id, None)

    def prune_owners(self, predicate: Callable[[str], bool]) -> list[str]:
        """Evict every park whose owner satisfies *predicate*.

        Iterates unique owners (deduped, first-seen order), calls
        ``predicate(owner_id)`` at most once per owner, and drops all
        ``_parked`` entries owned by matching tasks.

        Returns the list of evicted owner IDs in first-seen order.
        """
        # Collect unique owners in first-seen order.
        seen: dict[str, bool] = {}
        for owner, _rank in self._parked.values():
            if owner not in seen:
                seen[owner] = predicate(owner)
        # Evict matching owners.
        evicted: list[str] = []
        for owner, should_evict in seen.items():
            if should_evict:
                evicted.append(owner)
        if evicted:
            evicted_set = set(evicted)
            self._parked = {
                m: (owner, rank)
                for m, (owner, rank) in self._parked.items()
                if owner not in evicted_set
            }
            for owner in evicted:
                self._park_install_at.pop(owner, None)
        return evicted

    # --- Snapshot helpers (public accessors for observability) ---

    def snapshot_parks(self) -> dict[str, dict]:
        """Return a snapshot of current parks: ``{task_id: {modules, installed_at}}``.

        Builds a fresh dict from ``_parked`` and ``_park_install_at`` so callers
        cannot mutate internal state.  Preferred over direct ``_parked`` access in
        :meth:`Scheduler.get_state_snapshot`.
        """
        result: dict[str, dict] = {}
        for module, (owner, _rank) in self._parked.items():
            if owner not in result:
                result[owner] = {
                    'modules': [],
                    'installed_at': self._park_install_at.get(owner, ''),
                }
            result[owner]['modules'].append(module)
        return result

    def snapshot_holders(self) -> dict[str, str]:
        """Return a snapshot of current lock holders: ``{module: task_id}``.

        Builds a fresh dict from ``_held`` so callers cannot mutate internal state.
        Preferred over direct ``_held`` access in :meth:`Scheduler.get_state_snapshot`.
        """
        result: dict[str, str] = {}
        for task_id, modules in self._held.items():
            for m in modules:
                result[m] = task_id
        return result

    # --- Limit lookup (unchanged) ---

    def _limit_for(self, module: str) -> int:
        module = normalize_lock(module, self._config.lock_depth)
        if module not in self._limits:
            mc = self._config.for_module(module)
            if mc and mc.module_overrides and module in mc.module_overrides:
                self._limits[module] = mc.module_overrides[module]
            elif module in self._config.module_overrides:
                self._limits[module] = self._config.module_overrides[module]
            elif mc and mc.max_per_module is not None:
                self._limits[module] = mc.max_per_module
            else:
                self._limits[module] = self._config.max_per_module
        return self._limits[module]

    # --- Public API ---

    def is_held(self, task_id: str) -> bool:
        """Return True if task_id currently holds any module locks."""
        return task_id in self._held

    def try_acquire(self, task_id: str, modules: list[str]) -> bool:
        """Non-blocking attempt to acquire all module locks.

        Uses hierarchical conflict detection: a lock on ``A/B`` conflicts with
        ``A/B/C`` (and vice-versa) but NOT with ``A/D``.  Also refuses if any
        requested module hierarchically conflicts with an active reservation
        owned by a different task (see ``install_parks``).

        Returns True if all acquired, False if any unavailable.
        """
        depth = self._config.lock_depth
        normalized = list({normalize_lock(m, depth) for m in modules})

        # Check every requested module against all other tasks' held locks and
        # active reservations owned by other tasks.
        for module in normalized:
            if self._count_conflicts(module, exclude_task=task_id) >= self._limit_for(module):
                return False
            if self._is_parked_blocks(module, task_id):
                return False

        self._held[task_id] = set(normalized)
        logger.info(f'Task {task_id} acquired locks: {normalized}')
        return True

    def release(self, task_id: str) -> None:
        """Release all module locks held by a task."""
        modules = self._held.pop(task_id, set())
        if modules:
            logger.info(f'Task {task_id} released locks: {list(modules)}')

    def release_subset(self, task_id: str, modules: list[str]) -> list[str]:
        """Drop a subset of the task's held modules. Returns the normalized
        modules actually released (may be empty). Removes the task's entry
        entirely when no held modules remain so downstream iteration over
        ``_held`` behaves the same as after a full ``release``.
        """
        held = self._held.get(task_id)
        if not held:
            return []
        depth = self._config.lock_depth
        to_drop = {normalize_lock(m, depth) for m in modules} & held
        if not to_drop:
            return []
        held.difference_update(to_drop)
        if not held:
            del self._held[task_id]
        released = sorted(to_drop)
        logger.info(f'Task {task_id} released subset: {released}')
        return released

    def try_acquire_additional(self, task_id: str, additional: list[str]) -> bool:
        """Non-blocking attempt to expand lock set for a task."""
        depth = self._config.lock_depth
        current = self._held.get(task_id, set())
        new_modules = [
            normalize_lock(m, depth)
            for m in additional
            if normalize_lock(m, depth) not in current
        ]
        if not new_modules:
            return True

        for module in new_modules:
            if self._count_conflicts(module, exclude_task=task_id) >= self._limit_for(module):
                return False
            if self._is_parked_blocks(module, task_id):
                return False

        self._held.setdefault(task_id, set()).update(new_modules)
        logger.info(f'Task {task_id} expanded locks: {new_modules}')
        return True


class Scheduler:
    """Selects next eligible task and manages module locks."""

    def __init__(
        self,
        config: OrchestratorConfig,
        event_store: EventStore | None = None,
        *,
        mcp_session: McpSessionLike | None = None,
        time_source: Callable[[], float] | None = None,
        override_store: OverrideStore | None = None,
        monotonic_clock_source: Callable[[], float] | None = None,
    ):
        self.config = config
        self._time_source: Callable[[], float] = _resolve_time_source(time_source)
        # Monotonic clock source for the park-stop rolling-window transition
        # recorder.  time.monotonic avoids false-trip / stale-entry artefacts
        # from non-monotonic wall-clock skew (NTP steps, VM clock drift).
        # Injectable via the ``monotonic_clock_source`` kwarg so deterministic
        # tests can inject a fixed lambda without touching production semantics.
        # NOTE: callers MUST inject a monotonic-style source (no epoch relation,
        # immune to NTP/clock skew).  Injecting ``time.time`` will break trip
        # semantics under clock adjustments.
        self._park_stop_clock: Callable[[], float] = (
            monotonic_clock_source if monotonic_clock_source is not None else time.monotonic
        )
        self.lock_table = ModuleLockTable(config, time_source=self._time_source)
        self.event_store = event_store
        self._mcp_session = mcp_session
        self._dispatched: set[str] = set()
        self._memory_url = config.fused_memory.url
        self._project_root = str(config.project_root)
        self._module_cache: dict[str, list[str]] = {}  # task_id -> expanded modules
        self._fallback_warned: set[str] = set()  # task IDs already warned about fallback
        self._requeue_until: dict[str, float] = {}  # task_id -> monotonic deadline
        # Per-task dispatch timestamps (monotonic) for the dispatch-cooldown
        # gate.  Set immediately after a successful dispatch; cleared when the
        # task transitions to a terminal status.  Process-local — an
        # orchestrator restart is an acceptable implicit reset (reconciliation
        # re-marks tasks needing the gate).
        self._last_dispatch_at: dict[str, float] = {}  # task_id -> monotonic ts
        # Per-task retry-cap tracking.  Count of REQUEUED outcomes since the
        # last DONE (or cap-exhaust); history is the per-attempt record used
        # by the cap-exhaust escalation report.  Both are process-local — an
        # orchestrator restart is an acceptable implicit reset.
        self._requeue_counts: dict[str, int] = {}
        self._transient_requeue_counts: dict[str, int] = {}
        self._requeue_history: dict[str, list[RequeueRecord]] = {}
        # --- Fairness state (see orchestrator.config.FairnessConfig) ---
        self._skip_count: dict[str, int] = {}  # task_id -> consecutive top-skip count
        # Per-tier cap bookkeeping: remember the effective priority of every
        # currently-dispatched task so acquire_next can count slots at-or-below
        # a candidate's tier without re-walking the full task graph.
        self._dispatched_priority: dict[str, str] = {}
        # Age-anchor bookkeeping for score(): first time we see a task as
        # pending, we record its age baseline.  Cleared on transition to any
        # non-pending status so a cancelled->pending resurrection starts
        # fresh (no accumulated age).
        self._pending_anchor: dict[str, int] = {}
        self._was_non_pending: set[str] = set()
        # Effective-priority cache: populated at the end of each acquire_next tick
        # so get_state_snapshot() can include it without re-fetching tasks.
        # Empty dict before the first tick.
        self._last_effective_priorities: dict[str, str] = {}
        # --- Priority-override state ---
        self._override_store: OverrideStore | None = override_store
        # Public alias for shutdown-path callers (harness checkpoint) — read-only.
        self.override_store: OverrideStore | None = override_store
        # Snapshot from the previous tick, used to diff-detect override changes
        # and emit the priority_override_* / task_pinned / pin_queue_reordered events.
        self._prev_overrides_snapshot: dict[str, OverrideRow] = {}
        # Whether the snapshot has been seeded from the store on the first tick.
        # On scheduler restart, pre-existing overrides must NOT fire spurious
        # priority_override_set / task_pinned events — they represent state that
        # was already known, not fresh user actions.  The first tick seeds the
        # snapshot without emitting events; subsequent ticks diff-emit normally.
        self._overrides_initialized: bool = False
        # --- Park-and-stop pause state (task 1322) ---
        self._paused: bool = False
        self._pause_reason: str | None = None
        # Rolling deque of (task_id, monotonic_timestamp) pairs for each
        # successful blocked transition.  Entries older than
        # park_stop_parked_window_hours * 3600s are lazily evicted on each
        # _record_blocked_transition() call.  Storing task_id alongside the
        # timestamp enables per-task de-duplication: idempotent re-sets
        # (blocked→blocked, e.g. from a recovery loop or post-restart replay)
        # count as ONE transition per task within the window rather than N.
        self._blocked_transitions: deque[tuple[str, float]] = deque()
        # Companion set for O(1) de-dup lookup: contains the task_ids of all
        # entries currently in _blocked_transitions.  Kept in sync with the
        # deque — entries are added/removed together.
        self._blocked_task_ids_in_window: set[str] = set()
        # Callback installed by the Harness so trip → persistence + event.
        self._on_park_stop_trip: Callable[[str], Any] | None = None
        # --- Cross-project external-dep escalation (task 1580) ---
        # Callback installed by the Harness: (task_id, *, summary, detail,
        # category) → block the task + submit L1.  Default None so bare-Harness
        # unit tests (and park_gc) are unaffected.
        self._on_external_dep_block: Callable[..., Any] | None = None
        # --- Action-teardown suppression (task 1620, β Pair E / C3.2) ---
        # Predicate installed by the Harness: when set, set_task_status('blocked',
        # ...) returns early (before dispatch_tool) if the predicate returns True
        # for that task_id.  Absorbs racing 'blocked' writes emitted by a workflow
        # being killed during action teardown (park→deferred / restart→pending) so
        # the action's target status is not clobbered.  Mirrors the
        # _on_park_stop_trip / _on_external_dep_block callback-install pattern:
        # declared here, installed by Harness.__init__.
        self._suppress_blocked_write: Callable[[str], bool] | None = None
        # Per-(task_id, dep_string) count of consecutive ticks where the dep
        # resolved to a sentinel (unknown_project/unknown_task/malformed).
        # Process-local — a scheduler restart is an acceptable implicit reset,
        # matching _requeue_counts/_skip_count idioms above.
        self._external_unresolved_counts: dict[tuple[str, str], int] = {}
        # Per-(task_id, dep_id) count of consecutive ticks where the LOCAL dep
        # backfill get_statuses call degraded (parse failure or empty result).
        # Mirrors _external_unresolved_counts for local deps.  GC'd in the
        # per-tick stale-id sweep alongside _external_unresolved_counts.
        self._local_backfill_unresolved_counts: dict[tuple[str, str], int] = {}
        # Per-task_id count of consecutive ticks where the external-dep gate
        # held dispatch (either because the resolver was degraded, or because
        # all deps returned live non-done statuses).  Keyed by task_id (str).
        # Process-local — same rationale as _external_unresolved_counts.
        # GC'd alongside _external_unresolved_counts in the per-tick stale-id sweep.
        self._external_hold_streak: dict[str, int] = {}
        # Tracks the most-recent cause for _external_hold_streak[task_id].
        # Reset alongside the streak.  When the cause changes tick-over-tick
        # (e.g. resolver degraded → dep live) the streak resets to zero so the
        # emitted external_dep_gate_held.cause always reflects the dominant
        # (consecutive-run) reason, not a mixed accumulation.
        self._external_hold_cause: dict[str, str] = {}
        # --- Snapshot write throttle (task 1332) ---
        # Monotonic timestamp of the last successful _write_snapshot_best_effort
        # disk write.  None before the first write; the first write always
        # proceeds regardless of the throttle interval.
        self._last_snapshot_write_ts: float | None = None
        # Serialised payload from the last disk write.  Used for content-diff:
        # if the new payload is byte-identical, the disk write is skipped even
        # after the time gate passes (populated in step-6; kept here for
        # structural completeness and future use).
        self._last_snapshot_payload: str | None = None
        # Serialises concurrent _write_snapshot_best_effort invocations so that
        # tick writes and flush writes never race on the shared .json.tmp path.
        self._snapshot_write_lock: asyncio.Lock = asyncio.Lock()
        # One-time dedup flag for the project_root guard warning so it logs at
        # most once per scheduler instance, not every tick.
        self._snapshot_guard_warned: bool = False
        # One-time dedup flag for override-store read failures: a broken store
        # is a static deployment condition, so one traceback per instance is
        # enough — callers are time-throttled by _write_snapshot_best_effort.
        self._override_store_warned: bool = False

    # --- Park-and-stop pause API (task 1322) ---

    @property
    def is_paused(self) -> bool:
        """True when the scheduler is paused and acquire_next() returns None."""
        return self._paused

    @property
    def pause_reason(self) -> str | None:
        """Human-readable reason for the current pause, or None if not paused."""
        return self._pause_reason

    @property
    def parked_live_count(self) -> int:
        """Count of task_ids currently in the park-stop sliding window (de-duped).

        Equal to len(_blocked_task_ids_in_window).  Exposed as a public property
        so the digest subsystem (and any other observer) does not need to access
        the private set directly.  Task 1327 encapsulation.
        """
        return len(self._blocked_task_ids_in_window)

    @property
    def parked_window_churn_count(self) -> int:
        """Count of (task_id, timestamp) entries in the park-stop rolling deque.

        Equal to len(_blocked_transitions).  Counts raw transitions in the window
        (may exceed parked_live_count if the deque has not been evicted yet).
        Exposed as a public property for the digest subsystem.  Task 1327.
        """
        return len(self._blocked_transitions)

    def pause(self, reason: str) -> None:
        """Pause the scheduler.  acquire_next() will return None until resume().

        Idempotent: if already paused, the original reason is kept and a DEBUG
        log is emitted.  The pause state is in-memory only; callers that need
        persistence (e.g. Harness.pause_scheduler) must persist separately.
        """
        if self._paused:
            logger.debug(
                'Scheduler.pause() called while already paused '
                '(existing reason=%r, new reason=%r) — keeping original',
                self._pause_reason, reason,
            )
            return
        self._paused = True
        self._pause_reason = reason
        logger.info('Scheduler paused: %s', reason)

    def resume(self) -> None:
        """Resume the scheduler.  Next acquire_next() tick will dispatch normally.

        Idempotent: if not paused, this is a no-op.

        Clears the rolling ``_blocked_transitions`` deque so the operator's
        resume establishes a clean baseline.  Without this, a still-full deque
        (the window is a monotonic-clock 1h by default) would cause the next
        blocked transition — e.g. an in-flight workflow finishing shortly after
        resume — to immediately re-trip the park-stop pause, silently undoing
        the operator's action.  Requiring fresh transitions post-resume keeps the
        circuit breaker observable: trips correspond to bursts after resume,
        not stale history from before it.
        """
        if not self._paused:
            logger.debug('Scheduler.resume() called while not paused — no-op')
            return
        self._paused = False
        self._pause_reason = None
        self._blocked_transitions.clear()
        self._blocked_task_ids_in_window.clear()
        logger.info('Scheduler resumed')

    def _maybe_fire_park_stop_trip(self) -> None:
        """Check if the park-stop trip threshold is met and fire the callback.

        Guards (checked in order, earliest exit first):
          1. Already paused — no re-trip.
          2. park_stop_enabled=False — trip suppressed (but deque still records).
          3. Callback not wired — no-op.
          4. Count < threshold — not yet.

        When all guards pass, this method:
          a. SYNCHRONOUSLY calls self.pause(reason) — the latch step.  This
             immediately sets _paused=True so any concurrent coroutine that has
             already appended its timestamp but hasn't yet called
             _maybe_fire_park_stop_trip will see _paused=True and return at
             guard 1 without scheduling a duplicate callback.  This prevents the
             race where N concurrent set_task_status('blocked') calls each
             observe _paused=False (the async callback hasn't run yet) and each
             schedule their own ensure_future — resulting in N-threshold+1
             duplicate callbacks and an equal number of duplicate
             run_store.save_scheduler_pause / scheduler_paused event writes.
          b. Formats a human-readable reason string and logs a WARNING.
          c. Schedules the full callback (harness.pause_scheduler) via
             asyncio.ensure_future() — fire-and-forget so the status write is
             never delayed.  The callback's own scheduler.pause(reason) call
             becomes a no-op because _paused is already True (idempotent).
             Persistence (run_store) and event emission still fire exactly once.

        The synchronous latch is the key invariant: pause() BEFORE ensure_future.

        Crash semantics — at-most-once on disk: the in-memory ``_paused`` latch
        is set synchronously, but the ``run_store.save_scheduler_pause`` write
        and ``scheduler_paused`` event emission both live inside the scheduled
        ``harness.pause_scheduler`` coroutine.  If the orchestrator crashes
        (or the loop is shut down) AFTER the latch is set but BEFORE that
        coroutine executes, the pause is observable in-memory and in the WARN
        log but never reaches disk.  The next restart will therefore NOT
        restore the pause — operators investigating a near-trip crash should
        verify ``run_store.load_scheduler_pause`` reflects the expected state
        before resuming dispatch.  This trade-off is intentional: keeping the
        status write off the synchronous path avoids blocking the caller
        (set_task_status) on a SQLite write.
        """
        if self._paused:
            return
        if not self.config.park_stop_enabled:
            return
        if self._on_park_stop_trip is None:
            return
        n = len(self._blocked_transitions)
        threshold = self.config.park_stop_parked_threshold
        if n < threshold:
            return
        window_hours = self.config.park_stop_parked_window_hours
        reason = (
            f'park-stop: {n} tasks transitioned to blocked within '
            f'{window_hours}h (threshold={threshold})'
        )
        # SYNCHRONOUS LATCH: set _paused=True immediately so concurrent
        # coroutines see the paused state before the async callback runs.
        # This must happen before asyncio.ensure_future to close the race
        # window between "trip detected" and "callback sets _paused".
        self.pause(reason)
        logger.warning('Park-stop trip: %s — pausing scheduler', reason)
        try:
            t = asyncio.ensure_future(self._on_park_stop_trip(reason))
            # Attach a done-callback so exceptions inside harness.pause_scheduler
            # (e.g. RunStore write failure, EventStore emit failure) are logged
            # immediately rather than surfacing as GC-collected Task warnings
            # with no context.  Without this the operator sees the scheduler is
            # paused but has no diagnostic for why persistence failed.
            t.add_done_callback(
                lambda f: logger.error(
                    'park-stop trip callback raised an exception',
                    exc_info=f.exception(),
                )
                if not f.cancelled() and f.exception() is not None
                else None
            )
        except RuntimeError:
            # No running event loop — should not happen in production since
            # set_task_status is always called from an async context.  Log and
            # skip rather than crashing the caller.
            logger.warning(
                'park-stop trip: no running event loop; callback not scheduled'
            )

    def _record_blocked_transition(self, task_id: str) -> None:
        """Record a successful blocked transition in the rolling deque.

        De-duplicates by *task_id*: if the same task is already counted within
        the rolling window (e.g. from an idempotent re-set or post-restart
        replay), this call is a no-op.  This prevents a single task being
        marked blocked multiple times from artificially inflating the trip
        counter — the design intent is "N *distinct* tasks transition to
        blocked", not "N writes that resolve to blocked".

        Evicts (task_id, timestamp) pairs older than
        ``park_stop_parked_window_hours * 3600`` seconds from the left of
        the deque.  O(k) where k is the number of expired entries (typically
        tiny).  The companion ``_blocked_task_ids_in_window`` set is kept in
        sync so de-dup checks remain O(1).
        """
        now = self._park_stop_clock()
        cutoff = now - self.config.park_stop_parked_window_hours * 3600

        # Evict expired (task_id, ts) pairs from the front of the deque,
        # keeping _blocked_task_ids_in_window consistent.
        while self._blocked_transitions and self._blocked_transitions[0][1] <= cutoff:
            evicted_id, _ = self._blocked_transitions.popleft()
            self._blocked_task_ids_in_window.discard(evicted_id)

        # De-dup: if this task is already counted in the window, skip.
        if task_id in self._blocked_task_ids_in_window:
            return

        self._blocked_transitions.append((task_id, now))
        self._blocked_task_ids_in_window.add(task_id)

    async def dispatch_tool(
        self,
        name: str,
        arguments: dict,
        *,
        timeout: float = 15,
    ) -> dict:
        """Route an MCP tool call through the injected session or HTTP fallback.

        When ``self._mcp_session`` is set (e.g. a ``_StubMcpSession`` in eval
        mode), the call is dispatched directly via its ``call_tool`` method.
        Otherwise the existing ``mcp_call`` HTTP path is used unchanged so
        production semantics, retries, and error handling are preserved.
        """
        if self._mcp_session is not None:
            return await self._mcp_session.call_tool(name, arguments, timeout=timeout)
        return await mcp_call(
            f'{self._memory_url}/mcp',
            'tools/call',
            {'name': name, 'arguments': arguments},
            timeout=timeout,
        )

    @staticmethod
    def _normalize_task_metadata(task: dict) -> None:
        """Coerce ``task['metadata']`` to a dict in place.

        The fused-memory wire format may surface metadata as a JSON string,
        a dict, or absent. Normalize once at this boundary so every consumer
        can assume ``isinstance(task['metadata'], dict)`` without re-parsing.

        A non-JSON or non-dict-shaped value collapses to ``{}`` — every
        consumer reads dict-keyed sub-fields, so a non-dict carries no
        information they can use.
        """
        raw = task.get('metadata')
        if isinstance(raw, dict):
            return
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            task['metadata'] = parsed if isinstance(parsed, dict) else {}
            return
        task['metadata'] = {}

    @staticmethod
    def carries_substrate_probe(task: dict) -> bool:
        """Return True iff *task* carries a non-empty ``substrate_probe`` descriptor.

        This is the single source of truth the harness uses to decide whether
        to run the dispatch-time substrate re-check gate (D4).  Delegates to
        ``substrate_gate.extract_probe_set`` so the descriptor-extraction
        logic lives in one place.

        Lazy import keeps substrate_gate dependency-light and avoids import
        cycles (substrate_gate imports nothing from orchestrator).
        """
        # Lazy import — substrate_gate is stdlib-only; deferred to avoid
        # loading it at scheduler import time (mirrors b3_gate lazy-import
        # of orchestrator.config at resolution time).
        from orchestrator.substrate_gate import extract_probe_set  # noqa: PLC0415
        return extract_probe_set(task) is not None

    async def tasks_by_train(self, train_id: str) -> list[dict]:
        """Return tasks belonging to ``train_id``, sorted ascending by train.order (root→tip).

        δ₂ member-discovery helper (PRD §δ₂).  Performs a FRESH ``get_tasks``
        read — deliberately not a stateful cache — so callers always see the
        latest member statuses.  For a three-member train the round-trip cost
        is negligible, and a stale cache would cause the all-deferred check to
        fire on outdated data.

        **Active-only filter rationale (γ3b):** uses
        ``statuses=ACTIVE_TASK_STATUSES`` to reduce payload.  Terminal members
        are intentionally excluded:

        - *done* — group-merge is the atomic done-transition, so no member is
          ever ``done`` while ``_maybe_enqueue_group_merge`` evaluates this
          list.  The authoritative done check at merge time uses
          ``get_statuses`` (workflow.py ``_status_check``), so the
          discovery/verify split the PRD prescribes is already in place.
        - *cancelled* — a human-cancelled member is dropped from the returned
          list.  The remaining active members can then all reach
          ``merge-deferred``, causing the group-merge guard in
          ``_maybe_enqueue_group_merge`` to fire on the reduced set.  Whether
          a partially-cancelled train should still group-merge is a policy
          decision in the trigger (workflow.py:770), not in this helper.

        Args:
            train_id: The ``metadata.train.id`` value to filter by.  Returns
                ``[]`` immediately when falsy (avoids a spurious get_tasks
                round-trip for empty/None callers).

        Returns:
            List of task dicts whose ``metadata.train.id == train_id``, sorted
            ascending by ``metadata.train.order`` (root→tip).  Members whose
            ``order`` is missing or non-integer are sorted last
            deterministically (stable, no crash).
        """
        if not train_id:
            return []

        tasks = await self.get_tasks(statuses=ACTIVE_TASK_STATUSES)

        def _order_key(t: dict) -> tuple[int, int]:
            train = (t.get('metadata') or {}).get('train') or {}
            order = train.get('order') if isinstance(train, dict) else None
            try:
                return (0, int(order))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return (1, 0)  # missing/non-int → sort last

        return sorted(
            [
                t for t in tasks
                if isinstance((t.get('metadata') or {}).get('train'), dict)
                and (t.get('metadata') or {}).get('train', {}).get('id') == train_id
            ],
            key=_order_key,
        )

    async def get_tasks(
        self,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[dict]:
        """Fetch tasks from fused-memory/taskmaster.

        Args:
            statuses: Optional iterable of status strings to filter by (server-side
                SQL ``status IN (...)`` predicate).  When ``None`` (default), the
                argument is omitted and the server returns the full task tree —
                byte-identical to the previous behaviour.  Pass
                ``ACTIVE_TASK_STATUSES`` on hot paths to shrink the payload.
        """
        try:
            arguments: dict = {'project_root': self._project_root}
            if statuses is not None:
                arguments['statuses'] = list(statuses)
            result = await self.dispatch_tool(
                'get_tasks',
                arguments,
                timeout=15,
            )
            tasks, tasks_err = parse_tool_result(result, 'tasks', list)
            if tasks_err is None and tasks is not None:
                for t in tasks:
                    if isinstance(t, dict):
                        self._normalize_task_metadata(t)
                return tasks
        except Exception as e:
            # logger.exception preserves the traceback + exception class so the
            # next time this fires we have more than str(e) to go on — str()
            # produces bare forms like "[Errno 2] No such file or directory"
            # with no indication of which layer raised it.
            logger.exception(
                'Failed to fetch tasks: %s: %s', type(e).__name__, e,
            )
        return []

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        *,
        done_provenance: dict | None = None,
        reopen_reason: str | None = None,
    ) -> None:
        """Update task status via fused-memory.

        Terminal-state enforcement lives on the server (fused-memory
        TaskInterceptor) — this method just forwards the call. Pass
        ``reopen_reason`` to exit a terminal status (done/cancelled);
        orchestrator automation never needs it.

        fused-memory rejects some transitions (terminal-exit gate,
        phantom-done gate, done_provenance validation) by returning a
        structured error dict rather than raising. We inspect the response
        and emit a WARNING so silent rejections don't leave tasks stuck in
        the wrong state.

        **Transient-failure retry.** When fused-memory's taskmaster child
        is restarting or the fused-memory process itself just crashed, the
        tool wrapper returns ``{'error': 'TimeoutError(...)', 'error_type':
        'TimeoutError'}`` (or similar).  Without this loop the workflow
        would treat the call as successful and proceed, leaving the task
        stranded in-progress.  Retry up to ``_TRANSIENT_RETRIES`` times
        with exponential back-off; raise on persistent transient failure
        so callers (notably ``handle_blast_radius_expansion``) can decide
        whether to release locks.
        """
        arguments: dict = {
            'id': task_id,
            'status': status,
            'project_root': self._project_root,
        }
        if done_provenance is not None:
            arguments['done_provenance'] = done_provenance
        if reopen_reason is not None:
            arguments['reopen_reason'] = reopen_reason

        # C3.2 (task 1620, β Pair E): action-teardown suppression guard.
        # When the Harness stamps _action_teardown_tasks for a task being killed
        # (park→deferred / restart→pending), absorb any racing 'blocked' write the
        # workflow may emit before the kill completes.  This guard MUST sit BEFORE
        # dispatch_tool so the write never reaches fused-memory — _record_blocked_transition
        # and _maybe_fire_park_stop_trip are then necessarily skipped for free.
        # (A guard placed at the post-write success branch :1190 would still let the
        # write reach fused-memory, defeating C3.2.)
        if (
            status == 'blocked'
            and self._suppress_blocked_write is not None
            and self._suppress_blocked_write(task_id)
        ):
            logger.info(
                'set_task_status(%s, blocked): suppressed by action-teardown — skipping write',
                task_id,
            )
            return

        last_rejection: str | None = None
        for attempt in range(_TRANSIENT_RETRIES):
            try:
                response = await self.dispatch_tool(
                    'set_task_status', arguments, timeout=15,
                )
            except Exception as e:
                logger.exception(
                    'set_task_status(%s, %s) raised on attempt %d/%d: %s: %s',
                    task_id, status, attempt + 1, _TRANSIENT_RETRIES,
                    type(e).__name__, e,
                )
                last_rejection = f'{type(e).__name__}: {e}'
                if attempt + 1 < _TRANSIENT_RETRIES:
                    await asyncio.sleep(_TRANSIENT_BACKOFF_BASE * (2 ** attempt))
                continue

            rejection = extract_rejection(response)
            if rejection is None:
                # Success — record the blocked transition if applicable and
                # fire the park-stop trip check.  These must run before return
                # so that every confirmed write is counted exactly once.
                if status == 'blocked':
                    self._record_blocked_transition(task_id)
                    self._maybe_fire_park_stop_trip()
                return  # success
            last_rejection = rejection
            if not is_transient_rejection(rejection):
                # Non-transient rejection: classify and raise so callers can
                # react instead of finding out via a divergent on-disk state.
                # Two terminal-target carve-outs preserve idempotency:
                #   1. ``terminal_exit_rejected`` for a TERMINAL target with
                #      no reopen_reason — the server treats this as a no-op,
                #      not a contradiction. Log and return.
                #   2. ``terminal_exit_rejected`` when the caller passed
                #      reopen_reason — the caller already acknowledged the
                #      terminal state; surface as a warning.
                structured = extract_structured_rejection(response)
                error_code = (
                    str(structured.get('error', ''))
                    if isinstance(structured, dict)
                    else ''
                )
                if error_code == 'terminal_exit_rejected':
                    if status in _TERMINAL_STATUSES or reopen_reason is not None:
                        logger.warning(
                            'set_task_status(%s, %s) rejected by fused-memory: %s',
                            task_id, status, rejection,
                        )
                        return
                    raise TerminalExitRejection(
                        task_id=task_id,
                        old_status=str(structured.get('from_status', ''))
                        if isinstance(structured, dict) else '',
                        target_status=status,
                        raw=rejection,
                    )
                if error_code == 'done_gate_missing_files':
                    missing = []
                    if isinstance(structured, dict):
                        raw_missing = structured.get('missing_files') or []
                        if isinstance(raw_missing, list):
                            missing = [
                                m for m in raw_missing if isinstance(m, str)
                            ]
                    raise DoneGateRejection(
                        task_id=task_id,
                        missing_files=missing,
                        raw=rejection,
                    )
                if error_code in (
                    'done_provenance_required',
                    'done_provenance_invalid',
                ):
                    raise ProvenanceValidationRejection(
                        task_id=task_id,
                        error_code=error_code,
                        raw=rejection,
                    )
                # Any other non-transient rejection: raise the family base
                # so callers can catch SetTaskStatusRejected uniformly.
                raise SetTaskStatusRejected(
                    task_id=task_id,
                    error_code=error_code or 'unknown',
                    raw=rejection,
                )
            logger.info(
                'set_task_status(%s, %s) transient rejection '
                '(attempt %d/%d): %s — retrying',
                task_id, status, attempt + 1, _TRANSIENT_RETRIES, rejection,
            )
            if attempt + 1 < _TRANSIENT_RETRIES:
                await asyncio.sleep(_TRANSIENT_BACKOFF_BASE * (2 ** attempt))

        raise RuntimeError(
            f'set_task_status({task_id}, {status}) failed after '
            f'{_TRANSIENT_RETRIES} transient retries: {last_rejection}'
        )

    async def mark_done(
        self,
        task_id: str,
        *,
        kind: str,
        sha: str,
        note: str | None = None,
    ) -> None:
        """Set ``task_id`` done with verified ``done_provenance``.

        Centralises the provenance-construction shape shared by every
        workflow / harness call site that marks a task done.  ``sha`` is
        mandatory: a ``kind='merged'`` (or ``'found_on_main'``) without a
        commit is a workflow bug, not a normal case — the server's
        provenance gate would reject it anyway.

        Exceptions from ``set_task_status`` propagate; callers handle the
        rejection family explicitly so a stuck-done can never be silently
        swallowed.
        """
        provenance: dict[str, str] = {'kind': kind, 'commit': sha}
        if note is not None:
            provenance['note'] = note
        await self.set_task_status(
            task_id, 'done', done_provenance=provenance,
        )

    async def get_status(self, task_id: str) -> str | None:
        """Return the current status of ``task_id``, or ``None`` on failure.

        Replaces the old client-side status cache. Each call is a fresh MCP
        round-trip; fused-memory's warm get_task is ~30 ms, so this is cheap
        at the handful of decision points that actually need the truth
        (post-steward terminal check).
        """
        try:
            result = await self.dispatch_tool(
                'get_task',
                {'id': task_id, 'project_root': self._project_root},
                timeout=15,
            )
        except Exception as e:
            logger.exception(
                'Failed to get task %s status: %s: %s',
                task_id, type(e).__name__, e,
            )
            return None
        status, status_err = parse_tool_result(result, 'status', str)
        return status if status_err is None else None

    async def get_task(self, task_id: str) -> dict | None:
        """Fetch the full task dict (including metadata) from fused-memory.

        Used by the workflow's bypass-detection path to inspect
        ``metadata.done_provenance`` after a ``terminal_exit_rejected``
        rejection. Returns ``None`` on failure or absence.
        """
        try:
            result = await self.dispatch_tool(
                'get_task',
                {'id': task_id, 'project_root': self._project_root},
                timeout=15,
            )
        except Exception as e:
            logger.exception(
                'Failed to get task %s: %s: %s',
                task_id, type(e).__name__, e,
            )
            return None
        # The MCP tool returns the task wrapped in {data: {...}}; unwrap via
        # the same text-block parser used by get_status.
        if not isinstance(result, dict):
            return None
        for block in result.get('result', {}).get('content', []) or []:
            if isinstance(block, dict) and block.get('type') == 'text':
                try:
                    data = json.loads(block.get('text') or '')
                except (ValueError, TypeError):
                    return None
                if isinstance(data, dict):
                    inner = (
                        data.get('data')
                        if isinstance(data.get('data'), dict)
                        else data
                    )
                    if isinstance(inner, dict):
                        # Normalise metadata at the boundary so all callers
                        # receive task['metadata'] as a dict — consistent with
                        # get_tasks / acquire_next which already do this via
                        # _normalize_task_metadata.
                        self._normalize_task_metadata(inner)
                        return inner
                    return None
                return None
        return None

    async def get_statuses(
        self, ids: list[str] | None = None
    ) -> tuple[dict[str, str], Exception | None]:
        """Return a ``(statuses, error)`` tuple from fused-memory.

        Uses the ``get_statuses`` MCP tool which returns ~95% less data than
        ``get_tasks``.  Suitable for hot-loop callers that only need status.

        Args:
            ids: Optional list of task ids to filter by (unknown ids silently
                 omitted).  Pass ``None`` for all tasks.

        Returns:
            A ``(statuses, error)`` tuple.  On success: ``({id: status}, None)``.
            On any failure: ``({}, exception)``.  Error state lives on the stack
            — no shared mutable attribute, safe under concurrent callers.
        """
        try:
            arguments: dict = {'project_root': self._project_root}
            if ids is not None:
                arguments['ids'] = list(ids)
            result = await self.dispatch_tool('get_statuses', arguments, timeout=15)
            statuses, err = parse_tool_result(result, 'statuses', dict)
            if err is not None:
                return {}, err
            assert statuses is not None  # invariant: parse_tool_result → (None, err) | (value, None)
            return statuses, None
        except Exception as e:
            logger.exception(
                'Failed to fetch task statuses: %s: %s', type(e).__name__, e,
            )
            return {}, e

    async def get_external_statuses(
        self, deps: list[str]
    ) -> tuple[dict[str, str], Exception | None]:
        """Return a ``(statuses, error)`` tuple for a list of cross-project deps.

        Issues a single ``get_external_statuses`` MCP call (no ``project_root``
        — the tool is cross-project by design).  The dep strings are passed
        verbatim; the returned dict is keyed by those same dep strings.

        Sentinels returned by the tool (``unknown_project``, ``unknown_task``,
        ``malformed``) are surfaced as-is — the caller decides policy.

        Returns:
            ``({dep: status}, None)`` on success (all requested dep keys present).
            ``(partial_dict, ExternalResolverError)`` when the response dict is
            missing one or more requested dep keys (resolver-degraded; partial
            dict preserved for logging but error slot forces fail-safe wait).
            ``({}, ExternalResolverError)`` when ``parse_tool_result``
            returns a parse error (unparseable JSON or missing 'statuses' key).
            ``({}, exception)`` on any raised exception — transient raises are
            swallowed into the error slot (fail-safe; caller should skip policy
            effects).
        """
        try:
            arguments: dict = {'deps': list(deps)}
            result = await self.dispatch_tool(
                'get_external_statuses', arguments, timeout=15
            )
            statuses, parse_err = parse_tool_result(result, 'statuses', dict)
            if parse_err is not None:
                # primitive already emitted the WARNING; preserve ExternalResolverError type.
                return {}, ExternalResolverError(str(parse_err))
            assert statuses is not None  # invariant: parse_tool_result → (None, err) | (value, None)
            # Guard: the real tool always keys its response by the verbatim
            # dep string and always sets a value (real status or a sentinel).
            # A missing dep key is a genuine contract violation, not normal
            # operation, so treat it as resolver-degraded (fail-safe wait).
            missing = [d for d in deps if d not in statuses]
            if missing:
                msg = (
                    f'get_external_statuses: response missing {len(missing)}'
                    f' of {len(deps)} requested dep keys '
                    f'(sample: {missing[:3]!r}) — resolver-degraded; '
                    'fail-safe wait this tick'
                )
                logger.warning(msg)
                return statuses, ExternalResolverError(msg)
            return statuses, None
        except Exception as e:
            logger.exception(
                'Failed to fetch external dep statuses: %s: %s',
                type(e).__name__,
                e,
            )
            return {}, e

    _EXTERNAL_SENTINEL_STATUSES: frozenset[str] = frozenset(
        {'unknown_project', 'unknown_task', 'malformed'}
    )

    def _note_external_hold(
        self,
        task_id: str,
        *,
        cause: str,
        threshold: int,
        detail: str | None = None,
    ) -> None:
        """Bump the hold-streak for ``task_id`` and emit an event at the threshold.

        Called once per held tick (either ``'resolver_degraded'`` or
        ``'deps_live'``).  Emits ``EventType.external_dep_gate_held`` the
        first time the streak reaches ``threshold`` and on each subsequent
        ``threshold``-multiple tick, so the event is durable and bounded.

        Does NOT touch ``_external_unresolved_counts`` (sentinel-counter)
        — that counter is owned by the sentinel-escalation path.
        """
        # Reset when the cause changes so emitted events always reflect a
        # single consecutive-run cause, not a mixed accumulation.
        if self._external_hold_cause.get(task_id) != cause:
            self._external_hold_streak[task_id] = 0
        self._external_hold_cause[task_id] = cause
        streak = self._external_hold_streak.get(task_id, 0) + 1
        self._external_hold_streak[task_id] = streak
        if streak >= threshold and streak % threshold == 0:
            logger.warning(
                'Task %s: external-dep gate has held dispatch for %d consecutive '
                'ticks (cause=%r, threshold=%d)',
                task_id,
                streak,
                cause,
                threshold,
            )
            if self.event_store is not None:
                self.event_store.emit(
                    EventType.external_dep_gate_held,
                    task_id=task_id,
                    data={'cause': cause, 'ticks': streak, 'detail': detail},
                )

    async def _apply_external_dep_policy(
        self,
        pending_tasks: list[dict],
        external_cache: dict[str, str],
        external_err: Exception | None,
    ) -> None:
        """Side-effecting pass over pending tasks' external deps.

        Run ONCE per tick from ``acquire_next``.  Per-task per-dep:

        - ``done`` → satisfied; no action.
        - ``cancelled`` → invoke ``_on_external_dep_block`` immediately with
          ``EXTERNAL_DEP_CANCELLED`` prefix (invariant 2 — strict escalation).
        - ``unknown_project`` / ``unknown_task`` / ``malformed`` → increment the
          per-(task, dep) unresolved counter; escalate at threshold with
          ``EXTERNAL_DEP_UNRESOLVED`` prefix; counter resets when the dep
          resolves to a real (non-sentinel) status.
        - Other live status (``pending``, ``in-progress``, …) → wait silently;
          reset the sentinel counter for that (task, dep) pair (the dep is now
          resolvable even if not done).
        - ``external_err is not None`` → short-circuit; no counter increment,
          no escalation (fail-safe wait — resolver will retry next tick).

        **Recovery note**: once ``_on_external_dep_block`` sets a task to
        ``blocked``, the task leaves ``pending`` and this policy no longer
        evaluates it.  Recovery requires manual unblock (set the task back to
        ``pending`` after confirming the upstream dep is resolvable).  This is
        intentional for ``cancelled`` deps (permanently terminal) and
        conservative-but-safe for sentinel deps (transient resolution lag may
        require manual intervention).  The ``EXTERNAL_DEP_UNRESOLVED``
        escalation detail text prompts the human accordingly.

        This method must NOT be called from ``_deps_satisfied`` or
        ``_eligible_for_dispatch``.  Those are pure predicates called per-candidate;
        side effects here would N-fire per tick.
        """
        threshold = self.config.max_external_dep_unresolved_cycles

        if external_err is not None:
            # Resolver-degraded tick: fail-safe wait with visibility at threshold.
            # - NO sentinel-counter bumps (fail-safe invariant from task 1580).
            # - NO escalation (may recover next tick).
            # - Bump hold streak for every pending task with external deps so the
            #   hold becomes dashboard-visible once it persists too long.
            for task in pending_tasks:
                task_id = str(task.get('id', '?'))
                external_deps: list = (
                    (task.get('metadata') or {}).get('external_deps') or []
                )
                if external_deps:
                    self._note_external_hold(
                        task_id,
                        cause='resolver_degraded',
                        threshold=threshold,
                        detail=str(external_err),
                    )
            return

        for task in pending_tasks:
            task_id = str(task.get('id', '?'))
            external_deps: list = (
                (task.get('metadata') or {}).get('external_deps') or []
            )
            # Track whether any dep left this task held in a live (non-done,
            # non-sentinel) status this tick — used to drive hold-streak
            # visibility after the dep loop.
            held_live = False
            # Set True in the cancelled / threshold-crossing-sentinel branches
            # so the post-loop hold-streak code knows not to emit a spurious
            # 'deps_live' gate-held event for a task being terminally blocked.
            blocked_this_tick = False
            for dep in external_deps:
                status = external_cache.get(dep)

                if status == 'done':
                    # Satisfied — reset any accumulated sentinel counter.
                    self._external_unresolved_counts.pop((task_id, dep), None)

                elif status == 'cancelled':
                    # Strict immediate escalation — no counter increment.
                    self._external_unresolved_counts.pop((task_id, dep), None)
                    summary = (
                        f'EXTERNAL_DEP_CANCELLED: task {task_id} blocked — '
                        f'external dep {dep!r} is cancelled'
                    )
                    detail = (
                        f'Cross-project dep {dep!r} reached terminal status '
                        f'cancelled.  Task {task_id} cannot proceed; it should '
                        f'be re-architected or cancelled itself.'
                    )
                    if self._on_external_dep_block is not None:
                        await self._on_external_dep_block(
                            task_id,
                            summary=summary,
                            detail=detail,
                            category='dependency_discovered',
                        )
                    else:
                        logger.warning(
                            'External dep %r cancelled for task %s — '
                            'no _on_external_dep_block callback installed',
                            dep,
                            task_id,
                        )
                    blocked_this_tick = True

                elif status in self._EXTERNAL_SENTINEL_STATUSES:
                    # Unknown/malformed — grace-then-escalate counter.
                    count = (
                        self._external_unresolved_counts.get((task_id, dep), 0) + 1
                    )
                    self._external_unresolved_counts[(task_id, dep)] = count
                    if count >= threshold:
                        # Pop so the next crossing (if it persists) fires again.
                        self._external_unresolved_counts.pop((task_id, dep), None)
                        summary = (
                            f'EXTERNAL_DEP_UNRESOLVED: task {task_id} — '
                            f'dep {dep!r} unresolvable for {count} ticks '
                            f'(status={status!r})'
                        )
                        detail = (
                            f'Cross-project dep {dep!r} has returned sentinel '
                            f'{status!r} for {count} consecutive ticks '
                            f'(threshold={threshold}).  The dep string may be '
                            f'malformed, or the referenced project/task may not '
                            f'exist.  Task {task_id} is gated until resolved.  '
                            f'Once the dep is resolvable or done, manually set '
                            f'task {task_id} back to pending to reopen it — '
                            f'blocked tasks are not re-evaluated automatically.'
                        )
                        if self._on_external_dep_block is not None:
                            await self._on_external_dep_block(
                                task_id,
                                summary=summary,
                                detail=detail,
                                category='dependency_discovered',
                            )
                        else:
                            logger.warning(
                                'External dep %r unresolved x%d for task %s — '
                                'no _on_external_dep_block callback installed',
                                dep,
                                count,
                                task_id,
                            )
                        blocked_this_tick = True
                    else:
                        logger.debug(
                            'Task %s: external dep %r status=%r (%d/%d ticks) — '
                            'waiting silently',
                            task_id,
                            dep,
                            status,
                            count,
                            threshold,
                        )

                else:
                    # Any other live status (pending, in-progress, blocked, …):
                    # wait silently and reset the sentinel counter so transient
                    # blips don't accumulate.  Mark this task as held by a live
                    # dep so we can emit a visibility event if this persists.
                    self._external_unresolved_counts.pop((task_id, dep), None)
                    held_live = True

            # After the dep loop: drive the hold-streak for live-status holds.
            # - If any dep held the task live AND no terminal action fired this
            #   tick: bump+emit streak (genuinely waiting).
            # - Otherwise (all deps done, OR task was blocked via
            #   cancelled/threshold-sentinel): pop the streak.  A blocked task
            #   is not "still waiting" so gate_held must not fire for it.
            if held_live and not blocked_this_tick:
                self._note_external_hold(
                    task_id,
                    cause='deps_live',
                    threshold=threshold,
                )
            else:
                self._external_hold_streak.pop(task_id, None)
                self._external_hold_cause.pop(task_id, None)

    async def update_task(
        self,
        task_id: str,
        metadata: str | dict,
        *,
        append: bool = False,
        metadata_mode: str | None = None,
    ) -> bool:
        """Update task metadata via fused-memory. Returns True on success.

        Parameters
        ----------
        task_id:
            The task whose metadata should be updated.
        metadata:
            New metadata as a dict or pre-serialised JSON string.
        metadata_mode:
            Explicit merge mode forwarded to fused-memory (#1827 contract).
            ``'merge'`` (default): shallow last-write-wins — omitted keys are
            preserved, supplied keys overwrite wholesale.  This is the #4271
            fix: no-append callers (prd-tagger, module-tagger, auto-eval
            back-link) now preserve sibling keys like _causation_id and
            memory_hints instead of silently clobbering them.
            ``'additive'``: recursive list-union, dict-recursive, scalar
            OLD-wins.  Use for list-growth writes (e.g. dry_run_proposals).
            ``'replace'``: whole-blob overwrite, delete-by-omission.  Also
            the sanctioned repair path if a task's persisted metadata is
            corrupt (non-dict): under ``'merge'`` or ``'additive'``, the
            backend will raise the corrupt-blob guard rather than silently
            overwriting, so a corrupted blob requires an explicit
            ``metadata_mode='replace'`` call to repair.
        append:
            Legacy shorthand kept for back-compat.  Resolved to
            ``'additive'`` when ``True``.  Ignored when ``metadata_mode``
            is set explicitly.  Precedence: metadata_mode > append > merge.
        """
        # Resolve mode: explicit metadata_mode wins; append=True → additive;
        # default → merge (the #4271 fix — NOT replace).
        # NEVER forward 'append' on the wire: append=False resolves to REPLACE
        # on the backend, which would re-introduce the sibling-clobber bug.
        mode = metadata_mode if metadata_mode is not None else (
            'additive' if append else 'merge'
        )
        # fused-memory update_task expects metadata as a JSON string
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)
        arguments: dict = {
            'id': task_id,
            'metadata': metadata,
            'project_root': self._project_root,
            'metadata_mode': mode,
        }
        try:
            result = await self.dispatch_tool(
                'update_task',
                arguments,
                timeout=15,
            )
            # MCP tool errors return in the response body, not as exceptions
            content = result.get('result', result) if isinstance(result, dict) else result
            if isinstance(content, dict) and content.get('isError'):
                text = ''
                for block in content.get('content', []):
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text = block.get('text', '')
                        break
                logger.error(f'Failed to update task {task_id}: {text}')
                return False
            return True
        except Exception as e:
            logger.exception(
                'Failed to update task %s: %s: %s',
                task_id, type(e).__name__, e,
            )
            return False

    def _deps_satisfied(
        self,
        task: dict,
        status_map: dict[str, str],
        tasks_by_id: dict[str, dict] | None = None,
        *,
        external_status_cache: dict[str, str] | None = None,
        external_resolver_failed: bool = False,
    ) -> bool:
        """Return True if every dependency of *task* is in a terminal status.

        A dep is satisfied when its status is in :data:`TERMINAL_STATUSES`
        (``done`` or ``cancelled``).  ``cancelled`` represents an obsolete
        or duplicate task and should not block its dependents — the
        dependent re-architects against current main and either finds the
        work already merged or escalates for a different reason.

        **Intra-train allowance (PRD § 9.3):** when *tasks_by_id* is supplied,
        a dep in status ``merge-deferred`` is also treated as satisfied provided
        that BOTH the dependent (*task*) and the dep carry ``metadata.train.id``
        AND their train IDs match.  This allows the next member of an atomic
        train to dispatch as soon as its predecessor has been parked by the
        merge-deferred gate, without waiting for a full ``done`` transition.

        When *tasks_by_id* is ``None`` (the default), the allowance is disabled
        and behaviour is byte-identical to today — ``merge-deferred`` blocks like
        any other non-terminal status.  This preserves backward compatibility for
        the existing unit tests (TestDepsSatisfied, TestDepsSatisfiedLogging) and
        any external callers that don't pass the new parameter.

        The allowance also requires the dep record to be present in *tasks_by_id*.
        A missing dep (stale snapshot) is treated conservatively as blocking.

        **Cross-project external dep gate:** when *external_status_cache* is not
        ``None``, after all local deps are satisfied the method checks
        ``task.metadata.external_deps``.  An external dep is satisfied iff its
        status in the cache is exactly ``'done'``.  Any other value (live status,
        sentinel, missing from cache) is NOT satisfied.  When
        *external_resolver_failed* is ``True``, all external deps are treated as
        not satisfied regardless of the cache (fail-safe wait).

        Defaulting both new parameters to ``None``/``False`` makes the legacy
        3-arg call from ``_park_gc`` and all existing tests byte-identical.
        Side effects (escalation, counter increments) MUST NOT live here — this
        method is a pure predicate called from multiple sites.

        Handles three dependency formats:
          - dict with 'id' key: ``{'id': 1}`` or ``{'id': '1'}``
          - integer: ``1``
          - string: ``'1'``

        Emits a DEBUG log when a dependency blocks dispatch, naming the dep
        ID and its current status to aid diagnosis of premature-dispatch issues.
        Emits a separate DEBUG log when the intra-train allowance fires.
        """
        deps = task.get('dependencies', [])
        task_id = str(task.get('id', '?'))
        # Resolve this task's train id once (None if not a train member).
        # Coerce empty string to None so that '' does not accidentally match
        # another task whose id is also '' (ill-formed but defensively handled).
        task_train = (task.get('metadata') or {}).get('train')
        task_train_id: str | None = (
            task_train.get('id') or None if isinstance(task_train, dict) else None
        )
        for d in deps:
            dep_id = str(d.get('id', d) if isinstance(d, dict) else d)
            dep_status = status_map.get(dep_id, 'unknown')
            if dep_status in TERMINAL_STATUSES:
                continue
            # Intra-train allowance: merge-deferred predecessor in same train.
            if (
                dep_status == 'merge-deferred'
                and task_train_id is not None
                and tasks_by_id is not None
            ):
                dep_task = tasks_by_id.get(dep_id)
                if dep_task is not None:
                    dep_train = (dep_task.get('metadata') or {}).get('train')
                    # Coerce empty string to None (see task_train_id note above).
                    dep_train_id: str | None = (
                        dep_train.get('id') or None if isinstance(dep_train, dict) else None
                    )
                    if dep_train_id == task_train_id:
                        logger.debug(
                            'Task %s: intra-train dep satisfied: dep=%s train_id=%s',
                            task_id,
                            dep_id,
                            task_train_id,
                        )
                        continue
            logger.debug(
                'Task %s blocked: dep %s has status %s, '
                'need done or cancelled',
                task_id,
                dep_id,
                dep_status,
            )
            return False

        # External-dep gate (cross-project).  Only active when the cache is
        # supplied (not None) — defaults reproduce byte-identical legacy behaviour.
        if external_status_cache is not None:
            external_deps: list = (
                (task.get('metadata') or {}).get('external_deps') or []
            )
            for ext_dep in external_deps:
                if external_resolver_failed:
                    logger.debug(
                        'Task %s blocked: external dep %r not checked '
                        '(resolver failed — fail-safe wait)',
                        task_id,
                        ext_dep,
                    )
                    return False
                ext_status = external_status_cache.get(ext_dep)
                if ext_status != 'done':
                    logger.debug(
                        'Task %s blocked: external dep %r has status %r, '
                        'need done',
                        task_id,
                        ext_dep,
                        ext_status,
                    )
                    return False
        return True

    def _dispatch_cooldown_signal(self, task: dict) -> str | None:
        """Return the signal label if *task* carries a dispatch-cooldown signal.

        Signals (OR semantics — any one arms the gate):
        - ``recon_reset_count > 1``: task has been reset by reconciliation
          more than once (first reset is still allowed to dispatch).
        - ``steward_clear_at``: truthy value → steward stash-pop resolution.
        - ``recon_stage2_blocked_at``: truthy value → stage-2 block.
        - ``reopen_reason`` containing the substring ``'steward'`` (case-insensitive).

        Returns the matched signal label (for log messages), or ``None`` if no
        signal is present.  Used both by :meth:`_dispatch_cooldown_active` and
        at the dispatch site to guard ``_last_dispatch_at`` arming.
        """
        metadata = task.get('metadata') or {}

        # recon_reset_count > 1 (strict — first reset is allowed).
        # float() is used for type flexibility; bool values collapse to 1.0/0.0
        # (True → gate NOT armed, False → 0.0 → gate NOT armed), which is the
        # safe default. Non-finite floats compare > 1 as False, also safe.
        recon_count = metadata.get('recon_reset_count', 0)
        try:
            recon_count = float(recon_count)
        except (TypeError, ValueError):
            recon_count = 0
        if recon_count > 1:
            return 'recon_reset_count'

        # steward_clear_at: any truthy value
        if metadata.get('steward_clear_at'):
            return 'steward_clear_at'

        # recon_stage2_blocked_at: any truthy value
        if metadata.get('recon_stage2_blocked_at'):
            return 'recon_stage2_blocked_at'

        # reopen_reason containing 'steward' (case-insensitive — field is
        # human-authored prose and future producers may use different casing).
        # Non-string values (e.g. a dict from a malformed producer) are treated
        # as no-signal rather than str()-coerced: a repr containing the substring
        # 'steward' (e.g. {'steward_unblock_failure': True}) would be an
        # accidental false positive under str() coercion.
        reopen_reason = metadata.get('reopen_reason') or ''
        if isinstance(reopen_reason, str) and 'steward' in reopen_reason.lower():
            return 'reopen_reason'

        return None

    def _dispatch_cooldown_active(
        self, tid: str, signal: str | None
    ) -> bool:
        """Return ``True`` when the per-task dispatch cooldown gate is active.

        The gate is active when ALL of:
        1. The task has a prior dispatch recorded in ``_last_dispatch_at``.
        2. The elapsed time since that dispatch is less than
           ``config.dispatch_cooldown_secs`` (strict less-than).
        3. ``signal`` is non-``None`` — the task carries a reset/steward signal
           indicating it was just touched by reconciliation or the steward.

        ``signal`` must be precomputed by the caller via
        :meth:`_dispatch_cooldown_signal` before calling this method.  The
        caller owns signal evaluation; this method owns only the time-window
        check and the ``_last_dispatch_at`` expiry sweep.

        **Timing note**: ``_last_dispatch_at`` is only armed when the *dispatch
        itself* is signal-bearing (see :meth:`acquire_next`).  A steward signal
        that arrives *after* a signal-free dispatch will not retroactively
        suppress re-dispatch within the prior dispatch window — the gate only
        guards against rapid re-dispatch of tasks that were *already* flagged
        at the moment they were first picked up.
        """
        last_dispatch = self._last_dispatch_at.get(tid)
        if last_dispatch is None:
            return False
        elapsed = self._time_source() - last_dispatch
        if elapsed >= self.config.dispatch_cooldown_secs:
            # Entry is past the window and no longer affects behaviour — drop it
            # to keep the dict bounded for tasks that remain visible past the
            # window.  Tasks deleted via remove_task before their window elapses
            # leave an orphan entry until the window expires naturally; this is
            # an acceptable trade-off (bounded by dispatch_cooldown_secs).
            self._last_dispatch_at.pop(tid, None)
            return False

        return signal is not None

    def _gc_expired_cooldowns(self) -> None:
        """Sweep ``self._requeue_until`` and remove entries whose deadline has
        passed.

        Called once per tick from :meth:`acquire_next` (alongside the other
        per-tick GC sweeps) so that :meth:`_eligible_for_dispatch` can remain a
        pure, side-effect-free predicate.  An entry is expired when
        ``self._time_source() >= deadline`` — i.e. the same boundary semantics
        used by the eligibility check (``time_source() < deadline`` means *still
        cooling*).  Iterates over a snapshot of the dict so removal during
        iteration is safe.
        """
        now = self._time_source()
        for tid, deadline in list(self._requeue_until.items()):
            if deadline <= now:
                del self._requeue_until[tid]

    def _eligible_for_dispatch(
        self,
        task: dict,
        tid: str,
        status_map: dict[str, str],
        tasks_by_id: dict[str, dict] | None = None,
        *,
        external_status_cache: dict[str, str] | None = None,
        external_resolver_failed: bool = False,
    ) -> tuple[bool, str | None]:
        """Check whether *task* passes all eligibility gates for dispatch.

        Consolidates the duplicate gate logic that previously existed in both
        the scored-candidate loop and the pin-dispatch loop.  A single source
        of truth ensures that future gate additions (e.g. a new suppression
        signal) apply to both dispatch paths automatically.

        This method is a **pure predicate** — it has no side effects and may be
        called any number of times without surprise.  In particular it does *not*
        remove expired entries from ``self._requeue_until``; that bookkeeping is
        the responsibility of :meth:`_gc_expired_cooldowns`, which is called
        once per tick from :meth:`acquire_next` before either candidate loop.

        *tasks_by_id* is forwarded to :meth:`_deps_satisfied` to enable the
        intra-train merge-deferred allowance (PRD § 9.3).  When ``None``
        (the default), the allowance is disabled — behaviour is identical to
        today.  Both ``acquire_next`` call sites pass the per-tick snapshot.

        *external_status_cache* and *external_resolver_failed* are forwarded
        to :meth:`_deps_satisfied` for the cross-project external dep gate.
        When both default (``None``/``False``), the external gate is skipped —
        behaviour is byte-identical to the pre-external-dep implementation.
        The ``_park_gc`` call site does NOT pass these params (preserving
        park-GC semantics, scope containment per design decision 4).

        Returns ``(True, signal_label)`` when all gates pass.
        Returns ``(False, None)`` when any gate fails.  ``signal_label`` is
        the dispatch-cooldown signal for the task (or None), forwarded so the
        caller can pass it to post-dispatch bookkeeping without a second
        evaluation.
        """
        if task.get('status') != 'pending':
            return False, None
        if tid in self._dispatched:
            return False, None
        cooldown_deadline = self._requeue_until.get(tid)
        if cooldown_deadline is not None and self._time_source() < cooldown_deadline:
            return False, None
        if not self._deps_satisfied(
            task, status_map, tasks_by_id,
            external_status_cache=external_status_cache,
            external_resolver_failed=external_resolver_failed,
        ):
            return False, None
        signal_label = self._dispatch_cooldown_signal(task)
        if self._dispatch_cooldown_active(tid, signal_label):
            remaining_secs = (
                self.config.dispatch_cooldown_secs
                - (self._time_source() - self._last_dispatch_at.get(tid, 0.0))
            )
            metadata = task.get('metadata') or {}
            logger.info(
                'Task %s dispatch suppressed by cooldown: signal=%s=%r, remaining=%.1fs',
                tid,
                signal_label,
                metadata.get(signal_label),
                remaining_secs,
            )
            return False, None
        return True, signal_label

    def _bump_skip_and_maybe_park(
        self,
        task_id: str,
        modules: list[str],
        tier: str = DEFAULT_TIER,
    ) -> None:
        """Increment *task_id*'s skip counter; install a reservation if it
        has just crossed ``skip_threshold`` and does not already hold parks.

        *tier* is the task's effective priority — it selects a per-tier
        threshold and lease multiplier.  When the per-tier threshold is
        >= ``_INF_SKIP_THRESHOLD`` (e.g. ``9999`` for low/polish in the
        default config) parking is effectively disabled and the
        ``task_skipped`` event stream is rate-limited to geometric counts.
        """
        if not task_id:
            return
        count = self._skip_count.get(task_id, 0) + 1
        self._skip_count[task_id] = count
        threshold = self.config.fairness.skip_threshold_for(tier)
        # Rate-limit task_skipped for tiers that will never park: emit only
        # at {1, 10, 100, 1000, 10000, ...} so the event store is not flooded.
        should_emit = (
            threshold < _INF_SKIP_THRESHOLD
            or count in _GEOMETRIC_SKIP_EMIT_COUNTS
        )
        if self.event_store and should_emit:
            self.event_store.emit(
                EventType.task_skipped,
                task_id=task_id,
                data={
                    'skip_count': count,
                    'modules': modules,
                    'priority': tier,
                    'threshold': threshold,
                },
            )
        if (
            count >= threshold
            and not self.lock_table.has_parks(task_id)
        ):
            installed, evicted_pairs = self.lock_table.install_parks(task_id, modules, tier)
            logger.info(
                'Task %s reserved modules %s (skip_count=%d, tier=%s)',
                task_id, installed, count, tier,
            )
            if self.event_store:
                self.event_store.emit(
                    EventType.reservation_installed,
                    task_id=task_id,
                    data={
                        'modules': installed,
                        'skip_count': count,
                        'priority': tier,
                    },
                )
                for victim, victim_modules in evicted_pairs:
                    self.event_store.emit(
                        EventType.reservation_evicted,
                        task_id=task_id,
                        data={
                            'modules': victim_modules,
                            'preempted_by': task_id,
                            'preempted_by_priority': tier,
                            'victim': victim,
                        },
                    )

    # --- Value/h scoring helpers (P1/P2/P3) -----------------------------

    @staticmethod
    def _build_reverse_index(tasks: list[dict]) -> dict[str, set[str]]:
        """Build ``{dep_id -> {tasks_that_depend_on_dep_id}}`` in one pass.

        Replaces the O(N^2) inline dependents scan the legacy sort key used.
        """
        rev: dict[str, set[str]] = {}
        for t in tasks:
            tid = str(t.get('id', ''))
            if not tid:
                continue
            for d in t.get('dependencies', []):
                dep_id = str(d.get('id', d) if isinstance(d, dict) else d)
                if dep_id:
                    rev.setdefault(dep_id, set()).add(tid)
        return rev

    @staticmethod
    def _compute_effective_priorities(
        tasks_by_id: dict[str, dict],
        reverse_index: dict[str, set[str]],
        status_map: dict[str, str],
        override_boosts: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Priority inheritance (P1) with optional boost overlay.

        ``effective_priority(t) = min-rank(own, boost, effective(d) for d in dependents(t))``
        walking only undone dependents (``status not in {done, cancelled}``).

        ``override_boosts`` maps ``task_id -> boost_tier`` for tasks with an
        active priority-override boost.  When provided, the boost tier competes
        in the same min-rank race as the task's own tier and inherited tiers
        from dependents.  Defaults to None (no boost overlay).

        Tri-state DFS guards against dependency cycles: on a cycle the task
        contributes only its own priority and a WARN is logged.
        """
        _boosts: dict[str, str] = override_boosts or {}
        memo: dict[str, str] = {}
        visiting: set[str] = set()
        walked: set[str] = set()

        def walk(tid: str) -> str:
            if tid in memo:
                return memo[tid]
            if tid in visiting:
                logger.warning(
                    'Priority inheritance: cycle detected at task %s; using own priority only',
                    tid,
                )
                task = tasks_by_id.get(tid, {})
                return coerce_tier(task.get('priority'))
            visiting.add(tid)
            task = tasks_by_id.get(tid, {})
            own = coerce_tier(task.get('priority'))
            best_rank = PRIORITY_RANK[own]
            # Fold in boost overlay before the inheritance race.
            boost_tier = _boosts.get(tid)
            if boost_tier is not None:
                boost_rank = PRIORITY_RANK[coerce_tier(boost_tier)]
                if boost_rank < best_rank:
                    best_rank = boost_rank
            for parent_id in reverse_index.get(tid, ()):
                parent_status = status_map.get(parent_id, '')
                if parent_status in ('done', 'cancelled'):
                    continue
                parent_eff = walk(parent_id)
                parent_rank = PRIORITY_RANK[parent_eff]
                if parent_rank < best_rank:
                    best_rank = parent_rank
            visiting.discard(tid)
            walked.add(tid)
            result = PRIORITY_TIERS[best_rank]
            memo[tid] = result
            return result

        for tid in tasks_by_id:
            if tid not in memo:
                walk(tid)
        return memo

    @staticmethod
    def _compute_transitive_counts(
        tasks_by_id: dict[str, dict],
        reverse_index: dict[str, set[str]],
        status_map: dict[str, str],
    ) -> dict[str, int]:
        """CPM proxy (P3): BFS over the reverse-dependency graph per task,
        counting undone descendants.  Memoized per cycle, O(N+E) overall.
        """
        memo: dict[str, int] = {}

        def bfs(root: str) -> int:
            seen: set[str] = set()
            queue: deque[str] = deque([root])
            count = 0
            while queue:
                current = queue.popleft()
                for child in reverse_index.get(current, ()):
                    if child in seen:
                        continue
                    seen.add(child)
                    if status_map.get(child, '') in ('done', 'cancelled'):
                        # Walk through to find further undone descendants (they
                        # may themselves unlock work).
                        queue.append(child)
                        continue
                    count += 1
                    queue.append(child)
            return count

        for tid in tasks_by_id:
            memo[tid] = bfs(tid)
        return memo

    def _compute_age(self, task_id: str, max_id: int) -> int:
        """Return this task's age, in "newer-task-count" units.

        Anchors are lazily initialized in :meth:`_update_age_anchors`; they
        reset to the *current* max_id on resurrection so a cancelled→pending
        task does not inherit accumulated age.
        """
        anchor = self._pending_anchor.get(task_id)
        if anchor is None:
            return 0
        return max(0, max_id - anchor)

    def _update_age_anchors(self, tasks: list[dict], max_id: int) -> None:
        """Maintain per-task age anchors across ticks.

        - First time we see a task as pending with no prior non-pending
          history, anchor to its own numeric id (so genuinely-old pending
          tasks carry accumulated age from the start).
        - First time we see a task as pending after having seen it
          non-pending, anchor to *current max_id* (resurrection resets age).
        - On any non-pending observation, drop the anchor and mark the task
          as ever-non-pending so the next pending appearance is a fresh start.
        """
        for t in tasks:
            tid = str(t.get('id', ''))
            if not tid:
                continue
            status = t.get('status', '')
            if status != 'pending':
                self._pending_anchor.pop(tid, None)
                if status:
                    self._was_non_pending.add(tid)
                continue
            if tid in self._pending_anchor:
                continue
            # First-seen pending for this tid.
            if tid in self._was_non_pending:
                # Resurrection — start fresh from now.
                self._pending_anchor[tid] = max_id
            elif tid.isdigit():
                self._pending_anchor[tid] = int(tid)
            else:
                self._pending_anchor[tid] = max_id

    def _compute_score(
        self,
        tier: str,
        age: int,
        transitive_count: int,
    ) -> float:
        """Compute the total dispatch score for a task.

        ``score = TIER_BASE[tier] + min(α*age + β*log1p(trans), TIER_WIDTH - 1)``

        The combined age+CPM bonus is capped below ``TIER_WIDTH`` so bonuses
        can never bump a task across a tier boundary — priority always wins.
        """
        tier = coerce_tier(tier)
        base = TIER_BASE[tier]
        age_bonus = self.config.age_alpha * float(age)
        cpm_bonus = self.config.cpm_beta * math.log1p(max(0, transitive_count))
        bonus = min(age_bonus + cpm_bonus, float(TIER_WIDTH - 1))
        return float(base) + bonus

    def _emit_override_diff_events(
        self,
        prev: dict[str, OverrideRow],
        cur: dict[str, OverrideRow],
    ) -> None:
        """Diff prev vs cur override snapshots and emit override change events.

        Called once per tick after all in-memory override mutations are complete.
        Each change in ``boost_tier`` emits exactly one event per task per tick.
        Pin-state diffs emit ``task_pinned`` / ``task_unpinned`` per task, and a
        SINGLE ``pin_queue_reordered`` event per tick when any ``pin_order`` shifts.

        **Pin-order observation contract** (decided in task 1290)

        These pin/override events are **observability-only** and may lag
        OverrideStore / MCP writes by one tick (they are emitted at the end of
        the first ``acquire_next()`` call that follows an override mutation).

        ``pin_queue_reordered`` is emitted **only** when a pure reorder occurs
        — that is, when one or more tasks that were already pinned shift to a
        different ``pin_order`` via ``reorder_pin_queue()``.  It carries the
        complete post-change ordering in ``data['new_order']`` (a list of
        task-ids, ascending by new pin_order).

        ``pin_queue_reordered`` is intentionally **NOT** emitted on pin add
        (``task_pinned``) or pin remove (``task_unpinned``): the add/remove is
        already fully described by those events, and emitting a reorder event
        on top would be redundant noise with no additional information.

        Consumer strategies for determining current pin order:

        (i)  **Event recomposition** — replay events in emission order:
             ``task_pinned`` → append ``task_id`` to the ordered list;
             ``task_unpinned`` → remove ``task_id`` from the list;
             ``pin_queue_reordered`` → replace the list with ``new_order``.
             This strategy is eventually consistent (lags by at most one tick).

        (ii) **Authoritative snapshot** (preferred for "always-current" needs)
             — call ``OverrideStore.get_pin_queue(project_root)`` or read
             ``snapshot.pin_queue`` from the scheduler's public API directly.
             No event recomposition needed; used by consumers such as MCP
             ``get_pin_queue`` tools and dashboard scheduler pages.

        See also: ``EventType`` pin-events comment block in ``event_store.py``
        for the same contract at the taxonomy definition.
        """
        if not self.event_store:
            return
        all_ids = set(prev) | set(cur)
        pin_queue_changed = False
        for tid in all_ids:
            prev_row = prev.get(tid)
            cur_row = cur.get(tid)

            # --- boost_tier diffs ---
            prev_boost = prev_row.boost_tier if prev_row else None
            cur_boost = cur_row.boost_tier if cur_row else None
            if prev_boost != cur_boost:
                if cur_boost is not None:
                    self.event_store.emit(
                        EventType.priority_override_set,
                        task_id=tid,
                        data={'boost_tier': cur_boost},
                    )
                else:
                    self.event_store.emit(
                        EventType.priority_override_cleared,
                        task_id=tid,
                        data={'previous_boost_tier': prev_boost},
                    )

            # --- pin state diffs ---
            prev_pinned = prev_row.pinned if prev_row else False
            cur_pinned = cur_row.pinned if cur_row else False
            if prev_pinned != cur_pinned:
                # A task was pinned or unpinned.  Emit task_pinned / task_unpinned
                # for the individual task but do NOT set pin_queue_changed.
                # pin_queue_reordered is deliberately withheld on add/remove —
                # see the "Pin-order observation contract" in the method docstring
                # for the full rationale and consumer guidance (task 1290).
                if cur_pinned:
                    self.event_store.emit(
                        EventType.task_pinned,
                        task_id=tid,
                        data={'pin_order': cur_row.pin_order if cur_row else None},
                    )
                else:
                    self.event_store.emit(
                        EventType.task_unpinned,
                        task_id=tid,
                        data={
                            'previous_pin_order': (
                                prev_row.pin_order if prev_row else None
                            )
                        },
                    )
            elif prev_pinned and cur_pinned:
                # Both pinned — detect pin_order shift (signals a pure reorder).
                # Only fires for tasks that were already in the queue and moved
                # to a different position (via reorder_pin_queue).
                prev_order = prev_row.pin_order if prev_row else None
                cur_order = cur_row.pin_order if cur_row else None
                if prev_order != cur_order:
                    pin_queue_changed = True

            # --- reserve_now diffs (False→True only; True→False is handled by
            # --- reserve_now_consumed at the short-circuit emit site)
            prev_rn = prev_row.reserve_now if prev_row else False
            cur_rn = cur_row.reserve_now if cur_row else False
            if not prev_rn and cur_rn:
                self.event_store.emit(
                    EventType.reserve_now_armed,
                    task_id=tid,
                    data={},
                )

        # One ``pin_queue_reordered`` event per tick for any pin_order change.
        # Derive the new order from the in-memory ``cur`` snapshot (no second
        # SQLite round-trip — the post-GC snapshot is already authoritative).
        if pin_queue_changed:
            new_order = [
                tid
                for tid, _ in sorted(
                    ((t, r) for t, r in cur.items() if r.pinned),
                    key=lambda x: (x[1].pin_order if x[1].pin_order is not None else 0),
                )
            ]
            self.event_store.emit(
                EventType.pin_queue_reordered,
                data={'new_order': new_order},
            )

    async def acquire_next(self) -> TaskAssignment | None:
        """Find next eligible task under the value/h scoring model.

        Dispatch order is determined by ``_compute_score()``: tier base is
        dominant, age + CPM bonuses order tasks within a tier, and
        per-tier slot caps reserve headroom for higher-value work.

        Returns ``None`` immediately when the scheduler is paused (park-stop).
        No MCP round-trips, no GC, no override snapshots are performed while
        paused — all resume cleanly on the first unpaused tick.
        """
        if self._paused:
            logger.debug(
                'acquire_next() short-circuit: scheduler is paused (reason=%r)',
                self._pause_reason,
            )
            return None

        tasks = await self.get_tasks(statuses=ACTIVE_TASK_STATUSES)
        if not tasks:
            return None

        # Status + id indices, built once per tick.
        status_map: dict[str, str] = {}
        tasks_by_id: dict[str, dict] = {}
        max_id = 0
        for t in tasks:
            tid = str(t.get('id', ''))
            if not tid:
                continue
            status_map[tid] = t.get('status', 'unknown')
            tasks_by_id[tid] = t
            if tid.isdigit():
                max_id = max(max_id, int(tid))

        # Maintain age anchors (resurrected tasks reset their anchor).
        self._update_age_anchors(tasks, max_id)

        # Correctness crux (γ2): the active-only filter drops terminal tasks
        # from the result, so dep-ids referencing DONE/CANCELLED tasks will be
        # absent from status_map. _deps_satisfied reads status_map.get(dep_id,
        # 'unknown'), so those deps would block dispatching forever. Fix: collect
        # local dep-ids referenced by the fetched tasks that are NOT already in
        # status_map, then backfill them via the lean get_statuses(ids=missing).
        # In production this backfill commonly fires every tick (most pending tasks
        # have at least one done dep), but the two-call total (active get_tasks +
        # compact get_statuses) is still a net win over the old single full get_tasks
        # call (~95% smaller payload per γ1's get_statuses path).  In unit tests
        # whose get_tasks mocks return the full set (incl. done deps), status_map
        # is already complete → missing_dep_ids is empty → zero get_statuses calls.
        _all_dep_ids: set[str] = set()
        for _t in tasks:
            for _d in (_t.get('dependencies') or []):
                _dep_id = str(
                    _d.get('id', _d) if isinstance(_d, dict) else _d
                )
                if _dep_id:
                    _all_dep_ids.add(_dep_id)
        _missing_dep_ids = sorted(_all_dep_ids - set(status_map))
        if _missing_dep_ids:
            _backfilled, _backfill_err = await self.get_statuses(
                ids=_missing_dep_ids
            )
            if resolver_failed(_backfilled, _backfill_err):
                logger.warning(
                    'acquire_next: dep-status backfill degraded '
                    '(err=%r, missing_dep_ids=%r) — affected pending tasks '
                    'held fail-safe-wait',
                    _backfill_err,
                    _missing_dep_ids,
                )
                for _t in tasks:
                    _tid = str(_t.get('id', ''))
                    if _t.get('status') != 'pending' or not _tid:
                        continue
                    for _d in (_t.get('dependencies') or []):
                        _dep_id = str(
                            _d.get('id', _d) if isinstance(_d, dict) else _d
                        )
                        if _dep_id in _missing_dep_ids:
                            _key = (_tid, _dep_id)
                            self._local_backfill_unresolved_counts[_key] = (
                                self._local_backfill_unresolved_counts.get(_key, 0) + 1
                            )
                            _cnt = self._local_backfill_unresolved_counts[_key]
                            # Reuse max_external_dep_unresolved_cycles as the
                            # grace threshold — same "consecutive ticks before
                            # loud escalation" semantics as the external-dep
                            # resolver-degraded path being mirrored here.
                            # A dedicated max_local_backfill_unresolved_cycles
                            # field is deferred (config.py is out of scope).
                            if _cnt >= self.config.max_external_dep_unresolved_cycles:
                                logger.warning(
                                    'acquire_next: local dep backfill unresolved '
                                    'for %d consecutive ticks '
                                    '(task=%s, dep=%s) — possible scheduler '
                                    'degradation',
                                    _cnt,
                                    _tid,
                                    _dep_id,
                                )
            else:
                status_map.update(_backfilled)
                # Reset the consecutive-tick counters for deps that resolved
                # successfully in this backfill — mirrors the
                # _external_unresolved_counts.pop(...) on the 'done' branch in
                # _apply_external_dep_policy.  Without this reset, a dep that
                # degrades → recovers → degrades again accumulates across the
                # gap, making the "consecutive" counters and warning messages
                # misreport the streak length.
                for _t in tasks:
                    _tid = str(_t.get('id', ''))
                    if _t.get('status') != 'pending' or not _tid:
                        continue
                    for _d in (_t.get('dependencies') or []):
                        _dep_id = str(
                            _d.get('id', _d) if isinstance(_d, dict) else _d
                        )
                        if _dep_id in _backfilled:
                            self._local_backfill_unresolved_counts.pop(
                                (_tid, _dep_id), None
                            )
                # Partial-response guard: get_statuses returned a valid
                # (non-error) dict that is still missing some of the requested
                # dep ids.  Treat those still-missing ids as degraded — warn +
                # bump counter — mirroring the missing-key guard in
                # get_external_statuses (~1545).  The absent ids stay out of
                # status_map so _deps_satisfied returns False → fail-safe-wait,
                # now VISIBLE rather than a silent idle.
                _still_missing = set(_missing_dep_ids) - set(_backfilled)
                if _still_missing:
                    logger.warning(
                        'acquire_next: dep-status backfill returned partial '
                        'result (missing %d/%d dep ids: %r) — affected '
                        'pending tasks held fail-safe-wait',
                        len(_still_missing),
                        len(_missing_dep_ids),
                        sorted(_still_missing),
                    )
                    for _t in tasks:
                        _tid = str(_t.get('id', ''))
                        if _t.get('status') != 'pending' or not _tid:
                            continue
                        for _d in (_t.get('dependencies') or []):
                            _dep_id = str(
                                _d.get('id', _d) if isinstance(_d, dict) else _d
                            )
                            if _dep_id in _still_missing:
                                _key = (_tid, _dep_id)
                                self._local_backfill_unresolved_counts[_key] = (
                                    self._local_backfill_unresolved_counts.get(_key, 0) + 1
                                )
                                _cnt = self._local_backfill_unresolved_counts[_key]
                                # Same threshold as the degraded path above.
                                if _cnt >= self.config.max_external_dep_unresolved_cycles:
                                    logger.warning(
                                        'acquire_next: local dep absent from '
                                        'backfill for %d consecutive ticks '
                                        '(task=%s, dep=%s) — possible '
                                        'scheduler degradation',
                                        _cnt,
                                        _tid,
                                        _dep_id,
                                    )

        # Owner-state park-GC sweep. Replaces the wall-clock lease mechanic:
        # a park whose owner is terminal / missing / deps-unsatisfied has no
        # reason to keep blocking other tasks, so it's evicted now.
        def _park_gc(tid: str) -> bool:
            status = status_map.get(tid)
            if status in TERMINAL_STATUSES:
                return True
            if tid not in tasks_by_id:
                return True
            return not self._deps_satisfied(tasks_by_id[tid], status_map, tasks_by_id)

        gc_evicted = self.lock_table.prune_owners(_park_gc)
        for owner in gc_evicted:
            self._skip_count.pop(owner, None)
            if self.event_store:
                owner_status = status_map.get(owner)
                if owner_status in TERMINAL_STATUSES:
                    reason = f'terminal:{owner_status}'
                elif owner not in tasks_by_id:
                    reason = 'missing'
                else:
                    reason = 'deps_unsatisfied'
                self.event_store.emit(
                    EventType.reservation_expired,
                    task_id=owner,
                    data={'reason': reason},
                )

        # Drop _last_dispatch_at, _skip_count, _module_cache, _pending_anchor,
        # and sub-threshold _external_unresolved_counts entries for tasks that are:
        #   (a) in a terminal status in status_map, OR
        #   (b) absent from tasks_by_id (active-only filter dropped them because
        #       they completed between ticks — γ2: previously the full get_tasks
        #       result kept completed tasks visible so the terminal sweep could
        #       clean them up; active-only filtering removes them from the result).
        # so a future legitimate re-dispatch (e.g. cancelled -> pending
        # re-architect, or a freshly-created task reusing the id) starts from a
        # clean slate.  Resurrection-safe: a re-queued task re-derives modules
        # and re-accumulates its skip count fresh.
        # _pending_anchor and _was_non_pending are handled here directly (not only
        # in _update_age_anchors) because active-only filtering means terminal
        # tasks are absent from the `tasks` list that _update_age_anchors iterates.
        # Without this, a task that goes pending → terminal (e.g. cancelled while
        # pending, never dispatched) leaks its _pending_anchor entry permanently.
        # Recording _was_non_pending preserves resurrection semantics: if the task
        # is re-queued to pending, it gets a fresh max_id anchor instead of
        # re-using its old (stale) numeric id as the age anchor.
        _stale_ids: set[str] = set()
        # Iterate the union of all tracked bookkeeping keys so we catch ids that
        # are absent from status_map entirely (completed, dropped by active filter).
        # _pending_anchor is included so anchor-only entries (tasks that went
        # pending → terminal without ever being dispatched) are also caught.
        _all_tracked: set[str] = (
            set(self._last_dispatch_at)
            | set(self._skip_count)
            | set(self._module_cache)
            | set(self._pending_anchor)
        )
        for tid_str in _all_tracked:
            if status_map.get(tid_str) in TERMINAL_STATUSES or tid_str not in tasks_by_id:
                self._last_dispatch_at.pop(tid_str, None)
                self._skip_count.pop(tid_str, None)
                self._module_cache.pop(tid_str, None)
                self._pending_anchor.pop(tid_str, None)
                self._was_non_pending.add(tid_str)
                _stale_ids.add(tid_str)
        # _external_unresolved_counts is keyed by (task_id, dep); sweep
        # separately to avoid mutating while iterating.  A sub-threshold counter
        # entry would otherwise leak permanently if the task terminates before
        # crossing the escalation threshold (e.g. manually cancelled while count=1).
        if _stale_ids and self._external_unresolved_counts:
            _stale_ext_keys = [
                k for k in self._external_unresolved_counts
                if k[0] in _stale_ids
            ]
            for k in _stale_ext_keys:
                del self._external_unresolved_counts[k]
        if _stale_ids and self._local_backfill_unresolved_counts:
            _stale_local_keys = [
                k for k in self._local_backfill_unresolved_counts
                if k[0] in _stale_ids
            ]
            for k in _stale_local_keys:
                del self._local_backfill_unresolved_counts[k]
        # _external_hold_streak and _external_hold_cause are keyed by task_id
        # (str); GC alongside _external_unresolved_counts so they stay bounded.
        if _stale_ids and (self._external_hold_streak or self._external_hold_cause):
            for tid in _stale_ids:
                self._external_hold_streak.pop(tid, None)
                self._external_hold_cause.pop(tid, None)

        # Per-tick GC of the requeue-cooldown dict — keeps the dict bounded
        # and lets _eligible_for_dispatch stay side-effect-free.  Runs before
        # both the scored-candidate loop and the pin-dispatch loop so both
        # observe post-GC state, matching the contract previously provided by
        # the lazy per-call delete inside _eligible_for_dispatch.
        self._gc_expired_cooldowns()

        # Cross-project external dep gate (invariant 5 — one batched call per tick).
        # Collect the union of metadata.external_deps across all pending tasks; if
        # non-empty issue ONE get_external_statuses call.  Zero deps → zero calls.
        # The per-tick cache is then forwarded to _eligible_for_dispatch at both
        # call sites below; _apply_external_dep_policy runs the side-effecting pass
        # (counter increments, escalation callbacks) exactly once per tick.
        # The _park_gc call site above does NOT receive the cache, preserving park-GC
        # semantics (design decision 4: scope containment).
        _pending_tasks_with_ext: list[dict] = [
            t for t in tasks
            if t.get('status') == 'pending'
            and (t.get('metadata') or {}).get('external_deps')
        ]
        _ext_dep_union: list[str] = list({
            dep
            for t in _pending_tasks_with_ext
            for dep in ((t.get('metadata') or {}).get('external_deps') or [])
        })
        if _ext_dep_union:
            external_cache, external_err = await self.get_external_statuses(_ext_dep_union)
        else:
            external_cache, external_err = {}, None
        try:
            await self._apply_external_dep_policy(
                _pending_tasks_with_ext, external_cache, external_err
            )
        except Exception:
            # A failure in the policy pass (e.g. set_task_status raising inside
            # the _on_external_dep_block callback) must not abort the whole tick.
            # Degrade to a fail-safe wait: the gate stays closed via
            # _external_resolver_failed below, and the policy retries next tick.
            logger.warning(
                'External dep policy pass raised — degrading to fail-safe wait this tick',
                exc_info=True,
            )
        _external_resolver_failed = external_err is not None

        # Load priority-override snapshot for this tick.
        current_overrides: dict[str, OverrideRow] = (
            self._override_store.get_overrides(self._project_root)
            if self._override_store
            else {}
        )

        # Override GC: remove override rows for tasks that are terminal or missing,
        # and remove rows whose TTL has elapsed.
        # Runs alongside the park_gc sweep so the rest of the tick sees post-GC state.
        if self._override_store and current_overrides:
            terminal_or_missing_ids: set[str] = (
                {tid for tid, st in status_map.items() if st in TERMINAL_STATUSES}
                | (set(current_overrides.keys()) - set(tasks_by_id.keys()))
            )
            if terminal_or_missing_ids:
                cleared_overrides = self._override_store.clear_terminal(
                    self._project_root, terminal_or_missing_ids
                )
                for tid in cleared_overrides:
                    current_overrides.pop(tid, None)
            # TTL sweep: clear any rows whose ttl_until has elapsed.
            expired_overrides = self._override_store.clear_expired(
                self._project_root, datetime.now(UTC)
            )
            for tid in expired_overrides:
                current_overrides.pop(tid, None)

        # Snapshot pre-short-circuit overrides for diff-detection.  The
        # short-circuit below mutates current_overrides (clears reserve_now) so
        # the diff would otherwise never see a False→True transition for
        # reserve_now.  The next tick's prev snapshot uses the post-short-circuit
        # state (current_overrides), keeping the per-tick diff semantics correct.
        overrides_for_diff = dict(current_overrides)

        # Reserve-Now short-circuit: for any task with reserve_now=1, eagerly
        # install parks on its modules then clear the flag.  This is single-tick
        # fire-and-forget — the parks will survive until the owner-GC sweep evicts
        # them (owner goes terminal/missing or its deps lapse).  The loop skips
        # only tasks that are absent from the task list entirely, or that are
        # already in TERMINAL_STATUSES (done/cancelled).  A blocked-but-non-terminal
        # task DOES get parks installed — reserve_now is an explicit user override
        # so holding the modules for it is intentional.  The park-GC sweep
        # (_park_gc) reclaims those parks once the owner transitions to terminal.
        if self._override_store:
            for rid, rrow in list(current_overrides.items()):
                if not rrow.reserve_now:
                    continue
                if rid not in tasks_by_id:
                    continue
                if status_map.get(rid) in TERMINAL_STATUSES:
                    continue
                r_task = tasks_by_id[rid]
                r_modules = self._get_modules(r_task)
                r_tier = coerce_tier(r_task.get('priority'))
                # Clear the flag BEFORE installing parks.  install_parks is
                # naturally idempotent (duplicate parks are a no-op), so if the
                # process crashes between clear and install, the next tick re-runs
                # install harmlessly.  The opposite order risks a duplicate
                # reservation_installed event if the clear fails after a
                # successful install.
                #
                # In-process exceptions from install_parks are handled separately:
                # the flag is restored via set_override so the next tick retries.
                self._override_store.clear_override(
                    self._project_root, rid, field='reserve_now'
                )
                try:
                    installed, _evicted = self.lock_table.install_parks(
                        rid, r_modules, r_tier
                    )
                    if self.event_store and installed:
                        self.event_store.emit(
                            EventType.reserve_now_consumed,
                            task_id=rid,
                            data={'modules': installed, 'priority': r_tier},
                        )
                    # Reflect the cleared flag in the in-memory snapshot so
                    # downstream diff-detection doesn't spuriously re-emit for
                    # this tick.
                    current_overrides[rid] = OverrideRow(
                        boost_tier=rrow.boost_tier,
                        pinned=rrow.pinned,
                        pin_order=rrow.pin_order,
                        reserve_now=False,
                        ttl_until=rrow.ttl_until,
                    )
                except Exception:
                    logger.warning(
                        'reserve_now: install_parks failed for task %s; restoring reserve_now flag',
                        rid,
                        exc_info=True,
                    )
                    try:
                        self._override_store.set_override(
                            self._project_root, rid, reserve_now=True
                        )
                    except Exception:
                        logger.warning(
                            'reserve_now: failed to restore reserve_now flag for task %s',
                            rid,
                            exc_info=True,
                        )
                        # Restore failed — DB still holds reserve_now=False (cleared
                        # above).  Mirror that in memory so the diff-layer doesn't
                        # fabricate a spurious priority_override_cleared event next tick.
                        current_overrides[rid] = OverrideRow(
                            boost_tier=rrow.boost_tier,
                            pinned=rrow.pinned,
                            pin_order=rrow.pin_order,
                            reserve_now=False,
                            ttl_until=rrow.ttl_until,
                        )
                    continue

        # Diff-detect override changes and emit priority_override_* events.
        # Uses the pre-short-circuit override snapshot (overrides_for_diff) so
        # reserve_now False→True transitions are visible even though the
        # short-circuit already cleared the flag in current_overrides.
        #
        # On the first tick after a scheduler restart the snapshot starts empty.
        # Diffing against {} would emit spurious priority_override_set / task_pinned
        # events for every pre-existing override, confusing downstream consumers
        # that interpret them as fresh user actions.  We skip the diff on the
        # first tick and seed the snapshot so subsequent ticks diff correctly.
        if self._overrides_initialized:
            self._emit_override_diff_events(self._prev_overrides_snapshot, overrides_for_diff)
        else:
            self._overrides_initialized = True
        self._prev_overrides_snapshot = dict(current_overrides)

        # Build reverse index + compute effective priorities + CPM counts
        # once per tick (O(N+E)).
        reverse_index = self._build_reverse_index(tasks)
        override_boosts = {
            tid: row.boost_tier
            for tid, row in current_overrides.items()
            if row.boost_tier
        }
        effective_priorities = self._compute_effective_priorities(
            tasks_by_id, reverse_index, status_map,
            override_boosts=override_boosts or None,
        )
        self._last_effective_priorities = dict(effective_priorities)
        transitive_counts = self._compute_transitive_counts(
            tasks_by_id, reverse_index, status_map
        )

        # Filter to pending tasks whose deps are all done and that aren't
        # dispatched or in their post-requeue cooldown window.
        # _eligible_for_dispatch encapsulates all gates so the pin-dispatch
        # loop below uses the same logic (single source of truth).
        candidates: list[dict] = []
        candidate_signals: dict[str, str | None] = {}
        for t in tasks:
            tid_str = str(t.get('id', ''))
            if not tid_str:
                continue
            eligible, signal_label = self._eligible_for_dispatch(
                t, tid_str, status_map, tasks_by_id,
                external_status_cache=external_cache,
                external_resolver_failed=_external_resolver_failed,
            )
            if not eligible:
                continue
            # signal_label is stashed and reused at the dispatch arm site so
            # _dispatch_cooldown_signal is not called a second time.
            # Note: cooldown-suppressed tasks intentionally bypass the fairness
            # skip-bookkeeping machinery for the duration of the settle window.
            # They are invisible to skip counters and parking logic until the
            # window elapses, at which point they re-enter the normal candidate
            # pool and can accumulate skips like any other task.
            candidates.append(t)
            candidate_signals[tid_str] = signal_label

        if not candidates:
            return None

        # Pin-dispatch: try pinned tasks in pin_order ASC before scoring.
        # Pinned candidates bypass scoring entirely but still respect lock
        # availability and eligibility checks (status, deps, cooldown).
        # On lock conflict, fall through to the next pinned candidate without
        # touching skip counters or arming parks (pins bypass fairness).
        #
        # The pin queue is derived from the in-memory current_overrides snapshot
        # (already loaded above) so we avoid a second SQLite round-trip on every
        # tick.  The post-GC snapshot is already authoritative.
        if self._override_store:
            pin_queue: list[tuple[str, OverrideRow]] = sorted(
                ((tid, row) for tid, row in current_overrides.items() if row.pinned),
                key=lambda x: (x[1].pin_order if x[1].pin_order is not None else 0),
            )
            for pin_tid, _pin_row in pin_queue:
                if pin_tid not in tasks_by_id:
                    continue
                pin_task = tasks_by_id[pin_tid]
                # Re-use the same eligibility helper as the scored-candidate
                # loop to keep both paths in sync.  A future gate addition only
                # needs to be added to _eligible_for_dispatch.
                eligible, pin_signal = self._eligible_for_dispatch(
                    pin_task, pin_tid, status_map, tasks_by_id,
                    external_status_cache=external_cache,
                    external_resolver_failed=_external_resolver_failed,
                )
                if not eligible:
                    continue
                # Eligible pinned candidate — try to acquire its modules.
                pin_modules = self._get_modules(pin_task)
                if self.lock_table.try_acquire(pin_tid, pin_modules):
                    self._dispatched.add(pin_tid)
                    if pin_signal is not None:
                        self._last_dispatch_at[pin_tid] = self._time_source()
                    pin_pri = effective_priorities.get(
                        pin_tid, coerce_tier(pin_task.get('priority'))
                    )
                    self._dispatched_priority[pin_tid] = pin_pri
                    if self.event_store:
                        self.event_store.emit(
                            EventType.lock_acquired,
                            task_id=pin_tid,
                            data={'modules': pin_modules, 'priority': pin_pri},
                        )
                    await self._write_snapshot_best_effort()
                    return TaskAssignment(
                        task_id=pin_tid, task=pin_task, modules=pin_modules
                    )
                # Lock conflict — fall through to next pinned candidate.
                # No skip-bookkeeping for pinned tasks (pins bypass fairness).

        # Score each candidate.  Higher score wins; ties broken by task_id
        # string order (stable, FIFO-ish for numeric ids).
        scored: list[tuple[float, str, dict, str]] = []
        for t in candidates:
            tid = str(t.get('id', ''))
            pri = effective_priorities.get(tid, coerce_tier(t.get('priority')))
            age = self._compute_age(tid, max_id)
            transitive = transitive_counts.get(tid, 0)
            score = self._compute_score(pri, age, transitive)
            scored.append((score, tid, t, pri))

        scored.sort(key=lambda entry: (-entry[0], entry[1]))

        # DEBUG: log the top 3 so α/β tuning is post-hoc diagnosable.
        if logger.isEnabledFor(logging.DEBUG):
            top3 = scored[:3]
            logger.debug(
                'acquire_next top candidates: %s',
                [
                    {
                        'id': e[1],
                        'score': round(e[0], 2),
                        'pri': e[3],
                    }
                    for e in top3
                ],
            )

        # Strict top is the highest-scoring eligible candidate.  We track it
        # for fairness bookkeeping (skip counter / park installation).
        top_score, top_id, top_task, top_pri = scored[0]
        top_modules = self._get_modules(top_task)
        top_had_parks = self.lock_table.has_parks(top_id)

        for _score, task_id, task, pri in scored:
            modules = self._get_modules(task)
            if self.lock_table.try_acquire(task_id, modules):
                self._dispatched.add(task_id)
                # arm cooldown gate — only for signal-bearing dispatches.
                # Steward signals that arrive *after* a signal-free dispatch
                # will not retroactively suppress re-dispatch; the gate is
                # intentionally scoped to tasks that were already flagged
                # when first picked up (bounded _last_dispatch_at size).
                if candidate_signals.get(task_id) is not None:
                    self._last_dispatch_at[task_id] = self._time_source()
                self._dispatched_priority[task_id] = pri
                if task_id == top_id:
                    self._skip_count.pop(task_id, None)
                    if top_had_parks:
                        self.lock_table.clear_parks_for(task_id)
                        if self.event_store:
                            self.event_store.emit(
                                EventType.reservation_used,
                                task_id=task_id,
                                data={'modules': modules, 'priority': pri},
                            )
                else:
                    # A lower-ranked task won — top was passed over this tick.
                    self._bump_skip_and_maybe_park(top_id, top_modules, top_pri)
                if self.event_store:
                    self.event_store.emit(
                        EventType.lock_acquired,
                        task_id=task_id,
                        data={'modules': modules, 'priority': pri},
                    )
                await self._write_snapshot_best_effort()
                return TaskAssignment(task_id=task_id, task=task, modules=modules)

        # Loop exhausted with no acquire — top candidate was also skipped.
        self._bump_skip_and_maybe_park(top_id, top_modules, top_pri)
        await self._write_snapshot_best_effort()
        return None

    def get_state_snapshot(self) -> dict:
        """Return a deep-copy snapshot of current in-memory scheduler state.

        Contains ten top-level keys:
        - skip_counts: {task_id: int}
        - parks: {task_id: {modules: [...], installed_at: str}}
        - effective_priorities: {task_id: str}
        - pin_queue: [{task_id: str, order: int}, ...]
        - overrides: {task_id: {boost_tier, pinned, reserve_now, ttl_until}}
        - current_holders: {module: task_id}
        - lock_depth: int — top-level normalization depth for lock keys, so
          consumers (e.g. the dashboard) can normalize file footprints the
          same way before matching against current_holders.
        - is_paused: bool — True when the scheduler is park-stop paused
        - pause_reason: str | None — human-readable reason, or None when not paused
        - snapshot_at: ISO8601 timestamp
        """
        # skip_counts — plain int values, safe to copy.
        skip_counts = dict(self._skip_count)

        # parks — delegate to the public accessor so ModuleLockTable owns its
        # own representation (no private-attribute access from Scheduler).
        parks = self.lock_table.snapshot_parks()

        # effective_priorities — already a shallow dict of str→str.
        effective_priorities = dict(self._last_effective_priorities)

        # pin_queue — read from the override store if available.
        pin_queue: list[dict] = []
        if self._override_store:
            try:
                for tid, row in self._override_store.get_pin_queue(self._project_root):
                    pin_queue.append({'task_id': tid, 'order': row.pin_order})
            except Exception:
                if not self._override_store_warned:
                    self._override_store_warned = True
                    logger.warning(
                        'override_store.get_pin_queue failed; pin_queue degraded to empty list',
                        exc_info=True,
                    )

        # overrides — convert OverrideRow dataclasses to plain dicts.
        overrides: dict[str, dict] = {}
        if self._override_store:
            try:
                for tid, row in self._override_store.get_overrides(
                    self._project_root
                ).items():
                    overrides[tid] = {
                        'boost_tier': row.boost_tier,
                        'pinned': row.pinned,
                        'reserve_now': row.reserve_now,
                        'ttl_until': row.ttl_until.isoformat() if row.ttl_until else None,
                    }
            except Exception:
                if not self._override_store_warned:
                    self._override_store_warned = True
                    logger.warning(
                        'override_store.get_overrides failed; overrides degraded to empty dict',
                        exc_info=True,
                    )

        # current_holders — delegate to the public accessor.
        current_holders = self.lock_table.snapshot_holders()

        return {
            'skip_counts': skip_counts,
            'parks': parks,
            'effective_priorities': effective_priorities,
            'pin_queue': pin_queue,
            'overrides': overrides,
            'current_holders': current_holders,
            'lock_depth': self.config.lock_depth,
            'is_paused': self.is_paused,
            'pause_reason': self.pause_reason,
            'snapshot_at': datetime.now(UTC).isoformat(),
        }

    def _build_snapshot_payload(self, state: dict | None = None) -> str:
        """Serialise the scheduler business state to a stable JSON string for dedup.

        Excludes ``snapshot_at`` from the output because that field is derived
        from ``datetime.now()`` and changes on every call — including it would
        make two consecutive calls with identical business state byte-different,
        defeating the content-dedup check in ``_write_snapshot_best_effort``.

        The ``snapshot_at`` field is included in the separate disk payload
        built by ``_write_snapshot_best_effort`` (via ``json.dumps(state, ...)``)
        or by ``_write_state_snapshot_raw`` when called directly without a pre-built
        payload.

        Args:
            state: Pre-fetched snapshot dict from ``get_state_snapshot()``.
                When provided, avoids a redundant ``get_state_snapshot()``
                call (the caller already has the state and will build the disk
                payload from it).  When ``None``, calls ``get_state_snapshot()``
                internally.

        Returns a sorted-keys JSON string of the snapshot minus ``snapshot_at``.
        Sorted keys ensure deterministic output regardless of insertion order.
        Does not mutate the passed-in ``state`` dict.
        """
        if state is None:
            state = self.get_state_snapshot()
        # Exclude snapshot_at without mutating the caller's dict.
        dedup_state = {k: v for k, v in state.items() if k != 'snapshot_at'}
        return json.dumps(dedup_state, default=str, sort_keys=True)

    def _write_state_snapshot_raw(self, path: Path, payload: str | None = None) -> None:
        """Atomically write the current state snapshot to *path* as JSON.

        Creates parent directories if missing.  Uses a tmp-file + os.replace
        atomic rename so concurrent readers never see a partial write.

        Exceptions propagate to the caller (``_write_snapshot_best_effort``),
        which swallows them via its own try/except so the scheduler never stops
        ticking due to a disk issue.  Propagating rather than swallowing here
        ensures bookkeeping (``_last_snapshot_payload``,
        ``_last_snapshot_write_ts``) is only advanced when the write actually
        succeeds — a swallowed failure would silently record a stale snapshot
        as the last-written state, causing subsequent content-identical checks
        to skip a write that never actually persisted.

        Args:
            path: Destination path for the snapshot file.
            payload: Pre-serialised JSON string to write.  When provided
                (the normal path from ``_write_snapshot_best_effort``), avoids
                a redundant ``get_state_snapshot()`` + serialisation call.
                When ``None``, serialises the current state inline (used by
                direct callers such as tests).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix('.json.tmp')
        # Serialise the full snapshot (including snapshot_at) for the on-disk
        # record.  Note: this is independent of the dedup payload built in
        # _write_snapshot_best_effort; when a pre-built payload is passed,
        # no second get_state_snapshot() call is needed.
        if payload is None:
            payload = json.dumps(self.get_state_snapshot(), default=str)
        tmp_path.write_text(payload, encoding='utf-8')
        os.replace(tmp_path, path)

    async def _write_snapshot_best_effort(self, force: bool = False) -> None:
        """Write the scheduler state snapshot to the default path off the event loop.

        Derives the path from ``_project_root`` and offloads the JSON
        serialise + atomic-rename to a thread via ``asyncio.to_thread`` so
        the event loop is not blocked during disk I/O.

        At the expected tick rate (~1 tick/s per agent, bursting to ~20/s
        during pin-queue drains) and typical snapshot sizes (< 1 MB for
        1 500 tasks), each write costs a few ms of disk I/O that is now
        transparent to the event loop.

        **Throttle (time gate)**: writes are coalesced to at most one per
        ``config.snapshot_min_write_interval_secs`` (default 250 ms).  The
        first-ever write always proceeds.  Throttled ticks are O(1) *when
        ``self._snapshot_write_lock`` is uncontended*: ``asyncio.Lock.acquire``
        does not yield when the lock is free, so the fast path is a single
        monotonic subtraction with no JSON serialisation.  Under contention
        (e.g. a throttled tick arriving while a concurrent flush holds the
        lock), the tick must first acquire the lock and may briefly wait
        behind the in-flight write before the throttle check coalesces it;
        see the latency-tradeoff comment immediately above
        ``async with self._snapshot_write_lock:`` below.  Pass
        ``force=True`` to bypass the throttle and guarantee an immediate
        write (used by ``flush_state_snapshot``).

        **Content dedup**: after the time gate passes, ``get_state_snapshot()``
        is called once and the dedup payload (without ``snapshot_at``) is built
        via ``_build_snapshot_payload(state)``.  If byte-identical to
        ``_last_snapshot_payload``, the disk write is skipped but
        ``_last_snapshot_write_ts`` is still updated to prevent
        re-serialisation on every tick of an unchanged steady state.  The full
        disk payload (with ``snapshot_at``) is only serialised when a write is
        actually needed, avoiding redundant work on dedup hits.
        ``force=True`` always writes regardless of content equality.

        Bookkeeping (``_last_snapshot_payload``, ``_last_snapshot_write_ts``)
        is advanced only after a confirmed successful write: exceptions from
        ``_write_state_snapshot_raw`` propagate to the outer try/except, which logs
        a warning and returns without touching bookkeeping — preventing a
        failed write from being recorded as the last-written state.

        Swallows all exceptions so the scheduler never stops ticking due to
        disk or serialisation errors.
        """
        # Guard: defensive belt-and-suspenders against a _project_root that
        # bypassed pydantic validation.  Under normal operation this branch is
        # unreachable: config.py:875 types project_root as
        #   Path = Field(default=Path('.'))
        # with an after-validator at config.py:880-883 that calls .resolve(),
        # and validate_assignment=True at config.py:948 — so pydantic rejects
        # None on both construction and assignment.  The only path that produces
        # 'None'/empty in _project_root (set at scheduler.py:692 via
        # str(config.project_root)) is a value that bypassed pydantic
        # validation entirely, e.g. via object.__setattr__ as the task-1334
        # guard tests do.  Without this check, Path('None') / 'data' /
        # 'orchestrator' / 'scheduler_state.json' would create a directory
        # literally named ./None/ under the process CWD.  Refuse the write
        # instead.  This guard MUST run before any timestamp bookkeeping so a
        # no-project-root scheduler does not advance _last_snapshot_write_ts
        # for a write that never happens.
        if not self._project_root or self._project_root == 'None':
            if not self._snapshot_guard_warned:
                self._snapshot_guard_warned = True
                logger.warning(
                    'scheduler state snapshot skipped: project_root unset/invalid (%r)',
                    self._project_root,
                )
            return
        # Lock serialises the time-gate read → await → bookkeeping write section
        # so concurrent tick/flush callers never race on the stale timestamp or
        # on the shared .json.tmp path.  The project-root guard above stays
        # outside: it is O(1), touches no shared state, and acquiring the lock
        # for a guaranteed no-op would needlessly block no-project-root callers.
        # Latency tradeoff: a tick's snapshot attempt may briefly wait here
        # behind an in-flight flush's disk write; that serialisation is the
        # deliberate cost of preventing concurrent .json.tmp corruption.
        async with self._snapshot_write_lock:
            # Leading-edge time throttle.  Coalesces ticks within the configured
            # minimum interval at O(1) cost (monotonic subtraction only).
            # force=True bypasses the gate to guarantee a fresh write (e.g. flush
            # on quiescence/shutdown).
            now = self._time_source()
            if not force:
                interval = self.config.snapshot_min_write_interval_secs
                if (
                    self._last_snapshot_write_ts is not None
                    and (now - self._last_snapshot_write_ts) < interval
                ):
                    return  # throttled: within the coalesce window, no I/O
            # Time gate passed (or force=True): build the payload once.
            # Note: payload is built ONLY after the gate passes so throttled ticks
            # remain O(1) — they never reach this serialisation point.
            try:
                # Compute the state dict once; _build_snapshot_payload uses it for
                # the dedup comparison and disk_payload carries it to disk — no
                # second get_state_snapshot() call on the write path.
                state = self.get_state_snapshot()
                payload = self._build_snapshot_payload(state)
                # Content dedup: skip the disk write if the business state has not
                # changed since the last write.  Still advance the timestamp so the
                # next throttle window starts from now (prevents re-serialisation
                # every tick during an unchanged steady state).
                # force=True always writes, regardless of content equality.
                if not force and payload == self._last_snapshot_payload:
                    self._last_snapshot_write_ts = now
                    return
                # Build the disk payload (full state including snapshot_at) only
                # after the dedup check — avoids serialising when a dedup skip
                # applies.  Passed directly to _write_state_snapshot_raw to avoid a
                # second get_state_snapshot() call on the actual write path.
                disk_payload = json.dumps(state, default=str)
                path = (
                    Path(self._project_root) / 'data' / 'orchestrator' / 'scheduler_state.json'
                )
                await asyncio.to_thread(self._write_state_snapshot_raw, path, disk_payload)
                # Bookkeeping advanced only after a confirmed successful write.
                # _write_state_snapshot_raw propagates exceptions so these lines are
                # unreachable when the disk write fails — preventing a stale
                # snapshot from being recorded as last-written.
                self._last_snapshot_payload = payload
                self._last_snapshot_write_ts = now
            except Exception:
                logger.warning('_write_snapshot_best_effort failed', exc_info=True)

    async def flush_state_snapshot(self) -> None:
        """Force an immediate state snapshot write, bypassing the throttle.

        Guarantees that the on-disk snapshot reflects the most recent in-memory
        state regardless of how recently the last throttled write occurred.
        Intended for shutdown or quiescence paths where a stale-on-disk read
        would be incorrect.

        The project_root guard still applies: if ``_project_root`` is unset,
        this is a no-op (there is nowhere to write, even under force).
        """
        await self._write_snapshot_best_effort(force=True)

    async def _persist_files_metadata(self, task_id: str, needed: list[str]) -> bool:
        """Persist ``needed`` as ``metadata['files']`` via a Stage-2-preserving
        read-modify-write.

        Sibling keys attached by Stage-2 reconciliation (``memory_hints``,
        ``_causation_id``) survive because the merge policy is
        ``{**fresh_md, 'files': needed}`` — existing keys win except ``files``.

        ``needed`` is module-granularity (the output of ``files_to_modules``),
        not raw file paths.  Writing module paths to ``metadata.files`` is
        idempotent on restart because ``normalize_lock`` on an
        already-depth-normalized path is identity, so the derived lock set is
        unchanged after a reload.  This mirrors the failure-branch convention.

        Returns ``True`` when ``update_task`` reports success, ``False``
        otherwise.  Logs a warning on failure but never raises — callers that
        have already applied the in-memory narrowing should continue even when
        the durable write fails (reconcile/next-plan will retry).
        """
        fresh = await self.get_task(task_id)
        fresh_md = (fresh.get('metadata') or {}) if isinstance(fresh, dict) else {}
        merged = {**fresh_md, 'files': needed}
        updated = bool(await self.update_task(task_id, merged))
        if not updated:
            logger.warning(
                'Task %s: metadata.files persist failed (non-critical — '
                'in-memory state already applied; reconcile/next-plan will retry).',
                task_id,
            )
        return updated

    async def handle_blast_radius_expansion(
        self,
        task_id: str,
        current: list[str],
        needed: list[str],
    ) -> bool:
        """Handle plan refining blast radius (widening, narrowing, or shift).

        1. Try acquire any additional locks (needed − current)
        2. On success, release any stale locks (current − needed) so other
           tasks can acquire modules the refined plan no longer touches
        3. On acquire failure: update task with new modules, reset to pending,
           release current locks
        """
        depth = self.config.lock_depth
        current_set = {normalize_lock(m, depth) for m in current}
        needed_set = {normalize_lock(m, depth) for m in needed}
        additional = sorted(needed_set - current_set)
        stale = sorted(current_set - needed_set)
        if not additional and not stale:
            return True

        released: list[str] = []
        if not additional or self.lock_table.try_acquire_additional(task_id, additional):
            if stale:
                released = self.lock_table.release_subset(task_id, stale)
                if released and self.event_store:
                    self.event_store.emit(
                        EventType.lock_released,
                        task_id=task_id,
                        data={'modules': released, 'reason': 'plan_refinement'},
                    )
                # Persist the narrowed set so it survives a restart.  Without
                # this, the scheduler re-reads the over-declared metadata.files
                # on startup and re-acquires the released modules, re-introducing
                # the over-claim (δ bug).  Best-effort: in-memory narrowing already
                # applied via release_subset; return True even on update failure.
                # Shares the same read-modify-write logic as the requeue branch
                # (see _persist_files_metadata below).
                updated = await self._persist_files_metadata(task_id, needed)
                # Emit set_to_plan to signal the DURABLE persist (lock_released
                # above signals the in-memory release; this signals durability).
                # persisted=False lets the reify ζ gate distinguish a failed
                # write from a successful one without a separate metadata read.
                if self.event_store:
                    self.event_store.emit(
                        EventType.set_to_plan,
                        task_id=task_id,
                        data={
                            'files': needed,
                            'released': released,
                            'acquired': additional,
                            'persisted': bool(updated),
                        },
                    )
            logger.info(f'Task {task_id} expanded to modules: {needed}')
            return True

        # Can't acquire — reset task
        logger.warning(
            f'Task {task_id} needs modules {needed} but locks unavailable. Requeuing.'
        )
        # Cache expanded modules in memory so _get_modules uses them on retry
        self._module_cache[task_id] = sorted(needed_set)
        # Read-modify-write so memory_hints / _causation_id attached by Stage-2
        # reconciliation survive the blast-radius-failure files write (task 1511).
        # get_task already swallows MCP errors → None, so the isinstance guard
        # degrades gracefully to the prior {'files': needed} write (see
        # _persist_files_metadata).  This is intentionally backend-only — the
        # scheduler holds no per-task in-memory metadata to merge with (unlike
        # workflow._reconcile_metadata_files_for_done which uses _merge_fresh_metadata).
        # Merge policy: {**fresh_md, 'files': needed} — Stage-2 sibling keys survive.
        await self._persist_files_metadata(task_id, needed)
        try:
            await self.set_task_status(task_id, 'pending')
        except RuntimeError as e:
            # Transient retries exhausted — keep locks held so the worktree
            # stays reserved for this task; the next reconcile cycle (mid-run
            # sweep or startup) will revert the in-progress status when the
            # backend recovers.  Releasing locks here would let another task
            # claim the modules while this one is still nominally in-progress.
            logger.warning(
                'Task %s: set_task_status(pending) failed during '
                'blast-radius requeue (%s) — keeping locks held for '
                'reconcile to recover.', task_id, e,
            )
            return False
        self.lock_table.release(task_id)
        return False

    def release(self, task_id: str, *, requeued: bool = False) -> None:
        """Release all module locks for a task and clear dispatch guard."""
        self._dispatched.discard(task_id)
        self._dispatched_priority.pop(task_id, None)
        if requeued:
            self._requeue_until[task_id] = (
                self._time_source() + self.config.requeue_cooldown_secs
            )
        modules = list(self.lock_table._held.get(task_id, set()))
        self.lock_table.release(task_id)
        # Defensive: clear any reservations still owned by this task.
        self.lock_table.clear_parks_for(task_id)
        if self.event_store and modules:
            self.event_store.emit(
                EventType.lock_released,
                task_id=task_id,
                data={'modules': modules},
            )

    # --- Retry cap (per-task REQUEUED counter) ---

    def record_requeue(
        self,
        task_id: str,
        *,
        phase: str,
        reason: str,
        detail: str,
        run_id: str,
        cost_usd: float,
    ) -> int:
        """Append a requeue record and return the new *genuine* cumulative count.

        Transient API requeues (HTTP 5xx "agent API error" summaries, classified
        by ``is_transient_api_requeue``) are routed to ``_transient_requeue_counts``
        and do NOT increment the genuine ``_requeue_counts`` that feeds
        ``config.requeue_cap``.  Genuine requeues behave exactly as before.
        The record is appended to ``_requeue_history`` either way so the
        cap-exhaust report shows the full attempt timeline.
        ``RequeueRecord.attempt`` is the overall chronological index
        (``len(history) + 1``) across both buckets, so the timeline is
        monotonic when genuine and transient records interleave.
        Returns the genuine count (0 for a transient requeue).
        """
        history = self._requeue_history.setdefault(task_id, [])
        overall_attempt = len(history) + 1
        if is_transient_api_requeue(reason):
            t_count = self._transient_requeue_counts.get(task_id, 0) + 1
            self._transient_requeue_counts[task_id] = t_count
        else:
            g_count = self._requeue_counts.get(task_id, 0) + 1
            self._requeue_counts[task_id] = g_count
        history.append(
            RequeueRecord(
                attempt=overall_attempt,
                phase=phase,
                reason=reason,
                detail=detail,
                run_id=run_id,
                cost_usd=cost_usd,
                timestamp=time.time(),
            )
        )
        return self._requeue_counts.get(task_id, 0)

    def transient_requeue_count(self, task_id: str) -> int:
        """Return the number of transient API requeues recorded for *task_id*."""
        return self._transient_requeue_counts.get(task_id, 0)

    def clear_requeue_count(self, task_id: str) -> None:
        """Clear the requeue counters and history for *task_id*.

        Invoked on a DONE outcome (task recovered) and at the end of
        ``trigger_retry_cap_exhausted`` (human-resolution starts from zero).
        """
        self._requeue_counts.pop(task_id, None)
        self._transient_requeue_counts.pop(task_id, None)
        self._requeue_history.pop(task_id, None)

    async def trigger_retry_cap_exhausted(
        self,
        task_id: str,
        *,
        run_id: str,
        cost_usd: float,
        escalation_queue=None,
        reports_dir: Path | None = None,
        cap: int | None = None,
    ) -> Path | None:
        """Handle cap exhaustion: write report, set blocked, submit L1 escalation.

        Args:
            task_id: Task whose requeue counter hit the cap.
            run_id: Orchestrator run for the report header + SQL hint.
            cost_usd: Cost for the current run's attempts (harness passes
                either the cumulative run-level cost or the current-attempt
                cost from the RunStore).
            escalation_queue: The shared ``EscalationQueue``; when None, the
                escalation step is skipped (tests inject a stub or None).
            reports_dir: Where to write the markdown report.  Defaults to
                ``<project_root>/data/orchestrator/retry_cap_reports/``.
            cap: The cap that actually fired.  Defaults to
                ``config.requeue_cap`` (backward compatible).  Pass
                ``config.transient_requeue_cap`` when the transient ceiling
                fired so the report/event/escalation show the correct cap.

        Returns the report path written, or None when writing fails.
        """
        history = list(self._requeue_history.get(task_id, ()))
        cap = cap if cap is not None else self.config.requeue_cap
        n_attempts = len(history)
        last_reason = history[-1].reason if history else 'unknown'
        # Per-bucket breakdown for human-readable reports/escalations so the
        # triaging engineer sees which ceiling fired and how many of each kind
        # accumulated (avoids the misleading "10 iterations (cap=10)" when only
        # 8 were transient but total history also includes genuine ones).
        n_transient = sum(
            1 for r in history if is_transient_api_requeue(r.reason)
        )
        n_genuine = n_attempts - n_transient

        if reports_dir is None:
            reports_dir = (
                Path(self.config.project_root)
                / 'data' / 'orchestrator' / 'retry_cap_reports'
            )
        report_path: Path | None = None
        try:
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / f'{task_id}_{run_id}.md'
            report_path.write_text(
                _render_retry_cap_report(
                    task_id=task_id,
                    run_id=run_id,
                    cap=cap,
                    history=history,
                    cost_usd=cost_usd,
                )
            )
        except Exception:
            logger.exception(
                'Failed to write retry-cap report for task %s (run %s)',
                task_id, run_id,
            )
            report_path = None

        # Best-effort set blocked BEFORE escalating so a resume-on-L1 flow
        # (pending→unblock) doesn't race an acquire.
        try:
            await self.set_task_status(task_id, 'blocked')
        except Exception:
            logger.exception(
                'Failed to set task %s status to blocked on retry-cap exhaust',
                task_id,
            )

        if self.event_store:
            self.event_store.emit(
                EventType.retry_cap_exhausted,
                task_id=task_id,
                cost_usd=cost_usd,
                data={
                    'requeue_count': n_attempts,
                    'cap': cap,
                    'last_reason': last_reason[:200],
                    'report_path': str(report_path) if report_path else '',
                },
            )

        if escalation_queue is not None:
            try:
                from escalation.models import Escalation
                summary = (
                    f'Retry cap hit: {n_attempts} REQUEUED iterations '
                    f'({n_genuine} genuine/{n_transient} transient, cap={cap}); '
                    f'last reason: {last_reason[:120]}'
                )
                detail_lines = [
                    f'Task {task_id} exceeded requeue cap after {n_attempts} '
                    f'REQUEUED outcomes '
                    f'({n_genuine} genuine, {n_transient} transient; cap={cap}).',
                    f'Run: {run_id}',
                    f'Cost-to-date: ${cost_usd:.2f}',
                    f'Last reason: {last_reason}',
                ]
                if report_path is not None:
                    detail_lines.append(f'See report: {report_path}')
                esc = Escalation(
                    id=escalation_queue.make_id(task_id),
                    task_id=task_id,
                    agent_role='orchestrator',
                    severity='blocking',
                    category='retry_cap_exhausted',
                    summary=summary[:200],
                    detail='\n'.join(detail_lines),
                    suggested_action='investigate_and_retry',
                    level=1,
                )
                escalation_queue.submit(esc)
            except Exception:
                logger.exception(
                    'Failed to submit L1 escalation for task %s retry-cap exhaust',
                    task_id,
                )

        self.clear_requeue_count(task_id)
        return report_path

    def _get_modules(self, task: dict) -> list[str]:
        """Extract module list from task metadata, normalized for locking.

        Priority: in-memory cache > metadata.files > fallback ``task-<id>``.
        """
        task_id = str(task.get('id', ''))
        depth = self.config.lock_depth
        # Check in-memory cache first (survives metadata update failures)
        if task_id in self._module_cache:
            return self._module_cache[task_id]
        metadata = task.get('metadata') or {}
        if isinstance(metadata, dict):
            files = metadata.get('files', [])
            if isinstance(files, list) and files:
                derived = files_to_modules(files, depth)
                if derived:
                    return derived
        # Fallback: use a generic module name based on task id
        if task_id not in self._fallback_warned:
            logger.warning(
                'Task %s: no module metadata found — using fallback lock task-%s',
                task_id,
                task_id or 'unknown',
            )
            self._fallback_warned.add(task_id)
        return [f'task-{task_id or "unknown"}']
