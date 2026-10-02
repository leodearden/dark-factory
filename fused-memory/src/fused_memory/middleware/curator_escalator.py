"""Route curator LLM failures to the orchestrator's escalation queue.

The :class:`TaskCurator` used to silently degrade to ``action='create'`` on
any LLM error, which meant a broken curator shipped for five days without
anyone noticing (see plans/floating-snuggling-pebble.md, R2). The curator
now raises :class:`CuratorFailureError` on LLM failure; this module decides
what happens next.

Routing policy (keyed off orchestrator liveness):

* **Orchestrator is running** for the target project — submit a level-1
  escalation to the project's queue and return; the escalation watcher
  runs ``/unblock`` against it. We degrade to ``action='create'`` so the
  current ``add_task`` call still succeeds.

* **No orchestrator** (typical interactive MCP usage) — re-raise the
  failure so the MCP caller sees a loud error instead of a silent
  curator outage.

* **Post-ZOT duplicate finding** (``report_zot_duplicate``) — one level-1
  ``curator_zot_duplicate`` record per (new task, near-duplicate) pair,
  cross-referencing the live zero-output-hang record. Filed only when an
  orchestrator is running; otherwise logged, never raised.

Liveness is probed via ``flock(LOCK_SH | LOCK_NB)`` on
``{project_root}/data/orchestrator/orchestrator.lock`` (the orchestrator
holds ``LOCK_EX`` on startup). Treat a missing file as "no orchestrator".

Per-project burst policy: escalate the first 3 failures within a rolling
1 h window, then suppress further escalations for the rest of the window.
Single-pin suppression previously hid a sustained outage behind a stale
L1 — surfacing the third failure with an explicit "further suppressed"
note makes the burst visible to operators without flooding the queue.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.middleware.curator_zot_duplicate_sweep import (
    DUPLICATE_METADATA_KEY,
    DuplicateFinding,
)
from fused_memory.middleware.task_curator import CuratorFailureError

if TYPE_CHECKING:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# ``escalation`` is a sibling workspace package. The main reconciliation
# harness also imports it defensively (harness.py:38-46) because historical
# deployments could lack the package. Mirror that pattern here so the
# curator still functions (without escalation routing) in minimal envs.
try:
    from escalation.dedupe import (  # type: ignore[import-untyped]
        DedupeConfig,
        compute_content_fingerprint,
        content_fingerprint_key,
        submit_or_dedupe,
    )
    from escalation.models import Escalation  # type: ignore[import-untyped,no-redef]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover - exercised only in minimal envs
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)


_DEFAULT_COOLDOWN_SECS = 3600.0
_LOCK_FILENAME = 'data/orchestrator/orchestrator.lock'
_QUEUE_DIRNAME = 'data/escalations'

# Short in-process dedup window for zero-output-timeout reports. A batch of N
# candidates that all hit ZOT bisects to N concurrent size-1 curate() calls,
# each of which calls report_failure independently. This window bounds ONE such
# outage event to ONE recurrence count on the folded record below; reports
# within it after the first are logged and dropped.
_ZOT_DEDUP_WINDOW_SECS = 60.0

# The class Leo ruled on in esc-task-curator-17 (2026-08-27, ACCEPT-AND-RETUNE):
# an accepted, recurring, transient pre-turn empty-output hang on the curator
# call shape. Its dedupe fingerprint is derived from this pinned key, so every
# recurrence folds into one pending record instead of minting a new L1.
_ZOT_CATEGORY = 'curator_zero_output_hang'
_ZOT_ROOT_CAUSE = 'curator-empty-output-pre-turn-hang'
_ZOT_DUPLICATE_CATEGORY = 'curator_zot_duplicate'


def _transcript_evidence_lines(
    transcript_turns: int | None, tools_used: tuple[str, ...] | None,
) -> list[str]:
    """Detail lines for what the failed run's transcript recorded.

    An absent measurement is spelled out, never rendered as zero:
    ``transcript_turns=0`` means a transcript-confirmed pre-turn stall.
    """
    turns = 'unknown (transcript unreadable)' if transcript_turns is None else str(transcript_turns)
    tools = 'unknown' if tools_used is None else (','.join(tools_used) or '(none)')
    return [f'transcript_turns={turns}', f'tools_used={tools}']


def _curator_escalation(
    queue: EscalationQueue, *, category: str, summary: str, detail: str, **extra: Any,
) -> Escalation:
    """Build a level-1 record in the fixed envelope every curator escalation shares."""
    return Escalation(
        id=queue.make_id('curator'),
        task_id='task-curator',
        agent_role='fused-memory/task-curator',
        severity='blocking',
        category=category,
        summary=summary,
        detail=detail,
        level=1,
        **extra,
    )


def _submit_logged(
    queue: EscalationQueue, escalation: Escalation, *, kind: str, project_id: str,
) -> bool:
    """Submit ``escalation``; on queue I/O failure log it and return ``False``.

    Never raises: the caller's task write must not fail because the queue broke.
    """
    try:
        queue.submit(escalation)
    except Exception:
        logger.exception(
            'curator_escalator: failed to submit %s escalation for project %s',
            kind, project_id,
        )
        return False
    return True


class CuratorEscalator:
    """Route :class:`CuratorFailureError` to the orchestrator or back to the caller."""

    # Queue the first N failures in a burst window; suppress the rest.
    _ESCALATE_FIRST_N = 3

    def __init__(
        self,
        cooldown_secs: float = _DEFAULT_COOLDOWN_SECS,
        state_path: str | Path | None = None,
    ) -> None:
        self._cooldown_secs = cooldown_secs
        self._state_path: Path | None = Path(state_path) if state_path is not None else None
        # (project_id, subtype) → wall-clock timestamps (time.time()) of recent
        # failures in the window.  Keyed by the composite so different subtypes
        # for the same project are counted independently (e.g. error_max_budget_usd
        # and error_max_turns don't share quota).  Wall-clock is used so timestamps
        # survive a process restart when persisted; monotonic resets on restart
        # making reloaded values meaningless.
        # Pruned on every report_failure call.
        self._failure_log: dict[tuple[str, str | None], list[float]] = {}
        self._queues: dict[str, EscalationQueue] = {}
        # project_id → (monotonic timestamp, live ZOT escalation id or None) of
        # the last ZOT submit attempt. Prevents a batch of N concurrent curate()
        # ZOT calls from flooding the escalation queue with N identical entries
        # for a single outage event, and hands every sibling the same id.
        # Stays monotonic — in-process dedup, intentionally reset on restart.
        self._zot_last_submitted: dict[str, tuple[float, str | None]] = {}
        # Serialise concurrent _persist_state calls so only one write is ever
        # in flight.  Mirrors the asyncio.Lock instances held by TaskInterceptor
        # (task_interceptor.py:288-289,1308,1332) for per-project mutations.
        # Constructed without a running loop (valid on py>=3.10).
        self._persist_lock: asyncio.Lock = asyncio.Lock()
        # Reload persisted burst log if state_path is set and exists.
        if self._state_path is not None and self._state_path.exists():
            self._load_state()

    # ------------------------------------------------------------------
    # State persistence helpers (burst log, restart-durable)
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load _failure_log from state_path JSON on construction.

        Format: a JSON array of records::

            [{"project_id": str, "subtype": str|null, "timestamps": [float, ...]}, ...]

        Entries older than cooldown_secs are pruned on load.  A missing,
        empty, or corrupt file is tolerated (logged at WARNING; starts empty)
        because the curator has a best-effort contract — a broken state file
        must never prevent ``add_task`` from succeeding.
        """
        assert self._state_path is not None
        now = time.time()
        cutoff = now - self._cooldown_secs
        try:
            text = self._state_path.read_text()
            if not text.strip():
                return
            records = json.loads(text)
            dropped = 0
            for rec in records:
                try:
                    project_id = rec['project_id']
                    subtype = rec.get('subtype')  # may be None
                    timestamps = [t for t in rec['timestamps'] if t >= cutoff]
                    if timestamps:
                        self._failure_log[(project_id, subtype)] = timestamps
                except (KeyError, TypeError, ValueError):
                    dropped += 1
                    continue
            if dropped:
                logger.warning(
                    'curator_escalator: skipped %d malformed record(s) while '
                    'loading state from %s; remaining records loaded normally',
                    dropped,
                    self._state_path,
                )
        except Exception:
            logger.warning(
                'curator_escalator: failed to load state from %s; starting empty',
                self._state_path,
                exc_info=True,
            )

    async def _persist_state(self) -> None:
        """Atomically write _failure_log to state_path as JSON.

        Prunes all globally stale entries (timestamps older than the cooldown
        window) before writing so the file stays bounded to the active window —
        not just the key that was most recently touched.  The write is offloaded
        via :func:`asyncio.to_thread` to avoid blocking the event loop under
        burst load.

        Concurrent ``report_failure`` calls (e.g. from a batch ZOT bisect that
        spawns N concurrent size-1 curate() calls) are serialised by
        ``_persist_lock``: only one persist is ever in flight at a time.  This
        means the later-acquiring coroutine takes a fresh snapshot of the full
        current ``_failure_log`` (containing all keys updated so far) — a stale
        snapshot can never overwrite a fresher one, and a single writer always
        touches the temp file at a time.

        Each ``_write`` closure gets a unique temp filename
        ``<state>.{pid}.{id(payload)}.tmp`` (mirroring event_queue.py's
        disjoint-path discipline) so two writers can never share a temp path,
        with ``finally`` cleanup to ensure a failed/garbage temp never lingers.

        Uses write-to-temp + os.replace so a crash mid-write never corrupts
        the existing file.  Only called when state_path is set.
        """
        if self._state_path is None:
            return
        async with self._persist_lock:
            # Prune globally before serialising so stale (project_id, subtype)
            # pairs do not accumulate in memory or on disk across long-lived
            # processes.  Snapshot is taken inside the lock so the later-
            # acquiring coroutine always sees the complete current log.
            now = time.time()
            cutoff = now - self._cooldown_secs
            for key in list(self._failure_log.keys()):
                pruned = [t for t in self._failure_log[key] if t >= cutoff]
                if pruned:
                    self._failure_log[key] = pruned
                else:
                    del self._failure_log[key]
            records = [
                {
                    'project_id': project_id,
                    'subtype': subtype,
                    'timestamps': timestamps,
                }
                for (project_id, subtype), timestamps in self._failure_log.items()
            ]
            # Serialise the snapshot on the calling thread; only the I/O runs
            # in the thread pool so we don't hold the lock during the blocking
            # write.
            payload = json.dumps(records)
            state_path = self._state_path

            def _write() -> None:
                # Unique temp path per write: pid + id(payload) ensures two
                # concurrent writers (if they somehow bypass the lock) never
                # share a temp file name.  Defense-in-depth: mirrors
                # event_queue.py's disjoint-indexed-path discipline.
                tmp_path = state_path.parent / (
                    f'{state_path.name}.{os.getpid()}.{id(payload)}.tmp'
                )
                try:
                    tmp_path.write_text(payload)
                    os.replace(tmp_path, state_path)
                except Exception:
                    logger.warning(
                        'curator_escalator: failed to persist state to %s',
                        state_path,
                        exc_info=True,
                    )
                finally:
                    tmp_path.unlink(missing_ok=True)

            await asyncio.to_thread(_write)

    def _orchestrator_running(self, project_root: str) -> bool:
        """Return True if the project's orchestrator holds its exclusive lock.

        We probe with a *shared* non-blocking lock so we don't perturb the
        orchestrator's lock state — a successful acquisition means nobody
        holds LOCK_EX, so no orchestrator is running; a block/EAGAIN means
        it is.
        """
        lock_path = Path(project_root) / _LOCK_FILENAME
        if not lock_path.exists():
            return False
        try:
            fd = lock_path.open('rb')
        except OSError:
            return False
        try:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            except OSError as exc:
                # EAGAIN / EWOULDBLOCK on some platforms map to OSError
                import errno
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    return True
                raise
            # Lock acquired → orchestrator is not running. Release promptly.
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            return False
        finally:
            fd.close()

    def _queue_for(self, project_root: str) -> EscalationQueue:
        q = self._queues.get(project_root)
        if q is None:
            q = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
            self._queues[project_root] = q
        return q

    async def report_failure(
        self,
        *,
        project_root: str,
        project_id: str,
        justification: str,
        candidate_title: str,
        timed_out: bool | None = None,
        duration_ms: int | None = None,
        schema_tool_denied: bool = False,
        zero_output_timeout: bool = False,
        account_name: str | None = None,
        proc_tree: str | None = None,
        subtype: str | None = None,
        cost_usd: float | None = None,
        pool_sizes: dict[str, int] | None = None,
        transcript_turns: int | None = None,
        tools_used: tuple[str, ...] | None = None,
    ) -> str | None:
        """Route a curator failure. Raises :class:`CuratorFailureError` when no
        orchestrator is running so the MCP caller sees a loud error; the raised
        error keeps ``zero_output_timeout`` so a caller that degrades to create
        can still mark the create as ZOT-degraded.

        Returns the live ZOT escalation id for a ``zero_output_timeout`` report
        (the pending record a human will find, i.e. the parent when the report
        folded), and ``None`` on every other branch or when no id is known.

        When an orchestrator *is* running for this project, submit a level-1
        escalation for each of the first :attr:`_ESCALATE_FIRST_N` failures
        in a rolling cooldown window. The third escalation carries an
        explicit "further suppressed" note with the absolute window-end
        timestamp so operators can see the burst is ongoing. Subsequent
        failures within the window log a WARNING and return — preventing
        queue spam while keeping diagnostics visible.

        ``schema_tool_denied`` overrides ALL of that: it signals the CLI-2.1.168
        regression (the synthetic ``StructuredOutput`` schema tool was
        permission-denied), which is a *systemic config break* — not a flaky
        candidate. Such failures take a distinct branch that ALWAYS submits a
        self-describing escalation (bypassing burst suppression, separate from
        the ordinary cooldown log) so the break is un-missable and immediately
        diagnosable. Burst suppression is exactly what made the original outage
        read as sporadic "1 of 3" blips.
        """
        if not HAS_ESCALATION:
            # No escalation package available — fall back to a loud raise so
            # operators don't miss a silent curator outage.
            raise CuratorFailureError(
                f'TaskCurator LLM failed and escalation package is unavailable. '
                f'No dedupe was applied for project {project_id!r}. '
                f'justification={justification!r} candidate_title={candidate_title!r}',
                zero_output_timeout=zero_output_timeout,
            )

        if not self._orchestrator_running(project_root):
            raise CuratorFailureError(
                f'TaskCurator LLM failed and no orchestrator is running for '
                f'project {project_id!r}. No dedupe was applied. '
                f'justification={justification!r} candidate_title={candidate_title!r}',
                zero_output_timeout=zero_output_timeout,
            )

        if schema_tool_denied:
            # Systemic break (CLI tool-exclusion semantics changed): always
            # surface, never suppress, with a distinct summary + concrete fix
            # location. A human/code fix is required (in build_claude_argv), so
            # this should reach attention rather than be auto-watcher-resolved.
            await self._submit_schema_tool_denied(
                project_root=project_root,
                project_id=project_id,
                justification=justification,
                candidate_title=candidate_title,
                timed_out=timed_out,
                duration_ms=duration_ms,
            )
            return None

        if zero_output_timeout:
            # The accepted recurring class ruled on in esc-task-curator-17.
            # Two hangs hours apart each read as "failure 1 of 3" under the
            # normal burst window, so this bypasses burst suppression (and
            # leaves _failure_log alone); recurrences fold into one record.
            return await self._submit_zero_output_timeout(
                project_root=project_root,
                project_id=project_id,
                justification=justification,
                candidate_title=candidate_title,
                timed_out=timed_out,
                duration_ms=duration_ms,
                account_name=account_name,
                proc_tree=proc_tree,
                transcript_turns=transcript_turns,
                tools_used=tools_used,
            )

        now = time.time()  # wall-clock: stable across restarts (unlike monotonic)
        cutoff = now - self._cooldown_secs
        log_key = (project_id, subtype)
        log = [t for t in self._failure_log.get(log_key, []) if t >= cutoff]
        log.append(now)
        self._failure_log[log_key] = log
        # Persist the updated burst log so the counter survives a watchdog restart.
        await self._persist_state()
        count = len(log)
        burst_started = log[0]

        if count > self._ESCALATE_FIRST_N:
            # Window still has >=3 prior failures within cooldown; suppress.
            logger.warning(
                'curator_escalator: suppressing escalation for project %s '
                '(failure %d in window; cooldown %.0fs remaining since '
                'burst start); failure=%s',
                project_id,
                count,
                self._cooldown_secs - (now - burst_started),
                justification[:200],
            )
            return None

        # ``failures_in_window`` is always present so operator triage can
        # see "N of 3" at a glance without reading logs.
        detail_lines = [
            f'candidate_title={candidate_title!r}',
            f'project_id={project_id!r}',
            f'failures_in_window={count} of {self._ESCALATE_FIRST_N}',
        ]
        if timed_out is not None:
            detail_lines.append(f'timed_out={timed_out}')
        if duration_ms is not None:
            detail_lines.append(f'duration_ms={duration_ms}')
        if cost_usd is not None:
            detail_lines.append(f'cost_usd={cost_usd}')
        if pool_sizes is not None:
            detail_lines.append(f'pool_sizes={pool_sizes}')
        detail_lines.extend(_transcript_evidence_lines(transcript_turns, tools_used))
        detail_lines.append(f'justification={justification}')

        if count == self._ESCALATE_FIRST_N:
            # Absolute resume time via wall-clock — monotonic cannot convert
            # directly. Window closes cooldown_secs after burst's first entry.
            resume_at = datetime.now(UTC).timestamp() + (
                self._cooldown_secs - (now - burst_started)
            )
            resume_iso = datetime.fromtimestamp(resume_at, tz=UTC).isoformat()
            detail_lines.append('')
            detail_lines.append(
                f'NOTE: this is the {self._ESCALATE_FIRST_N}rd curator failure '
                f'for this project within the last hour. Further curator '
                f'failures will be suppressed from escalation for 1h '
                f'(until {resume_iso}). Investigate immediately — dedupe is '
                f'intermittently disabled. See logs for `task_curator: '
                f'decision=create cost_usd=0.0000` entries.',
            )
        detail = '\n'.join(detail_lines)

        queue = self._queue_for(project_root)
        escalation = _curator_escalation(
            queue,
            category='curator_failure',
            summary=(
                'TaskCurator LLM failing; dedupe bypassed for this ticket '
                '(and for further filings while the outage persists).'
            ),
            detail=detail,
        )
        if not _submit_logged(queue, escalation, kind='curator-failure', project_id=project_id):
            return None

        logger.warning(
            'curator_escalator: queued L1 escalation %s for project %s '
            '(failure %d of %d in window)',
            escalation.id, project_id, count, self._ESCALATE_FIRST_N,
        )
        return None

    async def report_zot_duplicate(
        self,
        *,
        project_root: str,
        project_id: str,
        finding: DuplicateFinding,
        candidate_title: str,
        zot_escalation_id: str | None,
        stamped: bool,
    ) -> None:
        """File one L1 record for a near-duplicate created under a ZOT degrade.

        ``stamped`` says whether the finding landed on the new task's metadata;
        the record must not point a human at metadata that was never written.

        Unlike ``report_failure`` this never raises on the no-orchestrator or
        no-escalation-package gates: it runs after the task already exists, so
        raising could only turn a missed notice into noise on a successful write.
        """
        if not HAS_ESCALATION or not await asyncio.to_thread(
            self._orchestrator_running, project_root,
        ):
            logger.warning(
                'curator_escalator: no orchestrator/escalation queue for project %s; '
                'post-ZOT duplicate %s ~ %s (score %.3f) not escalated',
                project_id, finding.task_id, finding.duplicate_task_id, finding.score,
            )
            return

        zot_ref = (
            repr(zot_escalation_id) if zot_escalation_id is not None
            else 'none known (breaker-open short-circuit, or the ZOT submit failed)'
        )
        stamp_note = (
            f'The new task carries metadata.{DUPLICATE_METADATA_KEY}.' if stamped
            else f'Stamping metadata.{DUPLICATE_METADATA_KEY} on the new task FAILED, '
            'so this record is the only trace of the finding.'
        )
        detail_lines = [
            f'task_id={finding.task_id!r}',
            f'duplicate_task_id={finding.duplicate_task_id!r}',
            f'duplicate_title={finding.duplicate_title!r}',
            f'score={finding.score:.3f}',
            f'candidate_title={candidate_title!r}',
            f'project_id={project_id!r}',
            f'zot_escalation_id={zot_ref}',
            '',
            'NOTE: curator dedupe was degraded to create by a zero-output hang. '
            'The post-ZOT duplicate sweep flagged this pair and did NOT combine, '
            f'cancel or delete anything. {stamp_note} A human should decide '
            'whether to cancel one of the two as SUPERSEDED.',
        ]

        queue = self._queue_for(project_root)
        escalation = _curator_escalation(
            queue,
            category=_ZOT_DUPLICATE_CATEGORY,
            summary=(
                f'possible duplicate created while curator dedupe was degraded by a '
                f'zero-output hang: task {finding.task_id} ~ task '
                f'{finding.duplicate_task_id} (score {finding.score:.3f})'
            ),
            detail='\n'.join(detail_lines),
        )
        if not _submit_logged(queue, escalation, kind='post-ZOT duplicate', project_id=project_id):
            return

        logger.warning(
            'curator_escalator: queued post-ZOT duplicate L1 escalation %s for '
            'project %s — task %s ~ task %s (score %.3f)',
            escalation.id, project_id, finding.task_id,
            finding.duplicate_task_id, finding.score,
        )

    async def _submit_schema_tool_denied(
        self,
        *,
        project_root: str,
        project_id: str,
        justification: str,
        candidate_title: str,
        timed_out: bool | None,
        duration_ms: int | None,
    ) -> None:
        """Submit a distinct, un-suppressed escalation for the CLI-2.1.168
        schema-tool-denied break.

        Deliberately bypasses the rolling-window burst suppression (and does not
        touch ``_failure_log``): a systemic tool-scoping break must surface on every
        occurrence. The summary is unmistakable vs the generic "curator LLM
        failing" escalation, and the detail names the concrete fix location so
        whoever picks it up can act without re-diagnosing.
        """
        detail_lines = [
            f'candidate_title={candidate_title!r}',
            f'project_id={project_id!r}',
        ]
        if timed_out is not None:
            detail_lines.append(f'timed_out={timed_out}')
        if duration_ms is not None:
            detail_lines.append(f'duration_ms={duration_ms}')
        detail_lines.append(f'justification={justification}')
        detail_lines.append('')
        detail_lines.append(
            "FIX: the CLI tool-exclusion semantics changed again — the '*' -> "
            "--tools '' substitution in "
            'shared/src/shared/cli_invoke.py::build_claude_argv no longer leaves '
            'the synthetic StructuredOutput schema tool in the registry, so every '
            'structured-output curator/recon call is permission-denied. Fix that '
            'substitution so StructuredOutput is NOT blocked (the live check is '
            'shared/tests/test_wildcard_deny_live_inventory.py, -m integration), '
            'then restart fused-memory.service. Task dedupe is DISABLED for this '
            'project until it is fixed.',
        )
        detail = '\n'.join(detail_lines)

        queue = self._queue_for(project_root)
        escalation = _curator_escalation(
            queue,
            category='curator_schema_tool_denied',
            summary=(
                'CRITICAL: schema StructuredOutput tool DENIED — CLI '
                "tool-exclusion semantics changed; cli_invoke's --tools '' "
                'substitution no longer permits the schema tool. Dedupe disabled '
                'until it is fixed.'
            ),
            detail=detail,
        )
        if not _submit_logged(
            queue, escalation, kind='schema-tool-denied', project_id=project_id,
        ):
            return

        logger.error(
            'curator_escalator: queued schema-tool-denied L1 escalation %s for '
            "project %s — StructuredOutput tool blocked despite cli_invoke's --tools ''; "
            'dedupe disabled until fixed',
            escalation.id, project_id,
        )

    async def _submit_zero_output_timeout(
        self,
        *,
        project_root: str,
        project_id: str,
        justification: str,
        candidate_title: str,
        timed_out: bool | None,
        duration_ms: int | None,
        account_name: str | None,
        proc_tree: str | None,
        transcript_turns: int | None,
        tools_used: tuple[str, ...] | None,
    ) -> str | None:
        """File a zero-output/full-timeout curator hang, folding recurrences.

        Returns the live record id from ``submit_or_dedupe`` (the parent's id
        when folded), the remembered id inside the in-process window, or
        ``None`` when the submit failed.

        This is the accepted recurring class ruled on in esc-task-curator-17
        (ACCEPT-AND-RETUNE). The escalation carries the pinned ``_ZOT_ROOT_CAUSE``
        and a fingerprint derived from it and the project, and is filed through
        ``submit_or_dedupe`` with an unbounded window: a recurrence folds into
        the pending record, whose ``dedupe_count`` is the recurrence counter.
        Once that record is resolved it leaves the pending set, so the next
        recurrence is a new incident.

        Bypasses the rolling-window burst suppression (and does not touch
        ``_failure_log``). The detail carries forensic evidence (account_name,
        proc_tree, duration_ms, transcript turns and tools).

        The in-process ``_ZOT_DEDUP_WINDOW_SECS`` window makes one batch bisect
        (N concurrent size-1 curate() calls) count as one recurrence, not N.
        """
        now_mono = time.monotonic()
        last = self._zot_last_submitted.get(project_id)
        if last is not None and (now_mono - last[0]) < _ZOT_DEDUP_WINDOW_SECS:
            logger.info(
                'curator_escalator: deduplicating ZOT escalation for project %s '
                '(last submitted %.1fs ago < dedup window %.0fs); '
                'candidate_title=%r',
                project_id,
                now_mono - last[0],
                _ZOT_DEDUP_WINDOW_SECS,
                candidate_title,
            )
            return last[1]

        detail_lines = [
            f'candidate_title={candidate_title!r}',
            f'project_id={project_id!r}',
        ]
        if timed_out is not None:
            detail_lines.append(f'timed_out={timed_out}')
        if duration_ms is not None:
            detail_lines.append(f'duration_ms={duration_ms}')
        if account_name is not None:
            detail_lines.append(f'account_name={account_name!r}')
        detail_lines.extend(_transcript_evidence_lines(transcript_turns, tools_used))
        if proc_tree:
            # Truncate to avoid overwhelming the escalation body.
            snippet = proc_tree[:1500]
            detail_lines.append(f'proc_tree=\n{snippet}')
        detail_lines.append(f'justification={justification}')
        detail_lines.append('')
        detail_lines.append(
            'NOTE: ruled ACCEPT-AND-RETUNE on esc-task-curator-17 (2026-08-27): an '
            'accepted, recurring, transient pre-turn empty-output hang on the '
            f"curator's sonnet + json-schema call shape, pinned as root_cause "
            f'{_ZOT_ROOT_CAUSE!r}. Recurrences fold into this record, so its '
            'dedupe_count is the recurrence counter (one burst of concurrent hangs '
            'counts once). transcript_turns=0 is a transcript-confirmed pre-turn '
            'stall; unknown means the transcript was not read. Dedupe degraded to '
            'create for this candidate; the circuit breaker short-circuits further '
            'curator LLM calls if hangs come back to back.',
        )
        detail = '\n'.join(detail_lines)

        queue = self._queue_for(project_root)
        escalation = _curator_escalation(
            queue,
            category=_ZOT_CATEGORY,
            summary=(
                'curator zero-output/full-timeout hang (accepted recurring class, '
                'esc-task-curator-17) — dedupe degraded to create; recurrences fold '
                'here and dedupe_count counts them.'
            ),
            detail=detail,
            root_cause=_ZOT_ROOT_CAUSE,
            dedupe_fingerprint=compute_content_fingerprint(  # type: ignore[possibly-unbound]
                _ZOT_CATEGORY, _ZOT_ROOT_CAUSE, affected_ids=[f'project:{project_id}'],
            ),
        )
        config = DedupeConfig(  # type: ignore[possibly-unbound]
            infra_dedupe_enabled=True,
            # UNBOUNDED, on the dead_letter_escalator precedent: these recur
            # days and weeks apart, and a resolved parent leaves the pending
            # set, so a recurrence after resolution still mints a new record.
            infra_dedupe_window_secs=float('inf'),
            infra_dedupe_categories=(_ZOT_CATEGORY,),
            key_fn=content_fingerprint_key,  # type: ignore[possibly-unbound]
        )
        try:
            response = submit_or_dedupe(queue, escalation, config)  # type: ignore[possibly-unbound]
        except Exception:
            logger.exception(
                'curator_escalator: failed to submit zero-output-timeout '
                'escalation for project %s',
                project_id,
            )
            # Do not re-raise — falling through to action='create' is safer than
            # failing add_task just because queue I/O broke.
            self._zot_last_submitted[project_id] = (now_mono, None)
            return None

        escalation_id: str = response['id']
        self._zot_last_submitted[project_id] = (now_mono, escalation_id)
        if response['status'] == 'dedup_skipped':
            logger.warning(
                'curator_escalator: zero-output-timeout recurrence for project %s '
                'folded into %s — account=%s duration_ms=%s; dedupe degraded to create',
                project_id, escalation_id, account_name, duration_ms,
            )
            return escalation_id
        logger.error(
            'curator_escalator: queued zero-output-timeout L1 escalation %s for '
            'project %s — account=%s duration_ms=%s; dedupe degraded to create',
            escalation_id, project_id, account_name, duration_ms,
        )
        return escalation_id
