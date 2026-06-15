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

import fcntl
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fused_memory.middleware.task_curator import CuratorFailureError

if TYPE_CHECKING:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# ``escalation`` is a sibling workspace package. The main reconciliation
# harness also imports it defensively (harness.py:38-46) because historical
# deployments could lack the package. Mirror that pattern here so the
# curator still functions (without escalation routing) in minimal envs.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped,no-redef]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover - exercised only in minimal envs
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)


_DEFAULT_COOLDOWN_SECS = 3600.0
_LOCK_FILENAME = 'data/orchestrator/orchestrator.lock'
_QUEUE_DIRNAME = 'data/escalations'

# Short dedup window for zero-output-timeout escalations. A batch of N
# candidates that all hit ZOT bisects to N concurrent size-1 curate() calls,
# each of which calls report_failure independently. Without dedup, a single
# outage event on a batch of N can enqueue up to N un-suppressed escalations.
# Submissions within this window after the first are logged and dropped.
_ZOT_DEDUP_WINDOW_SECS = 60.0


class CuratorEscalator:
    """Route :class:`CuratorFailureError` to the orchestrator or back to the caller."""

    # Queue the first N failures in a burst window; suppress the rest.
    _ESCALATE_FIRST_N = 3

    def __init__(self, cooldown_secs: float = _DEFAULT_COOLDOWN_SECS) -> None:
        self._cooldown_secs = cooldown_secs
        # project_id → monotonic timestamps of recent failures in the window.
        # Pruned on every ``report_failure`` call; a server restart resets
        # the burst, which is desired (operator gets fresh visibility).
        self._failure_log: dict[str, list[float]] = {}
        self._queues: dict[str, EscalationQueue] = {}
        # project_id → monotonic timestamp of the last submitted ZOT escalation.
        # Prevents a batch of N concurrent curate() ZOT calls from flooding the
        # escalation queue with N identical entries for a single outage event.
        self._zot_last_submitted: dict[str, float] = {}

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
    ) -> None:
        """Route a curator failure. Raises :class:`CuratorFailureError` when no
        orchestrator is running so the MCP caller sees a loud error.

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
            )

        if not self._orchestrator_running(project_root):
            raise CuratorFailureError(
                f'TaskCurator LLM failed and no orchestrator is running for '
                f'project {project_id!r}. No dedupe was applied. '
                f'justification={justification!r} candidate_title={candidate_title!r}',
            )

        if schema_tool_denied:
            # Systemic break (CLI tool-exclusion semantics changed): always
            # surface, never suppress, with a distinct summary + concrete fix
            # location. A human/code fix is required (update the deny-list), so
            # this should reach attention rather than be auto-watcher-resolved.
            await self._submit_schema_tool_denied(
                project_root=project_root,
                project_id=project_id,
                justification=justification,
                candidate_title=candidate_title,
                timed_out=timed_out,
                duration_ms=duration_ms,
            )
            return

        if zero_output_timeout:
            # Transient Anthropic-backend INFRA hang on the curator's
            # sonnet+json-schema call shape (task 1550). Two hangs hours apart
            # each read as "failure 1 of 3" under the normal burst window —
            # the outage was invisible. Always surface, bypassing burst
            # suppression (don't touch _failure_log).
            await self._submit_zero_output_timeout(
                project_root=project_root,
                project_id=project_id,
                justification=justification,
                candidate_title=candidate_title,
                timed_out=timed_out,
                duration_ms=duration_ms,
                account_name=account_name,
                proc_tree=proc_tree,
            )
            return

        now = time.monotonic()
        cutoff = now - self._cooldown_secs
        log = [t for t in self._failure_log.get(project_id, []) if t >= cutoff]
        log.append(now)
        self._failure_log[project_id] = log
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
            return

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
        escalation = Escalation(
            id=queue.make_id('curator'),
            task_id='task-curator',
            agent_role='fused-memory/task-curator',
            severity='blocking',
            category='curator_failure',
            summary='TaskCurator LLM failing; dedupe disabled',
            detail=detail,
            level=1,
        )
        try:
            queue.submit(escalation)
        except Exception:
            logger.exception(
                'curator_escalator: failed to submit escalation for project %s',
                project_id,
            )
            # Do not re-raise — falling through to action='create' is safer
            # than failing the add_task just because queue I/O broke.
            return

        logger.warning(
            'curator_escalator: queued L1 escalation %s for project %s '
            '(failure %d of %d in window)',
            escalation.id, project_id, count, self._ESCALATE_FIRST_N,
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
        touch ``_failure_log``): a systemic deny-list break must surface on every
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
            'FIX: the CLI tool-exclusion semantics changed again — the deny-list '
            'in shared/src/shared/cli_invoke.py (_REAL_BUILTIN_TOOLS_DENYLIST and '
            "the '*'-expansion in _invoke_claude) no longer permits the synthetic "
            'StructuredOutput schema tool, so every structured-output curator/recon '
            'call is permission-denied. Update that deny-list so StructuredOutput '
            'is NOT blocked, then restart fused-memory.service. Task dedupe is '
            'DISABLED for this project until the deny-list is fixed.',
        )
        detail = '\n'.join(detail_lines)

        queue = self._queue_for(project_root)
        escalation = Escalation(
            id=queue.make_id('curator'),
            task_id='task-curator',
            agent_role='fused-memory/task-curator',
            severity='blocking',
            category='curator_schema_tool_denied',
            summary=(
                'CRITICAL: schema StructuredOutput tool DENIED — CLI '
                'tool-exclusion semantics changed; the cli_invoke deny-list no '
                'longer permits the schema tool. Dedupe disabled until the '
                'deny-list is fixed.'
            ),
            detail=detail,
            level=1,
        )
        try:
            queue.submit(escalation)
        except Exception:
            logger.exception(
                'curator_escalator: failed to submit schema-tool-denied '
                'escalation for project %s',
                project_id,
            )
            # Do not re-raise — falling through to action='create' is safer than
            # failing add_task just because queue I/O broke. The loud escalation
            # is best-effort; the degrade-to-create still keeps the system limping.
            return

        logger.error(
            'curator_escalator: queued schema-tool-denied L1 escalation %s for '
            'project %s — StructuredOutput tool blocked by cli_invoke deny-list; '
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
    ) -> None:
        """Submit a distinct, un-suppressed escalation for a zero-output/full-timeout
        curator INFRA hang.

        Deliberately bypasses the rolling-window burst suppression (and does not
        touch ``_failure_log``): two hangs hours apart each read as "failure 1 of 3"
        under the normal window — the outage is invisible. Each ZOT must surface
        so operators can see a pattern across occurrences. The summary is unmistakable
        vs the generic "curator LLM failing" escalation, and the detail includes
        forensic evidence (account_name, proc_tree, duration_ms) so the next
        occurrence is diagnosable without re-reading logs.

        A short dedup window (_ZOT_DEDUP_WINDOW_SECS) prevents escalation floods
        from a single batch outage (bisect produces N concurrent size-1 curate()
        calls which each call report_failure independently).
        """
        # Dedup: a batch bisect of N candidates all hitting ZOT can call this
        # concurrently N times for the same project. Only the first submission
        # within _ZOT_DEDUP_WINDOW_SECS actually enqueues; the rest are logged
        # and dropped so the operator sees a pattern across outage events but
        # not a per-candidate flood within a single event.
        now_mono = time.monotonic()
        last = self._zot_last_submitted.get(project_id)
        if last is not None and (now_mono - last) < _ZOT_DEDUP_WINDOW_SECS:
            logger.info(
                'curator_escalator: deduplicating ZOT escalation for project %s '
                '(last submitted %.1fs ago < dedup window %.0fs); '
                'candidate_title=%r',
                project_id,
                now_mono - last,
                _ZOT_DEDUP_WINDOW_SECS,
                candidate_title,
            )
            return
        self._zot_last_submitted[project_id] = now_mono

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
        if proc_tree:
            # Truncate to avoid overwhelming the escalation body.
            snippet = proc_tree[:1500]
            detail_lines.append(f'proc_tree=\n{snippet}')
        detail_lines.append(f'justification={justification}')
        detail_lines.append('')
        detail_lines.append(
            'NOTE: this bypasses the 1-hour burst-suppression window because '
            'zero-output/full-timeout hangs hours apart otherwise each read as '
            '"failure 1 of 3" and never cross the escalate threshold — the outage '
            'is invisible. Root cause: transient Anthropic-backend degradation on '
            'the curator\'s sonnet+json-schema call shape (task 1550). Dedupe '
            'degraded to create for this candidate. The circuit-breaker watchdog '
            'will short-circuit further curator LLM calls if this recurs.',
        )
        detail = '\n'.join(detail_lines)

        queue = self._queue_for(project_root)
        escalation = Escalation(
            id=queue.make_id('curator'),
            task_id='task-curator',
            agent_role='fused-memory/task-curator',
            severity='blocking',
            category='curator_zero_output_hang',
            summary=(
                'curator zero-output/full-timeout INFRA hang — dedupe degraded to '
                'create; NOT a flaky candidate. Transient Anthropic-backend hang on '
                'sonnet+json-schema call shape.'
            ),
            detail=detail,
            level=1,
        )
        try:
            queue.submit(escalation)
        except Exception:
            logger.exception(
                'curator_escalator: failed to submit zero-output-timeout '
                'escalation for project %s',
                project_id,
            )
            # Do not re-raise — falling through to action='create' is safer than
            # failing add_task just because queue I/O broke.
            return

        logger.error(
            'curator_escalator: queued zero-output-timeout L1 escalation %s for '
            'project %s — account=%s duration_ms=%s; dedupe degraded to create',
            escalation.id, project_id, account_name, duration_ms,
        )
