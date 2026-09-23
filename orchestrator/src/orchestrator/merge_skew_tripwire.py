"""Merge-skew δ: pipeline-landing tripwire (task 2382).

M3 of ``plans/merge-skew-attribution-prd.md`` (PRD task δ, invariant I6,
boundary rows 5-6) — the proactive tripwire. On each successful merge
landing, if the landing's changed files trip a project-configured
load-bearing oracle, emit exactly ONE advisory info escalation naming the
landing sha and the in-flight tasks whose branch diffs OVERLAP the landing's
changed set (the ones whose own edits must be PORTED, not merely rebased),
and attach a steward-visible note to those tasks' metadata.

Advisory-only: never blocks/reorders the queue, ≤1 escalation per landing,
oracle absent/erroring → logged no-op, never delays advance. This module
holds the testable, injectable logic; ``Harness._maybe_pipeline_landing_tripwire``
(orchestrator/harness.py) is the thin fail-open adapter that wires it to the
real config/scheduler/git_ops/escalation_queue.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mirrors GitConfig.load_bearing_oracle_timeout_secs' default (config.py) —
# the default applied when a caller doesn't thread the configured value
# through explicitly (e.g. a direct unit-test call).
DEFAULT_ORACLE_TIMEOUT_SECS = 60.0


@dataclasses.dataclass(frozen=True)
class TripwireHit:
    """One in-flight task whose branch diff overlaps a load-bearing landing.

    ``overlap_files`` is the sorted intersection of this task's own branch
    diff with the landing's changed files — the files whose landed edits
    this task's branch must PORT (not merely rebase over).
    """

    task_id: str
    branch: str
    overlap_files: tuple[str, ...]


# The event loop holds only a WEAK reference to a Task, so an
# otherwise-unreferenced one can be garbage-collected mid-flight; membership
# in this set is what supplies the strong reference until the task ends.
#
# Mirrors the sibling ``_ABANDONED_PROBES`` / ``_abandon_probe`` registry in
# dashboard/src/dashboard/app.py (task 4089). This module does NOT call that
# package's ``track_task`` helper (dashboard/src/dashboard/data/db.py) —
# orchestrator has no dependency edge onto the leaf dashboard UI package, and
# ``_log_abandoned_oracle_cleanup`` below already does everything
# ``track_task``'s ``_release`` does (consumes the task's exception) plus
# DEBUG diagnostics it doesn't, so only the strong reference was missing
# here. This is now the THIRD near-identical copy of this pattern (the other
# two live in dashboard/); a follow-up (fused-memory ticket
# tkt_0RSP5RCWJGY202HZ0ZA0QRYCHE, dark_factory project) tracks consolidating
# it into shared/.
_ABANDONED_ORACLES: set[asyncio.Task] = set()

# A handful of concurrently-abandoned oracles is already unusual — at most
# one runs per merge landing. Crossing this many pending cleanups signals a
# persistent backlog (e.g. a wedged oracle whose grandchild holds the
# stdout/stderr pipes open — see _run_load_bearing_oracle's docstring) worth
# a human looking at, not one slow oracle; see _abandon_oracle below.
_ABANDONED_ORACLES_WARN_THRESHOLD = 8


def _log_abandoned_oracle_cleanup(task: asyncio.Task) -> None:
    """Done-callback for a load-bearing-oracle task abandoned after an
    exceptional exit from ``_run_load_bearing_oracle``'s wait — a timeout or
    the caller being cancelled (see ``_run_load_bearing_oracle`` and
    ``_abandon_oracle``) — releases the strong reference held in
    ``_ABANDONED_ORACLES``, retrieves the task's result/exception so a
    background cleanup failure never surfaces as an "exception was never
    retrieved" warning, and logs at DEBUG for diagnostics. Never raises.
    """
    _ABANDONED_ORACLES.discard(task)
    try:
        if task.cancelled():
            logger.debug('_run_load_bearing_oracle: abandoned oracle task finished cancelling')
            return
        exc = task.exception()
        if exc is not None:
            logger.debug(
                '_run_load_bearing_oracle: abandoned oracle task raised %r '
                'during background cleanup', exc,
            )
    except Exception:
        logger.debug(
            '_run_load_bearing_oracle: error inspecting abandoned oracle task', exc_info=True,
        )


def _abandon_oracle(task: asyncio.Task) -> None:
    """Cancel *task* fire-and-forget and hold a strong reference until it ends.

    Mirrors ``_abandon_probe`` in dashboard/src/dashboard/app.py (task
    4089) — the landed sibling fix for the same defect class.

    Also logs a WARNING naming the registry size once it exceeds
    :data:`_ABANDONED_ORACLES_WARN_THRESHOLD`: the per-call WARNING already
    logged by ``_run_load_bearing_oracle`` on timeout reads as a transient
    hiccup, so a persistent pile-up would otherwise stay invisible until
    something else notices the memory.
    """
    task.cancel()  # fire-and-forget — do NOT await the unwinding
    _ABANDONED_ORACLES.add(task)
    task.add_done_callback(_log_abandoned_oracle_cleanup)
    backlog = len(_ABANDONED_ORACLES)
    if backlog > _ABANDONED_ORACLES_WARN_THRESHOLD:
        logger.warning(
            '_run_load_bearing_oracle: %d abandoned oracle tasks still '
            'pending cleanup — possible persistent backlog (e.g. a wedged '
            'oracle grandchild holding the stdout/stderr pipes open)',
            backlog,
        )


async def _run_load_bearing_oracle(
    project_root: Path,
    oracle_cmd: list[str] | None,
    changed_files: list[str],
    *,
    timeout_secs: float = DEFAULT_ORACLE_TIMEOUT_SECS,
) -> bool:
    """Return True iff the project-configured load-bearing oracle says
    *changed_files* are load-bearing.

    Shells out to ``[*oracle_cmd, *changed_files]`` in *project_root* and
    returns True when the command exits 0 (conventional Unix predicate: exit
    0 ⟹ condition is true ⟹ load-bearing). The subprocess is bounded by
    *timeout_secs* — ``git_ops._run`` has no built-in timeout (its docstring
    requires callers to wrap it), and this call sits synchronously in the
    merge-landed hot path ahead of the service-restart coordinator fan-out,
    so an unbounded wait on a hung operator-supplied oracle script would
    violate I6 (never block/delay the advance).

    Deliberately NOT a plain ``asyncio.wait_for(_run(...), timeout=...)``:
    ``_run``'s own cancellation cleanup (``proc.kill()`` + ``await
    proc.wait()``, task 2608) reaps the DIRECT child, but if the oracle
    command itself spawns a grandchild that inherits the stdout/stderr
    pipes (e.g. a script that shells out further — a realistic shape for an
    operator-supplied oracle), that reap blocks until the grandchild
    releases the pipe, which can vastly outlast *timeout_secs* — ``wait_for``
    cannot re-interrupt a cleanup await that starts only *after* its one
    cancellation was already delivered, so it would end up waiting for the
    full cleanup anyway, silently defeating the bound. Instead the oracle
    run is a genuine ``asyncio.Task`` bounded via ``asyncio.wait(...,
    timeout=timeout_secs)``: on timeout the task is cancelled and
    *abandoned* (not awaited) — this function returns False immediately —
    while ``_run``'s kill+reap cleanup completes independently in the
    background (logged via :func:`_log_abandoned_oracle_cleanup`, never
    blocking a caller).

    The task is abandoned this way on EVERY exceptional exit from the
    ``asyncio.wait`` above, not only budget expiry: if this coroutine's own
    caller is cancelled while parked there, ``asyncio.wait`` does not cancel
    what it was waiting on, so the oracle task would otherwise be left
    referenced only by the event loop's WEAK task set. Either way the task
    is cancelled and held in the module-level :data:`_ABANDONED_ORACLES`
    strong-reference registry (via :func:`_abandon_oracle`) until it ends,
    then released.

    Fail-open contract — returns False for ANY of:
    - ``oracle_cmd`` is ``None`` or empty (tripwire disabled / misconfigured).
    - ``changed_files`` is empty.
    - The command exits non-zero (oracle says not load-bearing).
    - The command exceeds *timeout_secs* (treated as fail-open, not
      load-bearing — logged WARNING; cleanup abandoned to the background as
      described above).
    - The command is missing, non-executable, or any other exception is
      raised (an oracle hiccup must never wedge the merge-landed hot path —
      log WARNING, return False).

    The one exit that does NOT fail open: if this coroutine's caller is
    cancelled, the oracle task is abandoned (as above) but the
    ``CancelledError`` itself PROPAGATES rather than being laundered into
    ``False`` — a cancelled caller must stay cancelled.

    Mirrors ``verify._verify_pipeline_guard_requires_full_gate``'s fail-open
    shape, but takes a config-driven command list instead of a hardcoded
    ``scripts/verify-pipeline-guard.sh`` path.
    """
    try:
        if not oracle_cmd or not changed_files:
            return False
        from orchestrator.git_ops import _run  # noqa: PLC0415, I001 — lazy, mirrors _verify_pipeline_guard_requires_full_gate

        task = asyncio.ensure_future(_run([*oracle_cmd, *changed_files], cwd=project_root))
        try:
            _done, pending = await asyncio.wait({task}, timeout=timeout_secs)
        except BaseException:
            # asyncio.wait() does not cancel what it waits on, and this
            # catches BaseException (not just Exception) because
            # CancelledError — the caller being cancelled out from under us
            # — is a BaseException subclass since 3.8, so the outer
            # `except Exception` below cannot see it. Without this,
            # cancellation jumps over the abandon and strands the task in
            # the loop's WEAK set.
            _abandon_oracle(task)
            raise
        if pending:
            logger.warning(
                '_run_load_bearing_oracle: oracle command exceeded %.1fs for '
                '%s; treating as not-load-bearing (fail-open) and abandoning '
                'cleanup to the background',
                timeout_secs, project_root,
            )
            _abandon_oracle(task)
            return False
        rc, _out, _err = task.result()
        return rc == 0
    except Exception:
        logger.warning(
            '_run_load_bearing_oracle: unexpected error for %s',
            project_root, exc_info=True,
        )
        return False


def compute_tripwire_overlap(
    landing_changed_files: list[str],
    inflight_diffs: list[tuple[str, str, list[str]]],
) -> list[TripwireHit]:
    """Return a :class:`TripwireHit` for each in-flight task whose branch
    diff overlaps *landing_changed_files*.

    *inflight_diffs* is ``[(task_id, branch, branch_changed_files), ...]``.
    A task with ZERO overlap is silently absent from the result (not a
    zero-``overlap_files`` hit) — Open Q2's resolution: the landing's own
    changed files ARE the concrete load-bearing set the oracle just
    flagged, so intersecting each branch's diff against them yields exactly
    the files whose landed edits that branch must port.

    Deterministic: preserves *inflight_diffs* input order; each hit's
    ``overlap_files`` is sorted independently.
    """
    landing_set = set(landing_changed_files)
    if not landing_set:
        return []
    hits: list[TripwireHit] = []
    for task_id, branch, branch_changed_files in inflight_diffs:
        overlap = tuple(sorted(set(branch_changed_files) & landing_set))
        if overlap:
            hits.append(TripwireHit(task_id=task_id, branch=branch, overlap_files=overlap))
    return hits


async def emit_pipeline_landing_tripwire(
    *,
    project_root: Path,
    oracle_cmd: list[str] | None,
    escalation_queue: Any,
    landing_sha: str,
    landing_task_id: str,
    landing_changed_files: list[str],
    inflight: list[tuple[str, str]],
    get_branch_diff: Callable[[str], Awaitable[list[str] | None]],
    update_task: Callable[..., Awaitable[bool]],
    oracle_timeout_secs: float = DEFAULT_ORACLE_TIMEOUT_SECS,
) -> None:
    """Advisory pipeline-landing tripwire (PRD task δ, boundary rows 5-6).

    If *landing_changed_files* trips the project-configured load-bearing
    oracle (*oracle_cmd*), emit exactly ONE advisory info escalation naming
    *landing_sha* and every in-flight task (from *inflight*, a list of
    ``(task_id, branch)`` pairs — the just-landed task already excluded by
    the caller) whose branch diff — fetched via *get_branch_diff* —
    overlaps *landing_changed_files*, and attach a steward-visible note to
    each such task's metadata via *update_task*.

    On-task note durability — last-writer-wins by design: the note is written
    under the single ``x_merge_skew_tripwire`` metadata key with
    ``metadata_mode='merge'`` (shallow last-write-wins — omitted keys are
    preserved, this key's value is overwritten wholesale). If the SAME
    in-flight task is flagged by two distinct load-bearing landings before it
    merges, the second write replaces the first, so the on-task note reflects
    only the MOST RECENT landing. This is intentional, not lossy: the on-task
    note is a convenience pointer ("this branch must port a recently-landed
    change — see the escalation"), while the **escalation is the durable,
    complete per-landing record** — each landing files its own escalation
    under a unique ``pipeline-landing-tripwire-{sha12}`` sentinel (≤1 per
    landing), and a task hit by N landings is named in N distinct escalations,
    so no landing's signal is ever lost. Accumulating every landing's overlap
    onto the task would require a racy read-modify-write (``merge`` mode does
    not deep-merge a nested value) on this deliberately fail-open path, for no
    signal the escalation stream does not already carry durably.

    *oracle_timeout_secs* bounds the oracle subprocess (see
    ``_run_load_bearing_oracle``) — defaults to
    :data:`DEFAULT_ORACLE_TIMEOUT_SECS`, matching
    ``GitConfig.load_bearing_oracle_timeout_secs``'s default; callers should
    thread the configured value through explicitly (as
    ``Harness._maybe_pipeline_landing_tripwire`` does) rather than rely on
    this default.

    Fully fail-open (I6): *oracle_cmd* absent → logged no-op; oracle
    negative or timed-out → no-op; zero overlapping in-flight tasks → no-op
    (nothing actionable to report); ``escalation_queue`` ``None`` → no-op; a
    pre-existing pending escalation for this landing → dedup no-op (≤1
    escalation per landing). Every external call (oracle subprocess inside
    ``_run_load_bearing_oracle``, each ``get_branch_diff``, ``submit``,
    each ``update_task``) is individually guarded, and the whole body is
    wrapped again as a backstop — this function never raises for any
    oracle/git/task-update FAILURE.

    Caller cancellation is the one exit this does NOT swallow: it
    propagates BY DESIGN (``_run_load_bearing_oracle`` re-raises
    ``CancelledError`` rather than failing open — see its docstring and
    ``_ABANDONED_ORACLES``), and the ``except Exception:`` guards above and
    below must never be widened to ``except BaseException:`` to suppress
    it. A cancelled merge-landed hot path must stay cancelled; swallowing
    ``CancelledError`` here would break cooperative cancellation and
    re-strand the very oracle task ``_ABANDONED_ORACLES`` now tracks.
    """
    try:
        if not oracle_cmd:
            logger.debug('emit_pipeline_landing_tripwire: no oracle_cmd configured, no-op')
            return
        if not landing_changed_files:
            return
        load_bearing = await _run_load_bearing_oracle(
            project_root, oracle_cmd, landing_changed_files,
            timeout_secs=oracle_timeout_secs,
        )
        if not load_bearing:
            return
        if escalation_queue is None:
            return

        sentinel = f'pipeline-landing-tripwire-{landing_sha[:12]}'
        if escalation_queue.get_by_task(sentinel, status='pending'):
            logger.debug(
                'emit_pipeline_landing_tripwire: dedup — open escalation already '
                'exists for %s', sentinel,
            )
            return

        inflight_diffs: list[tuple[str, str, list[str]]] = []
        for task_id, branch in inflight:
            try:
                branch_files = await get_branch_diff(branch)
            except Exception:
                logger.warning(
                    'emit_pipeline_landing_tripwire: get_branch_diff failed for '
                    'task %s branch %s', task_id, branch, exc_info=True,
                )
                continue
            if branch_files is None:
                continue
            inflight_diffs.append((task_id, branch, branch_files))

        hits = compute_tripwire_overlap(landing_changed_files, inflight_diffs)
        if not hits:
            return

        from escalation.models import Escalation  # noqa: PLC0415, I001 — local import, escalation optional dep

        hit_lines = '\n'.join(
            f'- task {hit.task_id} (branch {hit.branch}): {", ".join(hit.overlap_files)}'
            for hit in hits
        )
        overlapping_task_ids = ', '.join(hit.task_id for hit in hits)
        summary = (
            f'Pipeline landing {landing_sha[:12]} touches load-bearing files also '
            f'present on {len(hits)} in-flight branch(es): {overlapping_task_ids}'
        )
        detail = (
            f'Landing task: {landing_task_id} (sha {landing_sha[:12]})\n'
            f'Load-bearing changed files: {", ".join(landing_changed_files)}\n'
            '\n'
            f'In-flight branches with overlapping edits:\n{hit_lines}'
        )
        esc = Escalation(
            id=escalation_queue.make_id(sentinel),
            task_id=sentinel,
            agent_role='orchestrator-merge-skew-tripwire',
            severity='info',
            level=0,
            category='risk_identified',
            summary=summary,
            detail=detail,
            suggested_action=(
                "Port the landed change into each named in-flight branch's own "
                'edits (not merely rebase) before it merges, to avoid silently '
                'reverting or conflicting with the load-bearing change.'
            ),
        )
        try:
            escalation_queue.submit(esc)
        except Exception:
            logger.warning(
                'emit_pipeline_landing_tripwire: escalation submit failed for %s',
                sentinel, exc_info=True,
            )

        for hit in hits:
            try:
                await update_task(
                    hit.task_id,
                    {
                        # x_-prefixed forward-compat namespace (CLAUDE.md
                        # "Tier-C: ad-hoc keys") — this is a one-off advisory
                        # annotation, not a blessed load-bearing metadata
                        # convention, so it must not be a bare top-level key
                        # (that would emit a task_metadata.schema_warning
                        # code=unknown_key census line on every read/write).
                        #
                        # metadata_mode='merge' is shallow last-write-wins:
                        # a task flagged by two landings keeps only the most
                        # recent one here. Intentional — see the docstring;
                        # the per-landing escalation (unique sentinel) is the
                        # durable, complete record, so nothing is lost.
                        'x_merge_skew_tripwire': {
                            'landing_sha': landing_sha,
                            'overlap_files': list(hit.overlap_files),
                        },
                    },
                    metadata_mode='merge',
                )
            except Exception:
                logger.warning(
                    'emit_pipeline_landing_tripwire: update_task failed for task %s',
                    hit.task_id, exc_info=True,
                )
    except Exception:
        logger.warning(
            'emit_pipeline_landing_tripwire: unexpected error for landing %s',
            landing_sha, exc_info=True,
        )
