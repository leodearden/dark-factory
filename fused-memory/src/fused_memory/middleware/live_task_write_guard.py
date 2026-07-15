"""Code-enforced before/after live-task-write self-check (task 2624).

Stage 2 reconciliation (``task_knowledge_sync``) runs as an LLM-driven CLI
agent that makes ``update_task``/``set_task_status`` MCP calls directly
against :class:`fused_memory.middleware.task_interceptor.TaskInterceptor`.
The "read status/claimant_run_id/heartbeat_at via ``get_task`` immediately
BEFORE and AFTER a write to a live in-progress task, and self-file a finding
on unexpected divergence" rule (incident: dark_factory task 2588 un-claim,
mem0 5bde6829) was, before this module, only an LLM prompt convention —
skippable by an inattentive/budget-constrained run. This module makes that
rule code-enforced at the interceptor's recon-stage write seam.

This is a leaf ``middleware`` module (sibling of ``recon_write_policy.py``):
it imports nothing from ``reconciliation/`` or ``server/``. The finding
"filer" is an injected async callback (see :func:`guarded_recon_task_write`),
so this module stays channel-agnostic — it has no opinion on where a
:class:`LifecycleResetFinding` ends up.

Design notes
------------
- :func:`detect_lifecycle_reset` fires on unexpected ``claimant_run_id`` OR
  ``status`` divergence — the true lifecycle-reset signals — after
  subtracting what the write itself requested. ``heartbeat_at`` is recorded
  in a resulting finding's ``diverged_fields`` as corroborating evidence but
  is NEVER an independent trigger: a live in-progress task's owner
  heartbeats periodically, so a legitimate concurrent heartbeat refresh
  coinciding with a Stage-2 write must not false-positive.
- The guard only engages when the before-state ``has_live_claimant`` (i.e.
  ``status == 'in-progress'`` with a present/non-blank ``claimant_run_id``);
  otherwise the write is dormant and proceeds unguarded.
- Filing is best-effort: a filer/after-read exception is logged and never
  breaks the underlying task write, which has already succeeded by the time
  the after-read/detect/file sequence runs.
- Detection window: the before-read, ``do_write``, and after-read all run
  while the caller holds ``TaskInterceptor._write_lock(project_id)`` (see
  the ``update_task`` and ``_apply_status_transition`` call sites), so a
  single guarded write is fully serialized against any *other* write that
  also takes that same per-project lock — such a same-process,
  lock-respecting writer cannot land between the before- and after-read.
  This is not a gap for the incident this backstop targets: the
  incident-2588 un-claim class is ``TaskInterceptor.set_task_claimant``
  (used for stranded-task release and the periodic heartbeat refresh),
  which by design takes NO write-lock at all (see its own docstring —
  "mirrors get_statuses: no write-lock"). It, and any genuinely
  out-of-process writer (a second server instance, a direct backend
  write), therefore freely interleaves with a guarded write's window and
  remains observable. A hypothetical same-process writer that *does* take
  ``_write_lock`` for the same task during a guarded write's window would
  queue behind it instead of interleaving — that narrower case is outside
  what a single guarded write can observe.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from shared.task_statuses import TaskStatus

logger = logging.getLogger(__name__)

# Stable, machine-readable flag_type surfaced on a LifecycleResetFinding and
# consumed by the recon_report channel (server/recon_lifecycle_filer.py).
LIFECYCLE_RESET_FLAG_TYPE = 'task_lifecycle_reset_detected'


# ---------------------------------------------------------------------------
# LiveTaskSnapshot / extraction helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveTaskSnapshot:
    """A minimal point-in-time read of the fields that define "live claimant"."""

    status: str
    claimant_run_id: str | None
    heartbeat_at: str | None


def _unwrap_task_data(task_data: Any) -> Mapping:
    """Return the dict-like payload from a ``get_task`` result.

    Mirrors ``task_interceptor._extract_status``'s flat/nested shape
    handling exactly: the presence of a top-level ``'status'`` key decides
    flat vs. nested-under-``'data'`` — the same discriminator the
    interceptor already uses to parse the same ``get_task`` response, so
    this reads the identical shape.
    """
    if not isinstance(task_data, Mapping):
        return {}
    if 'status' in task_data:
        return task_data
    data = task_data.get('data')
    if isinstance(data, Mapping):
        return data
    return {}


def extract_live_snapshot(task_data: Any) -> LiveTaskSnapshot:
    """Extract ``(status, claimant_run_id, heartbeat_at)`` from a ``get_task`` result.

    Handles both flat (``{'status': ...}``) and nested
    (``{'data': {'status': ...}}``) response shapes. Missing/absent fields
    default safely (``status`` -> ``'unknown'``, the other two -> ``None``)
    rather than raising, since this runs on the write path and must never
    itself break a write.
    """
    payload = _unwrap_task_data(task_data)
    return LiveTaskSnapshot(
        status=str(payload.get('status', 'unknown')),
        claimant_run_id=payload.get('claimant_run_id'),
        heartbeat_at=payload.get('heartbeat_at'),
    )


def has_live_claimant(task_data: Any) -> bool:
    """True iff *task_data* is ``in-progress`` with a present/non-blank claimant.

    Mirrors the "live claimant" half of ``shared.task_claimant.is_stranded``'s
    field semantics (status gate + claimant present/non-blank check), applied
    to a single snapshot rather than a staleness window.
    """
    snapshot = extract_live_snapshot(task_data)
    if snapshot.status != TaskStatus.IN_PROGRESS.value:
        return False
    claimant = snapshot.claimant_run_id
    return isinstance(claimant, str) and claimant.strip() != ''


# ---------------------------------------------------------------------------
# LifecycleResetFinding / detect_lifecycle_reset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleResetFinding:
    """A detected unexpected lifecycle divergence on a live in-progress task."""

    task_id: str
    project_id: str
    op: str
    before: LiveTaskSnapshot
    after: LiveTaskSnapshot
    diverged_fields: tuple[str, ...]
    flag_type: str = LIFECYCLE_RESET_FLAG_TYPE
    description: str = ''


def detect_lifecycle_reset(
    before: Any,
    after: Any,
    *,
    op: str,
    task_id: str,
    project_id: str,
    requested_status: str | None,
    requested_claimant_write: bool,
    requested_heartbeat_write: bool,
) -> LifecycleResetFinding | None:
    """Pure detector: compare before/after snapshots for an unexpected reset.

    Returns ``None`` unless :func:`has_live_claimant` holds for *before* —
    the guard only concerns itself with writes to a live in-progress task.

    A finding fires iff ``claimant_run_id`` or ``status`` diverged
    *unexpectedly* — i.e. the divergence is not what the write itself
    requested:

    - ``claimant_run_id`` divergence is unexpected unless
      ``requested_claimant_write`` is True (the write explicitly intended to
      change it, e.g. a claimant clear/release).
    - ``status`` divergence is unexpected unless the resulting status is
      exactly ``requested_status`` (the write asked for that transition and
      got it).

    ``heartbeat_at`` divergence is never an independent trigger — it is
    recorded in the resulting finding's ``diverged_fields`` only as
    corroborating evidence, and only when the finding already fires for a
    claimant/status reason and the heartbeat change was not itself requested
    (``requested_heartbeat_write``).
    """
    if not has_live_claimant(before):
        return None

    before_snapshot = extract_live_snapshot(before)
    after_snapshot = extract_live_snapshot(after)

    diverged: list[str] = []

    claimant_changed = before_snapshot.claimant_run_id != after_snapshot.claimant_run_id
    if claimant_changed and not requested_claimant_write:
        diverged.append('claimant_run_id')

    status_changed = before_snapshot.status != after_snapshot.status
    status_matches_request = (
        requested_status is not None and after_snapshot.status == requested_status
    )
    if status_changed and not status_matches_request:
        diverged.append('status')

    if not diverged:
        return None

    heartbeat_changed = before_snapshot.heartbeat_at != after_snapshot.heartbeat_at
    if heartbeat_changed and not requested_heartbeat_write:
        diverged.append('heartbeat_at')

    # description is intentionally left at the dataclass default here: the
    # only consumer (server/recon_lifecycle_filer.py's _describe) builds its
    # own description from before/after/diverged_fields, so computing one
    # here too would be a dead, drift-prone duplicate (task 2624 amendment).
    return LifecycleResetFinding(
        task_id=task_id,
        project_id=project_id,
        op=op,
        before=before_snapshot,
        after=after_snapshot,
        diverged_fields=tuple(diverged),
    )


# ---------------------------------------------------------------------------
# guarded_recon_task_write
# ---------------------------------------------------------------------------

GetTaskFn = Callable[..., Awaitable[Any]]
DoWriteFn = Callable[[], Awaitable[Any]]
FileFindingFn = Callable[[LifecycleResetFinding], Awaitable[Any]]


async def guarded_recon_task_write(
    *,
    task_id: str,
    project_id: str,
    op: str,
    get_task: GetTaskFn,
    do_write: DoWriteFn,
    file_finding: FileFindingFn,
    project_root: str,
    tag: str | None = None,
    before_task: Any = None,
    requested_status: str | None = None,
    requested_claimant_write: bool = False,
    requested_heartbeat_write: bool = False,
) -> Any:
    """Run *do_write*, guarded by a before/after live-task self-check.

    Resolves the before-state from *before_task* when supplied (the caller
    already read it, e.g. for ``recon_write_policy.check``), otherwise reads
    it via ``get_task(task_id, project_root, tag)``. When the before-state is
    not a live in-progress claimant, the guard is dormant: *do_write* runs
    and its result is returned directly, with no after-read.

    Otherwise: *do_write* is awaited (its exceptions propagate unchanged —
    the guard never suppresses a real write failure); then, best-effort
    (any exception here is logged and swallowed, never re-raised — filing a
    diagnostic must never turn an already-succeeded write into a failure for
    the caller), the after-state is read, :func:`detect_lifecycle_reset` is
    run, and on a hit *file_finding* is awaited. The write's result is always
    returned unchanged.
    """
    before = (
        before_task if before_task is not None else await get_task(task_id, project_root, tag)
    )

    if not has_live_claimant(before):
        return await do_write()

    result = await do_write()

    try:
        after = await get_task(task_id, project_root, tag)
        finding = detect_lifecycle_reset(
            before,
            after,
            op=op,
            task_id=task_id,
            project_id=project_id,
            requested_status=requested_status,
            requested_claimant_write=requested_claimant_write,
            requested_heartbeat_write=requested_heartbeat_write,
        )
        if finding is not None:
            await file_finding(finding)
    except Exception:
        logger.exception(
            'live_task_write_guard: best-effort after-read/detect/file failed for '
            'task_id=%r project_id=%r op=%r; write already succeeded, not retrying',
            task_id,
            project_id,
            op,
        )

    return result
