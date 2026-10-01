"""The ``### Live-Workflow Signals`` payload section, shared by Stages 1 and 2.

This module owns that section: the per-render probe fan-out over the active
tasks, the vocabulary the section is rendered in, and the reading rules stated
over that vocabulary.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from fused_memory.mcp_tools.scheduler_state import read_scheduler_state
from fused_memory.models.scope import ProjectRoot
from fused_memory.reconciliation.task_filter import MAX_ACTIVE_TASKS_RENDERED
from fused_memory.services.live_workflow_detector import (
    corroboration_for_task,
    detect_live_workflow,
    is_pure_gate_metadata,
    worktree_index_kwargs,
)
from fused_memory.services.orchestrator_detector import (
    is_orchestrator_live_for,
    orchestrator_started_at,
)

logger = logging.getLogger(__name__)


async def render_live_workflow_section(
    tasks: list[dict],
    project_root: ProjectRoot,
    *,
    now: datetime | None = None,
) -> str:
    """Render the '### Live-Workflow Signals' payload section for *tasks*.

    For each task in *tasks*, calls :func:`detect_live_workflow` and collects
    tasks whose :attr:`WorkflowLiveness.is_live` is True.  Returns an empty
    string when no task is live (keeps the payload tight).

    Each live task is listed with the firing signal names so the Stage 2 LLM
    can see which evidence contributed to the live designation:

    ```
    ### Live-Workflow Signals
    - task/4321: worktree, recent-commit
    ```

    Detector errors per task are swallowed and logged at WARNING level (the task
    is treated as not-live for that call — fail-safe, matching the harness gate).

    Each task's ``status`` is forwarded to the detector, so never-dispatched
    statuses (deferred/done/cancelled — see
    :data:`~fused_memory.services.live_workflow_detector.ORCH_LIVE_INELIGIBLE_STATUSES`)
    drop the project-wide orchestrator_live signal for that task while leaving
    its per-task worktree/commit signals unaffected.

    Each task's ``metadata.task_kind`` is also forwarded.  A BLOCKED
    deterministic task (``task_kind == 'deterministic'``) never acquires a
    worktree/branch of its own — it is routed to ``DeterministicRunner``
    instead — so the project-wide orchestrator_live signal is dropped for it
    too, the same way it is for never-dispatched statuses.  A normal blocked
    task (``task_kind`` absent or not ``'deterministic'``) keeps the signal
    ONLY when it carries genuine per-task evidence (a registered worktree or a
    recent commit), since it may legitimately auto-unblock mid-pipeline; a
    blocked normal task whose ONLY evidence is the bare project-wide
    orchestrator lock also has the signal dropped (task 2409 — closes the
    repeated re-deferral loop this caused for tasks 2335/2196).

    A PENDING deterministic PURE GATE — ``always_escalates`` truthy with no
    ``before_done``, classified by :func:`is_pure_gate_metadata` and forwarded
    as ``pure_gate`` — likewise has the project-wide signal dropped (task
    3751).  Its entire ``DeterministicRunner`` run is "file one born-at-L2
    escalation, stamp ``gate_escalated_at``, set status blocked": no script, no
    systemd, no ``git_ops``, and (like every deterministic task) no
    worktree/branch, so the bare lock can never be task-specific evidence for
    it.  A pending deterministic task carrying a ``before_done`` KEEPS the
    signal: that path runs a blocking deploy/predicate script while the status
    is still ``'pending'`` (``Harness._run_deterministic_slot`` never flips it
    to ``'in-progress'``) with no git evidence to reveal it.  Confirmed
    incident: task 3845 was listed here with ONLY the bare ``orchestrator``
    signal for 3+ consecutive reconciliation cycles, blocking its disposition.

    **In-progress corroboration gate (task 2963).** For an ``in-progress`` task
    whose only live signals are a lingering registered worktree and/or the
    project-wide orchestrator lock (no ``recent_commit``), a fleet redeploy that
    KILLED the workflow leaves both those signals falsely asserting liveness.
    This renderer therefore computes an explicit per-task corroboration verdict
    (:func:`corroboration_for_task`) for every in-progress task and passes it to
    the detector as ``corroborated``.  Corroboration requires at least one FRESH
    per-task signal, ANY sufficient: (1) a live claimant/heartbeat, (2) the
    task_id present in the scheduler's ``current_holders``/``parks`` snapshot, or
    (3) a ``routing.latest.decided_at`` newer than the orchestrator's start time
    (parsed from the lock).  When none corroborates, the detector downgrades the
    task to ``indeterminate`` (``is_live=False``) and the
    ``if not liveness.is_live: continue`` below drops it from the section — so a
    stranded post-redeploy task is no longer reported live, unblocking recon's
    stranded-remediation path.  The scheduler-state snapshot and the
    orchestrator start-time are hoisted once per render (like the orchestrator
    hoist); both are fail-safe → ``None``.  Non-in-progress tasks pass
    ``corroborated=None`` so the gate stays inert (behavior unchanged).

    PER-RENDER HOISTS.  Four inputs to :func:`detect_live_workflow` are
    invariant across every task in one render, so each is computed ONCE here
    and threaded down through ``kwargs``:

    1. :func:`is_orchestrator_live_for` — one lock file per project_root.
    2. :func:`read_scheduler_state` — one snapshot per project_root.
    3. :func:`orchestrator_started_at` — one restart boundary per project_root.
    4. :func:`worktree_index_kwargs` — the whole-repo ``git worktree list
       --porcelain``.

    The fourth is the expensive one and the reason task 3778 exists.  The
    first three are local file reads; the fourth forks git and parses its
    entire output, and it was being re-run inside the detector for EVERY task.
    Measured on the dark_factory repo at ~513 registered worktrees: ~40 ms per
    call x ~500 tasks ≈ 20 s of a 29.2 s render — work that is not merely
    repeated but *identical* every time, and which blocked the event loop for
    its whole duration.

    The first three are batched behind ONE ``asyncio.to_thread`` hop.  They are
    small local file reads, but this coroutine exists to STOP occupying the
    event loop, and removing the blocking git loop while leaving stray
    synchronous file I/O behind would just shrink the stall rather than end it.
    One hop rather than three keeps the thread-pool churn flat.

    All four are wrapped fail-safe.  The worktree index owns its own wrapper,
    :func:`worktree_index_kwargs`, because the same three-valued contract has
    to hold for the harness integrity gate's identical hoist: *unknown* omits
    the kwarg and restores exactly the pre-hoist behaviour (each task probes for
    itself), while a known-empty repo arrives as ``{'worktree_index': {}}``, a
    real answer that suppresses the per-task probes.  Every route to *unknown*
    is logged at WARNING **by the detector, not here** — the anticipated
    failures (spawn error, non-zero rc, timeout) by
    :func:`worktree_index_for`, an unexpected exception by
    :func:`worktree_index_kwargs`.  None of them is swallowed, because an
    unknown index silently costs ~20 s per render, which is precisely the class
    of degradation this task was filed to make visible.

    FAN-OUT CAP.  Only the first
    :data:`~fused_memory.reconciliation.task_filter.MAX_ACTIVE_TASKS_RENDERED`
    tasks are probed; an overflow is clipped and reported at WARNING
    (``reconciliation.live_workflow_render_capped``, naming total/rendered/
    omitted — no silent truncation, mirroring the
    ``stages/task_knowledge_sync.py::TaskKnowledgeSync.MAX_DONE_AUDIT_RENDERED``
    treatment).

    A clipped render also says so IN THE SECTION HEADER (``### Live-Workflow
    Signals (probed the first 50 of 512 active tasks …)``), because the WARNING
    and the safety argument below are both invisible to the reader that acts on
    this payload.  Both stage prompts state the rule "absent from this section
    ⇒ no live signal"; under a cap, absence acquires a second meaning — *past
    the cap, never probed* — and the payload is the only place that can
    disclose which one applies.  The header is bare when nothing was clipped,
    so the common case reads exactly as before.

    Capping here is SAFE.  This section is *advisory* input to the Stage 2 LLM
    about tasks it can see in the Active Task Tree, and that tree is rendered
    from the identical prefix slice (``render_active_section`` does
    ``tree.active_tasks[:max_tasks]`` with the same constant, task_filter.py:1614).
    A task past the cap is therefore one the LLM was never shown and cannot act
    on, so declining to probe it removes work without removing information.
    The load-bearing guard against racing a live pipeline is NOT this section
    but :func:`recon_write_policy.check` Gate 2, which is per-task, uncapped,
    and evaluated at write time.

    The bound is the deterministic prefix slice, NOT ``render_active_section``'s
    returned ``visible_active`` list, for two reasons.  (1) That function
    returns ``[]`` whenever its 50_000-char budget clamp trips
    (task_filter.py:1622-1635) — reusing it would silently delete this entire
    section on exactly the largest, most contended cycles.  The prefix slice is
    the superset of what can appear and never collapses.  (2) It is computed in
    ``assemble_payload``, while ``memory_consolidator`` calls this renderer by a
    different path; the slice is reproducible from ``tasks`` alone.

    The cap lives in the RENDERER rather than at its two call sites so
    task_knowledge_sync and memory_consolidator cannot drift apart.

    Args:
        tasks: Task dicts from the active/proactive-sample pool.  Only tasks
            with a parseable ``id`` are inspected (non-int ids are skipped).
            Clipped to the first ``MAX_ACTIVE_TASKS_RENDERED`` entries — see
            the fan-out cap paragraph above.
        project_root: Absolute path to the project root, forwarded to the
            detector and used to read the orchestrator lock + scheduler-state
            snapshot for the in-progress corroboration gate.
        now: Injectable reference time for deterministic tests.  Also the
            reference used for the claimant-heartbeat freshness check in the
            in-progress corroboration gate.

    Returns:
        A Markdown section string (e.g. ``'### Live-Workflow Signals\\n...\\n'``),
        or ``''`` when no tasks are live.  The header carries a
        ``(probed the first N of M active tasks …)`` scope note when — and only
        when — the fan-out cap clipped the input; see the fan-out cap
        paragraph.  It stays a prefix of the bare header either way, so a
        consumer grepping for ``'### Live-Workflow Signals'`` is unaffected.
    """
    if not tasks:
        return ''

    # Bound the fan-out (task 3778). The caller hands us the FULL active-task
    # pool, but the Active Task Tree the Stage 2 LLM actually sees is rendered
    # from the identical prefix slice of the same constant, so probing past it
    # is git work whose result is discarded. Clip explicitly and report the
    # drop at WARNING — never a silent truncation. See the docstring's
    # "Fan-out cap" paragraph for why this is safe and why the prefix slice
    # (not render_active_section's visible_active) is the right bound.
    total_active = len(tasks)
    header_scope = ''
    if total_active > MAX_ACTIVE_TASKS_RENDERED:
        omitted = total_active - MAX_ACTIVE_TASKS_RENDERED
        tasks = tasks[:MAX_ACTIVE_TASKS_RENDERED]
        # Say so IN THE SECTION, not just in the log. Both stage prompts tell
        # the LLM that absence from this section means "no live signal"; once
        # the fan-out is capped, absence has a second meaning ("past the cap,
        # never probed") that only the payload itself can disclose to the
        # reader acting on it.
        header_scope = (
            f' (probed the first {MAX_ACTIVE_TASKS_RENDERED} of {total_active} '
            f'active tasks — the same cap the Active Task Tree applies, so every '
            f'task shown there was probed)'
        )
        logger.warning(
            'reconciliation.live_workflow_render_capped: probed %d of %d active '
            'task(s); %d omitted by the MAX_ACTIVE_TASKS_RENDERED=%d cap (the '
            'same cap the Active Task Tree applies, so no visible task is missed)',
            MAX_ACTIVE_TASKS_RENDERED,
            total_active,
            omitted,
            MAX_ACTIVE_TASKS_RENDERED,
            extra={
                'total_active': total_active,
                'rendered': MAX_ACTIVE_TASKS_RENDERED,
                'omitted': omitted,
            },
        )

    # Hoist the project-level orchestrator check: it is constant for this
    # project_root (one lock file regardless of how many tasks are inspected).
    # Swallow any detector errors here — the per-task detect_live_workflow calls
    # will gracefully degrade on subsequent orchestrator checks.
    def _read_local_hoists() -> tuple[bool | None, dict | None, datetime | None]:
        # Three small local-file reads, batched into ONE thread hop below.
        try:
            orch_live: bool | None = is_orchestrator_live_for(project_root)
        except Exception:
            orch_live = None  # let detect_live_workflow derive it per-task
        try:
            sched: dict | None = read_scheduler_state(Path(project_root))
        except Exception:
            sched = None
        try:
            started: datetime | None = orchestrator_started_at(project_root)
        except Exception:
            started = None
        return orch_live, sched, started

    project_orch_live, scheduler_state, orch_started = await asyncio.to_thread(
        _read_local_hoists
    )

    kwargs: dict = {} if now is None else {'now': now}
    if project_orch_live is not None:
        kwargs['_orchestrator_live'] = project_orch_live

    # Hoist the per-render corroboration inputs (task 2963), mirroring the
    # orchestrator hoist above: both the scheduler-state snapshot and the
    # orchestrator restart-boundary timestamp are constant for this
    # project_root, so read each once. Both are wrapped fail-safe → None on any
    # error (corroboration_for_task tolerates None inputs; a None simply means
    # that corroboration signal cannot fire — never a raise). now_eff is the
    # reference time threaded into the claimant-freshness check.
    now_eff = now or datetime.now(UTC)

    # Hoist the whole-repo worktree list (task 3778) — the FOURTH per-render
    # invariant and by far the most expensive. See the docstring's "Per-render
    # hoists" paragraph: this one `git worktree list --porcelain` was running
    # inside detect_live_workflow for EVERY task, ~40 ms x ~500 tasks ≈ 20 s of
    # a measured 29 s render. worktree_index_kwargs owns the whole three-valued
    # contract — fail-safe, logging, and the unknown → omit-the-kwarg rule.
    kwargs.update(await worktree_index_kwargs(str(project_root)))

    live_lines: list[str] = []

    for task in tasks:
        raw_id = task.get('id')
        if raw_id is None:
            continue
        task_id = str(raw_id)
        raw_metadata = task.get('metadata')
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        task_kind = metadata.get('task_kind')
        # `metadata` is already the isinstance-guarded dict above, so a task
        # with absent/non-dict metadata yields pure_gate=False — fail-safe
        # toward live. See the docstring's pending-pure-gate paragraph.
        pure_gate = is_pure_gate_metadata(metadata)

        # Per-task corroboration gate (task 2963). For an IN-PROGRESS task,
        # compute an explicit corroboration verdict so the detector can downgrade
        # a killed-but-lingering task — whose only live signals are a stale
        # registered worktree and/or the project-wide orchestrator lock (no
        # recent_commit) — to indeterminate (is_live=False), which the
        # `if not liveness.is_live: continue` below then drops from the section.
        # A fresh per-task signal (live claimant/heartbeat, scheduler
        # holder/park, or a post-restart routing decision) keeps corroborated
        # True and the task listed. Non-in-progress tasks pass corroborated=None
        # so the gate stays inert (behavior unchanged). Fail-safe TOWARD live:
        # any assembler error leaves corroborated=None.
        corroborated: bool | None = None
        if task.get('status') == 'in-progress':
            try:
                corroborated = corroboration_for_task(
                    task, task_id, now=now_eff,
                    scheduler_state=scheduler_state,
                    orchestrator_started_at=orch_started,
                )
            except Exception:
                corroborated = None

        try:
            liveness = await detect_live_workflow(
                task_id, project_root,
                status=task.get('status'), task_kind=task_kind,
                pure_gate=pure_gate,
                corroborated=corroborated, **kwargs
            )
        except Exception:
            logger.warning(
                'reconciliation.render_live_workflow_section: '
                'detector error for task_id=%s; treating as not-live',
                task_id,
            )
            continue

        if not liveness.is_live:
            continue

        # Collect which signals fired for human-readable display.
        signals: list[str] = []
        if liveness.worktree_registered:
            signals.append('worktree')
        if liveness.recent_commit:
            signals.append('recent-commit')
        if liveness.orchestrator_live:
            signals.append('orchestrator')
        signal_str = ', '.join(signals) if signals else 'live'
        live_lines.append(f'- {liveness.branch}: {signal_str}')

    if not live_lines:
        return ''

    return f'### Live-Workflow Signals{header_scope}\n' + '\n'.join(live_lines) + '\n'
