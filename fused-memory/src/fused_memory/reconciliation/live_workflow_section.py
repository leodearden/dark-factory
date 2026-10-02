"""The ``### Live-Workflow Signals`` payload section, shared by Stages 1 and 2.

This module owns that section: the per-render probe fan-out over the active
tasks, the vocabulary the section is rendered in, and the reading rules stated
over that vocabulary.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from shared.timestamps import parse_timestamp_or_warn

from fused_memory.mcp_tools.scheduler_state import read_scheduler_state
from fused_memory.models.scope import ProjectRoot
from fused_memory.reconciliation.task_filter import MAX_ACTIVE_TASKS_RENDERED
from fused_memory.services.landed_on_main import LandingQuery, LandingVerdict, probe_landing
from fused_memory.services.live_workflow_detector import (
    DEFAULT_BRANCH_PREFIX,
    DEFAULT_HEARTBEAT_TTL,
    ClaimantLabel,
    WorkflowLiveness,
    claimant_label,
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

LIVE_WORKFLOW_SECTION_HEADER = '### Live-Workflow Signals'

#: A listed task with no live signal: it is listed only because its work landed.
NOT_LIVE_TOKEN = 'not live'

#: A live task none of whose own signals fired: it is listed only through the
#: project-wide orchestrator lock, which is not per-task evidence.
NO_PER_TASK_SIGNAL_TOKEN = 'no per-task signal'

CLAIMANT_FIELD = 'claimant='

#: The project line's distinguishing prefixes; the lock is project-wide, so it is
#: rendered once for the section, never per task.
PROJECT_LOCK_HELD = 'Project-wide: orchestrator lock HELD'
PROJECT_LOCK_UNKNOWN = 'Project-wide: orchestrator lock UNKNOWN'

#: How much of a landing commit's sha a row quotes.
_COMMIT_ABBREV = 10

#: Where a Stage 1 finding citing a signal this section did not render is
#: annotated (``live_workflow_citation_guard``), and the stat counting them.
CITATION_CONTRADICTIONS_KEY = 'live_workflow_citation_contradictions'
CITATION_CONTRADICTIONS_STAT = 'stage1_live_workflow_citation_contradictions'


class LiveSignal(StrEnum):
    WORKTREE = 'worktree'
    RECENT_COMMIT = 'recent-commit'
    ORCHESTRATOR = 'orchestrator'


#: The signals a row can carry; ORCHESTRATOR belongs to the project line.
PER_TASK_SIGNALS = (LiveSignal.WORKTREE, LiveSignal.RECENT_COMMIT)


class LandedToken(StrEnum):
    """The ``landed=`` field of a row; a TRUE field also names its evidence."""

    TRUE = 'landed=true'
    FALSE = 'landed=false'
    UNKNOWN = 'landed=unknown'


LIVE_WORKFLOW_RULES_HEADING = '### Reading the Live-Workflow Signals section'

_HEARTBEAT_TTL_MINUTES = int(DEFAULT_HEARTBEAT_TTL.total_seconds() // 60)


def render_live_workflow_authority_rules() -> str:
    """The reading rules over this section's rendered tokens, for the Stage 1 and 2 prompts.

    Stage 3 never receives the section, so it must not get rules over it. The
    block uses ``###`` headings only, because each stage's
    ``## Live-Workflow Authority`` region is sliced at the next ``## `` heading.
    """
    claimant = {label: f'`{CLAIMANT_FIELD}{label}`' for label in ClaimantLabel}
    signals = ' / '.join(f'`{signal}`' for signal in LiveSignal)
    return (
        f'{LIVE_WORKFLOW_RULES_HEADING}\n'
        f'Each row reads `- task/<id>: <per-task signals>; {CLAIMANT_FIELD}<label>; '
        f'landed=<...>`. The per-task signals are `{LiveSignal.WORKTREE}` and '
        f'`{LiveSignal.RECENT_COMMIT}`. The orchestrator lock is project-wide: when it '
        f'is held the section says so once, on its own `{PROJECT_LOCK_HELD}` line, and '
        f'that line is project context, never evidence about any one task. A row '
        f'reading `{NO_PER_TASK_SIGNAL_TOKEN}` is listed only through that lock. A '
        f'registered worktree survives task completion, so it does not prove a task '
        f'is running either.\n\n'
        f'A listed row is a HINT that triggers the authoritative check. It never '
        f'substitutes for that check, and on its own it is never grounds for '
        f'suppressing a write. The authoritative per-task liveness test is '
        f'`mcp__fused-memory__get_task(task_id)`: the top-level `claimant_run_id` is '
        f'non-null AND `heartbeat_at` is fresh (within about {_HEARTBEAT_TTL_MINUTES} '
        f'minutes). Both are required. When the record disagrees with the listing, '
        f'the RECORD WINS: perform the write, or emit the finding, that you would '
        f'otherwise have made, and record in it that the tie-breaker was applied and '
        f'what `get_task` returned. The live-task rules in this section apply only '
        f'once the record confirms the listing. This tie-breaker settles liveness '
        f'only; it never overrides the `landed=` rules below. Task 3254 rendered '
        f'`worktree, orchestrator` for 3+ cycles while `get_task` read pending with a '
        f'null claimant and a null heartbeat, and a suppressed write stalled a '
        f'decision task.\n\n'
        f'The `{CLAIMANT_FIELD}` field is that same test precomputed at render time. '
        f'{claimant[ClaimantLabel.NONE]} (no claimant) and '
        f'{claimant[ClaimantLabel.STALE]} (a claimant whose heartbeat is old, missing '
        f'or unreadable) are grounds to proceed; {claimant[ClaimantLabel.LIVE]} is '
        f'grounds to hold; {claimant[ClaimantLabel.UNKNOWN]} means the task record '
        f'could not be read. A `get_task` read settles any of them.\n\n'
        f'Every row ends with a `landed=` field saying whether the task\'s work is '
        f'already on main. A row that reads `{NOT_LIVE_TOKEN}` is listed only because '
        f'its work landed.\n\n'
        f"`{LandedToken.TRUE}` means the task's work is already on main: either its "
        f'branch is gone and a fresh `Merge task/<id> into main` marker is on main, or '
        f'every commit on its branch has a rebased twin on main. The evidence is named '
        f'in parentheses. Such a task is NOT stranded work, whatever its status says. '
        f'Never recommend or perform any of these for it: resume it, redispatch it, '
        f'reset it to pending, or reopen it. The correct disposition is one '
        f'info-severity, non-actionable finding saying the work landed but the '
        f"task's status lags, citing the rendered evidence.\n\n"
        f'`{LandedToken.FALSE}` means there is no positive evidence of landing, and '
        f'`{LandedToken.UNKNOWN}` means the landing probe failed. Neither is evidence '
        f'that the work is unmerged; the stranded rules in this section apply '
        f'unchanged.\n\n'
        f'For a task whose status is done, `metadata.done_provenance` (read it with '
        f"`get_task`) is the merge lane's landing record, and it outranks any branch "
        f'state. Never set a done task back to pending or in-progress on branch-state '
        f'grounds. A refused reopen is not grounds to escalate; record an info '
        f'finding instead (the task-3838 reopen storm).\n\n'
        f'Never assert or cite a live-workflow signal ({signals}) for a task unless '
        f"that task has a row in THIS payload's `{LIVE_WORKFLOW_SECTION_HEADER}` "
        f'section. The section rendered in this payload is the sole authority for '
        f"this cycle: a signal quoted from a prior cycle's Stage 3 findings, a "
        f'buffered episode or a Mem0 memory is not evidence of current liveness. If a '
        f'task has no row, say that its signals are absent rather than describing '
        f'them. A `{NOT_LIVE_TOKEN}` row carries no liveness signal at all. A Stage 1 '
        f'finding that contradicts the section is detected after the fact, annotated '
        f'`{CITATION_CONTRADICTIONS_KEY}`, and counted in '
        f'`{CITATION_CONTRADICTIONS_STAT}` (reify run 6aa50844 cited task/5891 as '
        f'live while its section listed nothing).'
    )


def _project_line(project_orchestrator_live: bool | None) -> str | None:
    if project_orchestrator_live is None:
        return f'{PROJECT_LOCK_UNKNOWN} (probe failed)'
    if project_orchestrator_live:
        return (
            f'{PROJECT_LOCK_HELD} — it fires for every task in this project, '
            f'so it is not per-task evidence'
        )
    return None


def _legend() -> str:
    signals = ' | '.join(PER_TASK_SIGNALS)
    claimants = ' | '.join(f'{CLAIMANT_FIELD}{label}' for label in ClaimantLabel)
    landed = ' | '.join(LandedToken)
    return (
        f'Legend: per-task signals are {signals}, or "{NO_PER_TASK_SIGNAL_TOKEN}" when '
        f'only the project-wide lock lists the task; {claimants} is the task '
        f"record's claimant heartbeat; {landed} says whether the work is already on "
        f'main; a "{NOT_LIVE_TOKEN}" row is listed only because its work landed.'
    )


def _landed_token(verdict: LandingVerdict) -> str:
    if verdict.landed is None:
        return LandedToken.UNKNOWN
    if not verdict.landed:
        return LandedToken.FALSE
    evidence = [verdict.evidence.value] if verdict.evidence is not None else []
    if verdict.commit:
        evidence.append(verdict.commit[:_COMMIT_ABBREV])
    return f'{LandedToken.TRUE} ({" ".join(evidence)})'


@dataclass(frozen=True)
class LiveWorkflowRow:
    """One listed task: live, or not live with its work landed on main.

    *signals* are the per-task live signals that fired, and are empty for a row
    that is not live or is live only through the project-wide lock.
    """

    task_id: str
    branch: str
    is_live: bool
    signals: tuple[LiveSignal, ...]
    claimant: ClaimantLabel
    landing: LandingVerdict

    def render(self) -> str:
        return (
            f'- {self.branch}: {self._state()}; {CLAIMANT_FIELD}{self.claimant}; '
            f'{_landed_token(self.landing)}'
        )

    def _state(self) -> str:
        if not self.is_live:
            return NOT_LIVE_TOKEN
        return ', '.join(self.signals) or NO_PER_TASK_SIGNAL_TOKEN


@dataclass(frozen=True)
class LiveWorkflowSnapshot:
    """What one render found, exactly as the section shows it to the stage LLM.

    *probed* of *total_active* tasks were probed; fewer means the fan-out cap
    clipped the input, and the header says so. *project_orchestrator_live* is
    the hoisted project-wide lock check, None when that check failed.
    """

    rows: tuple[LiveWorkflowRow, ...]
    project_orchestrator_live: bool | None
    probed: int = 0
    total_active: int = 0

    def row_for(self, task_id: str) -> LiveWorkflowRow | None:
        return next((row for row in self.rows if row.task_id == task_id), None)

    def render(self) -> str:
        """The section text, or ``''`` when no task is listed.

        Neither the project line nor the legend starts with ``#`` or ``- `` or
        names a ``task/<id>``, so the rows stay the only per-task lines.
        """
        if not self.rows:
            return ''
        header = LIVE_WORKFLOW_SECTION_HEADER + self._header_scope()
        project_line = _project_line(self.project_orchestrator_live)
        preamble = [header, *([project_line] if project_line else []), _legend()]
        return '\n'.join([*preamble, *(row.render() for row in self.rows)]) + '\n'

    def _header_scope(self) -> str:
        if self.total_active <= self.probed:
            return ''
        return (
            f' (probed the first {self.probed} of {self.total_active} '
            f'active tasks — the same cap the Active Task Tree applies, so every '
            f'task shown there was probed)'
        )


async def render_live_workflow_section(
    tasks: list[dict],
    project_root: ProjectRoot,
    *,
    now: datetime | None = None,
) -> str:
    """Render the section for *tasks*: :func:`build_live_workflow_snapshot`, rendered."""
    snapshot = await build_live_workflow_snapshot(tasks, project_root, now=now)
    return snapshot.render()


async def build_live_workflow_snapshot(
    tasks: list[dict],
    project_root: ProjectRoot,
    *,
    now: datetime | None = None,
) -> LiveWorkflowSnapshot:
    """Probe *tasks* and return the '### Live-Workflow Signals' snapshot.

    Each listed task is rendered with the per-task live signals that fired, so
    the stage LLM can see which evidence contributed to the live designation,
    with its claimant label and with its landing verdict.  The project-wide
    orchestrator lock is one project line, and a legend names every token:

    ```
    ### Live-Workflow Signals
    Project-wide: orchestrator lock HELD — it fires for every task in this ...
    Legend: per-task signals are worktree | recent-commit, or ...
    - task/4321: worktree, recent-commit; claimant=live; landed=false
    - task/4322: not live; claimant=none; landed=true (merge-marker 1a2b3c4d5e)
    ```

    The claimant label (:func:`claimant_label`, computed from the task dict
    already in hand) and the project line are display-only: they change no
    ``is_live`` verdict, detector gate or ``recon_write_policy`` behaviour, so
    task 2964's render/Gate-2 agreement holds by construction (task 4430).

    Each task's ``status``, ``metadata.task_kind``, pure-gate shape
    (:func:`is_pure_gate_metadata`) and, for an in-progress task, its
    corroboration verdict (:func:`corroboration_for_task`, task 2963) are
    forwarded to :func:`detect_live_workflow`, whose docstring owns the rules
    those inputs drive.  A detector error for one task is logged at WARNING and
    that task is treated as not live (fail-safe, matching the harness gate).

    LANDED COLUMN AND ROW INCLUSION (task 4874).  Alongside the detector
    fan-out, ONE :func:`probe_landing` call over the probed prefix asks whether
    each task's work is already on main: a fresh merge marker, or a rebased twin
    for every one of the branch's own commits.  A task gets a row iff it is live
    OR its landing is positively true, so absence keeps its meaning (no live
    signal and no landing evidence: a stranded candidate).  Every row carries
    a ``landed=`` field.  A landing probe that raises, or a ``metadata.reopen_at``
    that does not parse, yields ``landed=unknown``, which is never false and
    never earns a not-live task a row.

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
        tasks: Task dicts from the active/proactive-sample pool.  Tasks without
            an ``id`` are skipped.  Clipped to the first
            ``MAX_ACTIVE_TASKS_RENDERED`` entries — see the fan-out cap
            paragraph above.
        project_root: Absolute path to the project root, forwarded to the
            detector and the landing probe, and used to read the orchestrator
            lock + scheduler-state snapshot for the in-progress corroboration
            gate.
        now: Injectable reference time for deterministic tests.  Also the
            reference used for the claimant-heartbeat freshness check in the
            in-progress corroboration gate.
    """
    total_active = len(tasks)
    if not tasks:
        return LiveWorkflowSnapshot(rows=(), project_orchestrator_live=None)
    probed = [
        (str(task['id']), task) for task in _clip_to_cap(tasks) if task.get('id') is not None
    ]
    now_eff = now or datetime.now(UTC)
    (project_orchestrator_live, livenesses), landings = await asyncio.gather(
        _detect_liveness(probed, project_root, now, now_eff),
        _landing_verdicts(project_root, probed),
    )
    rows = (
        _row(
            task_id, liveness,
            landing=landings.get(task_id, LandingVerdict.unknown()),
            claimant=claimant_label(task, now=now_eff),
        )
        for (task_id, task), liveness in zip(probed, livenesses, strict=True)
    )
    return LiveWorkflowSnapshot(
        rows=tuple(row for row in rows if row is not None),
        project_orchestrator_live=project_orchestrator_live,
        probed=min(total_active, MAX_ACTIVE_TASKS_RENDERED),
        total_active=total_active,
    )


def _clip_to_cap(tasks: list[dict]) -> list[dict]:
    """The probed prefix; an overflow is reported at WARNING, never silently dropped."""
    total_active = len(tasks)
    if total_active <= MAX_ACTIVE_TASKS_RENDERED:
        return tasks
    omitted = total_active - MAX_ACTIVE_TASKS_RENDERED
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
    return tasks[:MAX_ACTIVE_TASKS_RENDERED]


def _metadata_of(task: Mapping) -> Mapping:
    metadata = task.get('metadata')
    return metadata if isinstance(metadata, dict) else {}


def _read_local_hoists(
    project_root: ProjectRoot,
) -> tuple[bool | None, dict | None, datetime | None]:
    """The three local-file hoists, each fail-safe to None; run in ONE thread hop.

    A None orchestrator check lets :func:`detect_live_workflow` derive it per
    task; a None scheduler snapshot or start time means that corroboration
    signal cannot fire.
    """
    try:
        orch_live: bool | None = is_orchestrator_live_for(project_root)
    except Exception:
        orch_live = None
    try:
        sched: dict | None = read_scheduler_state(Path(project_root))
    except Exception:
        sched = None
    try:
        started: datetime | None = orchestrator_started_at(project_root)
    except Exception:
        started = None
    return orch_live, sched, started


async def _detect_liveness(
    probed: Sequence[tuple[str, dict]],
    project_root: ProjectRoot,
    now: datetime | None,
    now_eff: datetime,
) -> tuple[bool | None, list[WorkflowLiveness | None]]:
    """The hoisted project-wide lock check, and each probed task's liveness (None on error)."""
    project_orch_live, scheduler_state, orch_started = await asyncio.to_thread(
        _read_local_hoists, project_root
    )
    kwargs: dict = {} if now is None else {'now': now}
    if project_orch_live is not None:
        kwargs['_orchestrator_live'] = project_orch_live
    # worktree_index_kwargs owns the whole three-valued contract: fail-safe,
    # logging, and the unknown -> omit-the-kwarg rule.
    kwargs.update(await worktree_index_kwargs(str(project_root)))

    livenesses: list[WorkflowLiveness | None] = []
    for task_id, task in probed:
        corroborated = _corroboration(task, task_id, now_eff, scheduler_state, orch_started)
        livenesses.append(await _detect_one(task_id, task, project_root, corroborated, kwargs))
    return project_orch_live, livenesses


def _corroboration(
    task: dict,
    task_id: str,
    now: datetime,
    scheduler_state: dict | None,
    orchestrator_started: datetime | None,
) -> bool | None:
    """The task-2963 verdict for an in-progress task; None (gate inert) otherwise or on error.

    Failing to None is failing TOWARD live: only an explicit False downgrades.
    """
    if task.get('status') != 'in-progress':
        return None
    try:
        return corroboration_for_task(
            task, task_id, now=now,
            scheduler_state=scheduler_state,
            orchestrator_started_at=orchestrator_started,
        )
    except Exception:
        logger.warning(
            'reconciliation.render_live_workflow_section: '
            'corroboration error for task_id=%s; leaving the gate inert',
            task_id,
        )
        return None


async def _detect_one(
    task_id: str,
    task: dict,
    project_root: ProjectRoot,
    corroborated: bool | None,
    kwargs: dict,
) -> WorkflowLiveness | None:
    metadata = _metadata_of(task)
    try:
        return await detect_live_workflow(
            task_id, project_root,
            status=task.get('status'), task_kind=metadata.get('task_kind'),
            pure_gate=is_pure_gate_metadata(metadata),
            corroborated=corroborated, **kwargs
        )
    except Exception:
        logger.warning(
            'reconciliation.render_live_workflow_section: '
            'detector error for task_id=%s; treating as not-live',
            task_id,
        )
        return None


def _landing_query(task_id: str, task: dict) -> LandingQuery | None:
    """None when ``metadata.reopen_at`` is set but unparseable, leaving landing unknown."""
    raw_reopen_at = _metadata_of(task).get('reopen_at')
    if raw_reopen_at is None:
        return LandingQuery(task_id)
    reopened_at, parsed = parse_timestamp_or_warn(
        raw_reopen_at, context=f'live_workflow_section reopen_at of task {task_id}',
    )
    return LandingQuery(task_id, reopened_at=reopened_at) if parsed else None


async def _landing_verdicts(
    project_root: ProjectRoot, probed: Sequence[tuple[str, dict]],
) -> Mapping[str, LandingVerdict]:
    """One landing probe over the probed prefix; a task missing from the result is unknown."""
    queries = [
        query for task_id, task in probed
        if (query := _landing_query(task_id, task)) is not None
    ]
    try:
        return await probe_landing(str(project_root), queries)
    except Exception:
        logger.warning(
            'reconciliation.live_workflow_landing_probe_failed: the landing probe '
            'raised for %s; every row renders %s',
            project_root, LandedToken.UNKNOWN,
            exc_info=True,
        )
        return {}


def _per_task_signals(liveness: WorkflowLiveness) -> tuple[LiveSignal, ...]:
    fired = (
        (liveness.worktree_registered, LiveSignal.WORKTREE),
        (liveness.recent_commit, LiveSignal.RECENT_COMMIT),
    )
    return tuple(signal for is_set, signal in fired if is_set)


def _row(
    task_id: str,
    liveness: WorkflowLiveness | None,
    *,
    landing: LandingVerdict,
    claimant: ClaimantLabel,
) -> LiveWorkflowRow | None:
    """A row iff the task is live or its work positively landed."""
    live = liveness if liveness is not None and liveness.is_live else None
    if live is None and landing.landed is not True:
        return None
    return LiveWorkflowRow(
        task_id=task_id,
        branch=liveness.branch if liveness is not None else f'{DEFAULT_BRANCH_PREFIX}{task_id}',
        is_live=live is not None,
        signals=_per_task_signals(live) if live is not None else (),
        claimant=claimant,
        landing=landing,
    )
