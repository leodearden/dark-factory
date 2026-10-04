"""Link-heal runs (plans/write-triage-link-healing-prd.md H1).

A plan run decides every live link, ledgers the heals it stands behind and
writes the plan document: a deterministic rendering of every pending heal,
whose sha256 an operator can approve. It writes nothing to the store.

An apply run drains the pending heals oldest first. Each heal re-reads its
record live, and is skipped when the record no longer shows what it was
planned against; otherwise it sends exactly one write, then re-reads the
record to verify it.

The escape anchors below belong to these filers alone; no other filer may
share them.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from fused_memory.maintenance.link_heal import (
    COMPLETION_ACTIONS,
    LinkBasis,
    LinkImage,
    PlannedAction,
    RunCounts,
    StaleField,
    build_plan,
    corroboration_mismatch,
    journal_reason,
    verification_mismatch,
)
from fused_memory.maintenance.link_heal_ledger import (
    ActionRow,
    ActionState,
    LinkHealLedger,
    RunSource,
)
from fused_memory.maintenance.link_heal_store import (
    LinkCensus,
    LinkHealStore,
    MetadataChange,
    StoreReadFailed,
)

if TYPE_CHECKING:
    from fused_memory.config.schema import LinkHealConfig

BACKLOG_ANCHOR = 'link-heal-backlog'
WRITE_FAILURE_ANCHOR = 'link-heal-write-failure'

PLAN_FORMAT = 'link-heal-plan/1'


@dataclass(frozen=True)
class RunLimits:
    max_actions_per_run: int
    backlog_multiplier: int
    write_failure_streak: int

    @classmethod
    def from_config(cls, config: LinkHealConfig) -> RunLimits:
        return cls(
            max_actions_per_run=config.max_actions_per_run,
            backlog_multiplier=config.backlog_multiplier,
            write_failure_streak=config.write_failure_streak,
        )


@dataclass(frozen=True)
class Escape:
    """One condition a run escalates rather than absorbs."""

    anchor: str
    summary: str
    detail: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'detail', MappingProxyType(dict(self.detail)))


#: Files an :class:`Escape`; returns the escalation id, or ``None`` when none was filed.
EscapeFiler = Callable[[Escape], str | None]


def backlog_escape(
    pending_count: int, limits: RunLimits, *, projects: Sequence[str],
) -> Escape | None:
    """The backlog escape, when more heals are pending than the cap drains in its multiple."""
    threshold = limits.max_actions_per_run * limits.backlog_multiplier
    if pending_count <= threshold:
        return None
    return Escape(
        anchor=BACKLOG_ANCHOR,
        summary=(
            f'link-heal backlog: {pending_count} pending heals exceed {threshold} '
            f'(max_actions_per_run x backlog_multiplier)'
        ),
        detail={
            'pending': pending_count,
            'threshold': threshold,
            'max_actions_per_run': limits.max_actions_per_run,
            'backlog_multiplier': limits.backlog_multiplier,
            'projects': list(projects),
        },
    )


@dataclass(frozen=True)
class RunReport:
    run_id: str
    counts: RunCounts
    plan_sha256: str | None = None
    plan_path: Path | None = None


def render_plan_document(rows: Iterable[ActionRow]) -> bytes:
    """The canonical plan document for *rows*: the same rows always render the same bytes."""
    document = {'format': PLAN_FORMAT, 'actions': [_plan_entry(row) for row in rows]}
    return (json.dumps(document, sort_keys=True, indent=2) + '\n').encode('utf-8')


def plan_sha256(document: bytes) -> str:
    return hashlib.sha256(document).hexdigest()


def _plan_entry(row: ActionRow) -> dict[str, Any]:
    heal = row.planned
    return {
        'action_id': row.action_id,
        'project_id': heal.project_id,
        'child_id': heal.child_id,
        'action': heal.action.value,
        'pre_image': heal.pre_image.as_dict(),
        'post_image': heal.post_image.as_dict(),
        'basis_source': heal.basis_source.value,
        'basis_key': heal.basis_key,
        'child_sha256': heal.child_sha256,
        'parent_sha256': heal.parent_sha256,
    }


@dataclass(frozen=True)
class _StagedHeals:
    """A plan's heals sorted against the ledger."""

    standing: tuple[PlannedAction, ...]
    new: tuple[PlannedAction, ...]
    already_pending: int
    undo_suppressed: int


def _stage(actions: Iterable[PlannedAction], ledger: LinkHealLedger) -> _StagedHeals:
    """Drop the heals an undo took back, and find those already pending."""
    standing: list[PlannedAction] = []
    new: list[PlannedAction] = []
    suppressed = 0
    for action in actions:
        if ledger.is_undo_suppressed(action):
            suppressed += 1
            continue
        standing.append(action)
        if ledger.find_pending(action) is None:
            new.append(action)
    return _StagedHeals(
        standing=tuple(standing),
        new=tuple(new),
        already_pending=len(standing) - len(new),
        undo_suppressed=suppressed,
    )


async def run_plan(
    bases: Iterable[LinkBasis],
    *,
    store: LinkHealStore,
    census: LinkCensus,
    ledger: LinkHealLedger,
    limits: RunLimits,
    projects: Sequence[str],
    source: RunSource,
    plan_path: Path,
) -> RunReport:
    """A non-writing run: ledger the plan, write its document, report what would escape."""
    run_id = ledger.start_run(source, writes=False)
    plan = await build_plan(bases, store=store, census=census, projects=projects)
    staged = _stage(plan.actions, ledger)
    ledger.add_planned(run_id, staged.new)
    pending = ledger.pending_actions(source)
    document = render_plan_document(pending)
    plan_path.write_bytes(document)
    sha = plan_sha256(document)
    escape = backlog_escape(len(pending), limits, projects=projects)
    counts = replace(
        plan.counts,
        planned=len(staged.standing),
        planned_by_action=Counter(action.action.value for action in staged.standing),
        already_pending=staged.already_pending,
        undo_suppressed=staged.undo_suppressed,
        would_escape=() if escape is None else (escape.anchor,),
    )
    ledger.finish_run(run_id, counts=counts.as_json(), plan_sha256=sha)
    return RunReport(run_id=run_id, counts=counts, plan_sha256=sha, plan_path=plan_path)


def link_heal_agent_id(run_id: str) -> str:
    """The ``agent_id`` every write of run *run_id* carries."""
    return f'link-heal-{run_id[:8]}'


@dataclass(frozen=True)
class _Outcome:
    state: ActionState
    detail: Mapping[str, Any] | None = None


def _read_failure(phase: str, failure: StoreReadFailed) -> dict[str, Any]:
    return {
        'failure': phase,
        'tool': failure.tool,
        'memory_id': failure.memory_id,
        'error_type': failure.error_type,
        'error': failure.detail,
    }


async def _corroborate(store: LinkHealStore, heal: PlannedAction) -> StaleField | None:
    """Re-read the child, its parent and (for a completion) its children, live."""
    child = await store.read(heal.project_id, heal.child_id)
    parent_id = heal.pre_image.parent_id
    parent = None if parent_id is None else await store.read(heal.project_id, parent_id)
    child_count = (
        await store.count_children(heal.project_id, heal.child_id)
        if heal.action in COMPLETION_ACTIONS
        else None
    )
    return corroboration_mismatch(heal, child, parent, child_count)


async def _verify(
    store: LinkHealStore, project_id: str, child_id: str, image: LinkImage,
) -> StaleField | None:
    return verification_mismatch(image, await store.read(project_id, child_id))


async def _write_and_verify(
    store: LinkHealStore,
    project_id: str,
    child_id: str,
    change: MetadataChange,
    after: LinkImage,
    *,
    run_id: str,
    reason: str,
) -> _Outcome:
    """Send *change* as one write, then confirm the record shows *after*."""
    reply = await store.write(
        project_id, child_id, change,
        agent_id=link_heal_agent_id(run_id), causation_id=run_id, reason=reason,
    )
    if not reply.landed:
        return _Outcome(
            ActionState.FAILED,
            {'failure': 'write', 'error_type': reply.error_type, 'error': reply.error},
        )
    try:
        mismatch = await _verify(store, project_id, child_id, after)
    except StoreReadFailed as failure:
        return _Outcome(ActionState.FAILED, _read_failure('verify', failure))
    if mismatch is not None:
        return _Outcome(ActionState.FAILED, {'failure': 'verify', 'field': mismatch.value})
    return _Outcome(ActionState.APPLIED)


async def _heal_one(heal: PlannedAction, *, store: LinkHealStore, run_id: str) -> _Outcome:
    try:
        stale = await _corroborate(store, heal)
    except StoreReadFailed as failure:
        return _Outcome(ActionState.FAILED, _read_failure('read', failure))
    if stale is not None:
        return _Outcome(ActionState.SKIPPED_STALE, {'stale_field': stale.value})
    return await _write_and_verify(
        store, heal.project_id, heal.child_id, heal.change, heal.post_image,
        run_id=run_id,
        reason=journal_reason(run_id[:8], heal.action, heal.pre_image),
    )


async def run_apply(
    *,
    store: LinkHealStore,
    ledger: LinkHealLedger,
    limits: RunLimits,
    filer: EscapeFiler,
    source: RunSource,
    approved_plan_sha256: str | None = None,
) -> RunReport:
    """A writing run: heal every pending row of *source*, oldest first."""
    pending = ledger.pending_actions(source)
    run_id = ledger.start_run(source, writes=True)
    outcomes: Counter[ActionState] = Counter()
    for row in pending:
        outcome = await _heal_one(row.planned, store=store, run_id=run_id)
        ledger.set_outcome(row.action_id, outcome.state, run_id, outcome.detail)
        outcomes[outcome.state] += 1
    counts = RunCounts(
        planned=len(pending),
        planned_by_action=Counter(row.planned.action.value for row in pending),
        applied=outcomes[ActionState.APPLIED],
        skipped_stale=outcomes[ActionState.SKIPPED_STALE],
        failed=outcomes[ActionState.FAILED],
    )
    ledger.finish_run(run_id, counts=counts.as_json())
    return RunReport(run_id=run_id, counts=counts)
