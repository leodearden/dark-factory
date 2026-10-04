"""Link-heal runs (plans/write-triage-link-healing-prd.md H1).

A plan run decides every live link, ledgers the heals it stands behind and
writes the plan document: a deterministic rendering of every pending heal,
whose sha256 an operator can approve. It writes nothing to the store.

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

from fused_memory.maintenance.link_heal import LinkBasis, PlannedAction, RunCounts, build_plan
from fused_memory.maintenance.link_heal_ledger import ActionRow, LinkHealLedger, RunSource
from fused_memory.maintenance.link_heal_store import LinkCensus, LinkHealStore

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
