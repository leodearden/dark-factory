"""Link-heal runs (plans/write-triage-link-healing-prd.md H1).

A plan run decides every live link, ledgers the heals it stands behind and
writes the plan document: a deterministic rendering of every pending heal,
whose sha256 an operator can approve. It writes nothing to the store.

An apply run drains the pending heals oldest first. Each heal re-reads its
record live, and is skipped when the record no longer shows what it was
planned against; otherwise it sends exactly one write, then re-reads the
record to verify it. An unattended run makes at most ``max_actions_per_run``
write attempts and escalates a backlog beyond its multiple; an operator who
approves the exact pending plan by sha lifts both. Any run stops, and
escalates, after ``write_failure_streak`` consecutive failed heals.

The escape anchors below belong to these filers alone; no other filer may
share them.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from fused_memory.middleware._folded_escalation import file_folded_escalation

if TYPE_CHECKING:
    from fused_memory.config.schema import LinkHealConfig

logger = logging.getLogger(__name__)

BACKLOG_ANCHOR = 'link-heal-backlog'
WRITE_FAILURE_ANCHOR = 'link-heal-write-failure'

PLAN_FORMAT = 'link-heal-plan/1'

CAP_BIT = 'max_actions_per_run'
STREAK_STOP = 'write_failure_streak'


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


def write_failure_escape(
    run_id: str, *, streak: int, not_attempted: int, last_failure: Mapping[str, Any] | None,
    projects: Sequence[str],
) -> Escape:
    """The escape a run files when it stops on a failure streak."""
    return Escape(
        anchor=WRITE_FAILURE_ANCHOR,
        summary=f'link-heal run {run_id[:8]} stopped after {streak} consecutive failed heals',
        detail={
            'run_id': run_id,
            'consecutive_failures': streak,
            'not_attempted': not_attempted,
            'last_failure': dict(last_failure or {}),
            'projects': list(projects),
        },
    )


def render_escape_detail(detail: Mapping[str, Any]) -> str:
    """Sorted ``key: value`` lines; a non-string value renders as JSON."""
    return '\n'.join(
        f'{key}: {value if isinstance(value, str) else json.dumps(value, sort_keys=True)}'
        for key, value in sorted(detail.items())
    )


@dataclass(frozen=True)
class _EscapeRoute:
    category: str
    suggested_action: str


_ESCAPE_ROUTES: Mapping[str, _EscapeRoute] = MappingProxyType({
    BACKLOG_ANCHOR: _EscapeRoute(
        category='risk_identified',
        suggested_action=(
            'Run `fused-memory/scripts/link_heal.py status` and review the plan document; '
            'apply a reviewed plan with `apply --approved-plan-sha`, or raise '
            'link_heal.max_actions_per_run.'
        ),
    ),
    WRITE_FAILURE_ANCHOR: _EscapeRoute(
        category='infra_issue',
        suggested_action=(
            'Read the failed actions of the named run in link_heal.db (their detail '
            'carries the failure and error_type). Check that the server is healthy and '
            'that mem0_update.metadata_patch_allowed_agent_prefixes admits link-heal-, '
            'then re-run apply.'
        ),
    ),
})


class FoldedEscapeFiler:
    """Files an :class:`Escape` into *project_root*'s queue, folded under its anchor.

    Every project's escapes land in this one queue, naming the project in the detail.
    """

    AGENT_ROLE = 'fused-memory/link-heal'

    def __init__(self, project_root: str | None, log: logging.Logger = logger) -> None:
        self._project_root = project_root
        self._log = log

    def __call__(self, escape: Escape) -> str | None:
        route = _ESCAPE_ROUTES[escape.anchor]
        return file_folded_escalation(
            self._project_root,
            anchor_task_id=escape.anchor,
            agent_role=self.AGENT_ROLE,
            category=route.category,
            severity='blocking',
            summary=escape.summary,
            detail=render_escape_detail(escape.detail),
            suggested_action=route.suggested_action,
            logger=self._log,
            log_label='link_heal',
            level=1,
        )


def _file(filer: EscapeFiler, escape: Escape | None) -> tuple[Mapping[str, Any], ...]:
    """File *escape*, if any, and the ``escaped`` entry that discloses it."""
    if escape is None:
        return ()
    return ({'anchor': escape.anchor, 'escalation_id': filer(escape)},)


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
    """One heal's ledger state; ``wrote`` says whether a write was attempted."""

    state: ActionState
    detail: Mapping[str, Any] | None = None
    wrote: bool = False


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
        detail = {'failure': 'write', 'error_type': reply.error_type, 'error': reply.error}
        return _Outcome(ActionState.FAILED, detail, wrote=True)
    try:
        mismatch = await _verify(store, project_id, child_id, after)
    except StoreReadFailed as failure:
        return _Outcome(ActionState.FAILED, _read_failure('verify', failure), wrote=True)
    if mismatch is not None:
        detail = {'failure': 'verify', 'field': mismatch.value}
        return _Outcome(ActionState.FAILED, detail, wrote=True)
    return _Outcome(ActionState.APPLIED, wrote=True)


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


class ApprovalMismatch(ValueError):
    """The approved plan sha is not the sha of the plan pending now."""

    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f'approved plan sha256 {expected} is not the pending plan\'s sha256 {actual}; '
            'the pending heals changed since that plan was reviewed — re-plan and review it',
        )


def _check_approval(approved: str, pending: Sequence[ActionRow]) -> None:
    actual = plan_sha256(render_plan_document(pending))
    if actual != approved:
        raise ApprovalMismatch(expected=approved, actual=actual)


@dataclass
class _Drain:
    """What one apply run's drain has come to so far."""

    outcomes: Counter[ActionState] = field(default_factory=Counter)
    writes: int = 0
    streak: int = 0
    last_failure: Mapping[str, Any] | None = None
    not_attempted: int = 0
    stopped_by: str | None = None

    def record(self, outcome: _Outcome) -> None:
        """A failure extends the streak, an applied heal resets it, a skip leaves it."""
        self.outcomes[outcome.state] += 1
        if outcome.wrote:
            self.writes += 1
        if outcome.state is ActionState.FAILED:
            self.streak += 1
            self.last_failure = outcome.detail
        elif outcome.state is ActionState.APPLIED:
            self.streak = 0


async def _drain(
    pending: Sequence[ActionRow],
    *,
    store: LinkHealStore,
    ledger: LinkHealLedger,
    run_id: str,
    cap: int | None,
    streak_limit: int,
) -> _Drain:
    """Heal *pending* in order until *cap* write attempts or *streak_limit* failures."""
    drain = _Drain()
    for index, row in enumerate(pending):
        if cap is not None and drain.writes >= cap:
            outcome = _Outcome(ActionState.SKIPPED_CAP)
        else:
            outcome = await _heal_one(row.planned, store=store, run_id=run_id)
        ledger.set_outcome(row.action_id, outcome.state, run_id, outcome.detail)
        drain.record(outcome)
        if drain.streak >= streak_limit:
            drain.stopped_by = STREAK_STOP
            drain.not_attempted = len(pending) - index - 1
            break
    return drain


def _projects(pending: Sequence[ActionRow]) -> list[str]:
    return list(dict.fromkeys(row.planned.project_id for row in pending))


async def run_apply(
    *,
    store: LinkHealStore,
    ledger: LinkHealLedger,
    limits: RunLimits,
    filer: EscapeFiler,
    source: RunSource,
    approved_plan_sha256: str | None = None,
) -> RunReport:
    """A writing run: heal the pending rows of *source*, oldest first.

    Raises :class:`ApprovalMismatch` before any run row when an approved sha
    is not the pending plan's.
    """
    pending = ledger.pending_actions(source)
    attended = approved_plan_sha256 is not None
    if approved_plan_sha256 is not None:
        _check_approval(approved_plan_sha256, pending)
    run_id = ledger.start_run(source, writes=True)
    projects = _projects(pending)
    escaped = () if attended else _file(
        filer, backlog_escape(len(pending), limits, projects=projects),
    )
    drain = await _drain(
        pending, store=store, ledger=ledger, run_id=run_id,
        cap=None if attended else limits.max_actions_per_run,
        streak_limit=limits.write_failure_streak,
    )
    if drain.stopped_by is not None:
        escaped += _file(filer, write_failure_escape(
            run_id, streak=drain.streak, not_attempted=drain.not_attempted,
            last_failure=drain.last_failure, projects=projects,
        ))
    counts = _apply_counts(pending, drain, escaped, approved_plan_sha256)
    ledger.finish_run(run_id, counts=counts.as_json())
    return RunReport(run_id=run_id, counts=counts)


def _apply_counts(
    pending: Sequence[ActionRow],
    drain: _Drain,
    escaped: tuple[Mapping[str, Any], ...],
    approved_plan_sha256: str | None,
) -> RunCounts:
    skipped_cap = drain.outcomes[ActionState.SKIPPED_CAP]
    return RunCounts(
        planned=len(pending),
        planned_by_action=Counter(row.planned.action.value for row in pending),
        applied=drain.outcomes[ActionState.APPLIED],
        skipped_stale=drain.outcomes[ActionState.SKIPPED_STALE],
        skipped_cap=skipped_cap,
        failed=drain.outcomes[ActionState.FAILED],
        not_attempted=drain.not_attempted,
        escaped=escaped,
        caps_bit=(CAP_BIT,) if skipped_cap else (),
        stopped_by=drain.stopped_by,
        approved_plan_sha256=approved_plan_sha256,
    )
