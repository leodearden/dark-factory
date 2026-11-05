"""The per-PRD view of one root: segments, readiness, attribution and signal, composed.

PURE (PRD ``plans/dashboard-prd-lanes-prd.md`` §4.3–§4.5, §4.7, §4.9–§4.10,
§5.5): no I/O and no clock read but the injected *now*. It reads one root's
six inputs, handed in by the caller: its status map, its raw active rows, its
lineage index, its pending human-level escalations, its run-ledger activity,
its PRD documents and its session holders' liveness. It never imports the
modules that acquire them; the value types those modules serve are defined
here. Its one piece of state is a per-root memo of the attribution, which
does not depend on *now*; it is keyed on the very rows and lineage values it
was computed from, so it can only ever answer for those.

:func:`build_prd_view` serves per-PRD segments, totals, open members, ready
age, title and signal mark; closed PRDs; unattributed done; dark matter; the
lineage multi-link count; signal coverage; each input's provenance; and the
rule text of every figure. :func:`open_work_keys` names the PRDs it draws, so
the caller knows which documents to read.
"""

from __future__ import annotations

import enum
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, TypeVar

from shared.capability_manifest import CapabilityManifestDoc, ManifestTask
from shared.process_identity import Liveness
from shared.task_metadata import Milestone
from shared.task_statuses import TERMINAL, TaskStatus
from shared.timestamps import parse_timestamp_or_warn

from dashboard.data.census import VIEWS, CensusVocabularyError, TaskCensus, TaskView, build_census
from dashboard.data.datum import (
    Datum,
    DatumContractError,
    DatumInvariant,
    DatumState,
    aged_at,
    unknown_datum,
)
from dashboard.data.prd_attribution import (
    MAX_HOPS,
    Attribution,
    LineageEntry,
    LinkField,
    PrdKey,
    Unresolved,
    UnresolvedReason,
    attribute,
    lineage_entry,
)


class Segment(enum.StrEnum):
    """Where a task sits in its PRD's bar; declaration order is display order."""

    DONE = 'done'
    IN_PROGRESS = 'in_progress'
    BLOCKED = 'blocked'
    DEFERRED = 'deferred'
    READY = 'ready'
    WAITING = 'waiting'


SEGMENT_TONES: MappingProxyType[Segment, str] = MappingProxyType({
    Segment.DONE: 'ok',
    Segment.IN_PROGRESS: 'accent',
    Segment.BLOCKED: 'bad',
    Segment.DEFERRED: 'fg3',
    Segment.READY: 'warn',
    Segment.WAITING: 'info',
})

SEGMENT_LABELS: MappingProxyType[Segment, str] = MappingProxyType({
    Segment.DONE: 'done',
    Segment.IN_PROGRESS: 'in progress',
    Segment.BLOCKED: 'blocked',
    Segment.DEFERRED: 'deferred',
    Segment.READY: 'ready',
    Segment.WAITING: 'waiting',
})


class FigureId(enum.StrEnum):
    """Every figure the view serves; each value IS the figure's wire field name."""

    SEGMENTS = 'segments'
    TOTAL = 'total'
    DIRECT_OPEN = 'direct_open_ids'
    FOLLOWUP_OPEN = 'followup_open'
    READY = 'ready'
    CLOSED = 'closed'
    UNATTRIBUTED_DONE = 'unattributed_done'
    DARK_MATTER = 'dark_matter'
    MULTI_LINK = 'multi_link'
    TITLE = 'title'
    SIGNAL = 'signal'
    SIGNAL_COVERAGE = 'signal_coverage'


class ViewInput(enum.StrEnum):
    """The inputs a view is composed from, named as its ``inputs`` list names them."""

    STATUS_MAP = 'status_map'
    ACTIVE_ROWS = 'active_rows'
    LINEAGE = 'lineage'
    ESCALATIONS = 'escalations'
    ACTIVITY = 'activity'
    SIGNALS = 'signals'
    LIVENESS = 'liveness'
    SCHEDULER_STATE = 'scheduler_state'


HIGH_PRIORITIES = frozenset({'critical', 'high'})
TOP_SOURCES = 5
MOVING_WINDOW_DAYS = 14
"""How recent a member's start or completion must be for its PRD to be moving."""
UNRECORDED = 'unrecorded'
OTHER_SOURCES = 'other'

_LINK_CHAIN = ' > '.join(field.value for field in LinkField)
_ATTRIBUTION_RULE = (
    "A task belongs to the PRD its own metadata names (prd_path, else prd, else prd_ref); "
    f"failing that, to the first PRD up its {_LINK_CHAIN} chain within this project, "
    f"at most {MAX_HOPS} links; cross-project links are not followed."
)

FIGURE_RULES: MappingProxyType[FigureId, str] = MappingProxyType({
    FigureId.SEGMENTS: (
        "A PRD's members are the tasks attributed to it, cancelled excluded. Each falls in one "
        "segment: done; in progress (in-progress, review, merge-deferred, infra-hold); blocked; "
        "deferred; ready (pending and could dispatch, see ready); waiting (pending and not "
        "ready: an unmet or unknown dependency, an external dependency, an unfired milestone, "
        "a human gate, or no active row)."
    ),
    FigureId.TOTAL: "The PRD's members, cancelled excluded: the sum of its segments.",
    FigureId.DIRECT_OPEN: f"Open members whose own metadata names this PRD. {_ATTRIBUTION_RULE}",
    FigureId.FOLLOWUP_OPEN: (
        f"Open members reaching this PRD through lineage, with the links followed. "
        f"{_ATTRIBUTION_RULE}"
    ),
    FigureId.READY: (
        "Pending members that could dispatch: every dependency known and done or cancelled, no "
        "external dependency, no unfired milestone, and not a human gate (a deterministic task "
        "that always escalates). It does not evaluate delivered checks on dependencies, pins, "
        "parks, requeue cool-downs or external statuses, so it is an upper bound on what the "
        "scheduler dispatches. A pending task with no active row counts waiting. The age is the "
        "oldest ready member's time pending, not time ready, in whole days from "
        "metadata.pending_since; it is a lower bound (shown ≥) when any ready member's anchor "
        "is backfilled or missing."
    ),
    FigureId.CLOSED: (
        "PRDs whose attributed members are all done (cancelled excluded), and how many members "
        "they hold; a closed PRD draws no row."
    ),
    FigureId.UNATTRIBUTED_DONE: (
        "Done tasks no PRD owns: not drawn; closes the census sum (row totals + closed tasks + "
        "unattributed done + dark matter = every task not cancelled)."
    ),
    FigureId.DARK_MATTER: (
        "Open tasks no PRD owns, grouped by why: no link, a lineage root without a PRD, a "
        "cross-project link (not followed), an unparseable link, a missing link target, a "
        f"cycle, a chain past {MAX_HOPS} links, a PRD outside this project, or lineage not yet "
        "read. Counted by status, by priority and by metadata.source (the top five sources "
        "plus other); a missing priority or source is unrecorded. ready_high lists the ready "
        "ones at high or critical priority."
    ),
    FigureId.MULTI_LINK: (
        f"Tasks carrying two or more lineage link keys; the first present in {_LINK_CHAIN} "
        "order is the one followed."
    ),
    FigureId.TITLE: (
        "The first line of the PRD file starting '# ', the marker stripped; the PRD key when "
        "the file is missing or has no such line."
    ),
    FigureId.SIGNAL: (
        "Whether the PRD's goal signal is delivered, read from the signal_leaves its "
        "capability-manifest sidecar names and those tasks' status in the store: delivered when "
        "every leaf is done (each leaf's done kind shown), partial when some are done and none "
        "is cancelled, not yet when none is done and none is cancelled. Unknown, with its "
        "cause, when there is no sidecar or it fails to load, the PRD names no signal leaf, or "
        "a leaf is unbound, bound to a task absent from the store, bound to a task whose own "
        "PRD is another (a stale binding), bound to an external task, or cancelled. Leaves are "
        "checked in signal_leaves order and the first failing one names the cause."
    ),
    FigureId.SIGNAL_COVERAGE: (
        "How many drawn PRDs have a sidecar that loads and names a signal leaf, of all drawn "
        "PRDs. A mark unknown everywhere beside low coverage is a missing producer, not a "
        "signal not yet delivered."
    ),
})

_IN_PROGRESS_STATUSES = VIEWS[TaskView.IN_FLIGHT] - {TaskStatus.BLOCKED}
_STATUS_SEGMENTS: Mapping[str, Segment] = MappingProxyType({
    TaskStatus.DONE: Segment.DONE,
    TaskStatus.BLOCKED: Segment.BLOCKED,
    TaskStatus.DEFERRED: Segment.DEFERRED,
})
_SATISFIED_DEPENDENCY_STATUSES = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED})
_MILESTONE_ANCHOR_KEY = 'milestone_deps_satisfied_at'
_PENDING_SINCE_KEY = 'pending_since'
_PENDING_SINCE_BACKFILLED_KEY = 'pending_since_backfilled'
_SECONDS_PER_DAY = 86_400
_EMPTY_MAP_REASON = 'the status map measured no tasks'
_LINEAGE_PENDING = Attribution(prd=None, hops=0, unresolved=UnresolvedReason.LINEAGE_PENDING)
_STATE_SEVERITY: Mapping[DatumState, int] = MappingProxyType({
    DatumState.FRESH: 0,
    DatumState.LOWER_BOUND: 1,
    DatumState.STALE: 2,
    DatumState.UNKNOWN: 3,
})
_STRUCTURAL_VIEW_KEYS = frozenset({'prds', 'inputs', 'rules', 'segment_legend'})
_STRUCTURAL_ROW_KEYS = frozenset({'key'})
_LISTED_IDS_CAP = 5

V = TypeVar('V')


def is_human_gate(metadata: Mapping[str, Any]) -> bool:
    """A deterministic task that always escalates: it waits on a human, never a dispatch."""
    return metadata.get('task_kind') == 'deterministic' and metadata.get('always_escalates') is True


def is_ready(
    task_id: int,
    row: Mapping[str, Any] | None,
    status_map: Mapping[int, str],
    *,
    now: datetime,
) -> bool:
    """Whether pending *task_id* could be dispatched at *now*, as far as its row says.

    Ready means pending with an active row, every dependency known and done or
    cancelled, no external dependency, no unfired milestone, and not a human
    gate. It does not evaluate delivered checks on dependencies, pins, parks,
    requeue cool-downs or external statuses, so it is an upper bound on what
    the scheduler would dispatch.
    """
    if status_map.get(task_id) != TaskStatus.PENDING or row is None:
        return False
    metadata = _metadata_of(row)
    dependencies = row.get('dependencies') or ()
    external_deps = metadata.get('external_deps')
    return (
        all(status_map.get(dep) in _SATISFIED_DEPENDENCY_STATUSES for dep in dependencies)
        and not (isinstance(external_deps, list) and external_deps)
        and not _milestone_unfired(metadata, now)
        and not is_human_gate(metadata)
    )


def segment_of(
    task_id: int,
    status: str,
    row: Mapping[str, Any] | None,
    status_map: Mapping[int, str],
    *,
    now: datetime,
) -> Segment | None:
    """*task_id*'s segment; ``None`` for a cancelled task, which no bar counts."""
    if status == TaskStatus.PENDING:
        return Segment.READY if is_ready(task_id, row, status_map, now=now) else Segment.WAITING
    if status in _IN_PROGRESS_STATUSES:
        return Segment.IN_PROGRESS
    return _STATUS_SEGMENTS.get(status)


def _metadata_of(row: Mapping[str, Any] | None) -> Mapping[str, Any]:
    metadata = None if row is None else row.get('metadata')
    return metadata if isinstance(metadata, Mapping) else MappingProxyType({})


def _milestone_unfired(metadata: Mapping[str, Any], now: datetime) -> bool:
    """A milestone that has not fired, where a malformed one never fires."""
    spec = metadata.get('milestone')
    if not spec:
        return False
    anchor = metadata.get(_MILESTONE_ANCHOR_KEY)
    if anchor is not None and not isinstance(anchor, str):
        return True
    try:
        return not Milestone.model_validate(spec).fired(anchor, now)
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# The view's value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReadyAge:
    """How many members are ready, and how long the oldest of them has been pending."""

    count: int
    oldest_task_id: int | None
    oldest_pending_days: int | None
    oldest_is_lower_bound: bool

    def to_wire(self) -> dict[str, object]:
        return {
            'count': self.count,
            'oldest_task_id': self.oldest_task_id,
            'oldest_pending_days': self.oldest_pending_days,
            'oldest_is_lower_bound': self.oldest_is_lower_bound,
        }


@dataclass(frozen=True, slots=True)
class Followup:
    """An open member reaching its PRD through lineage, ``hops`` links on."""

    task_id: int
    hops: int

    def to_wire(self) -> dict[str, object]:
        return {'task_id': self.task_id, 'hops': self.hops}


@dataclass(frozen=True, slots=True)
class PrdRow:
    """One PRD with at least one open member."""

    key: str
    title: str
    segments: Mapping[Segment, int]
    total: int
    direct_open_ids: tuple[int, ...]
    followup_open: tuple[Followup, ...]
    ready: ReadyAge
    signal: Signal

    def to_wire(self) -> dict[str, object]:
        return {
            'key': self.key,
            FigureId.TITLE.value: self.title,
            FigureId.SIGNAL.value: self.signal.to_wire(),
            FigureId.SEGMENTS.value: {segment.value: n for segment, n in self.segments.items()},
            FigureId.TOTAL.value: self.total,
            FigureId.DIRECT_OPEN.value: list(self.direct_open_ids),
            FigureId.FOLLOWUP_OPEN.value: [followup.to_wire() for followup in self.followup_open],
            FigureId.READY.value: self.ready.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class Closed:
    """PRDs with no open member, and how many members they hold."""

    prds: int
    tasks: int

    def to_wire(self) -> dict[str, object]:
        return {'prds': self.prds, 'tasks': self.tasks}


@dataclass(frozen=True, slots=True)
class DarkMatter:
    """The open tasks no PRD owns, grouped by why."""

    total: int
    by_status: Mapping[TaskStatus, int]
    by_priority: Mapping[str, int]
    by_source: Mapping[str, int]
    by_reason: Mapping[UnresolvedReason, int]
    ready_high_ids: tuple[int, ...]
    members: Mapping[UnresolvedReason, tuple[int, ...]]

    def to_wire(self) -> dict[str, object]:
        return {
            'total': self.total,
            'by_status': {status.value: n for status, n in self.by_status.items()},
            'by_priority': dict(self.by_priority),
            'by_source': dict(self.by_source),
            'by_reason': {reason.value: n for reason, n in self.by_reason.items()},
            'ready_high_ids': list(self.ready_high_ids),
            'members': {reason.value: list(ids) for reason, ids in self.members.items()},
        }


@dataclass(frozen=True, slots=True)
class InputState:
    """One input's provenance as the view judged it at its serving instant."""

    name: ViewInput
    as_of: datetime | None
    state: DatumState
    reason: str | None

    def to_wire(self) -> dict[str, object]:
        return {
            'name': self.name.value,
            'as_of': None if self.as_of is None else self.as_of.astimezone(UTC).isoformat(),
            'state': self.state.value,
            'reason': self.reason,
        }


@dataclass(frozen=True, slots=True)
class PrdView:
    """One root's per-PRD view (the γ1 subset of PRD §5.5's ``PrdView``)."""

    prds: tuple[PrdRow, ...]
    closed: Closed
    unattributed_done: int
    dark_matter: DarkMatter
    multi_link: int
    signal_coverage: SignalCoverage
    inputs: tuple[InputState, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            'prds': [row.to_wire() for row in self.prds],
            FigureId.CLOSED.value: self.closed.to_wire(),
            FigureId.UNATTRIBUTED_DONE.value: self.unattributed_done,
            FigureId.DARK_MATTER.value: self.dark_matter.to_wire(),
            FigureId.MULTI_LINK.value: self.multi_link,
            FigureId.SIGNAL_COVERAGE.value: self.signal_coverage.to_wire(),
            'inputs': [state.to_wire() for state in self.inputs],
            'rules': {figure.value: text for figure, text in FIGURE_RULES.items()},
            'segment_legend': [
                {'segment': segment.value, 'label': SEGMENT_LABELS[segment],
                 'tone': SEGMENT_TONES[segment]}
                for segment in Segment
            ],
        }


@dataclass(frozen=True, slots=True)
class TaskActivity:
    """One project's task activity, as its run ledger records it.

    ``started`` holds each task's latest start inside the moving window;
    ``completed`` each task's latest done completion, ever; ``ever_started``
    every task that has started at all; ``first_event_at`` the ledger's
    earliest event, ``None`` for an empty ledger.
    """

    started: Mapping[int, datetime]
    completed: Mapping[int, datetime]
    ever_started: frozenset[int]
    first_event_at: datetime | None


class SignalMark(enum.StrEnum):
    """Whether a PRD's goal signal is delivered, read from its signal leaves' status (§4.7)."""

    DELIVERED = 'delivered'
    PARTIAL = 'partial'
    NOT_YET = 'not_yet'
    UNKNOWN = 'unknown'


class SignalCause(enum.StrEnum):
    """Why a PRD's signal mark is unknown: the closed set of §4.7."""

    NO_SIDECAR = 'no_sidecar'
    SIDECAR_INVALID = 'sidecar_invalid'
    NAMES_NO_LEAF = 'names_no_leaf'
    UNBOUND_LABEL = 'unbound_label'
    LEAF_NOT_IN_STORE = 'leaf_not_in_store'
    STALE_BINDING = 'stale_binding'
    EXTERNAL_LEAF = 'external_leaf'
    LEAF_CANCELLED = 'leaf_cancelled'


SIGNAL_CAUSE_TEXT: MappingProxyType[SignalCause, str] = MappingProxyType({
    SignalCause.NO_SIDECAR: 'no capability-manifest sidecar',
    SignalCause.SIDECAR_INVALID: 'sidecar fails to load',
    SignalCause.NAMES_NO_LEAF: 'PRD names no signal leaf',
    SignalCause.UNBOUND_LABEL: 'signal leaf bound to no task',
    SignalCause.LEAF_NOT_IN_STORE: 'signal leaf task absent from the store',
    SignalCause.STALE_BINDING: "signal leaf task's PRD is not this PRD",
    SignalCause.EXTERNAL_LEAF: 'signal leaf is an external task',
    SignalCause.LEAF_CANCELLED: 'signal leaf cancelled',
})

_UNREAD_SIDECAR_CAUSES = frozenset({SignalCause.NO_SIDECAR, SignalCause.SIDECAR_INVALID})


@dataclass(frozen=True, slots=True)
class SidecarUnread:
    """A PRD's sidecar that was not read: absent, or failing to load with ``detail`` as its error."""

    cause: SignalCause
    detail: str | None

    def __post_init__(self) -> None:
        if self.cause not in _UNREAD_SIDECAR_CAUSES:
            allowed = ', '.join(sorted(_UNREAD_SIDECAR_CAUSES))
            raise ValueError(f'SidecarUnread.cause must be one of {allowed}; got {self.cause!s}')
        if self.cause is SignalCause.SIDECAR_INVALID and not self.detail:
            raise ValueError('SidecarUnread: a sidecar that fails to load must carry its error')


@dataclass(frozen=True, slots=True)
class PrdDocument:
    """What a PRD's own files say: its H1 title, and its sidecar or why that is unread."""

    title: str | None
    sidecar: CapabilityManifestDoc | SidecarUnread


_NO_DOCUMENT = PrdDocument(
    title=None, sidecar=SidecarUnread(SignalCause.NO_SIDECAR, 'no document was read for this PRD'),
)


@dataclass(frozen=True, slots=True)
class SignalLeaf:
    """One signal leaf as the store reads it; ``done_kind`` only for a done leaf."""

    label: str
    task_id: int | None
    status: str | None
    done_kind: str | None

    def to_wire(self) -> dict[str, object]:
        return {
            'label': self.label,
            'task_id': self.task_id,
            'status': self.status,
            'done_kind': self.done_kind,
        }


@dataclass(frozen=True, slots=True)
class Signal:
    """A PRD's signal mark, the leaves it was read from, and why it is unknown when it is."""

    mark: SignalMark
    cause: SignalCause | None
    reason: str | None
    leaves: tuple[SignalLeaf, ...]

    def __post_init__(self) -> None:
        unknown = self.mark is SignalMark.UNKNOWN
        if unknown != (self.cause is not None) or unknown != (self.reason is not None):
            raise ValueError(
                'Signal: an unknown mark carries a cause and a reason, any other mark neither; '
                f'got mark={self.mark!s}, cause={self.cause!s}, reason={self.reason!r}'
            )

    def to_wire(self) -> dict[str, object]:
        return {
            'mark': self.mark.value,
            'cause': None if self.cause is None else self.cause.value,
            'reason': self.reason,
            'leaves': [leaf.to_wire() for leaf in self.leaves],
        }


@dataclass(frozen=True, slots=True)
class SignalCoverage:
    """How many drawn PRDs name a signal leaf in a sidecar that loads, of all drawn PRDs."""

    named: int
    of: int

    def to_wire(self) -> dict[str, object]:
        return {'named': self.named, 'of': self.of}


@dataclass(frozen=True)
class PrdViewInputs:
    """What one root's view is built from.

    ``census`` is the status-map half's provenance; ``status_map`` is the
    population and the only source of each task's status. ``escalations``
    maps a task to its pending human-level escalations; ``documents`` holds
    a document for every PRD :func:`open_work_keys` names; ``liveness``
    carries the active rows' provenance.
    """

    project_root: str
    status_map: Mapping[int, str]
    census: Datum[TaskCensus]
    rows: Datum[list[dict]]
    lineage: Datum[Mapping[int, LineageEntry]]
    escalations: Datum[Mapping[int, tuple[str, ...]]]
    activity: Datum[TaskActivity]
    documents: Datum[Mapping[str, PrdDocument]]
    liveness: Datum[Mapping[int, Liveness | None]]


# ---------------------------------------------------------------------------
# build_prd_view: the census join
# ---------------------------------------------------------------------------


def build_prd_view(inputs: PrdViewInputs, *, now: datetime) -> Datum[PrdView]:
    """One root's view at *now*, with the composite provenance of its inputs.

    The view is ``unknown`` when the status map is empty or its census was
    never measured (with the census's own reason), when the map carries a
    status outside the vocabulary, or when any input is ``unknown``. Otherwise
    its state is the worst input state, its ``as_of`` and bound the oldest
    input's. The value is checked by :func:`validate_prd_view` before return.
    """
    aged: dict[ViewInput, Datum[Any]] = {
        ViewInput.STATUS_MAP: aged_at(inputs.census, now),
        ViewInput.ACTIVE_ROWS: aged_at(inputs.rows, now),
        ViewInput.LINEAGE: aged_at(inputs.lineage, now),
        ViewInput.ESCALATIONS: aged_at(inputs.escalations, now),
        ViewInput.ACTIVITY: aged_at(inputs.activity, now),
        ViewInput.SIGNALS: aged_at(inputs.documents, now),
        ViewInput.LIVENESS: aged_at(inputs.liveness, now),
    }
    census = aged[ViewInput.STATUS_MAP]
    if census.value is None or not inputs.status_map:
        return unknown_datum(census.reason or _EMPTY_MAP_REASON, census.freshness_bound_seconds)
    try:
        build_census(inputs.status_map)
    except CensusVocabularyError as drift:
        return unknown_datum(str(drift), census.freshness_bound_seconds)
    unknown = [datum for datum in aged.values() if datum.state is DatumState.UNKNOWN]
    if unknown or any(datum.value is None for datum in aged.values()):
        bound = (unknown[0] if unknown else census).freshness_bound_seconds
        return unknown_datum(_degradation_reason(aged) or 'an input has no value', bound)
    states = tuple(
        InputState(name, datum.as_of, datum.state, datum.reason) for name, datum in aged.items()
    )
    view = _shape_view(
        inputs,
        aged[ViewInput.ACTIVE_ROWS].value,
        aged[ViewInput.LINEAGE].value,
        aged[ViewInput.SIGNALS].value,
        states,
        now,
    )
    validate_prd_view(view, inputs.status_map)
    return composite(view, aged)


def open_work_keys(
    project_root: str,
    status_map: Mapping[int, str],
    rows: Sequence[Mapping[str, Any]],
    lineage: Mapping[int, LineageEntry],
) -> tuple[str, ...]:
    """The keys of the PRDs the view draws a row for, those with an open member, sorted.

    Read through the same memoised attribution :func:`build_prd_view` uses.
    """
    attributions = _attributed_for(project_root, rows, lineage).attributions
    return tuple(sorted({
        attribution.prd
        for task_id, status in status_map.items()
        if status not in TERMINAL
        and (attribution := attributions.get(task_id)) is not None
        and attribution.prd is not None
    }))


def _degradation_reason(aged: Mapping[ViewInput, Datum[Any]]) -> str | None:
    named = [
        f'{name}: {datum.reason or datum.state.value}'
        for name, datum in aged.items()
        if datum.state is not DatumState.FRESH
    ]
    return '; '.join(named) or None


def composite(value: V, aged: Mapping[ViewInput, Datum[Any]]) -> Datum[V]:
    """*value* with the provenance of the *aged* inputs it was built from.

    Its state is the worst input state, its ``as_of`` and bound the oldest
    input's, and its reason names every input that is not fresh. Every input
    must be measured.
    """
    measured = [(datum.as_of, datum) for datum in aged.values() if datum.as_of is not None]
    as_of, oldest = min(measured, key=lambda pair: pair[0])
    state = max((datum.state for datum in aged.values()), key=_STATE_SEVERITY.__getitem__)
    return Datum(
        value=value,
        as_of=as_of,
        state=state,
        reason=_degradation_reason(aged),
        freshness_bound_seconds=oldest.freshness_bound_seconds,
    )


@dataclass(frozen=True, slots=True)
class _Member:
    """One non-cancelled task of the population, classified."""

    task_id: int
    row: Mapping[str, Any] | None
    segment: Segment
    attribution: Attribution


@dataclass(frozen=True, slots=True)
class _Attributed:
    """One root's attribution, and the rows and lineage values it was computed from."""

    rows: Sequence[Mapping[str, Any]]
    lineage: Mapping[int, LineageEntry]
    rows_by_id: Mapping[int, Mapping[str, Any]]
    entries: Mapping[int, LineageEntry]
    attributions: Mapping[int, Attribution]
    multi_link: int


_attributed: dict[str, _Attributed] = {}
"""The last attribution per root; ``_attributed_for`` owns it."""


def _attributed_for(
    project_root: str, rows: Sequence[Mapping[str, Any]], lineage: Mapping[int, LineageEntry],
) -> _Attributed:
    """*project_root*'s attribution, overlaying its active rows on its lineage index.

    The overlay and the walk are most of a build's cost and do not depend on
    *now*: on a synthetic 9,700-task root with 3,276 active rows (measured
    2026-11-04, loaded host) a build that re-attributes took at best 98 ms, one
    reusing the attribution 44 ms. Their inputs are cached
    values that are replaced, never mutated, when re-read: the snapshot unit's
    rows for 15 s, the lineage index for 300 s. So the result is reused for
    as long as both are the very same objects. The memo holds them, so
    neither identity can be reused by another value while it is held.
    """
    memo = _attributed.get(project_root)
    if memo is not None and memo.rows is rows and memo.lineage is lineage:
        return memo
    rows_by_id = {row['id']: row for row in rows}
    entries = {
        **lineage,
        **{
            task_id: lineage_entry(_metadata_of(row), project_root)
            for task_id, row in rows_by_id.items()
        },
    }
    attributions, multi_link = attribute(entries)
    memo = _Attributed(
        rows, lineage, MappingProxyType(rows_by_id), MappingProxyType(entries), attributions,
        multi_link,
    )
    _attributed[project_root] = memo
    return memo


def _shape_view(
    inputs: PrdViewInputs,
    rows: Sequence[Mapping[str, Any]],
    lineage: Mapping[int, LineageEntry],
    documents: Mapping[str, PrdDocument],
    states: tuple[InputState, ...],
    now: datetime,
) -> PrdView:
    attributed = _attributed_for(inputs.project_root, rows, lineage)

    def headline(key: str) -> tuple[str, Signal]:
        document = documents.get(key, _NO_DOCUMENT)
        signal = signal_of(key, document, inputs.status_map, attributed.entries)
        return document.title or key, signal

    by_prd: dict[str, list[_Member]] = {}
    dark: dict[UnresolvedReason, list[_Member]] = {}
    unattributed_done = 0
    for task_id, status in inputs.status_map.items():
        row = attributed.rows_by_id.get(task_id)
        segment = segment_of(task_id, status, row, inputs.status_map, now=now)
        if segment is None:
            continue
        attribution = attributed.attributions.get(task_id, _LINEAGE_PENDING)
        member = _Member(task_id, row, segment, attribution)
        if member.attribution.prd is not None:
            by_prd.setdefault(member.attribution.prd, []).append(member)
        elif segment is Segment.DONE:
            unattributed_done += 1
        elif member.attribution.unresolved is not None:
            dark.setdefault(member.attribution.unresolved, []).append(member)
    prd_rows, closed = _prd_rows(by_prd, headline, now)
    return PrdView(
        prds=prd_rows,
        closed=closed,
        unattributed_done=unattributed_done,
        dark_matter=_dark_matter(dark, inputs.status_map),
        multi_link=attributed.multi_link,
        signal_coverage=SignalCoverage(
            named=sum(_names_a_leaf(documents.get(row.key, _NO_DOCUMENT)) for row in prd_rows),
            of=len(prd_rows),
        ),
        inputs=states,
    )


def _names_a_leaf(document: PrdDocument) -> bool:
    sidecar = document.sidecar
    return isinstance(sidecar, CapabilityManifestDoc) and sidecar.signal_leaves is not None


def _prd_rows(
    by_prd: Mapping[str, list[_Member]],
    headline: Callable[[str], tuple[str, Signal]],
    now: datetime,
) -> tuple[tuple[PrdRow, ...], Closed]:
    rows: list[PrdRow] = []
    closed_prds = closed_tasks = 0
    for key in sorted(by_prd):
        members = by_prd[key]
        open_members = sorted(
            (member for member in members if member.segment is not Segment.DONE),
            key=lambda member: member.task_id,
        )
        if open_members:
            title, signal = headline(key)
            rows.append(_prd_row(key, title, signal, members, open_members, now))
        else:
            closed_prds += 1
            closed_tasks += len(members)
    return tuple(rows), Closed(prds=closed_prds, tasks=closed_tasks)


def _prd_row(
    key: str,
    title: str,
    signal: Signal,
    members: Sequence[_Member],
    open_members: Sequence[_Member],
    now: datetime,
) -> PrdRow:
    counts = Counter(member.segment for member in members)
    return PrdRow(
        key=key,
        title=title,
        signal=signal,
        segments=MappingProxyType({segment: counts[segment] for segment in Segment}),
        total=len(members),
        direct_open_ids=tuple(m.task_id for m in open_members if m.attribution.direct),
        followup_open=tuple(
            Followup(m.task_id, m.attribution.hops) for m in open_members if not m.attribution.direct
        ),
        ready=_ready_age([member for member in members if member.segment is Segment.READY], now),
    )


def _ready_age(ready: Sequence[_Member], now: datetime) -> ReadyAge:
    lower_bound = False
    oldest: tuple[datetime, int] | None = None
    for member in ready:
        metadata = _metadata_of(member.row)
        anchor = _pending_anchor(metadata)
        if anchor is None or metadata.get(_PENDING_SINCE_BACKFILLED_KEY) is True:
            lower_bound = True
        if anchor is not None and (oldest is None or (anchor, member.task_id) < oldest):
            oldest = (anchor, member.task_id)
    if oldest is None:
        return ReadyAge(len(ready), None, None, lower_bound)
    anchor, task_id = oldest
    days = int((now - anchor).total_seconds() // _SECONDS_PER_DAY)
    return ReadyAge(len(ready), task_id, days, lower_bound)


def _pending_anchor(metadata: Mapping[str, Any]) -> datetime | None:
    raw = metadata.get(_PENDING_SINCE_KEY)
    if not isinstance(raw, str):
        return None
    moment, ok = parse_timestamp_or_warn(raw, context='prd_view.pending_since')
    return moment if ok else None


def _dark_matter(
    dark: Mapping[UnresolvedReason, list[_Member]], status_map: Mapping[int, str],
) -> DarkMatter:
    everyone = [member for members in dark.values() for member in members]
    priorities = Counter(_recorded(_row_value(m.row, 'priority')) for m in everyone)
    sources = Counter(_recorded(_metadata_of(m.row).get('source')) for m in everyone)
    by_status = Counter(TaskStatus(status_map[m.task_id]) for m in everyone)
    return DarkMatter(
        total=len(everyone),
        by_status=MappingProxyType({status: by_status[status] for status in TaskStatus}),
        by_priority=MappingProxyType(dict(_ranked(priorities))),
        by_source=MappingProxyType(_top_sources(sources)),
        by_reason=MappingProxyType(
            {reason: len(dark.get(reason, ())) for reason in UnresolvedReason},
        ),
        ready_high_ids=tuple(sorted(
            m.task_id for m in everyone
            if m.segment is Segment.READY
            and _recorded(_row_value(m.row, 'priority')) in HIGH_PRIORITIES
        )),
        members=MappingProxyType({
            reason: tuple(sorted(m.task_id for m in dark[reason]))
            for reason in UnresolvedReason
            if dark.get(reason)
        }),
    )


# ---------------------------------------------------------------------------
# signal_of: the signal mark (§4.7)
# ---------------------------------------------------------------------------


def signal_of(
    key: str,
    document: PrdDocument,
    status_map: Mapping[int, str],
    entries: Mapping[int, LineageEntry],
) -> Signal:
    """*key*'s signal mark, its leaves resolved in ``signal_leaves`` order.

    *entries* are the lineage entries with the active rows overlaid, the same
    the attribution walks, so each leaf's own PRD and done kind read as the
    view's members do. The first leaf that cannot be read names the unknown
    cause; a cancelled leaf is one, so a PRD with one is never delivered.
    """
    sidecar = document.sidecar
    if isinstance(sidecar, SidecarUnread):
        return _unknown_signal(sidecar.cause, sidecar.detail, ())
    if sidecar.signal_leaves is None:
        return _unknown_signal(SignalCause.NAMES_NO_LEAF, None, ())
    blocks = {block.label: block for block in sidecar.tasks}
    resolved = [
        _resolve_leaf(key, blocks[label], status_map, entries) for label in sidecar.signal_leaves
    ]
    leaves = tuple(leaf for leaf, _ in resolved)
    failure = next((failure for _, failure in resolved if failure is not None), None)
    if failure is not None:
        return _unknown_signal(*failure, leaves)
    done = sum(leaf.status == TaskStatus.DONE for leaf in leaves)
    if done == len(leaves):
        return Signal(SignalMark.DELIVERED, None, None, leaves)
    return Signal(SignalMark.PARTIAL if done else SignalMark.NOT_YET, None, None, leaves)


_LeafFailure = tuple[SignalCause, str]


def _resolve_leaf(
    key: str,
    block: ManifestTask,
    status_map: Mapping[int, str],
    entries: Mapping[int, LineageEntry],
) -> tuple[SignalLeaf, _LeafFailure | None]:
    label = block.label
    if block.external_task_id is not None:
        failure = (SignalCause.EXTERNAL_LEAF, f'label {label!r} is bound to {block.external_task_id}')
        return SignalLeaf(label, None, None, None), failure
    if block.task_id is None:
        return SignalLeaf(label, None, None, None), (SignalCause.UNBOUND_LABEL, f'label {label!r}')
    task_id = block.task_id
    status = status_map.get(task_id)
    entry = entries.get(task_id)
    done_kind = entry.done_kind if entry is not None and status == TaskStatus.DONE else None
    leaf = SignalLeaf(label, task_id, status, done_kind)
    bound_to = f'label {label!r} is bound to task {task_id}'
    if status is None:
        return leaf, (SignalCause.LEAF_NOT_IN_STORE, bound_to)
    if entry is None or entry.own_prd != PrdKey(key):
        return leaf, (SignalCause.STALE_BINDING, f'{bound_to}, whose PRD is {_own_prd_of(entry)}')
    if status == TaskStatus.CANCELLED:
        return leaf, (SignalCause.LEAF_CANCELLED, bound_to)
    return leaf, None


def _own_prd_of(entry: LineageEntry | None) -> str:
    if entry is None:
        return 'unread'
    if isinstance(entry.own_prd, PrdKey):
        return entry.own_prd.key
    if isinstance(entry.own_prd, Unresolved):
        return f'unresolved ({entry.own_prd.reason})'
    return 'not named on the task'


def _unknown_signal(
    cause: SignalCause, detail: str | None, leaves: tuple[SignalLeaf, ...],
) -> Signal:
    text = SIGNAL_CAUSE_TEXT[cause]
    return Signal(
        SignalMark.UNKNOWN, cause, text if detail is None else f'{text}: {detail}', leaves,
    )


def _row_value(row: Mapping[str, Any] | None, key: str) -> object:
    return None if row is None else row.get(key)


def _recorded(value: object) -> str:
    return value if isinstance(value, str) and value.strip() else UNRECORDED


def _ranked(counts: Counter[str]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _top_sources(sources: Counter[str]) -> dict[str, int]:
    ranked = _ranked(sources)
    top = dict(ranked[:TOP_SOURCES])
    rest = sum(n for _, n in ranked[TOP_SOURCES:])
    if rest:
        top[OTHER_SOURCES] = top.get(OTHER_SOURCES, 0) + rest
    return top


# ---------------------------------------------------------------------------
# validate_prd_view: the γ1 subset of §5.5's shaping validators
# ---------------------------------------------------------------------------


class PrdViewInvariant(enum.StrEnum):
    """The rule a :class:`PrdViewContractError` reports."""

    SEGMENTS_SUM = 'segments_sum'
    OPEN_MEMBER = 'open_member'
    CENSUS_IDENTITY = 'census_identity'
    DARK_MATTER_TOTAL = 'dark_matter_total'
    OPEN_ID_ONCE = 'open_id_once'
    FIGURE_RULE = 'figure_rule'


class PrdViewContractError(DatumContractError):
    """A :class:`PrdView` broke one of its declared rules, named by :attr:`rule`."""

    def __init__(self, rule: PrdViewInvariant, message: str) -> None:
        super().__init__(DatumInvariant.VALUE_CONTRACT, f'{rule.value}: {message}')
        self.rule = rule


def validate_prd_view(view: PrdView, status_map: Mapping[int, str]) -> None:
    """Raise :class:`PrdViewContractError` if *view* breaks a declared rule.

    The census identity is checked against ``build_census(status_map)``,
    computed here over the very map the members were drawn from, never
    against a snapshot's census, which may be last-good while the map failed.
    """
    for row in view.prds:
        _check_row(row)
    _check_dark_matter_total(view.dark_matter)
    _check_census_identity(view, build_census(status_map))
    _check_open_ids_once(view, status_map)
    _check_figure_rules(view)


def _check_row(row: PrdRow) -> None:
    summed = sum(row.segments.values())
    if summed != row.total:
        raise PrdViewContractError(
            PrdViewInvariant.SEGMENTS_SUM,
            f'PRD {row.key!r} segments sum to {summed} but total={row.total} '
            f'(segments {_segment_counts(row)})',
        )
    done = row.segments.get(Segment.DONE, 0)
    if row.total - done < 1:
        raise PrdViewContractError(
            PrdViewInvariant.OPEN_MEMBER,
            f'PRD {row.key!r} has no open member (total={row.total}, done={done}); '
            'a PRD with no open member is counted in closed, not drawn as a row',
        )


def _segment_counts(row: PrdRow) -> dict[str, int]:
    return {segment.value: n for segment, n in row.segments.items()}


def _check_dark_matter_total(dark_matter: DarkMatter) -> None:
    listed = sum(len(ids) for ids in dark_matter.members.values())
    if dark_matter.total != listed:
        raise PrdViewContractError(
            PrdViewInvariant.DARK_MATTER_TOTAL,
            f'dark_matter total={dark_matter.total} but its members list {listed} id(s)',
        )


def _check_census_identity(view: PrdView, census: TaskCensus) -> None:
    drawn = sum(row.total for row in view.prds)
    summed = drawn + view.closed.tasks + view.unattributed_done + view.dark_matter.total
    cancelled = census.counts[TaskStatus.CANCELLED]
    expected = census.total - cancelled
    if summed != expected:
        raise PrdViewContractError(
            PrdViewInvariant.CENSUS_IDENTITY,
            f'rows {drawn} + closed {view.closed.tasks} + unattributed_done '
            f'{view.unattributed_done} + dark_matter {view.dark_matter.total} = {summed}, but '
            f'the status map holds {census.total} task(s) less {cancelled} cancelled = {expected}',
        )


def _check_open_ids_once(view: PrdView, status_map: Mapping[int, str]) -> None:
    listed = Counter(_listed_open_ids(view))
    expected = {task_id for task_id, status in status_map.items() if status not in TERMINAL}
    problems = {
        'duplicated': sorted(task_id for task_id, n in listed.items() if n > 1),
        'missing': sorted(expected - listed.keys()),
        'not open': sorted(listed.keys() - expected),
    }
    named = [f'{label} {_ids(ids)}' for label, ids in problems.items() if ids]
    if named:
        raise PrdViewContractError(
            PrdViewInvariant.OPEN_ID_ONCE,
            f'every open task must be listed exactly once: {"; ".join(named)}',
        )


def _listed_open_ids(view: PrdView) -> Iterable[int]:
    for row in view.prds:
        yield from row.direct_open_ids
        yield from (followup.task_id for followup in row.followup_open)
    for ids in view.dark_matter.members.values():
        yield from ids


def _check_figure_rules(view: PrdView) -> None:
    view_figures = set(view.to_wire()) - _STRUCTURAL_VIEW_KEYS
    row_figures = {key for row in view.prds for key in row.to_wire()} - _STRUCTURAL_ROW_KEYS
    unnamed = sorted((view_figures | row_figures) - {figure.value for figure in FigureId})
    if unnamed:
        raise PrdViewContractError(
            PrdViewInvariant.FIGURE_RULE,
            f'wire figures naming no FigureId, so carrying no rule text: {unnamed}',
        )


def _ids(ids: Sequence[int]) -> str:
    shown = ', '.join(str(task_id) for task_id in ids[:_LISTED_IDS_CAP])
    more = len(ids) - _LISTED_IDS_CAP
    return f'[{shown}]' + (f' (+{more} more)' if more > 0 else '')
