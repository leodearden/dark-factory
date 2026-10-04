"""The H1 heal plan: which action each live link earns, and what it writes.

Contract: ``plans/write-triage-link-healing-prd.md`` H1. The action table has
one home, here: :func:`_deterministic_row` holds rows 1-5, and
:data:`_VERDICT_TABLE` plus the half-link children guard in
:func:`_verdict_row` hold the rest. The kinds and the contested key are
imported from ``server/grouped_read.py`` and never respelled.

A verdict reaches the table only as a :class:`LinkBasis`. The committed
hand-link corpus is one source of them (:func:`load_corpus_bases`); other
sources plug in by producing the same rows.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fused_memory.maintenance.link_heal_store import (
    LinkCensus,
    LinkHealStore,
    LiveRecord,
    MetadataChange,
    StoreReadFailed,
)
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CHILD_KINDS,
    CONTESTED_METADATA_KEY,
    PARENT_ID_KEY,
    SIGHTING_KIND,
)

KIND_KEY = 'kind'
PEER_KIND = 'peer'

#: The record keys a heal reads and writes, in the order changes name them.
LINK_KEYS = (PARENT_ID_KEY, KIND_KEY, CONTESTED_METADATA_KEY)


class Verdict(StrEnum):
    """The rater brief's seven words."""

    SAME = 'SAME'
    EXTENDS = 'EXTENDS'
    SUBSUMED = 'SUBSUMED'
    CORRECTS = 'CORRECTS'
    RELATED = 'RELATED'
    UNRELATED = 'UNRELATED'
    UNCLEAR = 'UNCLEAR'


MISFILE_VERDICTS = frozenset({Verdict.RELATED, Verdict.UNRELATED})
BELONGS_VERDICTS = frozenset(
    {Verdict.SAME, Verdict.EXTENDS, Verdict.SUBSUMED, Verdict.CORRECTS},
)


class KindClass(StrEnum):
    SIGHTING = 'sighting'
    AMENDMENT = 'amendment'
    PEER = 'peer'
    HALF_LINK = 'half_link'


_KIND_CLASSES = MappingProxyType({
    SIGHTING_KIND: KindClass.SIGHTING,
    AMENDMENT_KIND: KindClass.AMENDMENT,
    PEER_KIND: KindClass.PEER,
})


_CHILD_KIND_CLASSES = frozenset(_KIND_CLASSES[kind] for kind in CHILD_KINDS)


def kind_class(kind: Any) -> KindClass:
    """Any kind outside the child kinds and ``peer``, or none, is a half-link."""
    if not isinstance(kind, str):
        return KindClass.HALF_LINK
    return _KIND_CLASSES.get(kind, KindClass.HALF_LINK)


class HealAction(StrEnum):
    DETACH = 'detach'
    RELABEL = 'relabel'
    RELABEL_FLAG = 'relabel_flag'
    FLAG = 'flag'
    COMPLETE_AMENDMENT = 'complete_amendment'
    COMPLETE_AMENDMENT_FLAG = 'complete_amendment_flag'
    COMPLETE_SIGHTING = 'complete_sighting'


class Report(StrEnum):
    STALE_RATING = 'stale_rating'
    CONTESTED_REPORTED = 'contested_reported'
    CROSS_PROJECT_REPORTED = 'cross_project_reported'
    CHAIN_REPORTED = 'chain_reported'
    UNCLEAR = 'unclear'
    PEER_REPORTED = 'peer_reported'
    HAS_CHILDREN = 'has_children'
    NO_ACTION = 'no_action'
    UNEXAMINED = 'unexamined'


class BasisSource(StrEnum):
    CORPUS = 'corpus'
    DETERMINISTIC = 'deterministic'
    UNDO = 'undo'


def link_parent_id(meta: Mapping[str, Any]) -> str | None:
    """A record's ``parent_id``, or ``None`` unless it is a non-empty string."""
    value = meta.get(PARENT_ID_KEY)
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class LinkImage:
    """The link keys of one record. ``None`` means the key is absent."""

    parent_id: str | None
    kind: Any = None
    contested: Any = None

    @classmethod
    def from_metadata(cls, meta: Mapping[str, Any]) -> LinkImage:
        return cls(
            parent_id=link_parent_id(meta),
            kind=meta.get(KIND_KEY),
            contested=meta.get(CONTESTED_METADATA_KEY),
        )

    def as_dict(self) -> dict[str, Any]:
        """The present keys only."""
        values = (self.parent_id, self.kind, self.contested)
        return {key: value for key, value in zip(LINK_KEYS, values, strict=True) if value is not None}


def apply_change(image: LinkImage, change: MetadataChange) -> LinkImage:
    keys = image.as_dict()
    keys.update(change.patch or {})
    for key in change.delete_keys or ():
        keys.pop(key, None)
    return LinkImage.from_metadata(keys)


#: What each patch-only action writes: (kind, or None to keep it; whether to contest).
_WRITE_TARGETS: Mapping[HealAction, tuple[str | None, bool]] = MappingProxyType({
    HealAction.RELABEL: (AMENDMENT_KIND, False),
    HealAction.RELABEL_FLAG: (AMENDMENT_KIND, True),
    HealAction.FLAG: (None, True),
    HealAction.COMPLETE_AMENDMENT: (AMENDMENT_KIND, False),
    HealAction.COMPLETE_AMENDMENT_FLAG: (AMENDMENT_KIND, True),
    HealAction.COMPLETE_SIGHTING: (SIGHTING_KIND, False),
})


def change_for(action: HealAction, pre_image: LinkImage) -> MetadataChange:
    """The one write *action* makes on a record showing *pre_image*.

    A detach drops ``kind`` only when it is a child kind: an agent-invented
    kind such as ``extension`` stays.
    """
    if action is HealAction.DETACH:
        if kind_class(pre_image.kind) in _CHILD_KIND_CLASSES:
            return MetadataChange.delete_only([PARENT_ID_KEY, KIND_KEY])
        return MetadataChange.delete_only([PARENT_ID_KEY])
    kind, contest = _WRITE_TARGETS[action]
    patch: dict[str, Any] = {}
    if kind is not None:
        patch[KIND_KEY] = kind
    if contest:
        patch[CONTESTED_METADATA_KEY] = True
    return MetadataChange.patch_only(patch)


def post_image_for(action: HealAction, pre_image: LinkImage) -> LinkImage:
    return apply_change(pre_image, change_for(action, pre_image))


def undo_changes(post_image: LinkImage, pre_image: LinkImage) -> list[MetadataChange]:
    """The writes that take *post_image* back to *pre_image*: delete first, then patch."""
    current, target = post_image.as_dict(), pre_image.as_dict()
    deletes = [key for key in LINK_KEYS if key in current and key not in target]
    patch = {
        key: target[key]
        for key in LINK_KEYS
        if key in target and (key not in current or current[key] != target[key])
    }
    changes = []
    if deletes:
        changes.append(MetadataChange.delete_only(deletes))
    if patch:
        changes.append(MetadataChange.patch_only(patch))
    return changes


_REASON_FIELD_CHARS = 64


def journal_reason(run8: str, action: str, pre_image: LinkImage) -> str:
    """The write journal's pointer back to the ledger, under 200 characters."""
    return (
        f'link-heal r={run8} a={action} '
        f'prev_parent={_reason_field(pre_image.parent_id)} '
        f'prev_kind={_reason_field(pre_image.kind)}'
    )


def _reason_field(value: Any) -> str:
    return 'none' if value is None else str(value)[:_REASON_FIELD_CHARS]


@dataclass(frozen=True)
class LinkBasis:
    """A verdict on one (child, parent) pair, at the text hashes it judged."""

    project_id: str
    child_id: str
    parent_id: str
    verdict: Verdict
    child_sha256: str
    parent_sha256: str
    source: BasisSource
    key: str
    rated_text_matches_live: bool = True


class CorpusFormatError(ValueError):
    """A hand-link corpus row this module refuses to plan from."""


_SHA256_HEX = re.compile(r'[0-9a-f]{64}')


def load_corpus_bases(path: Path) -> tuple[LinkBasis, ...]:
    """Every row of the committed hand-link corpus, parsed strictly."""
    bases: list[LinkBasis] = []
    seen: set[tuple[str, str, str]] = set()
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        basis = _corpus_basis(_corpus_row(line, line_no), line_no)
        link = (basis.project_id, basis.child_id, basis.parent_id)
        if link in seen:
            raise CorpusFormatError(f'{basis.key}: duplicate link {link}')
        seen.add(link)
        bases.append(basis)
    return tuple(bases)


def _corpus_row(line: str, line_no: int) -> dict[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise CorpusFormatError(f'line {line_no}: not JSON ({exc})') from exc
    if not isinstance(row, dict):
        raise CorpusFormatError(f'line {line_no}: not a JSON object')
    return row


def _corpus_basis(row: dict[str, Any], line_no: int) -> LinkBasis:
    key = row.get('item_id')
    if not isinstance(key, str) or not key:
        raise CorpusFormatError(f'line {line_no}: no item_id')
    try:
        return LinkBasis(
            project_id=_corpus_text(row, 'project'),
            child_id=_corpus_text(row, 'entry_id'),
            parent_id=_corpus_text(row, 'target_id'),
            verdict=Verdict(row.get('verdict')),
            child_sha256=_corpus_sha256(row, 'child_sha256'),
            parent_sha256=_corpus_sha256(row, 'parent_sha256'),
            source=BasisSource.CORPUS,
            key=key,
            rated_text_matches_live=_corpus_flag(row, 'rated_text_matches_live'),
        )
    except ValueError as exc:
        raise CorpusFormatError(f'{key}: {exc}') from exc


def _corpus_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f'{field} is not a non-empty string')
    return value


def _corpus_sha256(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
        raise ValueError(f'{field} is not a sha256 hex digest')
    return value


def _corpus_flag(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise ValueError(f'{field} is not a boolean')
    return value


class ParentPresence(StrEnum):
    FOUND = 'found'
    ABSENT = 'absent'
    OTHER_PROJECT = 'other_project'


@dataclass(frozen=True)
class LinkState:
    """One live link as read just now: the child, and its parent in the child's project."""

    project_id: str
    child: LiveRecord
    parent_presence: ParentPresence
    parent: LiveRecord | None
    has_children: bool

    @property
    def link(self) -> LinkImage:
        return LinkImage.from_metadata(self.child.metadata)


@dataclass(frozen=True)
class Decision:
    """A table row's outcome, and for a heal, the basis it rests on."""

    outcome: HealAction | Report
    basis_source: BasisSource | None = None


def decide(state: LinkState, basis: LinkBasis | None) -> Decision:
    """Evaluate the H1 action table for one link; the first matching row wins."""
    deterministic = _deterministic_row(state, basis)
    if deterministic is not None:
        return deterministic
    if basis is None:
        return Decision(Report.UNEXAMINED)
    outcome = _verdict_row(state, basis.verdict)
    return Decision(outcome, basis.source if isinstance(outcome, HealAction) else None)


def is_stale(state: LinkState, basis: LinkBasis) -> bool:
    """The verdict no longer judges the texts the link shows now."""
    if not basis.rated_text_matches_live:
        return True
    if state.child.text_sha256 != basis.child_sha256:
        return True
    return state.parent is not None and state.parent.text_sha256 != basis.parent_sha256


def _deterministic_row(state: LinkState, basis: LinkBasis | None) -> Decision | None:
    if basis is not None and is_stale(state, basis):
        return Decision(Report.STALE_RATING)
    if state.link.contested:
        return Decision(Report.CONTESTED_REPORTED)
    if state.parent_presence is ParentPresence.ABSENT:
        return Decision(HealAction.DETACH, BasisSource.DETERMINISTIC)
    if state.parent_presence is ParentPresence.OTHER_PROJECT:
        return Decision(Report.CROSS_PROJECT_REPORTED)
    if state.parent is not None and link_parent_id(state.parent.metadata) is not None:
        return Decision(Report.CHAIN_REPORTED)
    return None


def _verdict_row(state: LinkState, verdict: Verdict) -> HealAction | Report:
    kind = kind_class(state.link.kind)
    if kind is KindClass.HALF_LINK and state.has_children and verdict in BELONGS_VERDICTS:
        return Report.HAS_CHILDREN
    return _VERDICT_TABLE[(kind, verdict)]


def _every_kind(verdicts: Iterable[Verdict], outcome: HealAction | Report):
    return {(kind, verdict): outcome for kind in KindClass for verdict in verdicts}


_VERDICT_TABLE: Mapping[tuple[KindClass, Verdict], HealAction | Report] = MappingProxyType({
    **_every_kind(MISFILE_VERDICTS, HealAction.DETACH),
    **_every_kind([Verdict.UNCLEAR], Report.UNCLEAR),
    (KindClass.SIGHTING, Verdict.EXTENDS): HealAction.RELABEL,
    (KindClass.SIGHTING, Verdict.CORRECTS): HealAction.RELABEL_FLAG,
    (KindClass.SIGHTING, Verdict.SAME): Report.NO_ACTION,
    (KindClass.SIGHTING, Verdict.SUBSUMED): Report.NO_ACTION,
    (KindClass.AMENDMENT, Verdict.CORRECTS): HealAction.FLAG,
    (KindClass.AMENDMENT, Verdict.SAME): Report.NO_ACTION,
    (KindClass.AMENDMENT, Verdict.SUBSUMED): Report.NO_ACTION,
    (KindClass.AMENDMENT, Verdict.EXTENDS): Report.NO_ACTION,
    **{(KindClass.PEER, verdict): Report.PEER_REPORTED for verdict in BELONGS_VERDICTS},
    (KindClass.HALF_LINK, Verdict.EXTENDS): HealAction.COMPLETE_AMENDMENT,
    (KindClass.HALF_LINK, Verdict.CORRECTS): HealAction.COMPLETE_AMENDMENT_FLAG,
    (KindClass.HALF_LINK, Verdict.SAME): HealAction.COMPLETE_SIGHTING,
    (KindClass.HALF_LINK, Verdict.SUBSUMED): HealAction.COMPLETE_SIGHTING,
})


@dataclass(frozen=True)
class PlannedAction:
    """One heal, as planned: what the record must show before, and after.

    The write itself is never stored; :attr:`change` derives it from the action
    and the pre-image. ``parent_sha256`` is ``None`` exactly for a
    deterministic detach, whose parent is gone.
    """

    project_id: str
    child_id: str
    action: HealAction
    pre_image: LinkImage
    post_image: LinkImage
    basis_source: BasisSource
    basis_key: str | None
    child_sha256: str
    parent_sha256: str | None

    def __post_init__(self) -> None:
        if self.post_image != post_image_for(self.action, self.pre_image):
            raise ValueError(
                f'post_image {self.post_image} is not what {self.action} '
                f'writes over {self.pre_image}',
            )
        if (self.parent_sha256 is None) != (self.basis_source is BasisSource.DETERMINISTIC):
            raise ValueError('parent_sha256 is None exactly for a deterministic basis')

    @classmethod
    def planned(
        cls,
        *,
        project_id: str,
        child_id: str,
        action: HealAction,
        pre_image: LinkImage,
        basis_source: BasisSource,
        basis_key: str | None,
        child_sha256: str,
        parent_sha256: str | None,
    ) -> PlannedAction:
        return cls(
            project_id=project_id,
            child_id=child_id,
            action=action,
            pre_image=pre_image,
            post_image=post_image_for(action, pre_image),
            basis_source=basis_source,
            basis_key=basis_key,
            child_sha256=child_sha256,
            parent_sha256=parent_sha256,
        )

    @property
    def change(self) -> MetadataChange:
        return change_for(self.action, self.pre_image)


COMPLETION_ACTIONS = frozenset({
    HealAction.COMPLETE_AMENDMENT,
    HealAction.COMPLETE_AMENDMENT_FLAG,
    HealAction.COMPLETE_SIGHTING,
})


class StaleField(StrEnum):
    """What a live re-read found no longer as a heal expected it."""

    CHILD = 'child'
    PARENT_ID = PARENT_ID_KEY
    KIND = KIND_KEY
    CONTESTED = CONTESTED_METADATA_KEY
    CHILD_SHA256 = 'child_sha256'
    PARENT = 'parent'
    PARENT_SHA256 = 'parent_sha256'
    CHILDREN = 'children'


def verification_mismatch(image: LinkImage, record: LiveRecord | None) -> StaleField | None:
    """The first link key on which *record* does not show *image*, or ``None``."""
    if record is None:
        return StaleField.CHILD
    return _image_mismatch(image, record)


def _image_mismatch(image: LinkImage, record: LiveRecord) -> StaleField | None:
    shown, expected = LinkImage.from_metadata(record.metadata).as_dict(), image.as_dict()
    for key in LINK_KEYS:
        if shown.get(key) != expected.get(key):
            return StaleField(key)
    return None


def corroboration_mismatch(
    planned: PlannedAction,
    child: LiveRecord | None,
    parent: LiveRecord | None,
    child_count: int | None,
) -> StaleField | None:
    """The first way the live link differs from what *planned* was planned against.

    *parent* is the read of the pre-image's parent in the child's project;
    *child_count* is needed only for a completion.
    """
    if child is None:
        return StaleField.CHILD
    return (
        _image_mismatch(planned.pre_image, child)
        or _child_hash_mismatch(planned, child)
        or _parent_mismatch(planned, parent)
        or _children_mismatch(planned, child_count)
    )


def _child_hash_mismatch(planned: PlannedAction, child: LiveRecord) -> StaleField | None:
    return None if child.text_sha256 == planned.child_sha256 else StaleField.CHILD_SHA256


def _parent_mismatch(planned: PlannedAction, parent: LiveRecord | None) -> StaleField | None:
    """A deterministic detach needs its parent still gone; any other heal, unchanged."""
    if planned.basis_source is BasisSource.DETERMINISTIC:
        return None if parent is None else StaleField.PARENT
    if parent is None:
        return StaleField.PARENT
    return None if parent.text_sha256 == planned.parent_sha256 else StaleField.PARENT_SHA256


def _children_mismatch(planned: PlannedAction, child_count: int | None) -> StaleField | None:
    if planned.action in COMPLETION_ACTIONS and child_count != 0:
        return StaleField.CHILDREN
    return None


@dataclass(frozen=True)
class RunCounts:
    """The one H1 disclosure record, shared by plan, apply, undo and status.

    ``examined`` counts the live links that were read and decided.
    ``adjudicated`` counts the examined links whose basis still judges the texts they show.
    ``unexamined`` counts the examined links that reached the verdict rows with no basis.
    """

    links_total: int = 0
    examined: int = 0
    adjudicated: int = 0
    unexamined: int = 0
    planned: int = 0
    planned_by_action: Mapping[str, int] = field(default_factory=dict)
    already_pending: int = 0
    undo_suppressed: int = 0
    bases_unlinked: int = 0
    read_failed: int = 0
    applied: int = 0
    skipped_stale: int = 0
    skipped_cap: int = 0
    failed: int = 0
    not_attempted: int = 0
    stale_rating: int = 0
    contested_reported: int = 0
    chain_reported: int = 0
    peer_reported: int = 0
    has_children: int = 0
    cross_project_reported: int = 0
    unclear: int = 0
    no_action: int = 0
    would_escape: tuple[str, ...] = ()
    escaped: tuple[Mapping[str, Any], ...] = ()
    caps_bit: tuple[str, ...] = ()
    stopped_by: str | None = None
    approved_plan_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'planned_by_action', MappingProxyType(dict(self.planned_by_action)),
        )
        object.__setattr__(self, 'would_escape', tuple(self.would_escape))
        object.__setattr__(
            self, 'escaped', tuple(MappingProxyType(dict(entry)) for entry in self.escaped),
        )
        object.__setattr__(self, 'caps_bit', tuple(self.caps_bit))

    @property
    def complete(self) -> bool:
        partial = (self.read_failed, self.failed, self.skipped_cap, self.not_attempted)
        return not any(partial) and self.stopped_by is None

    def as_json(self) -> dict[str, Any]:
        document: dict[str, Any] = {item.name: getattr(self, item.name) for item in fields(self)}
        document['planned_by_action'] = dict(self.planned_by_action)
        document['would_escape'] = list(self.would_escape)
        document['escaped'] = [dict(entry) for entry in self.escaped]
        document['caps_bit'] = list(self.caps_bit)
        document['complete'] = self.complete
        return document


@dataclass(frozen=True)
class Plan:
    actions: tuple[PlannedAction, ...]
    counts: RunCounts


async def read_link_state(
    store: LinkHealStore, project_id: str, child_id: str, projects: Sequence[str],
) -> LinkState | None:
    """The link as it reads now, or ``None`` once the record is no longer a link.

    A failed read raises :class:`StoreReadFailed`: unknown is never absent.
    """
    child = await store.read(project_id, child_id)
    parent_id = None if child is None else link_parent_id(child.metadata)
    if child is None or parent_id is None:
        return None
    parent = await store.read(project_id, parent_id)
    presence = (
        ParentPresence.FOUND
        if parent is not None
        else await _presence_elsewhere(store, parent_id, project_id, projects)
    )
    children = await store.count_children(project_id, child_id)
    return LinkState(
        project_id=project_id,
        child=child,
        parent_presence=presence,
        parent=parent,
        has_children=children > 0,
    )


async def _presence_elsewhere(
    store: LinkHealStore, memory_id: str, home: str, projects: Sequence[str],
) -> ParentPresence:
    for project_id in projects:
        if project_id != home and await store.read(project_id, memory_id) is not None:
            return ParentPresence.OTHER_PROJECT
    return ParentPresence.ABSENT


async def build_plan(
    bases: Iterable[LinkBasis],
    *,
    store: LinkHealStore,
    census: LinkCensus,
    projects: Sequence[str],
) -> Plan:
    """Decide every live link of *projects* against *bases*. Writes nothing."""
    tally = _PlanTally(bases)
    for project_id in projects:
        for child_id in await census.linked_ids(project_id):
            try:
                state = await read_link_state(store, project_id, child_id, projects)
            except StoreReadFailed:
                tally.record_read_failure(project_id, child_id)
                continue
            if state is not None:
                tally.record(state)
    return tally.plan()


def _link_key(project_id: str, child_id: str, parent_id: str | None) -> tuple[str, str, str | None]:
    return (project_id, child_id, parent_id)


class _PlanTally:
    """What :func:`build_plan` has seen so far. Each :class:`Report` is
    disclosed by the :class:`RunCounts` counter its value names."""

    def __init__(self, bases: Iterable[LinkBasis]) -> None:
        self._bases = {
            _link_key(basis.project_id, basis.child_id, basis.parent_id): basis
            for basis in bases
        }
        self._matched: set[tuple[str, str, str | None]] = set()
        self._unread: set[tuple[str, str]] = set()
        self._reports: Counter[Report] = Counter()
        self._actions: list[PlannedAction] = []
        self._examined = 0
        self._adjudicated = 0

    def record_read_failure(self, project_id: str, child_id: str) -> None:
        self._unread.add((project_id, child_id))

    def record(self, state: LinkState) -> None:
        self._examined += 1
        key = _link_key(state.project_id, state.child.memory_id, state.link.parent_id)
        basis = self._bases.get(key)
        if basis is not None:
            self._matched.add(key)
            if not is_stale(state, basis):
                self._adjudicated += 1
        decision = decide(state, basis)
        if isinstance(decision.outcome, HealAction):
            self._actions.append(_planned_action(state, decision.outcome, decision.basis_source, basis))
        else:
            self._reports[decision.outcome] += 1

    def _bases_unlinked(self) -> int:
        return sum(
            1
            for key in self._bases
            if key not in self._matched and (key[0], key[1]) not in self._unread
        )

    def plan(self) -> Plan:
        reported: dict[str, Any] = {report.value: self._reports[report] for report in Report}
        counts = RunCounts(
            links_total=self._examined + len(self._unread),
            examined=self._examined,
            adjudicated=self._adjudicated,
            planned=len(self._actions),
            planned_by_action=Counter(action.action.value for action in self._actions),
            bases_unlinked=self._bases_unlinked(),
            read_failed=len(self._unread),
            **reported,
        )
        return Plan(actions=tuple(self._actions), counts=counts)


def _planned_action(
    state: LinkState,
    action: HealAction,
    basis_source: BasisSource | None,
    basis: LinkBasis | None,
) -> PlannedAction:
    """A deterministic detach rests on the live child; every other heal on its basis."""
    if basis_source is BasisSource.DETERMINISTIC:
        source, key = BasisSource.DETERMINISTIC, None
        child_sha256, parent_sha256 = state.child.text_sha256, None
    elif basis is not None:
        source, key = basis.source, basis.key
        child_sha256, parent_sha256 = basis.child_sha256, basis.parent_sha256
    else:
        raise ValueError(f'{action} for {state.child.memory_id} rests on no basis')
    return PlannedAction.planned(
        project_id=state.project_id,
        child_id=state.child.memory_id,
        action=action,
        pre_image=state.link,
        basis_source=source,
        basis_key=key,
        child_sha256=child_sha256,
        parent_sha256=parent_sha256,
    )
