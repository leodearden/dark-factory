"""Bound a long-running audit-trail task instead of letting it accrete forever.

``docs/task-authoring.md`` §10 is the normative rule; this module is the
harness side of it.  It plans a rotation of one task (a pure function of the
task and the clock) and executes the plan over two injected ports: an archive
that must read back byte-identical before anything leaves the task, and a
compare-and-swap commit.

Import-light by design, like ``consolidation_gate.py``: stdlib, ``shared`` and
the one ``context_assembler`` constant whose prefix it must never shed.  The
interceptor imports this module, and the memory service is duck-typed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, Protocol

from shared.task_statuses import TERMINAL

from fused_memory.reconciliation.context_assembler import HINT_QUERIES_EXECUTED

__all__ = [
    'ROTATE_THRESHOLD_BYTES',
    'ROTATE_TARGET_BYTES',
    'HISTORY_KEEP',
    'HISTORY_MAX',
    'ARCHIVE_AGENT_ID',
    'ARCHIVE_MAX_BYTES',
    'ARCHIVE_SOURCE',
    'ROLLUP_KEY',
    'STANDING_INSTRUCTION',
    'AuditTrailArchive',
    'AuditTrailCommit',
    'RotationOutcome',
    'RotationPlan',
    'TaskRewrite',
    'bound_audit_trail',
    'memory_service_archive',
    'near_duplicate_key',
    'plan_rotation',
    'task_fingerprint',
    'task_payload_bytes',
    'unrotatable_reason',
]

# docs §10 "The threshold": rotate above 20,000 B, rotate down to <=10,000 B.
ROTATE_THRESHOLD_BYTES = 20_000
ROTATE_TARGET_BYTES = 10_000
# The cap of the autopilot_video 654 / gate 657 hand rotation.
HISTORY_KEEP = 5
# Trim from 2x down to 1x, so an archive memory is written once per HISTORY_KEEP cycles.
HISTORY_MAX = 2 * HISTORY_KEEP
# text-embedding-3-small takes 8,191 tokens; mem0 embeds the whole archive (infer=False),
# so budget a conservative 2 bytes/token.
ARCHIVE_MAX_BYTES = 16_000
ROLLUP_KEY = 'audit_trail_rotation'
# The archive memory's blessed Mem0 'source'; it carries no 'kind' (KIND_REGISTRY is closed).
ARCHIVE_SOURCE = 'audit_trail_rotation'
# A recon-stage id, so reconciliation/internal_writers.py::is_internal_writer recognises it.
ARCHIVE_AGENT_ID = 'recon-stage-audit_trail_rotation'
STANDING_INSTRUCTION = (
    'Append each new cycle record to the arrays named in history_keys, newest first, '
    'instead of minting a dated top-level metadata key. The harness trims each array '
    f'from {HISTORY_MAX} to {HISTORY_KEEP} entries and archives what it sheds verbatim '
    '(docs/task-authoring.md §10).'
)

_TEXT_COLUMNS = ('title', 'description', 'details')


def _column_bytes(task: Mapping[str, Any]) -> dict[str, int]:
    sizes = {column: len((task.get(column) or '').encode()) for column in _TEXT_COLUMNS}
    metadata = task.get('metadata')
    sizes['metadata'] = 0 if metadata is None else len(json.dumps(metadata).encode())
    return sizes


def task_payload_bytes(task: Mapping[str, Any]) -> int:
    """Whole-task payload size as docs §10 measures it, in UTF-8 bytes."""
    return sum(_column_bytes(task).values())


def task_fingerprint(task: Mapping[str, Any]) -> str:
    """Compare-and-swap token: changes whenever any column a rotation reads changes."""
    digest = hashlib.sha256()
    for column in (*_TEXT_COLUMNS, 'status'):
        digest.update((task.get(column) or '').encode())
        digest.update(b'\0')
    digest.update(json.dumps(task.get('metadata'), sort_keys=True).encode())
    return digest.hexdigest()


# A dated per-cycle key: '<stem>_YYYY_MM_DD', '_' or '-' separated, one optional letter.
_DATED_KEY_RE = re.compile(
    r'^(?P<stem>.+?)_(?P<year>\d{4})[_-](?P<month>\d{2})[_-](?P<day>\d{2})(?P<suffix>[a-z]?)$'
)
_HISTORY_SUFFIX = '_history'
_METADATA_SECTION = '=== METADATA ENTRIES ROTATED OUT (verbatim JSON) ==='
_QUERY_SECTION = '=== MEMORY HINT QUERIES ROTATED OUT ==='
_DESCRIPTION_SECTION = '=== DESCRIPTION BLOCKS ROTATED OUT (verbatim) ==='
_BLOCK_SEPARATOR_RE = re.compile(r'\n[ \t]*\n\s*')
_LEAD_MAX_CHARS = 80
# mem0 ids are UUIDs; size projections stand one in before the real id exists.
_ARCHIVE_ID_STAND_IN = '00000000-0000-0000-0000-000000000000'

Refusal = Literal[
    'terminal_status', 'metadata_not_an_object', 'rollup_not_harness_shaped', 'nothing_sheddable'
]

# Volatile tokens that make two otherwise-identical hint queries look different,
# in the order they are replaced (a UUID holds hex runs, a date holds digit runs).
_VOLATILE_TOKENS = (
    (re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'), 'uuid'),
    (re.compile(r'\d{4}[-_/]\d{2}[-_/]\d{2}'), 'date'),
    (re.compile(r'(?<![0-9a-z])(?=[0-9a-f]*\d)[0-9a-f]{7,}(?![0-9a-z])'), 'hex'),
    (re.compile(r'\d+'), 'n'),
)
_SEPARATORS_RE = re.compile(r'[\W_]+')


def near_duplicate_key(query: str) -> str:
    """Equal for hint queries differing only in case, spacing, punctuation, dates, ids, numbers."""
    key = query.casefold()
    for pattern, placeholder in _VOLATILE_TOKENS:
        key = pattern.sub(f' {placeholder} ', key)
    return _SEPARATORS_RE.sub(' ', key).strip()


def archive_link_query(archive_memory_id: str, task_id: str) -> str:
    return f'audit-trail rotation archive memory {archive_memory_id} (task {task_id})'


@dataclass(frozen=True)
class _FamilyMember:
    key: str
    recorded_on: str | None
    suffix: str = ''

    def history_entry(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        return {'source_key': self.key, 'recorded_on': self.recorded_on, 'value': metadata[self.key]}

    @property
    def age_order(self) -> tuple[str, str]:
        return (self.recorded_on or '', self.suffix)


def _parse_dated_key(key: str) -> tuple[str, _FamilyMember] | None:
    match = _DATED_KEY_RE.match(key)
    if match is None:
        return None
    try:
        recorded_on = date(int(match['year']), int(match['month']), int(match['day']))
    except ValueError:
        return None
    return match['stem'], _FamilyMember(key, recorded_on.isoformat(), match['suffix'])


def _dated_families(metadata: Mapping[str, Any]) -> dict[str, list[_FamilyMember]]:
    families: dict[str, list[_FamilyMember]] = {}
    for key in metadata:
        parsed = _parse_dated_key(key)
        if parsed is not None:
            stem, member = parsed
            families.setdefault(stem, []).append(member)
    return families


@dataclass(frozen=True)
class _Fold:
    metadata: dict[str, Any]
    history_keys: tuple[str, ...]
    folded_keys: tuple[str, ...]


def _fold_dated_families(metadata: Mapping[str, Any], owned: tuple[str, ...]) -> _Fold:
    """Fold each dated-key family into its '<stem>_history' array, newest first.

    A family folds once it has two dated members, or one when its array is
    already harness-owned.  A '<stem>_history' key the harness does not own is
    never written to, so the family stays where it is.
    """
    folded = dict(metadata)
    history_keys = set(owned)
    folded_keys: list[str] = []
    for stem, dated in _dated_families(metadata).items():
        history_key = stem + _HISTORY_SUFFIX
        is_owned = history_key in owned and isinstance(metadata.get(history_key), list)
        if history_key in metadata and not is_owned:
            continue
        if len(dated) < 2 and not is_owned:
            continue
        members = dated + ([_FamilyMember(stem, None)] if stem in metadata else [])
        members.sort(key=lambda member: member.age_order, reverse=True)
        entries = [member.history_entry(metadata) for member in members]
        folded[history_key] = entries + list(metadata.get(history_key, []))
        for member in members:
            del folded[member.key]
        history_keys.add(history_key)
        folded_keys.extend(member.key for member in members)
    return _Fold(folded, tuple(sorted(history_keys)), tuple(folded_keys))


@dataclass(frozen=True)
class _PriorRollup:
    history_keys: tuple[str, ...]
    rotations: tuple[Any, ...]


def _prior_rollup(metadata: Mapping[str, Any]) -> _PriorRollup | None:
    """The rollup this harness wrote last time, or None when its shape is not ours."""
    rollup = metadata.get(ROLLUP_KEY, {})
    if not isinstance(rollup, dict):
        return None
    history_keys = rollup.get('history_keys', [])
    rotations = rollup.get('rotations', [])
    if not isinstance(rotations, list) or not isinstance(history_keys, list):
        return None
    if not all(isinstance(key, str) for key in history_keys):
        return None
    return _PriorRollup(tuple(history_keys), tuple(rotations))


def _block_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans = []
    start = 0
    for separator in _BLOCK_SEPARATOR_RE.finditer(text):
        if separator.start() > start:
            spans.append((start, separator.start()))
        start = separator.end()
    if start < len(text):
        spans.append((start, len(text)))
    return tuple(spans)


def _lead(block: str) -> str:
    line = block.strip().splitlines()[0]
    return line if len(line) <= _LEAD_MAX_CHARS else line[: _LEAD_MAX_CHARS - 1] + '…'


@dataclass(frozen=True)
class _DescriptionCut:
    """A description split into blank-line blocks; the oldest ``shed`` middle blocks leave.

    The first block (what the task is) and the last (the newest) always stay.
    The shed span is sliced from the original text, never re-joined.
    """

    text: str
    blocks: tuple[tuple[int, int], ...]
    shed: int = 0

    @classmethod
    def of(cls, text: str) -> _DescriptionCut:
        return cls(text, _block_spans(text))

    @property
    def can_shed(self) -> bool:
        return self.shed < len(self.blocks) - 2

    @property
    def shed_span(self) -> str:
        if not self.shed:
            return ''
        return self.text[self.blocks[1][0] : self.blocks[self.shed][1]]

    @property
    def shed_leads(self) -> list[str]:
        return [_lead(self.text[start:end]) for start, end in self.blocks[1 : self.shed + 1]]

    def render(self, pointer: str) -> str:
        head = self.text[: self.blocks[0][1]]
        rest = self.text[self.blocks[self.shed + 1][0] :]
        return f'{head}\n\n{pointer}\n\n{rest}'


@dataclass(frozen=True)
class TaskRewrite:
    """The two columns a rotation writes; ``description`` None leaves it untouched."""

    description: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RotationPlan:
    """What one rotation keeps and sheds.  Built by :func:`plan_rotation` only."""

    task_id: str
    rotated_at: str
    bytes_before: int
    untouched_bytes: int
    description: _DescriptionCut
    metadata: Mapping[str, Any]
    history_keys: tuple[str, ...]
    folded_keys: tuple[str, ...]
    prior_rotations: tuple[Any, ...]
    shed_history: Mapping[str, tuple[Any, ...]] = dataclasses.field(default_factory=dict)
    shed_queries: tuple[str, ...] = ()

    @property
    def changes_task(self) -> bool:
        return bool(self.folded_keys or self._sheds_content)

    @property
    def trigger(self) -> Literal['size', 'pattern']:
        return 'size' if self.bytes_before > ROTATE_THRESHOLD_BYTES else 'pattern'

    @property
    def _sheds_content(self) -> bool:
        return bool(self.shed_history or self.shed_queries or self.description.shed)

    @property
    def _shed_rotations(self) -> tuple[Any, ...]:
        """Prior rotation records that make room for this rotation's own record."""
        return self.prior_rotations[HISTORY_KEEP - 1 :] if self._sheds_content else ()

    @property
    def archive_metadata(self) -> dict[str, Any]:
        """Mem0 metadata for the archive: blessed keys plus the x_ namespace, no 'kind'."""
        previous = self.prior_rotations[0] if self.prior_rotations else None
        return {
            'source': ARCHIVE_SOURCE,
            'task_id': self.task_id,
            'x_rotated_at': self.rotated_at,
            'x_bytes_before': self.bytes_before,
            'x_previous_archive_memory_id': (
                previous.get('archive_memory_id') if isinstance(previous, dict) else None
            ),
        }

    @property
    def archive_text(self) -> str | None:
        """Everything leaving the task, verbatim; None when nothing leaves."""
        sections = []
        if self.description.shed:
            sections.append(f'{_DESCRIPTION_SECTION}\n{self.description.shed_span}')
        shed: dict[str, Any] = {key: list(entries) for key, entries in self.shed_history.items()}
        if self._shed_rotations:
            shed[ROLLUP_KEY] = {'rotations': list(self._shed_rotations)}
        if shed:
            sections.append(f'{_METADATA_SECTION}\n{_verbatim_json(shed)}')
        if self.shed_queries:
            sections.append('\n'.join([_QUERY_SECTION, *self.shed_queries]))
        if not sections:
            return None
        header = (
            f'Audit-trail rotation archive for task {self.task_id}, rotated at {self.rotated_at}.\n'
            'Rule: docs/task-authoring.md §10.'
        )
        return '\n\n'.join([header, *sections])

    @property
    def archive_bytes(self) -> int:
        return len((self.archive_text or '').encode())

    @property
    def projected_bytes(self) -> int:
        """Whole-task payload after this plan is committed."""
        stand_in = None if self.archive_text is None else _ARCHIVE_ID_STAND_IN
        return self._rewrite_bytes(self.render(stand_in))

    @property
    def over_threshold_after(self) -> bool:
        return self.projected_bytes > ROTATE_THRESHOLD_BYTES

    def render(self, archive_memory_id: str | None) -> TaskRewrite:
        metadata = dict(self.metadata)
        if archive_memory_id is not None:
            metadata = _with_archive_link(metadata, archive_link_query(archive_memory_id, self.task_id))
        description = None
        if self.description.shed:
            description = self.description.render(self._pointer_block(archive_memory_id))
        if not self._sheds_content:
            return TaskRewrite(description, {**metadata, ROLLUP_KEY: self._rollup(self.prior_rotations)})
        kept = self.prior_rotations[: len(self.prior_rotations) - len(self._shed_rotations)]

        def with_record(bytes_after: int) -> TaskRewrite:
            record = self._rotation_record(archive_memory_id, bytes_after)
            return TaskRewrite(description, {**metadata, ROLLUP_KEY: self._rollup((record, *kept))})

        return _settle_bytes_after(with_record, self._rewrite_bytes)

    def _rollup(self, rotations: tuple[Any, ...]) -> dict[str, Any]:
        return {
            'standing_instruction': STANDING_INSTRUCTION,
            'history_keys': list(self.history_keys),
            'rotations': list(rotations),
        }

    def _rotation_record(self, archive_memory_id: str | None, bytes_after: int) -> dict[str, Any]:
        cut = self.description
        return {
            'rotated_at': self.rotated_at,
            'trigger': self.trigger,
            'archive_memory_id': archive_memory_id,
            'bytes_before': self.bytes_before,
            'bytes_after': bytes_after,
            'description_blocks_kept': len(cut.blocks) - cut.shed,
            'description_blocks_shed': cut.shed,
            'shed_block_leads': cut.shed_leads,
            'history_entries_kept': {
                key: len(self.metadata[key])
                for key in self.history_keys
                if isinstance(self.metadata.get(key), list)
            },
            'history_entries_shed': {key: len(shed) for key, shed in self.shed_history.items()},
            'folded_keys': list(self.folded_keys),
            'memory_hint_queries_shed': len(self.shed_queries),
        }

    def _pointer_block(self, archive_memory_id: str | None) -> str:
        cut = self.description
        leads = '\n'.join(f'- {lead}' for lead in cut.shed_leads)
        return (
            f'[{ROLLUP_KEY} {self.rotated_at}] {cut.shed} older description blocks '
            f'({len(cut.shed_span.encode()):,} B) moved to archive memory '
            f'{archive_memory_id or "unrecorded"}; their first lines:\n{leads}'
        )

    def _rewrite_bytes(self, rewrite: TaskRewrite) -> int:
        description = self.description.text if rewrite.description is None else rewrite.description
        rewritten = {'description': description, 'metadata': rewrite.metadata}
        return self.untouched_bytes + task_payload_bytes(rewritten)


def _settle_bytes_after(
    with_record: Callable[[int], TaskRewrite], measure: Callable[[TaskRewrite], int]
) -> TaskRewrite:
    """The rewrite whose recorded bytes_after equals its own measured size.

    The record's digits count toward the size they report; the count settles
    within one change of digit width.
    """
    bytes_after = 0
    while True:
        rewrite = with_record(bytes_after)
        measured = measure(rewrite)
        if measured == bytes_after:
            return rewrite
        bytes_after = measured


def _verbatim_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _with_archive_link(metadata: dict[str, Any], link: str) -> dict[str, Any]:
    hints = metadata.get('memory_hints', {'entities': [], 'queries': []})
    if not isinstance(hints, dict) or not isinstance(hints.get('queries', []), list):
        return metadata
    return {**metadata, 'memory_hints': {**hints, 'queries': [*hints.get('queries', []), link]}}


def _hint_queries(metadata: Mapping[str, Any]) -> list[str] | None:
    """The dict-shaped ``memory_hints.queries``; None for any shape the bound leaves alone."""
    hints = metadata.get('memory_hints')
    queries = hints.get('queries') if isinstance(hints, dict) else None
    if not isinstance(queries, list) or not all(isinstance(query, str) for query in queries):
        return None
    return queries


def _bounded_query_indexes(queries: list[str]) -> set[int]:
    """First occurrences only; the executed prefix plus the newest, up to HISTORY_KEEP."""
    seen: set[str] = set()
    unique: list[int] = []
    for index, query in enumerate(queries):
        key = near_duplicate_key(query)
        if key not in seen:
            seen.add(key)
            unique.append(index)
    executed, rest = unique[:HINT_QUERIES_EXECUTED], unique[HINT_QUERIES_EXECUTED:]
    room = max(HISTORY_KEEP - len(executed), 0)
    return {*executed, *rest[max(len(rest) - room, 0):]}


_ShedOne = Callable[[RotationPlan], RotationPlan | None]


def _bound_hint_queries(plan: RotationPlan) -> RotationPlan | None:
    queries = _hint_queries(plan.metadata)
    if queries is None:
        return None
    kept = _bounded_query_indexes(queries)
    if len(kept) == len(queries):
        return None
    hints = {
        **plan.metadata['memory_hints'],
        'queries': [query for index, query in enumerate(queries) if index in kept],
    }
    shed = tuple(query for index, query in enumerate(queries) if index not in kept)
    return dataclasses.replace(
        plan,
        metadata={**plan.metadata, 'memory_hints': hints},
        shed_queries=(*plan.shed_queries, *shed),
    )


def _history_trim(history_key: str) -> _ShedOne:
    """Sheds the oldest entry of an owned array, newest first in the archive, down to KEEP."""

    def shed_one(plan: RotationPlan) -> RotationPlan | None:
        entries = plan.metadata.get(history_key)
        if not isinstance(entries, list) or len(entries) <= HISTORY_KEEP:
            return None
        shed = dict(plan.shed_history)
        shed[history_key] = (entries[-1], *shed.get(history_key, ()))
        return dataclasses.replace(
            plan, metadata={**plan.metadata, history_key: entries[:-1]}, shed_history=shed
        )

    return shed_one


def _shed_description_block(plan: RotationPlan) -> RotationPlan | None:
    cut = plan.description
    if not cut.can_shed:
        return None
    return dataclasses.replace(plan, description=dataclasses.replace(cut, shed=cut.shed + 1))


@dataclass(frozen=True)
class _ShedPolicy:
    shed_one: _ShedOne
    stops_at_target: bool


def _shed_policies(plan: RotationPlan) -> list[_ShedPolicy]:
    """Pattern bounds always apply; the size bound only above the threshold, toward the target."""
    owned = [key for key in plan.history_keys if isinstance(plan.metadata.get(key), list)]
    policies = [
        _ShedPolicy(_history_trim(key), stops_at_target=False)
        for key in owned
        if len(plan.metadata[key]) > HISTORY_MAX
    ]
    if len(_hint_queries(plan.metadata) or ()) > HISTORY_MAX:
        policies.append(_ShedPolicy(_bound_hint_queries, stops_at_target=False))
    if plan.bytes_before > ROTATE_THRESHOLD_BYTES:
        policies.extend(_ShedPolicy(_history_trim(key), stops_at_target=True) for key in owned)
        policies.append(_ShedPolicy(_bound_hint_queries, stops_at_target=True))
        policies.append(_ShedPolicy(_shed_description_block, stops_at_target=True))
    return policies


def _shed_within_archive_budget(plan: RotationPlan, policies: list[_ShedPolicy]) -> RotationPlan:
    """Apply each policy one unit at a time, oldest first; stop for good at the archive budget."""
    for policy in policies:
        while not (policy.stops_at_target and plan.projected_bytes <= ROTATE_TARGET_BYTES):
            candidate = policy.shed_one(plan)
            if candidate is None:
                break
            if candidate.archive_bytes > ARCHIVE_MAX_BYTES:
                return plan
            plan = candidate
    return plan


def _plan_or_refusal(task: Mapping[str, Any], now: datetime) -> RotationPlan | Refusal:
    if task.get('status') in TERMINAL:
        return 'terminal_status'
    metadata = task.get('metadata')
    if not isinstance(metadata, dict):
        return 'metadata_not_an_object'
    prior = _prior_rollup(metadata)
    if prior is None:
        return 'rollup_not_harness_shaped'
    working = {key: value for key, value in metadata.items() if key != ROLLUP_KEY}
    fold = _fold_dated_families(working, prior.history_keys)
    columns = _column_bytes(task)
    draft = RotationPlan(
        task_id=str(task.get('id')),
        rotated_at=now.isoformat(),
        bytes_before=sum(columns.values()),
        untouched_bytes=columns['title'] + columns['details'],
        description=_DescriptionCut.of(task.get('description') or ''),
        metadata=fold.metadata,
        history_keys=fold.history_keys,
        folded_keys=fold.folded_keys,
        prior_rotations=prior.rotations,
    )
    plan = _shed_within_archive_budget(draft, _shed_policies(draft))
    return plan if plan.changes_task else 'nothing_sheddable'


def plan_rotation(task: Mapping[str, Any], *, now: datetime) -> RotationPlan | None:
    """Plan a rotation of ``task``, or None when there is nothing to bound."""
    plan = _plan_or_refusal(task, now)
    return plan if isinstance(plan, RotationPlan) else None


def unrotatable_reason(task: Mapping[str, Any], *, now: datetime) -> dict[str, Any] | None:
    """Why ``plan_rotation(task, now=now)`` is None for a task over the threshold."""
    plan = _plan_or_refusal(task, now)
    return None if isinstance(plan, RotationPlan) else _unrotatable_report(task, plan)


def _unrotatable_report(task: Mapping[str, Any], refusal: Refusal) -> dict[str, Any] | None:
    columns = _column_bytes(task)
    payload = sum(columns.values())
    if payload <= ROTATE_THRESHOLD_BYTES:
        return None
    return {
        'reason': refusal,
        'payload_bytes': payload,
        'threshold_bytes': ROTATE_THRESHOLD_BYTES,
        'column_bytes': columns,
    }


class AuditTrailArchive(Protocol):
    """Where rotated-out text goes; ``read`` must return exactly what ``write`` stored."""

    async def write(
        self, *, project_id: str, content: str, metadata: dict[str, Any]
    ) -> str | None: ...

    async def read(self, *, project_id: str, memory_id: str) -> str | None: ...


class AuditTrailCommit(Protocol):
    """Writes the rewrite iff the task still has ``expected_fingerprint``; else returns None."""

    async def __call__(
        self, *, expected_fingerprint: str, description: str | None, metadata: dict[str, Any]
    ) -> Mapping[str, Any] | None: ...


RotationStatus = Literal['rotated', 'folded', 'archive_unconfirmed', 'superseded', 'unrotatable']


@dataclass(frozen=True)
class RotationOutcome:
    status: RotationStatus
    task_id: str
    bytes_before: int
    bytes_after: int | None = None
    archive_memory_id: str | None = None
    over_threshold_after: bool | None = None
    unrotatable: Mapping[str, Any] | None = None
    committed_task: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """The wire form; the committed task travels separately as ``updated_task``."""
        return {
            'status': self.status,
            'task_id': self.task_id,
            'bytes_before': self.bytes_before,
            'bytes_after': self.bytes_after,
            'archive_memory_id': self.archive_memory_id,
            'over_threshold_after': self.over_threshold_after,
            'unrotatable': None if self.unrotatable is None else dict(self.unrotatable),
        }


async def bound_audit_trail(
    task: Mapping[str, Any],
    *,
    project_id: str,
    now: datetime,
    archive: AuditTrailArchive,
    commit: AuditTrailCommit,
) -> RotationOutcome | None:
    """Rotate ``task``: archive and confirm the read-back first, then commit by fingerprint.

    None when there is nothing to bound.  Nothing leaves the task unless the
    archive read back byte-identical (docs §10 step 1).
    """
    plan = _plan_or_refusal(task, now)
    task_id = str(task.get('id'))
    if not isinstance(plan, RotationPlan):
        report = _unrotatable_report(task, plan)
        if report is None:
            return None
        return RotationOutcome('unrotatable', task_id, report['payload_bytes'], unrotatable=report)
    archive_memory_id = None
    if plan.archive_text is not None:
        archive_memory_id = await archive.write(
            project_id=project_id, content=plan.archive_text, metadata=plan.archive_metadata
        )
        landed = archive_memory_id is not None and plan.archive_text == await archive.read(
            project_id=project_id, memory_id=archive_memory_id
        )
        if not landed:
            return RotationOutcome(
                'archive_unconfirmed', task_id, plan.bytes_before, archive_memory_id=archive_memory_id
            )
    rewrite = plan.render(archive_memory_id)
    committed = await commit(
        expected_fingerprint=task_fingerprint(task),
        description=rewrite.description,
        metadata=rewrite.metadata,
    )
    if committed is None:
        return RotationOutcome(
            'superseded', task_id, plan.bytes_before, archive_memory_id=archive_memory_id
        )
    bytes_after = task_payload_bytes(committed)
    return RotationOutcome(
        'folded' if archive_memory_id is None else 'rotated',
        task_id,
        plan.bytes_before,
        bytes_after=bytes_after,
        archive_memory_id=archive_memory_id,
        over_threshold_after=bytes_after > ROTATE_THRESHOLD_BYTES,
        committed_task=committed,
    )


@dataclass(frozen=True)
class _MemoryServiceArchive:
    memory_service: Any

    async def write(self, *, project_id: str, content: str, metadata: dict[str, Any]) -> str | None:
        response = await self.memory_service.add_memory(
            content=content,
            category='observations_and_summaries',
            project_id=project_id,
            agent_id=ARCHIVE_AGENT_ID,
            metadata=dict(metadata),
            _source=ARCHIVE_SOURCE,
        )
        return response.memory_ids[0] if response.memory_ids else None

    async def read(self, *, project_id: str, memory_id: str) -> str | None:
        record = await self.memory_service.get_memory_by_id(project_id, memory_id)
        return None if record is None else record['content']


def memory_service_archive(memory_service: Any) -> AuditTrailArchive:
    """The archive port over a duck-typed MemoryService.

    add_memory reports a mem0 failure as empty ``memory_ids``, so ``write``
    returns None for it.  ``get_memory_by_id`` takes ``project_id`` FIRST and
    positionally; its TimeoutError propagates rather than reading as a miss.
    """
    return _MemoryServiceArchive(memory_service)
