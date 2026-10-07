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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from shared.task_statuses import TERMINAL

from fused_memory.reconciliation.context_assembler import HINT_QUERIES_EXECUTED

__all__ = [
    'ROTATE_THRESHOLD_BYTES',
    'ROTATE_TARGET_BYTES',
    'HISTORY_KEEP',
    'HISTORY_MAX',
    'ARCHIVE_MAX_BYTES',
    'ROLLUP_KEY',
    'RotationPlan',
    'TaskRewrite',
    'near_duplicate_key',
    'plan_rotation',
    'task_fingerprint',
    'task_payload_bytes',
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

_TEXT_COLUMNS = ('title', 'description', 'details')


def task_payload_bytes(task: Mapping[str, Any]) -> int:
    """Whole-task payload size as docs §10 measures it, in UTF-8 bytes."""
    text_bytes = sum(len((task.get(column) or '').encode()) for column in _TEXT_COLUMNS)
    metadata = task.get('metadata')
    metadata_bytes = 0 if metadata is None else len(json.dumps(metadata).encode())
    return text_bytes + metadata_bytes


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
    metadata: Mapping[str, Any]
    history_keys: tuple[str, ...]
    folded_keys: tuple[str, ...]
    prior_rotations: tuple[Any, ...]
    shed_history: Mapping[str, tuple[Any, ...]] = dataclasses.field(default_factory=dict)
    shed_queries: tuple[str, ...] = ()

    @property
    def archive_text(self) -> str | None:
        """Everything leaving the task, verbatim; None when nothing leaves."""
        sections = []
        if self.shed_history:
            shed = {key: list(entries) for key, entries in self.shed_history.items()}
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

    def render(self, archive_memory_id: str | None) -> TaskRewrite:
        metadata = dict(self.metadata)
        if archive_memory_id is not None:
            metadata = _with_archive_link(metadata, archive_link_query(archive_memory_id, self.task_id))
        metadata[ROLLUP_KEY] = {
            'history_keys': list(self.history_keys),
            'rotations': list(self.prior_rotations),
        }
        return TaskRewrite(description=None, metadata=metadata)


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


def _bound_hint_queries(plan: RotationPlan) -> RotationPlan:
    queries = _hint_queries(plan.metadata)
    if queries is None:
        return plan
    kept = _bounded_query_indexes(queries)
    if len(kept) == len(queries):
        return plan
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


def _shed_history_entry(plan: RotationPlan, history_key: str) -> RotationPlan:
    """Move the oldest entry of an owned history array into the archive, newest first."""
    entries = list(plan.metadata[history_key])
    oldest = entries.pop()
    shed = dict(plan.shed_history)
    shed[history_key] = (oldest, *shed.get(history_key, ()))
    return dataclasses.replace(
        plan, metadata={**plan.metadata, history_key: entries}, shed_history=shed
    )


def _owned_arrays(plan: RotationPlan) -> list[tuple[str, int]]:
    return [
        (key, len(plan.metadata[key]))
        for key in plan.history_keys
        if isinstance(plan.metadata.get(key), list)
    ]


def _trim_owned_arrays(plan: RotationPlan, *, over: int, keep: int) -> RotationPlan:
    for history_key, length in _owned_arrays(plan):
        if length <= over:
            continue
        for _ in range(length - keep):
            plan = _shed_history_entry(plan, history_key)
    return plan


def plan_rotation(task: Mapping[str, Any], *, now: datetime) -> RotationPlan | None:
    """Plan a rotation of ``task``, or None when there is nothing to bound."""
    if task.get('status') in TERMINAL:
        return None
    metadata = task.get('metadata')
    if not isinstance(metadata, dict):
        return None
    prior = _prior_rollup(metadata)
    if prior is None:
        return None
    working = {key: value for key, value in metadata.items() if key != ROLLUP_KEY}
    fold = _fold_dated_families(working, prior.history_keys)
    plan = RotationPlan(
        task_id=str(task.get('id')),
        rotated_at=now.isoformat(),
        metadata=fold.metadata,
        history_keys=fold.history_keys,
        folded_keys=fold.folded_keys,
        prior_rotations=prior.rotations,
    )
    plan = _trim_owned_arrays(plan, over=HISTORY_MAX, keep=HISTORY_KEEP)
    if len(_hint_queries(plan.metadata) or ()) > HISTORY_MAX:
        plan = _bound_hint_queries(plan)
    if not plan.folded_keys and not plan.shed_history and not plan.shed_queries:
        return None
    return plan
