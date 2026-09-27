"""Enumerate every question awaiting a human across the fleet from both stores, folded onto one project token.

The two stores are the per-project escalation queues (their pending level-2
records) and the fleet-global decision registry (its ``open`` records). An item
is scoped on two axes, project and queue, as
``skills/escalation-watcher/SKILL.md::Filing Parked Decisions to the Cockpit Registry (C8)``
lays down: escalation ids are unique only within one queue, so a decision folds
into a pending L2 only when its recorded queue AND its escalation id both match.

Read-only. Queues are read through the module-level scan helpers of
``escalation.queue``, never through ``EscalationQueue``, whose constructor
creates the directory it is handed. Only a queue's root tier is parsed: its
``archive/`` subtree holds resolved records by construction, so it is indexed
by filename (the id) and read only when a citation asks for it.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from _task_db_scan import TaskDbUnreadable, connect_ro, discover_project_roots, tasks_db_path
from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths, read_escalation_for_scan
from orchestrator.session_registry import (
    UNKNOWN_QUEUE,
    DecisionRecord,
    DecisionState,
    decision_path_for_id,
    decisions_dir,
    list_decisions,
    normalize_escalations_dir,
    normalize_project_token,
)

QUEUE_TAGS: Mapping[Path, str] = MappingProxyType({
    Path('data', 'escalations'): '',
    Path('data', 'reconciliation', 'escalations'): 'recon',
})
QUEUE_SUBDIRS: tuple[Path, ...] = tuple(QUEUE_TAGS)

ESC_ID_RE = re.compile(r'\besc-(?:[A-Za-z0-9_]+-)+\d+\b')
TASK_CITATION_RE = re.compile(r'\btask (\d+)\b', re.IGNORECASE)

_SCAN_CONTEXT = 'sitting.inventory'
_SCAN_PARSE_ERRORS: tuple[type[BaseException], ...] = (json.JSONDecodeError, KeyError, TypeError, ValueError)
_KEY_ARITY = {'esc': 3, 'decision': 2}

ItemKey = tuple[str, ...]


@dataclass(frozen=True)
class Shortfall:
    """One record or store this run could not read; counted, never only logged."""

    source: str
    path: str
    reason: str


@dataclass(frozen=True)
class OpenItem:
    """One question awaiting a human. ``key`` is its identity; see ``escalation_key`` / ``decision_key``.

    ``decision_project`` is the canonical project of the DecisionRecord named by
    ``decision_id``, as read: the compare-and-swap expectation a close payload
    sends. It is not ``project``, which is derived from the queue for an
    escalation item, and dark-factory's recon queue carries other projects'
    gates, so the two legitimately differ.
    """

    key: ItemKey
    decision_id: str | None = None
    decision_project: str = ''
    escalation_id: str | None = None
    queue_dir: str = ''
    project: str = ''
    task_id: str | None = None
    severity: str = ''
    text: str = ''
    options: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    root_cause: str = ''
    pin_declared_by: tuple[str, ...] = ()
    triage_note: str = ''
    filed_at: str = ''
    age_days: float | None = None

    def __post_init__(self) -> None:
        if self.decision_id is None and self.decision_project:
            raise ValueError(f'{self.key}: decision_project {self.decision_project!r} names no decision_id')

    @property
    def kind(self) -> str:
        return self.key[0]


@dataclass(frozen=True)
class Inventory:
    items: tuple[OpenItem, ...]
    escalation_index: Mapping[str, Mapping[str, Path]]
    shortfalls: tuple[Shortfall, ...]


@dataclass(frozen=True)
class Glossary:
    """Short glosses for cited ids, each scoped to its own namespace: queue for escalations, project for tasks."""

    escalations: Mapping[str, Mapping[str, str]]
    tasks: Mapping[str, Mapping[str, str]]
    shortfalls: tuple[Shortfall, ...]


def escalation_key(queue_dir: str, escalation_id: str) -> ItemKey:
    return ('esc', queue_dir, escalation_id)


def decision_key(decision_id: str) -> ItemKey:
    return ('decision', decision_id)


def key_str(key: ItemKey) -> str:
    return json.dumps(list(key))


def parse_key(text: str) -> ItemKey:
    parts = json.loads(text)
    well_formed = (
        isinstance(parts, list)
        and all(isinstance(part, str) for part in parts)
        and len(parts) == _KEY_ARITY.get(parts[0] if parts else '', -1)
    )
    if not well_formed:
        raise ValueError(f'not an open-item key: {text!r}')
    return tuple(parts)


def _known_queue_subdir(path: Path) -> Path | None:
    for subdir in QUEUE_SUBDIRS:
        depth = len(subdir.parts)
        if path.parts[-depth:] == subdir.parts and len(path.parts) > depth:
            return subdir
    return None


def queue_project_root(queue_dir: str) -> Path | None:
    path = Path(queue_dir)
    subdir = _known_queue_subdir(path)
    return path.parents[len(subdir.parts) - 1] if subdir is not None else None


def queue_tag(queue_dir: str) -> str:
    """A decision-id-safe tag naming *queue_dir*'s shape: its ``QUEUE_TAGS`` entry, else a digest of its path."""
    normalized = normalize_escalations_dir(queue_dir)
    subdir = _known_queue_subdir(Path(normalized))
    if subdir is not None:
        return QUEUE_TAGS[subdir]
    return 'q' + hashlib.sha256(normalized.encode()).hexdigest()[:10]


def queue_project(queue_dir: str) -> str:
    root = queue_project_root(queue_dir)
    return normalize_project_token(root.name) if root is not None else ''


def escalation_queue_dirs(
    project_roots: Sequence[str] | None,
    decisions: Iterable[DecisionRecord],
) -> tuple[str, ...]:
    """Every queue dir that exists: each root's queues, then each queue the registry recorded.

    *project_roots* ``None`` means ``_task_db_scan.discover_project_roots()``.
    """
    roots = discover_project_roots() if project_roots is None else list(project_roots)
    candidates: list[str | Path] = [Path(root) / subdir for root in roots for subdir in QUEUE_SUBDIRS]
    candidates += [record.escalations_dir for record in decisions]
    found: dict[str, None] = {}
    for candidate in candidates:
        queue_dir = normalize_escalations_dir(candidate)
        if queue_dir not in ('', UNKNOWN_QUEUE) and Path(queue_dir).is_dir():
            found.setdefault(queue_dir, None)
    return tuple(found)


def collect_open_items(
    *,
    queue_dirs: Iterable[str],
    decisions_root: Path | str | None,
    now: datetime,
    project: str | None = None,
) -> Inventory:
    """Pending L2 escalations plus open decisions, deduped on (queue, escalation id).

    *decisions_root* is the fleet root holding ``decisions/`` (see
    ``session_registry.fleet_root``). *project*, when given, is folded through
    ``normalize_project_token`` and keeps only that project's items.
    """
    if now.tzinfo is None:
        raise ValueError('now must be timezone-aware')
    shortfalls: list[Shortfall] = []
    index: dict[str, Mapping[str, Path]] = {}
    items: dict[ItemKey, OpenItem] = {}
    for queue_dir in dict.fromkeys(normalize_escalations_dir(q) for q in queue_dirs):
        queue_index, pending, queue_shortfalls = _scan_queue(queue_dir)
        index[queue_dir] = MappingProxyType(queue_index)
        shortfalls += queue_shortfalls
        for esc in pending:
            item = _escalation_item(queue_dir, esc, now)
            items[item.key] = item

    decisions, decision_shortfalls = read_decisions(decisions_root)
    shortfalls += decision_shortfalls
    for record in decisions:
        if record.state != DecisionState.OPEN:
            continue
        queue_dir = normalize_escalations_dir(record.escalations_dir)
        host_key = escalation_key(queue_dir, record.escalation_id) if record.escalation_id else None
        host = items.get(host_key) if host_key is not None else None
        if host_key is not None and host is not None and host.decision_id is None:
            items[host_key] = replace(
                host, decision_id=record.id, decision_project=normalize_project_token(record.project)
            )
        else:
            item = _decision_item(record, queue_dir, now)
            items[item.key] = item

    selected = items.values()
    if project is not None:
        wanted = normalize_project_token(project)
        selected = [item for item in selected if item.project == wanted]
    return Inventory(
        items=tuple(sorted(selected, key=_oldest_first)),
        escalation_index=MappingProxyType(index),
        shortfalls=tuple(shortfalls),
    )


def cited_ids(text: str) -> tuple[frozenset[str], frozenset[str]]:
    """(escalation ids, task ids) cited in *text*."""
    return frozenset(ESC_ID_RE.findall(text)), frozenset(TASK_CITATION_RE.findall(text))


def item_citations(item: OpenItem) -> tuple[frozenset[str], frozenset[str]]:
    esc_ids = {cited for cited in (item.escalation_id, *item.members) if cited}
    task_ids = {item.task_id} if item.task_id else set()
    for text in (item.text, item.triage_note, item.root_cause, *item.options):
        esc_cited, task_cited = cited_ids(text)
        esc_ids |= esc_cited
        task_ids |= task_cited
    return frozenset(esc_ids), frozenset(task_ids)


def build_glossary(items: Iterable[OpenItem], escalation_index: Mapping[str, Mapping[str, Path]]) -> Glossary:
    """Gloss every id the items cite: escalation id -> summary, task id -> tasks.db title."""
    esc_wanted: dict[str, set[str]] = defaultdict(set)
    task_wanted: dict[str, set[str]] = defaultdict(set)
    for item in items:
        esc_ids, task_ids = item_citations(item)
        esc_wanted[item.queue_dir] |= esc_ids
        task_wanted[item.project] |= task_ids
    return gloss(esc_wanted, task_wanted, escalation_index)


def gloss(
    esc_wanted: Mapping[str, Iterable[str]],
    task_wanted: Mapping[str, Iterable[str]],
    escalation_index: Mapping[str, Mapping[str, Path]],
) -> Glossary:
    """Gloss escalation ids per queue dir and task ids per project; a scope the index cannot place is skipped."""
    esc_scoped = {queue: set(ids) for queue, ids in esc_wanted.items() if queue in escalation_index}
    task_scoped = {project: set(ids) for project, ids in task_wanted.items() if project}
    shortfalls: list[Shortfall] = []
    escalations: dict[str, Mapping[str, str]] = {}
    for queue_dir, esc_ids in esc_scoped.items():
        if not esc_ids:
            continue
        summaries, queue_shortfalls = _summaries(escalation_index[queue_dir], esc_ids)
        escalations[queue_dir] = MappingProxyType(summaries)
        shortfalls += queue_shortfalls

    roots = {queue_project(q): root for q in escalation_index if (root := queue_project_root(q)) is not None}
    tasks: dict[str, Mapping[str, str]] = {}
    for project, task_ids in task_scoped.items():
        if not task_ids:
            continue
        root = roots.get(project)
        if root is None:
            shortfalls.append(Shortfall('tasks_db', '', f'no queue in the index names a root for {project!r}'))
            continue
        titles, problem = _task_titles(root, task_ids)
        if problem is not None:
            shortfalls.append(problem)
        else:
            tasks[project] = MappingProxyType(titles)
    return Glossary(
        escalations=MappingProxyType(escalations),
        tasks=MappingProxyType(tasks),
        shortfalls=tuple(shortfalls),
    )


def parse_stamp(stamp: str | None) -> datetime | None:
    """An ISO-8601 stamp as an aware datetime (naive reads as UTC), or None when it does not parse."""
    try:
        parsed = datetime.fromisoformat(stamp or '')
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def age_days(stamp: str, now: datetime) -> float | None:
    filed = parse_stamp(stamp)
    return None if filed is None else (now - filed).total_seconds() / 86400


def _oldest_first(item: OpenItem) -> tuple[bool, float, str]:
    return (item.age_days is None, -(item.age_days or 0.0), key_str(item.key))


def read_escalation(path: Path) -> tuple[Escalation | None, Shortfall | None]:
    """One record read unlocked; a vanished file is None with no shortfall, an unreadable one is a shortfall."""
    esc, outcome = read_escalation_for_scan(path, context=_SCAN_CONTEXT, parse_errors=_SCAN_PARSE_ERRORS)
    if esc is None and outcome != 'vanished':
        return None, Shortfall('escalation_queue', str(path), outcome)
    return esc, None


def _scan_queue(queue_dir: str) -> tuple[dict[str, Path], list[Escalation], list[Shortfall]]:
    root = Path(queue_dir)
    index: dict[str, Path] = {}
    pending: list[Escalation] = []
    shortfalls: list[Shortfall] = []
    for path in iter_all_escalation_paths(root):
        index[path.stem] = path
        if path.parent != root:
            continue
        esc, problem = read_escalation(path)
        if problem is not None:
            shortfalls.append(problem)
        if esc is not None and esc.status == 'pending' and esc.level == 2:
            pending.append(esc)
    return index, pending, shortfalls


def read_decisions(fleet: Path | str | None) -> tuple[list[DecisionRecord], list[Shortfall]]:
    """Every record under *fleet*'s ``decisions/``, plus a shortfall for each file ``list_decisions`` skipped."""
    # Listed BEFORE the read: a decision filed in between is then merely unlisted,
    # never misreported as unreadable.
    on_disk = sorted(decisions_dir(fleet).glob('*.json'))
    records = list_decisions(fleet)
    read = {decision_path_for_id(record.id, root=fleet) for record in records}
    shortfalls = [
        Shortfall('decision_registry', str(path), 'skipped by list_decisions')
        for path in on_disk if path not in read
    ]
    return records, shortfalls


def _escalation_item(queue_dir: str, esc: Escalation, now: datetime) -> OpenItem:
    return OpenItem(
        key=escalation_key(queue_dir, esc.id),
        escalation_id=esc.id,
        queue_dir=queue_dir,
        project=queue_project(queue_dir),
        task_id=esc.task_id or None,
        severity=esc.severity,
        text=esc.summary,
        options=tuple(esc.options),
        members=tuple(esc.members),
        root_cause=esc.root_cause,
        pin_declared_by=tuple(esc.pin_declared_by),
        triage_note=esc.triage_note,
        filed_at=esc.timestamp,
        age_days=age_days(esc.timestamp, now),
    )


def _decision_item(record: DecisionRecord, queue_dir: str, now: datetime) -> OpenItem:
    return OpenItem(
        key=decision_key(record.id),
        decision_id=record.id,
        decision_project=normalize_project_token(record.project),
        escalation_id=record.escalation_id,
        queue_dir=queue_dir,
        project=normalize_project_token(record.project),
        task_id=record.task_id,
        severity=record.severity,
        text=record.text,
        options=tuple(record.options or ()),
        filed_at=record.filed_at,
        age_days=age_days(record.filed_at, now),
    )


def _summaries(queue_index: Mapping[str, Path], esc_ids: Iterable[str]) -> tuple[dict[str, str], list[Shortfall]]:
    summaries: dict[str, str] = {}
    shortfalls: list[Shortfall] = []
    for esc_id in sorted(esc_ids):
        path = queue_index.get(esc_id)
        if path is None:
            continue
        esc, problem = read_escalation(path)
        if problem is not None:
            shortfalls.append(problem)
        if esc is not None:
            summaries[esc_id] = esc.summary
    return summaries, shortfalls


def _task_titles(root: Path, task_ids: Iterable[str]) -> tuple[dict[str, str], Shortfall | None]:
    db = tasks_db_path(str(root))
    numeric = sorted(int(task_id) for task_id in task_ids if task_id.isdigit())
    try:
        conn = connect_ro(db)
    except TaskDbUnreadable as exc:
        return {}, Shortfall('tasks_db', str(exc.path), exc.reason.value)
    except sqlite3.Error as exc:
        return {}, Shortfall('tasks_db', str(db), str(exc))
    try:
        placeholders = ','.join('?' * len(numeric))
        rows = conn.execute(
            f"SELECT id, title FROM tasks WHERE tag = 'master' AND id IN ({placeholders})", numeric,
        ).fetchall()
    except sqlite3.Error as exc:
        return {}, Shortfall('tasks_db', str(db), str(exc))
    finally:
        conn.close()
    return {str(task_id): title for task_id, title in rows}, None
