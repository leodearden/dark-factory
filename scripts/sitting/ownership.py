"""Find whoever already owns a pending question before it is put to Leo: amendment A's ownership and in-flight sweep.

Routing rule: a probe that answers ``owns`` routes the item to the brief's
standing footer; ``mentions`` is evidence the agent must read before putting the
item to Leo. The asymmetry is deliberate. Every ``owns`` source is a structured
field (a session record and its ``result.md`` ``outcome:`` header,
``x_coalesced_into``, the ruling keys, the follow-up keys); the one
``mentions``-only source, the handover file, is prose, and routing on its
headings would be an ad-hoc parser.

``unavailable`` means the store could not be read and is never counted as
``empty``, so a missing sessions root can never read as "nothing owns this".
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from _task_db_scan import TaskDbUnreadable, connect_ro, tasks_db_path
from orchestrator.session_registry import (
    RESULT_FILENAME,
    TERMINAL_STATUSES,
    SessionRecord,
    normalize_project_token,
)
from sitting.inventory import ItemKey, OpenItem, Shortfall, cited_ids, parse_stamp

ProbeStatus = Literal['owns', 'mentions', 'empty', 'unavailable']

PROBES: tuple[str, ...] = (
    'spawned_session', 'unblock_run', 'coalesce_fold', 'task_ruling', 'spawned_followup', 'handover',
)
RULING_METADATA_KEYS: tuple[str, ...] = ('x_ruling', 'x_operator_ruling', 'x_ruled_by')
COALESCE_KEY = 'x_coalesced_into'
FOLLOWUP_KEYS: tuple[str, ...] = ('x_origin_escalation', 'origin_escalation')
HANDOVER_CANDIDATES: tuple[str, ...] = ('data/escalations/l2-handover.md', 'plans/l2-watcher-handover-*.md')

RESULT_OUTCOMES: tuple[str, ...] = ('done', 'blocked', 'abandoned', 'handed-off')
OWNING_OUTCOMES = frozenset({'done', 'handed-off'})
TERMINAL_TASK_STATUSES = frozenset({'done', 'cancelled'})
UNBLOCK_ROLE = 'unblock'


@dataclass(frozen=True)
class ProbeResult:
    probe: str
    status: ProbeStatus
    measured_at: str
    owner: str = ''
    evidence: str = ''
    owner_task_id: str | None = None
    owner_task_status: str = ''


@dataclass(frozen=True)
class OwnershipFinding:
    item_key: ItemKey
    probes: tuple[ProbeResult, ...]

    @property
    def owners(self) -> tuple[ProbeResult, ...]:
        return tuple(result for result in self.probes if result.status == 'owns')

    @property
    def owned(self) -> bool:
        return bool(self.owners)

    @property
    def empty_probes(self) -> tuple[str, ...]:
        return tuple(result.probe for result in self.probes if result.status == 'empty')


@dataclass(frozen=True)
class IndexedSession:
    slug: str
    record: SessionRecord
    record_dir: Path


@dataclass(frozen=True)
class SessionIndex:
    root: Path
    available: bool
    by_escalation: Mapping[str, tuple[IndexedSession, ...]]
    by_task: Mapping[tuple[str, str], tuple[IndexedSession, ...]]
    shortfalls: tuple[Shortfall, ...] = ()


@dataclass(frozen=True)
class TaskRow:
    id: str
    status: str
    title: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TaskRows:
    rows: Mapping[str, TaskRow]
    unavailable: str = ''


def index_sessions(sessions_root: Path) -> SessionIndex:
    """Read every session record once, indexed by escalation id and by (canonical project, task id)."""
    root = Path(sessions_root)
    if not root.is_dir():
        return SessionIndex(root, False, MappingProxyType({}), MappingProxyType({}))
    by_escalation: dict[str, list[IndexedSession]] = defaultdict(list)
    by_task: dict[tuple[str, str], list[IndexedSession]] = defaultdict(list)
    shortfalls: list[Shortfall] = []
    for path in sorted(root.glob('*/record.json')):
        try:
            record = SessionRecord.from_json(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            continue
        except (OSError, ValueError, KeyError, TypeError) as exc:
            shortfalls.append(Shortfall('session_registry', str(path), type(exc).__name__))
            continue
        session = IndexedSession(record.session_slug, record, path.parent)
        if record.escalation_id:
            by_escalation[record.escalation_id].append(session)
        if record.task_id:
            by_task[(normalize_project_token(record.project), str(record.task_id))].append(session)
    return SessionIndex(
        root, True,
        MappingProxyType({key: tuple(found) for key, found in by_escalation.items()}),
        MappingProxyType({key: tuple(found) for key, found in by_task.items()}),
        tuple(shortfalls),
    )


def parse_result_outcome(text: str) -> str | None:
    """The spawn contract's ``outcome:`` header value, or None when absent or not a contract outcome."""
    lines = text.lstrip().splitlines()
    if lines and lines[0].strip() == '---':
        lines = lines[1:]
    for line in lines:
        if not line.strip() or line.strip() == '---':
            return None
        key, sep, value = line.partition(':')
        if sep and key.strip() == 'outcome':
            outcome = value.strip()
            return outcome if outcome in RESULT_OUTCOMES else None
    return None


def load_task_rows(project_root: Path | str, task_ids: Iterable[str] | None = None) -> TaskRows:
    """Task rows (all of them, or just *task_ids*) from the project's tasks.db, metadata parsed once."""
    db = tasks_db_path(str(project_root))
    try:
        conn = connect_ro(db)
    except TaskDbUnreadable as exc:
        return TaskRows(MappingProxyType({}), str(exc))
    except sqlite3.Error as exc:
        return TaskRows(MappingProxyType({}), f'{db}: {exc}')
    sql = "SELECT id, status, title, metadata FROM tasks WHERE tag = 'master'"
    params: list[int] = []
    if task_ids is not None:
        params = sorted({int(task_id) for task_id in task_ids if str(task_id).isdigit()})
        sql += f" AND id IN ({','.join('?' * len(params))})"
    try:
        fetched = conn.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        return TaskRows(MappingProxyType({}), f'{db}: {exc}')
    finally:
        conn.close()
    return TaskRows(MappingProxyType({
        str(task_id): TaskRow(str(task_id), status, title, _parse_metadata(raw))
        for task_id, status, title, raw in fetched
    }))


def resolve_handover_path(repo_root: Path) -> Path | None:
    """The first ``HANDOVER_CANDIDATES`` pattern that matches a file; the newest-named file wins within a glob."""
    for pattern in HANDOVER_CANDIDATES:
        matches = sorted(path for path in Path(repo_root).glob(pattern) if path.is_file())
        if matches:
            return matches[-1]
    return None


def sweep(
    item: OpenItem,
    *,
    sessions: SessionIndex,
    task_rows: TaskRows,
    handover_path: Path | None,
    now: datetime,
) -> OwnershipFinding:
    measured_at = now.isoformat()
    return OwnershipFinding(item.key, (
        _probe_spawned_session(item, sessions, measured_at),
        _probe_unblock_run(item, sessions, measured_at),
        _probe_coalesce_fold(item, task_rows, measured_at),
        _probe_task_ruling(item, task_rows, measured_at),
        _probe_spawned_followup(item, task_rows, measured_at),
        _probe_handover(item, handover_path, measured_at),
    ))


def _probe_spawned_session(item: OpenItem, sessions: SessionIndex, measured_at: str) -> ProbeResult:
    probe = 'spawned_session'
    if not sessions.available:
        return ProbeResult(probe, 'unavailable', measured_at, evidence=f'no sessions root at {sessions.root}')
    if not item.escalation_id:
        return ProbeResult(probe, 'empty', measured_at, evidence='the item carries no escalation id')
    matches = [s for s in sessions.by_escalation.get(item.escalation_id, ()) if _same_project(s, item)]
    return _strongest_session(probe, matches, measured_at, f'no session record names {item.escalation_id}')


def _probe_unblock_run(item: OpenItem, sessions: SessionIndex, measured_at: str) -> ProbeResult:
    probe = 'unblock_run'
    if not sessions.available:
        return ProbeResult(probe, 'unavailable', measured_at, evidence=f'no sessions root at {sessions.root}')
    if not item.task_id:
        return ProbeResult(probe, 'empty', measured_at, evidence='the item names no task')
    filed = parse_stamp(item.filed_at)
    matches = [
        s for s in sessions.by_task.get((item.project, item.task_id), ())
        if s.record.role == UNBLOCK_ROLE and _started_at_or_after(s, filed)
    ]
    return _strongest_session(probe, matches, measured_at, f'no /unblock run on task {item.task_id} since filing')


def _probe_coalesce_fold(item: OpenItem, task_rows: TaskRows, measured_at: str) -> ProbeResult:
    probe = 'coalesce_fold'
    subject = _subject_row(probe, item, task_rows, measured_at)
    if isinstance(subject, ProbeResult):
        return subject
    target_id = subject.metadata.get(COALESCE_KEY)
    if target_id is None:
        return ProbeResult(probe, 'empty', measured_at, evidence=f'task {subject.id} carries no {COALESCE_KEY}')
    target = task_rows.rows.get(str(target_id))
    status = target.status if target is not None else ''
    title = target.title if target is not None else 'not in the store'
    return ProbeResult(
        probe, 'owns', measured_at,
        owner=f'task {target_id} ({status or "unknown"}): {title}',
        evidence=f'task {subject.id} {COALESCE_KEY}={target_id!r}',
        owner_task_id=str(target_id),
        owner_task_status=status,
    )


def _probe_task_ruling(item: OpenItem, task_rows: TaskRows, measured_at: str) -> ProbeResult:
    probe = 'task_ruling'
    subject = _subject_row(probe, item, task_rows, measured_at)
    if isinstance(subject, ProbeResult):
        return subject
    present = [(key, subject.metadata[key]) for key in RULING_METADATA_KEYS if subject.metadata.get(key)]
    if not present:
        return ProbeResult(probe, 'empty', measured_at, evidence=f'task {subject.id} carries no ruling key')
    return ProbeResult(
        probe, 'owns', measured_at,
        owner=f'ruling on task {subject.id} ({", ".join(key for key, _ in present)})',
        evidence='\n'.join(f'{key}={_quote(value)}' for key, value in present),
        owner_task_id=subject.id,
        owner_task_status=subject.status,
    )


def _probe_spawned_followup(item: OpenItem, task_rows: TaskRows, measured_at: str) -> ProbeResult:
    probe = 'spawned_followup'
    if task_rows.unavailable:
        return ProbeResult(probe, 'unavailable', measured_at, evidence=task_rows.unavailable)
    if not item.escalation_id:
        return ProbeResult(probe, 'empty', measured_at, evidence='the item carries no escalation id')
    followups = sorted(
        (row for row in task_rows.rows.values()
         if any(row.metadata.get(key) == item.escalation_id for key in FOLLOWUP_KEYS)),
        key=_newest_task_first,
    )
    evidence = '\n'.join(f'task {row.id} ({row.status}): {row.title}' for row in followups)
    live = [row for row in followups if row.status not in TERMINAL_TASK_STATUSES]
    if not live:
        return ProbeResult(
            probe, 'empty', measured_at, evidence=evidence or f'no task was spawned from {item.escalation_id}',
        )
    owner = live[0]
    return ProbeResult(
        probe, 'owns', measured_at,
        owner=f'task {owner.id} ({owner.status}): {owner.title}',
        evidence=evidence,
        owner_task_id=owner.id,
        owner_task_status=owner.status,
    )


def _probe_handover(item: OpenItem, handover_path: Path | None, measured_at: str) -> ProbeResult:
    probe = 'handover'
    if handover_path is None:
        return ProbeResult(probe, 'unavailable', measured_at, evidence='no handover file exists')
    try:
        text = handover_path.read_text(encoding='utf-8')
    except (OSError, ValueError) as exc:
        return ProbeResult(probe, 'unavailable', measured_at, evidence=f'{handover_path}: {exc}')
    hits = [
        f'{heading}\n{paragraph}' if heading else paragraph
        for heading, paragraph in _paragraphs(text)
        if _paragraph_mentions(item, paragraph)
    ]
    if not hits:
        return ProbeResult(probe, 'empty', measured_at, evidence=f'{handover_path.name} does not mention it')
    return ProbeResult(probe, 'mentions', measured_at, owner=handover_path.name, evidence='\n\n'.join(hits))


def _strongest_session(
    probe: str, sessions: Sequence[IndexedSession], measured_at: str, empty_evidence: str,
) -> ProbeResult:
    judged = [_judge_session(probe, s, measured_at) for s in sorted(sessions, key=_newest_session_first)]
    for status in ('owns', 'mentions'):
        hits = [result for result in judged if result.status == status]
        if hits:
            return ProbeResult(
                probe, status, measured_at, owner=hits[0].owner,
                evidence='\n'.join(result.evidence for result in hits),
            )
    return ProbeResult(probe, 'empty', measured_at, evidence=empty_evidence)


def _judge_session(probe: str, session: IndexedSession, measured_at: str) -> ProbeResult:
    record = session.record
    described = f'session {session.slug} role={record.role!r} status={record.status} started {record.start_ts}'
    if record.status not in TERMINAL_STATUSES:
        return ProbeResult(probe, 'owns', measured_at, owner=f'session {session.slug} (in flight)', evidence=described)
    outcome = _read_outcome(session)
    evidence = f'{described} result outcome={outcome or "absent or unparsable"}'
    if outcome in OWNING_OUTCOMES:
        return ProbeResult(
            probe, 'owns', measured_at, owner=f'session {session.slug} finished: {outcome}', evidence=evidence,
        )
    return ProbeResult(probe, 'mentions', measured_at, owner=f'session {session.slug}', evidence=evidence)


def _read_outcome(session: IndexedSession) -> str | None:
    path = Path(session.record.result_file) if session.record.result_file else session.record_dir / RESULT_FILENAME
    try:
        return parse_result_outcome(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _same_project(session: IndexedSession, item: OpenItem) -> bool:
    return normalize_project_token(session.record.project) in ('', item.project)


def _started_at_or_after(session: IndexedSession, filed: datetime | None) -> bool:
    started = parse_stamp(session.record.start_ts)
    return filed is not None and started is not None and started >= filed


def _newest_session_first(session: IndexedSession) -> tuple[bool, float]:
    started = parse_stamp(session.record.start_ts)
    return (session.record.status in TERMINAL_STATUSES, -started.timestamp() if started is not None else 0.0)


def _newest_task_first(row: TaskRow) -> int:
    return -int(row.id) if row.id.isdigit() else 0


def _subject_row(probe: str, item: OpenItem, task_rows: TaskRows, measured_at: str) -> TaskRow | ProbeResult:
    """The item's own task row, or the finished probe result when there is none to read."""
    if task_rows.unavailable:
        return ProbeResult(probe, 'unavailable', measured_at, evidence=task_rows.unavailable)
    if not item.task_id:
        return ProbeResult(probe, 'empty', measured_at, evidence='the item names no task')
    row = task_rows.rows.get(item.task_id)
    if row is None:
        return ProbeResult(probe, 'empty', measured_at, evidence=f'task {item.task_id} is not in the store')
    return row


def _quote(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _parse_metadata(raw: str | None) -> Mapping[str, Any]:
    try:
        parsed = json.loads(raw or '{}')
    except (TypeError, ValueError):
        return MappingProxyType({})
    return MappingProxyType(parsed if isinstance(parsed, dict) else {})


def _paragraphs(text: str) -> list[tuple[str, str]]:
    """(nearest heading, paragraph) for every blank-line-separated block of markdown."""
    heading = ''
    block: list[str] = []
    found: list[tuple[str, str]] = []
    for line in [*text.splitlines(), '']:
        if line.startswith('#') or not line.strip():
            if block:
                found.append((heading, '\n'.join(block)))
                block = []
            if line.startswith('#'):
                heading = line.strip()
            continue
        block.append(line)
    return found


def _paragraph_mentions(item: OpenItem, paragraph: str) -> bool:
    esc_ids, task_ids = cited_ids(paragraph)
    return (item.escalation_id is not None and item.escalation_id in esc_ids) or (
        item.task_id is not None and item.task_id in task_ids
    )
