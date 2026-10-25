"""scripts/legibility/plan_deviation_census.py — the nightly plan-deviation
census: counts recorded plan deviations and plan-deviation notes across tasks
and escalates a rate rise or a dominant pattern.

Design and rationale: plans/plan-deviation-recording-prd.md D8-D11.
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from escalation.categories import EscalationCategory
from escalation.disposition import ObservationConsumed
from escalation.models import NOTE_TO_FILE_CLASS, Escalation, normalised_severity
from escalation.queue import iter_all_escalation_paths, read_escalation_for_scan
from legibility import state_docs
from legibility.config import PlanDeviationCensus
from shared.plan_deviation import DEVIATION_KINDS

logger = logging.getLogger('legibility.plan_deviation_census')

RECENT_WINDOW_DAYS = 7

_PLAN_SECTIONS = ('prerequisites', 'steps')
_NOTED_RESOLUTION_CLASSES = frozenset({NOTE_TO_FILE_CLASS, ObservationConsumed.resolution_class})
_FALLBACK_AGENT_ROLES = frozenset({'implementer'})
_NOTE_SCAN_CONTEXT = 'plan_deviation_census.read_archived_notes'

_MAX_REPR_CHARS = 80


def _bounded(text: str, limit: int = _MAX_REPR_CHARS) -> str:
    """*text* on one line, cut to *limit* characters with an elision marker."""
    flat = text.replace('\n', '\\n')
    return flat if len(flat) <= limit else f'{flat[:limit]}...'


# ---------------------------------------------------------------------------
# settings — the raw config block, validated per field, never coerced
# ---------------------------------------------------------------------------

def _number(name: str, raw: object) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f'{name}: expected a number, got {raw!r}')
    if not math.isfinite(raw):
        raise ValueError(f'{name}: expected a finite number, got {raw!r}')
    return raw


def _rate_rise_factor(name: str, raw: object) -> float:
    value = _number(name, raw)
    if value < 1.0:
        raise ValueError(f'{name}: expected a number >= 1.0, got {raw!r}')
    return value


def _share(name: str, raw: object) -> float:
    value = _number(name, raw)
    if not 0 < value <= 1:
        raise ValueError(f'{name}: expected a number in (0, 1], got {raw!r}')
    return value


def _positive_int(name: str, raw: object) -> int:
    value = state_docs.as_int(name, raw)
    if value < 1:
        raise ValueError(f'{name}: expected an integer >= 1, got {raw!r}')
    return value


def _horizon_days(name: str, raw: object) -> int:
    value = state_docs.as_int(name, raw)
    if value <= RECENT_WINDOW_DAYS:
        raise ValueError(f'{name}: expected an integer > {RECENT_WINDOW_DAYS}, got {raw!r}')
    return value


def _meta_roots(name: str, raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list | tuple) or not raw:
        raise ValueError(f'{name}: expected a non-empty list of paths, got {raw!r}')
    roots = tuple(state_docs.as_str(name, entry) for entry in raw)
    if not all(roots):
        raise ValueError(f'{name}: expected non-empty paths, got {raw!r}')
    return roots


_FIELD_PARSERS: Mapping[str, Callable[[str, object], object]] = {
    'enabled': state_docs.as_bool,
    'rate_rise_factor': _rate_rise_factor,
    'min_records': _positive_int,
    'pattern_share': _share,
    'prose_only_min': _positive_int,
    'cooldown_days': _positive_int,
    'horizon_days': _horizon_days,
    'meta_roots': _meta_roots,
}


def settings_from_mapping(raw: object) -> PlanDeviationCensus:
    """The census settings from the raw ``plan_deviation_census`` block.

    ``None`` means every default. Each known key is validated on its own; a
    rejected value keeps that field's default, so one typo cannot disarm the
    other triggers. Every rejection and unknown key is reported in exactly
    one WARNING for the block. Never raises."""
    if raw is None:
        return PlanDeviationCensus()
    if not isinstance(raw, Mapping):
        logger.warning(
            'plan_deviation_census block is a %s, not a mapping; using every default',
            type(raw).__name__,
        )
        return PlanDeviationCensus()
    accepted: dict[str, object] = {}
    rejected: list[str] = []
    unknown: list[str] = []
    for key, value in raw.items():
        parse = _FIELD_PARSERS.get(key) if isinstance(key, str) else None
        if parse is None:
            unknown.append(_bounded(repr(key)))
            continue
        try:
            accepted[key] = parse(key, value)
        except ValueError as exc:
            rejected.append(
                f'{key}={_bounded(repr(value))} ({type(value).__name__}): {exc}'
            )
    if rejected or unknown:
        logger.warning(
            'plan_deviation_census block: defaults kept for rejected value(s) [%s]; '
            'unknown key(s) ignored [%s]',
            '; '.join(rejected), ', '.join(unknown),
        )
    return PlanDeviationCensus.model_validate(accepted)


# ---------------------------------------------------------------------------
# corpus — one flat list of dated records from two sources
# ---------------------------------------------------------------------------

class RecordSource(StrEnum):
    PLAN_STEP = 'plan_step'
    ESCALATION = 'escalation'


@dataclass(frozen=True)
class CensusRecord:
    task_id: str
    recorded_at: datetime
    kind: str | None
    prose_only: bool
    source: RecordSource


@dataclass(frozen=True)
class CorpusRead:
    """What one reader returned: its records, how many inputs it parsed, and
    how many inputs or entries it could not use."""

    records: tuple[CensusRecord, ...]
    opened: int
    skipped: int


_NOTHING = CorpusRead(records=(), opened=0, skipped=0)
_ONE_SKIPPED = CorpusRead(records=(), opened=0, skipped=1)


def _combine(reads: Iterable[CorpusRead]) -> CorpusRead:
    records: list[CensusRecord] = []
    opened = skipped = 0
    for read in reads:
        records.extend(read.records)
        opened += read.opened
        skipped += read.skipped
    return CorpusRead(records=tuple(records), opened=opened, skipped=skipped)


def _is_str_in(value: object, allowed: frozenset[str]) -> bool:
    return isinstance(value, str) and value in allowed


def _utc_stamp(name: str, raw: object) -> datetime | None:
    try:
        return state_docs.as_datetime(name, raw).astimezone(UTC)
    except ValueError:
        return None


def read_plan_deviations(meta_roots: Sequence[Path], *, horizon: datetime) -> CorpusRead:
    """Every valid deviation entry in ``<root>/*/plan.json`` under each root.

    A plan is rewritten on every write, so one last modified before *horizon*
    cannot hold an in-window entry and is skipped without being opened. The
    task id is the lane directory's name."""
    cutoff = horizon.timestamp()
    return _combine(_read_meta_root(root, cutoff) for root in meta_roots)


def _read_meta_root(root: Path, cutoff: float) -> CorpusRead:
    if not root.is_dir():
        return _NOTHING
    try:
        lanes = sorted(root.iterdir())
    except OSError as exc:
        logger.debug('cannot list meta root %s: %s', root, exc)
        return _ONE_SKIPPED
    return _combine(_read_plan(lane / 'plan.json', cutoff) for lane in lanes)


def _read_plan(path: Path, cutoff: float) -> CorpusRead:
    try:
        if path.stat().st_mtime < cutoff:
            return _NOTHING
    except (FileNotFoundError, NotADirectoryError):
        return _NOTHING
    except OSError as exc:
        logger.debug('cannot stat %s: %s', path, exc)
        return _ONE_SKIPPED
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug('cannot read %s: %s', path, exc)
        return _ONE_SKIPPED
    if not isinstance(doc, dict):
        return CorpusRead(records=(), opened=1, skipped=1)
    task_id = path.parent.name
    parsed = [_plan_entry(task_id, entry) for entry in _deviation_entries(doc)]
    records = tuple(record for record in parsed if record is not None)
    return CorpusRead(records=records, opened=1, skipped=len(parsed) - len(records))


def _deviation_entries(doc: Mapping[str, object]) -> Iterator[object]:
    for section in _PLAN_SECTIONS:
        items = doc.get(section)
        if not isinstance(items, list):
            continue
        for item in items:
            deviations = item.get('deviations') if isinstance(item, dict) else None
            if isinstance(deviations, list):
                yield from deviations


def _plan_entry(task_id: str, entry: object) -> CensusRecord | None:
    if not isinstance(entry, dict):
        return None
    kind = entry.get('kind')
    prose_only = entry.get('prose_only')
    if not _is_str_in(kind, DEVIATION_KINDS) or not isinstance(prose_only, bool):
        return None
    recorded_at = _utc_stamp('recorded_at', entry.get('recorded_at'))
    if recorded_at is None:
        return None
    return CensusRecord(
        task_id=task_id, recorded_at=recorded_at, kind=kind, prose_only=prose_only,
        source=RecordSource.PLAN_STEP,
    )


def read_archived_notes(escalations_dir: Path) -> CorpusRead:
    """Every escalation, pending or archived, that records a plan-deviation
    note: a note-to-file or observation-consumed resolution, or an info
    ``design_concern`` filed by an implementer before those classes existed."""
    return _combine(_read_note(path) for path in iter_all_escalation_paths(escalations_dir))


def _read_note(path: Path) -> CorpusRead:
    esc, outcome = read_escalation_for_scan(path, context=_NOTE_SCAN_CONTEXT)
    if esc is None:
        return _NOTHING if outcome == 'vanished' else _ONE_SKIPPED
    if not _is_counted_note(esc):
        return CorpusRead(records=(), opened=1, skipped=0)
    recorded_at = _utc_stamp('timestamp', esc.timestamp)
    if recorded_at is None:
        return CorpusRead(records=(), opened=1, skipped=1)
    record = CensusRecord(
        task_id=str(esc.task_id), recorded_at=recorded_at, kind=None, prose_only=False,
        source=RecordSource.ESCALATION,
    )
    return CorpusRead(records=(record,), opened=1, skipped=0)


def _is_counted_note(esc: Escalation) -> bool:
    if _is_str_in(esc.resolution_class, _NOTED_RESOLUTION_CLASSES):
        return True
    return (
        normalised_severity(esc.severity) == 'info'
        and esc.category == EscalationCategory.DESIGN_CONCERN
        and _is_str_in(esc.agent_role, _FALLBACK_AGENT_ROLES)
    )
