"""scripts/legibility/plan_deviation_census.py — the nightly plan-deviation
census: counts recorded plan deviations and plan-deviation notes across tasks
and escalates a rate rise or a dominant pattern.

Design and rationale: plans/plan-deviation-recording-prd.md D8-D11.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from escalation.categories import EscalationCategory
from escalation.disposition import ObservationConsumed
from escalation.models import NOTE_TO_FILE_CLASS, Escalation, normalised_severity
from escalation.queue import iter_all_escalation_paths, read_escalation_for_scan
from legibility import alarm, state_docs, trickle_state
from legibility.config import LegibilityConfig, PlanDeviationCensus
from shared.mcp_session import SyncMcpSession
from shared.plan_deviation import DEVIATION_KINDS
from shared.project_layout import escalation_queue_dir

logger = logging.getLogger('legibility.plan_deviation_census')

RECENT_WINDOW_DAYS = 7

_PLAN_SECTIONS = ('prerequisites', 'steps')
# Mirrors orchestrator/src/orchestrator/artifacts.py::Plan's confirmation key;
# the census tests pin it by reading what the real plan-tools server writes.
_FINALIZED_AT_KEY = '_finalized_at'
_NOTED_RESOLUTION_CLASSES = frozenset({NOTE_TO_FILE_CLASS, ObservationConsumed.resolution_class})
_FALLBACK_AGENT_ROLES = frozenset({'implementer'})
_NOTE_SCAN_CONTEXT = 'plan_deviation_census.read_archived_notes'
_TOP_TASK_IDS = 5
_FIRE_SUFFIX = ' -> FIRE'

STATE_FILENAME = 'plan-deviation-census-state.json'
AGENT_ROLE = 'plan-deviation-census'
CATEGORY = EscalationCategory.DESIGN_CONCERN.value
SEVERITY = 'blocking'
_LINE_PREFIX = 'plan-deviation census'
_DESIGN_POINTER = 'plans/plan-deviation-recording-prd.md D9-D10'
_MCP_TIMEOUT_SECS = 10.0
_MAX_FAILURE_CHARS = 200
_MAX_REPR_CHARS = 80

CallTool = Callable[[str, dict], object]


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
class FinalizedPlan:
    task_id: str
    finalized_at: datetime


@dataclass(frozen=True)
class CorpusRead:
    """What one reader returned: its records, the finalized plans it saw, how
    many inputs it parsed, and how many inputs or entries it could not use."""

    records: tuple[CensusRecord, ...]
    opened: int
    skipped: int
    plans: tuple[FinalizedPlan, ...] = ()


_NOTHING = CorpusRead(records=(), opened=0, skipped=0)
_ONE_SKIPPED = CorpusRead(records=(), opened=0, skipped=1)
_OPENED_AND_SKIPPED = CorpusRead(records=(), opened=1, skipped=1)


def _combine(reads: Iterable[CorpusRead]) -> CorpusRead:
    records: list[CensusRecord] = []
    plans: list[FinalizedPlan] = []
    opened = skipped = 0
    for read in reads:
        records.extend(read.records)
        plans.extend(read.plans)
        opened += read.opened
        skipped += read.skipped
    return CorpusRead(
        records=tuple(records), opened=opened, skipped=skipped, plans=tuple(plans),
    )


def _is_str_in(value: object, allowed: frozenset[str]) -> bool:
    return isinstance(value, str) and value in allowed


def _utc_stamp(name: str, raw: object) -> datetime | None:
    try:
        return state_docs.as_datetime(name, raw).astimezone(UTC)
    except ValueError:
        return None


def read_plan_deviations(meta_roots: Sequence[Path], *, horizon: datetime) -> CorpusRead:
    """Every valid deviation entry in ``<root>/*/plan.json`` under each root,
    and each task's finalized plan.

    A plan is rewritten on every write, so one last modified before *horizon*
    cannot hold an in-window entry and is skipped without being opened. The
    task id is the plan document's own ``task_id``, never its directory's
    name, and one plan seen under several meta directories (``N/`` and
    ``task-N/``) counts once: identical records collapse, and a task's
    finalized plan is its latest ``_finalized_at``."""
    cutoff = horizon.timestamp()
    read = _combine(_read_meta_root(root, cutoff) for root in meta_roots)
    return CorpusRead(
        records=tuple(dict.fromkeys(read.records)),
        opened=read.opened,
        skipped=read.skipped,
        plans=_latest_per_task(read.plans),
    )


def _latest_per_task(plans: Iterable[FinalizedPlan]) -> tuple[FinalizedPlan, ...]:
    latest: dict[str, FinalizedPlan] = {}
    for plan in plans:
        kept = latest.get(plan.task_id)
        if kept is None or plan.finalized_at > kept.finalized_at:
            latest[plan.task_id] = plan
    return tuple(latest.values())


def _read_meta_root(root: Path, cutoff: float) -> CorpusRead:
    if not root.is_dir():
        return _NOTHING
    try:
        lanes = sorted(root.iterdir())
    except OSError as exc:
        logger.debug('cannot list meta root %s: %s', root, exc)
        return _ONE_SKIPPED
    return _combine(_read_plan(lane / 'plan.json', cutoff) for lane in lanes)


def _unopened(path: Path, cutoff: float) -> CorpusRead | None:
    """What *path* contributes without being opened -- nothing when it is
    absent or last modified before *cutoff*, one skip when it cannot be
    stat'ed -- or ``None`` when it is to be read."""
    try:
        if path.stat().st_mtime >= cutoff:
            return None
    except (FileNotFoundError, NotADirectoryError):
        return _NOTHING
    except OSError as exc:
        logger.debug('cannot stat %s: %s', path, exc)
        return _ONE_SKIPPED
    return _NOTHING


def _read_plan(path: Path, cutoff: float) -> CorpusRead:
    unopened = _unopened(path, cutoff)
    if unopened is not None:
        return unopened
    try:
        doc = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug('cannot read %s: %s', path, exc)
        return _ONE_SKIPPED
    if not isinstance(doc, dict):
        return _OPENED_AND_SKIPPED
    task_id = _document_task_id(doc.get('task_id'))
    if task_id is None:
        return _OPENED_AND_SKIPPED
    parsed = [_plan_entry(task_id, entry) for entry in _deviation_entries(doc)]
    records = tuple(record for record in parsed if record is not None)
    finalized_at = _utc_stamp(_FINALIZED_AT_KEY, doc.get(_FINALIZED_AT_KEY))
    return CorpusRead(
        records=records,
        opened=1,
        skipped=len(parsed) - len(records),
        plans=() if finalized_at is None else (FinalizedPlan(task_id, finalized_at),),
    )


def _document_task_id(raw: object) -> str | None:
    if isinstance(raw, str):
        return raw.strip() or None
    if isinstance(raw, int) and not isinstance(raw, bool):
        return str(raw)
    return None


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


def read_archived_notes(escalations_dir: Path, *, horizon: datetime) -> CorpusRead:
    """Every escalation, pending or archived, that records a plan-deviation
    note: a note-to-file or observation-consumed resolution, or an info
    ``design_concern`` filed by an implementer before those classes existed.
    The census's own escalations are never counted.

    A record's file is written at filing and rewritten at resolution, so one
    last modified before *horizon* was filed before it too and is skipped
    without being opened."""
    cutoff = horizon.timestamp()
    return _combine(
        _read_note(path, cutoff) for path in iter_all_escalation_paths(escalations_dir)
    )


def _read_note(path: Path, cutoff: float) -> CorpusRead:
    unopened = _unopened(path, cutoff)
    if unopened is not None:
        return unopened
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
    if esc.agent_role == AGENT_ROLE:
        return False
    if _is_str_in(esc.resolution_class, _NOTED_RESOLUTION_CLASSES):
        return True
    return (
        normalised_severity(esc.severity) == 'info'
        and esc.category == EscalationCategory.DESIGN_CONCERN
        and _is_str_in(esc.agent_role, _FALLBACK_AGENT_ROLES)
    )


# ---------------------------------------------------------------------------
# evaluate — pure: two windows measured back from now, a rate arm per source
# and a prose-only arm (records per finalized plan), and a kind pattern
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    fire: bool
    reasons: tuple[str, ...]
    evidence: tuple[dict, ...]


@dataclass(frozen=True)
class _Windows:
    """The records and the finalized plans in the recent window
    ``(now-7d, now]`` and the baseline window ``(now-horizon, now-7d]``, and
    each source's earliest record in ``(now-horizon, now]``."""

    now: datetime
    recent: tuple[CensusRecord, ...]
    baseline: tuple[CensusRecord, ...]
    recent_plans: int
    baseline_plans: int
    history_starts: Mapping[RecordSource, datetime | None]
    baseline_days: int

    @property
    def recent_mean(self) -> float:
        return len(self.recent) / RECENT_WINDOW_DAYS

    @property
    def baseline_mean(self) -> float:
        return len(self.baseline) / self.baseline_days

    def counts(self, predicate: Callable[[CensusRecord], bool]) -> tuple[int, int]:
        """How many recent and how many baseline records satisfy *predicate*."""
        return (
            sum(1 for record in self.recent if predicate(record)),
            sum(1 for record in self.baseline if predicate(record)),
        )

    def per_plan(self, recent: int, baseline: int) -> tuple[float | None, float | None]:
        """*recent* and *baseline* per finalized plan in their windows;
        ``None`` for a window with no finalized plan."""
        return (
            recent / self.recent_plans if self.recent_plans else None,
            baseline / self.baseline_plans if self.baseline_plans else None,
        )

    def empty_window(self) -> str | None:
        if not self.recent_plans:
            return 'recent'
        if not self.baseline_plans:
            return 'baseline'
        return None

    def history_days(self, source: RecordSource) -> float | None:
        start = self.history_starts[source]
        return None if start is None else (self.now - start) / timedelta(days=1)

    def is_warming_up(self, source: RecordSource) -> bool:
        """Whether *source*'s history is too short for its baseline to mean
        anything: no in-horizon record, or a first one under the baseline
        window's length ago."""
        history = self.history_days(source)
        return history is None or history < self.baseline_days

    def kind_counts(self) -> dict[str, int]:
        counts = Counter(record.kind for record in self.recent)
        return {kind: counts[kind] for kind in sorted(DEVIATION_KINDS)}


def _as_utc(now: datetime) -> datetime:
    return now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)


def _split(
    records: Sequence[CensusRecord],
    plans: Sequence[FinalizedPlan],
    *,
    now: datetime,
    horizon_days: int,
) -> _Windows:
    recent_start = now - timedelta(days=RECENT_WINDOW_DAYS)
    horizon = now - timedelta(days=horizon_days)
    in_horizon = [record for record in records if horizon < record.recorded_at <= now]
    return _Windows(
        now=now,
        recent=tuple(record for record in in_horizon if recent_start < record.recorded_at),
        baseline=tuple(record for record in in_horizon if record.recorded_at <= recent_start),
        recent_plans=sum(1 for plan in plans if recent_start < plan.finalized_at <= now),
        baseline_plans=sum(1 for plan in plans if horizon < plan.finalized_at <= recent_start),
        history_starts={
            source: min(
                (record.recorded_at for record in in_horizon if record.source is source),
                default=None,
            )
            for source in RecordSource
        },
        baseline_days=horizon_days - RECENT_WINDOW_DAYS,
    )


@dataclass(frozen=True)
class _RateArm:
    """A per-plan rate of the records *predicate* selects, measured against
    its own baseline; *source*'s warm-up gates it and *floor* is the fewest
    recent records that may fire it."""

    label: str
    source: RecordSource
    predicate: Callable[[CensusRecord], bool]
    floor: int


def _from_source(source: RecordSource) -> Callable[[CensusRecord], bool]:
    return lambda record: record.source is source


def _is_prose_only_plan_step(record: CensusRecord) -> bool:
    return record.source is RecordSource.PLAN_STEP and record.prose_only


def _source_arms(config: PlanDeviationCensus) -> tuple[_RateArm, ...]:
    return tuple(
        _RateArm(f'rate-rise[{source.value}]', source, _from_source(source), config.min_records)
        for source in RecordSource
    )


def _prose_only_arm(config: PlanDeviationCensus) -> _RateArm:
    return _RateArm(
        'prose-only', RecordSource.PLAN_STEP, _is_prose_only_plan_step, config.prose_only_min,
    )


def evaluate(
    records: Sequence[CensusRecord],
    *,
    plans: Sequence[FinalizedPlan],
    now: datetime,
    config: PlanDeviationCensus,
) -> Verdict:
    """Whether the recorded deviations rose per finalized plan, source by
    source, or formed a pattern. Records and plans after *now* or at or
    before the horizon are ignored. A source still warming up, or a window
    without a finalized plan, fires no rate arm."""
    windows = _split(records, plans, now=_as_utc(now), horizon_days=config.horizon_days)
    triggers = (
        *(_rate_arm(windows, arm, config) for arm in _source_arms(config)),
        _kind_pattern(windows, config),
        _rate_arm(windows, _prose_only_arm(config), config),
    )
    return Verdict(
        fire=any(fired for fired, _ in triggers),
        reasons=tuple(text + _FIRE_SUFFIX if fired else text for fired, text in triggers),
        evidence=_evidence(windows, config),
    )


def _rate_arm(windows: _Windows, arm: _RateArm, config: PlanDeviationCensus) -> tuple[bool, str]:
    recent, baseline = windows.counts(arm.predicate)
    if windows.is_warming_up(arm.source):
        return False, f'{arm.label}: {_warming_up_text(windows, arm.source)}'
    empty = windows.empty_window()
    if empty is not None:
        return False, (
            f'{arm.label}: no finalized plan in the {empty} window '
            f'({recent} recent, {baseline} baseline record(s))'
        )
    recent_rate = recent / windows.recent_plans
    baseline_rate = baseline / windows.baseline_plans
    bar = config.rate_rise_factor * baseline_rate
    fired = recent_rate > bar and recent >= arm.floor
    return fired, (
        f'{arm.label}: {RECENT_WINDOW_DAYS}d {recent_rate:.3f}/plan '
        f'({recent} over {windows.recent_plans} plan(s)) vs prior {windows.baseline_days}d '
        f'{baseline_rate:.3f}/plan ({baseline} over {windows.baseline_plans} plan(s)) '
        f'(bar x{float(config.rate_rise_factor)} = {bar:.3f}/plan; '
        f'{RECENT_WINDOW_DAYS}d count {recent}, min {arm.floor})'
    )


def _warming_up_text(windows: _Windows, source: RecordSource) -> str:
    history = windows.history_days(source)
    if history is None:
        horizon_days = windows.baseline_days + RECENT_WINDOW_DAYS
        return f'warming up (no record in the {horizon_days}d horizon)'
    return f'warming up (history {history:.1f}d < {windows.baseline_days}d)'


def _kind_pattern(windows: _Windows, config: PlanDeviationCensus) -> tuple[bool, str]:
    counts = windows.kind_counts()
    kinded = sum(counts.values())
    if not kinded:
        return False, (
            f'kind-pattern: no kinded records in {RECENT_WINDOW_DAYS}d '
            f'({len(windows.recent)} record(s))'
        )
    top_kind = min(counts, key=lambda kind: (-counts[kind], kind))
    share = counts[top_kind] / kinded
    fired = share >= config.pattern_share and counts[top_kind] >= config.min_records
    return fired, (
        f'kind-pattern: {top_kind} {counts[top_kind]}/{kinded} = {share:.2f} of '
        f'{RECENT_WINDOW_DAYS}d kinded records (bar {config.pattern_share}; '
        f'min {config.min_records})'
    )


def _evidence(windows: _Windows, config: PlanDeviationCensus) -> tuple[dict, ...]:
    measured_at = windows.now.isoformat()
    return tuple(
        {'observation': observation, 'measured_at': measured_at, 'ref': ref, **fields}
        for ref, observation, fields in (
            _window_evidence(windows, config),
            _kinds_evidence(windows),
            _prose_only_evidence(windows),
            _top_task_ids_evidence(windows),
            _sources_evidence(windows),
        )
    )


def _per_plan_text(rate: float | None) -> str:
    return 'no finalized plan' if rate is None else f'{rate:.3f}/plan'


def _window_evidence(windows: _Windows, config: PlanDeviationCensus) -> tuple[str, str, dict]:
    recent, baseline = len(windows.recent), len(windows.baseline)
    recent_rate, baseline_rate = windows.per_plan(recent, baseline)
    return 'window', (
        f'{recent} record(s) over {windows.recent_plans} finalized plan(s) in the '
        f'{RECENT_WINDOW_DAYS} days to {windows.now.isoformat()} '
        f'({_per_plan_text(recent_rate)}; mean {windows.recent_mean:.2f}/d); '
        f'{baseline} over {windows.baseline_plans} in the prior {windows.baseline_days} days '
        f'({_per_plan_text(baseline_rate)}; mean {windows.baseline_mean:.2f}/d)'
    ), {
        'window_end': windows.now.isoformat(),
        'recent_window_days': RECENT_WINDOW_DAYS,
        'baseline_window_days': windows.baseline_days,
        'recent_count': recent,
        'recent_plans': windows.recent_plans,
        'recent_per_plan': recent_rate,
        'recent_mean': windows.recent_mean,
        'baseline_count': baseline,
        'baseline_plans': windows.baseline_plans,
        'baseline_per_plan': baseline_rate,
        'baseline_mean': windows.baseline_mean,
        'rate_rise_factor': config.rate_rise_factor,
    }


def _kinds_evidence(windows: _Windows) -> tuple[str, str, dict]:
    counts = windows.kind_counts()
    kinded = sum(counts.values())
    unkinded = sum(record.kind is None for record in windows.recent)
    named = ', '.join(f'{kind} {n}' for kind, n in counts.items() if n) or 'none'
    return 'kinds', (
        f'recent records by kind: {named}; {kinded} with a kind, {unkinded} without'
    ), {
        'kind_counts': counts,
        'kinded_count': kinded,
        'unkinded': unkinded,
        'recent_count': len(windows.recent),
    }


def _prose_only_evidence(windows: _Windows) -> tuple[str, str, dict]:
    recent, baseline = windows.counts(_is_prose_only_plan_step)
    recent_rate, baseline_rate = windows.per_plan(recent, baseline)
    return 'prose_only', (
        f'{recent} recent record(s) were recorded in prose only '
        f'({_per_plan_text(recent_rate)}); {baseline} in the baseline '
        f'({_per_plan_text(baseline_rate)})'
    ), {
        'prose_only_count': recent,
        'baseline_prose_only_count': baseline,
        'recent_per_plan': recent_rate,
        'baseline_per_plan': baseline_rate,
    }


def _top_task_ids_evidence(windows: _Windows) -> tuple[str, str, dict]:
    counts = Counter(record.task_id for record in windows.recent)
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:_TOP_TASK_IDS]
    named = ', '.join(f'{task_id} {n}' for task_id, n in top) or 'none'
    return 'top_task_ids', (
        f'busiest tasks in the recent window: {named}'
    ), {'top_task_ids': [{'task_id': task_id, 'count': n} for task_id, n in top]}


def _sources_evidence(windows: _Windows) -> tuple[str, str, dict]:
    per_source = {source.value: _source_fields(windows, source) for source in RecordSource}
    warming_up = [source.value for source in RecordSource if windows.is_warming_up(source)]
    named = ', '.join(
        f'{fields["recent_count"]} {source}' for source, fields in per_source.items()
    )
    return 'sources', (
        f'recent records by source: {named}; warming up: {", ".join(warming_up) or "none"}'
    ), {**per_source, 'warming_up': warming_up}


def _source_fields(windows: _Windows, source: RecordSource) -> dict[str, object]:
    recent, baseline = windows.counts(_from_source(source))
    recent_rate, baseline_rate = windows.per_plan(recent, baseline)
    return {
        'recent_count': recent,
        'baseline_count': baseline,
        'recent_per_plan': recent_rate,
        'baseline_per_plan': baseline_rate,
        'history_days': windows.history_days(source),
        'warming_up': windows.is_warming_up(source),
    }


# ---------------------------------------------------------------------------
# state — when the census last filed, so it files at most once per cooldown
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CensusState:
    last_filed_at: datetime
    escalation_id: str

    def to_json(self) -> dict[str, object]:
        return {
            'last_filed_at': self.last_filed_at.isoformat(),
            'escalation_id': self.escalation_id,
        }


def _state_codec(raw: object) -> CensusState:
    doc = state_docs.json_object(raw, 'plan-deviation census state')
    return CensusState(
        last_filed_at=state_docs.required(doc, 'last_filed_at', state_docs.as_datetime),
        escalation_id=state_docs.required(doc, 'escalation_id', state_docs.as_str),
    )


def default_state_path(project_id: str) -> Path:
    return trickle_state.project_state_dir(project_id) / STATE_FILENAME


def synthetic_task_id(project_id: str) -> str:
    return f'plan-deviation-census-{project_id}'


# ---------------------------------------------------------------------------
# run — one census pass, one journal line, never raises
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Disposition:
    """What became of a FIRE verdict; *degraded* when a step meant to guard
    or record the filing could not do its job."""

    text: str
    degraded: bool


def _failure(exc: BaseException) -> str:
    return f'{type(exc).__name__}: {_bounded(str(exc), _MAX_FAILURE_CHARS)}'


def _default_call_tool(cfg: LegibilityConfig) -> CallTool:
    base_url = f'http://localhost:{cfg.escalation_port}'

    def call_tool(tool_name: str, arguments: dict) -> dict:
        alarm.refuse_real_post_under_test(base_url, tool_name)
        with SyncMcpSession(
            base_url, client_name='legibility-plan-deviation-census', timeout=_MCP_TIMEOUT_SECS,
        ) as session:
            return session.call_tool(tool_name, arguments)

    return call_tool


def run(
    cfg: LegibilityConfig,
    *,
    now: datetime | None = None,
    call_tool: CallTool | None = None,
    state_path: Path | None = None,
) -> str:
    """Run the census for *cfg*'s project: read both corpora, evaluate, and on
    FIRE file one escalation unless the cooldown or a pending census
    escalation holds it. Logs and returns one ``plan-deviation census:
    FIRE|NO-FIRE -- ...`` line. Never raises."""
    try:
        line, degraded = _census_line(
            cfg,
            now=datetime.now(UTC) if now is None else now,
            call_tool=_default_call_tool(cfg) if call_tool is None else call_tool,
            state_path=default_state_path(cfg.project_id) if state_path is None else state_path,
        )
    except Exception as exc:
        line, degraded = f'{_LINE_PREFIX}: NO-FIRE -- evaluation failed ({_failure(exc)})', True
    logger.log(logging.WARNING if degraded else logging.INFO, '%s', line)
    return line


def _census_line(
    cfg: LegibilityConfig, *, now: datetime, call_tool: CallTool, state_path: Path,
) -> tuple[str, bool]:
    settings = settings_from_mapping(cfg.plan_deviation_census)
    if not settings.enabled:
        return f'{_LINE_PREFIX}: NO-FIRE -- disabled by plan_deviation_census.enabled', False
    now = _as_utc(now)
    horizon = now - timedelta(days=settings.horizon_days)
    plan_corpus = read_plan_deviations(
        [Path(cfg.project_root) / root for root in settings.meta_roots], horizon=horizon,
    )
    notes = read_archived_notes(escalation_queue_dir(cfg.project_root), horizon=horizon)
    verdict = evaluate(
        [*plan_corpus.records, *notes.records], plans=plan_corpus.plans, now=now, config=settings,
    )
    summary = (
        f'{"; ".join(verdict.reasons)}; corpus: {len(plan_corpus.records)} plan-step + '
        f'{len(notes.records)} escalation record(s), {len(plan_corpus.plans)} finalized '
        f'plan(s), {plan_corpus.skipped + notes.skipped} skipped'
    )
    if not verdict.fire:
        return f'{_LINE_PREFIX}: NO-FIRE -- {summary}', False
    disposition = _dispose(
        verdict, project_id=cfg.project_id, settings=settings, now=now,
        call_tool=call_tool, state_path=state_path,
    )
    return f'{_LINE_PREFIX}: FIRE -- {summary}; {disposition.text}', disposition.degraded


def _dispose(
    verdict: Verdict,
    *,
    project_id: str,
    settings: PlanDeviationCensus,
    now: datetime,
    call_tool: CallTool,
    state_path: Path,
) -> _Disposition:
    """Cooldown, then the pending check, then the filing, then the state write."""
    held = _cooldown_hold(state_path, now=now, cooldown_days=settings.cooldown_days)
    if held is not None:
        return held
    task_id = synthetic_task_id(project_id)
    held = _pending_hold(call_tool, task_id)
    if held is not None:
        return held
    try:
        escalation_id, status = _confirmed_filing(
            call_tool('escalate_info', _filing_arguments(verdict, task_id))
        )
    except Exception as exc:
        return _Disposition(f'filing failed ({_failure(exc)}) -- not filed', degraded=True)
    return _record_filing(
        state_path, CensusState(last_filed_at=now, escalation_id=escalation_id), status=status,
    )


def _cooldown_hold(state_path: Path, *, now: datetime, cooldown_days: int) -> _Disposition | None:
    state = state_docs.read_state(state_path, _state_codec)
    if isinstance(state, state_docs.Malformed):
        return _Disposition(
            f'state {state.path} malformed ({state.reason}) -- not filed', degraded=True,
        )
    if isinstance(state, state_docs.Missing):
        return None
    age = now - state.value.last_filed_at
    if age >= timedelta(days=cooldown_days):
        return None
    return _Disposition(
        f'cooldown: last filed {age / timedelta(days=1):.1f}d ago as '
        f'{state.value.escalation_id} (cooldown {cooldown_days}d) -- not filed',
        degraded=False,
    )


def _pending_hold(call_tool: CallTool, task_id: str) -> _Disposition | None:
    try:
        pending = _pending_ids(
            call_tool('get_pending_escalations', {'task_id': task_id, 'compact': True})
        )
    except Exception as exc:
        return _Disposition(f'pending check failed ({_failure(exc)}) -- not filed', degraded=True)
    if not pending:
        return None
    return _Disposition(
        f'pending: {", ".join(pending)} already open on {task_id} -- not filed', degraded=False,
    )


def _pending_ids(response: object) -> list[str]:
    rows = response.get('result') if isinstance(response, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f'unexpected get_pending_escalations response {_bounded(repr(response))}')
    return [str(row.get('id', '?')) if isinstance(row, dict) else '?' for row in rows]


def _filing_arguments(verdict: Verdict, task_id: str) -> dict:
    return {
        'task_id': task_id,
        'agent_role': AGENT_ROLE,
        'category': CATEGORY,
        'severity': SEVERITY,
        'summary': 'plan-deviation census FIRE -- ' + '; '.join(verdict.reasons),
        'detail': '\n'.join([*verdict.reasons, f'Design: {_DESIGN_POINTER}.']),
        'suggested_action': (
            'Read the evidence (per-plan rates by source, warming-up sources, per-kind '
            'counts, prose-only rates, top task ids) and decide whether planning doctrine '
            'or tooling needs to change.'
        ),
        'evidence': list(verdict.evidence),
    }


def _confirmed_filing(response: object) -> tuple[str, str]:
    """The escalation id and status of a filing the server confirmed it
    persisted; ``ValueError`` naming why for any other response."""
    if not isinstance(response, dict):
        raise ValueError(f'unexpected escalate_info response {_bounded(repr(response))}')
    if 'error' in response:
        raise ValueError(f'server refused: {_bounded(repr(response["error"]))}')
    status = str(response.get('status'))
    if status == 'accepted_unpersisted':
        raise ValueError('status accepted_unpersisted: the server could not confirm it persisted')
    escalation_id = response.get('id')
    if not isinstance(escalation_id, str) or not escalation_id:
        raise ValueError(f'response carries no escalation id: {_bounded(repr(response))}')
    return escalation_id, status


def _record_filing(state_path: Path, state: CensusState, *, status: str) -> _Disposition:
    filed = f'filed {state.escalation_id} (status={status})'
    try:
        state_docs.write_state(state_path, state)
    except OSError as exc:
        return _Disposition(
            f'{filed}; state write failed ({_failure(exc)}) -- the pending check guards a refile',
            degraded=True,
        )
    return _Disposition(filed, degraded=False)
