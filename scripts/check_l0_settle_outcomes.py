#!/usr/bin/env python3
"""Count stranded and wrongly dismissed blocking L0 escalations in one escalation queue directory.

``--queue-dir DIR --window-days N --baseline-stranded-per-100 F --ratio R --min-stranded K
--max-reaper-wrong M --max-pending-hours H`` reads DIR's live records and its dated archive, writes
nothing, and prints human lines followed by a compact JSON object as the last line. It exits 1 when a
threshold is breached, 0 otherwise, and 2 when it cannot judge: the queue directory could not be read,
or no blocking L0 was filed in the post window. The rules and thresholds are
plans/review-all-s9-workflow-phase-spine-and-recovery-ledger-prd.md §C10.
"""

from __future__ import annotations

import argparse
import enum
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Final

# orchestrator/src/orchestrator/harness.py::Harness._reap_orphan_l0_escalations
REAPER_RESOLVER: Final = 'harness-orphan-reaper'
PROMOTION_PREFIX: Final = 'Auto-promoted to level 1'
# escalation/src/escalation/queue.py::EscalationQueue.dismiss_all_pending and
# orchestrator/src/orchestrator/harness.py::Harness._dismiss_stale_escalations
AUTO_DISMISS_RESOLVER: Final = 'auto-dismissed'
RESTART_PREFIX: Final = 'Auto-dismissed: orchestrator restarted'
STALE_STRAND_CLASS: Final = 'stale-strand'
# escalation/src/escalation/archive.py::ARCHIVE_SUBDIR and ::DATE_FORMAT
ARCHIVE_SUBDIR: Final = 'archive'
ARCHIVE_DAY_FORMAT: Final = '%Y-%m-%d'
LIVE_RECORD_GLOB: Final = 'esc-*.json'
PENDING_STATUS: Final = 'pending'

COUNTED_LEVEL: Final = 0
COUNTED_SEVERITY: Final = 'blocking'
FOLLOWER_WINDOW: Final = timedelta(hours=24)
FOLLOWER_LEVELS: Final = frozenset({0, 1})
REDESIGN_FOLLOW_UP: Final = 'open the joint L0 end-of-life redesign with the harness plan (S8)'
RATE_DIGITS: Final = 2
CANNOT_JUDGE_EXIT: Final = 2
UNBOUNDED: Final = datetime.max.replace(tzinfo=UTC)


class CannotJudge(Exception):
    headline: ClassVar[str]

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f'{path}: {reason}')
        self.path = path
        self.reason = reason


class UnreadableQueue(CannotJudge):
    headline = 'unreadable queue directory'


class RecordVanished(UnreadableQueue):
    """A listed record file was gone by the time it was read."""


class NothingFiled(CannotJudge):
    headline = 'nothing to judge'


class Closure(enum.StrEnum):
    OPEN = 'open'
    STEWARD_OR_OTHER = 'steward_or_other'
    REAPER_PROMOTION = 'reaper_promotion'
    OTHER_PROMOTION = 'other_promotion'
    REAPER_DISMISSAL = 'reaper_dismissal'
    STARTUP_STRAND = 'startup_strand'
    STARTUP_DISMISSAL = 'startup_dismissal'
    WORKFLOW_AUTO_DISMISSAL = 'workflow_auto_dismissal'


def classify_closure(
    status: str, resolved_by: str | None, resolution: str | None, resolution_class: str | None,
) -> Closure:
    if status == PENDING_STATUS:
        return Closure.OPEN
    if resolution_class == STALE_STRAND_CLASS:
        return Closure.STARTUP_STRAND
    text = resolution or ''
    if resolved_by == REAPER_RESOLVER:
        return Closure.REAPER_PROMOTION if text.startswith(PROMOTION_PREFIX) else Closure.REAPER_DISMISSAL
    if resolved_by == AUTO_DISMISS_RESOLVER:
        return Closure.STARTUP_DISMISSAL if text.startswith(RESTART_PREFIX) else Closure.WORKFLOW_AUTO_DISMISSAL
    if text.startswith(PROMOTION_PREFIX):
        return Closure.OTHER_PROMOTION
    return Closure.STEWARD_OR_OTHER


@dataclass(frozen=True)
class Record:
    id: str
    task_id: str | None
    level: int
    severity: str
    category: str
    filed_at: datetime
    resolved_at: datetime | None
    closure: Closure
    late_resolved: bool

    @property
    def is_counted_l0(self) -> bool:
        return (
            self.level == COUNTED_LEVEL
            and self.severity == COUNTED_SEVERITY
            and self.task_id is not None
            and self.task_id.isdigit()
        )


def _required(raw: Mapping[str, Any], key: str, path: Path) -> object:
    if key not in raw:
        raise UnreadableQueue(path, f'missing required key {key!r}')
    return raw[key]


def _text(raw: Mapping[str, Any], key: str, path: Path) -> str:
    value = _required(raw, key, path)
    if not isinstance(value, str):
        raise UnreadableQueue(path, f'{key} is {type(value).__name__}, not str')
    return value


def _text_or_null(value: object, key: str, path: Path) -> str | None:
    if value is not None and not isinstance(value, str):
        raise UnreadableQueue(path, f'{key} is {type(value).__name__}, not str or null')
    return value


def _integer(raw: Mapping[str, Any], key: str, path: Path) -> int:
    value = _required(raw, key, path)
    if not isinstance(value, int) or isinstance(value, bool):
        raise UnreadableQueue(path, f'{key} is {type(value).__name__}, not int')
    return value


def _moment(raw: Mapping[str, Any], key: str, path: Path) -> datetime:
    value = _text(raw, key, path)
    try:
        moment = datetime.fromisoformat(value)
        return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
    except (ValueError, OverflowError) as exc:
        raise UnreadableQueue(path, f'{key} {value!r} is not a usable ISO timestamp') from exc


def _late_resolved(raw: Mapping[str, Any], path: Path) -> bool:
    entries = raw.get('late_resolutions', [])
    if not isinstance(entries, list):
        raise UnreadableQueue(path, f'late_resolutions is {type(entries).__name__}, not list')
    return bool(entries)


def parse_record(raw: Mapping[str, Any], path: Path) -> Record:
    """The one boundary where an escalation file's persisted fields become a typed Record."""
    status = _text(raw, 'status', path)
    is_open = status == PENDING_STATUS
    return Record(
        id=_text(raw, 'id', path),
        task_id=_text_or_null(_required(raw, 'task_id', path), 'task_id', path),
        level=_integer(raw, 'level', path),
        severity=_text(raw, 'severity', path),
        category=_text(raw, 'category', path),
        filed_at=_moment(raw, 'timestamp', path),
        resolved_at=None if is_open else _moment(raw, 'resolved_at', path),
        closure=classify_closure(
            status,
            _text_or_null(raw.get('resolved_by'), 'resolved_by', path),
            _text_or_null(raw.get('resolution'), 'resolution', path),
            _text_or_null(raw.get('resolution_class'), 'resolution_class', path),
        ),
        late_resolved=_late_resolved(raw, path),
    )


def read_record(path: Path) -> Record:
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise RecordVanished(path, 'gone since it was listed') from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnreadableQueue(path, f'{type(exc).__name__}: {exc}') from exc
    if not isinstance(raw, dict):
        raise UnreadableQueue(path, f'holds a JSON {type(raw).__name__}, not an object')
    return parse_record(raw, path)


def _is_archive_day(name: str) -> bool:
    try:
        datetime.strptime(name, ARCHIVE_DAY_FORMAT)
    except ValueError:
        return False
    return True


def _sorted_glob(directory: Path, pattern: str) -> list[Path]:
    try:
        return sorted(directory.glob(pattern))
    except OSError as exc:
        raise UnreadableQueue(directory, f'{type(exc).__name__}: {exc}') from exc


def _archive_days(archive_root: Path) -> list[Path]:
    try:
        children = list(archive_root.iterdir())
    except OSError as exc:
        raise UnreadableQueue(archive_root, f'{type(exc).__name__}: {exc}') from exc
    return sorted(child for child in children if child.is_dir() and _is_archive_day(child.name))


def _archived_record_paths(archive_root: Path) -> list[Path]:
    return [path for day in _archive_days(archive_root) for path in _sorted_glob(day, '*.json')]


def _read_live_record(path: Path, archive_root: Path) -> Record:
    """A live record the queue archived after the root was listed is read from its archived copy."""
    try:
        return read_record(path)
    except RecordVanished:
        copies = [day / path.name for day in _archive_days(archive_root) if (day / path.name).is_file()]
        if not copies:
            raise UnreadableQueue(path, 'left the queue root without an archived copy') from None
        return read_record(copies[-1])


def load_records(queue_dir: Path) -> list[Record]:
    """Every record in the dated archive and the live queue root, one per id, the root copy winning.

    The root is listed only once the archive has been read, so the queue archiving a live record mid-scan
    leaves that record readable from its archived copy.
    """
    if not queue_dir.is_dir():
        raise UnreadableQueue(queue_dir, 'not a directory')
    archive_root = queue_dir / ARCHIVE_SUBDIR
    if not archive_root.is_dir():
        raise UnreadableQueue(archive_root, 'not a directory')
    by_id = {record.id: record for record in map(read_record, _archived_record_paths(archive_root))}
    for path in _sorted_glob(queue_dir, LIVE_RECORD_GLOB):
        record = _read_live_record(path, archive_root)
        by_id[record.id] = record
    return list(by_id.values())


FollowerIndex = Mapping[tuple[str | None, str], Sequence[Record]]


def index_followers(records: Iterable[Record]) -> FollowerIndex:
    followers: defaultdict[tuple[str | None, str], list[Record]] = defaultdict(list)
    for record in records:
        if record.level in FOLLOWER_LEVELS:
            followers[(record.task_id, record.category)].append(record)
    return followers


def wrongly_dismissed(record: Record, followers_by_task_category: FollowerIndex) -> bool:
    """Late-resolved, or followed within FOLLOWER_WINDOW of its close by a same-task, same-category filing."""
    if record.late_resolved:
        return True
    resolved_at = record.resolved_at
    if resolved_at is None:
        return False
    return any(
        follower.id != record.id and resolved_at < follower.filed_at <= resolved_at + FOLLOWER_WINDOW
        for follower in followers_by_task_category.get((record.task_id, record.category), ())
    )


@dataclass(frozen=True)
class WindowCounts:
    filed: int
    reaper_promotions: int
    stale_strands: int
    other_promotions: int
    reaper_dismissals: int
    reaper_wrong: int
    startup_dismissals: int
    startup_wrong: int
    workflow_auto_dismissals: int
    workflow_auto_wrong: int

    @property
    def stranded(self) -> int:
        return self.reaper_promotions + self.stale_strands

    @property
    def stranded_per_100(self) -> float | None:
        return 100 * self.stranded / self.filed if self.filed else None

    def as_json(self) -> dict[str, int | float | None]:
        rate = self.stranded_per_100
        return {
            **asdict(self),
            'stranded': self.stranded,
            'stranded_per_100': None if rate is None else round(rate, RATE_DIGITS),
        }


def count_window(records: Sequence[Record], followers: FollowerIndex, *, start: datetime, end: datetime) -> WindowCounts:
    """Counted blocking L0s filed in [start, end); followers index every record, not just this window's."""
    counted = [record for record in records if record.is_counted_l0 and start <= record.filed_at < end]

    def closed_as(*closures: Closure) -> list[Record]:
        return [record for record in counted if record.closure in closures]

    def wrong(group: Iterable[Record]) -> int:
        return sum(wrongly_dismissed(record, followers) for record in group)

    reaper_dismissals = closed_as(Closure.REAPER_DISMISSAL)
    startup_dismissals = closed_as(Closure.STARTUP_STRAND, Closure.STARTUP_DISMISSAL)
    workflow_auto_dismissals = closed_as(Closure.WORKFLOW_AUTO_DISMISSAL)
    return WindowCounts(
        filed=len(counted),
        reaper_promotions=len(closed_as(Closure.REAPER_PROMOTION)),
        stale_strands=len(closed_as(Closure.STARTUP_STRAND)),
        other_promotions=len(closed_as(Closure.OTHER_PROMOTION)),
        reaper_dismissals=len(reaper_dismissals),
        reaper_wrong=wrong(reaper_dismissals),
        startup_dismissals=len(startup_dismissals),
        startup_wrong=wrong(startup_dismissals),
        workflow_auto_dismissals=len(workflow_auto_dismissals),
        workflow_auto_wrong=wrong(workflow_auto_dismissals),
    )


@dataclass(frozen=True)
class Report:
    window_days: int
    max_pending_hours: float
    pre: WindowCounts
    post: WindowCounts
    pending_over_h: int

    def as_json(self) -> dict[str, Any]:
        return {
            'window_days': self.window_days,
            'max_pending_hours': self.max_pending_hours,
            'pre': self.pre.as_json(),
            'post': self.post.as_json(),
            'pending_over_h': self.pending_over_h,
        }


def measure(records: Sequence[Record], *, now: datetime, window_days: int, max_pending_hours: float) -> Report:
    """post = [now - N days, unbounded), so a clock-skewed filing lands in post; pre = the N days before."""
    post_start = now - timedelta(days=window_days)
    pre_start = post_start - timedelta(days=window_days)
    max_pending_age = timedelta(hours=max_pending_hours)
    followers = index_followers(records)
    return Report(
        window_days=window_days,
        max_pending_hours=max_pending_hours,
        pre=count_window(records, followers, start=pre_start, end=post_start),
        post=count_window(records, followers, start=post_start, end=UNBOUNDED),
        pending_over_h=sum(
            1
            for record in records
            if record.is_counted_l0 and record.closure is Closure.OPEN and now - record.filed_at > max_pending_age
        ),
    )


class Breach(enum.StrEnum):
    STRANDED_RATE = 'stranded_rate'
    REAPER_WRONG = 'reaper_wrong'
    PENDING_OVER_H = 'pending_over_h'


@dataclass(frozen=True)
class Thresholds:
    baseline_stranded_per_100: float
    ratio: float
    min_stranded: int
    max_reaper_wrong: int


def stranded_bar_per_100(report: Report, thresholds: Thresholds) -> float:
    """R * max(pre rate, F): an empty pre window counts as 0, so the baseline floor F applies."""
    return thresholds.ratio * max(report.pre.stranded_per_100 or 0.0, thresholds.baseline_stranded_per_100)


def judge(report: Report, thresholds: Thresholds) -> list[Breach]:
    post_rate = report.post.stranded_per_100
    breaches: list[Breach] = []
    if (
        post_rate is not None
        and post_rate >= stranded_bar_per_100(report, thresholds)
        and report.post.stranded >= thresholds.min_stranded
    ):
        breaches.append(Breach.STRANDED_RATE)
    if report.post.reaper_wrong >= thresholds.max_reaper_wrong:
        breaches.append(Breach.REAPER_WRONG)
    if report.pending_over_h > 0:
        breaches.append(Breach.PENDING_OVER_H)
    return breaches


def resolve_exit_code(breaches: Sequence[Breach]) -> int:
    return 1 if breaches else 0


def measure_queue(queue_dir: Path, *, now: datetime, window_days: int, max_pending_hours: float) -> Report:
    """measure() over queue_dir, refusing a post window with nothing filed: no producer wrote, so nothing to judge."""
    report = measure(load_records(queue_dir), now=now, window_days=window_days, max_pending_hours=max_pending_hours)
    if report.post.filed == 0:
        raise NothingFiled(queue_dir, f'no blocking L0 filed in the last {window_days} days')
    return report


def _shown_rate(rate: float | None) -> str:
    return 'n/a' if rate is None else f'{rate:.{RATE_DIGITS}f}/100'


def _window_lines(label: str, counts: WindowCounts) -> list[str]:
    return [
        f'{label}: filed {counts.filed}, stranded {counts.stranded} '
        f'({counts.reaper_promotions} reaper promotions + {counts.stale_strands} stale-strand) '
        f'= {_shown_rate(counts.stranded_per_100)}',
        f'  reaper dismissals {counts.reaper_dismissals}, wrong {counts.reaper_wrong}',
        f'  promotions by other resolvers {counts.other_promotions} (reported, not judged)',
        f'  startup dismissals {counts.startup_dismissals}, wrong {counts.startup_wrong} (reported, not judged)',
        f'  workflow auto-dismissals {counts.workflow_auto_dismissals}, '
        f'wrong {counts.workflow_auto_wrong} (reported, not judged)',
    ]


def render(report: Report) -> list[str]:
    days = report.window_days
    return [
        f'blocking L0s on numeric task ids, two {days}-day windows',
        *_window_lines(f'post (last {days} days)', report.post),
        *_window_lines(f'pre ({days} days before that)', report.pre),
        f'pending longer than {report.max_pending_hours:g}h: {report.pending_over_h}',
    ]


def breach_line(breach: Breach, report: Report, thresholds: Thresholds) -> str:
    post = report.post
    match breach:
        case Breach.STRANDED_RATE:
            measured = (
                f'post {_shown_rate(post.stranded_per_100)} >= bar '
                f'{_shown_rate(stranded_bar_per_100(report, thresholds))} '
                f'(ratio {thresholds.ratio:g} x max(pre {_shown_rate(report.pre.stranded_per_100)}, '
                f'baseline {thresholds.baseline_stranded_per_100:g}/100)) '
                f'and post stranded {post.stranded} >= {thresholds.min_stranded}'
            )
        case Breach.REAPER_WRONG:
            measured = f'post reaper_wrong {post.reaper_wrong} >= {thresholds.max_reaper_wrong}'
        case Breach.PENDING_OVER_H:
            measured = f'{report.pending_over_h} blocking L0s pending longer than {report.max_pending_hours:g}h'
    return f'breach: {breach}: {measured}'


def verdict_json(report: Report, thresholds: Thresholds, breaches: Sequence[Breach]) -> dict[str, Any]:
    return {
        **report.as_json(),
        'stranded_bar_per_100': round(stranded_bar_per_100(report, thresholds), RATE_DIGITS),
        'breaches': [breach.value for breach in breaches],
        'follow_up': REDESIGN_FOLLOW_UP if breaches else None,
    }


def compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(',', ':'), sort_keys=True)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument('--queue-dir', type=Path, required=True)
    parser.add_argument('--window-days', type=int, required=True)
    parser.add_argument('--baseline-stranded-per-100', type=float, required=True)
    parser.add_argument('--ratio', type=float, required=True)
    parser.add_argument('--min-stranded', type=int, required=True)
    parser.add_argument('--max-reaper-wrong', type=int, required=True)
    parser.add_argument('--max-pending-hours', type=float, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = Thresholds(
        baseline_stranded_per_100=args.baseline_stranded_per_100,
        ratio=args.ratio,
        min_stranded=args.min_stranded,
        max_reaper_wrong=args.max_reaper_wrong,
    )
    try:
        report = measure_queue(
            args.queue_dir,
            now=datetime.now(UTC),
            window_days=args.window_days,
            max_pending_hours=args.max_pending_hours,
        )
    except CannotJudge as exc:
        print(f'{exc.headline}: {exc.path} ({exc.reason})')
        return CANNOT_JUDGE_EXIT
    breaches = judge(report, thresholds)
    lines = [*render(report), *(breach_line(breach, report, thresholds) for breach in breaches)]
    if breaches:
        lines.append(f'follow-up: {REDESIGN_FOLLOW_UP}')
    for line in lines:
        print(line)
    print(compact_json(verdict_json(report, thresholds, breaches)))
    return resolve_exit_code(breaches)


if __name__ == '__main__':
    sys.exit(main())
