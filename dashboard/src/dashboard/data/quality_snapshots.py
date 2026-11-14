"""The one reader of both quality-metrics snapshot homes: the committed baselines under
``<root>/plans/quality-metrics/`` and the nightly files under ``<nightly_dir>/<project_id>/``
(scripts/quality_snapshot_cadence.py::DEFAULT_DATA_DIR).

Every figure is the instrument's own: scripts/quality_metrics_snapshot.py runs as a subprocess
(``--summary --json``, ``--current --diff``) on a loop thread, and its output is decoded here at
the boundary. Nothing re-derives a measure. Design: plans/quality-dashboard-prd.md decisions 11,
17 and 20.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any

from shared.project_id import fold_project_token
from shared.quality_series.git_lines import GitObjectMissing, GitTimeout, git_output

from dashboard.data.datum import Datum, DatumState, aged_at, unknown_datum
from dashboard.data.quality_markers import SnapshotIdentity
from dashboard.data.quality_root_tick import FACTORY_REPO

BASELINE_HOME = Path('plans/quality-metrics')
"""The committed baselines' home, relative to a project root."""

INSTRUMENT_SCRIPT = FACTORY_REPO / 'scripts' / 'quality_metrics_snapshot.py'

INSTRUMENT_TIMEOUT_SECONDS = 60.0
"""Bounds one instrument run. Measured 2026-11-14: --summary of df's 2.1 MB baseline takes 0.25 s,
and a --diff about 2.2 s."""

SNAPSHOT_GIT_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# The two homes.


@dataclass(frozen=True)
class SnapshotFile:
    """One snapshot file; *point_id* is opaque, stable, and unique across both homes."""

    point_id: str
    path: Path
    baseline: bool


def _point_id(home: str, relative: str) -> str:
    return hashlib.sha256(f'{home}\0{relative}'.encode()).hexdigest()[:16]


def _json_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.glob('*.json') if path.is_file())


def _nightly_dirs(nightly_dir: Path, project_id: str) -> list[Path]:
    """Every subdirectory of *nightly_dir* whose name folds to *project_id*."""
    if not nightly_dir.is_dir():
        return []
    wanted = fold_project_token(project_id)
    return sorted(
        directory for directory in nightly_dir.iterdir()
        if directory.is_dir() and fold_project_token(directory.name) == wanted
    )


def snapshot_files(root: Path, nightly_dir: Path, project_id: str) -> list[SnapshotFile]:
    """Both homes' ``*.json`` files, baselines first; a missing home contributes nothing."""
    baselines = [
        SnapshotFile(_point_id('baseline', path.name), path, baseline=True)
        for path in _json_files(root / BASELINE_HOME)
    ]
    nightly = [
        SnapshotFile(_point_id('nightly', f'{directory.name}/{path.name}'), path, baseline=False)
        for directory in _nightly_dirs(nightly_dir, project_id)
        for path in _json_files(directory)
    ]
    return baselines + nightly


# ---------------------------------------------------------------------------
# The instrument, run as a subprocess.


class InstrumentRefused(Exception):
    """The instrument exited non-zero over *path*; its exit code and stderr, verbatim."""

    def __init__(self, path: Path, exit_code: int, stderr: str) -> None:
        super().__init__(f'{path}: the instrument exited {exit_code}: {stderr.strip()}')
        self.path = path
        self.exit_code = exit_code
        self.stderr = stderr


def _run_instrument(instrument: Path, args: Sequence[str], path: Path, timeout_s: float) -> str:
    proc = subprocess.run(
        [sys.executable, str(instrument), *args],
        capture_output=True, text=True, timeout=timeout_s, check=False,
    )
    if proc.returncode != 0:
        raise InstrumentRefused(path, proc.returncode, proc.stderr)
    return proc.stdout


def diff_text(
    current: Path, previous: Path, *,
    instrument: Path = INSTRUMENT_SCRIPT, timeout_s: float = INSTRUMENT_TIMEOUT_SECONDS,
) -> str:
    """The instrument's ``--current CURRENT --diff PREVIOUS`` report, verbatim."""
    return _run_instrument(
        instrument, ['--current', str(current), '--diff', str(previous)], current, timeout_s
    )


def commit_time(root: Path, sha: str, timeout_s: float) -> datetime:
    """*sha*'s committer time in *root*; GitObjectMissing when *root* has no such commit."""
    stdout = git_output(root, ['show', '-s', '--format=%cI', sha], timeout_s, objects=(sha,))
    return datetime.fromisoformat(stdout.decode().strip())


# ---------------------------------------------------------------------------
# --summary --json, decoded at the boundary.

CellValue = int | tuple[int, int]
"""A count or total, or the (max, total) cognitive pair."""


@dataclass(frozen=True)
class SummaryColumn:
    slug: str
    header: str


@dataclass(frozen=True)
class SummaryCell:
    value: CellValue
    text: str  # exactly as the instrument's markdown --summary prints it


@dataclass(frozen=True)
class MemberRow:
    member: str
    pseudo: bool
    cells: tuple[SummaryCell, ...]  # one per column, in the instrument's column order


@dataclass(frozen=True)
class H14Counts:
    """Heuristic 14's two marks as the snapshot's params record them, and the files at each."""

    soft_ceiling_lines: int
    alarm_lines: int
    at_soft_ceiling: int
    at_alarm: int


@dataclass(frozen=True)
class CycleCounts:
    """None where the snapshot's schema does not record the set: unknown, never zero."""

    cycles: int
    hidden_cycles: int | None
    typing_cycles: int | None


@dataclass(frozen=True)
class SnapshotSummary:
    """One snapshot as the instrument's --summary defines it; no domain line totals are kept.

    *document* is the instrument's JSON verbatim, which is what quality.db stores.
    """

    as_of_sha: str
    run_id: str
    params: Mapping[str, object]
    unknown_params: frozenset[str]
    complete: bool
    columns: tuple[SummaryColumn, ...]
    members: tuple[MemberRow, ...]
    h14: H14Counts
    cycles: CycleCounts
    deferred_imports: int
    cycle_closing_imports: int | None
    private_patch_targets: int
    document: str = field(compare=False, repr=False)

    @classmethod
    def from_document(cls, document: str) -> SnapshotSummary:
        """Decode the instrument's --summary --json; a ValueError names the first field refused."""
        return _decode_summary(_object(json.loads(document), 'the summary'), document)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _object(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'{where} is {type(value).__name__}, expected a JSON object')
    return value


def _list(value: object, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f'{where} is {type(value).__name__}, expected a JSON array')
    return value


def _field(obj: Mapping[str, Any], key: str, where: str, accepts: Callable[[object], bool], expected: str) -> Any:
    if key not in obj:
        raise ValueError(f'{where}.{key} is missing from the instrument summary')
    value = obj[key]
    if not accepts(value):
        raise ValueError(f'{where}.{key} is {value!r}, expected {expected}')
    return value


def _str(obj: Mapping[str, Any], key: str, where: str) -> str:
    return _field(obj, key, where, lambda value: isinstance(value, str), 'a string')


def _bool(obj: Mapping[str, Any], key: str, where: str) -> bool:
    return _field(obj, key, where, lambda value: isinstance(value, bool), 'a boolean')


def _int(obj: Mapping[str, Any], key: str, where: str) -> int:
    return _field(obj, key, where, _is_int, 'an integer')


def _count_or_unknown(obj: Mapping[str, Any], key: str, where: str) -> int | None:
    return _field(obj, key, where, lambda value: value is None or _is_int(value), 'an integer or null')


def _cell_value(value: object, where: str) -> CellValue:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, list) and len(value) == 2 and all(_is_int(part) for part in value):
        return value[0], value[1]
    raise ValueError(f'{where}.value is {value!r}, expected an integer or a [max, total] pair')


def _cell(value: object, where: str) -> SummaryCell:
    obj = _object(value, where)
    return SummaryCell(_cell_value(obj.get('value'), where), _str(obj, 'text', where))


def _cells(value: object, where: str, width: int) -> tuple[SummaryCell, ...]:
    cells = _list(value, where)
    if len(cells) != width:
        raise ValueError(f'{where} has {len(cells)} cells, expected one per column ({width})')
    return tuple(_cell(cell, f'{where}[{index}]') for index, cell in enumerate(cells))


def _column(value: object, where: str) -> SummaryColumn:
    obj = _object(value, where)
    return SummaryColumn(_str(obj, 'slug', where), _str(obj, 'header', where))


def _member_row(value: object, where: str, width: int) -> MemberRow:
    obj = _object(value, where)
    return MemberRow(_str(obj, 'member', where), _bool(obj, 'pseudo', where), _cells(obj.get('cells'), f'{where}.cells', width))


def _rows(summary: Mapping[str, Any], key: str, decode: Callable[[object, str], Any]) -> tuple[Any, ...]:
    where = f'the summary.{key}'
    return tuple(decode(item, f'{where}[{index}]') for index, item in enumerate(_list(summary.get(key), where)))


def _domain_count(columns: Sequence[SummaryColumn], cells: Sequence[SummaryCell], slug: str) -> int:
    """The domain row's integer cell in column *slug*."""
    index = next((at for at, column in enumerate(columns) if column.slug == slug), None)
    if index is None:
        raise ValueError(f'the summary.columns has no {slug!r} column')
    value = cells[index].value
    if not isinstance(value, int):
        raise ValueError(f'the summary.domain {slug!r} cell is {value!r}, expected an integer')
    return value


def _unknown_params(summary: Mapping[str, Any]) -> frozenset[str]:
    names = _field(
        summary, 'unknown_params', 'the summary',
        lambda value: isinstance(value, list) and all(isinstance(name, str) for name in value),
        'a list of strings',
    )
    return frozenset(names)


def _decode_summary(summary: Mapping[str, Any], document: str) -> SnapshotSummary:
    top = 'the summary'
    columns: tuple[SummaryColumn, ...] = _rows(summary, 'columns', _column)
    params = _object(summary.get('params'), f'{top}.params')
    domain = _cells(_object(summary.get('domain'), f'{top}.domain').get('cells'), f'{top}.domain.cells', len(columns))
    graph_at = f'{top}.import_graph'
    graph = _object(summary.get('import_graph'), graph_at)
    return SnapshotSummary(
        as_of_sha=_str(summary, 'as_of_sha', top),
        run_id=_str(summary, 'run_id', top),
        params=params,
        unknown_params=_unknown_params(summary),
        complete=_bool(summary, 'complete', top),
        columns=columns,
        members=_rows(summary, 'members', lambda row, where: _member_row(row, where, len(columns))),
        h14=H14Counts(
            soft_ceiling_lines=_int(params, 'h14_soft_ceiling_lines', f'{top}.params'),
            alarm_lines=_int(params, 'h14_alarm_lines', f'{top}.params'),
            at_soft_ceiling=_domain_count(columns, domain, 'files_at_soft_ceiling'),
            at_alarm=_domain_count(columns, domain, 'files_at_alarm'),
        ),
        cycles=CycleCounts(
            cycles=_int(graph, 'cycles', graph_at),
            hidden_cycles=_count_or_unknown(graph, 'hidden_cycles', graph_at),
            typing_cycles=_count_or_unknown(graph, 'typing_cycles', graph_at),
        ),
        deferred_imports=_int(graph, 'deferred_imports', graph_at),
        cycle_closing_imports=_count_or_unknown(graph, 'cycle_closing_imports', graph_at),
        private_patch_targets=_domain_count(columns, domain, 'private_patch_targets'),
        document=document,
    )


def summarise(
    path: Path, *, instrument: Path = INSTRUMENT_SCRIPT, timeout_s: float = INSTRUMENT_TIMEOUT_SECONDS,
) -> SnapshotSummary:
    """The instrument's ``--summary PATH --json``, decoded; InstrumentRefused when it refuses *path*."""
    document = _run_instrument(instrument, ['--summary', str(path), '--json'], path, timeout_s)
    return SnapshotSummary.from_document(document)



# ---------------------------------------------------------------------------
# One root's scan on a worker thread: cached by file state, then by content digest plus summariser
# digest; diffs by identity pair plus summariser digest.


def summariser_digest(instrument: Path) -> str:
    """sha256 of the instrument and the source_measures beside it: what a cached figure was computed by."""
    digest = hashlib.sha256()
    for path in (instrument, instrument.with_name('source_measures.py')):
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


class DiffState(StrEnum):
    FIRST = 'first'
    READY = 'ready'
    PENDING = 'pending'
    FAILED = 'failed'


@dataclass(frozen=True)
class FileState:
    """One snapshot file as last read, and the summariser that read it."""

    point_id: str
    file: Path
    baseline: bool
    size: int
    mtime_ns: int
    digest: str
    summariser: str


def _state_fields(state: FileState) -> dict[str, Any]:
    return {name.name: getattr(state, name.name) for name in fields(FileState)}


def cache_identity(summary: SnapshotSummary, digest: str) -> str:
    """(as_of_sha, canonical params) when every params key is known; else the content digest,
    because unknown equals no identity."""
    if summary.unknown_params:
        return f'digest:{digest}'
    return f'{summary.as_of_sha} {json.dumps(summary.params, sort_keys=True)}'


def pair_key(previous_identity: str, current_identity: str, summariser: str) -> str:
    """The diff cache key: both cache identities plus the summariser digest."""
    return hashlib.sha256(json.dumps([previous_identity, current_identity, summariser]).encode()).hexdigest()[:32]


def point_order(committed_at: datetime, baseline: bool, point_id: str) -> tuple[datetime, bool, str]:
    """The series' order: by commit time, a baseline before a nightly file at the same commit, then id."""
    return committed_at, not baseline, point_id


def _utc_iso(at: datetime) -> str:
    return at.astimezone(UTC).isoformat()


@dataclass(frozen=True)
class PointDiff:
    """A point's diff against the point before it; no previous point and no pair key for the first."""

    previous_id: str | None
    pair_key: str | None
    state: DiffState
    reason: str | None = None

    def to_wire(self) -> dict[str, object]:
        return {'previous_id': self.previous_id, 'state': self.state.value, 'reason': self.reason}


@dataclass(frozen=True)
class ScannedPoint(FileState):
    """A placed snapshot: summarised by the instrument, and its as_of_sha's commit time known."""

    summary: SnapshotSummary
    committed_at: datetime
    diff: PointDiff

    def identity(self) -> SnapshotIdentity:
        return SnapshotIdentity(
            as_of_sha=self.summary.as_of_sha, committed_at=self.committed_at,
            params=self.summary.params, unknown_params=self.summary.unknown_params,
        )

    def to_wire(self) -> dict[str, object]:
        """The Contract's SnapshotPoint; member cells as the instrument's texts only, and no domain row."""
        summary = self.summary
        return {
            'id': self.point_id,
            'file': str(self.file),
            'baseline': self.baseline,
            'as_of_sha': summary.as_of_sha,
            'committed_at': _utc_iso(self.committed_at),
            'run_id': summary.run_id,
            'params': dict(summary.params),
            'unknown_params': sorted(summary.unknown_params),
            'complete': summary.complete,
            'members': {
                'columns': [column.header for column in summary.columns],
                'rows': [
                    {'member': row.member, 'pseudo': row.pseudo, 'cells': [cell.text for cell in row.cells]}
                    for row in summary.members
                ],
            },
            'h14': asdict(summary.h14),
            'cycles': asdict(summary.cycles),
            'deferred_imports': summary.deferred_imports,
            'cycle_closing_imports': summary.cycle_closing_imports,
            'private_patch_targets': summary.private_patch_targets,
            'diff': self.diff.to_wire(),
        }


@dataclass(frozen=True)
class UnplacedFile(FileState):
    """A file served with its reason instead of placed: the instrument refused it (no summary), or
    git cannot resolve its as_of_sha."""

    reason: str
    summary: SnapshotSummary | None = None

    def to_wire(self) -> dict[str, object]:
        return {'id': self.point_id, 'file': str(self.file), 'baseline': self.baseline, 'reason': self.reason}


@dataclass(frozen=True)
class SettledDiff:
    state: DiffState  # READY or FAILED
    reason: str | None = None


@dataclass(frozen=True)
class ComputedDiff:
    """A diff this scan ran: READY with the instrument's text, or FAILED with its refusal."""

    state: DiffState
    text: str | None
    reason: str | None


_SETTLED = frozenset({DiffState.READY, DiffState.FAILED})


@dataclass(frozen=True)
class RootScan:
    points: tuple[ScannedPoint, ...]
    unplaced: tuple[UnplacedFile, ...]
    new_diffs: Mapping[str, ComputedDiff]
    pending_summaries: int
    pending_diffs: int
    baseline_home: Path
    nightly_home: Path


@dataclass(frozen=True)
class SnapshotCache:
    """A root's stored file readings by point id, and its settled diffs by pair key."""

    files: Mapping[str, ScannedPoint | UnplacedFile]
    diffs: Mapping[str, SettledDiff]

    @classmethod
    def empty(cls) -> SnapshotCache:
        return cls({}, {})

    @classmethod
    def from_scan(cls, scan: RootScan) -> SnapshotCache:
        return cls(
            {reading.point_id: reading for reading in (*scan.points, *scan.unplaced)},
            {
                point.diff.pair_key: SettledDiff(point.diff.state, point.diff.reason)
                for point in scan.points
                if point.diff.pair_key is not None and point.diff.state in _SETTLED
            },
        )


@dataclass(frozen=True)
class _Refusal:
    stderr: str


@dataclass(frozen=True)
class _Summarised:
    """A file the instrument has read, and its commit time once known."""

    state: FileState
    outcome: SnapshotSummary | _Refusal
    committed_at: datetime | None = None


@dataclass(frozen=True)
class _Placed:
    state: FileState
    summary: SnapshotSummary
    committed_at: datetime

    @property
    def identity(self) -> str:
        return cache_identity(self.summary, self.state.digest)

    def order(self) -> tuple[datetime, bool, str]:
        return point_order(self.committed_at, self.state.baseline, self.state.point_id)


def _past(deadline: float) -> bool:
    return time.monotonic() >= deadline


def _reused(cached: ScannedPoint | UnplacedFile) -> _Summarised:
    state = FileState(**_state_fields(cached))
    if isinstance(cached, ScannedPoint):
        return _Summarised(state, cached.summary, cached.committed_at)
    return _Summarised(state, cached.summary if cached.summary is not None else _Refusal(cached.reason))


def _stats(files: Sequence[SnapshotFile]) -> list[tuple[SnapshotFile, os.stat_result]]:
    """Each file still present with its stat, newest first."""
    stats = []
    for file in files:
        with contextlib.suppress(FileNotFoundError):
            stats.append((file, file.path.stat()))
    return sorted(stats, key=lambda pair: pair[1].st_mtime_ns, reverse=True)


class _Summariser:
    """Summarises a root's files, reusing what the cache or this scan already holds for the same bytes."""

    def __init__(self, cache: SnapshotCache, summariser: str, instrument: Path, deadline: float) -> None:
        self.cache, self.summariser, self.instrument, self.deadline = cache, summariser, instrument, deadline
        self.by_digest = {
            cached.digest: _reused(cached) for cached in cache.files.values() if cached.summariser == summariser
        }
        self.pending = 0

    def read(self, file: SnapshotFile, stat: os.stat_result) -> _Summarised | None:
        """*file*'s summary or refusal; None, counted pending, when the deadline or the instrument ran out."""
        cached = self.cache.files.get(file.point_id)
        if cached is not None and (cached.size, cached.mtime_ns, cached.summariser) == (
            stat.st_size, stat.st_mtime_ns, self.summariser
        ):
            return _reused(cached)
        if _past(self.deadline):
            self.pending += 1
            return None
        data = file.path.read_bytes()
        state = FileState(file.point_id, file.path, file.baseline, stat.st_size, stat.st_mtime_ns,
                          hashlib.sha256(data).hexdigest(), self.summariser)
        twin = self.by_digest.get(state.digest)
        summarised = _Summarised(state, twin.outcome, twin.committed_at) if twin else self._summarise(state)
        if summarised is not None:
            self.by_digest[state.digest] = summarised
        return summarised

    def _summarise(self, state: FileState) -> _Summarised | None:
        try:
            return _Summarised(state, summarise(state.file, instrument=self.instrument))
        except InstrumentRefused as refused:
            return _Summarised(state, _Refusal(refused.stderr))
        except subprocess.TimeoutExpired:
            self.pending += 1
            return None


def _place(root: Path, summarised: _Summarised, deadline: float) -> _Placed | UnplacedFile | None:
    """*summarised* with its commit time, unplaced with its reason, or None when time ran out.

    A sha git cannot resolve is asked again every scan: a later fetch may bring it.
    """
    state, outcome = summarised.state, summarised.outcome
    if isinstance(outcome, _Refusal):
        return UnplacedFile(**_state_fields(state), reason=outcome.stderr)
    if summarised.committed_at is not None:
        return _Placed(state, outcome, summarised.committed_at)
    if _past(deadline):
        return None
    try:
        return _Placed(state, outcome, commit_time(root, outcome.as_of_sha, SNAPSHOT_GIT_TIMEOUT_SECONDS))
    except GitObjectMissing:
        reason = f'as_of_sha {outcome.as_of_sha} is not a commit in {root}'
        return UnplacedFile(**_state_fields(state), reason=reason, summary=outcome)
    except GitTimeout:
        return None


class _Differ:
    """Each placed point's diff against the one before it, newest pair first, settled pairs reused."""

    def __init__(self, cache: SnapshotCache, instrument: Path, deadline: float) -> None:
        self.cache, self.instrument, self.deadline = cache, instrument, deadline
        self.new_diffs: dict[str, ComputedDiff] = {}
        self.pending = 0

    def diff(self, previous: _Placed, current: _Placed) -> PointDiff:
        key = pair_key(previous.identity, current.identity, current.state.summariser)
        settled = self.cache.diffs.get(key) or self.new_diffs.get(key) or self._compute(key, previous, current)
        if settled is None:
            self.pending += 1
            return PointDiff(previous.state.point_id, key, DiffState.PENDING)
        return PointDiff(previous.state.point_id, key, settled.state, settled.reason)

    def _compute(self, key: str, previous: _Placed, current: _Placed) -> ComputedDiff | None:
        if _past(self.deadline):
            return None
        try:
            text = diff_text(current.state.file, previous.state.file, instrument=self.instrument)
        except InstrumentRefused as refused:
            self.new_diffs[key] = ComputedDiff(DiffState.FAILED, None, refused.stderr)
        except subprocess.TimeoutExpired:
            return None
        else:
            self.new_diffs[key] = ComputedDiff(DiffState.READY, text, None)
        return self.new_diffs[key]


def _scanned(placed: _Placed, diff: PointDiff) -> ScannedPoint:
    return ScannedPoint(
        **_state_fields(placed.state), summary=placed.summary, committed_at=placed.committed_at, diff=diff
    )


def _diffed(placed: Sequence[_Placed], differ: _Differ) -> tuple[ScannedPoint, ...]:
    """The points in order, their diffs computed newest pair first."""
    diffs = {
        index: differ.diff(placed[index - 1], placed[index]) for index in range(len(placed) - 1, 0, -1)
    }
    first = PointDiff(None, None, DiffState.FIRST)
    return tuple(_scanned(point, diffs.get(index, first)) for index, point in enumerate(placed))


def scan_root(
    root: Path, project_id: str, nightly_dir: Path, cache: SnapshotCache, *,
    deadline: float, instrument: Path = INSTRUMENT_SCRIPT,
) -> RootScan:
    """One root's snapshot scan under a time.monotonic() *deadline*, newest input first: summaries,
    then commit times, then diffs. What the deadline leaves undone is counted pending, never guessed."""
    summariser = _Summariser(cache, summariser_digest(instrument), instrument, deadline)
    read = [summarised for file, stat in _stats(snapshot_files(root, nightly_dir, project_id))
            if (summarised := summariser.read(file, stat)) is not None]
    placements = [_place(root, summarised, deadline) for summarised in read]
    placed = sorted((point for point in placements if isinstance(point, _Placed)), key=_Placed.order)
    differ = _Differ(cache, instrument, deadline)
    return RootScan(
        points=_diffed(placed, differ),
        unplaced=tuple(point for point in placements if isinstance(point, UnplacedFile)),
        new_diffs=differ.new_diffs,
        pending_summaries=summariser.pending + placements.count(None),
        pending_diffs=differ.pending,
        baseline_home=root / BASELINE_HOME,
        nightly_home=nightly_dir / fold_project_token(project_id),
    )



# ---------------------------------------------------------------------------
# The stored series and the QUALITY_SNAPSHOTS datum; the request path only reads these.


@dataclass(frozen=True)
class SnapshotSeries:
    """Every placed point in point_order, every unplaced file with its reason, and what is pending."""

    points: tuple[ScannedPoint, ...]
    unplaced: tuple[UnplacedFile, ...]
    pending_summaries: int
    pending_diffs: int

    def __post_init__(self) -> None:
        for before, after in pairwise(self.points):
            if point_order(before.committed_at, before.baseline, before.point_id) >= point_order(
                after.committed_at, after.baseline, after.point_id
            ):
                raise ValueError(
                    'snapshot points must be in (committed_at, baseline first, point_id) order: '
                    f'{before.point_id} ({_utc_iso(before.committed_at)}) comes before '
                    f'{after.point_id} ({_utc_iso(after.committed_at)})'
                )

    def to_wire(self) -> dict[str, object]:
        return {
            'points': [point.to_wire() for point in self.points],
            'unplaced': [file.to_wire() for file in self.unplaced],
            'pending': {'summaries': self.pending_summaries, 'diffs': self.pending_diffs},
        }


@dataclass(frozen=True)
class StoredScan:
    """The newest successful scan of a project's two homes."""

    series: SnapshotSeries
    scanned_at: datetime
    baseline_home: Path
    nightly_home: Path


@dataclass(frozen=True)
class StoredSnapshots:
    """A project's newest successful scan, if any, and its newest attempt."""

    scan: StoredScan | None
    attempted_at: datetime
    failure: str | None


@dataclass(frozen=True)
class StoredDiff:
    """One point's diff as stored: its state, the point it is against, and the text when ready."""

    state: DiffState
    previous_id: str | None
    text: str | None
    reason: str | None


def _no_snapshot_reason(scan: StoredScan) -> str:
    return (f'no snapshot yet: neither {scan.baseline_home}/ nor {scan.nightly_home}/ holds a '
            f'snapshot file (scanned at {_utc_iso(scan.scanned_at)})')


def _newer_failure(stored: StoredSnapshots, scan: StoredScan) -> str | None:
    if stored.failure is None or stored.attempted_at <= scan.scanned_at:
        return None
    return (f'the newest snapshot scan (at {_utc_iso(stored.attempted_at)}) failed: {stored.failure}; '
            f'serving the scan of {_utc_iso(scan.scanned_at)}')


def snapshot_datum(
    project_id: str, stored: StoredSnapshots | None, *, now: datetime, freshness_bound_seconds: int,
) -> Datum[SnapshotSeries]:
    """QUALITY_SNAPSHOTS for one project, as of its newest successful scan."""
    if stored is None:
        return unknown_datum(f'the snapshot homes of {project_id} have not been scanned yet', freshness_bound_seconds)
    scan = stored.scan
    if scan is None:
        return unknown_datum(f'the snapshot scan of {project_id} failed: {stored.failure}', freshness_bound_seconds)
    series = scan.series
    if not (series.points or series.unplaced or series.pending_summaries):
        return unknown_datum(_no_snapshot_reason(scan), freshness_bound_seconds)
    stale = _newer_failure(stored, scan)
    if stale is None and series.pending_summaries:
        stale = f'{series.pending_summaries} snapshot files not yet summarised'
    if stale is not None:
        return Datum(series, scan.scanned_at, DatumState.STALE, stale, freshness_bound_seconds)
    return aged_at(Datum(series, scan.scanned_at, DatumState.FRESH, None, freshness_bound_seconds), now)
