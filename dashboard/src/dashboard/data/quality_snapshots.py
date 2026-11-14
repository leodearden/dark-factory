"""The one reader of both quality-metrics snapshot homes: the committed baselines under
``<root>/plans/quality-metrics/`` and the nightly files under ``<nightly_dir>/<project_id>/``
(scripts/quality_snapshot_cadence.py::DEFAULT_DATA_DIR).

Every figure is the instrument's own: scripts/quality_metrics_snapshot.py runs as a subprocess
(``--summary --json``, ``--current --diff``) on a loop thread, and its output is decoded here at
the boundary. Nothing re-derives a measure. Design: plans/quality-dashboard-prd.md decisions 11,
17 and 20.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from shared.project_id import fold_project_token
from shared.quality_series.git_lines import git_output

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
