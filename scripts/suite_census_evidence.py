"""Evidence readers for the suite census (task 5414): retained verify artefacts become typed records.

The I/O rim of ``suite_census_outcomes``. Each artefact format -- archived and
live junit, pytest logs, the runs.db ``flake_occurrence`` table -- is parsed
once, here, into Observation / Unresolved values. An artefact that cannot be
read or attributed to a tracked test becomes an Unresolved record, so it is
counted rather than dropped (INV-11).
"""
from __future__ import annotations

import gzip
import re
import sqlite3
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from merge_lane_metrics import tracked_files
from suite_census_outcomes import (
    Evidence,
    Observation,
    Outcome,
    Record,
    SourceWindow,
    TestId,
    Unresolved,
)

ARCHIVED_JUNIT = 'data/verify-logs/*/attempt-*.junit-*.xml.gz'
LIVE_JUNIT = '.worktrees/*/.df-verify-junit/report*.xml'
VERIFY_LOGS = 'data/verify-logs/*/attempt-*.test-*.log'
RUNS_DB = 'data/orchestrator/runs.db'
FLAKE_QUERY = 'SELECT rowid, observed_at, test_id FROM flake_occurrence ORDER BY rowid'

# The producer is orchestrator/src/orchestrator/verify.py::_archive_junit_report.
_ARCHIVE_NAME = re.compile(
    r'^attempt-\d+(?:\.(?P<infix>.+))?\.(?:junit|test)-(?P<stamp>\d{8}T\d{6})(?:_\d+)?Z'
    r'\.(?:xml\.gz|log)$'
)
_LIVE_NAME = re.compile(r'^report(?:\.(?P<infix>.+))?\.xml$')
_FUNCTION = re.compile(r'[A-Za-z_]\w*')
_LOG_FAILURE = re.compile(r'^(?:FAILED|ERROR) (?P<nodeid>\S.*?)(?: - .*)?$')
_UNREADABLE_JUNIT = (OSError, EOFError, zlib.error, ET.ParseError)


def _utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def _iso_utc(text: str | None) -> str | None:
    """*text* as a UTC second-precision ISO stamp; unparseable text is shown as given."""
    if text is None:
        return None
    try:
        return _utc(datetime.fromisoformat(text))
    except ValueError:
        return text


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _window(source: str, pattern: str, artefacts: int, stamps: Iterable[str]) -> SourceWindow:
    stamps = sorted(stamps)
    return SourceWindow(
        source=source, pattern=pattern, present=artefacts > 0, artefacts=artefacts,
        first=stamps[0] if stamps else None, last=stamps[-1] if stamps else None,
    )


@dataclass(frozen=True)
class _Artefact:
    """One verify artefact file, its name parsed once into its module infix and stamp."""

    path: Path
    relative: str
    infix: str | None
    stamp: str | None


def _archived(project_root: Path, pattern: str) -> tuple[_Artefact, ...]:
    def artefact(path: Path) -> _Artefact:
        relative = _relative(path, project_root)
        parsed = _ARCHIVE_NAME.match(path.name)
        if parsed is None:
            return _Artefact(path=path, relative=relative, infix=None, stamp=None)
        return _Artefact(
            path=path, relative=relative, infix=parsed['infix'] or '',
            stamp=_utc(datetime.strptime(parsed['stamp'], '%Y%m%dT%H%M%S')),
        )

    return tuple(artefact(path) for path in sorted(project_root.glob(pattern)))


def _live(project_root: Path) -> tuple[_Artefact, ...]:
    def artefact(path: Path) -> _Artefact:
        parsed = _LIVE_NAME.match(path.name)
        return _Artefact(
            path=path, relative=_relative(path, project_root),
            infix=(parsed['infix'] or '') if parsed else None,
            stamp=_utc(datetime.fromtimestamp(path.stat().st_mtime, UTC)),
        )

    return tuple(artefact(path) for path in sorted(project_root.glob(LIVE_JUNIT)))


def _artefact_window(source: str, pattern: str, artefacts: Sequence[_Artefact]) -> SourceWindow:
    return _window(
        source, pattern, len(artefacts), (a.stamp for a in artefacts if a.stamp is not None),
    )


def _test_id(path: str, classes: Sequence[str], function: str) -> TestId:
    return TestId(package=path.split('/', 1)[0], name='::'.join((path, *classes, function)))


def _join(base: str, relative: str) -> str:
    return f'{base}/{relative}' if base else relative


@dataclass(frozen=True)
class _PytestTree:
    """The tracked .py files of the measured tree and its verify modules, by infix."""

    tracked: frozenset[str]
    module_by_infix: MappingProxyType[str, str]

    @classmethod
    def of(cls, tree_root: Path) -> _PytestTree:
        configs = tracked_files(tree_root, 'orchestrator.yaml', '*/orchestrator.yaml')
        dirs = (config.rpartition('/')[0] for config in configs)
        # Forward only: orchestrator/src/orchestrator/verify.py::_make_infix is the producer.
        return cls(
            tracked=frozenset(tracked_files(tree_root, '*.py')),
            module_by_infix=MappingProxyType(
                {d.replace('/', '_').replace(' ', '_'): d for d in dirs}
            ),
        )

    def module_of(self, infix: str | None) -> str | None:
        if infix is None:
            return None
        known = [k for k in self.module_by_infix if infix == k or infix.startswith(k + '.')]
        return self.module_by_infix[max(known, key=len)] if known else None

    def bases(self, module: str | None) -> tuple[str, ...]:
        if module is None:
            return (*sorted(self.module_by_infix.values()), '')
        return (module, '')

    def resolve_classname(self, classname: str, bases: Sequence[str]) -> tuple[str, tuple[str, ...]] | None:
        parts = classname.split('.')
        for base in bases:
            for cut in range(len(parts), 0, -1):
                path = _join(base, '/'.join(parts[:cut]) + '.py')
                if path in self.tracked:
                    return path, tuple(parts[cut:])
        return None

    def resolve_nodeid(self, nodeid: str, bases: Sequence[str], *, every: bool) -> list[TestId]:
        """The tests a pytest node id names: the first base's, or *every* base's when ambiguous."""
        parts = nodeid.split('[', 1)[0].split('::')
        function = _FUNCTION.match(parts[-1]) if len(parts) > 1 else None
        if function is None:
            return []
        paths = [path for base in bases if (path := _join(base, parts[0])) in self.tracked]
        if not every:
            paths = paths[:1]
        return [_test_id(path, parts[1:-1], function.group()) for path in paths]


def _junit_outcome(case: ET.Element) -> Outcome:
    tags = {child.tag for child in case}
    if 'failure' in tags or 'error' in tags:
        return Outcome.FAILED
    if 'skipped' in tags:
        return Outcome.SKIPPED
    return Outcome.PASSED


def _seconds(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class _JunitReader:
    """Reads junit reports for ONE records() pass; a run seen twice is read once."""

    def __init__(self, tree: _PytestTree) -> None:
        self._tree = tree
        self._ids: dict[tuple[str | None, str, str], TestId | None] = {}
        self._seen: set[tuple[str, str]] = set()

    def read(self, artefact: _Artefact, source: str) -> list[Record]:
        path = artefact.path
        try:
            with gzip.open(path, 'rb') if path.name.endswith('.gz') else path.open('rb') as handle:
                return self._parse(ET.iterparse(handle, events=('start', 'end')), artefact, source)
        except _UNREADABLE_JUNIT:
            return [Unresolved(source=source, raw_id=artefact.relative)]

    def _parse(
        self, events: Iterator[tuple[str, ET.Element]], artefact: _Artefact, source: str,
    ) -> list[Record]:
        module = self._tree.module_of(artefact.infix)
        run_key: tuple[str, str] | None = None
        records: list[Record] = []
        for event, element in events:
            if event == 'start':
                if element.tag == 'testsuite' and run_key is None and element.get('timestamp'):
                    run_key = (module or artefact.infix or '', element.get('timestamp', ''))
                    if run_key in self._seen:
                        return []
            elif element.tag == 'testcase':
                records.append(self._testcase(element, module, artefact.relative, source))
                element.clear()
        if run_key is not None:
            self._seen.add(run_key)
        return records

    def _testcase(self, case: ET.Element, module: str | None, run: str, source: str) -> Record:
        classname, name = case.get('classname', ''), case.get('name', '')
        function = _FUNCTION.match(name)
        test = self._test(module, classname, function.group()) if function else None
        if test is None:
            return Unresolved(source=source, raw_id=f'{classname}::{name}')
        return Observation(
            source=source, run=run, test=test, outcome=_junit_outcome(case),
            seconds=_seconds(case.get('time')),
        )

    def _test(self, module: str | None, classname: str, function: str) -> TestId | None:
        key = (module, classname, function)
        if key not in self._ids:
            resolved = self._tree.resolve_classname(classname, self._tree.bases(module))
            self._ids[key] = resolved and _test_id(resolved[0], resolved[1], function)
        return self._ids[key]


def _failure_records(
    source: str, run: str, raw_id: str, tests: Sequence[TestId],
) -> Iterator[Record]:
    if not tests:
        yield Unresolved(source=source, raw_id=raw_id)
    for test in tests:
        yield Observation(source=source, run=run, test=test, outcome=Outcome.FAILED, seconds=None)


def _pytest_log_records(artefact: _Artefact, tree: _PytestTree) -> Iterator[Record]:
    module = tree.module_of(artefact.infix)
    bases = tree.bases(module)
    try:
        with artefact.path.open(encoding='utf-8', errors='replace') as log:
            lines = [line.rstrip('\n') for line in log if line.startswith(('FAILED ', 'ERROR '))]
    except OSError:
        yield Unresolved(source='pytest-logs', raw_id=artefact.relative)
        return
    for line in lines:
        parsed = _LOG_FAILURE.match(line)
        if parsed is None:
            continue
        nodeid = parsed['nodeid']
        tests = tree.resolve_nodeid(nodeid, bases, every=module is None)
        yield from _failure_records('pytest-logs', artefact.relative, nodeid, tests)


# ---------------------------------------------------------------------------
# runs.db flake_occurrence, shared by both ecosystems.

def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)


def _flake_window(project_root: Path) -> SourceWindow:
    db = project_root / RUNS_DB
    pattern = f'{RUNS_DB}: {FLAKE_QUERY}'
    absent = SourceWindow('flake_occurrence', pattern, False, 0, None, None)
    if not db.is_file():
        return absent
    try:
        with closing(_connect_ro(db)) as conn:
            count, first, last = conn.execute(
                'SELECT COUNT(*), MIN(observed_at), MAX(observed_at) FROM flake_occurrence'
            ).fetchone()
    except sqlite3.Error:
        return absent
    return SourceWindow('flake_occurrence', pattern, True, count, _iso_utc(first), _iso_utc(last))


def _flake_records(
    project_root: Path, resolve: Callable[[str], Sequence[TestId]],
) -> Iterator[Record]:
    db = project_root / RUNS_DB
    if not db.is_file():
        return
    try:
        with closing(_connect_ro(db)) as conn:
            rows = conn.execute(FLAKE_QUERY).fetchall()
    except sqlite3.Error as exc:
        yield Unresolved(source='flake_occurrence', raw_id=f'{RUNS_DB}: {exc}')
        return
    for rowid, _observed_at, test_id in rows:
        yield from _failure_records(
            'flake_occurrence', f'flake_occurrence#{rowid}', test_id, resolve(test_id),
        )


# ---------------------------------------------------------------------------
# The pytest ecosystem.

def pytest_evidence(project_root: Path, tree_root: Path) -> Evidence:
    """Junit (archived, then live), pytest logs and flake_occurrence under *project_root*,
    resolved against the tracked tests of *tree_root*."""
    tree = _PytestTree.of(tree_root)
    archived = _archived(project_root, ARCHIVED_JUNIT)
    live = _live(project_root)
    logs = _archived(project_root, VERIFY_LOGS)

    def records() -> Iterator[Record]:
        junit = _JunitReader(tree)
        for artefact in archived:
            yield from junit.read(artefact, 'archived-junit')
        for artefact in live:
            yield from junit.read(artefact, 'live-junit')
        for artefact in logs:
            yield from _pytest_log_records(artefact, tree)
        all_bases = tree.bases(None)
        yield from _flake_records(
            project_root, lambda test_id: tree.resolve_nodeid(test_id, all_bases, every=True),
        )

    return Evidence(
        windows=(
            _artefact_window('archived-junit', ARCHIVED_JUNIT, archived),
            _artefact_window('live-junit', LIVE_JUNIT, live),
            _artefact_window('pytest-logs', VERIFY_LOGS, logs),
            _flake_window(project_root),
        ),
        records=records,
        uncosted_universe=frozenset(),
    )
