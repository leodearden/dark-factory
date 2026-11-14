"""Evidence readers for the suite census (task 5414): retained verify artefacts become typed records.

The I/O rim of ``suite_census_outcomes``. Each artefact format -- archived and
live junit, pytest logs, nextest status and run_all lines, the flaky ledger, the
runs.db ``flake_occurrence`` table -- is parsed once, here, into Observation /
Unresolved values. An artefact that cannot be read, or a test id naming no test
function the measured tree defines, becomes an Unresolved record, so it is
counted rather than dropped (INV-11).
"""
from __future__ import annotations

import ast
import gzip
import json
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

import _checkout as _checkout
from shared.project_layout import runs_db
from source_measures import tracked_files
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
# Every row is a failure of the test it names, whatever its call site or verdict: each
# follows a run in which that test went red (offline-lane runs included), and the failed
# floor errs high. The ledger's test-less sentinel resolves to no test, so stays Unresolved.
# Pinned by scripts/tests/test_suite_census_evidence.py::TestFlakeOccurrenceCallSites.
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
_UNPARSEABLE_SOURCE = (SyntaxError, ValueError, OSError)


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
class _Scope:
    """The functions and classes defined directly in one module or class body, and a
    class's bases: each a bare name, or None for any other expression."""

    functions: frozenset[str]
    classes: MappingProxyType[str, _Scope]
    bases: tuple[str | None, ...] = ()

    @classmethod
    def of(cls, body: Sequence[ast.stmt], bases: tuple[str | None, ...] = ()) -> _Scope:
        return cls(
            functions=frozenset(
                node.name for node in body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            classes=MappingProxyType({
                node.name: cls.of(node.body, tuple(_base_name(base) for base in node.bases))
                for node in body if isinstance(node, ast.ClassDef)
            }),
            bases=bases,
        )


def _base_name(base: ast.expr) -> str | None:
    return base.id if isinstance(base, ast.Name) else None


def _supplies(module: _Scope, scope: _Scope, function: str, seen: frozenset[str]) -> bool:
    """Whether *scope*, a class in *module*, defines or inherits *function*. A base the
    file does not define may supply any function; an in-file base cycle supplies none."""
    if function in scope.functions:
        return True
    for base in scope.bases:
        if base is None or base not in module.classes:
            return True
        if base not in seen and _supplies(module, module.classes[base], function, seen | {base}):
            return True
    return False


class _Definitions:
    """What each .py file of the measured tree defines: a memoising reader that parses a
    file on first ask. Its cache fills in place, but no answer it gives ever changes."""

    def __init__(self, tree_root: Path) -> None:
        self._tree_root = tree_root
        self._cache: dict[str, _Scope] = {}

    def defines(self, path: str, classes: Sequence[str], function: str) -> bool:
        module = scope = self._module(path)
        for name in classes:
            if name not in scope.classes:
                return False
            scope = scope.classes[name]
        return _supplies(module, scope, function, frozenset())

    def _module(self, path: str) -> _Scope:
        if path not in self._cache:
            try:
                body = ast.parse((self._tree_root / path).read_bytes(), filename=path).body
            except _UNPARSEABLE_SOURCE:
                body = []
            self._cache[path] = _Scope.of(body)
        return self._cache[path]


@dataclass(frozen=True)
class _PytestTree:
    """The measured tree's tracked .py files, its verify modules by infix, and what each file defines."""

    tracked: frozenset[str]
    module_by_infix: MappingProxyType[str, str]
    definitions: _Definitions

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
            definitions=_Definitions(tree_root),
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

    def _defines(self, path: str, classes: Sequence[str], function: str) -> bool:
        return path in self.tracked and self.definitions.defines(path, classes, function)

    def resolve_classname(self, classname: str, function: str, bases: Sequence[str]) -> TestId | None:
        """The test named by a junit classname and function name: the first base and cut whose
        file defines it."""
        parts = classname.split('.')
        for base in bases:
            for cut in range(len(parts), 0, -1):
                path, classes = _join(base, '/'.join(parts[:cut]) + '.py'), parts[cut:]
                if self._defines(path, classes, function):
                    return _test_id(path, classes, function)
        return None

    def resolve_nodeid(self, nodeid: str, bases: Sequence[str], *, every: bool) -> list[TestId]:
        """The tests a pytest node id names: the first defining base's test, or *every*
        defining base's when ambiguous."""
        parts = nodeid.split('[', 1)[0].split('::')
        matched = _FUNCTION.match(parts[-1]) if len(parts) > 1 else None
        if matched is None:
            return []
        function, classes = matched.group(), parts[1:-1]
        paths = [
            path for base in bases
            if self._defines(path := _join(base, parts[0]), classes, function)
        ]
        if not every:
            paths = paths[:1]
        return [_test_id(path, classes, function) for path in paths]


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
            self._ids[key] = self._tree.resolve_classname(
                classname, function, self._tree.bases(module),
            )
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


def _flake_store(project_root: Path) -> tuple[Path, str]:
    db = runs_db(project_root)
    return db, db.relative_to(project_root).as_posix()


def _flake_window(project_root: Path) -> SourceWindow:
    db, label = _flake_store(project_root)
    pattern = f'{label}: {FLAKE_QUERY}'
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
    db, label = _flake_store(project_root)
    if not db.is_file():
        return
    try:
        with closing(_connect_ro(db)) as conn:
            rows = conn.execute(FLAKE_QUERY).fetchall()
    except sqlite3.Error as exc:
        yield Unresolved(source='flake_occurrence', raw_id=f'{label}: {exc}')
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


# ---------------------------------------------------------------------------
# The nextest ecosystem.

FLAKY_LEDGER = 'data/verify-logs/flaky-ledger.jsonl'
INFRA_DIR = 'tests/infra'

_NEXTEST_STATUS = re.compile(
    r'^\s+(?P<status>[A-Z][A-Z-]*) \[\s*>?(?P<secs>[\d.]+)s\]\s+(?:\([^)]*\)\s+)?'
    r'(?P<binary>\S+)\s+(?P<test>\S+)\s*$'
)
_RUN_ALL_FAILED = re.compile(r'^FAILED ((?:\S+\.sh\s*)+)$')

# The one home of the nextest status vocabulary. None marks a progress line,
# which says nothing about an outcome; a status missing here is Unresolved.
_NEXTEST_OUTCOME: MappingProxyType[str, Outcome | None] = MappingProxyType({
    'PASS': Outcome.PASSED,
    'LEAK': Outcome.PASSED,
    **dict.fromkeys(
        ('FAIL', 'TIMEOUT', 'LEAK-FAIL', 'SIGSEGV', 'SIGABRT', 'SIGBUS', 'SIGKILL',
         'SIGTERM', 'ABORT'),
        Outcome.FAILED,
    ),
    **dict.fromkeys(('SLOW', 'TERMINATING', 'START', 'RETRY'), None),
})


@dataclass(frozen=True)
class _InfraTests:
    """The tracked ``tests/infra/*.sh`` scripts run_all.sh drives."""

    tracked: frozenset[str]

    @classmethod
    def of(cls, tree_root: Path) -> _InfraTests:
        scripts = tracked_files(tree_root, f'{INFRA_DIR}/*.sh')
        return cls(frozenset(path for path in scripts if path.rpartition('/')[0] == INFRA_DIR))

    def resolve(self, name: str) -> list[TestId]:
        path = f'{INFRA_DIR}/{name}'
        return [TestId(package=INFRA_DIR, name=path)] if path in self.tracked else []

    def universe(self) -> frozenset[TestId]:
        return frozenset(
            TestId(package=INFRA_DIR, name=path)
            for path in self.tracked
            if path.rpartition('/')[2].startswith('test_')
        )


def _status_record(status: re.Match[str], run: str) -> Record | None:
    if status['status'] not in _NEXTEST_OUTCOME:
        return Unresolved(source='nextest-logs', raw_id=status.string.strip())
    outcome = _NEXTEST_OUTCOME[status['status']]
    if outcome is None:
        return None
    binary = status['binary']
    return Observation(
        source='nextest-logs', run=run,
        test=TestId(package=binary.split('::', 1)[0], name=f'{binary} {status["test"]}'),
        outcome=outcome, seconds=_seconds(status['secs']),
    )


def _nextest_line_records(line: str, run: str, infra: _InfraTests) -> Iterator[Record]:
    status = _NEXTEST_STATUS.match(line)
    if status is not None:
        record = _status_record(status, run)
        if record is not None:
            yield record
        return
    run_all = _RUN_ALL_FAILED.match(line)
    for name in run_all[1].split() if run_all else ():
        yield from _failure_records('nextest-logs', run, name, infra.resolve(name))


def _nextest_log_records(artefact: _Artefact, infra: _InfraTests) -> Iterator[Record]:
    try:
        lines = artefact.path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        yield Unresolved(source='nextest-logs', raw_id=artefact.relative)
        return
    for line in lines:
        yield from _nextest_line_records(line, artefact.relative, infra)


@dataclass(frozen=True)
class _LedgerEntry:
    ts: str
    test: str
    run_id: str


def _ledger_line(line: str) -> _LedgerEntry | Unresolved:
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        entry = None
    fields = ('ts', 'test', 'run_id')
    if not isinstance(entry, dict) or not all(isinstance(entry.get(f), str) for f in fields):
        return Unresolved(source='flaky-ledger', raw_id=line)
    return _LedgerEntry(ts=entry['ts'], test=entry['test'], run_id=entry['run_id'])


def _ledger(project_root: Path) -> tuple[_LedgerEntry | Unresolved, ...]:
    path = project_root / FLAKY_LEDGER
    if not path.is_file():
        return ()
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return (Unresolved(source='flaky-ledger', raw_id=FLAKY_LEDGER),)
    return tuple(_ledger_line(line) for line in text.splitlines() if line.strip())


def _ledger_window(ledger: Sequence[_LedgerEntry | Unresolved]) -> SourceWindow:
    stamps = (
        stamp for line in ledger
        if isinstance(line, _LedgerEntry) and (stamp := _iso_utc(line.ts)) is not None
    )
    return _window('flaky-ledger', FLAKY_LEDGER, len(ledger), stamps)


def _ledger_records(
    ledger: Sequence[_LedgerEntry | Unresolved], infra: _InfraTests,
) -> Iterator[Record]:
    by_run: dict[str, list[_LedgerEntry]] = {}
    for line in ledger:
        if isinstance(line, Unresolved):
            yield line
        else:
            by_run.setdefault(line.run_id, []).append(line)
    for run_id, entries in by_run.items():
        for entry in entries:
            yield from _failure_records('flaky-ledger', run_id, entry.test, infra.resolve(entry.test))


def nextest_evidence(project_root: Path, tree_root: Path) -> Evidence:
    """Nextest status and run_all lines in the verify logs, the flaky ledger and
    flake_occurrence under *project_root*; infra scripts resolve against *tree_root*."""
    infra = _InfraTests.of(tree_root)
    logs = _archived(project_root, VERIFY_LOGS)
    ledger = _ledger(project_root)

    def records() -> Iterator[Record]:
        for artefact in logs:
            yield from _nextest_log_records(artefact, infra)
        yield from _ledger_records(ledger, infra)
        yield from _flake_records(project_root, infra.resolve)

    return Evidence(
        windows=(
            _artefact_window('nextest-logs', VERIFY_LOGS, logs),
            _ledger_window(ledger),
            _flake_window(project_root),
        ),
        records=records,
        uncosted_universe=infra.universe(),
    )
