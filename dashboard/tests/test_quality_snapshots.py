"""The quality-metrics snapshot reader: both homes, read through the REAL instrument.

Every summary and diff here runs scripts/quality_metrics_snapshot.py as a subprocess, so the
instrument proves the fixtures (dashboard/tests/_quality_snapshots.py) as well as the reader.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest
from _quality_fixture import GitRepo
from _quality_snapshots import (
    DEFAULT_GRAPH,
    DEFAULT_MEMBERS,
    default_params,
    snapshot_doc,
    write_snapshot,
)
from shared.quality_series.git_lines import GitObjectMissing

from dashboard.data.datum import DatumState, validate_datum
from dashboard.data.quality_snapshots import (
    BASELINE_HOME,
    INSTRUMENT_SCRIPT,
    SNAPSHOT_GIT_TIMEOUT_SECONDS,
    DiffState,
    InstrumentRefused,
    RootScan,
    SnapshotCache,
    SnapshotSeries,
    SnapshotSummary,
    commit_time,
    diff_text,
    scan_root,
    snapshot_datum,
    snapshot_files,
    summarise,
)
from dashboard.data.quality_store import (
    read_snapshot_cache,
    read_snapshot_diff,
    read_snapshot_identities,
    read_snapshot_state,
    record_snapshot_failure,
    write_snapshot_scan,
)
from dashboard.loops import _QualityStore

SHA_1 = '1' * 40
SHA_2 = '2' * 40
SHA_3 = '3' * 40

SLUGS = [
    'src_files', 'src_lines', 'src_prose_lines', 'src_cognitive_max_total',
    'tests_files', 'tests_lines', 'tests_cognitive_total', 'files_at_soft_ceiling',
    'files_at_alarm', 'function_local_imports', 'reexport_names', 'reach_back_imports',
    'private_patch_targets', 'private_reads',
]


# ---------------------------------------------------------------------------
# The two homes.


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{}', encoding='utf-8')
    return path


class TestSnapshotFiles:
    def test_both_homes_are_listed_with_their_baseline_flag(self, tmp_path: Path) -> None:
        root, nightly = tmp_path / 'root', tmp_path / 'nightly'
        baseline = _touch(root / BASELINE_HOME / 'review-all-dark_factory-20261008.json')
        _touch(root / BASELINE_HOME / 'notes.md')
        first = _touch(nightly / 'dark_factory' / f'{SHA_1}.json')
        second = _touch(nightly / 'Dark-Factory' / f'{SHA_2}.json')
        _touch(nightly / 'dark_factory' / 'README.txt')
        _touch(nightly / 'reify' / f'{SHA_3}.json')

        files = snapshot_files(root, nightly, 'dark_factory')

        assert {(file.path, file.baseline) for file in files} == {
            (baseline, True), (first, False), (second, False),
        }

    def test_point_ids_are_stable_and_unique_across_both_homes(self, tmp_path: Path) -> None:
        root, nightly = tmp_path / 'root', tmp_path / 'nightly'
        _touch(root / BASELINE_HOME / f'{SHA_1}.json')
        _touch(nightly / 'dark_factory' / f'{SHA_1}.json')
        _touch(nightly / 'dark-factory' / f'{SHA_1}.json')

        files = snapshot_files(root, nightly, 'dark_factory')

        ids = [file.point_id for file in files]
        assert len(set(ids)) == len(ids) == 3
        assert [file.point_id for file in snapshot_files(root, nightly, 'dark_factory')] == ids

    def test_a_missing_home_contributes_nothing(self, tmp_path: Path) -> None:
        root, nightly = tmp_path / 'root', tmp_path / 'nightly'
        root.mkdir()
        assert snapshot_files(root, nightly, 'dark_factory') == []

        only = _touch(nightly / 'dark_factory' / f'{SHA_1}.json')
        assert [(file.path, file.baseline) for file in snapshot_files(root, nightly, 'dark_factory')] == [
            (only, False),
        ]


# ---------------------------------------------------------------------------
# summarise: the instrument's --summary --json, decoded at the boundary.


def _domain_figures() -> dict[str, int]:
    """The domain-level figures, worked out from the fixture's file records."""
    figures = [figure for member in DEFAULT_MEMBERS.values() for figure in member.files.values()]
    return {
        'at_soft_ceiling': sum(figure.lines >= 1500 for figure in figures),
        'at_alarm': sum(figure.lines >= 2000 for figure in figures),
        'private_patch_targets': sum(len(figure.private_patch_targets) for figure in figures),
    }


def _summary(tmp_path: Path, schema: int) -> SnapshotSummary:
    path = write_snapshot(tmp_path / f'schema-{schema}.json', snapshot_doc(schema, as_of_sha=SHA_1, run_id='r1'))
    return summarise(path)


class TestSummarise:
    @pytest.mark.parametrize('schema', [1, 2, 3])
    def test_the_header_fields(self, tmp_path: Path, schema: int) -> None:
        summary = _summary(tmp_path, schema)

        assert (summary.as_of_sha, summary.run_id, summary.complete) == (SHA_1, 'r1', True)
        assert dict(summary.params) == snapshot_doc(schema, as_of_sha=SHA_1, run_id='r1')['params']
        assert [column.slug for column in summary.columns] == SLUGS
        assert summary.columns[0].header == 'src files'

    @pytest.mark.parametrize('schema', [1, 2])
    def test_an_older_schema_names_its_unknown_params(self, tmp_path: Path, schema: int) -> None:
        assert {'instrument', 'python_version'} <= _summary(tmp_path, schema).unknown_params

    def test_a_schema_3_summary_has_no_unknown_params(self, tmp_path: Path) -> None:
        assert _summary(tmp_path, 3).unknown_params == frozenset()

    @pytest.mark.parametrize('schema', [1, 2, 3])
    def test_the_domain_extracts(self, tmp_path: Path, schema: int) -> None:
        summary = _summary(tmp_path, schema)
        expected = _domain_figures()

        assert (summary.h14.soft_ceiling_lines, summary.h14.alarm_lines) == (1500, 2000)
        assert (summary.h14.at_soft_ceiling, summary.h14.at_alarm) == (
            expected['at_soft_ceiling'], expected['at_alarm'],
        )
        assert summary.private_patch_targets == expected['private_patch_targets']
        assert summary.cycles.cycles == len(DEFAULT_GRAPH['cycles'])
        assert summary.deferred_imports == len(DEFAULT_GRAPH['deferred'])

    @pytest.mark.parametrize('schema', [2, 3])
    def test_the_cycle_sets_a_schema_records(self, tmp_path: Path, schema: int) -> None:
        summary = _summary(tmp_path, schema)

        assert (summary.cycles.hidden_cycles, summary.cycles.typing_cycles) == (
            len(DEFAULT_GRAPH['hidden_cycles']), len(DEFAULT_GRAPH['typing_cycles']),
        )
        assert summary.cycle_closing_imports == sum(
            entry['closes_cycle'] for entry in DEFAULT_GRAPH['deferred']
        )

    def test_schema_1_cycle_sets_are_unknown(self, tmp_path: Path) -> None:
        summary = _summary(tmp_path, 1)

        assert (summary.cycles.hidden_cycles, summary.cycles.typing_cycles) == (None, None)
        assert summary.cycle_closing_imports is None

    def test_member_rows_are_the_instruments_cells_in_its_order(self, tmp_path: Path) -> None:
        summary = _summary(tmp_path, 3)

        assert [(row.member, row.pseudo) for row in summary.members] == [
            ('alpha', False), ('beta', False), ('scripts', True),
        ]
        alpha = dict(zip(SLUGS, summary.members[0].cells, strict=True))
        alpha_lines = sum(
            figure.lines for figure in DEFAULT_MEMBERS['alpha'].files.values() if figure.kind == 'src'
        )
        assert (alpha['src_lines'].value, alpha['src_lines'].text) == (alpha_lines, str(alpha_lines))
        assert (alpha['src_cognitive_max_total'].value, alpha['src_cognitive_max_total'].text) == (
            (1, 2), '1 / 2',
        )
        assert alpha['private_patch_targets'].value == 2

    def test_an_invalid_snapshot_is_refused_with_the_instruments_words(self, tmp_path: Path) -> None:
        path = tmp_path / 'foreign.json'
        path.write_text('{"instrument": "other"}', encoding='utf-8')

        with pytest.raises(InstrumentRefused) as raised:
            summarise(path)

        proc = subprocess.run(
            [sys.executable, str(INSTRUMENT_SCRIPT), '--summary', str(path), '--json'],
            capture_output=True, text=True, check=False,
        )
        assert (raised.value.path, raised.value.exit_code, raised.value.stderr) == (
            path, 2, proc.stderr,
        )
        assert str(path) in raised.value.stderr


# ---------------------------------------------------------------------------
# diff_text and commit_time.


class TestDiffText:
    def test_the_instruments_diff_verbatim(self, tmp_path: Path) -> None:
        previous = write_snapshot(tmp_path / 'p.json', snapshot_doc(2, as_of_sha=SHA_1, run_id='r1'))
        current = write_snapshot(tmp_path / 'c.json', snapshot_doc(3, as_of_sha=SHA_2, run_id='r2'))

        text = diff_text(current, previous)

        proc = subprocess.run(
            [sys.executable, str(INSTRUMENT_SCRIPT), '--current', str(current), '--diff', str(previous)],
            capture_output=True, text=True, check=True,
        )
        assert text == proc.stdout
        assert text.startswith('params differ: ')

    def test_a_refused_diff_raises(self, tmp_path: Path) -> None:
        current = write_snapshot(tmp_path / 'c.json', snapshot_doc(3, as_of_sha=SHA_2, run_id='r2'))
        missing = tmp_path / 'absent.json'

        with pytest.raises(InstrumentRefused) as raised:
            diff_text(current, missing)

        assert raised.value.exit_code == 2
        assert str(missing) in raised.value.stderr


class TestCommitTime:
    def test_the_committer_time(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GIT_COMMITTER_DATE', '2026-10-08T12:30:00+02:00')
        repo = GitRepo(tmp_path / 'repo')
        sha = repo.init()

        assert commit_time(repo.root, sha, SNAPSHOT_GIT_TIMEOUT_SECONDS) == datetime(
            2026, 10, 8, 10, 30, tzinfo=UTC
        )

    def test_an_unknown_sha_is_missing(self, tmp_path: Path) -> None:
        repo = GitRepo(tmp_path / 'repo')
        repo.init()

        with pytest.raises(GitObjectMissing):
            commit_time(repo.root, 'f' * 40, SNAPSHOT_GIT_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# scan_root: one root's scan on a worker thread, cached by file state, digest and identity pair.

_COUNTING_INSTRUMENT = """\
import pathlib
import subprocess
import sys

with open(pathlib.Path(__file__).with_name('calls.log'), 'a', encoding='utf-8') as log:
    log.write(sys.argv[1] + '\\n')
sys.exit(subprocess.call([sys.executable, {real!r}, *sys.argv[1:]]))
"""


@dataclass(frozen=True)
class Scene:
    """A git root with three commits; a schema-2 baseline at sha1 and schema-3 nightlies at sha2, sha3."""

    root: Path
    nightly: Path
    shas: tuple[str, str, str]
    times: tuple[datetime, datetime, datetime]
    instrument: Path

    @property
    def home(self) -> Path:
        return self.nightly / 'dark_factory'

    def calls(self) -> list[str]:
        """The instrument modes run since the last call (--summary or --current)."""
        log = self.instrument.with_name('calls.log')
        if not log.exists():
            return []
        modes = log.read_text(encoding='utf-8').splitlines()
        log.unlink()
        return modes

    def scan(self, cache: SnapshotCache | None = None, *, deadline: float | None = None) -> RootScan:
        return scan_root(
            self.root, 'dark_factory', self.nightly, cache or SnapshotCache.empty(),
            deadline=time.monotonic() + 120 if deadline is None else deadline,
            instrument=self.instrument,
        )


@pytest.fixture
def scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scene:
    repo = GitRepo(tmp_path / 'root')
    times = (
        datetime(2026, 10, 8, 12, tzinfo=UTC),
        datetime(2026, 11, 1, 3, tzinfo=UTC),
        datetime(2026, 11, 2, 3, tzinfo=UTC),
    )
    monkeypatch.setenv('GIT_COMMITTER_DATE', times[0].isoformat())
    sha1 = repo.init()
    monkeypatch.setenv('GIT_COMMITTER_DATE', times[1].isoformat())
    sha2 = repo.commit({'a.txt': 'two\n'}, 'two')
    monkeypatch.setenv('GIT_COMMITTER_DATE', times[2].isoformat())
    sha3 = repo.commit({'a.txt': 'three\n'}, 'three')
    monkeypatch.delenv('GIT_COMMITTER_DATE')
    write_snapshot(repo.root / BASELINE_HOME / 'review-all-dark_factory-20261008.json',
                   snapshot_doc(2, as_of_sha=sha1, run_id='r1'))
    nightly = tmp_path / 'nightly'
    for sha, run_id in ((sha2, 'r2'), (sha3, 'r3')):
        write_snapshot(nightly / 'dark_factory' / f'{sha}.json', snapshot_doc(3, as_of_sha=sha, run_id=run_id))
    instrument_dir = tmp_path / 'instrument'
    instrument_dir.mkdir()
    instrument = instrument_dir / 'quality_metrics_snapshot.py'
    instrument.write_text(_COUNTING_INSTRUMENT.format(real=str(INSTRUMENT_SCRIPT)), encoding='utf-8')
    (instrument_dir / 'source_measures.py').symlink_to(INSTRUMENT_SCRIPT.with_name('source_measures.py'))
    return Scene(repo.root, nightly, (sha1, sha2, sha3), times, instrument)


def _rewrite(path: Path, doc: dict[str, object]) -> None:
    before = path.stat().st_mtime_ns
    write_snapshot(path, doc)
    os.utime(path, ns=(before + 1_000_000_000, before + 1_000_000_000))


class TestScanRoot:
    def test_a_cold_scan_places_every_point_with_its_diff(self, scene: Scene) -> None:
        scan = scene.scan()

        assert [point.summary.as_of_sha for point in scan.points] == list(scene.shas)
        assert [point.baseline for point in scan.points] == [True, False, False]
        assert [point.committed_at for point in scan.points] == list(scene.times)
        assert [point.diff.state for point in scan.points] == [DiffState.FIRST, DiffState.READY, DiffState.READY]
        for previous, point in zip(scan.points, scan.points[1:], strict=False):
            assert point.diff.previous_id == previous.point_id
            assert point.diff.pair_key is not None
            assert scan.new_diffs[point.diff.pair_key].text == diff_text(point.file, previous.file)
        assert (scan.unplaced, scan.pending_summaries, scan.pending_diffs) == ((), 0, 0)
        assert sorted(scene.calls()) == ['--current', '--current', '--summary', '--summary', '--summary']

    def test_a_warm_scan_recomputes_nothing(self, scene: Scene) -> None:
        first = scene.scan()
        scene.calls()

        second = scene.scan(SnapshotCache.from_scan(first))

        assert scene.calls() == []
        assert second.new_diffs == {}
        assert second.points == first.points

    def test_new_content_resummarises_that_file_and_rediffs_only_a_changed_identity_pair(
        self, scene: Scene
    ) -> None:
        first = scene.scan()
        scene.calls()
        sha1, sha2, sha3 = scene.shas

        _rewrite(scene.home / f'{sha2}.json', snapshot_doc(3, as_of_sha=sha2, run_id='r2-again'))
        same_identity = scene.scan(SnapshotCache.from_scan(first))

        assert scene.calls() == ['--summary']
        assert same_identity.new_diffs == {}
        assert same_identity.points[1].summary.run_id == 'r2-again'

        changed = {**default_params(3), 'complexipy_version': '6.3.0'}
        _rewrite(scene.home / f'{sha3}.json', snapshot_doc(3, as_of_sha=sha3, run_id='r3', params=changed))
        new_identity = scene.scan(SnapshotCache.from_scan(same_identity))

        assert scene.calls() == ['--summary', '--current']
        assert list(new_identity.new_diffs) == [new_identity.points[2].diff.pair_key]
        assert new_identity.points[2].diff.pair_key != first.points[2].diff.pair_key

    def test_a_removed_file_drops_out(self, scene: Scene) -> None:
        first = scene.scan()
        (scene.home / f'{scene.shas[2]}.json').unlink()

        second = scene.scan(SnapshotCache.from_scan(first))

        assert [point.point_id for point in second.points] == [point.point_id for point in first.points[:2]]

    def test_unplaceable_files_are_listed_with_their_reason(self, scene: Scene) -> None:
        refused = scene.home / 'foreign.json'
        refused.write_text('{"instrument": "other"}', encoding='utf-8')
        unknown_sha = 'f' * 40
        write_snapshot(scene.home / f'{unknown_sha}.json', snapshot_doc(3, as_of_sha=unknown_sha, run_id='r9'))

        scan = scene.scan()

        reasons = {reading.file.name: reading.reason for reading in scan.unplaced}
        assert set(reasons) == {'foreign.json', f'{unknown_sha}.json'}
        with pytest.raises(InstrumentRefused) as raised:
            summarise(refused)
        assert reasons['foreign.json'] == raised.value.stderr
        assert unknown_sha in reasons[f'{unknown_sha}.json']
        assert len(scan.points) == 3

    def test_a_passed_deadline_computes_nothing(self, scene: Scene) -> None:
        scan = scene.scan(deadline=time.monotonic() - 1)

        assert (scan.points, scan.new_diffs, scan.pending_summaries) == ((), {}, 3)
        assert scene.calls() == []

    def test_a_byte_identical_copy_reuses_the_summary_and_the_pair_key(self, scene: Scene) -> None:
        first = scene.scan()
        scene.calls()
        nightly = scene.home / f'{scene.shas[2]}.json'
        copy = scene.root / BASELINE_HOME / nightly.name
        copy.write_bytes(nightly.read_bytes())

        second = scene.scan(SnapshotCache.from_scan(first))

        assert scene.calls() == ['--current']
        assert [(point.file, point.baseline) for point in second.points[2:]] == [(copy, True), (nightly, False)]
        assert second.points[2].diff.pair_key == first.points[2].diff.pair_key
        assert second.points[2].diff.state is DiffState.READY
        assert list(second.new_diffs) == [second.points[3].diff.pair_key]


# ---------------------------------------------------------------------------
# The snapshot store and the datum: a real quality.db, opened with its schema applied.

SCANNED_AT = datetime(2026, 11, 14, 6, 0, tzinfo=UTC)
BOUND = 1200


@pytest.fixture
async def conn(tmp_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    async with _QualityStore(tmp_path / 'quality.db') as store:
        yield store.connection


async def _stored(conn: aiosqlite.Connection, scan: RootScan, at: datetime = SCANNED_AT) -> None:
    await write_snapshot_scan(conn, 'dark_factory', scan, at)
    await conn.commit()


async def _datum(conn: aiosqlite.Connection, now: datetime = SCANNED_AT):
    stored = await read_snapshot_state(conn, 'dark_factory')
    return snapshot_datum('dark_factory', stored, now=now, freshness_bound_seconds=BOUND)


class TestTheSnapshotDatum:
    async def test_nothing_stored_is_unknown_until_scanned(self, conn: aiosqlite.Connection) -> None:
        datum = await _datum(conn)

        assert datum.state is DatumState.UNKNOWN
        assert datum.reason is not None and 'not been scanned yet' in datum.reason
        assert 'dark_factory' in datum.reason

    async def test_a_scan_finding_no_file_says_no_snapshot_yet_naming_both_homes(
        self, conn: aiosqlite.Connection, tmp_path: Path
    ) -> None:
        root, nightly = tmp_path / 'empty-root', tmp_path / 'nightly'
        root.mkdir()
        scan = scan_root(root, 'dark_factory', nightly, SnapshotCache.empty(), deadline=time.monotonic() + 60)
        await _stored(conn, scan)

        datum = await _datum(conn)

        assert datum.state is DatumState.UNKNOWN
        assert datum.reason is not None and datum.reason.startswith('no snapshot yet')
        assert f'{root / BASELINE_HOME}/' in datum.reason
        assert f'{nightly / "dark_factory"}/' in datum.reason

    async def test_a_failed_scan_with_no_points_names_the_failure(self, conn: aiosqlite.Connection) -> None:
        await record_snapshot_failure(conn, 'dark_factory', SCANNED_AT, 'snapshot scan exceeded the 120.0s per-root budget')
        await conn.commit()

        datum = await _datum(conn)

        assert datum.state is DatumState.UNKNOWN
        assert datum.reason is not None and 'snapshot scan exceeded the 120.0s per-root budget' in datum.reason

    async def test_a_newer_failed_attempt_serves_the_points_stale(
        self, conn: aiosqlite.Connection, scene: Scene
    ) -> None:
        await _stored(conn, scene.scan())
        attempted = SCANNED_AT + timedelta(minutes=10)
        await record_snapshot_failure(conn, 'dark_factory', attempted, 'OSError: boom')
        await conn.commit()

        datum = await _datum(conn, attempted)

        assert datum.state is DatumState.STALE
        assert datum.reason is not None and 'OSError: boom' in datum.reason
        assert attempted.isoformat() in datum.reason
        assert datum.value is not None and len(datum.value.points) == 3

    async def test_pending_summaries_are_stale(self, conn: aiosqlite.Connection, scene: Scene) -> None:
        await _stored(conn, scene.scan(deadline=time.monotonic() - 1))

        datum = await _datum(conn)

        assert datum.state is DatumState.STALE
        assert datum.reason == '3 snapshot files not yet summarised'

    async def test_a_complete_scan_is_fresh_then_ages(self, conn: aiosqlite.Connection, scene: Scene) -> None:
        await _stored(conn, scene.scan())

        fresh = await _datum(conn, SCANNED_AT + timedelta(seconds=60))
        aged = await _datum(conn, SCANNED_AT + timedelta(seconds=BOUND + 1))

        assert (fresh.state, fresh.as_of, fresh.reason) == (DatumState.FRESH, SCANNED_AT, None)
        assert aged.state is DatumState.STALE

    async def test_the_wire_shape(self, conn: aiosqlite.Connection, scene: Scene) -> None:
        scan = scene.scan()
        await _stored(conn, scan)

        datum = await _datum(conn)
        validate_datum(datum, SCANNED_AT)
        wire = datum.to_wire()['value']

        assert isinstance(wire, dict) and set(wire) == {'points', 'unplaced', 'pending'}
        assert wire['unplaced'] == [] and wire['pending'] == {'summaries': 0, 'diffs': 0}
        first, second = wire['points'][:2]
        assert set(first) == {
            'id', 'file', 'baseline', 'as_of_sha', 'committed_at', 'run_id', 'params', 'unknown_params',
            'complete', 'members', 'h14', 'cycles', 'deferred_imports', 'cycle_closing_imports',
            'private_patch_targets', 'diff',
        }
        summary = scan.points[0].summary
        assert first['members'] == {
            'columns': [column.header for column in summary.columns],
            'rows': [
                {'member': row.member, 'pseudo': row.pseudo, 'cells': [cell.text for cell in row.cells]}
                for row in summary.members
            ],
        }
        assert first['unknown_params'] == sorted(summary.unknown_params)
        assert first['h14'] == {
            'soft_ceiling_lines': 1500, 'alarm_lines': 2000,
            'at_soft_ceiling': summary.h14.at_soft_ceiling, 'at_alarm': summary.h14.at_alarm,
        }
        assert first['cycles'] == {'cycles': 2, 'hidden_cycles': 1, 'typing_cycles': 1}
        assert first['diff'] == {'previous_id': None, 'state': 'first', 'reason': None}
        assert second['diff'] == {'previous_id': first['id'], 'state': 'ready', 'reason': None}
        assert first['committed_at'] == scene.times[0].isoformat()
        assert first['baseline'] is True and second['baseline'] is False

    def test_unordered_points_are_refused_naming_the_ids(self, scene: Scene) -> None:
        points = scene.scan().points

        with pytest.raises(ValueError, match='order') as raised:
            SnapshotSeries(points=(points[1], points[0], points[2]), unplaced=(), pending_summaries=0, pending_diffs=0)

        assert points[0].point_id in str(raised.value) and points[1].point_id in str(raised.value)


class TestTheSnapshotStore:
    async def test_a_second_write_replaces_a_vanished_point(self, conn: aiosqlite.Connection, scene: Scene) -> None:
        first = scene.scan()
        await _stored(conn, first)
        (scene.home / f'{scene.shas[2]}.json').unlink()
        second = scene.scan(await read_snapshot_cache(conn, 'dark_factory'))
        await _stored(conn, second, SCANNED_AT + timedelta(minutes=10))

        cache = await read_snapshot_cache(conn, 'dark_factory')

        assert cache == SnapshotCache.from_scan(second)
        assert set(cache.files) == {point.point_id for point in first.points[:2]}

    async def test_the_cache_round_trips(self, conn: aiosqlite.Connection, scene: Scene) -> None:
        scan = scene.scan()
        await _stored(conn, scan)

        assert await read_snapshot_cache(conn, 'dark_factory') == SnapshotCache.from_scan(scan)

    async def test_unplaced_files_round_trip(self, conn: aiosqlite.Connection, scene: Scene) -> None:
        (scene.home / 'foreign.json').write_text('{"instrument": "other"}', encoding='utf-8')
        scan = scene.scan()
        await _stored(conn, scan)

        stored = await read_snapshot_state(conn, 'dark_factory')

        assert stored is not None and stored.scan is not None
        assert stored.scan.series.unplaced == scan.unplaced
        assert (await read_snapshot_cache(conn, 'dark_factory')).files.keys() >= {scan.unplaced[0].point_id}

    async def test_the_diff_of_a_point(self, conn: aiosqlite.Connection, scene: Scene) -> None:
        scan = scene.scan()
        await _stored(conn, scan)
        first, second = scan.points[:2]

        ready = await read_snapshot_diff(conn, 'dark_factory', second.point_id)
        opening = await read_snapshot_diff(conn, 'dark_factory', first.point_id)

        assert ready is not None and (ready.state, ready.previous_id) == (DiffState.READY, first.point_id)
        assert ready.text == diff_text(second.file, first.file)
        assert opening is not None and (opening.state, opening.text) == (DiffState.FIRST, None)
        assert await read_snapshot_diff(conn, 'dark_factory', 'no-such-point') is None

    async def test_the_identities_follow_the_points(self, conn: aiosqlite.Connection, scene: Scene) -> None:
        scan = scene.scan()
        await _stored(conn, scan)

        identities = await read_snapshot_identities(conn, 'dark_factory')

        assert identities == [point.identity() for point in scan.points]
        assert identities[0].unknown_params >= {'instrument', 'python_version'}
        assert identities[1].unknown_params == frozenset()

    async def test_a_pending_diff_reads_pending(self, conn: aiosqlite.Connection, scene: Scene) -> None:
        cold = scene.scan()
        without_diffs = replace(cold, new_diffs={})
        await _stored(conn, without_diffs)

        stored = await read_snapshot_state(conn, 'dark_factory')

        assert stored is not None and stored.scan is not None
        assert [point.diff.state for point in stored.scan.series.points] == [
            DiffState.FIRST, DiffState.PENDING, DiffState.PENDING,
        ]
