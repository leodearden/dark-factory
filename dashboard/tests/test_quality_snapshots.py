"""The quality-metrics snapshot reader: both homes, read through the REAL instrument.

Every summary and diff here runs scripts/quality_metrics_snapshot.py as a subprocess, so the
instrument proves the fixtures (dashboard/tests/_quality_snapshots.py) as well as the reader.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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

from dashboard.data.quality_snapshots import (
    BASELINE_HOME,
    INSTRUMENT_SCRIPT,
    SNAPSHOT_GIT_TIMEOUT_SECONDS,
    DiffState,
    InstrumentRefused,
    RootScan,
    SnapshotCache,
    SnapshotSummary,
    commit_time,
    diff_text,
    scan_root,
    snapshot_files,
    summarise,
)

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
