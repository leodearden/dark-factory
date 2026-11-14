"""The quality-metrics snapshot reader: both homes, read through the REAL instrument.

Every summary and diff here runs scripts/quality_metrics_snapshot.py as a subprocess, so the
instrument proves the fixtures (dashboard/tests/_quality_snapshots.py) as well as the reader.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _quality_fixture import GitRepo
from _quality_snapshots import DEFAULT_GRAPH, DEFAULT_MEMBERS, snapshot_doc, write_snapshot
from dashboard.data.quality_snapshots import (
    BASELINE_HOME,
    INSTRUMENT_SCRIPT,
    SNAPSHOT_GIT_TIMEOUT_SECONDS,
    InstrumentRefused,
    SnapshotSummary,
    commit_time,
    diff_text,
    snapshot_files,
    summarise,
)
from shared.quality_series.git_lines import GitObjectMissing

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
