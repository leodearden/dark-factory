"""Tests of ``scripts/quality_metrics_snapshot.py --summary SNAPSHOT --json``: the per-member table as typed data.

The JSON is the markdown ``--summary`` rendered a second way, so every
assertion here compares the two renderings of one REAL measured fixture repo
(quality_snapshot_fixtures.py) cell for cell: the dashboard reads the figures
exactly as the instrument's ``--summary`` defines them, never re-derived.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import quality_metrics_snapshot as snapshot
from quality_snapshot_fixtures import (
    MOD_BEFORE,
    TRIVIAL_TEST,
    as_schema_1,
    as_schema_2,
    committed_repo,
    read_json,
    run_snapshot,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_FILES: dict[str, str] = {
    'pyproject.toml': '[tool.uv.workspace]\nmembers = ["beta", "alpha"]\n',
    'alpha/src/alpha/__init__.py': 'THING = 1\n',
    'alpha/src/alpha/mod.py': MOD_BEFORE,
    'alpha/src/alpha/big.py': ''.join(f'V{index} = {index}\n' for index in range(2010)),
    'alpha/src/alpha/late.py': 'def g():\n    import alpha.mod\n    return alpha.mod\n',
    'alpha/src/alpha/rb.py': 'from alpha import THING\nY = THING\n',
    'alpha/tests/test_mod.py': (
        'from unittest.mock import patch\nfrom alpha import mod\n\ndef test_it():\n'
        "    with patch('alpha.mod._x'):\n        assert mod._y\n"
    ),
    'beta/src/beta/b.py': 'B = 1\n',
    'beta/tests/test_b.py': TRIVIAL_TEST,
}

_SLUGS = [
    'src_files', 'src_lines', 'src_prose_lines', 'src_cognitive_max_total',
    'tests_files', 'tests_lines', 'tests_cognitive_total', 'files_at_soft_ceiling',
    'files_at_alarm', 'function_local_imports', 'reexport_names', 'reach_back_imports',
    'private_patch_targets', 'private_reads',
]

#: The markdown graph line's label for each import_graph slug.
_GRAPH_LABELS = {
    'edges': 'edges',
    'reach-backs': 'reach_backs',
    'deferred imports': 'deferred_imports',
    'cycle-closing deferred imports': 'cycle_closing_imports',
    'cycles': 'cycles',
    'hidden cycles': 'hidden_cycles',
    'typing cycles': 'typing_cycles',
}


def _markdown(path: Path, capsys: pytest.CaptureFixture[str]) -> list[str]:
    capsys.readouterr()
    assert snapshot.main(['--summary', str(path)]) == 0
    return capsys.readouterr().out.splitlines()


def _summary_json(path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    capsys.readouterr()
    assert snapshot.main(['--summary', str(path), '--json']) == 0
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 1, out
    return json.loads(out)


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip('|').split('|')]


def _table(markdown: list[str]) -> tuple[list[str], list[list[str]]]:
    """The header cells and the data rows' cells of the markdown table."""
    start = next(index for index, line in enumerate(markdown) if line.startswith('| member |'))
    rows = []
    for line in markdown[start + 2:]:
        if not line.startswith('|'):
            break
        rows.append(_cells(line))
    return _cells(markdown[start]), rows


def _graph_counts(markdown: list[str]) -> dict[str, int | None]:
    """The markdown graph line's counts by slug; an unknown kind is None."""
    line = next(line for line in markdown if line.startswith('import graph: '))
    counted, _, unknown = line.removeprefix('import graph: ').partition('; ')
    counts: dict[str, int | None] = {}
    for part in counted.split(', '):
        number, label = part.split(' ', 1)
        counts[_GRAPH_LABELS[label]] = int(number)
    if unknown:
        named = re.sub(r' unknown \(schema \d+\)$', '', unknown)
        counts.update({_GRAPH_LABELS[label]: None for label in named.split(', ')})
    return counts


@pytest.fixture
def measured(tmp_path: Path) -> Path:
    path = tmp_path / 'out' / 'snapshot.json'
    assert run_snapshot(committed_repo(tmp_path, _FILES), path) == 0
    return path


def _older(measured: Path, tmp_path: Path, schema: int) -> Path:
    convert = {1: as_schema_1, 2: as_schema_2}[schema]
    path = tmp_path / f'schema-{schema}.json'
    path.write_text(json.dumps(convert(read_json(measured))), encoding='utf-8')
    return path


class TestTheSummaryJson:
    def test_one_object_with_the_snapshot_header_fields(
        self, measured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        taken = read_json(measured)
        summary = _summary_json(measured, capsys)
        assert set(summary) == {
            'run_id', 'as_of_sha', 'since', 'schema_version', 'params', 'unknown_params',
            'complete', 'measured_files', 'domain_files', 'unreadable', 'columns', 'members',
            'domain', 'import_graph',
        }
        evidence = taken['evidence']
        assert (summary['run_id'], summary['as_of_sha'], summary['since']) == (
            'run-1', taken['as_of_sha'], 'none'
        )
        assert summary['schema_version'] == snapshot.SCHEMA_VERSION
        assert summary['params'] == taken['params']
        assert (summary['complete'], summary['measured_files'], summary['domain_files']) == (
            evidence['complete'], evidence['measured_files'], evidence['domain_files']
        )
        assert summary['unreadable'] == evidence['unreadable']

    def test_columns_are_the_markdown_headers_with_their_slugs(
        self, measured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        summary = _summary_json(measured, capsys)
        headers, _rows = _table(_markdown(measured, capsys))
        assert [column['header'] for column in summary['columns']] == headers[2:]
        assert [column['slug'] for column in summary['columns']] == _SLUGS
        assert all(set(column) == {'slug', 'header'} for column in summary['columns'])

    def test_member_cells_are_the_markdown_rows_typed(
        self, measured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        summary = _summary_json(measured, capsys)
        _headers, rows = _table(_markdown(measured, capsys))
        members = summary['members']
        assert [member['member'] for member in members] == sorted(member['member'] for member in members)
        assert [
            [member['member'], json.dumps(member['pseudo']), *(cell['text'] for cell in member['cells'])]
            for member in members
        ] == rows[:-1]
        for member in members:
            for slug, cell in zip(_SLUGS, member['cells'], strict=True):
                assert set(cell) == {'value', 'text'}
                if slug == 'src_cognitive_max_total':
                    highest, total = cell['value']
                    assert cell['text'] == f'{highest} / {total}'
                else:
                    assert type(cell['value']) is int, (member['member'], slug)
                    assert cell['text'] == str(cell['value'])

    def test_the_fixture_reaches_every_kind_of_figure(
        self, measured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        alpha = next(member for member in _summary_json(measured, capsys)['members'] if member['member'] == 'alpha')
        values = dict(zip(_SLUGS, (cell['value'] for cell in alpha['cells']), strict=True))
        assert values['files_at_alarm'] == 1
        assert values['private_patch_targets'] == 1
        assert values['function_local_imports'] == 1
        assert values['reach_back_imports'] == 1

    def test_domain_cells_are_the_markdown_domain_row(
        self, measured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        summary = _summary_json(measured, capsys)
        _headers, rows = _table(_markdown(measured, capsys))
        assert rows[-1][0] == '(domain)'
        assert [cell['text'] for cell in summary['domain']['cells']] == rows[-1][2:]

    def test_the_import_graph_counts_are_the_markdown_graph_line(
        self, measured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        summary = _summary_json(measured, capsys)
        assert summary['import_graph'] == _graph_counts(_markdown(measured, capsys))
        assert set(summary['import_graph']) == set(_GRAPH_LABELS.values())

    def test_a_schema_3_snapshot_has_no_unknown_params(
        self, measured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _summary_json(measured, capsys)['unknown_params'] == []

    def test_a_schema_1_snapshot_names_its_unknowns(
        self, measured: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        old = _older(measured, tmp_path, 1)
        summary = _summary_json(old, capsys)
        graph = summary['import_graph']
        assert (graph['cycle_closing_imports'], graph['hidden_cycles'], graph['typing_cycles']) == (
            None, None, None
        )
        assert graph == _graph_counts(_markdown(old, capsys))
        assert {'instrument', 'python_version'} <= set(summary['unknown_params'])
        assert summary['unknown_params'] == sorted(summary['unknown_params'])

    def test_a_schema_2_snapshot_names_its_unknown_params(
        self, measured: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        summary = _summary_json(_older(measured, tmp_path, 2), capsys)
        assert {'instrument', 'python_version'} <= set(summary['unknown_params'])
        assert summary['import_graph']['typing_cycles'] is not None

    def test_the_markdown_summary_is_unchanged_by_the_flag(
        self, measured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        markdown = _markdown(measured, capsys)
        _summary_json(measured, capsys)
        assert _markdown(measured, capsys) == markdown
        assert markdown[0].startswith('run: run-1 as_of ')
        assert any(line.startswith('| member | pseudo |') for line in markdown)


class TestTheJsonFlagNeedsTheSummary:
    @pytest.mark.parametrize(
        'argv',
        [
            pytest.param(['--json'], id='alone'),
            pytest.param(['--current', 'a.json', '--diff', 'b.json', '--json'], id='with-current'),
            pytest.param(['--run-id', 'r', '--out', 'o.json', '--diff', 'b.json', '--json'], id='with-diff'),
            pytest.param(['--params', '--json'], id='with-params'),
        ],
    )
    def test_json_without_summary_is_a_usage_error(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as raised:
            snapshot.main(argv)
        assert raised.value.code == 2

    def test_an_invalid_snapshot_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / 'given.json'
        path.write_text('{"instrument": "other"}', encoding='utf-8')
        assert snapshot.main(['--summary', str(path), '--json']) == 2
        captured = capsys.readouterr()
        assert str(path) in captured.err
        assert captured.out == ''
