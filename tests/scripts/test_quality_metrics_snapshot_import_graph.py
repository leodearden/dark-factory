"""Tests of the import graph ``scripts/quality_metrics_snapshot.py`` measures.

Its sections and per-file graph fields, the wider cycle sets and
``closes_cycle``, its ``--diff`` 'import graph:' lines, and reading the
schema-1 snapshots recorded before the cycle sets. As in
test_quality_metrics_snapshot.py, every fixture is a small REAL git repository
built and measured through quality_snapshot_fixtures.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import quality_metrics_snapshot as snapshot
import source_measures
from quality_snapshot_fixtures import (
    BASE_FILES,
    GRAPH_FILES,
    SnapshotEdit,
    as_schema_1,
    commit,
    committed_repo,
    diff_chain,
    diff_of,
    diff_section,
    measured_snapshot,
    read_json,
    rewriting,
    run_snapshot,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


# ---------------------------------------------------------------------------
# The import graph: nodes are module import names.

_RELATIVE: dict[str, str] = {
    'alpha/src/rel/__init__.py': 'NAME = 1\n',
    'alpha/src/rel/other.py': 'y = 1\n',
    'alpha/src/rel/sub/__init__.py': 'from .. import NAME\n',
    'alpha/src/rel/sub/n.py': 'thing = 1\n',
    'alpha/src/rel/sub/m.py': (
        'import rel\nfrom . import n\nfrom .n import thing\nfrom ..other import y\n'
    ),
    'beta/src/beta/b.py': 'B = 1\n',
    'scripts/z.py': 'from . import q\n',
}

#: Six function-local imports, each in a different place inside a function; one class-level import.
_DEFERRED_SHAPES = (
    'from typing import TYPE_CHECKING\n'
    'def outer():\n'
    '    def inner():\n'
    '        import os\n'
    '    class Local:\n'
    '        from alpha import mod\n'
    '    try:\n'
    '        import json\n'
    '    except ImportError:\n'
    '        from alpha import mod as m2\n'
    '    if TYPE_CHECKING:\n'
    '        import typing\n'
    'class C:\n'
    '    import sys\n'
    '    def m(self):\n'
    '        import re\n'
)


class TestTheImportGraph:
    @pytest.fixture
    def measured(self, tmp_path: Path) -> dict[str, Any]:
        return measured_snapshot(tmp_path, GRAPH_FILES)

    def test_it_has_exactly_the_six_sections(self, measured: dict[str, Any]) -> None:
        assert list(measured['import_graph']) == [
            'edges', 'reach_back', 'deferred', 'cycles', 'hidden_cycles', 'typing_cycles',
        ]

    def test_edges_are_explicit_first_party_src_imports(self, measured: dict[str, Any]) -> None:
        # Module-level, TYPE_CHECKING and function-local imports all count; no
        # implicit parent package, no stdlib, nothing from a tests file.
        assert measured['import_graph']['edges'] == [
            ['a', 'b'],
            ['b', 'a'],
            ['beta.b', 'pkg.c'],
            ['c', 'd'],
            ['d', 'c'],
            ['e', 'a'],
            ['pkg.a', 'pkg'],
            ['pkg.b', 'pkg.c'],
            ['x', 'legibility.y'],
        ]

    def test_cycles_are_over_runtime_module_level_edges_only(
        self, measured: dict[str, Any]
    ) -> None:
        # c <-> d exists only under TYPE_CHECKING, and e -> a is function-local.
        assert measured['import_graph']['cycles'] == [['a', 'b']]

    def test_the_wider_cycle_sets_add_function_local_then_type_checking_edges(
        self, measured: dict[str, Any]
    ) -> None:
        # e -> a is function-local, but a never reaches e; c <-> d is TYPE_CHECKING only.
        graph = measured['import_graph']
        assert graph['hidden_cycles'] == [['a', 'b']]
        assert graph['typing_cycles'] == [['a', 'b'], ['c', 'd']]

    def test_a_name_from_the_package_init_is_a_reach_back_and_a_submodule_is_not(
        self, measured: dict[str, Any]
    ) -> None:
        assert measured['import_graph']['reach_back'] == [
            {'from': 'pkg.a', 'to': 'pkg', 'names': ['THING'], 'line': 1},
        ]

    def test_deferred_imports_are_listed_by_site(self, measured: dict[str, Any]) -> None:
        deferred = measured['import_graph']['deferred']
        assert deferred == [{'from': 'e', 'line': 5, 'imports': ['a'], 'closes_cycle': False}]
        assert (
            len([entry for entry in deferred if entry['from'] == 'e'])
            == measured['files']['alpha/src/e.py']['function_local_imports']
        )

    def test_each_src_records_function_local_imports_are_its_deferred_entries(
        self, tmp_path: Path
    ) -> None:
        measured = measured_snapshot(tmp_path, {**BASE_FILES, 'alpha/src/alpha/late.py': _DEFERRED_SHAPES})
        deferred_from = [entry['from'] for entry in measured['import_graph']['deferred']]
        src_records = [r for r in measured['files'].values() if r['kind'] == 'src']
        for record in src_records:
            assert record['function_local_imports'] == deferred_from.count(record['module'])
        assert measured['files']['alpha/src/alpha/late.py']['function_local_imports'] == 6

    def test_the_per_file_graph_fields(self, measured: dict[str, Any]) -> None:
        files = measured['files']
        assert files['alpha/src/pkg/a.py']['reach_back_imports'] == 1
        assert files['alpha/src/pkg/b.py']['reach_back_imports'] == 0
        assert files['alpha/src/pkg/b.py']['fan_out'] == 1
        assert files['alpha/src/e.py']['fan_out'] == 1
        assert files['alpha/src/pkg/c.py']['fan_in_src'] == 2
        assert files['alpha/src/a.py']['fan_in_src'] == 2
        assert files['alpha/src/pkg/c.py']['fan_in_tests'] == 1
        assert files['alpha/src/pkg/a.py']['fan_in_tests'] == 1
        assert files['alpha/src/b.py']['fan_in_tests'] == 0

    def test_a_src_record_has_exactly_the_contract_fields(
        self, measured: dict[str, Any]
    ) -> None:
        assert set(measured['files']['alpha/src/pkg/a.py']) == {
            'member', 'kind', 'blob', 'lines', 'prose_lines', 'prose_ratio',
            'cognitive_total', 'cognitive_max', 'cognitive_max_function', 'functions',
            'module', 'package_init', 'function_local_imports', 'reexport_names',
            'reach_back_imports', 'cycle_closing_imports', 'fan_out', 'fan_in_src', 'fan_in_tests',
        }

    def test_relative_imports_resolve(self, tmp_path: Path) -> None:
        graph = measured_snapshot(tmp_path, _RELATIVE)['import_graph']
        edges = graph['edges']
        for edge in (
            ['rel.sub', 'rel'],
            ['rel.sub.m', 'rel'],
            ['rel.sub.m', 'rel.other'],
            ['rel.sub.m', 'rel.sub.n'],
        ):
            assert edges.count(edge) == 1, edge
        # An __init__ importing a name from an ancestor package is a reach-back;
        # a bare `import rel` and a lateral `from ..other import y` are not, and
        # scripts/z.py's unresolvable relative import is neither edge nor error.
        assert graph['reach_back'] == [
            {'from': 'rel.sub', 'to': 'rel', 'names': ['NAME'], 'line': 1},
        ]
        assert not [edge for edge in edges if edge[0] == 'z']

    def test_an_unreadable_module_is_still_a_node(self, tmp_path: Path) -> None:
        measured = measured_snapshot(
            tmp_path,
            {**BASE_FILES, 'alpha/src/broken.py': 'def (:\n', 'alpha/src/user.py': 'import broken\n'},
        )
        assert ['user', 'broken'] in measured['import_graph']['edges']
        assert 'alpha/src/broken.py' in measured['evidence']['unreadable']
        assert 'alpha/src/broken.py' not in measured['files']
        assert measured['files']['alpha/src/user.py']['fan_out'] == 1


#: Cycles that only function-local (j <-> m, f <-> g) or TYPE_CHECKING (t <-> v) imports close.
_CYCLE_SETS: dict[str, str] = {
    'alpha/src/f.py': 'import g\n',
    'alpha/src/g.py': 'def h():\n    import w, f\n    return f\n',
    'alpha/src/w.py': 'W = 1\n',
    'alpha/src/j.py': 'def k():\n    import m\n    return m\n',
    'alpha/src/m.py': 'def n():\n    import j\n    return j\n',
    'alpha/src/t.py': (
        'from typing import TYPE_CHECKING\n\n\ndef u():\n    if TYPE_CHECKING:\n        import v\n'
    ),
    'alpha/src/v.py': 'import t\n',
    'alpha/src/e.py': 'def late():\n    import f\n    return f\n',
    'beta/src/beta/b.py': 'B = 1\n',
}


class TestTheCycleSets:
    @pytest.fixture
    def measured(self, tmp_path: Path) -> dict[str, Any]:
        return measured_snapshot(tmp_path, _CYCLE_SETS)

    def test_each_set_adds_the_edges_of_a_later_moment(self, measured: dict[str, Any]) -> None:
        graph = measured['import_graph']
        assert graph['cycles'] == []
        assert graph['hidden_cycles'] == [['f', 'g'], ['j', 'm']]
        assert graph['typing_cycles'] == [['f', 'g'], ['j', 'm'], ['t', 'v']]

    def test_a_deferred_import_closes_a_cycle_when_it_lies_on_a_hidden_one(
        self, measured: dict[str, Any]
    ) -> None:
        # g: one target in the importer's hidden cycle suffices; j and m: both
        # function-local edges of one cycle are marked; t: a TYPE_CHECKING import
        # lies on a typing cycle only; e: its target never reaches back.
        assert measured['import_graph']['deferred'] == [
            {'from': 'e', 'line': 2, 'imports': ['f'], 'closes_cycle': False},
            {'from': 'g', 'line': 2, 'imports': ['f', 'w'], 'closes_cycle': True},
            {'from': 'j', 'line': 2, 'imports': ['m'], 'closes_cycle': True},
            {'from': 'm', 'line': 2, 'imports': ['j'], 'closes_cycle': True},
            {'from': 't', 'line': 6, 'imports': ['v'], 'closes_cycle': False},
        ]

    def test_each_src_record_counts_its_cycle_closing_imports(self, measured: dict[str, Any]) -> None:
        counts = {
            record['module']: record['cycle_closing_imports']
            for record in measured['files'].values()
            if record['kind'] == 'src'
        }
        assert counts == {
            'beta.b': 0, 'e': 0, 'f': 0, 'g': 1, 'j': 1, 'm': 1, 't': 0, 'v': 0, 'w': 0,
        }


_GRAPH_BASE: dict[str, str] = {
    **BASE_FILES,
    'alpha/src/a.py': 'A = 1\n',
    'alpha/src/b.py': 'B = 1\n',
    'alpha/src/e.py': 'def f():\n    return 1\n',
    'alpha/src/pkg/__init__.py': '',
    'alpha/src/pkg/a.py': 'Y = 1\n',
}

_GRAPH_CHANGED: dict[str, str] = {
    'alpha/src/a.py': 'import b\nA = 1\n',
    'alpha/src/b.py': 'import a\nB = 1\n',
    'alpha/src/e.py': 'def f():\n    import a\n    return a\n',
    'alpha/src/pkg/__init__.py': 'THING = 1\n',
    'alpha/src/pkg/a.py': 'from pkg import THING\nY = THING\n',
}


class TestImportGraphChanges:
    _GRAPH = 'import graph:'

    def test_additions_then_their_removals(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reverted = {path: _GRAPH_BASE[path] for path in _GRAPH_CHANGED}
        chain = diff_chain(
            tmp_path,
            capsys,
            _GRAPH_BASE,
            lambda root: commit(root, 'couple', write=_GRAPH_CHANGED),
            lambda root: commit(root, 'decouple', write=reverted),
        )
        added = diff_section(chain[1][1], self._GRAPH)
        for line in (
            '  edge added: a -> b',
            '  edge added: b -> a',
            '  cycle added: a, b',
            '  hidden cycle added: a, b',
            '  typing cycle added: a, b',
            '  reach-back added: pkg.a -> pkg (THING)',
            '  deferred import added: e: a',
        ):
            assert line in added, (line, added)
        removed = diff_section(chain[2][1], self._GRAPH)
        for line in (
            '  edge removed: a -> b',
            '  edge removed: b -> a',
            '  cycle removed: a, b',
            '  hidden cycle removed: a, b',
            '  typing cycle removed: a, b',
            '  reach-back removed: pkg.a -> pkg (THING)',
            '  deferred import removed: e: a',
        ):
            assert line in removed, (line, removed)

    def test_line_numbers_are_not_compared(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        coupled = {**_GRAPH_BASE, **_GRAPH_CHANGED}
        shifted = '\n' + _GRAPH_CHANGED['alpha/src/pkg/a.py']
        diff = diff_of(tmp_path, capsys, coupled, rewriting('alpha/src/pkg/a.py', shifted))
        assert diff_section(diff, self._GRAPH) == ['  (none)']

    def test_a_function_local_import_closing_a_cycle_is_a_hidden_cycle_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**BASE_FILES, 'alpha/src/f.py': 'import g\n', 'alpha/src/g.py': 'G = 1\n'}
        diff = diff_of(
            tmp_path, capsys, base, rewriting('alpha/src/g.py', 'def h():\n    import f\n    return f\n')
        )
        assert diff_section(diff, self._GRAPH) == [
            '  edge added: g -> f',
            '  deferred import added: g: f',
            '  cycle-closing deferred import added: g: f',
            '  hidden cycle added: f, g',
            '  typing cycle added: f, g',
        ]

    def test_a_deferred_import_that_starts_closing_a_cycle_is_a_cycle_closing_one_added(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {
            **BASE_FILES,
            'alpha/src/f.py': 'F = 1\n',
            'alpha/src/g.py': 'def h():\n    import f\n    return f\n',
        }
        diff = diff_of(tmp_path, capsys, base, rewriting('alpha/src/f.py', 'import g\nF = 1\n'))
        assert diff_section(diff, self._GRAPH) == [
            '  edge added: f -> g',
            '  cycle-closing deferred import added: g: f',
            '  hidden cycle added: f, g',
            '  typing cycle added: f, g',
        ]

    def test_a_deferred_import_leaving_type_checking_onto_a_hidden_cycle_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Edges, cycle sets and the deferred imports by (from, imports) all stay the same.
        guarded = (
            'from typing import TYPE_CHECKING\n\n\ndef h():\n    import f\n    return f\n\n\n'
            'def k():\n    if TYPE_CHECKING:\n        import f\n    return TYPE_CHECKING\n'
        )
        base = {**BASE_FILES, 'alpha/src/f.py': 'import g\n', 'alpha/src/g.py': guarded}
        unguarded = guarded.replace('    if TYPE_CHECKING:\n        import f\n', '    import f\n')
        diff = diff_of(tmp_path, capsys, base, rewriting('alpha/src/g.py', unguarded))
        assert diff_section(diff, self._GRAPH) == ['  cycle-closing deferred import added: g: f']

    def test_a_type_checking_only_cycle_is_a_typing_cycle_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**BASE_FILES, 'alpha/src/c.py': 'X = 1\n', 'alpha/src/d.py': 'X = 1\n'}

        def couple(root: Path) -> None:
            commit(root, 'couple', write={
                'alpha/src/c.py': GRAPH_FILES['alpha/src/c.py'],
                'alpha/src/d.py': GRAPH_FILES['alpha/src/d.py'],
            })

        assert diff_section(diff_of(tmp_path, capsys, base, couple), self._GRAPH) == [
            '  edge added: c -> d',
            '  edge added: d -> c',
            '  typing cycle added: c, d',
        ]


def _unknown_in_schema_1(side: str) -> list[str]:
    return [
        f'  cycle-closing deferred imports unknown: the {side} snapshot is schema 1',
        f'  hidden cycles unknown: the {side} snapshot is schema 1',
        f'  typing cycles unknown: the {side} snapshot is schema 1',
    ]


class TestASchema1SnapshotIsStillRead:
    @pytest.fixture
    def taken(self, tmp_path: Path) -> dict[str, Any]:
        return measured_snapshot(tmp_path, GRAPH_FILES)

    def test_it_is_valid(self, taken: dict[str, Any]) -> None:
        old = as_schema_1(taken)
        assert snapshot.validate_snapshot(old, origin='v1') == old

    def test_its_absent_cycle_sets_are_unknown_on_either_side(self, taken: dict[str, Any]) -> None:
        # Nothing else in the section: deferred entries compare equal without closes_cycle.
        old = as_schema_1(taken)
        assert diff_section(snapshot.diff_lines(taken, old), 'import graph:') == _unknown_in_schema_1(
            'previous'
        )
        assert diff_section(snapshot.diff_lines(old, taken), 'import graph:') == _unknown_in_schema_1(
            'current'
        )

    def test_a_measuring_run_diffs_against_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = committed_repo(tmp_path, GRAPH_FILES)
        first, second, previous = tmp_path / 's1.json', tmp_path / 's2.json', tmp_path / 'prev.json'
        assert run_snapshot(root, first) == 0
        previous.write_text(json.dumps(as_schema_1(read_json(first))), encoding='utf-8')
        capsys.readouterr()
        assert run_snapshot(root, second, '--diff', str(previous)) == 0
        printed = capsys.readouterr().out.splitlines()
        assert diff_section(printed, 'import graph:') == _unknown_in_schema_1('previous')
        assert snapshot.main(['--current', str(second), '--diff', str(previous)]) == 0
        printed = capsys.readouterr().out.splitlines()
        assert diff_section(printed, 'import graph:') == _unknown_in_schema_1('previous')

    @pytest.mark.parametrize(
        ('named', 'edit'),
        [
            pytest.param(
                'import_graph.deferred[0] keys',
                lambda old: old['import_graph']['deferred'][0].update(closes_cycle=False),
                id='closes_cycle',
            ),
            pytest.param(
                "files['alpha/src/a.py'] keys",
                lambda old: old['files']['alpha/src/a.py'].update(cycle_closing_imports=0),
                id='cycle_closing_imports',
            ),
        ],
    )
    def test_one_carrying_a_schema_2_field_is_refused(
        self, taken: dict[str, Any], named: str, edit: SnapshotEdit
    ) -> None:
        old = as_schema_1(taken)
        edit(old)
        with pytest.raises(source_measures.MetricsError) as raised:
            snapshot.validate_snapshot(old, origin='v1')
        assert named in str(raised.value)

    def test_an_unread_version_is_refused_naming_the_read_ones(self, taken: dict[str, Any]) -> None:
        with pytest.raises(source_measures.MetricsError) as raised:
            snapshot.validate_snapshot({**taken, 'schema_version': 4}, origin='v4')
        assert 'schema_version is 4' in str(raised.value)
        assert 'one of [1, 2, 3]' in str(raised.value)

    def test_the_instrument_is_checked_before_the_version(self) -> None:
        with pytest.raises(source_measures.MetricsError) as raised:
            snapshot.validate_snapshot({'schema_version': 7, 'instrument': 'other'}, origin='foreign')
        assert 'instrument' in str(raised.value)
