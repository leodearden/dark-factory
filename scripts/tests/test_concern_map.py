"""Unit tests of ``scripts/concern_map.py``, the per-module concern map.

The module under measurement is INLINE and synthetic, with neutral names, so
these tests pin what a unit, a ref and the reconciliation are, and know
nothing of git_ops.py. The one real-tree run is an explicit anti-vacuity
anchor, not a definition.
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import MutableMapping
from pathlib import Path

import _checkout
import concern_map
import pytest
import source_measures

ENGINE_MODULE = '''\
"""Synthetic module."""
from __future__ import annotations

import os

LIMIT = 3


def helper(value):
    return value + LIMIT


class Failure(Exception):
    """A synthetic failure."""


class Engine:
    """A synthetic engine."""

    KIND = 'engine'

    def __init__(self, pool):
        self._pool = pool
        self._count = 0

    def poll(self):
        if self._pool.x > LIMIT:
            raise Failure(os.sep)
        return self.helper_method()

    def helper_method(self):
        return helper(self._pool)

    @property
    def size(self):
        return self._count

    def describe(self):
        return f'{self.size}'

    def bump(self):
        def inner():
            self._count += 1

        step = lambda: self._count
        inner()
        return step()

    # Why other is registered:
    # it is the callback.
    @staticmethod
    def other():
        return None

    def wire(self, register):
        register(self.other)
'''

METHOD_NAMES = (
    'Engine.__init__',
    'Engine.poll',
    'Engine.helper_method',
    'Engine.size',
    'Engine.describe',
    'Engine.bump',
    'Engine.other',
    'Engine.wire',
)


def _measure(source: str = ENGINE_MODULE, class_name: str = 'Engine') -> concern_map.ConcernMap:
    return concern_map.measure_module(source, path='engine.py', class_name=class_name)


def _units(source: str = ENGINE_MODULE) -> dict[str, concern_map.Unit]:
    return {unit.name: unit for unit in _measure(source).units}


# ---------------------------------------------------------------------------
# Units: one per method of the target class, one per top-level binding.


class TestUnits:
    def test_each_method_of_the_class_is_a_method_unit_named_by_the_class(self) -> None:
        methods = [
            unit.name for unit in _measure().units if unit.kind is concern_map.UnitKind.METHOD
        ]
        assert methods == list(METHOD_NAMES)

    def test_each_top_level_def_class_and_assignment_is_a_module_unit(self) -> None:
        modules = [
            unit.name for unit in _measure().units if unit.kind is concern_map.UnitKind.MODULE
        ]
        assert modules == ['LIMIT', 'helper', 'Failure']

    def test_imports_and_the_target_class_itself_are_not_units(self) -> None:
        names = set(_units())
        assert not names & {'os', 'annotations', 'Engine'}

    def test_units_come_in_source_order(self) -> None:
        assert [unit.name for unit in _measure().units] == [
            'LIMIT', 'helper', 'Failure', *METHOD_NAMES,
        ]

    def test_a_property_getter_and_setter_are_one_unit(self) -> None:
        source = (
            'class Engine:\n'
            '    @property\n'
            '    def size(self):\n'
            '        return self._size\n'
            '\n'
            '    @size.setter\n'
            '    def size(self, value):\n'
            '        self._size = value\n'
        )
        (unit,) = _measure(source).units
        assert unit.name == 'Engine.size'
        assert unit.lines == 6
        assert unit.writes == frozenset({'_size'})
        assert unit.reads == frozenset({'_size'})


# ---------------------------------------------------------------------------
# What each unit touches: self attributes written and read, and units referenced.


class TestSelfAccessAndRefs:
    @pytest.mark.parametrize(
        ('name', 'writes', 'reads', 'refs'),
        [
            ('Engine.__init__', {'_pool', '_count'}, set(), set()),
            ('Engine.poll', set(), {'_pool'}, {'Engine.helper_method', 'LIMIT', 'Failure'}),
            ('Engine.helper_method', set(), {'_pool'}, {'helper'}),
            ('Engine.size', set(), {'_count'}, set()),
            ('Engine.describe', set(), set(), {'Engine.size'}),
            ('Engine.bump', {'_count'}, {'_count'}, set()),
            ('Engine.other', set(), set(), set()),
            ('Engine.wire', set(), set(), {'Engine.other'}),
            ('LIMIT', set(), set(), set()),
            ('helper', set(), set(), {'LIMIT'}),
            ('Failure', set(), set(), set()),
        ],
    )
    def test_writes_reads_and_refs(self, name, writes, reads, refs) -> None:
        unit = _units()[name]
        assert (unit.writes, unit.reads, unit.refs) == (
            frozenset(writes), frozenset(reads), frozenset(refs),
        )

    def test_the_sets_are_immutable(self) -> None:
        unit = _units()['Engine.poll']
        assert all(
            isinstance(field, frozenset) for field in (unit.writes, unit.reads, unit.refs)
        )

    def test_a_member_reached_through_self_is_a_ref_never_a_read(self) -> None:
        # describe reads the property and poll calls a method: neither is state.
        units = _units()
        assert 'size' not in units['Engine.describe'].reads
        assert 'helper_method' not in units['Engine.poll'].reads

    def test_a_bound_method_passed_as_a_callback_is_a_ref(self) -> None:
        assert _units()['Engine.wire'].refs == frozenset({'Engine.other'})

    def test_nested_defs_and_lambdas_count_toward_the_enclosing_method(self) -> None:
        bump = _units()['Engine.bump']
        assert '_count' in bump.writes  # from the nested def
        assert '_count' in bump.reads  # from the lambda

    def test_a_local_binding_shadows_a_top_level_unit(self) -> None:
        source = (
            'def helper():\n'
            '    return 1\n'
            '\n'
            'class Engine:\n'
            '    def by_param(self, helper):\n'
            '        return helper\n'
            '\n'
            '    def by_assignment(self):\n'
            '        helper = 2\n'
            '        return helper\n'
            '\n'
            '    def by_global(self):\n'
            '        return helper()\n'
        )
        units = _units(source)
        assert units['Engine.by_param'].refs == frozenset()
        assert units['Engine.by_assignment'].refs == frozenset()
        assert units['Engine.by_global'].refs == frozenset({'helper'})

    def test_a_member_reached_through_the_class_name_is_a_ref(self) -> None:
        source = (
            'class Engine:\n'
            '    def first(self):\n'
            '        return 1\n'
            '\n'
            '    def second(self):\n'
            '        return Engine.first(self)\n'
        )
        assert _units(source)['Engine.second'].refs == frozenset({'Engine.first'})

    def test_a_classmethod_receiver_resolves_like_self(self) -> None:
        source = (
            'class Engine:\n'
            '    _registry = {}\n'
            '\n'
            '    @classmethod\n'
            '    def build(cls):\n'
            '        cls._registry = {}\n'
            '        return cls.first()\n'
            '\n'
            '    def first(self):\n'
            '        return 1\n'
        )
        build = _units(source)['Engine.build']
        assert build.writes == frozenset({'_registry'})
        assert build.refs == frozenset({'Engine.first'})


# ---------------------------------------------------------------------------
# Lines: a unit's span, and the prose inside it.


class TestLines:
    @pytest.mark.parametrize(
        ('name', 'lines', 'prose_lines'),
        [
            ('LIMIT', 1, 0),
            ('helper', 2, 0),
            ('Failure', 2, 1),
            ('Engine.__init__', 3, 0),
            ('Engine.poll', 4, 0),
            ('Engine.size', 3, 0),
            ('Engine.bump', 7, 0),
            ('Engine.other', 5, 2),
        ],
    )
    def test_lines_and_prose_lines(self, name, lines, prose_lines) -> None:
        unit = _units()[name]
        assert (unit.lines, unit.prose_lines) == (lines, prose_lines)

    def test_a_unit_spans_from_its_first_decorator(self) -> None:
        # @property, def size, return: the decorator line is the unit's.
        assert _units()['Engine.size'].lines == 3

    def test_a_contiguous_comment_block_above_the_decorator_belongs_to_the_unit(self) -> None:
        # Two comment lines, @staticmethod, def other, return None.
        other = _units()['Engine.other']
        assert (other.lines, other.prose_lines) == (5, 2)

    def test_a_comment_block_separated_by_a_blank_line_is_not_attached(self) -> None:
        source = (
            'class Engine:\n'
            '    # loose\n'
            '\n'
            '    def first(self):\n'
            '        return 1\n'
        )
        assert _units(source)['Engine.first'].lines == 2

    def test_prose_lines_come_from_the_one_prose_definition(self) -> None:
        tree = source_measures.parse_source(ENGINE_MODULE, path='engine.py')
        prose = source_measures.prose_line_numbers_in_tree(ENGINE_MODULE, tree, path='engine.py')
        outside_units = 2  # the module docstring and Engine's own docstring
        assert sum(unit.prose_lines for unit in _measure().units) == len(prose) - outside_units


# ---------------------------------------------------------------------------
# Reconciliation (INV-11): every line of the file is measured exactly once.


class TestReconciliation:
    def test_unit_lines_plus_the_residual_buckets_are_the_file_total(self) -> None:
        cmap = _measure()
        measured = sum(unit.lines for unit in cmap.units) + sum(cmap.residual.values())
        assert measured == cmap.total_lines == len(ENGINE_MODULE.splitlines())

    def test_the_residual_buckets_are_named_and_measured(self) -> None:
        assert dict(_measure().residual) == {
            'module_preamble': 3,
            'class_non_method': 12,
            'module_other': 0,
            'inter_unit_gaps': 8,
        }

    def test_a_top_level_statement_binding_no_name_is_module_other(self) -> None:
        source = (
            'import sys\n'
            'if sys.argv:\n'
            '    pass\n'
            'class Engine:\n'
            '    pass\n'
        )
        assert _measure(source).residual['module_other'] == 2

    def test_two_units_sharing_a_line_are_a_named_failure(self) -> None:
        source = 'class Engine:\n    pass\na = 1; b = 2\n'
        with pytest.raises(concern_map.MetricsError, match=r'engine\.py.*\b4\b.*\b3\b'):
            _measure(source)


# ---------------------------------------------------------------------------
# Failures are named, never an empty map (INV-11).


class TestFailures:
    def test_an_absent_class_raises_naming_the_path_and_the_class(self) -> None:
        with pytest.raises(concern_map.MetricsError) as excinfo:
            _measure(class_name='Missing')
        message = str(excinfo.value)
        assert 'engine.py' in message
        assert 'Missing' in message

    def test_a_class_nested_below_top_level_is_absent(self) -> None:
        source = 'def outer():\n    class Engine:\n        pass\n'
        with pytest.raises(concern_map.MetricsError, match='Engine'):
            _measure(source)

    def test_the_error_is_source_measures_own(self) -> None:
        assert concern_map.MetricsError is source_measures.MetricsError

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(concern_map.MetricsError, match='engine.py'):
            _measure('class Engine(:\n')

    def test_the_map_is_immutable(self) -> None:
        cmap = _measure()
        for field in dataclasses.fields(cmap):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(cmap, field.name, getattr(cmap, field.name))
        assert not isinstance(cmap.residual, MutableMapping)


# ---------------------------------------------------------------------------
# evaluate_partition: what a proposed split of the units would cost.
#
# A refs B (poll -> helper_method), B refs A (describe -> size), B refs C
# (helper -> LIMIT), and A and B both write _count (__init__, bump).

PARTITION = {
    'A': ['Engine.__init__', 'Engine.poll', 'Engine.size'],
    'B': ['Engine.helper_method', 'Engine.describe', 'Engine.bump', 'helper'],
    'C': ['LIMIT', 'Failure', 'Engine.other', 'Engine.wire'],
}

# A -> B -> C -> A: poll -> helper_method -> helper -> LIMIT.
THREE_CYCLE = {
    'A': ['Engine.poll', 'LIMIT'],
    'B': ['Engine.helper_method'],
    'C': ['helper'],
    'D': [
        'Failure', 'Engine.__init__', 'Engine.size', 'Engine.describe',
        'Engine.bump', 'Engine.other', 'Engine.wire',
    ],
}


def _report(partition=PARTITION) -> concern_map.PartitionReport:
    return concern_map.evaluate_partition(_measure(), partition)


class TestEvaluatePartition:
    def test_cluster_totals_sum_to_the_measured_units(self) -> None:
        cmap = _measure()
        clusters = _report().clusters
        assert sum(size.units for size in clusters.values()) == len(cmap.units)
        assert sum(size.lines for size in clusters.values()) == sum(u.lines for u in cmap.units)
        assert sum(size.prose_lines for size in clusters.values()) == sum(
            u.prose_lines for u in cmap.units
        )

    def test_a_cluster_total_is_the_sum_of_its_units(self) -> None:
        # __init__ 3 + poll 4 + size 3; C holds Failure's docstring and other's comments.
        clusters = _report().clusters
        assert (clusters['A'].units, clusters['A'].lines, clusters['A'].prose_lines) == (3, 10, 0)
        assert clusters['C'].prose_lines == 3

    def test_edges_count_cross_cluster_refs_with_a_sample_pair(self) -> None:
        edges = _report().edges
        assert {key: edge.count for key, edge in edges.items()} == {
            ('A', 'B'): 1,
            ('A', 'C'): 2,
            ('B', 'A'): 1,
            ('B', 'C'): 1,
        }
        assert edges[('A', 'B')].sample == ('Engine.poll', 'Engine.helper_method')
        assert edges[('A', 'C')].sample in {
            ('Engine.poll', 'LIMIT'), ('Engine.poll', 'Failure'),
        }

    def test_intra_cluster_refs_are_not_edges(self) -> None:
        # wire -> other and helper_method -> helper stay inside their clusters.
        assert not {key for key in _report().edges if key[0] == key[1]}
        assert ('C', 'C') not in _report().edges

    def test_a_two_cluster_cycle_is_one_cyclic_group(self) -> None:
        assert _report().cyclic_groups == (frozenset({'A', 'B'}),)

    def test_a_three_cluster_cycle_is_one_cyclic_group(self) -> None:
        assert _report(THREE_CYCLE).cyclic_groups == (frozenset({'A', 'B', 'C'}),)

    def test_an_acyclic_partition_has_no_cyclic_group(self) -> None:
        acyclic = {
            'top': ['Engine.poll', 'Engine.describe', 'Engine.wire', 'Engine.bump'],
            'bottom': [
                'Engine.helper_method', 'helper', 'LIMIT', 'Failure', 'Engine.size',
                'Engine.other', 'Engine.__init__',
            ],
        }
        assert _report(acyclic).cyclic_groups == ()

    def test_an_attribute_written_from_two_clusters_is_a_shared_write(self) -> None:
        assert dict(_report().shared_writes) == {'_count': frozenset({'A', 'B'})}

    def test_module_unit_users_are_the_clusters_referencing_each_module_unit(self) -> None:
        assert dict(_report().module_unit_users) == {
            'LIMIT': frozenset({'A', 'B'}),
            'helper': frozenset({'B'}),
            'Failure': frozenset({'A'}),
        }

    def test_the_report_is_immutable(self) -> None:
        report = _report()
        for field in dataclasses.fields(report):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(report, field.name, getattr(report, field.name))
        assert not isinstance(report.clusters, MutableMapping)


class TestPartitionCoverage:
    def test_the_error_is_a_metrics_error(self) -> None:
        assert issubclass(concern_map.PartitionError, source_measures.MetricsError)

    def test_every_unassigned_duplicate_and_unknown_unit_is_named(self) -> None:
        broken = {
            'A': ['Engine.__init__', 'Engine.poll', 'Engine.size', 'helper', 'Engine.ghost'],
            'B': ['Engine.helper_method', 'Engine.describe', 'Engine.bump', 'helper'],
            'C': ['LIMIT', 'Failure', 'nope'],
        }
        with pytest.raises(concern_map.PartitionError) as excinfo:
            _report(broken)
        message = str(excinfo.value)
        for name in ('Engine.other', 'Engine.wire', 'helper', 'Engine.ghost', 'nope'):
            assert name in message

    def test_a_unit_listed_twice_in_one_cluster_is_a_duplicate(self) -> None:
        twice = {**PARTITION, 'C': [*PARTITION['C'], 'LIMIT']}
        with pytest.raises(concern_map.PartitionError, match='LIMIT'):
            _report(twice)



# ---------------------------------------------------------------------------
# The command line: JSON and markdown renderings, and the exit-code contract.


@pytest.fixture
def engine_root(tmp_path: Path) -> Path:
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'engine.py').write_text(ENGINE_MODULE, encoding='utf-8')
    return tmp_path


def _partition_file(tmp_path: Path, content: object) -> str:
    target = tmp_path / 'partition.json'
    target.write_text(json.dumps(content), encoding='utf-8')
    return str(target)


def _run(capsys, root: Path, *extra: str) -> tuple[int, str, str]:
    code = concern_map.main(['pkg/engine.py', '--class', 'Engine', '--root', str(root), *extra])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestJsonRendering:
    def test_the_map_renders_as_a_json_object(self, capsys, engine_root) -> None:
        code, out, _ = _run(capsys, engine_root, '--format', 'json')
        assert code == 0
        rendered = json.loads(out)
        assert (rendered['path'], rendered['class_name'], rendered['total_lines']) == (
            'pkg/engine.py', 'Engine', 56,
        )
        assert rendered['residual'] == dict(_measure().residual)
        assert 'partition' not in rendered

    def test_each_unit_renders_its_measures_with_sorted_lists(self, capsys, engine_root) -> None:
        _, out, _ = _run(capsys, engine_root, '--format', 'json')
        units = {unit['name']: unit for unit in json.loads(out)['units']}
        assert [unit['name'] for unit in json.loads(out)['units']][:3] == ['LIMIT', 'helper', 'Failure']
        assert units['Engine.poll'] == {
            'name': 'Engine.poll',
            'kind': 'method',
            'lines': 4,
            'prose_lines': 0,
            'writes': [],
            'reads': ['_pool'],
            'refs': ['Engine.helper_method', 'Failure', 'LIMIT'],
        }
        assert units['Engine.__init__']['writes'] == ['_count', '_pool']
        assert units['LIMIT']['kind'] == 'module'

    def test_a_partition_adds_its_report(self, capsys, engine_root, tmp_path) -> None:
        code, out, _ = _run(
            capsys, engine_root, '--format', 'json',
            '--partition', _partition_file(tmp_path, PARTITION),
        )
        assert code == 0
        partition = json.loads(out)['partition']
        assert partition['clusters']['A'] == {'units': 3, 'lines': 10, 'prose_lines': 0}
        assert partition['edges']['A -> B'] == {
            'count': 1, 'sample': ['Engine.poll', 'Engine.helper_method'],
        }
        assert set(partition['edges']) == {'A -> B', 'A -> C', 'B -> A', 'B -> C'}
        assert partition['cyclic_groups'] == [['A', 'B']]
        assert partition['shared_writes'] == {'_count': ['A', 'B']}
        assert partition['module_unit_users']['LIMIT'] == ['A', 'B']

    def test_the_rendering_is_deterministic(self, capsys, engine_root, tmp_path) -> None:
        partition = _partition_file(tmp_path, PARTITION)
        first = _run(capsys, engine_root, '--format', 'json', '--partition', partition)[1]
        second = _run(capsys, engine_root, '--format', 'json', '--partition', partition)[1]
        assert first == second


class TestMarkdownRendering:
    def test_the_unit_table_has_a_row_per_unit(self, capsys, engine_root) -> None:
        code, out, _ = _run(capsys, engine_root, '--format', 'markdown')
        assert code == 0
        assert '| unit | kind | lines | prose | writes | reads | refs |' in out
        rows = [line for line in out.splitlines() if line.startswith('| Engine.') or line.startswith('| LIMIT')]
        assert len(rows) == len(METHOD_NAMES) + 1
        assert '| Engine.poll | method | 4 | 0 |  | _pool | Engine.helper_method, Failure, LIMIT |' in out

    def test_markdown_is_the_default_format(self, capsys, engine_root) -> None:
        assert _run(capsys, engine_root)[1] == _run(capsys, engine_root, '--format', 'markdown')[1]

    def test_a_partition_adds_cluster_edge_and_cycle_sections(
        self, capsys, engine_root, tmp_path
    ) -> None:
        _, out, _ = _run(
            capsys, engine_root, '--format', 'markdown',
            '--partition', _partition_file(tmp_path, PARTITION),
        )
        assert '| cluster | units | lines | prose |' in out
        assert '| A | 3 | 10 | 0 |' in out
        assert '| from | to | refs | sample |' in out
        assert '| A | B | 1 | Engine.poll -> Engine.helper_method |' in out
        cycles = out.split('## Cyclic cluster groups', 1)[1]
        assert '- A, B' in cycles.split('##', 1)[0]
        assert '`_count`: A, B' in out

    @pytest.mark.parametrize('with_partition', [False, True])
    def test_every_table_row_has_as_many_cells_as_its_header(
        self, capsys, engine_root, tmp_path, with_partition
    ) -> None:
        extra = ['--partition', _partition_file(tmp_path, PARTITION)] if with_partition else []
        _, out, _ = _run(capsys, engine_root, *extra)
        tables = [block for block in out.split('\n\n') if block.startswith('|')]
        assert len(tables) == (4 if with_partition else 1)
        for table in tables:
            widths = {line.count('|') for line in table.splitlines()}
            assert len(widths) == 1, table

    def test_an_acyclic_partition_says_none_explicitly(self, capsys, engine_root, tmp_path) -> None:
        acyclic = {'all': [unit.name for unit in _measure().units]}
        _, out, _ = _run(
            capsys, engine_root, '--format', 'markdown',
            '--partition', _partition_file(tmp_path, acyclic),
        )
        cycles = out.split('## Cyclic cluster groups', 1)[1].split('##', 1)[0]
        assert cycles.strip() == 'none'


class TestExitCodes:
    def test_a_missing_class_is_an_instrument_failure(self, capsys, engine_root) -> None:
        code = concern_map.main(['pkg/engine.py', '--class', 'Missing', '--root', str(engine_root)])
        err = capsys.readouterr().err
        assert code == 2
        assert 'Missing' in err

    def test_a_missing_source_is_an_instrument_failure(self, capsys, engine_root) -> None:
        code = concern_map.main(['pkg/absent.py', '--class', 'Engine', '--root', str(engine_root)])
        assert code == 2
        assert 'pkg/absent.py' in capsys.readouterr().err

    def test_an_unreadable_partition_file_is_an_instrument_failure(
        self, capsys, engine_root, tmp_path
    ) -> None:
        absent = str(tmp_path / 'no-such-partition.json')
        code, _, err = _run(capsys, engine_root, '--partition', absent)
        assert code == 2
        assert 'no-such-partition.json' in err

    @pytest.mark.parametrize(
        'content',
        [
            pytest.param('{not json', id='not-json'),
            pytest.param('["A"]', id='not-an-object'),
            pytest.param('{"A": "Engine.poll"}', id='members-not-a-list'),
            pytest.param('{"A": ["Engine.poll", 3]}', id='member-not-a-string'),
        ],
    )
    def test_a_malformed_partition_file_is_an_instrument_failure(
        self, capsys, engine_root, tmp_path, content
    ) -> None:
        target = tmp_path / 'partition.json'
        target.write_text(content, encoding='utf-8')
        code, out, err = _run(capsys, engine_root, '--partition', str(target))
        assert code == 2
        assert 'partition.json' in err
        assert out == ''

    def test_a_partition_error_is_an_instrument_failure(self, capsys, engine_root, tmp_path) -> None:
        partial = {'A': ['Engine.poll']}
        code, _, err = _run(capsys, engine_root, '--partition', _partition_file(tmp_path, partial))
        assert code == 2
        assert 'Engine.wire' in err


class TestTheRealTree:
    def test_git_ops_measures_with_at_least_one_method_unit(self, capsys) -> None:
        # ANTI-VACUITY anchor only: independent of git_ops.py's size, which the
        # git-ops split PRD exists to shrink.
        code = concern_map.main([
            'orchestrator/src/orchestrator/git_ops.py', '--class', 'GitOps',
            '--format', 'json', '--root', str(_checkout.ROOT),
        ])
        rendered = json.loads(capsys.readouterr().out)
        assert code == 0
        assert any(unit['kind'] == 'method' for unit in rendered['units'])
