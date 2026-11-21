"""Unit tests of ``scripts/concern_map.py``, the per-module concern map.

The module under measurement is INLINE and synthetic, with neutral names, so
these tests pin what a unit, a ref and the reconciliation are, and know
nothing of git_ops.py. The one real-tree run is an explicit anti-vacuity
anchor, not a definition.
"""
from __future__ import annotations

import dataclasses

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
        with pytest.raises(dataclasses.FrozenInstanceError):
            cmap.total_lines = 0  # type: ignore[misc]
        with pytest.raises(TypeError):
            cmap.residual['inter_unit_gaps'] = 0  # type: ignore[index]
