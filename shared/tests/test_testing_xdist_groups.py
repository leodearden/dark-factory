"""Contract of shared/src/shared/testing_xdist_groups.py."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest

from shared.testing_xdist_groups import single_xdist_group, xdist_group_of


@dataclass(frozen=True)
class _StubItem:
    """The two attributes of a collected item that xdist's group rule reads."""

    nodeid: str
    marks: tuple[pytest.Mark, ...] = field(default=())

    def iter_markers(self, name: str) -> Iterator[pytest.Mark]:
        return (mark for mark in self.marks if mark.name == name)


def _item(nodeid: str, *marks: pytest.MarkDecorator) -> _StubItem:
    return _StubItem(nodeid, tuple(decorator.mark for decorator in marks))


class TestXdistGroupOf:
    def test_an_item_without_the_mark_has_no_group(self):
        assert xdist_group_of(_item('t.py::a')) is None

    def test_a_positional_name_is_the_group(self):
        assert xdist_group_of(_item('t.py::a', pytest.mark.xdist_group('a'))) == 'a'

    def test_a_keyword_name_is_the_group(self):
        assert xdist_group_of(_item('t.py::a', pytest.mark.xdist_group(name='b'))) == 'b'

    def test_a_bare_mark_is_the_default_group(self):
        assert xdist_group_of(_item('t.py::a', pytest.mark.xdist_group())) == 'default'

    def test_several_marks_join_their_sorted_names(self):
        item = _item('t.py::a', pytest.mark.xdist_group('b'), pytest.mark.xdist_group('a'))

        assert xdist_group_of(item) == 'a_b'

    def test_a_repeated_name_counts_once(self):
        item = _item(
            't.py::a',
            pytest.mark.xdist_group('a'),
            pytest.mark.xdist_group(name='a'),
        )

        assert xdist_group_of(item) == 'a'

    def test_marks_of_another_name_are_ignored(self):
        item = _item('t.py::a', pytest.mark.timeout(5), pytest.mark.xdist_group('a'))

        assert xdist_group_of(item) == 'a'


class TestSingleXdistGroup:
    def test_items_sharing_a_group_resolve_it(self):
        items = [
            _item('t.py::a', pytest.mark.xdist_group('g')),
            _item('t.py::b', pytest.mark.xdist_group(name='g')),
        ]

        assert single_xdist_group(items) == 'g'

    def test_a_generator_of_items_is_accepted(self):
        items = (_item(f't.py::{n}', pytest.mark.xdist_group('g')) for n in 'ab')

        assert single_xdist_group(items) == 'g'

    def test_items_in_different_groups_name_every_item(self):
        items = [
            _item('t.py::a', pytest.mark.xdist_group('g')),
            _item('t.py::b', pytest.mark.xdist_group('h')),
        ]

        with pytest.raises(AssertionError) as excinfo:
            single_xdist_group(items)

        message = str(excinfo.value)
        assert 'share one xdist_group' in message
        assert 't.py::a -> g' in message
        assert 't.py::b -> h' in message

    def test_an_ungrouped_item_is_named_as_such(self):
        items = [
            _item('t.py::a', pytest.mark.xdist_group('g')),
            _item('t.py::b'),
        ]

        with pytest.raises(AssertionError) as excinfo:
            single_xdist_group(items)

        message = str(excinfo.value)
        assert 't.py::a -> g' in message
        assert 't.py::b -> no xdist_group' in message

    def test_a_lone_ungrouped_item_fails(self):
        with pytest.raises(AssertionError, match='t.py::a -> no xdist_group'):
            single_xdist_group([_item('t.py::a')])

    def test_no_items_is_the_callers_decision(self):
        with pytest.raises(ValueError):
            single_xdist_group([])
