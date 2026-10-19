"""Contract of shared/src/shared/pytest_mark_grammar.py, over inline AST fixtures."""

from __future__ import annotations

import ast

import pytest

from shared.pytest_mark_grammar import (
    mark_elements,
    marker_name,
    pytestmark_elements,
    pytestmark_value,
)


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode='eval').body


def _statement(source: str) -> ast.stmt:
    return ast.parse(source).body[0]


def _body(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


@pytest.mark.parametrize(('source_expr', 'expected'), [
    ('pytest.mark.slow', 'slow'),
    ('pytest.mark.timeout(120)', 'timeout'),
    ('pytest.mark', None),
    ('other.mark.thing', None),
    ('functools.wraps(f)', None),
    ('pytest.raises', None),
    ('slow', None),
])
def test_marker_name_reads_only_the_pytest_mark_namespace(source_expr, expected):
    assert marker_name(_expr(source_expr)) == expected


@pytest.mark.parametrize('source', [
    'pytestmark = pytest.mark.slow\n',
    'pytestmark: list = [pytest.mark.slow]\n',
])
def test_pytestmark_value_hands_back_the_bound_value_node_itself(source):
    statement = _statement(source)
    assert isinstance(statement, ast.Assign | ast.AnnAssign)
    assert pytestmark_value(statement) is statement.value


@pytest.mark.parametrize('source', [
    'pytestmark: list\n',
    'x = 1\n',
    'x: int = 1\n',
    'pytest.mark.slow\n',
])
def test_pytestmark_value_is_none_when_nothing_binds_pytestmark(source):
    assert pytestmark_value(_statement(source)) is None


@pytest.mark.parametrize('source', [
    '[pytest.mark.slow, pytest.mark.timeout(5)]',
    '(pytest.mark.slow, pytest.mark.timeout(5))',
])
def test_mark_elements_unwraps_a_list_or_tuple_into_its_elements(source):
    value = _expr(source)
    assert isinstance(value, ast.List | ast.Tuple)
    elements = mark_elements(value)
    assert len(elements) == 2
    assert all(element is original for element, original in zip(elements, value.elts, strict=True))


def test_mark_elements_wraps_a_single_expression():
    value = _expr('pytest.mark.timeout(5)')
    assert mark_elements(value) == [value]


def test_pytestmark_elements_hands_back_a_single_bound_element_itself():
    body = _body('pytestmark = pytest.mark.slow\n')
    statement = body[0]
    assert isinstance(statement, ast.Assign)
    elements = pytestmark_elements(body)
    assert len(elements) == 1
    assert elements[0] is statement.value


def test_pytestmark_elements_unwraps_a_list_binding():
    elements = pytestmark_elements(_body('pytestmark = [pytest.mark.slow, pytest.mark.timeout(5)]\n'))
    assert [marker_name(element) for element in elements] == ['slow', 'timeout']


def test_pytestmark_elements_takes_the_last_binding():
    body = _body('pytestmark = pytest.mark.slow\npytestmark = [pytest.mark.timeout(5)]\n')
    assert [marker_name(element) for element in pytestmark_elements(body)] == ['timeout']


def test_pytestmark_elements_ignores_a_later_valueless_annotation():
    body = _body('pytestmark = pytest.mark.slow\npytestmark: list\n')
    assert [marker_name(element) for element in pytestmark_elements(body)] == ['slow']


@pytest.mark.parametrize('source', [
    'if True:\n    pytestmark = pytest.mark.slow\n',
    'x = 1\n',
    '',
])
def test_pytestmark_elements_is_empty_without_a_direct_child_binding(source):
    assert pytestmark_elements(_body(source)) == []
