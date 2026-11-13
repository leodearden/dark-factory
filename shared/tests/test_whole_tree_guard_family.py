"""Contract of shared/src/shared/whole_tree_guard_family.py, over inline source fixtures.

Membership of the whole-tree guard family is decided by the NAME the
module-level timeout mark is spelled with, never by its value, so every
spelling here is a source string and nothing is imported from the tree.
"""

from __future__ import annotations

import ast

import pytest

import shared.whole_tree_guard_family as family
from shared.whole_tree_guard_family import (
    WHOLE_TREE_SCAN_TEST_TIMEOUT,
    WHOLE_TREE_SCAN_TIMEOUT_NAME,
    is_whole_tree_guard_ast,
    is_whole_tree_guard_module,
    min_leg_budget_for_family,
    module_level_timeout_arguments,
)

_HOUSE_GUARD_SOURCE = (
    'import pytest\n'
    'from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
    'pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)\n'
)

_ADMITTED_SOURCES = [
    pytest.param(_HOUSE_GUARD_SOURCE, id='house-spelling'),
    pytest.param(
        'import pytest\n'
        'import shared.whole_tree_guard_family as family\n'
        'pytestmark = pytest.mark.timeout(family.WHOLE_TREE_SCAN_TEST_TIMEOUT)\n',
        id='dotted',
    ),
    pytest.param(
        'import pytest\n'
        'from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
        'pytestmark = pytest.mark.timeout(timeout=WHOLE_TREE_SCAN_TEST_TIMEOUT)\n',
        id='keyword',
    ),
    pytest.param(
        'import pytest\n'
        'from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
        'pytestmark = [pytest.mark.slow, pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)]\n',
        id='list-form',
    ),
    pytest.param(
        'import pytest\n'
        'from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
        'pytestmark: list = [pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)]\n',
        id='annotated',
    ),
    pytest.param(
        'import pytest\n'
        'from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
        'pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)\n',
        id='legacy-orch-helpers-import',
    ),
    pytest.param(
        'import pytest\n'
        'import _orch_helpers\n'
        'pytestmark = pytest.mark.timeout(_orch_helpers.WHOLE_TREE_SCAN_TEST_TIMEOUT)\n',
        id='legacy-orch-helpers-dotted',
    ),
]

_REJECTED_SOURCES = [
    pytest.param(
        'import pytest\npytestmark = pytest.mark.timeout(30)\n',
        id='small-literal',
    ),
    pytest.param(
        'import pytest\npytestmark = pytest.mark.timeout(1200)\n',
        id='literal-equal-to-the-ceiling',
    ),
    pytest.param(
        'import pytest\n'
        'from somewhere import SOME_OTHER_TIMEOUT\n'
        'pytestmark = pytest.mark.timeout(SOME_OTHER_TIMEOUT)\n',
        id='different-constant',
    ),
    pytest.param(
        'import pytest\n'
        'from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
        'pytestmark = pytest.mark.timeout()\n',
        id='zero-arg-timeout',
    ),
    pytest.param(
        '"""Mentions WHOLE_TREE_SCAN_TEST_TIMEOUT in prose only."""\n'
        'import pytest\n'
        'from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
        '# pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)\n'
        'pytestmark = pytest.mark.slow\n',
        id='import-comment-docstring-only',
    ),
    pytest.param(
        'import pytest\n'
        'from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT\n'
        'def f():\n'
        '    pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)\n'
        '    return pytestmark\n',
        id='function-local-pytestmark',
    ),
    pytest.param(
        _HOUSE_GUARD_SOURCE + 'pytestmark = pytest.mark.slow\n',
        id='rebound-without-timeout',
    ),
    pytest.param(None, id='none'),
    pytest.param('', id='empty'),
    pytest.param('def f(:\n', id='syntax-error'),
    pytest.param(
        'pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT\n',
        id='syntax-error-mentioning-the-name',
    ),
]


@pytest.mark.parametrize('source', _ADMITTED_SOURCES)
def test_guard_family_admits_every_sanctioned_spelling(source: str) -> None:
    assert is_whole_tree_guard_module(source) is True


@pytest.mark.parametrize('source', _REJECTED_SOURCES)
def test_guard_family_rejects_everything_else_without_raising(source: str | None) -> None:
    assert is_whole_tree_guard_module(source) is False


def _parseable(params: list) -> list:
    parseable = []
    for param in params:
        (source,) = param.values
        if source is None:
            continue
        try:
            ast.parse(source)
        except SyntaxError:
            continue
        parseable.append(param)
    return parseable


@pytest.mark.parametrize('source', _parseable(_ADMITTED_SOURCES + _REJECTED_SOURCES))
def test_the_tree_level_core_agrees_with_the_source_predicate(source: str) -> None:
    assert is_whole_tree_guard_ast(ast.parse(source)) is is_whole_tree_guard_module(source)


def test_the_spelling_names_the_module_own_binding() -> None:
    assert getattr(family, WHOLE_TREE_SCAN_TIMEOUT_NAME) == WHOLE_TREE_SCAN_TEST_TIMEOUT


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        pytest.param('import pytest\npytestmark = pytest.mark.timeout(30)\n', ['30'], id='positional'),
        pytest.param(
            'import pytest\npytestmark = pytest.mark.timeout(timeout=30)\n', ['30'], id='keyword',
        ),
        pytest.param(
            'import pytest\n'
            'pytestmark = [pytest.mark.timeout(300), pytest.mark.timeout(30)]\n',
            ['300', '30'],
            id='two-marks-in-element-order',
        ),
        pytest.param(
            'import pytest\n'
            'pytestmark = [pytest.mark.slow, pytest.mark.timeout(_orch_helpers.X)]\n',
            ['_orch_helpers.X'],
            id='non-timeout-siblings-skipped',
        ),
        pytest.param(
            'import pytest\n'
            'pytestmark = pytest.mark.timeout(30)\n'
            'pytestmark = pytest.mark.timeout(45)\n',
            ['45'],
            id='last-binding-wins',
        ),
        pytest.param('import pytest\n', [], id='no-pytestmark'),
        pytest.param('import pytest\npytestmark = pytest.mark.asyncio\n', [], id='no-timeout'),
        pytest.param('import pytest\npytestmark = pytest.mark.timeout()\n', [], id='zero-arg'),
        pytest.param('def f(:\n', [], id='syntax-error'),
        pytest.param(None, [], id='none'),
    ],
)
def test_module_level_timeout_arguments_reads_the_seconds_expressions(
    source: str | None, expected: list[str],
) -> None:
    arguments = module_level_timeout_arguments(source)
    assert isinstance(arguments, tuple)
    assert [ast.unparse(argument) for argument in arguments] == expected


def test_module_level_timeout_arguments_returns_the_name_node_unevaluated() -> None:
    (argument,) = module_level_timeout_arguments(_HOUSE_GUARD_SOURCE)
    assert isinstance(argument, ast.Name)
    assert argument.id == 'WHOLE_TREE_SCAN_TEST_TIMEOUT'


_RECORDED_WORST_GREEN_RUNS = [0.0, 296.88, 397.47, 930.59, 1695.08, 4626.17]


@pytest.mark.parametrize('worst_green_run_secs', _RECORDED_WORST_GREEN_RUNS)
def test_min_leg_budget_is_the_next_whole_hundred_past_a_late_hang_reaching_the_ceiling(
    worst_green_run_secs: float,
) -> None:
    budget = min_leg_budget_for_family(worst_green_run_secs)
    late_hang_fires_at = worst_green_run_secs + WHOLE_TREE_SCAN_TEST_TIMEOUT
    assert isinstance(budget, int)
    assert budget % 100 == 0
    assert budget >= late_hang_fires_at
    assert budget - late_hang_fires_at < 100


def test_min_leg_budget_adds_nothing_when_the_sum_is_already_a_whole_hundred() -> None:
    worst_green_run_secs = float(100 - WHOLE_TREE_SCAN_TEST_TIMEOUT % 100)
    assert min_leg_budget_for_family(worst_green_run_secs) == (
        worst_green_run_secs + WHOLE_TREE_SCAN_TEST_TIMEOUT
    )


def test_min_leg_budget_never_shrinks_as_the_worst_green_run_grows() -> None:
    budgets = [min_leg_budget_for_family(w) for w in sorted(_RECORDED_WORST_GREEN_RUNS)]
    assert budgets == sorted(budgets)
