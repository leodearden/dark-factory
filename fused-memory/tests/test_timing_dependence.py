"""Tests for _timing_dependence.py, Rule D ``timing-dependence`` of the test-quality lint.

The subject is that module alone, driven through its public per-node API:
``timing_facts`` over every node of one walk, then ``timing_violations``.  How the
main checker composes Rule D with Rules A-C is tested beside that script, in
test_check_bare_magicmock_config.py.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

# fused-memory/scripts/ is not on PYTHONPATH, so load the modules by path.  The
# default module names keep _timing_dependence's own `from _lint_cli import ...`
# bound to this same _lint_cli module object.
_SCRIPTS = Path(__file__).parent.parent / 'scripts'
_lint_cli = load_script_module(_SCRIPTS / '_lint_cli.py')
_timing = load_script_module(_SCRIPTS / '_timing_dependence.py')

_NEW_FILE = 'orchestrator/tests/test_brand_new.py'
_FOREIGN_VOCABULARY = ('wall-clock-deadline', 'bare-magicmock', 'bare-dataclass-double', '# noqa:')


_NO_DEBT = _lint_cli.DebtBaseline.from_mapping({})


def _scan(source: str, filename: str = _NEW_FILE, budgets=_NO_DEBT):
    tree = ast.parse(textwrap.dedent(source))
    facts = [
        fact
        for node in ast.walk(tree)
        for fact in _timing.timing_facts(node, deadline_already_reported=False)
    ]
    return _timing.timing_violations(facts, filename=filename, budgets=budgets)


def _assert_owns_its_vocabulary(message: str) -> None:
    assert 'timing-dependence' in message, message
    for foreign in _FOREIGN_VOCABULARY:
        assert foreign not in message, f'Rule D must not carry {foreign!r}: {message!r}'


_SLEEP_PRELUDE = (
    'import asyncio\n'
    'import math\n'
    'import time\n'
    '\n'
    '\n'
    'async def test_case(interval, fake_clock):\n'
)
_SLEEP_LINENO = _SLEEP_PRELUDE.count('\n') + 1
_SLEEP_CALL = re.compile(r'(time|asyncio)\.sleep\(')


def _in_test(statement: str) -> str:
    return _SLEEP_PRELUDE + f'    {statement}\n'


class TestSleepLeg:
    """A real sleep through the time or asyncio module is a site; a yield or a hang simulator is not."""

    @pytest.mark.parametrize(
        'statement',
        [
            'time.sleep(0.1)',
            'await asyncio.sleep(0.05)',
            'await asyncio.sleep(interval)',
            'await asyncio.sleep(delay=0.2)',
            'time.sleep(30)',
        ],
    )
    def test_a_real_sleep_is_one_violation_at_the_call(self, statement: str):
        found = _scan(_in_test(statement))
        call_col = 4 + _SLEEP_CALL.search(statement).start()
        assert [(v.lineno, v.col_offset) for v in found] == [(_SLEEP_LINENO, call_col)], found

    @pytest.mark.parametrize(
        'statement',
        [
            'await asyncio.sleep(0)',
            'time.sleep(0.0)',
            'await asyncio.sleep(60)',
            'await asyncio.sleep(3600)',
            "await asyncio.sleep(float('inf'))",
            'await asyncio.sleep(math.inf)',
            'fake_clock.sleep(5)',
            'self.sleep(1)',
            'loop.call_soon(time.sleep, 0.1)',
        ],
    )
    def test_a_yield_a_hang_simulator_or_another_receiver_is_not_a_site(self, statement: str):
        assert _scan(_in_test(statement)) == []


class TestSleepMessage:
    @pytest.fixture
    def message(self) -> str:
        [violation] = _scan(_in_test('time.sleep(0.1)'))
        return violation.message

    def test_names_the_rule_and_the_call_and_no_other_rules_vocabulary(self, message: str):
        _assert_owns_its_vocabulary(message)
        assert 'time.sleep(0.1)' in message

    @pytest.mark.parametrize(
        'remedy',
        [
            'event',
            'polled predicate',
            'shared/src/shared/pytest_fake_clock.py::fake_clock',
            'shared/src/shared/testing_virtual_clock.py::virtual_clock_test',
            '@pytest.mark.timing(reason=',
            'behaviour under test',
        ],
    )
    def test_names_each_remedy(self, message: str, remedy: str):
        assert remedy in message

    def test_states_the_hang_simulator_floor(self, message: str):
        assert '60 s' in message
