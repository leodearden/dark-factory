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


def _assert_positions(source: str) -> list[tuple[int, int]]:
    tree = ast.parse(textwrap.dedent(source))
    return sorted((node.lineno, node.col_offset) for node in ast.walk(tree) if isinstance(node, ast.Assert))


_ELAPSED_FLAGGED = {
    'direct': """
        import time

        def test_case():
            t0 = time.monotonic()
            assert time.monotonic() - t0 < 0.5
        """,
    'elapsed_binding': """
        import time

        def test_case(start):
            elapsed = time.monotonic() - start
            assert elapsed < 2
        """,
    'clock_bindings': """
        import time

        def test_case():
            start = time.perf_counter()
            end = time.perf_counter()
            assert end - start < 1.0
        """,
    'rounded_loop_read': """
        def test_case(loop, t0):
            took = round(loop.time() - t0, 2)
            assert took >= 0.1
        """,
    'chained': """
        import time

        def test_case(t0):
            elapsed = time.monotonic() - t0
            assert 0.4 < elapsed < 0.6
        """,
    'nested_async_scope': """
        import asyncio

        def test_case():
            async def scenario(work):
                t0 = asyncio.get_running_loop().time()
                await work()
                elapsed = asyncio.get_running_loop().time() - t0
                assert elapsed < 1
        """,
    'attribute_loop_receiver': """
        class TestCase:
            def test_case(self, t0):
                elapsed = self._loop.time() - t0
                assert elapsed <= 3
        """,
    'nanosecond_read': """
        import time

        def test_case(t0):
            assert time.time_ns() - t0 < 5_000_000
        """,
}

_ELAPSED_NOT_FLAGGED = {
    'fake_clock_read': """
        def test_case(fake_clock, t0):
            elapsed = fake_clock.monotonic() - t0
            assert elapsed == 5
        """,
    'no_literal': """
        import time

        def test_case(t0, budget):
            elapsed = time.monotonic() - t0
            assert elapsed < budget * 2
        """,
    'not_a_numeric_comparison': """
        import time

        def test_case(t0):
            elapsed = time.monotonic() - t0
            assert elapsed is not None
        """,
    'bound_in_another_function': """
        import time

        def helper(t0):
            elapsed = time.monotonic() - t0
            return elapsed

        def test_case(elapsed):
            assert elapsed < 2
        """,
    'no_subtraction': """
        import time

        def test_case():
            assert time.monotonic() > 0
        """,
    'no_clock_binding': """
        def test_case(x, y):
            assert x - y < 1
        """,
}


class TestElapsedAssertLeg:
    """An assert comparing a REAL elapsed time with a numeric literal, resolved in its own scope."""

    @pytest.mark.parametrize('source', _ELAPSED_FLAGGED.values(), ids=_ELAPSED_FLAGGED.keys())
    def test_one_violation_at_the_assert(self, source: str):
        found = _scan(source)
        assert [(v.lineno, v.col_offset) for v in found] == _assert_positions(source), found

    @pytest.mark.parametrize('source', _ELAPSED_NOT_FLAGGED.values(), ids=_ELAPSED_NOT_FLAGGED.keys())
    def test_not_a_real_elapsed_time_against_a_literal(self, source: str):
        assert _scan(source) == []


class TestElapsedAssertMessage:
    @pytest.fixture
    def message(self) -> str:
        [violation] = _scan(_ELAPSED_FLAGGED['chained'])
        return violation.message

    def test_names_the_rule_and_the_comparison_and_no_other_rules_vocabulary(self, message: str):
        _assert_owns_its_vocabulary(message)
        assert '0.4 < elapsed < 0.6' in message

    @pytest.mark.parametrize(
        'needle',
        [
            'merge gate',
            'shared/src/shared/pytest_fake_clock.py::fake_clock',
            'shared/src/shared/testing_virtual_clock.py::virtual_clock_test',
            '@pytest.mark.timing(reason=',
        ],
    )
    def test_names_why_and_each_remedy(self, message: str, needle: str):
        assert needle in message


_DEADLINE_PRELUDE = (
    'import asyncio\n'
    'import subprocess\n'
    'import time\n'
    '\n'
    '\n'
    'async def test_case(client, task, cmd, thread, proc, pred, loop, fake_clock, pid, done):\n'
)
_DEADLINE_LINENO = _DEADLINE_PRELUDE.count('\n') + 1


def _in_async_test(statement: str) -> str:
    return _DEADLINE_PRELUDE + f'    {statement}\n'


def _deadline_kind_sites(facts) -> list:
    return [
        fact
        for fact in facts
        if isinstance(fact, _timing.TimingSite) and fact.kind is _timing.TimingKind.LITERAL_DEADLINE
    ]


class TestLiteralDeadlineLeg:
    """A numeric literal bound to a wall-clock DEADLINE PARAMETER, keyed structurally, never by callee."""

    @pytest.mark.parametrize(
        ('statement', 'anchor'),
        [
            ('await asyncio.wait_for(client.fetch(), timeout=5)', 'asyncio.wait_for'),
            ('await asyncio.wait_for(task, 2.5)', 'asyncio.wait_for'),
            ('subprocess.run(cmd, timeout=30)', 'subprocess.run'),
            ('thread.join(timeout=1.0)', 'thread.join'),
            ('proc.wait(timeout=10)', 'proc.wait'),
            ('await poll_until(pred, timeout=5)', 'poll_until'),
            ('async with asyncio.timeout(3):\n        await client.fetch()', 'async with'),
            ('deadline = time.monotonic() + 10.0', 'time.monotonic'),
            ('deadline = loop.time() + 60', 'loop.time'),
            ('deadline = 5 + asyncio.get_running_loop().time()', '5 +'),
        ],
    )
    def test_one_violation_at_the_deadline(self, statement: str, anchor: str):
        found = _scan(_in_async_test(statement))
        expected = (_DEADLINE_LINENO, 4 + statement.index(anchor))
        assert [(v.lineno, v.col_offset) for v in found] == [expected], found

    @pytest.mark.parametrize(
        'statement',
        [
            'await asyncio.wait_for(client.fetch(), timeout=0)',
            'await asyncio.wait_for(client.fetch(), timeout=None)',
            'await asyncio.wait_for(client.fetch(), timeout=TIMEOUT)',
            'raise subprocess.TimeoutExpired(cmd, timeout=5)',
            'client = _FakeFalkorClient(timeout=3)',
            'mark = pytest.mark.timeout(timeout=60)',
            'deadline = fake_clock.monotonic() + 5',
            'async with asyncio.timeout(None):\n        await client.fetch()',
            'run_on_own_time(cmd, budget=5.0)',
            'wait_on_own_time(pid, done, budget=5.0)',
        ],
    )
    def test_not_a_literal_wall_clock_deadline(self, statement: str):
        assert _scan(_in_async_test(statement)) == []

    def test_a_pytest_timeout_mark_decorator_is_not_a_deadline(self):
        source = """
            import pytest

            @pytest.mark.timeout(timeout=60)
            def test_case():
                pass
            """
        assert _scan(source) == []


class TestLiteralDeadlinePrecedence:
    """A node another rule already reported is not a second Rule D deadline site."""

    @staticmethod
    def _facts(source: str, node_type: type, *, already_reported: bool) -> list:
        tree = ast.parse(textwrap.dedent(source))
        [node] = [
            node
            for node in ast.walk(tree)
            if isinstance(node, node_type)
            and (node_type is not ast.Call or ast.unparse(node).startswith('asyncio.wait_for'))
        ]
        return list(_timing.timing_facts(node, deadline_already_reported=already_reported))

    @pytest.mark.parametrize(
        ('statement', 'node_type'),
        [
            ('await asyncio.wait_for(client.fetch(), timeout=5)', ast.Call),
            ('async with asyncio.timeout(3):\n        await client.fetch()', ast.AsyncWith),
        ],
    )
    def test_an_already_reported_deadline_node_yields_no_deadline_site(self, statement, node_type):
        source = _in_async_test(statement)
        assert _deadline_kind_sites(self._facts(source, node_type, already_reported=False))
        assert _deadline_kind_sites(self._facts(source, node_type, already_reported=True)) == []

    @pytest.mark.parametrize(
        'statement',
        [
            'await asyncio.wait_for(asyncio.sleep(0.1), timeout=5)',
            'async with asyncio.timeout(3):\n        await asyncio.sleep(0.1)',
        ],
    )
    def test_a_sleep_inside_an_already_reported_deadline_is_still_a_site(self, statement: str):
        tree = ast.parse(_in_async_test(statement))
        facts = [
            fact
            for node in ast.walk(tree)
            for fact in _timing.timing_facts(node, deadline_already_reported=True)
        ]
        sites = _timing.uncovered_timing_sites(facts)
        assert [site.kind for site in sites] == [_timing.TimingKind.SLEEP]


class TestLiteralDeadlineMessage:
    @pytest.mark.parametrize(
        ('statement', 'site_text'),
        [
            ('subprocess.run(cmd, timeout=30)', 'subprocess.run(cmd, timeout=30)'),
            ('async with asyncio.timeout(3):\n        await client.fetch()', 'asyncio.timeout(3)'),
            ('deadline = time.monotonic() + 10.0', 'time.monotonic() + 10.0'),
        ],
    )
    def test_names_the_rule_and_the_site_and_no_other_rules_vocabulary(self, statement, site_text):
        [violation] = _scan(_in_async_test(statement))
        _assert_owns_its_vocabulary(violation.message)
        assert site_text in violation.message

    @pytest.mark.parametrize(
        'remedy',
        [
            'load-tolerant named constant',
            'shared/src/shared/testing_virtual_clock.py::virtual_clock_test',
            'shared/src/shared/testing_own_time.py::run_on_own_time',
            'subprocess.run',
            '::wait_on_own_time',
            'Popen',
            '@pytest.mark.timing(reason=',
            'behaviour under test',
        ],
    )
    def test_names_each_remedy(self, remedy: str):
        [violation] = _scan(_in_async_test('thread.join(timeout=1.0)'))
        assert remedy in violation.message


_COVERAGE_PRELUDE = 'import asyncio\nimport time\n\nimport pytest\n\n\n'
_REASON = "reason='the 5 s stop grace is the behaviour under test'"
_ALL_KINDS_BODY = (
    'time.sleep(0.1)\n'
    'await asyncio.sleep(0.1)\n'
    'elapsed = time.monotonic() - t0\n'
    'assert elapsed < 2\n'
    'await asyncio.wait_for(fut, timeout=5)\n'
)
_ALL_KINDS_SITE_COUNT = 4
_VIRTUAL_LOOP_BODY = _ALL_KINDS_BODY.replace('time.sleep(0.1)\n', '')


def _coverage_source(header: str, body: str = _ALL_KINDS_BODY, *, method: bool = False) -> str:
    """*header* then a test holding *body*: a module function, or a method when *method*."""
    if method:
        return _COVERAGE_PRELUDE + header + '    async def test_case(self, t0, fut):\n' + textwrap.indent(body, ' ' * 8)
    return _COVERAGE_PRELUDE + header + 'async def test_case(t0, fut):\n' + textwrap.indent(body, ' ' * 4)


_COVERED = {
    'marker_on_the_test': _coverage_source(f'@pytest.mark.timing({_REASON})\n'),
    'marker_on_the_class': _coverage_source(f'@pytest.mark.timing({_REASON})\nclass TestCase:\n', method=True),
    'marker_on_a_module_helper': _coverage_source(f'@pytest.mark.timing({_REASON})\n').replace(
        'def test_case', 'def _helper'
    ),
    'marker_on_a_fake_class': _coverage_source(
        f'@pytest.mark.timing({_REASON})\nclass _FakeWorker:\n', method=True
    ).replace('def test_case', 'def run'),
    'module_pytestmark': _coverage_source(f'pytestmark = pytest.mark.timing({_REASON})\n\n\n'),
    'module_pytestmark_list': _coverage_source(
        f'pytestmark = [pytest.mark.slow, pytest.mark.timing({_REASON})]\n\n\n'
    ),
    'virtual_clock_test': _coverage_source(
        'from shared.testing_virtual_clock import virtual_clock_test\n\n\n@virtual_clock_test\n',
        _VIRTUAL_LOOP_BODY,
    ),
    'virtual_clock_test_attribute': _coverage_source(
        'from shared import testing_virtual_clock\n\n\n@testing_virtual_clock.virtual_clock_test\n',
        _VIRTUAL_LOOP_BODY,
    ),
}

_NOT_COVERED = {
    'bare_marker': _coverage_source('@pytest.mark.timing\n'),
    'no_reason': _coverage_source('@pytest.mark.timing()\n'),
    'empty_reason': _coverage_source("@pytest.mark.timing(reason='')\n"),
    'blank_reason': _coverage_source("@pytest.mark.timing(reason='   ')\n"),
    'non_literal_reason': _coverage_source('@pytest.mark.timing(reason=REASON)\n'),
    'positional_reason': _coverage_source("@pytest.mark.timing('x')\n"),
    'marker_on_a_sibling': _coverage_source(
        f'@pytest.mark.timing({_REASON})\ndef test_sibling():\n    pass\n\n\n'
    ),
}


class TestCoverage:
    """Only ``@pytest.mark.timing(reason='<non-blank literal>')`` or a virtual clock excuses a site."""

    def test_the_uncovered_control_carries_one_site_per_statement(self):
        assert len(_scan(_coverage_source(''))) == _ALL_KINDS_SITE_COUNT

    @pytest.mark.parametrize('source', _COVERED.values(), ids=_COVERED.keys())
    def test_a_valid_marker_or_a_virtual_clock_covers_every_site(self, source: str):
        assert _scan(source) == []

    def test_a_class_body_pytestmark_covers_only_that_class(self):
        covered = _coverage_source(
            f'class TestCovered:\n    pytestmark = pytest.mark.timing({_REASON})\n\n', method=True
        )
        uncovered = _coverage_source('class TestUncovered:\n', method=True)
        source = covered + '\n\n' + uncovered.removeprefix(_COVERAGE_PRELUDE)
        uncovered_first_line = source.splitlines().index('class TestUncovered:') + 1
        found = _scan(source)
        assert len(found) == _ALL_KINDS_SITE_COUNT, found
        assert all(v.lineno > uncovered_first_line for v in found), found

    @pytest.mark.parametrize('source', _NOT_COVERED.values(), ids=_NOT_COVERED.keys())
    def test_an_invalid_or_misplaced_marker_covers_nothing(self, source: str):
        found = _scan(source)
        assert len(found) == _ALL_KINDS_SITE_COUNT, found
        for violation in found:
            assert 'the marker covers only with a non-empty literal reason=' in violation.message

    def test_a_virtual_clock_does_not_cover_a_real_thread_sleep(self):
        source = _coverage_source(
            'from shared.testing_virtual_clock import virtual_clock_test\n\n\n@virtual_clock_test\n'
        )
        [violation] = _scan(source)
        assert 'time.sleep(0.1)' in violation.message


_DEBT_FILE = 'orchestrator/tests/test_x.py'
_DEBT_KEY = f"TIMING_DEPENDENCE_DEBT['{_DEBT_FILE}']"
_BUDGET_OF_TWO = _lint_cli.DebtBaseline.from_mapping({_DEBT_FILE: 2})
_SLEEP_SITE = '    time.sleep(0.1)\n'
_COVERED_HELPER = (
    f'\n\n@pytest.mark.timing({_REASON})\ndef _covered_helper():\n{_SLEEP_SITE}'
)


def _sleeps(count: int, *, covered: int = 0) -> str:
    return (
        'import time\n\nimport pytest\n\n\ndef test_case():\n'
        + _SLEEP_SITE * count
        + '    pass\n'
        + _COVERED_HELPER * covered
    )


def _budgeted(source: str, filename: str = _DEBT_FILE):
    return _scan(source, filename=filename, budgets=_BUDGET_OF_TWO)


class TestTimingDependenceBudget:
    """Rule D's own shrink-only per-file budget, counted over UNCOVERED sites only."""

    def test_a_debt_file_at_its_budget_is_silent(self):
        assert _budgeted(_sleeps(2)) == []

    def test_an_overrun_reports_the_last_site(self):
        found = _budgeted(_sleeps(3))
        assert [(v.lineno, v.col_offset) for v in found] == [(9, 4)], found

    @pytest.mark.parametrize(
        'needle',
        [
            'recorded budget of 2',
            '3 were found',
            '1 over budget',
            'positional',
            'shared/src/shared/pytest_fake_clock.py::fake_clock',
            'shared/src/shared/testing_virtual_clock.py::virtual_clock_test',
            '@pytest.mark.timing(reason=',
            'Do NOT raise',
            'TIMING_DEPENDENCE_DEBT',
        ],
    )
    def test_the_overrun_message_names_the_budget_its_remedies_and_forbids_raising_it(self, needle):
        [violation] = _budgeted(_sleeps(3))
        _assert_owns_its_vocabulary(violation.message)
        assert needle in violation.message

    def test_slack_is_one_file_level_violation_naming_the_lowered_number(self):
        [violation] = _budgeted(_sleeps(1))
        assert (violation.lineno, violation.col_offset) == (1, 0)
        _assert_owns_its_vocabulary(violation.message)
        assert (
            f'lower {_DEBT_KEY} from 2 to 1 in fused-memory/scripts/_timing_dependence_debt.py'
            in violation.message
        )

    def test_a_fully_migrated_file_is_told_to_delete_its_entry(self):
        [violation] = _budgeted(_sleeps(0))
        _assert_owns_its_vocabulary(violation.message)
        assert f'delete {_DEBT_KEY}' in violation.message
        assert 'fused-memory/tests/test_fused_memory_governed_lists.py::TIMING_DEPENDENCE_BASELINE' in (
            violation.message
        )

    def test_a_non_debt_file_reports_every_site(self):
        assert len(_budgeted(_sleeps(3), filename=_NEW_FILE)) == 3

    def test_a_covered_site_neither_fills_the_budget_nor_hides_an_overrun(self):
        assert _budgeted(_sleeps(2, covered=1)) == []
        [slack] = _budgeted(_sleeps(1, covered=1))
        assert (slack.lineno, slack.col_offset) == (1, 0)

    def test_the_live_budgets_index_exactly_the_debt_mapping(self):
        debt = load_script_module(_SCRIPTS / '_timing_dependence_debt.py').TIMING_DEPENDENCE_DEBT
        rebuilt = _lint_cli.DebtBaseline.from_mapping(debt)
        for key in debt:
            assert _timing.TIMING_DEPENDENCE_BUDGETS.entry_for(key) == rebuilt.entry_for(key)
