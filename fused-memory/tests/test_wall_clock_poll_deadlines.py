"""Rule C (``wall-clock-deadline``) over the poll shapes of a wall-clock deadline.

fused-memory/scripts/check_bare_magicmock_config.py's Rule C flags a load-bearing
wait whose deadline is charged in wall clock. This module pins the shapes beyond
the ``asyncio.wait_for`` / ``wait_responsive`` call leg: an async ``while`` poll
that gives up against a wall-clock read, an ``asyncio.timeout`` block, and
``poll_responsive`` as a call shape of its own. Every case drives the public
``find_violations`` over a synthetic source.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'check_bare_magicmock_config.py'
_checker = load_script_module(SCRIPT_PATH, mod_name='check_bare_magicmock_config')

_NOQA = '# noqa: wall-clock-deadline — deliberate, for the test'


def _rule_c(source: str, filename: str = 'orchestrator/tests/test_brand_new.py') -> list:
    """Return only the Rule C violations find_violations reports for *source*."""
    return [
        v
        for v in _checker.find_violations(textwrap.dedent(source), filename)
        if 'wall-clock-deadline' in v.message
    ]


def _line_of(source: str, needle: str) -> int:
    """1-based line number of the first line of dedented *source* containing *needle*."""
    for lineno, line in enumerate(textwrap.dedent(source).splitlines(), start=1):
        if needle in line:
            return lineno
    raise AssertionError(f'{needle!r} not in source')


_LOOP_TIME_POLL = """
    async def test_x():
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 60
        while loop.time() < deadline:
            await asyncio.sleep(0.05)
"""


class TestClockDeadlinePollLeg:
    def test_a_loop_time_deadline_poll_flags_once_at_the_while_line(self):
        violations = _rule_c(_LOOP_TIME_POLL)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(_LOOP_TIME_POLL, 'while loop.time()')

    @pytest.mark.parametrize(
        'clock',
        [
            'asyncio.get_running_loop().time()',
            'time.monotonic()',
            'time.time()',
            'time.perf_counter()',
        ],
    )
    def test_every_wall_clock_receiver_flags_identically(self, clock):
        source = f"""
            async def test_x():
                deadline = {clock} + 60
                while {clock} < deadline:
                    await asyncio.sleep(0.05)
        """

        assert len(_rule_c(source)) == 1, f'{clock} deadline poll must flag once'

    def test_a_compound_while_test_flags(self):
        source = """
            async def test_x():
                deadline = loop.time() + 60
                while not done() and loop.time() < deadline:
                    await asyncio.sleep(0.05)
        """

        assert len(_rule_c(source)) == 1

    def test_the_comparison_in_the_body_flags(self):
        source = """
            async def poll_until():
                deadline = loop.time() + 5
                while True:
                    if loop.time() > deadline:
                        pytest.fail('x')
                    await asyncio.sleep(0.01)
        """

        violations = _rule_c(source)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(source, 'while True')

    @pytest.mark.parametrize(
        'source',
        [
            pytest.param(
                """
                def test_x():
                    while time.monotonic() < d:
                        time.sleep(0.1)
                """,
                id='sync-loop',
            ),
            pytest.param(
                """
                async def fake_worker():
                    t0 = loop.time()
                    while loop.time() - t0 < duration:
                        await asyncio.sleep(tick)
                """,
                id='elapsed-duration',
            ),
            pytest.param(
                """
                async def test_x():
                    while not ready():
                        await asyncio.sleep(0.01)
                """,
                id='no-clock',
            ),
            pytest.param(
                """
                async def test_x():
                    while loop.time() < deadline:
                        async def _inner():
                            await asyncio.sleep(0.01)
                        spawn(_inner)
                """,
                id='await-only-in-nested-scope',
            ),
        ],
    )
    def test_shapes_that_are_not_a_clock_deadline_poll_stay_silent(self, source):
        assert _rule_c(source) == []

    def test_a_nested_poll_flags_once_at_the_inner_while(self):
        source = """
            async def test_x():
                while True:
                    deadline = loop.time() + 5
                    while loop.time() < deadline:
                        await asyncio.sleep(0.01)
                    await step()
        """

        violations = _rule_c(source)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(source, 'while loop.time()')

    def test_a_noqa_above_the_while_suppresses_it(self):
        source = f"""
            async def test_x():
                deadline = loop.time() + 60
                {_NOQA}
                while loop.time() < deadline:
                    await asyncio.sleep(0.05)
        """

        assert _rule_c(source) == []

    def test_the_message_names_the_poll_remedy_and_no_other_rule(self):
        [violation] = _rule_c(_LOOP_TIME_POLL)
        message = violation.message

        for phrase in (
            'WALL CLOCK',
            'poll_responsive',
            'orchestrator/tests/_poll_responsive.py::poll_responsive',
            'awaited event did happen',
            'importable only under orchestrator/tests',
            '# noqa: wall-clock-deadline',
        ):
            assert phrase in message, f'{phrase!r} missing from: {message}'
        for foreign in ('mock_orch_config', 'fake_verify_result', 'bare asyncio.wait_for'):
            assert foreign not in message, f'{foreign!r} must not appear in: {message}'
