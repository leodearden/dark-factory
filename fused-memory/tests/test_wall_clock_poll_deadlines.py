"""Rule C (``wall-clock-deadline``) over the poll shapes of a wall-clock deadline.

fused-memory/scripts/check_bare_magicmock_config.py's Rule C flags a load-bearing
wait whose deadline is charged in wall clock. This module pins the shapes beyond
the ``asyncio.wait_for`` / ``wait_responsive`` call leg: an async ``while`` poll
that gives up against a wall-clock read, an ``asyncio.timeout`` block, and
``poll_responsive`` as a call shape of its own. Every case drives the public
``find_violations`` over a synthetic source.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_REPO_ROOT = Path(__file__).parent.parent.parent
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
                async def fake_worker():
                    t0 = loop.time()
                    while True:
                        if loop.time() - t0 >= duration:
                            break
                        await asyncio.sleep(tick)
                """,
                id='elapsed-duration-whose-only-break-is-the-clock',
            ),
            pytest.param(
                """
                async def fake_worker():
                    t0 = loop.time()
                    while loop.time() - t0 < duration:
                        for item in items:
                            if item.done:
                                break
                        await asyncio.sleep(tick)
                """,
                id='elapsed-duration-break-ends-only-a-nested-loop',
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

    @pytest.mark.parametrize(
        'source',
        [
            pytest.param(
                """
                async def test_x():
                    t0 = loop.time()
                    while not ready() and loop.time() - t0 < timeout:
                        await asyncio.sleep(0.01)
                """,
                id='a-predicate-conjunct',
            ),
            pytest.param(
                """
                async def test_x():
                    t0 = loop.time()
                    while True:
                        if loop.time() - t0 > timeout:
                            pytest.fail('x')
                        if ready():
                            break
                        await asyncio.sleep(0.01)
                """,
                id='a-break-outside-the-clock-check',
            ),
            pytest.param(
                """
                async def test_x():
                    t0 = loop.time()
                    while True:
                        if loop.time() - t0 > timeout:
                            pytest.fail('x')
                        elif ready():
                            return
                        await asyncio.sleep(0.01)
                """,
                id='a-return-in-the-clock-checks-else-arm',
            ),
        ],
    )
    def test_an_elapsed_time_poll_with_another_exit_flags_once(self, source):
        """`loop.time() - t0 < timeout` is the one-line rewrite of `loop.time() < deadline`.

        It counts as a poll's give-up when the loop can also end some other way,
        so it cannot evade the leg the direct spelling trips.
        """
        violations = _rule_c(source)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(source, 'while ')

    @pytest.mark.parametrize(
        'source',
        [
            pytest.param(
                """
                async def test_x():
                    while deadline - loop.time() > 0:
                        await asyncio.sleep(0.01)
                """,
                id='in-the-test',
            ),
            pytest.param(
                """
                async def test_x():
                    while True:
                        if deadline - time.monotonic() <= 0:
                            pytest.fail('x')
                        await asyncio.sleep(0.01)
                """,
                id='in-the-body',
            ),
        ],
    )
    def test_a_remaining_time_operand_flags_like_a_direct_read(self, source):
        """`deadline - loop.time() > 0` is the one-line rewrite of `loop.time() < deadline`.

        A remaining time is always measured against a deadline, so unlike an
        elapsed time it needs no other exit to count as a poll's give-up.
        """
        violations = _rule_c(source)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(source, 'while ')

    def test_a_clock_check_in_a_for_loop_belongs_to_the_enclosing_while(self):
        source = """
            async def test_x():
                while True:
                    for probe in probes:
                        if loop.time() > deadline:
                            pytest.fail('x')
                    await asyncio.sleep(0.01)
        """

        violations = _rule_c(source)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(source, 'while True')

    def test_a_for_loop_is_not_a_leg_of_its_own(self):
        """Deliberate negative space: a top-level `for` clock poll is silent.

        A `for` loop is searched as part of its enclosing `while` (above), so making
        it a leg too would count one poll twice.  Zero clock-deadline `for` polls
        were measured in the scanned tree on 2026-11-02 (task 7800).  Widening the
        leg to `for` loops must be a deliberate change to this pin.
        """
        source = """
            async def test_x():
                deadline = loop.time() + 5
                for _ in range(1000):
                    if loop.time() > deadline:
                        break
                    await asyncio.sleep(0.01)
        """

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


_PARKED_VERIFY_POLL = """
    async def _await_parked_verify(verifier, task_id, timeout):
        async with asyncio.timeout(timeout):
            while task_id not in verifier.verified:
                await asyncio.sleep(0.01)
"""

_RULE_A_AND_B_VOCABULARY = (
    'mock_orch_config',
    'pydantic_spec',
    'fake_verify_result',
    'spec=VerifyResult',
    'bare-magicmock',
    'bare-dataclass-double',
)


def _timeout_block(context: str, body: str) -> str:
    """An async test whose *body* lines run inside ``async with <context>:``."""
    inner = ''.join(f'        {line}\n' for line in body.splitlines())
    return f'async def test_x():\n    async with {context}:\n{inner}'


_POLL_BODY = 'while not ready():\n    await asyncio.sleep(0.01)'


class TestTimeoutBlockLeg:
    def test_a_named_delay_around_a_poll_flags_routing_only_at_the_block(self):
        violations = _rule_c(_PARKED_VERIFY_POLL)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(_PARKED_VERIFY_POLL, 'async with')

    def test_a_literal_delay_adds_the_raw_literal_kind(self):
        violations = _rule_c(_timeout_block('asyncio.timeout(5)', _POLL_BODY))

        assert len(violations) == 2, violations
        literal = [v.message for v in violations if 'RAW wall-clock literal' in v.message]
        assert len(literal) == 1, violations
        assert 'delay' in literal[0], literal[0]
        assert 'Derive the bound from MERGE_RESULT_TIMEOUT' in literal[0], literal[0]

    def test_the_keyword_delay_spelling_flags_identically(self):
        assert len(_rule_c(_timeout_block('asyncio.timeout(delay=5)', _POLL_BODY))) == 2

    def test_a_load_bearing_await_with_no_poll_flags(self):
        barrier = _rule_c(_timeout_block('asyncio.timeout(5)', 'await done.wait()'))
        future = _rule_c(_timeout_block('asyncio.timeout(t)', 'await req.result'))

        assert len(barrier) == 2, barrier
        assert len(future) == 1, future
        for v in barrier:
            assert 'done.wait() (.wait() barrier)' in v.message, v.message

    @pytest.mark.parametrize(
        ('context', 'body'),
        [
            pytest.param('asyncio.timeout(5)', 'await task', id='bare-name-join'),
            pytest.param(
                'asyncio.timeout(30)', 'result = await run_verification(x)', id='plain-call'
            ),
            pytest.param('asyncio.timeout(5)', 'await asyncio.gather(a, b)', id='gather'),
            pytest.param('asyncio.timeout(None)', _POLL_BODY, id='no-deadline'),
            pytest.param('other.timeout(5)', _POLL_BODY, id='not-asyncio'),
        ],
    )
    def test_blocks_that_bound_no_load_bearing_wait_stay_silent(self, context, body):
        assert _rule_c(_timeout_block(context, body)) == []

    def test_a_multi_item_block_flags_its_deadline_item(self):
        assert len(_rule_c(_timeout_block('asyncio.timeout(5), slot()', _POLL_BODY))) == 2

    @pytest.mark.parametrize(
        'context', ['asyncio.timeout_at(when)', 'asyncio.timeout_at(when=deadline)']
    )
    def test_an_absolute_deadline_flags_routing_only(self, context):
        violations = _rule_c(_timeout_block(context, _POLL_BODY))

        assert len(violations) == 1, violations
        assert 'RAW wall-clock literal' not in violations[0].message

    @pytest.mark.parametrize(
        'context', ['asyncio.timeout(when=5)', 'asyncio.timeout_at(delay=5)']
    )
    def test_the_other_contexts_keyword_is_not_a_deadline(self, context):
        """Each context's argument is read by its own keyword: the other is a TypeError."""
        assert _rule_c(_timeout_block(context, _POLL_BODY)) == []

    def test_one_noqa_above_the_block_suppresses_both_kinds(self):
        source = f'async def test_x():\n    {_NOQA}\n' + _timeout_block(
            'asyncio.timeout(5)', _POLL_BODY
        ).removeprefix('async def test_x():\n')

        assert _rule_c(source) == []

    def test_the_routing_message_names_both_remedies_and_no_other_rule(self):
        [violation] = _rule_c(_PARKED_VERIFY_POLL)
        message = violation.message

        for phrase in (
            'asyncio.timeout',
            'WALL CLOCK',
            'wait_responsive',
            'poll_responsive',
            'importable only under orchestrator/tests',
            '# noqa: wall-clock-deadline',
        ):
            assert phrase in message, f'{phrase!r} missing from: {message}'
        for foreign in (*_RULE_A_AND_B_VOCABULARY, 'bare asyncio.wait_for'):
            assert foreign not in message, f'{foreign!r} must not appear in: {message}'


class TestNestedDeadlineSpellings:
    """The innermost deadline spelling owns the waits inside it: one wait counts once."""

    def test_nested_timeout_blocks_count_the_wait_once_at_the_inner_block(self):
        source = """
            async def test_x():
                async with asyncio.timeout(5):
                    async with asyncio.timeout(3):
                        await ev.wait()
        """

        violations = _rule_c(source)

        assert len(violations) == 2, violations
        assert {v.lineno for v in violations} == {_line_of(source, 'asyncio.timeout(3)')}

    def test_a_clock_poll_inside_a_timeout_block_counts_once_at_the_while(self):
        source = """
            async def test_x():
                async with asyncio.timeout(t):
                    while loop.time() < deadline:
                        await asyncio.sleep(0.01)
        """

        violations = _rule_c(source)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(source, 'while loop.time()')

    def test_a_timeout_block_inside_a_clock_poll_counts_once_at_the_block(self):
        source = """
            async def test_x():
                while loop.time() < deadline:
                    async with asyncio.timeout(t):
                        await ev.wait()
        """

        violations = _rule_c(source)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(source, 'asyncio.timeout(t)')

    def test_the_outer_block_still_owns_a_wait_outside_the_inner_one(self):
        source = """
            async def test_x():
                async with asyncio.timeout(t):
                    async with asyncio.timeout(u):
                        await a.wait()
                    await b.wait()
        """

        violations = _rule_c(source)

        assert sorted(v.lineno for v in violations) == [
            _line_of(source, 'asyncio.timeout(t)'),
            _line_of(source, 'asyncio.timeout(u)'),
        ], violations

    def test_an_inner_block_with_no_deadline_owns_nothing(self):
        source = """
            async def test_x():
                async with asyncio.timeout(t):
                    async with asyncio.timeout(None):
                        await ev.wait()
        """

        violations = _rule_c(source)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(source, 'asyncio.timeout(t)')

    def test_a_per_attempt_block_that_flags_nothing_leaves_the_clock_poll_flagged(self):
        source = """
            async def test_x():
                while loop.time() < deadline:
                    async with asyncio.timeout(1):
                        if await probe():
                            break
        """

        violations = _rule_c(source)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(source, 'while loop.time()')

    def test_a_per_attempt_block_that_flags_nothing_leaves_the_outer_block_flagged(self):
        source = """
            async def test_x():
                async with asyncio.timeout(t):
                    while True:
                        async with asyncio.timeout(1):
                            if await probe():
                                break
        """

        violations = _rule_c(source)

        assert len(violations) == 1, violations
        assert violations[0].lineno == _line_of(source, 'asyncio.timeout(t)')


class TestResponsiveHelperRegistry:
    """The responsive helpers Rule C recognises and cites exist where it says they do."""

    def test_each_helper_is_a_top_level_async_def_at_its_registered_home(self):
        """Parsed, never imported: the homes are orchestrator test helpers.

        `RESPONSIVE_WAIT_HELPERS` is a stdlib-only literal copy, so a rename or a
        move would otherwise silently blind the call leg to that helper and leave
        every message citing a path that no longer resolves.
        """
        for name, helper in _checker.RESPONSIVE_WAIT_HELPERS.items():
            home = _REPO_ROOT / helper.home
            assert home.is_file(), (
                f'{name}: home {helper.home!r} is not a file; update '
                '`RESPONSIVE_WAIT_HELPERS` in fused-memory/scripts/check_bare_magicmock_config.py'
            )
            module = ast.parse(home.read_text(encoding='utf-8'))
            async_defs = {
                node.name for node in module.body if isinstance(node, ast.AsyncFunctionDef)
            }
            assert name in async_defs, (
                f'{name} is not a top-level async def of {helper.home!r}; update '
                '`RESPONSIVE_WAIT_HELPERS` in fused-memory/scripts/check_bare_magicmock_config.py'
            )

    @pytest.mark.parametrize(
        ('source', 'helper'),
        [
            pytest.param(
                'async def t():\n    await asyncio.wait_for(done.wait(), timeout=t)\n',
                'wait_responsive',
                id='bare-wait-for',
            ),
            pytest.param(_LOOP_TIME_POLL, 'poll_responsive', id='clock-deadline-poll'),
            pytest.param(_PARKED_VERIFY_POLL, 'poll_responsive', id='timeout-block'),
        ],
    )
    def test_each_routing_remedy_cites_its_helper_at_its_registered_home(self, source, helper):
        [violation] = _rule_c(source)
        citation = f'{_checker.RESPONSIVE_WAIT_HELPERS[helper].home}::{helper}'

        assert citation in violation.message, violation.message


class TestPollResponsiveCallShape:
    def test_a_literal_timeout_flags_the_raw_literal_kind_only(self):
        violations = _rule_c("poll_responsive(lambda: ready(), timeout=45.0, label='x')\n")

        assert len(violations) == 1, violations
        message = violations[0].message
        assert 'RAW wall-clock literal' in message, message
        assert 'timeout' in message, message
        assert 'polled predicate' in message, message
        assert 'bare asyncio.wait_for' not in message, message

    def test_a_literal_wall_cap_names_max_wall_s(self):
        violations = _rule_c("poll_responsive(pred, label='x', max_wall_s=30.0)\n")

        assert len(violations) == 1, violations
        assert 'max_wall_s' in violations[0].message

    @pytest.mark.parametrize(
        'source',
        [
            "poll_responsive(pred, label='x')\n",
            "poll_responsive(pred, timeout=MERGE_RESULT_TIMEOUT, label='x')\n",
        ],
    )
    def test_a_derived_or_default_bound_stays_silent(self, source):
        assert _rule_c(source) == []

    def test_a_noqa_above_a_literal_call_suppresses_it(self):
        source = f"{_NOQA}\npoll_responsive(pred, timeout=45.0, label='x')\n"

        assert _rule_c(source) == []

    def test_a_call_with_no_predicate_stays_silent(self):
        assert _rule_c('poll_responsive()\n') == []
