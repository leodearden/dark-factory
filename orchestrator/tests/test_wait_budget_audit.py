"""Unit tests for the AST wait-budget audit engine in _wait_budget_audit.py.

Every case drives the engine with a SYNTHETIC source string (and, where a
name must resolve, a literal namespace), never a real test module, so each
case is non-vacuous and does not shift when another file is edited.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
from collections.abc import Mapping

import pytest
from _wait_budget_audit import (
    AMBIENT_TEST_TIMEOUT,
    ItemWaitAudit,
    LimitSource,
    audit_module,
    is_module_level_item,
    item_wait_budgets,
)


class TestWaitBudgetAuditHelper:
    """Unit tests for `item_wait_budgets`, task 3492's audit engine -- see
    its docstring and `_method_wait_budget`'s in _wait_budget_audit.py for
    the per-item and shape-floor/branch-blind rationale; this class only
    pins the observable behaviour case by case.
    """

    def test_two_sequential_wait_for_calls_sum_within_one_method(self) -> None:
        """Two sequential `asyncio.wait_for(..., timeout=45.0)` calls in the
        SAME method sum to 90.0 -- pytest-timeout enforces the whole method
        body, so sequential waits stack.
        """
        source = '''
class TestTwoSequentialWaits:
    async def test_two_waits(self):
        await asyncio.wait_for(x, timeout=45.0)
        await asyncio.wait_for(y, timeout=45.0)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestTwoSequentialWaits::test_two_waits': 90.0}, (
            f'Expected two sequential 45.0s waits in one method to sum to '
            f'90.0, got {budgets!r}.'
        )

    def test_mutually_exclusive_branches_are_summed_not_maxed(self) -> None:
        """Two 45.0s waits in opposite arms of an `if`/`else` -- which can
        never both execute in a single real run -- sum to 90.0, not the
        45.0 a control-flow-aware count would give.

        This pins a deliberate, tolerated LIMITATION, not a design goal:
        `_method_wait_budget` walks the whole method body via `ast.walk`
        with no awareness of branches, so mutually exclusive arms over-count
        relative to any real execution path (see `_method_wait_budget`'s
        docstring). It is why
        `test_merge_queue_concurrent_verify.py::TestOverlapSignal`'s
        computed budget exceeds a control-flow-aware hand count over the
        same method's try/except paths.
        """
        source = '''
class TestBranchesSummed:
    async def test_branches(self, cond):
        if cond:
            await asyncio.wait_for(x, timeout=45.0)
        else:
            await asyncio.wait_for(y, timeout=45.0)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestBranchesSummed::test_branches': 90.0}, (
            f'Expected mutually exclusive if/else branches to be SUMMED '
            f'(90.0), not maxed (45.0) -- this pins the branch-blind '
            f'limitation documented on _method_wait_budget, not a design '
            f'goal; got {budgets!r}.'
        )

    def test_non_asyncio_wait_for_is_not_counted(self) -> None:
        """A `.wait_for(...)` call on something other than `asyncio` (e.g.
        `gate.wait_for(x, timeout=999.0)`) contributes 0.0 -- the match is
        deliberately narrowed to `asyncio.wait_for` specifically, so an
        unrelated same-named method can't be mistaken for the real thing.
        """
        source = '''
class TestNonAsyncioWaitFor:
    async def test_it(self, gate):
        await gate.wait_for(x, timeout=999.0)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestNonAsyncioWaitFor::test_it': 0.0}, (
            f'Expected a non-asyncio .wait_for(...) call to contribute 0.0, '
            f'got {budgets!r}.'
        )

    def test_each_method_is_billed_separately_not_class_wide_sum(self) -> None:
        """A class with TWO methods, each with one 45.0 wait, bills each
        method at its own 45.0 -- NOT a 90.0 class-wide sum.

        This is the distinction that makes the audit correct: pytest-timeout
        applies the class mark to each method independently. Summing across
        methods would misidentify a class like
        `test_merge_queue_concurrent_verify.py::TestRunInflightVerifyAbortPoll`
        (four 45s methods, a 180s class-wide sum but only 45s per method)
        as a false-positive breach.
        """
        source = '''
class TestTwoLightMethods:
    async def test_one(self):
        await asyncio.wait_for(x, timeout=45.0)

    async def test_two(self):
        await asyncio.wait_for(y, timeout=45.0)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {
            'TestTwoLightMethods::test_one': 45.0,
            'TestTwoLightMethods::test_two': 45.0,
        }, (
            f'Expected each method billed at its own 45.0, not the class-wide '
            f'sum (90.0); got {budgets!r}. Summing across methods would '
            f'misidentify the breaching set the guard is meant to compute.'
        )

    def test_bare_await_outcome_contributes_hidden_default_budget(self) -> None:
        """A bare `_await_outcome(req, label='x')` call with no explicit
        `timeout=` contributes the hidden default, MERGE_RESULT_TIMEOUT
        (45.0) -- invisible to a naive `timeout=` grep, which is exactly
        why the guard must special-case this call shape.
        """
        source = '''
class TestBareAwaitOutcome:
    async def test_bare_default(self):
        await _await_outcome(req, label='x')
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestBareAwaitOutcome::test_bare_default': 45.0}, (
            f'Expected a bare _await_outcome() call to contribute the '
            f'hidden MERGE_RESULT_TIMEOUT default (45.0), got {budgets!r}.'
        )

    def test_await_outcome_explicit_timeout_overrides_default(self) -> None:
        """`_await_outcome(req, label='x', timeout=10.0)` contributes its
        explicit 10.0, not the 45.0 default -- proves the explicit
        `timeout=` kwarg is read, not just assumed.
        """
        source = '''
class TestExplicitAwaitOutcome:
    async def test_explicit_timeout(self):
        await _await_outcome(req, label='x', timeout=10.0)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestExplicitAwaitOutcome::test_explicit_timeout': 10.0}, (
            f'Expected an explicit timeout=10.0 kwarg to override the '
            f'hidden default, got {budgets!r}.'
        )

    def test_dynamic_timeout_value_is_unbounded_and_does_not_raise(self) -> None:
        """A non-literal/dynamic timeout such as `timeout=some_var` bills
        `math.inf` and does NOT raise: the scan cannot read the value, so
        the wait may run for any wall clock, and no mark clears that.
        """
        source = '''
class TestDynamicTimeout:
    async def test_dynamic(self):
        await asyncio.wait_for(x, timeout=some_var)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestDynamicTimeout::test_dynamic': math.inf}, (
            f'Expected an unresolvable dynamic timeout to bill math.inf '
            f'without raising, got {budgets!r}.'
        )

    def test_wait_for_positional_dynamic_timeout_is_unbounded(self) -> None:
        """The positional spelling `asyncio.wait_for(x, some_var)` is the
        same unreadable wait, so it bills `math.inf` too.
        """
        source = '''
class TestPositionalDynamicTimeout:
    async def test_dynamic(self):
        await asyncio.wait_for(x, some_var)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestPositionalDynamicTimeout::test_dynamic': math.inf}, (
            f'Expected an unresolvable positional timeout to bill math.inf, '
            f'got {budgets!r}.'
        )

    def test_wait_for_none_timeout_is_unbounded(self) -> None:
        """`asyncio.wait_for(x, timeout=None)` waits forever, so it bills
        `math.inf`, not the 0.0 a literal zero would.
        """
        source = '''
class TestNoneTimeout:
    async def test_forever(self):
        await asyncio.wait_for(x, timeout=None)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestNoneTimeout::test_forever': math.inf}, (
            f'Expected timeout=None to bill math.inf, got {budgets!r}.'
        )

    def test_await_outcome_unresolvable_timeout_is_unbounded(self) -> None:
        """An explicit `_await_outcome(..., timeout=some_var)` replaces the
        hidden default with a value the scan cannot read, so it bills
        `math.inf`.
        """
        source = '''
class TestAwaitOutcomeUnknownTimeout:
    async def test_it(self):
        await _await_outcome(req, label='x', timeout=some_var)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestAwaitOutcomeUnknownTimeout::test_it': math.inf}, (
            f'Expected an unresolvable _await_outcome timeout to bill '
            f'math.inf, got {budgets!r}.'
        )

    def test_merge_result_timeout_name_resolves_to_45(self) -> None:
        """`timeout=MERGE_RESULT_TIMEOUT` resolves the name reference to
        its known numeric value, 45.0.
        """
        source = '''
class TestMergeResultTimeoutName:
    async def test_it(self):
        await asyncio.wait_for(x, timeout=MERGE_RESULT_TIMEOUT)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestMergeResultTimeoutName::test_it': 45.0}, (
            f'Expected timeout=MERGE_RESULT_TIMEOUT to resolve to 45.0, '
            f'got {budgets!r}.'
        )

    def test_heavy_barrier_test_timeout_name_resolves_to_its_value(self) -> None:
        """`timeout=HEAVY_BARRIER_TEST_TIMEOUT` resolves the name reference
        through the scanned module's namespace to the constant's numeric
        value rather than billing it as unbounded.
        """
        source = '''
class TestHeavyBarrierTimeoutName:
    async def test_it(self):
        await asyncio.wait_for(x, timeout=HEAVY_BARRIER_TEST_TIMEOUT)
'''
        budgets = item_wait_budgets(
            source,
            module_constants={'HEAVY_BARRIER_TEST_TIMEOUT': 540},
        )

        assert math.isfinite(budgets['TestHeavyBarrierTimeoutName::test_it']), (
            f'Expected timeout=HEAVY_BARRIER_TEST_TIMEOUT to resolve to a '
            f'finite budget, got {budgets!r}: the name was billed as unbounded.'
        )
        assert budgets == {'TestHeavyBarrierTimeoutName::test_it': 540.0}, (
            f'Expected timeout=HEAVY_BARRIER_TEST_TIMEOUT to resolve to '
            f'540.0, got {budgets!r}.'
        )

    def test_module_constant_name_resolves_from_the_namespace(self) -> None:
        """A module-local constant the static table cannot know resolves
        through the scanned module's namespace: `timeout=_SETTLE` with
        `_SETTLE = 30.0` bills 30.0, not unbounded.
        """
        source = '''
class TestModuleConstant:
    async def test_it(self):
        await asyncio.wait_for(x, timeout=_SETTLE)
'''
        budgets = item_wait_budgets(source, module_constants={'_SETTLE': 30.0})

        assert budgets == {'TestModuleConstant::test_it': 30.0}, (
            f'Expected timeout=_SETTLE to resolve to its namespace value '
            f'30.0, got {budgets!r}.'
        )

    def test_wait_responsive_nominal_resolves_from_the_namespace(self) -> None:
        """A `wait_responsive` nominal spelled as a module constant
        (`_GATE = 15`) is stretched like a literal one: 2 x 15 = 30.0.
        """
        source = '''
class TestModuleGate:
    async def test_it(self, g):
        await wait_responsive(g.wait(), timeout=_GATE, label='x')
'''
        budgets = item_wait_budgets(source, module_constants={'_GATE': 15})

        assert budgets == {'TestModuleGate::test_it': 30.0}, (
            f'Expected a namespace-resolved nominal of 15 to bill the '
            f'stretched 30.0, got {budgets!r}.'
        )

    @pytest.mark.parametrize(
        'method',
        [
            pytest.param(
                'async def test_it(self, _SETTLE):\n'
                '        await asyncio.wait_for(x, timeout=_SETTLE)\n',
                id='parameter',
            ),
            pytest.param(
                'async def test_it(self):\n'
                '        _SETTLE = compute()\n'
                '        await asyncio.wait_for(x, timeout=_SETTLE)\n',
                id='local-assignment',
            ),
            pytest.param(
                'async def test_it(self):\n'
                '        for _SETTLE in (1, 2):\n'
                '            await asyncio.wait_for(x, timeout=_SETTLE)\n',
                id='for-target',
            ),
        ],
    )
    def test_a_name_the_method_binds_is_never_resolved(self, method: str) -> None:
        """A name the method binds locally is not the module constant of
        the same name, so it bills unbounded even though the namespace
        holds it: the scan cannot read the local value.
        """
        source = f'class TestShadowed:\n    {method}'
        budgets = item_wait_budgets(source, module_constants={'_SETTLE': 30.0})

        assert budgets == {'TestShadowed::test_it': math.inf}, (
            f'Expected a locally bound _SETTLE to shadow the namespace value '
            f'and bill math.inf, got {budgets!r}.'
        )

    def test_a_shadowed_static_table_name_is_never_resolved(self) -> None:
        """The shadow rule is uniform: a method that rebinds a name the
        static known-constant table holds bills unbounded too.
        """
        source = '''
class TestShadowedStaticName:
    async def test_it(self):
        MERGE_RESULT_TIMEOUT = 999
        await asyncio.wait_for(x, timeout=MERGE_RESULT_TIMEOUT)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestShadowedStaticName::test_it': math.inf}, (
            f'Expected a locally rebound MERGE_RESULT_TIMEOUT to bill '
            f'math.inf, got {budgets!r}.'
        )

    @pytest.mark.parametrize('value', [True, '30', None], ids=['bool', 'str', 'none'])
    def test_a_non_numeric_namespace_value_is_unbounded(self, value: object) -> None:
        """Only an int or float that is not a bool resolves; any other
        namespace value is unreadable as a wait, so it bills unbounded.
        """
        source = '''
class TestNonNumeric:
    async def test_it(self):
        await asyncio.wait_for(x, timeout=_SETTLE)
'''
        budgets = item_wait_budgets(source, module_constants={'_SETTLE': value})

        assert budgets == {'TestNonNumeric::test_it': math.inf}, (
            f'Expected a namespace value of {value!r} to bill math.inf, '
            f'got {budgets!r}.'
        )

    def test_a_namespace_value_wins_over_the_static_table(self) -> None:
        """The namespace is what the scanned module's code actually sees, so
        it wins over the static table for the same name.
        """
        source = '''
class TestNamespaceWins:
    async def test_it(self):
        await asyncio.wait_for(x, timeout=MERGE_GATE_BARRIER_TIMEOUT)
'''
        budgets = item_wait_budgets(
            source, module_constants={'MERGE_GATE_BARRIER_TIMEOUT': 7}
        )

        assert budgets == {'TestNamespaceWins::test_it': 7.0}, (
            f'Expected the namespace value 7 to win over the static table, '
            f'got {budgets!r}.'
        )

    @pytest.mark.parametrize(
        ('timeout', 'bill'),
        [
            pytest.param('2 * _SETTLE', 10.0, id='mult'),
            pytest.param('_SETTLE + 1.5', 6.5, id='add'),
            pytest.param('(_SETTLE - 1) / 2', 2.0, id='sub-div'),
            pytest.param('_SETTLE - _NEGATIVE', 8.0, id='exact-operands'),
            pytest.param('_NEGATIVE * 2', 0.0, id='negative-result'),
        ],
    )
    def test_arithmetic_over_readable_operands_bills_its_value(
        self, timeout: str, bill: float
    ) -> None:
        """`+ - * /` over readable operands is exact, so it bills its value:
        operands keep their sign and only the result is clamped at 0.0.
        """
        source = f'''
class TestArithmetic:
    async def test_it(self):
        await asyncio.wait_for(x, {timeout})
'''
        budgets = item_wait_budgets(
            source, module_constants={'_SETTLE': 5.0, '_NEGATIVE': -3}
        )

        assert budgets == {'TestArithmetic::test_it': bill}, (
            f'Expected timeout={timeout} to bill {bill}, got {budgets!r}.'
        )

    @pytest.mark.parametrize(
        'timeout',
        [
            pytest.param('2 * some_var', id='unknown-operand'),
            pytest.param('_SETTLE / 0', id='division-by-zero'),
            pytest.param('_SETTLE // 2', id='unsupported-operator'),
            pytest.param('-_SETTLE', id='unary'),
        ],
    )
    def test_arithmetic_the_scan_cannot_read_is_unbounded(self, timeout: str) -> None:
        """Arithmetic with an unreadable operand or outside `+ - * /` bills
        `math.inf`, like any other value the scan cannot read.
        """
        source = f'''
class TestUnreadableArithmetic:
    async def test_it(self):
        await asyncio.wait_for(x, {timeout})
'''
        budgets = item_wait_budgets(source, module_constants={'_SETTLE': 5.0})

        assert budgets == {'TestUnreadableArithmetic::test_it': math.inf}, (
            f'Expected timeout={timeout} to bill math.inf, got {budgets!r}.'
        )

    def test_wait_responsive_bare_bills_the_stretched_hidden_default(self) -> None:
        """A `wait_responsive(...)` with no `timeout=` bills its hidden
        MERGE_RESULT_TIMEOUT default stretched by RESPONSIVE_WAIT_STRETCH
        (2 x 45 = 90.0), the wall clock the helper can actually consume.
        """
        source = '''
class TestWaitResponsiveBare:
    async def test_it(self):
        await wait_responsive(req.result, label='x')
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestWaitResponsiveBare::test_it': 90.0}, (
            f'Expected the stretched hidden default (90.0), got {budgets!r}.'
        )

    def test_wait_responsive_gate_barrier_bills_twice_its_nominal(self) -> None:
        """`timeout=MERGE_GATE_BARRIER_TIMEOUT` (15) bills 2 x 15 = 30.0."""
        source = '''
class TestWaitResponsiveGate:
    async def test_it(self, g):
        await wait_responsive(g.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label='x')
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestWaitResponsiveGate::test_it': 30.0}, (
            f'Expected a stretched gate-barrier nominal (30.0), got {budgets!r}.'
        )

    def test_wait_responsive_large_nominal_is_clamped_to_the_wall_cap(self) -> None:
        """A nominal whose stretch exceeds RESPONSIVE_WAIT_WALL_CAP bills
        the cap (90.0): the helper's default cap never exceeds it.
        """
        source = '''
class TestWaitResponsiveClamped:
    async def test_it(self):
        await wait_responsive(req.result, timeout=500.0, label='x')
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestWaitResponsiveClamped::test_it': 90.0}, (
            f'Expected the RESPONSIVE_WAIT_WALL_CAP clamp (90.0), got {budgets!r}.'
        )

    def test_wait_responsive_explicit_max_wall_is_billed_verbatim(self) -> None:
        """An explicit `max_wall_s` replaces the helper's scaled default cap,
        so it is the wall clock the site may consume and is billed as is --
        not the 2 x nominal a default-cap site would get.
        """
        source = '''
class TestWaitResponsiveExplicitCap:
    async def test_it(self):
        await wait_responsive(req.result, timeout=1.0, max_wall_s=1000.0, label='x')
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestWaitResponsiveExplicitCap::test_it': 1000.0}, (
            f'Expected an explicit max_wall_s=1000.0 to be billed verbatim, '
            f'got {budgets!r}.'
        )

    def test_wait_responsive_explicit_max_wall_name_resolves(self) -> None:
        """`max_wall_s=RESPONSIVE_WAIT_WALL_CAP` resolves the name and bills
        it (90.0).
        """
        source = '''
class TestWaitResponsiveNamedCap:
    async def test_it(self):
        await wait_responsive(
            req.result, timeout=1.0, max_wall_s=RESPONSIVE_WAIT_WALL_CAP, label='x'
        )
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestWaitResponsiveNamedCap::test_it': 90.0}, (
            f'Expected max_wall_s=RESPONSIVE_WAIT_WALL_CAP to bill 90.0, '
            f'got {budgets!r}.'
        )

    def test_wait_responsive_explicit_none_max_wall_is_the_default(self) -> None:
        """`max_wall_s=None` IS the helper's default, so the site bills the
        stretched nominal (2 x 15 = 30.0), not an unknown cap.
        """
        source = '''
class TestWaitResponsiveNoneCap:
    async def test_it(self, g):
        await wait_responsive(
            g.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, max_wall_s=None, label='x'
        )
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestWaitResponsiveNoneCap::test_it': 30.0}, (
            f'Expected max_wall_s=None to bill the stretched nominal (30.0), '
            f'got {budgets!r}.'
        )

    def test_wait_responsive_unresolvable_max_wall_is_unbounded(self) -> None:
        """An explicit `max_wall_s` the auditor cannot resolve could allow
        any wall clock at all, so the site bills `math.inf` -- which no mark
        clears, making the coverage guard fail loudly rather than quietly
        under-billing.
        """
        source = '''
class TestWaitResponsiveUnknownCap:
    async def test_it(self):
        await wait_responsive(req.result, timeout=1.0, max_wall_s=some_var, label='x')
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestWaitResponsiveUnknownCap::test_it': math.inf}, (
            f'Expected an unresolvable max_wall_s to bill math.inf, got {budgets!r}.'
        )

    def test_wait_responsive_unresolvable_nominal_bills_the_wall_cap(self) -> None:
        """With a default cap the helper never exceeds
        RESPONSIVE_WAIT_WALL_CAP whatever its nominal, so an unresolvable
        `timeout=` bills that exact bound (90.0) rather than 0.0.
        """
        source = '''
class TestWaitResponsiveUnknownNominal:
    async def test_it(self):
        await wait_responsive(req.result, timeout=some_var, label='x')
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestWaitResponsiveUnknownNominal::test_it': 90.0}, (
            f'Expected an unresolvable wait_responsive nominal to bill '
            f'RESPONSIVE_WAIT_WALL_CAP (90.0), got {budgets!r}.'
        )

    def test_stop_worker_bills_its_hidden_default_join(self) -> None:
        """`_stop_worker(worker, worker_task)` joins the worker task for up
        to its hidden default (5.0s), so a teardown through it is billed.
        """
        source = '''
class TestStopWorkerDefault:
    async def test_it(self):
        await _stop_worker(worker, worker_task)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestStopWorkerDefault::test_it': 5.0}, (
            f'Expected the hidden default join (5.0), got {budgets!r}.'
        )

    def test_stop_worker_bills_an_explicit_join_timeout(self) -> None:
        """An explicit `join_timeout=` is the join actually waited (20.0)."""
        source = '''
class TestStopWorkerExplicit:
    async def test_it(self):
        await _stop_worker(worker, worker_task, join_timeout=20.0)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestStopWorkerExplicit::test_it': 20.0}, (
            f'Expected an explicit join_timeout=20.0 to be billed, got {budgets!r}.'
        )

    def test_stop_worker_unresolvable_join_timeout_is_unbounded(self) -> None:
        """The teardown join is real wall clock, so a `join_timeout=` the
        scan cannot read bills `math.inf` rather than vanishing from the
        budget.
        """
        source = '''
class TestStopWorkerUnknownJoin:
    async def test_it(self):
        await _stop_worker(worker, worker_task, join_timeout=some_var)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestStopWorkerUnknownJoin::test_it': math.inf}, (
            f'Expected an unresolvable join_timeout to bill math.inf, '
            f'got {budgets!r}.'
        )

    def test_asyncio_sleep_is_outside_the_counted_shape_set(self) -> None:
        """`asyncio.sleep(999)` contributes 0.0 -- sleeps are deliberately
        OUT of the counted shape set (the guard is a conservative floor,
        not a full audit; see the floor/superset design decision).
        """
        source = '''
class TestSleepNotCounted:
    async def test_sleeps(self):
        await asyncio.sleep(999)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestSleepNotCounted::test_sleeps': 0.0}, (
            f'Expected asyncio.sleep(...) to contribute 0.0 (not a counted '
            f'call shape), got {budgets!r}.'
        )

    def test_non_test_methods_do_not_contribute(self) -> None:
        """A non-`test_*` method (a private helper or a fixture) contributes
        nothing -- only real test methods drive the budget.
        """
        source = '''
class TestHelperMethodsIgnored:
    async def _helper(self):
        await asyncio.wait_for(x, timeout=999.0)

    @pytest.fixture
    def some_fixture(self):
        return object()

    async def test_actual(self):
        await asyncio.wait_for(y, timeout=5.0)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'TestHelperMethodsIgnored::test_actual': 5.0}, (
            f'Expected only test_* methods to contribute to the budget '
            f'(5.0), with the 999.0s wait in a non-test helper ignored; '
            f'got {budgets!r}.'
        )

    def test_class_with_no_test_methods_yields_no_bill(self) -> None:
        """A class with no test methods at all yields no non-zero item --
        no crash.
        """
        source = '''
class TestNoTestMethods:
    async def _helper(self):
        await asyncio.wait_for(x, timeout=45.0)
'''
        budgets = item_wait_budgets(source)

        assert not any(budgets.values()), (
            f'Expected a class with no test_* methods to yield no non-zero '
            f'item, got {budgets!r}.'
        )

    def test_class_with_no_waits_yields_no_bill(self) -> None:
        """A class whose test methods never wait yields no non-zero item --
        no crash."""
        source = '''
class TestNoWaits:
    async def test_nothing(self):
        x = 1 + 1
        return x
'''
        budgets = item_wait_budgets(source)

        assert not any(budgets.values()), (
            f'Expected a class with no waits at all to yield no non-zero '
            f'item, got {budgets!r}.'
        )

    def test_syntactically_odd_but_parseable_input_does_not_raise(self) -> None:
        """A syntactically odd but still-parseable module -- no `Test*`
        classes at all, only module-level and conditionally-nested code --
        does not raise.
        """
        source = '''
x = 1
def free_function():
    return 42
if x:
    class NotATestClass:
        def method(self):
            pass
'''
        budgets = item_wait_budgets(source)

        assert budgets == {}, (
            f'Expected a module with no Test* classes to yield an empty '
            f'result without raising, got {budgets!r}.'
        )

    def test_unparseable_source_returns_empty_dict_without_raising(self) -> None:
        """Unparseable source returns `{}` rather than raising: a crash here
        would fail the whole suite over an unrelated edit to a scanned file.
        """
        budgets = item_wait_budgets('def not valid python(:')

        assert budgets == {}, (
            f'Expected unparseable source to return {{}} without raising, '
            f'got {budgets!r}.'
        )

    def test_item_ids_tell_module_level_functions_from_methods(self) -> None:
        source = (
            'async def test_x():\n    await asyncio.wait_for(a, timeout=1)\n'
            'class TestY:\n    async def test_z(self):\n'
            '        await asyncio.wait_for(a, timeout=1)\n'
        )
        kinds = {item: is_module_level_item(item) for item in item_wait_budgets(source)}

        assert kinds == {'test_x': True, 'TestY::test_z': False}, kinds

    def test_module_level_test_function_is_keyed_by_its_bare_name(self) -> None:
        """pytest collects a module-level `test*` function as an item of
        its own, so it is billed under its bare name; a module-level helper
        that is not named `test*` is not an item and contributes nothing.
        """
        source = '''
async def test_x():
    await asyncio.wait_for(x, timeout=45)
    await asyncio.wait_for(y, timeout=45)

async def settle():
    await asyncio.wait_for(z, timeout=999)
'''
        budgets = item_wait_budgets(source)

        assert budgets == {'test_x': 90.0}, (
            f'Expected the module-level test_x billed at 90.0 under its bare '
            f'name and the non-test helper ignored, got {budgets!r}.'
        )


# Synthetic wait and mark values for TestAuditModule, derived from the
# ambient budget so a change to it cannot silently move a case across its
# threshold.
_HEAVY_WAIT = AMBIENT_TEST_TIMEOUT / 2  # two of these reach the ambient budget
_LIGHT_WAIT = AMBIENT_TEST_TIMEOUT / 4  # one of these stays under it
_GENEROUS_MARK = AMBIENT_TEST_TIMEOUT * 2  # clears every synthetic bill here
_TIGHT_MARK = _LIGHT_WAIT / 2  # clears none of them

_HEAVY_BODY = (
    f'        await asyncio.wait_for(a, timeout={_HEAVY_WAIT})\n'
    f'        await asyncio.wait_for(b, timeout={_HEAVY_WAIT})\n'
)
_LIGHT_BODY = f'        await asyncio.wait_for(a, timeout={_LIGHT_WAIT})\n'


def _exec_module(source: str) -> dict[str, object]:
    """The runtime namespace of *source* executed as a module, so its marks
    and constants come from the very text the engine parses."""
    namespace: dict[str, object] = {'pytest': pytest, 'asyncio': asyncio}
    exec(compile(source, '<synthetic test module>', 'exec'), namespace)
    return namespace


def _audits(
    source: str, namespace: Mapping[str, object] | None = None
) -> list[ItemWaitAudit]:
    """*source*'s audit rows against *namespace*, else against the namespace
    of *source* executed."""
    loaded = _exec_module(source) if namespace is None else namespace
    return audit_module(source, lambda: loaded) or []


def _offenders(
    source: str, namespace: Mapping[str, object] | None = None
) -> list[ItemWaitAudit]:
    return [audit for audit in _audits(source, namespace) if audit.offends]


def _limit_sources(source: str) -> dict[str, LimitSource]:
    return {audit.item: audit.limit_source for audit in _audits(source)}


class TestAuditModule:
    """`audit_module` judges each pytest item's bill against the
    timeout pytest-timeout actually applies to it: the closest timeout mark
    (the test's own, then its class's, then the module's `pytestmark`), else
    the ambient budget. Every case runs a synthetic module whose waits are
    defined, never awaited.
    """

    def test_an_unmarked_module_level_test_reaching_the_ambient_budget_offends(
        self,
    ) -> None:
        source = 'async def test_x():\n' + _HEAVY_BODY.replace('        ', '    ')
        offenders = _offenders(source)

        assert [(o.item, o.limit_source) for o in offenders] == [
            ('test_x', LimitSource.AMBIENT)
        ], f'expected test_x to offend under the ambient budget, got {offenders!r}'
        assert offenders[0].budget == AMBIENT_TEST_TIMEOUT
        assert offenders[0].limit == AMBIENT_TEST_TIMEOUT

    def test_an_unmarked_method_under_the_ambient_budget_is_covered(self) -> None:
        source = 'class TestLight:\n    async def test_it(self):\n' + _LIGHT_BODY

        assert _offenders(source) == []

    def test_a_class_mark_covers_its_methods_only_when_it_clears_the_bill(
        self,
    ) -> None:
        adequate = (
            f'@pytest.mark.timeout({_GENEROUS_MARK})\n'
            'class TestCovered:\n    async def test_it(self):\n' + _HEAVY_BODY
        )
        too_tight = (
            f'@pytest.mark.timeout({_TIGHT_MARK})\n'
            'class TestTight:\n    async def test_it(self):\n' + _LIGHT_BODY
        )

        assert _offenders(adequate) == []
        offenders = _offenders(too_tight)
        assert [(o.item, o.limit_source, o.limit) for o in offenders] == [
            ('TestTight::test_it', LimitSource.CLASS_MARK, _TIGHT_MARK)
        ], (
            f'a class mark below the bill offends even under the ambient '
            f'budget, because the mark replaces it; got {offenders!r}'
        )

    def test_a_method_mark_replaces_its_class_mark(self) -> None:
        tighter_method = (
            f'@pytest.mark.timeout({_GENEROUS_MARK})\n'
            'class TestMixed:\n'
            f'    @pytest.mark.timeout({_TIGHT_MARK})\n'
            '    async def test_tight(self):\n' + _LIGHT_BODY
            + '    async def test_inherits(self):\n' + _LIGHT_BODY
        )
        adequate_method = (
            'class TestUnmarked:\n'
            f'    @pytest.mark.timeout({_GENEROUS_MARK})\n'
            '    async def test_heavy(self):\n' + _HEAVY_BODY
        )

        offenders = _offenders(tighter_method)
        assert [(o.item, o.limit_source) for o in offenders] == [
            ('TestMixed::test_tight', LimitSource.FUNCTION_MARK)
        ], (
            f'the method-level mark replaces the generous class mark, so only '
            f'test_tight is short of its bill; got {offenders!r}'
        )
        assert _offenders(adequate_method) == []

    def test_a_module_level_test_runs_under_its_own_mark(self) -> None:
        source = (
            f'@pytest.mark.timeout({_GENEROUS_MARK})\n'
            'async def test_marked():\n' + _HEAVY_BODY.replace('        ', '    ')
            + 'async def test_unmarked():\n' + _HEAVY_BODY.replace('        ', '    ')
        )
        offenders = _offenders(source)

        assert [(o.item, o.limit_source) for o in offenders] == [
            ('test_unmarked', LimitSource.AMBIENT)
        ], (
            f'the same bill clears test_marked\'s own mark but not the '
            f'ambient budget test_unmarked runs under; got {offenders!r}'
        )

    @pytest.mark.parametrize(
        'pytestmark',
        [
            pytest.param(f'pytest.mark.timeout({_GENEROUS_MARK})', id='single'),
            pytest.param(
                f'[pytest.mark.asyncio, pytest.mark.timeout({_GENEROUS_MARK})]',
                id='list',
            ),
        ],
    )
    def test_a_module_pytestmark_covers_an_unmarked_class(self, pytestmark: str) -> None:
        source = (
            f'pytestmark = {pytestmark}\n'
            'class TestCovered:\n    async def test_it(self):\n' + _HEAVY_BODY
        )
        assert _offenders(source) == []
        assert _limit_sources(source) == {'TestCovered::test_it': LimitSource.MODULE_MARK}

    def test_a_too_tight_module_mark_offends_under_the_ambient_budget(self) -> None:
        source = (
            f'pytestmark = pytest.mark.timeout({_TIGHT_MARK})\n'
            'class TestLight:\n    async def test_it(self):\n' + _LIGHT_BODY
        )
        offenders = _offenders(source)

        assert [(o.item, o.limit_source) for o in offenders] == [
            ('TestLight::test_it', LimitSource.MODULE_MARK)
        ], f'got {offenders!r}'

    def test_a_tighter_class_mark_overrides_a_generous_module_mark(self) -> None:
        source = (
            f'pytestmark = pytest.mark.timeout({_GENEROUS_MARK})\n'
            f'@pytest.mark.timeout({_TIGHT_MARK})\n'
            'class TestTight:\n    async def test_it(self):\n' + _LIGHT_BODY
        )
        offenders = _offenders(source)

        assert [(o.item, o.limit_source) for o in offenders] == [
            ('TestTight::test_it', LimitSource.CLASS_MARK)
        ], f'got {offenders!r}'

    def test_a_module_pytestmark_without_a_timeout_leaves_the_ambient_budget(
        self,
    ) -> None:
        source = (
            'pytestmark = pytest.mark.asyncio\n'
            'class TestHeavy:\n    async def test_it(self):\n' + _HEAVY_BODY
        )
        offenders = _offenders(source)

        assert [(o.item, o.limit_source) for o in offenders] == [
            ('TestHeavy::test_it', LimitSource.AMBIENT)
        ], f'got {offenders!r}'

    def test_an_unbounded_bill_offends_under_any_mark(self) -> None:
        source = (
            '@pytest.mark.timeout(10**6)\n'
            'class TestUnbounded:\n'
            '    async def test_it(self):\n'
            '        await asyncio.wait_for(a, timeout=unknown_name)\n'
        )
        offenders = _offenders(source)

        assert [o.item for o in offenders] == ['TestUnbounded::test_it'], f'got {offenders!r}'
        assert math.isinf(offenders[0].budget)
        assert 'unbounded' in offenders[0].describe(), offenders[0].describe()

    def test_a_module_constant_resolves_end_to_end_through_the_namespace(self) -> None:
        source = (
            '_SETTLE = 30.0\n'
            'class TestSettle:\n'
            '    async def test_it(self):\n'
            '        await asyncio.wait_for(a, timeout=_SETTLE)\n'
        )
        namespace = _exec_module(source)

        assert item_wait_budgets(source, module_constants=namespace) == {
            'TestSettle::test_it': 30.0
        }
        assert _offenders(source, namespace) == [], (
            'a namespace-resolved 30.0s bill clears the ambient budget; billed '
            'unbounded it would offend under any timeout'
        )

    def test_a_class_missing_from_the_namespace_is_an_unresolved_offender(self) -> None:
        source = 'class TestGone:\n    async def test_it(self):\n' + _LIGHT_BODY
        namespace = _exec_module(source)
        del namespace['TestGone']
        offenders = _offenders(source, namespace)

        assert [(o.item, o.limit_source, o.limit) for o in offenders] == [
            ('TestGone::test_it', LimitSource.UNRESOLVED, None)
        ], f'an item whose marks cannot be read must offend loudly; got {offenders!r}'

    def test_a_mark_with_unreadable_seconds_offends(self) -> None:
        source = (
            "@pytest.mark.timeout('soon')\n"
            'class TestUnreadable:\n    async def test_it(self):\n' + _LIGHT_BODY
        )
        offenders = _offenders(source)

        assert [(o.item, o.limit) for o in offenders] == [
            ('TestUnreadable::test_it', None)
        ], f'got {offenders!r}'

    @pytest.mark.parametrize(
        'mark',
        [
            pytest.param("@pytest.mark.timeout(method='thread')", id='method-only'),
            pytest.param('@pytest.mark.timeout(timeout=None)', id='timeout-none'),
        ],
    )
    def test_a_closest_mark_without_seconds_leaves_the_ambient_budget(
        self, mark: str
    ) -> None:
        """pytest-timeout reads only the closest timeout mark, and one that
        sets no seconds falls back to the ini/CLI timeout -- not to a
        farther mark's seconds, so the generous class mark does not apply.
        """
        source = (
            f'@pytest.mark.timeout({_GENEROUS_MARK})\n'
            'class TestMixed:\n'
            f'    {mark}\n'
            '    async def test_heavy(self):\n' + _HEAVY_BODY
            + f'    {mark}\n'
            '    async def test_light(self):\n' + _LIGHT_BODY
        )
        offenders = _offenders(source)

        assert [(o.item, o.limit, o.limit_source) for o in offenders] == [
            ('TestMixed::test_heavy', AMBIENT_TEST_TIMEOUT, LimitSource.AMBIENT)
        ], f'got {offenders!r}'

    @pytest.mark.parametrize('seconds', [0, -1])
    @pytest.mark.parametrize(
        'template',
        [
            pytest.param(
                'class TestOff:\n    @pytest.mark.timeout({seconds})\n'
                '    async def test_it(self):\n',
                id='function-mark',
            ),
            pytest.param(
                '@pytest.mark.timeout({seconds})\nclass TestOff:\n'
                '    async def test_it(self):\n',
                id='class-mark',
            ),
            pytest.param(
                'pytestmark = pytest.mark.timeout({seconds})\nclass TestOff:\n'
                '    async def test_it(self):\n',
                id='module-mark',
            ),
        ],
    )
    def test_a_mark_at_or_below_zero_disables_the_timeout(
        self, template: str, seconds: int
    ) -> None:
        """pytest-timeout arms no timer for seconds <= 0, so nothing can kill
        the item, whatever its bill -- even an unbounded one."""
        source = template.format(seconds=seconds) + (
            '        await asyncio.wait_for(a, timeout=unknown_name)\n'
        )

        assert _offenders(source) == []

    def test_a_base_class_mark_reaches_a_subclass_decorated_with_other_marks(
        self,
    ) -> None:
        """Decorating the subclass gives it a ``pytestmark`` of its own that
        hides the base class's from a plain ``getattr``; pytest still
        applies the base class's timeout mark."""
        source = (
            f'@pytest.mark.timeout({_GENEROUS_MARK})\n'
            'class _Base:\n    pass\n'
            '@pytest.mark.asyncio\n'
            'class TestSub(_Base):\n    async def test_it(self):\n' + _HEAVY_BODY
        )
        assert _offenders(source) == []
        assert _limit_sources(source) == {'TestSub::test_it': LimitSource.CLASS_MARK}

    def test_a_base_class_timeout_mark_wins_over_the_subclass_own_one(self) -> None:
        """pytest collects class marks base class first and takes the first
        timeout mark, so a base class's mark beats the subclass's own --
        measured on pytest 9.0.3. The scan mirrors that order rather than
        the intuitive one."""
        source = (
            f'@pytest.mark.timeout({_TIGHT_MARK})\n'
            'class _Base:\n    pass\n'
            f'@pytest.mark.timeout({_GENEROUS_MARK})\n'
            'class TestSub(_Base):\n    async def test_it(self):\n' + _LIGHT_BODY
        )
        offenders = _offenders(source)

        assert [(o.item, o.limit, o.limit_source) for o in offenders] == [
            ('TestSub::test_it', _TIGHT_MARK, LimitSource.CLASS_MARK)
        ], f'got {offenders!r}'

    def test_only_the_three_mark_levels_count_as_a_mark(self) -> None:
        assert {source for source in LimitSource if source.is_mark} == {
            LimitSource.FUNCTION_MARK,
            LimitSource.CLASS_MARK,
            LimitSource.MODULE_MARK,
        }

    def test_every_billed_item_gets_a_row_offending_or_not(self) -> None:
        source = (
            'class TestMixed:\n'
            '    async def test_heavy(self):\n' + _HEAVY_BODY
            + '    async def test_light(self):\n' + _LIGHT_BODY
            + '    async def test_idle(self):\n        pass\n'
        )

        assert [(audit.item, audit.offends) for audit in _audits(source)] == [
            ('TestMixed::test_heavy', True),
            ('TestMixed::test_light', False),
        ], 'an item with no recognised wait bills nothing and gets no row'

    @pytest.mark.parametrize(
        'source',
        [
            pytest.param('def not valid python(:', id='unparseable'),
            pytest.param('class TestIdle:\n    def test_it(self):\n        pass\n', id='no-wait'),
            pytest.param(
                'class TestZero:\n    async def test_it(self):\n'
                '        await asyncio.wait_for(a, timeout=0)\n',
                id='zero-wait',
            ),
        ],
    )
    def test_a_module_with_nothing_to_bill_is_never_loaded(self, source: str) -> None:
        """Loading a module means importing it, so the audit skips that
        when the parse alone shows nothing can bill."""
        loads: list[None] = []

        def load_namespace() -> Mapping[str, object]:
            loads.append(None)
            return {}

        assert audit_module(source, load_namespace) is None
        assert loads == [], 'the namespace must not be loaded'

    def test_a_module_whose_namespace_cannot_be_loaded_is_not_audited(self) -> None:
        source = 'async def test_x():\n' + _HEAVY_BODY.replace('        ', '    ')

        assert audit_module(source, lambda: None) is None

    def test_an_audit_row_is_frozen(self) -> None:
        source = 'async def test_x():\n' + _HEAVY_BODY.replace('        ', '    ')
        audit = _audits(source)[0]

        field = dataclasses.fields(audit)[0].name

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(audit, field, 'other')
