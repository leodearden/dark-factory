"""Unit tests for the AST wait-budget audit engine in _wait_budget_audit.py.

Every case drives the engine with a SYNTHETIC source string (and, where a
name must resolve, a literal namespace), never a real test module, so each
case is non-vacuous and does not shift when another file is edited.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math

import pytest
from _wait_budget_audit import (
    AMBIENT_TEST_TIMEOUT,
    LimitSource,
    WaitBudgetOffender,
    item_limit_sources,
    item_wait_budgets,
    suppressed_result_wait_methods,
    unguarded_worker_teardown_methods,
    wait_budget_offenders,
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


# ---------------------------------------------------------------------------
# task 4219: suppressed-wait debt-ratchet scanners
# ---------------------------------------------------------------------------


class TestSuppressedWaitScanner:
    """Unit tests for `suppressed_result_wait_methods` -- the executable
    form of the `outcome is None` anti-pattern that
    `test_merge_queue_concurrent_verify.py::_await_outcome`'s docstring
    names ("The shape this replaces"): either a `with
    contextlib.suppress(TimeoutError): outcome = await
    asyncio.wait_for(fut, timeout=N)` shape, or a `try: outcome = await
    asyncio.wait_for(...) except TimeoutError: outcome = None` shape. Both
    convert a genuine pipeline hang or a missing cascade into a confusing
    `outcome is None` assertion failure instead of failing loudly by label
    (task 4219; the try/except spelling was added in the task 4219
    amendment pass).
    """

    def test_suppressed_timeout_error_around_wait_for_is_reported(self) -> None:
        """A `with contextlib.suppress(TimeoutError):` block enclosing an
        `asyncio.wait_for(...)` call IS reported -- this is the anti-pattern
        itself.
        """
        source = '''
class TestSuppressedWait:
    async def test_it(self):
        with contextlib.suppress(TimeoutError):
            outcome = await asyncio.wait_for(fut, timeout=5.0)
'''
        result = suppressed_result_wait_methods(source)

        assert result == {'TestSuppressedWait': {'test_it'}}, (
            f'Expected the suppress(TimeoutError)-wrapped wait_for to be '
            f'reported, got {result!r}.'
        )

    def test_suppress_asyncio_timeout_error_attribute_form_is_reported(self) -> None:
        """The `asyncio.TimeoutError` attribute spelling (as opposed to the
        bare `TimeoutError` Name form) inside `suppress(...)` IS also
        recognised. No site in test_merge_queue_concurrent_verify.py used
        this spelling when the case was written --
        this pins the branch so it does not silently rot un-covered.
        """
        source = '''
class TestSuppressedWaitAttributeForm:
    async def test_it(self):
        with contextlib.suppress(asyncio.TimeoutError):
            outcome = await asyncio.wait_for(fut, timeout=5.0)
'''
        result = suppressed_result_wait_methods(source)

        assert result == {'TestSuppressedWaitAttributeForm': {'test_it'}}, (
            f'Expected the suppress(asyncio.TimeoutError)-wrapped wait_for '
            f'to be reported, got {result!r}.'
        )

    def test_bare_suppress_name_import_form_is_reported(self) -> None:
        """`suppress(TimeoutError):` -- the bare `Name` call from a
        `from contextlib import suppress` import, rather than the
        `contextlib.suppress` attribute form -- IS also recognised.
        """
        source = '''
class TestBareSuppressImport:
    async def test_it(self):
        with suppress(TimeoutError):
            outcome = await asyncio.wait_for(fut, timeout=5.0)
'''
        result = suppressed_result_wait_methods(source)

        assert result == {'TestBareSuppressImport': {'test_it'}}, (
            f'Expected the bare suppress(TimeoutError)-wrapped wait_for to '
            f'be reported, got {result!r}.'
        )

    def test_try_except_timeout_error_around_wait_for_is_reported(self) -> None:
        """The second spelling of the anti-pattern: a bare `try: outcome =
        await asyncio.wait_for(...) except TimeoutError: outcome = None`
        block IS reported -- this is the exact shape found (undetected,
        until this amendment) at
        `test_merge_queue_concurrent_verify.py::TestHaltAndUnavailable::test_runner_unavailable_quarantine_fallback`.
        """
        source = '''
class TestTryExceptTimeoutError:
    async def test_it(self):
        try:
            outcome = await asyncio.wait_for(fut, timeout=5.0)
        except TimeoutError:
            outcome = None
'''
        result = suppressed_result_wait_methods(source)

        assert result == {'TestTryExceptTimeoutError': {'test_it'}}, (
            f'Expected the try/except TimeoutError-wrapped wait_for to be '
            f'reported, got {result!r}.'
        )

    def test_try_except_other_exception_is_not_reported(self) -> None:
        """A `try/except` around a wait_for call whose handler catches a
        DIFFERENT exception (not `TimeoutError`) is NOT reported -- the
        anti-pattern is specifically about swallowing a timeout, not about
        exception handling in general.
        """
        source = '''
class TestTryExceptOtherException:
    async def test_it(self):
        try:
            outcome = await asyncio.wait_for(fut, timeout=5.0)
        except ValueError:
            outcome = None
'''
        result = suppressed_result_wait_methods(source)

        assert result == {}, (
            f'Expected a try/except catching ValueError (not TimeoutError) '
            f'to NOT be reported, got {result!r}.'
        )

    def test_try_except_timeout_error_that_fails_loudly_is_not_reported(self) -> None:
        """A `try/except TimeoutError:` handler that does NOT swallow --
        e.g. it runs cleanup and then calls `pytest.fail(...)` with a clear
        diagnostic message, the exact shape used in
        test_merge_queue_concurrent_verify.py at
        `TestOverlapSignal::test_both_verifies_enter_before_either_released`
        and `TestLastItemOfBurstFinalizes::test_single_item_resolves_with_free_remote_slot`
        -- is NOT reported. This is already the DESIRED loud-failure
        behaviour, not the `outcome is None` anti-pattern; an earlier
        version of this scanner (task 4219 amendment) matched on the
        caught exception type alone and false-positived on both real
        sites.
        """
        source = '''
class TestTryExceptLoudFailure:
    async def test_it(self):
        try:
            outcome = await asyncio.wait_for(fut, timeout=5.0)
        except TimeoutError:
            with contextlib.suppress(Exception):
                await cleanup()
            pytest.fail('descriptive RED message explaining the failure')
'''
        result = suppressed_result_wait_methods(source)

        assert result == {}, (
            f'Expected a try/except TimeoutError handler that fails loudly '
            f'via pytest.fail(...) (not a swallow) to NOT be reported, '
            f'got {result!r}.'
        )

    def test_try_except_timeout_error_without_wait_for_is_not_reported(self) -> None:
        """A `try/except TimeoutError:` block whose `try:` body encloses no
        `asyncio.wait_for` call at all is NOT reported.
        """
        source = '''
class TestTryExceptNoWaitFor:
    async def test_it(self):
        try:
            do_something_unrelated()
        except TimeoutError:
            pass
'''
        result = suppressed_result_wait_methods(source)

        assert result == {}, (
            f'Expected a try/except TimeoutError block with no enclosed '
            f'wait_for call in the try body to NOT be reported, '
            f'got {result!r}.'
        )

    def test_multiple_classes_and_methods_are_all_aggregated(self) -> None:
        """Offenders across MULTIPLE classes and MULTIPLE methods within
        one class are all accumulated into the returned mapping -- not just
        the first one found. Pins the dict-of-sets aggregation and the
        per-class method-name set, which the single-offender cases above
        cannot distinguish from an implementation that stops at the first
        hit.
        """
        source = '''
class TestFirstOffender:
    async def test_one(self):
        with contextlib.suppress(TimeoutError):
            outcome = await asyncio.wait_for(fut, timeout=5.0)

    async def test_two(self):
        try:
            outcome = await asyncio.wait_for(fut, timeout=5.0)
        except TimeoutError:
            outcome = None

    async def test_clean(self):
        outcome = await asyncio.wait_for(fut, timeout=5.0)

class TestSecondOffender:
    async def test_only(self):
        with contextlib.suppress(TimeoutError):
            outcome = await asyncio.wait_for(fut, timeout=5.0)
'''
        result = suppressed_result_wait_methods(source)

        assert result == {
            'TestFirstOffender': {'test_one', 'test_two'},
            'TestSecondOffender': {'test_only'},
        }, (
            f'Expected offenders from both classes and both offending '
            f'methods in TestFirstOffender to be aggregated together, '
            f'got {result!r}.'
        )

    def test_suppress_exception_is_not_reported(self) -> None:
        """`contextlib.suppress(Exception):` (not `TimeoutError`
        specifically) around a wait_for call is NOT reported -- this is the
        blessed best-effort teardown-join shape (the `worker_task` join in
        `test_merge_queue_concurrent_verify.py::_stop_worker`), which
        asserts nothing and so isn't the None-outcome
        anti-pattern.
        """
        source = '''
class TestSuppressException:
    async def test_it(self):
        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=5.0)
'''
        result = suppressed_result_wait_methods(source)

        assert result == {}, (
            f'Expected suppress(Exception) (not TimeoutError) to NOT be '
            f'reported, got {result!r}.'
        )

    def test_bare_wait_for_without_suppress_is_not_reported(self) -> None:
        """A bare `await asyncio.wait_for(...)` with no enclosing suppress
        at all is NOT reported -- a loud TimeoutError is exactly the
        desired behaviour, not the anti-pattern.
        """
        source = '''
class TestBareWaitFor:
    async def test_it(self):
        outcome = await asyncio.wait_for(fut, timeout=5.0)
'''
        result = suppressed_result_wait_methods(source)

        assert result == {}, (
            f'Expected a bare, unsuppressed wait_for to NOT be reported, '
            f'got {result!r}.'
        )

    def test_suppress_timeout_error_without_wait_for_is_not_reported(self) -> None:
        """A `suppress(TimeoutError):` block that encloses no
        `asyncio.wait_for` call at all is NOT reported -- the anti-pattern
        is specifically about swallowing a result-future wait, not about
        `suppress(TimeoutError)` used for some unrelated purpose.
        """
        source = '''
class TestSuppressNoWaitFor:
    async def test_it(self):
        with contextlib.suppress(TimeoutError):
            do_something_unrelated()
'''
        result = suppressed_result_wait_methods(source)

        assert result == {}, (
            f'Expected a suppress(TimeoutError) block with no enclosed '
            f'wait_for call to NOT be reported, got {result!r}.'
        )

    def test_non_test_method_and_non_test_class_are_skipped(self) -> None:
        """A non-`test_*` method (even carrying the anti-pattern) and a
        class whose name does not start with `Test` are both skipped
        entirely.
        """
        source = '''
class TestHelperMethodSkipped:
    async def _helper(self):
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(fut, timeout=5.0)

class NotATestClass:
    async def test_it(self):
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(fut, timeout=5.0)
'''
        result = suppressed_result_wait_methods(source)

        assert result == {}, (
            f'Expected a non-test_* method and a non-Test* class to both '
            f'be skipped, got {result!r}.'
        )

    def test_unparseable_source_returns_empty_dict_without_raising(self) -> None:
        """Unparseable source returns `{}` rather than raising -- mirrors
        `item_wait_budgets`'s must-never-raise contract: a crash here would
        fail the whole suite over an unrelated edit to a scanned file.
        """
        result = suppressed_result_wait_methods('def not valid python(:')

        assert result == {}, (
            f'Expected unparseable source to return {{}} without raising, '
            f'got {result!r}.'
        )


class TestWorkerTeardownGuardScanner:
    """Unit tests for `unguarded_worker_teardown_methods` -- the executable
    form of the esc-3980-4 leak invariant recorded in
    `test_merge_queue_concurrent_verify.py::_stop_worker`'s docstring: `wait_responsive` gives up by raising `_pytest.outcomes.Failed`, so a
    give-up on a straight-line body (like a mid-body `assert`) skips the
    worker-stop call entirely and leaks a live SpeculativeMergeWorker plus
    its run task into pytest-asyncio teardown -- how one red test used to
    cascade into unrelated failures elsewhere in the session (task 4219).

    A stop call counts as guarded only when the `finally:`'s guarding
    `try:` body encloses EVERY `wait_responsive` call in the method (not
    merely when SOME finally exists somewhere in the method), and either
    `worker.stop()` or the shared `_stop_worker(...)` helper counts as a
    stop call (task 4219 amendment; both fixed after review found the
    original implementation's blind spots).
    """

    def test_straight_line_stop_after_wait_responsive_is_reported(self) -> None:
        """A `test_*` method that calls `wait_responsive(...)` and then
        calls `worker.stop()` from straight-line code (not inside a
        `finally:`) IS reported -- a give-up mid-body would skip the stop
        call entirely.
        """
        source = '''
class TestUnguardedTeardown:
    async def test_it(self):
        await wait_responsive(fut, timeout=5.0, label='x')
        await worker.stop()
'''
        result = unguarded_worker_teardown_methods(source)

        assert result == ['TestUnguardedTeardown::test_it'], (
            f'Expected the straight-line worker.stop() after wait_responsive '
            f'to be reported, got {result!r}.'
        )

    def test_stop_inside_finally_is_not_reported(self) -> None:
        """The same method with `worker.stop()` moved into a
        `try: ... finally:` block is NOT reported -- a give-up mid-try
        still runs the finally, so the worker is always stopped.
        """
        source = '''
class TestGuardedTeardown:
    async def test_it(self):
        try:
            await wait_responsive(fut, timeout=5.0, label='x')
        finally:
            await worker.stop()
'''
        result = unguarded_worker_teardown_methods(source)

        assert result == [], (
            f'Expected worker.stop() inside a finally: block to NOT be '
            f'reported, got {result!r}.'
        )

    def test_finally_that_does_not_enclose_the_wait_is_still_reported(self) -> None:
        """A `finally:` block that stops the worker IS present, but the
        `try:` guarding it does NOT enclose the `wait_responsive` call
        (which sits before the try, as straight-line code) -- this IS
        reported. A give-up from that wait_responsive call raises before
        the try is ever entered, so its finally never runs and the worker
        still leaks; crediting ANY finally-wrapped stop() in the method,
        regardless of what its try actually encloses, would make the guard
        vacuous for exactly this shape (task 4219 amendment; reviewer
        finding, empirically verified against the pre-fix implementation).
        """
        source = '''
class TestFinallyDoesNotEncloseWait:
    async def test_it(self):
        await wait_responsive(fut, timeout=5.0, label='x')
        try:
            await something()
        finally:
            await worker.stop()
'''
        result = unguarded_worker_teardown_methods(source)

        assert result == ['TestFinallyDoesNotEncloseWait::test_it'], (
            f'Expected a finally that does not enclose the wait_responsive '
            f'call to still be reported, got {result!r}.'
        )

    def test_one_guarded_and_one_unguarded_stop_call_is_reported(self) -> None:
        """A method with TWO worker-stop calls -- one correctly
        finally-guarded, one straight-line -- IS reported. Pins the
        `any(call not in guarded ...)` branch: a single well-guarded stop
        call must not paper over a second, unguarded one.
        """
        source = '''
class TestMixedGuardedAndUnguarded:
    async def test_it(self):
        try:
            await wait_responsive(fut, timeout=5.0, label='x')
        finally:
            await worker.stop()
        await worker.stop()
'''
        result = unguarded_worker_teardown_methods(source)

        assert result == ['TestMixedGuardedAndUnguarded::test_it'], (
            f'Expected the method with one guarded and one unguarded '
            f'worker.stop() call to be reported, got {result!r}.'
        )

    def test_straight_line_stop_worker_helper_is_reported(self) -> None:
        """A straight-line `await _stop_worker(worker, worker_task)` --
        the shared teardown helper -- IS reported when not inside a
        finally, exactly like a straight-line `worker.stop()`. Without this, a future migration
        that adopts the OTHER blessed teardown shape would make
        the guard silently vacuous instead of rewarding the correct usage.
        """
        source = '''
class TestUnguardedStopWorkerHelper:
    async def test_it(self):
        await wait_responsive(fut, timeout=5.0, label='x')
        await _stop_worker(worker, worker_task)
'''
        result = unguarded_worker_teardown_methods(source)

        assert result == ['TestUnguardedStopWorkerHelper::test_it'], (
            f'Expected the straight-line _stop_worker(...) helper call to '
            f'be reported, got {result!r}.'
        )

    def test_stop_worker_helper_inside_finally_is_not_reported(self) -> None:
        """The same `_stop_worker(...)` helper call, moved into a
        `try: ... finally:` block whose try encloses the wait_responsive
        call, is NOT reported.
        """
        source = '''
class TestGuardedStopWorkerHelper:
    async def test_it(self):
        try:
            await wait_responsive(fut, timeout=5.0, label='x')
        finally:
            await _stop_worker(worker, worker_task)
'''
        result = unguarded_worker_teardown_methods(source)

        assert result == [], (
            f'Expected a finally-wrapped _stop_worker(...) helper call to '
            f'NOT be reported, got {result!r}.'
        )

    def test_stop_without_wait_responsive_is_not_reported(self) -> None:
        """A method that calls `worker.stop()` outside a finally but never
        calls `wait_responsive` at all is NOT reported -- the guard is
        deliberately scoped to migrated methods, so it starts empty and
        cannot mass-fail the pre-existing straight-line teardown sites in
        test_merge_queue_concurrent_verify.py that have not been migrated
        yet.
        """
        source = '''
class TestUnmigratedTeardown:
    async def test_it(self):
        await asyncio.wait_for(fut, timeout=5.0)
        await worker.stop()
'''
        result = unguarded_worker_teardown_methods(source)

        assert result == [], (
            f'Expected worker.stop() with no wait_responsive call in the '
            f'method to NOT be reported (guard scoped to migrated methods '
            f'only), got {result!r}.'
        )

    def test_wait_responsive_without_stop_is_not_reported(self) -> None:
        """A method that calls `wait_responsive` but never calls
        `worker.stop()` at all is NOT reported -- there is no stop call to
        be unguarded.
        """
        source = '''
class TestNoStopCall:
    async def test_it(self):
        await wait_responsive(fut, timeout=5.0, label='x')
'''
        result = unguarded_worker_teardown_methods(source)

        assert result == [], (
            f'Expected a method with no worker.stop() call at all to NOT '
            f'be reported, got {result!r}.'
        )

    def test_unparseable_source_returns_empty_list_without_raising(self) -> None:
        """Unparseable source returns `[]` rather than raising -- mirrors
        `item_wait_budgets`'s must-never-raise contract.
        """
        result = unguarded_worker_teardown_methods('def not valid python(:')

        assert result == [], (
            f'Expected unparseable source to return [] without raising, '
            f'got {result!r}.'
        )


# Synthetic wait and mark values for TestWaitBudgetOffenders, derived from the
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


def _offenders(source: str) -> list[WaitBudgetOffender]:
    return wait_budget_offenders(source, _exec_module(source))


class TestWaitBudgetOffenders:
    """`wait_budget_offenders` judges each pytest item's bill against the
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
        namespace = _exec_module(source)

        assert wait_budget_offenders(source, namespace) == []
        assert item_limit_sources(source, namespace) == {
            'TestCovered::test_it': LimitSource.MODULE_MARK
        }

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
        assert wait_budget_offenders(source, namespace) == [], (
            'a namespace-resolved 30.0s bill clears the ambient budget; billed '
            'unbounded it would offend under any timeout'
        )

    def test_a_class_missing_from_the_namespace_is_an_unresolved_offender(self) -> None:
        source = 'class TestGone:\n    async def test_it(self):\n' + _LIGHT_BODY
        namespace = _exec_module(source)
        del namespace['TestGone']
        offenders = wait_budget_offenders(source, namespace)

        assert [(o.item, o.limit_source, o.limit) for o in offenders] == [
            ('TestGone::test_it', LimitSource.UNRESOLVED, None)
        ], f'an item whose marks cannot be read must offend loudly; got {offenders!r}'

    def test_a_mark_with_unreadable_seconds_offends(self) -> None:
        source = (
            '@pytest.mark.timeout(timeout=None)\n'
            'class TestUnreadable:\n    async def test_it(self):\n' + _LIGHT_BODY
        )
        offenders = _offenders(source)

        assert [(o.item, o.limit) for o in offenders] == [
            ('TestUnreadable::test_it', None)
        ], f'got {offenders!r}'

    def test_unparseable_source_has_no_offenders_and_does_not_raise(self) -> None:
        assert wait_budget_offenders('def not valid python(:', {}) == []

    def test_an_offender_is_frozen(self) -> None:
        source = 'async def test_x():\n' + _HEAVY_BODY.replace('        ', '    ')
        offender = _offenders(source)[0]

        field = dataclasses.fields(offender)[0].name

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(offender, field, 'other')
