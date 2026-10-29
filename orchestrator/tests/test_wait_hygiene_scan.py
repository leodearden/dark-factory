"""Unit tests for the wait anti-pattern scans in _wait_hygiene_scan.py.

Every case drives a scan with a SYNTHETIC source string, never a real test
module, so each case is non-vacuous and does not shift when another file is
edited.
"""

from __future__ import annotations

from _wait_hygiene_scan import suppressed_result_wait_items, unguarded_worker_teardown_items


class TestSuppressedWaitScanner:
    """Unit tests for `suppressed_result_wait_items` -- the executable
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
        result = suppressed_result_wait_items(source)

        assert result == ['TestSuppressedWait::test_it'], (
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
        result = suppressed_result_wait_items(source)

        assert result == ['TestSuppressedWaitAttributeForm::test_it'], (
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
        result = suppressed_result_wait_items(source)

        assert result == ['TestBareSuppressImport::test_it'], (
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
        result = suppressed_result_wait_items(source)

        assert result == ['TestTryExceptTimeoutError::test_it'], (
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
        result = suppressed_result_wait_items(source)

        assert result == [], (
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
        result = suppressed_result_wait_items(source)

        assert result == [], (
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
        result = suppressed_result_wait_items(source)

        assert result == [], (
            f'Expected a try/except TimeoutError block with no enclosed '
            f'wait_for call in the try body to NOT be reported, '
            f'got {result!r}.'
        )

    def test_multiple_classes_and_methods_are_all_aggregated(self) -> None:
        """Offenders across MULTIPLE classes and MULTIPLE methods within
        one class are all reported, sorted -- not just the first one found.
        The single-offender cases above cannot distinguish that from an
        implementation that stops at the first hit.
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
        result = suppressed_result_wait_items(source)

        assert result == [
            'TestFirstOffender::test_one',
            'TestFirstOffender::test_two',
            'TestSecondOffender::test_only',
        ], (
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
        result = suppressed_result_wait_items(source)

        assert result == [], (
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
        result = suppressed_result_wait_items(source)

        assert result == [], (
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
        result = suppressed_result_wait_items(source)

        assert result == [], (
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
        result = suppressed_result_wait_items(source)

        assert result == [], (
            f'Expected a non-test_* method and a non-Test* class to both '
            f'be skipped, got {result!r}.'
        )

    def test_a_module_level_test_function_is_scanned(self) -> None:
        """pytest collects a module-level `test*` function as an item of its
        own, so the scan reports it under its bare name.
        """
        source = '''
async def test_x():
    with contextlib.suppress(TimeoutError):
        outcome = await asyncio.wait_for(fut, timeout=5.0)
'''
        result = suppressed_result_wait_items(source)

        assert result == ['test_x'], (
            f'Expected the module-level test_x to be reported, got {result!r}.'
        )

    def test_unparseable_source_returns_empty_list_without_raising(self) -> None:
        """Unparseable source returns `[]` rather than raising -- mirrors
        `item_wait_budgets`'s must-never-raise contract: a crash here would
        fail the whole suite over an unrelated edit to a scanned file.
        """
        result = suppressed_result_wait_items('def not valid python(:')

        assert result == [], (
            f'Expected unparseable source to return [] without raising, '
            f'got {result!r}.'
        )


class TestWorkerTeardownGuardScanner:
    """Unit tests for `unguarded_worker_teardown_items` -- the executable
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
        result = unguarded_worker_teardown_items(source)

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
        result = unguarded_worker_teardown_items(source)

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
        result = unguarded_worker_teardown_items(source)

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
        result = unguarded_worker_teardown_items(source)

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
        result = unguarded_worker_teardown_items(source)

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
        result = unguarded_worker_teardown_items(source)

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
        result = unguarded_worker_teardown_items(source)

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
        result = unguarded_worker_teardown_items(source)

        assert result == [], (
            f'Expected a method with no worker.stop() call at all to NOT '
            f'be reported, got {result!r}.'
        )

    def test_a_module_level_test_function_is_scanned(self) -> None:
        """pytest collects a module-level `test*` function as an item of its
        own, so the scan reports it under its bare name.
        """
        source = '''
async def test_x():
    await wait_responsive(fut, timeout=5.0, label='x')
    await worker.stop()
'''
        result = unguarded_worker_teardown_items(source)

        assert result == ['test_x'], (
            f'Expected the module-level test_x to be reported, got {result!r}.'
        )

    def test_unparseable_source_returns_empty_list_without_raising(self) -> None:
        """Unparseable source returns `[]` rather than raising -- mirrors
        `item_wait_budgets`'s must-never-raise contract.
        """
        result = unguarded_worker_teardown_items('def not valid python(:')

        assert result == [], (
            f'Expected unparseable source to return [] without raising, '
            f'got {result!r}.'
        )
