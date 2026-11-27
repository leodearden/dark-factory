"""Tests for the helpers in _fm_helpers.py."""

import asyncio
import concurrent.futures
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import (
    _LOADED_SCRIPT_MODULE_NAMES,
    LoopFreedomProbe,
    load_script_module,
    submit_and_resolve,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub_interceptor(
    *,
    submit_result: dict,
    resolve_result: dict,
    ticket_store_row,
) -> MagicMock:
    """Return a minimal MagicMock interceptor wired for submit_and_resolve tests."""
    interceptor = MagicMock()
    interceptor.submit_task = AsyncMock(return_value=submit_result)
    interceptor.resolve_ticket = AsyncMock(return_value=resolve_result)
    interceptor._ticket_store = MagicMock()
    interceptor._ticket_store.get = AsyncMock(return_value=ticket_store_row)
    return interceptor


# ---------------------------------------------------------------------------
# Guard tests — submit_task return-shape routing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_and_resolve_proceeds_when_ticket_present_regardless_of_other_keys():
    """submit_and_resolve must route on 'ticket' presence, not on the absence of other keys.

    Forward-compatibility test: a future evolution of submit_task may return a success dict
    that carries an additional advisory key alongside 'ticket'.  The helper must use the
    *presence of 'ticket'* to decide whether to proceed to resolve_ticket, regardless of what
    other keys appear in the dict.  A guard that required the dict to have *only* 'ticket'
    would break when an extra key is added.
    """
    expected = {'id': '99', 'title': 'Done'}
    interceptor = _make_stub_interceptor(
        submit_result={'ticket': 'tkt_x', 'unrelated_advisory_field': 'foo'},
        resolve_result={'status': 'created', 'task_id': '99'},
        ticket_store_row={'result_json': json.dumps(expected)},
    )

    result = await submit_and_resolve(interceptor, '/project', title='T')

    # Must have proceeded past the guard and returned the parsed result_json
    assert result == expected, f'expected {expected!r}, got {result!r}'
    interceptor.resolve_ticket.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'non_dict_value',
    [None, [], 'err', 42],
    ids=['none', 'list', 'str', 'int'],
)
async def test_submit_and_resolve_asserts_when_submit_result_is_non_dict(non_dict_value):
    """submit_and_resolve must raise AssertionError (not silently return) when submit_task returns a non-dict.

    submit_task is contract-bound to always return a dict; a non-dict indicates a regression in
    the production code.  The helper is test-only, so a loud AssertionError is preferred over
    silent pass-through that would mask the underlying bug.

    The error message must contain repr(submit_result) so the diagnostician
    can see exactly what was returned.

    Execution must blow up BEFORE any resolve_ticket call.
    """
    interceptor = _make_stub_interceptor(
        submit_result=non_dict_value,  # type: ignore[arg-type]
        resolve_result={},
        ticket_store_row=None,
    )

    with pytest.raises(AssertionError) as excinfo:
        await submit_and_resolve(interceptor, '/project', title='T')

    message = str(excinfo.value)
    assert repr(non_dict_value) in message, (
        f"repr of non-dict value {non_dict_value!r} not in error message: {message!r}"
    )
    interceptor.resolve_ticket.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_and_resolve_returns_submit_error_when_no_ticket():
    """submit_and_resolve must return the submit-error dict verbatim and NOT call resolve_ticket.

    This is the current-production behavior: submit_task returns an error dict with no 'ticket'
    key (e.g. backlog gate rejection or closed-server error) and the helper passes it through
    immediately.  All callers in test_task_interceptor.py and test_backlog_policy.py assert on
    result.get('error') / result.get('error_type') in this path.
    """
    submit_result = {'error': 'closed', 'error_type': 'ClosedError'}
    interceptor = _make_stub_interceptor(
        submit_result=submit_result,
        resolve_result={},
        ticket_store_row=None,
    )

    result = await submit_and_resolve(interceptor, '/project', title='T')

    assert result == submit_result, f'expected submit-error dict {submit_result!r}, got {result!r}'
    interceptor.resolve_ticket.assert_not_awaited()


# ---------------------------------------------------------------------------
# Step-3: fallback test — helper must raise AssertionError when no result_json
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    'row_value',
    [None, {}, {'result_json': ''}],
    ids=['row-none', 'row-empty', 'result-json-empty'],
)
async def test_submit_and_resolve_raises_when_no_result_json(row_value):
    """submit_and_resolve must raise AssertionError (not silently return resolve_result) when
    the worker never wrote a result_json.

    The AssertionError message must name the ticket id and include enough of resolve_result
    to diagnose the root cause.  Currently fails because line 178 silently returns resolve_result.
    """
    resolve_result = {'status': 'failed', 'reason': 'timeout', 'task_id': None}
    interceptor = _make_stub_interceptor(
        submit_result={'ticket': 'tkt_y'},
        resolve_result=resolve_result,
        ticket_store_row=row_value,
    )

    with pytest.raises(AssertionError) as excinfo:
        await submit_and_resolve(interceptor, '/project', title='T')

    message = str(excinfo.value)
    assert 'tkt_y' in message, f"ticket id 'tkt_y' not in error message: {message!r}"
    # resolve_result must be dumped verbatim so the diagnostician can see what the worker returned.
    # Check for the structural marker 'resolve_result=' (from the format string) AND 'failed'
    # (from resolve_result['status']), rather than a coincidental inner-value substring.
    assert 'resolve_result=' in message and 'failed' in message, (
        f"resolve_result not surfaced in error message: {message!r}"
    )


# ---------------------------------------------------------------------------
# Tests for 8df8bdcd shared scenario builder + parse helpers
# ---------------------------------------------------------------------------
# Helpers are imported lazily inside each test (not at module level) so that
# a future missing or renamed name surfaces as a localized test failure
# rather than a collection error that breaks unrelated tests.

# Invalid 'kind' values shared by the two ValueError parametrize tests below.
# Keep in one place so the contract stays in sync if spelling/cases change.
_INVALID_KINDS = ['actve', '', 'Active', 'provenence']


class TestMake8df8Scenario:
    """Unit tests for the shared make_8df8_scenario() builder."""

    @pytest.mark.parametrize('id_type', [int, str])
    @pytest.mark.parametrize('status', ['done', 'in-progress', 'pending'])
    def test_builder_returns_canonical_ids_and_titles(self, id_type, status):
        """builder produces tasks for ids 1369/1355/1361 with distinct canonical titles."""
        from _fm_helpers import make_8df8_scenario

        tasks, title_by_id = make_8df8_scenario(id_type=id_type, status=status)

        # Three tasks
        assert len(tasks) == 3, f'Expected 3 tasks, got {len(tasks)}'

        # List order is completion order: 1369, 1355, 1361 (NOT id-sort order 1355,1361,1369)
        ids = [id_type(t['id']) for t in tasks]
        assert ids == [id_type(1369), id_type(1355), id_type(1361)], (
            f'Completion order must be 1369→1355→1361, got {ids}'
        )

        # All three expected ids present
        assert set(ids) == {id_type(1369), id_type(1355), id_type(1361)}

    @pytest.mark.parametrize('id_type', [int, str])
    @pytest.mark.parametrize('status', ['done', 'in-progress', 'pending'])
    def test_builder_titles_are_distinct(self, id_type, status):
        """builder gives each task a DISTINCT title (no shared/copied titles)."""
        from _fm_helpers import make_8df8_scenario

        tasks, title_by_id = make_8df8_scenario(id_type=id_type, status=status)

        titles = [t['title'] for t in tasks]
        assert len(set(titles)) == 3, (
            f'All titles must be distinct, got {titles}'
        )

    @pytest.mark.parametrize('id_type', [int, str])
    @pytest.mark.parametrize('status', ['done', 'in-progress', 'pending'])
    def test_builder_title_by_id_keys_match_id_type(self, id_type, status):
        """title_by_id keys have the requested id_type (int or str)."""
        from _fm_helpers import make_8df8_scenario

        tasks, title_by_id = make_8df8_scenario(id_type=id_type, status=status)

        for key in title_by_id:
            assert isinstance(key, id_type), (
                f'title_by_id key {key!r} is {type(key).__name__}, expected {id_type.__name__}'
            )

    @pytest.mark.parametrize('id_type', [int, str])
    @pytest.mark.parametrize('status', ['done', 'in-progress', 'pending'])
    def test_builder_title_by_id_matches_task_list(self, id_type, status):
        """title_by_id[id] == task['title'] for every task in the list."""
        from _fm_helpers import make_8df8_scenario

        tasks, title_by_id = make_8df8_scenario(id_type=id_type, status=status)

        for task in tasks:
            tid = id_type(task['id'])
            assert tid in title_by_id, f'Key {tid!r} missing from title_by_id'
            assert title_by_id[tid] == task['title'], (
                f'title_by_id[{tid!r}]={title_by_id[tid]!r} != task title={task["title"]!r}'
            )

    @pytest.mark.parametrize('id_type', [int, str])
    @pytest.mark.parametrize('status', ['done', 'in-progress', 'pending'])
    def test_builder_status_applied(self, id_type, status):
        """All tasks have the requested status."""
        from _fm_helpers import make_8df8_scenario

        tasks, _ = make_8df8_scenario(id_type=id_type, status=status)

        for task in tasks:
            assert task['status'] == status, (
                f'Task {task["id"]}: status={task["status"]!r}, expected {status!r}'
            )

    def test_builder_with_provenance_attaches_done_provenance(self):
        """with_provenance=True attaches metadata.done_provenance matching test_stages.py fixture."""
        from _fm_helpers import make_8df8_scenario

        tasks, _ = make_8df8_scenario(id_type=int, status='done', with_provenance=True)

        task_by_id = {t['id']: t for t in tasks}

        # id=1369: commit branch
        assert 'metadata' in task_by_id[1369], 'Task 1369 must have metadata'
        prov_1369 = task_by_id[1369]['metadata']['done_provenance']
        assert prov_1369.get('commit') == 'abc123deadbeef', (
            f'1369 commit mismatch: {prov_1369}'
        )

        # id=1355: note branch
        assert 'metadata' in task_by_id[1355], 'Task 1355 must have metadata'
        prov_1355 = task_by_id[1355]['metadata']['done_provenance']
        assert prov_1355.get('note') == 'Covered by sibling task 1354', (
            f'1355 note mismatch: {prov_1355}'
        )

        # id=1361: legacy/none (no metadata.done_provenance key)
        assert 'metadata' not in task_by_id[1361] or 'done_provenance' not in task_by_id[1361].get('metadata', {}), (
            f'Task 1361 should NOT have done_provenance (legacy branch), got: {task_by_id[1361]}'
        )

    def test_builder_without_provenance_has_no_done_provenance(self):
        """Default (with_provenance=False): no task has metadata.done_provenance."""
        from _fm_helpers import make_8df8_scenario

        tasks, _ = make_8df8_scenario(id_type=int, status='done')

        for task in tasks:
            meta = task.get('metadata', {})
            assert 'done_provenance' not in meta, (
                f'Task {task["id"]}: unexpected done_provenance in {meta}'
            )


class TestLineRegexConstants:
    """ID_TITLE_LINE_RE and PROVENANCE_LINE_RE constants exist and parse correctly."""

    def test_id_title_line_re_matches_active_line(self):
        """ID_TITLE_LINE_RE matches a typical active-task rendered line."""
        from _fm_helpers import ID_TITLE_LINE_RE

        line = '- [1369] (in-progress) Refactor event dispatch to async deps=[]'
        m = ID_TITLE_LINE_RE.search(line)
        assert m is not None, f'ID_TITLE_LINE_RE did not match: {line!r}'
        assert m.group(1) == '1369'
        assert 'Refactor event dispatch to async' in m.group(2)

    def test_provenance_line_re_matches_em_dash_line(self):
        """PROVENANCE_LINE_RE matches a legacy-branch provenance line with em-dash."""
        from _fm_helpers import PROVENANCE_LINE_RE

        line = '- [1361] Add retry logic for database connections — provenance: unknown'
        m = PROVENANCE_LINE_RE.search(line)
        assert m is not None, f'PROVENANCE_LINE_RE did not match: {line!r}'
        assert m.group(1) == '1361'
        assert 'Add retry logic for database connections' in m.group(2)

    def test_provenance_line_re_matches_eol_line(self):
        """PROVENANCE_LINE_RE matches a commit-branch provenance line ending at EOL."""
        from _fm_helpers import PROVENANCE_LINE_RE

        line = '- [1369] Refactor event dispatch to async'
        m = PROVENANCE_LINE_RE.search(line)
        assert m is not None, f'PROVENANCE_LINE_RE did not match: {line!r}'
        assert m.group(1) == '1369'
        assert 'Refactor event dispatch to async' in m.group(2)


class TestParseRenderedIdTitlePairs:
    """parse_rendered_id_title_pairs extracts {id: title} from rendered output."""

    def test_parse_active_extracts_correct_pairs(self):
        """kind='active' extracts int id → title pairs from active-task lines."""
        from _fm_helpers import parse_rendered_id_title_pairs

        rendered = (
            '- [1369] (in-progress) Refactor event dispatch to async deps=[]\n'
            '- [1355] (in-progress) Implement rate limiter middleware deps=[]\n'
            '- [1361] (in-progress) Add retry logic for database connections deps=[]\n'
        )
        found = parse_rendered_id_title_pairs(rendered, kind='active')

        assert set(found.keys()) == {1369, 1355, 1361}
        assert found[1369] == 'Refactor event dispatch to async'
        assert found[1355] == 'Implement rate limiter middleware'
        assert found[1361] == 'Add retry logic for database connections'

    def test_parse_provenance_extracts_correct_pairs(self):
        """kind='provenance' extracts int id → title pairs from provenance-section lines."""
        from _fm_helpers import parse_rendered_id_title_pairs

        rendered = (
            '### Done-task Provenance\n'
            '- [1369] Refactor event dispatch to async\n'
            '  commit: abc123deadbeef\n'
            '- [1361] Add retry logic for database connections — provenance: unknown (legacy)\n'
        )
        found = parse_rendered_id_title_pairs(rendered, kind='provenance')

        assert 1369 in found, f'id 1369 missing from {found}'
        assert 1361 in found, f'id 1361 missing from {found}'
        assert found[1369] == 'Refactor event dispatch to async'
        assert found[1361] == 'Add retry logic for database connections'

    def test_parse_active_returns_empty_when_no_matches(self):
        """parse_rendered_id_title_pairs returns {} when no active-task lines are present."""
        from _fm_helpers import parse_rendered_id_title_pairs

        found = parse_rendered_id_title_pairs('No tasks here.', kind='active')
        assert found == {}

    def test_parse_provenance_returns_empty_when_no_matches(self):
        """parse_rendered_id_title_pairs returns {} when no provenance lines are present."""
        from _fm_helpers import parse_rendered_id_title_pairs

        found = parse_rendered_id_title_pairs('No tasks here.', kind='provenance')
        assert found == {}

    @pytest.mark.parametrize('bad_kind', _INVALID_KINDS)
    def test_parse_raises_value_error_for_invalid_kind(self, bad_kind):
        """parse_rendered_id_title_pairs raises ValueError for any kind that is not
        'active' or 'provenance', and the error message includes repr(bad_kind)."""
        from _fm_helpers import parse_rendered_id_title_pairs

        with pytest.raises(ValueError) as excinfo:
            parse_rendered_id_title_pairs('- [1369] Some title\n', kind=bad_kind)

        assert repr(bad_kind) in str(excinfo.value), (
            f"repr({bad_kind!r}) not found in error message: {excinfo.value!r}"
        )

    def test_parse_valid_kinds_do_not_raise(self):
        """parse_rendered_id_title_pairs does NOT raise for the two valid kinds,
        and returns {} for input with no matching lines (empty-input contract)."""
        from _fm_helpers import parse_rendered_id_title_pairs

        assert parse_rendered_id_title_pairs('No tasks.', kind='active') == {}
        assert parse_rendered_id_title_pairs('No tasks.', kind='provenance') == {}

    @pytest.mark.parametrize('bad_kind', _INVALID_KINDS)
    def test_assert_id_title_pairing_raises_value_error_transitively(self, bad_kind):
        """assert_id_title_pairing fires ValueError transitively for invalid kind
        (it delegates to parse_rendered_id_title_pairs, so the guard covers both)."""
        from _fm_helpers import assert_id_title_pairing

        with pytest.raises(ValueError) as excinfo:
            assert_id_title_pairing(
                '- [1369] Some title\n',
                {1369: 'Some title'},
                kind=bad_kind,
            )

        assert repr(bad_kind) in str(excinfo.value), (
            f"repr({bad_kind!r}) not found in error message: {excinfo.value!r}"
        )


class TestAssertIdTitlePairing:
    """assert_id_title_pairing raises on mismatches and passes on correct renders."""

    def test_passes_on_correct_active_render(self):
        """assert_id_title_pairing succeeds when every id pairs with its OWN title."""
        from _fm_helpers import assert_id_title_pairing

        rendered = (
            '- [1369] (in-progress) Refactor event dispatch to async deps=[]\n'
            '- [1355] (in-progress) Implement rate limiter middleware deps=[]\n'
            '- [1361] (in-progress) Add retry logic for database connections deps=[]\n'
        )
        title_by_id = {
            1369: 'Refactor event dispatch to async',
            1355: 'Implement rate limiter middleware',
            1361: 'Add retry logic for database connections',
        }
        # Should NOT raise
        assert_id_title_pairing(rendered, title_by_id, kind='active')

    def test_raises_on_zero_matches_anti_vacuity(self):
        """assert_id_title_pairing raises AssertionError on zero matches (anti-vacuity guard)."""
        from _fm_helpers import assert_id_title_pairing

        title_by_id = {
            1369: 'Refactor event dispatch to async',
            1355: 'Implement rate limiter middleware',
            1361: 'Add retry logic for database connections',
        }
        # Rendered output with no task lines → vacuous
        with pytest.raises(AssertionError, match='[Nn]o.*match|zero.*match|matched.*nothing|vacuous|empty'):
            assert_id_title_pairing('No tasks.', title_by_id, kind='active')

    def test_raises_on_neighbor_title_swap(self):
        """assert_id_title_pairing raises AssertionError when a task carries a neighbor's title."""
        from _fm_helpers import assert_id_title_pairing

        # Task 1369 renders with 1355's title (the cycle-8df8bdcd bug pattern)
        rendered = (
            '- [1369] (in-progress) Implement rate limiter middleware deps=[]\n'
            '- [1355] (in-progress) Refactor event dispatch to async deps=[]\n'
            '- [1361] (in-progress) Add retry logic for database connections deps=[]\n'
        )
        title_by_id = {
            1369: 'Refactor event dispatch to async',
            1355: 'Implement rate limiter middleware',
            1361: 'Add retry logic for database connections',
        }
        with pytest.raises(AssertionError):
            assert_id_title_pairing(rendered, title_by_id, kind='active')

    def test_raises_on_unknown_id_when_expected_ids_not_given(self):
        """assert_id_title_pairing fails when a rendered id is absent from title_by_id
        and expected_ids is not provided — guards the latent stray/bleed-id footgun."""
        from _fm_helpers import assert_id_title_pairing

        # Rendered output includes id 9999 which is NOT in title_by_id.
        rendered = (
            '- [1369] (in-progress) Refactor event dispatch to async deps=[]\n'
            '- [9999] (in-progress) Stray task from another context deps=[]\n'
        )
        title_by_id = {
            1369: 'Refactor event dispatch to async',
        }
        with pytest.raises(AssertionError, match='9999.*absent|absent.*9999|stray|bleed'):
            assert_id_title_pairing(rendered, title_by_id, kind='active')


# ---------------------------------------------------------------------------
# Tests for the shared LoopFreedomProbe oracle (task 5920)
# ---------------------------------------------------------------------------

#: Only a held loop ever reaches the ceiling, so shortening it cannot turn a
#: False verdict True; it only keeps the held-loop cases fast.
_HELD_LOOP_CEILING_SECONDS = 0.01


def _drive_without_yielding_to_loop(coro) -> None:
    with pytest.raises(StopIteration):
        while True:
            coro.send(None)


def _run_on_a_worker_while_the_loop_thread_waits(fn, *args) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(fn, *args).result()


class TestLoopFreedomProbe:
    """Unit tests for LoopFreedomProbe.suspend() / .block() / .observations / .assert_loop_stayed_free()."""

    @pytest.mark.asyncio
    async def test_a_free_loop_reads_true(self):
        probe = LoopFreedomProbe()

        await probe.suspend()

        assert probe.observations == (True,)
        probe.assert_loop_stayed_free()

    @pytest.mark.asyncio
    async def test_a_call_site_that_resumes_the_coroutine_itself_reads_false(self):
        probe = LoopFreedomProbe()

        _drive_without_yielding_to_loop(probe.suspend())

        assert probe.observations == (False,)

    @pytest.mark.asyncio
    async def test_a_call_site_waiting_on_another_threads_loop_reads_false(self):
        probe = LoopFreedomProbe(ceiling_seconds=_HELD_LOOP_CEILING_SECONDS)

        _run_on_a_worker_while_the_loop_thread_waits(asyncio.run, probe.suspend())

        assert probe.observations == (False,)

    @pytest.mark.asyncio
    async def test_a_call_site_awaiting_another_threads_loop_reads_true(self):
        probe = LoopFreedomProbe()

        await asyncio.to_thread(asyncio.run, probe.suspend())

        assert probe.observations == (True,)

    @pytest.mark.asyncio
    async def test_a_blocking_call_offloaded_from_the_loop_reads_true(self):
        probe = LoopFreedomProbe()

        await asyncio.to_thread(probe.block)

        assert probe.observations == (True,)

    @pytest.mark.asyncio
    async def test_a_blocking_call_on_the_loop_thread_reads_false(self):
        probe = LoopFreedomProbe()

        probe.block()

        assert probe.observations == (False,)

    @pytest.mark.asyncio
    async def test_a_blocking_call_the_loop_thread_waits_on_reads_false(self):
        probe = LoopFreedomProbe(ceiling_seconds=_HELD_LOOP_CEILING_SECONDS)

        _run_on_a_worker_while_the_loop_thread_waits(probe.block)

        assert probe.observations == (False,)

    @pytest.mark.asyncio
    async def test_assert_loop_stayed_free_rejects_an_unreached_probe(self):
        probe = LoopFreedomProbe()

        with pytest.raises(AssertionError, match='never reached'):
            probe.assert_loop_stayed_free()

    @pytest.mark.asyncio
    async def test_assert_loop_stayed_free_rejects_a_held_loop(self):
        probe = LoopFreedomProbe()
        _drive_without_yielding_to_loop(probe.suspend())

        with pytest.raises(AssertionError, match='held'):
            probe.assert_loop_stayed_free()

    @pytest.mark.asyncio
    async def test_each_suspension_is_recorded(self):
        probe = LoopFreedomProbe()

        await probe.suspend()
        await probe.suspend()

        assert probe.observations == (True, True)


# ---------------------------------------------------------------------------
# Tests for the shared load_script_module() loader (task 3738)
# ---------------------------------------------------------------------------
# load_script_module imports a script that is NOT a package member and NOT on
# PYTHONPATH (anything under `scripts/`) by file path, so its pure functions
# can be tested without sys.path pollution. It was hoisted out of
# test_sweep_toolcall_xml_leak.py and test_toolcall_xml_leak_sweep_artifacts.py,
# which carried two independent copies of it.
#
# The REUSE guard is the reason the more-robust of those two copies was the one
# promoted, and it is precisely the behaviour nothing pinned: both sweep modules
# register the SAME sys.modules key, so an unconditional second load would
# re-execute the 965-line script and silently replace the first module object,
# leaving two live module objects whose identity depends on collection order.
# That guard is currently exercised only INCIDENTALLY, as a side effect of
# collection order across those two modules — drop it and both still pass. These
# tests make it load-bearing directly.
#
# Everything here drives throwaway tmp_path scripts, never a real repo script,
# so no case depends on the content of anything under scripts/.
#
# load_script_module is imported at module level (with submit_and_resolve),
# deliberately NOT lazily per-test the way the 8df8 section above does it:
# every test in this section needs it, so seven copies of one import line bought
# nothing.

@pytest.fixture
def registered_module_names():
    """Yield a registrar for sys.modules keys a test is about to create.

    Every registered key is popped afterwards, pass or fail, so a test that
    loads a throwaway tmp_path script cannot leak a module object into sibling
    tests (or shadow a real module for the rest of the session).

    The helper's own record of which keys it installed is cleared for the same
    keys, so a tmp_path name cannot arrive at a sibling test already marked as
    helper-owned and quietly disarm the shadowing guard there.
    """
    names: list[str] = []
    yield names.append
    for name in names:
        sys.modules.pop(name, None)
        _LOADED_SCRIPT_MODULE_NAMES.discard(name)


class TestLoadScriptModule:
    """Unit tests for load_script_module(script_path, mod_name=None)."""

    def test_module_level_names_are_reachable_on_the_returned_module(
        self, tmp_path, registered_module_names
    ):
        """The whole point: a by-path script's constants and functions come back
        as ordinary attributes of the returned module."""
        script = tmp_path / 'probe_script.py'
        script.write_text('VALUE = 41\n\n\ndef bump(n):\n    return n + 1\n')
        registered_module_names('probe_script')

        mod = load_script_module(script)

        assert mod.VALUE == 41
        assert mod.bump(41) == 42

    def test_registers_under_the_explicit_mod_name(
        self, tmp_path, registered_module_names
    ):
        """An explicit mod_name is the sys.modules key — this is what lets two
        test modules deliberately SHARE one key for the same script."""
        script = tmp_path / 'stem_is_ignored.py'
        script.write_text("ORIGIN = 'explicit'\n")
        registered_module_names('chosen_key')

        mod = load_script_module(script, mod_name='chosen_key')

        assert sys.modules['chosen_key'] is mod
        assert 'stem_is_ignored' not in sys.modules

    def test_registers_under_the_file_stem_when_mod_name_is_omitted(
        self, tmp_path, registered_module_names
    ):
        """mod_name defaults to the file stem."""
        script = tmp_path / 'defaults_to_stem.py'
        script.write_text("ORIGIN = 'stem'\n")
        registered_module_names('defaults_to_stem')

        mod = load_script_module(script)

        assert sys.modules['defaults_to_stem'] is mod

    def test_a_second_load_of_the_same_file_reuses_the_module_object(
        self, tmp_path, registered_module_names
    ):
        """REUSE — the guard this loader exists for.

        The second call must return the IDENTICAL object without re-executing
        the file. Asserted by mutating the first module and looking for the
        mutation afterwards: a re-execution would hand back a freshly built
        module object with no SENTINEL on it, so `is`-identity plus the
        surviving attribute together rule out both replacement and re-exec.
        """
        script = tmp_path / 'reused_script.py'
        script.write_text('VALUE = 1\n')
        registered_module_names('reused_script')

        first = load_script_module(script)
        first.SENTINEL = 'set-by-first-caller'  # type: ignore[attr-defined]

        second = load_script_module(script)

        assert second is first
        assert second.SENTINEL == 'set-by-first-caller'

    def test_reuse_survives_a_non_normalised_path_to_the_same_file(
        self, tmp_path, registered_module_names
    ):
        """The guard compares RESOLVED paths, so an unnormalised spelling of the
        same file still reuses.

        Callers build these paths with `Path(__file__).parent.parent / ...`, so
        the second caller's spelling need not be string-equal to the first's.
        An implementation comparing the cached ``__file__`` as a raw string
        would re-execute here.
        """
        subdir = tmp_path / 'subdir'
        subdir.mkdir()
        script = tmp_path / 'roundabout_script.py'
        script.write_text('VALUE = 1\n')
        registered_module_names('roundabout_script')

        first = load_script_module(script)
        first.SENTINEL = 'set-by-first-caller'  # type: ignore[attr-defined]

        second = load_script_module(subdir / '..' / 'roundabout_script.py')

        assert second is first
        assert second.SENTINEL == 'set-by-first-caller'

    def test_a_different_file_under_a_used_mod_name_is_not_reused(
        self, tmp_path, registered_module_names
    ):
        """The guard keys on the FILE, not on the name.

        Reusing on a name match alone would hand the second caller the first
        script's module — silently serving the wrong code under the right key.

        Replacement is allowed here because the helper installed that key
        itself; the shadowing test below covers the case where it did not.
        """
        first_script = tmp_path / 'first_source.py'
        first_script.write_text("ORIGIN = 'first'\n")
        second_script = tmp_path / 'second_source.py'
        second_script.write_text("ORIGIN = 'second'\n")
        registered_module_names('shared_key')

        first = load_script_module(first_script, mod_name='shared_key')
        assert first.ORIGIN == 'first'

        second = load_script_module(second_script, mod_name='shared_key')

        assert second is not first
        assert second.ORIGIN == 'second'
        assert sys.modules['shared_key'] is second

    def test_an_exception_during_exec_propagates_and_leaves_no_partial_entry(
        self, tmp_path, registered_module_names
    ):
        """A script that blows up mid-exec must not leave its half-built module
        registered.

        The module is inserted into sys.modules BEFORE exec_module (so the
        script can import itself), which means the failure path has to undo
        that. Otherwise the next caller's reuse check would find the cached
        entry, match its __file__, and return a module whose top level never
        finished running — a partially-initialised module served as a good one.
        """
        script = tmp_path / 'exploding_script.py'
        script.write_text("LOADED_SO_FAR = 1\nraise RuntimeError('boom during exec')\n")
        registered_module_names('exploding_script')

        with pytest.raises(RuntimeError, match='boom during exec'):
            load_script_module(script)

        assert 'exploding_script' not in sys.modules

    def test_a_module_the_helper_did_not_install_is_never_shadowed(
        self, tmp_path, registered_module_names
    ):
        """A key some OTHER importer owns must raise, not be overwritten.

        mod_name defaults to the file stem, and `scripts/` is full of names
        that are also ordinary module names (config.py, utils.py, types.py).
        Silently replacing `sys.modules['config']` with a script would persist
        for the rest of the pytest process and surface as an unrelated test
        failing far away, so the helper refuses and says which name collided.
        """
        script = tmp_path / 'shadow_target.py'
        script.write_text("ORIGIN = 'the script'\n")
        registered_module_names('shadow_target')
        pre_existing = types.ModuleType('shadow_target')
        pre_existing.__file__ = str(tmp_path / 'somewhere_else.py')
        pre_existing.ORIGIN = 'the real module'  # type: ignore[attr-defined]
        sys.modules['shadow_target'] = pre_existing

        with pytest.raises(ImportError, match='refusing to shadow'):
            load_script_module(script)

        assert sys.modules['shadow_target'] is pre_existing
        assert sys.modules['shadow_target'].ORIGIN == 'the real module'

    def test_a_foreign_module_without_a_file_attribute_is_also_not_shadowed(
        self, tmp_path, registered_module_names
    ):
        """Namespace packages and C/builtin modules have no usable __file__.

        The reuse check reads __file__ to decide whether the cached entry IS
        this script; a None there must fall through to the ownership guard
        rather than to a replacement.
        """
        script = tmp_path / 'fileless_target.py'
        script.write_text("ORIGIN = 'the script'\n")
        registered_module_names('fileless_target')
        pre_existing = types.ModuleType('fileless_target')
        sys.modules['fileless_target'] = pre_existing

        with pytest.raises(ImportError, match='refusing to shadow'):
            load_script_module(script, mod_name='fileless_target')

        assert sys.modules['fileless_target'] is pre_existing

    def test_ownership_lapses_when_a_load_fails_mid_exec(
        self, tmp_path, registered_module_names
    ):
        """A failed load leaves nothing installed, so it must leave no claim.

        Otherwise the helper would still consider the name its own, and a
        genuine `import <name>` landing in between would be replaced by the
        next by-path load — the exact shadowing the guard exists to stop,
        reachable through a failed load.
        """
        exploding = tmp_path / 'lapsed_key.py'
        exploding.write_text("raise RuntimeError('boom during exec')\n")
        script = tmp_path / 'later_script.py'
        script.write_text("ORIGIN = 'the script'\n")
        registered_module_names('lapsed_key')

        with pytest.raises(RuntimeError, match='boom during exec'):
            load_script_module(exploding)

        pre_existing = types.ModuleType('lapsed_key')
        pre_existing.__file__ = str(tmp_path / 'a_real_module.py')
        sys.modules['lapsed_key'] = pre_existing

        with pytest.raises(ImportError, match='refusing to shadow'):
            load_script_module(script, mod_name='lapsed_key')

        assert sys.modules['lapsed_key'] is pre_existing

    def test_a_path_with_no_importable_loader_raises_import_error(self, tmp_path):
        """The helper's only explicit error contract.

        spec_from_file_location returns None when no loader claims the suffix,
        and the helper must turn that into a named ImportError rather than an
        AttributeError on None further down.
        """
        with pytest.raises(ImportError, match='Cannot load'):
            load_script_module(tmp_path / 'not_a_module.txt')
