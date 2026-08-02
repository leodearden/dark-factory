"""Tests for the submit_and_resolve helper in _fm_helpers.py."""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from _fm_helpers import submit_and_resolve

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
# Tests for shared Qdrant test scaffolding (task 2277)
# ---------------------------------------------------------------------------
# De-duplicates the _qdrant_available()/QDRANT_URL/skip-marker scaffolding that
# was previously copy-pasted verbatim in test_recon_dedup_premise.py and
# test_mem0_qdrant_integration.py (flagged by esc-2221-26). No real Qdrant is
# used here — qdrant_client.QdrantClient is monkeypatched throughout.

class TestQdrantHelpers:
    """Unit tests for QDRANT_URL, _qdrant_available(), and qdrant_skipif()."""

    def test_qdrant_url_constant(self):
        """QDRANT_URL matches the local-Qdrant default both integration files used."""
        from _fm_helpers import QDRANT_URL

        assert QDRANT_URL == 'http://localhost:6333'

    def test_qdrant_available_true_when_client_ok(self, monkeypatch):
        """_qdrant_available() returns True when construction, get_collections, and close all succeed."""
        from _fm_helpers import _qdrant_available

        fake_client = MagicMock()
        fake_client.get_collections.return_value = None
        fake_client.close.return_value = None
        monkeypatch.setattr('qdrant_client.QdrantClient', MagicMock(return_value=fake_client))

        assert _qdrant_available() is True
        fake_client.get_collections.assert_called_once()
        fake_client.close.assert_called_once()

    def test_qdrant_available_false_when_get_collections_raises(self, monkeypatch):
        """_qdrant_available() returns False (never raises) when get_collections() raises."""
        from _fm_helpers import _qdrant_available

        fake_client = MagicMock()
        fake_client.get_collections.side_effect = Exception('boom')
        monkeypatch.setattr('qdrant_client.QdrantClient', MagicMock(return_value=fake_client))

        assert _qdrant_available() is False

    def test_qdrant_available_false_when_construct_raises(self, monkeypatch):
        """_qdrant_available() returns False (never raises) when QdrantClient(...) itself raises."""
        from _fm_helpers import _qdrant_available

        def _boom(*args, **kwargs):
            raise Exception('connection refused')

        monkeypatch.setattr('qdrant_client.QdrantClient', _boom)

        assert _qdrant_available() is False

    def test_qdrant_skipif_default_reason(self):
        """qdrant_skipif() returns a pytest.mark.skipif marker with a non-empty default reason."""
        from _fm_helpers import qdrant_skipif

        m = qdrant_skipif()

        assert m.name == 'skipif'
        assert isinstance(m.kwargs['reason'], str) and m.kwargs['reason']

    def test_qdrant_skipif_custom_reason(self):
        """qdrant_skipif(reason=...) propagates a caller-supplied reason string."""
        from _fm_helpers import qdrant_skipif

        assert qdrant_skipif(reason='x').kwargs['reason'] == 'x'

    def test_qdrant_skipif_condition_tracks_probe(self, monkeypatch):
        """qdrant_skipif()'s skip condition is `not _qdrant_available()`, evaluated at call time.

        Pins the behaviour-preserving contract: the marker must track whatever
        _qdrant_available() currently reports, not a value cached at import time.
        """
        import _fm_helpers

        monkeypatch.setattr(_fm_helpers, '_qdrant_available', lambda: False)
        assert _fm_helpers.qdrant_skipif().args[0] is True

        monkeypatch.setattr(_fm_helpers, '_qdrant_available', lambda: True)
        assert _fm_helpers.qdrant_skipif().args[0] is False


# ---------------------------------------------------------------------------
# Tests for the shared poll_until() helper (task 2377)
# ---------------------------------------------------------------------------
# Converts fixed "sleep(N) then assert background work is done" sites (fragile
# under host CPU oversubscription) into a bounded poll on the same
# post-condition. These tests pin the contract using ONLY predicate-call-count
# assertions and raise assertions — never wall-clock-elapsed assertions — so
# this suite stays load-independent and does not reintroduce the very
# fragility class poll_until exists to remove.

class TestPollUntil:
    """Unit tests for the shared poll_until(predicate, *, timeout, interval, message) helper."""

    @pytest.mark.asyncio
    async def test_already_true_sync_predicate_returns_after_one_call(self):
        """An already-true sync predicate returns immediately after exactly 1 call (check-before-sleep)."""
        from _fm_helpers import poll_until

        calls = 0

        def predicate():
            nonlocal calls
            calls += 1
            return True

        result = await poll_until(predicate, timeout=5.0)

        assert result is True
        assert calls == 1, f'expected exactly 1 call, got {calls}'

    @pytest.mark.asyncio
    async def test_sync_predicate_flips_true_on_nth_call(self):
        """A sync predicate that flips true on the Nth call returns after exactly N calls.

        Proves check-before-sleep and that polling stops as soon as the
        post-condition holds, rather than over-polling or racing ahead.
        """
        from _fm_helpers import poll_until

        calls = 0
        n = 4

        def predicate():
            nonlocal calls
            calls += 1
            return calls >= n

        result = await poll_until(predicate, timeout=5.0, interval=0.001)

        assert result is True
        assert calls == n, f'expected exactly {n} calls, got {calls}'

    @pytest.mark.asyncio
    async def test_async_predicate_is_awaited_and_flips_true_on_nth_call(self):
        """An async/coroutine predicate is awaited and behaves identically to the sync case.

        If poll_until failed to await the coroutine, the (always-truthy)
        coroutine object itself would satisfy the truthiness check on the
        first call and `calls` would remain 0 — so this also proves the
        coroutine body actually ran.
        """
        from _fm_helpers import poll_until

        calls = 0
        n = 3

        async def predicate():
            nonlocal calls
            calls += 1
            return calls >= n

        result = await poll_until(predicate, timeout=5.0, interval=0.001)

        assert result is True
        assert calls == n, f'expected exactly {n} calls, got {calls}'

    @pytest.mark.asyncio
    async def test_return_value_is_truthy_predicate_result_verbatim(self):
        """poll_until returns the predicate's truthy result verbatim, not coerced to True.

        Lets callers chain on the returned value (e.g. a stats dict) instead
        of re-fetching state after the poll succeeds.
        """
        from _fm_helpers import poll_until

        payload = {'done': True, 'value': 42}

        def predicate():
            return payload

        result = await poll_until(predicate, timeout=5.0)

        assert result is payload, f'expected the exact object {payload!r}, got {result!r}'

    @pytest.mark.asyncio
    async def test_never_true_predicate_raises_assertion_error_at_deadline(self):
        """A never-true predicate raises AssertionError once the deadline passes.

        Uses a tiny timeout and asserts only the raise (never elapsed time) —
        a slow host makes this test slower, never falsely passing or failing.
        """
        from _fm_helpers import poll_until

        with pytest.raises(AssertionError):
            await poll_until(lambda: False, timeout=0.05, interval=0.01)

    @pytest.mark.asyncio
    async def test_caller_message_appears_in_raised_assertion_error(self):
        """A caller-supplied `message` appears verbatim in the raised AssertionError."""
        from _fm_helpers import poll_until

        with pytest.raises(AssertionError, match='custom diagnostic xyz'):
            await poll_until(
                lambda: False, timeout=0.05, interval=0.01, message='custom diagnostic xyz',
            )

    @pytest.mark.asyncio
    async def test_default_message_names_the_timeout(self):
        """When message=None, the default AssertionError message names the timeout value."""
        from _fm_helpers import poll_until

        with pytest.raises(AssertionError, match='0.05'):
            await poll_until(lambda: False, timeout=0.05, interval=0.01)


# ---------------------------------------------------------------------------
# Tests for the shared ensure_fresh_collection() helper (task 2773, item 3)
# ---------------------------------------------------------------------------
# Idempotent + 409-tolerant collection (re)creation, used by the qdrant
# integration fixture. No real Qdrant is used here — the client is a
# MagicMock and qdrant_client.http.exceptions.UnexpectedResponse is
# constructed directly — so this suite runs in the default merge-verify lane
# even though its one live consumer (the qdrant fixture) is integration-only.
# Assertions are on call counts and ordering only, never on live Qdrant.

class TestEnsureFreshCollection:
    """Unit tests for ensure_fresh_collection(client, collection_name, *, size, distance)."""

    def test_creates_once_when_absent(self):
        """collection_exists()->False: create_collection called exactly once, no delete."""
        from _fm_helpers import ensure_fresh_collection

        client = MagicMock()
        client.collection_exists.return_value = False

        ensure_fresh_collection(client, 'c', size=8)

        client.create_collection.assert_called_once()
        client.delete_collection.assert_not_called()

    def test_deletes_then_recreates_when_present(self):
        """collection_exists()->True: delete_collection precedes create_collection."""
        from _fm_helpers import ensure_fresh_collection

        client = MagicMock()
        client.collection_exists.return_value = True

        ensure_fresh_collection(client, 'c', size=8)

        names = [c[0] for c in client.method_calls]
        assert names == ['collection_exists', 'delete_collection', 'create_collection']

    def test_conflict_on_create_triggers_single_delete_and_retry(self):
        """A 409 on the first create_collection triggers exactly one delete then a second create.

        Models the swallowed-teardown stale-collection case the helper exists
        to self-heal: the collection was reported absent but a concurrent /
        prior run left it behind, so the create 409-conflicts; the helper
        deletes and recreates rather than propagating the conflict.
        """
        from _fm_helpers import ensure_fresh_collection
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.collection_exists.return_value = False
        client.create_collection.side_effect = [
            UnexpectedResponse(409, 'Conflict', b'{}', httpx.Headers()),
            None,
        ]

        # Must not raise — the 409 is tolerated.
        ensure_fresh_collection(client, 'c', size=8)

        assert client.create_collection.call_count == 2
        assert client.delete_collection.call_count == 1
        names = [c[0] for c in client.method_calls]
        assert names == [
            'collection_exists',
            'create_collection',
            'delete_collection',
            'create_collection',
        ]

    def test_non_conflict_create_error_propagates(self):
        """A non-409 UnexpectedResponse (e.g. 500) propagates — fail loud, no retry."""
        from _fm_helpers import ensure_fresh_collection
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.collection_exists.return_value = False
        client.create_collection.side_effect = UnexpectedResponse(500, 'err', b'{}', httpx.Headers())

        with pytest.raises(UnexpectedResponse):
            ensure_fresh_collection(client, 'c', size=8)

        client.delete_collection.assert_not_called()
        assert client.create_collection.call_count == 1


# ---------------------------------------------------------------------------
# Tests for the shared collection_vector_size() helper (task 2773, item 4)
# ---------------------------------------------------------------------------
# Race-safe read of a collection's vector dimension: under `-n auto` load a
# just-created collection's get_collection() can transiently 404 (or the HTTP
# layer can raise ResponseHandlingException) before it is durably visible, so
# a single naive read flakes. The helper wraps the read in poll_until so the
# transient window is a bounded retry, while a non-404 error (e.g. a genuine
# 500) is re-raised immediately — fail loud. No real Qdrant is used here: the
# client is a MagicMock and get_collection is scripted. Following
# TestPollUntil's discipline, cases assert only on call counts and raises,
# never on elapsed wall-clock, so the suite stays load-independent.

class TestCollectionVectorSize:
    """Unit tests for async collection_vector_size(client, collection, *, timeout, interval)."""

    @staticmethod
    def _info(size: int):
        """A fake get_collection() return value exposing .config.params.vectors."""
        from qdrant_client.models import Distance, VectorParams

        info = MagicMock()
        info.config.params.vectors = VectorParams(size=size, distance=Distance.COSINE)
        return info

    @pytest.mark.asyncio
    async def test_returns_size_after_one_call_when_readable(self):
        """A readable collection returns its vector size after exactly 1 get_collection call."""
        from _fm_helpers import collection_vector_size

        client = MagicMock()
        client.get_collection.return_value = self._info(8)

        size = await collection_vector_size(client, 'c', timeout=5.0, interval=0.001)

        assert size == 8
        assert client.get_collection.call_count == 1

    @pytest.mark.asyncio
    async def test_tolerates_transient_404_then_returns(self):
        """Two transient 404s then a readable collection: returns 8 after exactly 3 calls.

        Proves the transient-404 window is a bounded retry, not a single naive
        read — the call count is deterministic (scripted side_effect), so this
        assertion is load-independent.
        """
        from _fm_helpers import collection_vector_size
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.get_collection.side_effect = [
            UnexpectedResponse(404, 'Not Found', b'{}', httpx.Headers()),
            UnexpectedResponse(404, 'Not Found', b'{}', httpx.Headers()),
            self._info(8),
        ]

        size = await collection_vector_size(client, 'c', timeout=5.0, interval=0.001)

        assert size == 8
        assert client.get_collection.call_count == 3

    @pytest.mark.asyncio
    async def test_persistent_404_raises_assertion_error(self):
        """A get_collection that always 404s raises AssertionError once the deadline passes.

        Uses a tiny timeout and asserts only the raise and that the probe ran
        at least once (never elapsed wall-clock or an exact poll count) — a
        slow host makes this slower, never falsely passing or failing.
        """
        from _fm_helpers import collection_vector_size
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.get_collection.side_effect = UnexpectedResponse(404, 'Not Found', b'{}', httpx.Headers())

        with pytest.raises(AssertionError):
            await collection_vector_size(client, 'c', timeout=0.05, interval=0.01)

        assert client.get_collection.call_count >= 1

    @pytest.mark.asyncio
    async def test_non_404_error_propagates_immediately(self):
        """A non-404 UnexpectedResponse (e.g. 500) propagates on the first probe — fail loud."""
        from _fm_helpers import collection_vector_size
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.get_collection.side_effect = UnexpectedResponse(500, 'err', b'{}', httpx.Headers())

        with pytest.raises(UnexpectedResponse):
            await collection_vector_size(client, 'c', timeout=5.0, interval=0.001)

        assert client.get_collection.call_count == 1


# ---------------------------------------------------------------------------
# Tests for the shared await_index_operational() barrier (task 3377)
# ---------------------------------------------------------------------------
# FalkorDB builds indices asynchronously, so a test that queries an index
# immediately after creating it can silently succeed for a query the engine
# would otherwise reject (measured while characterising task 3334: 2 of 6 runs
# falsely reported success). await_index_operational is the shared barrier that
# closes that window; it was extracted out of
# test_falkor_fulltext_integration.py so both live-index modules route through
# one implementation.
#
# These tests drive the barrier against FAKE graph objects — no live FalkorDB —
# so the whole RED/GREEN cycle runs in the DEFAULT `-m 'not integration'` lane,
# on exactly the CI configuration least able to notice the regression this
# barrier exists to prevent. Following TestPollUntil's discipline, cases assert
# only on `db.indexes()` call counts and raises, never on elapsed wall-clock,
# so the suite stays load-independent.

# The measured live header (FalkorDB module v41800 / 4.18.0). `result.header`
# rows are (type, name) 2-tuples, so the column NAME is `col[1]`; `status` sits
# at index 7 of 9 today — a position the barrier must NOT hardcode.
_LIVE_INDEX_HEADER: list[tuple[int, str]] = [
    (1, 'label'),
    (1, 'properties'),
    (1, 'types'),
    (1, 'options'),
    (1, 'language'),
    (1, 'stopwords'),
    (1, 'entitytype'),
    (1, 'status'),
    (1, 'info'),
]

# The real measured not-ready value — NOT a bare 'UNDER CONSTRUCTION'. It
# carries a varying progress prefix, which is why the readiness predicate must
# match the READY side exactly rather than substring-testing the not-ready one.
_UNDER_CONSTRUCTION = '[Indexing] 3/200000: UNDER CONSTRUCTION'
_OPERATIONAL = 'OPERATIONAL'


class _FakeIndexResult:
    """A scripted ``CALL db.indexes()`` result mirroring the measured live shape."""

    def __init__(self, header: list[tuple[int, str]], result_set: list[list]):
        self.header = header
        self.result_set = result_set


class _FakeIndexGraph:
    """A fake FalkorDB graph replaying a scripted sequence of index results.

    Counts ``query()`` calls so tests can prove the barrier genuinely polled
    (rather than returning early) and that an already-ready index costs exactly
    one round-trip. Once the script is exhausted the final entry repeats
    forever, so a "never ready" case is expressed by a one-entry script.
    """

    def __init__(self, results: list[_FakeIndexResult]):
        assert results, 'scripted result sequence must be non-empty'
        self._results = results
        self.calls = 0
        self.queries: list[str] = []

    async def query(self, q: str, params=None) -> _FakeIndexResult:
        self.calls += 1
        self.queries.append(q)
        return self._results[min(self.calls - 1, len(self._results) - 1)]


def _index_row(status: str, header: list[tuple[int, str]] = _LIVE_INDEX_HEADER) -> list:
    """One ``db.indexes()`` row carrying *status* in the header's `status` column."""
    return [status if name == 'status' else f'<{name}>' for _, name in header]


def _index_result(
    *statuses: str, header: list[tuple[int, str]] = _LIVE_INDEX_HEADER
) -> _FakeIndexResult:
    """A result whose rows carry *statuses*, one row per status."""
    return _FakeIndexResult(header, [_index_row(s, header) for s in statuses])


class TestAwaitIndexOperational:
    """Unit tests for async await_index_operational(graph, timeout_s)."""

    @pytest.mark.asyncio
    async def test_already_operational_returns_after_one_query(self):
        """An already-OPERATIONAL index returns after exactly 1 db.indexes() call.

        Proves check-before-sleep: the barrier costs a single round-trip in the
        common case, so adding it to a fixture is nearly free.
        """
        from _fm_helpers import await_index_operational

        graph = _FakeIndexGraph([_index_result(_OPERATIONAL, _OPERATIONAL)])

        await await_index_operational(graph, timeout_s=5.0)

        assert graph.calls == 1, f'expected exactly 1 query, got {graph.calls}'
        assert graph.queries == ['CALL db.indexes()']

    @pytest.mark.asyncio
    async def test_polls_until_status_flips_to_operational(self):
        """Not-ready for the first 3 polls then OPERATIONAL: returns after exactly 4 calls.

        The call count is what proves the barrier actually blocked rather than
        returning on the first (under-construction) read.
        """
        from _fm_helpers import await_index_operational

        graph = _FakeIndexGraph(
            [
                _index_result(_UNDER_CONSTRUCTION),
                _index_result(_UNDER_CONSTRUCTION),
                _index_result(_UNDER_CONSTRUCTION),
                _index_result(_OPERATIONAL),
            ]
        )

        await await_index_operational(graph, timeout_s=5.0)

        assert graph.calls == 4, f'expected exactly 4 queries, got {graph.calls}'

    @pytest.mark.asyncio
    async def test_mixed_rows_do_not_satisfy_the_barrier(self):
        """One OPERATIONAL row plus one still building must NOT be treated as ready.

        EVERY index on the graph has to be ready; an `any()`-style predicate
        would let a half-built graph through and reinstate the false-green.
        """
        from _fm_helpers import await_index_operational

        graph = _FakeIndexGraph([_index_result(_OPERATIONAL, _UNDER_CONSTRUCTION)])

        with pytest.raises(AssertionError):
            await await_index_operational(graph, timeout_s=0.05)

    @pytest.mark.asyncio
    async def test_status_column_is_resolved_by_name_not_position(self):
        """A reordered header with a decoy 'OPERATIONAL' at index 7 must still block.

        `status` is moved to index 0 and the literal 'OPERATIONAL' is planted at
        index 7 (its position in today's live header). A barrier that read the
        column positionally would see the decoy and return success; one that
        resolves `status` by name reads the real, still-building value and
        blocks. This is the assertion that stops a FalkorDB column reorder from
        silently turning the barrier into a no-op.
        """
        from _fm_helpers import await_index_operational

        reordered = [(1, 'status')] + [c for c in _LIVE_INDEX_HEADER if c[1] != 'status']
        assert reordered[7][1] != 'status', 'decoy must not land on the real status column'

        row = _index_row(_UNDER_CONSTRUCTION, reordered)
        row[7] = _OPERATIONAL  # decoy in the position `status` occupies live

        graph = _FakeIndexGraph([_FakeIndexResult(reordered, [row])])

        with pytest.raises(AssertionError):
            await await_index_operational(graph, timeout_s=0.05)

    @pytest.mark.asyncio
    async def test_missing_status_column_raises_immediately_naming_the_header(self):
        """A header with no `status` column fails loudly and at once — not as a timeout.

        The two failure modes carry different operator actions: "index never
        became ready" is a slow or wedged build, while "db.indexes() has no
        status column" means FalkorDB changed its result shape and the barrier
        can no longer be trusted at all. Collapsing the second into a generic
        timeout message would hide a silent no-op behind a plausible-looking
        timeout, so this must raise on the FIRST query with the observed header
        names in the message — and must never skip.
        """
        from _fm_helpers import await_index_operational

        header = [(1, 'label'), (1, 'properties'), (1, 'state')]
        graph = _FakeIndexGraph([_FakeIndexResult(header, [['Entity', ['name'], 'READY']])])

        with pytest.raises(AssertionError) as excinfo:
            await await_index_operational(graph, timeout_s=5.0)

        message = str(excinfo.value)
        assert 'status' in message
        for name in ('label', 'properties', 'state'):
            assert name in message, f'header name {name!r} missing from {message!r}'
        assert graph.calls == 1, (
            f'expected the header-shape failure to raise on the first query, got '
            f'{graph.calls} queries — it must not be retried until the deadline and '
            f'reported as a timeout.'
        )

    @pytest.mark.asyncio
    async def test_never_operational_raises_with_last_seen_statuses(self):
        """An index that never becomes ready raises AssertionError naming its last statuses.

        Raising (rather than skipping or passing) is deliberate: a
        never-operational index must be a loud failure, since passing would
        reintroduce exactly the false-green this barrier removes.
        """
        from _fm_helpers import await_index_operational

        graph = _FakeIndexGraph([_index_result(_UNDER_CONSTRUCTION)])

        with pytest.raises(AssertionError) as excinfo:
            await await_index_operational(graph, timeout_s=0.05)

        assert _UNDER_CONSTRUCTION in str(excinfo.value)
        assert graph.calls >= 1

    @pytest.mark.asyncio
    async def test_empty_result_set_is_not_ready(self):
        """A graph reporting ZERO indices must never satisfy the barrier — fail closed.

        `all()` over an empty sequence is vacuously True, so without an explicit
        non-empty guard a graph whose index vanished (or was never created)
        would return success having gated nothing.
        """
        from _fm_helpers import await_index_operational

        graph = _FakeIndexGraph([_FakeIndexResult(_LIVE_INDEX_HEADER, [])])

        with pytest.raises(AssertionError):
            await await_index_operational(graph, timeout_s=0.05)
