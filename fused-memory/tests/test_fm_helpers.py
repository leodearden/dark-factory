"""Tests for the submit_and_resolve helper in _fm_helpers.py."""

import json
from unittest.mock import AsyncMock, MagicMock

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
        """qdrant_skipif() returns a pytest.mark.skipif marker with the default reason."""
        from _fm_helpers import qdrant_skipif

        m = qdrant_skipif()

        assert m.name == 'skipif'
        assert m.kwargs['reason'] == 'Qdrant not reachable'

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
