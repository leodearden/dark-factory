"""Tests for the submit_and_resolve helper in _fm_helpers.py."""

import asyncio
import concurrent.futures
import json
import subprocess
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from _fm_helpers import (
    _LOADED_SCRIPT_MODULE_NAMES,
    LoopFreedomProbe,
    _init_git_repo,
    git_stdout,
    init_ab_commit_repo,
    init_merge_history_repo,
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
# Tests for shared FalkorDB test scaffolding (task 3502)
# ---------------------------------------------------------------------------
# No real FalkorDB is used here — falkordb.FalkorDB is monkeypatched
# throughout, so this suite runs in the default `-m 'not integration'` lane.

class TestFalkorHelpers:
    """Unit tests for FALKOR_HOST/FALKOR_PORT, _falkor_available(), falkor_skipif()."""

    def test_falkor_host_from_env_default(self, monkeypatch):
        """_falkor_host_from_env() falls back to the local-FalkorDB default.

        The ``delenv`` is what makes the assertion deterministic under an
        operator environment that sets FALKOR_HOST.
        """
        from _fm_helpers import _falkor_host_from_env

        monkeypatch.delenv('FALKOR_HOST', raising=False)

        assert _falkor_host_from_env() == 'localhost'

    def test_falkor_host_from_env_override(self, monkeypatch):
        """FALKOR_HOST is honoured — the normal shape when FalkorDB is not on loopback."""
        from _fm_helpers import _falkor_host_from_env

        monkeypatch.setenv('FALKOR_HOST', 'falkordb.internal')

        assert _falkor_host_from_env() == 'falkordb.internal'

    def test_falkor_port_from_env_default(self, monkeypatch):
        """_falkor_port_from_env() falls back to 6379 *as an int*."""
        from _fm_helpers import _falkor_port_from_env

        monkeypatch.delenv('FALKOR_PORT', raising=False)

        assert _falkor_port_from_env() == 6379
        assert isinstance(_falkor_port_from_env(), int)

    def test_falkor_port_from_env_coerces_to_int(self, monkeypatch):
        """An overridden FALKOR_PORT is coerced from str to int.

        The coercion is load-bearing: a str port silently breaks
        ``FalkorDB(port=...)``. This is the assertion that fails if ``int()``
        is ever dropped — ``'7000' == 7000`` is False.
        """
        from _fm_helpers import _falkor_port_from_env

        monkeypatch.setenv('FALKOR_PORT', '7000')

        assert _falkor_port_from_env() == 7000
        assert isinstance(_falkor_port_from_env(), int)

    def test_falkor_port_from_env_treats_empty_as_unset(self, monkeypatch):
        """An empty FALKOR_PORT takes the default instead of exploding.

        ``FALKOR_PORT: ${FALKOR_PORT}`` in a compose/CI template with the
        variable unset produces exactly this shape. Since conftest.py imports
        _fm_helpers on every session, a ValueError here would abort the entire
        run — including the unit lane, which never touches FalkorDB.
        """
        from _fm_helpers import _falkor_port_from_env

        monkeypatch.setenv('FALKOR_PORT', '')

        assert _falkor_port_from_env() == 6379

    def test_falkor_port_from_env_rejects_non_integer(self, monkeypatch):
        """A malformed FALKOR_PORT fails loudly, naming the variable and value.

        Not silently defaulted: a typo'd port must not be papered over into a
        probe against 6379. The RuntimeError replaces a bare ``int()``
        ValueError whose traceback pointed into a shared helper.
        """
        from _fm_helpers import _falkor_port_from_env

        monkeypatch.setenv('FALKOR_PORT', 'notaport')

        with pytest.raises(RuntimeError, match=r"FALKOR_PORT.*'notaport'"):
            _falkor_port_from_env()

    def test_falkor_constants_have_the_types_consumers_require(self):
        """The module constants carry the types ``FalkorDB(...)`` requires.

        Scoped deliberately to the types, which is the real hazard: ``os.environ``
        yields str, and a str port silently breaks ``FalkorDB(port=...)``. This
        holds under any environment and pins no literal default, which would
        couple this lane to the operator's environment.

        It does NOT pin that the constants are *derived* from the environment.
        That cannot be asserted in-process without an ``importlib.reload`` of a
        module the whole session already holds references into (it would hand
        earlier-imported tests stale dataclass identities), and a subprocess
        import costs a measured ~8.6s here. The derivation functions themselves
        are covered directly by the tests above; what is untested is only the
        two assignment lines that call them.
        """
        from _fm_helpers import FALKOR_HOST, FALKOR_PORT

        assert isinstance(FALKOR_HOST, str)
        assert isinstance(FALKOR_PORT, int)

    def test_falkor_available_true_when_client_ok(self, monkeypatch):
        """_falkor_available() returns True when construct, query, and close all succeed."""
        from _fm_helpers import _falkor_available

        fake_graph = MagicMock()
        fake_graph.query.return_value = None
        fake_client = MagicMock()
        fake_client.select_graph.return_value = fake_graph
        fake_client.close.return_value = None
        monkeypatch.setattr('falkordb.FalkorDB', MagicMock(return_value=fake_client))

        assert _falkor_available() is True
        fake_client.close.assert_called_once()

    def test_falkor_available_false_when_query_raises(self, monkeypatch):
        """_falkor_available() returns False (never raises) when the probe query raises."""
        from _fm_helpers import _falkor_available

        fake_graph = MagicMock()
        fake_graph.query.side_effect = Exception('boom')
        fake_client = MagicMock()
        fake_client.select_graph.return_value = fake_graph
        monkeypatch.setattr('falkordb.FalkorDB', MagicMock(return_value=fake_client))

        assert _falkor_available() is False
        # The fork wraps close() in contextlib.suppress inside a `finally`, so a
        # probe against a half-dead server must still not leak the connection.
        fake_client.close.assert_called_once()

    def test_falkor_available_false_when_construct_raises(self, monkeypatch):
        """_falkor_available() returns False (never raises) when FalkorDB(...) itself raises."""
        from _fm_helpers import _falkor_available

        def _boom(*args, **kwargs):
            raise Exception('connection refused')

        monkeypatch.setattr('falkordb.FalkorDB', _boom)

        assert _falkor_available() is False

    def test_falkor_available_probe_wiring(self, monkeypatch):
        """_falkor_available() probes with the same host/port/timeout and query the forks used.

        Pins the behaviour-preserving contract: a bounded ~2s connect against
        FALKOR_HOST/FALKOR_PORT, then ``select_graph('_probe').query('RETURN 1')``.
        """
        from _fm_helpers import FALKOR_HOST, FALKOR_PORT, _falkor_available

        fake_graph = MagicMock()
        fake_client = MagicMock()
        fake_client.select_graph.return_value = fake_graph
        fake_ctor = MagicMock(return_value=fake_client)
        monkeypatch.setattr('falkordb.FalkorDB', fake_ctor)

        assert _falkor_available() is True

        fake_ctor.assert_called_once_with(
            host=FALKOR_HOST, port=FALKOR_PORT, socket_connect_timeout=2
        )
        fake_client.select_graph.assert_called_once_with('_probe')
        fake_graph.query.assert_called_once_with('RETURN 1')

    def test_falkor_skipif_default_reason(self):
        """falkor_skipif() returns a pytest.mark.skipif marker with a non-empty default reason.

        The exact wording is deliberately not pinned — matching the Qdrant
        sibling above. A reword is cosmetic and nothing depends on the prose.
        """
        from _fm_helpers import falkor_skipif

        m = falkor_skipif()

        assert m.name == 'skipif'
        assert isinstance(m.kwargs['reason'], str) and m.kwargs['reason']

    def test_falkor_skipif_custom_reason(self):
        """falkor_skipif(reason=...) propagates a caller-supplied reason string."""
        from _fm_helpers import falkor_skipif

        assert falkor_skipif(reason='x').kwargs['reason'] == 'x'

    def test_falkor_skipif_condition_tracks_probe(self, monkeypatch):
        """falkor_skipif()'s condition is `not _falkor_available()`, evaluated at call time.

        The load-bearing property: the factory resolves _falkor_available
        through the module global rather than capturing it, so each consuming
        module's probe fires at that module's own collection time.
        """
        import _fm_helpers

        monkeypatch.setattr(_fm_helpers, '_falkor_available', lambda: False)
        assert _fm_helpers.falkor_skipif().args[0] is True

        monkeypatch.setattr(_fm_helpers, '_falkor_available', lambda: True)
        assert _fm_helpers.falkor_skipif().args[0] is False

    def test_unique_graph_name_shape(self):
        """unique_graph_name(slug) is '_test_<slug>_<8 lowercase hex>'.

        Asserted as a regex on the CONTRACT rather than a fixed string, since
        the suffix is random; this pins the prefix, the slug placement and the
        8-hex width all at once.
        """
        import re

        from _fm_helpers import unique_graph_name

        assert re.fullmatch(r'_test_myslug_[0-9a-f]{8}', unique_graph_name('myslug'))

    def test_unique_graph_name_is_unique_per_call(self):
        """Two calls with the same slug return different names.

        This is the whole point of the idiom: concurrent runs (pytest-xdist
        -n auto, a CI matrix against one shared FalkorDB) must never share a
        graph and wipe each other's fixtures.
        """
        from _fm_helpers import unique_graph_name

        assert unique_graph_name('myslug') != unique_graph_name('myslug')

    def test_unique_graph_name_preserves_existing_namespace(self):
        """The helper reproduces the migrated modules' existing graph-name namespace.

        Byte-compatibility check: the migration must not silently rename live
        graphs, and the 'task id in the graph name' convention (which makes an
        orphaned graph from a killed run traceable to a module) must survive.
        """
        from _fm_helpers import unique_graph_name

        assert unique_graph_name('530_list_indices_integration').startswith(
            '_test_530_list_indices_integration_'
        )


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

        Matches the TIMEOUT path's message: IndexHeaderError also subclasses
        AssertionError, so a bare `pytest.raises(AssertionError)` would stay
        green if a regression made the header check fire spuriously here.
        """
        from _fm_helpers import await_index_operational

        graph = _FakeIndexGraph([_index_result(_OPERATIONAL, _UNDER_CONSTRUCTION)])

        with pytest.raises(AssertionError, match='not OPERATIONAL'):
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

        Matches the TIMEOUT path's message so the case cannot be satisfied by
        the header-shape path instead: the reordered header DOES contain a
        `status` column, so reaching IndexHeaderError here would mean the
        by-name resolution broke, not that the barrier blocked.
        """
        from _fm_helpers import await_index_operational

        reordered = [(1, 'status')] + [c for c in _LIVE_INDEX_HEADER if c[1] != 'status']
        assert reordered[7][1] != 'status', 'decoy must not land on the real status column'

        row = _index_row(_UNDER_CONSTRUCTION, reordered)
        row[7] = _OPERATIONAL  # decoy in the position `status` occupies live

        graph = _FakeIndexGraph([_FakeIndexResult(reordered, [row])])

        with pytest.raises(AssertionError, match='not OPERATIONAL'):
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

        Asserts the EXACT type (IndexHeaderError, not the AssertionError it
        subclasses): the whole point of the distinct type is that a caller can
        tell the two paths apart, so a test that accepted the base type would
        stay green if the paths were collapsed back together.
        """
        from _fm_helpers import IndexHeaderError, await_index_operational

        header = [(1, 'label'), (1, 'properties'), (1, 'state')]
        graph = _FakeIndexGraph([_FakeIndexResult(header, [['Entity', ['name'], 'READY']])])

        with pytest.raises(IndexHeaderError) as excinfo:
            await await_index_operational(graph, timeout_s=5.0)

        message = str(excinfo.value)
        assert 'not OPERATIONAL' not in message, (
            f'the header-shape failure must not be dressed up as a timeout: {message!r}'
        )
        assert 'status' in message
        for name in ('label', 'properties', 'state'):
            assert name in message, f'header name {name!r} missing from {message!r}'
        assert graph.calls == 1, (
            f'expected the header-shape failure to raise on the first query, got '
            f'{graph.calls} queries — it must not be retried until the deadline and '
            f'reported as a timeout.'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('header', 'offending_entry'),
        [
            pytest.param([(1, 'label'), (7,), (1, 'status')], (7,), id='short-pair'),
            pytest.param([(1, 'label'), None, (1, 'status')], None, id='non-sequence'),
            pytest.param(['label', 'status'], 'label', id='bare-str'),
        ],
    )
    async def test_malformed_header_entry_raises_index_header_error_on_first_query(
        self, header, offending_entry
    ):
        """A header entry that is not a (type, name) pair must fail as the named
        shape error, not as a raw IndexError/TypeError or a mangled
        missing-column report.
        """
        from _fm_helpers import IndexHeaderError, await_index_operational

        graph = _FakeIndexGraph([_FakeIndexResult(header, [['x', _OPERATIONAL]])])

        with pytest.raises(IndexHeaderError) as excinfo:
            await await_index_operational(graph, timeout_s=5.0)

        message = str(excinfo.value)
        assert graph.calls == 1, f'expected a raise on the first query, got {graph.calls}'
        assert 'not OPERATIONAL' not in message, (
            f'the header-shape failure must not be dressed up as a timeout: {message!r}'
        )
        assert repr(offending_entry) in message, (
            f'the error must name the entry it saw, {offending_entry!r}: {message!r}'
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

        Matches the TIMEOUT path's message: the header here is the well-formed
        live one, so an IndexHeaderError would mean the header check fired
        spuriously — a bare `pytest.raises(AssertionError)` could not tell.
        """
        from _fm_helpers import await_index_operational

        graph = _FakeIndexGraph([_FakeIndexResult(_LIVE_INDEX_HEADER, [])])

        with pytest.raises(AssertionError, match='not OPERATIONAL'):
            await await_index_operational(graph, timeout_s=0.05)


# ---------------------------------------------------------------------------
# Tests for the shared real-git repo builders (task 4847)
# ---------------------------------------------------------------------------
# Real git repos in tmp_path, read back through git itself, so the commit graph
# test_stages.py's done-provenance tests rely on is pinned by observation.

class TestGitStdout:
    """Unit tests for git_stdout(path, *args)."""

    def test_returns_stripped_stdout(self, tmp_path):
        git_stdout(tmp_path, 'init', '-q')

        assert git_stdout(tmp_path, 'rev-parse', '--is-inside-work-tree') == 'true'

    def test_nonzero_exit_raises_called_process_error(self, tmp_path):
        git_stdout(tmp_path, 'init', '-q')

        with pytest.raises(subprocess.CalledProcessError):
            git_stdout(tmp_path, 'rev-parse', '--verify', 'no-such-ref')


def _hostile_global_git_config(tmp_path, monkeypatch) -> None:
    """Point git's global config at one that forces signing through a failing
    gpg program and routes every hook a builder's commands can trigger to
    ``exit 1``, including the ones ``--no-verify`` does not bypass."""
    hooks = tmp_path / 'hostile-hooks'
    hooks.mkdir()
    for hook in (
        'pre-commit', 'commit-msg', 'pre-merge-commit', 'prepare-commit-msg', 'post-checkout',
    ):
        script = hooks / hook
        script.write_text('#!/bin/sh\nexit 1\n')
        script.chmod(0o755)
    config = tmp_path / 'hostile-gitconfig'
    config.write_text(
        f'[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = false\n[core]\n\thooksPath = {hooks}\n'
    )
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(config))


class TestRepoBuildersIgnoreDeveloperGitConfig:
    """Every builder survives a developer's global gpgsign and commit hooks."""

    @pytest.mark.parametrize(
        'builder', [_init_git_repo, init_ab_commit_repo, init_merge_history_repo],
    )
    def test_builder_succeeds_under_global_gpgsign_and_failing_hooks(
        self, tmp_path, monkeypatch, builder,
    ):
        _hostile_global_git_config(tmp_path, monkeypatch)
        repo = tmp_path / 'repo'

        builder(repo)

        assert git_stdout(repo, 'config', '--get', 'commit.gpgsign') == 'false'
        assert git_stdout(repo, 'status', '--porcelain') == ''


class TestProvenanceRepoBuilders:
    """Unit tests for init_ab_commit_repo(path) and init_merge_history_repo(path)."""

    def test_ab_commit_repo_is_one_root_commit_on_main_adding_a_and_b(self, tmp_path):
        from _fm_helpers import init_ab_commit_repo

        sha = init_ab_commit_repo(tmp_path)

        assert sha == git_stdout(tmp_path, 'rev-parse', 'main')
        assert git_stdout(tmp_path, 'rev-list', '--count', 'main') == '1'
        assert git_stdout(tmp_path, 'ls-tree', '-r', '--name-only', sha).splitlines() == [
            'a.txt',
            'b.txt',
        ]
        assert git_stdout(tmp_path, 'config', '--get', 'commit.gpgsign') == 'false'

    def test_merge_history_single_is_the_ab_commit_prefix(self, tmp_path):
        from _fm_helpers import init_ab_commit_repo, init_merge_history_repo

        merge_repo = tmp_path / 'm'
        prefix_repo = tmp_path / 'p'
        shas = init_merge_history_repo(merge_repo)
        prefix_sha = init_ab_commit_repo(prefix_repo)

        assert set(shas) == {'single', 'merge', 'conflict_merge'}
        single = shas['single']
        assert len(git_stdout(merge_repo, 'rev-list', '--parents', '-n', '1', single).split()) == 1
        assert git_stdout(merge_repo, 'rev-parse', f'{single}^{{tree}}') == git_stdout(
            prefix_repo, 'rev-parse', f'{prefix_sha}^{{tree}}'
        )

    def test_merge_history_merges_report_full_first_parent_file_lists(self, tmp_path):
        from _fm_helpers import init_merge_history_repo

        shas = init_merge_history_repo(tmp_path)
        merge, conflict_merge = shas['merge'], shas['conflict_merge']

        for sha in (merge, conflict_merge):
            parents = git_stdout(tmp_path, 'rev-list', '--parents', '-n', '1', sha).split()[1:]
            assert len(parents) == 2, f'{sha} must be a two-parent merge, got {parents}'
        assert git_stdout(tmp_path, 'rev-parse', f'{merge}^1') == shas['single']
        merge_files = set(
            git_stdout(tmp_path, 'diff', '--name-only', f'{merge}^1', merge).splitlines()
        )
        assert merge_files == {'feature.py', *(f'extra_{i}.py' for i in range(1, 5))}
        conflict_files = set(
            git_stdout(
                tmp_path, 'diff', '--name-only', f'{conflict_merge}^1', conflict_merge
            ).splitlines()
        )
        assert {'a.txt', 'conflict_only.py'} <= conflict_files
        assert git_stdout(tmp_path, 'rev-parse', 'main') == conflict_merge


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
