"""Tests for the fan-out every task row read rides.

The file covers the fan-out and offline-marker policy ``fetch_tasks`` and
``fetch_task_page`` share, that every read is live (PRD decision 20), and
per-root failure-streak isolation. The streak class also covers
``fetch_statuses``' fan-out.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from _canned_mcp import canned_get_tasks_result

import dashboard.data.tasks as tasks_mod


class TestFanoutStreakIsolationAcrossProjectRoots:
    """One broken project_root must not re-arm its WARNING every poll cycle.

    The mcp_fanout throttle key is ``(log_label, url)`` and ONE fused-memory
    URL serves every project_root, so a fixed literal ``log_label`` collapses
    all roots onto one key. ``note_fanout_success`` *pops* that key, so a
    healthy root's success in the same poll cycle clears the broken root's
    open streak — its next failure is ``streak == 1`` again, emitting both an
    opening and a 'recovered' WARNING every cycle, indefinitely. That is the
    exact sustained flood the transition-only policy (task 3871) exists to
    prevent, reintroduced through the key rather than the level.
    """

    @pytest.fixture(autouse=True)
    def _clean_state(self):
        """Give each test clean streak state.

        Without reset_sessions an earlier test's open streak would silently
        demote this test's expected opening WARNING to DEBUG — the exact
        failure mode reset_failure_streaks was added for.
        """
        from dashboard.data.memory import reset_sessions

        reset_sessions()
        yield
        reset_sessions()

    @staticmethod
    def _per_root_side_effect(root_a: str, payload: dict):
        """Raise ConnectError for root A; return *payload* for any other root."""
        async def _call(client, url, tool, args, **_kw):
            if args.get('project_root') == root_a:
                raise httpx.ConnectError('refused')
            return payload
        return _call

    @staticmethod
    def _fanout_records(caplog):
        records = [r for r in caplog.records if r.name == 'dashboard.data.mcp_fanout']
        warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
        return warnings

    async def test_fetch_tasks_broken_root_warns_once_across_poll_cycles(
        self, dummy_client, dummy_config, tmp_path, caplog
    ):
        """Three poll cycles, one shared URL, root A broken and root B healthy."""
        from dashboard.data.tasks import fetch_tasks

        assert len(dummy_config.fused_memory_urls) == 1, (
            'the collapse only shows up when both roots share one URL'
        )
        root_a = str(tmp_path / 'proj-a')
        root_b = str(tmp_path / 'proj-b')

        mock_mcp = AsyncMock(
            side_effect=self._per_root_side_effect(root_a, canned_get_tasks_result())
        )
        with caplog.at_level(logging.DEBUG, logger='dashboard.data.mcp_fanout'), \
                patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            for _ in range(3):
                a_result = await fetch_tasks(dummy_client, dummy_config, root_a)
                b_result = await fetch_tasks(dummy_client, dummy_config, root_b)

        # fetch_tasks returns ``list[dict] | dict``; narrow to the offline
        # marker branch before reading it.
        assert isinstance(a_result, dict), 'root A must return the offline marker'
        assert a_result.get('offline') is True, 'root A must be failing'
        assert isinstance(b_result, list), 'root B must be healthy'

        warnings = self._fanout_records(caplog)
        assert len(warnings) == 1, (
            f'a broken root must warn once, not once per poll cycle, got {warnings}'
        )
        assert not [m for m in warnings if 'recovered' in m], (
            f"root B's success must not close root A's streak, got {warnings}"
        )
        assert 'proj-a' in warnings[0], (
            'the WARNING must name the failing project_root — with one shared '
            f'URL it is the only way to tell which root is down, got {warnings[0]}'
        )

    async def test_fetch_statuses_broken_root_warns_once_across_poll_cycles(
        self, dummy_client, dummy_config, tmp_path, caplog
    ):
        """Same contract for the burndown collector's fetch_statuses path."""
        from dashboard.data.tasks import fetch_statuses

        root_a = str(tmp_path / 'proj-a')
        root_b = str(tmp_path / 'proj-b')

        mock_mcp = AsyncMock(
            side_effect=self._per_root_side_effect(
                root_a, {'statuses': {'1': 'done', '2': 'pending'}}
            )
        )
        # fetch_statuses holds no cache of its own (task 5587), so BOTH
        # roots genuinely re-read every cycle — which is exactly the streak
        # this asserts on.
        with caplog.at_level(logging.DEBUG, logger='dashboard.data.mcp_fanout'), \
                patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            for _ in range(3):
                a_result = await fetch_statuses(dummy_client, dummy_config, root_a)
                b_result = await fetch_statuses(dummy_client, dummy_config, root_b)

        assert a_result.get('offline') is True, 'root A must be failing'
        assert b_result == {1: 'done', 2: 'pending'}, 'root B must be healthy'

        warnings = self._fanout_records(caplog)
        assert len(warnings) == 1, (
            f'a broken root must warn once, not once per poll cycle, got {warnings}'
        )
        assert not [m for m in warnings if 'recovered' in m], (
            f"root B's success must not close root A's streak, got {warnings}"
        )
        assert 'proj-a' in warnings[0], (
            f'the WARNING must name the failing project_root, got {warnings[0]}'
        )


class TestFanoutCore:
    """The fan-out and offline-marker policy both public row reads share.

    Driven through ``fetch_tasks`` and ``fetch_task_page`` with only
    ``mcp_tool_call`` replaced, so what is pinned is the policy a caller
    observes, not the private core that implements it.
    """

    @pytest.fixture(autouse=True)
    def _clean_streaks(self):
        from dashboard.data.memory import reset_sessions

        reset_sessions()
        yield
        reset_sessions()

    async def test_urls_are_tried_once_each_in_config_order(
        self, dummy_client, two_url_config
    ):
        """The first success wins, so the order decides which server is preferred."""
        seen: list[str] = []

        async def _mcp(_client, url, *_args, **_kwargs):
            seen.append(url)
            raise httpx.ConnectError('refused')

        with patch('dashboard.data.tasks.mcp_tool_call', new=_mcp):
            result = await tasks_mod.fetch_tasks(
                dummy_client, two_url_config, '/proj/core-order',
            )

        assert seen == list(two_url_config.fused_memory_urls)
        assert isinstance(result, dict) and result.get('offline') is True

    async def test_a_soft_failure_falls_through_to_the_next_url(
        self, dummy_client, two_url_config
    ):
        """A structured MCP error costs the next url an attempt, not the whole read."""
        first_url = two_url_config.fused_memory_urls[0]

        async def _mcp(_client, url, *_args, **_kwargs):
            if url == first_url:
                return {'error': 'first url said no'}
            return canned_get_tasks_result()

        with patch('dashboard.data.tasks.mcp_tool_call', new=_mcp):
            result = await tasks_mod.fetch_task_page(
                dummy_client, two_url_config, '/proj/core-fallthrough',
                page_size=10, offset=0,
            )

        assert isinstance(result, list)
        assert [row['id'] for row in result] == [7, 8]

    async def test_every_url_failing_yields_the_marker_naming_each_failure(
        self, dummy_client, two_url_config
    ):
        async def _mcp(_client, url, *_args, **_kwargs):
            raise httpx.ConnectError(f'refused by {url}')

        with patch('dashboard.data.tasks.mcp_tool_call', new=_mcp):
            result = await tasks_mod.fetch_tasks(
                dummy_client, two_url_config, '/proj/core-marker',
            )

        assert isinstance(result, dict) and result.get('offline') is True, result
        for url in two_url_config.fused_memory_urls:
            assert url in result['error'], (
                f'the marker must say what each server answered; {url} is '
                f'missing from {result["error"]!r}'
            )

    async def test_each_public_read_fans_out_under_its_own_name(
        self, dummy_client, dummy_config, caplog
    ):
        """The fan-out WARNING names the read that failed, per read.

        `first_success` keys its per-url failure streak on `(log_label, url)`,
        so a label shared by both public reads would throttle them as one
        stream and report a failed page read under `fetch_tasks[...]` —
        telling an operator the wrong function broke.
        """
        async def _mcp(*_args, **_kwargs):
            raise httpx.ConnectError('refused')

        with caplog.at_level(logging.WARNING, logger='dashboard.data.mcp_fanout'), \
                patch('dashboard.data.tasks.mcp_tool_call', new=_mcp):
            await tasks_mod.fetch_tasks(dummy_client, dummy_config, '/proj/labelled')
            await tasks_mod.fetch_task_page(
                dummy_client, dummy_config, '/proj/labelled', page_size=5, offset=0,
            )

        warnings = [
            r.getMessage() for r in caplog.records
            if r.name == 'dashboard.data.mcp_fanout' and r.levelno == logging.WARNING
        ]
        assert any(m.startswith('fetch_tasks[labelled]') for m in warnings), warnings
        assert any(m.startswith('fetch_task_page[labelled]') for m in warnings), warnings


# ---------------------------------------------------------------------------
# TestEveryReadIsLive — PRD decision 20: one cache per datum, owned by the
# snapshot unit, so nothing beneath it holds a task read
# ---------------------------------------------------------------------------


class TestEveryReadIsLive:
    """Every task read reaches the substrate on every call.

    The snapshot unit (``task_snapshot.SNAPSHOT_TTL_SECONDS``) is the one cache
    a task datum has. A TTL under it would let a read answer with rows older
    than the ``as_of`` the unit stamps on them, and a retry-suppression window
    under it would hide a recovery the unit is about to ask for.

    Each case uses its own project root, so no other test's reads can answer
    for these.
    """

    async def test_two_identical_whole_tree_reads_both_reach_the_substrate(
        self, dummy_client, dummy_config
    ):
        mock_mcp = AsyncMock(side_effect=lambda *_a, **_k: canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            first = await tasks_mod.fetch_tasks(dummy_client, dummy_config, '/proj/live-tree')
            second = await tasks_mod.fetch_tasks(dummy_client, dummy_config, '/proj/live-tree')

        assert mock_mcp.call_count == 2, (
            f'two identical fetch_tasks calls made {mock_mcp.call_count} '
            'substrate call(s); a stored answer served the repeat'
        )
        assert isinstance(first, list) and isinstance(second, list)
        assert first == second

    async def test_two_identical_page_reads_both_reach_the_substrate(
        self, dummy_client, dummy_config
    ):
        mock_mcp = AsyncMock(side_effect=lambda *_a, **_k: canned_get_tasks_result())
        with patch('dashboard.data.tasks.mcp_tool_call', new=mock_mcp):
            for _ in range(2):
                page = await tasks_mod.fetch_task_page(
                    dummy_client, dummy_config, '/proj/live-page',
                    page_size=2, offset=0,
                )
                assert isinstance(page, list)

        assert mock_mcp.call_count == 2, (
            f'two identical fetch_task_page calls made {mock_mcp.call_count} '
            'substrate call(s); a stored answer served the repeat'
        )

    async def test_a_failed_read_is_retried_on_the_very_next_call(
        self, dummy_client, dummy_config
    ):
        reachable = False
        calls: list[bool] = []

        async def _mcp(*_args, **_kwargs):
            calls.append(reachable)
            if not reachable:
                raise httpx.ReadTimeout('canned get_tasks read timeout')
            return canned_get_tasks_result()

        with patch('dashboard.data.tasks.mcp_tool_call', new=_mcp):
            failed = await tasks_mod.fetch_tasks(dummy_client, dummy_config, '/proj/live-retry')
            reachable = True
            recovered = await tasks_mod.fetch_tasks(dummy_client, dummy_config, '/proj/live-retry')

        assert isinstance(failed, dict) and failed.get('offline') is True, failed
        assert True in calls, (
            'the read after a failure never reached the substrate: the failure '
            'suppressed its own retry, so a recovered root stays reported '
            f'offline (substrate calls, by reachability: {calls})'
        )
        assert isinstance(recovered, list) and len(recovered) == 2, recovered

    def test_the_terminal_window_is_read_fresh_per_request(
        self, client, tmp_path, monkeypatch
    ):
        """Its ``as_of`` is the request instant, so the rows must be that instant's.

        The window is asked for on a user action, not polled, so a read per
        request is the whole cost of making the stamp true.
        """
        from _canned_mcp import CannedMCP, _raw_row

        from dashboard.config import DashboardConfig

        root = tmp_path / 'live-window'
        root.mkdir()
        client.app.state.config = DashboardConfig(project_root=root)
        monkeypatch.setattr(
            'dashboard.data.active_tasks.fetch_task_runtime', AsyncMock(return_value={}),
        )
        pairs = [(1, 'in-progress'), (2, 'done'), (3, 'done'), (4, 'cancelled')]
        canned = CannedMCP(
            rows=[_raw_row(task_id, status) for task_id, status in pairs],
            status_map=dict(pairs),
        )

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            for _ in range(2):
                resp = client.get('/api/v2/dashboard/tasks?terminal=live-window')
                assert resp.status_code == 200, resp.text
                assert resp.json()['TASKS_TERMINAL:live-window']['value'], resp.json()

        window_reads = [
            call for call in canned.calls_to('get_tasks')
            if 'page_size' in call['args']
        ]
        assert len(window_reads) == 2, (
            f'two ?terminal= requests issued {len(window_reads)} terminal page '
            'read(s); a stored page answered a request whose as_of claims now'
        )
