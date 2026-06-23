"""Tests for get_scheduler_state and get_scheduler_events MCP tools.

All tools read from orchestrator-owned files under
``<project_root>/data/orchestrator/``.  Tests use a ``tmp_path``-rooted
project_root with the ``passthrough_main_checkout`` autouse fixture so the
path isn't rejected by the git-worktree validator.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from fused_memory.mcp_tools.scheduler_state import read_scheduler_state
from fused_memory.server.tools import create_mcp_server

# ---------------------------------------------------------------------------
# Fixtures (mirror test_scheduler_overrides_tools.py pattern)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    These tests use synthetic project_root values rooted in tmp_path that
    aren't real git working trees; the real resolver would reject them.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def memory_service():
    """Mocked MemoryService with all methods as AsyncMocks."""
    svc = AsyncMock()
    svc.add_memory = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def task_interceptor():
    """Mocked task interceptor."""
    ti = AsyncMock()
    ti.set_task_status = AsyncMock(return_value={'success': True})
    return ti


@pytest.fixture
def mcp_server(memory_service, task_interceptor):
    """MCP server with mocked MemoryService and task interceptor."""
    return create_mcp_server(memory_service, task_interceptor=task_interceptor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_SKELETON = {
    'skip_counts': {},
    'parks': {},
    'park_stacks': {},
    'effective_priorities': {},
    'pin_queue': [],
    'overrides': {},
    'current_holders': {},
    'is_paused': False,
    'pause_reason': None,
    'snapshot_at': None,
}


_PARK_STACKS_FIXTURE = {
    'mod/a': [
        {'owner': 'L', 'rank': 5, 'shadowed': True,
         'installed_at': '2026-06-01T00:00:00+00:00'},
        {'owner': 'H', 'rank': 9, 'shadowed': False,
         'installed_at': '2026-06-01T00:00:00+00:00'},
    ],
}

_SNAPSHOT_WITH_PARK_STACKS = {
    'skip_counts': {},
    'parks': {},
    'park_stacks': _PARK_STACKS_FIXTURE,
    'effective_priorities': {},
    'pin_queue': [],
    'overrides': {},
    'current_holders': {},
    'is_paused': False,
    'pause_reason': None,
    'snapshot_at': '2026-06-01T12:00:00+00:00',
}


def _write_snapshot(project_root: Path, data: dict) -> Path:
    """Write a synthetic scheduler_state.json under project_root."""
    path = project_root / 'data' / 'orchestrator' / 'scheduler_state.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def _runs_db_path(project_root: Path) -> Path:
    return project_root / 'data' / 'orchestrator' / 'runs.db'


# ===========================================================================
# Step-19: get_scheduler_state tool
# ===========================================================================


class TestGetSchedulerStateTool:
    """get_scheduler_state reads the on-disk JSON snapshot."""

    @pytest.mark.asyncio
    async def test_returns_snapshot_from_disk(self, tmp_path, mcp_server):
        """Tool round-trips a synthetic snapshot dict written to disk."""
        synthetic = {
            'skip_counts': {'T1': 3},
            'parks': {'T2': {'modules': ['m/src'], 'installed_at': '2026-01-01T00:00:00+00:00'}},
            'effective_priorities': {'T1': 'high'},
            'pin_queue': [{'task_id': 'T3', 'order': 1}],
            'overrides': {'T1': {'boost_tier': 'high', 'pinned': False,
                                 'reserve_now': False, 'ttl_until': None}},
            'current_holders': {'m/src': 'T4'},
            'is_paused': True,
            'pause_reason': 'park-stop: 5 tasks parked in 1h',
            'snapshot_at': '2026-05-14T12:00:00+00:00',
        }
        _write_snapshot(tmp_path, synthetic)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_state',
            {'project_root': str(tmp_path)},
        )
        assert result == synthetic, (
            f'Expected round-tripped snapshot, got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_returns_empty_skeleton_when_file_missing(
        self, tmp_path, mcp_server
    ):
        """Tool returns the empty skeleton when scheduler_state.json is missing."""
        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_state',
            {'project_root': str(tmp_path)},
        )
        assert result == _EMPTY_SKELETON, (
            f'Expected empty skeleton, got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_invalid_project_root_returns_validation_error(
        self, tmp_path, mcp_server
    ):
        """Empty string project_root triggers the ValidationError envelope."""
        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_state',
            {'project_root': ''},
        )
        assert isinstance(result, dict), f'Expected dict, got {type(result)}'
        assert result.get('error_type') == 'ValidationError', (
            f'Expected ValidationError, got {result!r}'
        )
        assert 'error' in result

    def test_read_scheduler_state_empty_skeleton_includes_park_stacks(
        self, tmp_path
    ):
        """read_scheduler_state returns park_stacks == {} when no snapshot file exists.

        RED driver for the _empty_skeleton() gap: before the fix, park_stacks is
        absent from the skeleton dict and this assertion fails with KeyError.
        """
        result = read_scheduler_state(tmp_path)
        assert result['park_stacks'] == {}, (
            f"Expected park_stacks == {{}} in file-absent skeleton, got {result.get('park_stacks')!r}"
        )

    def test_read_scheduler_state_passes_park_stacks_through(self, tmp_path):
        """read_scheduler_state returns park_stacks verbatim from a snapshot on disk.

        GREEN-from-start regression guard: the file-present path is a verbatim
        json.loads pass-through; this test locks that contract so a future
        key-whitelist or schema-coercion step cannot silently strip park_stacks.

        park_stacks shape per scheduler.py:3461 — full LIFO bottom→top:
          {module: [{owner, rank, shadowed, installed_at}, ...]}
        """
        _write_snapshot(tmp_path, _SNAPSHOT_WITH_PARK_STACKS)

        result = read_scheduler_state(tmp_path)
        assert result['park_stacks'] == _PARK_STACKS_FIXTURE, (
            f'Expected park_stacks to be returned verbatim, got {result["park_stacks"]!r}'
        )

    @pytest.mark.asyncio
    async def test_get_scheduler_state_surfaces_park_stacks(
        self, tmp_path, mcp_server
    ):
        """get_scheduler_state MCP tool surfaces park_stacks from a snapshot on disk.

        GREEN-from-start end-to-end regression guard: verifies the full chain
        (json.loads → asyncio.to_thread → MCP tool) does not drop park_stacks.
        Catches a future whitelist/filter at any layer in the stack.
        """
        _write_snapshot(tmp_path, _SNAPSHOT_WITH_PARK_STACKS)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_state',
            {'project_root': str(tmp_path)},
        )
        assert result['park_stacks'] == _PARK_STACKS_FIXTURE, (
            f'Expected park_stacks surfaced end-to-end, got {result.get("park_stacks")!r}'
        )


# ===========================================================================
# Step-21: get_scheduler_events tool
# ===========================================================================

_EVENTS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""


def _seed_runs_db(project_root: Path, rows: list[dict]) -> Path:
    """Create runs.db and insert rows with controlled timestamps.

    Each row dict supports keys: timestamp, run_id, task_id, event_type, data.
    """
    db_path = project_root / 'data' / 'orchestrator' / 'runs.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_EVENTS_SCHEMA)
        for row in rows:
            conn.execute(
                'INSERT INTO events '
                '(timestamp, run_id, task_id, event_type, data) '
                'VALUES (?, ?, ?, ?, ?)',
                (
                    row['timestamp'],
                    row.get('run_id', 'run-1'),
                    row.get('task_id'),
                    row['event_type'],
                    json.dumps(row.get('data', {})),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


class TestGetSchedulerEventsTool:
    """get_scheduler_events reads events from runs.db."""

    @pytest.mark.asyncio
    async def test_returns_events_from_runs_db(self, tmp_path, mcp_server):
        """Tool returns three events in newest-first order."""
        rows = [
            {'timestamp': '2026-05-14T10:00:00+00:00', 'event_type': 'task_skipped',
             'task_id': 'T1', 'data': {'reason': 'held'}},
            {'timestamp': '2026-05-14T11:00:00+00:00', 'event_type': 'lock_acquired',
             'task_id': 'T2', 'data': {'module': 'm/src'}},
            {'timestamp': '2026-05-14T12:00:00+00:00', 'event_type': 'lock_released',
             'task_id': 'T3', 'data': {}},
        ]
        _seed_runs_db(tmp_path, rows)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': str(tmp_path)},
        )
        assert isinstance(result, dict), f'Expected dict, got {type(result)!r}'
        events = result['events']
        assert result['count'] == 3, f'Expected count=3, got {result["count"]}'
        # newest-first order
        assert events[0]['event_type'] == 'lock_released'
        assert events[1]['event_type'] == 'lock_acquired'
        assert events[2]['event_type'] == 'task_skipped'
        # data is a parsed dict, not a string
        assert isinstance(events[2]['data'], dict)
        assert events[2]['data'] == {'reason': 'held'}

    @pytest.mark.asyncio
    async def test_event_types_filter(self, tmp_path, mcp_server):
        """Tool returns only rows matching the requested event_types."""
        rows = [
            {'timestamp': '2026-05-14T10:00:00+00:00', 'event_type': 'task_skipped',
             'task_id': 'T1'},
            {'timestamp': '2026-05-14T11:00:00+00:00', 'event_type': 'lock_acquired',
             'task_id': 'T2'},
            {'timestamp': '2026-05-14T12:00:00+00:00', 'event_type': 'task_skipped',
             'task_id': 'T3'},
        ]
        _seed_runs_db(tmp_path, rows)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': str(tmp_path), 'event_types': ['task_skipped']},
        )
        assert result['count'] == 2, f'Expected count=2, got {result["count"]}'
        assert all(e['event_type'] == 'task_skipped' for e in result['events'])

    @pytest.mark.asyncio
    async def test_since_filter(self, tmp_path, mcp_server):
        """Tool returns only events at or after the `since` timestamp."""
        rows = [
            {'timestamp': '2026-05-14T09:00:00+00:00', 'event_type': 'task_skipped',
             'task_id': 'T1'},
            {'timestamp': '2026-05-14T11:00:00+00:00', 'event_type': 'lock_acquired',
             'task_id': 'T2'},
            {'timestamp': '2026-05-14T12:00:00+00:00', 'event_type': 'lock_released',
             'task_id': 'T3'},
        ]
        _seed_runs_db(tmp_path, rows)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': str(tmp_path), 'since': '2026-05-14T10:00:00+00:00'},
        )
        assert result['count'] == 2, f'Expected count=2, got {result["count"]}'
        # Only T2 and T3 are >= 10:00
        returned_tasks = {e['task_id'] for e in result['events']}
        assert returned_tasks == {'T2', 'T3'}

    @pytest.mark.asyncio
    async def test_limit_caps_results(self, tmp_path, mcp_server):
        """Tool caps results to the `limit` parameter."""
        rows = [
            {'timestamp': f'2026-05-14T0{i}:00:00+00:00', 'event_type': 'task_skipped',
             'task_id': f'T{i}'}
            for i in range(5)
        ]
        _seed_runs_db(tmp_path, rows)

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': str(tmp_path), 'limit': 2},
        )
        assert result['count'] == 2, f'Expected count=2, got {result["count"]}'
        assert len(result['events']) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_when_db_missing(self, tmp_path, mcp_server):
        """Tool returns empty result when runs.db does not exist."""
        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': str(tmp_path)},
        )
        assert result == {'events': [], 'count': 0}, (
            f'Expected empty result, got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_invalid_project_root_returns_validation_error(
        self, tmp_path, mcp_server
    ):
        """Empty string project_root triggers the ValidationError envelope."""
        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': ''},
        )
        assert isinstance(result, dict), f'Expected dict, got {type(result)}'
        assert result.get('error_type') == 'ValidationError', (
            f'Expected ValidationError, got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_read_scheduler_events_builds_uri_via_path_as_uri(
        self, tmp_path, mcp_server
    ):
        """read_scheduler_events must build the SQLite connect URI in canonical file:/// form.

        Verifies that the connect string passed to aiosqlite.connect starts with
        ``file:///``, ends with ``?mode=ro``, carries ``uri=True``, and that the
        path portion contains no unencoded ``?`` characters.  Exact formula is not
        re-derived (that would be a change-detector); the implementation-independent
        checks below fully enforce the URI canonicalization contract.
        """
        from unittest.mock import patch

        import aiosqlite as _aiosqlite

        _seed_runs_db(
            tmp_path,
            [
                {
                    'timestamp': '2026-05-14T12:00:00+00:00',
                    'event_type': 'lock_released',
                    'task_id': 'T1',
                    'data': {},
                }
            ],
        )

        real_connect = _aiosqlite.connect
        captured_uri = None
        captured_uri_kwarg = None

        # spy must be a plain function (not async): aiosqlite.connect() returns a
        # _ConnectionContextManager that supports `async with`, not a coroutine.
        # Wrapping in `async def` would return a coroutine instead, breaking __aexit__.
        def spy(*args, **kwargs):
            nonlocal captured_uri, captured_uri_kwarg
            captured_uri = args[0] if args else None
            captured_uri_kwarg = kwargs.get('uri')
            return real_connect(*args, **kwargs)

        with patch('aiosqlite.connect', spy):
            result = await mcp_server._tool_manager.call_tool(
                'get_scheduler_events',
                {'project_root': str(tmp_path)},
            )

        assert isinstance(result, dict), f'Expected dict, got {type(result)!r}'
        assert 'count' in result, (
            f"Expected 'count' key (got error envelope instead): {result!r}"
        )
        assert result['count'] == 1, f'Expected count=1, got {result!r}'
        assert captured_uri is not None
        assert captured_uri.startswith('file:///'), (
            f'Expected file:/// form (Path.as_uri()), got {captured_uri!r}'
        )
        assert captured_uri_kwarg is True, (
            f'Expected uri=True kwarg, got uri={captured_uri_kwarg!r}'
        )
        # Implementation-independent checks (don't mirror as_uri() formula):
        assert captured_uri.endswith('?mode=ro'), (
            f'URI must end with ?mode=ro, got {captured_uri!r}'
        )
        # Path portion must not contain a literal '?' (reserved chars must be %-encoded).
        path_portion = captured_uri.removesuffix('?mode=ro')
        assert '?' not in path_portion, (
            f'Unencoded ? in path portion of URI: {captured_uri!r}'
        )
        # Canonicalization check: path portion must equal the *resolved* db path URI.
        # If .resolve() were dropped from production, this would catch the regression on
        # platforms where tmp_path contains symlink components (e.g. /tmp -> /private/tmp
        # on macOS).
        resolved_db = (tmp_path / 'data' / 'orchestrator' / 'runs.db').resolve()
        assert path_portion == resolved_db.as_uri(), (
            f'URI path portion does not match resolved-path URI; '
            f'got {path_portion!r}, want {resolved_db.as_uri()!r}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('fragment', ['has?question', 'has#hash', 'has%41literal'])
    async def test_handles_uri_reserved_chars_in_project_root(
        self, tmp_path, mcp_server, fragment
    ):
        """Tool returns events even when project_root path contains URI-reserved chars.

        Linux filesystems accept '?', '#', and '%' in directory names.  Without
        URL-encoding the db_path before embedding it in the SQLite URI
        (``file:<path>?mode=ro``), those characters are misinterpreted: '?' and
        '#' as URI query-string / fragment delimiters (causing aiosqlite to fail
        or open the wrong file), and a literal '%41' as the percent-encoded form
        of 'A' (silently opening a different file path).  All three cases surface
        as ``{'error': ..., 'error_type': ...}`` (no 'count' key) instead of
        the expected ``{'events': [...], 'count': 1}`` shape.  This test is the
        regression canary for task 1331.
        """
        project_root = tmp_path / fragment
        project_root.mkdir()
        _seed_runs_db(
            project_root,
            [
                {
                    'timestamp': '2026-05-14T12:00:00+00:00',
                    'event_type': 'lock_released',
                    'task_id': 'T1',
                    'data': {},
                }
            ],
        )

        result = await mcp_server._tool_manager.call_tool(
            'get_scheduler_events',
            {'project_root': str(project_root)},
        )
        assert isinstance(result, dict), f'Expected dict, got {type(result)!r}'
        assert 'count' in result, (
            f"Expected 'count' key (got error envelope instead): {result!r}"
        )
        assert result['count'] == 1, f'Expected count=1, got {result!r}'
        events = result['events']
        assert len(events) == 1
        assert events[0]['event_type'] == 'lock_released'


# ===========================================================================
# Step-23: get_scheduler_state performance
# ===========================================================================


class TestSnapshotPerformance:
    """`read_scheduler_state` must serve a 1500-task snapshot under 50ms.

    Orchestrator snapshot-read budget; regression canary for task 1230.
    """

    def test_read_scheduler_state_under_50ms_for_1500_tasks(self, tmp_path):
        """Median latency for a 1500-task snapshot is < 50ms (task 1230 acceptance criterion)."""
        n = 1500
        snapshot = {
            'skip_counts': {f'T{i}': i % 5 for i in range(n)},
            'parks': {
                f'T{i}': {'modules': [f'm{i}/src'], 'installed_at': '2026-05-14T00:00:00+00:00'}
                for i in range(0, n, 10)
            },
            'effective_priorities': {f'T{i}': 'medium' for i in range(n)},
            'pin_queue': [{'task_id': f'T{i}', 'order': i} for i in range(10)],
            'overrides': {
                f'T{i}': {'boost_tier': 'high', 'pinned': False,
                           'reserve_now': False, 'ttl_until': None}
                for i in range(0, n, 5)
            },
            'current_holders': {f'm{i}/src': f'T{i}' for i in range(n)},
            'snapshot_at': '2026-05-14T12:00:00+00:00',
        }
        _write_snapshot(tmp_path, snapshot)

        result: dict = {}
        samples: list[float] = []
        # NOTE: We deliberately measure the warm-cache path (no page-cache eviction between
        # samples) because the orchestrator reads this snapshot many times per cycle in steady
        # state, so warm-cache cost is what the 50ms budget is actually about.
        # 2 warm-up samples (discarded) + 20 measured samples.
        # Two warm-ups absorb OS page-cache warm-up and CPU cache warm-up on
        # the file buffer.  20 measured samples make the median robust against
        # isolated tail spikes (vs. the previous 9-sample window).
        for i in range(22):
            t0 = time.perf_counter()
            result = read_scheduler_state(tmp_path)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if i >= 2:  # discard the first two samples as warm-up
                samples.append(elapsed_ms)

        median_ms = statistics.median(samples)
        # Guard against false-positive perf passes.  The two failure modes that
        # would make this test trivially fast and therefore meaningless are:
        #   (1) read_scheduler_state silently falls back to _empty_skeleton() —
        #       json.loads of a non-existent path is microseconds, so the
        #       median bound would pass without exercising real read+parse.
        #   (2) A future bug truncates the deserialized payload (e.g. partial
        #       parse, streaming decode that stops early).
        # `'snapshot_at' in result` does NOT catch (1) because _empty_skeleton()
        # also contains the key `'snapshot_at'` (value None).  Checking the
        # full size of `skip_counts` catches both: skeleton has skip_counts={}
        # (len 0), a truncated payload has len < n, and a healthy read has len == n.
        assert len(result['skip_counts']) == n, (
            f'Expected {n} entries in skip_counts (full 1500-task payload), got '
            f'{len(result["skip_counts"])} — read_scheduler_state may have fallen '
            f'back to the empty skeleton or returned a truncated payload, which '
            f'would make the perf bound trivially pass.'
        )
        # Regression canary for the orchestrator snapshot-read budget (task 1230
        # acceptance criterion).  The bound is 50ms; the actual cost of
        # json.loads(path.read_bytes()) for a ~500KB file is single-digit ms on
        # any reasonable disk, leaving large headroom.  Do NOT loosen this to
        # 150-250ms — that silently inflates the contract.  Do NOT confuse this
        # with an MCP-layer perf bound; this test specifically times the sync
        # helper (not mcp_server._tool_manager.call_tool) because the
        # asyncio.to_thread handoff + FastMCP dispatch are CI-load-sensitive and
        # flaky under pytest-xdist -n auto with 32 workers, making them
        # unsuitable for a tight latency canary.
        assert median_ms < 50, (
            f'Median latency {median_ms:.1f}ms exceeds 50ms acceptance criterion '
            f'(regression canary for orchestrator snapshot-read budget, task 1230)'
        )
