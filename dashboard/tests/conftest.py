"""pytest configuration — ensure local src takes precedence over installed package.

Non-fixture helpers (RECONCILIATION_SCHEMA, make_recon_db, …) live in
`_dashboard_helpers.py` — a uniquely-named sibling module — so they can be
imported from test files without conflicting with sibling subprojects'
conftests under `sys.modules['conftest']`.
"""

import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path guard — MUST stay above all non-stdlib imports.
#
# Insert this worktree's src directory at the front of sys.path so that
# `import dashboard` loads the local (possibly modified) code rather than
# whatever editable install the dashboard .venv has pinned to the main tree.
#
# Do NOT move this block below any third-party or local imports.  If a future
# maintainer adds `from dashboard import ...` above this block it would
# silently resolve against the installed editable package instead of the
# local worktree src/, causing hard-to-diagnose test failures.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Also front-load this worktree's shared/src so `import shared` (e.g. the
# shared.mcp_idempotency helper the McpSession twin imports, task 2766) loads
# the local worktree copy rather than the editable install pinned to the main
# tree — otherwise a brand-new shared submodule that has not yet landed on
# main is unimportable here, breaking collection of tests that touch it.
# Mirrors orchestrator/tests/conftest.py's shared/src insertion.
_SHARED_SRC = Path(__file__).parent.parent.parent / 'shared' / 'src'
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))
_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Suite-wide git isolation (task 3355, incident esc-3072-3).  The verify lane
# runs `cd dashboard && uv run pytest tests/`, which makes rootdir the
# SUBPROJECT — the repo-root conftest.py is never loaded, so each test-root
# conftest wires the defence itself.  APPEND the repo root, never insert(0, ...):
# at sys.path[0] it would make the subproject directories resolve as namespace
# packages pointing at the project folder instead of src/<pkg>/, defeating the
# guard block above.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import aiosqlite  # noqa: E402
import httpx  # noqa: E402
import pytest  # noqa: E402
from _dashboard_helpers import RECONCILIATION_SCHEMA  # noqa: E402
from df_pytest_isolation import (  # noqa: E402
    _df_git_ceiling_at_basetemp,  # noqa: F401  — the binding IS the wiring
    reject_unsafe_basetemp,
)
from starlette.testclient import TestClient  # noqa: E402


def pytest_configure(config):
    """Refuse a --basetemp aimed inside a live task worktree (esc-3072-3)."""
    reject_unsafe_basetemp(config)


@pytest.fixture()
def dashboard_config(tmp_path):
    """Create a DashboardConfig with tmp_path-based project_root."""
    from dashboard.config import DashboardConfig

    return DashboardConfig(project_root=tmp_path)


@pytest.fixture()
async def dummy_client():
    async with httpx.AsyncClient() as c:
        yield c


@pytest.fixture()
def dummy_config(tmp_path):
    from dashboard.config import DashboardConfig

    return DashboardConfig(project_root=tmp_path)


@pytest.fixture()
def two_url_config(tmp_path):
    """Create a DashboardConfig with two test URLs (ports 9000, 9001).

    Port 9000 is used as the failing server in fallback tests; port 9001
    responds successfully. Using ports distinct from the default (8002)
    makes the test intent explicit.
    """
    from dashboard.config import DashboardConfig

    return DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=['http://localhost:9000', 'http://localhost:9001'],
    )


@pytest.fixture()
def client():
    """Create a TestClient for the dashboard FastAPI app."""
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def two_url_client(monkeypatch):
    """TestClient running the lifespan with a two-URL fused-memory config.

    Uses monkeypatch.setenv so DashboardConfig.from_env() inside the lifespan
    produces the two-URL list naturally — no post-hoc mutation of app.state.
    monkeypatch teardown restores os.environ automatically after each test.
    """
    from dashboard.app import app

    monkeypatch.setenv(
        'DASHBOARD_FUSED_MEMORY_URLS',
        'http://localhost:9000,http://localhost:9001',
    )
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def reconciliation_db(tmp_path):
    """Create a temporary SQLite reconciliation DB with schema and sample data.

    Returns the path to the database file.
    """
    db_path = tmp_path / 'reconciliation.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(RECONCILIATION_SCHEMA)

    now = datetime.now(UTC)

    # Watermark row
    conn.execute(
        """INSERT INTO watermarks
           (project_id, last_full_run_id, last_full_run_completed,
            last_episode_timestamp, last_memory_timestamp, last_task_change_timestamp)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            'dark_factory',
            'run-001',
            (now - timedelta(hours=1)).isoformat(),
            (now - timedelta(minutes=30)).isoformat(),
            (now - timedelta(minutes=20)).isoformat(),
            (now - timedelta(minutes=10)).isoformat(),
        ),
    )

    # Runs: one completed, one still running
    completed_started = now - timedelta(hours=2)
    completed_finished = completed_started + timedelta(minutes=5)
    conn.execute(
        """INSERT INTO runs
           (id, project_id, run_type, trigger_reason, started_at, completed_at,
            events_processed, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            'run-001',
            'dark_factory',
            'full',
            'staleness_timer',
            completed_started.isoformat(),
            completed_finished.isoformat(),
            7,
            'completed',
        ),
    )
    running_started = now - timedelta(minutes=10)
    conn.execute(
        """INSERT INTO runs
           (id, project_id, run_type, trigger_reason, started_at, completed_at,
            events_processed, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            'run-002',
            'dark_factory',
            'incremental',
            'event_threshold',
            running_started.isoformat(),
            None,
            3,
            'running',
        ),
    )

    # Event buffer: 3 events with varying timestamps
    for i, minutes_ago in enumerate([60, 30, 5]):
        conn.execute(
            """INSERT INTO event_buffer
               (id, project_id, event_type, event_source, agent_id, timestamp, payload, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f'evt-{i + 1:03d}',
                'dark_factory',
                'memory_write',
                'interceptor',
                f'agent-{i + 1}',
                (now - timedelta(minutes=minutes_ago)).isoformat(),
                '{}',
                'buffered',
            ),
        )

    # Burst state: 2 rows
    conn.execute(
        """INSERT INTO burst_state (agent_id, state, last_write_at, burst_started_at)
           VALUES (?, ?, ?, ?)""",
        (
            'agent-1',
            'bursting',
            (now - timedelta(minutes=2)).isoformat(),
            (now - timedelta(minutes=10)).isoformat(),
        ),
    )
    conn.execute(
        """INSERT INTO burst_state (agent_id, state, last_write_at, burst_started_at)
           VALUES (?, ?, ?, ?)""",
        (
            'agent-2',
            'idle',
            (now - timedelta(hours=1)).isoformat(),
            None,
        ),
    )

    # Journal entries for run-001
    conn.execute(
        """INSERT INTO journal_entries
           (id, run_id, stage, timestamp, operation, target_system,
            before_state, after_state, reasoning, evidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            'je-001',
            'run-001',
            'memory_consolidation',
            (now - timedelta(hours=1, minutes=55)).isoformat(),
            'consolidate',
            'mem0',
            '{"count": 5}',
            '{"count": 3}',
            'Merged duplicate memories',
            '[{"source": "mem0", "id": "m-1"}]',
        ),
    )
    conn.execute(
        """INSERT INTO journal_entries
           (id, run_id, stage, timestamp, operation, target_system,
            before_state, after_state, reasoning, evidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            'je-002',
            'run-001',
            'task_knowledge_sync',
            (now - timedelta(hours=1, minutes=50)).isoformat(),
            'sync',
            'graphiti',
            None,
            '{"entities": 2}',
            '',
            '[]',
        ),
    )

    # Judge verdict
    conn.execute(
        """INSERT INTO judge_verdicts (run_id, reviewed_at, severity, findings, action_taken)
           VALUES (?, ?, ?, ?, ?)""",
        (
            'run-001',
            (now - timedelta(hours=1)).isoformat(),
            'low',
            '[{"issue": "minor drift"}]',
            'logged',
        ),
    )

    conn.commit()
    conn.close()

    yield db_path


@pytest.fixture()
def empty_reconciliation_db(tmp_path):
    """Create a reconciliation DB with schema but no data.

    Returns the path to the database file.
    """
    db_path = tmp_path / 'empty_reconciliation.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(RECONCILIATION_SCHEMA)
    conn.commit()
    conn.close()

    yield db_path


@pytest.fixture()
def missing_db_path(tmp_path):
    """Return a path to a non-existent database file."""
    return tmp_path / 'nonexistent' / 'reconciliation.db'


@pytest.fixture()
async def recon_conn(reconciliation_db):
    """Open an aiosqlite connection to the populated reconciliation DB."""
    async with aiosqlite.connect(str(reconciliation_db)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


@pytest.fixture()
async def empty_recon_conn(empty_reconciliation_db):
    """Open an aiosqlite connection to the empty reconciliation DB."""
    async with aiosqlite.connect(str(empty_reconciliation_db)) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn
