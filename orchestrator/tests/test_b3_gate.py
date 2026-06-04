"""Tests for orchestrator.b3_gate module.

Covers: state load/save, record_launch idempotency, charge rolling-24h cap,
concurrency serialisation, check_proposal mechanical classification, freshness
P1/P2 against real git fixtures, _read_latest_proposal, _resolve_cap, CLI
main() wiring, and the two-way boundary test against the real _build_entry
producer from dry_run_unblock.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared helpers (adapted from test_dry_run_unblock.py)
# ---------------------------------------------------------------------------

def _init_git_repo(path: Path) -> str:
    """Init a minimal git repo at *path* (on 'main' branch), return HEAD sha."""
    p = str(path)
    subprocess.run(['git', 'init', '-b', 'main', p], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'config', 'user.name', 'Test User'],
                   check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'config', 'user.email', 'test@example.com'],
                   check=True, capture_output=True)
    (path / 'README.md').write_text('init')
    subprocess.run(['git', '-C', p, 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'commit', '-m', 'initial commit'],
                   check=True, capture_output=True)
    result = subprocess.run(['git', '-C', p, 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _diverge_feature_branch(path: Path) -> str:
    """Create a feature branch with one extra commit so HEAD != main.

    Returns the new HEAD sha (feature commit).
    """
    p = str(path)
    subprocess.run(['git', '-C', p, 'checkout', '-b', 'feature'],
                   check=True, capture_output=True)
    (path / 'extra.txt').write_text('feature work')
    subprocess.run(['git', '-C', p, 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'commit', '-m', 'feature commit'],
                   check=True, capture_output=True)
    result = subprocess.run(['git', '-C', p, 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _make_agent_result(*, success=True, cost_usd=0.50, structured_output=None,
                       subtype='', output='', duration_ms=1000):
    """Create a minimal MagicMock AgentResult stand-in for _build_entry tests."""
    r = MagicMock()
    r.success = success
    r.cost_usd = cost_usd
    r.structured_output = structured_output
    r.subtype = subtype
    r.output = output
    r.duration_ms = duration_ms
    return r


def _seed_tasks_db(project_root: Path, task_id: int, proposals: list) -> None:
    """Seed a minimal tasks.db at <project_root>/.taskmaster/tasks/tasks.db.

    Creates the table with columns: tag, id, status, priority, metadata, updated_at.
    Inserts one row with tag='master', the given task_id, and the proposals list
    serialised into metadata JSON as {'dry_run_proposals': proposals}.
    """
    db_dir = project_root / '.taskmaster' / 'tasks'
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / 'tasks.db'
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            tag TEXT NOT NULL,
            id INTEGER NOT NULL,
            status TEXT,
            priority TEXT,
            metadata TEXT,
            updated_at TEXT,
            PRIMARY KEY (tag, id)
        )
    """)
    metadata = json.dumps({'dry_run_proposals': proposals})
    conn.execute(
        "INSERT OR REPLACE INTO tasks (tag, id, status, priority, metadata, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ('master', task_id, 'blocked', 'medium', metadata, '2026-06-04T00:00:00+00:00'),
    )
    conn.commit()
    conn.close()
