"""Shared test fixtures for reviewer_trial mining/adjudication unit tests.

These builders reproduce the REAL schemas of the offline, gitignored mining
sources (``data/orchestrator/runs.db``, ``data/escalations/*.json``) so the
miner and diff-recovery unit tests are fully deterministic and never touch
the real gitignored data.  See
``orchestrator/src/orchestrator/evals/reviewer_trial/mining.py``.

Uniquely named (like ``_orch_helpers.py``) so it can be imported from test
files without colliding with sibling subprojects' ``conftest`` modules.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

# Schema mirrors data/orchestrator/runs.db (main-repo, gitignored) — only the
# three tables mining.py actually reads (runs, task_results, events).
_SCHEMA = """
CREATE TABLE runs (
    run_id         TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    prd_path       TEXT,
    started_at     TEXT NOT NULL,
    completed_at   TEXT,
    total_tasks    INTEGER DEFAULT 0,
    completed      INTEGER DEFAULT 0,
    blocked        INTEGER DEFAULT 0,
    escalated      INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    paused_for_cap INTEGER DEFAULT 0,
    cap_pause_secs REAL DEFAULT 0.0
);
CREATE TABLE task_results (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    task_id             TEXT NOT NULL,
    project_id          TEXT NOT NULL,
    title               TEXT,
    outcome             TEXT NOT NULL,
    cost_usd            REAL DEFAULT 0.0,
    duration_ms         INTEGER DEFAULT 0,
    agent_invocations   INTEGER DEFAULT 0,
    execute_iterations  INTEGER DEFAULT 0,
    verify_attempts     INTEGER DEFAULT 0,
    review_cycles       INTEGER DEFAULT 0,
    steward_cost_usd    REAL DEFAULT 0.0,
    steward_invocations INTEGER DEFAULT 0,
    completed_at        TEXT,
    PRIMARY KEY (run_id, task_id)
);
CREATE TABLE events (
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
"""


def build_synthetic_runs_db(tmp_path: Path, rows: list[dict]) -> Path:
    """Create a temp sqlite ``runs.db`` mirroring the real runs/task_results/events schema.

    Each entry in *rows* describes one ``task_results`` row, with optional keys:

    - ``task_id`` (str, required)
    - ``project_id`` (str, default ``'dark_factory'``)
    - ``run_id`` (str, default ``'run-test'``)
    - ``title`` (str, default ``''``)
    - ``outcome`` (str, default ``'done'``) — one of done/requeued/blocked/escalated/cancelled
    - ``review_cycles`` (int, default ``1``)
    - ``verify_attempts`` (int, default ``1``)
    - ``completed_at`` (str, default ``'2026-01-01T00:00:00+00:00'``)
    - ``merge_sha`` (str | None) — when set, writes a ``merge_finalized`` event
      carrying ``data.merge_sha``
    - ``escalation_created`` (bool, default ``False``) — when True, writes an
      ``escalation_created`` event for this task
    - ``escalation_id`` (str | None) — id for the escalation_created event's
      ``data.escalation_id``; defaults to ``f'esc-{task_id}-1'``

    Returns the path to the created ``runs.db``.
    """
    db_path = tmp_path / 'runs.db'
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA)

        seen_runs: set[str] = set()
        for row in rows:
            run_id = row.get('run_id', 'run-test')
            if run_id not in seen_runs:
                conn.execute(
                    'INSERT INTO runs (run_id, project_id, started_at) VALUES (?, ?, ?)',
                    (run_id, row.get('project_id', 'dark_factory'), '2026-01-01T00:00:00+00:00'),
                )
                seen_runs.add(run_id)

            task_id = row['task_id']
            completed_at = row.get('completed_at', '2026-01-01T00:00:00+00:00')
            conn.execute(
                """
                INSERT INTO task_results
                    (run_id, task_id, project_id, title, outcome, verify_attempts,
                     review_cycles, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task_id,
                    row.get('project_id', 'dark_factory'),
                    row.get('title', ''),
                    row.get('outcome', 'done'),
                    row.get('verify_attempts', 1),
                    row.get('review_cycles', 1),
                    completed_at,
                ),
            )

            merge_sha = row.get('merge_sha')
            if merge_sha:
                data = {
                    'request_id': f'mr-{task_id}',
                    'branch': f'task/{task_id}',
                    'state': 'done',
                    'merge_sha': merge_sha,
                }
                conn.execute(
                    """
                    INSERT INTO events (timestamp, run_id, task_id, event_type, phase, data)
                    VALUES (?, ?, ?, 'merge_finalized', 'merge', ?)
                    """,
                    (completed_at, run_id, task_id, json.dumps(data)),
                )

            if row.get('escalation_created'):
                esc_id = row.get('escalation_id', f'esc-{task_id}-1')
                data = {
                    'escalation_id': esc_id,
                    'category': row.get('escalation_category', 'review_suggestions'),
                    'severity': row.get('escalation_severity', 'info'),
                    'count': 1,
                }
                conn.execute(
                    """
                    INSERT INTO events (timestamp, run_id, task_id, event_type, phase, data)
                    VALUES (?, ?, ?, 'escalation_created', 'review', ?)
                    """,
                    (completed_at, run_id, task_id, json.dumps(data)),
                )

        conn.commit()
    finally:
        conn.close()

    return db_path


def write_sample_escalations(esc_dir: Path) -> list[Path]:
    """Write a couple of ``esc-<id>.json`` records (real schema) plus noise files.

    Noise mirrors what actually lives in ``data/escalations/``: a ``.lock``
    sidecar, a non-escalation state file (``b3-state.json``), and a malformed
    JSON ``esc-*.json`` file — all of which ``mine_escalation_refs`` must
    tolerate/skip without raising.

    Returns the list of written well-formed ``esc-*.json`` paths (does not
    include the noise files).
    """
    esc_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    esc_1 = {
        'id': 'esc-101-1',
        'task_id': '101',
        'agent_role': 'reviewer',
        'severity': 'blocking',
        'category': 'review_suggestions',
        'summary': 'Sample escalation summary 1',
        'detail': 'Sample escalation detail 1',
        'level': 0,
        'status': 'resolved',
    }
    esc_2 = {
        'id': 'esc-202-1',
        'task_id': '202',
        'agent_role': 'steward',
        'severity': 'info',
        'category': 'cleanup_needed',
        'summary': 'Sample escalation summary 2',
        'detail': 'Sample escalation detail 2',
        'level': 0,
        'status': 'pending',
    }
    for esc in (esc_1, esc_2):
        p = esc_dir / f'{esc["id"]}.json'
        p.write_text(json.dumps(esc, indent=2))
        written.append(p)

    # Noise the miner must skip/tolerate:
    (esc_dir / 'esc-101-1.json.lock').write_text('')
    (esc_dir / 'b3-state.json').write_text(json.dumps({'unrelated': True}))
    (esc_dir / 'esc-303-1.json').write_text('{not valid json')

    return written


def write_archived_escalations(esc_dir: Path) -> list[Path]:
    """Write an escalation record nested under a dated ``archive/`` subdirectory.

    Mirrors the REAL ``data/escalations/`` layout: the flat top level holds
    only ``*.lock``/``*.seq`` sentinels for currently-open escalations (see
    ``write_sample_escalations``'s ``.lock`` noise) — resolved records
    rotate into ``archive/<YYYY-MM-DD>/esc-<id>.json``. ``mine_escalation_refs``
    must recurse to find them.
    """
    archive_dir = esc_dir / 'archive' / '2026-07-01'
    archive_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    esc = {
        'id': 'esc-404-1',
        'task_id': '404',
        'agent_role': 'reviewer',
        'severity': 'blocking',
        'category': 'review_suggestions',
        'summary': 'Archived escalation summary',
        'detail': 'Archived escalation detail',
        'level': 0,
        'status': 'resolved',
    }
    p = archive_dir / f'{esc["id"]}.json'
    p.write_text(json.dumps(esc, indent=2))
    written.append(p)

    # A resolved escalation's lock sentinel can remain alongside it in the
    # archive dir too -- must still be skipped by mine_escalation_refs.
    (archive_dir / 'esc-404-1.json.lock').write_text('')

    return written


def make_git_repo_with_merge(tmp_path: Path) -> tuple[Path, str]:
    """Init a temp git repo and create a real (2-parent) merge commit.

    Returns ``(repo_path, merge_sha)``.  ``git show`` alone produces no diff
    for a plain merge commit (no direct content change vs. either parent
    individually is guaranteed) — exercising this shape validates
    ``recover_diff``'s first-parent-diff fallback path, not just the trivial
    single-parent-commit case.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()

    def _git(*args: str) -> str:
        result = subprocess.run(
            ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()

    _git('init', '-q')
    _git('config', 'user.email', 'test@example.com')
    _git('config', 'user.name', 'Test')

    (repo / 'file.txt').write_text('line1\nline2\n')
    _git('add', 'file.txt')
    _git('commit', '-q', '-m', 'initial commit')
    base_branch = _git('rev-parse', '--abbrev-ref', 'HEAD')

    _git('checkout', '-q', '-b', 'feature')
    (repo / 'file.txt').write_text('line1\nline2\nline3 from feature\n')
    _git('add', 'file.txt')
    _git('commit', '-q', '-m', 'feature change')

    _git('checkout', '-q', base_branch)
    (repo / 'other.txt').write_text('unrelated base-branch change\n')
    _git('add', 'other.txt')
    _git('commit', '-q', '-m', 'base branch change')

    _git('merge', '--no-ff', '-q', '-m', 'Merge feature', 'feature')
    merge_sha = _git('rev-parse', 'HEAD')

    return repo, merge_sha
