"""Fixture inputs for orchestrator.sandbox_soak: an event store, a task store and a git repo."""
from __future__ import annotations

import json
import sqlite3
import subprocess

from orchestrator import event_store as _event_store
from orchestrator.event_store import EventType

PROBE_REPORT_PATH = "docs/sandbox-containment-probe-report.md"


def build_events_db(path):
    """Fixture events store — the real event_store `_SCHEMA`, rows spread across
    TWO run_ids so the run-agnostic aggregation is exercised. Plain
    DELETE-journal connection (no WAL) so the mode=ro readers open cleanly."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_event_store._SCHEMA)
        rows = [
            # (run_id, task_id, event_type, data)
            ("run-A", "1", EventType.sandbox_applied.value, {"backend": "bwrap", "digest": "d1"}),
            ("run-A", "2", EventType.sandbox_applied.value, {"backend": "bwrap"}),
            ("run-A", "9", EventType.sandbox_unavailable.value, {"reason": "no backend"}),
            ("run-A", "9", EventType.escalation_created.value,
             {"category": "sandbox_denial", "summary": "write denied: EACCES on /etc/x"}),
            # run-B: a distinct run_id — union must aggregate across both.
            ("run-B", "2", EventType.sandbox_applied.value, {"backend": "bwrap"}),  # dup task across runs
            ("run-B", "3", EventType.sandbox_applied.value, {"backend": "bwrap"}),
            ("run-B", None, EventType.sandbox_applied.value, {"note": "no task"}),  # NULL task_id -> excluded
            ("run-B", "10", EventType.sandbox_unavailable.value, {"reason": "policy"}),
            ("run-B", "7", EventType.escalation_created.value,
             {"category": "milestone_gate", "summary": "milestone check failed"}),
        ]
        conn.executemany(
            "INSERT INTO events (timestamp, run_id, task_id, event_type, data) "
            "VALUES (?, ?, ?, ?, ?)",
            [("2026-07-23T00:00:00+00:00", r, t, et, json.dumps(d)) for (r, t, et, d) in rows],
        )
        conn.commit()
    finally:
        conn.close()


def build_tasks_db(path, rows, tag="master"):
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE tasks (tag TEXT NOT NULL, id INTEGER NOT NULL, "
            "status TEXT NOT NULL, PRIMARY KEY (tag, id))"
        )
        conn.executemany(
            "INSERT INTO tasks (tag, id, status) VALUES (?, ?, ?)",
            [(tag, i, s) for i, s in rows],
        )
        conn.commit()
    finally:
        conn.close()


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )


def init_repo_on_main(repo):
    """Init a git repo whose default branch is deterministically `main`."""
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init", "-q")
    git(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("init\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "init")


def write_report(repo):
    p = repo / PROBE_REPORT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# containment probe report\n")
    return p


def commit_probe_report(repo):
    write_report(repo)
    git(repo, "add", PROBE_REPORT_PATH)
    git(repo, "commit", "-q", "-m", "add probe report")
