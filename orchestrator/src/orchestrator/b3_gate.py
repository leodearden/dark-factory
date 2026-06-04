"""b3_gate — mechanical freshness check, durable rolling cap, per-proposal launch records.

CLI usage (three verbs, JSON to stdout):
  python -m orchestrator.b3_gate check         --task-id ID --worktree DIR --project-root DIR [options]
  python -m orchestrator.b3_gate record-launch --task-id ID --worktree DIR --project-root DIR [options]
  python -m orchestrator.b3_gate charge        --task-id ID --project-root DIR [options]

PRD: plans/b3-low-risk-auto-unblock-hardening-prd.md T2 §2.2, §4.1, §4.2, §4.4, §4.5
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_REL_PATH = 'data/escalations/b3-state.json'
B3_CATEGORIES: frozenset[str] = frozenset({'task_failure', 'review_issues'})

FRESH = 'fresh'
DRIFT = 'drift'
ABORT = 'abort'

DEFAULT_TAG = 'master'
DEFAULT_CAP = 6

_EMPTY_STATE: dict[str, list] = {'launches': [], 'charges': []}


# ---------------------------------------------------------------------------
# State path
# ---------------------------------------------------------------------------

def _state_path(project_root: str | Path) -> Path:
    """Return the canonical state-file path for a given project root."""
    return Path(project_root) / STATE_REL_PATH


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------

def _load_state(path: Path) -> dict[str, Any]:
    """Load b3 state from *path*; return empty state on any error."""
    try:
        text = path.read_text(encoding='utf-8')
        if not text.strip():
            return {'launches': [], 'charges': []}
        data = json.loads(text)
        # Ensure both keys present
        return {
            'launches': data.get('launches', []),
            'charges': data.get('charges', []),
        }
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        return {'launches': [], 'charges': []}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    """Atomically write *state* to *path* (tmp + os.replace).

    Modelled on digest.py:482 — writes to a sibling .tmp file then
    os.replace so the file is never partially written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state), encoding='utf-8')
    os.replace(str(tmp), str(path))


@contextmanager
def _locked_state(state_path: Path) -> Iterator[None]:
    """Context manager: acquire an exclusive flock on *state_path*, yield, release.

    The lock serializes the full read-modify-write for concurrent callers.
    Modelled on harness.py:189-223 (_acquire_project_lock).

    We lock a sidecar `.lock` file so the lock file is never the state file
    itself (which gets replaced atomically). fcntl.flock is per open-file-
    description, so two independent open()+flock() calls in the same process
    serialize correctly.
    """
    lock_path = state_path.with_suffix('.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, 'w')  # noqa: SIM115
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


# ---------------------------------------------------------------------------
# record_launch
# ---------------------------------------------------------------------------

def record_launch(
    task_id: str,
    head_sha: str,
    investigated_at: str,
    *,
    state_path: Path,
    now: datetime,
) -> dict[str, Any]:
    """Record a per-proposal launch entry under a lock.

    Idempotent: same (task_id, head_sha, investigated_at) triple is not re-added.
    Returns ``{'recorded': bool, 'already_attempted': bool}``.
    """
    with _locked_state(state_path):
        state = _load_state(state_path)
        # Check for existing entry with the same key
        for entry in state['launches']:
            if (entry.get('task_id') == task_id
                    and entry.get('head_sha') == head_sha
                    and entry.get('investigated_at') == investigated_at):
                return {'recorded': False, 'already_attempted': True}
        # Not found — append new launch record
        state['launches'].append({
            'task_id': task_id,
            'head_sha': head_sha,
            'investigated_at': investigated_at,
            'recorded_at': now.isoformat(),
        })
        _save_state(state_path, state)
    return {'recorded': True, 'already_attempted': False}


# ---------------------------------------------------------------------------
# charge
# ---------------------------------------------------------------------------

def _count_charges_in_window(state: dict[str, Any], now: datetime) -> int:
    """Count charges in the rolling 24-hour window before *now*."""
    cutoff = now - timedelta(hours=24)
    count = 0
    for c in state.get('charges', []):
        try:
            ts = datetime.fromisoformat(c['charged_at'])
            if ts > cutoff:
                count += 1
        except (KeyError, ValueError):
            pass
    return count


def _cap_remaining(state: dict[str, Any], cap: int, now: datetime) -> int:
    """Return how many merge slots remain in the current 24-hour window."""
    return max(0, cap - _count_charges_in_window(state, now))


def charge(
    task_id: str,
    *,
    state_path: Path,
    cap: int,
    now: datetime,
) -> dict[str, Any]:
    """Charge one rolling-24h merge slot.

    Returns ``{'charged': bool, 'remaining': int}`` (plus 'reason' on refusal).
    Over-cap calls return ``{'charged': False, 'remaining': 0, 'reason': 'cap exceeded'}``.
    """
    with _locked_state(state_path):
        state = _load_state(state_path)
        cutoff = now - timedelta(hours=24)
        in_window = [c for c in state.get('charges', [])
                     if _ts_in_window(c.get('charged_at', ''), cutoff)]
        remaining_before = cap - len(in_window)
        if remaining_before <= 0:
            return {'charged': False, 'remaining': 0, 'reason': 'cap exceeded'}
        # Prune out-of-window charges to bound file growth, then append new one
        state['charges'] = in_window + [{'task_id': task_id, 'charged_at': now.isoformat()}]
        _save_state(state_path, state)
    return {'charged': True, 'remaining': remaining_before - 1}


def _ts_in_window(ts_str: str, cutoff: datetime) -> bool:
    """Return True if *ts_str* parses to a datetime strictly after *cutoff*."""
    try:
        return datetime.fromisoformat(ts_str) > cutoff
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# check_proposal (verdict core)
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: str) -> tuple[int, str]:
    """Run a git command; return (returncode, stdout_stripped)."""
    raise NotImplementedError


def check_proposal(
    entry: dict[str, Any] | None,
    *,
    worktree: str,
    category: str | None,
    run_git: Any = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Classify a proposal entry and return a verdict dict.

    Keys: verdict (fresh|drift|abort), reason, head_sha, main_sha, age_seconds.
    The 'run_git' parameter defaults to _run_git; tests inject a fake.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# _read_latest_proposal
# ---------------------------------------------------------------------------

def _read_latest_proposal(
    task_id: int | str,
    project_root: str | Path,
    *,
    tag: str = DEFAULT_TAG,
) -> dict[str, Any] | None:
    """Read the latest dry_run_proposals entry from tasks.db for the given task.

    Returns None on any error (missing db, missing row, empty proposals list).
    Uses stdlib sqlite3 only — no fused_memory dependency.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# _resolve_cap
# ---------------------------------------------------------------------------

def _resolve_cap(config_path: str | None) -> int:
    """Return the b3_merge_cap_per_24h value from config, or DEFAULT_CAP."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI verb runners
# ---------------------------------------------------------------------------

def run_check(args: argparse.Namespace) -> None:
    """Execute the 'check' verb: print JSON verdict to stdout."""
    raise NotImplementedError


def run_record_launch(args: argparse.Namespace) -> None:
    """Execute the 'record-launch' verb: print JSON result to stdout."""
    raise NotImplementedError


def run_charge(args: argparse.Namespace) -> None:
    """Execute the 'charge' verb: print JSON result to stdout."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns an exit code."""
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
