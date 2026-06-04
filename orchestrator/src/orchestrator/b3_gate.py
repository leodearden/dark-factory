"""b3_gate — mechanical freshness check, durable rolling cap, per-proposal launch records.

CLI usage (three verbs, JSON to stdout):
  python -m orchestrator.b3_gate check         --task-id ID --worktree DIR --project-root DIR [options]
  python -m orchestrator.b3_gate record-launch --task-id ID --worktree DIR --project-root DIR [options]
  python -m orchestrator.b3_gate charge        --task-id ID --project-root DIR [options]

PRD: plans/b3-low-risk-auto-unblock-hardening-prd.md T2 §2.2, §4.1, §4.2, §4.4, §4.5
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

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
    try:
        result = subprocess.run(
            ['git', '-C', cwd] + args,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout.strip()
    except Exception as exc:
        return 1, str(exc)


def check_proposal(
    entry: dict[str, Any] | None,
    *,
    worktree: str,
    category: str | None,
    run_git: Any = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Classify a proposal entry and return a verdict dict.

    Precedence:
      1. entry falsy -> abort 'no proposal to gate'
      2. risk_label != 'low' -> abort
      3. 'status' key present (failure entry) -> abort
      4. category not in B3_CATEGORIES (when not None) -> abort
      5. head_sha or main_sha missing/None -> drift 'no sha anchor'
      6. HEAD != head_sha -> abort (git-anchored; P1)
      7. diff main_sha..main -- files_referenced non-empty -> drift (P2)
      8. else fresh

    Keys: verdict, reason, head_sha, main_sha, age_seconds.
    The 'run_git' parameter defaults to _run_git; tests inject a fake.
    """
    if run_git is None:
        run_git = _run_git

    # --- (1) No proposal ---
    if not entry:
        return {
            'verdict': ABORT,
            'reason': 'no proposal to gate',
            'head_sha': None,
            'main_sha': None,
            'age_seconds': None,
        }

    head_sha = entry.get('head_sha')
    main_sha = entry.get('main_sha')

    def _result(verdict, reason):
        age = None
        try:
            investigated_at = entry.get('investigated_at')
            if investigated_at and now is not None:
                ts = datetime.fromisoformat(investigated_at)
                age = (now - ts).total_seconds()
        except Exception:
            pass
        return {
            'verdict': verdict,
            'reason': reason,
            'head_sha': head_sha,
            'main_sha': main_sha,
            'age_seconds': age,
        }

    # --- (2) Risk label ---
    if entry.get('risk_label') != 'low':
        return _result(ABORT, f'risk_label is not low: {entry.get("risk_label")!r}')

    # --- (3) Status key (failure entry) ---
    if 'status' in entry:
        return _result(ABORT, f'proposal is a failure entry (status={entry["status"]!r})')

    # --- (4) Category check ---
    if category is not None and category not in B3_CATEGORIES:
        return _result(ABORT, f'category {category!r} not in B3_CATEGORIES')

    # --- (5) SHA anchor ---
    if not head_sha or not main_sha:
        return _result(DRIFT, 'no sha anchor — re-investigate to refresh shas')

    # --- (6) P1: HEAD must not have moved ---
    rc, current_head = run_git(['rev-parse', 'HEAD'], worktree)
    if rc != 0:
        return _result(ABORT, f'git rev-parse HEAD failed (rc={rc})')
    if current_head != head_sha:
        return _result(
            ABORT,
            f'HEAD has moved: recorded={head_sha!r} current={current_head!r}',
        )

    # --- (7) P2: file-scoped main drift ---
    files = entry.get('files_referenced', [])
    if files:
        diff_args = ['diff', f'{main_sha}..main', '--'] + list(files)
        rc, diff_out = run_git(diff_args, worktree)
        if rc != 0:
            return _result(DRIFT, f'could not verify footprint drift (git diff rc={rc})')
        if diff_out.strip():
            return _result(DRIFT, 'main moved within proposal footprint (files_referenced)')

    # --- (8) Fresh ---
    return _result(FRESH, 'sha anchors valid and footprint unchanged')


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
    try:
        db_path = Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'
        if not db_path.exists():
            return None
        # Open read-only via URI
        uri = f'file:{db_path}?mode=ro'
        try:
            conn = sqlite3.connect(uri, uri=True)
        except sqlite3.OperationalError:
            return None
        try:
            cursor = conn.execute(
                'SELECT metadata FROM tasks WHERE tag=? AND id=?',
                (tag, int(task_id)),
            )
            row = cursor.fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        data = json.loads(row[0])
        proposals = data.get('dry_run_proposals', [])
        if not proposals:
            return None
        return proposals[-1]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# _resolve_cap
# ---------------------------------------------------------------------------

def _resolve_cap(config_path: str | None) -> int:
    """Return the b3_merge_cap_per_24h value from config, or DEFAULT_CAP.

    Falls back to DEFAULT_CAP on any load failure so the gate never hard-fails
    on a bad/foreign config path (§4.5).
    """
    if config_path is None:
        return DEFAULT_CAP
    try:
        from orchestrator.config import load_config
        return load_config(Path(config_path)).unblock_auto.b3_merge_cap_per_24h
    except Exception:
        return DEFAULT_CAP


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_now(now_str: str | None) -> datetime:
    """Parse an ISO datetime string, or return current UTC time if None."""
    if now_str is None:
        return datetime.now(UTC)
    return datetime.fromisoformat(now_str)


def _already_attempted(state: dict[str, Any], task_id: str,
                        head_sha: str | None, investigated_at: str | None) -> bool:
    """Return True if a launch entry with the given key exists in state."""
    for entry in state.get('launches', []):
        if (entry.get('task_id') == task_id
                and entry.get('head_sha') == head_sha
                and entry.get('investigated_at') == investigated_at):
            return True
    return False


# ---------------------------------------------------------------------------
# CLI verb runners
# ---------------------------------------------------------------------------

def run_check(args: argparse.Namespace) -> None:
    """Execute the 'check' verb: print JSON verdict to stdout."""
    now = _parse_now(args.now)
    cap = _resolve_cap(args.config)
    sp = _state_path(args.project_root)
    state = _load_state(sp)

    tag = getattr(args, 'tag', DEFAULT_TAG)
    entry = _read_latest_proposal(args.task_id, args.project_root, tag=tag)
    category = getattr(args, 'category', None)

    result = check_proposal(
        entry,
        worktree=args.worktree,
        category=category,
        now=now,
    )

    # Add state-derived fields
    result['cap_remaining'] = _cap_remaining(state, cap, now)
    head_sha = entry.get('head_sha') if entry else None
    investigated_at = entry.get('investigated_at') if entry else None
    result['already_attempted'] = _already_attempted(
        state, str(args.task_id), head_sha, investigated_at,
    )

    print(json.dumps(result))


def run_record_launch(args: argparse.Namespace) -> None:
    """Execute the 'record-launch' verb: print JSON result to stdout."""
    now = _parse_now(args.now)
    sp = _state_path(args.project_root)
    tag = getattr(args, 'tag', DEFAULT_TAG)

    entry = _read_latest_proposal(args.task_id, args.project_root, tag=tag)
    if entry is None:
        print(json.dumps({
            'recorded': False,
            'already_attempted': False,
            'error': 'no proposal found',
        }))
        return

    head_sha = entry.get('head_sha', '')
    investigated_at = entry.get('investigated_at', '')

    result = record_launch(
        str(args.task_id),
        head_sha,
        investigated_at,
        state_path=sp,
        now=now,
    )
    print(json.dumps(result))


def run_charge(args: argparse.Namespace) -> None:
    """Execute the 'charge' verb: print JSON result to stdout."""
    now = _parse_now(args.now)
    cap = _resolve_cap(args.config)
    sp = _state_path(args.project_root)

    result = charge(str(args.task_id), state_path=sp, cap=cap, now=now)
    print(json.dumps(result))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the three CLI verbs."""
    p = argparse.ArgumentParser(
        prog='python -m orchestrator.b3_gate',
        description=(
            'B3 gate: mechanical freshness check, durable rolling cap, '
            'per-proposal launch records.'
        ),
    )
    sub = p.add_subparsers(dest='verb', required=True)

    # --- Shared option factory ---
    def _add_common(sp_parser, *, worktree=False):
        sp_parser.add_argument('--task-id', required=True,
                               help='task ID (integer)')
        if worktree:
            sp_parser.add_argument('--worktree', required=True,
                                   help='path to the blocked task\'s worktree')
        sp_parser.add_argument('--project-root', required=True,
                               help='project root dir (contains data/escalations/)')
        sp_parser.add_argument('--config', default=None,
                               help='optional path to orchestrator config YAML')
        sp_parser.add_argument('--now', default=None,
                               help='inject current time as ISO string (tests/debugging)')
        sp_parser.add_argument('--tag', default=DEFAULT_TAG,
                               help=f'taskmaster tag (default: {DEFAULT_TAG!r})')

    # --- check ---
    check_p = sub.add_parser('check', help='classify latest proposal and return JSON verdict')
    _add_common(check_p, worktree=True)
    check_p.add_argument('--category', default=None,
                         help='block category (task_failure or review_issues)')

    # --- record-launch ---
    rl_p = sub.add_parser('record-launch', help='record per-proposal launch entry')
    _add_common(rl_p, worktree=True)

    # --- charge ---
    charge_p = sub.add_parser('charge', help='charge one rolling-24h merge slot')
    _add_common(charge_p, worktree=False)

    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns an exit code (0 on success)."""
    p = _build_parser()
    args = p.parse_args(argv)

    if args.verb == 'check':
        run_check(args)
    elif args.verb == 'record-launch':
        run_record_launch(args)
    elif args.verb == 'charge':
        run_charge(args)

    return 0


if __name__ == '__main__':
    sys.exit(main())
