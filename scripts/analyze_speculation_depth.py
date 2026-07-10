#!/usr/bin/env python3
"""Calibrate merge-verify p_good/p_flake from an orchestrator runs.db.

Mines the EXISTING event store (no schema migration) for:
  - per-attempt merge_verify pass rate
  - p_flake, computed two independent ways (must approximately agree):
      (1) per-attempt pass rate over ALL merge_verify events
      (2) 1/E[attempts-to-first-pass] among eventually-landed tasks
  - p_good bracket: [1 - genuine-conflict-rate, eventual-verify-pass-rate]
  - attempts-to-first-pass retry histogram
  - terminal merge_attempt outcome tally
  - real per-depth P(pass|depth) when merge_verify events carry a depth
    field (task 2340 instrumentation); otherwise a CONFOUNDED runner-split
    fallback (runner is a broken depth proxy — flips sign across windows).

Ported from the Phase-0 calibration-mining scratchpad prototype.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from typing import Any


def load_events(
    conn: sqlite3.Connection, event_type: str, since_days: int,
) -> list[dict[str, Any]]:
    """Load events of *event_type* from the last *since_days* days.

    Returns a list of ``{'timestamp', 'task_id', 'data'}`` dicts, ordered by
    insertion (``id``).  Malformed JSON in the ``data`` column degrades to an
    empty dict rather than raising — the event store's ``emit`` is
    fire-and-forget, so a truncated/corrupt row is possible and must not
    abort the whole read.
    """
    rows = conn.execute(
        "SELECT timestamp, task_id, data FROM events "
        "WHERE event_type=? AND timestamp > datetime('now', ?) ORDER BY id",
        (event_type, f'-{since_days} days'),
    ).fetchall()
    out = []
    for ts, task_id, data in rows:
        try:
            parsed = json.loads(data) if data else {}
        except json.JSONDecodeError:
            parsed = {}
        out.append({'timestamp': ts, 'task_id': task_id, 'data': parsed})
    return out


def compute_calibration(
    merge_verify_events: list[dict[str, Any]],
    merge_attempt_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the robust (event-store-only) calibration estimators.

    ``merge_verify_events`` / ``merge_attempt_events`` are lists of
    ``{'task_id', 'data': {...}}`` dicts (the shape returned by
    :func:`load_events`).  Pure function — no DB, no clock.

    Returns a dict with keys:
      per_attempt_pass_rate  — pass rate over ALL merge_verify events
      p_flake_landed         — 1/E[attempts-to-first-pass] among landed tasks
      retry_histogram        — {n_attempts: count} among landed tasks
      landed / never         — distinct-task counts
      land_rate              — landed / (landed + never); p_good upper bound
      terminal_outcomes      — Counter of merge_attempt.data.outcome
      genuine_conflict_rate  — terminal 'conflict' fraction, or None
      p_good_bracket         — (1 - genuine_conflict_rate, land_rate), or None
    """
    n = len(merge_verify_events)
    npass = sum(1 for e in merge_verify_events if e['data'].get('passed'))
    per_attempt_pass_rate = npass / n if n else None

    task_attempts: dict[str, int] = defaultdict(int)
    task_any_pass: dict[str, bool] = defaultdict(bool)
    for e in merge_verify_events:
        tid = e['task_id']
        task_attempts[tid] += 1
        task_any_pass[tid] |= bool(e['data'].get('passed'))

    tasks = list(task_attempts)
    landed = [t for t in tasks if task_any_pass[t]]
    never = [t for t in tasks if not task_any_pass[t]]

    p_flake_landed = None
    retry_histogram: dict[int, int] = {}
    if landed:
        atts = [task_attempts[t] for t in landed]
        mean_attempts = sum(atts) / len(atts)
        p_flake_landed = 1 / mean_attempts
        retry_histogram = dict(Counter(atts))

    land_rate = len(landed) / len(tasks) if tasks else None

    terminal_outcomes = dict(
        Counter(e['data'].get('outcome', '?') for e in merge_attempt_events)
    )
    genuine_conflict_rate = None
    if merge_attempt_events:
        total_ma = sum(terminal_outcomes.values())
        conflict = terminal_outcomes.get('conflict', 0)
        genuine_conflict_rate = conflict / total_ma

    p_good_bracket = None
    if genuine_conflict_rate is not None and land_rate is not None:
        p_good_bracket = (1 - genuine_conflict_rate, land_rate)

    return {
        'per_attempt_pass_rate': per_attempt_pass_rate,
        'p_flake_landed': p_flake_landed,
        'retry_histogram': retry_histogram,
        'landed': len(landed),
        'never': len(never),
        'land_rate': land_rate,
        'terminal_outcomes': terminal_outcomes,
        'genuine_conflict_rate': genuine_conflict_rate,
        'p_good_bracket': p_good_bracket,
    }
