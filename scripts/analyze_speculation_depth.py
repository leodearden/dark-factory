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

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
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


def compute_per_depth(
    merge_verify_events: list[dict[str, Any]],
) -> dict[int, dict[str, Any]] | None:
    """Bucket merge_verify events by depth and compute real P(pass|depth).

    Coerces each event's ``data['depth']`` with a None-safe ``int()``:
    absent, ``None``, or unparseable depths are skipped rather than
    raising — merge_verify emits a native int, ``_emit_speculative``
    str-converts (speculative_merge stores depth as e.g. ``"1"``), and all
    historical (pre-task-2340) events carry no depth field at all.

    Returns ``None`` when no event contributes a usable depth (the trigger
    for :func:`format_report`'s CONFOUNDED runner-split fallback), else a
    dict ``{depth: {'pass': int, 'total': int, 'rate': float}}``.
    """
    buckets: dict[int, list[int]] = defaultdict(lambda: [0, 0])  # depth -> [pass, total]
    for e in merge_verify_events:
        raw_depth = e['data'].get('depth')
        if raw_depth is None:
            continue
        try:
            depth = int(raw_depth)
        except (TypeError, ValueError):
            continue
        buckets[depth][1] += 1
        buckets[depth][0] += bool(e['data'].get('passed'))

    if not buckets:
        return None

    return {
        depth: {'pass': p, 'total': t, 'rate': p / t}
        for depth, (p, t) in buckets.items()
    }


def format_report(
    calibration: dict[str, Any], per_depth: dict[int, dict[str, Any]] | None,
) -> str:
    """Render a human-readable report from a :func:`compute_calibration`
    result and an optional :func:`compute_per_depth` result.

    When *per_depth* is non-empty (>=1 windowed merge_verify event carried a
    real ``depth``), prints the real per-depth P(pass|depth) block.
    Otherwise falls back to a runner-split section explicitly labelled
    CONFOUNDED with a warning — runner is a broken depth proxy (prefer-local
    scheduling correlates runner with depth, but the correlation is
    confounded by host-env/cold-build effects and flips sign across
    windows), so no numeric runner split is printed, only the warning.
    """
    lines: list[str] = []
    lines.append('=== merge-verify calibration ===')
    lines.append('')

    rate = calibration['per_attempt_pass_rate']
    if rate is not None:
        lines.append(
            f'p_flake, method 1 (per-attempt pass rate over ALL merge_verify '
            f'events): {rate:.3f}'
        )
    else:
        lines.append('p_flake, method 1 (per-attempt pass rate): n/a (no merge_verify events)')

    p_flake_landed = calibration['p_flake_landed']
    if p_flake_landed is not None:
        lines.append(
            f'p_flake, method 2 (1/E[attempts-to-first-pass] among landed '
            f'tasks): {p_flake_landed:.3f}'
        )
    else:
        lines.append('p_flake, method 2 (attempts-to-first-pass): n/a (no landed tasks)')

    if rate is not None and p_flake_landed is not None:
        agree = 'agree' if abs(rate - p_flake_landed) < 0.05 else 'DIVERGE'
        lines.append(f'  -> the two methods {agree} (diff={abs(rate - p_flake_landed):.3f})')

    p_good_bracket = calibration['p_good_bracket']
    if p_good_bracket is not None:
        lo, hi = p_good_bracket
        lines.append(
            f'p_good bracket [1 - genuine_conflict_rate, land_rate]: '
            f'[{lo:.3f}, {hi:.3f}]'
        )
    else:
        lines.append('p_good bracket: n/a')

    retry_histogram = calibration['retry_histogram']
    if retry_histogram:
        hist_str = ', '.join(
            f'{n}x:{count}' for n, count in sorted(retry_histogram.items())
        )
        lines.append(f'attempts-to-first-pass histogram (landed tasks): {hist_str}')
    else:
        lines.append('attempts-to-first-pass histogram: n/a')

    lines.append(
        f"distinct tasks: landed={calibration['landed']} never={calibration['never']}"
    )

    terminal_outcomes = calibration['terminal_outcomes']
    if terminal_outcomes:
        oc_str = ', '.join(f'{k}={v}' for k, v in terminal_outcomes.items())
        lines.append(f'terminal merge_attempt outcomes: {oc_str}')
    else:
        lines.append('terminal merge_attempt outcomes: n/a')

    lines.append('')

    if per_depth:
        lines.append('=== per-depth P(pass|depth) [real depth signal] ===')
        for depth in sorted(per_depth):
            entry = per_depth[depth]
            lines.append(
                f"  depth={depth}: {entry['pass']}/{entry['total']} = "
                f"{entry['rate']:.3f}"
            )
    else:
        lines.append('=== runner-split (depth proxy) — CONFOUNDED ===')
        lines.append(
            'WARNING: runner is a broken depth proxy — it tracks laptop '
            'infra/host-env health, not verify-frontier depth, and flips '
            'sign across windows. No merge_verify event in this window '
            'carries a depth field, so no per-depth P(pass|depth) can be '
            'computed; deepen instrumentation (task 2340) before trusting '
            'any runner-based split.'
        )

    return '\n'.join(lines)


def main(argv: Sequence[str]) -> int:
    """CLI entry point: load, calibrate, and print the report for a runs.db.

    Returns 0 always (this is a read-only analysis; a missing/unreadable DB
    surfaces as an uncaught exception from sqlite3, not a handled exit code
    — matching the tool's `scripts/` siblings, which do not swallow I/O
    errors on their positional path argument).
    """
    parser = argparse.ArgumentParser(
        description='Calibrate merge-verify p_good/p_flake from an orchestrator runs.db.'
    )
    parser.add_argument('db_path', help='Path to the orchestrator runs.db')
    parser.add_argument(
        '--since', type=int, default=30,
        help='Window size in days (default: %(default)s)',
    )
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db_path)
    try:
        merge_verify_events = load_events(conn, 'merge_verify', args.since)
        merge_attempt_events = load_events(conn, 'merge_attempt', args.since)
    finally:
        conn.close()

    calibration = compute_calibration(merge_verify_events, merge_attempt_events)
    per_depth = compute_per_depth(merge_verify_events)
    print(format_report(calibration, per_depth))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
