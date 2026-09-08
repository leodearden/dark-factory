#!/usr/bin/env python3
"""Cut the merge-leg pytest `-n` A/B (16 vs 8) from the orchestrator event store.

Context: 2026-09-08 load-tuning session (Leo). The MERGE test leg's `-n auto`
was pinned via ``verify_env.PYTEST_XDIST_AUTO_NUM_WORKERS`` (16 first, then a
scheduled switch to 8 via scripts/merge-pytest-n-ab-switch.sh). The question:
does 8 cost the serial merge bottleneck material wall-clock now that task-lane
fan-out is bounded (verify_admission_task_slots=1), or is it within noise (in
which case 8 wins on memory and CPU-seconds)?

Arms are derived from ``config_reload`` events whose ``applied.verify_env``
changed the key, so the cut is by the value actually in force at each merge
verify's timestamp, not by calendar date. An orchestrator restart re-reads the
same file, so the file value stays authoritative across restarts.

Stdlib only. Read-only against runs.db (opened with ``mode=ro``). Writes a
markdown report under plans/ (``--no-report`` to skip) and prints a summary
table plus a trailing single-line JSON verdict.

Decision rule (non-speculative, passed merge verifies, n >= MIN_N per arm):
  median(8) <= (1 + TOL_SAME) * median(16)  -> recommend 8   (memory/CPU win, wall-clock within noise)
  median(8) >= (1 + TOL_WORSE) * median(16) -> recommend keep 16
  otherwise                                  -> ambiguous: extend the A/B one more cycle
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

KEY = 'PYTEST_XDIST_AUTO_NUM_WORKERS'
DEFAULT_DB = '/home/leo/src/dark-factory/data/orchestrator/runs.db'
DEFAULT_SINCE = '2026-09-08T10:18:00+00:00'   # the reload that pinned 16 (commit 35b8916784)
MIN_N = 10
TOL_SAME = 0.05
TOL_WORSE = 0.15


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace('Z', '+00:00'))


def load_arms(cur: sqlite3.Cursor, since: datetime) -> list[tuple[datetime, str]]:
    """[(ts, value)] boundaries from config_reload events that changed the key."""
    arms: list[tuple[datetime, str]] = []
    for ts, data in cur.execute(
        "select timestamp, data from events where event_type='config_reload' and timestamp >= ? order by timestamp",
        (since.isoformat(),),
    ):
        try:
            entry = (json.loads(data).get('applied') or {}).get('verify_env')
        except (TypeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        new = entry.get('new') or {}
        arms.append((parse_ts(ts), str(new.get(KEY, 'unset'))))
    return arms


def arm_at(arms: list[tuple[datetime, str]], ts: datetime) -> str:
    cur = 'pre'
    for b_ts, value in arms:
        if b_ts <= ts:
            cur = value
        else:
            break
    return cur


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float('nan')
    ys = sorted(xs)
    k = max(0, min(len(ys) - 1, int(round(p * (len(ys) - 1)))))
    return ys[k]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--since', default=DEFAULT_SINCE, help='ISO-8601; default = the reload that pinned 16')
    ap.add_argument('--report-dir', default='/home/leo/src/dark-factory/plans')
    ap.add_argument('--no-report', action='store_true', help='print only; write no report file')
    ap.add_argument('--commit', action='store_true', help='git commit --only the report (for the scheduled run)')
    args = ap.parse_args()

    since = parse_ts(args.since)
    con = sqlite3.connect(f'file:{args.db}?mode=ro', uri=True)
    cur = con.cursor()
    arms = load_arms(cur, since)

    rows = []
    for ts, data in cur.execute(
        "select timestamp, data from events where event_type='merge_verify' and timestamp >= ? order by timestamp",
        (since.isoformat(),),
    ):
        d = json.loads(data)
        if d.get('runner') not in (None, 'local'):
            continue
        t = parse_ts(ts)
        rows.append({
            'ts': t, 'arm': arm_at(arms, t), 'secs': (d.get('duration_ms') or 0) / 1000.0,
            'passed': bool(d.get('passed')), 'speculative': bool(d.get('speculative')),
            'attempt': d.get('attempt'), 'chain_items': d.get('chain_items'),
        })

    def stats(sel):
        xs = [r['secs'] for r in sel]
        return {
            'n': len(xs), 'median_s': round(statistics.median(xs)) if xs else None,
            'p90_s': round(pct(xs, 0.9)) if xs else None, 'max_s': round(max(xs)) if xs else None,
            'ge_3500s': sum(1 for x in xs if x >= 3500),
        }

    per_arm: dict[str, dict] = {}
    for arm in sorted({r['arm'] for r in rows}):
        sel = [r for r in rows if r['arm'] == arm]
        per_arm[arm] = {
            'all': stats(sel),
            'passed_nonspec': stats([r for r in sel if r['passed'] and not r['speculative']]),
            'failed': sum(1 for r in sel if not r['passed']),
            'speculative': sum(1 for r in sel if r['speculative']),
            'first': min(r['ts'] for r in sel).isoformat(timespec='minutes'),
            'last': max(r['ts'] for r in sel).isoformat(timespec='minutes'),
        }

    a16, a8 = per_arm.get('16', {}).get('passed_nonspec', {}), per_arm.get('8', {}).get('passed_nonspec', {})
    verdict: dict = {'rule': f'median(8) vs median(16), passed non-speculative, n>={MIN_N}/arm, same<=+{TOL_SAME:.0%}, worse>=+{TOL_WORSE:.0%}'}
    if a16.get('n', 0) >= MIN_N and a8.get('n', 0) >= MIN_N and a16.get('median_s'):
        ratio = a8['median_s'] / a16['median_s']
        verdict['ratio_8_over_16'] = round(ratio, 3)
        if ratio <= 1 + TOL_SAME:
            verdict['recommendation'] = 'switch merge leg to 8 (wall-clock within noise; take the memory/CPU-seconds win) and amend task 3589 to land -n 8 in addopts without a merge carve-out'
        elif ratio >= 1 + TOL_WORSE:
            verdict['recommendation'] = 'keep merge leg at 16 (8 costs the serial bottleneck material wall-clock); amend task 3589 to inject an explicit -n 16 on the merge role'
        else:
            verdict['recommendation'] = 'ambiguous: revert to 16 for two more days, then 8 for two more days, and re-run this analysis'
    else:
        verdict['recommendation'] = f'insufficient data: need >={MIN_N} passed non-speculative merge verifies per arm (16: {a16.get("n", 0)}, 8: {a8.get("n", 0)})'

    now = datetime.now(timezone.utc)
    lines = [f'# Merge-leg pytest `-n` A/B (16 vs 8) — cut {now.isoformat(timespec="minutes")}', '',
             f'Source: `{args.db}` (`merge_verify` events, runner=local, since {since.isoformat(timespec="minutes")}).',
             'Arms from `config_reload` events changing `verify_env.' + KEY + '`:', '']
    for b_ts, v in arms:
        lines.append(f'- {b_ts.isoformat(timespec="minutes")} → `{v}`')
    lines += ['', '| arm | window | n (all) | failed | spec | n (passed, non-spec) | median s | p90 s | max s | ≥3500 s |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for arm, s in per_arm.items():
        p = s['passed_nonspec']
        lines.append(f"| {arm} | {s['first']} → {s['last']} | {s['all']['n']} | {s['failed']} | {s['speculative']} | {p['n']} | {p['median_s']} | {p['p90_s']} | {p['max_s']} | {p['ge_3500s']} |")
    lines += ['', '## Verdict', '', f"- rule: {verdict['rule']}"]
    if 'ratio_8_over_16' in verdict:
        lines.append(f"- median ratio 8/16: **{verdict['ratio_8_over_16']}**")
    lines += [f"- **recommendation: {verdict['recommendation']}**", '',
              'Caveats: `merge_verify.duration_ms` is the whole merge verify (all modules, test+lint+type), not the',
              'orchestrator test leg alone; concurrent load is not controlled for beyond the 2026-09-08 bounds',
              '(task slots 1, offline lane/agent shells capped at 8 after the first redeploy). A restart re-reads the',
              'same file, so arms survive restarts. Duration under the merge cold ceiling only; a timed-out verify',
              'is a failed row here.']
    report = '\n'.join(lines) + '\n'
    print(report)
    if not args.no_report:
        out = Path(args.report_dir) / f'merge-pytest-n-ab-report-{now.strftime("%Y-%m-%d")}.md'
        out.write_text(report, encoding='utf-8')
        print(f'report: {out}', file=sys.stderr)
        if args.commit:
            repo = str(Path(args.report_dir).parent)
            subprocess.run(['git', '-C', repo, 'add', '--', str(out)], check=True)
            subprocess.run(['git', '-C', repo, 'commit', '--only', str(out), '-q', '-m',
                            f'plans: merge-leg pytest -n A/B report {now.strftime("%Y-%m-%d")} (scripts/merge-pytest-n-ab-analysis.py)'], check=True)
    print(json.dumps({'per_arm': {k: v['passed_nonspec'] for k, v in per_arm.items()}, **verdict}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
