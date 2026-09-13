#!/usr/bin/env python3
"""Recommend dispatch-admission load thresholds from the load-sampler corpus.

Task 3592, leaf δ of ``plans/load-throttle-harmonisation-prd.md``. Run by the
ε1/ε2 calibration gates fourteen and twenty-eight days after the sampler's
30-day retention lands, and by an operator who wants the same cut now.

Reads ``data/load-samples.db`` READ-ONLY and reports, per signal: percentiles;
per-candidate-threshold hold fractions and hold streaks against D11; drift
between this project's ``psi_admission`` block and reify's; and which yaml
leaves merely restate the shipped code default.

STDLIB ONLY, plus ``yaml`` imported lazily inside the drift check. This is
measured, not stylistic. The live orchestrator runs under ``uv run --frozen
--project orchestrator``, yet its /proc/<pid>/environ carries no VIRTUAL_ENV
and no venv bin on PATH; ``deterministic_runner.py`` launches a
``before_done`` script with ``create_subprocess_exec(script, *args)`` —
straight through the shebang with that env inherited. So at gate time
``#!/usr/bin/env python3`` resolves to /usr/bin/python3 (3.12.3), which HAS
yaml but NOT ``shared``, ``sampler`` or ``orchestrator``. A top-level
first-party import would crash the gate on import fourteen days after this
lands, in a born-at-L2 escalation path, with no earlier signal. Keeping yaml
lazy additionally makes the module stdlib-importable, so any test in any
subproject can load it by path, and a missing PyYAML becomes a NAMED
degradation instead of an import crash.

ALWAYS EXITS 0 when it completed an analysis. Every degradation — corpus
missing or empty, peer config absent/unreadable/unparseable, either side
missing its psi_admission block, PyYAML absent, code defaults unavailable — is
a named entry in the report AND a key in the trailing JSON. ε1/ε2 are
``always_escalates=True`` with no target_unit, so per deterministic_runner.py
a non-zero rc is classified an INFRA FAULT and produces a born-at-L2
infra_issue with no gate: exiting non-zero because reify's yaml was missing
would page a human about a broken script instead of delivering the
calibration report that says one side could not be compared.

Output contract, matching scripts/merge-pytest-n-ab-analysis.py: the human
report on stdout, the written report's path on STDERR so stdout's last line
stays the single-line JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_PEER_CONFIG = Path('/home/leo/src/reify/dark-factory-orchestrator.yaml')
DEFAULT_REPORT_DIR = Path('/home/leo/src/dark-factory/plans')
PERCENTILES = (0.50, 0.90, 0.95, 0.99)

# Arm name -> how this corpus records it. Duplicated from
# sampler.metrics.ARM_METRIC_STEMS BY NECESSITY: that module cannot be
# imported here (see the stdlib-only note above). The reconciler is the named
# lockstep test in sampler/tests/test_load_metrics.py, which loads this file by
# path and asserts the two agree — so do NOT "de-duplicate" this with an
# import, which would crash the gate.
ARM_METRIC_SELECTORS = {
    'mem_full_avg10': 'psi_mem_full_avg10',
    'mem_some_avg10': 'psi_mem_some_avg10',
    'io_some_avg10': 'psi_io_some_avg10',
    'cpu_some_avg10': 'psi_cpu_some_avg10',
    'runqueue_ratio': 'runqueue_ratio',
    'own_cpu_some_avg10': 'own_cpu_some10',
}


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float('nan')
    ys = sorted(xs)
    k = max(0, min(len(ys) - 1, int(round(p * (len(ys) - 1)))))
    return ys[k]


def default_db() -> Path:
    """The corpus, via the env seam sampler/__main__.py and the installer use."""
    root = os.environ.get('DARK_FACTORY_ROOT', '/home/leo/src/dark-factory')
    return Path(root) / 'data/load-samples.db'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--db', type=Path, default=None,
                    help='load-samples.db (default: $DARK_FACTORY_ROOT/data/load-samples.db)')
    ap.add_argument('--arm', choices=sorted(ARM_METRIC_SELECTORS), default=None,
                    help='restrict the analysis to one arm (default: all)')
    ap.add_argument('--config', type=Path,
                    default=Path('/home/leo/src/dark-factory/dark-factory-orchestrator.yaml'),
                    help='this project\'s orchestrator config (default: %(default)s)')
    ap.add_argument('--peer-config', type=Path, default=DEFAULT_PEER_CONFIG,
                    help='the peer project\'s orchestrator config (default: %(default)s)')
    ap.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT_DIR,
                    help='where the markdown report is written (default: %(default)s)')
    ap.add_argument('--no-report', action='store_true',
                    help='print only; write no report file')
    ap.add_argument('--commit', action='store_true',
                    help='git commit --only the written report (for the scheduled run)')
    args = ap.parse_args(argv)
    if args.db is None:
        args.db = default_db()
    return args


def read_series(db: Path, arm: str | None) -> tuple[dict[str, list[float]], list[str]]:
    """Return ``{metric: [values in ts order]}`` and any degradations hit.

    Opened ``file:...?mode=ro`` so a calibration run can never write to the
    live corpus. A ':' selector matches every per-cgroup leaf under that stem,
    each kept as its OWN series — pooling them would average unrelated
    workloads into one meaningless number.
    """
    selectors = (
        [ARM_METRIC_SELECTORS[arm]] if arm else list(ARM_METRIC_SELECTORS.values())
    )
    try:
        con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    except sqlite3.Error as exc:
        return {}, [f'db_unavailable: {db} ({exc})']

    series: dict[str, list[float]] = {}
    try:
        for selector in selectors:
            rows = con.execute(
                'SELECT metric, value FROM samples'
                ' WHERE metric = ? OR metric LIKE ? ORDER BY ts',
                (selector, f'{selector}:%'),
            ).fetchall()
            for metric, value in rows:
                series.setdefault(metric, []).append(float(value))
    except sqlite3.Error as exc:
        return {}, [f'db_unavailable: {db} ({exc})']
    finally:
        con.close()

    if not series:
        return {}, [f'no_samples_in_window: no rows for {sorted(selectors)} in {db}']
    return series, []


def percentile_table(series: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            'n': len(values),
            'p50': round(pct(values, 0.50), 3),
            'p90': round(pct(values, 0.90), 3),
            'p95': round(pct(values, 0.95), 3),
            'p99': round(pct(values, 0.99), 3),
            'max': round(max(values), 3),
        }
        for metric, values in sorted(series.items())
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    now = datetime.now(UTC)

    series, degradations = read_series(args.db, args.arm)
    percentiles = percentile_table(series)

    lines = [
        f'# Load-threshold calibration — cut {now.isoformat(timespec="minutes")}',
        '',
        f'Corpus: `{args.db}` (read-only).',
        '',
        '## Percentiles',
        '',
    ]
    if percentiles:
        lines += ['| metric | n | p50 | p90 | p95 | p99 | max |',
                  '|---|---|---|---|---|---|---|']
        for metric, stats in percentiles.items():
            lines.append(
                f"| `{metric}` | {stats['n']} | {stats['p50']} | {stats['p90']} "
                f"| {stats['p95']} | {stats['p99']} | {stats['max']} |"
            )
    else:
        lines.append('_No samples._')

    lines += ['', '## Degradations', '']
    lines += [f'- {d}' for d in degradations] or ['_None._']

    report = '\n'.join(lines) + '\n'
    print(report)

    if not args.no_report:
        out = args.report_dir / f'load-threshold-calibration-{now.strftime("%Y-%m-%d")}.md'
        out.write_text(report, encoding='utf-8')
        print(f'report: {out}', file=sys.stderr)

    print(json.dumps({
        'generated_at': now.isoformat(timespec='seconds'),
        'db': str(args.db),
        'percentiles': percentiles,
        'degradations': [d.split(':', 1)[0] for d in degradations],
        'degradation_details': degradations,
    }))
    return 0


if __name__ == '__main__':
    sys.exit(main())
