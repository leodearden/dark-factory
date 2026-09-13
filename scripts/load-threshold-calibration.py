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
from typing import NamedTuple

DEFAULT_PEER_CONFIG = Path('/home/leo/src/reify/dark-factory-orchestrator.yaml')
DEFAULT_REPORT_DIR = Path('/home/leo/src/dark-factory/plans')
PERCENTILES = (0.50, 0.90, 0.95, 0.99)

class ArmSpec(NamedTuple):
    """Everything the report needs about one gate arm.

    ``selector`` is the sampler metric recording this arm. ``is_stem`` says
    whether it names a metric outright or a ':' prefix with one series per
    cgroup leaf — the two are read differently and reported separately, never
    pooled. ``ladder`` is the candidate thresholds to evaluate hold fractions
    at. ``unit`` labels the numbers for the human reading the escalation.
    """

    selector: str
    is_stem: bool
    ladder: tuple[float, ...]
    unit: str


# Percentage-pressure arms share one ladder: a PSI avg10 is a percentage of
# wall time stalled, so the same rungs mean the same thing for all of them.
_PRESSURE_LADDER = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
_PRESSURE_UNIT = '% of wall time stalled (PSI avg10)'

# Arm name -> its ArmSpec. The `selector` half is duplicated from
# sampler.metrics.ARM_METRIC_STEMS BY NECESSITY: that module cannot be
# imported here, because at gate time this script runs under the system
# python3 (see the stdlib-only note above). The reconciler is the named
# lockstep test in sampler/tests/test_load_metrics.py, which loads this file by
# path and asserts the two agree in both directions — so do NOT
# "de-duplicate" this with an import, which would crash the gate.
ARM_METRIC_SELECTORS = {
    'mem_full_avg10': ArmSpec(
        'psi_mem_full_avg10', False, _PRESSURE_LADDER, _PRESSURE_UNIT),
    'mem_some_avg10': ArmSpec(
        'psi_mem_some_avg10', False, _PRESSURE_LADDER, _PRESSURE_UNIT),
    'io_some_avg10': ArmSpec(
        'psi_io_some_avg10', False, _PRESSURE_LADDER, _PRESSURE_UNIT),
    'cpu_some_avg10': ArmSpec(
        'psi_cpu_some_avg10', False, _PRESSURE_LADDER, _PRESSURE_UNIT),
    # A RATIO, not a percentage: procs_running / len(sched_getaffinity(0)).
    # 1.0 is "as many runnable threads as CPUs"; 4.0 is PRD D9's provisional.
    'runqueue_ratio': ArmSpec(
        'runqueue_ratio', False,
        (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0),
        'runnable threads per CPU (ratio)'),
    # One series per cgroup leaf, so ':' — reported per leaf, never pooled.
    'own_cpu_some_avg10': ArmSpec(
        'own_cpu_some10', True, _PRESSURE_LADDER, _PRESSURE_UNIT),
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


def read_series(
    db: Path, arm: str | None
) -> tuple[dict[str, list[tuple[int, float]]], list[str]]:
    """Return ``{metric: [(ts, value) in ts order]}`` and any degradations hit.

    Opened ``file:...?mode=ro`` so a calibration run can never write to the
    live corpus. A ':' selector matches every per-cgroup leaf under that stem,
    each kept as its OWN series — pooling them would average unrelated
    workloads into one meaningless number.
    """
    specs = (
        [ARM_METRIC_SELECTORS[arm]] if arm else list(ARM_METRIC_SELECTORS.values())
    )
    selectors = [spec.selector for spec in specs]
    try:
        con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    except sqlite3.Error as exc:
        return {}, [f'db_unavailable: {db} ({exc})']

    series: dict[str, list[tuple[int, float]]] = {}
    try:
        for selector in selectors:
            rows = con.execute(
                'SELECT metric, ts, value FROM samples'
                ' WHERE metric = ? OR metric LIKE ? ORDER BY ts',
                (selector, f'{selector}:%'),
            ).fetchall()
            for metric, ts, value in rows:
                series.setdefault(metric, []).append((int(ts), float(value)))
    except sqlite3.Error as exc:
        return {}, [f'db_unavailable: {db} ({exc})']
    finally:
        con.close()

    if not series:
        return {}, [f'no_samples_in_window: no rows for {sorted(selectors)} in {db}']
    return series, []


def percentile_table(
    series: dict[str, list[float]]
) -> dict[str, dict[str, float]]:
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


# PRD D11: "the gate holds on a minority of ticks (target <= 20%)".
D11_HOLD_FRACTION_TARGET = 0.20

# The paired .timer's OnUnitActiveSec. Only a FALLBACK for a corpus too short
# to measure spacing from; the real value is read off the data.
NOMINAL_TICK_SECONDS = 5


def hold_fraction(values: list[float], threshold: float) -> float:
    """Fraction of samples at or over *threshold*.

    ``>=`` mirrors shared.psi's arm comparison (``arm.value(self) >=
    threshold``), so this report and the live gate cannot disagree about the
    boundary — a report using ``>`` would recommend a threshold the gate then
    behaves differently at.
    """
    if not values:
        return 0.0
    return sum(1 for v in values if v >= threshold) / len(values)


def within_d11_target(fraction: float) -> bool:
    return fraction <= D11_HOLD_FRACTION_TARGET


def observed_spacing(timestamps: list[int]) -> int:
    """Median gap between consecutive samples, in seconds.

    Measured rather than assumed: 5 s is the UNIT's setting, not a property of
    the corpus, and a corpus written at another cadence would otherwise get
    wrong wall-clock. Too few samples to measure falls back to the nominal
    cadence rather than raising — a one-row corpus is a thin report, not an
    error.
    """
    gaps = sorted(
        b - a for a, b in zip(timestamps, timestamps[1:], strict=False) if b > a
    )
    if not gaps:
        return NOMINAL_TICK_SECONDS
    return gaps[len(gaps) // 2]


def _human_duration(seconds: int) -> str:
    if seconds >= 3600:
        return f'{seconds // 3600}h{(seconds % 3600) // 60:02d}m'
    if seconds >= 60:
        return f'{seconds // 60}m'
    return f'{seconds}s'


def longest_hold_run(
    values: list[float], threshold: float, *, spacing_seconds: int
) -> dict:
    """Longest CONSECUTIVE run of holding ticks, in ticks and wall-clock.

    This is D11's second clause — "never sits at the floor for hours" — read
    operationally. The literal quantity that clause names, in-flight sitting
    at ``min_inflight_floor``, is NOT recorded in load-samples.db and cannot be
    recovered from it; the hold streak is the closest thing this corpus
    supports and it answers the question the clause is asking, without
    inventing a signal. A reviewer reading the number should know it is a hold
    streak, not a floor-occupancy measurement.

    It is reported BESIDE the fraction, not instead of it, because a fraction
    alone cannot tell 20% delivered as single-tick blips from 20% delivered as
    one continuous block — opposite verdicts for a dispatch throttle.
    """
    longest = current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        longest = max(longest, current)
    seconds = longest * spacing_seconds
    return {'ticks': longest, 'seconds': seconds, 'human': _human_duration(seconds)}


def load_psi_admission_block(path: Path, side: str) -> tuple[dict | None, list[str]]:
    """Parse one orchestrator config and return its ``psi_admission`` mapping.

    Every failure mode gets its OWN named degradation carrying the path and
    the reason, and none of them raises: the consumer is a human reading an
    escalation, and "reify's yaml was missing" must arrive as a sentence in the
    report rather than as a crash.

    ``yaml`` is imported HERE rather than at module scope. At gate time this
    script runs under the system python3, which happens to have PyYAML — but
    the module must stay stdlib-importable so any subproject's test can load
    it by path, and so a missing PyYAML degrades by name instead of crashing
    on import.

    An ABSENT block returns ``None``, never ``{}``: "this project has not
    configured psi_admission" and "it configured an empty block" are different
    facts, and collapsing them would let a one-sided comparison report a
    spurious match.
    """
    try:
        import yaml
    except ImportError as exc:
        return None, [f'pyyaml_absent: cannot parse {side} config {path} ({exc})']
    if not path.is_file():
        return None, [f'{side}_config_missing: {path}']
    try:
        parsed = yaml.safe_load(path.read_text())
    except OSError as exc:
        return None, [f'{side}_config_unreadable: {path} ({exc})']
    except yaml.YAMLError as exc:
        return None, [f'{side}_config_unparseable: {path} ({exc})']
    if not isinstance(parsed, dict):
        return None, [f'{side}_config_unparseable: {path} (top level is not a mapping)']
    block = parsed.get('psi_admission')
    if block is None:
        return None, [f'{side}_psi_admission_absent: {path}']
    if not isinstance(block, dict):
        return None, [f'{side}_psi_admission_absent: {path} (not a mapping)']
    return block, []


def arm_thresholds(block: dict | None) -> dict[str, float]:
    """The arm thresholds a parsed block actually sets.

    Reported beside each ladder so the report always evaluates the value in
    force, not only the hypotheticals. Non-arm leaves (``enabled``,
    ``min_inflight_floor``) and non-numeric values are skipped; ``bool`` is
    excluded explicitly because in Python it IS an int, and `enabled: true`
    would otherwise read as the threshold 1.
    """
    if block is None:
        return {}
    return {
        arm: float(value) for arm, value in block.items()
        if arm in ARM_METRIC_SELECTORS and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def compare_blocks(local: dict | None, peer: dict | None) -> dict | None:
    """Compare two parsed ``psi_admission`` mappings; ``None`` if either is absent.

    Returns ``{'drift': [{leaf, local, peer}], 'local_only': [...],
    'peer_only': [...]}``.

    Over PARSED MAPPINGS, never text (INV-10): key order, indentation,
    comments, `4.00` against `4.0`, `7.0e1` against `70.0` and `yes` against
    `true` are all yaml SPELLINGS of the same value, and a textual diff would
    report every one of them as drift and bury the one that matters.

    ``None`` in means ``None`` out. An absent block is NOT an empty match: a
    project with no ``psi_admission`` at all would otherwise report perfect
    agreement with one that has six leaves. An actually-empty block on both
    sides is a real comparison that happens to find nothing.

    A ONE-SIDED LEAF is reported separately from a value mismatch — "the two
    projects disagree about this number" and "only one project has this knob"
    call for different operator actions.
    """
    if local is None or peer is None:
        return None
    shared_leaves = sorted(set(local) & set(peer))
    return {
        'drift': [
            {'leaf': leaf, 'local': local[leaf], 'peer': peer[leaf]}
            for leaf in shared_leaves
            if local[leaf] != peer[leaf]
        ],
        'local_only': sorted(set(local) - set(peer)),
        'peer_only': sorted(set(peer) - set(local)),
    }


def _arm_for(metric: str, specs: list[ArmSpec]) -> str | None:
    """The arm name a recorded metric belongs to (':' stems included)."""
    stem = metric.split(':', 1)[0]
    for arm, spec in ARM_METRIC_SELECTORS.items():
        if spec in specs and spec.selector in (metric, stem):
            return arm
    return None


def _spec_for(metric: str, specs: list[ArmSpec]) -> ArmSpec | None:
    arm = _arm_for(metric, specs)
    return ARM_METRIC_SELECTORS[arm] if arm else None


def hold_table(
    series: dict[str, list[tuple[int, float]]],
    specs: list[ArmSpec],
) -> dict[str, list[dict]]:
    """Per metric, one entry per candidate threshold on its arm's ladder."""
    ladders = {spec.selector: spec.ladder for spec in specs}
    out: dict[str, list[dict]] = {}
    for metric, points in series.items():
        stem = metric.split(':', 1)[0]
        ladder = ladders.get(stem) or ladders.get(metric)
        if ladder is None:
            continue
        timestamps = [ts for ts, _ in points]
        values = [v for _, v in points]
        spacing = observed_spacing(timestamps)
        out[metric] = [
            {
                'threshold': threshold,
                'hold_fraction': round(hold_fraction(values, threshold), 4),
                'within_d11_target': within_d11_target(hold_fraction(values, threshold)),
                'longest_hold_run': longest_hold_run(
                    values, threshold, spacing_seconds=spacing
                ),
            }
            for threshold in ladder
        ]
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    now = datetime.now(UTC)

    series, degradations = read_series(args.db, args.arm)
    specs = (
        [ARM_METRIC_SELECTORS[args.arm]] if args.arm
        else list(ARM_METRIC_SELECTORS.values())
    )
    percentiles = percentile_table({m: [v for _, v in pts] for m, pts in series.items()})
    holds = hold_table(series, specs)
    local_block, local_degradations = load_psi_admission_block(args.config, 'local')
    peer_block, peer_degradations = load_psi_admission_block(args.peer_config, 'peer')
    degradations += local_degradations + peer_degradations
    configured = arm_thresholds(local_block)
    drift = compare_blocks(local_block, peer_block)

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

    lines += ['', '## Candidate thresholds (PRD D11)', '',
              f'Target: the gate holds on <= {D11_HOLD_FRACTION_TARGET:.0%} of ticks. '
              'The longest hold RUN is reported beside the fraction because the same '
              'fraction delivered as one long block and as isolated blips are opposite '
              'verdicts for a dispatch throttle. It is a hold streak, not a '
              'floor-occupancy measurement — in-flight floor occupancy is not in this '
              'corpus.', '']
    for metric, candidates in holds.items():
        spec = _spec_for(metric, specs)
        in_force = configured.get(_arm_for(metric, specs), None)
        lines += [
            f'### `{metric}` — {spec.unit if spec else ""}',
            '',
            f'Configured value in force: '
            f'{in_force if in_force is not None else "(none — see degradations)"}',
            '',
            '| candidate | hold fraction | <= target | longest hold run |',
            '|---|---|---|---|',
        ]
        for candidate in candidates:
            run = candidate['longest_hold_run']
            mark = 'yes' if candidate['within_d11_target'] else 'NO'
            lines.append(
                f"| {candidate['threshold']} | {candidate['hold_fraction']:.1%} "
                f"| {mark} | {run['ticks']} ticks ({run['human']}) |"
            )
        lines.append('')

    lines += ['', '## Config drift against the peer project', '',
              f'Local: `{args.config}`  ·  Peer: `{args.peer_config}`', '']
    if drift is None:
        lines.append(
            'NOT COMPARED — at least one side has no `psi_admission` block. See '
            'the degradations below; this is a one-sided run, not a match.')
    elif not (drift['drift'] or drift['local_only'] or drift['peer_only']):
        lines.append('No drift: both blocks set the same leaves to the same values.')
    else:
        for entry in drift['drift']:
            lines.append(
                f"- `{entry['leaf']}`: local `{entry['local']}` vs peer `{entry['peer']}`")
        for leaf in drift['local_only']:
            lines.append(f'- `{leaf}`: set locally only')
        for leaf in drift['peer_only']:
            lines.append(f'- `{leaf}`: set by the peer only')

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
        'holds': holds,
        'configured': configured,
        'drift': drift,
        'degradations': [d.split(':', 1)[0] for d in degradations],
        'degradation_details': degradations,
    }))
    return 0


if __name__ == '__main__':
    sys.exit(main())
