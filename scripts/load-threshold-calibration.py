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
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

DEFAULT_PEER_CONFIG = Path('/home/leo/src/reify/dark-factory-orchestrator.yaml')
DEFAULT_REPORT_DIR = Path('/home/leo/src/dark-factory/plans')
PERCENTILES = (0.50, 0.90, 0.95, 0.99)

# {metric: [(ts, value) in ts order]} — what every reader here hands around.
Series = dict[str, list[tuple[int, float]]]

class ArmSpec(NamedTuple):
    """Everything the report needs about one gate arm.

    ``selector`` is the sampler metric recording this arm. ``is_stem`` says
    whether it names a metric outright or a ':' prefix with one series per
    cgroup leaf — the two are read differently and reported separately, never
    pooled. ``ladder`` is the candidate thresholds to evaluate hold fractions
    at. ``unit`` labels the numbers for the human reading the escalation.

    ``readability`` names the ``*_read_ok`` metric recording whether this arm
    was readable at all, or ``None`` when the collector emits none. It has no
    default, deliberately: a new arm cannot be added without deciding the
    question, and an arm whose readability is unknown must SAY so rather than
    be reported as fully covered.
    """

    selector: str
    is_stem: bool
    ladder: tuple[float, ...]
    unit: str
    readability: str | None


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
        'psi_mem_full_avg10', False, _PRESSURE_LADDER, _PRESSURE_UNIT, None),
    'mem_some_avg10': ArmSpec(
        'psi_mem_some_avg10', False, _PRESSURE_LADDER, _PRESSURE_UNIT, None),
    'io_some_avg10': ArmSpec(
        'psi_io_some_avg10', False, _PRESSURE_LADDER, _PRESSURE_UNIT, None),
    'cpu_some_avg10': ArmSpec(
        'psi_cpu_some_avg10', False, _PRESSURE_LADDER, _PRESSURE_UNIT, None),
    # A RATIO, not a percentage: procs_running / len(sched_getaffinity(0)).
    # 1.0 is "as many runnable threads as CPUs"; 4.0 is PRD D9's provisional.
    'runqueue_ratio': ArmSpec(
        'runqueue_ratio', False,
        (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0),
        'runnable threads per CPU (ratio)', 'runqueue_read_ok'),
    # One series per cgroup leaf, so ':' — reported per leaf, never pooled.
    'own_cpu_some_avg10': ArmSpec(
        'own_cpu_some10', True, _PRESSURE_LADDER, _PRESSURE_UNIT, 'own_read_ok'),
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
    ap.add_argument('--uv-bin', type=Path, default=DEFAULT_UV_BIN,
                    help='uv binary used to ask the live model for its code '
                         'defaults (default: %(default)s)')
    ap.add_argument('--commit', action='store_true',
                    help='git commit --only the written report (for the scheduled run)')
    args = ap.parse_args(argv)
    if args.db is None:
        args.db = default_db()
    return args


def _fetch(con: sqlite3.Connection, selectors: list[str]) -> Series:
    """``{metric: [(ts, value) in ts order]}`` for every selector, one home.

    Shared by the value series and the readability series so the GLOB spelling
    and its reasoning live in exactly one place (heuristic 11).
    """
    series: Series = {}
    for selector in selectors:
        rows = con.execute(
            'SELECT metric, ts, value FROM samples'
            ' WHERE metric = ? OR metric GLOB ? ORDER BY ts',
            (selector, f'{selector}:*'),
        ).fetchall()
        for metric, ts, value in rows:
            series.setdefault(metric, []).append((int(ts), float(value)))
    return series


def read_series(
    db: Path, arm: str | None
) -> tuple[Series, Series, list[str]]:
    """Return the value series, the READABILITY series, and any degradations.

    The two are returned apart and never merged: a ``*_read_ok`` row is
    evidence ABOUT a series, not a sample of it, so pooling them would corrupt
    the very percentiles and hold fractions it exists to qualify.

    Opened ``file:...?mode=ro`` so a calibration run can never write to the
    live corpus. A ':' selector matches every per-cgroup leaf under that stem,
    each kept as its OWN series — pooling them would average unrelated
    workloads into one meaningless number.

    The stem match is ``GLOB``, not ``LIKE``, for two independently sufficient
    reasons. Correctness: ``_`` is a single-character wildcard in LIKE and
    every selector contains one, so ``own_cpu_some10:%`` also matches
    ``own-cpu-some10:leaf`` — and this repo spells unit and slice names with
    hyphens throughout, making that a realistic next metric name. GLOB has no
    ``_`` wildcard. Cost: LIKE is ASCII-case-INsensitive by default, so it
    cannot use a BINARY-collated index and this query planned as
    ``SCAN samples``; GLOB is always case-sensitive, so the prefix
    optimisation applies and the plan becomes ``MULTI-INDEX OR`` over
    ``SEARCH samples USING INDEX idx_samples_metric_ts (metric=?)`` plus
    ``(metric>? AND metric<?)``. Measured on a 2.16M-row probe with this exact
    schema. At the 30-day steady state the corpus is ~13M rows, so the LIKE
    spelling was a full scan per selector — the same shape of cost regression
    the dashboard's ``/api/load`` query carried before it was bounded.
    """
    specs = (
        [ARM_METRIC_SELECTORS[arm]] if arm else list(ARM_METRIC_SELECTORS.values())
    )
    selectors = [spec.selector for spec in specs]
    try:
        con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    except sqlite3.Error as exc:
        return {}, {}, [f'db_unavailable: {db} ({exc})']

    try:
        series = _fetch(con, selectors)
        readability = _fetch(
            con, [spec.readability for spec in specs if spec.readability]
        )
    except sqlite3.Error as exc:
        return {}, {}, [f'db_unavailable: {db} ({exc})']
    finally:
        con.close()

    if not series:
        return {}, {}, [
            f'no_samples_in_window: no rows for {sorted(selectors)} in {db}'
        ]
    return series, readability, []


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

# Below this readable fraction, a series' hold fractions are reported with a
# named degradation. It is a REPORTING threshold, not a decision one: the exact
# coverage is printed either way, so no verdict depends on where this sits — it
# decides only when the report shouts.
D11_READABILITY_FLOOR = 0.95

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


_DEFAULTS_DUMP = (
    'import json;'
    ' from orchestrator.config import PsiAdmissionConfig as C;'
    ' print(json.dumps({k: f.default for k, f in C.model_fields.items()},'
    ' default=str))'
)
DEFAULT_UV_BIN = Path('/home/leo/.local/bin/uv')
_DEFAULTS_TIMEOUT_SECONDS = 120


def default_defaults_command(uv_bin: Path) -> list[str]:
    """The command that asks the live model for its own defaults.

    `--no-sync` is LOAD-BEARING, not cosmetic: a plain `uv run --project
    shared` was measured REMOVING orchestrator from the shared root venv, and
    this script may run while orchestrators are live. `--frozen` additionally
    pins the lockfile.

    The defaults are FETCHED rather than imported or ast-parsed. Imported is
    impossible — at gate time this runs under the system python3, which has no
    `orchestrator`. An ast walk over orchestrator/config.py would be an ad-hoc
    parser of source (heuristic 12) that breaks silently on a `Field()`
    respelling. So the script asks the authoritative object, via the one tool
    that IS on the inherited PATH.
    """
    return [
        str(uv_bin), 'run', '--frozen', '--no-sync',
        '--project', 'orchestrator', 'python', '-c', _DEFAULTS_DUMP,
    ]


def fetch_code_defaults(
    *, command: list[str], cwd: Path
) -> tuple[dict | None, list[str]]:
    """Run *command* and parse its JSON stdout as the shipped code defaults.

    Any failure — the binary absent, a non-zero exit, unparseable output, a
    timeout — returns ``(None, [code_defaults_unavailable: <reason>])``. The
    reason is carried so an operator can tell "uv is not installed" from "the
    model moved"; nothing is ever assumed in place of a real answer.
    """
    try:
        proc = subprocess.run(
            command, cwd=str(cwd), capture_output=True, text=True,
            timeout=_DEFAULTS_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, [f'code_defaults_unavailable: {exc}']
    if proc.returncode != 0:
        return None, [
            f'code_defaults_unavailable: {command[0]} exited {proc.returncode} '
            f'({proc.stderr.strip()[:200]})'
        ]
    try:
        parsed = json.loads(proc.stdout)
    except ValueError as exc:
        return None, [f'code_defaults_unavailable: unparseable output ({exc})']
    if not isinstance(parsed, dict):
        return None, ['code_defaults_unavailable: output is not a JSON object']
    return parsed, []


def compare_to_code_defaults(block: dict | None, defaults: dict | None) -> dict | None:
    """Which of *block*'s arm leaves merely restate the shipped code default.

    PRD §6.2 writes three of the memory/io thresholds at values that are
    ALREADY the shipped code defaults, giving one fact three homes with no
    reconciler (INV-9). This is the report line that says so. It stays a
    REPORT line: the script never edits either yaml and never removes a leaf.

    The numbers themselves are deliberately not written here, not even as
    prose: a docstring copy goes stale exactly as silently as a code copy.

    ``defaults`` is required and has no fallback — a built-in copy here would
    make this script the fourth home of the very fact the check exists to
    police, and would go on reporting against its own stale numbers long after
    the model changed.

    A leaf the defaults mapping does not know is ``unknown_to_schema``, not a
    silent drop: measured today, PsiAdmissionConfig has no ``runqueue_ratio``
    field (β unlanded), and "this arm is not in the model yet" is a distinct
    fact from both "restates the default" and "differs from it".
    """
    if block is None or defaults is None:
        return None
    restates, unknown = [], []
    for leaf, value in block.items():
        if leaf not in ARM_METRIC_SELECTORS:
            continue
        if leaf not in defaults:
            unknown.append(leaf)
        elif value == defaults[leaf]:
            restates.append(leaf)
    return {'restates_default': sorted(restates), 'unknown_to_schema': sorted(unknown)}


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


def coverage_table(
    series: Series,
    readability: Series,
    specs: list[ArmSpec],
) -> dict[str, dict[str, float] | None]:
    """Per VALUE metric, the readable-tick coverage its numbers rest on.

    A failed read emits no value row at all, so ``hold_fraction``'s denominator
    is the number of SUCCESSFUL reads rather than the number of ticks. A
    ``*_read_ok`` row IS emitted every tick, so its row count is the tick count
    and its 1.0 count is the readable count — which is the whole reason
    ``collect_load_metrics`` persists it as a metric instead of a log line.
    Without this, "holds on 12% of samples" reads identically whether the
    corpus covered a fortnight or the 3% of it that was readable, and those are
    opposite verdicts for setting a dispatch threshold.

    ``None`` for an arm whose collector emits no readability metric — the four
    host-PSI arms. Reporting a fabricated 1.0 there would be the same class of
    defect as persisting α's fail-open 0.0 as a ratio.

    Keyed by the value metric so a ':' stem reports PER LEAF: one cgroup can be
    unreadable while its siblings are fine, which is exactly the case worth
    seeing.
    """
    out: dict[str, dict[str, float] | None] = {}
    for metric in series:
        arm = _arm_for(metric, specs)
        spec = ARM_METRIC_SELECTORS[arm] if arm else None
        if spec is None:
            continue
        if spec.readability is None:
            out[metric] = None
            continue
        _stem, separator, tail = metric.partition(':')
        key = f'{spec.readability}:{tail}' if separator else spec.readability
        points = readability.get(key, [])
        ticks = len(points)
        readable = sum(1 for _ts, value in points if value == 1.0)
        out[metric] = {
            'ticks': ticks,
            'readable': readable,
            'readable_fraction': round(readable / ticks, 4) if ticks else 0.0,
        }
    return out


def readability_degradations(
    coverage: dict[str, dict[str, float] | None]
) -> list[str]:
    """One named degradation per series whose coverage is below the floor."""
    return [
        f"low_readability: {metric} readable on {stats['readable']}/"
        f"{stats['ticks']} ticks ({stats['readable_fraction']:.1%}), below the "
        f'{D11_READABILITY_FLOOR:.0%} floor — read its hold fractions against '
        'that coverage, not as a fortnight'
        for metric, stats in sorted(coverage.items())
        if stats is not None and stats['readable_fraction'] < D11_READABILITY_FLOOR
    ]


def commit_report(path: Path, stamp: str) -> list[str]:
    """`git add --` then `git commit --only <path>`; degrade named on failure.

    `--only` and not a bare `git commit`: the repo this runs in is
    machine-operated — the merge worker, the startup reconciler and git hooks
    all act on it — so a bare commit would sweep in whatever a concurrent
    process happens to have staged. That is a live hazard here, not a
    stylistic preference.

    A git failure is a NAMED degradation, never a non-zero exit. The analysis
    is the deliverable and committing it is a convenience; ε1/ε2 classify a
    non-zero rc as an INFRA FAULT with no gate, so letting git turn a
    delivered calibration into a born-at-L2 page would be exactly backwards.
    """
    repo = str(path.parent.parent)
    subject = (
        f'plans: load-threshold calibration report {stamp} '
        '(scripts/load-threshold-calibration.py)'
    )
    try:
        for argv in (
            ['git', '-C', repo, 'add', '--', str(path)],
            ['git', '-C', repo, 'commit', '--only', str(path), '-q', '-m', subject],
        ):
            proc = subprocess.run(argv, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                return [
                    f'report_commit_failed: {" ".join(argv[:4])} exited '
                    f'{proc.returncode} ({proc.stderr.strip()[:200]})'
                ]
    except (OSError, subprocess.SubprocessError) as exc:
        return [f'report_commit_failed: {exc}']
    return []


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    now = datetime.now(UTC)
    stamp = now.strftime('%Y-%m-%d')

    series, readability, degradations = read_series(args.db, args.arm)
    specs = (
        [ARM_METRIC_SELECTORS[args.arm]] if args.arm
        else list(ARM_METRIC_SELECTORS.values())
    )
    percentiles = percentile_table({m: [v for _, v in pts] for m, pts in series.items()})
    holds = hold_table(series, specs)
    coverage = coverage_table(series, readability, specs)
    degradations += readability_degradations(coverage)
    local_block, local_degradations = load_psi_admission_block(args.config, 'local')
    peer_block, peer_degradations = load_psi_admission_block(args.peer_config, 'peer')
    degradations += local_degradations + peer_degradations
    configured = arm_thresholds(local_block)
    drift = compare_blocks(local_block, peer_block)
    code_defaults, defaults_degradations = fetch_code_defaults(
        command=default_defaults_command(args.uv_bin), cwd=args.config.parent)
    degradations += defaults_degradations
    restatements = {
        'local': compare_to_code_defaults(local_block, code_defaults),
        'peer': compare_to_code_defaults(peer_block, code_defaults),
    }

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
        arm = _arm_for(metric, specs)
        spec = ARM_METRIC_SELECTORS[arm] if arm else None
        in_force = configured.get(arm) if arm else None
        stats = coverage.get(metric)
        if stats is None:
            readable = (
                'Coverage: no readability metric for this arm, so the hold '
                'fractions below are over readable ticks of unknown count.'
            )
        else:
            readable = (
                f"Coverage: readable on {stats['readable']}/{stats['ticks']} "
                f"ticks ({stats['readable_fraction']:.1%})"
                + ('' if stats['readable_fraction'] >= D11_READABILITY_FLOOR
                   else ' — **BELOW THE FLOOR**, see degradations')
            )
        lines += [
            f'### `{metric}` — {spec.unit if spec else ""}',
            '',
            f'Configured value in force: '
            f'{in_force if in_force is not None else "(none — see degradations)"}',
            '',
            # A hold fraction's denominator is readable ticks, not ticks, so it
            # is printed beside the coverage it was computed over — never alone.
            readable,
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

    lines += ['', '## Leaves that merely restate the shipped code default', '',
              'PRD §6.2 writes three values that are already the code defaults, so '
              'one fact gains three homes with no reconciler (INV-9). This section '
              'names them; it never edits either file.', '']
    for side, verdict in restatements.items():
        if verdict is None:
            lines.append(f'- **{side}**: not compared (see degradations).')
            continue
        lines.append(
            f"- **{side}**: restates the default: "
            f"{', '.join(f'`{leaf}`' for leaf in verdict['restates_default']) or 'none'}"
            f"; not in the model's schema: "
            f"{', '.join(f'`{leaf}`' for leaf in verdict['unknown_to_schema']) or 'none'}"
        )

    lines += ['', '## Degradations', '']
    lines += [f'- {d}' for d in degradations] or ['_None._']

    report = '\n'.join(lines) + '\n'
    print(report)

    if not args.no_report:
        # ONE clock read, reused for the filename, the heading above and the
        # commit subject below — three reads could straddle midnight and
        # produce a report whose own name disagrees with its heading.
        out = args.report_dir / f'load-threshold-calibration-{stamp}.md'
        # A NAMED degradation, never a non-zero exit — commit_report's shape,
        # and for its reason: the analysis is the deliverable and filing it is
        # a convenience, while ε1/ε2 classify a non-zero rc as an INFRA FAULT
        # with no gate. The report text is already on stdout above, so nothing
        # the run produced is lost when only the filing fails.
        try:
            out.write_text(report, encoding='utf-8')
        except OSError as exc:
            degradations.append(f'report_unwritable: {out} ({exc})')
        else:
            # To STDERR, so stdout's last line stays the single-line JSON.
            print(f'report: {out}', file=sys.stderr)
            if args.commit:
                degradations += commit_report(out, stamp)

    # A degradation raised by the commit above lands in the JSON but not in the
    # already-written report text — the report cannot narrate its own commit.
    print(json.dumps({
        'generated_at': now.isoformat(timespec='seconds'),
        'db': str(args.db),
        'percentiles': percentiles,
        'holds': holds,
        'coverage': coverage,
        'configured': configured,
        'drift': drift,
        'restates_code_default': restatements,
        'degradations': [d.split(':', 1)[0] for d in degradations],
        'degradation_details': degradations,
    }))
    return 0


if __name__ == '__main__':
    sys.exit(main())
