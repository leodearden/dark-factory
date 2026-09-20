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

That contract is scoped by its own first clause, and the scope is the point: a
DEGRADATION is an unusual INPUT -- an absent config, a non-UTF-8 byte, a yaml
date scalar -- and belongs in the named vocabulary, which is why each of those
is caught where it arises. An unanticipated exception is not that. It means no
analysis was completed, so there is nothing to deliver and an INFRA FAULT page
is the honest signal; a blanket `except Exception` at __main__ returning 0
would buy the letter of the contract by reporting a broken script as a
delivered report with a degradation. Loud beats silent here, so there is no
top-level guard, deliberately.

Output contract, matching scripts/merge-pytest-n-ab-analysis.py: the human
report on stdout, the written report's path on STDERR so stdout's last line
stays the single-line JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, TypedDict

# The PEER checkout, and so the one path here that must NOT follow
# $DARK_FACTORY_ROOT: the whole point of the comparison is that it lives in a
# different project. Every other default is derived from the seam below.
DEFAULT_PEER_CONFIG = Path('/home/leo/src/reify/dark-factory-orchestrator.yaml')
PERCENTILES = (0.50, 0.90, 0.95, 0.99)

# {metric: [(ts, value) in ts order]} — what every reader here hands around.
Series = dict[str, list[tuple[int, float]]]

class ArmSpec(NamedTuple):
    """Everything the report needs about one gate arm.

    ``selector`` is the sampler metric recording this arm. It may name a
    metric outright or be a ':' prefix with one series per cgroup leaf; which
    one it is is read off the RECORDED metric name at every use site, never
    declared here, because a declared copy of a derivable fact is free to
    contradict the data. ``ladder`` is the candidate thresholds to evaluate
    hold fractions at. ``unit`` labels the numbers for the human reading the
    escalation.

    ``readability`` names the ``*_read_ok`` metric recording whether this arm
    was readable at all, or ``None`` when the collector emits none. It has no
    default, deliberately: a new arm cannot be added without deciding the
    question, and an arm whose readability is unknown must SAY so rather than
    be reported as fully covered.
    """

    selector: str
    ladder: tuple[float, ...]
    unit: str
    readability: str | None


# Percentage-pressure arms share one ladder: a PSI avg10 is a percentage of
# wall time stalled, so the same rungs mean the same thing for all of them.
_PRESSURE_LADDER = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
_PRESSURE_UNIT = '% of wall time stalled (PSI avg10)'

# The corpus's TICK CLOCK: the one metric ``collect_load_metrics`` writes on
# every tick it completes, readable or not — an unreadable /proc/stat is a 0.0
# row, never a missing one. Every ``own_read_ok:<leaf>`` comes out of that same
# call but only for the leaves discovered that tick, so a leaf's own row count
# is not the tick count; this metric's is. Public because the lockstep test in
# sampler/tests/test_load_metrics.py asserts the sampler really does emit it on
# every completed tick, which is the whole of what makes it a clock.
TICK_METRIC = 'runqueue_read_ok'

# Arm name -> its ArmSpec. The `selector` half is duplicated from
# sampler.metrics.ARM_METRIC_STEMS BY NECESSITY: that module cannot be
# imported here, because at gate time this script runs under the system
# python3 (see the stdlib-only note above). The reconciler is the named
# lockstep test in sampler/tests/test_load_metrics.py, which loads this file by
# path and asserts the two agree in both directions — so do NOT
# "de-duplicate" this with an import, which would crash the gate.
ARM_METRIC_SELECTORS = {
    'mem_full_avg10': ArmSpec(
        'psi_mem_full_avg10', _PRESSURE_LADDER, _PRESSURE_UNIT, None),
    'mem_some_avg10': ArmSpec(
        'psi_mem_some_avg10', _PRESSURE_LADDER, _PRESSURE_UNIT, None),
    'io_some_avg10': ArmSpec(
        'psi_io_some_avg10', _PRESSURE_LADDER, _PRESSURE_UNIT, None),
    'cpu_some_avg10': ArmSpec(
        'psi_cpu_some_avg10', _PRESSURE_LADDER, _PRESSURE_UNIT, None),
    # A RATIO, not a percentage: procs_running / len(sched_getaffinity(0)).
    # 1.0 is "as many runnable threads as CPUs"; 4.0 is PRD D9's provisional.
    'runqueue_ratio': ArmSpec(
        'runqueue_ratio',
        (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0),
        'runnable threads per CPU (ratio)', TICK_METRIC),
    # One series per cgroup leaf, so ':' — reported per leaf, never pooled.
    'own_cpu_some_avg10': ArmSpec(
        'own_cpu_some10', _PRESSURE_LADDER, _PRESSURE_UNIT, 'own_read_ok'),
}


def pct(xs: list[float], p: float) -> float:
    """NEAREST-RANK percentile: always a value the corpus actually contains.

    Deliberate, and not the same choice as ``statistics.quantiles`` or
    ``dashboard/src/dashboard/data/stats_utils.py``, both of which INTERPOLATE.
    What this feeds is a candidate THRESHOLD ladder, and an interpolated p99 is
    a number no tick ever produced -- "hold above 4.03" where the corpus only
    ever holds 4.0 and 4.1. Nearest-rank keeps every reported figure an observed
    reading, which is what makes the report's numbers checkable against the
    corpus by hand.

    ``+ 0.5`` and ``math.floor``, not ``round``: round() is banker's, so an
    exact .5 index alternated between the lower and upper median with n --
    ``pct([1, 2], .5)`` was 1 while ``pct([1, 2, 3, 4], .5)`` was 3. Same
    definition either way for every non-tie index (the four ladder rungs on a
    100-sample series are identical), but a rule that changes with the parity of
    n is one no reader can state.

    SHARED ORIGIN, and it cannot currently be shared as code: this is a copy of
    ``scripts/merge-pytest-n-ab-analysis.py::pct``, which has the un-fixed
    tie-break. Both files must run under the system python3 with stdlib only,
    and ``scripts/`` is not a package, so there is no module for them to import
    from and no reconciler test to write against one. Change this and the other
    one drifts silently -- read them together.
    """
    if not xs:
        return float('nan')
    ys = sorted(xs)
    k = max(0, min(len(ys) - 1, math.floor(p * (len(ys) - 1) + 0.5)))
    return ys[k]


def default_project_root() -> Path:
    """Which checkout owns the code, via the env seam the installer uses."""
    return Path(os.environ.get('DARK_FACTORY_ROOT', '/home/leo/src/dark-factory'))


def default_db() -> Path:
    """The corpus, via the env seam sampler/__main__.py and the installer use."""
    return default_project_root() / 'data/load-samples.db'


def default_config() -> Path:
    """THIS project's orchestrator config, via the same seam as the corpus.

    Honouring the seam here is not symmetry for its own sake. With the path
    hardcoded, ``DARK_FACTORY_ROOT=/other/checkout`` read the other checkout's
    corpus and code defaults and then compared them against the ORIGINAL
    checkout's yaml -- a report that silently mixes two checkouts and says so
    nowhere. ``--config`` still names a FILE independently of ``--project-root``
    (which names the checkout whose CODE supplies the shipped defaults); those
    stay two axes, they just now share one origin when neither is given.
    """
    return default_project_root() / 'dark-factory-orchestrator.yaml'


def default_report_dir() -> Path:
    """Where the report lands -- and, with --commit, which repo receives it."""
    return default_project_root() / 'plans'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--db', type=Path, default=None,
                    help='load-samples.db (default: $DARK_FACTORY_ROOT/data/load-samples.db)')
    ap.add_argument('--arm', choices=sorted(ARM_METRIC_SELECTORS), default=None,
                    help='restrict the analysis to one arm (default: all)')
    ap.add_argument('--config', type=Path, default=None,
                    help='this project\'s orchestrator config '
                         '(default: $DARK_FACTORY_ROOT/dark-factory-orchestrator.yaml)')
    ap.add_argument('--peer-config', type=Path, default=DEFAULT_PEER_CONFIG,
                    help='the peer project\'s orchestrator config (default: %(default)s)')
    ap.add_argument('--project-root', type=Path, default=None,
                    help='checkout whose orchestrator code supplies the shipped '
                         'defaults (default: $DARK_FACTORY_ROOT). Independent of '
                         '--config, which only says where the yaml lives.')
    ap.add_argument('--report-dir', type=Path, default=None,
                    help='where the markdown report is written '
                         '(default: $DARK_FACTORY_ROOT/plans)')
    ap.add_argument('--no-report', action='store_true',
                    help='print only; write no report file')
    ap.add_argument('--uv-bin', type=Path, default=DEFAULT_UV_BIN,
                    help='uv binary used to ask the live model for its code '
                         'defaults (default: %(default)s)')
    ap.add_argument('--commit', action='store_true',
                    help='git commit --only the written report (for the scheduled run)')
    args = ap.parse_args(argv)
    # Resolved HERE and not as argparse defaults, so the env seam is read at
    # call time rather than frozen at import. All four, not just the corpus: a
    # path that ignores the seam sends the run across two checkouts at once.
    if args.project_root is None:
        args.project_root = default_project_root()
    if args.db is None:
        args.db = default_db()
    if args.config is None:
        args.config = default_config()
    if args.report_dir is None:
        args.report_dir = default_report_dir()
    return args


# The two halves of a selector, as SEPARATE statements. Spelled
# `WHERE metric = ? OR metric GLOB ?` they plan as MULTI-INDEX OR plus
# USE TEMP B-TREE FOR ORDER BY, because one ORDER BY over the union has to be
# sorted globally -- while the caller only ever needs ts order WITHIN a metric.
# Apart, each is a bare index range scan with no sort at all. Measured on a
# 700k-row probe of one stem (7 leaves x 100k ticks): 1,930 ms together against
# 1,288 ms apart, byte-identical series. At the 30-day steady state the script's
# docstring cites, the own_cpu_some10 stem alone is ~3.6M rows, so the temp sort
# is the larger part of a read this gate makes on a 14-day clock.
#
# The sort is not the only cost avoided: a temp B-tree needs temp-file space,
# and SQLite raises `unable to open database file` when it cannot get any --
# turning a delivered analysis into a non-zero rc, which ε1/ε2 classify as an
# infra fault. Observed while benchmarking this very change.
#
# `ORDER BY metric, ts` on the stem half, not `ORDER BY ts`: the leading column
# is what lets the index supply the order, and the caller splits by metric
# anyway. The cursor is iterated rather than .fetchall()'d so the row tuples do
# not have to exist all at once alongside the Series they are copied into.
_EXACT_SQL = 'SELECT metric, ts, value FROM samples WHERE metric = ? ORDER BY ts'
_STEM_SQL = (
    'SELECT metric, ts, value FROM samples WHERE metric GLOB ? ORDER BY metric, ts'
)

# The corpus TICK COUNT, counted rather than fetched. ``coverage_table`` wants
# only how MANY ticks the corpus holds, and TICK_METRIC is written on every
# completed tick -- ~518k rows at the 30-day steady state this script's
# docstring cites. Reading that series to take its `len()` materialised ~518k
# (ts, value) tuples on every run, including the ε2 `--arm own_cpu_some_avg10`
# cut and the four PSI arms -- none of which DECLARES the clock as its
# readability metric, and none of which looks at the series at all.
#
# Measured in this worktree against the real schema: this statement plans as
# `SEARCH samples USING COVERING INDEX idx_samples_metric_ts (metric=?)` --
# index-only, no table access and no temp B-tree. The series fetch it replaces
# plans as a NON-covering `SEARCH samples USING INDEX idx_samples_metric_ts
# (metric=?)`, because it has to read `value` off the table for every row it
# then discards.
_COUNT_SQL = 'SELECT COUNT(*) FROM samples WHERE metric = ?'


def _fetch(con: sqlite3.Connection, selectors: list[str]) -> Series:
    """``{metric: [(ts, value) in ts order]}`` for every selector, one home.

    Shared by the value series and the readability series so the GLOB spelling
    and its reasoning live in exactly one place (heuristic 11).
    """
    series: Series = {}
    for selector in selectors:
        for sql, param in (
            (_EXACT_SQL, selector),
            (_STEM_SQL, f'{selector}:*'),
        ):
            for metric, ts, value in con.execute(sql, (param,)):
                series.setdefault(metric, []).append((int(ts), float(value)))
    return series


class CorpusRead(NamedTuple):
    """One read of the corpus: the values, the evidence about them, the clock.

    A named record rather than a four-slot tuple because two of those slots
    are the same type: a caller that transposed ``series`` and ``readability``
    would be silently wrong in both directions, and ``read.ticks_in_corpus``
    says at the call site what a fourth position does not. ``ArmSpec`` is this
    file's existing precedent for the shape, so this adds no new idiom, and a
    NamedTuple still unpacks positionally for a caller that prefers it.
    """

    series: Series
    readability: Series
    ticks_in_corpus: int
    degradations: list[str]


def read_series(db: Path, arm: str | None) -> CorpusRead:
    """Read one arm's values, the readability evidence about them, and the clock.

    The first two are returned apart and never merged: a ``*_read_ok`` row is
    evidence ABOUT a series, not a sample of it, so pooling them would corrupt
    the very percentiles and hold fractions it exists to qualify.

    ``ticks_in_corpus`` is the corpus tick count — the denominator of every
    coverage row (``coverage_table``), which an ``--arm own_cpu_some_avg10``
    run needs as much as a full one does. It is COUNTED, not fetched (see
    ``_COUNT_SQL``), so the readability dict carries exactly the metrics the
    selected arms DECLARE and nothing else: ``TICK_METRIC`` appears there for
    the runqueue arm, which declares it, and for no other.

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
    optimisation applies and the stem half plans as
    ``SEARCH samples USING INDEX idx_samples_metric_ts (metric>? AND
    metric<?)``. Measured on a 2.16M-row probe with this exact schema. At the
    30-day steady state the corpus is ~13M rows, so the LIKE spelling was a
    full scan per selector — the same shape of cost regression the dashboard's
    ``/api/load`` query carried before it was bounded. Why the two halves are
    issued as separate statements is on ``_STEM_SQL``.
    """
    specs = (
        [ARM_METRIC_SELECTORS[arm]] if arm else list(ARM_METRIC_SELECTORS.values())
    )
    selectors = [spec.selector for spec in specs]
    try:
        con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    except sqlite3.Error as exc:
        return CorpusRead({}, {}, 0, [f'db_unavailable: {db} ({exc})'])

    try:
        series = _fetch(con, selectors)
        readability = _fetch(
            con, sorted({spec.readability for spec in specs if spec.readability})
        )
        # Same connection and same guard as the fetches, so an unreadable
        # corpus still degrades by name instead of raising past the caller.
        (ticks_in_corpus,) = con.execute(_COUNT_SQL, (TICK_METRIC,)).fetchone()
    except sqlite3.Error as exc:
        return CorpusRead({}, {}, 0, [f'db_unavailable: {db} ({exc})'])
    finally:
        con.close()

    if not series:
        return CorpusRead({}, {}, 0, [
            f'no_samples_in_window: no rows for {sorted(selectors)} in {db}'
        ])
    return CorpusRead(series, readability, int(ticks_in_corpus), [])


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

# Below this fraction, a cause of a series' coverage shortfall — too few
# corpus ticks with a row, or too few readable ticks among those — is reported
# as a named degradation (see ``_below_floor``). It is a REPORTING threshold,
# not a decision one: the exact coverage is printed either way, so no verdict
# depends on where this sits — it decides only when the report shouts.
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


# A run survives a tick arriving late, but not a hole in the corpus. Three
# times the observed spacing is wide enough that ordinary jitter (the timer is
# OnUnitActiveSec, so a slow tick pushes the next one out) never splits a real
# run, and narrow enough that a restart or a crashed collector always does.
_HOLD_RUN_GAP_MULTIPLE = 3


def longest_hold_run(
    points: list[tuple[int, float]], threshold: float, *, spacing_seconds: int
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

    It takes (ts, value) POINTS, not bare values, because a gap in the corpus
    is not a continuation. Walking values alone and multiplying by the median
    spacing reported holds at ts 1000/1005/1010, a multi-hour outage, then
    holds at 100000/100005 as one run of "5 ticks (25s)" — for an interval
    actually spanning ~27 h. Welding across an outage is wrong in precisely the
    direction the clause cares about.

    Wall-clock is likewise READ from the timestamps rather than multiplied out
    of the tick count, so a run whose ticks arrived late reports the time it
    really spanned. The run's own final tick is added back: without it a
    single-tick hold would report 0 s for a hold that did happen, and for an
    evenly-spaced run the number stays exactly ticks x spacing.

    The winner is the longest in SECONDS, not in ticks, because seconds is what
    the clause asks about; on a regular cadence the two coincide.
    """
    max_gap = _HOLD_RUN_GAP_MULTIPLE * spacing_seconds
    best_ticks = best_seconds = 0
    ticks = 0
    start_ts = previous_ts = 0
    for ts, value in points:
        if value < threshold:
            ticks = 0
        elif ticks and ts - previous_ts <= max_gap:
            ticks += 1
        else:
            ticks, start_ts = 1, ts
        previous_ts = ts
        if ticks:
            seconds = ts - start_ts + spacing_seconds
            if seconds > best_seconds:
                best_ticks, best_seconds = ticks, seconds
    return {
        'ticks': best_ticks,
        'seconds': best_seconds,
        'human': _human_duration(best_seconds),
    }


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
        # encoding= explicitly: read_text() would use the LOCALE codec, so a
        # config with a non-UTF-8 byte raises UnicodeDecodeError under LANG=C
        # and not under LANG=*.UTF-8 -- a degradation whose existence depends on
        # the gate's environment. UnicodeDecodeError is a ValueError, not an
        # OSError, so it is named in the except too: uncaught it would leave
        # main() by a path that has no top-level guard, and a non-zero rc here
        # is an INFRA FAULT page rather than the delivered report.
        parsed = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError) as exc:
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


class ArmThresholds(NamedTuple):
    """What a block sets, what it tried and failed to set, and why.

    ``unusable`` is carried as structured data rather than recovered by
    re-reading ``degradations`` (INV-12, no ad-hoc parsers): the report needs
    per-arm membership to choose its wording, and the degradation strings exist
    for humans.
    """
    in_force: dict[str, float]
    unusable: dict[str, str]
    degradations: list[str]


def arm_thresholds(block: dict | None, side: str = 'local') -> ArmThresholds:
    """The arm thresholds a parsed block actually sets, plus the ones it botched.

    Reported beside each ladder so the report always evaluates the value in
    force, not only the hypotheticals. Non-arm leaves (``enabled``,
    ``min_inflight_floor``) are skipped silently -- they are not thresholds and
    their presence is not a mistake.

    A NON-NUMERIC value on an ARM leaf is a different thing and is NAMED, never
    skipped. The author asked for a threshold and did not get one, so silence
    here would print "no value in force" for a config that visibly sets one --
    the silent-fail-soft this module's own contract forbids (see the module
    docstring: every degradation is a named entry in the report AND a key in the
    trailing JSON). The trigger is not hypothetical: PyYAML implements YAML 1.1,
    whose float regex requires a SIGN in the exponent, so `7.0e1` parses as the
    STRING '7.0e1' while `7.0e+1` parses as 70.0 (pinned by
    test_a_numeric_looking_string_is_reported_as_drift_not_silently_coerced).
    A quoted `"4.0"` does the same. An operator can write what reads as a
    number and have the calibration quietly ignore it.

    ``bool`` is excluded from the numeric case explicitly because in Python it
    IS an int, and `enabled: true` would otherwise read as the threshold 1. It
    lands in ``unusable`` rather than being dropped, for the same reason: on an
    ARM leaf a bool is a mistake worth seeing.
    """
    if block is None:
        return ArmThresholds({}, {}, [])
    in_force: dict[str, float] = {}
    unusable: dict[str, str] = {}
    degradations: list[str] = []
    for arm, value in block.items():
        if arm not in ARM_METRIC_SELECTORS:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            in_force[arm] = float(value)
            continue
        shape = f'{value!r} ({type(value).__name__})'
        unusable[arm] = shape
        degradations.append(f'{side}_threshold_not_numeric: {arm} = {shape}')
    return ArmThresholds(in_force, unusable, degradations)


_DEFAULTS_DUMP = (
    'import json;'
    ' from orchestrator.config import PsiAdmissionConfig as C;'
    ' print(json.dumps({k: f.default for k, f in C.model_fields.items()},'
    ' default=str))'
)
DEFAULT_UV_BIN = Path('/home/leo/.local/bin/uv')
_DEFAULTS_TIMEOUT_SECONDS = 120

# Every git call is bounded, because an unbounded one defeats the named
# degradation exactly as thoroughly as a traceback would -- the gate would hang
# instead of delivering the analysis. Sized by the slowest real case: the commit
# runs in the machine-operated project_root, where CLAUDE.md budgets pre-commit
# at up to 300 s. TimeoutExpired is a SubprocessError, so the existing handlers
# already name it once a timeout exists at all.
_GIT_TIMEOUT_SECONDS = 360


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


def _in_force_label(arm: str | None, in_force: float | None,
                    unusable: dict[str, str]) -> str:
    """The three states an arm's configured value can be in, said apart.

    The single "(none -- see degradations)" this replaces was wrong in BOTH
    directions at once: it pointed at degradations for an unconfigured arm,
    where there are none to read and nothing is wrong, and it used the same
    words for a leaf that WAS set and could not be used -- the one case where
    there is something to go and read.
    """
    if in_force is not None:
        return str(in_force)
    if arm is not None and arm in unusable:
        return f'set but UNUSABLE ({unusable[arm]}) — see degradations'
    return '(not configured)'


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


def _value_metric_for(metric: str, specs: list[ArmSpec]) -> str | None:
    """The VALUE series a ``*_read_ok`` metric is evidence ABOUT, or None.

    The mirror image of the two lines ``coverage_table`` uses in the forward
    direction, written against the same ArmSpec fields and the same ':'
    partition — so the stem/non-stem rule is stated once per direction and a
    metric reached from either side computes an identical row.

    Needed because a series readable on NO tick writes no value row at all, so
    the readability side is the only side it appears on. ``None`` for anything
    no SELECTED spec claims, which is what keeps an ``--arm`` run reporting
    only the arm it was asked about.
    """
    stem, separator, tail = metric.partition(':')
    for spec in specs:
        if spec.readability == stem:
            return f'{spec.selector}:{tail}' if separator else spec.selector
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
        values = [v for _, v in points]
        spacing = observed_spacing([ts for ts, _ in points])
        out[metric] = [
            _rung(points, values, threshold, spacing)
            for threshold in ladder
        ]
    return out


def _rung(
    points: list[tuple[int, float]],
    values: list[float],
    threshold: float,
    spacing: int,
) -> dict:
    """One ladder rung. The fraction is computed ONCE and reused.

    It used to be computed twice per rung — once to report and once to judge
    against the target — which doubled 8 full passes per metric into 16 over a
    series that is ~518k points at the 30-day steady state.
    """
    fraction = hold_fraction(values, threshold)
    return {
        'threshold': threshold,
        'hold_fraction': round(fraction, 4),
        'within_d11_target': within_d11_target(fraction),
        'longest_hold_run': longest_hold_run(
            points, threshold, spacing_seconds=spacing
        ),
    }


class Coverage(TypedDict):
    """One metric's readable-tick coverage, as reported and as serialised.

    A TypedDict rather than a ``dict[str, float]``: the entry carries three
    counts, a fraction that is ``None`` when it cannot be known, and the NAME
    of the readability metric the counts came from, and every consumer reads
    those back at different types. At runtime it is a plain dict, which is
    what lands in the JSON payload.

    ``ticks_in_corpus`` is the fraction's denominator; ``ticks_with_a_row`` is
    not, and is reported beside it so "the leaf existed for 3 of 14 days" stays
    visible instead of being rounded into a single number.
    """

    ticks_in_corpus: int
    ticks_with_a_row: int
    readable: int
    readable_fraction: float | None
    readability_metric: str


def coverage_table(
    series: Series,
    readability: Series,
    specs: list[ArmSpec],
    *,
    ticks_in_corpus: int,
) -> dict[str, Coverage | None]:
    """Per VALUE metric, the readable-tick coverage its numbers rest on.

    A failed read emits no value row at all, so ``hold_fraction``'s denominator
    is the number of SUCCESSFUL reads rather than the number of ticks. Without
    this, "holds on 12% of samples" reads identically whether the corpus covered
    a fortnight or the 3% of it that was readable, and those are opposite
    verdicts for setting a dispatch threshold.

    The fraction is readable ticks over *ticks_in_corpus*, the corpus tick
    count ``read_series`` counted — not over the row count of the arm's own
    ``*_read_ok`` metric. The two agree only for a readability metric written
    on every tick, which ``runqueue_read_ok`` is and ``own_read_ok:<leaf>`` is
    not: that one is written only on ticks its leaf was discovered, so a leaf
    present for 3 days of a 14-day corpus has 3 days of rows, all readable, and
    dividing by its own row count would call that full coverage. Taken as a
    parameter rather than derived here, so the denominator is a fact about the
    CORPUS and not about which arms this run happened to select.

    ``None`` for an arm whose collector emits no readability metric — the four
    host-PSI arms. Reporting a fabricated 1.0 there would be the same class of
    defect as persisting α's fail-open 0.0 as a ratio.

    Keyed by the value metric so a ':' stem reports PER LEAF: one cgroup can be
    unreadable while its siblings are fine, which is exactly the case worth
    seeing. A series with READABILITY rows and no value rows gets a row too,
    reached backwards through ``_value_metric_for`` — because the fully
    unreadable leaf is precisely the one that writes no value row: the sampler
    records ``own_read_ok:<leaf>`` = 0.0 on every tick it discovered the leaf
    and nothing else. Iterating the value series alone therefore hid the exact
    case this keying exists to expose.
    """
    # The UNION of both sides, because each carries a case the other cannot.
    # Without the readability side, a series readable on no tick is invisible
    # (no value rows to iterate). Without the value side, the four PSI arms
    # vanish — they declare no readability metric, so nothing reaches them
    # backwards, and their coverage is a reported ``None`` rather than nothing.
    # Sorted, so the JSON payload's key order does not depend on which side a
    # metric arrived from.
    covered = {
        metric for metric in series if _arm_for(metric, specs) is not None
    } | {
        value_metric
        for evidence in readability
        if (value_metric := _value_metric_for(evidence, specs)) is not None
    }
    out: dict[str, Coverage | None] = {}
    for metric in sorted(covered):
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
        readable = sum(1 for _ts, value in points if value == 1.0)
        out[metric] = {
            'ticks_in_corpus': ticks_in_corpus,
            'ticks_with_a_row': len(points),
            'readable': readable,
            # None, not 0.0, when either count is zero. No clock is an unknown
            # denominator and no readability row is no evidence about the
            # series; 0.0 is the claim "we looked and it was never readable",
            # and the floor check below would then report absence of evidence
            # as a below-floor verdict about the corpus. Same class of defect
            # as the fabricated 1.0 refused above.
            'readable_fraction': (
                round(readable / ticks_in_corpus, 4)
                if ticks_in_corpus and points else None
            ),
            'readability_metric': key,
        }
    return out


def _below_floor(stats: Coverage) -> list[tuple[str, str]]:
    """Each cause of a KNOWN coverage's shortfall that is below the D11 floor.

    ``(degradation, detail)`` pairs, one per cause, each judged on its own
    ratio. Presence is rows over corpus ticks: a tick with no ``*_read_ok`` row
    is a tick the series' leaf was not discovered on — a unit restarted, added,
    removed or renamed. Readability is readable ticks over the ticks the series
    was present, and a ZERO numerator there is reported as its own cause rather
    than as the extreme of that one: a series read on no tick has no hold
    fractions to read against its coverage, which is the action
    ``low_readability`` asks for. A shortfall split between the two can leave
    both above the floor while ``readable_fraction`` dips below it; that
    fraction is still printed beside every hold ladder, but neither cause alone
    is a finding.

    An unknown coverage (either count zero) has no cause to name, so it yields
    nothing here rather than a division by zero — which would be a non-zero rc,
    an infra fault to ε1/ε2.
    """
    corpus, rows, readable = (
        stats['ticks_in_corpus'], stats['ticks_with_a_row'], stats['readable'])
    if not corpus or not rows:
        return []
    out = []
    if rows / corpus < D11_READABILITY_FLOOR:
        out.append((
            'partial_presence',
            f'present on only {rows}/{corpus} corpus ticks ({rows / corpus:.1%}), '
            f'below the {D11_READABILITY_FLOOR:.0%} floor — '
            f"`{stats['readability_metric']}` has no row on the rest (its leaf "
            'was not discovered), so its hold fractions describe that span, not '
            'the whole corpus',
        ))
    if readable == 0:
        out.append((
            'never_readable',
            f'was discovered on {rows}/{corpus} corpus ticks and readable on '
            'none of them — a tick with no readable value writes no value row, '
            'so this series has no candidate-threshold section above; its '
            'absence there is failed reads, not a leaf that was never '
            'discovered',
        ))
    elif readable / rows < D11_READABILITY_FLOOR:
        out.append((
            'low_readability',
            f'readable on {readable}/{rows} of the ticks it was present '
            f'({readable / rows:.1%}), below the {D11_READABILITY_FLOOR:.0%} '
            'floor — read its hold fractions against that coverage',
        ))
    return out


def _coverage_line(stats: Coverage | None) -> str:
    """The report's one-line coverage verdict printed beside a hold ladder."""
    if stats is None:
        return (
            'Coverage: no readability metric for this arm, so the hold '
            'fractions below are over readable ticks of unknown count.'
        )
    if stats['readable_fraction'] is None:
        return (
            f"Coverage: UNKNOWN — the corpus holds {stats['ticks_with_a_row']} "
            f"`{stats['readability_metric']}` rows and {stats['ticks_in_corpus']} "
            f'`{TICK_METRIC}` clock rows, and a coverage needs both, so the '
            'hold fractions below are over readable ticks of unknown count. '
            'See degradations.'
        )
    causes = [cause for cause, _detail in _below_floor(stats)]
    return (
        f"Coverage: readable on {stats['readable']}/{stats['ticks_in_corpus']} "
        f"corpus ticks ({stats['readable_fraction']:.1%}), present on "
        f"{stats['ticks_with_a_row']}/{stats['ticks_in_corpus']}"
        + (f" — **BELOW THE FLOOR**: {', '.join(causes)}, see degradations"
           if causes else '')
    )


def readability_degradations(
    coverage: dict[str, Coverage | None]
) -> list[str]:
    """Name each series whose coverage is below the floor, or NOT KNOWN.

    Four separate degradations, because they call for four different operator
    readings: ``low_readability`` says the collector ran on the series and often
    failed, so read the hold fractions against that coverage;
    ``never_readable`` says it failed EVERY time, so there are no hold
    fractions to read at all and the series has no candidate-threshold section
    above to read them in; ``partial_presence`` says the series existed for
    only part of the corpus,
    so its hold fractions describe that span, not the whole window; and
    ``unknown_readability`` says the corpus carries no evidence either way,
    which usually means the collector never ran at all. Folding any one into
    another sends an operator hunting a flaky read that never happened.
    """
    out = []
    for metric, stats in sorted(coverage.items()):
        if stats is None:
            continue
        if stats['readable_fraction'] is None:
            out.append(
                f"unknown_readability: {metric} coverage is unknown — not zero: "
                f"the corpus holds {stats['ticks_with_a_row']} "
                f"{stats['readability_metric']} rows and "
                f"{stats['ticks_in_corpus']} {TICK_METRIC} clock rows, and a "
                'coverage needs both. Its hold fractions below are over '
                'readable ticks of unknown count.'
            )
            continue
        out += [f'{cause}: {metric} {detail}' for cause, detail in _below_floor(stats)]
    return out


def _discover_repo(directory: Path) -> tuple[str | None, list[str]]:
    """The git repo enclosing *directory*, ASKED of git rather than computed.

    The repo used to be ``report_path.parent.parent``, which is right only for
    the default ``<root>/plans/<file>.md`` layout: ``--report-dir <root>``
    resolved to the directory ABOVE the repo and the commit degraded for no
    reason an operator could see. Where the report goes and which repo encloses
    it are not the same fact, and only git knows the second one.
    """
    try:
        proc = subprocess.run(
            ['git', '-C', str(directory), 'rev-parse', '--show-toplevel'],
            capture_output=True, text=True, check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, [f'report_commit_failed: {exc}']
    if proc.returncode != 0:
        return None, [
            f'report_commit_failed: {directory} is inside no git repository '
            f'({proc.stderr.strip()[:200]})'
        ]
    return proc.stdout.strip(), []


def _unstage(repo: str, path: Path) -> list[str]:
    """Restore the index after a failed commit; degrade if even that fails.

    `git add` is genuinely required above -- `git commit --only` rejects an
    untracked pathspec -- so a commit that fails after it leaves the report
    STAGED. CLAUDE.md names leftover staged state in `project_root` as a live
    hazard and not a tidiness question: that checkout is machine-operated, and
    a concurrent bare `git commit` from the merge worker or a hook would sweep
    this file into an unrelated commit. The realistic triggers are ordinary --
    a pre-commit hook rejection, or `.git/index.lock` still held past the
    grace window.

    Scoped to this one path, so it cannot disturb anything another process
    staged. A failure to unstage is itself named rather than swallowed: the
    caller is already returning a degradation, and "the commit failed AND the
    index is still dirty" is a different operator action from "the commit
    failed".
    """
    try:
        proc = subprocess.run(
            ['git', '-C', repo, 'reset', '-q', '--', str(path)],
            capture_output=True, text=True, check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f'report_unstage_failed: {path} ({exc})']
    if proc.returncode != 0:
        return [
            f'report_unstage_failed: {path} is still staged in {repo} '
            f'({proc.stderr.strip()[:200]})'
        ]
    return []


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
    repo, degradations = _discover_repo(path.parent)
    if repo is None:
        return degradations
    subject = (
        f'plans: load-threshold calibration report {stamp} '
        '(scripts/load-threshold-calibration.py)'
    )
    try:
        for argv in (
            ['git', '-C', repo, 'add', '--', str(path)],
            ['git', '-C', repo, 'commit', '--only', str(path), '-q', '-m', subject],
        ):
            proc = subprocess.run(
                argv, capture_output=True, text=True, check=False,
                timeout=_GIT_TIMEOUT_SECONDS,
            )
            if proc.returncode != 0:
                return [
                    f'report_commit_failed: {" ".join(argv[:4])} exited '
                    f'{proc.returncode} ({proc.stderr.strip()[:200]})'
                ] + _unstage(repo, path)
    except (OSError, subprocess.SubprocessError) as exc:
        return [f'report_commit_failed: {exc}'] + _unstage(repo, path)
    return []


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    now = datetime.now(UTC)
    stamp = now.strftime('%Y-%m-%d')

    corpus = read_series(args.db, args.arm)
    series = corpus.series
    degradations = list(corpus.degradations)
    specs = (
        [ARM_METRIC_SELECTORS[args.arm]] if args.arm
        else list(ARM_METRIC_SELECTORS.values())
    )
    percentiles = percentile_table({m: [v for _, v in pts] for m, pts in series.items()})
    holds = hold_table(series, specs)
    coverage = coverage_table(
        series, corpus.readability, specs, ticks_in_corpus=corpus.ticks_in_corpus)
    degradations += readability_degradations(coverage)
    local_block, local_degradations = load_psi_admission_block(args.config, 'local')
    peer_block, peer_degradations = load_psi_admission_block(args.peer_config, 'peer')
    degradations += local_degradations + peer_degradations
    configured, unusable_thresholds, threshold_degradations = arm_thresholds(
        local_block, 'local')
    degradations += threshold_degradations
    drift = compare_blocks(local_block, peer_block)
    code_defaults, defaults_degradations = fetch_code_defaults(
        command=default_defaults_command(args.uv_bin), cwd=args.project_root)
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
        lines += [
            f'### `{metric}` — {spec.unit if spec else ""}',
            '',
            f'Configured value in force: {_in_force_label(arm, in_force, unusable_thresholds)}',
            '',
            # A hold fraction's denominator is readable ticks, not ticks, so it
            # is printed beside the coverage it was computed over — never alone.
            _coverage_line(coverage.get(metric)),
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
    # default=str for the same reason fetch_code_defaults passes it: `drift`
    # carries RAW parsed yaml values straight through from compare_blocks, and
    # yaml resolves an unquoted `2026-09-17` to datetime.date, which json cannot
    # serialise. Without it a TypeError is raised AFTER the whole analysis has
    # run and printed -- the most expensive possible moment to lose the rc.
    print(json.dumps({
        'generated_at': now.isoformat(timespec='seconds'),
        'db': str(args.db),
        'percentiles': percentiles,
        'holds': holds,
        'coverage': coverage,
        'configured': configured,
        # The exact counterpart of 'configured', and load-bearing for a machine
        # consumer: without it the JSON cannot distinguish "this arm was never
        # set" from "it was set and rejected" -- the confusion this key pair
        # exists to remove. Naming it here beats re-parsing degradation strings.
        'unusable_thresholds': unusable_thresholds,
        'drift': drift,
        'restates_code_default': restatements,
        'degradations': [d.split(':', 1)[0] for d in degradations],
        'degradation_details': degradations,
    }, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
