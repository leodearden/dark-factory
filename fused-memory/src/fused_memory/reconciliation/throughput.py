"""Reconciliation drain/inflow throughput analysis (task 3049).

Answers three operator questions about a project's reconciliation pipeline:

* **inflow** — how many events actually arrive, per hour and per day, broken
  down by event type;
* **drain** — how much wall-clock the pipeline spends in each run mode
  (backlog chunk, steady state, remediation, targeted) and what that costs
  per event;
* **capacity** — the sustainable events/day that falls out of those two, and
  whether it clears the observed burst inflow.

Everything here is read-only over the reconciliation SQLite DB plus pure
arithmetic, so the report can be run out-of-process against a live production
database (see ``main`` / ``python -m``).

METHOD NOTE — the ISO8601 offset trap
-------------------------------------
``event_buffer.timestamp`` is an ISO8601 string that **carries an offset**::

    2026-07-25T13:23:22.383113+00:00

SQLite's ``datetime('now', ...)`` renders **space-separated and offset-free**::

    2026-07-25 13:23:22

Comparing the two in SQL (``WHERE timestamp < datetime('now', '-1 hour')``) is
a *string* comparison, and ``'T'`` (0x54) sorts above ``' '`` (0x20).  Every
same-day row therefore compares greater than the literal, so a whole day
collapses into a single bucket and the measurement silently reads as one giant
hour.  This is not hypothetical — it is the trap the task's METHOD NOTE calls
out.

The rule this module enforces: **bucket by parsing, never by SQL string
comparison against a ``datetime('now')`` literal.**  ``utc_hour_bucket`` is the
single place that parsing happens; every reader in this module and the rollup
in ``event_buffer.cleanup_drained`` route through it.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    'FITTED_CYCLE_FIXED_SECONDS',
    'FITTED_CYCLE_MARGINAL_SECONDS',
    'OBSERVED_BURST_EVENTS_PER_DAY',
    'STEADY_STATE_AMORTISATION_MIN_BATCH',
    'build_report',
    'capacity_verdict',
    'cycle_seconds',
    'main',
    'seconds_per_event',
    'sustainable_events_per_day',
    'DRAIN_MODES',
    'classify_run',
    'drain_stats',
    'inflow_daily',
    'remediation_duty_cycle',
    'inflow_hourly',
    'utc_hour_bucket',
]


def utc_hour_bucket(ts: str) -> str:
    """Normalise an ISO8601 timestamp to a UTC hour key ``'YYYY-MM-DDTHH'``.

    Parses (never string-compares — see the module METHOD NOTE), treats a naive
    timestamp as UTC, converts any offset-bearing timestamp *to* UTC rather
    than truncating at its local hour, and formats an hour key that sorts
    lexicographically in true chronological order.

    Args:
        ts: An ISO8601 timestamp, with or without an offset.

    Returns:
        The UTC hour key, e.g. ``'2026-07-25T13'``.

    Raises:
        ValueError: If ``ts`` is not a parseable ISO8601 timestamp.  Bucketing
            garbage into a plausible-looking wrong hour would silently corrupt
            the measurement, so this fails loudly instead.
    """
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime('%Y-%m-%dT%H')


def _connect_ro(db_path: str | Path) -> sqlite3.Connection:
    """Open the reconciliation DB read-only.

    The report is meant to run out-of-process against a live production
    database, so it must never be able to write.  ``mode=ro`` enforces that at
    the SQLite level rather than by convention.

    Raises:
        FileNotFoundError: If ``db_path`` does not exist.  SQLite's own error
            for a missing read-only file ('unable to open database file') is
            indistinguishable from a permissions problem, so the check is
            explicit — the CLI turns this into a clear message rather than a
            traceback.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f'reconciliation database not found: {path}')
    conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """True if ``table`` is present.

    A DB written by an older build has no ``event_arrival_hourly``, and one
    that has never run reconciliation has no ``runs``.  Readers degrade to
    'no rows from that source' rather than raising, so a partially-populated
    production DB still yields a report.
    """
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _normalise_since(since: str | None) -> str | None:
    """Normalise a ``since`` argument to an hour-bucket key.

    Accepts either an hour bucket ('2026-07-25T14') or a full ISO8601
    timestamp, with or without an offset.  Both route through
    ``utc_hour_bucket``, so an offset-bearing value is CONVERTED to UTC rather
    than lexically compared — the METHOD NOTE trap, again.
    """
    if since is None:
        return None
    return utc_hour_bucket(since)


def _hour_floor_instant(since: str | None) -> str | None:
    """``since`` floored to the start of its UTC hour, as an ISO8601 instant.

    The inflow half of the report reads hourly-bucketed data, so it can only
    ever honour a floor at hour granularity — a sub-hour ``since`` is silently
    rounded DOWN there.  The drain half compares an exact instant against
    ``runs.started_at`` and would honour it precisely.  Left alone, the two
    halves would cover different windows (``--since 14:30`` counting arrivals
    from 14:00 but only runs from 14:30), which quietly breaks the
    inflow-vs-capacity comparison the tool exists to make.

    So :func:`build_report` floors ONCE, here, and threads the floored instant
    into both halves.  Flooring rather than rejecting a sub-hour value keeps
    the CLI forgiving; the floored value is echoed in the payload as
    ``since_effective`` so the widening is visible rather than assumed.
    """
    bucket = _normalise_since(since)
    return None if bucket is None else f'{bucket}:00:00+00:00'


def _arrival_counts(
    db_path: str | Path,
    project_id: str,
    since: str | None = None,
) -> Counter[tuple[str, str]]:
    """Per-(hour_bucket, event_type) arrival counts for one project.

    The UNION of two sources that partition the event set exactly:

    * ``event_arrival_hourly`` — events already swept out of the buffer by
      ``cleanup_drained``, which rolled them up in the same transaction as the
      DELETE;
    * ``event_buffer`` — every row still live, of ANY status ('buffered' and
      'drained' alike), because a drained row that has not yet aged past
      ``max_age_seconds`` is still an arrival that has not been rolled up.

    Those two sets are disjoint and exhaustive by construction, so summing
    them double-counts nothing and loses nothing.
    """
    floor = _normalise_since(since)
    counts: Counter[tuple[str, str]] = Counter()

    conn = _connect_ro(db_path)
    try:
        if _table_exists(conn, 'event_arrival_hourly'):
            for row in conn.execute(
                """SELECT hour_bucket, event_type, event_count
                   FROM event_arrival_hourly WHERE project_id = ?""",
                (project_id,),
            ):
                if floor is not None and row['hour_bucket'] < floor:
                    continue
                counts[(row['hour_bucket'], row['event_type'])] += row['event_count']

        if _table_exists(conn, 'event_buffer'):
            for row in conn.execute(
                'SELECT timestamp, event_type FROM event_buffer WHERE project_id = ?',
                (project_id,),
            ):
                # Bucket in Python.  Never filter these by a SQL comparison
                # against a datetime('now') literal — see the METHOD NOTE.
                bucket = utc_hour_bucket(row['timestamp'])
                if floor is not None and bucket < floor:
                    continue
                counts[(bucket, row['event_type'])] += 1
    finally:
        conn.close()

    return counts


def inflow_hourly(
    db_path: str | Path,
    project_id: str,
    since: str | None = None,
) -> dict[str, int]:
    """Per-hour event arrival counts for one project, oldest hour first.

    Args:
        db_path: Path to reconciliation.db (opened read-only).
        project_id: Project to report on.
        since: Optional inclusive floor — an hour bucket or an ISO8601
            timestamp; normalised to a bucket key before comparison.

    Returns:
        ``{hour_bucket: event_count}``, insertion-ordered chronologically.
        Hour-bucket keys sort lexicographically in true chronological order,
        so the ordering is exact rather than approximate.
    """
    counts = _arrival_counts(db_path, project_id, since)
    hourly: Counter[str] = Counter()
    for (bucket, _event_type), n in counts.items():
        hourly[bucket] += n
    return {bucket: hourly[bucket] for bucket in sorted(hourly)}


def inflow_daily(
    db_path: str | Path,
    project_id: str,
    since: str | None = None,
) -> dict[str, Any]:
    """Per-day arrival totals plus the per-event_type composition.

    Args:
        db_path: Path to reconciliation.db (opened read-only).
        project_id: Project to report on.
        since: Optional inclusive floor (see :func:`inflow_hourly`).

    Returns:
        ``{'daily': {'YYYY-MM-DD': int}, 'composition': {event_type: int},
        'total_events': int}``.  The composition map is the evidence needed to
        size lever 3 (coalescing redundant task_status_changed / task_modified
        events for one task id) — it says how much of the inflow is actually
        those two types.
    """
    counts = _arrival_counts(db_path, project_id, since)

    daily: Counter[str] = Counter()
    composition: Counter[str] = Counter()
    for (bucket, event_type), n in counts.items():
        daily[bucket[:10]] += n  # 'YYYY-MM-DDTHH' -> 'YYYY-MM-DD'
        composition[event_type] += n

    return {
        'daily': {day: daily[day] for day in sorted(daily)},
        'composition': {
            etype: composition[etype]
            for etype in sorted(composition, key=lambda e: (-composition[e], e))
        },
        'total_events': sum(daily.values()),
    }


# ── drain statistics ───────────────────────────────────────────────────
#
# Unlike event_buffer, the `runs` table has NO pruning anywhere in the
# codebase, so drain history is fully derivable retrospectively — no new
# instrumentation was needed for this half of the measurement.

#: The four run modes a `runs` row is classified into.  ``targeted`` is
#: reported but deliberately excluded from the drain-capacity totals: targeted
#: runs never acquire the reconciliation lock (targeted.py:213), so they
#: execute concurrently with a drain and consume none of its capacity
#: (ADDENDUM 2 finding 5).
DRAIN_MODES: tuple[str, ...] = (
    'backlog_chunk',
    'steady_state',
    'remediation',
    'targeted',
)

#: Modes that drain buffered events — lock-held capacity MINUS remediation.
#: This is the numerator of the remediation-FREE rate, and it is what the
#: step-18 deferral gate leaves behind when it defers the inline remediation
#: tail during backlog mode.
_DRAIN_ONLY_MODES: tuple[str, ...] = ('backlog_chunk', 'steady_state')

#: Modes that actually consume lock-held drain capacity.  Defined FROM
#: :data:`_DRAIN_ONLY_MODES` so the two sets cannot drift apart: adding a new
#: event-draining mode to one automatically adds it to the other, and the only
#: intentional difference stays the single 'remediation' entry.
_CAPACITY_MODES: tuple[str, ...] = (*_DRAIN_ONLY_MODES, 'remediation')

#: BacklogIterator stamps each chunk with this trigger_reason prefix
#: (harness.py:4848, 'backlog_chunk:{n}:{events}').  The final consolidation
#: pass (harness.py:4873, 'backlog_final_consolidation') deliberately does NOT
#: match — it runs after the chunks with a drained buffer and is a normal full
#: cycle, so it belongs in steady_state.
_BACKLOG_CHUNK_PREFIX = 'backlog_chunk:'


def classify_run(run_type: str, trigger_reason: str) -> str:
    """Classify one `runs` row into a :data:`DRAIN_MODES` bucket.

    Args:
        run_type: The row's ``run_type`` ('full' / 'targeted' / 'remediation').
        trigger_reason: The row's ``trigger_reason``.

    Returns:
        One of :data:`DRAIN_MODES`.
    """
    if run_type == 'remediation':
        return 'remediation'
    if run_type == 'targeted':
        return 'targeted'
    if trigger_reason.startswith(_BACKLOG_CHUNK_PREFIX):
        return 'backlog_chunk'
    return 'steady_state'


def _seconds_between(started_at: str, completed_at: str) -> float:
    """Wall-clock seconds between two ISO8601 timestamps, offset-aware."""
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(completed_at)
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return (end - start).total_seconds()


def _per_event(wall_clock_seconds: float, events: int) -> float | None:
    """Seconds per event, or None when the mode drained no events.

    Remediation runs legitimately process zero events (they act on Stage-3
    findings, not buffered events), and a mode with no runs at all is reported
    present-but-empty.  Both must read as 'not applicable', not raise.
    """
    if events <= 0:
        return None
    return wall_clock_seconds / events


def drain_stats(
    db_path: str | Path,
    project_id: str,
    since: str | None = None,
) -> dict[str, Any]:
    """Per-mode drain statistics over the `runs` table for one project.

    Args:
        db_path: Path to reconciliation.db (opened read-only).
        project_id: Project to report on.
        since: Optional inclusive floor on ``started_at`` — an ISO8601
            timestamp; rows that started earlier are ignored.

    Returns:
        ``{'modes': {mode: {events, wall_clock_seconds, run_count,
        seconds_per_event}}, 'runs_in_flight': int,
        'drain_wall_clock_seconds': float, 'drain_events': int,
        'drain_seconds_per_event': float | None,
        'drain_only_wall_clock_seconds': float,
        'drain_only_seconds_per_event': float | None}``.

        Every mode in :data:`DRAIN_MODES` is always present, empty if unseen,
        so a consumer never has to guard on a missing key.

        A run with ``completed_at IS NULL`` is still in flight: it has no
        measurable wall-clock, so it is excluded from every aggregate (which
        would otherwise be diluted by a partial run) and surfaced in
        ``runs_in_flight`` instead — loud rather than silently dropped.

        The top-level ``drain_*`` totals cover :data:`_CAPACITY_MODES` only.
        Targeted runs hold no lock and consume no drain capacity, so folding
        their wall-clock in would understate real throughput.

    TWO RATES, AND WHICH ONE TO USE — read this before doing arithmetic:

    * ``drain_seconds_per_event`` is **remediation-INCLUSIVE**.  Remediation
      wall-clock is in the numerator (it is lock-held capacity) while
      remediation's zero events add nothing to the denominator, so the
      remediation penalty is **already applied exactly once** in this rate.  A
      caller must therefore NOT multiply it by ``(1 - duty_cycle)`` again —
      that discounts twice and understates capacity by a factor of
      ``1/(1 - duty)``.  Pair it with
      ``sustainable_events_per_day(rate, remediation_duty_cycle=0.0)``.
    * ``drain_only_seconds_per_event`` is the **remediation-FREE** rate:
      :data:`_DRAIN_ONLY_MODES` wall-clock over the same ``drain_events``.
      This is the one to pair with a *modelled* duty cycle, and it is the
      post-lever-1 capacity — lever 1 (the step-18 backlog deferral gate)
      removes remediation from the drain path, which is exactly what
      excluding it from the numerator expresses.

    The two are related by ``drain_only = inclusive * (1 - duty)`` exactly.
    That identity is the algebraic statement that the inclusive rate already
    carries the penalty; it is not an invitation to re-derive one from the
    other in a caller.
    """
    floor = None
    if since is not None:
        floor = datetime.fromisoformat(since)
        if floor.tzinfo is None:
            floor = floor.replace(tzinfo=UTC)

    modes: dict[str, dict[str, Any]] = {
        mode: {'events': 0, 'wall_clock_seconds': 0.0, 'run_count': 0}
        for mode in DRAIN_MODES
    }
    runs_in_flight = 0

    conn = _connect_ro(db_path)
    try:
        if _table_exists(conn, 'runs'):
            rows = conn.execute(
                """SELECT run_type, trigger_reason, started_at, completed_at,
                          events_processed
                   FROM runs WHERE project_id = ?""",
                (project_id,),
            ).fetchall()
        else:
            rows = []

        for row in rows:
            if floor is not None:
                started = datetime.fromisoformat(row['started_at'])
                if started.tzinfo is None:
                    started = started.replace(tzinfo=UTC)
                if started < floor:
                    continue

            if row['completed_at'] is None:
                runs_in_flight += 1
                continue

            mode = modes[classify_run(row['run_type'] or '', row['trigger_reason'] or '')]
            mode['run_count'] += 1
            mode['events'] += row['events_processed'] or 0
            mode['wall_clock_seconds'] += _seconds_between(
                row['started_at'], row['completed_at'],
            )
    finally:
        conn.close()

    for mode in modes.values():
        mode['seconds_per_event'] = _per_event(
            mode['wall_clock_seconds'], mode['events'],
        )

    drain_wall_clock = sum(modes[m]['wall_clock_seconds'] for m in _CAPACITY_MODES)
    drain_events = sum(modes[m]['events'] for m in _CAPACITY_MODES)
    # Same event denominator, remediation wall-clock dropped from the
    # numerator — remediation drains zero events, so the denominators are
    # identical by construction and the two rates differ only by (1 - duty).
    drain_only_wall_clock = sum(
        modes[m]['wall_clock_seconds'] for m in _DRAIN_ONLY_MODES
    )

    return {
        'modes': modes,
        'runs_in_flight': runs_in_flight,
        'drain_wall_clock_seconds': drain_wall_clock,
        'drain_events': drain_events,
        'drain_seconds_per_event': _per_event(drain_wall_clock, drain_events),
        'drain_only_wall_clock_seconds': drain_only_wall_clock,
        'drain_only_seconds_per_event': _per_event(
            drain_only_wall_clock, drain_events),
    }


def remediation_duty_cycle(drain_stats_result: dict[str, Any]) -> float:
    """Fraction of lock-held drain wall-clock consumed by remediation passes.

    Remediation is an unconditional inline tail of every completed
    ``run_full_cycle`` (``_maybe_remediate``, harness.py:2963), and
    ``BacklogIterator`` runs each backlog chunk as its own full cycle — so
    every chunk drags in its own remediation pass.  Those passes process zero
    buffered events, so every second they hold is a second the backlog is not
    draining.  That is the loss lever 1 recovers.

    Args:
        drain_stats_result: The return value of :func:`drain_stats`.

    Returns:
        ``remediation / (remediation + backlog_chunk + steady_state)`` over
        wall-clock seconds, in ``[0.0, 1.0]``.  Targeted runs are excluded:
        they never acquire the reconciliation lock, so counting them would
        inflate the denominator and make remediation preemption look cheaper
        than it is.  Returns ``0.0`` when there is no drain wall-clock at all,
        rather than raising on an empty denominator.
    """
    modes = drain_stats_result['modes']
    denominator = sum(modes[m]['wall_clock_seconds'] for m in _CAPACITY_MODES)
    if denominator <= 0:
        return 0.0
    return modes['remediation']['wall_clock_seconds'] / denominator


# ── capacity arithmetic ────────────────────────────────────────────────
#
# MEASUREMENT PROVENANCE for every constant below — 2026-07-25, project reify
# (the ADDENDUM 2 sequence, reproduced verbatim as a test fixture):
#
#   steady state    50 events / 945s  = 18.90 s/event
#   backlog chunk  393 events / 954s  =  2.43 s/event
#   backlog chunk  326 events / ~960s =  2.94 s/event
#
# These are THREE points from TWO different model tiers (steady vs opus), so
# any fit across them mixes regimes.  The constants here are deliberately
# CONSERVATIVE rather than the optimistic fit the raw data supports, and they
# are a snapshot of one day on one project.  They must be recomputed from live
# data via build_report / the module CLI rather than trusted indefinitely —
# the config invariant test in tests/test_config_schema.py is what forces the
# shipped config and this claim to move together, but neither can detect that
# the underlying system got slower.

#: Observed burst inflow for reify, events/day.  The figure the sustainable
#: capacity claim is checked against by :func:`capacity_verdict`.
#: Provenance: task 3049's ~3.5k/day burst observation.  Recompute from
#: :func:`inflow_daily` once the event_arrival_hourly rollup has history.
OBSERVED_BURST_EVENTS_PER_DAY: int = 3500

#: Fixed per-cycle cost in seconds — the part of a reconciliation cycle that
#: does NOT scale with batch size (stage orchestration, context assembly, LLM
#: round trips).
#: Provenance: the 2026-07-25 reify points above.  Fitting T(B) = F + c*B
#: across the 50-event steady-state point and the 393-event chunk gives
#: c = 9/343 ~ 0.03 s/event, i.e. essentially ALL fixed cost — but those two
#: points ran on different tiers, so that fit would over-claim.  This value
#: pairs with the deliberately pessimistic marginal cost below and reproduces
#: the measured steady-state point to within 8.5%.
FITTED_CYCLE_FIXED_SECONDS: float = 900.0

#: Marginal per-event cost in seconds.
#: Provenance: chunk 2's observed 2.94 s/event taken as an UPPER BOUND on
#: marginal cost, rounded down to 2.5.  Deliberately conservative: the raw fit
#: supports ~0.03 s/event, so the real amortisation is very likely better than
#: this model predicts.  Being wrong in this direction under-promises capacity
#: rather than over-promising it.
FITTED_CYCLE_MARGINAL_SECONDS: float = 2.5

#: Minimum steady-state batch size at which the fixed per-cycle cost is
#: adequately amortised.
#: Derivation: at B=150 the model gives (900 + 2.5*150)/150 = 8.5 s/event
#: against the 18.9 s/event measured at B=50 — a 2.2x amortisation, which
#: clears OBSERVED_BURST_EVENTS_PER_DAY with headroom.  At the shipped
#: buffer_size_threshold of 250 this requires conditional_trigger_ratio >= 0.6.
#: tests/test_config_schema.py asserts the shipped config satisfies that, so
#: the constant and the config cannot drift apart silently.
STEADY_STATE_AMORTISATION_MIN_BATCH: int = 150


def cycle_seconds(
    batch_size: int,
    fixed_seconds: float = FITTED_CYCLE_FIXED_SECONDS,
    marginal_seconds: float = FITTED_CYCLE_MARGINAL_SECONDS,
) -> float:
    """Total wall-clock for one reconciliation cycle: ``T(B) = F + c*B``.

    Args:
        batch_size: Events processed in the cycle.
        fixed_seconds: The per-cycle cost that does not scale with the batch.
        marginal_seconds: The per-event cost.

    Returns:
        Predicted cycle wall-clock in seconds.  Note that ``batch_size = 0``
        still costs the full fixed component — that is the finding this whole
        model exists to express.
    """
    return fixed_seconds + marginal_seconds * batch_size


def seconds_per_event(
    batch_size: int,
    fixed_seconds: float = FITTED_CYCLE_FIXED_SECONDS,
    marginal_seconds: float = FITTED_CYCLE_MARGINAL_SECONDS,
) -> float:
    """Amortised per-event cost at ``batch_size``: ``T(B) / B``.

    Raises:
        ValueError: If ``batch_size <= 0``.  Per-event cost is undefined for an
            empty batch, so this raises rather than dividing by zero or
            returning a misleading sentinel.
    """
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}')
    return cycle_seconds(batch_size, fixed_seconds, marginal_seconds) / batch_size


def sustainable_events_per_day(
    seconds_per_event: float,
    remediation_duty_cycle: float = 0.0,
    utilisation: float = 1.0,
) -> float:
    """Events/day the pipeline can drain at a given per-event cost.

    ``86400 * utilisation * (1 - remediation_duty_cycle) / seconds_per_event``.

    APPLY THE REMEDIATION PENALTY EXACTLY ONCE.  ``remediation_duty_cycle``
    exists to model remediation overhead **on top of a remediation-free rate**.
    Pass ``0.0`` whenever the supplied rate ALREADY includes remediation
    wall-clock — which ``drain_stats``' ``drain_seconds_per_event`` does, and
    which its ``drain_only_seconds_per_event`` does not.  Passing a measured
    duty cycle alongside a measured inclusive rate double-discounts and
    understates capacity by a factor of ``1/(1 - duty)``; see the two-rates
    contract on :func:`drain_stats`.

    Args:
        seconds_per_event: Amortised drain cost per event.  See above for
            whether it should be paired with a non-zero duty cycle.
        remediation_duty_cycle: Fraction of lock-held drain wall-clock spent in
            zero-event remediation passes (see :func:`remediation_duty_cycle`),
            to be applied to a rate that does NOT already include it.  Lever 1
            drives this toward 0, which is worth ``1/(1-duty)`` on the same
            per-event rate — a pure scheduling win, no work saved.
        utilisation: Fraction of the day the project is actually holding the
            reconciliation lock and draining.  1.0 is the theoretical ceiling,
            not an observed value.

    Returns:
        Sustainable events/day.

    Raises:
        ValueError: On a non-positive rate, a duty cycle outside ``[0, 1)``, or
            a utilisation outside ``(0, 1]`` — all caller bugs that would
            otherwise produce a silently nonsensical capacity claim.
    """
    if seconds_per_event <= 0:
        raise ValueError(
            f'seconds_per_event must be positive, got {seconds_per_event}',
        )
    if not 0.0 <= remediation_duty_cycle < 1.0:
        raise ValueError(
            f'remediation_duty_cycle must be in [0, 1), got {remediation_duty_cycle}',
        )
    if not 0.0 < utilisation <= 1.0:
        raise ValueError(f'utilisation must be in (0, 1], got {utilisation}')
    return 86400.0 * utilisation * (1.0 - remediation_duty_cycle) / seconds_per_event


def capacity_verdict(
    sustainable_per_day: float,
    observed_burst_per_day: float = OBSERVED_BURST_EVENTS_PER_DAY,
) -> dict[str, Any]:
    """Does the sustainable drain rate clear the observed burst inflow?

    Args:
        sustainable_per_day: Output of :func:`sustainable_events_per_day`.
        observed_burst_per_day: Measured burst inflow, events/day.

    Returns:
        ``{'verdict': 'sufficient' | 'insufficient', 'headroom_ratio': float,
        'sustainable_per_day': float, 'observed_burst_per_day': float}``.
        Break-even counts as sufficient (``headroom_ratio >= 1.0``).

    Raises:
        ValueError: If ``observed_burst_per_day <= 0`` — there is no verdict to
            render against a zero burst, and returning infinity would read as
            'lots of headroom'.
    """
    if observed_burst_per_day <= 0:
        raise ValueError(
            f'observed_burst_per_day must be positive, got {observed_burst_per_day}',
        )
    headroom = sustainable_per_day / observed_burst_per_day
    return {
        'verdict': 'sufficient' if headroom >= 1.0 else 'insufficient',
        'headroom_ratio': headroom,
        'sustainable_per_day': sustainable_per_day,
        'observed_burst_per_day': observed_burst_per_day,
    }


# ── report ─────────────────────────────────────────────────────────────


def _retention_note() -> str:
    """The report's own measurement caveat.

    ``cleanup_drained`` is the only ``DELETE FROM event_buffer`` in the
    codebase, so before task 3049 added the ``event_arrival_hourly`` rollup a
    project's arrival history survived only as long as that cleanup window —
    NOT the week the task's ask (a) assumed.  (``event_journal`` is no help
    either: it is a write-ahead log whose ``mark_processed`` deletes the row.)
    Inflow from before the rollup landed is therefore gone for good, and a
    reader of this report has to know that before drawing a trend from it.

    Deliberately states NO window figure (amendment).  An earlier shape read
    the live default off ``EventBuffer.cleanup_drained``'s signature, which
    bought one interpolated number at the price of a report-build-time coupling
    to a parameter NAME in another module, a lazy import to dodge a circular
    one, and a degradation path so an unrelated rename could not take the whole
    report down.  The caveat is qualitative anyway: what a reader must act on
    is that pre-rollup inflow is UNRECOVERABLE, not how many seconds of it the
    sweep kept.  Callers that need the number have the config in hand.
    """
    return (
        'Inflow retention: EventBuffer.cleanup_drained deletes drained rows '
        'past a configured age, and is the only DELETE FROM event_buffer in '
        'the codebase. Task 3049 added the event_arrival_hourly rollup INSIDE '
        'that delete transaction, so from then on arrival history is durable '
        'and exact. Inflow from before the rollup landed was destroyed by that '
        'delete and is unrecoverable — treat any window extending back past '
        'the rollup as truncated, not as a period of low inflow.'
    )


def build_report(
    db_path: str | Path,
    project_id: str,
    since: str | None = None,
) -> dict[str, Any]:
    """Full inflow + drain + capacity report for one project.

    Args:
        db_path: Path to reconciliation.db (opened read-only).
        project_id: Project to report on.
        since: Optional inclusive floor — an ISO8601 timestamp or hour bucket.
            FLOORED to the start of its UTC hour and threaded into BOTH halves,
            so inflow and drain cover the same window and can be compared
            directly.  Inflow is hourly-bucketed data and cannot honour a
            sub-hour floor; applying the same floor to drain is what keeps the
            two halves aligned.  The floored value is echoed as
            ``since_effective``; ``since`` echoes what was asked for.

    Returns:
        A JSON-serialisable dict with ``project_id``, ``db_path``, ``since``,
        ``since_effective``,
        ``inflow`` (hourly / daily / composition / total_events), ``drain``
        (the :func:`drain_stats` payload), ``remediation_duty_cycle``,
        ``sustainable_events_per_day``, ``capacity_verdict``,
        ``remediation_free_events_per_day``,
        ``remediation_free_capacity_verdict`` and ``retention_note``.

        TWO capacity figures, each with its own verdict:

        * ``sustainable_events_per_day`` — the OBSERVED status quo, computed
          from the remediation-inclusive rate.  This stays the headline
          deliberately: it is what the measured window actually demonstrated,
          not a projection.
        * ``remediation_free_events_per_day`` — the post-lever-1 capacity,
          computed from the remediation-free rate, i.e. what the step-18
          backlog deferral gate leaves once remediation is off the drain path.

        ``remediation_duty_cycle`` stays in the payload as the explanation of
        the gap: the two figures differ by exactly ``1/(1 - duty)``.

        Every capacity key is ``None`` when the observed data cannot support a
        claim — no drained events (so no measured per-event rate), or a duty
        cycle of exactly 1.0 (only remediation runs in the window).  The
        arithmetic deliberately raises on those inputs rather than inventing a
        number; the REPORT's job is to say 'unknown' instead of propagating
        that to an operator who merely pointed the tool at a fresh database.
    """
    # ONE floor, applied to both halves — see _hour_floor_instant.
    effective_since = _hour_floor_instant(since)
    drain = drain_stats(db_path, project_id, effective_since)
    duty = remediation_duty_cycle(drain)
    daily = inflow_daily(db_path, project_id, effective_since)

    # BOTH duty arguments below are 0.0, and that is NOT a bug to 'fix'.
    # Per the drain_stats two-rates contract, each rate already carries its own
    # remediation treatment: drain_seconds_per_event has remediation wall-clock
    # in its numerator (penalty applied once, in the measurement), and
    # drain_only_seconds_per_event has it excluded entirely (no penalty to
    # apply — that is the point).  Passing `duty` here would discount a second
    # time and understate the observed capacity by 1/(1 - duty) ~ 1.78x on the
    # measured reify window.  The duty cycle's role in this payload is to
    # EXPLAIN the gap between the two figures, not to scale either of them.
    rate = drain['drain_seconds_per_event']
    sustainable: float | None = None
    verdict: dict[str, Any] | None = None
    if rate is not None and rate > 0 and duty < 1.0:
        sustainable = sustainable_events_per_day(rate, 0.0)
        verdict = capacity_verdict(sustainable, OBSERVED_BURST_EVENTS_PER_DAY)

    drain_only_rate = drain['drain_only_seconds_per_event']
    remediation_free: float | None = None
    remediation_free_verdict: dict[str, Any] | None = None
    if drain_only_rate is not None and drain_only_rate > 0:
        remediation_free = sustainable_events_per_day(drain_only_rate, 0.0)
        remediation_free_verdict = capacity_verdict(
            remediation_free, OBSERVED_BURST_EVENTS_PER_DAY)

    return {
        'project_id': project_id,
        'db_path': str(db_path),
        'since': since,
        'since_effective': effective_since,
        'inflow': {
            'hourly': inflow_hourly(db_path, project_id, effective_since),
            'daily': daily['daily'],
            'composition': daily['composition'],
            'total_events': daily['total_events'],
        },
        'drain': drain,
        'remediation_duty_cycle': duty,
        'sustainable_events_per_day': sustainable,
        'capacity_verdict': verdict,
        'remediation_free_events_per_day': remediation_free,
        'remediation_free_capacity_verdict': remediation_free_verdict,
        'retention_note': _retention_note(),
    }


def _format_report(report: dict[str, Any]) -> str:
    """Render a report as a short operator-readable summary.

    Deliberately terse: the per-mode drain breakdown is the line an operator
    reads to tell a THROUGHPUT overrun (healthy pipeline, drain simply slower
    than inflow) apart from a judge halt (task 2920), which is a different
    failure with a different fix.
    """
    lines = [f'project: {report["project_id"]}  db: {report["db_path"]}']
    if report['since']:
        lines.append(f'since:   {report["since"]}')
        if report['since_effective'] != report['since']:
            lines.append(f'         (floored to {report["since_effective"]} for '
                         f'BOTH inflow and drain — inflow is hourly)')

    inflow = report['inflow']
    lines.append(f'inflow:  {inflow["total_events"]} events over '
                 f'{len(inflow["hourly"])} hour(s)')
    for etype, n in inflow['composition'].items():
        lines.append(f'           {etype}: {n}')

    lines.append('drain:')
    for mode in DRAIN_MODES:
        stats = report['drain']['modes'][mode]
        per_event = stats['seconds_per_event']
        rate = 'n/a' if per_event is None else f'{per_event:.2f}s/event'
        lines.append(
            f'           {mode:<15} runs={stats["run_count"]:<4} '
            f'events={stats["events"]:<6} '
            f'wall={stats["wall_clock_seconds"]:.0f}s  {rate}'
        )
    if report['drain']['runs_in_flight']:
        lines.append(f'           (in flight, excluded: '
                     f'{report["drain"]["runs_in_flight"]})')

    lines.append(f'remediation duty cycle: {report["remediation_duty_cycle"]:.3f} '
                 f'of lock-held drain wall-clock')

    if report['sustainable_events_per_day'] is None:
        lines.append('capacity: unknown (no drained events in this window)')
    else:
        verdict = report['capacity_verdict']
        lines.append(
            f'capacity: {report["sustainable_events_per_day"]:.0f} events/day '
            f'sustainable vs {verdict["observed_burst_per_day"]:.0f}/day burst '
            f'=> {verdict["verdict"]} '
            f'(headroom {verdict["headroom_ratio"]:.2f}x)'
        )

    # The post-lever-1 figure gets its own line rather than replacing the
    # observed one: an operator needs to see what the pipeline demonstrably
    # sustains AND where removing the remediation preemption takes it.
    if report['remediation_free_events_per_day'] is None:
        lines.append('          post-lever-1 (remediation-free): unknown')
    else:
        rf_verdict = report['remediation_free_capacity_verdict']
        lines.append(
            f'          post-lever-1 (remediation-free): '
            f'{report["remediation_free_events_per_day"]:.0f} events/day '
            f'=> {rf_verdict["verdict"]} '
            f'(headroom {rf_verdict["headroom_ratio"]:.2f}x)'
        )

    lines.append('')
    lines.append(report['retention_note'])
    return '\n'.join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Report reconciliation inflow, per-mode drain throughput and the '
            'resulting sustainable events/day for one project (task 3049).'
        ),
    )
    parser.add_argument(
        '--db', required=True,
        help='Path to reconciliation.db, e.g. data/reconciliation/reconciliation.db',
    )
    parser.add_argument(
        '--project', required=True,
        help='Project id to report on, e.g. reify',
    )
    parser.add_argument(
        '--since', default=None,
        help='Inclusive floor: an ISO8601 timestamp or an hour bucket '
             "('2026-07-25T14'). Applies to inflow and drain alike.",
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Emit the raw JSON report instead of the readable summary.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Run as::

        python -m fused_memory.reconciliation.throughput \\
            --db data/reconciliation/reconciliation.db --project reify --json

    Returns:
        0 on success; 1 when the database is missing, a ``since`` value cannot
        be parsed, or the database holds a malformed timestamp.  Each surfaces
        as a one-line stderr message and a non-zero exit rather than a
        traceback — and each names the thing that was actually wrong.

    The ``--since`` parse is validated EAGERLY, on its own, before the report
    runs.  ``build_report`` raises ValueError of its own accord on a corrupt
    ``event_buffer.timestamp`` or ``runs.started_at`` row, so a blanket
    ``except ValueError`` would report a data-corruption diagnosis as a
    ``--since`` parse failure — pointing an operator at an argument they may
    never have supplied and hiding the row that is actually broken.
    """
    args = _build_parser().parse_args(argv)
    try:
        _hour_floor_instant(args.since)
    except ValueError as e:
        print(f'error: could not parse --since {args.since!r}: {e}', file=sys.stderr)
        return 1

    try:
        report = build_report(args.db, args.project, args.since)
    except FileNotFoundError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    except ValueError as e:
        print(f'error: malformed timestamp in {args.db}: {e}', file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2) if args.json else _format_report(report))
    return 0


if __name__ == '__main__':
    sys.exit(main())
