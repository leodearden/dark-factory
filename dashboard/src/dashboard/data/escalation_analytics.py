"""Escalation lifecycle analytics — archive aggregator for the dashboard.

Backend data layer for plans/escalation-lifecycle-dashboard-prd.md Seam 2
(task gamma / 2658). Produces the payload for
``GET /api/v2/dashboard/escalation-analytics``: per-project origin/lifespan/
workflow aggregates over the escalation archive, plus regime markers and a
``parse_failures`` count (INV-4 — every skipped record is loud AND counted).

This module is a PURE-SYNC core: :func:`build_escalation_analytics` does a
per-request archive walk with no ``asyncio``. The route (``dashboard.app``)
wraps the call in ``asyncio.to_thread`` behind a short TTL cache so a cold
~10k-record walk never blocks the event loop. Tests exercise this module's
functions directly.

Clock discipline: the only permitted clock read is via
:func:`dashboard.data.utils.resolve_now`, threaded through once from
:func:`build_escalation_analytics` — see ``test_clock_discipline.py``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from bisect import bisect_left
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml
from escalation.classify import classify_resolver_tier, effective_benign
from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths

from dashboard.data.stats_utils import percentile
from dashboard.data.utils import parse_utc, resolve_now

logger = logging.getLogger(__name__)

# Module-relative default: dashboard/regime-markers.yaml (the top-level
# `dashboard/` package dir — three levels up from
# dashboard/src/dashboard/data/escalation_analytics.py).
_DEFAULT_REGIME_MARKERS_PATH = Path(__file__).resolve().parents[3] / 'regime-markers.yaml'


def load_regime_markers(path: Path | None = None) -> tuple[list[dict], int]:
    """Load hand-curated regime markers from a committed YAML file.

    Args:
        path: Path to the YAML file. Defaults to the committed
            ``dashboard/regime-markers.yaml``.

    Returns:
        ``(markers, parse_failures_delta)``. Never raises:

        - Missing file -> ``([], 0)`` (absent is not a failure).
        - Unparseable YAML or a non-list top level -> ``([], 1)`` + WARNING
          (row 9 substrate: the endpoint must never 500 on a broken markers
          file — it degrades to an empty list and counts the failure).

        Each returned marker is normalized to ``{date, label, tasks}``
        (``tasks`` defaults to ``[]``). ``date`` is coerced to ``str`` when
        YAML parses an unquoted ``YYYY-MM-DD`` scalar as a ``datetime.date``
        object — otherwise the payload would fail JSON serialization at the
        route layer.
    """
    p = path if path is not None else _DEFAULT_REGIME_MARKERS_PATH
    if not p.exists():
        return [], 0

    try:
        with p.open() as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning('load_regime_markers: failed to read/parse %s: %s', p, exc)
        return [], 1

    if not isinstance(data, list):
        logger.warning(
            'load_regime_markers: %s top level is not a list (got %s)',
            p, type(data).__name__,
        )
        return [], 1

    markers: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning('load_regime_markers: skipping non-mapping entry in %s: %r', p, item)
            continue
        raw_date = item.get('date')
        markers.append({
            'date': raw_date.isoformat() if isinstance(raw_date, date) else raw_date,
            'label': item.get('label'),
            'tasks': item.get('tasks') or [],
        })
    return markers, 0


# ---------------------------------------------------------------------------
# runs.db done-per-day (esc_per_done_daily substrate)
# ---------------------------------------------------------------------------


def _done_by_day(runs_db: Path) -> dict[str, int]:
    """Return ``{date: count}`` of ``outcome='done'`` task_results rows.

    Bucketed by ``date(completed_at)``. Sync, read-only (``mode=ro`` URI),
    fail-open — mirrors ``orchestrator.digest._query_events_ro``'s discipline
    applied to ``runs.db`` instead of an events DB: a missing DB logs at
    DEBUG and returns ``{}``; any other failure logs at WARNING and returns
    ``{}``. Never raises.
    """
    runs_db = Path(runs_db)
    try:
        if not runs_db.exists():
            logger.debug('_done_by_day: DB not found (fail-open): %s', runs_db)
            return {}
        db_uri = runs_db.resolve().as_uri() + '?mode=ro'
        conn = sqlite3.connect(db_uri, uri=True)
        try:
            rows = conn.execute(
                "SELECT date(completed_at), COUNT(*) FROM task_results "
                "WHERE outcome = 'done' AND completed_at IS NOT NULL AND completed_at != '' "
                "GROUP BY date(completed_at)"
            ).fetchall()
            return {row[0]: row[1] for row in rows if row[0] is not None}
        finally:
            conn.close()
    except Exception:
        # TOCTOU guard: re-detect a since-vanished DB as missing (DEBUG)
        # rather than an unexpected failure (WARNING).
        if not runs_db.exists():
            logger.debug('_done_by_day: DB not found (fail-open): %s', runs_db)
            return {}
        logger.warning('_done_by_day: query failed for %s', runs_db, exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Archive walk (record loading)
# ---------------------------------------------------------------------------


def _load_escalation_records(escalations_dir: Path) -> tuple[list[Escalation], int]:
    """Walk *escalations_dir* (queue root + archive) parsing every escalation file.

    Returns ``(records, parse_failures)``. ``triaged_at``/``triaged_by`` (2555)
    are first-class ``Escalation`` dataclass fields (``escalation.models``), so
    ``Escalation.from_dict``'s ``__dataclass_fields__`` filter already
    preserves them on the parsed record — no raw dict needs to be carried
    alongside it for later blocks (``triage_segments``) to read.

    A file that is not valid JSON, or whose parsed JSON cannot construct an
    ``Escalation`` (missing required field, non-dict top level, etc.), is
    counted in ``parse_failures`` and skipped — never raises. INV-4: the
    dashboard is the loud surface for a corrupt archive file (skipped AND
    counted), not a silent drop or a 500.
    """
    records: list[Escalation] = []
    parse_failures = 0
    for path in iter_all_escalation_paths(Path(escalations_dir)):
        try:
            esc = Escalation.from_dict(json.loads(path.read_text()))
        except Exception as exc:
            logger.warning('_load_escalation_records: failed to parse %s: %s', path, exc)
            parse_failures += 1
            continue
        records.append(esc)
    return records, parse_failures


# ---------------------------------------------------------------------------
# Origin block
# ---------------------------------------------------------------------------

# predictably_benign: benign_rate > _MIN_RATE AND n >= _MIN_N, computed over a
# trailing window keyed by resolved_at — a SEPARATE, windowed calculation
# from the all-time sources[].benign_rate field (see _origin_block docstring).
_PREDICTABLY_BENIGN_WINDOW_DAYS = 28
_PREDICTABLY_BENIGN_MIN_N = 20
_PREDICTABLY_BENIGN_MIN_RATE = 0.9


def _origin_block(records: list[Escalation], *, now: datetime) -> dict:
    """Per-source (``agent_role``) origin aggregates over *records*.

    ``sources[].filings`` counts every record for that source, regardless of
    status. ``benign``/``actionable``/``stamped_share``/``benign_rate`` are
    computed over ALL-TIME terminal (``resolved``/``dismissed``) records only
    (pending records are excluded — nothing to classify yet), via
    :func:`escalation.classify.effective_benign`.

    ``predictably_benign`` is a separate, windowed computation — trailing
    ``_PREDICTABLY_BENIGN_WINDOW_DAYS`` days keyed by ``resolved_at`` — True
    iff the windowed benign rate exceeds ``_PREDICTABLY_BENIGN_MIN_RATE`` AND
    the windowed classified count is at least ``_PREDICTABLY_BENIGN_MIN_N``.

    ``daily_by_source`` maps ``date(timestamp) -> {source: filing_count}``;
    ``daily_spark`` is each source's ascending-by-date filing-count series
    read back from that same map (sparkline shape). The two are deliberately
    redundant: ``daily_by_source`` is the cross-source-by-date shape (donut/
    stacked-bar rendering) and ``daily_spark`` is the pre-extracted per-source
    series (sparkline rendering) — both are provided so neither consumer has
    to reconstruct one from the other.
    """
    window_cutoff = now - timedelta(days=_PREDICTABLY_BENIGN_WINDOW_DAYS)

    filings: dict[str, int] = {}
    classified: dict[str, int] = {}
    benign: dict[str, int] = {}
    actionable: dict[str, int] = {}
    stamped: dict[str, int] = {}
    window_n: dict[str, int] = {}
    window_benign: dict[str, int] = {}
    daily_by_source: dict[str, dict[str, int]] = {}

    for esc in records:
        source = esc.agent_role
        filings[source] = filings.get(source, 0) + 1

        try:
            filed_date = parse_utc(esc.timestamp).date().isoformat()
        except (TypeError, ValueError):
            logger.warning(
                '_origin_block: unparseable timestamp on %s: %r', esc.id, esc.timestamp,
            )
        else:
            daily_by_source.setdefault(filed_date, {})
            daily_by_source[filed_date][source] = daily_by_source[filed_date].get(source, 0) + 1

        if esc.status not in ('resolved', 'dismissed'):
            continue  # pending (open) -> nothing to classify yet

        cls, provenance = effective_benign(esc)
        if cls is None:
            continue
        classified[source] = classified.get(source, 0) + 1
        if provenance == 'stamped':
            stamped[source] = stamped.get(source, 0) + 1
        if cls == 'benign':
            benign[source] = benign.get(source, 0) + 1
        elif cls == 'actionable':
            actionable[source] = actionable.get(source, 0) + 1

        try:
            resolved_dt = parse_utc(esc.resolved_at)
        except (TypeError, ValueError):
            logger.warning(
                '_origin_block: unparseable resolved_at on terminal %s: %r',
                esc.id, esc.resolved_at,
            )
            continue
        if resolved_dt >= window_cutoff:
            window_n[source] = window_n.get(source, 0) + 1
            if cls == 'benign':
                window_benign[source] = window_benign.get(source, 0) + 1

    sources = []
    for source, n_filings in sorted(filings.items()):
        n_classified = classified.get(source, 0)
        n_benign = benign.get(source, 0)
        n_window = window_n.get(source, 0)
        n_window_benign = window_benign.get(source, 0)
        window_rate = n_window_benign / n_window if n_window else 0.0
        dates = sorted(d for d, by_source in daily_by_source.items() if source in by_source)

        sources.append({
            'source': source,
            'filings': n_filings,
            'benign': n_benign,
            'actionable': actionable.get(source, 0),
            'stamped_share': stamped.get(source, 0) / n_classified if n_classified else 0.0,
            'benign_rate': n_benign / n_classified if n_classified else 0.0,
            'predictably_benign': (
                window_rate > _PREDICTABLY_BENIGN_MIN_RATE
                and n_window >= _PREDICTABLY_BENIGN_MIN_N
            ),
            'daily_spark': [daily_by_source[d][source] for d in dates],
        })

    return {'daily_by_source': daily_by_source, 'sources': sources}


# ---------------------------------------------------------------------------
# Lifespan block
# ---------------------------------------------------------------------------

_BREACH_SECONDS = 6 * 3600


def _downsample_stratified(samples: list[list], threshold: int) -> list[list]:
    """Deterministically downsample *samples* to at most *threshold* rows.

    Stratified by tier (``samples[i][1]``): each tier gets a quota
    proportional to its share of the total, allocated by the largest-
    remainder method (capped at the tier's own size) so quotas sum to
    exactly ``threshold`` whenever enough cross-tier slack exists — which it
    always does here, since *threshold* < ``len(samples)`` is the caller's
    precondition. Ties in the remainder distribution break deterministically
    on tier name (no RNG). Within each tier, rows are sorted by date
    (``samples[i][0]``) and the quota is picked via a fixed, evenly-spaced
    stride, so the selected rows stay in ascending date order. The returned
    list is grouped by tier (sorted tier name), date-ascending within tier.

    Callers must only invoke this when ``len(samples) > threshold`` — the
    precondition is asserted, not silently handled.
    """
    total = len(samples)
    assert total > threshold, '_downsample_stratified requires len(samples) > threshold'

    by_tier: dict[str, list[list]] = {}
    for row in samples:
        by_tier.setdefault(row[1], []).append(row)
    for rows in by_tier.values():
        rows.sort(key=lambda r: r[0])

    exact_quota = {tier: len(rows) * threshold / total for tier, rows in by_tier.items()}
    quota = {tier: min(int(exact_quota[tier]), len(rows)) for tier, rows in by_tier.items()}
    remaining = threshold - sum(quota.values())

    # Largest-remainder distribution of the leftover units: round-robin over
    # tiers in descending-fractional-part order (tie-broken by tier name) so
    # `remaining` can exceed the tier count without under-allocating.
    # Terminates because remaining < total-sum(quota) (cross-tier slack)
    # always holds when threshold < total; the guard is defensive only.
    priority = sorted(by_tier, key=lambda t: (-(exact_quota[t] - int(exact_quota[t])), t))
    idx = 0
    guard = 0
    while remaining > 0 and guard <= total:
        tier = priority[idx % len(priority)]
        if quota[tier] < len(by_tier[tier]):
            quota[tier] += 1
            remaining -= 1
        idx += 1
        guard += 1

    downsampled: list[list] = []
    for tier in sorted(by_tier):
        rows = by_tier[tier]
        q = quota[tier]
        if q <= 0:
            continue
        n = len(rows)
        if q >= n:
            downsampled.extend(rows)
            continue
        # Evenly-spaced index selection: int(i * n / q) is strictly
        # increasing for i in range(q) whenever q < n, so this always picks
        # q distinct, ascending (date-ordered, since rows are pre-sorted)
        # indices — no RNG, fully reproducible.
        downsampled.extend(rows[int(i * n / q)] for i in range(q))
    return downsampled


def _lifespan_block(
    records: list[Escalation],
    *,
    now: datetime,
    downsample_threshold: int = 10_000,
) -> dict:
    """Lifespan aggregates over *records*: percentiles, samples, open items, promotion.

    - ``percentiles_by_level``: per-``level`` (stringified) p50/p90 of
      ``resolved_at - timestamp`` seconds, over terminal-with-valid-times
      records.
    - ``samples``: one ``[date, tier, level, secs]`` row per terminal-with-
      valid-times record, ``date`` = ``date(resolved_at)`` (matches
      ``flow_daily``'s key so the two reconcile — row 11) and ``tier`` =
      :func:`~escalation.classify.classify_resolver_tier` of ``resolved_by``.
    - ``open_items``: one ``{id, task_id, level, age_secs, breach_6h}`` per
      pending record, ``age_secs`` = ``now - timestamp``,
      ``breach_6h`` = ``age_secs > 6h``.
    - ``l1_to_l2_promotion``: ``{count, p50_secs, p90_secs}`` computed from
      every ``level == 2`` record's ``members`` — ``L2.timestamp -
      member.timestamp`` per member id found in the by-id index (uses the
      L2's OWN ``timestamp``, i.e. promotion time, not its resolution —
      this is independent of whether the L2 itself is terminal or still
      pending). Missing members and negative deltas are skipped with a
      WARNING. ``p50_secs``/``p90_secs`` are ``None`` when no deltas were
      collected. No L0->L1 metric — see design_decisions (open question 7):
      the model has no machine-readable L0->L1 link.
    - ``triage_segments`` (render-when-present, 2555 forward-compat):
      ``{count, filed_to_triaged:{p50,p90}, triaged_to_resolved:{p50,p90}}``
      over terminal-with-valid-times records carrying a parseable
      ``esc.triaged_at`` (a first-class ``Escalation`` dataclass field).
      ``filed_to_triaged`` = ``triaged_at - timestamp``;
      ``triaged_to_resolved`` = ``resolved_at - triaged_at``. Omitted
      entirely (no key) when no record in the project carries a parseable
      ``triaged_at`` — this is a render-when-present field, not a
      zero-filled one.

    When ``len(samples) > downsample_threshold``, ``samples`` is
    deterministically downsampled (see :func:`_downsample_stratified`) and
    the result carries ``samples_downsampled=True`` plus ``samples_total``
    (the pre-downsample count) — a loud marker, never silent truncation.
    Below/at threshold, ``samples`` is untouched and neither key appears.
    """
    by_id: dict[str, Escalation] = {esc.id: esc for esc in records}

    secs_by_level: dict[int, list[float]] = {}
    samples: list[list] = []
    open_items: list[dict] = []
    promotion_deltas: list[float] = []
    filed_to_triaged_deltas: list[float] = []
    triaged_to_resolved_deltas: list[float] = []

    for esc in records:
        if esc.level == 2 and esc.members:
            try:
                l2_filed_at = parse_utc(esc.timestamp)
            except (TypeError, ValueError):
                logger.warning(
                    '_lifespan_block: unparseable timestamp on L2 cluster %s: %r',
                    esc.id, esc.timestamp,
                )
                l2_filed_at = None
            if l2_filed_at is not None:
                for member_id in esc.members:
                    member = by_id.get(member_id)
                    if member is None:
                        logger.warning(
                            '_lifespan_block: l1_to_l2_promotion member %s of %s not found '
                            'in archive', member_id, esc.id,
                        )
                        continue
                    try:
                        member_filed_at = parse_utc(member.timestamp)
                    except (TypeError, ValueError):
                        logger.warning(
                            '_lifespan_block: unparseable timestamp on member %s of %s',
                            member_id, esc.id,
                        )
                        continue
                    delta = (l2_filed_at - member_filed_at).total_seconds()
                    if delta < 0:
                        logger.warning(
                            '_lifespan_block: negative l1_to_l2_promotion delta (%.1fs) for '
                            'member %s of %s', delta, member_id, esc.id,
                        )
                        continue
                    promotion_deltas.append(delta)

        if esc.status == 'pending':
            try:
                filed_at = parse_utc(esc.timestamp)
            except (TypeError, ValueError):
                logger.warning(
                    '_lifespan_block: unparseable timestamp on pending %s: %r',
                    esc.id, esc.timestamp,
                )
                continue
            age_secs = (now - filed_at).total_seconds()
            open_items.append({
                'id': esc.id,
                'task_id': esc.task_id,
                'level': esc.level,
                'age_secs': age_secs,
                'breach_6h': age_secs > _BREACH_SECONDS,
            })
            continue

        if esc.status not in ('resolved', 'dismissed'):
            continue

        try:
            filed_at = parse_utc(esc.timestamp)
            resolved_at = parse_utc(esc.resolved_at)
        except (TypeError, ValueError):
            logger.warning(
                '_lifespan_block: unparseable timestamp(s) on terminal %s (timestamp=%r, '
                'resolved_at=%r)', esc.id, esc.timestamp, esc.resolved_at,
            )
            continue

        secs = (resolved_at - filed_at).total_seconds()
        tier = classify_resolver_tier(esc.resolved_by)
        secs_by_level.setdefault(esc.level, []).append(secs)
        samples.append([resolved_at.date().isoformat(), tier, esc.level, secs])

        triaged_at_raw = esc.triaged_at
        if triaged_at_raw:
            try:
                triaged_at = parse_utc(triaged_at_raw)
            except (TypeError, ValueError):
                logger.warning(
                    '_lifespan_block: unparseable triaged_at on terminal %s: %r',
                    esc.id, triaged_at_raw,
                )
            else:
                filed_to_triaged_deltas.append((triaged_at - filed_at).total_seconds())
                triaged_to_resolved_deltas.append((resolved_at - triaged_at).total_seconds())

    percentiles_by_level = {
        str(level): {
            'p50': percentile(sorted(secs_list), 50),
            'p90': percentile(sorted(secs_list), 90),
        }
        for level, secs_list in secs_by_level.items()
    }

    promotion_deltas.sort()
    l1_to_l2_promotion = {
        'count': len(promotion_deltas),
        'p50_secs': percentile(promotion_deltas, 50) if promotion_deltas else None,
        'p90_secs': percentile(promotion_deltas, 90) if promotion_deltas else None,
    }

    samples_total = len(samples)
    downsampled = samples_total > downsample_threshold
    if downsampled:
        samples = _downsample_stratified(samples, downsample_threshold)

    result = {
        'percentiles_by_level': percentiles_by_level,
        'l1_to_l2_promotion': l1_to_l2_promotion,
        'samples': samples,
        'open_items': open_items,
    }
    if downsampled:
        result['samples_downsampled'] = True
        result['samples_total'] = samples_total

    if filed_to_triaged_deltas:
        filed_to_triaged_deltas.sort()
        triaged_to_resolved_deltas.sort()
        result['triage_segments'] = {
            'count': len(filed_to_triaged_deltas),
            'filed_to_triaged': {
                'p50': percentile(filed_to_triaged_deltas, 50),
                'p90': percentile(filed_to_triaged_deltas, 90),
            },
            'triaged_to_resolved': {
                'p50': percentile(triaged_to_resolved_deltas, 50),
                'p90': percentile(triaged_to_resolved_deltas, 90),
            },
        }

    return result


# ---------------------------------------------------------------------------
# Workflow block
# ---------------------------------------------------------------------------

_CHURN_LOOKBACK = timedelta(hours=24)


def _workflow_block(records: list[Escalation], runs_db: Path) -> dict:
    """Workflow aggregates over *records*: tier/action mix, churn, throughput, flow cube.

    - ``tier_weekly``: ISO-week (``YYYY-Www``) -> ``{tier: count}``, over
      terminal records with a parseable ``resolved_at``
      (:func:`~escalation.classify.classify_resolver_tier` of ``resolved_by``).
    - ``action_mix``: terminal ``resolution_action`` counts, ``None`` ->
      ``'unspecified'`` (visible adoption gap — INV-4).
    - ``churn_daily``: ``date(timestamp) -> count`` of filings whose
      ``task_id`` had ANY other escalation resolved/dismissed within the
      24h before this filing's own ``timestamp``.
    - ``esc_per_done_daily``: per ``date(timestamp)``,
      ``{date, filings, done, ratio}`` — ``done`` from :func:`_done_by_day`
      against *runs_db*; ``ratio`` is ``None`` when ``done == 0`` (no div0).
    - ``flow_daily``: sparse ``[{date,source,level,tier,class,n}]`` cube over
      terminal-with-valid-times records (same population as
      ``lifespan.samples`` — both gate on parseable ``timestamp`` AND
      ``resolved_at`` — so their counts reconcile per row 11).
    """
    by_task: dict[str, list[Escalation]] = {}
    for esc in records:
        by_task.setdefault(esc.task_id, []).append(esc)

    tier_weekly: dict[str, dict[str, int]] = {}
    action_mix: dict[str, int] = {}
    flow_cube: dict[tuple[str, str, int, str, str], int] = {}

    for esc in records:
        if esc.status not in ('resolved', 'dismissed'):
            continue

        try:
            resolved_at = parse_utc(esc.resolved_at)
        except (TypeError, ValueError):
            logger.warning(
                '_workflow_block: unparseable resolved_at on terminal %s: %r',
                esc.id, esc.resolved_at,
            )
            continue

        tier = classify_resolver_tier(esc.resolved_by)
        iso = resolved_at.isocalendar()
        week_key = f'{iso.year}-W{iso.week:02d}'
        week_bucket = tier_weekly.setdefault(week_key, {})
        week_bucket[tier] = week_bucket.get(tier, 0) + 1

        action_key = esc.resolution_action or 'unspecified'
        action_mix[action_key] = action_mix.get(action_key, 0) + 1

        cls, _provenance = effective_benign(esc)
        if cls is None:
            continue
        try:
            parse_utc(esc.timestamp)  # gate: same "valid-times" population as lifespan.samples
        except (TypeError, ValueError):
            logger.warning(
                '_workflow_block: unparseable timestamp on terminal %s: %r',
                esc.id, esc.timestamp,
            )
            continue
        key = (resolved_at.date().isoformat(), esc.agent_role, esc.level, tier, cls)
        flow_cube[key] = flow_cube.get(key, 0) + 1

    # Pre-parse + sort each task's terminal resolved_at once so the per-filing
    # churn check below can bisect a [window_start, filed_at) range instead of
    # rescanning (and re-parsing timestamps in) that task's full escalation
    # history for every filing — O(log k + matches) rather than O(k) per
    # filing, where k = escalations ever filed against that task_id. Without
    # this, a task with a long re-filing history would make this block
    # quadratic in k.
    resolved_index: dict[str, tuple[list[datetime], list[str]]] = {}
    for task_id, task_records in by_task.items():
        pairs: list[tuple[datetime, str]] = []
        for other in task_records:
            if other.status not in ('resolved', 'dismissed'):
                continue
            try:
                other_resolved_at = parse_utc(other.resolved_at)
            except (TypeError, ValueError):
                continue
            pairs.append((other_resolved_at, other.id))
        pairs.sort(key=lambda p: p[0])
        resolved_index[task_id] = ([p[0] for p in pairs], [p[1] for p in pairs])

    churn_daily: dict[str, int] = {}
    filings_by_date: dict[str, int] = {}
    for esc in records:
        try:
            filed_at = parse_utc(esc.timestamp)
        except (TypeError, ValueError):
            logger.warning(
                '_workflow_block: unparseable timestamp on %s: %r', esc.id, esc.timestamp,
            )
            continue

        date_key = filed_at.date().isoformat()
        filings_by_date[date_key] = filings_by_date.get(date_key, 0) + 1

        window_start = filed_at - _CHURN_LOOKBACK
        resolved_ats, resolved_ids = resolved_index.get(esc.task_id, ([], []))
        lo = bisect_left(resolved_ats, window_start)
        hi = bisect_left(resolved_ats, filed_at)
        churned = any(resolved_ids[i] != esc.id for i in range(lo, hi))
        if churned:
            churn_daily[date_key] = churn_daily.get(date_key, 0) + 1

    done_by_date = _done_by_day(Path(runs_db))
    esc_per_done_daily = [
        {
            'date': d,
            'filings': filings_by_date.get(d, 0),
            'done': done_by_date.get(d, 0),
            'ratio': (
                filings_by_date.get(d, 0) / done_by_date[d] if done_by_date.get(d) else None
            ),
        }
        for d in sorted(set(filings_by_date) | set(done_by_date))
    ]

    flow_daily = [
        {'date': d, 'source': source, 'level': level, 'tier': tier, 'class': cls, 'n': n}
        for (d, source, level, tier, cls), n in sorted(flow_cube.items())
    ]

    return {
        'tier_weekly': tier_weekly,
        'action_mix': action_mix,
        'churn_daily': churn_daily,
        'esc_per_done_daily': esc_per_done_daily,
        'flow_daily': flow_daily,
    }


# ---------------------------------------------------------------------------
# Per-project aggregation
# ---------------------------------------------------------------------------


def _aggregate_project(
    project: str,
    escalations_dir: Path,
    runs_db: Path,
    *,
    now: datetime,
    downsample_threshold: int = 10_000,
) -> tuple[dict, int]:
    """Aggregate one project's escalation archive into a Seam-2 payload entry.

    Returns ``(entry, parse_failures)``.
    """
    records, parse_failures = _load_escalation_records(Path(escalations_dir))
    entry = {
        'project': project,
        'origin': _origin_block(records, now=now),
        'lifespan': _lifespan_block(records, now=now, downsample_threshold=downsample_threshold),
        'workflow': _workflow_block(records, Path(runs_db)),
    }
    return entry, parse_failures


# ---------------------------------------------------------------------------
# Top-level payload assembly
# ---------------------------------------------------------------------------


def build_escalation_analytics(
    project_dirs: list[tuple[str, Path, Path]],
    *,
    now: datetime | None = None,
    regime_markers_path: Path | None = None,
    downsample_threshold: int = 10_000,
) -> dict:
    """Build the full Seam-2 escalation-analytics payload across *project_dirs*.

    *project_dirs* is ``[(label, escalations_dir, runs_db), ...]``, primary
    project first. This is the PURE-SYNC core — the only permitted clock
    read in this module: ``now`` is resolved once via :func:`resolve_now`
    and threaded through every per-project aggregation, so no per-project
    call reads the live clock independently.

    ``parse_failures`` combines archive-record parse failures (summed
    across all project dirs) AND the regime-markers load's
    ``parse_failures_delta`` into a single loud count (INV-4) — there is
    only one ``parse_failures`` field in the payload.

    ``downsample_threshold`` bounds each project's ``lifespan.samples`` —
    see :func:`_lifespan_block` and :func:`_downsample_stratified`.

    Two archive-reach signals, both computed from ONE ``is_dir`` stat per
    project.  They exist because nothing else in the payload can answer either
    question — :func:`iter_all_escalation_paths` returns silently on a missing
    dir and :meth:`Path.glob` swallows ``PermissionError``, so an absent
    archive, an unreadable one and a genuinely empty one otherwise produce
    byte-identical entries.

    * ``archives_present`` (``all``) — the completeness DIAGNOSTIC: did this
      scan reach EVERY configured archive?  An operator reads it to find a
      root that is unmounted or misconfigured.
    * ``archives_reached`` (``any``) — the CACHE predicate (see
      :func:`archive_scan_succeeded`): did this scan reach ANY archive at all?
      A build that walked nothing is free to re-derive and must not pin an
      empty view for a TTL window past the moment the volume mounts; a build
      that walked something paid for that walk, and the cache is what stops it
      being paid again on the next poll.

    They are separate fields because ``all`` is UNSOUND as a cache predicate,
    which is not hypothetical: measured 2026-08-01 against the installed
    unit's own ``DASHBOARD_KNOWN_PROJECT_ROOTS`` (9 roots), 2 roots have no
    ``data/escalations`` dir at all while the other 7 hold ~9.1k records
    between them.  Keying on ``all`` there means the TTL never stores
    anything and every 3s poll re-runs the whole multi-second walk — ``all``
    reports "this scan was free to redo" at precisely the moment it was most
    expensive.  A root that has simply never escalated must not delete the
    cache in front of the other seven.

    Their shared LIMIT, stated because a signal is only useful if its blind
    spot is known: one ``is_dir`` stat distinguishes the ABSENT/unmounted
    case, not every permission flap — a directory that exists but whose own
    mode is stripped still passes ``is_dir`` and then globs empty.  Both are
    deliberately not ``parse_failures``, which counts unparseable RECORDS:
    that count is a permanent property of a corrupt file, so a cache keyed on
    it would be defeated forever in front of the ~10k-record walk the cache
    exists to prevent.  All three fields answer different questions and none
    is derived from another.
    """
    resolved_now = resolve_now(now)

    parse_failures = 0
    per_project = []
    for project, escalations_dir, runs_db in project_dirs:
        entry, project_parse_failures = _aggregate_project(
            project, escalations_dir, runs_db, now=resolved_now,
            downsample_threshold=downsample_threshold,
        )
        parse_failures += project_parse_failures
        per_project.append(entry)

    markers, markers_parse_failures = load_regime_markers(regime_markers_path)
    parse_failures += markers_parse_failures

    # ONE pass, one stat per project, feeding both signals — so the two can
    # never drift apart by disagreeing about what was on disk when.
    reached = [
        Path(escalations_dir).is_dir() for _label, escalations_dir, _runs_db in project_dirs
    ]

    return {
        'generated_at': resolved_now.isoformat(),
        'parse_failures': parse_failures,
        'regime_markers': markers,
        'per_project': per_project,
        # ``all`` — the completeness diagnostic an operator reads.
        'archives_present': all(reached),
        # ``any`` — the cache predicate; see archive_scan_succeeded.
        'archives_reached': any(reached),
    }


def archive_scan_succeeded(payload: dict) -> bool:
    """Did this scan reach any escalation archive at all?

    The cacheability predicate for the analytics route's TTL cache
    (``cache_ok=archive_scan_succeeded``).  False for exactly one payload: the
    one built without reaching a single archive, where every configured
    ``escalations_dir`` failed its :meth:`Path.is_dir` check and nothing was
    walked.  That is O(1) to recompute — one negative stat per project — so
    re-deriving it on every poll costs nothing, while caching it keeps the tab
    reporting an empty archive for a full TTL window after the volume mounts.

    A PARTIAL scan is cacheable, and that is the whole point of keying on
    ``archives_reached`` (``any``) rather than ``archives_present`` (``all``):
    the ordinary multi-project config has roots that have simply never
    escalated, and one of those must not delete the cache sitting in front of
    every other root's ~10k-record walk.  ``archives_present: False`` rides
    along in the payload as the diagnostic and has no say here.  The accepted
    trade-off is the narrow one: an archive that disappears mid-life leaves
    that project's panel up to one TTL window stale.

    Deliberately not keyed on ``parse_failures`` — that counts unparseable
    RECORDS and is permanent for a corrupt file, so gating on it would defeat
    the cache forever in front of the very walk it protects.

    Reads through ``.get`` for the same reason
    :func:`dashboard.data.memory_evals.root_scan_succeeded` does: it runs
    inside the cache write path, where a partially-built or older-shaped
    payload must degrade to "don't cache" rather than raise where a raise
    would surface as a 500 on a 3s dashboard poll.
    """
    return bool(payload.get('archives_reached'))
