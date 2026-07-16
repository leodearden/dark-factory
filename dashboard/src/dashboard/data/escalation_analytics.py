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
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml
from escalation.classify import classify_resolver_tier, effective_benign
from escalation.models import Escalation
from escalation.queue import iter_all_escalation_paths

from dashboard.data.stats_utils import percentile
from dashboard.data.utils import parse_utc

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


def _load_escalation_records(escalations_dir: Path) -> tuple[list[tuple[Escalation, dict]], int]:
    """Walk *escalations_dir* (queue root + archive) parsing every escalation file.

    Returns ``(records, parse_failures)``: each record pairs the parsed
    :class:`~escalation.models.Escalation` with its *raw* dict. The raw dict
    is retained so forward-compat fields dropped by ``Escalation.from_dict``'s
    ``__dataclass_fields__`` filter (e.g. 2555's ``triaged_at``/``triaged_by``)
    survive for later blocks (``triage_segments``) with no model change.

    A file that is not valid JSON, or whose parsed JSON cannot construct an
    ``Escalation`` (missing required field, non-dict top level, etc.), is
    counted in ``parse_failures`` and skipped — never raises. INV-4: the
    dashboard is the loud surface for a corrupt archive file (skipped AND
    counted), not a silent drop or a 500.
    """
    records: list[tuple[Escalation, dict]] = []
    parse_failures = 0
    for path in iter_all_escalation_paths(Path(escalations_dir)):
        try:
            raw = json.loads(path.read_text())
            esc = Escalation.from_dict(raw)
        except Exception as exc:
            logger.warning('_load_escalation_records: failed to parse %s: %s', path, exc)
            parse_failures += 1
            continue
        records.append((esc, raw))
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


def _origin_block(records: list[tuple[Escalation, dict]], *, now: datetime) -> dict:
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
    read back from that same map (sparkline shape).
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

    for esc, _raw in records:
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


def _lifespan_block(records: list[tuple[Escalation, dict]], *, now: datetime) -> dict:
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
    """
    by_id: dict[str, Escalation] = {esc.id: esc for esc, _raw in records}

    secs_by_level: dict[int, list[float]] = {}
    samples: list[list] = []
    open_items: list[dict] = []
    promotion_deltas: list[float] = []

    for esc, _raw in records:
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

    return {
        'percentiles_by_level': percentiles_by_level,
        'l1_to_l2_promotion': l1_to_l2_promotion,
        'samples': samples,
        'open_items': open_items,
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
) -> tuple[dict, int]:
    """Aggregate one project's escalation archive into a Seam-2 payload entry.

    Returns ``(entry, parse_failures)``. *runs_db* is accepted here (rather
    than only by the workflow block) so the per-project signature is stable
    across steps — it is not yet read by the origin/lifespan blocks; the
    workflow block's ``esc_per_done_daily`` (via ``_done_by_day``) consumes
    it in a later step.
    """
    records, parse_failures = _load_escalation_records(Path(escalations_dir))
    entry = {
        'project': project,
        'origin': _origin_block(records, now=now),
        'lifespan': _lifespan_block(records, now=now),
    }
    return entry, parse_failures
