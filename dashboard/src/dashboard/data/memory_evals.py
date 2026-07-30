"""Backend data layer for the Memory-evals dashboard section (task 3215).

Reads the memory-eval program's on-disk artifacts and shapes them for the
``/api/v2/dashboard/memory-evals`` route.  See
``docs/prds/memory-eval-program.md`` (M1 metric series, M2 limits + verdicts,
M3 escalation contract) and ``docs/prds/memory-eval-dashboard.md``.

**Artifacts only, never the module (G6/INV-5).**  This reader uses plain
``json.load`` + dict access and deliberately does NOT import
``shared.memory_eval_metrics`` / ``shared.memory_eval_limits`` — exactly as
``shared/tests/fixtures/memory_eval/README.md`` describes the dashboard-shaped
reader.  The artifact series on disk is the published contract; importing the
producer would couple the dashboard to the producer's in-memory objects and
invite a second implementation of what the producer already decided.  For the
same reason there is **no statistics of any kind** here: no ``math.comb``, no
``math.lgamma``, no p-value, no threshold.  Verdicts are *read*, never
re-derived (INV-1: same file the evaluator read).

**Loud, never silent (DD6/INV-2).**  Every artifact this reader cannot use is
recorded in the payload's structured ``issues`` list — named, with its
``eval_id`` and path — not merely logged and not silently dropped.  Nothing in
here raises: a degraded tree yields a degraded payload, never a 500.  Staleness
is *displayed*, never alarmed on; this module files no escalations.

**Same-host file reads (DD1).**  The dashboard and the eval runner share a
filesystem; there is no RPC in this path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

# One screen of trend == one alpha-derivation window.  Matches
# ``runs_per_quarter=90`` in the committed limits artifact, which is what the
# program's derived alpha is computed over, so the chart shows exactly the
# window the limits govern.  When more runs exist on disk the payload says so
# (``truncated`` / ``runs_on_disk``) — a dropped run is never silent.
_TREND_RUN_CAP = 90

# The closed metric-kind vocabulary (M1).  A kind outside this set is a
# RENDERING failure — there is no chart primitive for it — so it earns an
# issue.  Other semantic violations of the M1 schema are the producer's to
# reject at emit time and are passed through verbatim here.
_KNOWN_KINDS = frozenset({'tripwire', 'proportion', 'count', 'scalar'})

# The limits artifact carries a great deal more than this — ``alarms``,
# ``alarmed_metric_count``, ``grandfather_set``, ``snapshotted_metric_ids`` and
# its own embedded ``verdicts[]``.  Only this whitelist is surfaced, and it is
# PROVENANCE: what alpha the program derived, over which baseline runs, against
# which grandfather set.  See ``_read_limits`` for why the rest is ignored.
_LIMITS_PROVENANCE_KEYS = (
    'alpha',
    'false_alarm_budget',
    'runs_per_quarter',
    'min_samples',
    'baseline_window',
    'baseline_run_stamps',
    'grandfather_set_hash',
    'run_stamp',
    'generator',
)


def _load_json(path: Path) -> Any:
    """Parse *path*, or raise for the caller's narrow handler to record."""
    return json.loads(path.read_text())


def _issue(
    issues: list[dict[str, Any]],
    kind: str,
    *,
    eval_id: str | None = None,
    path: Path | str | None = None,
    detail: str = '',
) -> None:
    """Record one degraded artifact — named, located, and counted (DD6/INV-2)."""
    issues.append({
        'kind': kind,
        'eval_id': eval_id,
        'path': str(path) if path is not None else None,
        'detail': detail,
    })


def _read_limits(
    eval_dir: Path,
    latest_run_stamp: str | None,
    issues: list[dict[str, Any]],
    *,
    has_runs: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return ``(provenance_block, metric_id -> limits record)`` for one eval.

    Per-eval, NOT at the root: ``shared.memory_eval_limits.limits_artifact_path``
    returns ``<root>/<eval_id>/limits-current.json`` and the committed exemplar
    sits at ``e1-retrieval-health/limits-current.json``.  The committed
    fixtures are the contract.

    The artifact's embedded ``verdicts[]`` supplies ONE dashboard input —
    ``rule_kind``, the rule that governs each metric.  Its ``status`` /
    ``p_value`` / ``baseline`` / ``alarms`` are deliberately NOT read as verdict
    state: that vocabulary (``baseline_snapshot|ok|alarm|improved|
    insufficient_data``) is not the M2 verdict vocabulary the dashboard
    displays, and these records carry no ``fingerprint``, so they cannot
    participate in the escalation join at all.  Reading both sources would hand
    the UI two disagreeing alarm truths and force a dashboard-side mapping
    between the vocabularies — exactly the re-derivation G6/INV-5 forbids.
    ``verdicts-current.json`` is the sole verdict source.
    """
    path = eval_dir / 'limits-current.json'
    if not path.is_file():
        if has_runs:
            # Metrics on disk with no limits beside them means the evaluator is
            # not completing — a real silent-failure mode, since the UI would
            # otherwise show trends with a blank provenance block and look fine.
            _issue(
                issues,
                'missing_limits',
                eval_id=eval_dir.name,
                path=path,
                detail=f'{eval_dir.name} has metrics runs but no limits artifact',
            )
        return None, {}

    body = _load_json(path)
    if not isinstance(body, dict):
        return None, {}

    # ``.get()`` throughout: a schema addition upstream is inert here, never
    # fatal, and an omitted field reads as absent rather than crashing.
    block = {key: body.get(key) for key in _LIMITS_PROVENANCE_KEYS}
    # One string comparison, not a re-derivation.  The committed exemplar
    # exhibits this skew itself (limits stamped 20260704, newest metrics
    # 20260705); displaying the alpha/baseline provenance beside a newer
    # current value without disclosing it would present stale provenance as
    # though it governed the displayed run.
    block['stale_for_latest_run'] = body.get('run_stamp') != latest_run_stamp

    by_metric: dict[str, Any] = {}
    for record in body.get('verdicts') or []:
        if isinstance(record, dict) and isinstance(record.get('metric_id'), str):
            by_metric[record['metric_id']] = record
    return block, by_metric


def _metric_rows(body: Any) -> list[dict]:
    """The metric records of one parsed run body, or ``[]`` if unusable."""
    if not isinstance(body, dict):
        return []
    metrics = body.get('metrics')
    if not isinstance(metrics, list):
        return []
    return [
        m for m in metrics
        if isinstance(m, dict) and isinstance(m.get('metric_id'), str) and m.get('metric_id')
    ]


def _build_eval(eval_dir: Path, issues: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble one eval's trend payload from its ``metrics-*.json`` series.

    Runs are ordered by FILENAME — the producer's own contract
    (``shared.memory_eval_metrics.load_series_window`` sorts the same way,
    because the stamp is a zero-padded UTC string) — so the dashboard never
    invents a second ordering rule.  The x-axis label for each run is the
    artifact's in-body ``run_stamp``.
    """
    eval_id = eval_dir.name
    all_paths = sorted(eval_dir.glob('metrics-*.json'))

    # Trailing window: keep the MOST RECENT runs, drop the oldest.  Dropping
    # the newest instead would leave a trend that ends N runs ago reading as
    # current.  ``_TREND_RUN_CAP`` is looked up in the module namespace here
    # (not captured at import) so it stays one adjustable knob.
    #
    # No silent caps: whenever this window drops anything, the count dropped
    # is disclosed IN THE PAYLOAD (``truncated`` + ``runs_on_disk`` beside
    # ``run_count``), never only in a log the operator will not read.
    runs_on_disk = len(all_paths)
    paths = all_paths[-_TREND_RUN_CAP:] if _TREND_RUN_CAP else all_paths
    truncated = len(paths) < runs_on_disk

    runs: list[tuple[str, dict[str, dict]]] = []
    corpus: dict | None = None
    for path in paths:
        body = _load_json(path)
        rows = _metric_rows(body)
        stamp = body.get('run_stamp') if isinstance(body, dict) else None
        if isinstance(body, dict) and isinstance(body.get('corpus'), dict):
            corpus = dict(body['corpus'])
        runs.append((stamp, {row['metric_id']: row for row in rows}))

    run_stamps = [stamp for stamp, _ in runs]

    # Metric identity is the metric_id, unioned across the whole window: a
    # metric that appears only in some runs still gets a full-width series
    # (with holes), so every series stays index-aligned to the shared axis.
    metric_ids: list[str] = []
    for _, by_id in runs:
        for metric_id in by_id:
            if metric_id not in metric_ids:
                metric_ids.append(metric_id)

    latest_run_stamp = run_stamps[-1] if run_stamps else None
    limits, limits_by_metric = _read_limits(
        eval_dir, latest_run_stamp, issues, has_runs=bool(all_paths),
    )

    latest = runs[-1][1] if runs else {}
    metrics: list[dict[str, Any]] = []
    for metric_id in sorted(metric_ids):
        # A metric missing from a run contributes a None hole at that index
        # rather than being dropped — dropping would silently shift this
        # metric's points against its neighbours'.
        values = [by_id.get(metric_id, {}).get('value') for _, by_id in runs]
        current = latest.get(metric_id, {})
        metrics.append({
            'metric_id': metric_id,
            'kind': current.get('kind'),
            # Looked up by metric_id, never zipped positionally: the two lists
            # are ordered independently and a metric may be absent from either.
            'rule_kind': limits_by_metric.get(metric_id, {}).get('rule_kind'),
            'current_value': current.get('value'),
            'n': current.get('n'),
            'denominator': current.get('denominator'),
            'direction': current.get('direction'),
            'trend': {'labels': list(run_stamps), 'values': values},
            # The verdict column exists even when empty — absent verdict state
            # renders as an explicit gap, never as an implied "no alarm".
            'verdict': None,
        })

    return {
        'eval_id': eval_id,
        'latest_run_stamp': latest_run_stamp,
        'limits': limits,
        'run_stamps': run_stamps,
        'run_count': len(run_stamps),
        'runs_on_disk': runs_on_disk,
        'truncated': truncated,
        'corpus': corpus,
        'metrics': metrics,
    }


def build_memory_evals(
    memory_evals_dir: Path,
    escalations_dir: Path,
    *,
    now: Any = None,
) -> dict[str, Any]:
    """Aggregate the memory-eval artifact tree into one dashboard payload.

    Args:
        memory_evals_dir: The memory-eval artifact root
            (``<project_root>/fused-memory/data/memory-evals``); enumerated
            generically — one entry per subdirectory, no per-eval code (DD5).
        escalations_dir: The recon escalation queue dir, joined by fingerprint.
        now: Request-scoped reference timestamp, resolved ONCE via
            :func:`~dashboard.data.utils.resolve_now` and threaded through
            every derived time field.  Injectable so staleness is testable
            without freezing the clock.

    Returns:
        The ``MEMORY_EVALS`` payload body.  Never raises.
    """
    resolved_now = resolve_now(now)

    issues: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        'generated_at': resolved_now.isoformat(),
        'root_present': True,
        'evals': [],
        'issues': issues,
        'issue_count': 0,
        'unmatched_escalations': [],
    }

    eval_dirs = sorted(p for p in memory_evals_dir.iterdir() if p.is_dir())
    payload['evals'] = [_build_eval(d, issues) for d in eval_dirs]
    payload['issue_count'] = len(issues)
    return payload
