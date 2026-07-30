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


def _load_json(path: Path) -> Any:
    """Parse *path*, or raise for the caller's narrow handler to record."""
    return json.loads(path.read_text())


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


def _build_eval(eval_dir: Path) -> dict[str, Any]:
    """Assemble one eval's trend payload from its ``metrics-*.json`` series.

    Runs are ordered by FILENAME — the producer's own contract
    (``shared.memory_eval_metrics.load_series_window`` sorts the same way,
    because the stamp is a zero-padded UTC string) — so the dashboard never
    invents a second ordering rule.  The x-axis label for each run is the
    artifact's in-body ``run_stamp``.
    """
    eval_id = eval_dir.name
    paths = sorted(eval_dir.glob('metrics-*.json'))

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
            'current_value': current.get('value'),
            'n': current.get('n'),
            'denominator': current.get('denominator'),
            'direction': current.get('direction'),
            'trend': {'labels': list(run_stamps), 'values': values},
        })

    return {
        'eval_id': eval_id,
        'run_stamps': run_stamps,
        'run_count': len(run_stamps),
        'runs_on_disk': len(paths),
        'truncated': False,
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

    payload: dict[str, Any] = {
        'generated_at': resolved_now.isoformat(),
        'root_present': True,
        'evals': [],
        'issues': [],
        'issue_count': 0,
        'unmatched_escalations': [],
    }

    eval_dirs = sorted(p for p in memory_evals_dir.iterdir() if p.is_dir())
    payload['evals'] = [_build_eval(d) for d in eval_dirs]
    return payload
