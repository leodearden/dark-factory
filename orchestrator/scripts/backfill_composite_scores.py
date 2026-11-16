"""Backfill composite_score on existing eval result JSON files.

Re-scores every cell of the packaged result store
(``orchestrator/src/orchestrator/evals/results.py::ResultStore.packaged``), in
place, using the current ``compute_composite`` implementation. Used after the ε fix
that removed ``plan_completion_pct`` from the composite, so vLLM runs with
T/T/T gates but plan-bookkeeping=0 score > 0 again.

Usage::

    uv run --project orchestrator python orchestrator/scripts/backfill_composite_scores.py
"""

from __future__ import annotations

import dataclasses

from orchestrator.evals.metrics import EvalMetrics, compute_composite
from orchestrator.evals.results import ResultStore

_KNOWN_METRICS = frozenset(f.name for f in dataclasses.fields(EvalMetrics))


def _rescored(cell: dict) -> dict:
    metrics = cell.get('metrics') or {}
    if metrics:
        clean = {k: v for k, v in metrics.items() if k in _KNOWN_METRICS}
        metrics['composite_score'] = compute_composite(EvalMetrics(**clean))
    return cell


def main() -> int:
    store = ResultStore.packaged()
    files = store.cell_paths()
    updated = sum(store.rewrite(path, _rescored) for path in files)

    print(f'Scanned: {len(files)}')
    print(f'Updated: {updated}')
    print(f'Unchanged: {len(files) - updated}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
