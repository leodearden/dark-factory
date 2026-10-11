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
import json

from orchestrator.evals.metrics import EvalMetrics, compute_composite
from orchestrator.evals.results import ResultStore


def main() -> int:
    known = {f.name for f in dataclasses.fields(EvalMetrics)}
    files = ResultStore.packaged().cell_paths()
    scanned = 0
    updated = 0
    unchanged = 0

    for path in files:
        scanned += 1
        with open(path) as f:
            data = json.load(f)

        metrics_dict = data.get('metrics') or {}
        if not metrics_dict:
            unchanged += 1
            continue

        clean = {k: v for k, v in metrics_dict.items() if k in known}
        em = EvalMetrics(**clean)
        new_score = compute_composite(em)
        old_score = metrics_dict.get('composite_score')

        if old_score == new_score:
            unchanged += 1
            continue

        data['metrics']['composite_score'] = new_score
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        updated += 1

    print(f'Scanned: {scanned}')
    print(f'Updated: {updated}')
    print(f'Unchanged: {unchanged}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
