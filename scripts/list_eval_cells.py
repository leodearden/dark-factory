#!/usr/bin/env python3
"""Print one task's persisted eval cells, one path per line, as orchestrator/src/orchestrator/evals/results.py::ResultStore names them."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bind orchestrator to THIS checkout rather than whichever copy an editable install put on
# sys.path (scripts/backfill_decision_queue_stamp.py::_ORCH_SRC). A bare sys.path.insert,
# which ruff's E402 exempts, so the import below needs no noqa.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'orchestrator' / 'src'))

from orchestrator.evals.results import ResultStore


def store_for(results_dir: Path | None) -> ResultStore:
    return ResultStore.packaged() if results_dir is None else ResultStore(results_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_id')
    parser.add_argument(
        '--results-dir', type=Path, default=None,
        help='the store to list (default: ResultStore.packaged(), where eval-ofat writes)',
    )
    args = parser.parse_args(argv)
    for path in store_for(args.results_dir).paths_for_task(args.task_id):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
