"""Entry point for ``python -m sampler`` — the ExecStart target of the systemd oneshot.

Invoked by the dark-factory-load-sampler.service unit every 5 seconds via the
paired .timer.  Each invocation is a fresh process; state persists only via the
SQLite DB at data/load-samples.db (relative to the repo root WorkingDirectory).

This module is intentionally thin — all business logic lives in sampler.sampler
and sampler.store so it can be unit-tested.  The live integration signal
(``systemctl --user is-active dark-factory-load-sampler.timer`` plus
``sqlite3 data/load-samples.db 'SELECT COUNT(DISTINCT metric) FROM samples
WHERE ts > strftime(...)' `` returning >= 7) validates this shell.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from sampler.metrics import collect_psi, collect_process_metrics
from sampler.sampler import run_tick
from sampler.store import LoadSampleStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
logger = logging.getLogger('sampler')


def main() -> None:
    """Collect one tick of metrics and write them to data/load-samples.db."""
    db_path = Path('data/load-samples.db')
    store = LoadSampleStore(db_path)

    now = int(time.time())

    try:
        psi = collect_psi()
    except Exception:
        logger.exception('Failed to collect PSI metrics')
        sys.exit(1)

    try:
        process_metrics = collect_process_metrics()
    except Exception:
        logger.exception('Failed to collect process metrics')
        sys.exit(1)

    run_tick(store, now, psi=psi, process_metrics=process_metrics)
    store.maybe_vacuum(now)

    logger.info(
        'tick ts=%d psi=%s process=%s',
        now,
        {k: f'{v:.2f}' for k, v in psi.items()},
        {k: f'{v:.0f}' for k, v in process_metrics.items()},
    )


if __name__ == '__main__':
    main()
