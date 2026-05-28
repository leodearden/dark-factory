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
import os
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
    """Collect one tick of metrics and write them to data/load-samples.db.

    The DB path is ``<root>/data/load-samples.db`` where ``<root>`` defaults to
    the process's CWD (set by the systemd ``WorkingDirectory=`` to the repo
    root) or is overridden by the ``DARK_FACTORY_ROOT`` environment variable.
    This allows relocating the checkout without editing the unit file.
    """
    root = os.environ.get('DARK_FACTORY_ROOT', '.')
    db_path = Path(root) / 'data/load-samples.db'

    try:
        store = LoadSampleStore(db_path)
    except Exception:
        logger.exception('Failed to open/create store at %s; aborting tick', db_path)
        sys.exit(1)

    now = int(time.time())

    # Degrade-and-continue: a failure in one collection group (e.g. kernel
    # lacking PSI support) should not discard the other group's metrics.
    # Each tick is independent, so partial writes are better than no write.
    # Only store-construction failure justifies a non-zero exit.
    try:
        psi = collect_psi()
    except Exception:
        logger.exception('Failed to collect PSI metrics; writing process metrics only')
        psi = {}

    try:
        process_metrics = collect_process_metrics()
    except Exception:
        logger.exception('Failed to collect process metrics; writing PSI metrics only')
        process_metrics = {}

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
