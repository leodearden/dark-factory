"""Entry point for ``python -m sampler`` — the ExecStart target of the systemd oneshot.

Invoked by the dark-factory-load-sampler.service unit every 5 seconds via the
paired .timer.  Each invocation is a fresh process; state persists only via the
SQLite DB at data/load-samples.db (relative to the repo root WorkingDirectory).

This module is intentionally thin — all business logic lives in sampler.sampler
and sampler.store so it can be unit-tested.  The live integration signal
(``systemctl --user is-active dark-factory-load-sampler.timer`` plus a
``SELECT COUNT(DISTINCT metric) FROM samples WHERE ts > ...`` query against
data/load-samples.db) validates this shell; ``scripts/install-load-sampler.sh``
runs exactly that check after arming the timer.

That count depends on how many cgroup leaves a host has (the per-tick row
formula is stated on ``sampler.metrics.METRIC_GROUPS``), so the floor worth
asserting is that rows arrived at all, not that a particular count did.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from sampler.metrics import METRIC_GROUPS, MetricGroup, Samples
from sampler.sampler import run_tick
from sampler.store import LoadSampleStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)
logger = logging.getLogger('sampler')


def main(groups: Sequence[MetricGroup] = METRIC_GROUPS) -> None:
    """Collect one tick of *groups* and write it to data/load-samples.db.

    The DB path is ``<root>/data/load-samples.db`` where ``<root>`` defaults to
    the process's CWD (set by the systemd ``WorkingDirectory=`` to the repo
    root) or is overridden by the ``DARK_FACTORY_ROOT`` environment variable.
    This allows relocating the checkout without editing the unit file.

    A group whose collector fails writes no rows this tick and the others are
    still written; only a store that cannot be opened exits non-zero.
    """
    root = os.environ.get('DARK_FACTORY_ROOT', '.')
    db_path = Path(root) / 'data/load-samples.db'

    try:
        store = LoadSampleStore(db_path)
    except Exception:
        logger.exception('Failed to open/create store at %s; aborting tick', db_path)
        sys.exit(1)

    now = int(time.time())
    collected = {group: _collect(group) for group in groups}
    written = run_tick(store, now, collected)
    store.maybe_vacuum(now)

    logger.info(
        'tick ts=%d %s',
        now,
        ' '.join(f'{group.name}={_render(written.get(group))}' for group in groups),
    )


def _collect(group: MetricGroup) -> Samples:
    """The one degrade handler: a failed collection is ``{}``, never a fabricated 0.0."""
    try:
        return group.collect()
    except Exception:
        logger.exception('Failed to collect %s metrics; writing the other groups only', group.name)
        return {}


def _render(samples: Samples | None) -> str:
    if samples is None:
        return 'dropped'
    return str({key.label(): f'{value:.2f}' for key, value in samples.items()})


if __name__ == '__main__':
    main()
