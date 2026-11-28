"""The single definition of a routing-experiment arm.

Shared by the orchestrator resolver
(``orchestrator/src/orchestrator/routing.py::_rule_matches``), the experiment
readout script and the fused-memory curator, so all three always agree on
which arm a key falls in. Contract: plans/routing-experiment-substrate-prd.md
§"Contract: split_bucket".
"""

from __future__ import annotations

import hashlib


def split_bucket(salt: str, key: str, buckets: int) -> int:
    """Return the arm, in ``range(buckets)``, that ``key`` falls in under ``salt``.

    ``key`` is the caller's stable key: the task id as ``RouteInputs`` carries
    it for experiment 1, the candidate_key for experiment 2. Pure, with no I/O,
    and stable across processes and hosts.
    """
    if buckets < 2:
        raise ValueError(f'split_bucket: buckets must be >= 2, got {buckets!r}')
    digest = hashlib.sha256(f'{salt}\x00{key}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % buckets
