"""FalkorDB index-provisioning drift diagnostics (task 3709, PRD δ).

Surfaces a silent-failure tail that went unobserved for four months: graphiti's
index provisioning was stubbed out, so graphs served queries with none of the
indices they were supposed to have.  Nothing failed — reads just got slower and
no signal was ever emitted.

`summarize_index_health()` diffs the ACTUAL spec set (what `CALL db.indexes()`
reported, projected through α's `normalize_index_records()`) against the
EXPECTED set (α's `expected_index_set()`) and produces a machine-readable health
record.  Reconciliation surfaces that record in Stage 1 `report.stats` and
WARNING logs, and the harness's drift detector escalates on it — making the
missing indices observable without any change to the read path.

There is deliberately NO second source of truth here for what should exist
(INV-5 / PRD D3): the expected set is α's, passed in by the caller.

Design: no I/O except a WARNING log on a degenerate `expected` set.  All other
I/O is performed by the harness in `_check_index_health()`; only interpretation
lives here — the same split `queue_health.summarize_graphiti_queue_health()`
uses.

Imports NOTHING from `backends.graphiti_client` (and nothing that would pull the
graphiti/LLM/embedder stack in at import time), matching `falkor_indices`'
isolation rule.  Index specs are typed structurally as
`tuple[str, str, str, str]` — the same shape as `falkor_indices.IndexSpec` —
rather than imported, for the same reason.
"""

from __future__ import annotations

import logging
from collections.abc import Set as AbstractSet

logger = logging.getLogger(__name__)

# (label, entity_type, field, index_type) — structurally identical to
# falkor_indices.IndexSpec, restated here to keep this module import-light.
_IndexSpec = tuple[str, str, str, str]


def summarize_index_health(
    actual: AbstractSet[_IndexSpec],
    expected: AbstractSet[_IndexSpec],
) -> dict:
    """Classify a graph's actual index set against the expected one.

    Args:
        actual: Specs the graph actually has, from
            `normalize_index_records(await graphiti.list_indices(group_id=...))`.
        expected: Specs the graph should have, from `expected_index_set()`.

    Returns:
        dict with:
            healthy (bool): True when nothing expected is missing AND *expected*
                is non-empty (see the fail-closed note below).
            missing (list): Expected-but-absent specs, SORTED.  This is the
                drift signal, and the sort makes the escalation payload
                deterministic and diffable — set iteration order varies between
                processes, so two identical findings would otherwise serialise
                differently.
            unexpected (list): Present-but-unexpected specs, SORTED.  Reported,
                never acted on (PRD D8): an operator-added index is not drift to
                repair, so it does NOT flip *healthy*.
            expected_total (int): len(expected).
            actual_total (int): len(actual).

    Emits a WARNING and forces `healthy=False` when *expected* is empty.

    Why the empty-`expected` guard exists: `healthy` keys off `missing == []`,
    and `missing = expected - actual` is vacuously empty whenever *expected* is
    empty — so a degenerate expected set would report every graph on the server
    as perfectly healthy.  That is a silent false all-clear, and precisely this
    PRD's own failure mode (a stub that produced no signal for four months)
    recurring inside the detector built to catch it.  `expected_index_set()`
    raises `UnparsedIndexStatementError` rather than shortening its result, so
    this case should be unreachable; failing CLOSED costs one branch and
    guarantees that if it ever DOES become reachable — a refactor, a caller
    passing the wrong argument, a graphiti upgrade that changes the statement
    source — the detector goes loud instead of quietly certifying the fleet
    green.
    """
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))

    degenerate = not expected
    if degenerate:
        logger.warning(
            'summarize_index_health: expected index set is EMPTY'
            ' (actual_total=%d); forcing healthy=False rather than reporting a'
            ' vacuous all-clear',
            len(actual),
        )

    return {
        'healthy': missing == [] and not degenerate,
        'missing': missing,
        'unexpected': unexpected,
        'expected_total': len(expected),
        'actual_total': len(actual),
    }
