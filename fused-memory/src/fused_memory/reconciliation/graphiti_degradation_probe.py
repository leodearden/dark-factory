"""The probe protocol a recon stage follows before it says anything about
Graphiti's health (task 4644).

## The incident

Run cd53b227-18ac-4432-b22f-2b5ac0913cf2 (2026-08-23) ran exactly ONE
mixed-store probe — query "FalkorDB index provisioning and graph read
latency", ``limit=3`` — reported it as a bare negative, and concluded both
that the degradation "did not reproduce this cycle" and that no persistent
Graphiti problem existed. Stage 3 of the SAME run then reproduced it on a
``limit=8`` query ("recent completed work session summary and decisions"):
``degraded: true``, ``failed_stores: ['graphiti']``. The run probed negative
at Stage 2 and observed positive at Stage 3, inside one cycle.

Raising N alone would not have saved it. Run
45b9a919-ac25-431b-bb0f-39c28b1d1906 then ran three probes (``limit`` 15, 3
and 8 — the last replaying the exact Stage-3 query and limit that HAD fired)
and 0 of 3 reproduced. A ladder buys coverage, never clearance.

## Why a negative set can never clear the fault

``server/tools.py::search`` surfaces ``degraded`` / ``failed_stores`` under
FAULT-ONLY LOUDNESS — the keys appear only when a store actually failed, so a
healthy probe returns no health key at all. There is no positive "graphiti is
fine" observation available to any stage. Every negative probe is an absence
of evidence by construction, and no number of them becomes evidence of
absence. That is the whole reason :data:`NEGATIVE_SET_VERDICT_TEMPLATE` is
worded as a count rather than a verdict.

## Why these numbers

:data:`PROBE_LIMIT_LADDER` is derived from the recorded probe log, not chosen
to satisfy a requirement's wording. ``8`` is the fan-out BOTH positive
sightings used, so it is the size with demonstrated power to fire. ``3`` is
the limit whose lone probe produced the cd53b227 false negative — it stays in
the ladder because the defect being fixed is that result size was a silently
avoided variable rather than a controlled one, and controlling a variable
means spanning it, including the value already known to miss. ``15`` extends
past 8, matching the widest rung run 45b9a919 tried.

Reporting :data:`GRAPHITI_DEGRADATION_REPRODUCED_STAT_KEY` alongside
:data:`GRAPHITI_MIXED_STORE_PROBES_RUN_STAT_KEY` follows the same convention as
``graphiti_writes_queued`` beside ``writes_dead_lettered`` in
``reconciliation/stage_stats.py``: a bare ``0`` reads as a no-op when it is
actually an unresolved question, so the denominator is published with it and
the pair is self-explaining.

## Scope

This is a defect in reconciliation's own probe methodology. It is NOT part of
task 3708 (FalkorDB index provisioning), which shares only the subject matter.
The full corrected probe log — both positive sightings and all four negative
probes — lives in Mem0 record da16ef9d-b9f1-4be4-af5e-0299ff9a65f2.
"""

from __future__ import annotations

__all__ = [
    'GRAPHITI_DEGRADATION_REPRODUCED_STAT_KEY',
    'GRAPHITI_MIXED_STORE_PROBES_RUN_STAT_KEY',
    'HIGH_FANOUT_LIMIT_FLOOR',
    'MIN_PROBES_PER_CYCLE',
    'NEGATIVE_SET_VERDICT_TEMPLATE',
    'PROBE_LIMIT_LADDER',
]

# The denominator pair. Defined once here and reaching the stage prompts only
# through the renderer, so the name an operator reads out of a cycle report's
# `stats` is the same object the stage was told to emit.
GRAPHITI_MIXED_STORE_PROBES_RUN_STAT_KEY = 'graphiti_mixed_store_probes_run'
GRAPHITI_DEGRADATION_REPRODUCED_STAT_KEY = 'graphiti_degradation_reproduced'

# The `limit` values one cycle's probes sweep — see "Why these numbers".
PROBE_LIMIT_LADDER: tuple[int, ...] = (3, 8, 15)

MIN_PROBES_PER_CYCLE = 3

HIGH_FANOUT_LIMIT_FLOOR = 8

# The only conclusion a negative probe set licenses (task 4644 requirement 4).
NEGATIVE_SET_VERDICT_TEMPLATE = (
    '0 of {n} probes reproduced; the fault is intermittent and load-dependent, '
    'so a negative set does not clear it.'
)
