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
    'NEGATIVE_SET_VERDICT',
    'NEGATIVE_SET_VERDICT_TEMPLATE',
    'PROBE_LIMIT_LADDER',
    'render_graphiti_degradation_probe_section',
]

# The denominator pair. Defined once here and reaching the stage prompts only
# through the renderer, so the name an operator reads out of a cycle report's
# `stats` is the same object the stage was told to emit.
GRAPHITI_MIXED_STORE_PROBES_RUN_STAT_KEY = 'graphiti_mixed_store_probes_run'
GRAPHITI_DEGRADATION_REPRODUCED_STAT_KEY = 'graphiti_degradation_reproduced'

# The `limit` values one cycle's probes sweep — see "Why these numbers".
PROBE_LIMIT_LADDER: tuple[int, ...] = (3, 8, 15)

MIN_PROBES_PER_CYCLE = len(PROBE_LIMIT_LADDER)

HIGH_FANOUT_LIMIT_FLOOR = 8

# The only conclusion a negative probe set licenses (task 4644 requirement 4).
NEGATIVE_SET_VERDICT_TEMPLATE = (
    '0 of {n} probes reproduced; the fault is intermittent and load-dependent, '
    'so a negative set does not clear it.'
)
# The wording as the stage prompts and the premise-lint rejection show it. The
# count is left to the stage: "at least" lets it run more probes than rungs.
NEGATIVE_SET_VERDICT = NEGATIVE_SET_VERDICT_TEMPLATE.format(n='N')


def render_graphiti_degradation_probe_section(*, runs_probes: bool) -> str:
    """Render the probe-protocol section for one stage.

    Follows the :func:`prompts.render_finding_provenance_section` /
    :func:`prompts.render_escalation_boundary_note` precedent: ONE shared body
    plus a single clause selected by a keyword-only capability flag, so the
    half both stages need exists exactly once (INV-5 ``no-lockstep-duplication``).

    The split is a capability split, not a style choice. The verdict rule and
    the cross-stage caveat govern any stage that says anything about Graphiti's
    health; the ladder and the two counters belong only to the stage that
    actually runs probes and reports ``stats``. Naming a counter to a stage is
    a live instruction to emit it, so handing the read-only stage the probe
    text would order it to report numbers it has no way to produce.

    Every limit and both counter names are interpolated from this module's
    constants. Nothing here may be hand-typed: the number an operator reads out
    of a cycle report and the number the stage was told to emit have to be the
    same object, or the protocol drifts silently.

    Args:
        runs_probes: True for the stage that runs the ladder and owns the
            counters; False for a read-only stage, which inherits the verdict
            rule and the caveat alone.

    Returns:
        The shared body plus the matching capability clause, interpolated into
        a stage's system prompt at build time.
    """
    ladder_phrase = ', '.join(str(limit) for limit in PROBE_LIMIT_LADDER)

    if runs_probes:
        capability_clause = (
            'You run this probe. Each cycle, run at least '
            f'{MIN_PROBES_PER_CYCLE} mixed-store probes — one at each of these '
            f'`search` limits: {ladder_phrase} — and vary the query text across '
            'them, so that fan-out rather than a single cached query is what '
            'differs between probes. Result size is the confound: the lone probe '
            'behind the false conclusion was the narrowest one available, while '
            'both recorded reproductions came from wider queries — so never let '
            f'your widest probe fall below `limit` {HIGH_FANOUT_LIMIT_FLOOR}, the '
            'fan-out they used.\n\n'
            'Report BOTH counters in your structured `stats`, always together '
            'and never one without the other:\n'
            f'- `{GRAPHITI_MIXED_STORE_PROBES_RUN_STAT_KEY}` — how many probes '
            'you ran.\n'
            f'- `{GRAPHITI_DEGRADATION_REPRODUCED_STAT_KEY}` — how many of them '
            'reproduced the degradation.\n'
            'Published alone, a reproduced count of zero reads as a no-op. '
            'Published beside its denominator it reads as "0 of N", which is '
            'what it actually means.'
        )
    else:
        capability_clause = (
            'You do not run this probe and do not report its counters — running '
            'the ladder belongs to the stage that owns those stats. Report the '
            'store health you actually observe in your own searches, and do not '
            "treat another stage's negative probe set as having cleared "
            'anything.'
        )

    return (
        '## Graphiti Mixed-Store Degradation Probe\n'
        'The Graphiti degradation this probe looks for is INTERMITTENT and '
        'load-dependent: it fires on some queries and not others, within one '
        'cycle, against the same store.\n\n'
        'There is no positive observation of health available to you. `search` '
        'reports `degraded` and `failed_stores` only WHEN a store has already '
        'failed, so a clean probe returns no health key at all — its silence is '
        'the absence of a fault report, not a report of no fault. A negative '
        'probe set is therefore an absence of evidence by construction, and no '
        'number of negatives converts it into evidence of absence.\n\n'
        'A probe set in which nothing reproduced licenses exactly one '
        'conclusion, and this is the permitted wording, with N the number of '
        'probes actually run:\n'
        f'    {NEGATIVE_SET_VERDICT}\n'
        'It does NOT license "the degradation did not reproduce", "no '
        'persistent Graphiti problem", or any other claim that the fault is '
        'absent, cleared, resolved or historical.\n\n'
        "Neither stage's probe result binds the other. In run cd53b227 one "
        'stage probed negative and a later stage of the SAME cycle observed '
        "`degraded: true` with `failed_stores: ['graphiti']` — so a stage that "
        "observes the fault must not defer to a sibling's negative, and no "
        "stage may cite another stage's negative as corroboration.\n\n"
        + capability_clause
    )
