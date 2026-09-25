"""The probe protocol a recon stage follows before it says anything about
Graphiti's health, held as data so the stage prompts render it rather than
restate it (task 4644). It answers run cd53b227, whose Stage 2 turned one
negative probe into "no persistent Graphiti problem" in a cycle where Stage 3
then observed the fault; the full probe log is in Mem0 record
da16ef9d-b9f1-4be4-af5e-0299ff9a65f2.

## Why these numbers

:data:`PROBE_LIMIT_LADDER` comes from that probe log. ``8`` is the fan-out
both positive sightings used, so it is the size with demonstrated power to
fire and the floor the widest probe may not drop below
(:data:`HIGH_FANOUT_LIMIT_FLOOR`). ``3`` is the limit whose lone probe
produced the false negative; it stays because controlling a variable means
spanning it, including the value already known to miss. ``15`` extends past
8, matching the widest rung a later three-probe set tried.

:data:`NEGATIVE_SET_VERDICT_TEMPLATE` is a count, not a verdict, because
``server/tools.py::search`` reports ``degraded``/``failed_stores`` only when a
store has already failed: a clean probe carries no health signal, so no number
of negatives is evidence of absence. That later three-probe set was 0 of 3,
one of them replaying the exact query and limit that had fired.

The two stat keys are published together for the reason
``reconciliation/stage_stats.py`` publishes ``graphiti_writes_queued`` beside
``writes_dead_lettered``: a bare ``0`` reads as a no-op, and the denominator
makes it read as "0 of N".
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

# The denominator pair, always published together.
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
    """Render the probe-protocol section for one stage: a shared body (the
    verdict rule and the cross-stage caveat) plus one capability clause, the
    shape of :func:`prompts.render_finding_provenance_section`.

    Args:
        runs_probes: True for the stage that runs the ladder and reports its
            counters; False for a read-only stage, whose clause tells it not
            to.
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
