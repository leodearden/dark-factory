"""scripts/legibility/trickle_state.py — the nightly trickle run-state
record: WHY a night produced nothing, written by the pipeline itself.

See plans/confusion-reduction-prd.md §6 decision 7. That decision bans
inferring pipeline health from the REPO's contents (git history, codebook
mtime) because a legitimately quiet night commits nothing, so an EXTERNAL
observer cannot tell "produced nothing because there was nothing" from
"produced nothing because it is broken". That information is not missing
from the world — only from the observer. It already exists INSIDE the
pipeline, as :class:`legibility.sampling.SampleResult`'s conservation
invariant. So this module has the pipeline RECORD its own reason in
machine-readable form; :mod:`check_trickle_progress` reads it. A quiet
night is recorded AS quiet and never alarms, so decision 7's stated
rationale is satisfied rather than circumvented.

Division of labour between the two probes:

- ``check_trickle_liveness.sh``  — did the UNIT RUN? (systemd unit state)
- ``check_trickle_progress.py``  — did SIGNAL FLOW? (this state file)

STDLIB-ONLY IMPORTS ARE A HARD CONSTRAINT, NOT A STYLE PREFERENCE.
``check_trickle_progress.py`` imports this module and, when bound as a
``before_done`` predicate, is EXEC'd directly by
``deterministic_runner._default_run_script``
(``asyncio.create_subprocess_exec(script, *args)``) — no ``uv run``
wrapper, no project venv, bare ``#!/usr/bin/env python3``. Meanwhile
``nightly.py`` and ``legibility.config`` both pull in PyYAML. A stray
third-party import HERE would break the predicate at DISPATCH time, not
at test time — i.e. it would fail in exactly the silent way this module
exists to prevent. Same reason ``DEFAULT_MAX_BARREN_RUNS`` is a module
constant rather than a ``legibility.yaml`` field: reading that config
would drag pydantic + PyYAML onto the predicate path.

Keeping the classifier in ONE module (rather than inlining it in
``nightly.py``) also stops nightly's escalation and the probe's verdict
from drifting apart (INV-5, no lockstep duplication) — the exact drift
that let a suppressed night read like a quiet one for 14 nights
(2026-07-16..29).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outcome vocabulary
# ---------------------------------------------------------------------------

OUTCOME_PRODUCTIVE = 'productive'
"""Digests were built: ``selected > 0``."""

OUTCOME_QUIET = 'quiet'
"""Nothing was digested and nothing reached the sampling/budget stage — a
legitimately quiet or dormant night. NEVER an alarm."""

OUTCOME_BARREN = 'barren'
"""Nothing was digested even though real, distinct, non-duplicate signal
DID reach the sampling/budget stage. This is the absence mode that looks
identical to a quiet night from outside the pipeline."""


# ---------------------------------------------------------------------------
# classify_run — the three-valued absence classifier
# ---------------------------------------------------------------------------

def classify_run(
    *,
    total_records: int,
    zero_signal_dropped: int,
    dedupe_collapsed: int,
    below_sampling_cut: int,
    budget_skipped: int,
    selected_count: int,
) -> str:
    """Classify one nightly trickle run as productive / barren / quiet.

    DERIVED FROM, not tuned against, :class:`sampling.SampleResult`'s
    conservation identity::

        total_records == zero_signal_dropped + dedupe_collapsed
                         + below_sampling_cut + budget_skipped
                         + len(selected)

    Three branches:

    1. ``selected_count > 0``                                -> productive.
       Digests were built. This deliberately INCLUDES a night that also
       skipped records on budget: a partially-truncated night is the byte
       budget working as designed, never an absence.
    2. ``(budget_skipped + below_sampling_cut) > 0``          -> barren.
       Both are doors that only records with real, distinct,
       non-duplicate signal can leave by, so reaching this branch proves
       genuine signal existed and NOTHING was digested. The two doors are
       kept separate in the recorded counters because they have DIFFERENT
       remedies (``budgets.max_daily_digest_bytes`` vs
       ``sampling.top_fraction``/``per_stratum_min`` — SampleResult's own
       docstring is explicit that conflating them is wrong), but for the
       PRESENCE question they are one signal: real signal in, nothing out.
    3. otherwise                                             -> quiet.

    WHY BRANCH 3 IS PROVABLY SAFE — the no-false-alarm guarantee. Reaching
    the ``else`` means ``selected_count == 0`` and both cut counters are
    0, so by the identity ``total_records == zero_signal_dropped +
    dedupe_collapsed``: every enumerated record left by the zero-signal or
    dedupe door, or nothing was enumerated at all. That is EXACTLY the
    "genuinely quiet night" PRD decision 7 protects, so a quiet or dormant
    project can never be classified barren. This is a proof from the
    invariant, not a threshold someone picked — which is what lets a
    progress probe exist without re-opening decision 7's false-alarm
    objection.

    ``total_records`` and ``zero_signal_dropped`` are accepted (and
    RECORDED by :func:`record_run`) but deliberately NOT consulted by the
    branch logic — they are what an operator reads to understand the shape
    of a night, and the identity above is what makes the classification
    auditable after the fact. Do not "simplify" them out of the signature
    or out of the recorded state.
    """
    if selected_count > 0:
        return OUTCOME_PRODUCTIVE
    if (budget_skipped + below_sampling_cut) > 0:
        return OUTCOME_BARREN
    return OUTCOME_QUIET
