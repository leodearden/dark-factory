"""Write-time triage for ``add_memory`` — redirect, don't reject (task 3127).

The successor to :mod:`fused_memory.server.near_duplicate_guard`. That guard
answers a near-duplicate write by REJECTING it: the tool returns a structured
soft-block and the submitted content is gone unless the agent re-submits with
an override. Triage answers the same question by REDIRECTING: a restatement is
attached as a SIGHTING child of the memory it restates, so the text survives,
the rediscovery is counted, and nothing the agent wrote is lost.

Contract C1 is absolute and every function here is written to it:

* **never lose content** — every path ends in a write, either standalone or
  as a child of a canonical;
* **never block a write** — no path returns an error and no path raises;
* **never edit a canonical** — triage issues no ``update_memory`` and no
  ``delete_memory``, so a wrong attach is always re-parentable.

Fail-open (INV-4) is how C1 survives contact with a broken dependency: any
exception anywhere in the pipeline yields ``stored``, plus a counted and
eventually escalated fail-open event. Silent degradation is the thing being
prevented, so the fallback is loud in the logs and quiet in the response.

SEAM NOTE (INV-5). Candidate retrieval goes through ``MemoryService.search``
and nothing else. Task 3111 lands topic-anchored recall AT that seam; routing
this leaf's retrieval through it means the improvement arrives here for free,
whereas a topic-aware retrieval built inside this module would be a second
implementation to keep in sync. 3111 is confirmed NOT landed as of this leaf's
base, which is the expected state — plain cross-category search is the correct
retrieval today.

The pure/impure split mirrors ``near_duplicate_guard``: pure synchronous
selectors and defensive ``getattr``-at-every-hop config resolvers, with the
one async retrieval helper and the async orchestrator kept separate from them.
"""

from __future__ import annotations

from typing import Any

# --- ack contract (INV-1: one home for the wire names) ---------------------
#
# Leaf gamma (the judge) and the tool-level boundary tests IMPORT these rather
# than restating the strings, so a rename has exactly one place to fail
# instead of drifting between the tool, the judge, and the tests asserting on
# both.

#: Key on the add_memory ack carrying what triage did with the write.
ROUTED_KEY = 'routed'

#: Key on the add_memory ack naming the memory a write was attached to.
#: PRESENT ONLY on an attach outcome — omitted entirely, never emitted as
#: null, for a plain ``stored``. An absent key is an unambiguous signal; a
#: null is a value the reader then has to disambiguate.
CANONICAL_ID_KEY = 'canonical_id'

#: The write was stored as a new standalone memory. Also the fail-open
#: outcome and the deliberate-stub outcome — from the caller's side those are
#: indistinguishable from a genuine "nothing matched", which is the point.
OUTCOME_STORED = 'stored'

#: The write restates an existing memory and was attached to it as a sighting.
OUTCOME_RESTATED = 'restated'

#: The write adds to an existing memory and was attached as an amendment.
#: Produced by leaf gamma's judge, never by this leaf's stub.
OUTCOME_AMENDED = 'amended'

#: The write contradicts an existing memory and was attached as a contested
#: child. Produced by leaf gamma's judge, never by this leaf's stub.
OUTCOME_CONTESTED = 'contested'

#: The closed set of ack outcomes. Closed deliberately (D3): the judge's
#: output is a fixed vocabulary, so an unrecognised value is a bug rather
#: than an extension point.
TRIAGE_OUTCOMES: frozenset[str] = frozenset({
    OUTCOME_STORED,
    OUTCOME_RESTATED,
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
})

# --- config defaults --------------------------------------------------------

#: Default when config is absent/partial/non-bool. OFF is the safe direction
#: AND the deliberate one: an unreadable config must leave today's guard
#: behaviour in place, never silently enable a path whose judge is a stub.
_DEFAULT_WRITE_TRIAGE_ENABLED = False

#: Default retrieval width. Measured same-category recall on the live corpus:
#: 26.1% @5, 43.9% @10, 69.4% @20, 88.5% @50. This is a RANK property, not a
#: threshold — a candidate that never enters the result set cannot be scored
#: by any band — so it caps everything t_high and t_low can achieve. The one
#: number this must NOT be is the retired near-dup guard's hardcoded
#: ``limit=5``, at which three quarters of the duplicates triage exists to
#: catch are invisible to it. Counted as a TOTAL across the three
#: Mem0-primary categories, so effective per-category depth is lower.
_DEFAULT_CANDIDATE_K = 20


def _write_triage_attr(memory_service: Any, attr: str) -> Any:
    """Navigate ``memory_service.config.write_triage.<attr>`` defensively.

    ``getattr`` at every hop, mirroring
    ``near_duplicate_guard._reconciliation_attr``: a missing service, a
    missing config, a missing section and a missing leaf all read as ``None``
    rather than raising into a write path that must never fail.

    Read LIVE on every call, never captured at import or construction. That is
    what makes the green-tier ``RELOADABLE_FIELDS`` registration of these
    leaves real: ``apply_reload`` mutates the shared config object in place,
    and a captured value cannot observe an in-place mutation — which would
    leave the kill switch restart-only while sitting in the allowlist as if it
    were hot.
    """
    config = getattr(memory_service, 'config', None)
    write_triage = getattr(config, 'write_triage', None)
    return getattr(write_triage, attr, None)


def resolve_write_triage_enabled(memory_service: Any) -> bool:
    """Read the staged-rollout kill switch from *memory_service*'s config.

    Returns :data:`_DEFAULT_WRITE_TRIAGE_ENABLED` (False) unless the leaf is a
    real ``bool``. ``isinstance(bool)`` only, deliberately: a truthy ``1``
    from YAML, or the Mock an unspecced test double auto-generates for any
    attribute, would otherwise enable triage by accident — and enabling it
    while the judge is still a stub is the wrong direction to fail in.
    """
    value = _write_triage_attr(memory_service, 'enabled')
    if isinstance(value, bool):
        return value
    return _DEFAULT_WRITE_TRIAGE_ENABLED


def resolve_candidate_k(memory_service: Any) -> int:
    """Read the retrieval width from *memory_service*'s config.

    Returns :data:`_DEFAULT_CANDIDATE_K` unless the leaf is a real positive
    ``int``. ``bool`` is excluded despite being an ``int`` subclass — a
    ``candidate_k=True`` would resolve to a width of 1, which is triage with
    almost no recall and no error anywhere to explain it. A non-positive width
    is refused for the same reason: it would read as "no comparable candidate"
    on every write and route everything to ``stored``, i.e. triage silently
    disabled. The schema bounds this ``ge=1``, so a 0 can only arrive from a
    hand-built config object or a partially-applied reload.
    """
    value = _write_triage_attr(memory_service, 'candidate_k')
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return _DEFAULT_CANDIDATE_K


def resolve_bands(memory_service: Any) -> tuple[float | None, float | None]:
    """Read ``(t_high, t_low)`` from *memory_service*'s config.

    Each is a real ``float``/``int`` coerced to ``float``, or ``None``.
    ``None`` is a FIRST-CLASS reading, not an error — the landed schema uses
    it to mean UNCALIBRATED, and leaf alpha measured a corpus on which no
    deterministic band is derivable at all. Both readings must survive the
    resolver for :func:`decide_band` to act on them.

    ``bool`` is excluded despite being an ``int`` subclass: ``t_high=True``
    would coerce to a cutoff of 1.0 (silently emptying the deterministic
    band) and ``t_high=False`` to 0.0 (making every candidate a restatement).
    Neither is a measurement, and both fail invisibly.

    Read live per call, same green-tier reload requirement as the flag: a
    re-calibration must take effect on a running server without a restart.
    """
    return (
        _coerce_band(_write_triage_attr(memory_service, 't_high')),
        _coerce_band(_write_triage_attr(memory_service, 't_low')),
    )


def _coerce_band(value: Any) -> float | None:
    """One band edge as a float, or ``None`` for anything that is not a number."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None
