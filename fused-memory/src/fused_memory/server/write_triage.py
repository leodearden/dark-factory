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
and nothing else. A topic-aware — or any other — retrieval built inside this
module would be a second implementation to keep in sync forever, so the seam
discipline is binding and unconditional.

Going THROUGH the seam is not the same as taking everything it offers.
Task 3111's topic-anchored recall HAS landed at that seam
(``MemoryService.search(anchor_topics=...)``, defaulting to ``True``), and
this module explicitly opts OUT of it: the pin PROMOTES rather than adds, so
on a consolidated topic it spends candidate slots on records triage can never
route to. That is an agent-facing read improvement and a machine-consumer
regression; see :func:`retrieve_candidates` for the full reasoning. Use the
seam, and pass the flags a candidate-set consumer needs.

THE JUDGE SLOT IS FILLED. ``server/write_triage_judge.py`` (leaf gamma) is
the real middle-band judge, and ``server/tools.py`` is the SINGLE place that
wires it in as ``triage_write(..., judge=judge_write)``. The import direction
is one-way and load-bearing: that module imports this one for the ``OUTCOME_*``
constants, so this module must NEVER import it — making the real judge
``triage_write``'s own default would be a circular import. ``_stub_judge``
remains the default here for direct callers and for the contract tests that
inject their own fakes.

The pure/impure split mirrors ``near_duplicate_guard``: pure synchronous
selectors and defensive ``getattr``-at-every-hop config resolvers, with the
one async retrieval helper and the async orchestrator kept separate from them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shared.storm_counter import StormCounter

from fused_memory.models.enums import MEM0_PRIMARY
from fused_memory.server.grouped_read import (
    CHILD_KINDS,
    CONTESTED_METADATA_KEY,
    PARENT_ID_KEY,
)

# The per-store-cosine reader, IMPORTED rather than re-implemented (INV-5) —
# the same treatment PARENT_ID_KEY and CHILD_KINDS get above. This module
# retires ``near_duplicate_guard``, so a copy here would have been the obvious
# shortcut; it is also exactly how task 3658's field move (relevance_score →
# metadata['store_score']) would go wrong a second time. One copy updated and
# the other left behind does not raise — it scores every candidate as
# uncomparable, which routes every write to ``stored`` and is indistinguishable
# from a genuinely novel corpus. If that module is ever deleted rather than
# left dormant, this import fails LOUDLY at import time, which is the right
# way to be told to hoist the helper. (Hoisting it into a shared home is the
# better end state and is deliberately NOT done here: near_duplicate_guard.py
# is outside this task's lock set.)
from fused_memory.server.near_duplicate_guard import _cosine_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

    from fused_memory.models.memory import MemoryResult
    from fused_memory.services.memory_service import SearchResults

# Defensive import of the optional ``escalation`` workspace package, copied
# from markup_tripwire.py (which took it from middleware/candidate_key_
# escalation.py): when it is missing — minimal CI envs, deployments that never
# installed it — the storm escalation degrades to a logged no-op. This module
# sits on the MCP write path, and by the time escalation is attempted the
# write has ALREADY been stored, so nothing here may change its outcome.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)

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

#: Key on the add_memory ack naming the escalation filed when a fail-open
#: BURST crossed the threshold on this call. Present only on that call —
#: rare by construction, and additive like the other two.
FAIL_OPEN_ESCALATION_ID_KEY = 'triage_fail_open_escalation_id'

#: The metadata keys an ATTACH outcome OVERWRITES when it reroutes a write
#: into a child record. This is the definition the force-store predicate
#: below is derived from, so the two cannot drift: every key the attach
#: writes is a key a caller must be allowed to keep.
#:
#: THREE keys, not two, since task 3128 wired the ``contested`` outcome: that
#: child is an amendment PLUS ``CONTESTED_METADATA_KEY``, so a caller who set
#: the contested flag themselves must force-store like any other. Note the
#: outcomes no longer write the SAME keys — only ``contested`` writes the
#: third — so this set is the UNION over outcomes, which is what the force
#: store has to defend. The gate suite pins it that way, sweeping every attach
#: outcome and unioning what each persisted.
#:
#: ``'kind'`` is spelled as a literal because ``grouped_read`` exports the
#: kind VALUES (``AMENDMENT_KIND``/``SIGHTING_KIND``) but no constant for the
#: key itself, and that module is outside this task's scope to extend.
#: ``CONTESTED_METADATA_KEY`` by contrast IS exported, so it is imported —
#: ``grouped_read`` owns the read-side predicate that has to recognise what
#: the write side stamps, and two spellings of that key would produce children
#: flagged in a way nothing reads.
#: ``tests/server/test_add_memory_write_triage_gate.py`` pins this set against
#: the keys ``tools.py`` actually writes, so a FOURTH key added to the attach
#: without widening this set fails there rather than silently.
ATTACH_OWNED_KEYS: frozenset[str] = frozenset({
    PARENT_ID_KEY, 'kind', CONTESTED_METADATA_KEY,
})

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

#: INTERNAL routing sentinel: the write landed in the middle band and must be
#: adjudicated. Deliberately NOT a member of :data:`TRIAGE_OUTCOMES` — the ack
#: says what triage DID with a write, and "sent it to a judge" is not an answer
#: to that. The judge's own verdict is what a caller eventually sees.
OUTCOME_JUDGE = 'judge'

#: The closed set of ack outcomes. Closed deliberately (D3): the judge's
#: output is a fixed vocabulary, so an unrecognised value is a bug rather
#: than an extension point.
TRIAGE_OUTCOMES: frozenset[str] = frozenset({
    OUTCOME_STORED,
    OUTCOME_RESTATED,
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
})


def attach_write_landed(result: Any) -> bool:
    """Did the child write for an ATTACH outcome actually persist?

    The ack may only claim :data:`CANONICAL_ID_KEY` for a link that exists, so
    the write path needs this BEFORE building the ack.

    A raise is only half the failure surface, and the smaller half — the same
    asymmetry :func:`triage_write` handles for retrieval.
    ``MemoryService.add_memory`` does not raise when a store fails: it catches
    the Graphiti/Mem0 exception into ``_graphiti_error``/``_mem0_error``, folds
    it into ``message``, and returns an ordinary ``AddMemoryResponse``. What it
    does NOT do on that path is return any ``memory_ids`` — the mem0 arm
    appends ids only after ``mem0.add`` resolves without raising — so an
    EXPLICITLY empty ``memory_ids`` is the honest "nothing landed" signal for
    the Mem0-primary categories triage covers. (An empty ``memory_ids`` from a
    non-raising mem0 call is the silent dedup/infer drop
    ``MemoryService.add_memory`` itself logs as anomalous; treating it as
    not-landed here is the same reading.)

    Answers ``True`` for anything that is not an explicitly empty sequence —
    a missing attribute, or the ``Mock`` an unspecced test double
    auto-generates. Ambiguity resolves toward "landed" on purpose: this
    predicate only ever DOWNGRADES an ack, and downgrading a real attach
    because the response shape was unreadable would invent a failure that did
    not happen. Production always returns a real ``AddMemoryResponse``, whose
    lists are real lists, so the check is exact where it matters.
    """
    memory_ids = getattr(result, 'memory_ids', None)
    return not (isinstance(memory_ids, list | tuple) and not memory_ids)


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


# --- pure band routing ------------------------------------------------------


@dataclass(frozen=True)
class BandDecision:
    """Which band a candidate set fell in, and the numbers that decided it.

    Frozen so a downstream stage cannot rewrite the routing after the fact.
    Carries the inputs alongside the verdict so the ack (and a log line, and
    leaf gamma's judge prompt) can quote WHY without recomputing anything —
    the same reason ``build_near_duplicate_block`` emits ``similarity`` beside
    ``threshold``: a decision the reader cannot check reads as a malfunction.
    """

    #: One of the outcome constants, or the internal :data:`OUTCOME_JUDGE`.
    outcome: str
    #: The best comparable candidate's id, or ``None`` when nothing compared.
    canonical_id: str | None
    #: That candidate's per-store cosine, or ``None``.
    similarity: float | None
    #: The band edges this decision was made against, echoed as read.
    t_high: float | None
    t_low: float | None


def declares_attach_keys(metadata: Any) -> bool:
    """True when the CALLER's own *metadata* already sets a key an attach owns.

    ``parent_id`` and ``kind`` are first-class caller-supplied Tier-A metadata
    keys — ``memory_metadata`` validates ``kind`` against ``KIND_REGISTRY``
    and ``parent_id`` for UUID shape, and ``MemoryService`` resolves parent
    liveness — so an agent CAN and does set both through ``add_memory``. An
    ATTACH outcome overwrites them (see :data:`ATTACH_OWNED_KEYS`), so
    whatever the caller put there is destroyed with no log line and no
    fail-open count. Under a contract whose first clause is *never lose
    content*, a write that would cost the caller its own metadata force-stores
    instead.

    Scoped to those keys and NOT to metadata generally: triage must still
    fire for the ordinary write that carries a ``source`` or a ``topic``,
    which is nearly all of them. The keys here are exactly the ones the attach
    clobbers, no wider.

    PRESENCE is the test, never truthiness — which matters most for the third
    key: a caller who explicitly wrote ``x_contested: False`` has made a
    claim about the record, and letting a contested attach flip it to ``True``
    would be the same silent overwrite in the one direction where the value
    reverses the record's meaning.

    ANY ``kind`` counts, not just ``CHILD_KINDS``. The narrower rule looked
    sufficient — an ``amendment`` demoted to a ``sighting`` is the loudest
    case, because ``grouped_read`` digests amendment TEXT while sightings are
    only COUNTED — but ``kind`` is an open free-text vocabulary (the census
    measured 329 distinct values, 242 of them singletons), so a
    ``cycle_summary`` or ``completion_note`` write is just as much a
    declaration of what the record IS. Relabelling one ``sighting`` erases
    that classification AND folds the record into some canonical's sighting
    count. The coverage this costs is small and measured: the census found
    95.0% of records (47,150 of 49,628) carry no ``kind`` at all, so triage
    still sees ~19 of every 20 writes.

    Likewise ANY ``parent_id``, well-formed or not: a malformed link is a
    caller bug for ``memory_metadata`` validation to report, and quietly
    converting it into a triage attach would hide it. Contrast
    ``grouped_read._parent_id_in_meta``, which requires both keys well-formed
    because it answers a different question — "does this record group?" rather
    than "did the caller set something an attach would overwrite?".
    """
    if not isinstance(metadata, dict):
        return False
    return any(key in metadata for key in ATTACH_OWNED_KEYS)


def _canonical_id_of(result: MemoryResult) -> str:
    """The id a write should attach to for *result* — hoisting a CHILD.

    A sighting or amendment record is itself attached to something, and
    ``grouped_read`` resolves exactly ONE level of parentage. Attaching to a
    child would therefore create a GRANDCHILD that can never fold under the
    true canonical — a child that exists but never groups, which reads as
    content loss without being one. So when *result* is a child, this returns
    its ``parent_id``; otherwise it returns the record's own id.

    Why hoist rather than EXCLUDE children from the candidate set: a sighting
    child holds the restatement text VERBATIM, which makes it the strongest
    available evidence that this exact restatement was seen before, and the
    likeliest max-cosine hit on the second restatement of a fact. Dropping it
    would lose that signal entirely and route a genuine restatement to
    ``stored`` whenever the canonical happens to be worded differently.

    The other half of the trade, stated honestly: hoisting can attach to a
    parent id whose record was since deleted. That is at worst the
    pre-existing dangling-parent condition ``grouped_read`` already tolerates,
    whereas attaching to a child creates a guaranteed-broken grandchild that
    one-level grouping can NEVER fold.

    The child rule mirrors ``grouped_read._parent_id_in_meta`` rule-for-rule —
    a child ``kind`` AND a non-empty ``str`` ``parent_id``, with a malformed
    parent link falling back to the record's own id — and both the kind set
    and the metadata key are IMPORTED from that module rather than spelled as
    literals (INV-5), because a drift between the write side and the read side
    is exactly what produces an unfoldable child.
    """
    meta = result.metadata or {}
    if meta.get('kind') in CHILD_KINDS:
        parent_id = meta.get(PARENT_ID_KEY)
        if isinstance(parent_id, str) and parent_id:
            return parent_id
    return result.id


def decide_band(
    results: list[MemoryResult],
    *,
    t_high: float | None,
    t_low: float | None,
) -> BandDecision:
    """Route a write by the MAXIMUM per-store cosine among *results*.

    ``s >= t_high`` is a deterministic :data:`OUTCOME_RESTATED` with no judge
    call; ``t_low <= s < t_high`` routes to :data:`OUTCOME_JUDGE`; anything
    below ``t_low``, and an empty or wholly uncomparable candidate set, is
    :data:`OUTCOME_STORED`. Both edges are INCLUSIVE, matching how the
    calibrator derives them: each is an order statistic of measured pairs, so
    the boundary value is itself an observed duplicate.

    The score read is the COSINE from ``metadata['store_score']``, NOT
    ``relevance_score``. Since task 3658 the latter is an ordinal RRF value —
    a single-store rank-1 hit scores 1/(60+1) ~ 0.0164 — which would never
    clear a cosine band, silently disabling triage for every input, and would
    pick the wrong canonical even where it did fire. Same warning
    ``near_duplicate_guard`` carries on ``find_near_duplicate_memory``.

    A candidate with no numeric cosine can never qualify at ANY threshold. A
    missing cosine means NOT COMPARABLE, which is not the same as a measured
    similarity of 0.0 that a low floor might clear.

    Candidates are NOT filtered by category. That is the point of this leaf:
    the retired guard restricted matching to the write's own category, and
    reify esc-5547/esc-5560 were both cross-category duplicates it therefore
    could not see.

    Two boundary-of-configuration readings, both of which return a decision
    rather than raising:

    ``t_low is None`` — UNCALIBRATED. Fail open to ``stored``, matching the
    landed schema's own reading of ``None``. With no measured lower edge there
    is no evidence that any candidate is a duplicate, and inventing a floor
    would attach real writes to unrelated canonicals.

    ``t_high is None`` — an EMPTY DETERMINISTIC BAND, and a FIRST-CLASS
    configuration rather than a broken one. ``calibrate_write_triage.py``
    derives ``t_high`` as "the smallest measured duplicate score that strictly
    exceeds every measured negative", an objective only satisfiable when the
    two distributions actually separate. On the esc-3181 cluster leaf alpha
    measured that they do not: the unrelated-pair MAX (0.8672) sits ABOVE the
    true-pair MAX (0.8532). So on such a corpus there is honestly no
    deterministic cutoff, everything at or above ``t_low`` routes to the
    judge, and NOTHING takes the autonomous ``restated`` path. Do not "fix"
    this into an assertion that a deterministic band always exists.

    ``canonical_id`` is a CANONICAL id, not merely "the winning result's id":
    a winner that is itself a sighting or amendment child is HOISTED to its
    parent by :func:`_canonical_id_of`, so the attach can never produce a
    grandchild that one-level grouping cannot fold. Only the attach TARGET
    moves — ``similarity`` stays the winner's own measured cosine, because
    that is the evidence actually observed against the submitted text.

    Pure and synchronous: does no I/O, takes no service, and raises nothing on
    empty input.
    """
    scored = [
        (cosine, r)
        for r in results
        if (cosine := _cosine_of(r)) is not None
    ]
    if not scored:
        return BandDecision(OUTCOME_STORED, None, None, t_high, t_low)

    similarity, best = max(scored, key=lambda pair: pair[0])

    # UNCALIBRATED: no measured lower edge, so nothing is evidenced as a
    # duplicate. Reported with no canonical, so the caller cannot accidentally
    # attach to a candidate this did not endorse.
    if t_low is None or similarity < t_low:
        return BandDecision(OUTCOME_STORED, None, None, t_high, t_low)

    # t_high None => empty deterministic band => the judge takes everything
    # from t_low up. See the docstring: this is measured, not a fallback.
    if t_high is not None and similarity >= t_high:
        return BandDecision(
            OUTCOME_RESTATED, _canonical_id_of(best), similarity, t_high, t_low,
        )

    return BandDecision(
        OUTCOME_JUDGE, _canonical_id_of(best), similarity, t_high, t_low,
    )


# --- candidate retrieval ----------------------------------------------------

#: The one store triage searches. Graphiti exposes no similarity scores, so a
#: Graphiti hit carries no ``store_score`` and can never be a comparable
#: candidate — asking for it would cost a fan-out that cannot affect the
#: decision.
_TRIAGE_STORES = ['mem0']


async def retrieve_candidates(
    memory_service: Any,
    content: str,
    project_id: str,
    k: int,
) -> SearchResults:
    """Fetch up to *k* comparable candidates for *content*, cross-category.

    Two things about this call are load-bearing.

    (1) The categories are ALL THREE Mem0-primary categories, deliberately.
    The retired near-dup guard filtered candidates to the WRITE's own category
    (``near_duplicate_guard.find_near_duplicate_memory``'s ``category``
    parameter), which was a measured blind spot: reify esc-5547 and esc-5560
    were both cross-category duplicates it structurally could not see. The set
    is imported from :data:`fused_memory.models.enums.MEM0_PRIMARY` rather
    than spelled out here, so there is one home for it (INV-5) and a category
    added later is triaged without an edit to this module.

    (2) Retrieval goes through the ``MemoryService.search`` SEAM and nowhere
    else, and that discipline is unchanged and still binding: do NOT build a
    topic-aware — or any other — retrieval here, because a second
    implementation is a second thing to keep in sync, and the seam note in
    this module's docstring exists to forbid exactly that.

    What was wrong was the INHERITANCE claim that used to sit here — that
    task 3111's topic-anchored recall would simply "arrive for free" once it
    landed. It landed, and it is not free for THIS consumer.
    ``anchor_topics=False`` is therefore passed explicitly below, with the
    reasoning at the call. Topic anchoring is an agent-facing READ
    improvement; a machine consumer that thresholds and post-filters its
    window must opt out or silently lose genuine candidates to displacement.

    COST, stated plainly as the retired guard's call site stated it: this is
    an embedding plus a vector round-trip on EVERY triaged write, so it is on
    the latency path of ordinary memory writes. ``write_triage.enabled: false``
    is the escape hatch, and it is the shipped default.

    Exceptions PROPAGATE. Fail-open is :func:`triage_write`'s job, and
    swallowing here would turn a wiring bug — a renamed kwarg, a changed
    signature — into a silent "no candidates found" that routes every write to
    ``stored`` with nothing to tell it apart from a genuinely novel corpus.

    BUT AN EXCEPTION IS ONLY HALF THE FAILURE SURFACE, and the smaller half.
    ``MemoryService.search`` does NOT raise on a store outage: it catches the
    store exception (or cancels the store on timeout), logs
    ``search.store_failed``, and returns an EMPTY ``SearchResults`` carrying
    ``degraded=True`` and ``failed_stores``. So a mem0 outage — the dominant
    retrieval failure — reaches this function as an ordinary empty result,
    not as a raise. The `except` arm in :func:`triage_write` therefore catches
    only wiring bugs (a renamed kwarg → ``TypeError``, a bad category/store
    string → ``ValueError``); the outage is caught by that function's explicit
    ``degraded`` check instead.

    This is why the ``SearchResults`` object is returned UN-TRANSFORMED — no
    slice, no comprehension, no ``sorted()``. Those all return a plain
    ``list`` and silently drop ``degraded``/``failed_stores``, which would
    re-hide the outage from the only code positioned to count it. The return
    type is annotated ``SearchResults`` rather than ``list[MemoryResult]`` so
    that invariant is checkable rather than merely argued: under the looser
    annotation a "cleanup" to a slice or a comprehension type-checks fine,
    and the resulting outage reads as healthy-and-empty.
    """
    return await memory_service.search(
        query=content,
        project_id=project_id,
        categories=sorted(category.value for category in MEM0_PRIMARY),
        stores=list(_TRIAGE_STORES),
        limit=k,
        # OPT OUT of topic-anchored recall (task 3111), for the same reason
        # the retired near-duplicate write guard's call site opts out. These
        # `k` slots are a CANDIDATE SET, not a presentation: the pin PROMOTES
        # rather than adds, so the window stays exactly `k` long and each
        # pinned canonical evicts the lowest-ranked genuine cosine hit. Worse,
        # a pinned canonical deliberately carries no metadata['store_score']
        # (services/topic_anchor.py), and `decide_band` drops every candidate
        # whose cosine is non-numeric -- so a pin can never qualify at ANY
        # threshold. Every pin is a slot spent on a record triage must ignore,
        # on exactly the consolidated topics where pins exist.
        anchor_topics=False,
    )


# --- fail-open + storm counter (INV-4) --------------------------------------

#: Fail-opens inside the window before the burst is escalated. One fail-open is
#: routine (a transient mem0 blip); a burst means every write in the window
#: silently fell back to pre-triage behaviour, which is the silent degradation
#: INV-4 exists to make visible. Module constants rather than config leaves,
#: like ``markup_tripwire``'s: an alarm an operator can tune down is an alarm
#: that gets tuned down.
_FAIL_OPEN_STORM_THRESHOLD = 10
_FAIL_OPEN_STORM_WINDOW_SECONDS = 3600.0

_FAIL_OPEN_HINT = (
    'add_memory write triage failed open repeatedly: every write in this '
    'window was stored WITHOUT triage, i.e. with the pre-triage behaviour. No '
    'write was lost or blocked (contract C1), but no restatement was detected '
    'either. Check candidate retrieval (MemoryService.search / mem0 '
    'reachability) and the judge. To stop triage deliberately rather than '
    'letting it degrade silently, set write_triage.enabled: false — it is '
    'green-tier hot-reloadable via reload_config.'
)


class TriageFailOpenCounter:
    """Rolling-window burst detector over triage fail-opens.

    A THIN ADAPTER over :class:`shared.storm_counter.StormCounter`, not a
    fifth copy of the append/prune/count/rate-limit body (INV-5). The
    construction idiom matches the live per-instance consumers in
    ``services/memory_service.py`` and ``shared/mcp_markup_middleware.py``:
    ``StormCounter(time_provider=…)``, with threshold and window passed to
    :meth:`StormCounter.record` per call rather than captured.

    (An earlier draft of this leaf named ``markup_tripwire.MarkupStormCounter``
    as the template. Task 4458 DELETED that class — do not go looking for it.)

    State is process-local and per-instance, and resets on restart: this
    catches a live burst, it is not durable statistics. One instance is built
    per ``create_mcp_server`` call so nothing bleeds between servers or between
    tests.
    """

    def __init__(self, time_provider: Callable[[], float] = time.time) -> None:
        self._counter = StormCounter(time_provider=time_provider)
        self._pending_storm: dict[str, Any] | None = None

    def record(self, *, project: str | None = None) -> dict[str, Any] | None:
        """Record one fail-open; return a storm summary iff a burst just fired.

        The summary renames ``StormCounter``'s generic ``labels`` to
        ``projects`` and adds a ``hint``, so what lands in the escalation
        detail reads as "which projects are degraded" rather than requiring
        the reader to know the label convention. A project that could not be
        resolved still counts toward the burst — there is simply nothing to
        name it against.
        """
        summary = self._counter.record(
            threshold=_FAIL_OPEN_STORM_THRESHOLD,
            window_seconds=_FAIL_OPEN_STORM_WINDOW_SECONDS,
            label=project,
        )
        if summary is None:
            return None
        storm = {
            'count': summary['count'],
            'threshold': summary['threshold'],
            'window_seconds': summary['window_seconds'],
            'projects': summary['labels'],
            'hint': _FAIL_OPEN_HINT,
        }
        # Stashed as well as returned. Most fail-opens are recorded DEEP inside
        # `triage_write`, whose return value is a routing decision — so the
        # party that can resolve a project_root and file the escalation (the
        # tool body, which holds the known_projects map) is not the party that
        # made the record() call. Without this the alarm would be built,
        # counted, and then dropped on the floor.
        self._pending_storm = storm
        return storm

    def drain_storm(self) -> dict[str, Any] | None:
        """Return and CLEAR the storm summary the last :meth:`record` produced.

        Draining rather than peeking: a storm is filed once per crossing, and
        leaving it in place would re-file it on every subsequent write until
        the window rolled — turning one alarm into a stream of them.
        """
        storm, self._pending_storm = self._pending_storm, None
        return storm

    def live_count(self) -> int:
        """Fail-opens currently inside the window, without recording one.

        Read-only, for the tool-level assertions and for an operator probe.
        """
        return self._counter.prune(_FAIL_OPEN_STORM_WINDOW_SECONDS)


async def _stub_judge(
    *,
    memory_service: Any,
    content: str,
    project_id: str,
    decision: BandDecision | None,
    candidates: Any = (),
) -> str:
    """Middle-band adjudication — leaf gamma's ``write_triage_judge`` replaces this.

    Returns :data:`OUTCOME_STORED` unconditionally. That is a DELIBERATE STUB,
    explicitly NOT a fail-open event: it must never be routed through
    :class:`TriageFailOpenCounter`. If it were, the first
    ``_FAIL_OPEN_STORM_THRESHOLD`` middle-band writes after the flag flip would
    guarantee a storm escalation describing an outage that is not happening,
    which trains an operator to ignore the alarm that exists to catch a real
    one.

    The real judge (D3) is synchronous-in-``add_memory``, fail-open,
    closed-output over :data:`TRIAGE_OUTCOMES`, and DETECTS rather than
    adjudicates — it classifies the relationship between the write and the
    candidate, it does not decide which text is true.

    Storing is also the right stub answer on the merits: with no judge, the
    only alternative is to attach on a similarity the calibration explicitly
    declined to call deterministic.

    *candidates* carries the retrieved records and is accepted with a DEFAULT,
    so a four-keyword call from a direct caller stays valid. This stub ignores
    them; leaf gamma's real judge consumes them and trims to PRD C1's top 3-5
    itself.

    STILL THE DEFAULT, deliberately, now that the real judge has landed.
    ``server/tools.py`` is the single place that passes
    ``judge=write_triage_judge.judge_write``, which keeps this module free of
    that import and the dependency acyclic (``write_triage_judge`` imports
    THIS module; this module must never import that one). It also keeps the
    judge-slot contract tests above meaningful — they inject their own fakes
    and must not accidentally reach a live LLM.
    """
    return OUTCOME_STORED


async def triage_write(
    memory_service: Any,
    *,
    content: str,
    project_id: str,
    counter: TriageFailOpenCounter,
    judge: Any = None,
    allow_near_duplicate: bool = False,
    caller_owns_attach_keys: bool = False,
    is_recon_stage_agent: bool = False,
) -> BandDecision:
    """Route one ``add_memory`` write. NEVER raises, NEVER blocks, NEVER errors.

    Force-store → retrieve → band → (judge slot) → decision. Every stage is
    wrapped after the force-store checks: any
    exception whatsoever yields :data:`OUTCOME_STORED` plus exactly one
    counted fail-open and a logged warning with ``exc_info``. Nothing escapes,
    because C1 is absolute — from the caller's side a fail-open is
    indistinguishable from "nothing matched", which is what keeps the write
    unblocked.

    This is where the module deliberately DIVERGES from the retired guard's
    call site, which re-raises ``TypeError``/``AttributeError``/``NameError``
    so a wiring bug surfaces loudly rather than being swallowed. Re-raising
    here would be an errored write, i.e. a blocked write. The loudness is
    preserved a different way: those three classes are logged at ERROR on one
    greppable line naming the exception type, and counted like any other
    fail-open — so a changed ``MemoryService.search`` signature surfaces as a
    storm escalation rather than as a stream of errored writes.

    *allow_near_duplicate*, *caller_owns_attach_keys* and
    *is_recon_stage_agent* are passed IN rather than recomputed:
    ``add_memory`` already derives all three from the metadata and the
    agent_id it holds, and a second derivation here is a second place for them
    to disagree about who is exempt. :func:`declares_attach_keys` is the one
    spelling of the *caller_owns_attach_keys* predicate.
    """
    if allow_near_duplicate:
        # D2 reinterprets the retired guard's bypass flag as triage's
        # force-store escape hatch: under the guard it meant "do not reject
        # me", here it means "do not reroute me". Same writer intent — the
        # content is genuinely distinct — expressed against the mechanism that
        # replaced the one it was built for.
        #
        # Returned BEFORE retrieval, not after: retrieval is an embedding +
        # vector round-trip, and a writer who has already declared the content
        # distinct should not pay for a lookup whose answer cannot change the
        # outcome.
        return BandDecision(OUTCOME_STORED, None, None, None, None)

    if caller_owns_attach_keys:
        # The caller already set `parent_id` and/or `kind`, and an attach
        # OVERWRITES both. Routing this write would re-parent the record under
        # whatever canonical triage picked and replace the caller's `kind`
        # with `sighting`/`amendment` — destroying, in the `kind` case, the
        # record's own declared classification and folding it into a
        # canonical's sighting count. A declared `amendment` demoted to a
        # `sighting` is the sharpest version: `grouped_read` digests amendment
        # TEXT while sightings are only counted, so the submitted content
        # stops being readable in the grouped document.
        #
        # All of that is C1 content loss caused by the C1 mechanism, so the
        # caller's own metadata wins: triage does not get a second vote on a
        # classification the caller already made.
        #
        # Returned before retrieval for the same reason as the flag above: no
        # candidate can change the answer, so no round-trip is spent asking.
        return BandDecision(OUTCOME_STORED, None, None, None, None)

    if is_recon_stage_agent:
        # The recon-stage exemption SURVIVES this leaf; LEAF IOTA owns its
        # removal, and its explicit signal is "a recon-agent direct near-dup
        # add_memory now triages like anyone else".
        #
        # Why it must survive until then: Stage-1 consolidation writes a merged
        # canonical that is EXPECTED to closely resemble the duplicates it
        # replaces, and there is no ordering guarantee that those duplicates are
        # deleted first. Attaching the merged entry as a sighting of one of them
        # would INVERT consolidation — the entry written to supersede a memory
        # would become its child — and the inversion would be invisible, because
        # a sighting of a near-identical parent is exactly what a real
        # restatement looks like.
        return BandDecision(OUTCOME_STORED, None, None, None, None)

    try:
        k = resolve_candidate_k(memory_service)
        t_high, t_low = resolve_bands(memory_service)
        results = await retrieve_candidates(memory_service, content, project_id, k)
        # A RETRIEVAL OUTAGE DOES NOT ARRIVE AS AN EXCEPTION. This check is
        # not defensive tidiness — without it the fail-open apparatus cannot
        # see the failure mode it was built for. ``MemoryService.search``
        # ABSORBS every store exception and every store timeout: it logs
        # `search.store_failed`, appends to `failed_stores`, and returns an
        # EMPTY ``SearchResults`` with ``degraded=True`` rather than raising.
        # So when mem0 is down or slow — the dominant retrieval failure, and
        # the one INV-4 names first — the `except` arm below never fires,
        # `decide_band([])` returns `stored`, the counter never increments,
        # and every write in the outage is stored untriaged and
        # indistinguishable from a genuinely novel corpus. That is precisely
        # the silent degradation this module exists to prevent.
        #
        # The degrade metadata is readable here ONLY because
        # `retrieve_candidates` returns the SearchResults object
        # un-transformed: `degraded`/`failed_stores` do NOT survive a slice,
        # a comprehension, or a sorted() (see `memory_service.SearchResults`'s
        # own warning). Do not "clean up" that return into a list.
        #
        # ANY degraded retrieval is a fail-open here, with no partial-result
        # subtlety, because `_TRIAGE_STORES` is the single store `mem0`:
        # degraded can only mean the one store triage depends on failed, so
        # the candidate slate is empty-or-unusable, never merely thinner.
        if getattr(results, 'degraded', False):
            failed = getattr(results, 'failed_stores', None)
            _record_fail_open(
                counter, project_id,
                RuntimeError(f'search degraded: failed_stores={failed!r}'),
                stage='retrieve',
            )
            return BandDecision(OUTCOME_STORED, None, None, t_high, t_low)
        decision = decide_band(results, t_high=t_high, t_low=t_low)
    except Exception as exc:  # noqa: BLE001 — C1: nothing escapes this path.
        _record_fail_open(counter, project_id, exc, stage='retrieve')
        return BandDecision(OUTCOME_STORED, None, None, None, None)

    if decision.outcome != OUTCOME_JUDGE:
        return decision

    try:
        verdict = await (judge or _stub_judge)(
            memory_service=memory_service,
            content=content,
            project_id=project_id,
            decision=decision,
            # PRD C1's judge input is "the new entry + top 3-5 candidates", so
            # the retrieved records have to survive the trip. Passed WHOLE and
            # un-transformed, for the same reason `retrieve_candidates` returns
            # it whole: a slice or a comprehension yields a plain list and
            # silently drops `degraded`/`failed_stores`. Trimming to the top
            # few is the judge's own job (write_triage_judge.
            # select_judge_candidates), which is also why no width is imposed
            # here -- one home for that decision.
            candidates=results,
        )
    except Exception as exc:  # noqa: BLE001 — C1: nothing escapes this path.
        _record_fail_open(counter, project_id, exc, stage='judge')
        return BandDecision(OUTCOME_STORED, None, None, decision.t_high, decision.t_low)

    if verdict not in TRIAGE_OUTCOMES:
        # A closed output set (D3) means an unrecognised verdict is a BUG, not
        # an extension point — counted as a fail-open so it cannot pass as a
        # routing decision nobody notices.
        _record_fail_open(
            counter, project_id,
            ValueError(f'judge returned {verdict!r}, not in TRIAGE_OUTCOMES'),
            stage='judge',
        )
        return BandDecision(OUTCOME_STORED, None, None, decision.t_high, decision.t_low)

    # `stored` carries no canonical: nothing was attached, so naming one would
    # invite a caller to attach to a candidate the judge declined to endorse.
    canonical_id = None if verdict == OUTCOME_STORED else decision.canonical_id
    return BandDecision(
        verdict, canonical_id, decision.similarity, decision.t_high, decision.t_low,
    )


#: The exception classes the retired guard's call site re-raised as wiring
#: bugs. Triage cannot re-raise (C1), so it logs them at ERROR instead of
#: WARNING to keep the same signal at a different volume.
_WIRING_BUG_CLASSES = (TypeError, AttributeError, NameError)


def _record_fail_open(
    counter: TriageFailOpenCounter,
    project_id: str | None,
    exc: BaseException,
    *,
    stage: str,
) -> dict[str, Any] | None:
    """Log and count one fail-open; return a storm summary iff a burst fired.

    Never raises: a counter that failed while recording a failure would turn
    a degraded write path into a broken one.
    """
    level = logging.ERROR if isinstance(exc, _WIRING_BUG_CLASSES) else logging.WARNING
    try:
        logger.log(
            level,
            'write_triage fail-open at stage=%s project=%s exc=%s — write stored '
            'WITHOUT triage (contract C1: never block a write)',
            stage, project_id, type(exc).__name__,
            exc_info=exc,
        )
        return counter.record(project=project_id)
    except Exception:  # noqa: BLE001 — the alarm must not break the write path.
        logger.exception('write_triage fail-open counter itself failed')
        return None


# --- the storm escalation (INV-4) ------------------------------------------
#
# Escalation wiring copied shape-for-shape from
# ``markup_tripwire.emit_markup_storm_escalation``, which took it from
# ``middleware/candidate_key_escalation.py``.
#
# _ANCHOR_TASK_ID is a stable per-project anchor (not a real task id) so the
# resulting ids form one greppable ``esc-write-triage-fail-open-N`` series and
# the dedup check has something to key on.
#
# It is THIS LEAF'S OWN and is shared with nobody, which is load-bearing rather
# than tidy. Measured: the L1 escalation watcher files its own cluster records
# under the ``markup-tripwire`` anchor and SQUATS it — the tripwire filed
# nothing 2026-08-16..2026-08-19 while 41 rejections occurred, all 17 records
# sitting at dedupe_count 0. A filer that dedupes against an anchor somebody
# else keeps open never files again, and the resulting silence is
# indistinguishable from health. That incident is why
# ``emit_markup_storm_escalation`` grew its ``anchor_task_id`` parameter, and
# it is why "simplifying" this into a shared anchor would disable the alarm.
_QUEUE_DIRNAME: str = 'data/escalations'
_ANCHOR_TASK_ID: str = 'write-triage-fail-open'
_AGENT_ROLE: str = 'fused-memory/write-triage'
_CATEGORY: str = 'write_triage_fail_open_storm'
_PRD_PATH: str = 'docs/prds/memory-write-path-convergence.md'


def emit_triage_fail_open_storm_escalation(
    project_root: str | None,
    storm: dict[str, Any],
) -> str | None:
    """File a ``write_triage_fail_open_storm`` escalation for a burst (INV-4).

    Returns the escalation id — freshly filed, or the id of an already-open
    escalation under this leaf's anchor (dedup) — or ``None`` when filing is
    impossible or fails.

    NEVER raises. By the time this runs the write's outcome is already decided
    AND the write has already been stored, so escalation is purely additive:
    every failure mode degrades to ``None`` plus a log line. An exception
    escaping here would convert a successfully-degraded write into a failed
    one, which is the exact C1 violation the whole fail-open apparatus exists
    to prevent.

    A ``None`` *project_root* is a quiet no-op: ``add_memory`` takes a
    ``project_id``, and an unknown project resolves to no root at all.

    The anchor dedup matters beyond the counter's own per-window rate limit: a
    retrieval outage running for hours would otherwise file one escalation per
    window, so those collapse into the single open record until an operator
    resolves it.
    """
    if project_root is None:
        logger.debug(
            'write_triage: no project_root resolved; fail-open storm %r will '
            'not be escalated',
            storm,
        )
        return None
    if not HAS_ESCALATION:
        logger.debug(
            'write_triage: escalation package unavailable; fail-open storm %r '
            'in project_root=%r will not be escalated',
            storm, project_root,
        )
        return None

    try:
        queue = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
    except Exception:
        logger.exception(
            'write_triage: failed to open the escalation queue for '
            'project_root=%r; fail-open storm %r not escalated',
            project_root, storm,
        )
        return None

    # Best-effort dedup: a read failure falls THROUGH to filing rather than
    # bailing out — losing duplicate-suppression is strictly better than losing
    # the alarm for a degradation that is happening right now.
    try:
        existing = queue.get_by_task(_ANCHOR_TASK_ID, status='pending')
    except Exception:
        logger.exception(
            'write_triage: failed to check for an existing open fail-open '
            'escalation for project_root=%r; proceeding to file a new one',
            project_root,
        )
        existing = []
    if existing:
        logger.info(
            'write_triage: %s already open for project_root=%r (storm %r now); '
            'not filing a duplicate',
            existing[0].id, project_root, storm,
        )
        return existing[0].id

    count = storm.get('count')
    window_seconds = storm.get('window_seconds')
    detail = '\n'.join([
        f'project_root={project_root!r}',
        f'fail_opens_in_window={count!r}',
        f'threshold={storm.get("threshold")!r}',
        f'window_seconds={window_seconds!r}',
        # The counter is per-server, not per-project, so the count above may
        # span more than this project_root. Naming them all keeps the record
        # honest and points at the co-affected queues, rather than blaming
        # whichever write happened to cross the threshold.
        f'projects_in_window={storm.get("projects")!r}',
        '',
        'add_memory write triage (PRD leaf beta, task 3127) FAILED OPEN '
        'repeatedly inside one rolling window. This is a DEGRADATION, not a '
        'data-loss incident, and the distinction decides the remediation:',
        '',
        '  * Every write in the window still LANDED — stored WITHOUT triage, '
        'i.e. with the pre-triage behaviour. No write was lost and no write '
        'was blocked (contract C1 holds on every fail-open path).',
        '  * What was lost is DETECTION: no restatement was recognised, so '
        'writes that should have been attached as sightings of an existing '
        'memory were stored as new standalone entries instead. Nothing is '
        'corrupted and nothing needs unwinding — the corpus simply grew the '
        'duplicates triage exists to prevent.',
        '',
        'A burst therefore means a DEPENDENCY is down, not that triage is '
        "misfiring. Check candidate retrieval first (MemoryService.search / "
        'mem0 reachability / the embedding provider), then the judge. Grep the '
        "server logs for 'write_triage fail-open at stage=' — the stage= field "
        'says which of the two it was, and a stage the logs record at ERROR '
        '(TypeError/AttributeError/NameError) is a WIRING BUG, most likely a '
        'changed MemoryService.search signature, not an outage.',
        '',
        'To stop triage DELIBERATELY rather than letting it degrade silently, '
        'set write_triage.enabled: false in fused-memory/config/config.yaml. '
        'That is the one lever — it is green-tier hot-reloadable, so '
        'reload_config applies it without a restart. Disabling it restores '
        "today's pre-triage behaviour exactly, which is already what these "
        'writes got; the difference is that it stops being silent.',
        '',
        f'Owner: {_PRD_PATH} (leaf beta / task 3127). Attach the stage, the '
        'exception type and the affected project from the log lines above '
        "against that PRD's open leaves.",
    ])

    try:
        esc = Escalation(  # type: ignore[possibly-unbound]
            id=queue.make_id(_ANCHOR_TASK_ID),
            task_id=_ANCHOR_TASK_ID,
            agent_role=_AGENT_ROLE,
            severity='blocking',
            category=_CATEGORY,
            summary=(
                f'{count} add_memory write(s) triaged WITHOUT triage in '
                f'{window_seconds}s — write triage is failing open '
                f'(see {_PRD_PATH})'
            ),
            detail=detail,
            suggested_action=(
                'check MemoryService.search / mem0 reachability, then the '
                "judge; grep the logs for 'write_triage fail-open at stage=' "
                'for the failing stage. To stop triage deliberately, set '
                'write_triage.enabled: false (green-tier hot-reloadable)'
            ),
            level=1,
        )
        esc_id = queue.submit(esc)
    except Exception:
        # A queue I/O failure must not propagate: the write has already been
        # stored, and the WARNING/ERROR log at the fail-open site has already
        # recorded the burst. The operator simply loses the queued heads-up.
        logger.exception(
            'write_triage: failed to submit fail-open storm escalation for '
            'project_root=%r (storm %r)',
            project_root, storm,
        )
        return None

    logger.warning(
        'write_triage: queued %s for project_root=%r (fail-open storm %r)',
        esc_id, project_root, storm,
    )
    return esc_id
