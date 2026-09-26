"""Single-source constants for the entity-standing-decision batch (task 2894 α,
PRD plans/stage1-entity-standing-decision-prd.md).

INV-5 (single source, no lockstep duplication): this module is the sole import
home for the standing-decision record kind, the closed grounds enum, the
grounds→token-family binding, the ``investigation_outcome`` mem0 kind, the
state/expiry_reason vocabularies, and the 90-day TTL. Consumers:

* **α** (this batch, ``recon_ledger``) — imports the record kind, ``STATE_*``,
  ``EXPIRY_REASON_TTL``, and ``GROUNDS_ENUM`` for the ledger substrate and its
  TTL-flip gc().
* **β** (writer + authorization gate) — applies ``STANDING_DECISION_TTL_DAYS``
  to ``decided_at`` and consumes ``MEM0_KIND_INVESTIGATION_OUTCOME`` /
  ``GROUNDS_ENUM``.
* **γ/δ** (Hook A filter / Hook B annotation) — consume ``GROUNDS_ENUM`` and
  ``GROUNDS_TOKEN_FAMILIES`` for the fallback token-family match; γ also
  consumes ``SUPPRESSION_STORM_THRESHOLD_PER_CYCLE`` and
  ``CATEGORY_STANDING_DECISION_STORM`` for its storm escape.
* **ε** (prompt/self-model renderer) — consumes the grounds enum and
  ``MEM0_KIND_INVESTIGATION_OUTCOME``.
* **ζ** (growth/merge sweeps) — consume ``STATE_*`` and the growth/merge
  ``EXPIRY_REASON_*`` values.

α owns only the binding STRUCTURE of ``GROUNDS_TOKEN_FAMILIES`` (every grounds
value has a bound family); the actual seed token contents are γ-tactical (PRD
Open Question 5, decided in γ). α defines ``STANDING_DECISION_TTL_DAYS`` but
does NOT compute ``decided_at + 90d`` — ``recon_ledger`` is clock-injected with
no ``datetime.now()``; the writer β applies the default while α enforces the
never-None-``expires_at`` invariant at the ledger write boundary.
"""

from __future__ import annotations

# --- Ledger record kind (persisted contract — must not drift) ---------------
RECORD_KIND_ENTITY_STANDING_DECISION = 'entity_standing_decision'

# --- Grounds vocabulary (closed enum, initially one value) ------------------
# PRD Open Question 2 / row 2: a closed enum, initially the single value
# ``structural_size_conflation``. Complaints citing a specific edge uuid are by
# definition outside it (the b0057f3d escape hatch). Additional grounds values
# are added here (single source) when a new recurring false-positive class is
# characterized.
GROUNDS_STRUCTURAL_SIZE_CONFLATION = 'structural_size_conflation'
GROUNDS_ENUM: frozenset[str] = frozenset({GROUNDS_STRUCTURAL_SIZE_CONFLATION})

# --- Grounds → token-family binding (INV-5) ---------------------------------
# α owns the binding STRUCTURE only: every grounds value in GROUNDS_ENUM has a
# bound token-family tuple. The seed token *contents* are γ-tactical (PRD Open
# Question 5, decided in γ) — γ consumes this map for the Hook A/B fallback
# match ("the flag_type matches the token-family list bound to the row's
# grounds").
#
# γ (task 2896, PRD Open Question 5) finalizes the seed for
# ``structural_size_conflation`` to a curated list of DISTINCTIVE stems covering
# the "entity is too big / its edges conflate multiple topics" false-positive
# class. Each entry is a lowercase substring stem matched (casefolded) against a
# flag_type by :func:`~fused_memory.reconciliation.flag_dedup._flag_type_in_grounds_family`
# — e.g. ``'conflat'`` matches ``topic_conflation``/``conflated_entity``,
# ``'monolith'`` matches ``monolithic_entity``. Kept deliberately conservative
# (distinctive stems, not generic filler words) so an unrelated flag_type citing
# the entity uuid in free text is NOT fallback-suppressed — the under-suppression
# bias (PRD decision 10): a fallback miss costs one cycle of noise, never a
# hidden finding. Preserves α's structure invariant (keys == GROUNDS_ENUM; each
# family a tuple of str), so test_standing_decision_constants.py stays green.
#
# DISTINCTIVE means "does not occur in a flag_type outside this class", and that
# is a checkable property, not a slogan (reviewer finding correctness, amendment
# pass).  The first cut also carried ``'count'``, ``'topic'``, ``'scope'`` and
# ``'broad'``; each matches live flag_types that have nothing to do with entity
# size — ``'count'`` hits ``recon_stale_task_count_snapshot`` and every other
# ``*_count_*`` flag, ``'scope'`` hits ``scope_violation`` and
# ``consolidated_scope_correction``.  A stem that fires there does not merely
# add noise: it silently DROPS that finding for any entity under an active
# standing decision, which is the exact failure the under-suppression bias
# forbids.  They are dropped rather than made exact-token matches because
# splitting on ``_`` would not save them (``scope_violation`` and
# ``recon_stale_task_count_snapshot`` both contain the bare token) while it
# WOULD break the stems that carry this family (``'conflat'`` inside
# ``conflation``, ``'size'`` inside ``oversized``).  The surviving eight cover
# the class; ``topic_conflation`` still matches via ``'conflat'``.  Add a stem
# only after checking it against the live flag_type vocabulary.
GROUNDS_TOKEN_FAMILIES: dict[str, tuple[str, ...]] = {
    GROUNDS_STRUCTURAL_SIZE_CONFLATION: (
        'size',
        'large',
        'magnitude',
        'conflat',
        'sprawl',
        'bloat',
        'monolith',
        'overload',
    ),
}

# --- Suppression-storm threshold (γ, task 2896; PRD Open Question 4) ---------
# Hook A (Stage-1 filter, γ) files ONE recon "storm escape" escalation per
# ACTIVE standing decision that suppresses MORE THAN this many flags in a single
# reconciliation cycle (strict ``>``): an active decision hiding a flood of
# flags in one cycle is a signal the decision may be over-broad or the entity's
# situation has changed, warranting a human look. The parenthetical PRD variant
# ("across a streak of cycles") requires persistent per-decision cross-cycle
# state and is deferred; γ implements the self-contained per-cycle N.
SUPPRESSION_STORM_THRESHOLD_PER_CYCLE = 5

# Escalation category of that storm escape, single-sourced HERE rather than
# spelled as a literal at each use site (reviewer finding architecture-coherence,
# amendment pass).  ``Escalation.category`` is free-form prose validated by
# nothing at submit time (see the REFACTOR TRIGGER comment on
# ``escalation.models.Escalation``), and the correctness of a categorized
# detector rests entirely on its filer and its reader spelling the category
# identically: γ's filer stamps it on the record while the fold gate
# (``DedupeConfig.infra_dedupe_categories``) and any operator query read it back.
# A shared constant makes that agreement structural instead of a discipline
# every future reader has to re-verify.
CATEGORY_STANDING_DECISION_STORM = 'reconciliation_standing_decision_storm'

# --- Evidence mem0 kind (Arm 2 of the β authorization gate) -----------------
# The ledger kind is the sole machine-consulted standing-decision form; the
# ad-hoc mem0 kinds are demoted to evidence-only (PRD row 6). This is the mem0
# kind the β authorization gate counts (≥3 matching records) and ε emits.
MEM0_KIND_INVESTIGATION_OUTCOME = 'investigation_outcome'

# --- State vocabulary -------------------------------------------------------
STATE_ACTIVE = 'active'
STATE_EXPIRED = 'expired'
STATE_REVOKED = 'revoked'
STANDING_DECISION_STATES: frozenset[str] = frozenset(
    {STATE_ACTIVE, STATE_EXPIRED, STATE_REVOKED}
)

# --- Expiry-reason vocabulary (INV-2: structured fact at the transition) ----
# A non-active row carries an expiry_reason ∈ {ttl, growth, merge, operator},
# stamped structurally at the state flip — never re-derived from logs.
EXPIRY_REASON_TTL = 'ttl'
EXPIRY_REASON_GROWTH = 'growth'
EXPIRY_REASON_MERGE = 'merge'
EXPIRY_REASON_OPERATOR = 'operator'
EXPIRY_REASONS: frozenset[str] = frozenset(
    {EXPIRY_REASON_TTL, EXPIRY_REASON_GROWTH, EXPIRY_REASON_MERGE, EXPIRY_REASON_OPERATOR}
)

# --- TTL --------------------------------------------------------------------
# β applies ``decided_at + STANDING_DECISION_TTL_DAYS`` to compute expires_at;
# α owns the constant and the never-None-expires_at invariant only.
STANDING_DECISION_TTL_DAYS = 90
