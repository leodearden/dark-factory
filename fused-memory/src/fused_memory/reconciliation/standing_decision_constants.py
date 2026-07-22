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
  ``GROUNDS_TOKEN_FAMILIES`` for the fallback token-family match.
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
# grounds"). The seed below is a minimal, non-empty placeholder γ replaces.
GROUNDS_TOKEN_FAMILIES: dict[str, tuple[str, ...]] = {
    GROUNDS_STRUCTURAL_SIZE_CONFLATION: ('size', 'large', 'count', 'magnitude'),
}

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
