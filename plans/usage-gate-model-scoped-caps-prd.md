# PRD: Model-scoped account caps in the UsageGate (fable-aware failover)

**Status:** active — authored 2026-07-20 (autonomous session; design input is the
verified brief `~/.claude/spawn-briefs/fable-architect-followup-prd-2026-07-20.md`,
which pre-answers G1–G6 and records Leo's constraints; honored here, not
re-litigated). **Project:** dark_factory. **Approach:** **B+H** (contract +
boundary-test sketch). G5 heuristic hit: the UsageGate governs **all fleet
dispatch** (the load-bearing seam), cross-module blast radius 3
(`shared/usage_gate.py`, `shared/cli_invoke.py`, `orchestrator/routing.py` +
`workflow.py`), and ≥2 cross-PRD consumers (adaptive-routing ξ, the
fable-architect eval/admission PRD).

## Goal

Give the UsageGate **per-(account, model-scope) cap awareness** so that:

1. A **fable cap-hit does NOT cap the account's remaining general capacity** —
   the account keeps serving non-fable dispatches.
2. A **fable-needing invocation fails over separately**: it skips accounts whose
   fable half is exhausted even when their general capacity remains, and lands
   on an account with fable headroom.
3. **All-fable-capped never freezes the fleet**: fable-needing work waits (gate
   level) or degrades to the next routing layer (resolver level) while general
   sessions keep proceeding untouched.

**Enabling fact (Leo, 2026-07-20):** all pool accounts have `claude-fable-5`
access, each capped at **50% of that account's token spend**. This updates the
adaptive-routing PRD's decision 4 ("only the interactive account is proven";
per-(account,model) cap rework out of scope) — this PRD **lifts that deferral
and owns the rework**.

## Background

Current semantics (`shared/src/shared/usage_gate.py`): one `AccountPhase` per
account (AVAILABLE/PROBING/PROBE_IN_FLIGHT/CAPPED/AUTH_FAILED); cap detection
is reactive (stderr pattern matching via `classify_invocation`); resume is
timer-based on parsed `resets_at` with optimistic uncap + reactive re-detect;
`before_invoke()` returns the first available account's lease and **blocks only
when all accounts are capped**. There is **no model dimension anywhere**: a cap
hit while running fable would CAP the whole account, wasting its untouched
general half and blocking general work behind fable exhaustion; conversely a
fable-needing session cannot skip a fable-exhausted-but-generally-available
account.

This is the safety prerequisite for admitting fable to production routing
(adaptive-routing task ξ = task 2544): without scope separation, any
override-pinned or retry-rung fable dispatch risks knocking a whole account out
of the pool for hours.

## Sketch of approach

A minimal **scope overlay** on the existing account machinery — deliberately
NOT a per-scope clone of the phase state machine (G7 `no-lockstep-duplication`):

- **Scope derivation.** New config `usage_cap.scoped_cap_models: list[str]`
  (default `['claude-fable-5']`). An invocation's scope is its model string if
  listed, else `None` (general). With production routing not yet admitting
  fable, shipping the default changes **no production behavior** (general
  invocations derive scope `None` and take today's exact paths).
- **Scope cap state.** `AccountState` gains `scope_caps: dict[str, ScopeCap]`
  where `ScopeCap = {capped, resets_at, near_cap, capped_at}` — a flag + timer,
  not a phase machine. The account-level phase machine, lease/generation
  machinery, probe loops, and auth handling are untouched.
- **Attribution by invoked model** (the load-bearing design decision): a
  cap-hit is attributed to the scope of the invocation that observed it. A
  scoped (fable) invocation's CapHit marks **only that account's fable scope**
  capped; a general invocation's CapHit caps the **whole account** (existing
  path, unchanged). This makes the design **independent of the cap message
  text** — no unverified assumption about a fable-specific stderr shape (G3).
  If the whole account is in fact capped, the next general invocation on it
  re-detects and account-caps it — one wasted attempt, the module's existing
  optimistic-uncap philosophy.
- **Scope-aware selection.** `before_invoke(scope=None)` keeps today's
  predicate; `before_invoke(scope=m)` additionally skips accounts whose scope
  `m` is capped (and not past `resets_at`). Account-level CAPPED/AUTH_FAILED
  always excludes the account for every scope.
- **Scope-aware waiting that never touches the general gate.** When every
  account is scope-capped for `m` but some are generally available, the scoped
  waiter sleeps toward the soonest scope `resets_at` on a **separate wake
  mechanism** — it must never clear `_open` (the fleet pause gate). Timer-based
  optimistic scope-uncap + reactive re-cap; **no fable probe loop** (the
  account resume probe hardcodes haiku and cannot confirm fable headroom; a
  real fable dispatch is the probe).
- **Resolver capacity fail-safe** (production degrade-don't-wait):
  `resolve_route` gains a scope-capacity check with the exact mechanics of the
  existing per-model ceiling check — a layer resolving to a model whose scope
  has no account headroom is skipped, `rejected` records
  `"<layer>:model-capacity-exhausted"`, resolution falls through, dispatch is
  never blocked. Fed by a gate snapshot threaded from `TaskWorkflow._invoke`;
  snapshot absent → check skipped (fail-safe).

## Resolved design decisions

1. **Scope overlay, not phase-machine clone.** Only account-level state needs
   PROBING/PROBE_IN_FLIGHT/AUTH_FAILED; a scope needs capped+resets_at+near_cap.
   Cloning the transition table per scope would be lock-step duplication and
   double the invariant surface for no benefit.
2. **Attribution by invoked model, never by message text.** The fable cap-hit
   stderr shape is unverified substrate; the design must not depend on it. The
   generic CAP_HIT/NEAR_CAP pattern family + `_parse_resets_at` are reused
   as-is. Residual risk (monitored, not load-bearing): if a fable cap-hit
   surfaces a message the patterns don't match at all, it classifies Failure →
   existing retry/failover, no scope state change; the existing cap-like-prefix
   breadcrumb logging covers capture for a pattern extension.
3. **General cap implies scope-unusable; scope cap implies nothing about
   general.** Selection checks account phase first, scope second. A general
   CapHit does not write scope state (the account-level CAPPED already excludes
   every scope).
4. **No scope probe loop.** Timer-based optimistic uncap at `resets_at`
   (unknown `resets_at` → conservative fixed backoff, tactical default decided
   in γ), re-cap reactively on the next scoped invocation. Bounded by the
   existing `consecutive_cap_hits` / `max_cap_retries` / `cap_wait_sanity_secs`
   machinery (G7 storm-escape).
5. **Two-layer wait-vs-degrade split.** Production fable dispatches degrade at
   resolve time (capacity check → next layer → opus) and effectively never
   scope-wait; gate-level scope-waiting remains correct and bounded for
   direct/pinned callers (eval campaigns, operator sessions) where waiting for
   fable is the intent.
6. **Byte-equivalence shipping.** `scoped_cap_models` default lists fable, but
   no production dispatch resolves fable pre-admission, so behavior is
   unchanged on ship. `scoped_cap_models: []` disables the mechanism entirely
   (kill switch). Existing UsageGate test suites must stay green unmodified
   except where they construct config directly.
7. **Claude-backend scope only.** Non-claude backends bypass account failover
   entirely (harness-backend decision 3, single-account); the scope machinery
   is OAuth-pool/claude-only. `classify_invocation`'s backend branching is
   unchanged.
8. **Scoped observability via existing event stream.** Cost events
   (`cap_hit`/`near_cap`/`failover`) gain a `scope` field in their JSON
   details; a new read API `scope_status()` exposes per-(account × scope) state
   for the digest/dashboard, but rendering a panel is out of scope here.

## Pre-conditions for activating

None external — all substrate verified on main this session (G3):

- `model` is in scope before slot acquisition: `cli_invoke.py:968`
  (`invoke_kwargs.get('model', ...)`) precedes `usage_gate.invoke_slot()` at
  :1104 — threading the scope into the lease/report path is additive.
- Attribution seam: `InvokeSlot.report(outcome)` / `detect_cap_hit` dispatch to
  `_handle_cap_detected` / `_handle_near_cap_warning` with the account token —
  scope rides alongside.
- `_parse_resets_at` handles generic "resets …" phrases;
  `_refresh_capped_accounts` is the existing timer-sweep precedent.
- Resolver fail-safe mechanics: `resolve_route`'s allowlist + per-model-ceiling
  layer-skip + `RoutingDecision.rejected` (task 2535, landed) — the capacity
  check is a third check with identical semantics.
- `FABLE_CANDIDATE_MODEL = 'claude-fable-5'` + `probe_models` per-account ×
  model probing (task 2532, landed) — availability evidence for admission.
- The 50%/account fable cap is Leo's stated fact (external substrate; the
  probe artifact confirms access, and scope semantics are cap-fraction-agnostic
  — nothing here encodes "50%").

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/adaptive-model-routing-prd.md` | resolves its deferral; extends its resolver | Decision 4 / Out-of-scope "Per-(account,model) UsageGate cap states" is **lifted** — owned here (paired edit marks it). δ extends `resolve_route` with the capacity check (same fail-safe shape as its ceiling check); ξ = task **2544** consumes this PRD's integration gate ε as a hard dep (wired at decompose, task amended per the fable-architect-eval PRD) | **this-prd** owns the cap-state + capacity check; adaptive-routing keeps the resolver contract + admission flip | paired edit + dep wiring at decompose |
| `plans/fable-architect-eval-admission-prd.md` (sibling, authored same session) | produces | scope-safe fable dispatch is the safety prerequisite its ratification gate (τ3) and the amended 2544 depend on; the bounded eval campaign itself deliberately does NOT depend (small spend, accepted pre-scope risk) | **this-prd** | dep wired at decompose |
| `plans/harness-backend-reconnect-pi-prd.md` | boundary only | backend axis stays theirs (provider pools, `_MODEL_COSTS`/price table, non-claude cap semantics); this PRD's scope state is claude/OAuth-pool-only and touches no backend selection. No contested ownership — that PRD never claims UsageGate cap-state, and its decision 3 leaves non-claude backends outside failover | each PRD its axis | no seam task owed |

## Contract (H)

### C1 — Scope model

```python
# shared/src/shared/config_models.py
class UsageCapConfig:
    scoped_cap_models: list[str] = ['claude-fable-5']   # [] disables scoping

# shared/src/shared/usage_gate.py
@dataclass
class ScopeCap:
    capped: bool = False
    resets_at: datetime | None = None
    near_cap: bool = False
    capped_at: datetime | None = None

class AccountState:
    scope_caps: dict[str, ScopeCap]      # keys ⊆ scoped_cap_models; lazily created

def scope_for(model: str, config: UsageCapConfig) -> str | None:
    # model in scoped_cap_models → model; else None (general)
```

### C2 — Selection & waiting

```python
async def before_invoke(self, scope: str | None = None) -> AccountLease | None
```

- **Invariant S1 (general untouched):** `scope=None` behaves byte-identically
  to today for every input.
- **Invariant S2 (scope skip):** `scope=m` never returns an account whose
  `scope_caps[m].capped` holds with `resets_at` in the future, nor any account
  the general predicate would skip.
- **Invariant S3 (no fleet freeze):** an all-scope-capped wait never clears
  `_open`, never sets `paused_reason` for the fleet, and never delays a
  concurrent `scope=None` caller. The scoped wait uses its own wake mechanism
  (tactical choice in γ) with timeout toward the soonest scope `resets_at`.
- **Invariant S4 (account cap dominates):** an account-level CAPPED/AUTH_FAILED
  excludes the account for every scope; a scope cap excludes it only for that
  scope.

### C3 — Attribution

`InvokeSlot` carries the invocation's scope (derived once in
`invoke_with_cap_retry` from `invoke_kwargs['model']`).

- **Invariant S5:** CapHit reported through a slot with scope `m` writes
  `scope_caps[m]` (capped + resets_at) and does **not** transition the account
  phase; CapHit with scope `None` takes the existing `_handle_cap_detected`
  path unchanged. NearCap annotates the matching scope (or account, scope
  `None`). AuthFailed and all non-cap outcomes are scope-blind (account-level,
  unchanged).
- **Invariant S6 (optimistic uncap):** a scope cap past `resets_at` is cleared
  by the selection-time sweep; a premature clear is re-capped by the next
  scoped invocation's re-detection (bounded by existing retry machinery).

### C4 — Resolver capacity check

```python
# orchestrator/src/orchestrator/routing.py
RouteInputs gains: scope_capacity: Mapping[str, bool] | None   # model → any-account-headroom
# UsageGate read API feeding it:
def scope_capacity_snapshot(self) -> dict[str, bool]   # for each scoped model
```

- **Invariant S7 (fail-safe, never blocks):** when a layer would set model `m`
  with `scope_capacity[m] == False`, that layer's model assignment is skipped
  and `"<layer>:model-capacity-exhausted"` appended to `rejected` — identical
  mechanics to `model-ceiling-exhausted`. `scope_capacity=None` (or `m` absent
  from the mapping) skips the check entirely. `routing_decision` events carry
  the rejection.
- **Invariant S8 (advisory only):** the snapshot is resolve-time advisory; the
  gate's own scope predicate at invoke time remains authoritative (a stale
  snapshot degrades to a scope-wait/failover, never a crash).

### Boundary-test sketch (two-way; task ε's observable signal)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Fable cap leaves general open | account A fable-scope capped (via a scoped CapHit) | `before_invoke()` (general) returns A; a general invocation on A completes normally |
| B2 | Scoped failover | A fable-capped, B has fable headroom | `before_invoke(scope=fable)` returns B; `failover` cost event carries `scope: fable` |
| B3 | All-fable-capped ≠ fleet freeze | A and B both fable-capped, both generally available | a `scope=fable` caller waits (or its resolver layer degrades); a concurrent general caller returns immediately; `_open` stays set; fleet `paused_reason` empty |
| B4 | General cap dominates | A account-CAPPED via a general CapHit | `before_invoke(scope=fable)` and `before_invoke()` both skip A |
| B5 | Timer uncap + reactive re-cap | A's fable `resets_at` passed; the API still caps | next `scope=fable` selection returns A; the invocation's CapHit re-caps the scope with the new `resets_at`; failover proceeds; no tight loop (retry counters advance) |
| B6 | Byte-equivalence kill switch | `scoped_cap_models: []` | every existing UsageGate test green; no scope state allocated; `before_invoke(scope='claude-fable-5')` behaves as general |
| B7 | Resolver degrade | override pins fable; snapshot says fable exhausted | resolved model falls to next layer; `rejected` names `metadata_override:model-capacity-exhausted`; dispatch proceeds; `routing_decision` event shows it |
| B8 | Non-claude bypass unchanged | `invoke_fn` supplied, `backend='codex'` | no scope derivation/state writes; T1 reconnect contract shapes preserved |

## Decomposition plan

Chained α→β→γ (all edit `usage_gate.py`; the narrow module lock serializes them
anyway — chaining avoids rebase churn, per the eval-revival Phase-1 precedent).

- **α — Scope substrate: config + state + derivation** *(intermediate →
  unlocks β)*. `scoped_cap_models` config field (validated list[str]),
  `ScopeCap`, `AccountState.scope_caps`, `scope_for()`. Signal (intermediate):
  config loads with the default; `scope_for` unit-covered; existing suites
  green (B6 shape). Modules: shared/src/shared/{config_models,usage_gate}.py,
  shared/tests.
- **β — Scope-aware attribution** *(intermediate → unlocks γ)*. Thread the
  invocation's scope from `invoke_with_cap_retry` (model already in scope at
  cli_invoke.py:968) through `InvokeSlot` into `report`/`detect_cap_hit`;
  scoped CapHit/NearCap write `scope_caps` per invariant S5; scoped cost-event
  fields. Signal: a fake-invoke test drives a scoped CapHit and asserts scope
  state written + account phase untouched (B1's write half). Modules:
  shared/src/shared/{usage_gate,cli_invoke}.py, shared/tests.
- **γ — Scope-aware selection, waiting, uncap + read API** *(intermediate →
  unlocks δ, ε)*. `before_invoke(scope=)`, the S3 wait mechanism, the
  `resets_at` sweep, `scope_capacity_snapshot()` + `scope_status()`. Signal:
  B1/B2/B4/B5 green at the gate level. Modules: shared/src/shared/usage_gate.py,
  shared/tests.
- **δ — Resolver capacity fail-safe** *(intermediate → unlocks ε)*.
  `RouteInputs.scope_capacity` + the layer-skip check mirroring the ceiling
  check; `TaskWorkflow._invoke` threads the gate snapshot. Signal: B7 green —
  a fable-pinned resolution with an exhausted snapshot falls back one layer and
  records the rejection. Modules: orchestrator/src/orchestrator/{routing,
  workflow}.py, orchestrator/tests.
- **ε — Scope integration gate (B+H boundary suite B1–B8)** *(leaf —
  C-as-integration-gate)*. Two-way suite through the cap-retry loop with a fake
  `invoke_fn` injecting cap stderr per scope, plus the resolver path. Signal:
  B1–B8 green end-to-end; this is the dep the fable admission (task 2544) and
  the sibling PRD's ratification gate consume. Modules: shared/tests,
  orchestrator/tests.
- **ζ — Docs + operator surface** *(leaf)*. CLAUDE.md (usage-cap scoping +
  `model-capacity-exhausted` in the routing rejected vocabulary),
  skills/orchestrate. Signal: docs committed naming `scoped_cap_models` and the
  scope semantics. Modules: CLAUDE.md, skills/orchestrate/SKILL.md.

## Out of scope

- Fable admission itself (allowlist/ladder/ceiling flip + any fleet routing
  rule) — adaptive-routing task **2544**, ratification-gated per the sibling
  eval PRD.
- Provider/backend axis: non-claude cap semantics, credential pools, price
  tables (`harness-backend-reconnect-pi`).
- Per-model daily **USD ceilings** — already live (`per_model_daily_ceiling_usd`,
  adaptive-routing); orthogonal to token-spend cap scopes.
- Dashboard/digest **rendering** of scope state (the `scope_status()` API is
  the substrate; a panel is a follow-up).
- Persisting scope caps across restarts — reactive re-detection suffices,
  matching account-cap behavior.
- Encoding the 50% fraction anywhere — scope semantics are fraction-agnostic.

## Open questions (tactical — surfaced, not blocking)

1. **S3 wake mechanism** — per-scope `asyncio.Event` vs a single
   `asyncio.Condition` notified on any state write, with a
   soonest-scope-reset timeout. Either satisfies S3. Decide in **γ**.
2. **Unknown-`resets_at` scope backoff default** — a scoped CapHit whose
   message yields no parseable reset time; suggested: reuse
   `max_probe_interval_secs` as the optimistic re-open interval. Decide in **γ**.
3. **Scoped cost-event naming** — reuse `cap_hit`/`near_cap`/`failover` with a
   `scope` detail field (suggested; keeps consumers working) vs new event
   names. Decide in **β**.
4. **Snapshot staleness tolerance in δ** — snapshot at resolve time vs a
   read-through callable; suggested: plain snapshot (S8 makes staleness
   harmless). Decide in **δ**.
