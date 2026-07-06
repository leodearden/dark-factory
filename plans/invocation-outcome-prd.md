# Invocation-Outcome Seam (W4) — PRD

**Program:** bug-hotspot remediation 2026-07-06, stream **W4** (wave 1).
**Status:** active — decompose-ready.
**Slug:** `invocation-outcome`. **Approach:** B + H (contract + two-way boundary tests).
**Authoritative seam map:** `plans/bug-hotspot-remediation-program-2026-07-06.md`.
**Survey evidence:** `plans/bug-hotspot-survey-2026-07-06.md` cluster *shared-infra*
(full findings `…-full-findings.json` cluster 8, findings 0–5).
**Write-tag:** `agent_id="claude-prd-invocation-outcome"`.

---

## 1. Goal

Collapse the six overlapping structures that answer **"what did this CLI invocation
mean, and what should happen to the account?"** into **one seam** in the repo's
highest-fix-ratio file (`shared/usage_gate.py`, 68%) plus its co-conspirator
`shared/cli_invoke.py`. Today that question is answered in **five classifier sites with
divergent rules** and driven into account state through **an unenforced six-flag state
machine mutated at ~10 sites**, with **two forked cap-retry loops**, an **attribution
predicate that disagrees with the selection predicate**, and a **shared mutable probe
credentials file**. Every one of these is a live fix-generating structure with a long
whack-a-mole commit history (10× cap-message fixes, 4× probe-slot-leak fixes).

**What an operator/developer observes when this lands:**
- A single `classify_invocation()` is the only place any cap/auth/wedge/error verdict is
  computed; a new CLI wording is added in **one** table, not patched into 2–3.
- `grep` finds `CAP_HIT_PREFIXES` / `CAP_CONFIRM_KEYWORDS` / `NON_CAP_CLI_ERROR_MARKERS`
  / backend patterns in exactly **one** module.
- Illegal account transitions (e.g. `AUTH_FAILED → PROBE_IN_FLIGHT`) **raise loudly** in
  tests instead of silently corrupting state.
- Cost/log attribution **always names the account actually invoked** — the
  `PROBE_IN_FLIGHT` name-skew is gone.
- The steward inherits the zero-output wedge guard, auth routing, and marker exclusions
  it silently lacks today (a wedged steward is no longer re-resumed forever).
- Probe credentials no longer race a single shared `/tmp/claude-config-usage-gate-probe/
  .credentials.json` across the fleet.

## 2. Background

`shared.usage_gate.UsageGate` is a per-process in-memory async account pool over a
**host-wide shared 6-account pool**; `shared.cli_invoke.invoke_with_cap_retry` is a
~340-line policy loop wrapping it. `orchestrator.usage_gate` is a 10-line **pure
re-export shim** of `shared.usage_gate` — *not* a fork (no drift risk; leave it alone).
The layering and the six findings this PRD resolves are documented in the survey's
`architecture_notes` / `cross_system_notes` for cluster 8. Prior point-fixes
(tasks 297, 313, 300, 381, 680, 729, 786, 898 — all `done`) are the *evidence* of the
defect classes, not competitors: each patched one instance of a class this PRD makes
structurally impossible.

**Duplication doctrine (program decision #4):** `task_interceptor.py:115`'s "duplication
is cheaper than cross-package coupling" is retired. `shared/` is the sanctioned home for
cross-process contracts (precedent: `shared.usage_gate`, `shared.locking`). The new
`shared/invocation_outcome.py` follows that precedent.

## 3. Sketch of approach

Six mechanisms, one seam:

1. **`shared/src/shared/invocation_outcome.py`** — a sum type
   `InvocationOutcome = OK | CapHit(resets_at, reason) | NearCap(reason) |
   AuthFailed(status) | CliLocalError(marker) | ZeroOutputWedge | Failure(kind)` and
   **one** classifier `classify_invocation(result: AgentResult, *, strict_confirm: bool,
   backend: str) -> InvocationOutcome` holding **all** string tables adjacent
   (`CAP_HIT_PREFIXES`, `CAP_CONFIRM_KEYWORDS`, `NON_CAP_CLI_ERROR_MARKERS`,
   `CODEX_CAP_PATTERNS`, `GEMINI_CAP_PATTERNS`, `NEAR_CAP_PREFIXES`). The probe's
   deliberate **prefix-only** policy is expressed as `strict_confirm=False`, *not* a
   separate loop.

2. **`AccountPhase(StrEnum)`** — `AVAILABLE, PROBING, PROBE_IN_FLIGHT, CAPPED,
   AUTH_FAILED` (`near_cap` stays a separate annotation). A **legal-transition table**
   plus **`UsageGate._transition(acct, new_phase, *, resets_at=None, reason='')` as the
   ONLY writer** of phase state: it (a) asserts legality (raises/logs loudly on illegal
   transitions), (b) owns per-phase side effects (cancel/start resume+reprobe tasks, fire
   the cost event, stamp `pause_started_at`), (c) recomputes the global `_open` event from
   `any(a.phase in {AVAILABLE, PROBING})` **in one place** (currently 10 sites).

3. **`InvokeSlot.report(outcome: InvocationOutcome)`** — performs the matching gate
   transition **and** settles the slot atomically, making *"slot settled iff gate
   informed"* an enforced invariant. Deletes `cli_invoke`'s reach-ins to gate privates
   `_handle_auth_failure` / `_handle_cap_detected` and the manual `slot.settle()`.

4. **Delete `steward._invoke_steward`'s forked loop** and route the steward through
   `invoke_with_cap_retry`, extended with the two hooks it actually needs:
   `rebuild_prompt: Callable[[bool], Awaitable[str]]` and `max_cap_retries: int | None`.

5. **`AccountLease`** — `before_invoke()` returns an immutable lease (name+token+
   generation) instead of a bare token; `InvokeSlot` carries the lease; the
   `active_account_name` re-derivation in the invoke path is deleted. *"The account you
   report on is the account you invoked with"* becomes a type-level invariant.

6. **Probe-dir isolation (F5 cheap-now half)** — `TaskConfigDir(f'usage-gate-probe-
   {acct.name}-{os.getpid()}')` so concurrent probes stop racing one shared
   `.credentials.json`; plus a regression test pinning env-token precedence.

Plus one lifecycle invariant surfaced in `cross_system_notes`:
**`UsageGate.shutdown()` refuses to spawn new probe tasks once shutting down** (enforced
inside the gate via a `_shutting_down` guard checked by `_transition`), removing the
harness's reliance on cancel-before-shutdown call ordering.

## 4. Resolved design decisions

- **DD-1 — F5 host-wide coordination scope.** The **cheap-now** half is IN scope: unique
  probe config dir per `(account, pid)` + an env-precedence regression test (removes a
  *confirmed-in-code* shared-mutable-file race; low effort; preserves fail-safe wait).
  The **systemic** half — a file-backed / SQLite `account_status` shared cap-state with
  mtime refresh + single-prober election — is **OUT of scope** (§10), because: its
  triggering incident is *speculative* (survey: "no confirmed incident"; only the N×
  rediscovery *cost* is confirmed), it is `effort=high`, and it introduces a **new
  cross-process coordination substrate** (shared SQLite written by ~6 orchestrators +
  fused-memory, plus a distributed single-prober election) that is its own PRD-sized
  design with its own failure surface. Fail-safe wait semantics are **preserved either
  way** because `before_invoke()`'s block-until-`_open` behaviour is untouched. *(AFK safe
  default per program doc; see Open questions Q1.)*

- **DD-2 — probe prefix-only asymmetry is a parameter, not a loop.** The deliberate
  do-not-fix asymmetry between `detect_cap_hit` (prefix + `CAP_CONFIRM_KEYWORDS` guard)
  and `_run_probe` (prefix-only) is preserved as `strict_confirm=True|False` on the one
  classifier. The 20-line do-not-fix comment moves adjacent to the `strict_confirm`
  branch so the rationale survives.

- **DD-3 — `classify_agent_failure` becomes a projection.** The steward's parallel
  `AgentFailureKind` taxonomy (`cli_invoke.py:425/453`) is re-expressed as a **projection
  of `InvocationOutcome`**, not a fifth independent classifier. `AgentFailureKind`/
  `AgentFailureClass` remain the public type W9 consumes (seam-map: W9's `BlockDisposition`
  consumes `AgentFailureKind`/outcomes); only its *derivation* changes.

- **DD-4 — staged steward deletion (migration caution).** The steward-loop deletion (η)
  is gated behind the shared-loop hooks (ζ) landing green. SIGHUP reload semantics are
  preserved: today's 9-field wholesale reset becomes one `_transition(acct, AVAILABLE)`
  per account; a test asserts SIGHUP still uncaps all accounts.

- **DD-5 — `_open` invariant is the property under test.** The equivalence
  `_open.is_set() ⟺ any(a.phase in {AVAILABLE, PROBING})` is the load-bearing invariant of
  the phase machine and is asserted by a property test over random transition sequences,
  not just example-based cases.

- **DD-6 — lease generation.** `AccountLease` carries a monotonic `generation` bumped on
  every `_transition` of that account, so a stale lease (account re-transitioned mid-flight
  by another task) is detectable rather than silently mis-attributing. Token-equality
  resolution (`_resolve_account` / `_find_account_by_token`) survives only as a **fallback
  for token-only external callers** (fused-memory recon `agent_loop`/`judge` still pass
  bare tokens).

## 5. Pre-conditions for activating

- **None upstream** (brief: W4 has no upstream deps). Wave-1; may run immediately.
- **Downstream (not a precondition for W4):** W9 (wave 2) consumes W4's
  `InvocationOutcome` / `AgentFailureKind`; W9 wires its dep on W4's task ids at
  wave-2 decompose time. W4 files **no** edge to W9.
- Substrate all verified on main 2026-07-06 (§ G3 in Cross-refs / manifest).

## 6. Cross-PRD relationship (G4)

W4 **owns** the classification/phase/lease seam; the only cross-PRD consumer is W9, and
the program doc's seam map makes W4 the sole owner with no reciprocal-ownership ambiguity.

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| W9 `workflow-state-machine` | consumes | `InvocationOutcome`, `AgentFailureKind`, `classify_invocation` | **W4 (this)** | W9 wires dep at wave-2 decompose; W4 files no edge |
| W9 `workflow-state-machine` | consumes | `InvokeSlot.report`, `AccountPhase`, `AccountLease` | **W4 (this)** | same |

No other stream redefines invocation classification (seam map: "nothing else redefines
classification"). `orchestrator.usage_gate` is a re-export shim, not a consumer to
reconcile.

## 7. Contract section (B + H)

### 7.1 `InvocationOutcome` (sum type, `shared/invocation_outcome.py`)

```python
class InvocationOutcome:                # frozen dataclasses / a tagged union
    OK                                  # invocation succeeded, real tokens consumed
    CapHit(resets_at: datetime | None,  # account hit its usage cap
           reason: str)                 #   resets_at=None ⇒ parse failed (see 7.1.a)
    NearCap(reason: str)                # approaching cap; annotation only, not a phase
    AuthFailed(status: int)             # 401/403 — account not authorised
    CliLocalError(marker: str)          # local CLI error (session-id collision, etc.)
                                        #   — explicitly NOT a cap (reify-3604 class)
    ZeroOutputWedge                     # pre-first-turn wedge (zero turns produced)
    Failure(kind: str)                  # everything else
```

- **7.1.a — no silent fabrication.** `CapHit.resets_at is None` is a **first-class**
  distinction, not a `now+1h` fabrication. `_parse_resets_at`'s current silent `now+1h`
  fallback is replaced by returning `None` + a loud `resets_parse_failed` cost event; the
  scheduler/dashboard treat `None` as "unknown reset" explicitly. *(This is the F4
  observability improvement; the full prose-vs-structured inversion is a §10 follow-up —
  see Open questions Q2.)*

### 7.2 `classify_invocation`

```python
def classify_invocation(
    result: AgentResult, *, strict_confirm: bool, backend: str = 'claude',
) -> InvocationOutcome: ...
```

- **Total & pure** — no gate mutation, no I/O; a function of `result` + the two kwargs.
- Holds **all** string tables. Backend switch covers `claude` (default),
  `codex` (`CODEX_CAP_PATTERNS`), `gemini` (`GEMINI_CAP_PATTERNS`).
- `strict_confirm=True` ⇒ cap requires prefix **and** a `CAP_CONFIRM_KEYWORDS` hit
  (the `detect_cap_hit` regime); `strict_confirm=False` ⇒ prefix-only (the `_run_probe`
  regime, DD-2).
- Precedence (highest first): `AuthFailed` (4xx) → `CliLocalError`
  (`NON_CAP_CLI_ERROR_MARKERS`) → `CapHit`/`NearCap` → `ZeroOutputWedge` → `OK` →
  `Failure`. This precedence **is** the reify-3604 fix ("don't treat CLI errors as caps")
  made structural.

### 7.3 `AccountPhase` + `_transition`

```python
class AccountPhase(StrEnum):
    AVAILABLE; PROBING; PROBE_IN_FLIGHT; CAPPED; AUTH_FAILED

# The ONLY writer of acct.phase anywhere in usage_gate.py.
def _transition(self, acct, new_phase, *, resets_at=None, reason='') -> None: ...
```

Legal transitions (illegal ⇒ raise in tests / loud-log in prod):

| From \ To | AVAILABLE | PROBING | PROBE_IN_FLIGHT | CAPPED | AUTH_FAILED |
|---|---|---|---|---|---|
| AVAILABLE | – | – | – | ✓ | ✓ |
| PROBING | ✓ (claim→PIF) | – | ✓ | ✓ | ✓ |
| PROBE_IN_FLIGHT | ✓ (confirm) | – | – | ✓ | ✓ |
| CAPPED | – | ✓ (probe uncap) | – | – | – |
| AUTH_FAILED | ✓ (reprobe ok) | – | – | ✓ (demote) | – |

- **Invariants:** exactly one phase per account; `_open.is_set() ⟺ any(a.phase in
  {AVAILABLE, PROBING})` after every call (DD-5); `near_cap` cleared by the transition
  function; background-task handles (`resume_task`, `auth_reprobe_task`) are
  started/cancelled **only** inside `_transition`.
- **`AUTH_FAILED → CAPPED` demotion** (reprobe sees a cap message) keeps its ~40 lines of
  ordering intent but expressed as the single legal edge above; `capped + auth_failed`
  can no longer coexist by construction (they are one `phase`).
- **Shutdown guard (DD, cross-system note):** a `_shutting_down` flag set by `shutdown()`
  makes `_transition` **refuse to start** new resume/reprobe background tasks; a
  post-shutdown transition to `CAPPED` spawns **no** probe task.

### 7.4 `InvokeSlot.report` + `AccountLease`

```python
@dataclass(frozen=True)
class AccountLease:
    name: str; token: str | None; generation: int

def before_invoke(self) -> AccountLease | None: ...   # was: -> str | None (bare token)

class InvokeSlot:
    lease: AccountLease
    def report(self, outcome: InvocationOutcome) -> None:
        """Apply the matching gate transition AND settle the slot, atomically."""
```

- `report` maps `outcome → phase` via `_transition` (CapHit→CAPPED, AuthFailed→
  AUTH_FAILED, OK→confirm/AVAILABLE, NearCap→annotation, others→no phase change) **and**
  settles the slot — replacing the caller-discipline pair (`_handle_*` reach-in +
  `slot.settle()`). "Settled iff gate informed" is enforced, not documented.
- `slot.account_name` is derived from `lease.name` (no `active_account_name` re-derivation).
- Extended `invoke_with_cap_retry` hooks: `rebuild_prompt(session_lost: bool)` (called on
  cap-retry when the session can't be resumed) and `max_cap_retries: int | None` (bound
  for callers that must not wait 14 days; `None` = existing patient behaviour).

## 8. Boundary-test sketch (B + H — the integration-gate signal)

The integration-gate leaf (κ) is green iff this two-way suite passes. Each row faces
**both** the producer side (`UsageGate`) and a consumer side (`cli_invoke` /
`steward` / cost-store).

| # | Scenario | Preconditions | Postconditions (asserts) |
|---|---|---|---|
| B1 | Illegal transition raises | account in `AUTH_FAILED` | `_transition(acct, PROBE_IN_FLIGHT)` raises; state unchanged |
| B2 | `_open` equivalence holds | random legal transition sequence (property test) | after every step `_open.is_set() == any(phase∈{AVAILABLE,PROBING})` |
| B3 | Classifier golden corpus, per backend | historical cap/error strings from the fix commits, for `claude`/`codex`/`gemini` | each string → the expected `InvocationOutcome` variant; `strict_confirm` toggles prefix-only vs confirm-guarded |
| B4 | reify-3604 non-cap | a `NON_CAP_CLI_ERROR_MARKERS` stderr | `classify_invocation` ⇒ `CliLocalError`, **never** `CapHit`; no cap transition; no infinite retry |
| B5 | Attribution under PROBE_IN_FLIGHT skew | `account[0]` is `PROBE_IN_FLIGHT`, caller handed `account[1]` | `slot.lease.name == account[1]`; `cost_store.save_invocation` records `account[1]` (not `[0]`) |
| B6 | `report` atomicity | any non-`OK` outcome | slot settled **and** the matching `_transition` applied; no reach-in to `_handle_*`; `probe_in_flight` released on every exit path incl. exception |
| B7 | Steward inherits wedge guard | steward session returns zero turns | fresh-session fallback fires (via `rebuild_prompt`), **not** infinite re-resume; auth routing + marker exclusions active |
| B8 | SIGHUP still uncaps | all accounts `CAPPED`, then SIGHUP | each account → `AVAILABLE` via one `_transition`; `_open` set |
| B9 | Probe-dir isolation + env precedence | two concurrent probes, different accounts; env `CLAUDE_CODE_OAUTH_TOKEN` set + conflicting config-dir cred | distinct config dirs per `(account,pid)`; **observed** env token wins (escalate if not — see manifest) |
| B10 | Shutdown refuses probes | `shutdown()` called, then a `CapHit` transition | no new probe task spawned; gate-enforced, not caller-ordered |

## 9. Decomposition plan

Labels are Greek; task ids assigned at decompose. `metadata.files` is always file-level.
`shared/usage_gate.py` is edited by γ, β, δ, ε, θ, ι — these **serialize on the file
lock** by design (single-file refactor); the DAG encodes only the *semantic* order and
each task's §7 contract keeps rebases clean.

| Label | Title | Files | Prereqs | Leaf? | Observable signal |
|---|---|---|---|---|---|
| **α** | `invocation_outcome.py`: `InvocationOutcome` sum type + `classify_invocation` + per-backend golden corpus | `shared/src/shared/invocation_outcome.py` (new), `shared/tests/test_invocation_outcome.py` (new) | — | no (unlocks β, ε) | B3 golden-corpus test green: each historical string → expected variant, per backend, `strict_confirm` honoured |
| **γ** | `AccountPhase` + legal-transition table + `_transition` sole-writer; migrate all handlers; single `_open` recompute; `_shutting_down` guard | `shared/src/shared/usage_gate.py`, `shared/tests/test_usage_gate.py` | — | no (unlocks β, δ, ε, θ, ι) | B1 illegal-raise + B2 `_open` property + B8 SIGHUP-uncap + B10 shutdown-refuses-probe green |
| **β** | Rewire `cli_invoke` + `usage_gate.detect_cap_hit`/`_run_probe`/`classify_agent_failure` onto `classify_invocation` (probe = `strict_confirm=False`); collapse the 5 sites | `shared/src/shared/cli_invoke.py`, `shared/src/shared/usage_gate.py`, `shared/tests/test_cli_invoke.py` | α, γ | no (unlocks ε, κ) | single-source grep: `CAP_HIT_PREFIXES`/`CAP_CONFIRM_KEYWORDS`/`NON_CAP_CLI_ERROR_MARKERS` live only in `invocation_outcome.py`; B4 reify-3604 regression green |
| **δ** | `AccountLease` + `before_invoke` returns lease; `InvokeSlot` carries lease; delete `active_account_name` re-derivation | `shared/src/shared/usage_gate.py`, `shared/src/shared/cli_invoke.py`, `shared/tests/test_usage_gate_exhaustive.py` | γ | no (unlocks ε, κ) | B5 attribution test green: PROBE_IN_FLIGHT-skew reports invoked account in `lease.name` + cost store |
| **ε** | `InvokeSlot.report(outcome)` atomic transition+settle; delete `cli_invoke` reach-ins to `_handle_auth_failure`/`_handle_cap_detected` + manual `slot.settle()` | `shared/src/shared/usage_gate.py`, `shared/src/shared/cli_invoke.py`, `shared/tests/test_cap_retry.py` | β, δ | no (unlocks ζ, κ) | B6: grep shows zero reach-ins/`slot.settle()` in `cli_invoke.py`; "settled iff gate informed" + probe-slot-leak-on-exception green |
| **ζ** | Extend `invoke_with_cap_retry` with `rebuild_prompt` + `max_cap_retries` hooks | `shared/src/shared/cli_invoke.py`, `shared/tests/test_cap_retry.py` | ε | no (unlocks η) | `max_cap_retries` bound raises after N (not 14-day wait); `rebuild_prompt` invoked on unresumable cap-retry (stubbed unit test) |
| **η** | Delete `steward._invoke_steward` fork; route steward through `invoke_with_cap_retry` (rebuild_prompt = fresh-escalations prompt; max_cap_retries bound) | `orchestrator/src/orchestrator/steward.py`, `orchestrator/tests/test_steward.py` | ζ | no (unlocks κ) | B7: steward inherits wedge guard (zero-turn ⇒ fresh-session fallback, not infinite resume); grep shows no `invoke_slot`/`detect_cap_hit` direct calls in `steward.py` |
| **θ** | Probe config dir unique per `(account, pid)`; env-precedence regression test | `shared/src/shared/usage_gate.py`, `shared/tests/test_usage_gate.py` | γ | no (unlocks κ) | B9: distinct config dirs per `(account,pid)`; env token precedence **observed** (escalate if config-dir wins) |
| **ι** | `UsageGate.shutdown()` refuses new probe tasks (`_shutting_down` enforced inside gate) | `shared/src/shared/usage_gate.py`, `shared/tests/test_usage_gate_exhaustive.py` | γ | no (unlocks κ) | B10: post-`shutdown()` `CapHit` transition spawns no probe task (gate-enforced) |
| **κ** | Integration gate: the §8 two-way boundary suite (end-to-end selection→lease→classify→report→transition through a fake-CLI harness) | `shared/tests/test_invocation_outcome_boundary.py` (new) | ε, η, θ, ι | **yes** | Full B1–B10 boundary suite green facing both producer and consumer sides |

DAG edges: β→{α,γ}; δ→γ; ε→{β,δ}; ζ→ε; η→ζ; θ→γ; ι→γ; κ→{ε,η,θ,ι}. Roots: α, γ.
Sole leaf: κ.

## 10. Out of scope

- **F5 systemic host-wide cap-state coordination** — shared/SQLite `account_status`
  table with mtime refresh + single-prober election (DD-1). Speculative incident,
  `effort=high`, new cross-process substrate deserving its own PRD. Fail-safe wait
  preserved without it.
- **W9's exception-ladder consumption** of these types (`BlockDisposition`,
  `WorkflowStateMachine`) — wave 2, W9-owned.
- **Any change to what the Claude CLI emits** — external substrate; the whole prose-
  scraping stack exists because the CLI gives no structured contract. This PRD converts
  the arms race into a *bounded, observable* fallback (7.1.a) but cannot eliminate it.
- **Full F4 prose→structured inversion** (primary classification from structured
  `AgentResult` fields with prose as monitored fallback + versioned per-CLI fixture
  corpus). This PRD does the observable-fallback half (loud `resets_parse_failed` /
  `cap_prose_only` events + `resets_at=None` first-class); the full inversion + CI
  fixture-corpus-per-CLI-bump is a follow-up (Open questions Q2).
- **`orchestrator.usage_gate` shim** — pure re-export, untouched.
- **fused-memory recon `agent_loop`/`judge`** token-only callers — keep working via the
  `_resolve_account` token fallback (DD-6); not migrated to leases here.

## 11. Open questions (tactical — surfaced, not blocking; AFK safe-defaults taken)

1. **Q1 — F5 systemic coordination.** Deferred out-of-scope per DD-1 (safe default:
   ship the cheap-now probe-dir isolation θ, defer the shared cap-state substrate). If a
   *confirmed* concurrent-probe misattribution or a costly N× rediscovery incident lands,
   promote the systemic half to its own PRD. **Decide when:** first confirmed F5 incident.
2. **Q2 — F4 full inversion.** 7.1.a ships the observable-fallback half; the full
   structured-primary inversion + versioned CLI fixture corpus is deferred. **Suggested
   resolution:** file as a follow-up once the CLI exposes any structured rate-limit field
   (`_parse_claude_output` extension point). **Decide during:** a future CLI-contract PRD.
3. **Q3 — `max_cap_retries` value for the steward.** η passes a bound so a wedged steward
   can't wait 14 days. **Suggested resolution:** reuse today's `_MAX_CAP_RETRIES = 16`
   (steward.py:54) as the bound to preserve current behaviour. **Decide during:** η impl.
4. **Q4 — `AccountLease.generation` staleness policy.** DD-6 makes stale leases
   *detectable*; whether a stale-lease `report` should hard-error vs log-and-proceed is
   tactical. **Suggested resolution:** log-and-proceed (fail-safe) + a cost event, matching
   the project's loud-observable-over-hard-fail norm. **Decide during:** δ/ε impl.
5. **Q5 — golden-corpus source of truth.** B3's corpus is extracted from the fix-commit
   strings named in the finding histories. **Suggested resolution:** materialise it as a
   checked-in fixture (`shared/tests/fixtures/cap_strings/`) so future CLI wordings append
   there. **Decide during:** α impl.
