# Operational-ask routing: sub-classify `operational_mode`, boundary-enforce coercion

**Status:** active — 2026-07-19. Contract extending the shipped operational-ask
coercion (tasks 2085 substring registry, 2687 execution_class axis, 2225/η
execution_class declaration guard). Greenlit in the analyse-then-discuss
session `discuss-operational-deterministic-coercion` (esc-2785-2).

## Goal

An `execution_class='operational'` ask is routed **by what can execute it**, and
that routing is a **submit-boundary invariant that no path can bypass**:

- An operational ask that is a mechanical **gate** (run a committed script /
  purely gate on a human) — and every `execution_class='decision'` ask — is
  coerced to `task_kind='deterministic'` + `always_escalates=true` at the
  submit boundary, on **every** submit path (normal, `planning_mode`, direct),
  surviving a stale registry or a curator error.
- An operational ask that needs **LLM judgment** with live-store tools and no
  repo diff (Mem0 curation, triage) — `operational_mode='llm'` — routes to a
  **distinct human-gate** whose born-at-L2 escalation is unambiguously marked
  "needs LLM-operational handling; no automated lane yet", instead of bouncing
  off the code-first architect (the recurring esc-2785-2 failure) or being
  silently dumped on a human as an ordinary gate.
- The recon stage that *discovers* memory-curation work — it already holds the
  memory-mutation tools — **completes the safe merges inline** and files only
  the residual irreversible judgment call as an `operational` +
  `operational_mode='gate'` task, so the redundant relay-then-bounce class is
  cut off at its origin.

User-observable: `get_task` on a `planning_mode`-filed `operational` task shows
`task_kind='deterministic'` (proving the boundary — not the fail-open curator —
coerced it); an `operational_mode='llm'` task yields a born-at-L2 escalation
carrying a stable marker token and is never dispatched to an architect.

## Background

Today the operational→deterministic coercion lives **only** in the
`TaskCurator` (`_maybe_route_deterministic` → `route_deterministic` →
`task_interceptor._inject_deterministic_pure_gate`), a best-effort, fail-open,
per-process-cached component. Two defects follow (both verified in code, session
`discuss-operational-deterministic-coercion`):

1. **The routing is bypassable.** The *declaration* (`execution_class`) is
   hard-enforced at the submit boundary (`execution_class_guard`, task 2225/η),
   but the *routing coercion* is soft — skipped on a stale/unreadable registry,
   a curator error, a process running pre-2687 code, or the **`planning_mode`
   curator-bypassing path**. esc-2785-2 fired because the live fused-memory
   process pre-dated the 2687 axis (curator caches the registry once per
   process); the mechanical recurrence self-healed on restart, but the
   bypass surface is structural.
2. **The one target mis-serves the LLM-judgment class.** `operational` →
   deterministic **pure-gate** = "file a born-at-L2 escalation, block until a
   human does it by hand." For Mem0 editorial curation (task 2785: case-by-case
   judgment over 12 memory entries) that forces a human to do LLM-able work. No
   orchestrator role runs an LLM agent on a zero-repo-footprint task — every
   `ROLES` entry (`architect`/`implementer`/`simple_task`/…) is code-first
   (produces/verifies/merges a diff). So the `operational_llm` class is
   unserved: the architect bounces it, the pure-gate offloads it.

Crucially, the recon stages that *file* these tasks already hold the exact
tools: Stage 1 (`memory_consolidator`) and Stage 2 (`task_knowledge_sync`) both
have `add_memory`/`delete_memory`/`merge_entities`
(`cli_stage_runner.py` `DISALLOW_MEMORY_WRITES` applies only to Stage 3). Stage
2 in fact already performed most of task 2785's curation inline; the filed task
was a redundant relay that then bounced.

## Sketch of approach

Five mechanisms:

1. **`metadata.operational_mode: 'gate' | 'llm'`** — a recognized, validated
   metadata field (handled exactly as `execution_class` is — a known field, not
   a `_BLESSED_METADATA_KEYS` allowlist entry, so `model_dump()` grows no
   `None`-valued noise), default absent ≡ `'gate'` (backward-compatible: every
   existing `operational` task stays a gate, no migration). Orthogonal to
   `execution_class` — **not** an enum split of it.
2. **`inject_operational_routing(metadata)`** — a deterministic submit-boundary
   transform in `tools.py:submit_task`, placed immediately after
   `inject_execution_class` (plus the interceptor-side twin for the direct
   path), that maps the *declared* `(execution_class, operational_mode)` to
   routing metadata **before the ticket blob is persisted**. Because it runs in
   `tools.py:submit_task` before the interceptor's `planning_mode` branch, it
   fires on every path. Reuses `_inject_deterministic_pure_gate` (no lock-step
   re-implementation of the pure-gate stamp).
3. **`operational_mode='llm'` distinct human-gate** — the injector stamps a
   stable marker on the `llm`-mode task; the `DeterministicRunner`'s pure-gate
   born-at-L2 escalation surfaces that marker in a distinguishable
   summary/detail (a fixed token, e.g. `operational_llm_needs_lane`) so a human
   (and any future automation) can tell an LLM-operational gate from a plain
   one. Phase-1 target; the general non-architect LLM lane is **out of scope**
   (deferred until a *non-memory* `operational_llm` ask recurs).
4. **Demote the curator's execution_class axis to a legacy fallback** — with
   the boundary owning tagged routing, `operational_ask_registry.match_candidate`'s
   execution_class axis (task 2687) no longer fires for tagged asks (it would be
   a redundant second coercion site — INV-5). The fuzzy **substring** entries
   stay in the curator as the *untagged-legacy* fallback only.
5. **Recon source-completion** — recon Stage 1/2 prompt/brief (`recon_self_model`
   render + stage prompts) instructs the tool-holding stage to complete the
   *safe* memory merges inline and file only the *residual irreversible
   judgment call* as an `operational` + `operational_mode='gate'` task, and to
   declare `operational_mode` on its `submit_task` calls.

Net: the human-gate is reached only for genuinely human decisions (irreversible
judgment, go/no-go) — never for LLM-able bulk work or code changes.

## Resolved design decisions

1. **Orthogonal discriminator, not an enum split** (Leo, 2026-07-19). Keep
   `execution_class='operational'` unchanged; add
   `operational_mode: 'gate'|'llm'` (default `'gate'`). Backward-compatible with
   zero migration; mirrors how `before_done.kind` was added alongside the
   existing deterministic contract. Rejected: splitting `EXECUTION_CLASSES` into
   `operational_gate`/`operational_llm` (hard-enforced enum → migration + prompt
   churn for in-flight `operational` tasks).
2. **Coercion is a boundary invariant, not a curator decision** (Leo,
   2026-07-19). Move the *authoritative* `(execution_class, operational_mode)`→
   routing coercion out of the fail-open curator to a deterministic
   submit-boundary inject. The curator keeps only the untagged-substring
   fallback. This is what makes it unbypassable (incl. `planning_mode`).
3. **`operational_llm` phase-1 = distinct human-gate; general LLM lane
   deferred.** Building a new `task_kind`/agent-role LLM lane is deferred until a
   *non-memory* `operational_llm` ask recurs (the registry's own "only after it
   RECURS" discipline). The dominant recurring case — memory curation — is
   handled by decision #4 instead, so no new lane is needed now.
4. **Memory curation is source-completed, not lane-served.** The recon stage
   that already holds the tools does the safe merges inline and gates only the
   residual; this cuts the redundant relay-then-bounce class off at origin and
   is why the LLM lane can be deferred.
5. **`llm`-mode marker is a structured token, not a log-scrape** (INV-2). The
   distinct escalation reason is a fixed token the runner emits from the stamped
   marker — not a phrase a consumer must parse out of free text.

## Pre-conditions for activating

None — all substrate exists on `main`:
- `execution_class` field handling + `execution_class_guard`
  (`inject_execution_class`) — the pattern `operational_mode` mirrors.
- `_inject_deterministic_pure_gate`, the deterministic `task_kind` contract, and
  the pure-gate born-at-L2 escalation (`DeterministicRunner`).
- `tools.py:submit_task` guard chain (`inject_task_kind` → `inject_execution_class`)
  running before the interceptor's `planning_mode` branch.
- Recon Stage 1/2 memory-mutation tool grants (`cli_stage_runner.py`).

## Cross-PRD relationship

No cross-PRD seams. Self-contained within `fused-memory` (middleware + recon
prompts + shared metadata) and one `orchestrator` module
(`deterministic_runner.py` escalation reason). Adjacent-but-independent:
`plans/capability-delivered-checks-prd.md` (this PRD's manifest sidecar uses that
machinery; no shared mechanism).

## Approach: B + H

High-stakes (touches the `submit_task` boundary — a load-bearing seam — and a
producer→consumer routing seam) → contract + boundary-test sketch below.

### Contract

**Field.** `metadata.operational_mode ∈ {'gate','llm'}`, optional; absent ≡
`'gate'`. Recognized + validated by `parse_metadata`
(`shared/src/shared/task_metadata.py`) exactly as `execution_class` is; an
invalid value is a `ValidationError` at the write boundary, never a lint
warning. Meaningful only when `execution_class='operational'`; ignored (recorded
but inert) otherwise.

**Boundary transform `inject_operational_routing(metadata) -> dict`.** Pure,
deterministic, no I/O. Runs in `tools.py:submit_task` immediately after
`inject_execution_class` (and the interceptor-side twin). Resolution table
(first match):

| `execution_class` | `operational_mode` | Result |
|---|---|---|
| `operational` | `gate` (or absent) | `_inject_deterministic_pure_gate` (task_kind=deterministic, always_escalates=true, before_done dropped) |
| `decision` | — | `_inject_deterministic_pure_gate` |
| `operational` | `llm` | `_inject_deterministic_pure_gate` **+** stamp `llm`-gate marker (drives the distinct escalation reason) |
| `code_tdd` / absent | — | untouched (architect/TDD path) |

Idempotent (re-running on an already-coerced task is a byte-identical no-op) and
override-safe (a hand-supplied `task_kind`/`before_done` is overwritten to the
pure-gate shape, matching `_inject_deterministic_pure_gate`'s existing contract).

**Distinct escalation reason.** For an `llm`-marked pure-gate, the
`DeterministicRunner` born-at-L2 escalation carries a stable token
(`operational_llm_needs_lane`) in its summary/detail (and/or a distinct
category), so the gate is machine-distinguishable from a plain gate.

**Curator demotion.** `match_candidate`'s execution_class axis no longer routes
a tagged `operational`/`decision` candidate (the boundary already did); the
substring entries remain for untagged legacy asks.

### Boundary-test sketch (the ζ integration matrix)

| Scenario | Preconditions | Postconditions asserted |
|---|---|---|
| gate via normal submit | recon-stage submit, `execution_class=operational`, `operational_mode=gate` | filed task `task_kind=deterministic`, `always_escalates=true`, no `before_done` |
| gate via **planning_mode** | same, `planning_mode=True` | same — proves the boundary (not curator) coerced; curator never consulted |
| `decision` | `execution_class=decision` | `task_kind=deterministic` pure-gate |
| **llm** | `execution_class=operational`, `operational_mode=llm` | `task_kind=deterministic`; born-at-L2 escalation detail contains `operational_llm_needs_lane`; task never routed to `architect` |
| untagged legacy | no `execution_class`, title/desc matches a substring registry entry | curator still routes to a pure-gate (fallback intact) |
| invalid mode rejected | `operational_mode='bogus'` | `submit_task` returns `ValidationError`; no task created |
| recon source-completion | recon stage finds a duplicate-cluster curation need | rendered stage prompt instructs inline safe-merge + files only residual as `operational_mode=gate` |

## Decomposition plan

- **α — `operational_mode` schema field + validation.** Add `operational_mode`
  as a recognized, validated metadata field (mirror `execution_class`) in
  `shared/src/shared/task_metadata.py`; `{gate,llm}` accepted, default absent≡gate,
  invalid rejected with `ValidationError`; the fused-memory write boundary honors
  it. *Modules:* `shared`. *Signal (leaf):* `submit_task` with
  `operational_mode='bogus'` → `ValidationError`; `'gate'`/`'llm'` persist with no
  `task_metadata.schema_warning`. *Prereqs:* —.
- **β — submit-boundary `inject_operational_routing`.** New deterministic inject
  in `tools.py:submit_task` after `inject_execution_class` + the interceptor
  twin; implements the contract table; reuses `_inject_deterministic_pure_gate`.
  *Modules:* `fused-memory` (`server/tools.py`, `middleware/task_interceptor.py`,
  new `middleware/operational_routing_guard.py`). *Signal (leaf):* a
  `planning_mode` submit with `execution_class=operational,operational_mode=gate`
  → `get_task` shows `task_kind=deterministic`+`always_escalates` (proves the
  boundary coerced, not the curator). *Prereqs:* α.
- **γ — `operational_llm` distinct human-gate.** `llm`-mode marker → the
  `DeterministicRunner` pure-gate born-at-L2 escalation carries the stable
  `operational_llm_needs_lane` token; the task is never dispatched to an
  architect. *Modules:* `fused-memory` (injector marker), `orchestrator`
  (`deterministic_runner.py` escalation reason). *Signal (leaf):* an
  `operational`+`llm` submit yields a born-at-L2 escalation whose detail contains
  `operational_llm_needs_lane`; `task_kind=deterministic`. *Prereqs:* β.
- **δ — demote curator execution_class axis to legacy fallback.** `match_candidate`
  no longer routes a tagged `operational`/`decision` candidate (boundary owns it);
  substring entries stay for untagged asks. *Modules:* `fused-memory`
  (`operational_ask_registry.py`, `task_curator.py`). *Signal (leaf):* a tagged
  operational candidate is NOT re-routed by the curator (boundary already
  coerced); an untagged substring-matching ask still routes via the curator.
  *Prereqs:* β.
- **ε — recon source-completion brief.** Update recon Stage 1/2 prompt
  (`recon_self_model` render + stage prompts) to instruct the tool-holding stage
  to complete safe memory merges inline and file only the residual irreversible
  judgment as `operational`+`operational_mode=gate`, declaring `operational_mode`
  on its submits. *Modules:* `fused-memory` (reconciliation prompts). *Signal
  (leaf):* a rendered-prompt test asserts the source-completion + `operational_mode`
  guidance is present in the Stage 1/2 prompt text. *Prereqs:* α.
- **ζ — boundary integration gate (B+H).** The boundary-test matrix above, green
  in CI. *Modules:* `fused-memory` tests (+ `orchestrator`). *Signal (leaf):* the
  boundary-matrix suite is green. *Prereqs:* β, γ, δ, ε.

DAG: α → β → {γ, δ}; α → ε; {β, γ, δ, ε} → ζ.

## Out of scope for this PRD

- **A general non-architect LLM-operational lane** (new `task_kind`/agent role
  running an LLM agent with live-store tools, no diff, closing via
  `done_provenance.kind='operational-verified'`). Deferred until a *non-memory*
  `operational_llm` ask recurs; phase-1 routes `operational_llm` to a distinct
  human-gate instead. When it recurs, that PRD consumes the
  `operational_llm_needs_lane` marker γ emits as its trigger.
- **Auto-executing irreversible memory mutations from recon** — recon
  source-completes only the *safe* merges; irreversible judgment calls remain a
  human gate by design.
- **Migrating existing `operational` tasks** — none needed (default ≡ gate).

## Open questions (surfaced but not decided in this session)

1. **Exact `llm`-marker carrier.** Whether the `llm` marker is a dedicated
   metadata field vs. an `x_`-namespaced note vs. a description-prepended banner,
   and whether the distinct escalation uses a new `category` or only a detail
   token. Tactical — γ picks the least-surface option that keeps the token
   machine-checkable (INV-2). Decide during γ.
2. **Where the recon "safe vs irreversible" line is drawn.** The precise
   predicate a recon stage uses to decide inline-merge vs escalate-as-gate.
   Tactical — ε encodes the conservative rule already embodied by Stage 2's
   task-2785 behavior (merge exact/near-exact duplicates inline; escalate any
   content-losing judgment call). Decide during ε.
