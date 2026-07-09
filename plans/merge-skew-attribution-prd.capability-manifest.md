# Capability manifest — merge-skew-attribution-prd

Mechanizes G3 (assumed-substrate verified) + G6 (premise validity) for
`plans/merge-skew-attribution-prd.md`. One block per task; each capability the
task's signal asserts is bound to evidence. Any FAIL value blocks the batch.

Verified 2026-07-09 against `orchestrator/src/orchestrator/` on main tip
`e060990b04`. Grep line numbers are anchors, not guarantees of stability.

Legend: `grep:file:line wired` = referenced from the production entry path;
`producer:task-<label>` = delivered by an upstream batch task in the dep closure;
`existing` = stdlib / git plumbing / already-wired MCP.

---

## α — `classify_merge_failure_disposition` + `SkewEvidence` + `MergeOutcome.disposition`

Intermediate (foundation). Roped into the β integration gate (G2 escape hatch).
Producer of the net-new type surface — those symbols are α's *deliverable*, not a
substrate dependency.

| Capability asserted | Binding | Verdict |
|---|---|---|
| `verify_failure_is_preexisting_on_main` (I1 probe order; classifier only refines the non-preexisting bucket) | `grep:verify.py:3673` def; `grep:merge_queue.py:165` import, `:730` called at the gate | PASS (wired) |
| `VerifyResult` (contract input type) | `grep:verify.py:2062` class def; `cause_hint`/per-test fields `:499,:974` | PASS |
| git plumbing (`merge-base`, `log --name-only`, `diff`) — the whole classifier is git-only, read-only (I2/D2) | `existing` (subprocess) | PASS |
| **"branch's most recent pre-merge verify was green" fact (I5 precondition for INTEGRATION_SKEW)** | `producer:event-store history` — worker has `event_store` (`grep:harness.py:3277`); precedent: task 1720 (done) gates on the branch's terminal merge/verify outcome from the event store. NOT scheduler-only task metadata (out of worker reach by design, per 1720). | PASS **with noop-trap note** ↓ |
| net-new: `MergeFailureDisposition`, `SkewEvidence`, `MergeOutcome.disposition` field (default `INDETERMINATE`) | α is the **producer** (anti-inversion: adding the `disposition` field here lets both β and γ depend only on α) | PASS (α delivers) |

**Noop-trap note (G6 branch-3, field-population twin):** INTEGRATION_SKEW fires
only when the "branch pre-merge verify green" fact is *readable* at the classifier
call site. If that value is never populated/readable, the verdict is dead and the
whole PRD is a silent no-op (field declared, never populated). α must **verify at
least one source (event-store history) actually yields the fact** before declaring
done; if truly unavailable, degrade to INDETERMINATE (fail-open, I3) — but that
degradation must be a proven fallback, not the default outcome.

## β — wire disposition into the merge-gate failure path + surfaces (THE integration gate)

Intermediate (ε depends on it) **and** the B+H integration-gate leaf — its signal is
§Boundary-test rows 1–4 + end-to-end (escalation body + `merge_status`).

| Capability asserted | Binding | Verdict |
|---|---|---|
| α's `classify_merge_failure_disposition` + enum + `MergeOutcome.disposition` | `producer:α` — α upstream of β (DAG-direction OK) | PASS |
| merge-gate failure call site (invoke classifier after the existing preexisting probe returns "not preexisting", I1) | `grep:merge_queue.py:730` (probe call in the gate; classifier slots in after) | PASS (wired) |
| `MergeOutcome.failure_diagnostic` surface (I4) | `grep:merge_queue.py:8509` `_build_merge_failure_diagnostic`, `:8539` render, `:7740` set on outcome | PASS (wired) |
| skew escalation body (sibling of the "fix main:" composer) + distinct category | `grep:merge_queue.py:514` fix-main composer + `:709-751` preexisting escalation/dedup path; `escalate_info`/`escalate_blocker` MCP `existing` | PASS |
| task block reason + dry-run-proposal context (I4: "port landed commit X, don't hunt your own diff") | block-reason machinery `existing` (workflow routes on reason prefixes) | PASS |
| **populate `MergeOutcome.disposition` in production** so γ's runs.db field is non-sentinel (anti-orphan wiring for γ) | β writes the computed disposition onto the shared outcome; γ reads it at emission | PASS **with anti-orphan note** ↓ |

**Anti-orphan note (G1/G5):** γ builds the runs.db `disposition` field; β is the
producer that *populates* it on the production path. β's wiring MUST set
`MergeOutcome.disposition`, else γ's field ships permanently `INDETERMINATE` (an
orphan producer). Cross-referenced in γ.

**2357 constraint (G4):** any skew verdict keys on **dispatch-time base facts**
(`merge_base(branch,main)`, `branch_base_sha`), never the snapshot
`two_layer_invariants` surface (stale-cache artifact — see DF 2357).

## γ — disposition in runs.db events + stats separation  (LEAF)

| Capability asserted | Binding | Verdict |
|---|---|---|
| α's `MergeFailureDisposition` enum + `MergeOutcome.disposition` | `producer:α` upstream | PASS |
| runs.db merge event write site (add `disposition` to the payload) | `grep:merge_queue.py:1492` `_emit_merge_attempt` (production emission path; called `:1606,:1647,:2704,:2732`) | PASS (wired) |
| persisted-field readable through the stats/dashboard read path (row 7: `GROUP BY disposition`) | `event_store` persistence `existing`; γ adds the stats query | PASS |
| field-population (non-sentinel) | γ's own signal injects a real disposition (α's enum) → sqlite3 shows non-null; **production** population is β's job (see β anti-orphan note) | PASS |

γ depends on **α only** (parallel to β). Its signal is provable independently by
emitting one event with a disposition set; it does not require β's forced-skew
scenario. Physical note: γ and β both edit `merge_queue.py`, so the narrow-file
lock serializes them — not a logical dependency.

## δ — pipeline-landing tripwire (M3)  (LEAF)

Advisory only (I6): ≤1 info escalation per landing; oracle absent/erroring → logged
no-op; never delays advance.

| Capability asserted | Binding | Verdict |
|---|---|---|
| post-advance hook site (after a landing) | `grep:merge_gates.py` `_finalize_advanced_merge` + `merge_queue.py` push_main/advance path `existing` (δ adds the hook) | PASS |
| load-bearing oracle `verify-pipeline-guard.sh requires-full-gate <files...>` (exit-code contract) | `producer:reify` — reify-owned primitive, **already consumed by DF done task 1774**; exercised 2026-07-08 (PRD precond). Config-gated per project; absent → no-op | PASS |
| per-project oracle-command config knob | δ's own deliverable (new config leaf) | PASS (δ delivers) |
| `escalate_info` (one info escalation naming the landing + overlapping tasks) | escalation MCP `existing` | PASS |
| in-flight branch-diff overlap vs load-bearing set (git diff) | `existing` git plumbing; in-flight-set source + overlap definition are tactical (Open Q2) — advisory/fail-open covers unavailability | PASS |
| attach steward-visible note to overlapping tasks' metadata | `update_task` MCP `existing` | PASS |

## ε — docs/triage guidance  (LEAF, simple)

| Capability asserted | Binding | Verdict |
|---|---|---|
| escalation-watcher triage guidance surface (add disposition vocabulary + "skew ⇒ port, don't debug") | `grep:skills/escalation-watcher/SKILL.md` (exists; also escalation-watcher-auto) | PASS |

No numeric/exactness/rejection/end-to-end premise — pure operator-facing docs
surface. Depends on β (the vocabulary it documents must exist first).

---

## Result

No FAIL bindings. Two G6 notes baked into task descriptions (α noop-trap;
β↔γ anti-orphan population). Batch clears G3 + G6. reify-side M4 (`/prd` overlay
ordering check) is deliberately **not** in this batch — it is a reify-tree task.
