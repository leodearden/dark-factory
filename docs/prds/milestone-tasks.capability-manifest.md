# Capability manifest — milestone-tasks.md

Per-leaf capability→evidence bindings mechanizing G3 (assumed-substrate) + G6
(premise validity). Any FAIL binding blocks the batch until resolved. Grep
evidence is against `main` as of 2026-07-08. **Status: DRAFT** — re-verify each
grep at decompose time (drift check).

Evidence commands used:
- wiring grep: `grep -n '<symbol>' orchestrator/src/orchestrator/scheduler.py` (etc.)
- provenance kind: `shared/src/shared/task_metadata.py:68` (`DoneProvenance.kind` Literal)
- empty-value sentinel: N/A (no field-population signal in this PRD — the
  predicate verdict is an exit code, not a sampled result field)

---

## α — Typed milestone + predicate + provenance schema (shared)

| Capability asserted by signal | Binding | Evidence | Verdict |
|---|---|---|---|
| `register_metadata_submodel` extension point exists | wired | `shared/src/shared/task_metadata.py:260` | **PASS** |
| `TaskMetadata`/`BeforeDone` carry `extra='allow'` (new field survives round-trip) | wired | `task_metadata.py:226,48` | **PASS** |
| `DoneProvenance.kind` is the single valid-kinds authority (adding a kind is local) | wired | `task_metadata.py:68` Literal + `deterministic_runner.py:180` `_build_done_provenance` imports it | **PASS** |
| `Milestone` sub-model + `'deterministic-milestone'` kind + `BeforeDone.kind` | producer:α (self-delivered) | this task adds them; grep after landing | **PASS** (self) |
| `parse_metadata(direction='read/write')` warns/rejects per enforce=true | wired | `task_metadata.enforce=true` live (commit ee97613a96) | **PASS** |

*G6:* no numeric/exactness/rejection premise beyond α's own schema validators
(self-delivered). DAG-direction: α is upstream of β,γ,δ — correct.

## β — Scheduler milestone time-gate (orchestrator)

| Capability asserted by signal | Binding | Evidence | Verdict |
|---|---|---|---|
| `_eligible_for_dispatch` is the single gate chain for both dispatch loops | wired | `scheduler.py:2857`, called from both scored + pin loops | **PASS** |
| `_deferred_watch_gated` sibling-gate pattern to extend | wired | `scheduler.py:2824,2902` | **PASS** |
| `_deps_satisfied` evaluator (reused for the anchor) | wired | `scheduler.py:2596`; already used at `:2907` | **PASS** |
| per-tick sweep hook in `acquire_next` (mirror `_gc_expired_cooldowns`) | wired | `scheduler.py:2807`, called once/tick before candidate loops | **PASS** |
| wall-clock `datetime.now(UTC)` available in scheduler (NOT `_time_source`) | wired | `scheduler.py:356,628,4202` use `datetime.now(UTC)`; `_time_source`=`time.monotonic` (`:433`) — **must not** be used for the anchor | **PASS** |
| `update_task(..., metadata_mode='merge')` to stamp the anchor | wired | used by DeterministicRunner `deterministic_runner.py:1072` | **PASS** |
| `metadata.milestone` typed shape | producer:α (upstream dep) | α delivers `Milestone`; DAG-direction correct | **PASS** |

*G6 branch 3 (end-to-end):* every capability the β signal needs is delivered by α
(upstream) or exists on main. No downstream-owned capability. **PASS.**

## γ — Predicate deterministic mode (orchestrator)

| Capability asserted by signal | Binding | Evidence | Verdict |
|---|---|---|---|
| `DeterministicRunner.run` before-deploy branch point | wired | `deterministic_runner.py:919` (`before_done is not None`) — insert predicate branch above it | **PASS** |
| `_default_run_script` runs a script under timeout, returns `(rc, tail)` | wired | `deterministic_runner.py:446` | **PASS** |
| born-at-L2 escalation submit with sentinel role (level stays 2) | wired | `deterministic_runner.py:583`, role `orchestrator-deterministic` (`:177`) | **PASS** |
| escalation `category` accepts a new value `milestone_check_failed` | wired | `escalation/.../models.py:55` category is open `str` | **PASS** |
| section-1 resume-to-done machinery (quiescence + resolve→done) | wired | `deterministic_runner.py:862-914` | **PASS** |
| `deterministic-milestone` provenance kind | producer:α (upstream dep) | α adds it to `DoneProvenance.kind`; `_build_done_provenance` raises if absent → hard dep | **PASS** |

*G6 branch 4 (verdict):* the "exit 1 → escalation fires" is a **positive**
capability (escalation queue submit exists) delivered by γ itself + α's kind.
No false rejection premise (the orchestrator asserts no threshold; the script
owns it). DAG-direction: γ depends on α — correct. **PASS.**

## δ — submit_task validation (fused-memory)

| Capability asserted by signal | Binding | Evidence | Verdict |
|---|---|---|---|
| `deterministic_task_error` guard extension point at submit boundary | wired | `deterministic_task_guard.py:115` | **PASS** |
| `_validate_before_done` structural/fs checks to extend for `kind` | wired | `deterministic_task_guard.py:193` | **PASS** |
| structured `ValidationError` dict return convention | wired | `deterministic_task_guard.py:99` `_validation_error` | **PASS** |
| rejection mechanism fires on malformed milestone (B11) | producer:δ (self) + `rejection-check` | δ builds the rejection; its test authors each malformed submit and observes the `ValidationError` — bind the observed rejection, do NOT log it as mere motivation | **PASS** (self, must-observe) |
| `metadata.milestone` typed shape (for the well-formed-persists half) | producer:α (upstream dep) | α delivers `Milestone` | **PASS** |

*G6 branch 4 (rejection):* the signal asserts malformed specs are **rejected**.
δ IS the task that builds the rejection, so the capability is self-delivered;
the manifest binds `rejection-check` — the δ test must **author X and observe the
`ValidationError` fire**, never assert absence-of-persist alone. **PASS.**

## ε — End-to-end integration gate (orchestrator) — LEAF

| Capability asserted by signal | Binding | Evidence | Verdict |
|---|---|---|---|
| time-gate holds/releases (B1–B6,B12) | producer:β (upstream) | β in transitive dep closure; covers the scheduler-gate extent | **PASS** |
| predicate verdict paths (B7–B10) | producer:γ (upstream) | γ covers the runner-verdict extent | **PASS** |
| submit validation (B11) | producer:δ (upstream) | δ covers the validation extent | **PASS** |
| exemplar check script `scripts/check_merge_flakiness.sh` exists+executable | **producer:ε (self)** | ε authors the exemplar script/fixture; before_done guard requires it exist+`X_OK` at submit — ε's own test creates it | **PASS** (self) |

*G6 branch 3:* every capability ε's signal needs is delivered by β/γ/δ (all
**upstream** — DAG-direction correct) or authored by ε itself (the exemplar
fixture). No capability owned by a task that depends on ε. **PASS.**

## ζ — Operator docs (docs)

Docs-only leaf roped to ε's shipped surface (documents behaviour ε proves). No
substrate capability asserted beyond "the §5/§6 behaviour ε landed exists."
Signal = `CLAUDE.md` names the milestone schema + exit-code contract. **PASS.**

---

## Summary

No FAIL bindings. Every runtime capability is either wired on `main` (verified
grep) or delivered by an **upstream** task in the dependency closure (α→{β,γ,δ}→ε→ζ).
The two self-delivered rejection/fixture capabilities (δ's `ValidationError`,
ε's exemplar script) are bound to must-observe evidence, not motivation. Batch is
clear to queue once α's schema lands and the greps are re-confirmed at decompose.
