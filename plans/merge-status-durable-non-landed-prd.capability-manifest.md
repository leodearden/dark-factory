# Capability manifest — `plans/merge-status-durable-non-landed-prd.md`

Mechanizes G3 + G6 for the batch. Every binding below was **re-derived by grep at decompose time**
(2026-08-28, main `3e64918b53`), not carried over from the PRD's prose. Capabilities are cited by
**symbol**, never `path:line`.

Machine-readable twin: `plans/merge-status-durable-non-landed-prd.capability-manifest.yaml`.

**Verdict summary: 20 bindings across 6 tasks — 20 PASS, 0 FAIL, 0 OPEN; 12 carry a mechanical
`delivered_check`.** Counts verified by loading the sidecar through
`shared/src/shared/capability_manifest.py::load_capability_manifest`, not by hand-tallying the tables
below. No binding had to be re-scoped, re-homed or relaxed to clear the gate.

---

## α — Vocabulary one-home + drift guard *(intermediate; unlocks β, δ)*

| Capability | Binding | Verdict |
|---|---|---|
| `shared-vocabulary-module-convention` | capability→producer (wired) — `shared/src/shared/task_statuses.py::TaskStatus` exists on main; `plans/task-status-authority-prd.md` is the precedent that collapsed four duplicated status-set copies into it | PASS |
| `doc-consistency-guard-precedent` | capability→producer (wired) — `scripts/tests/test_design_invariants_consistency.py::PINNED_SITES` (9 refs) derives a vocabulary from one normative doc and cross-checks restating artifacts | PASS |
| `guard-already-pins-a-skills-markdown-file` | capability→producer (wired) — that same guard pins `skills/prd/references/gates.md` (4 refs), so pinning a `skills/*/SKILL.md` list is an **established** pattern, not a new capability | PASS |
| `drift-guard-fires-on-seeded-violation` | rejection-mechanism (G6 branch 4) — built **and bound** by α itself: the seeded-violation fixture must be observed to FAIL the guard, mirroring `docs/legibility/design-invariants-fixtures.md` | PASS *(manual — rejection quality is judged by the fixture, not a grep)* |

**Note on the two-vocabulary split.** α homes `MergeState` **and** `MergeSubmitStatus`. Verified at
decompose: `skills/merge-queue/SKILL.md`'s "Terminal at submit time" set interleaves submit-only
values (`already_merged`, `unknown_branch`) with poll states, so a guard pinning those lists against
a poll-only enum would fire falsely. Homing only the poll half is a `producer-extent-short` shape —
caught here, resolved by widening α's extent rather than by relaxing the guard.

## γ — Cross-run reads + alias resolution *(intermediate; unlocks β)*

| Capability | Binding | Verdict |
|---|---|---|
| `latest_merge_finalized-exists` | capability→producer (wired) — `orchestrator/src/orchestrator/event_store.py::EventStore.latest_merge_finalized` | PASS |
| `finalize-payload-carries-superseded_by` | capability→producer (wired) — the `merge_finalized` emit in `orchestrator/src/orchestrator/merge_queue.py` writes `superseded_by` into `data`, so the train reverse-index needs **no new write** | PASS |
| `event-type-index-covers-the-cross-run-predicate` | capability→producer (wired) — `idx_events_type ON events(event_type)` exists; `merge_finalized` is 2,620 of 364,134 rows (0.7%) | PASS |
| `cross-run-query-is-affordable` | numeric floor — `floor: 50 ms > ~0.05 ms measured`. Measured 2026-08-28 on the live 161 MB / 364,134-row store: cross-run 0.0 ms vs run-scoped 0.2 ms. ~3 orders of margin | PASS |

## β — The read-path rebuild *(intermediate; unlocks δ, ε)*

| Capability | Binding | Verdict |
|---|---|---|
| `git-authority-tier-importable` | capability→producer, DAG-direction — `producer:task-4649` **upstream** (out-of-batch hard prereq, wired as a real edge). β re-points at it; it does not extract it | PASS |
| `MergeQueueStore-readable-from-escalation` | capability→producer (wired) — `orchestrator/src/orchestrator/merge_queue_store.py::MergeQueueStore`, and `escalation/src/escalation/server.py` already lazily imports `orchestrator.*` at 9 sites, so the journal tier introduces **no new layering violation** | PASS |
| `journal-is-the-same-file-recovery-reads` | capability→producer (wired) — `merge_queue_store.py::recover_pending_merges`; tier 4 reads exactly what startup recovery reads (one fact, one home) | PASS |
| `merge-state-vocabulary` | capability→producer, DAG-direction — `producer:task-α` **upstream** | PASS |
| `cross-run-read-mode` | capability→producer, DAG-direction — `producer:task-γ` **upstream**. β cannot emit `stale_record` without it; this is why β depends on γ | PASS |

## δ — Collapse the skills' decision trees onto `reason` *(LEAF)*

| Capability | Binding | Verdict |
|---|---|---|
| `reason-field-emitted-by-the-server` | capability→producer, DAG-direction — `producer:task-β` **upstream**. The signal reads a field β populates; it is not re-derived in prose | PASS |
| `three-consumer-files-exist` | capability→producer (wired) — `skills/merge-queue/SKILL.md`, `skills/unblock/SKILL.md`, `skills/unblock-low-risk/SKILL.md` all present | PASS |
| `vocabulary-lists-already-normalized` | capability→producer, DAG-direction — `producer:task-α` **upstream**. α does the mechanical list edits so δ only rewrites reasoning prose — the two never contend on the same lines | PASS |

## ε — Delete the retention ring *(intermediate; unlocks ζ)*

| Capability | Binding | Verdict |
|---|---|---|
| `ring-is-genuinely-unwired` | **rejection-mechanism / `expect: absent`** — re-derived at decompose, NOT taken from task 3149: `_terminal_retention\s*=` has **zero** production matches and `TerminalOutcomeRetention(` has **zero** production constructions (tests only). The deletion premise is independently confirmed | PASS |
| `tier2-read-already-removed` | capability→producer, DAG-direction — `producer:task-β` **upstream**. β removes the Tier-2 read path before ε removes the type, so neither step leaves a dangling reference | PASS |

## ζ — Integration gate: boundary suite B1–B10 *(LEAF)*

| Capability | Binding | Verdict |
|---|---|---|
| `all-boundary-rows-have-upstream-producers` | DAG-direction — every row's producer (α, γ, β, δ, ε, and out-of-batch 4649) is **upstream** of ζ. No row depends on a task that depends on ζ | PASS |
| `B5-asserts-a-non-call` | capability→producer (wired) — the landing primitives (`is_ancestor`, `find_merge_marker`) exist today in `orchestrator/src/orchestrator/git_ops.py`; B5 asserts this module does **not** call them (invariant I5), which is checkable by spy/mock, not by needing new substrate | PASS |
| `B10-latency-bound-is-above-the-floor` | numeric floor — `floor: p99 < 50 ms > ~0.05 ms measured`; basis and margin as in γ | PASS |

---

## G6 premise notes

- **B10 is the only numeric assertion in the batch** and it carries a measured achievability basis
  with ~3 orders of magnitude of margin. It is deliberately scoped to the **cross-run event-store
  read** — it does **not** bound the git tier, which this PRD neither introduces nor changes and
  which measures 0.84–1.11 s per full-history scan on this repo's 62,928 commits.
- **B9 and ε's `ring-is-genuinely-unwired` are the two rejection-style assertions.** Both are bound:
  B9 by α's seeded-violation fixture (the guard must be *observed* to fail), ε by an `expect: absent`
  grep re-run at decompose.
- **No signal in this batch samples a result field**, so the field-population sub-check does not fire.

## G7 walk — all nine invariants, every task, no waivers

| Invariant | Disposition |
|---|---|
| INV-1 `contracts-machine-checked` | **Resolved by α** — this is the defect being retired. The vocabulary moves from prose in three SKILL.md files to an enum plus a submit-time-checkable guard. |
| INV-2 `structured-facts-at-failure` | **Resolved by β** — Tier 4 today discards everything it knows behind a constant hint; `reason` + `last_outcome` are exactly this invariant. |
| INV-3 `corroborate-before-acting` | **No hit.** `stale_record` reports a labelled record and never acts on it. The rejected recency-window alternative *was* this shape — recorded in the PRD as rejected for that reason. |
| INV-4 `storm-escape-required` | **N/A.** No task adds a detector, suppressor or fallback that emits escalations. β adds read tiers; ε deletes; α is a test. |
| INV-5 `no-lockstep-duplication` | **Resolved by α** (three prose copies → one home) and **guarded in β** by invariant I5: β consumes the landing verdict and must never grow a second landing authority beside `branch_work_landed`. |
| INV-6 `status-matches-liveness` | **Resolved by D1's partition.** An epistemic state can never imply an owner or a live merge; `journaled` and `stale_record` are explicitly not-live. |
| INV-7 `holds-owned-and-bounded` | **N/A.** No new hold, park, or wait state. |
| INV-8 `loop-thread-occupancy-bounded` | **Addressed, not waived.** β adds a sync sqlite read and a small atomic JSON read inside an async MCP tool. Measured: the cross-run query is ~0.05 ms and the journal is a single small file; the pre-existing git tier is already `await`ed. Pinned by B10. β carries an explicit note to keep the journal read a bounded single-file load and revisit if the journal ever grows unbounded. |
| INV-9 `one-fact-one-home` | **Resolved by D6 + the read-path-only framing.** The merge outcome's home is the `merge_finalized` row; this PRD reads it rather than building a projection table, and ε *removes* the competing RAM home. |
