# Module Tagger Retirement PRD

**Status:** active — retirement ratified by Leo 2026-08-20 after a 5-agent
measurement investigation (session `investigate-df-3595126`). Approach:
bare B (G5 prompt resolved: no new seam contract is created; the
`metadata.files` writer-charter seam is owned by DF 3260).

## Goal

Delete the LLM module tagger (`Harness._tag_task_modules` and its entire
supporting surface) from dark-factory, close the one real coverage gap its
absence opens (undeclared `complexity='simple'` submissions), and hand the
existing repair pipeline a census of the records the tagger already
damaged. After this PRD lands, an operator observes:

- Orchestrator startup no longer runs the "Tag tasks with code modules"
  step; `--retag-modules` is an unknown CLI argument.
- A `submit_task` with `complexity='simple'` and empty/absent
  `metadata.files` is rejected at the fused-memory boundary with a
  structured error; the same submission with declared files succeeds.
- A committed census artifact enumerates every tagger-stamped record
  across all six project corpora, classified for the repair pipeline.

## Background — why delete (measured 2026-08-20 unless marked inherited)

The tagger predicted, from task title+description+a directory listing, the
files a task would touch, and persisted `metadata.files` (+
`files_tagged_at`) — the sole durable input to scope-lock derivation
(`Scheduler._get_modules` → `derive_modules` → `try_acquire`).

- **Serving population is tiny.** Stamp-era cohort: 3.2% of new DF tasks
  (16/504) and 17% of reify tasks (199/1,169) ever reach it; author-declared
  files at submit has been the primary channel since 2026-03-27
  (`564ab0221f` demoted the tagger to a fallback).
- **Predictions are poor on its intended population.** 200 scored genuine
  cold starts: median Jaccard 0.25 vs the plan, 10% exact match, 18–22%
  complete misses, micro recall 0.23; 33.6% of predicted paths did not
  exist at tag time. Corroborates the in-repo haiku-trial verdict
  (F1≈0.37, "MARGINAL", `plans/module-tagger-haiku-trial-report.md`).
- **Its lock protects a window in which nothing can write.** Pre-plan
  window is 10–14 min median; the architect has Edit/Write disallowed,
  edits are worktree-local, and module locks are pure admission
  bookkeeping (never enforced by any sandbox). Plan-boundary
  `_reconcile_scope_locks` re-derives locks from real scope before
  EXECUTE, and overlap is caught there (121 reify / 245 DF conflict
  requeues observed).
- **Counterfactual contribution ≈ zero.** In ~5 weeks of live operation, a
  tagger-derived pre-plan lock was the sole blocker of a dispatch 0 times
  in either project (1 DF event cited a tagged holder alongside other
  blockers). Author-declared pre-plan locks blocked 416 (reify) / 240 (DF)
  pairs over the same lens — the mechanism works; the tagger adds nothing.
- **It actively corrupts.** 8.6% (DF, 25/292) to 24.6% (reify, 52/211) of
  its stamps overwrite authoritative scope destroyed by an upstream wiper
  (lower bounds), converting recoverable wipes into permanent,
  plausible-looking records that `repair_wiped_metadata_files.py` then
  refuses (SKIP_FILES_PRESENT). 33 non-terminal victims live now (11 DF,
  22 reify). Inherited, consistent: esc-6068-4's stratified n=120 sample —
  11/11 stamped tasks diverged from real scope, Fisher p=0.0007.
- **Fallback is the accepted status quo.** 21–33% of tagger-era dispatches
  already ran on the conflict-with-nothing synthetic `task-<id>` lock;
  realized overlap harm across whole corpora: ~7 merge-conflict pairs
  (reify), 0 (DF). Only 10 (reify) / 7 (DF) modern-era episodes ever ran
  to done entirely under a synthetic lock.

Six projects ran the tagger (DF, reify, autopilot-video, know-live,
pump-web-ui, solar-challenge-platform). All six live
`dark-factory-orchestrator.yaml` files carry zero `module_tagger` keys
(verified 2026-08-20; defaults-only), so deleting the config-schema fields
breaks no deployed config.

## Sketch of approach

Three independent leaves plus decompose-session coordination actions:

1. **β — delete the tagger** (orchestrator + shared + config + tests +
   docs), ordered after in-flight task 3122 lands.
2. **δ — hard gate**: reject `complexity='simple'` submissions with
   empty/absent `metadata.files` at the fused-memory write boundary
   (γ lock-charter-guard territory; `lock_charter_error` structured-error
   precedent, `routing_intent_guard` lint precedent nearby).
3. **ε — debris census**: deterministic sweep of all six corpora emitting
   a committed, machine-readable artifact for the repair pipeline (DF 3113
   P4a, DF 3427) — repair itself stays with its existing owners.

No replacement predictor is built. Undeclared never-planned tasks run on
the synthetic `task-<id>` lock until plan reconcile — accepted on the
measured basis above.

## Resolved design decisions

1. **Delete, not narrow.** DF 4504 (gate the tagger against overwriting
   planned scope) is superseded and will be cancelled; a deleted tagger
   overwrites nothing. DF 4191 (re-tag on fingerprint change) is mooted.
2. **Hard gate for simple tasks** (Leo, 2026-08-20; chosen over no-gate
   and warn-only). The SIMPLE_TASK path plans *and* edits under the
   dispatch-time lock only (`workflow.py:5030-5074`), so it is the one
   population whose lock quality matters during edits. δ makes the
   author-declared channel mandatory there. δ includes a pre-flip census
   of the historical would-reject rate, stated in its PR.
3. **Census only for debris** (Leo, 2026-08-20). ~503 stamped records and
   the tagger-guessed wrong scopes on non-terminal tasks are enumerated
   and classified by ε; repair/blanking stays with DF 3113 P4a and DF 3427
   (divergence-keyed, per the wiped-field lesson: audit on divergence from
   the authoritative source, never on emptiness).
4. **Land 3122 first** (Leo, 2026-08-20). Task 3122 (fileless-misfile
   soft signals) is in-progress and its branch persists
   `files_tagged_empty`, one of its three soft signals. β depends on 3122
   reaching done, then deletes the `files_tagged_empty` write path along
   with the tagger. Misfile detection retains its other two signals
   (title convention, foreign-root-in-prose) and the PathScopeAdjudicator
   wiring; 3122's own text rates the tagger signal non-gating
   ("not a gate on its own").
5. **Vocabulary entries are retained, annotated historical.**
   `files_tagged_at` stays a blessed Tier-A key
   (`shared/src/shared/task_metadata.py:903`) — 153 observed metadata
   amendments re-carry existing stamps, and unblessing would reject those
   writes. `module_tagger` stays in `KNOWN_ROLE_NAMES` (`:82`) for the
   same old-record-tolerance reason. Both get "retired 2026-08 — historical
   records only, no live writer" comments; no production code writes either
   anew.
6. **Read paths keep tolerating historical rows.** `digest.py`'s
   NULL-`task_id` `module_tagger` invocation handling (:467, :516) and its
   fixtures stay — runs.db retains historical tagger invocation rows that
   the model×role rollup must still render.
7. **Config-schema `module_tagger` fields are deleted** (Models/Budgets/
   MaxTurns/Effort/Timeouts/Backends + `defaults.yaml` entries + eval-mode
   pins). Whether an unknown `module_tagger:` key in a user yaml then
   errors or is ignored follows the schema's existing extra-key policy —
   verify at implementation; do not assert rejection in any signal without
   observing it fire (G6 branch 4).
8. **Coordination is performed by the decompose session, not filed as
   tasks**: cancel DF 4504, DF 4191, DF 3380 (each after a
   dependent-check — cancelling a task arms its dependents); amend DF 3260
   (details append) noting the tagger writer is removed. DF 3380's
   residual idea (validating author-declared paths) goes to Open
   questions, not into any leaf.

## Pre-conditions for activating

- DF 3122 `done` (β's dependency; in-progress with a live claimant as of
  2026-08-20 12:18Z).
- fused-memory MCP running for decompose filing (standard).

## Cross-PRD / cross-task relationships

| Other | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| DF 3260 (owned-field single-writer redesign) | this PRD shrinks the writer set 3260 charters | `metadata.files` writer charter | 3260 | amend 3260's record at decompose |
| DF 3122 / 3121 (misfile soft signals) | this PRD deletes signal (c) `files_tagged_empty` after 3122 lands | tagger empty-verdict persistence in `harness.py` | this PRD (β) | β depends on 3122 |
| DF 3113 P4a / DF 3427 (repair pipeline) | ε produces their input | census artifact (victim list keyed by project/task/classification) | this PRD (ε) produces; 3113/3427 own repair | queued |
| DF 4504, 4191, 3380 | superseded / mooted | — | this PRD | cancel at decompose |
| `plans/capability-delivered-checks-prd.md` | consumes | manifest YAML sidecar | that PRD's stamper | standard |

## Decomposition plan

All three tasks are leaves (no intra-batch dependencies).

- **β — Delete the LLM module tagger end-to-end** · high ·
  `task_kind=normal` · depends on: DF 3122.
  Remove: `_tag_task_modules` (harness.py:2883-3067), its 3 call sites
  (:2477 startup, :10080, :10135), the `retag_modules` param (:2172) and
  `--retag-modules` CLI flag (cli.py:228-269), import :83;
  `module_tagger_prompt.py`; `Scheduler.seed_modules` (scheduler.py:
  8812-8824 — verify sole caller is the tagger before removing);
  config-schema `module_tagger` fields (config.py:229,245,261,277,303,354)
  and `defaults.yaml` entries (:192,344,366,391,411,425); eval-runner pins
  (evals/runner.py:306-336); `routing_dispatch.py` batch-contract comments
  (:159-269); `workflow.py:12197-12198` comment; `module_charter.py:14-15`
  docstring; the `files_tagged_empty` persistence 3122 lands; dedicated
  tests (`test_harness_module_tagging.py`, `test_harness_module_tagger_cost.py`,
  `test_module_tagger_prompt.py`), `scripts/trial_module_tagger_haiku.py`
  + `tests/scripts/test_trial_module_tagger_haiku.py` +
  `shared/pyproject.toml:20` collection entry, and the
  `_tag_task_modules` mocks in the 8 harness-startup test files. Annotate
  (do not remove) the two vocabulary entries per decision 5; keep digest
  handling + fixtures per decision 6. Update docs: `ARCHITECTURE.md:667`
  role table, `OPERATIONS.md:588`, `docs/task-authoring.md`
  (:364,:782,:794,:797,:844 — mark `files_tagged_at` historical),
  `skills/orchestrate/SKILL.md:327,:593`.
  *(Line anchors are at HEAD `eba215060c`, 2026-08-20 — re-grep before
  editing; anchors drift.)*
  **Observable signal:** a fresh orchestrator startup log shows the
  adjacent startup steps but no module-tagging step (baseline stated in
  the signal); the CLI rejects `--retag-modules` as unknown; `rg -n
  module_tagger` across `orchestrator/src fused-memory/src shared/src
  dashboard/src` returns only the two annotated vocabulary retention
  sites and digest's historical-row handling; full test suite green.
- **δ — Hard-gate simple-task submissions on declared files** · high ·
  `task_kind=normal` · no deps (parallel-safe with β; different package).
  In the fused-memory write boundary (`task_interceptor.py`, beside the γ
  lock-charter guard at :5909-5927), reject `submit_task` where the
  authored complexity is `simple` and `metadata.files` is empty/absent,
  with a structured error (sibling of `lock_charter_error`) whose message
  states the requirement. Verify the exact complexity field/spelling
  against `docs/task-authoring.md` and `plans/author-declared-complexity-prd.md`
  at implementation. Update the submit-instruction prompts of roles that
  file simple tasks (`agents/roles.py` submit stamps) to state the
  requirement. Include a census of the historical would-reject rate in
  the PR description. Decide the `update_task`-sets-simple-later edge per
  Open question 1.
  **Observable signal:** a live `submit_task` with `complexity='simple'`
  and no files returns the structured error (observed through the MCP
  response — G6 branch 4: the rejection is exercised, not assumed); the
  same submission with declared files succeeds; a non-simple fileless
  submission still succeeds (no scope creep).
- **ε — Tagger-debris census across all six corpora** · medium ·
  `task_kind=normal` · no deps.
  A deterministic script (reusing `scripts/audit_wiped_metadata_files.py`'s
  event lens and its corpus-access pattern) sweeps DF, reify,
  autopilot-video, know-live, pump-web-ui, solar-challenge-platform for
  records carrying `files_tagged_at`, classifying each: terminal vs
  non-terminal; plan-reconciled since tagging vs never-reconciled
  (tagger guess still live); known post-wipe signature. Emits a committed
  machine-readable artifact (+ md summary) keyed for DF 3113 P4a / DF 3427
  consumption. Read-only against task corpora (fused-memory read path,
  never raw `tasks.db`).
  **Observable signal:** the committed artifact exists with per-project
  counts; it contains the known positives (reify 6068/5602/5632, DF 3113's
  own record); re-running the script reproduces the counts (exit 0).

**G7 (advisory walk, author mode):** δ's rejection is machine-checked, not
prose (`contracts-machine-checked` ✓) and is a stateless per-write check —
no detector loop, no storm surface; ε is read-only and emits structured
facts (`structured-facts-at-failure` ✓); β's cancellations happen via
status transitions that trigger reconciliation (`status-matches-liveness`
✓). No hits identified; decompose re-walks the full list in
`docs/legibility/design-invariants.md` as the blocking check.

## Out of scope

- **Repair of tagger-damaged records** — DF 3113 P4a (divergence-keyed
  self-heal) and DF 3427 (second-pass sweep) own it; ε feeds them.
- **Path-existence validation of author-declared `metadata.files`** — DF
  3380's residual idea; its own measurement (naive existence assertion
  ~85% FP because tasks legitimately create new files) bounds any future
  design. Not filed here.
- **Any replacement predictor** (deterministic or LLM) — the measured
  basis says none is needed.
- Unmasking terminal historical records; deleting historical plans/docs
  (`module-tagger-haiku-trial-report.md` stays as history).

## Open questions (tactical)

1. **Does δ also cover `update_task` later setting `complexity='simple'`
   on a fileless task?** Suggested: yes if the interceptor sees that write
   cheaply, else file a follow-up. Decide during δ.
2. **Unknown `module_tagger:` yaml key behavior after schema-field
   deletion** (error vs ignore) — follows the schema's existing extra-key
   policy; verify during β and state in the PR.
3. **`.worktrees/3380` and branch `task/3380` cleanup** after 3380's
   cancellation — normally orchestrator gc's job; confirm it collects
   cancelled-task worktrees, else note for ops.
