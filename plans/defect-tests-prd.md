# Defect tests: every declared defect gets a test that is proven to catch it, and kept

**Status:** active, authored 2026-10-16. Approach **B + H** (contracts + two-way boundary tests).
**Code anchors** are cited by symbol and verified against main `478b57d4f7` (2026-10-16). Main moves
fast, so re-locate symbols at implementation time.

## Goal

When a defect is found during a build, by whatever route, it is declared as structured data and
stays declared until a recorded outcome closes it. That outcome is one of three:

- **Proven:** a test the factory has *proven* fails on the pre-fix tree and passes on the fix.
- **Exempted:** an owned, reviewer-accepted exemption.
- **Withdrawn:** a reviewer-accepted withdrawal, for a defect declared in error.

A proven test is **kept**: no later merge may remove it silently. The rule holds for every project
the factory operates. It is proven by machine for pytest projects and visibly "unsupported"
elsewhere. It ships in **shadow** mode (recorded, not enforced) and is flipped to **enforce** by a
human gate after a soak.

User-observable when it lands:

- `get_task <id>` shows `metadata.defects`. Each entry has its origin, ref and status, plus its
  latest proof (`PROVEN` with base and tip SHAs, or a typed shortfall), its exemption or withdrawal,
  and the reviewer's ruling.
- `runs.db` `events` carries a `defect_plan_check` and a `defect_proof` event for every declared
  defect, and a `defect_keep_check` event at every merge. `scripts/defect_tests_soak_report.py`
  prints the soak statistics from them.
- In **enforce** mode:
  - A plan that omits a reproducing step is sent back to the architect.
  - A fix whose test does not fail on the pre-fix tree is sent back for a replan.
  - A merge with an unresolved defect, or one that drops a defect test without a disposition, is
    refused with a typed `DefectMergeRefusal`.
- A client project's feed (client-feed PRD) lists each defect ref, its test ids and its outcome per
  merge.
- Gate ξ reminds Leo when the LDE knowledge sheet may restore its trimmed sentence ("every bug or gap
  … gets its own test, kept and run on every later version").

## Background

`plans/test-per-bug-investigation-2026-10-15.md` (report) is the evidence base. In short:

- No rule or gate gives a bug its own test today. The habit is strong on planned fixes: 33 of 33
  code defects got a test after a blocking review (report M4).
- **No rule on these paths:** the steward, `/unblock`, the amender, the merger and hotfix paths.
- **Unchecked:**
  - The debugger may edit a failing test it judges wrong.
  - No code checks that a RED step failed (`plan_tools.py::_mark_step_done`).
  - Nothing protects a test once written.
- **Scoped merge verify** runs only the test files a diff touches. Task **7008** makes `full` the
  shipped default (Leo's ruling, 2026-10-15).
- **The sheet** was trimmed accordingly (scoping-intake `6b68b78`).
- **The report also measured** that most first blocking verdicts concern tests, not code (40 of
  75), and that all four recent red-main fix-forwards corrected a wrong test. A declared defect
  therefore needs an honest way to close as "not a defect".

## Consumers and user-observable surfaces (G1)

| Mechanism | Consumer |
|---|---|
| `defects` / `defect_test_retirements` fields and vocabulary (α1) | Every reader and writer below; `get_task` readers (Leo, the dashboard, the escalation watcher) |
| `declare_defect` / `record_defect_outcome` write path (α2) | δ2's writer, ε's declaration copy, ζ1/ζ2's proofs, δ2's rulings, `scripts/prove_defect.py`, θ1's filers |
| Injected pytest plugin: markers, collection map, outcome recorder (β) | The proof (ζ1), the presence check (η), the client feed (ι, and client-feed ε 6972) |
| Static defect enumerator `shared/src/shared/defect_enumerator.py` (β) | The proof's ref → node-id resolution (ζ1), the keep-ratchet (η) |
| `DEFECT_TEST_RULE` constant (γ) | Six dispatched role prompts; `skills/unblock`, `skills/unblock-low-risk`, `CONTRIBUTING.md` (rendered excerpts) |
| Reviewer `kind` and `defect_rulings` contract (δ1) | δ2's writer and ruling application; θ's `review_completed.by_kind` (client-feed 6967) |
| Plan tools for `reproduces` and declarations, and the plan predicate (ε) | The workflow plan, replan and simple paths; the `defect_plan_check` event; the architect re-prompt in enforce |
| Proof runner, `DefectProof` and `scripts/prove_defect.py` (ζ1) | `defects[].proof`; the presence check (η); the client feed (ι); the human paths (`/unblock`, hand carries) |
| Workflow proof integration (ζ2) | The `defect_proof` events; the typed replan; the fleet streak escalation |
| Merge presence check and keep-ratchet (η) | The merge refusal (`DefectMergeRefusal`) in `merge_status`; the `defect_keep_check` event |
| `defect_tests:` config group (ε) | ε, ζ2, η read it; μ flips `mode` |

## Resolved design decisions

Leo's rulings, from the 2026-10-15/16 sessions:

1. **Everywhere, not client-only.** One rule for every project, with no per-project switch for
   whether the rule applies (`docs/code-quality.md` heuristic 3). A rule exercised only on rare
   paid work would fail there first.
2. **Pytest first.**
   - The proof and the keep-ratchet cover projects whose test command the orchestrator classifies as
     pytest.
   - Any other runner records `PROOF_UNSUPPORTED_RUNNER` / `KEEP_CHECK_UNSUPPORTED`, visibly, never as
     green. That includes an `OPAQUE`-classified pytest command (pump-web-ui today) and reify
     (cargo-nextest via an OPAQUE `scripts/verify.sh`).
   - Reify is a follow-up PRD. The ref grammar is runner-neutral.
3. **Shadow, then flip.** `defect_tests.mode` ∈ {`off`, `shadow`, `enforce`}, shipped `shadow`;
   human gate μ flips it. See I5 for exactly what the mode governs.
4. **The reviewer judges** every exemption, withdrawal and test retirement through a structured
   verdict field (δ1, applied by δ2). It is recorded on the task and shown on the client page as typed fields, never
   prose.
5. **Proof site: VERIFY plus a merge presence check.**
   - The proof runs after the task's VERIFY phase (both arms: verify ran, or the verified-green
     checkpoint hit). Every orchestrated fix path re-enters it: debugger, replan, amender, steward
     resume.
   - The merge gate checks that no declared defect is unresolved, which covers `/unblock` and hand
     carries.
   - `scripts/prove_defect.py` produces the proof record for those human paths.
6. **Client defect bridge.**
   - A **client-feed PRD revision** owns the bridge: a `defect_reported` inbox event and a narrowly
     privileged filer outside the publisher's read-only isolation boundary.
   - This PRD owns only the `client` origin and its ref format.
   - Until the revision lands, Leo files client defects by hand per `docs/task-authoring.md` (θ2).
     Human gate ν schedules the revision.

Design decisions taken in this session:

7. **One record, one store, one writer.**
   - **Store:** every declared defect, task-level or found in-task, is an entry in the task's
     `metadata.defects`. This is a registered submodel in `shared/src/shared/task_metadata.py`, with
     list cardinality.
   - **Writers:** all writes after creation go through two server-side operations (α2). Both merge
     per entry by ref inside one write unit and apply the transition table (I2), whose single home is
     `shared/src/shared/defects.py`. Generic `update_task` cannot do this: its `merge` mode is shallow
     last-write-wins on the whole key, and `additive` unions whole dicts, which would duplicate a
     ref.
     - `declare_defect(task_id, event)` carries declarations, closure declarations (exemption,
       withdrawal) and test-retirement declarations. Any filer may call it.
     - `record_defect_outcome(task_id, ref, event)` carries `Proof` and `Rule` events.
   - **Creation:** `submit_task` accepts `defects` only as `open` entries with no proof and no ruling.
     `update_task` rejects any change to `defects` or `defect_test_retirements` for every caller.
   - **Authority is by tool exposure.** fused-memory has no authenticated caller identity, so
     authority cannot key on one. The precedent is
     `fused-memory/src/fused_memory/backends/task_backend_errors.py::DoneProvenanceWriteAuthorityError`,
     where `set_task_status` is the sole sanctioned writer.
     - `record_defect_outcome` is in no dispatched role's `allowed_tools` (`roles.py`), and a drift
       test asserts that.
     - Only the orchestrator workflow and `scripts/prove_defect.py` call it.
     - Interactive sessions are Leo's trust domain and are not machine-guarded.
   - **No silent drop:** a declared defect leaves `open` only through a recorded transition, so a
     later review cannot drop it silently, as happened to task 2358's (`one-fact-one-home`).
   - **No writes to terminal tasks:** an entry is never written after its task is terminal.
8. **The static marker is the binding.** `@pytest.mark.defect("<ref>")` on a test is the single home
   of the fact "this test reproduces `<ref>`".
   - The **static enumerator** (β, built on `shared/src/shared/pytest_mark_grammar.py`) is
     authoritative for which node ids carry a ref, for both the proof and the ratchet. The runtime
     collection map only confirms that those ids were collected.
   - A runtime-only mark (applied dynamically, invisible to the static scan) is reported as
     `MARKER_NOT_STATIC`, never counted.
   - The plan step's `reproduces` records the planned intent and is checked at plan time (ε). The
     proof re-resolves from the tree at use time (`corroborate-before-acting`).
9. **Proof against the pre-fix tree.**
   - The proof builds a detached worktree at `git merge-base <tip> main`, recomputed at proof time.
     Over it, it overlays the tip's **test-side diff**: every changed path the project's test-file
     rule matches, conftest and helpers included.
   - It runs the defect's node ids there and at the tip, with β's plugin injected.
   - Each side's per-id outcome comes from the plugin's in-process hooks (`pytest_runtest_logreport`,
     `pytest_collectreport`). They read `excinfo.value` attributes (`ImportError.name`,
     `AttributeError.name` / `.obj`), never junit text or tracebacks (heuristic 12).
   - A base-side collection error of a node's module counts as `error_attributed` when the missing
     symbol is absent at base and defined at tip. This is the ordinary "the test imports what the fix
     adds" case.
   - Proving against the base, not the step's own commit, survives the rebase and WIP-squash
     re-pointing of step commits (`workflow.py::_reconcile_done_step_commits`).
10. **Outcomes are a closed type** (`structured-facts-at-failure`, `no-silent-fail-soft`):
    `PROVEN`, `PASSES_AT_BASE`, `FAILS_AT_TIP`, `NOT_COLLECTED`, `MARKER_NOT_STATIC`,
    `PROOF_UNAVAILABLE(reason)`, `PROOF_UNSUPPORTED_RUNNER`.
    - `PASSES_AT_BASE`, `FAILS_AT_TIP`, `NOT_COLLECTED` and `MARKER_NOT_STATIC` are findings about
      the fix. In enforce they drive a replan.
    - `PROOF_UNAVAILABLE` (worktree, sync, timeout, crash, unattributed base error) retries once and
      is never blamed on the plan. In enforce it holds the merge with a typed infra refusal (η), an
      owned hold, not a silent jam.
11. **Bounded, persisted counters and a storm escape** (`storm-escape-required`).
    - **Plan refusal:** gets one architect repair per dispatch (precedent
      `workflow.py::TaskWorkflow._repair_plan_schema`).
    - **Proof finding:** gets a replan through a typed `ReplanCause = ReviewAggregation | DefectProof`
      rendered by `build_replan_prompt`.
    - **Persistence:** both are bounded by task-lifetime totals `defect_plan_repairs_total` and
      `defect_proof_replans_total`, persisted beside the review counters
      (`artifacts.py::TaskArtifacts.set_review_counters`, the task-2749 precedent). A steward resume
      grants no fresh allowance. Past the bound, the task blocks to the steward with the record.
    - **Storm escape:** `PROOF_UNAVAILABLE` alone feeds a fleet streak (`shared/src/shared/streak.py::KeyedStreaks`).
      It files one info L1 per task and one L2 per streak of `defect_tests.streak_threshold`
      consecutive distinct tasks.
12. **Scope: correctness defects only.** A defect is wrong runtime behaviour.
    - These are not defects: maintainability findings (`docs/code-quality.md` heuristics), doc,
      prompt or comment defects (reviewer `kind: docs`), test-quality findings
      (`kind: test_quality`), and lint and type errors. Lint and type errors are caught by checks
      that run anyway, so auto-healed `compile_error` tasks carry no `defects`.
    - De-flake tasks are owned by the flake-ledger PRD. A defect-marked test that flakes keeps
      counting as kept while ledger suppression lands merges.
    - A red-main test failure is a defect whose test already exists and fails at base. Marking it is
      enough for the proof. A red main caused by a *wrong test* is either not declared or is
      withdrawn as `test_was_wrong`.
13. **Every way to close a defect without a proof, or to remove its test, is owned and ruled.** All
    of them are declared through ε's plan tools (or `declare_defect` on non-planning paths). They are
    accepted or rejected by the reviewer's `defect_rulings` (δ1/δ2). On a path where no reviewer runs
    (`/unblock`, hand carries, steward fixes sent straight to `merge_request`), the operator rules with
    `scripts/prove_defect.py <task> --rule <ref> accept|reject --reason …`. That records a `Rule`
    marked `ruled_by: operator`, counted separately in the soak report.
    - **Exemption** `not_reproducible`: reason ∈ {`nondeterministic_environment`,
      `needs_external_system`, `no_runnable_harness`} plus `Debt(owner: TaskRef | TicketRef)` from
      `shared/src/shared/governed_exceptions.py`. `Policy(ratified)` is unavailable until the INV-12
      ratification table lands.
    - **Withdrawal**: reason ∈ {`not_a_defect`, `test_was_wrong`, `duplicate_of:<ref>`}.
      - "Live" means declared on some task, found through α2's lookup by ref.
      - `duplicate_of` is also the steward's exit when a rebase brings in an independent fix from
        main and the proof flips `proven → open` via `PASSES_AT_BASE`.
    - **Retirement** of a proven defect's test is **not** a state of the defect's entry. That entry
      usually lives on a terminal task and is never rewritten.
      - It is a record on the task whose diff removes the test:
        `metadata.defect_test_retirements: [Retirement(ref, reason, ruling)]`, with reason ∈
        {`feature_removed`, `superseded_by:<ref>`, `test_relocated`}.
      - It is declared with ε's `retire_defect_test` through `declare_defect` and ruled on that task.
      - A `superseded_by` must name a live ref.
    - A closure awaiting its ruling is `pending`. The proof skips pending refs, and the presence check
      treats them as unresolved. Exemptions and withdrawals are counted per project
      (`exceptions-owned-or-ratified`).
14. **Keep-ratchet compares the candidate with its own parent.**
    - At merge, η enumerates refs statically on the candidate tree and on the tree it lands on: main,
      or a speculative-train base. Both are read from git objects, never a working copy.
    - **Removal:** a ref present on the parent and absent from the candidate is a removal. A ref
      whose only carriers are statically `skip`/`xfail` counts as removed.
    - **Attribution:** a removal is attributed to the task whose own diff removes it, and needs an
      accepted retirement in **that task's** `defect_test_retirements`. There is no baseline file to
      edit.
    - **Robustness:** keyed on ref, so a rename or move is not a removal.
    - **Incompleteness:** a file that cannot be parsed makes the result `KEEP_CHECK_INCOMPLETE`, never
      clean (`readers-prove-their-producer`). This applies only to files that carried a ref on the
      parent tree or that the candidate's diff touches; an unrelated file is not counted.
      `KEEP_CHECK_INCOMPLETE` feeds the same `KeyedStreaks` storm escape as `PROOF_UNAVAILABLE`.
15. **Ref grammar** (`<origin>:<id>`): runner-neutral, minted once, never re-minted.

    | Origin | Ref | Note |
    |---|---|---|
    | `review` (in-task) | `review:<task>:<issue_hash>` | Idempotent by issue content; repeat sightings increment a counter on the existing entry |
    | `review_instrument` | `finding:<fk-key>` | |
    | `red_main` | `redmain:<main_health_signature>` | |
    | `agent_followup` | `followup:<spawned_from>:<suggestion_hash>` | |
    | `human` | `human:<filed_at_utc>:<hash8>` | No dependency on an id that does not exist yet |
    | `client` | `portal:<project>:<block>:<n>` | |
    | `exploratory` | `explore:<run>:<n>` | |

    `origin` is authoritative; `metadata.source` stays a free label.
16. **Reviewer contract.** Every issue carries a required `kind` ∈ {`defect`, `test_quality`,
    `docs`, `process`}. The verdict carries
    `defect_rulings: [{ref, subject: exemption | withdrawal | retirement, accepted: bool}]`.
    - A missing or unknown value is a retryable `status: error` from `submit_review_verdict` that
      names the allowed values.
    - A legacy verdict being normalised defaults to `defect`, the fail-closed value, and is flagged.
    - Every enum is rendered into the contract from `shared/src/shared/defects.py`.
    - δ2's writer declares blocking `kind=defect` issues **before** the `max_review_cycles` cap check,
      so a final-cycle defect is still declared.
17. **The rule text has one home.** `DEFECT_TEST_RULE` in `roles.py` is spliced by one helper into
    each role prompt that fixes code. Skills and `CONTRIBUTING.md` carry a pointer plus a rendered
    excerpt mirrored by a test (INV-10 mirror tier). The guard asserts the splice by identity with a
    sentinel (`guards-exercise-behaviour`).
18. **Client-facing display carries typed fields only.** The feed shows ref, test ids, outcome,
    exemption or withdrawal reason, and owner. It never shows `summary`. "No defect map recorded"
    renders as `not_recorded`, distinct from "zero defects".
19. **The plugin reaches target projects by injection.**
    - β's plugin is a single dependency-free file, uniquely named and alone in its own directory:
      `shared/src/shared/_pytest_inject/_df_defect_plugin.py`.
    - Only that directory goes on `PYTHONPATH`, set in `verify.py::_target_subprocess_env`, so no
      other shared module can shadow a target's own modules.
    - The orchestrator adds `-p _df_defect_plugin` to every structured pytest verify leg and to the
      proof's legs. A raw-retained or OPAQUE pytest leg cannot take the flag, so it records
      `KEEP_CHECK_UNSUPPORTED` / `PROOF_UNSUPPORTED_RUNNER`.
    - Injection is skipped when `defect_tests.mode == off`.
    - It registers the `defect` and `criterion` markers through `config.addinivalue_line`, so a
      target's own `markers` list is never overridden and `--strict-markers` or `-W error` targets do
      not break.
    - It writes the collection map, and its outcome recorder runs only on proof legs.
    - A target project needs no install and no config edit. Node ids are module-qualified
      (`<module_root>/<path>::<name>`) everywhere they are stored.

## Sketch of approach

- **α1 Vocabulary.**
  - `shared/src/shared/defects.py` holds the enums, the frozen records, the ref grammar and the
    transition table.
  - The `defects` and `defect_test_retirements` submodels are registered in
    `shared/src/shared/task_metadata.py`.
- **α2 Write path.** In `fused-memory`:
  - the MCP operations `declare_defect` and `record_defect_outcome`;
  - lookup by ref;
  - the creation floor in `submit_task`;
  - `update_task` rejection of edits to both fields.
- **β Plugin and enumerator.**
  - `shared/src/shared/_pytest_inject/_df_defect_plugin.py` is the single-file plugin: markers, the
    collection map `<junit>.tests.json` = `{nodeid: {defects: [ref], criterion: str|null}}`, and the
    outcome recorder. Client-feed ε (6972) consumes the map instead of writing its own.
  - Injection: the `-p` flag in `orchestrator/src/orchestrator/verify_cmd.py`; the plugin directory
    on `PYTHONPATH` in `verify.py::_target_subprocess_env`.
  - The static enumerator `shared/src/shared/defect_enumerator.py` is built on
    `shared/src/shared/pytest_mark_grammar.py`, with per-file completeness and skip/xfail detection.
- **γ Rule.** The `DEFECT_TEST_RULE` constant and splice helper; `_format_task` rendering of
  declared defects into the architect, replan and simple-task briefings.
- **δ1 Reviewer contract.** `kind` and `defect_rulings`, with validation.
- **δ2 Reviewer consumption.** The in-task declaration writer, the application of rulings, and the
  reviewer briefing showing declared defects, pending closures and retirements.
- **ε Plan.**
  - Plan tools: `reproduces` on `add_plan_step`, plus `declare_not_reproducible`,
    `declare_withdrawn` and `retire_defect_test`.
  - `step_type` is validated as an enum.
  - `plan_reproduces_declared_defects(plan, defects) -> PlanCheck`, at every plan-producing site.
  - In enforce, the simple-path veto.
  - The `defect_tests:` config group.
- **ζ1 Proof runner.** The `PROOF_BASE` worktree kind, the overlay, the injected-plugin pytest legs,
  `DefectProof`, and `scripts/prove_defect.py`, all recording through `record_defect_outcome` (α2).
- **ζ2 Workflow proof integration.** The hook after both VERIFY arms, the typed replan, the persisted
  counter, the streak, and the soak report.
- **η Merge.** `orchestrator/src/orchestrator/merge_lane/defect_checks.py`: the presence check and
  the keep-ratchet, with one gate call and a typed refusal.
- **θ1 / θ2.** Auto-filing sites stamp declarations, with the curator combine policy; the docs and
  skills.
- **ι Feed display.**
- **κ Debugger-edit fact.**
- **λ Integration gate.**
- **μ / ν / ξ Human gates.**

## Contract

### Types (`shared/src/shared/defects.py`, all frozen)

```
DefectOrigin    = review | review_instrument | red_main | agent_followup | human | client | exploratory
IssueKind       = defect | test_quality | docs | process
DefectStatus    = open | pending | proven | exempted | withdrawn
ExemptionReason = nondeterministic_environment | needs_external_system | no_runnable_harness
WithdrawalReason = not_a_defect | test_was_wrong | duplicate_of(ref)
RetirementReason = feature_removed | superseded_by(ref) | test_relocated
ProofOutcome    = PROVEN | PASSES_AT_BASE | FAILS_AT_TIP | NOT_COLLECTED | MARKER_NOT_STATIC
                | PROOF_UNAVAILABLE | PROOF_UNSUPPORTED_RUNNER
SideOutcome     = passed | failed | error_attributed | error_unattributed | skipped | not_collected
KeepOutcome     = KEPT | REMOVED_RETIRED | REMOVED_UNRETIRED | KEEP_CHECK_INCOMPLETE | KEEP_CHECK_UNSUPPORTED
DefectMergeRefusal = defect_unresolved | defect_proof_unavailable | defect_test_removed
                   | keep_check_incomplete

NodeId(module_root, path, name)
DefectDeclaration(ref, origin, summary, declared_at, declared_by, status, sightings: int,
                  proof: DefectProof|None, closure: Exemption|Withdrawal|None, ruling: Ruling|None)
Exemption(reason: ExemptionReason, owner: Debt)          Withdrawal(reason: WithdrawalReason)
Retirement(ref, reason: RetirementReason, ruling: Ruling|None)   # on the REMOVING task
Ruling(subject: closure | retirement, accepted: bool, ruled_by: reviewer | operator, verdict_cycle|None)
DefectProof(ref, outcome, base_sha, tip_sha, measured_at, node_ids: tuple[NodeId, ...],
            base: tuple[(NodeId, SideOutcome, exc_type|None, missing_symbol|None)], tip: (same),
            unavailable_reason|None, runner_rc|None, duration_s)
DeclareEvent = Declare(DefectDeclaration) | DeclareClosure(ref, Exemption|Withdrawal)
             | DeclareRetirement(Retirement) | Sighting(ref)        # declare_defect (any filer)
OutcomeEvent = Proof(DefectProof) | Rule(ref, Ruling)                # record_defect_outcome (orchestrator, prove_defect.py)
PlanCheck(outcome: ok | missing | reproduces_after_fix | unknown_ref, defect_refs)
KeepCheck(outcome, removed_refs, retired_refs, incomplete_files, parent_sha, candidate_sha)
```

Every refusal, event, replan prompt and escalation renders from these records, never from formatted
prose.

### Invariants

- **I1** A `defects` entry's ref parses under the grammar (decision 15) and is unique within the
  task. `declare_defect` and `record_defect_outcome` are the only post-creation writers, and
  neither writes to a terminal task.
- **I2** The transition table for a `defects` entry is the single home, applied server-side:
  - `open → proven`: a `PROVEN` proof for the current tip.
  - `proven → open`: a later non-`PROVEN` proof for a newer tip.
  - `open → pending`: a closure (exemption or withdrawal) is declared.
  - `pending → exempted | withdrawn`: an accepted ruling.
  - `pending → open`: a rejected ruling.

  No other transition exists. Retirement is not a transition: it is a `Retirement` record on the
  removing task, which only its own ruling changes.
- **I3** The proof re-reads `defects` from the live store, recomputes the merge-base, and
  re-resolves refs to node ids statically at the tip.
- **I4** `PROVEN` holds iff all of these hold for every ref:
  - at least one statically carried node id;
  - every id collected and `passed` at tip;
  - every id either collected and `failed` at base, or `error_attributed` there (at run or
    collection).
- **I5** The mode governs exactly three gates: the plan refusal (ε), the proof-driven replan (ζ2)
  and the merge refusal (η). In `shadow` they record and do not act. Contract validations (`kind`,
  `defect_rulings`, `step_type`, the `defects` shape and authority) and the `PROOF_UNAVAILABLE`
  escalations are mode-independent.
- **I6** The keep-ratchet compares the candidate tree with the tree it lands on (main or a
  speculative base). A removal is attributed to the task whose diff removes it.
- **I7** A check that cannot complete reports that as its own outcome (`PROOF_UNAVAILABLE`,
  `KEEP_CHECK_INCOMPLETE`), never a pass.
- **I8** The presence check requires two things of the merging task:
  - none of its declared defects is `open` or `pending`;
  - for each of its `proven` refs, the candidate tree still statically carries that proof's node ids
    under the ref, unless the merging task's own `defect_test_retirements` holds an accepted
    retirement for the ref.

  `tip_sha` equality is not required, so rebases and merger commits are tolerated.

### Hook points (S1/S4-verified)

| Hook | Site |
|---|---|
| Plan check | `workflow.py::TaskWorkflow._plan`, after `_validate_prerequisites_or_block('initial plan')`, before `stamp_plan_provenance`/`lock_plan` |
| | after `_replan` beside the "Architect replan produced no valid steps" check in `_execute_verify_review_loop` |
| | `_apply_revalidation_skip` (its `None` return falls through to the architect path) |
| | `_should_run_simple_task`: in enforce, a veto for any task with an `open` defect; in shadow, a post-hoc recording check in `_run_simple_task` |
| | The order check considers only `pending` steps added since the defect was declared |
| In-task declarations (δ2) | `_execute_verify_review_loop`'s blocking branch, before the `max_review_cycles` check |
| Proof | `_execute_verify_review_loop` after both VERIFY arms (`_verify_debugfix_loop` ran, or `_verify_checkpoint_hit`); skipped only when a `PROVEN` proof exists for the current tip |
| Merge checks | one call from the merge gate into `merge_lane/defect_checks.py` |
| Reviewer briefing | `briefing.py::build_reviewer_prompt`, on the task block task 5369 adds |

### Config (`config.py::OrchestratorConfig`, group `defect_tests:`, in the reload green tier)

- `mode` (`shadow`)
- `max_plan_repairs` (1)
- `max_proof_replans` (1)
- `proof_timeout_secs`
- `streak_threshold` (5)
- `test_file_pattern` (default: the report's M1 rule)

## Boundary-test sketch (the integration gate λ)

Rows run against the real workflow and merge seams. They use fake agents, a real git fixture repo
holding a pytest workspace, and one **non-workspace** pytest fixture project (no `shared` installed).

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Planned fix, proven | `human` defect; plan has a `test` step `reproduces=[ref]`; marked test fails at base | `PlanCheck=ok`; `DefectProof=PROVEN` (base/tip SHAs); status `proven`; presence passes; events recorded |
| B2 | Plan omits reproduction | Same; no `reproduces` step | shadow: `missing` recorded, plan proceeds. enforce: one repair prompt with the `PlanCheck`; a second miss blocks to the steward; the counter survives a re-dispatch |
| B3 | Vacuous test | Marked test passes at base | `PASSES_AT_BASE`; enforce: replan via `ReplanCause=DefectProof`; past the bound, blocks with the record |
| B4 | New-symbol import | Test module imports a function the fix adds (top-level import) | Base: collection `error_attributed` (`ImportError.name` absent at base, defined at tip); `PROVEN` |
| B5 | Unrelated base error | Base conftest broken for an unrelated reason | `error_unattributed` → `PROOF_UNAVAILABLE` after one retry; one info L1; no replan spent; enforce: merge held as `defect_proof_unavailable` |
| B6 | In-task review defect | Reviewer blocking issue `kind=defect` on the final allowed cycle | Entry `review:<task>:<hash>` declared before the cap exit; a later review omitting it leaves it `open` |
| B7 | Contract refusal | Issue without `kind`, or a malformed `defect_rulings` entry | `submit_review_verdict` returns `status: error` naming the enum; the reviewer retries; mode-independent |
| B8 | Exemption | `declare_not_reproducible(needs_external_system, owner=ticket)` | Proof skips it; `pending`; reviewer accepts → `exempted`, presence passes; reviewer rejects → `open` |
| B9 | Withdrawal | Red-main defect whose test was wrong; `declare_withdrawn(test_was_wrong)` | Reviewer accepts → `withdrawn`; no proof required; counted |
| B10 | Unretired removal | A later task deletes a marked test with no retirement | shadow: `defect_keep_check REMOVED_UNRETIRED` naming the ref; enforce: refused `defect_test_removed` |
| B11 | Rename / relocate | Marked test moves files, same ref | `KEPT` |
| B12 | Skip-marked | Only carrier gains `@pytest.mark.skip` | Treated as removed (B10) |
| B13 | Unsupported runner | OPAQUE test-command project | `PROOF_UNSUPPORTED_RUNNER` / `KEEP_CHECK_UNSUPPORTED`; the presence check counts the defect as unresolved; enforce requires an exemption |
| B14 | `/unblock` path | Fix merged through `merge_request`, no workflow verify | Presence: `defect_unresolved` (enforce refuses, shadow records); `scripts/prove_defect.py <task>` records `PROVEN`; the retry passes |
| B15 | Curator combine | Suggestion carrying `defects` combined into a `pending` target / an in-progress target | Pending target gains the entries via `declare_defect`; in-progress target → the curator creates a separate task instead |
| B16 | Speculative train | Task A retires ref R with an accepted retirement; task B unrelated; both in one train | Both land; B's `KeepCheck=KEPT` (B is compared with its own parent, main+A) |
| B17 | Authority | Any dispatched role's tool allowlist; any caller's `update_task` on `defects` or `defect_test_retirements` | The drift test shows `record_defect_outcome` is in no role's `allowed_tools`; `update_task` is rejected for every caller; `declare_defect` is accepted |
| B18 | Non-workspace target | The fixture project with `filterwarnings=error` and its own `markers` list | Verify passes with the injected plugin; `defect` registered; target markers intact; B1 holds |
| B19 | Runtime-only mark | `pytest.param(..., marks=pytest.mark.defect(ref))` invisible to the static scan | `MARKER_NOT_STATIC` |
| B20 | Operator ruling | A withdrawal declared on the `/unblock` path, where no reviewer runs | `scripts/prove_defect.py --rule <ref> accept` records `ruled_by: operator`; status `withdrawn`; the merge retry passes |
| B21 | No shadowing | The fixture project has its own top-level module `testing` | With the plugin directory on `PYTHONPATH`, the project's `testing` still imports |
| B22 | Raw-retained leg | A pytest leg inside a raw-retained shell chain | No injection; `PROOF_UNSUPPORTED_RUNNER` / `KEEP_CHECK_UNSUPPORTED` recorded; the verify itself is unaffected |
| B23 | Retirement on the removing task | Task C removes a proven ref's test with an accepted retirement in its own `defect_test_retirements` | `KeepCheck=REMOVED_RETIRED`; C merges; the original done task's entry is not written |

## Pre-conditions for activating

These in-flight tasks edit the same surfaces, so the leaves named depend on them (S2, S4,
2026-10-16):

- **5752** (plan-deviation β; `plan_tools.py`) → ε.
- **5753** (plan-deviation γ; `roles.py` doctrine), **5532** (architect recipe SPOT) and **5357**
  (debugger failure text) → γ.
- **5494** (verdict issue validation), **6739** (`review_payload.py`) and **5224** (verdict contract
  `note_dispositions`; one harness-version bump per landing) → δ1.
- **5369** (reviewer task record) and **6967** (client-feed θ, same loop exits) → δ2.
- **6972** (client-feed ε) → amended at decompose to consume β's map and to depend on β, with a
  paired edit of `plans/client-feed-prd.md` §3, §7.7, decision 17, the ε row and its sidecar block
  (done 2026-10-16, client-feed §12). 7002 (α2) needs no change: its contract work does not touch
  the map.
- **7008** → ξ only.

δ2, ε and ζ2 all edit `workflow.py::TaskWorkflow._execute_verify_review_loop`, so they are chained
δ2 → ε → ζ2. Tier 1 (α1 → β → γ) waits on none of δ, ε or α2.

No novel substrate is assumed beyond what the leaves build. Per-leaf G3 evidence goes in the
capability manifest at decompose.

## Cross-PRD relationship

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/client-feed-prd.md` ε 6972, §3, §7.7, decision 17 | this → other | the injected plugin's collection map `<junit>.tests.json` `{nodeid: {defects, criterion}}`; β also owns the `criterion` marker registration | this PRD (β); paired edit and 6972 amendment made at decompose (client-feed §12, 2026-10-16) | amended; 6972 depends on β |
| `plans/client-feed-prd.md` display | this → other | `cases[].defects`, `defects_results` fold, `not_recorded` | this PRD (ι) | queued |
| `plans/client-feed-prd.md` θ 6967 | both | `review_completed.by_kind` counts | client-feed θ, amended at decompose | at decompose |
| client-feed PRD **revision** (unwritten) | other → this | `defect_reported` inbox event + privileged filer calling `submit_task`/`declare_defect` with `origin: client` | client-feed revision (ruling 6) | human gate ν |
| `~/src/scoping-intake/docs/prds/client-portal.md` | other → this | report id and criterion id exposed to the revision's filer | portal PRD, one-line amendment via ν | via ν |
| `plans/flake-ledger-prd.md` §6 | both | a defect-marked test is ineligible for any quarantine lane; suppression does not shrink the ratchet | this PRD (rule); flake-ledger gets the reciprocal sentence (θ2) | queued |
| `docs/quality-findings-contract.md` §8 | this → other | filed **correctness** findings declare `defects` with `ref = finding:<fk>` | this PRD (θ2, one pointer sentence) | queued |
| task 7008 | other → this | full merge breadth gives "run on every later version" and merge-run junit | task 7008 | pending |

## Decomposition plan

Labels only; ids are assigned at decompose. Sizing follows the overlay. Tier 1, for the first client
block, is α1 → β → γ. Each brief cites the binding `docs/code-quality.md` heuristics (10, 11, 12, 13)
and the merge-lane ratchet where it applies.

**α1 Defect vocabulary** (intermediate).
- *Scope:* `shared/src/shared/defects.py` (types, ref grammar, transition table); `defects` and
  `defect_test_retirements` registration in `shared/src/shared/task_metadata.py`; the
  `docs/task-authoring.md` §8 entry.
- *Unlocks:* β, γ, α2, δ1. Roped to λ.
- *Prereqs:* none.

**α2 Defect write path.**
- *Scope:* in `fused-memory`, the `declare_defect` and `record_defect_outcome` MCP operations
  (per-ref merge in one write unit, applying α1's transition table, never writing a terminal task);
  lookup by ref; the `submit_task` creation floor; `update_task` rejection for every caller; a
  drift test that `record_defect_outcome` is in no dispatched role's `allowed_tools`.
- *Signal:* B17; a declared defect round-trips through `get_task`; a malformed declaration is
  rejected synchronously, naming the allowed values.
- *Prereqs:* α1.

**β Injected pytest plugin and static enumerator (Tier 1).**
- *Scope:* `shared/src/shared/_pytest_inject/_df_defect_plugin.py` (alone in its directory, no
  imports beyond stdlib and pytest: markers via `addinivalue_line`, the collection map, the outcome
  recorder); the `-p` injection into structured pytest legs in `verify_cmd.py` and the `PYTHONPATH`
  entry in `verify.py::_target_subprocess_env`, both skipped when mode is `off`;
  `shared/src/shared/defect_enumerator.py` on `pytest_mark_grammar.py`; dark-factory's members also
  list the marker in pyproject, with a drift test.
- *Signal:* B18, B21, B22 (a non-workspace project verifies with the marker registered and its own
  modules unshadowed, and its map lists a marked test); `python -m shared.defect_enumerator <tree>`
  prints refs and completeness.
- *Prereqs:* α1.

**γ Defect-test rule in every fixing prompt (Tier 1).**
- *Scope:* `DEFECT_TEST_RULE` and the splice helper; splices into ARCHITECT, `build_replan_prompt`,
  SIMPLE_TASK, DEBUGGER, STEWARD and `build_amender_prompt`; `_format_task` renders declared
  defects; rendered excerpts plus a mirror test in `skills/unblock`, `skills/unblock-low-risk` and
  `CONTRIBUTING.md` (the human paths' only lever).
- *Signal:* a persisted architect or debugger briefing for a task carrying `defects` lists them and
  contains the rule; the sentinel identity test passes for all six builders.
- *Prereqs:* α1, β, 5753, 5532, 5357.

**δ1 Reviewer contract.**
- *Scope:* `kind` and `defect_rulings` in `_REVIEWER_CONTRACT_TEMPLATE`, rendered from α1, with the
  harness-version bump and its literal pins (`orchestrator/tests/test_reviewer_prompt_split.py`,
  `evals/reviewer_trial`); validation in `verdict_tools.py::_submit_review_verdict` on 5494's
  validator; the legacy `kind` default in the review-payload normaliser; `_format_issue_section`
  renders `kind`.
- *Signal:* B7.
- *Prereqs:* α1, 5494, 6739, 5224.

**δ2 Reviewer consumption.**
- *Scope:* the in-task declaration writer (idempotent hash refs, before the cap check, via α2's
  `declare_defect`); rulings applied via `record_defect_outcome`; the reviewer briefing shows
  declared defects, pending closures and the task's retirements.
- *Signal:* B6; the reviewer half of B8 and B23.
- *Prereqs:* δ1, α2, 5369, 6967.

**ε Plan tools, plan check and mode.**
- *Scope:* `reproduces` on `add_plan_step` and the replace path; `declare_not_reproducible`,
  `declare_withdrawn` and `retire_defect_test` plan tools, copied to the task via `declare_defect`
  by the workflow;
  `step_type` enum validation; `orchestrator/src/orchestrator/defect_tests.py::plan_reproduces_declared_defects`
  at the plan sites; the simple-path veto; the persisted `defect_plan_repairs_total`; the
  `defect_tests:` config group in the reload green tier; the `defect_plan_check` event.
- *Signal:* B2 in shadow and enforce; the declaration half of B8, B9 and B23.
- *Prereqs:* α2, δ2, 5752.

**ζ1 Proof runner.**
- *Scope:* the `WorktreeKind.PROOF_BASE` and `PROTECTED_PREFIXES` entry in `git_ops.py` (a
  merge-lane ratchet cluster file, so the raise is recorded in
  `orchestrator/tests/merge_lane_ratchet_authorized_raises.json` in the same commit); the test-side
  overlay; injected-plugin pytest legs at base and tip; the `DefectProof` assembly from the plugin's
  outcome records; `scripts/prove_defect.py <task> [--rule <ref> accept|reject --reason …]`
  recording through `record_defect_outcome`; a measured cold and warm duration recorded in the
  completion note.
- *Signal:* `prove_defect.py <task>` on a fixture branch records `PROVEN` (B1, B4) and
  `PASSES_AT_BASE` (B3), visible in `get_task`; B20.
- *Prereqs:* α2, β.

**ζ2 Workflow proof integration.**
- *Scope:* the hook after both VERIFY arms; the skip on a current `PROVEN`; I2 transitions via
  `record_defect_outcome`;
  `ReplanCause` and `build_replan_prompt` rendering; the persisted `defect_proof_replans_total`;
  the `PROOF_UNAVAILABLE` retry, the info L1 and the `KeyedStreaks` L2; `scripts/defect_tests_soak_report.py`
  (read-only, from `runs.db` events).
- *Signal:* B1/B3/B5 `defect_proof` events in `runs.db`; the soak report prints them.
- *Prereqs:* ε, ζ1.

**η Merge presence check and keep-ratchet.**
- *Scope:* `merge_lane/defect_checks.py` (functions at or below cognitive complexity 15, per the
  ratchet's `new_function_cognitive_ceiling`) with one gate call, its raise recorded in the
  authorized-raises file; the I8 presence check; the I6 keep-ratchet against the candidate's parent;
  the typed `DefectMergeRefusal` on the merge outcome (not the free-text `reason`); the
  `defect_keep_check` event; soak report rows.
- *Signal:* B10–B14, B16 and B23 at the merge lane.
- *Prereqs:* α2, β, ζ1.

**θ1 Auto-filing sites declare defects.**
- *Scope:* `roles.py::submit_only_instructions` gains a structured `defects` parameter used by
  `_FOLLOWUP_FILING_INSTRUCTIONS`, the steward templates, DEEP_REVIEWER and
  `review_checkpoint.py::_build_prompt`; the curator `_build_candidate` carries declarations;
  `_execute_combine` adds them through `declare_defect` only to a `pending` target and otherwise files
  a separate task.
- *Signal:* B15; an implementer follow-up carries `origin: agent_followup`.
- *Prereqs:* α2.

**θ2 Docs and skills** (`complexity: simple` candidate).
- *Scope:* `docs/quality-findings-contract.md` §8 (one pointer sentence: correctness findings only,
  `ref = finding:<fk>`); the `skills/review`, `skills/hotspot-survey` and `skills/review-all`
  templates point there; `docs/task-authoring.md` hand-filing recipes for `human`, `red_main` (naming
  the escalation field that carries `main_health_signature`, which θ2 locates) and interim `client`;
  `plans/flake-ledger-prd.md` §6 reciprocal sentence.
- *Signal:* a `/review` filing per the template produces a task whose `defects` validates; the
  mirror test passes.
- *Prereqs:* α1, θ1.

**ι Client-feed defect display.**
- *Scope:* `cases[].defects` and the `defects_results` fold in `client-feed`, with the `not_recorded`
  state and typed fields only.
- *Signal:* the publisher run against dark-factory's own merge-run junit and map, as client-feed ε
  does, shows a real defect ref with its outcome. The row is observed, not seeded.
- *Prereqs:* β, ζ2, 6972, 7002, 6969, 7008.

**κ Debugger-edit fact to the reviewer.**
- *Scope:* parse the task-verify "cost" junit into failing ids; track the debugger's changed files
  per iteration (`git_ops.py::GitOps.get_changed_files`); pass `edited_failing_tests` with a
  `could_not_determine` state to `build_reviewer_prompt`.
- *Signal:* a review after a debugger edit of a failing test shows the fact in the persisted
  reviewer briefing.
- *Prereqs:* δ2, 5357.

**λ Integration gate.**
- *Scope:* one boundary module implementing B1–B23 against the real seams.
- *Signal:* the module passes in the orchestrator suite.
- *Prereqs:* δ2, ε, ζ2, η, θ1.

**μ Human gate: flip `defect_tests.mode` to `enforce`.**
- *Scope:* `task_kind: deterministic`, `always_escalates`. Soak criteria, read from
  `scripts/defect_tests_soak_report.py`:
  - at least 14 days in shadow **and** at least 10 recorded proofs, or 60 days, whichever comes
    first;
  - no unexplained `PROOF_UNAVAILABLE` streak;
  - a hand audit of 5 `PROVEN` records and every `PASSES_AT_BASE`.
- *Basis:* M4 measured about 11 review-found code defects a month.
- *Prereqs:* λ, κ.

**ν Human gate: author the client-feed PRD revision.**
- *Scope:* the `defect_reported` inbox event and the privileged filer (ruling 6), including the
  portal PRD's one-line amendment. `always_escalates`.
- *Prereqs:* α2.

**ξ Human gate: restore the sheet's per-bug sentence?**
- *Scope:* shows Leo the soak report and asks whether `about-lde.md` may say again "every bug or gap
  found during the build gets its own test, kept and run on every later version".
  `always_escalates`.
- *Prereqs:* μ, ι, 7008.

**G7 (advisory; walked by S2, re-checked by S4, 2026-10-16).** Resolutions:

| Invariant | Resolved by |
|---|---|
| `no-silent-fail-soft` | 10, I7 |
| `structured-facts-at-failure` | Contract types; decision 9's attribute-based outcomes |
| `corroborate-before-acting` | I3, I8, 14 |
| `storm-escape-required` | 11 |
| `contracts-machine-checked` | 7, 16, I5, η |
| `one-fact-one-home` | 7, 8, 15, 17 |
| `guards-exercise-behaviour` | 17 |
| `exceptions-owned-or-ratified` | 13, 14 |
| `readers-prove-their-producer` | 14, 18, ι's signal |
| `loop-thread-occupancy-bounded` | ζ1 uses the async subprocess runner with `proof_timeout_secs` |
| `status-matches-liveness` | η's refusals route through the existing merge-lane blocked/requeue choke point with typed reasons |

Waiver: `contracts-machine-checked` for **undeclared** defects (one an agent fixes inline without
declaring it). No machine can see them; γ and `_FOLLOWUP_FILING_INSTRUCTIONS` are the only levers.

## Out of scope

- Non-pytest runners for the proof and the keep-ratchet (reify, cargo, jest/vitest, shell, Kotlin).
  These get a follow-up PRD.
- Automatic client defect intake (ν).
- Machine-checked RED for ordinary feature `test` steps. ζ1's runner could extend to them, which
  would be a separate decision.
- Retrofitting `defects` onto terminal tasks.
- Leo's direct-to-main commits (`git commit --only`), which bypass the merge lane. γ's
  `CONTRIBUTING.md` excerpt is the only lever.

## Open questions (tactical)

1. **The proof worktree's venv cost.** Settle whether `PROOF_BASE` reuses the warm seed or
   `_preprovision_shared_venv`. ζ1 measures and records it.
2. **The default test-file rule for the overlay** (`test_file_pattern`). Does it suffice for each
   onboarded project? Check in ζ1.
3. **The soak report's output format.** Decide in ζ2.
