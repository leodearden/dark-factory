# A test per bug: what the factory does, and what the client-block promise needs

2026-10-15. An investigation run as a `/team` session at Leo's brief
`~/.claude/spawn-briefs/2026-10-13-scoping/brief-test-per-bug.md`. Read-only: no code
was changed and no task was filed. Code is cited as `path::symbol` against main
`005b0a20c7` (every cited file is unchanged since `0e2408f8fa`, where the seats read it).
An opus critic seat reviewed the assembled draft, and its findings are folded in.

## The answer in brief

The sheet says: *"Every bug or gap found during the build gets its own test, kept and
run on every later version."* Today no rule says this and no gate checks it. **Trim the
sentence now** (§6). Restore it only when the machine checks of §5 Tier 2 and the
breadth default of D6 exist.

**What holds in spirit.**

- **Fixes that go through planning.** The architect's general rule, "Every behavior gets
  a test first", reaches every fix the factory plans. After a blocking review, **all 33
  code defects in the measured sample were fixed with a test that drives them, and none
  without one** (§3, M4). 32 tests were new and 1 extended an existing test. Ordering was
  checked on 9 of them: in 7 the test came first, marked RED.
- **Failures caught by an existing check.** Most failures the debugger fixes were found
  by lint, type or test checks that already exist and keep running. 229 of 325 landed
  debugger fixes answered lint or type failures only (M1).

**What does not hold.**

- **Paths with no test rule.** The steward, `/unblock`, the amender, the merger and Leo's
  direct hotfix have none, and on the simple path a test is optional.
- **The debugger's licence to edit tests.** It may edit a failing test it judges wrong,
  unchecked. 56 of the 96 landed debugger fixes for a failing test changed only test
  files. Whether any of them weakened the test that found a bug is unmeasured.
- **"Kept".** Nothing protects a test once written.
- **"Run on every later version".** Under the default `merge_verify_breadth: scoped`,
  which every newly onboarded project gets, the merge gate re-runs only the test files the
  merging diff touches, and none for a source-only diff. The full suite runs on main
  every 30 minutes, after the fact.
- **Bugs found outside the factory.** Nothing in the factory runs the acceptance suite or
  the exploratory check, and nothing takes in a client's defect report.
- **Unmeasured transfer.** Nothing checks any of this, and the measured habit is
  dark-factory's own, with the architect and implementer on opus at xhigh. A client
  block on shipped defaults runs the implementer and debugger on sonnet.

**Recommendation.** One `/prd`, applied to every project rather than client projects
only, in two tiers:

- **Tier 1 (D4, D6), small, before the first client block.** It makes the trimmed wording
  true.
  - **D4:** one prompt rule, held in one constant. It is spliced into the architect,
    replan, simple-task, debugger, steward and amender prompts, and pointed at from the
    `/unblock` skills and `CONTRIBUTING.md`. It does not reach the merger.
  - **D6:** `factory-init` writes `merge_verify_breadth: full` for every new project.
- **Tier 2 (D1, D2, D3, D5).** It is what restoring the full sentence requires:
  - **D1:** a defect is structured data, including a `kind` on each reviewer issue.
  - **D2:** one workflow predicate requires a reproducing test step in the plan of every
    declared defect.
  - **D3:** the machine proves that test fails on the pre-fix tree and passes at the tip.
  - **D5:** a marker and ratchet keep the test.

§5 gives each part, its cost and what it cannot catch.

## 1. Where a bug is found, and what happens there

"Required / suggested / not mentioned" is about **a test that captures the bug found**.
Running existing tests does not count.

| # | Where the bug is found | Who fixes it | Test for the bug | Enforced by |
|---|---|---|---|---|
| a | The implementer's own test step (EXECUTE) | Implementer, following the architect's plan | Required where the plan has a `test` step; the implementer may not add steps | Prompt |
| b | Task VERIFY fails | Debugger, at most 4 times (5 verify runs) | Not mentioned | Nothing |
| c | Reviewer reports a **blocking** defect | Architect replan, then implementer and debugger | Not mentioned in the replan prompt; only the architect's general rule 4 | Nothing specific |
| d | Reviewer **suggestion**, in scope | Implementer as amender, 1 round per task | Not mentioned; test coverage is a review area, so a missing test is at most a suggestion | Nothing |
| e | Simple path (`complexity='simple'`, which the rubric offers for "a one-spot bug fix") | Single simple-task agent | Suggested ("For tasks that need a new test: add `test` then `impl` step pair") | Prompt |
| f | Task blocked and escalated | Steward (L0) | Not mentioned ("verify with tests" means run them) | Nothing |
| g | Completion judge | none (off by default) | Not mentioned | Nothing |
| h | Post-merge red main | `compile_error`: an auto-filed "fix main" task. Every other category: escalation to a human or auto-watcher | Not mentioned in the fix-task text | Nothing |
| i | `/review` finding filed as a task | Normal pipeline (architect) | Not mentioned ("how to verify the fix" only) | Nothing |
| j | `/hotspot-survey` or `/review-all` finding filed as a task | Normal pipeline | Not mentioned | Nothing |
| k | Deep reviewer (every 40 merges) | Files a task | Suggested only as a finding class ("Integration test gaps") | Prompt |
| l | Human `/unblock`, `/unblock-low-risk` | Human or sub-agent | Not mentioned; low-risk *permits* touching "direct test files" | Nothing |
| m | Leo's direct hotfix | Leo | Not mentioned; "RED/GREEN pairs" are a commit-message convention | Nothing |
| n | Client-reported defect, 90-day window | No factory actor: the portal's `defect_reported` stage has no bridge to a factory task | Not mentioned | Nothing |
| o | An agent notices a latent bug outside its task | Architect or implementer files a low-priority task (`source: agent-followup`); the filed task goes through the normal pipeline | Not mentioned in the filing instruction | Nothing |
| p | Out-of-scope reviewer suggestions | Routed to the curator as follow-up tasks | Not mentioned | Nothing |
| q | Merge-conflict resolution | Merger (rule 3, "Run tests after resolving.") | Not mentioned | Nothing |
| r | Acceptance suite or exploratory check during the build | Nothing in the factory runs either. A suite placed in the project's `test_command` would run in every verify, and a failure would go through row b with its test already present | none | Nothing |

Evidence for each row, quoted from the current text:

- **a.** `orchestrator/src/orchestrator/agents/roles.py::IMPLEMENTER`, rule 2: "For `test`
  steps: write the test, run it, confirm it fails." The check is self-attested.
  `orchestrator/src/orchestrator/mcp/plan_tools.py::_mark_step_done` checks only that the
  step id exists and the commit is reachable from HEAD. No runtime code reads a step's
  type or runs a test step's commit. The survey's "no code checks RED" row is
  **confirmed**.
- **b.** `roles.py::DEBUGGER` and `orchestrator/src/orchestrator/agents/briefing.py::build_debugger_prompt`.
  The whole instruction is "Analyze the root cause … Make minimal, targeted fixes … Run
  the verification commands … Commit your fixes". Rule 3 is the only test rule: "Don't
  change test expectations unless the test itself is wrong", and no gate checks that
  judgement. The debugger sees the plan title and analysis, not the task description.
  Task 5357 (pending, high) measured 12.7% of debugger briefings carrying no failure text
  at all.
- **c.** `workflow.py::TaskWorkflow._execute_verify_review_loop` calls `_replan(reviews)`.
  `briefing.py::build_replan_prompt` says "Add the steps that address the blocking issues
  with `add_plan_step(...)`". It says nothing about a test. The first blocking verdict
  gets one replan. The second hits `max_review_cycles=2` and escalates to the steward.
- **c, d.** The reviewer is *not told* whether a change is a bug fix: its prompt is the
  memory context and the diff, cut at 50,000 characters (`briefing.py::build_reviewer_prompt`).
  It can read the plan and `git log`. `_REVIEWER_HEURISTICS_TEMPLATE` rule 2, "Blocking
  means broken", and rule 3, "When in doubt, suggest", make a missing test a suggestion
  at most. The only blocking rule about tests runs the other way ("Reject docstring /
  wording meta-tests").
- **e.** `roles.py::SIMPLE_TASK` and `roles.py::_COMPLEXITY_RUBRIC`.
- **f.** `roles.py::STEWARD`: "Fix the code, verify with tests"; "Stay in scope. Only fix
  what the escalation describes."
- **h.** `merge_lane/worker.py::compose_fix_main_brief` writes "Automated main-health fix
  task.", the category, the cause and 800 bytes of detail, with no test clause.
  `AUTO_HEAL_MECHANICAL_CATEGORIES = {'compile_error'}`. Its comment excludes
  `test_failure` because it "needs human judgement on test expectations".
- **i.** `skills/review/references/phase3-triage.md` Step 5.
- **j.** `skills/hotspot-survey/SKILL.md` and `skills/review-all/SKILL.md` filing templates.
- **k.** `roles.py::DEEP_REVIEWER`.
- **l.** `skills/unblock/SKILL.md`, `skills/unblock-low-risk/SKILL.md`.
- **m.** `CONTRIBUTING.md` §1(b) and §7. `hooks/pre-commit` runs no pytest.
- **n.** `~/src/scoping-intake/docs/prds/client-portal.md` §5 has the edge "defect_window →
  defect_reported, actor system". `plans/client-feed-prd.md` §7.4: the portal-to-factory
  inbox carries only `reveal_requested`.
- **o.** `roles.py::_FOLLOWUP_FILING_INSTRUCTIONS`, spliced into `ARCHITECT` and
  `IMPLEMENTER`: "a latent bug in an adjacent module … do NOT do it inline … File it".
- **q.** `roles.py::MERGER`.
- **r.** `plans/client-feed-prd.md` §11 lists "Running the acceptance suite against
  staging" as out of scope. `client_feed` is unbuilt.

The merge-gate verify failure *before* landing was not traced. Its fix runs through rows
b–d.

No path has a gate. The architect's plan path is the one place a test-first habit reaches
a fix, and only because the same prompt applies to every task.
`roles.py::ARCHITECT` rule 4: "Every behavior gets a test first." No fix-task branch
exists: `build_architect_prompt` takes the task dict and has no task-kind switch.

## 2. What is already true in spirit

- **Fixes through a replan.** After a blocking review, all 33 code defects in the measured
  windows were fixed with a test that drives them: 32 new, 1 extended, none without
  (M4). The commits read like "test(4878): RED — main() itself pairs a gated slug (the review's regression
  test)".
- **Fix-titled tasks.** 58 of 62 fix-titled tasks merged in the last 90 days changed some
  test file (M2). That is not necessarily a new test for the bug. Of the 4 that did not,
  3 were doc or prompt-text fixes and 1 (task 3297, a script bug) was a real code fix
  with no test.
- **A failure found by an existing check keeps its check.**
  - Lint and type failures (229 of 325 landed debugger fixes) are caught by checks that
    run again on later changes.
  - Of the 96 landed debugger fixes for a failing test:
    - 16 changed only source. The failing test was kept.
    - 56 changed only test files. That is a lower bound, since tip-only ranges undercount.
    - 24 changed both.
  - Whether the 56 repaired the task's own new test or weakened a pre-existing test that
    had found a real bug is unmeasured. That is the one way the debugger can lose a
    bug's test, and nothing checks it.
- **Red main in this repo has been test fragility, not product bugs.** All four
  red-main fix-forwards merged in the last 90 days (6567, 5425, 5230, 4583) edited the
  failing test itself. The test was wrong against the ambient config or an enumeration
  pin.

Where it does not hold:

- **Fixes that never plan.** The steward, `/unblock`, the amender, the merger, Leo's
  direct hotfix, and the simple path, where a test is optional.
  - Six tasks' first blocking verdict was followed by no replan commits. Each was traced
    (§3, M4):
    - one was fixed in the amendment pass, with the test written after the fix (2709);
    - one test-only finding was removed by an implementer review-fix iteration (4354);
    - Leo reverted one doc-only change (5164);
    - one branch contamination was removed by a rebase (4880);
    - one was a false positive from a dirty worktree (3984);
    - **one was never fixed (2358).** Later reviews downgraded it to a suggestion, and the
      substring match it named is still on main
      (`orchestrator/src/orchestrator/chronic_flake.py::_has_open_dedup_match`,
      `if test in (result.get('title') or '')`). A blocking verdict can be overruled by a
      later review with no test and no record of why.
- **"Kept".** Any later task may delete or weaken a test. The reviewer is not told to
  object. The judge, which does treat "deletes tests that were previously failing" as
  non-work, is off by default (`config.py::OrchestratorConfig.judge_after_each_iteration`).
- **"Run on every later version".** It is true per merge only under
  `merge_verify_breadth: full`, where every registered module's full suite runs on the
  merged tree.
  - Under the default `scoped` (`config.py::OrchestratorConfig.merge_verify_breadth`;
    `defaults.yaml`), the merge gate keeps the "legacy FILE_SCOPED/SKIPPED shape"
    (`orchestrator/src/orchestrator/verify_plan.py::_derive_module_runs`). Pytest runs only
    the test files the merging diff touches. A source-only diff runs none at merge
    ("Arm 5 is reachable only at role='merge'").
  - Under `scoped`, the owning module's full suite runs only at the task's own VERIFY,
    before rebase. The full suite runs on the main tip every 30 minutes
    (`main_tip_sweep_enabled`, default on), which escalates on drift but is not a gate.
  - `skills/factory-init/SKILL.md` never writes the key. `full` for client projects
    exists only as a manual install step in Leo's design note
    (`~/mission-control/consulting/notes/client-dashboard-design.md` §10).

## 3. Measurements

Every figure carries its rule. The figures are about dark-factory's own build. reify
could not be measured: its ledgers hold 1 debugger entry in total.

**M1: debugger commits that land without touching a test file. 49 of 325 (15.1%).**

- *Population:* every `"agent": "debugger"` entry with summary "Debug fix for …" in
  `.worktrees/.task-meta/*/iterations.jsonl`, 2026-07-06 to 2026-10-15 (552 entries).
- *Committed:* 435 entries made a commit.
- *Landed:* 325, meaning the task has a `Merge task/<id>` commit on main. Only 51 debugger
  shas are on main literally, because branches are rebased before merge. The other 274
  were matched by surviving pre-rebase objects (129) or by author-time windows over the
  merge range (145, the weakest tier). Excluded: 117 uncommitted, 110 committed but not
  landed.
- *Diff range:* previous ledger commit to this commit, falling back to the author-time
  window or the tip commit alone. Tip-only is a lower bound on test touches.
- *Test file:* path matches `(^|/)tests?/`, `test_*.py`, `*_test.py`, `*_test.go`,
  `*.test.ts(x)`, `*.spec.ts(x)` or `conftest.py`.
- *Last 90 days:* 28 of 225 (12.4%).
- *By failure kind:* fixes whose failure included "tests failed", 16 of 96 (16.7%);
  lint-only, type-only or lint+type, 33 of 229 (14.4%).
- *Test-failure split:* 16 source only, 56 test files only, 24 both.
- *Reading:* this is the brief's requested number, but it measures test-file *touches*,
  not new regression tests. Most debugger test-file touches are lint or pyright repairs of
  the task's own tests.
- *Ledger coverage:* 552 of the roughly 1,020 debugger invocations `runs.db` records since
  2026-07-06.

**M2: fix-titled tasks that change no test file. 4 of 62 (6.5%) in the last 90 days;
26 of 320 (8.1%) since 2026-03-18.**

- *Title rule:* case-insensitive, `^(fix|fixes|fixed)\b`, or contains "regression",
  "post-merge", "fix-forward" or "red main".
- *Population:* status done, with a `Merge task/<id>` commit dated on or after 2026-07-17
  (1,579 tasks merged in that window).
- *Diff:* the whole task, `merge^1...merge^2`, so the task's own RED commits are included.
  "Changes a test file" is not "adds a test for the bug": the red-main fix-forwards in it
  edit the failing test.

**M4: after a blocking review, does the fix carry a test for the defect? 33 of 33 code
defects got a test driving them (32 new, 1 extended); 0 were fixed without one.**

- *Population:* the 103 dark_factory tasks with `task_results.review_cycles >= 1`
  completed on or after 2026-07-17. 75 were measured. 22 have no merge commit. 6 have no
  branch commits after the blocking review and were traced one by one (below).
- *Window:* commits on the task's merged branch authored after the reviewer invocation
  that produced the **first blocking** verdict, and before the next reviewer invocation
  started. That reviewer invocation is the last one completing at or before the mtime of
  `.task-meta/<id>/reviews-cycle-1/reviewer_comprehensive.json`. Subjects matching
  `^amend[:(]` are excluded.
- *First-draft correction:* the first draft keyed every window on the *first* reviewer
  invocation. That dropped 15 multi-review tasks and misaligned 4. The critic seat caught
  it.
- *Classification:* blocking issues set against the window diff, case by case, by a sonnet
  seat. No test was run, so "drives the defect" is read from the diff, the test names and
  the `RED` commit labels.

| Class (75 measured windows) | Count |
|---|---|
| Every blocking issue was about a test itself (mostly wording or introspection meta-tests the reviewer is told to block); no new test is owed | 40 |
| A code defect, and the window adds a new test function driving it | 32 |
| A code defect, and the window extends an existing test that drives it | 1 |
| A code defect fixed with no test | 0 |
| A doc or prompt defect, where no runtime test is meaningful | 2 |

- *Test-first order:* checked only for the 9 new code-defect cases of the second pass. In
  7, a RED or test-only commit precedes the fix. In 2 (4848, 6524) the order is mixed.
  For the first-pass 23 it was not checked case by case.
- *The 6 with no branch commits after the blocking review:*

  | Task | What became of the blocking issue |
  |---|---|
  | 2709 | Fixed in the amendment pass. A new test was added after the fix |
  | 4354 | A test-only finding, removed by an implementer review-fix commit that the window logic missed |
  | 5164 | A doc-only rename, reverted by Leo |
  | 4880 | Foreign commits on the branch, removed by a rebase |
  | 3984 | A false positive: an uncommitted change in the review worktree |
  | **2358** | **Never fixed.** Downgraded to a suggestion by later reviews. The cited substring match is still on main (verified by the lead) |

- *Lead's spot-check:* tasks 4878 and 5311 both show a `test(...): RED …` commit
  reproducing the review's finding before the fix.
- *Scope:* M4 covers the replan path (row c) on dark-factory's own build, with the
  architect and implementer on opus at xhigh. It says nothing about rows f, l and m, or
  about a client block on shipped sonnet defaults.
- *Side finding:* 40 of 75 first blocking verdicts concern tests, not product code,
  chiefly the wording meta-tests the reviewer is told to block.

## 4. Candidate mechanisms weighed

The brief's candidates are weighed below. Two facts narrow the field:

- The reviewer is not told the task.
- Every *planned* fix converges on the same place: the replan, a filed fix task, or the
  simple path all produce a plan `test` step.

| Candidate | Where it would live | Machine or prompt | What it costs | What it cannot catch | Verdict |
|---|---|---|---|---|---|
| Debugger rule, "a fix lands with a test that fails before and passes after" | `roles.py::DEBUGGER`, `briefing.py::build_debugger_prompt` | Prompt | Negligible tokens; a few more debugger turns | Most debugger failures come from an existing check, so the rule mostly asks for a test that exists. It cannot run without failure text (5357) | Keep as part of D4: never weaken a failing test without stating in the commit why the test was wrong; add a test only when the fix changes behaviour no failing test exercises |
| Reviewer blocking criterion, "a fix commit without a test" | `roles.py::_REVIEWER_HEURISTICS_TEMPLATE` | Prompt, judged by an agent | The reviewer must first be given the task's defect context, which is itself a change. Each new block is a replan, and a second block is a steward escalation | Judgement drift; adds blocks where a deterministic check costs nothing | Reject as the primary mechanism |
| Merge-lane check that fix-shaped tasks (by title) touch a test file | `merge_lane/gates.py` | Machine | One diff inspection | Touching a test file is not capturing the bug: M1's lint-only touches would count. Title routing breaks heuristic 12 (structured data instead of meaningful strings). 94% already pass (M2) | Reject |
| Task-authoring rule plus a `delivered_check` per fix task | `docs/task-authoring.md` §3.3 | Machine | Per-task authoring work | The author cannot name a test the architect has not written yet. `shared/src/shared/delivered_check_polarity.py` evaluates `grep` and `path` checks against a tree; it does not run tests | Reject; borrow only its polarity idea (fail before, pass after) |
| Structured defect, a reproducing plan step, a base-vs-tip proof, and a ratchet (§5 Tier 2) | Task metadata vocabulary; the reviewer verdict contract; a workflow predicate; a proof runner on the detached-worktree pattern of `verify.py::verify_failure_is_preexisting_on_main` | Machine, plus D4 | See §5 | See §5 | **Recommend** |
| Project-config flag (client only, or all) | `config.py::OrchestratorConfig` | Config | One key, plus a branch at each consumer | none in itself | Not needed (§5) |

## 5. Recommendation

**One rule for every project, as six parts in two tiers, authored as one `/prd`.**

### Tier 1: makes the trimmed wording true (two to three small tasks)

- **D4. One prompt rule, held in one constant** (heuristic 11, SPOT).
  - It lives in `roles.py`, alongside `SCOPE_BOUNDARY_GUIDANCE`, and is spliced into
    `ARCHITECT`, `build_replan_prompt`, `SIMPLE_TASK`, `DEBUGGER`, `STEWARD` and
    `build_amender_prompt`.
  - It says: a defect fix starts from a test that fails without it. Never weaken a failing
    test without stating in the commit why the test, not the code, was wrong.
  - `skills/unblock/SKILL.md`, `skills/unblock-low-risk/SKILL.md` and `CONTRIBUTING.md`
    §1(b)/§7 do not restate it; they point to the constant as its single home (INV-9).
    Skills and Leo's hotfixes cannot import a Python constant, so this is the only
    no-copy form.
  - About 150 tokens per prompt.
  - It does not reach the merger, and for the human paths it is a convention only.
- **D6. Breadth by default.** *Superseded by Leo's ruling of 2026-10-15, filed as
  task 7008:* full merge verify and scoped task verify become the shipped project-policy
  defaults (`config.py` and `defaults.yaml`), not a `factory-init` write. A project opts
  down to `scoped` explicitly.
  - That puts every test in every merge gate, so no "client" flag is needed for breadth.
  - Its cost (a full suite per merge) is visible per project, which is the axis
    `merge_verify_breadth` already exists for.
  - dark-factory already runs `full`.

### Tier 2: what restoring "every … kept and run on every later version" requires

- **D1. A defect is structured data** (heuristic 12, structured data instead of meaningful
  strings).
  - *Task level:* `metadata.defect` carries an origin of `review`, `red_main`,
    `review_instrument`, `agent_followup`, `human`, `client` or `exploratory`, a summary
    and a ref. It is set at the filing sites that exist today:
    - `merge_lane/worker.py::compose_fix_main_brief`;
    - the `/review`, `/hotspot-survey` and `/review-all` templates;
    - `DEEP_REVIEWER`'s filing;
    - `_FOLLOWUP_FILING_INSTRUCTIONS`;
    - the curator's routing of reviewer suggestions;
    - hand filing per `docs/task-authoring.md`.

    It is validated at write time, as the existing metadata vocabulary is.
  - *In-task:* the reviewer's `category` is free text today, so it cannot tell a code
    defect from a test-quality finding without routing on strings. Add an enum `kind` of
    `defect`, `test_quality`, `docs` or `process` to `_REVIEWER_CONTRACT_TEMPLATE`, and
    bump `_REVIEWER_PROMPT_HARNESS_VERSION`. A blocking `defect` is then the declaration
    for the replan. Test-quality and docs findings owe no reproduction; on M4's evidence
    they are most first blocking verdicts.
- **D2. One predicate, called at every point a plan is made** (heuristic 10, clear
  invariants informatively, redundantly, uniformly enforced).
  - `add_plan_step` gains a `reproduces` field.
  - A workflow-side `plan_reproduces_declared_defects(plan, defects)` refuses a plan in
    which a declared defect has no `test` step with `reproduces` ahead of its fixing
    `impl` step, and names the defect in the refusal. A declared `not_reproducible` with
    a reason is the alternative, and it is shown to the reviewer and on the client page.
  - It is called after `_plan`, after `_replan` (beside the existing "Architect replan
    produced no valid steps" check), and after `_run_simple_task`.
  - A declared defect stays declared until it is reproduced and fixed, or given a
    recorded disposition. A later review cannot drop it silently, as happened to task
    2358's (M4).
  - `confirm_plan` alone would not do: the replan and the simple path never call it, and
    the plan-tools server cannot see task metadata.
  - It is deterministic and runs no tests.
- **D3. The machine proves fail-before, pass-after.**
  - In a detached worktree at the merge-base, overlay the reproducing test file(s) from
    the tip. The reproducing test ids must fail there and pass at the tip.
  - An import error counts as a failure only when the imported name is absent at the base
    and added by the diff.
  - Proving against the base, not the step's own commit, survives the rebase and WIP-squash
    re-pointing that `workflow.py::_reconcile_done_step_commits` does to step commits.
  - It reuses the detached-worktree and `run_scoped_verification` pattern of
    `verify.py::verify_failure_is_preexisting_on_main`.
  - A failed proof goes to the architect for a replan, bounded by the existing
    review-cycle counter. It does not go to the debugger, which cannot create an earlier
    failing commit.
  - It shares the *idea* of `delivered_check_polarity` (fail before, pass after) but not
    its code, which evaluates grep and path checks. So it is not a SPOT reuse.
- **D5. "Kept".**
  - The reproducing test carries its defect ref as a marker (`@pytest.mark.defect("<ref>")`).
  - A deterministic ratchet refuses any merge after which the set of collected defect refs
    shrinks without a recorded disposition. The precedent is the inline-suppression
    ratchet, `scripts/inline_suppressions.py`.
  - Showing each bug, its test and its result on the client page reuses the client feed's
    planned `criterion` marker plugin (`plans/client-feed-prd.md`). That feed is unbuilt,
    so this display half is a seam the `/prd` must own (G4). The ratchet itself does not
    depend on it.
  - Pytest only at first, as the client feed is.

### Everywhere, not client-only

- **Dogfooding.** A rule run only on client blocks would run mostly on rare, paid work,
  so its own bugs would surface there first. dark-factory's self-build is the factory's
  test bed.
- **No flag needed.** Adding a per-project switch would add a second axis to vary
  (heuristic 3, carefully factored orthogonal dimensions of variability). That is worth
  it only when projects genuinely differ. Here the one real cost difference, a full suite
  per merge, already has its own key (D6).
- **One mechanism for planned fixes.** The defect's origin is a value, so every path that
  *files or replans* a defect shares D1–D3. The in-task paths that never plan (rows b, d,
  f, l, m) rely on D4.
- **If Leo still wants client-only:** a single `Literal` key on `OrchestratorConfig`, read
  by the D2 predicate and the D3 proof, not a flag per site.

### Costs

- *Tokens:* D4 adds about 150 tokens to six prompts.
- *Declarations:* D2 adds a reproducing step only where one is missing. On the replan path
  M4 found none missing. How many filed fix tasks lack one is unmeasured. Without D1's
  `kind` enum, every test-quality finding would need a `not_reproducible` declaration.
- *Compute:* D3 costs one scoped test run on the base tree per declared defect; seconds
  for pytest.
- *Escalations:* a defect with no reproducible test is declared, not looped. A failed proof
  spends one replan inside the existing cap.
- *Size:* roughly seven tasks: D1 vocabulary and filing sites; D1 verdict enum; D2
  predicate; D3 proof runner; D4 constant and pointers; D5 ratchet; D6 factory-init.

### What it still cannot catch

- A bug an agent fixes inline instead of filing it (against `_FOLLOWUP_FILING_INSTRUCTIONS`).
- A test that discriminates the fix but under-describes the bug.
- A defect no automated test can reproduce. It is declared and shown, not proven.
- Non-pytest stacks, until the proof runner and the marker support them.
- The client defect window (row n). It needs a portal-to-factory bridge that files a
  `client`-origin task. That is portal work. Once filed, the task falls under D1–D3.
- Whether the habit transfers to a client block on shipped models. Re-measure by M4's
  method after the first block.

## 6. What the sentence over-claims, and what to say instead

**Recommended now:** trim. The table gives wording true today. The last column says what
lets the original stand.

| Clause as written | Mechanical today | By prompt | Not at all | Say now | Original stands once |
|---|---|---|---|---|---|
| "Every change starts from a failing test." | none | Planned `test` steps (`ARCHITECT` rule 4, `IMPLEMENTER` rule 2), self-attested | Doc-only changes (architect rule 5 plans "a single `impl` step … (no test)"), the simple path, debugger, steward, amender and merger changes, lint and type fixes | "New behaviour is written test-first, as standard practice." | Never literally ("every change"); D3 extended to every `test` step would make "each planned change" mechanical |
| "Huge numbers of automatic tests are written" | none | none | Leo's words; for one block it depends on the block's size. `docs/code-quality.md` lists "Test count or test-to-code ratio" under "Do not steer by" | "Many automatic tests are written" | Leo's call |
| "most of them before the code that they test (test driven development)" | none | Yes | Not measured as a ratio | Keep | none |
| "for all the individual parts of the code (unit tests)" | none | none | "all": there is no coverage target; `docs/code-quality.md` counts coverage "only when the fixtures are real" | "for the individual parts of the code (unit tests)" | none |
| "all tests are run before every significant change is accepted (continuous integration)" | Only under `full` (whole suite on the merged tree per merge; docs-only trivial pass excepted). Under `scoped`, the merge gate runs only touched test files | none | No CI service (`CONTRIBUTING.md`: "There is no CI."); the merge gate does that job | "the project's tests are run before every change is accepted" | D6 is in place for the project |
| "Every bug or gap found during the build gets its own test, kept and run on every later version." | Nothing | Gaps via tests-first; bugs by habit on planned paths (M4) | No per-bug rule; "kept" unprotected; "every later version" false under `scoped` | "A bug found during the build is normally captured by a test before it is fixed, and that test joins the project's suite." After Tier 1, add: "…which runs on every merge." | Tier 2 (D1–D3, D5) and D6 are built |
| "The project's tests, lint and type checks run on every merge" | Literally yes, but under `scoped` they run on touched files only | none | none | "The project's tests, lint and type checks run before every merge" | D6 |
| "an AI reviewer reads every change of code" | Yes for task work (normal and simple paths), on a diff cut at 50,000 characters | none | Deterministic tasks and Leo's direct hotfixes (prior survey, not re-verified). Merge-conflict resolutions and steward fixes: re-review not checked | "an AI reviewer reads every task's change" | Re-review of steward and merger changes is confirmed |
| "The acceptance suite runs against the whole." | No factory mechanism runs it (`plans/client-feed-prd.md` §11) | none | Unless the block's set-up puts the suite in the project's `test_command` | "Leo runs the acceptance suite against the whole before the demonstration", or keep if the set-up puts the suite in the test command | The suite is in `test_command` for every client project |

## 7. Corrections to the prior survey and the brief

- **"Clients get `merge_verify_breadth: full` per the client-portal design."** Only as a
  manual install step (`client-dashboard-design.md` §10, "Per client project:
  `merge_verify_breadth: full` …"). Neither PRD (`client-portal.md`,
  `plans/client-feed-prd.md`) nor `factory-init` sets it.
- **"MERGE re-verifies touched modules."** Under `scoped` the merge gate re-runs touched
  *test files* (lint and type on touched files). The task-role VERIFY is what runs touched
  modules' full suites, before rebase. The full suite also runs on the main tip every 30
  minutes by default, whatever the breadth.
- **"VERIFY failures … capped at 5 attempts."** That means 5 verify runs and at most 4
  debugger runs. Three identical failure signatures escalate sooner
  (`max_failure_signature_repeat`).
- **"Up to 2 review cycles."** That means one replan; the second blocking verdict escalates
  to the steward. Both counters run over the task's lifetime.
- **"Implementer (Sonnet)."** Sonnet is what `defaults.yaml` ships for the implementer and
  debugger. This repo overrides both to opus at xhigh (`dark-factory-orchestrator.yaml`
  `models`). The reviewer is opus at high in `defaults.yaml`.

## Appendix: method and artefacts

| Seat | Model / effort | Tokens | Minutes | Output |
|---|---|---|---|---|
| A, inner-loop paths | sonnet / high | 199k | 31 | Rows a–g and the survey checks (all four confirmed) |
| B, outer-loop paths | sonnet / high | 181k | 29 | Rows h–n, breadth, CI, configuration loci |
| C, measurement | sonnet / high | 119k | 21 | M1, M2. reify was not measurable |
| E, M4 classification (two passes) | sonnet / medium | 176k | 20 | M4: first pass 63 cases; after the critic's window fix, 16 re-classified and 6 no-window cases traced |
| D, critic of the assembled draft | opus / high | 238k | 36 | 19 findings: 4 high, 8 medium, 7 low. All folded in; the M4 window fix was re-measured |

The lead recounted seat C's figures, wrote the M4 window scripts, and spot-checked M4
cases. Scripts and per-case data are under
`/tmp/claude-1000/-home-leo-src-dark-factory/971a4a12-4594-49f0-9cf5-03286944ac73/scratchpad/`
(`seatC/`, `lead/m4v2.py`, `lead/m4v2_cases.json`, `lead/m4v2_status.json`). That is
session scratch, so re-derive from the stated rules if it is gone.
