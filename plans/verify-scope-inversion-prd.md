# PRD — verify-scope inversion: narrow task verify, broad merge gate (reify parity)

**Status:** queued · 2026-07-13 · agent-legibility survey follow-on
(`plans/agent-legibility-survey-2026-07-13.md` §1.4 + Addendum Q2;
codebook `docs/legibility/confusion-codebook.yaml` id `verify-scope-asymmetry`,
16 incidents). Owner (Leo) ratified the direction 2026-07-13: adopt reify's
pattern — narrow task-specific verification in the task worktree, broad
verification at the merge gate — for Dark Factory, expressed through the W7
verify-plan layer. Coverage at the gate must not drop; wall-clock wins come
from the inner loop.
**Approach:** B + H (high stakes — merge-gate correctness; blast radius
verify.py/verify_plan.py/merge_queue.py/workflow.py/config; cross-PRD seams
with W7, 2564-2567, 2549, 2501). Contract + boundary-test sketch below.
**Owns (G4 authoritative):** role-differentiated scope policy in
`derive_verify_plan`, plan-authoritative execution in
`run_scoped_verification`, the `merge_verify_breadth` knob, full-suite
baseline attribution at the merge gate, and infra-outcome attempt-consumption
policy.

## Goal

DF's verify scope policy is **role-blind**: `scope_module_config`
(verify.py:1432) applies the same narrowing at task-level verify and at the
merge gate, so both ends are wrong in opposite directions (all verified on
main 2026-07-13):

- **The gate is narrow where it must be broad.** Pytest runs a module's full
  suite only when conftest/test-data files are touched; otherwise it is
  file-scoped to touched **test** files; a **source-only diff runs zero pytest
  at the gate** (`verify_plan.py:318-322` — "no collectable test files touched
  — nothing to run"; same decision in `scope_module_config`,
  verify.py:1538-1541 `test_cmd = None`). Sibling-test breakage lands green;
  broad coverage exists only as post-merge unscoped *typechecks*
  (`_run_unscoped_typechecks`, merge_queue.py:1547) and after-the-fact
  main-tip sweeps. The mined red-mains came exactly from this hole.
- **Task verify is broad where it should be narrow.** Whenever scope widens —
  conftest/test-data → full module suite, structural file → unscoped pyright,
  unregistered path / derive-failure → the global whole-repo fan-out chain
  (orchestrator/config.yaml:41) — the task inherits main's pre-existing reds
  and infra flakes as its own failure, on every iteration (task verify rebases
  onto main before each attempt, workflow.py:5313-5321). Infra-shaped
  outcomes (semaphore timeouts, ENOSPC, heartbeat kills) still consume verify
  attempts and dispatch debuggers who produce zero-diff churn.

Reify, after heavy verify investment, runs the inversion and it works in
production: task role `--scope branch` / profile=debug (fast TDD); merge path
injects `DF_VERIFY_ROLE=merge` → `--scope all` / profile=both; merge trains
(max 3) amortize gate cost (`reify/orchestrator.yaml` ~:97-131, ~:335).
Rationale ratified: a gate false-negative (red main) taxes every subsequent
task and human attention; a task-verify false-negative costs one iteration.
DF's full warm suite is **~2 minutes** (orchestrator/config.yaml:55 comment)
vs reify's ~90-minute builds — if reify affords a broad gate, DF trivially
does.

**User/operator-observable outcome when this lands:**
- A source-only diff that breaks a sibling module's test is **rejected at the
  merge gate** (today: lands green, discovered by a sweep hours later; red-main
  triage is the most attention-expensive escalation class).
- A task touching only source files gets **pre-gate pytest signal from its
  owning module's suite** (today: zero pytest until after merge).
- A merge-gate failure that is **pre-existing on main** never blames the
  branch (baseline-diff attribution over failing test ids, not category
  guesswork).
- An **infra-classified** verify outcome consumes zero verify attempts and
  dispatches no debugger, at both the task and merge consumers.
- A task scoped to an unregistered path runs a **scoped fallback**, never the
  whole-repo fan-out chain (the measurable task-verify wall-clock win).

## Consumers (G1)

- **The merge queue / main branch health** — the broad gate closes the
  red-main class (16 mined incidents).
- **Task agents (implementer/debugger)** — owning-module pytest signal before
  the gate; no unrelated-red misattribution churn.
- **Leo / AFK operation** — red-main triage and false-blame escalations are
  the most attention-expensive classes; both shrink.
- Per-mechanism consumers are named in the decomposition plan.

## Sketch of approach

Six mechanisms + one deploy capstone. The verify.py/verify_plan.py work
chains linearly **after the W7 spine tip** (θ = task 2147, ι = task 2148) —
same file-lock discipline as W7 (its α→…→θ chain, Resolved decision 1 there).

1. **κ — plan-authoritative execution.** `run_scoped_verification` becomes
   derive→execute→aggregate: the `VerifyPlan` from `derive_verify_plan` is
   **executed**, not merely attached as a diagnostic. `scope_module_config`'s
   independent decision tree (the hand-mirrored twin documented by the drift
   notes at verify_plan.py:243-251 and verify.py:1495-1507) dies; predicates
   fold into the plan layer. Behaviour byte-identical (goldens): this task
   changes *who decides*, not *what is decided*.
2. **λ — role-differentiated scope policy + breadth knob.**
   `derive_verify_plan(role=...)` (parameter exists; today it only sets
   `needs_pipeline_guard_check`, verify_plan.py:601) becomes the policy fork:
   - `role='merge'` under `merge_verify_breadth: full`: every **registered**
     ModuleConfig contributes FULL_SUITE runs for each configured command
     (pytest+ruff+pyright), on the merged tree. Per-module commands — never
     the OPAQUE global &&-chain. TRIVIAL (docs-only) short-circuit preserved.
   - `role='task'`: current narrowing, plus **owning-module pytest for
     source-only diffs** (the Python analogue of reify's `--scope branch`,
     package granularity): source files under a module prefix → that module's
     full `test_command`; touched-test-only diffs keep file-scoped selection;
     conftest/test-data/structural widening rules unchanged. Never widens
     beyond the owning modules of the diff. Selection errors are safe — the
     broad gate is the net.
   - New config knob `merge_verify_breadth: Literal['scoped','full']`,
     default `'scoped'` (byte-identical legacy) until σ flips it. Task-role
     policy is unconditional (strictly better signal).
   - Train verifies (`_do_train_merge` workspace verify; train-member
     `force_workspace=True` at workflow.py:5261) route through the same
     merge-role plan when breadth=full (per-module commands replace the
     opaque global chain).
3. **μ — broad-gate baseline attribution.** Extend
   `verify_failure_is_preexisting_on_main` / `_classify_main_health_red`
   (merge_queue.py:738) from category+cause_hint matching to **failing-test-id
   diff against a per-main-SHA full-suite baseline**: merge-role pytest runs
   get `--junitxml` injected structurally (VerifyCmd `base_flags`); failing
   ids are parsed from XML (stdlib); a gate failure blocks the branch **only
   for failures not in the baseline**; pre-existing failures route to the
   existing MAIN_HEALTH_RED path. Baseline cache is **seeded by each
   successful gate run** (the just-verified merged tree *is* the next main
   tip — zero marginal probe cost by induction); a cache miss pays one main
   full-suite probe, cached per main SHA. Coordinates with 2564-2567 (probe
   scheduling/transport — see G4).
4. **ν — infra outcomes never consume attempts.** A verify outcome whose
   `CategoryPolicy.is_infra_transient` is true increments no
   `verify_attempt` in `_verify_debugfix_loop`, dispatches no debugger, and
   consumes no merge verify attempt — requeue/hold instead, keeping the
   existing infra-hold/escalation pathways. Depends on 2549 (classifier
   patterns for semaphore/ENOSPC/SIGBUS/psi-gate) so coverage is real.
5. **ξ — B+H integration gate** — two-way boundary tests over the whole
   inversion (sketch below), including one exact historical red-main-shaped
   incident diff as a plan golden (W7's testing style, its Resolved
   decision 6).
6. **σ — config flip:** `merge_verify_breadth: full` +
   `merge_train_former_enabled: true` + `merge_train_coalesce_enabled: true`
   + explicit `merge_train_max_members: 3` (reify GO-N3 precedent) in
   orchestrator/config.yaml.
7. **τ — deterministic deploy capstone:** restart the DF orchestrator fleet
   (config.yaml is not in `orchestrator_restart_watch_prefixes`, so a
   config-only merge does not auto-restart; these knobs are not
   hot-reloadable).

## Resolved design decisions (do not relitigate)

1. **The plan becomes authoritative (κ), reversing γ-2126's diagnostic-only
   trade-off.** W7's γ landed `derive_verify_plan` as an observability mirror
   with a documented "Fidelity" trade-off — acceptable while the two trees
   computed the *same* answer. Role-differentiated policy makes them compute
   **different** answers by design; hand-mirroring a policy fork across two
   decision trees is exactly the "same bug fixed twice" class W7 exists to
   kill. The drift notes at both sites already name this hazard.
2. **Broad gate = every registered module's own commands, not the global
   &&-chain.** The chain (orchestrator/config.yaml:41) is OPAQUE to VerifyCmd
   (P1: never scoped/mutated — no junitxml injection, no per-tool
   classification). Per-module commands are parseable PYTEST/RUFF/PYRIGHT,
   classify per-tool (δ-2131), and admit structured flag injection. The chain
   survives only as the legacy no-module-configs fallback.
3. **Package-granularity reverse-dep selection at task role, not an import
   graph.** Reify's `--scope branch` is crate-granularity; DF module suites
   are seconds-to-a-minute. A static import-graph selector is over-machinery
   with real failure modes; the gate is the correctness net either way.
   Cross-module breadth at task level is deliberately omitted.
4. **Baseline attribution is per-failing-test-id, seeded by gate successes.**
   Category-granularity would mask a NEW failure in a module that already has
   a pre-existing red of the same category. Seeding from the winning gate
   run's own result makes steady-state baseline cost zero; only the first
   failure against an unseeded main SHA pays a probe (bounded, cached —
   existing `_PROBE_CACHE` pattern generalized). OPAQUE commands degrade
   gracefully to today's category-level attribution.
5. **Staged rollout behind `merge_verify_breadth`, default `'scoped'`.**
   The gate is the most load-bearing lane; orchestrator restart-on-merge
   auto-fires on orchestrator/src changes, so code lands must not flip
   behaviour implicitly. σ flips the knob only after μ (attribution) and ν
   (infra policy) are landed and ξ is green — without μ, a broad gate would
   re-charge every legacy pre-existing red to innocent branches.
6. **Trains on at N=3.** The train former/coalescer is DF code (tasks
   1704-1708), production-proven in reify (GO at N=3; s(3)=0.962, coupling
   failure 0/104). Broad gate cost is amortized across up to 3 members —
   same correctness gate as single merges (train tip verified before
   advance).
7. **Infra non-consumption is policy-table-driven,** keyed on α-2123's
   `CategoryPolicy.is_infra_transient` — no new string registries, one row
   per category, import-time exhaustiveness already enforced (F1).

## Pre-conditions for activating

- **Substrate (all verified on main 2026-07-13):** `derive_verify_plan` with
  `role` param + `ScopeKind.TRIVIAL` + docs-only short-circuit
  (verify_plan.py:138-144, :541-601; verify.py:3838); `scope_module_config` +
  fallback twin (verify.py:1432, :1693); `VerifyCmd.base_flags` structural
  flag surface (verify_cmd.py:71); pytest `--junitxml` (pytest builtin,
  stdlib XML parse); `verify_failure_is_preexisting_on_main` +
  `_PROBE_CACHE` + `escalate_preexisting_main_break` default-true
  (merge_queue.py:738; config.py:1611); `CategoryPolicy.is_infra_transient`
  (verify_categories.py:74); train code present (`_do_train_merge`
  merge_queue.py:3227; knobs default-false in defaults.yaml:498-500);
  `scripts/restart-all-orchestrators.sh` exists and is executable (deploy
  capstone `before_done`); role plumbing end-to-end (workflow.py:5262
  role='task'; merge_queue role='merge' sites).
- **Hard external deps (filed, wired at decompose):** W7 spine tip
  **2147 (θ)** + gate **2148 (ι)** — κ mutates the layer ι proves; **2564**
  (mainprobe off critical path, design-first) — μ co-touches the probe
  neighborhood; **2549** (infra classifier patterns) — ν's coverage; **2501**
  (per-project verify-slot scoping) — σ raises gate load on a host shared by
  ~6 orchestrators.
- W7's δ-2131 (in-progress) and ε-2133 sit between the current verify.py tip
  and 2147 in W7's own chain; our chain inherits them transitively.

## Cross-PRD relationship (G4)

| Other stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| **W7** verify-plan (`plans/verify-plan-prd.md`, wave 1) | extends | `derive_verify_plan`/`VerifyPlan`/`ScopeKind`; `run_scoped_verification` decision layer | **W7** owns until its spine completes; **this PRD** owns plan-execution authority + role policy after 2147/2148 (W7 declared scope-widening out of scope — this is that follow-on) | wired — κ deps 2147+2148 |
| **2564-2567** mainprobe-stall program | co-touches | `verify_failure_is_preexisting_on_main` / `_classify_main_health_red` | **2564** owns probe scheduling/transport (off-critical-path, host-affinity, warm probe); **this PRD (μ)** owns the baseline-diff decision policy | wired — μ deps 2564 |
| **2549** verify classifier residuals | consumes | CATEGORY_POLICY infra patterns + failure_report excerpts | **2549** owns patterns; ν consumes `is_infra_transient` rows | wired — ν deps 2549 |
| **2501** per-project verify slots | consumes | per-project slot dirs for verify admission | **2501** owns; σ deps it (contention headroom before breadth+trains flip) | wired — σ deps 2501 |
| **W9** workflow-state-machine (wave 2, unfiled) | future co-touch | workflow.py block/attempt paths (ν edits attempt-consumption) | ν limited to attempt-count policy at existing sites; W9 wires deps to ν's id at its own decompose | noted |

No reciprocal-ownership ambiguity: every seam has exactly one owner.

## Contract section (B + H)

**Role policy (`derive_verify_plan`)**
- **R1 (divergence):** with `merge_verify_breadth='full'` and a non-TRIVIAL
  diff, `role='merge'` ⇒ every registered ModuleConfig contributes a
  FULL_SUITE PlannedRun per configured command; `role='task'` ⇒ runs are
  confined to the owning modules of the diff (plus the existing
  structural-pyright widening), never all-modules.
- **R2 (TRIVIAL parity):** a no-source diff ⇒ `ScopeKind.TRIVIAL` at both
  roles — the docs-only fast-path survives the inversion.
- **R3 (task-role pytest floor):** a source-only diff under a registered
  module ⇒ that module's full `test_command` at task role (today: zero
  pytest). Touched-test-only diffs keep file-scoped selection.
- **R4 (knob):** `merge_verify_breadth='scoped'` ⇒ merge-role plans are
  byte-identical to legacy (golden-tested rollback path). Validated as a
  `Literal` at config load.

**Plan authority (`run_scoped_verification`)**
- **A1:** the runs executed are exactly `plan.runs` (execution-time skips
  recorded on the plan); `VerifyResult.plan` is the *executed* plan. The
  fidelity drift notes die with the second decision tree.
- **A2:** OPAQUE commands are never scoped/mutated (W7 P1 preserved); an
  OPAQUE module command at merge role runs verbatim and degrades μ to
  category-level attribution for that module only.

**Baseline attribution (μ)**
- **B1:** a merge-gate verify failure blocks the branch only for failing
  test ids **absent from the baseline** for the same main SHA; wholly
  pre-existing failures route to the existing MAIN_HEALTH_RED escalation
  path (fingerprint-deduped), never to the branch.
- **B2 (seeding):** a successful gate run's full-suite result seeds the
  baseline for the main SHA it advances to; steady-state baseline cost is
  zero.
- **B3 (degradation):** no junitxml (OPAQUE / non-pytest tools) ⇒ fall back
  to today's category+cause_hint probe for that run — never a crash, never
  silent branch-blame.

**Infra policy (ν)**
- **I1:** `is_infra_transient=True` outcomes increment no verify-attempt
  counter, dispatch no debugger, and consume no merge verify attempt at
  either consumer; they requeue/hold via existing pathways
  (infra-hold/requeue), preserving loud escalation on exhaustion.

**Trains (λ/σ)**
- **T1:** `_do_train_merge`'s verify and single-merge verify run the same
  merge-role plan under breadth=full (per-module commands); train members'
  task-role verifies keep union scope.

## Boundary-test sketch (B + H — task ξ's signal)

| # | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|
| 1 | **The hole, closed** | one historical red-main-shaped source-only diff (mined per ξ's recipe; constructed two-module shape as fallback) breaking a sibling module's test; breadth=full | merge gate REJECTS: blocked outcome, report names the failing sibling test; plan shows FULL_SUITE for the sibling module |
| 2 | Task-role signal | same diff, role='task' | plan contains the owning module's full test_command; no cross-module runs (R3, R1-task) |
| 3 | Docs-only | .md-only diff, both roles | TRIVIAL, zero commands executed (R2) |
| 4 | New-vs-preexisting | baseline has module-X red; branch adds NEW failure in module Y | branch blocked citing only Y; X routes MAIN_HEALTH_RED (B1) |
| 5 | Wholly pre-existing | baseline red only, branch clean | MAIN_HEALTH_RED path, branch not charged, no debugfix dispatch; second merge same main SHA hits baseline cache (B1/B2) |
| 6 | Infra non-consumption | semaphore-timeout-shaped output (2549 pattern) at task verify AND merge verify | attempt counters unchanged, no debugger; requeue/hold taken (I1) |
| 7 | Train amortization | 3 line-stackable members, breadth=full | exactly ONE full-breadth verify of the train tip; per-module commands, not the opaque chain (T1) |
| 8 | Rollback path | breadth='scoped' | merge-role plan byte-identical to legacy goldens (R4) |
| 9 | Fallback narrowing | diff touching only tests/scripts/ (registered scripts module), role='task' | scoped commands only — the whole-repo fan-out chain never appears in the plan (the wall-clock win, asserted structurally) |
| 10 | Plan authority | mixed diff, both roles | executed commands == plan.runs (spy); VerifyResult.plan is the executed plan (A1) |

## Decomposition plan

Fresh greek labels (κ…τ — W7 used α…ι). **verify.py/verify_plan.py spine is
linear** (κ→λ, chained after W7's 2147/2148); **merge_queue spine** μ→ν;
ξ is the B+H gate; σ→τ the staged enable. Task ids assigned at decompose;
capability manifest beside this PRD.

- **κ — plan-authoritative execution** (verify.py, verify_plan.py).
  *force_full_path* (looks mechanical; replaces a load-bearing decision
  tree). **Signal:** the drift-note twin is gone — `scope_module_config`'s
  independent tree no longer exists; scope-behaviour goldens (conftest /
  test-data / structural / source-only / fallback) byte-identical
  pre-vs-post; executed == planned (A1 spy test). **Consumer:** λ (the
  policy fork needs one tree), VerifyResult.plan consumers, ι-2148's proven
  contract. **Prereq:** 2147 (θ, verify.py lock tip), 2148 (ι — W7 batch
  proven before its layer is mutated).
- **λ — role-differentiated policy + `merge_verify_breadth` knob**
  (verify_plan.py policy; config.py + defaults.yaml knob; train routing:
  merge_queue.py `_do_train_merge` verify + workflow.py:5261 train-member
  `force_workspace` consume the role plan under breadth=full).
  **Signal:** plan goldens — source-only diff yields (merge+full: FULL_SUITE
  every registered module; merge+scoped: legacy-identical; task:
  owning-module suite); docs-only yields TRIVIAL at both roles; plan
  `reason` strings name the role and coverage (the survey's "sibling tests
  NOT run" signpost, now role-aware). **Consumer:** the merge gate + task
  verify call sites (μ, workflow), σ. **Prereq:** κ.
- **μ — broad-gate baseline attribution** (merge_queue.py probe
  neighborhood; verify.py probe; verify_cmd junitxml injection).
  **Signal:** boundary rows 4/5 — NEW-only blame over failing-test-id diff;
  baseline seeded by a successful gate run (cache hit observable on the
  next failure, zero extra probe); OPAQUE degradation to category-level
  (B3). **Consumer:** merge worker block path, MAIN_HEALTH_RED consumers,
  the AFK operator. **Prereq:** λ; external 2564 (probe
  scheduling/transport owner — avoids rebase-thrash on the same
  neighborhood).
- **ν — infra outcomes never consume attempts** (workflow.py
  `_verify_debugfix_loop` attempt accounting; merge_queue verify-attempt
  consumption; policy read from CATEGORY_POLICY only). **Signal:** boundary
  row 6 — is_infra_transient outcome leaves attempt counters unchanged at
  both consumers, no debugger dispatch, requeue/hold taken; exhaustion
  still escalates loudly. **Consumer:** workflow/merge consumers, W9 (wave
  2). **Prereq:** μ (merge_queue lock chain); external 2549 (patterns).
- **ξ — B+H integration gate** (ONE new test module,
  orchestrator/tests/test_verify_scope_inversion_boundary.py; boundary rows
  1-10). Row 1's golden diff: mine one incident from the
  verify-scope-asymmetry cluster (main-sweep failures / MAIN_HEALTH_RED
  records / `git log` fix-forward commits); if no minimal reproducible diff
  is extractable, construct the minimal two-module shape (source change
  breaking a sibling module's test) — the cluster's 16 incidents validate
  the premise either way (G6). **Signal (the leaf):** the boundary module
  passes, driving real merge-queue/workflow seams both ways. **Consumer:**
  the merge-gate correctness guarantee / CI; σ's flip evidence. **Prereq:**
  ν (transitively κ/λ/μ).
- **σ — config flip** (orchestrator/config.yaml only; complexity=simple):
  `merge_verify_breadth: full`, `merge_train_former_enabled: true`,
  `merge_train_coalesce_enabled: true`, `merge_train_max_members: 3` + a
  comment block citing this PRD. **Signal:** config committed; drift test
  green; knobs documented restart-required. **Consumer:** τ (the running
  fleet). **Prereq:** ξ; external 2501.
- **τ — deterministic deploy capstone** (`task_kind='deterministic'`,
  `before_done` = `scripts/restart-all-orchestrators.sh`,
  `target_unit='orchestrator-dark-factory.service'` → detached
  self-restart, done=`scheduled`). **Signal:** fresh
  ActiveEnterTimestamp across the fleet; the next merge attempt's logged
  plan carries breadth=full FULL_SUITE runs. **Consumer:** the live gate.
  **Prereq:** σ.

## Out of scope

- **Import-graph / coverage-map test selection** at task role (decision 3 —
  package granularity only; revisit only if owning-module suites grow
  painful).
- **Reify-side anything** — reify already runs the inversion; its configs
  are untouched.
- **Probe scheduling/transport** (2564-2567 own it: off-critical-path,
  host-affinity, watchdog, warm probe). μ only changes the decision policy.
- **Classifier pattern additions and failure-report excerpting** (2549 owns).
- **Per-project verify-slot mechanics** (2501 owns).
- **W7's remaining spine** (δ-2131, ε-2133, θ-2147, ι-2148) — upstream deps,
  not re-filed.
- **Hot-reload support for the new knobs** — restart-only tier; τ restarts.
- The env-transient/venv-isolation root cause (same exclusion as W7).

## Open questions (tactical — AFK defaults recorded)

1. **Failing-test-id extraction shape.** Default: inject
   `--junitxml=<attempt-dir>/junit.xml` into merge-role PYTEST VerifyCmds
   structurally; parse with stdlib `xml.etree`. Fallback considered
   (`-rf` FAILED-line parsing) only if junitxml proves incompatible with a
   module's pytest config. Decide at μ impl.
2. **Baseline persistence.** Default: in-process per-main-SHA cache
   (generalizing `_PROBE_CACHE`), seeded by gate successes; a restart loses
   the seed and the next failure re-probes once (bounded). Persist to disk
   only if restart-frequency makes re-probes noisy. Decide at μ impl.
3. **Owning-module granularity for multi-module source diffs.** Default:
   union of owning modules' suites (matches existing multi-module fan-out
   semantics). Decide at λ impl.
4. **`merge_verify_breadth` name/values.** Default as specified; if config
   review prefers reify-parity naming (`scope: branch|all`), rename at λ
   impl — the contract (R1-R4) is name-agnostic.
5. **Coalescer on at flip vs later.** Default: both train knobs on at σ
   (reify runs both); if train-formation churn appears, drop coalesce first.

## Notes for the decompose session

- Every task filed `planning_mode=True`; wire all deps (intra-batch bare
  ints + external 2147/2148/2564/2549/2501 bare ints) while deferred; flip
  the whole batch in one `commit_planning`.
- `metadata.files` file-level (Contract-1): κ/λ share verify_plan.py —
  linear chain, no parallel dispatch; μ/ν share merge_queue.py.
- Metadata `user_observable_signal` / `consumer_ref` / substrate-confirmed
  flags written for the future tracking-infra session; the orchestrator does
  not read them yet.
- Capability manifest committed beside this PRD as
  `plans/verify-scope-inversion-prd.capability-manifest.md`.
- Concurrency: sibling /prd sessions share this checkout — `git commit
  --only <paths>`, retry on ref-lock, never stash in the project root.
