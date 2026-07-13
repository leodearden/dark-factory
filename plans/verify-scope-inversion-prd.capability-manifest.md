# Capability manifest — verify-scope-inversion PRD

Mechanizes G3 + G6 per task: every capability each task's user-observable
signal asserts, bound to evidence. Verified 2026-07-13 against `main`
(417b30d40c author commit; code refs at 0691d13263). Any binding resolving to
a FAIL value (`declared-only`/`test-only`/`producer-absent`/
`producer-downstream`/`producer-extent-short`/`rejection-absent`) blocks the
batch. **Result: no FAIL bindings.**

Evidence vocabulary: `grep:<file>:<line>` = wired/present on main today ·
`producer:<label/task-id> upstream` = delivered by an upstream task in this
batch's dependency closure · `substrate:<x>` = language/tool substrate ·
`self` = a property the task's own code produces (bound by its RED test) ·
`config:<file>:<line>` = operator-config evidence.

External upstream producers (all filed, all pending, all **upstream** —
DAG-direction PASS): 2147 (W7 θ, verify.py lock tip), 2148 (W7 ι, decision-
layer contract proven), 2564 (mainprobe off-critical-path owner), 2549
(infra classifier patterns), 2501 (per-project verify slots).

---

## κ — plan-authoritative execution

Signal: `scope_module_config`'s independent decision tree is gone; scope
goldens byte-identical pre/post; executed commands == `plan.runs` (spy).

| Capability asserted | Evidence | Verdict |
|---|---|---|
| A `VerifyPlan` exists to promote to authority | `grep:verify_plan.py:549` (derive_verify_plan, role param), `:601` (plan construction); landed by W7 γ (task 2126, done) | PASS |
| The diagnostic-only twin + drift hazard exist to delete | `grep:verify_plan.py:243-251` (drift note), `grep:verify.py:1495-1507` (mirror note in scope_module_config), `grep:verify.py:3574-3597` (`_safe_derive_verify_plan_dict` "diagnostic-only … never consulted") | PASS |
| The execution loop to re-anchor exists | `grep:verify.py:3607` (run_scoped_verification), `:2906` (run_verification) | PASS |
| W7 spine tip + contract proof upstream | `producer:2147 upstream` (θ, last verify.py-lock task), `producer:2148 upstream` (ι boundary suite proves the layer before κ mutates it) | PASS |
| Golden inputs (conftest/test-data/structural/source-only/fallback) derivable from history | W7 DD6 corpus already encodes 1077/1852 diffs (`grep:plans/verify-plan-prd.md:158-161`); source-only shape at `grep:verify_plan.py:318-322` | PASS |

## λ — role-differentiated policy + `merge_verify_breadth` knob

Signal: plan goldens — source-only diff ⇒ (merge+full: FULL_SUITE every
registered module; merge+scoped: legacy-identical; task: owning-module
suite); docs-only ⇒ TRIVIAL both roles; reasons name role+coverage.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `role` already threads end-to-end (plumbing exists; policy fork is the delta) | `grep:verify_plan.py:549`, `grep:verify.py:2695-2708` (DF_VERIFY_ROLE env), `grep:workflow.py:2584-2590` (role='task'), merge sites role='merge' (`grep:merge_queue.py:1589` unscoped typechecks role='merge') | PASS |
| TRIVIAL/docs-only fast-path exists to preserve | `grep:verify_plan.py:138-144` (ScopeKind.TRIVIAL), `:541` (_TRIVIAL_REASON), `grep:verify.py:3838` (docs-only mirror) | PASS |
| Every registered module has full commands for FULL_SUITE emission | `config:escalation/orchestrator.yaml:5-8` and siblings (shared/scripts/dashboard/orchestrator/sampler/fused-memory all carry test/lint/type commands; verified 2026-07-13) | PASS |
| Owning-module suite for source-only is a policy change, not new machinery | `grep:verify.py:1538-1541` (today: `test_cmd = None`), `grep:verify_plan.py:318-322` (today: SKIPPED) — the widening target `mc.test_command` is the same field the conftest branch already uses (`grep:verify.py:1532-1537`) | PASS |
| Config knob surface exists | `grep:orchestrator/src/orchestrator/config.py:1611` (sibling bool field pattern; pydantic Literal validation standard in config.py) | PASS |
| Train verify sites exist to route | `grep:merge_queue.py:3227` (_do_train_merge, workspace verify per docstring (d)), `grep:workflow.py:5260-5262` (force_workspace=train) | PASS |
| One decision tree to fork (no dual maintenance) | `producer:κ upstream` (plan authority) | PASS |

## μ — broad-gate baseline attribution

Signal: NEW-only blame over failing-test-id diff; baseline seeded by a
successful gate run (cache hit on next failure); OPAQUE degrades to
category-level.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Preexisting-on-main probe + gate block path exist to extend | `grep:merge_queue.py:738` (verify_failure_is_preexisting_on_main call), `:748-760` (MAIN_HEALTH_RED blocked outcome + fingerprint), probe cache `grep:merge_queue.py:726-730` (_PROBE_CACHE docstring) | PASS |
| Probe enabled by default | `grep:orchestrator/src/orchestrator/config.py:1611` (escalate_preexisting_main_break default True) | PASS |
| Structural flag injection for junitxml | `grep:orchestrator/src/orchestrator/verify_cmd.py:71` (base_flags tuple field; mutators are structured edits per W7 Contract §VerifyCmd) | PASS |
| Failing-test-id extraction substrate | `substrate:pytest --junitxml` (pytest builtin, no plugin) + `substrate:xml.etree` (stdlib) | PASS |
| OPAQUE degradation path (P1 never mutated) | `grep:plans/verify-plan-prd.md:202-203` (Invariant P1, landed by β-2125 done) — category-level fallback is today's behaviour, retained | PASS |
| Broad merge-role plan upstream | `producer:λ upstream` | PASS |
| Probe scheduling/transport not contested | `producer:2564 upstream` (owns off-critical-path/warm-probe; μ owns decision policy only — G4 seam) | PASS |

## ν — infra outcomes never consume attempts

Signal: `is_infra_transient` outcome leaves attempt counters unchanged at
both consumers; no debugger dispatch; requeue/hold taken; exhaustion still
escalates.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Policy table + flag exist (one row per category, exhaustiveness asserted) | `grep:orchestrator/src/orchestrator/verify_categories.py:74` (is_infra_transient field), F1 import-time assert (W7 α-2123 done) | PASS |
| Task-side attempt counter + debugfix dispatch site | `grep:workflow.py:5305-5335` (verify_attempt loop), `:5429` (max_verify_attempts consumption) | PASS |
| Merge-side attempt consumption site | `grep:merge_queue.py:1024-1137` (verify dispatch + retry accounting neighborhood) | PASS |
| Hold/requeue pathways exist (no new escalation machinery) | `grep:workflow.py:5214-5301` (VerifyInfraError retry + infra-hold stamp via _mark_blocked(block_status='infra-hold')) | PASS |
| Infra categories actually classified (semaphore/ENOSPC/SIGBUS/psi-gate) | `producer:2549 upstream` (pattern rows); existing infra categories already flagged in CATEGORY_POLICY | PASS |

## ξ — B+H integration gate (leaf)

Signal: boundary rows 1-10 green, driving real merge-queue/workflow seams
both ways.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Row 1 rejection mechanism (source-only sibling-breaking diff → blocked) | rejection-check: gate blocks on verify RED today (`grep:merge_queue.py:748-760` MergeOutcome('blocked')); the missing piece is breadth, delivered by `producer:λ upstream`; ξ's test drives the real gate and observes the blocked outcome + named sibling test | PASS |
| Row 1 golden-diff premise (such an incident exists) | 16 mined incidents (`grep:docs/legibility/confusion-codebook.yaml:49-63` verify-scope-asymmetry); mining recipe in-task; constructed two-module shape as G6-honest fallback | PASS |
| Rows 4/5 baseline semantics | `producer:μ upstream` | PASS |
| Row 6 infra non-consumption | `producer:ν upstream` + `producer:2549 upstream` (transitively via ν) | PASS |
| Row 7 train amortization | `grep:merge_queue.py:3227` (_do_train_merge present) + `producer:λ upstream` (routing); knobs test-settable (config object, no daemon needed) | PASS |
| Rows 3/8/9/10 (TRIVIAL parity, rollback golden, fallback narrowing, plan authority) | `producer:λ upstream` (R2/R4), `producer:κ upstream` (A1); fallback lane exists `grep:verify.py:1693` | PASS |
| Boundary-suite harness precedent (drives real seams, no real ssh) | `grep:plans/verify-plan-prd.md:249-266` (W7 ι sketch), prior gates 1737/2260/2309 all landed this shape | PASS |

## σ — config flip

Signal: orchestrator/config.yaml carries breadth=full + train knobs; drift
tests green; knobs documented restart-required.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Knob exists to flip | `producer:λ upstream` (merge_verify_breadth) | PASS |
| Train knobs exist and default off | `grep:orchestrator/src/orchestrator/defaults.yaml:498-500` (merge_train_former_enabled/coalesce false; max_members default 3) | PASS |
| Train machinery landed (DF code, production-proven in reify at N=3) | `grep:merge_queue.py:3227` + reify GO-N3 precedent (reify orchestrator.yaml merge-train block, user-authorized 2026-06-10) | PASS |
| Contention headroom on the shared host | `producer:2501 upstream` (per-project verify-slot dirs) | PASS |
| Gate proven before flip | `producer:ξ upstream` | PASS |

## τ — deterministic deploy capstone (leaf)

Signal: fleet restart scheduled (done_provenance kind
'deterministic-deploy-scheduled'); next merge attempt's logged plan carries
breadth=full FULL_SUITE runs.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Restart script exists + executable (submit-time validation requirement) | `grep:scripts/restart-all-orchestrators.sh` (-rwxrwxr-x, verified 2026-07-13); self-unit deferred last by the script per `config:orchestrator/config.yaml:96-106` | PASS |
| Restart is actually required (not hot-reloadable; not auto-restart-covered) | `config:orchestrator/config.yaml:109-111` (ACTIVATION NOTE pattern: config knobs outside hot-reload allowlist are inert until restart), `:122-127` (watch_prefixes exclude config.yaml — no auto-restart on config-only merges) | PASS |
| DeterministicRunner + detached self-unit path | CLAUDE.md §Deterministic task kind (born-at-L2 + target_unit-own → systemd-run detached, done='scheduled'); precedent deploys plans/1793/1800/1863/1875/1897-deploy-*-restart.md all landed | PASS |
| Config flipped upstream | `producer:σ upstream` | PASS |
