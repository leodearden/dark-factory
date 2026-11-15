# Capability manifest — S16 orchestrator config shape, defaults and routing

PRD: `plans/review-all-s16-orchestrator-config-shape-defaults-and-routing-prd.md`. Machine-readable twin:
`plans/review-all-s16-orchestrator-config-shape-defaults-and-routing-prd.capability-manifest.yaml`
(every `task_id: null` until decompose stamps it; signal leaf ω; 28 leaves after the fold-verify of 2026-11-13).

**Code anchors** verified against main `ba17934861` (2026-11-11) and re-verified against main `ce6673c7e1`
(2026-11-13) by the ruling fold, and at main `fceb951e7e` by the fold-verify. Every `grep:` pointer below was
resolved at `fceb951e7e`, and every check was evaluated there through `shared.delivered_check_polarity`: all
101 producer checks FAIL at that tree, all 11
`*-substrate` checks PASS, and no `absent` check matches a file outside its leaf's declared files. Every
open question is ruled, so no row is OPEN; the gate σ's rows and the live-fleet rows carry `kind: manual`
checks because they describe files outside this tree. Verdict PASS = binding held at authoring. A row
reading "substrate at authoring (deleted by <leaf>)" is never copied to a task as a producer check.


## β0 — Shared config-field declarations (reload tier, legacy spelling, role-table pivot) and the generic hot-reload engine

| Capability | Evidence | Verdict |
|---|---|---|
| annotation-descent-port-substrate | `grep:fused-memory/src/fused_memory/config/reload.py::_ABSENT` substrate at authoring (deleted by β2) — the annotation-driven descent β0 generalises exists on main | PASS |
| reload-engine-in-shared | built here — shared/src/shared/config_reload.py::apply_reload (C4) | PASS |
| reload-tier-derivation-in-shared | built here — shared/src/shared/config_fields.py::reloadable_paths (C3) | PASS |
| legacy-spelling-pivot-in-shared | built here — shared/src/shared/config_fields.py::pivot_legacy_spellings (C3) | PASS |
| restart-only-rows-marker | built here — shared/src/shared/config_fields.py::RestartOnlyRows, per-row tiers on a GREEN role-table field (C3, D4) | PASS |
| table-pivot-declaration-in-shared | built here — shared/src/shared/config_fields.py::TableLegacy, the role-table pivot form (C3, boundary row 4b) | PASS |

## γ0 — Shared role and backend vocabulary: Backend, RoleName, config_role, RoleSettings

| Capability | Evidence | Verdict |
|---|---|---|
| backend-enum-in-shared | built here — shared/src/shared/backends.py::Backend (C1) | PASS |
| backend-enum-carries-zcode | built here — shared/src/shared/backends.py::Backend.ZCODE (C1; task 5399 landed the zcode backend) | PASS |
| backend-enum-carries-mimo | built here — shared/src/shared/backends.py::Backend.MIMO (C1; task 7993 landed the mimo backend) | PASS |
| role-name-enum-in-shared | built here — shared/src/shared/roles.py::RoleName (C2) | PASS |
| reviewer-collapse-declared-once | built here — shared/src/shared/roles.py::config_role (C2, D8) | PASS |
| reviewer-variants-declared | built here — shared/src/shared/roles.py::REVIEWER_VARIANTS and DISPATCHABLE_ROLES (C2, D8) | PASS |
| role-settings-model-in-shared | built here — shared/src/shared/roles.py::RoleSettings (C2) | PASS |

## α — Config defaults have one home: field defaults are authoritative, defaults.yaml keeps only environment-layered leaves

| Capability | Evidence | Verdict |
|---|---|---|
| census-walk-substrate | `grep:orchestrator/src/orchestrator/config.py::census_config_keys` substrate — the model_fields walk the guard mirrors exists on main | PASS |
| defaults-yaml-sheds-restated-concurrency | built here — the restating max_concurrent_tasks leaf leaves orchestrator/src/orchestrator/defaults.yaml | PASS |
| one-home-guard-test | built here — orchestrator/tests/test_config_default_one_home.py (C7) | PASS |
| dashboard-stops-restating-24 | built here — `grep:dashboard/src/dashboard/data/orchestrator.py::_ORCHESTRATOR_DEFAULT_MAX_CONCURRENT_TASKS` is deleted | PASS |
| dashboard-stops-restating-lock-depth | built here — `grep:dashboard/src/dashboard/data/scheduler.py::_DEFAULT_LOCK_DEPTH` (2, against the survivor 4) is deleted or repointed | PASS |
| cross-package-defaults-module | built here — shared/src/shared/config_defaults.py (C8, OQ-4 ruled by the lead) holds DEFAULT_MAX_CONCURRENT_TASKS and DEFAULT_LOCK_DEPTH; created here | PASS |
| concurrency-field-default-names-the-constant | built here — the max_concurrent_tasks Field default names C8's constant (value 24, the D2 survivor) | PASS |
| dashboard-reads-shared-default | built here — the dashboard imports the cross-package default from shared.config_defaults (OQ-4) | PASS |

## ι — Promote the config-default-one-home design invariant through the four-site lockstep edit

| Capability | Evidence | Verdict |
|---|---|---|
| house-pattern-guard | producer:α upstream — orchestrator/tests/test_config_default_one_home.py (C7) | PASS |
| invariant-minted | built here — docs/legibility/design-invariants.md section for config-default-one-home | PASS |
| trigger-shape-registered | built here — skills/prd/references/gates.md family row and inv-trigger-shapes span | PASS |
| fixtures-registered | built here — docs/legibility/design-invariants-fixtures.md PRD and CODE fixtures plus the addendum walk | PASS |

## β2 — Fused-memory declares reload tiers and legacy spellings on its fields; its engine copy config/reload.py is deleted

| Capability | Evidence | Verdict |
|---|---|---|
| fm-reload-tool-substrate | `grep:fused-memory/src/fused_memory/server/tools.py` substrate — the reload_config tool registration β2 repoints; scoped to server/ because S15 ε1 may move it to server/tool_families/ops.py first | PASS |
| fm-engine-copy-gone | built here — `grep:fused-memory/src/fused_memory/config/reload.py::apply_reload` and the module are deleted | PASS |
| fm-reloads-through-shared-engine | built here — fused-memory's reload_config tool imports shared.config_reload (wherever S15 placed the tool under server/) | PASS |
| fm-pivots-legacy-spellings | built here — FusedMemoryConfig's root mode='before' validator calls shared.config_fields.pivot_legacy_spellings (B1 of the critique; boundary row 13b) | PASS |

## β1 — The orchestrator declares reload tiers and legacy spellings on its fields; its engine copy goes

| Capability | Evidence | Verdict |
|---|---|---|
| hand-allowlist-gone | built here — the 338-line RELOADABLE_FIELDS literal in `grep:orchestrator/src/orchestrator/config.py::RELOADABLE_FIELDS` becomes a derivation | PASS |
| derived-reloadable-set | built here — RELOADABLE_FIELDS derived by shared.config_fields.reloadable_paths | PASS |
| orchestrator-engine-copy-gone | built here — `grep:orchestrator/src/orchestrator/config.py::_iter_leaves` (runtime-value descent) is deleted | PASS |
| harness-reloads-through-shared-engine | built here — `grep:orchestrator/src/orchestrator/harness.py::Harness.reload_config` calls shared.config_reload | PASS |
| model-copy-pivots-legacy-keys | built here — OrchestratorConfig.model_copy pivots a declared legacy spelling in update= and raises on an unknown key (boundary row 18b) | PASS |
| census-reports-deprecated-spellings | built here — census_config_keys gains a deprecated bucket (name decided in β1, tactical 2); boundary row 5 | PASS |

## γ1 — Orchestrator backend fields are Backend; invoke_agent dispatches through one total table

| Capability | Evidence | Verdict |
|---|---|---|
| invoke-dispatch-substrate | `grep:orchestrator/src/orchestrator/agents/invoke.py::invoke_agent` substrate — the dispatch entry γ1 rewires | PASS |
| invoke-if-elif-chain-gone | built here — the backend if/elif chain in invoke_agent becomes a Mapping[Backend, Invoker] | PASS |
| backend-fields-typed | built here — UnblockAutoConfig.backend and watcher_backend typed Backend | PASS |

## γ2 — One role-keyed RoleSettings table replaces the knob-major submodels, the unblock-auto role knobs and the watcher role knobs

| Capability | Evidence | Verdict |
|---|---|---|
| knob-major-submodels-gone | built here — `grep:orchestrator/src/orchestrator/config.py::ModelsConfig` and its five siblings are deleted | PASS |
| role-table-on-config | built here — OrchestratorConfig.roles: dict[RoleName, RoleSettings] (D7), with or without its Reload marker | PASS |
| inline-reviewer-collapse-gone | built here — `grep:orchestrator/src/orchestrator/routing.py::_config_key` and the inline startswith in TaskWorkflow._invoke go | PASS |
| watcher-role-knobs-gone | built here — watcher_model and its four sibling role knobs become roles.watcher | PASS |
| known-role-names-derived | built here — shared.task_metadata.KNOWN_ROLE_NAMES derived from DISPATCHABLE_ROLES and REVIEWER_VARIANTS (D8) | PASS |

## δ — The route resolver and the allowlist move to shared; orchestrator.routing keeps the availability probe

| Capability | Evidence | Verdict |
|---|---|---|
| resolver-substrate | `grep:orchestrator/src/orchestrator/routing.py::resolve_route` substrate at authoring (deleted by δ) — the resolver δ moves | PASS |
| structured-rejections | producer:task-6766 upstream — shared.route_rejection.RouteRejection (S25 C4) | PASS |
| invocation-cost-floor | producer:S20-δ upstream — shared.claude_argv.INVOCATION_COST_FLOOR_USD (S20 C6) | PASS |
| resolver-in-shared | built here — shared/src/shared/route_resolver.py::resolve_route (C6) | PASS |
| route-inputs-for-a-role | built here — shared/src/shared/route_resolver.py::RouteInputs.for_role, the fused-memory caller's constructor (C6) | PASS |
| policy-in-shared | built here — shared/src/shared/route_policy.py::RoutingPolicy (C5) | PASS |
| role-budgets-checked-against-floor | built here — shared.route_policy.validate_role_models refuses a Claude row whose budget is below S20's floor (C5) | PASS |
| orchestrator-stock-rules-default | built here — orchestrator/src/orchestrator/config.py::OrchestratorRoutingPolicy carries the stock rule table as its rules default (C5, D2) | PASS |
| config-stops-importing-routing | built here — config.py's import of DEFAULT_ALLOWED_MODELS/DEFAULT_LADDER goes | PASS |
| type-checking-cycle-gone | built here — routing.py's TYPE_CHECKING import of OrchestratorConfig goes | PASS |

## ε1 — Recon agent and judge providers are Backend; the dead SDK arms go

| Capability | Evidence | Verdict |
|---|---|---|
| live-cli-judge-path-substrate | `grep:fused-memory/src/fused_memory/reconciliation/judge.py::Judge._call_judge_cli` substrate — the production path that stays | PASS |
| judge-typo-to-openai-arm-gone | built here — the else → OpenAI arm of `grep:fused-memory/src/fused_memory/reconciliation/judge.py::Judge._call_llm` goes | PASS |
| agentloop-sdk-adapters-gone | built here — `grep:fused-memory/src/fused_memory/reconciliation/agent_loop.py::_OpenAIResponseAdapter` and the SDK arms go | PASS |
| recon-provider-fields-typed | built here — agent_llm_provider/judge_llm_provider stop being bare str | PASS |
| shared-recon-fixture-on-cli | built here — fused-memory/tests/reconciliation/harness_fixtures.py stops configuring the anthropic SDK provider the leaf removes | PASS |
| judge-anthropic-arm-and-prose-path-retired | built here — the judge becomes claude-CLI-only and JUDGE_VERDICT_SCHEMA is the one verdict contract home (OQ-3, ruled by the lead) | PASS |

## ε2a — Recon LLM calls declare a role and receive a validated route

| Capability | Evidence | Verdict |
|---|---|---|
| tier-selection-substrate | `grep:fused-memory/src/fused_memory/reconciliation/harness.py::ReconciliationHarness._select_tier` substrate — the tier choice ε2a routes | PASS |
| shared-resolver | producer:δ upstream — shared.route_resolver.resolve_route, RouteInputs.for_role and shared.route_policy.validate_role_models | PASS |
| recon-budget-hardcode-gone | built here — cli_stage_runner's max_budget_usd=5.0 becomes the stage role's budget_usd | PASS |
| recon-model-fields-gone | built here — sonnet_model, opus_model, agent_llm_model, judge_llm_model become legacy spellings of roles rows | PASS |
| fm-allowlist-validation | built here — FusedMemoryConfig validates its role table against model_routing at load and reload | PASS |

## ε2b — Curator, adjudicators, write-triage judge and the memory LLM take their route from the role table

| Capability | Evidence | Verdict |
|---|---|---|
| write-triage-inheritance-substrate | `grep:fused-memory/src/fused_memory/server/write_triage_judge.py::resolve_judge_model` substrate at authoring (deleted by ε2b) — the inheritance rule OQ-8 retires | PASS |
| write-triage-inheritance-resolvers-gone | built here — resolve_judge_provider/resolve_judge_model are replaced by roles.write_triage_judge (OQ-8, ruled by the lead) | PASS |
| adjudicator-model-field-gone | built here — LinkHealConfig.adjudicator_model becomes a legacy spelling of roles.link_adjudicator.model | PASS |
| curator-routes-per-call | built here — the curator resolves its route per call through shared.route_resolver | PASS |

## η — Service-path knobs leave ReconciliationConfig; fused-memory consumers trust the validated schema

| Capability | Evidence | Verdict |
|---|---|---|
| storm-threshold-bounds-substrate | `grep:fused-memory/src/fused_memory/config/schema.py::Mem0UpdateConfig` substrate — storm_threshold is already Field(default=20, gt=0) (multi-line), so the schema is the invariant home | PASS |
| reconciliation-attr-copies-gone | built here — `grep:fused-memory/src/fused_memory/server/near_duplicate_guard.py::_reconciliation_attr` and its two copies go | PASS |
| restated-anchor-default-gone | built here — `grep:fused-memory/src/fused_memory/services/topic_anchor.py::_DEFAULT_TOPIC_ANCHOR_ENABLED` goes | PASS |
| seed-data-out-of-schema | built here — `grep:fused-memory/src/fused_memory/config/schema.py::_default_topic_guard_clusters` moves to its own module | PASS |

## ζ1 — Harness supervision knobs become submodels: service restart keyed by service, fleet redeploy, watcher, digest, escalation revalidation, info-L0 router

| Capability | Evidence | Verdict |
|---|---|---|
| restart-flat-fields-gone | built here — the 18 restart flat fields S24 γ leaves become service_restart[<service>] / fleet_redeploy (the submodel names absorb the whole prefix) | PASS |
| watcher-flat-fields-gone | built here — the remaining watcher_* flat fields become the watcher submodel | PASS |
| digest-flat-fields-gone | built here — digest_* become the digest submodel | PASS |
| service-restart-declared | built here — OrchestratorConfig.service_restart keyed by RestartedService | PASS |

## ζ2 — Recovery sweeps become one submodel per sweep

| Capability | Evidence | Verdict |
|---|---|---|
| sweep-flat-fields-gone | built here — the orphan-L0, stranded, main-tip, terminal-status, no-landings and warm-lane-GC flat fields become recovery_sweeps.<sweep> (the nested names absorb the prefix) | PASS |
| recovery-sweeps-declared | built here — OrchestratorConfig.recovery_sweeps | PASS |

## ζ3 — The scheduler submodel, and the two double-duty knobs get their own fields

| Capability | Evidence | Verdict |
|---|---|---|
| double-duty-reads-substrate | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_backfill_terminal_dep_records` substrate — the second reader ζ3 repoints | PASS |
| local-backfill-threshold-field | built here — scheduler.max_local_backfill_unresolved_cycles (declared in config.py or config_sections/ (OQ-9, ruled by the lead)) | PASS |
| record-fetch-budget-field | built here — delivered_checks.max_record_fetches_per_tick | PASS |
| deferral-comment-gone | built here — the scheduler comment deferring the dedicated field goes | PASS |
| scheduler-readers-on-submodel | built here — no production reader names a moved scheduler knob at the config root (attribute or getattr string); config.scheduler.<field> cannot match | PASS |
| scheduler-submodel-declared | built here — OrchestratorConfig.scheduler | PASS |

## ζ4a — Verify admission, clock-stop and cgroup knobs become submodels

| Capability | Evidence | Verdict |
|---|---|---|
| verify-admission-flat-fields-gone | built here — verify_admission_*, verify_clock_stop_*, verify_cgroup_cpu_weight_* and verify_use_cgroup_scope become submodels (the submodel names absorb the prefix) | PASS |
| verify-admission-declared | built here — OrchestratorConfig.verify_admission | PASS |

## ζ6a — Workflow iteration caps and review cadence become the workflow submodel

| Capability | Evidence | Verdict |
|---|---|---|
| workflow-cap-readers-on-submodel | built here — no production reader names a moved iteration cap at the config root; config.workflow.<field> cannot match; shared/ is scoped to cli_result.py, where S20 β moves the one docstring that names a cap (ζ6a no longer declares cli_invoke.py) | PASS |
| workflow-submodel-declared | built here — OrchestratorConfig.workflow | PASS |

## ζ6b — Steward knobs become the steward submodel; rebase, revalidation and cleanup knobs join the workflow submodel

| Capability | Evidence | Verdict |
|---|---|---|
| steward-and-workflow-knob-readers-on-submodel | built here — no production reader names a moved steward/rebase/revalidation knob at the config root | PASS |
| steward-submodel-declared | built here — OrchestratorConfig.steward (non-role steward knobs) | PASS |

## ζ4b — Verify timeouts, retries, cold commands and concurrency become the verify submodel; per-module overrides map onto it

| Capability | Evidence | Verdict |
|---|---|---|
| verify-timeout-readers-on-submodel | built here — no production reader names a moved verify timeout/retry/cold-command knob on the root config (ModuleConfig keeps its flat override names, so module_config.<name> is excluded) | PASS |
| module-override-targets-declared | built here — orchestrator/src/orchestrator/config.py::MODULE_OVERRIDE_TARGETS maps each flat module key to its root path (D5) | PASS |
| spec-overrides-take-dotted-paths | built here — orchestrator/src/orchestrator/verify_spec_shared.py::apply_config_overrides, the dotted-path override helper S4 η's apply_merge_verify_spec calls (task 9135 deletes apply_spec_to_host_config); polarity re-measured at ζ4b's filing, after 9135 has landed | PASS |
| verify-submodel-declared | built here — OrchestratorConfig.verify | PASS |

## ζ5a — Merge-lane knobs become the merge_lane submodel

| Capability | Evidence | Verdict |
|---|---|---|
| merge-lane-readers-on-submodel | built here — no production reader names a moved merge-lane knob at the config root (attribute or getattr string, gates.py's merge_verify_breadth included) | PASS |
| merge-lane-submodel-declared | built here — OrchestratorConfig.merge_lane | PASS |

## ζ5b — Verify-host knobs become the verify_hosts submodel

| Capability | Evidence | Verdict |
|---|---|---|
| verify-host-readers-on-submodel | built here — no production reader names a moved verify-host knob at the config root | PASS |
| verify-hosts-submodel-declared | built here — OrchestratorConfig.verify_hosts | PASS |

## κ — Dark-factory's operator yaml, its dashboard module yaml and its operator docs use the new setting names

| Capability | Evidence | Verdict |
|---|---|---|
| repo-yaml-knob-major-blocks-gone | built here — dark-factory-orchestrator.yaml's models/budgets/max_turns/effort/timeouts blocks become roles | PASS |
| repo-yaml-flat-watcher-gone | built here — the flat watcher_* keys move under watcher / roles.watcher | PASS |
| dashboard-module-yaml-root-keys-moved | built here — dashboard/orchestrator.yaml's project-level role blocks move to roles (D5); its module-override keys stay flat | PASS |
| operator-prose-names-new-paths | built here — the instruction-bearing operator prose stops naming moved orchestrator settings (94 lines in 10 files at ce6673c7e1) | PASS |

## χ — The escalation watcher spawns the session of a gate Leo pre-authorised, in attended and AFK posture alike

| Capability | Evidence | Verdict |
|---|---|---|
| watcher-honours-preauthorised-gate | built here — skills/escalation-watcher/SKILL.md §"Always ask — keyed on record content" gains the x_preauthorised_spawn exception, and AFK shift 2 names it | PASS |

## σ — Operator gate: a pre-authorised /team session rewrites the nine project yamls and the two leo-laptop host yamls to the new setting names

| Capability | Evidence | Verdict |
|---|---|---|
| sweep-session-pre-authorised | gate shape — the description quotes Leo's pre-authorisation of 2026-11-12 and states the spawn may run while he is away | PASS |
| deploy-check-before-any-edit | one-decision-partition-per-area (INV-23) — step 1 (every running unit runs ζ5b's code) is decided by code before any file is edited | PASS |
| operated-yamls-carry-new-names | the session's report records one yaml-only commit sha or skip reason per repository and check-config per file | PASS |

## ξ — Old setting names are refused at load and at hot reload in both services, with an error naming the new path; the translation path is deleted

| Capability | Evidence | Verdict |
|---|---|---|
| refusal-type-in-shared | built here — shared/src/shared/config_fields.py::LegacySpellingRefused (C3 end state) | PASS |
| refusal-function-in-shared | built here — shared/src/shared/config_fields.py::refuse_legacy_spellings replaces the pivot (C3) | PASS |
| orchestrator-refuses-old-names | built here — OrchestratorConfig's root validator calls refuse_legacy_spellings | PASS |
| fused-memory-refuses-old-names | built here — FusedMemoryConfig's root validator calls refuse_legacy_spellings | PASS |
| golden-fixture-asserts-refusal | built here — the row-18 golden test asserts the old-spelled copies are refused (rows 21-23) | PASS |
| operated-yamls-integration-check | the refusing check-config over the nine project yamls and both laptop files, recorded in the commit message | PASS |

## π — The scoped-cap model has one spelling: one shared Fable id, one alias decoder, and scope_for matches through it

| Capability | Evidence | Verdict |
|---|---|---|
| scope-for-substrate | `grep:shared/src/shared/usage_gate.py::scope_for` substrate — the exact-match scope derivation π widens (S20 ε moves it to shared/account_state.py first) | PASS |
| fable-model-one-home | built here — shared/src/shared/model_names.py::FABLE_MODEL (C9) | PASS |
| alias-decoder | built here — shared/src/shared/model_names.py::canonical_model, the one alias decoder (C9) | PASS |
| scope-default-names-the-constant | built here — UsageCapConfig.scoped_cap_models defaults to [FABLE_MODEL] | PASS |
| stale-fable-literal-gone | built here — the ['claude-fable-5'] default literal goes | PASS |
| repo-yaml-scope-override-gone | built here — dark-factory's scoped_cap_models override (it would restate the default) goes | PASS |
| admitted-literal-named-once | built here — orchestrator/routing.py stops restating the admitted Fable id | PASS |

## ω — Integration gate: both services load, reload, refuse old names and route through the shared config contracts

| Capability | Evidence | Verdict |
|---|---|---|
| orchestrator-integration-gate | built here — orchestrator/tests/test_config_shape_integration_gate.py | PASS |
| fused-memory-integration-gate | built here — fused-memory/tests/test_routing_integration_gate.py | PASS |
| live-fleet-loads-and-reloads | INV-13 producer proof — after deploy, both running services' reload_config with unchanged yamls applies nothing, and check-config across the known roots and the two remote-verify host yamls lists no old name, unknown key or conflict | PASS |
